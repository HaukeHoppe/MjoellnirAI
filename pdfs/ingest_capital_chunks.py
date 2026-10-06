"""
Offline ingestion: builds a LangChain FAISS index from faiss_capital_index/extracted_v2_all.json.

Each chunk is indexed HyPE-style - one vector for its `embedding_text` plus one
vector per `hypothetical_questions` entry - but every vector's docstore entry
holds the chunk's full `content` and metadata, so any hit resolves to the whole
chunk. The pipeline de-duplicates hits by `chunk_id`.

Steps on top of the plain embedding:

1. Concept / opinion classification (custom LLM tagging step). Every chunk gets
   `content_class`: "concept" (evergreen explanation) or "opinion" (time-bound
   view / forecast / positioning). Mixed chunks count as opinion, so a stale
   view is never presented as timeless. Results are cached in
   chunk_classes.json (keyed by chunk id + content hash), so re-runs only
   classify new or changed chunks. Each relation is also tagged
   `relation_class`: "timeless" (general mechanism) or "time-bound", cached in
   relation_classes.json - an opinion chunk can still contain a timeless mechanism.
2. RAPTOR-style summaries, adapted from raptor.py: concept chunks that explain
   the same thing in different videos are clustered (average-linkage on the
   embedding_text vectors instead of raptor.py's GMM, so unrelated chunks are
   never forced into a cluster) and summarized into one canonical node
   (chunk_type "summary", sources = union of its children). Opinion chunks are
   never merged. Written to summaries.json and added to the index.
3. graph.json: a concept graph built from every chunk's `relations`
   (nodes = concepts, edges = source -> target per direction, each edge listing
   the chunks that assert it with their mechanism / conditions / time_horizon).

Overwrites index.faiss / index.pkl / graph.json / summaries.json in the index dir
(extracted_v2_all.json is kept as the source of truth). Run inside the capital pipelines
container, then restart it so capital_rag_pipeline.py picks up the new index:

    docker exec -it open-webui-pipelines-capital python /data/ingest_capital_chunks.py
    docker restart open-webui-pipelines-capital
"""

import argparse
import hashlib
import json
import os
import re
from typing import List, Literal

import numpy as np
from pydantic import BaseModel, Field
from sklearn.cluster import AgglomerativeClustering
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INDEX_DIR = os.path.join(THIS_DIR, "faiss_capital_index")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a LangChain FAISS index from capital-markets extracted_v2_all.json."
    )
    parser.add_argument("--index_dir", default=DEFAULT_INDEX_DIR)
    parser.add_argument(
        "--chunks", default=None, help="Defaults to <index_dir>/extracted_v2_all.json"
    )
    # Must match CAPITAL_EMBEDDING_MODEL in the pipeline's valves.
    parser.add_argument("--embedding_model", default="text-embedding-3-large")
    parser.add_argument("--llm_model", default="gpt-4o-mini")
    # Cosine similarity of the embedding_text vectors above which two concept
    # chunks count as the same explanation. Measured on this corpus: real
    # cross-video repeats sit at ~0.74+, different topics start mixing at ~0.70.
    parser.add_argument("--cluster_similarity", type=float, default=0.73)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Concept / opinion classification
# ---------------------------------------------------------------------------


class ChunkClass(BaseModel):
    content_class: Literal["concept", "opinion"] = Field(
        ..., description="concept = evergreen explanation, opinion = time-bound view"
    )
    reason: str = Field(..., description="One short sentence explaining the choice.")


CLASSIFY_PROMPT = """You classify a chunk from a transcribed capital-markets video by one creator.

concept = evergreen teaching content: how markets, instruments or economic mechanisms work, historical regularities, general principles or strategies. It stays correct no matter when it was said.
opinion = time-bound content: the creator's view of the current market, forecasts, expectations, positioning, trade ideas, or commentary on current prices, recent moves or upcoming events. It decays over time.

If the chunk contains ANY current market assessment, forecast or recommendation, classify it as opinion, even if it also explains a concept. Presenting a stale view as timeless is the worse mistake.

Title: {title}

Chunk:
{content}"""


def content_hash(chunk: dict) -> str:
    return hashlib.sha256((chunk["title"] + chunk["content"]).encode("utf-8")).hexdigest()


# The org's gpt-4o-mini limit is 200k tokens/min; 8 parallel requests hit it.
MAX_CONCURRENCY = 4
# Classification progress is written to the cache after every batch, so a crash
# or rerun only redoes the unfinished batch.
CACHE_BATCH = 250


def save_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def classify_chunks(chunks: list, llm: ChatOpenAI, cache_path: str) -> dict:
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)

    todo = [
        c
        for c in chunks
        if cache.get(str(c["id"]), {}).get("hash") != content_hash(c)
    ]
    print(f"Classifying {len(todo)} chunks ({len(chunks) - len(todo)} cached) ...")
    if todo:
        classifier = llm.with_structured_output(ChunkClass)
        for start in range(0, len(todo), CACHE_BATCH):
            batch = todo[start : start + CACHE_BATCH]
            results = classifier.batch(
                [CLASSIFY_PROMPT.format(title=c["title"], content=c["content"]) for c in batch],
                config={"max_concurrency": MAX_CONCURRENCY},
            )
            for chunk, result in zip(batch, results):
                cache[str(chunk["id"])] = {
                    "hash": content_hash(chunk),
                    "content_class": result.content_class,
                    "reason": result.reason,
                }
            save_json(cache_path, cache)
            print(f"  {start + len(batch)}/{len(todo)} chunks classified")

    # Drop entries for chunks that no longer exist.
    ids = {str(c["id"]) for c in chunks}
    cache = {k: v for k, v in cache.items() if k in ids}
    save_json(cache_path, cache)
    return {int(k): v for k, v in cache.items()}


class RelationClass(BaseModel):
    relation_class: Literal["timeless", "time-bound"] = Field(
        ..., description="timeless = general mechanism, time-bound = tied to the current situation"
    )


RELATION_PROMPT = """You classify one cause -> effect relation extracted from a capital-markets video.

timeless = a general mechanism that holds no matter when it was said (e.g. "Steigende Zinsen drücken die Anleihepreise").
time-bound = refers to the current market situation, specific current levels, dates or events, a forecast, or a positioning / trade idea.

Judge by the mechanism text. The time_horizon field only says how fast the effect unfolds (e.g. medium-term) - it does NOT make a relation time-bound. A mechanism phrased as a general rule ("Steigende Zinsen erhöhen die Finanzierungskosten, was die Aktienbewertungen drückt") is timeless.

Relation: {source} -> {target} ({direction}, {time_horizon})
Mechanism: {mechanism}
Conditions: {conditions}"""


def relation_hash(chunk_id, rel: dict) -> str:
    key = json.dumps([chunk_id, rel], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def classify_relations(chunks: list, llm: ChatOpenAI, cache_path: str) -> None:
    # A chunk tagged "opinion" can still contain timeless mechanisms. Tagging each
    # relation lets the answer state those generally while the rest of the chunk
    # stays attributed to its video. Sets rel["relation_class"] in place.
    # One relation per request on purpose: labelling 25 per request misclassified
    # general mechanisms as time-bound (17/50 agreement with single requests).
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)

    keys = {id(rel): relation_hash(c["id"], rel) for c in chunks for rel in c["relations"]}
    todo = [rel for c in chunks for rel in c["relations"] if keys[id(rel)] not in cache]
    print(f"Classifying {len(todo)} relations ({len(keys) - len(todo)} cached) ...")
    if todo:
        classifier = llm.with_structured_output(RelationClass)
        for start in range(0, len(todo), CACHE_BATCH):
            batch = todo[start : start + CACHE_BATCH]
            results = classifier.batch(
                [RELATION_PROMPT.format(**rel) for rel in batch],
                config={"max_concurrency": MAX_CONCURRENCY},
            )
            for rel, result in zip(batch, results):
                cache[keys[id(rel)]] = result.relation_class
            save_json(cache_path, cache)
            print(f"  {start + len(batch)}/{len(todo)} relations classified")

    cache = {k: v for k, v in cache.items() if k in set(keys.values())}
    save_json(cache_path, cache)
    for chunk in chunks:
        for rel in chunk["relations"]:
            rel["relation_class"] = cache[keys[id(rel)]]


# ---------------------------------------------------------------------------
# RAPTOR-style canonical summaries for repeated concept explanations
# ---------------------------------------------------------------------------


class ClusterSummary(BaseModel):
    title: str = Field(..., description="Short German title of the concept.")
    content: str = Field(..., description="The canonical explanation in German.")


SUMMARIZE_PROMPT = """The passages below are the same creator explaining the same concept in different videos.
Write ONE canonical explanation in German that combines them.
- Use only statements contained in the passages. Do not add outside knowledge.
- Keep the mechanisms (cause -> effect) and the conditions under which they hold.
- If the passages disagree, state the disagreement explicitly.

{passages}"""


def video_of(source: str) -> str:
    # "Marktgespräch [123456789] (de-x-autogen) #1" -> "Marktgespräch [123456789] (de-x-autogen)"
    # "Marktkommentar Mai 2026_transcript #3" -> "Marktkommentar Mai 2026_transcript"
    return re.sub(r"\s*#\d+$", "", source)


def build_summaries(
    chunks: list, classes: dict, primary_vectors: dict, llm: ChatOpenAI, similarity: float
) -> list:
    concept_chunks = [c for c in chunks if classes[c["id"]]["content_class"] == "concept"]
    if len(concept_chunks) < 2:
        return []

    X = np.array([primary_vectors[c["id"]] for c in concept_chunks])
    labels = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=1 - similarity,
    ).fit_predict(X)

    clusters = []
    for label in sorted(set(labels)):
        members = [c for c, l in zip(concept_chunks, labels) if l == label]
        videos = {video_of(s) for c in members for s in c["sources"]}
        # Only repeated explanations across videos become a canonical node.
        if len(members) >= 2 and len(videos) >= 2:
            clusters.append(members)

    print(f"Summarizing {len(clusters)} cross-video concept clusters ...")
    if not clusters:
        return []

    summarizer = llm.with_structured_output(ClusterSummary)
    results = summarizer.batch(
        [
            SUMMARIZE_PROMPT.format(
                passages="\n\n---\n\n".join(
                    f"[{video_of(c['sources'][0])}] {c['title']}\n{c['content']}"
                    for c in members
                )
            )
            for members in clusters
        ],
        config={"max_concurrency": MAX_CONCURRENCY},
    )

    summaries = []
    for n, (members, result) in enumerate(zip(clusters, results)):
        summaries.append(
            {
                "id": f"summary-{n}",
                "chunk_type": "summary",
                "content_class": "concept",
                "title": result.title,
                "content": result.content,
                "concepts": sorted({k for c in members for k in c["concepts"]}),
                "market_domain": sorted({d for c in members for d in c["market_domain"]}),
                "sources": sorted({s for c in members for s in c["sources"]}),
                "support": len(members),
                "child_chunk_ids": [c["id"] for c in members],
            }
        )
    return summaries


# ---------------------------------------------------------------------------
# Index + graph
# ---------------------------------------------------------------------------


def chunk_metadata(chunk: dict, chunk_class: dict) -> dict:
    return {
        "chunk_id": chunk["id"],
        "chunk_type": chunk["chunk_type"],
        "content_class": chunk_class["content_class"],
        "title": chunk["title"],
        "concepts": chunk["concepts"],
        "relations": chunk["relations"],
        "conditions": chunk["conditions"],
        "evidence": chunk["evidence"],
        "market_domain": chunk["market_domain"],
        "sources": chunk["sources"],
        # extracted_v2 chunks carry no support count; each has exactly one source.
        "support": chunk.get("support", len(chunk["sources"])),
    }


def summary_metadata(summary: dict) -> dict:
    return {
        "chunk_id": summary["id"],
        "chunk_type": "summary",
        "content_class": "concept",
        "title": summary["title"],
        "concepts": summary["concepts"],
        "relations": [],
        "conditions": "",
        "evidence": "summary",
        "market_domain": summary["market_domain"],
        "sources": summary["sources"],
        "support": summary["support"],
        "child_chunk_ids": summary["child_chunk_ids"],
        "matched_text": summary["content"],
    }


def build_graph(chunks: list) -> dict:
    nodes = {}
    edges = {}
    for chunk in chunks:
        for concept in chunk["concepts"]:
            nodes.setdefault(concept, {"id": concept, "chunk_ids": []})
            nodes[concept]["chunk_ids"].append(chunk["id"])
        for rel in chunk["relations"]:
            for concept in (rel["source"], rel["target"]):
                nodes.setdefault(concept, {"id": concept, "chunk_ids": []})
            key = (rel["source"], rel["target"], rel["direction"])
            edge = edges.setdefault(
                key,
                {
                    "source": rel["source"],
                    "target": rel["target"],
                    "direction": rel["direction"],
                    "weight": 0,
                    "evidence": [],
                },
            )
            edge["weight"] += 1
            edge["evidence"].append(
                {
                    "chunk_id": chunk["id"],
                    "mechanism": rel["mechanism"],
                    "conditions": rel["conditions"],
                    "time_horizon": rel["time_horizon"],
                    "relation_class": rel["relation_class"],
                }
            )
    for node in nodes.values():
        node["chunk_ids"] = sorted(set(node["chunk_ids"]))
    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


def main():
    args = parse_args()
    chunks_path = args.chunks or os.path.join(args.index_dir, "extracted_v2_all.json")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set in the environment. "
            "It should already be set via docker-compose.yaml for this container."
        )

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    # max_retries: back off on 429s instead of aborting the whole run.
    llm = ChatOpenAI(model=args.llm_model, temperature=0, max_retries=30)
    classes = classify_chunks(
        chunks, llm, os.path.join(args.index_dir, "chunk_classes.json")
    )
    n_opinion = sum(1 for c in classes.values() if c["content_class"] == "opinion")
    print(f"  {len(classes) - n_opinion} concept, {n_opinion} opinion")
    classify_relations(chunks, llm, os.path.join(args.index_dir, "relation_classes.json"))
    rel_classes = [r["relation_class"] for c in chunks for r in c["relations"]]
    print(
        f"  {rel_classes.count('timeless')} timeless, "
        f"{rel_classes.count('time-bound')} time-bound relations"
    )

    embed_inputs, contents, metadatas = [], [], []
    primary_positions = {}
    for chunk in chunks:
        primary_positions[chunk["id"]] = len(embed_inputs)
        for text in [chunk["embedding_text"], *chunk["hypothetical_questions"]]:
            embed_inputs.append(text)
            contents.append(chunk["content"])
            metadatas.append(
                {**chunk_metadata(chunk, classes[chunk["id"]]), "matched_text": text}
            )

    print(
        f"Embedding {len(embed_inputs)} texts for {len(chunks)} chunks "
        f"with {args.embedding_model} ..."
    )
    embeddings = OpenAIEmbeddings(model=args.embedding_model, max_retries=10)
    vectors = embeddings.embed_documents(embed_inputs)

    primary_vectors = {cid: vectors[pos] for cid, pos in primary_positions.items()}
    summaries = build_summaries(
        chunks, classes, primary_vectors, llm, args.cluster_similarity
    )
    if summaries:
        summary_vectors = embeddings.embed_documents([s["content"] for s in summaries])
        for summary, vector in zip(summaries, summary_vectors):
            contents.append(summary["content"])
            vectors.append(vector)
            metadatas.append(summary_metadata(summary))
    summaries_path = os.path.join(args.index_dir, "summaries.json")
    with open(summaries_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=1)
    print(f"Saved {len(summaries)} canonical concept summaries to {summaries_path}")

    # from_embeddings stores `contents` (the full chunk) as page_content while
    # indexing the embedding_text / hypothetical-question vectors.
    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(contents, vectors)),
        embedding=embeddings,
        metadatas=metadatas,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        normalize_L2=True,
    )
    vectorstore.save_local(args.index_dir)
    print(f"Saved LangChain FAISS index to {args.index_dir}")

    graph = build_graph(chunks)
    graph_path = os.path.join(args.index_dir, "graph.json")
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=1)
    print(
        f"Saved concept graph ({len(graph['nodes'])} nodes, "
        f"{len(graph['edges'])} edges) to {graph_path}"
    )


if __name__ == "__main__":
    main()

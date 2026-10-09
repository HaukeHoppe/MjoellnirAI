"""
Exports the concept graph of the public index as a slim JSON file for the graph explorer (graph-explorer/).

Reads graph.json (concepts and cause -> effect relations), extracted_v2_all.json (chunk titles and sources),
concept_map.json (German surface forms of each concept) and attribution.json (Wikipedia URLs) from the index
and writes data/explorer/graph.json, which the graph-explorer container serves to the browser.

- Only concepts that take part in at least one relation are kept; the explorer is about chains.
- Each concept gets a German label: the surface form (from concept_map.json) that occurs most often in the
  texts of its chunks, so "Interest Rates" is shown as "Zinsen".
- Each relation keeps its evidence (mechanism sentence, conditions, time horizon, timeless or time-bound) and
  the chunk it comes from, with the chunk's title, source and Wikipedia link.

The file is published as is. Only export the public index (Wikipedia and own texts); the private video corpus
(faiss_capital_index) must not be exported, so another index needs --allow_private.

Run on the host or in the pipelines-capital container (no API calls, a few seconds):
    python data/export_explorer_graph.py
    docker exec open-webui-pipelines-capital python /data/export_explorer_graph.py --index_dir /data/faiss_public_index --out /data/explorer/graph.json
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index_dir", default=os.path.join(HERE, "faiss_public_index"))
    parser.add_argument("--out", default=os.path.join(HERE, "explorer", "graph.json"))
    parser.add_argument(
        "--allow_private", action="store_true", help="export an index other than faiss_public_index"
    )
    return parser.parse_args()


def load(index_dir: str, name: str):
    with open(os.path.join(index_dir, name), encoding="utf-8") as f:
        return json.load(f)


def german_labels(nodes: list, chunk_text: dict, concept_map: dict) -> dict:
    # Surface forms per concept, counted in the texts of the concept's chunks; the most frequent one wins,
    # the shorter one on a tie. A concept without any form in its texts keeps its English id.
    forms = defaultdict(set)
    for surface, concept in concept_map.items():
        forms[concept].add(surface)
    labels = {}
    for node in nodes:
        concept = node["id"]
        text = " ".join(chunk_text.get(c, "") for c in node["chunk_ids"]).lower()
        counts = Counter()
        for form in forms.get(concept, ()):
            counts[form] = len(re.findall(r"(?<!\w)" + re.escape(form.lower()) + r"(?!\w)", text))
        best = max(counts.items(), key=lambda kv: (kv[1], -len(kv[0])), default=(concept, 0))
        if best[1] > 0:
            labels[concept] = best[0]
        else:
            # No form occurs verbatim (the extractor often rephrased): take the form that looks most German.
            others = [f for f in forms.get(concept, ()) if f != concept]
            labels[concept] = max(others, key=lambda f: (german_score(f), -len(f)), default=concept)
    return labels


def german_score(form: str) -> int:
    # Umlauts and German function words are a clear sign; English titles capitalize every word,
    # German ones only the nouns ("Rechte der Aktionäre" vs. "First Shares To Outsiders").
    words = form.split()
    score = 2 if re.search(r"[äöüß]", form, re.I) else 0
    score += 2 if any(w in {"der", "die", "das", "des", "den", "dem", "und", "von", "im", "am", "zur", "zum"} for w in words) else 0
    score += 1 if len(words) == 1 or not all(w[:1].isupper() for w in words) else 0
    return score


def source_of(chunk: dict, urls: dict) -> dict:
    # "Wikipedia: Aktie #0" -> the article "Aktie" with its URL; own texts ("Mjoelnir-Erklärtext: Aktie #0")
    # have no URL.
    raw = chunk["sources"][0] if chunk["sources"] else ""
    kind, _, rest = raw.partition(": ")
    article = rest.rsplit(" #", 1)[0]
    entry = {"title": chunk["title"], "source": f"{kind}: {article}" if article else kind}
    if kind == "Wikipedia" and article in urls:
        entry["url"] = urls[article]
    return entry


def main():
    args = parse_args()
    if os.path.basename(os.path.normpath(args.index_dir)) != "faiss_public_index" and not args.allow_private:
        sys.exit(f"{args.index_dir} is not the public index; the export is published. Use --allow_private.")

    graph = load(args.index_dir, "graph.json")
    chunks = {int(c["id"]): c for c in load(args.index_dir, "extracted_v2_all.json")}
    concept_map = load(args.index_dir, "concept_map.json")
    urls = {a["title"]: a["url"] for a in load(args.index_dir, "attribution.json")}

    linked = {e["source"] for e in graph["edges"]} | {e["target"] for e in graph["edges"]}
    nodes = [n for n in graph["nodes"] if n["id"] in linked]
    labels = german_labels(nodes, {i: c["content"] for i, c in chunks.items()}, concept_map)

    edges, used_chunks = [], set()
    for edge in graph["edges"]:
        evidence = []
        for ev in edge["evidence"]:
            chunk_id = int(ev["chunk_id"])
            used_chunks.add(chunk_id)
            evidence.append(
                {
                    "chunk": chunk_id,
                    "mechanism": ev["mechanism"],
                    "conditions": ev["conditions"],
                    "horizon": ev["time_horizon"],
                    "timeless": ev["relation_class"] == "timeless",
                }
            )
        edges.append(
            {
                "source": edge["source"],
                "target": edge["target"],
                "direction": edge["direction"],
                "weight": edge["weight"],
                "evidence": evidence,
            }
        )

    out = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "concepts": [{"id": n["id"], "label": labels[n["id"]], "chunks": len(n["chunk_ids"])} for n in nodes],
        "edges": edges,
        "chunks": {str(i): source_of(chunks[i], urls) for i in sorted(used_chunks) if i in chunks},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    size = os.path.getsize(args.out) / 1e6
    print(
        f"Wrote {len(out['concepts'])} concepts, {len(edges)} relations and {len(out['chunks'])} source chunks "
        f"to {args.out} ({size:.1f} MB)"
    )


if __name__ == "__main__":
    main()

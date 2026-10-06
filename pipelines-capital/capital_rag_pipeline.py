"""
title: Capital Markets RAG
author: Mjoelnir AI
date: 2026-09-25
version: 0.4
license: MIT
description: RAG over the capital-markets chunks in faiss_public_index (Wikipedia articles and own explanatory texts, built by build_public_kb.py and ingest_capital_chunks.py; INDEX_DIR can point to another index such as faiss_capital_index). Query flow - fusion retrieval (HyPE dense vectors + BM25, adapted from fusion_retrieval.py), concept-graph expansion (graph.json), causal chain search (cause -> effect paths of up to 4 steps through the directed concept graph, with the chunks stating each step), LLM reranking + relevance/grounding grading per chunk (reranking.py / reliable_rag), generation that keeps time-bound opinions attributed to their source, a post-answer grounding check, and a source list that explains why each source was used (explainable_retrieval.py). Explorer mode: without a grounded answer, related covered topics are suggested as verified follow-up questions. Runs in its own pipelines container (pipelines-capital) next to the climate test pipeline.
requirements: langchain-community,langchain-openai,langchain-core,faiss-cpu,openai,pydantic,rank-bm25,numpy
"""

import json
import os
import re
from collections import Counter, OrderedDict
from difflib import SequenceMatcher
from typing import Generator, Iterator, List, Literal, Union

import numpy as np
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class Grade(BaseModel):
    relevance: int = Field(..., ge=0, le=10, description="Relevance to the question, 0-10.")
    supports_answer: bool = Field(
        ..., description="True only if the document directly helps answer the question."
    )
    states_step: bool = Field(
        ...,
        description="True if the document states at least one cause -> effect link of the chain the question asks about, even if it does not answer the whole question.",
    )
    reason: str = Field(
        ..., description="One short sentence, in the language of the question, why."
    )


class Claim(BaseModel):
    claim: str = Field(..., description="One claim from the answer.")
    kind: Literal["fact", "meta"] = Field(
        ...,
        description="meta = only says what the sources do/don't cover, or compares claims from the context without adding information.",
    )
    quote: str = Field(
        ...,
        description="Verbatim passage copied from the context that states the claim; empty if none.",
    )
    claim_cause: str = Field(
        "", description="If the claim states a cause -> effect: the cause named in the claim."
    )
    quote_cause: str = Field(
        "", description="If the claim states a cause -> effect: the cause the quote gives for that effect."
    )
    same_cause: bool = Field(
        True, description="False if claim_cause and quote_cause are different causes."
    )
    supported: bool = Field(
        ..., description="fact: true only if the quote states it. meta: true if it is accurate."
    )
    answer_sentence: str = Field(
        ..., description="The sentence of the answer that contains the claim, copied verbatim."
    )


class AnswerCheck(BaseModel):
    claims: List[Claim] = Field(default_factory=list)


class Suggestion(BaseModel):
    topic: int = Field(..., description="Number of the topic in the list.")
    label: str = Field(..., description="Short topic name in the language of the user's question.")
    question: str = Field(
        ...,
        description="A question in the language of the user's question that the excerpt of this topic answers.",
    )


class Suggestions(BaseModel):
    suggestions: List[Suggestion] = Field(default_factory=list)


class QueryConcepts(BaseModel):
    causes: List[str] = Field(default_factory=list, description="Triggers / causes named in the question")
    effects: List[str] = Field(default_factory=list, description="Effects the question asks about; empty if none")


QUERY_CONCEPTS_PROMPT = """Name the cause -> effect concepts in this capital-markets question as short English
Title Case names of economic quantities, actors or events (e.g. "Stock Market Crash", "Gold Price",
"Interest Rates", "Bank Lending"). causes: the trigger(s) the question starts from. effects: what the question
asks the consequences for; leave empty if it asks for consequences in general.

Question: {question}"""


GRADE_PROMPT = """Rate how well the document helps answer the question.
Consider the specific intent of the question, not just keyword or topic overlap.

relevance: 0-10.
supports_answer: true only if the document contains information that directly helps answer the question.
states_step: true if the document states at least one cause -> effect link that belongs to the chain the question asks about - the trigger, an intermediate step or the final effect (e.g. for "How can a rate decision in Japan hit US tech stocks?": a document stating that Japan's low rates lead to borrowing in yen, or that rising rates lower stock valuations). It does not need to answer the whole question. False for documents that only share the topic or keywords without stating such a link.
reason: one short sentence in the language of the question.

Question: {question}

Document: {title}
{content}"""

ANSWER_PROMPT = """Answer the question using ONLY the context below. If the context doesn't support an answer, say so.
Cite the context blocks you use as [1], [2], ... after every statement that relies on them.

Do NOT use your own knowledge - not even for well-known textbook facts or to explain a mechanism. Every statement, including every reason ("weil ...", "because ..."), must be stated in the context. If the context gives the effect but not the reason (or only part of what was asked), answer only that part and say explicitly that the sources do not explain the rest.

Stay faithful to the wording of the sources:
- Keep their degree of certainty: "könnte" / "kann" stays a possibility, never "typischerweise" or "immer".
- Only state a cause -> effect link that the context states as such. Do not join two separate statements into a new cause -> effect link.
- You may build a chain of several steps when every single step is stated in the context: A -> B in one block, B -> C in another. Write each step as its own sentence with its own citation. Never state the final effect as a direct result of the trigger (A -> C) unless the context states that link itself.

Mechanism questions (why / how / what happens if) are answered as a chain, as long as the context supports it:
**<Short heading naming the mechanism>**
1. **<Step>**: cause -> effect and how it works [n]
2. **<Step>**: ... (as many steps as the context supports, in causal order)
**Bedingungen und Gegeneffekte**: when the chain holds or breaks, and what works against it, as stated in the context.
No introduction or summary sentence that links the trigger directly to the final effect - the steps make that link.
Phrase each step close to the wording of its block and give only the reason the block gives; do not add reasons, actors or effects of your own.
The context may start with "Causal paths": chains of relations found across the sources, each step with the blocks that state it. Use them to find and order the steps; every step still needs its block citation. Other questions are answered in normal prose.
Build the chain from all blocks, not only from the block that covers the most. Even if one block describes the whole chain, read the other blocks: cite every block that states a step (e.g. [1][3]), and use them for links that block leaves out, such as the link from the trigger named in the question to the first step, or from the last step to the effect asked about.

The context blocks come from the indexed capital-markets sources (each block label names its source) and come in two kinds:
- CONCEPT blocks are timeless explanations. You may state them as general explanations.
- OPINION blocks are time-bound views, forecasts or positioning as stated in the named source. Always attribute them to that source in the same paragraph, using the source title exactly as written in the block label without the [id] and #n (e.g. "Laut Marktkommentar Mai 2026 ..."). Never present them as current facts or as a current recommendation.
- Exception: a "Relation [TIMELESS]" line is a general mechanism, even inside an OPINION block. You may state it as a general explanation. Everything else in an OPINION block, including "Relation [TIME-BOUND]" lines, must be attributed to its source.
- If opinion blocks from different sources disagree, show each view with its source and say that they differ. Do not merge them into one view, and do not end with a summary or conclusion that combines views from different sources.

Answer in the language of the question.

Context:
{context}

Question: {question}"""

CHECK_PROMPT = """Check every factual claim in the answer against the context.

Split the answer into individual claims. Every reason or explanation ("weil ...", "because ...", "dadurch ...") is a separate claim. A sentence that joins two statements ("..., wobei ...", "... und ...", "..., während ...") is two claims.
Write every claim self-contained: replace "dadurch", "dies", "was", "somit", "this" etc. with what they refer to, so each claim names its own cause (e.g. "Sinkende Zinsen steigern die Nachfrage nach Anleihen", not "Die Nachfrage steigt dadurch").

For each claim:
- kind: "meta" if it only says what the sources do or do not cover, or only compares / contrasts claims from the context (e.g. "the two sources recommend different positions") without adding any information of its own. Everything else is "fact". A meta statement that adds any new information is a "fact".
- quote (facts only): copy, character for character, ONE sentence from the context that states the claim. Never join sentences from different places. Only exception: if the statement continues in the directly following sentence(s) of the same passage, which refer back with a pronoun ("Sie", "Er", "Dies", "Diese", "It", "This", ...) or a connector ("deshalb", "dadurch", "daher", "somit", "therefore", ...), copy up to three consecutive sentences. If no single sentence states it, the claim is unsupported and quote stays empty. For meta claims leave it empty.
- supported: for a fact, true only if the quote states the claim - general knowledge does NOT count, even if the claim is true. For a meta claim, true if it accurately describes the context.

The quote must state the claim itself: same cause, same effect, same degree of certainty. A fact is unsupported if it:
- turns a possibility ("könnte", "kann", "may") into a rule ("typischerweise", "immer", "typically");
- links a cause and an effect that the context mentions only separately, or links the effect to a different cause. If the quote names a different cause than the claim (claim: falling rates raise demand; quote: investor confidence raises demand), the claim is unsupported;
- attributes a view to the wrong source.
A statement that merges different views from different sources into one is unsupported (fact or meta).

claim_cause / quote_cause / same_cause: for every claim that states a cause -> effect, write down the cause the claim names and the cause the quote gives for the same effect, then set same_cause. Example: claim "Sinkende Zinsen steigern die Nachfrage nach Anleihen", quote "Wenn Investoren Vertrauen zurückgewinnen, ... steigt die Nachfrage" -> claim_cause "sinkende Zinsen", quote_cause "Vertrauen der Investoren", same_cause false.
answer_sentence: the sentence of the answer the claim was taken from, copied verbatim (before you rewrote the claim).

Context:
{context}

Answer:
{answer}"""

REVISE_PROMPT = """Rewrite the answer to fix the problems listed below.
- "not stated in the sources": remove the statement. Do not replace it with a sentence that repeats or denies it ("Die Quellen erklären nicht, dass ...") - that is a new claim about the sources and often wrong. Only if the removal leaves part of the question unanswered, you may add one general note such as "Zu <Aspekt der Frage> sagen die Quellen nichts."
- "missing source attribution": keep the statement but name the given source title in the same paragraph (e.g. "Laut <title> ...").
- "merges separate statements": keep the content but write each source statement as its own sentence with its citation; do not connect them with "wobei", "da", "weil" etc.
Do not add any new information. Keep the citations [1], [2], ... of the remaining statements. Answer in the language of the question.
Return only the rewritten answer text, without a heading or label such as "Answer:".

Context:
{context}

Question: {question}

Answer:
{answer}

Problems:
{unsupported}"""

EXPLORE_PROMPT = """{situation}
Below are topics that ARE covered in the indexed capital-markets sources, each with an excerpt from the sources.

Pick up to {n} topics {goal}.
For each picked topic:
- topic: its number from the list.
- label: a short topic name in the language of the user's question.
- question: one follow-up question in the language of the user's question that the excerpt of this topic answers. Ask only for what the excerpt actually explains - do not ask about details it does not contain.
Skip topics that have nothing to do with the user's interest. Returning fewer topics, or none, is fine.

User's question: {question}

Topics:
{topics}"""

EXPLORE_NO_ANSWER = (
    "The user asked a question that the sources do not answer.",
    "that are most closely related to what the user wanted to know, so they can explore nearby material instead",
)
EXPLORE_ANSWERED = (
    "The user asked a question and already got an answer from the sources.",
    "the user would most likely want to explore next to deepen or broaden that answer. Do not pick topics that only repeat the question",
)

NO_ANSWER = (
    "Dazu habe ich in den Quellen leider keine belegte Antwort gefunden. "
    "Ich antworte nur mit dem, was dort tatsächlich steht – deshalb rate ich hier nicht."
)

# Open WebUI sends its background tasks (title, tags, follow-ups) to the chat
# model itself, as a prompt starting with "### Task:".
FOLLOW_UP_TASK = re.compile(r"^### Task:\s*Suggest .*follow-up questions", re.IGNORECASE)


def normalize(text: str) -> str:
    # Parentheticals ("(Trading Halts)") are dropped: quotes often leave them out.
    text = re.sub(r"\s*\([^)]*\)", "", text.lower())
    text = re.sub(r"[\"'„“”‚‘’«»]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# A sentence continues the previous one if it refers back to it early on: with a
# pronoun ("Volatilität ... beschreibt das Risiko. Sie wird ... gemessen.") or a
# causal connector ("Der Wert der Sicherheiten sinkt. Broker fordern deshalb
# Nachschüsse."). Then the source itself states the link between the sentences.
ANAPHORS = set(
    """
    sie er es dies diese dieser dieses diesen diesem ihr ihre ihren ihrem ihrer
    sein seine seinen seinem seiner deren dessen
    it its they their these this
    """.split()
)
CONNECTORS = set(
    """
    deshalb dadurch daher darum deswegen somit folglich infolgedessen dementsprechend
    therefore thus hence consequently
    """.split()
)


def continues(sentence: str) -> bool:
    words = re.findall(r"\w+", sentence)
    return bool(words) and (words[0] in ANAPHORS or any(w in CONNECTORS for w in words[:8]))


# A numbered step of a chain text ("2. Euro-Abwertung → Importpreise: ..."): one line stating one
# cause -> effect, so its sentences belong together even without a connector.
CHAIN_STEP = re.compile(r"^\s*\d+\.\s")


def in_chain_step(quote: str, context: str) -> bool:
    return any(CHAIN_STEP.match(line) and quote in normalize(line) for line in context.splitlines())


def quote_in_context(quote: str, context: str) -> bool:
    # The checker's quote must really appear in the context, so the check cannot
    # be passed with an invented quote. Minor copy differences are tolerated.
    # A quote is one sentence, up to three consecutive sentences of the same
    # passage where each further sentence refers back to the previous one
    # (continues()), or up to four consecutive sentences within one numbered
    # chain step. Anything else would let a cause from one sentence be attached
    # to an effect from an unrelated one.
    raw_context = context
    quote, context = normalize(quote), normalize(context)
    sentences = re.split(r"(?<=[.!?])\s+", quote)
    if len(quote) < 15 or len(sentences) > 4:
        return False
    if not all(continues(s) for s in sentences[1:]):
        return len(sentences) > 1 and in_chain_step(quote, raw_context)
    if len(sentences) > 3:
        return False
    if quote in context:
        return True
    match = SequenceMatcher(None, context, quote, autojunk=False).find_longest_match(
        0, len(context), 0, len(quote)
    )
    return match.size >= 0.9 * len(quote)


def joins_sentences(quote: str, context: str) -> bool:
    # True if the quote is several sentences that each appear in the context:
    # the claim merges separate source statements and should be split, rather
    # than being unsupported.
    sentences = re.split(r"(?<=[.!?])\s+", quote.strip())
    return len(sentences) > 1 and all(quote_in_context(s, context) for s in sentences)


# Video attribution is checked in code, not by the checker LLM: the LLM rewrites
# claims to be self-contained, dropping the "Im Video ..." it should look for.


def video_key(text: str) -> str:
    # "LIVE zur FED-Entscheidung_ Webinar" and "LIVE zur FED-Entscheidung: Webinar" match.
    return re.sub(r"\W+", "", text.lower())


def parse_video(source: str) -> tuple:
    # "Marktgespräch [123456789] (de-x-autogen) #1" -> ("Marktgespräch", "123456789")
    # "Marktkommentar Mai 2026_transcript #3" -> ("Marktkommentar Mai 2026", "")
    match = re.match(r"^(.*?)\s*\[(\d+)\]", source)
    if match:
        return (match.group(1), match.group(2))
    return (re.sub(r"_transcript$", "", re.sub(r"\s*#\d+$", "", source)), "")


def required_videos(quote: str, blocks: List[dict]) -> List[tuple]:
    # The videos a quoted statement must be attributed to; empty if it comes from
    # a CONCEPT block or a TIMELESS relation line (may be stated generally).
    matched = [b for b in blocks if quote_in_context(quote, b["text"])]
    for block in matched:
        if not block["opinion"]:
            return []
        for line in block["timeless"]:
            if quote_in_context(quote, line) or normalize(line) in normalize(quote):
                return []
    return sorted({video for block in matched for video in block["videos"]})


def paragraph_of(answer: str, sentence: str) -> str:
    paragraphs = [p for p in re.split(r"\n\s*\n", answer) if p.strip()] or [answer]
    sentence = normalize(sentence)

    def overlap(paragraph: str) -> int:
        paragraph = normalize(paragraph)
        return SequenceMatcher(None, paragraph, sentence, autojunk=False).find_longest_match(
            0, len(paragraph), 0, len(sentence)
        ).size

    return max(paragraphs, key=overlap)


def drop_sentences(answer: str, sentences: List[str]) -> str:
    # Removes the answer sentences the checker flagged (matched fuzzily, since
    # the checker copies them only roughly verbatim). Line structure is kept, so
    # bullet lists stay lists; lines left empty are dropped.
    targets = [normalize(s) for s in sentences if s.strip()]

    def flagged(sentence: str) -> bool:
        sentence = normalize(sentence)
        return any(SequenceMatcher(None, sentence, t, autojunk=False).ratio() >= 0.8 for t in targets)

    lines = []
    for line in answer.split("\n"):
        if not line.strip():
            lines.append("")
            continue
        kept = [s for s in re.split(r"(?<=[.!?])\s+", line.strip()) if not flagged(s)]
        if kept:
            lines.append(" ".join(kept))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def names_video(paragraph: str, videos: List[tuple]) -> bool:
    key = video_key(paragraph)
    return any(video_key(title) in key or (vid and vid in paragraph) for title, vid in videos)


# Function words would otherwise make BM25 match almost every chunk.
STOPWORDS = set(
    """
    der die das den dem des ein eine einen einem einer eines und oder aber als auch
    an auf aus bei bis durch für gegen in im ins mit nach ohne über um unter von vor
    zu zum zur ist sind war waren wird werden wurde hat haben hatte kann können
    muss soll sollte würde ich du er sie es wir ihr man mich mir sich was wie wo
    wann warum welche welcher welches wer nicht kein keine noch nur so sehr schon
    dass wenn dann denn da hier jetzt mal ja nein doch es gibt
    the a an and or but of to in on at for with by from is are was were be been
    it this that these those what how why when where which who do does did can
    """.split()
)


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


def cite_source(source: str) -> str:
    # "Marktgespräch [123456789] (de-x-autogen) #1" -> "Marktgespräch [123456789] #1"
    # "Marktkommentar Mai 2026_transcript #3" -> "Marktkommentar Mai 2026 #3"
    source = re.sub(r"_transcript(\s*#\d+)$", r"\1", source)
    return re.sub(r"\s+\([^)]*\)(\s*#\d+)$", r"\1", source)


def same_question(a: str, b: str, idf: dict) -> bool:
    # A suggestion that only rephrases the user's question ("Wie wichtig sind
    # IPO-Daten für ...?" -> "Wie helfen IPO-Daten bei ...?") is no new topic.
    # Shared words are weighted by rarity (BM25 idf), so sharing "IPO" counts,
    # sharing "Daten" or "wichtig" hardly.
    if SequenceMatcher(None, normalize(a), normalize(b), autojunk=False).ratio() >= 0.75:
        return True
    top = max(idf.values(), default=1.0)

    def weight(tokens: set) -> float:
        return sum(idf.get(t, top) for t in tokens)

    tokens_a, tokens_b = set(tokenize(a)), set(tokenize(b))
    lighter = min(weight(tokens_a), weight(tokens_b))
    return lighter > 0 and weight(tokens_a & tokens_b) >= 0.75 * lighter


def cited_numbers(answer: str, count: int) -> set:
    # Context block numbers cited in the answer: "[1]", "[1][3]", "[1], [2]", "[1, 3]".
    numbers = set()
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer):
        numbers.update(int(n) for n in group.split(","))
    return {n for n in numbers if 1 <= n <= count}


def minmax(scores: dict) -> dict:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class Pipeline:
    class Valves(BaseModel):
        # Public knowledge base (build_public_kb.py); any index built by ingest_capital_chunks.py works.
        INDEX_DIR: str = "/data/faiss_public_index"
        # Must match --embedding_model in ingest_capital_chunks.py.
        EMBEDDING_MODEL: str = "text-embedding-3-large"
        # gpt-4.1 follows multi-step chains across blocks better than gpt-4o-mini.
        LLM_MODEL: str = "gpt-4.1"
        # Model used for reranking/grading of retrieved chunks and the query-concept step.
        # gpt-4.1-mini: own rate limits (gpt-4o-mini's daily request limit is shared with builds).
        GRADER_MODEL: str = "gpt-4.1-mini"
        # Model for the claim-by-claim grounding check of the answer. A stronger
        # model than LLM_MODEL, so it catches textbook knowledge the answer model
        # slipped in.
        CHECK_MODEL: str = "gpt-4o"
        # Every chunk has ~9-13 vectors (embedding_text + hypothetical questions),
        # so fetch more raw hits than we need and de-duplicate by chunk_id.
        FETCH_K: int = 40
        # The index uses MAX_INNER_PRODUCT on unit-length vectors, so the score is
        # cosine similarity (higher = closer). Dense hits below this are ignored;
        # a chunk can still come in through BM25 (e.g. exact ticker match).
        MIN_SIMILARITY: float = 0.35
        # Fusion weight: ALPHA * dense + (1 - ALPHA) * BM25 (both min-max normalized).
        ALPHA: float = 0.6
        # Fused candidates passed on to graph expansion and grading.
        CANDIDATE_K: int = 12
        # Graph expansion: chunks asserting the same relation as the top seeds.
        GRAPH_SEED_K: int = 3
        GRAPH_EXPAND_K: int = 4
        # Grading: keep chunks with supports_answer and relevance >= this.
        MIN_RELEVANCE: int = 6
        TOP_K: int = 6
        # Causal chain search: the question's causes and effects are matched to graph concepts, and
        # directed cause -> effect paths of up to CHAIN_MAX_HOPS steps between them are followed through
        # the concept graph. The chunks stating the steps join the candidates; a path chunk only needs
        # CHAIN_MIN_RELEVANCE, because a single step rarely answers the whole question by itself.
        CHAIN_SEARCH: bool = True
        CHAIN_MAX_HOPS: int = 4
        CHAIN_PATHS: int = 3
        CHAIN_EXTRA_K: int = 8
        CHAIN_TOP_K: int = 6
        CHAIN_MIN_RELEVANCE: int = 4
        # Cosine similarity for matching a question concept to a graph concept.
        CONCEPT_MIN_SIMILARITY: float = 0.55
        # The answer is generated in full, then every claim must be backed by a
        # verbatim quote from the context before anything is shown. Unsupported
        # claims are removed (up to MAX_REVISIONS rewrites); if some remain, no
        # answer is given. Disabling this streams the unchecked answer directly.
        CHECK_ANSWER_GROUNDING: bool = True
        MAX_REVISIONS: int = 2
        # Progress is shown as one status line that updates step by step. This
        # additionally writes the full step log (incl. rejected claims) into a
        # collapsible "thinking" block - useful for debugging.
        SHOW_THINKING_LOG: bool = False
        # Explorer mode: when there is no grounded answer, suggest related topics
        # the index does cover, each as a follow-up question. A suggestion is only
        # shown if the grader confirms its excerpt answers that question.
        EXPLORE_MODE: bool = True
        EXPLORE_SUGGESTIONS: int = 4
        # A topic (graph concept) must appear in at least this many chunks to be
        # suggested, so the user is not sent to a one-off mention.
        EXPLORE_MIN_CHUNKS: int = 2

    def __init__(self):
        self.name = "Capital Markets RAG"
        self.valves = self.Valves()
        self.vectorstore = None
        self.docs = {}
        self.bm25 = None
        self.bm25_ids = []
        self.relation_chunks = {}
        self.concept_chunks = {}
        self.concept_neighbors = {}
        self.concept_summaries = {}
        # Directed graph for the chain search: source -> {target: total edge weight}, and unit vectors
        # of the concept names (rows in concept_names order) to match question concepts.
        self.out_edges = {}
        self.concept_names = []
        self.concept_vectors = None
        self.concept_extractor = None
        # Recent questions -> what the follow-up task needs (see _follow_ups).
        self.recent = OrderedDict()
        self.llm = None
        self.answer_llm = None
        self.reviser = None
        self.grader = None
        self.checker = None

    def _load(self):
        # Loads the index built by ingest_capital_chunks.py. Does NOT run ingestion.
        # Missing index must not raise here, or the whole pipelines server fails
        # to start - pipe() reports it instead.
        if not os.path.exists(os.path.join(self.valves.INDEX_DIR, "index.faiss")):
            print(f"[capital_rag] No index in {self.valves.INDEX_DIR} - run ingest_capital_chunks.py")
            self.vectorstore = None
        else:
            # distance_strategy / normalize_L2 are not stored in index.pkl, so they
            # have to be passed again to match how the index was built.
            self.vectorstore = FAISS.load_local(
                self.valves.INDEX_DIR,
                OpenAIEmbeddings(model=self.valves.EMBEDDING_MODEL),
                allow_dangerous_deserialization=True,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                normalize_L2=True,
            )
            self._build_keyword_index()
            self._load_graph()

        self.llm = ChatOpenAI(model=self.valves.LLM_MODEL, temperature=0, streaming=True)
        self.answer_llm = ChatOpenAI(model=self.valves.LLM_MODEL, temperature=0)
        grader_llm = ChatOpenAI(model=self.valves.GRADER_MODEL, temperature=0)
        self.grader = grader_llm.with_structured_output(Grade)
        check_llm = ChatOpenAI(model=self.valves.CHECK_MODEL, temperature=0)
        self.checker = check_llm.with_structured_output(AnswerCheck)
        self.explorer = grader_llm.with_structured_output(Suggestions)
        self.concept_extractor = grader_llm.with_structured_output(QueryConcepts)
        # Revisions use the stronger model too: gpt-4o-mini tended to keep the
        # flagged statement in slightly different words.
        self.reviser = check_llm

    def _build_keyword_index(self):
        # One BM25 document per chunk: title + concepts + content + every text it
        # was embedded under (embedding_text, hypothetical questions).
        self.docs, texts = {}, {}
        self.concept_summaries = {}
        for docstore_id in self.vectorstore.index_to_docstore_id.values():
            doc = self.vectorstore.docstore.search(docstore_id)
            chunk_id = doc.metadata["chunk_id"]
            if chunk_id not in self.docs:
                self.docs[chunk_id] = doc
                meta = doc.metadata
                texts[chunk_id] = [meta["title"], " ".join(meta["concepts"]), doc.page_content]
                if meta["chunk_type"] == "summary":
                    for concept in meta["concepts"]:
                        self.concept_summaries.setdefault(concept, []).append(chunk_id)
            texts[chunk_id].append(doc.metadata.get("matched_text", ""))
        self.bm25_ids = list(texts)
        self.bm25 = BM25Okapi([tokenize(" ".join(texts[i])) for i in self.bm25_ids])

    def _load_graph(self):
        self.relation_chunks = {}
        self.concept_chunks = {}
        self.concept_neighbors = {}
        graph_path = os.path.join(self.valves.INDEX_DIR, "graph.json")
        if not os.path.exists(graph_path):
            print(f"[capital_rag] No graph.json in {self.valves.INDEX_DIR} - graph expansion disabled")
            return
        with open(graph_path, encoding="utf-8") as f:
            graph = json.load(f)
        for node in graph["nodes"]:
            self.concept_chunks[node["id"]] = [c for c in node["chunk_ids"] if c in self.docs]
        self.out_edges = {}
        for edge in graph["edges"]:
            key = (edge["source"], edge["target"], edge["direction"])
            self.relation_chunks[key] = [e["chunk_id"] for e in edge["evidence"]]
            for a, b in ((edge["source"], edge["target"]), (edge["target"], edge["source"])):
                self.concept_neighbors.setdefault(a, Counter())[b] += edge["weight"]
            if edge["source"] != edge["target"]:
                self.out_edges.setdefault(edge["source"], Counter())[edge["target"]] += edge["weight"]
        if self.valves.CHAIN_SEARCH:
            self._load_concept_vectors()

    def _load_concept_vectors(self):
        # Embeddings of all concept names, cached next to the index (concept_vectors.npz) and only
        # recomputed when the graph's concepts change.
        names = sorted(self.concept_chunks)
        path = os.path.join(self.valves.INDEX_DIR, "concept_vectors.npz")
        if os.path.exists(path):
            cached = np.load(path, allow_pickle=False)
            if list(cached["names"]) == names:
                self.concept_names, self.concept_vectors = names, cached["vectors"]
                return
        try:
            vectors = np.array(OpenAIEmbeddings(model=self.valves.EMBEDDING_MODEL).embed_documents(names))
        except Exception as e:
            print(f"[capital_rag] Concept vectors failed, chain search disabled: {e}")
            return
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        self.concept_names, self.concept_vectors = names, vectors.astype(np.float32)
        try:
            np.savez(path, names=np.array(names), vectors=self.concept_vectors)
        except OSError as e:
            print(f"[capital_rag] Could not cache concept vectors: {e}")

    async def on_startup(self):
        self._load()

    async def on_shutdown(self):
        pass

    async def on_valves_updated(self):
        self._load()

    def _fusion_candidates(self, query: str) -> List[tuple]:
        # Adapted from fusion_retrieval.py: min-max normalized dense and BM25
        # scores, combined with ALPHA. Dense scores are per chunk (best vector).
        dense = {}
        for doc, score in self.vectorstore.similarity_search_with_score(
            query, k=self.valves.FETCH_K
        ):
            if score < self.valves.MIN_SIMILARITY:
                continue
            chunk_id = doc.metadata["chunk_id"]
            dense[chunk_id] = max(score, dense.get(chunk_id, score))

        bm25_raw = self.bm25.get_scores(tokenize(query))
        top = np.argsort(bm25_raw)[::-1][: self.valves.FETCH_K]
        keyword = {self.bm25_ids[i]: float(bm25_raw[i]) for i in top if bm25_raw[i] > 0}

        dense_n, keyword_n = minmax(dense), minmax(keyword)
        alpha = self.valves.ALPHA
        fused = {
            chunk_id: alpha * dense_n.get(chunk_id, 0.0)
            + (1 - alpha) * keyword_n.get(chunk_id, 0.0)
            for chunk_id in set(dense) | set(keyword)
        }
        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        return ranked[: self.valves.CANDIDATE_K]

    def _graph_expand(self, candidates: List[tuple]) -> List[tuple]:
        # Chunks from other passages/videos that assert the same relation
        # (source -> target, direction) as the top seeds.
        present = {chunk_id for chunk_id, _ in candidates}
        extra = []
        for chunk_id, _ in candidates[: self.valves.GRAPH_SEED_K]:
            for rel in self.docs[chunk_id].metadata.get("relations", []):
                key = (rel["source"], rel["target"], rel["direction"])
                for other in self.relation_chunks.get(key, []):
                    if other in self.docs and other not in present:
                        present.add(other)
                        extra.append((other, 0.0))
        return candidates + extra[: self.valves.GRAPH_EXPAND_K]

    def _match_concepts(self, phrases: List[str]) -> dict:
        # Question concept phrases -> graph concepts (top 3 per phrase above CONCEPT_MIN_SIMILARITY).
        if not phrases or self.concept_vectors is None:
            return {}
        vectors = np.array(self.vectorstore.embedding_function.embed_documents(phrases), dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        matched = {}
        for row in vectors @ self.concept_vectors.T:
            for i in np.argsort(row)[::-1][:3]:
                if row[i] >= self.valves.CONCEPT_MIN_SIMILARITY:
                    name = self.concept_names[i]
                    matched[name] = max(float(row[i]), matched.get(name, 0.0))
        return matched

    def _causal_paths(self, question: str) -> List[List[str]]:
        # Directed cause -> effect paths through the concept graph, from the question's causes to its
        # effects (breadth-first, so shorter paths first). Without effects: the strongest onward chains.
        asked = self.concept_extractor.invoke(QUERY_CONCEPTS_PROMPT.format(question=question))
        starts = self._match_concepts(asked.causes)
        ends = self._match_concepts(asked.effects)
        if not starts:
            return []
        max_hops, wanted = self.valves.CHAIN_MAX_HOPS, self.valves.CHAIN_PATHS
        paths = []
        if ends:
            # Breadth-first with up to 3 predecessors per concept (strongest edges first), then every
            # reached effect is traced back to the causes.
            preds, frontier, seen = {}, list(starts), set(starts)
            for _ in range(max_hops):
                nxt = []
                for node in frontier:
                    for target, _ in self.out_edges.get(node, Counter()).most_common(30):
                        if target in starts:
                            continue
                        if len(preds.setdefault(target, [])) < 3 and node not in preds[target]:
                            preds[target].append(node)
                        if target not in seen:
                            seen.add(target)
                            nxt.append(target)
                frontier = nxt

            def back(node: str, path: List[str]) -> Iterator[List[str]]:
                if node in starts:
                    yield [node] + path
                    return
                if len(path) >= max_hops:
                    return
                for prev in preds.get(node, []):
                    if prev not in path:
                        yield from back(prev, [node] + path)

            for end in sorted(ends, key=ends.get, reverse=True):
                for path in back(end, []):
                    if len(path) > 1 and path not in paths:
                        paths.append(path)
        else:
            # Strongest onward chain from each cause, branching at the first step.
            for start in sorted(starts, key=starts.get, reverse=True)[:2]:
                for first, _ in self.out_edges.get(start, Counter()).most_common(2):
                    path = [start, first]
                    while len(path) <= max_hops:
                        options = [t for t, _ in self.out_edges.get(path[-1], Counter()).most_common(5)
                                   if t not in path]
                        if not options:
                            break
                        path.append(options[0])
                    paths.append(path)
        # Shorter and better-supported paths first.
        support = lambda p: sum(self.out_edges[a][b] for a, b in zip(p, p[1:]))
        paths.sort(key=lambda p: (len(p), -support(p)))
        return paths[:wanted]

    def _path_chunks(self, paths: List[List[str]], present: set) -> List[int]:
        # The chunks that state the steps of the paths: a greedy cover, preferring chunks that state
        # several steps at once (typically a chain text).
        step_chunks = {}
        for path in paths:
            for a, b in zip(path, path[1:]):
                ids = {c for (s, t, _), chunks in self.relation_chunks.items() if s == a and t == b for c in chunks}
                step_chunks[(a, b)] = {c for c in ids if c in self.docs}
        uncovered, chosen = set(step_chunks), []
        while uncovered and len(chosen) < self.valves.CHAIN_EXTRA_K:
            best = max(
                {c for step in uncovered for c in step_chunks[step]},
                key=lambda c: (sum(c in step_chunks[s] for s in uncovered), c in present),
                default=None,
            )
            if best is None:
                break
            chosen.append(best)
            uncovered = {s for s in uncovered if best not in step_chunks[s]}
        return chosen

    def _grade(self, query: str, candidates: List[tuple], path_ids: frozenset = frozenset()) -> List[tuple]:
        # reranking.py-style 0-10 LLM scoring combined with reliable_rag's
        # relevance grading: unsupported chunks are dropped before generation.
        docs = [self.docs[chunk_id] for chunk_id, _ in candidates]
        grades = self.grader.batch(
            [
                GRADE_PROMPT.format(
                    question=query, title=doc.metadata["title"], content=doc.page_content
                )
                for doc in docs
            ],
            config={"max_concurrency": 8},
            return_exceptions=True,
        )
        kept, steps = [], []
        for (chunk_id, fused), doc, grade in zip(candidates, docs, grades):
            if isinstance(grade, Exception):
                print(f"[capital_rag] Grading failed for chunk {chunk_id}: {grade}")
                continue
            if grade.supports_answer and grade.relevance >= self.valves.MIN_RELEVANCE:
                kept.append((doc, grade, fused))
            elif (chunk_id in path_ids or grade.states_step) and grade.relevance >= self.valves.CHAIN_MIN_RELEVANCE:
                # A step of the asked chain (from the graph's causal paths or as judged by the grader):
                # relevant as a link even if it alone does not answer, so a chain can be built from several
                # sources instead of the one block that covers the most.
                steps.append((doc, grade, fused))
        kept.sort(key=lambda hit: (hit[1].relevance, hit[2]), reverse=True)
        steps.sort(key=lambda hit: hit[1].relevance, reverse=True)
        return kept[: self.valves.TOP_K] + steps[: self.valves.CHAIN_TOP_K]

    def _explore(
        self, query: str, candidates: List[tuple], answered: bool = False, exclude: set = frozenset()
    ) -> List[tuple]:
        # Explorer mode: the concepts of the closest chunks (and graph neighbours
        # of those) are topics the index covers. Without an answer they are
        # offered instead; after an answer as further topics, with the chunks the
        # answer used (exclude) not offered again as excerpts.
        # Returns [(label, question)], each question verified by the grader
        # against the excerpt it was written from.
        pool = [chunk_id for chunk_id, _ in candidates]
        if not pool:
            # Nothing passed MIN_SIMILARITY or BM25: take the nearest chunks anyway.
            for doc, _ in self.vectorstore.similarity_search_with_score(
                query, k=self.valves.FETCH_K
            ):
                if doc.metadata["chunk_id"] not in pool:
                    pool.append(doc.metadata["chunk_id"])

        scores = Counter()
        for rank, chunk_id in enumerate(pool):
            for concept in self.docs[chunk_id].metadata.get("concepts", []):
                scores[concept] += 1 / (rank + 1)
        for concept, score in scores.most_common(5):
            for neighbor, _ in self.concept_neighbors.get(concept, Counter()).most_common(5):
                scores[neighbor] += 0.3 * score

        topics, used = [], set(exclude)
        for concept, _ in scores.most_common():
            if len(topics) >= 3 * self.valves.EXPLORE_SUGGESTIONS:
                break
            if len(self.concept_chunks.get(concept, [])) < self.valves.EXPLORE_MIN_CHUNKS:
                continue
            # Excerpt: a nearby chunk on this concept, else its summary, else any chunk.
            options = (
                [c for c in pool if concept in self.docs[c].metadata.get("concepts", [])]
                + self.concept_summaries.get(concept, [])
                + self.concept_chunks[concept]
            )
            chunk_id = next((c for c in options if c not in used), None)
            if chunk_id is not None:
                used.add(chunk_id)
                topics.append((concept, self.docs[chunk_id]))
        if not topics:
            return []

        listing = "\n\n".join(
            f"{i}. {concept} - {doc.metadata['title']}\n{doc.page_content[:600]}"
            for i, (concept, doc) in enumerate(topics, start=1)
        )
        situation, goal = EXPLORE_ANSWERED if answered else EXPLORE_NO_ANSWER
        picked = self.explorer.invoke(
            EXPLORE_PROMPT.format(
                situation=situation,
                goal=goal,
                n=self.valves.EXPLORE_SUGGESTIONS,
                question=query,
                topics=listing,
            )
        ).suggestions
        seen, proposals = set(), []
        for s in picked:
            if 1 <= s.topic <= len(topics) and s.topic not in seen and s.question.strip():
                if same_question(s.question, query, self.bm25.idf):
                    print(f"[capital_rag] Explorer: skip rephrased question {s.question!r}")
                    continue
                seen.add(s.topic)
                proposals.append((s, topics[s.topic - 1][1]))
        proposals = proposals[: self.valves.EXPLORE_SUGGESTIONS]

        # Only suggest questions the pipeline can answer: grade each question
        # against its excerpt with the same bar as normal retrieval.
        grades = self.grader.batch(
            [
                GRADE_PROMPT.format(
                    question=s.question, title=doc.metadata["title"], content=doc.page_content
                )
                for s, doc in proposals
            ],
            config={"max_concurrency": 8},
            return_exceptions=True,
        )
        return [
            (s.label, s.question)
            for (s, _), grade in zip(proposals, grades)
            if not isinstance(grade, Exception)
            and grade.supports_answer
            and grade.relevance >= self.valves.MIN_RELEVANCE
        ]

    def _remember(self, query: str, candidates: List[tuple], exclude: set, suggestions) -> None:
        # suggestions None = not computed yet; _follow_ups computes them lazily,
        # in the background task after the answer is already shown.
        self.recent.pop(query, None)
        self.recent[query] = {
            "candidates": candidates,
            "exclude": exclude,
            "suggestions": suggestions,
        }
        while len(self.recent) > 50:
            self.recent.popitem(last=False)

    def _follow_ups(self, prompt: str) -> str:
        # Answers Open WebUI's follow-up task: it runs after every answer and
        # stores the result as clickable follow-up chips. The prompt contains the
        # chat history; the remembered question found latest in it is the one
        # just answered.
        query, entry, pos = None, None, -1
        for q, e in self.recent.items():
            p = prompt.rfind(q)
            if p > pos:
                query, entry, pos = q, e, p
        if entry is None:
            return json.dumps({"follow_ups": []})
        if entry["suggestions"] is None:
            entry["suggestions"] = self._explore(
                query, entry["candidates"], answered=True, exclude=entry["exclude"]
            )
        return json.dumps(
            {"follow_ups": [q for _, q in entry["suggestions"]]}, ensure_ascii=False
        )

    def _unsupported_claims(self, context: str, blocks: List[dict], answer: str) -> tuple:
        # reliable_rag-style hallucination check, per claim: a claim only counts
        # as supported if the checker says so AND (for facts) its quote really is
        # in the context. Meta claims ("the sources don't explain why") need no quote.
        # Also returns whether the answer has any fact claim at all - an answer of
        # only meta claims ("the sources don't cover this") is a refusal - and the
        # answer sentences the problems were found in.
        check = self.checker.invoke(CHECK_PROMPT.format(context=context, answer=answer))
        has_facts = any(claim.kind == "fact" for claim in check.claims)
        unsupported, sentences = [], []
        for claim in check.claims:
            has_evidence = claim.kind == "meta" or quote_in_context(claim.quote, context)
            if claim.supported and claim.same_cause and not has_evidence and joins_sentences(claim.quote, context):
                problem = (
                    "merges separate statements from the sources into one sentence: split it "
                    "into separate sentences, one per source statement, without linking them"
                )
            elif not (claim.supported and has_evidence and claim.same_cause):
                problem = "not stated in the sources"
            elif claim.kind == "fact" and (videos := required_videos(claim.quote, blocks)):
                if names_video(paragraph_of(answer, claim.answer_sentence), videos):
                    continue
                titles = " / ".join(f'"{title}"' for title, _ in videos)
                problem = f"missing source attribution: name the source {titles} in this paragraph"
            else:
                continue
            print(f"[capital_rag] {problem}: {claim.claim!r} (quote: {claim.quote!r})")
            unsupported.append(f"{claim.claim} ({problem})")
            sentences.append(claim.answer_sentence)
        return unsupported, has_facts, sentences

    def _grounded_answer(self, question: str, context: str, blocks: List[dict]) -> Iterator[tuple]:
        # Yields ("status", short, detail) progress steps and finally
        # ("answer", text or None, has_facts).
        yield ("status", "Formuliere Antwort nur aus den Quellen …", None)
        answer = self.answer_llm.invoke(
            ANSWER_PROMPT.format(context=context, question=question)
        ).content
        for attempt in range(self.valves.MAX_REVISIONS + 1):
            yield ("status", "Prüfe jede Aussage gegen die Quellen …", None)
            unsupported, has_facts, sentences = self._unsupported_claims(context, blocks, answer)
            if not unsupported:
                yield ("status", "Alle Aussagen sind durch die Quellen belegt ✓", None)
                yield ("answer", answer, has_facts)
                return
            yield (
                "status",
                f"{len(unsupported)} Aussage(n) nicht sauber belegt",
                f"{len(unsupported)} Aussage(n) nicht sauber belegt:\n"
                + "\n".join(f"  - {c}" for c in unsupported),
            )
            if attempt == self.valves.MAX_REVISIONS:
                break
            yield ("status", "Überarbeite die Antwort …", None)
            answer = self.reviser.invoke(
                REVISE_PROMPT.format(
                    context=context,
                    question=question,
                    answer=answer,
                    unsupported="\n".join(f"- {c}" for c in unsupported),
                )
            ).content
            # The prompt labels the draft "Answer:"; the model sometimes echoes it.
            answer = re.sub(r"^\s*\**(answer|antwort)\**\s*:\s*\**\s*", "", answer, flags=re.IGNORECASE)
        # Last resort instead of dropping the whole answer: cut the flagged
        # sentences and check what is left once more, with the same bar.
        trimmed = drop_sentences(answer, sentences)
        if trimmed and trimmed != answer:
            yield ("status", "Entferne nicht belegte Sätze und prüfe erneut …", None)
            unsupported, has_facts, _ = self._unsupported_claims(context, blocks, trimmed)
            if not unsupported and has_facts:
                yield ("status", "Verbleibende Aussagen sind belegt ✓", None)
                yield ("answer", trimmed, has_facts)
                return
        yield ("status", "Keine vollständig belegte Antwort möglich.", None)
        yield ("answer", None, False)

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        if user_message.startswith("### Task:"):
            # Open WebUI background task, not a question: follow-ups come from the
            # explorer, everything else (title, tags) from the plain LLM.
            try:
                if FOLLOW_UP_TASK.match(user_message):
                    yield self._follow_ups(user_message) if self.valves.EXPLORE_MODE else "{}"
                else:
                    yield self.answer_llm.invoke(user_message).content
            except Exception as e:
                print(f"[capital_rag] Task failed: {e}")
                yield "{}"
            return

        if self.vectorstore is None:
            yield (
                f"Der Kapitalmarkt-Index fehlt in {self.valves.INDEX_DIR}. Erstelle ihn mit\n\n"
                "`docker exec -it open-webui-pipelines-capital python /data/ingest_capital_chunks.py`\n\n"
                "und starte danach den Container pipelines-capital neu."
            )
            return

        # Progress is sent as Open WebUI status events: a chunk with an "event"
        # key is forwarded to the chat's event emitter, so each step replaces the
        # previous one in a single line above the answer. Only when streaming -
        # a non-streamed response is joined as text. SHOW_THINKING_LOG also writes
        # the full step log (incl. rejected claims) into a collapsible <think> block.
        streaming = body.get("stream", True)
        log_open = False

        def step(short: str, detail: str = None, done: bool = False) -> list:
            nonlocal log_open
            out = []
            if streaming:
                out.append(
                    {"event": {"type": "status", "data": {"description": short, "done": done}}}
                )
            if self.valves.SHOW_THINKING_LOG:
                if not log_open:
                    out.append("<think>\n")
                    log_open = True
                out.append(f"{detail or short}\n\n")
            return out

        def close_log() -> list:
            return ["</think>\n\n"] if log_open else []

        def find_related(reason: str, exclude: set = frozenset()) -> Generator:
            # Yields status steps; returns the explorer suggestions.
            if not self.valves.EXPLORE_MODE:
                return []
            yield from step(f"{reason} – suche verwandte Themen …")
            try:
                return self._explore(user_message, candidates, exclude=exclude)
            except Exception as e:
                print(f"[capital_rag] Explorer failed: {e}")
                return []

        def related_block(suggestions) -> list:
            if not suggestions:
                return []
            out = [
                "\n\n**Verwandte Themen, zu denen es Material gibt:**\n"
                + "\n".join(f"- **{label}**: {question}" for label, question in suggestions)
            ]
            if streaming:
                # Shown as clickable follow-up chips under the message.
                out.append(
                    {
                        "event": {
                            "type": "chat:message:follow_ups",
                            "data": {"follow_ups": [q for _, q in suggestions]},
                        }
                    }
                )
            return out

        def no_answer(reason: str, exclude: set = frozenset()) -> Iterator:
            suggestions = yield from find_related(reason, exclude)
            self._remember(user_message, candidates, set(exclude), suggestions)
            if suggestions:
                reason = f"{reason} – {len(suggestions)} verwandte Themen gefunden"
            yield from step(reason, done=True)
            yield from close_log()
            yield NO_ANSWER
            yield from related_block(suggestions)

        yield from step("Durchsuche die Quellen …")
        candidates = self._fusion_candidates(user_message)
        yield from step(f"{len(candidates)} Kandidaten gefunden")
        expanded = self._graph_expand(candidates)
        if len(expanded) > len(candidates):
            yield from step(
                f"Konzeptgraph: {len(expanded) - len(candidates)} verwandte Abschnitte ergänzt"
            )
        candidates = expanded
        paths, path_ids = [], frozenset()
        if self.valves.CHAIN_SEARCH and self.concept_vectors is not None:
            yield from step("Suche Wirkungsketten im Konzeptgraphen …")
            try:
                paths = self._causal_paths(user_message)
            except Exception as e:
                print(f"[capital_rag] Chain search failed: {e}")
            if paths:
                present = {chunk_id for chunk_id, _ in candidates}
                path_ids = frozenset(self._path_chunks(paths, present))
                candidates = candidates + [(c, 0.0) for c in path_ids if c not in present]
                yield from step(
                    f"{len(paths)} Wirkungskette(n) mit bis zu {max(len(p) - 1 for p in paths)} Schritten gefunden",
                    "Wirkungsketten:\n" + "\n".join(f"  - {' → '.join(p)}" for p in paths),
                )
        grounded = []
        if candidates:
            yield from step(f"Bewerte die Relevanz von {len(candidates)} Abschnitten …")
            grounded = self._grade(user_message, candidates, path_ids)

        if not grounded:
            yield from no_answer("Keine relevanten Abschnitte gefunden")
            return
        yield from step(
            f"{len(grounded)} relevante Abschnitte ausgewählt",
            f"{len(grounded)} relevante Abschnitte ausgewählt:\n"
            + "\n".join(f"  - {doc.metadata['title']}" for doc, _, _ in grounded),
        )

        blocks = []
        for i, (doc, _, _) in enumerate(grounded, start=1):
            meta = doc.metadata
            videos = "; ".join(sorted({cite_source(s) for s in meta["sources"]}))
            timeless = []
            if meta.get("content_class") == "opinion":
                label = f"OPINION (time-bound, stated in: {videos})"
            else:
                label = f"CONCEPT (from: {videos})"
            block = f"[{i}] {meta['title']} - {label}\n{doc.page_content}"
            for rel in meta.get("relations", []):
                rel_label = "TIMELESS" if rel.get("relation_class") == "timeless" else "TIME-BOUND"
                block += (
                    f"\nRelation [{rel_label}]: {rel['source']} -> {rel['target']} "
                    f"({rel['direction']}, {rel['time_horizon']}): {rel['mechanism']}"
                )
                if rel.get("conditions"):
                    block += f" Conditions: {rel['conditions']}"
                if rel_label == "TIMELESS":
                    timeless.append(rel["mechanism"])
            if meta.get("conditions"):
                block += f"\nConditions: {meta['conditions']}"
            blocks.append(
                {
                    "text": block,
                    "opinion": meta.get("content_class") == "opinion",
                    "timeless": timeless,
                    "videos": sorted({parse_video(s) for s in meta["sources"]}),
                }
            )
        context = "\n\n".join(b["text"] for b in blocks)
        # Causal paths whose every step is stated by a selected block, with those block numbers.
        path_lines = []
        for path in paths:
            refs = []
            for a, b in zip(path, path[1:]):
                nums = [
                    i for i, (doc, _, _) in enumerate(grounded, start=1)
                    if any(r["source"] == a and r["target"] == b for r in doc.metadata.get("relations", []))
                ]
                refs.append(nums)
            if all(refs):
                line = path[0] + "".join(
                    f" -> {b} [{', '.join(map(str, nums))}]" for b, nums in zip(path[1:], refs)
                )
                path_lines.append(f"- {line}")
        if path_lines:
            context = (
                "Causal paths (relations found across the sources; [n] = block that states the step):\n"
                + "\n".join(path_lines) + "\n\n" + context
            )

        # None = follow-ups are computed later by the follow-up task (_follow_ups).
        suggestions = None
        if self.valves.CHECK_ANSWER_GROUNDING:
            # Never show an unchecked answer: if the check itself fails, refuse.
            answer, has_facts = None, False
            try:
                for kind, value, detail in self._grounded_answer(user_message, context, blocks):
                    if kind == "status":
                        yield from step(value, detail)
                    else:
                        answer, has_facts = value, detail
            except Exception as e:
                print(f"[capital_rag] Grounding check failed: {e}")
                yield from step("Die Prüfung ist fehlgeschlagen")
            if answer is None:
                # The answer failed on these chunks, so don't offer them again.
                failed = {doc.metadata["chunk_id"] for doc, _, _ in grounded}
                yield from no_answer("Keine belegte Antwort gefunden", failed)
                return
            if not has_facts:
                # The answer only says the sources don't cover the question.
                suggestions = yield from find_related("Die Quellen beantworten die Frage nicht")
            cited = cited_numbers(answer, len(grounded))
            yield from step(f"Antwort geprüft ✓ ({len(cited)} Quellen)", done=True)
            yield from close_log()
            yield answer
        else:
            yield from step(f"Antwort aus {len(grounded)} Quellen (ungeprüft)", done=True)
            yield from close_log()
            answer = ""
            for chunk in self.llm.stream(
                ANSWER_PROMPT.format(context=context, question=user_message)
            ):
                if chunk.content:
                    answer += chunk.content
                    yield chunk.content
            cited = cited_numbers(answer, len(grounded))

        # Only the sources the answer actually cites, under their citation numbers.
        source_lines = []
        for i, (doc, grade, _) in enumerate(grounded, start=1):
            if i not in cited:
                continue
            meta = doc.metadata
            if meta["chunk_type"] == "summary":
                kind = f"Konzept, Zusammenfassung aus {meta['support']} Abschnitten"
            elif meta.get("content_class") == "opinion":
                kind = "Meinung (zeitgebunden)"
            else:
                kind = "Konzept"
            videos = "; ".join(sorted({cite_source(s) for s in meta["sources"]}))
            source_lines.append(
                f"- [{i}] {meta['title']} ({kind}) - {videos}\n"
                f"  Relevanz {grade.relevance}/10: {grade.reason}"
            )
        if source_lines:
            yield "\n\nQuellen:\n" + "\n".join(source_lines)
        yield from related_block(suggestions)
        used = {grounded[i - 1][0].metadata["chunk_id"] for i in cited}
        self._remember(user_message, candidates, used, suggestions)

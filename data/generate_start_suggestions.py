"""
Generates the start-page topic suggestions for the Capital Markets RAG model.

--style chains (default): picks cause -> effect chain texts (build_public_kb.py) across market
domains and lets an LLM write a demanding question about how the trigger propagates, so the start
page shows what the chat is for: connections, not definitions.
--style summaries: picks broad, well-covered topics from summaries.json (timeless concept summaries).
Either way only questions the grader confirms the text answers are kept (same GRADE_PROMPT and
MIN_RELEVANCE as the pipeline). Writes start_suggestions.json next to the index;
apply_start_suggestions.py puts them on the model in Open WebUI.

Run inside the pipelines-capital container (index at /data, pipeline at /app/pipelines):
    docker exec open-webui-pipelines-capital python /data/generate_start_suggestions.py
    docker exec -w /app/backend open-webui python /data/apply_start_suggestions.py
"""

import argparse
import json
import os
import sys
from itertools import zip_longest
from typing import List

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

sys.path.insert(0, "/app/pipelines")
from capital_rag_pipeline import GRADE_PROMPT, Grade, Pipeline, tokenize  # noqa: E402


class Starter(BaseModel):
    # Lengths are set by the prompt (short topic suggestions vs. long chain questions).
    title: str = Field(..., description="Topic name, German, as the prompt asks.")
    subtitle: str = Field(..., description="Second line shown under the title, German, as the prompt asks.")
    question: str = Field(
        ..., description="The question in German that the text answers, as a user would ask it."
    )


STARTER_PROMPT = """Below is a summary of a topic from a capital-markets knowledge base.
Write a starter question a user could ask to learn about this topic.
Ask only for what the summary actually explains - do not ask about details it does not contain.
Keep the question general and timeless (no dates, no current market calls).

title: the topic name, 2-4 words. subtitle: a short phrase that continues it (shown in a second line).
Everything in German.

Topic: {title}
{content}"""


def pick_topics(summaries: List[dict], count: int) -> List[dict]:
    # Best-covered topics first, alternating between market domains so the
    # start page is not all one area.
    by_domain = {}
    for s in sorted(summaries, key=lambda s: s.get("support", 0), reverse=True):
        domains = s.get("market_domain") or [""]
        by_domain.setdefault(domains[0] if isinstance(domains, list) else domains, []).append(s)
    mixed = [s for row in zip_longest(*by_domain.values()) for s in row if s]
    return mixed[:count]


CHAIN_STARTER_PROMPT = """Below is a text explaining a capital-markets cause -> effect chain.
Write a start-page suggestion that invites the user to explore the chain, not to look up a term.
title: 2-5 words naming trigger and end effect, e.g. "Aktiencrash → Gold".
subtitle: the core question in one line, at most 90 characters, e.g. "Warum fällt Gold, wenn die Aktienmärkte crashen?"
question: a demanding question of 1-3 sentences as a user would ask it: name the trigger, ask how it propagates
step by step and what it means for the end effect (and, if the text covers it, under which conditions the chain
breaks). Ask only for what the text explains. Timeless: no dates, no current market calls.
Everything in German.

Topic: {title}
{content}"""


def chain_topics(index_dir: str, count: int) -> List[dict]:
    # Cause -> effect chain texts (build_public_kb.py), alternating between market domains.
    with open(os.path.join(index_dir, "extracted_v2_all.json"), encoding="utf-8") as f:
        chunks = [c for c in json.load(f) if c["sources"][0].startswith("Mjoelnir-Wirkungskette")]
    by_domain = {}
    for c in chunks:
        by_domain.setdefault((c.get("market_domain") or [""])[0], []).append(c)
    mixed = [c for row in zip_longest(*by_domain.values()) for c in row if c]
    return mixed[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", default="/data/faiss_public_index")
    parser.add_argument("--count", type=int, default=12, help="Suggestions to keep.")
    parser.add_argument("--model", default="gpt-4.1-mini")
    # summaries: one topic explanation each; chains: multi-step cause -> effect questions.
    parser.add_argument("--style", choices=["summaries", "chains"], default="chains")
    args = parser.parse_args()

    # Twice as many candidates as needed: some fail the grader or repeat a topic.
    if args.style == "chains":
        topics, prompt = chain_topics(args.index_dir, 2 * args.count), CHAIN_STARTER_PROMPT
    else:
        with open(os.path.join(args.index_dir, "summaries.json"), encoding="utf-8") as f:
            topics, prompt = pick_topics(json.load(f), 2 * args.count), STARTER_PROMPT

    llm = ChatOpenAI(model=args.model, temperature=0)
    starters = llm.with_structured_output(Starter).batch(
        [prompt.format(title=s["title"], content=s["content"]) for s in topics],
        config={"max_concurrency": 8},
        return_exceptions=True,
    )
    pairs = [(s, st) for s, st in zip(topics, starters) if not isinstance(st, Exception)]
    grades = llm.with_structured_output(Grade).batch(
        [
            GRADE_PROMPT.format(question=st.question, title=s["title"], content=s["content"])
            for s, st in pairs
        ],
        config={"max_concurrency": 8},
        return_exceptions=True,
    )

    min_relevance = Pipeline.Valves().MIN_RELEVANCE
    suggestions, seen = [], set()
    for (s, st), grade in zip(pairs, grades):
        if isinstance(grade, Exception) or not grade.supports_answer or grade.relevance < min_relevance:
            print(f"skip: {st.question} ({'error' if isinstance(grade, Exception) else grade.reason})")
            continue
        # "Diversifikation in der Anlagestrategie" / "Diversifikation im
        # Portfoliomanagement" are one topic: compare the first content word
        # (for chains the whole title: "Zinsen → Banken" and "Zinsen → Immobilien" differ).
        words = tokenize(st.title) or [st.title.lower()]
        key = " ".join(words) if args.style == "chains" else words[0]
        if key in seen:
            continue
        seen.add(key)
        # Open WebUI's suggestion format: title = [bold line, second line].
        suggestions.append({"title": [st.title, st.subtitle], "content": st.question})
        if len(suggestions) == args.count:
            break

    out_path = os.path.join(args.index_dir, "start_suggestions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(suggestions, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(suggestions)} suggestions to {out_path}")
    for s in suggestions:
        print(f"  - {s['title'][0]} ({s['title'][1]}): {s['content']}")


if __name__ == "__main__":
    main()

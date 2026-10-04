"""
Generates the start-page topic suggestions for the Capital Markets RAG model.

Picks broad, well-covered topics from summaries.json (timeless concept summaries,
each built from several chunks), lets an LLM write one starter question per
topic, and keeps only questions the grader confirms the summary answers (same
GRADE_PROMPT and MIN_RELEVANCE as the pipeline). Writes start_suggestions.json
next to the index; apply_start_suggestions.py puts them on the model in Open WebUI.

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
    title: str = Field(..., description="Topic name, 2-4 words, German.")
    subtitle: str = Field(..., description="Short phrase continuing the title, 3-6 words, German.")
    question: str = Field(
        ..., description="One question in German that the summary answers, as a user would ask it."
    )


STARTER_PROMPT = """Below is a summary of a topic from videos by a capital-markets creator.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", default="/data/faiss_capital_index")
    parser.add_argument("--count", type=int, default=12, help="Suggestions to keep.")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    with open(os.path.join(args.index_dir, "summaries.json"), encoding="utf-8") as f:
        summaries = json.load(f)
    # Twice as many candidates as needed: some fail the grader or repeat a topic.
    topics = pick_topics(summaries, 2 * args.count)

    llm = ChatOpenAI(model=args.model, temperature=0)
    starters = llm.with_structured_output(Starter).batch(
        [STARTER_PROMPT.format(title=s["title"], content=s["content"]) for s in topics],
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
        # Portfoliomanagement" are one topic: compare the first content word.
        key = (tokenize(st.title) or [st.title.lower()])[0]
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

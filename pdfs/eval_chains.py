"""
Measures how well the Capital Markets RAG pipeline explains multi-step cause -> effect chains.

Sends a fixed set of chain questions to the pipeline (through the pipelines server's OpenAI-compatible
API) and lets an LLM judge each answer: number of consecutive causal steps, whether the chain reaches
from the trigger to the final effect, and whether it was a refusal. Writes the answers and scores to
a JSON file, so runs before and after a change can be compared.

Run inside the capital pipelines container:
    docker exec open-webui-pipelines-capital python /data/eval_chains.py --out /data/eval_before.json
"""

import argparse
import json
import os
import time
import urllib.request

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

QUESTIONS = [
    "Warum kann bei einem starken Aktiencrash auch der Goldpreis kurzfristig fallen?",
    "Wie können steigende Leitzinsen über Kursverluste bei Anleihen eine Bankenkrise auslösen?",
    "Was passiert mit Schwellenländern, wenn der US-Dollar stark aufwertet?",
    "Wie kann die Auflösung von Carry-Trades in Yen weltweit die Aktienmärkte unter Druck setzen?",
    "Warum fallen Wachstumsaktien stärker als Value-Aktien, wenn die Inflation überraschend steigt?",
    "Wie überträgt sich ein Ölpreisschock auf Inflation, Zinsen und Aktienbewertungen?",
    "Wie kann ein Anstieg der Volatilität über risikobasierte Anlagestrategien zu weiteren Verkäufen führen?",
    "Wie wirkt quantitative Lockerung auf Anleiherenditen, Aktienkurse und Wechselkurse?",
    "Warum können steigende Kreditaufschläge von Unternehmensanleihen eine Rezession ankündigen?",
    "Wie kann eine inverse Zinsstrukturkurve über die Kreditvergabe der Banken die Konjunktur bremsen?",
    "Was passiert bei einer Liquiditätskrise am Geldmarkt mit Anleihen und Aktien?",
    "Wie entsteht ein Short Squeeze und welche Folgen hat er für den Aktienkurs?",
    "Wie wirkt eine Abwertung des Euro auf exportorientierte Unternehmen und auf die Inflation im Euroraum?",
    "Warum kann Deflation trotz sinkender Preise für Unternehmen und Kreditnehmer schädlich sein?",
    "Wie kann eine kreditfinanzierte Spekulationsblase entstehen und wie läuft ihr Platzen ab?",
]

JUDGE_PROMPT = """You assess an answer to a question about a capital-markets cause -> effect mechanism.

steps: the number of distinct consecutive cause -> effect steps the answer explains as a chain (A -> B -> C
counts 2). Separate facts that are not linked into a chain do not count.
complete: 2 = the chain reaches from the trigger in the question to the effect asked about, 1 = partly,
0 = no chain.
refused: true if the answer says it cannot answer or only suggests other topics.
depth: 1-5, how well the answer explains the mechanism, its conditions and counter-effects.

Question: {question}

Answer:
{answer}"""


class Judgement(BaseModel):
    steps: int = Field(..., ge=0)
    complete: int = Field(..., ge=0, le=2)
    refused: bool
    depth: int = Field(..., ge=1, le=5)
    note: str = Field(..., description="One sentence on what is missing, German")


def ask(question: str) -> str:
    body = json.dumps({
        "model": "capital_rag_pipeline",
        "messages": [{"role": "user", "content": question}],
        "stream": False,
    }).encode("utf-8")
    request = urllib.request.Request(
        "http://localhost:9099/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['PIPELINES_API_KEY']}"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--judge_model", default="gpt-4.1-mini")
    args = parser.parse_args()
    judge = ChatOpenAI(model=args.judge_model, temperature=0).with_structured_output(Judgement)

    results = []
    for question in QUESTIONS:
        start = time.time()
        try:
            answer = ask(question)
        except Exception as e:
            answer = f"ERROR: {e}"
        seconds = round(time.time() - start)
        text = answer.split("\n\nQuellen:")[0]
        verdict = judge.invoke(JUDGE_PROMPT.format(question=question, answer=text))
        results.append({"question": question, "seconds": seconds, **verdict.model_dump(), "answer": answer})
        print(f"steps {verdict.steps} complete {verdict.complete} depth {verdict.depth} "
              f"refused {verdict.refused} {seconds:4d}s  {question[:60]}", flush=True)
        json.dump(results, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    n = len(results)
    print(f"\nmean steps {sum(r['steps'] for r in results) / n:.1f} | complete chains "
          f"{sum(r['complete'] == 2 for r in results)}/{n} | mean depth {sum(r['depth'] for r in results) / n:.1f} "
          f"| refused {sum(r['refused'] for r in results)}/{n} | mean time {sum(r['seconds'] for r in results) / n:.0f}s")


if __name__ == "__main__":
    main()

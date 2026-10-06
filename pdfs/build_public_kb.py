"""
Builds the public knowledge base for the Capital Markets RAG pipeline from sources it may use:

1. German Wikipedia articles (CC BY-SA 4.0): one article per topic, split into sections. The section
   text is kept verbatim as chunk content; an LLM only adds the retrieval metadata (title, concepts,
   relations, embedding text, questions). Every article is listed with author/licence links in
   lizenzen.html, as CC BY-SA requires.
2. Own explanatory texts: an LLM writes one timeless explanation per topic from general knowledge,
   without market calls or forecasts.

Output is the chunk format of extracted_v2_all.json, so ingest_capital_chunks.py builds the index
unchanged. Run inside the capital pipelines container (/data = ./pdfs):

    docker exec open-webui-pipelines-capital python /data/build_public_kb.py --stage fetch
    # check /data/faiss_public_index/wiki_resolved.json (topic -> article), then:
    docker exec open-webui-pipelines-capital python /data/build_public_kb.py --stage build
    docker exec open-webui-pipelines-capital python /data/ingest_capital_chunks.py --index_dir /data/faiss_public_index
"""

import argparse
import html
import json
import os
import re
import time
import urllib.parse
import urllib.request
from typing import List, Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

OUT_DIR = "/data/faiss_public_index"
WIKI_API = "https://de.wikipedia.org/w/api.php"
# Wikimedia asks API clients to identify themselves with a contact.
USER_AGENT = "MjoelnirKB/1.0 (https://mjoelnir.ai/impressum)"

# Neutral topic list: general capital-markets knowledge, written for this project.
TOPICS = [
    # Instruments and markets
    "Aktie", "Anleihe", "Investmentfonds", "Exchange-traded Fund", "Indexfonds", "Aktienindex",
    "DAX", "S&P 500", "MSCI World", "Börse", "Orderbuch", "Limit-Order", "Market Maker",
    "Geld-Brief-Spanne", "Liquidität (Finanzmarkt)", "Depot (Bank)", "Broker (Finanzwesen)",
    "Einlagensicherung", "Derivat (Wirtschaft)", "Option (Wirtschaft)", "Future (Finanzwirtschaft)",
    "Optionsschein", "Zertifikat (Finanzprodukt)", "Hedging", "Leerverkauf", "Short Squeeze",
    "Leverage-Effekt", "Wertpapierkredit", "Stop-Loss-Order",
    # Valuation and analysis
    "Dividende", "Dividendenrendite", "Kurs-Gewinn-Verhältnis", "Kurs-Buchwert-Verhältnis",
    "Eigenkapitalrentabilität", "Free Cashflow", "Unternehmensbewertung", "Discounted-Cashflow-Verfahren",
    "Fundamentalanalyse", "Technische Analyse", "Gleitender Durchschnitt", "Unterstützung und Widerstand",
    "Trendlinie", "Relative Strength Index", "MACD", "Bollinger-Bänder", "Candlestick-Chart",
    # Risk and portfolio
    "Volatilität", "Beta-Faktor", "Sharpe-Quotient", "Maximum Drawdown", "Value at Risk",
    "Diversifikation (Wirtschaft)", "Portfoliotheorie", "Effizienzkurve", "Capital Asset Pricing Model",
    "Arbitragepreistheorie", "Effizienzmarkthypothese", "Random-Walk-Theorie", "Risikoprämie",
    "Aktienrisikoprämie", "Asset Allocation", "Rebalancing", "Cost-Average-Effekt", "Zinseszins",
    "Value Investing", "Wachstumsaktie", "Momentum (Finanzmarkt)", "Market Timing", "Buy and Hold",
    # Behavioural finance
    "Behavioral Finance", "Neue Erwartungstheorie", "Verlustaversion", "Herdenverhalten",
    "Dispositionseffekt", "Ankereffekt", "Bestätigungsfehler",
    # Rates, money and macro
    "Inflation", "Deflation", "Stagflation", "Leitzins", "Geldpolitik", "Europäische Zentralbank",
    "Federal Reserve System", "Quantitative Lockerung", "Zinsstrukturkurve", "Inverse Zinsstrukturkurve",
    "Effektivzins", "Duration (Finanzwirtschaft)", "Bonität", "Rating", "Ratingagentur", "Ausfallrisiko",
    "Staatsanleihe", "Unternehmensanleihe", "Hochzinsanleihe", "Inflationsindexierte Anleihe",
    "Konjunktur", "Konjunkturzyklus", "Bruttoinlandsprodukt", "Rezession", "Einkaufsmanagerindex",
    "Wechselkurs", "Devisenmarkt", "Carry Trade",
    # Asset classes
    "Gold als Kapitalanlage", "Rohstoff", "Rohöl", "Offener Immobilienfonds", "Real Estate Investment Trust",
    "Kryptowährung", "Bitcoin", "Private Equity", "Risikokapital",
    # Market history and seasonality
    "Spekulationsblase", "Dotcom-Blase", "Finanzkrise ab 2007", "Börsenkrach", "Saisonalität (Börse)",
    "Januareffekt", "Präsidentschaftszyklus",
    # Taxes and saving (Germany)
    "Abgeltungsteuer", "Sparerpauschbetrag", "Vorabpauschale", "Altersvorsorge", "Kapitallebensversicherung",
]

# Topics whose title or search hit is the wrong article (checked by hand in wiki_resolved.json).
# None = no fitting article; the topic only gets an own explanatory text.
WIKI_OVERRIDES = {
    "Depot (Bank)": "Wertpapierdepot",
    "Future (Finanzwirtschaft)": "Terminkontrakt",
    "Wachstumsaktie": "Wachstumswert",
    "Duration (Finanzwirtschaft)": "Duration",
    "Unterstützung und Widerstand": None,
    "Trendlinie": None,
    "Saisonalität (Börse)": None,
    "Präsidentschaftszyklus": None,
}

SKIP_SECTIONS = {
    "literatur", "weblinks", "einzelnachweise", "siehe auch", "anmerkungen", "quellen", "fußnoten",
    "rezeption", "filme", "trivia", "belege", "weiterführende literatur", "nachweise",
}
MAX_CHUNKS_PER_ARTICLE = 8
TARGET_CHARS, MAX_CHARS, MIN_CHARS = 1800, 2600, 300

DOMAINS = Literal["equities", "macro", "credit", "commodities", "rates", "fx", "banking", "derivatives", "crypto"]


# ---------------------------------------------------------------------------
# Stage 1: Wikipedia
# ---------------------------------------------------------------------------


def wiki(params: dict) -> dict:
    query = urllib.parse.urlencode({**params, "format": "json", "formatversion": 2})
    request = urllib.request.Request(f"{WIKI_API}?{query}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        time.sleep(0.3)  # stay well below the API's rate limits
        return json.load(response)


def resolve(topic: str):
    # Manual override, then the exact title (following redirects), otherwise the best search hit.
    if topic in WIKI_OVERRIDES:
        return WIKI_OVERRIDES[topic]
    page = wiki({"action": "query", "titles": topic, "redirects": 1})["query"]["pages"][0]
    if page.get("missing") is None and "invalid" not in page:
        return page["title"]
    hits = wiki({"action": "query", "list": "search", "srsearch": topic, "srlimit": 1})["query"]["search"]
    return hits[0]["title"] if hits else None


def fetch_article(title: str) -> dict:
    page = wiki({
        "action": "query", "titles": title, "redirects": 1, "prop": "extracts|info|revisions",
        "explaintext": 1, "exsectionformat": "wiki", "inprop": "url", "rvprop": "ids",
    })["query"]["pages"][0]
    return {
        "title": page["title"],
        "url": page["fullurl"],
        "revid": page["revisions"][0]["revid"],
        "text": clean_formulas(page.get("extract", "")),
    }


def clean_formulas(text: str) -> str:
    # The plain-text extract renders math as one symbol per line followed by "{\displaystyle ...}".
    # Both become a single "[Formel]" marker.
    out, i = [], 0
    while (start := text.find("{\\displaystyle", i)) != -1:
        depth, end = 0, start
        for end in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[end], 0)
            if depth == 0:
                break
        out.append(text[i:start] + " [Formel] ")
        i = end + 1
    text = "".join(out) + text[i:]
    lines, kept, run = text.splitlines(), [], []
    for line in lines + ["<end>"]:
        if len(line.strip()) <= 3 and line != "<end>":
            run.append(line)
            continue
        # A run of four or more symbol lines is a rendered formula; shorter runs are normal blank lines.
        kept += [""] if sum(1 for r in run if r.strip()) >= 4 else run
        run = []
        if line != "<end>":
            kept.append(line)
    text = "\n".join(kept)
    text = re.sub(r"(\s*\[Formel\]\s*)+", " [Formel] ", text)
    return re.sub(r"[ \t]+", " ", text)


def split_long(para: str) -> List[str]:
    # Paragraphs above MAX_CHARS are cut at sentence ends.
    if len(para) <= MAX_CHARS:
        return [para]
    parts, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", para):
        if len(current) + len(sentence) > TARGET_CHARS and current:
            parts.append(current)
            current = ""
        current += sentence + " "
    return parts + [current] if current.strip() else parts


def split_sections(text: str) -> List[tuple]:
    # "== Heading ==" / "=== Sub ===" lines start a section; the intro has no heading.
    sections, heading, lines, skip_level = [], "Einleitung", [], None
    for line in text.splitlines():
        m = re.match(r"^(=+)\s*(.*?)\s*=+$", line)
        if m:
            level = len(m[1])
            if skip_level and level > skip_level:
                continue
            if lines and not skip_level:
                sections.append((heading, "\n".join(lines).strip()))
            heading, lines = m[2], []
            skip_level = level if m[2].lower() in SKIP_SECTIONS else None
        elif not skip_level:
            lines.append(line)
    if lines and not skip_level:
        sections.append((heading, "\n".join(lines).strip()))
    return [(h, t) for h, t in sections if t]


def chunk_article(sections: List[tuple]) -> List[dict]:
    # Consecutive sections are merged up to TARGET_CHARS; long sections are cut at paragraphs.
    chunks, current, heads = [], "", []

    def flush():
        nonlocal current, heads
        if len(current) >= MIN_CHARS:
            chunks.append({"sections": heads, "text": current.strip()})
        current, heads = "", []

    for heading, text in sections:
        for para in [p for block in re.split(r"\n\s*\n", text) for p in split_long(block)]:
            if len(current) + len(para) > MAX_CHARS and current:
                flush()
            if heading not in heads:
                heads.append(heading)
            current += para.strip() + "\n\n"
            if len(current) >= TARGET_CHARS:
                flush()
    flush()
    return chunks[:MAX_CHUNKS_PER_ARTICLE]


def stage_fetch():
    resolved, articles = {}, []
    for topic in TOPICS:
        title = resolve(topic)
        resolved[topic] = title
        if not title or any(a["title"] == title for a in articles):
            continue
        article = fetch_article(title)
        article["chunks"] = chunk_article(split_sections(article.pop("text")))
        articles.append(article)
        print(f"{topic:38s} -> {title:45s} {len(article['chunks'])} chunks")
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(resolved, open(f"{OUT_DIR}/wiki_resolved.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(articles, open(f"{OUT_DIR}/wiki_raw.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(articles)} articles, {sum(len(a['chunks']) for a in articles)} chunks")


# ---------------------------------------------------------------------------
# Stage 2: chunk metadata and own texts
# ---------------------------------------------------------------------------


class Relation(BaseModel):
    source: str = Field(..., description="Cause concept, English, Title Case")
    target: str = Field(..., description="Effect concept, English, Title Case")
    direction: Literal["positive", "negative", "ambiguous", "non-linear"]
    mechanism: str = Field(..., description="German sentence explaining how the cause affects the effect")
    conditions: str = Field("", description="German: when the relation holds, empty if not stated")
    time_horizon: Literal["immediate", "short-term", "medium-term", "long-term", "unspecified"]


class Meta(BaseModel):
    title: str = Field(..., description="Short German title of what this text explains")
    concepts: List[str] = Field(..., description="2-6 key concepts, English, Title Case (e.g. 'Interest Rates')")
    relations: List[Relation] = Field(default_factory=list, description="Cause -> effect links stated in the text")
    embedding_text: str = Field(..., description="German summary of the text in 2-3 sentences")
    hypothetical_questions: List[str] = Field(..., description="8 German questions this text answers")
    market_domain: List[DOMAINS]


class OwnText(Meta):
    content: str = Field(
        ...,
        description="German explanation, 1200-1800 characters, with the parts 'Kontext:', 'Wirkungskette:' and "
        "'Bedingungen & Fazit:' separated by blank lines",
    )


META_PROMPT = """Below is a passage from the German Wikipedia article "{article}" (section: {sections}).
Create retrieval metadata for it. Use only what the passage says: relations only for cause -> effect links the
passage states, mechanisms and questions only about its content. Questions in German, as a private investor
would ask them.

{text}"""

OWN_PROMPT = """Write a timeless German explanation of the capital-markets topic "{topic}" for private investors.
- Write in your own words from general financial knowledge. Do not reproduce text from any book, article,
  course or video, and do not name any author, course, company offering or website.
- Explain how it works, what it is used for, cause -> effect mechanisms and their limits.
- No current market assessment, forecast, price target or recommendation; no dates of recent events.
Then create the retrieval metadata for your text (relations only for links your text states)."""


def run_batch(llm, prompts: List[str]) -> list:
    results = llm.batch(prompts, config={"max_concurrency": 8}, return_exceptions=True)
    # One retry for failed calls (rate limits, truncated output).
    retry = [i for i, r in enumerate(results) if isinstance(r, Exception)]
    if retry:
        again = llm.batch([prompts[i] for i in retry], config={"max_concurrency": 4}, return_exceptions=True)
        for i, r in zip(retry, again):
            results[i] = r
    return results


def to_chunk(meta: Meta, content: str, source: str, chunk_id: int) -> dict:
    return {
        "chunk_type": "topic",
        "title": meta.title,
        "concepts": meta.concepts,
        "content": content,
        "relations": [r.model_dump() for r in meta.relations],
        "conditions": "",
        "evidence": "explicit",
        "embedding_text": meta.embedding_text,
        "hypothetical_questions": meta.hypothetical_questions[:10],
        "market_domain": list(meta.market_domain) or ["equities"],
        "new_concepts": [],
        "sources": [source],
        "id": chunk_id,
    }


def lizenzen_html(articles: List[dict], own_topics: List[str]) -> str:
    rows = "\n".join(
        f'\t\t\t<li><a href="{html.escape(a["url"])}" rel="noopener">{html.escape(a["title"])}</a> '
        f'(<a href="{html.escape(a["url"])}?oldid={a["revid"]}" rel="noopener">verwendete Version</a>, '
        f'<a href="{html.escape(a["url"])}?action=history" rel="noopener">Autoren</a>)</li>'
        for a in sorted(articles, key=lambda a: a["title"].lower())
    )
    return LIZENZ_TEMPLATE.replace("{rows}", rows).replace("{n_articles}", str(len(articles))).replace(
        "{n_own}", str(len(own_topics)))


def stage_build(model: str):
    articles = json.load(open(f"{OUT_DIR}/wiki_raw.json", encoding="utf-8"))
    llm_meta = ChatOpenAI(model=model, temperature=0, max_retries=10).with_structured_output(Meta)
    llm_own = ChatOpenAI(model=model, temperature=0.3, max_retries=10).with_structured_output(OwnText)

    wiki_items = [(a, i, c) for a in articles for i, c in enumerate(a["chunks"])]
    print(f"Metadata for {len(wiki_items)} Wikipedia chunks ...")
    metas = run_batch(llm_meta, [
        META_PROMPT.format(article=a["title"], sections=", ".join(c["sections"]), text=c["text"])
        for a, _, c in wiki_items
    ])
    print(f"Own texts for {len(TOPICS)} topics ...")
    owns = run_batch(llm_own, [OWN_PROMPT.format(topic=re.sub(r"\s*\(.*\)$", "", t)) for t in TOPICS])

    chunks = []
    for (article, i, chunk), meta in zip(wiki_items, metas):
        if isinstance(meta, Exception):
            print(f"  skipped {article['title']} #{i}: {meta}")
            continue
        content = f'Aus dem Wikipedia-Artikel „{article["title"]}“ ({", ".join(chunk["sections"])}):\n\n{chunk["text"]}'
        chunks.append(to_chunk(meta, content, f"Wikipedia: {article['title']} #{i}", len(chunks)))
    own_topics = []
    for topic, own in zip(TOPICS, owns):
        if isinstance(own, Exception):
            print(f"  skipped own text {topic}: {own}")
            continue
        name = re.sub(r"\s*\(.*\)$", "", topic)
        own_topics.append(name)
        chunks.append(to_chunk(own, own.content, f"Mjoelnir-Erklärtext: {name} #0", len(chunks)))

    json.dump(chunks, open(f"{OUT_DIR}/extracted_v2_all.json", "w", encoding="utf-8"), ensure_ascii=False)
    attribution = [{k: a[k] for k in ("title", "url", "revid")} for a in articles]
    json.dump(attribution, open(f"{OUT_DIR}/attribution.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(f"{OUT_DIR}/lizenzen.html", "w", encoding="utf-8").write(lizenzen_html(articles, own_topics))
    n_wiki = sum(1 for c in chunks if c["sources"][0].startswith("Wikipedia"))
    print(f"{len(chunks)} chunks ({n_wiki} Wikipedia, {len(chunks) - n_wiki} own texts) -> {OUT_DIR}")


LIZENZ_TEMPLATE = """<!doctype html>
<html lang="de">
<head>
	<meta charset="utf-8" />
	<meta name="viewport" content="width=device-width, initial-scale=1" />
	<title>Quellen und Lizenzen – Mjoelnir</title>
	<style>
		:root { --bg: #ffffff; --fg: #1f2328; --muted: #656d76; --link: #0969da; }
		@media (prefers-color-scheme: dark) { :root { --bg: #171717; --fg: #e6e6e6; --muted: #9a9a9a; --link: #6cb6ff; } }
		body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.6 system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; }
		main { max-width: 44rem; margin: 0 auto; padding: 3rem 16px; overflow-wrap: break-word; }
		h1 { font-size: 1.75rem; margin: 0 0 1.5rem; }
		h2 { font-size: 1.3rem; margin: 2.5rem 0 0.75rem; }
		p, ul { margin: 0 0 1rem; }
		ul { padding-left: 1.2rem; }
		a { color: var(--link); }
		.back { display: inline-block; margin-top: 2rem; color: var(--muted); }
	</style>
</head>
<body>
	<main>
		<h1>Quellen und Lizenzen</h1>
		<p>
			Der KI-Chat beantwortet Fragen ausschließlich auf Grundlage der folgenden Quellen. Die Antworten sind
			allgemeine Informationen und keine Anlageberatung.
		</p>

		<h2>Wikipedia-Artikel</h2>
		<p>
			Verwendet werden Texte aus {n_articles} Artikeln der deutschsprachigen Wikipedia. Die Texte stehen unter der
			Lizenz <a href="https://creativecommons.org/licenses/by-sa/4.0/deed.de" rel="noopener">Creative Commons
			Namensnennung – Weitergabe unter gleichen Bedingungen 4.0 (CC BY-SA 4.0)</a>. Urheber sind die jeweiligen
			Wikipedia-Autoren, die über den Link „Autoren“ einsehbar sind.
		</p>
		<p>
			Änderungen: Die Artikel wurden in Abschnitte zerlegt; Literatur- und Einzelnachweise wurden entfernt. Zu
			jedem Abschnitt wurden automatisch Stichworte, Zusammenfassungen und Beispielfragen erzeugt, und Antworten
			des Chats können Abschnitte zusammenfassen oder umformulieren. Diese aus Wikipedia abgeleiteten Inhalte
			stehen ebenfalls unter CC BY-SA 4.0.
		</p>
		<ul>
{rows}
		</ul>

		<h2>Eigene Erklärtexte</h2>
		<p>
			Zusätzlich enthält die Wissensbasis {n_own} eigene Erklärtexte zu allgemeinen Kapitalmarktthemen. Sie
			wurden für diese Website mit Hilfe eines KI-Sprachmodells erstellt und sind in der Quellenliste des Chats
			als „Mjoelnir-Erklärtext“ gekennzeichnet.
		</p>

		<a class="back" href="/">← Zurück zum Chat</a> · <a class="back" href="/impressum">Impressum</a> · <a class="back" href="/datenschutz">Datenschutz</a>
	</main>
</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", choices=["fetch", "build"], required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()
    stage_fetch() if args.stage == "fetch" else stage_build(args.model)

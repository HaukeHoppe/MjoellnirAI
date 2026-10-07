"""
Builds the public knowledge base for the Capital Markets RAG pipeline from sources it may use:

1. German Wikipedia articles (CC BY-SA 4.0): one article per topic, split into sections. The section
   text is kept verbatim as chunk content; an LLM only adds the retrieval metadata (title, concepts,
   relations, embedding text, questions). Every article is listed with author/licence links in
   lizenzen.html, as CC BY-SA requires.
2. Own explanatory texts: an LLM writes one timeless explanation per topic from general knowledge,
   without market calls or forecasts.
3. Own cause -> effect chains: a stronger LLM explains how a shock propagates through markets step by step
   (trigger, numbered steps, conditions, counter-effects), one relation per step. These carry the
   multi-step links that encyclopedia articles rarely state.
4. Concept normalization: every concept name (chunk concepts and relation ends) is mapped to one canonical
   English name, and rarely used relation ends are linked to a core concept that means the same or is its
   general form, so the concept graph connects chunks from different sources and chains can be followed
   across them.

Unchanged chunks are taken over from the previous build, so a rebuild only pays for new material.

Output is the chunk format of extracted_v2_all.json, so ingest_capital_chunks.py builds the index
unchanged. Run inside the capital pipelines container (/data = ./pdfs):

    docker exec open-webui-pipelines-capital python /data/build_public_kb.py --stage fetch
    # check /data/faiss_public_index/wiki_resolved.json (topic -> article), then:
    docker exec open-webui-pipelines-capital python /data/build_public_kb.py --stage build
    docker exec open-webui-pipelines-capital python /data/build_public_kb.py --stage normalize
    docker exec open-webui-pipelines-capital python /data/ingest_capital_chunks.py --index_dir /data/faiss_public_index
"""

import argparse
import hashlib
import html
import json
import os
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from typing import List, Literal

import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from sklearn.cluster import AgglomerativeClustering

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

# Wikipedia only (no own text): articles on how shocks spread, the links between the topics above.
WIKI_MECHANISMS = [
    "Nachschusspflicht", "Margin Call", "Deleveraging", "Flight to Quality", "Finanzielle Ansteckung",
    "Pensionsgeschäft", "Credit Spread", "Bank Run", "Kreditklemme", "Systemisches Risiko", "Prozyklizität",
    "VIX", "Long-Term Capital Management", "Schwarzer Montag", "Flash Crash", "Vermögenseffekt",
    "Transmissionsmechanismus der Geldpolitik", "Phillips-Kurve", "Taylor-Regel", "Zinsparität",
    "Kaufkraftparität", "Ölkrise", "Schuldendeflation", "Asienkrise", "Eurokrise", "Silicon Valley Bank",
    "Schattenbank", "Verbriefung", "Collateralized Debt Obligation", "Credit Default Swap", "Fallen Angel",
    "Lohn-Preis-Spirale", "Crowding-out", "Zinsänderungsrisiko", "Liquiditätsfalle",
]

# Own cause -> effect chains: how a trigger propagates through markets (written for this project).
CHAIN_TOPICS = [
    # Crashes, leverage and liquidity
    "Aktiencrash, Margin Calls und Notverkäufe von Gold",
    "Deleveraging: Wie Schuldenabbau Kursrückgänge verstärkt",
    "Volatilitätsanstieg und risikobasierte Strategien (Volatility Targeting, Risk Parity)",
    "Flucht in sichere Häfen: Staatsanleihen, US-Dollar, Schweizer Franken und Gold in Krisen",
    "Ansteckung zwischen Anlageklassen in Krisen: steigende Korrelationen",
    "Liquiditätsspirale: Fallende Preise, höhere Sicherheitsabschläge, Zwangsverkäufe",
    "Short Squeeze: Leerverkäufer, Eindeckungskäufe und Kursexplosion",
    "Gamma-Effekte am Optionsmarkt: Absicherungsgeschäfte der Händler verstärken Kursbewegungen",
    "Fondsabflüsse und Notverkäufe bei offenen Fonds",
    "Bank-Run: Vertrauensverlust, Einlagenabzug und Notverkäufe von Vermögenswerten",
    "Hebelprodukte und kreditfinanzierte Wertpapierkäufe als Verstärker von Abwärtsbewegungen",
    "Rebalancing in Crashphasen: Wie Portfolio-Umschichtungen Märkte stützen oder belasten",
    # Rates and banks
    "Steigende Leitzinsen, Kursverluste bei Anleihen und Bankenstress",
    "Zinsanstieg und Bewertung von Wachstumsaktien über den Diskontierungssatz",
    "Inverse Zinsstrukturkurve, Bankmargen und Kreditvergabe",
    "Zinswende und Immobilienpreise über Finanzierungskosten",
    "Steigende Zinsen und hoch verschuldete Unternehmen (Refinanzierungsrisiko)",
    "Kreditaufschläge als Frühindikator: Spreads, Finanzierungsbedingungen, Investitionen",
    "Kreditklemme: Bankverluste, Eigenkapital, eingeschränkte Kreditvergabe und Rezession",
    "Stress am Geldmarkt: knappe Liquidität, Geldmarktzinsen und Eingriffe der Zentralbank",
    "Zinsänderungsrisiko von Anleihen: Duration, Kursverlust und Wiederanlage",
    # Central banks and states
    "Transmissionsmechanismus der Geldpolitik: vom Leitzins zur Inflation",
    "Quantitative Lockerung: Anleihekäufe, Renditen, Portfolioumschichtung und Vermögenspreise",
    "Quantitative Straffung: Bilanzabbau, Liquidität und Risikoprämien",
    "Zinserwartungen und Anleihekurse: Wie Erwartungen Märkte bewegen, bevor die Zentralbank handelt",
    "Glaubwürdigkeit der Zentralbank und Inflationserwartungen",
    "Lohn-Preis-Spirale: Inflation, Lohnforderungen, Kosten und weitere Preissteigerungen",
    "Staatsverschuldung, Defizite und Anleiherenditen",
    "Steigende Staatsanleiherenditen und Verdrängung privater Investitionen",
    "Staatsschuldenkrise: Zinsanstieg, Bankbilanzen und Teufelskreis zwischen Staat und Banken",
    # Currencies and global flows
    "Starker US-Dollar und Schwellenländer mit Schulden in Dollar",
    "Auflösung von Carry-Trades: Yen-Aufwertung, Zwangsverkäufe und globale Marktturbulenzen",
    "Euro-Abwertung, Exportunternehmen und importierte Inflation",
    "Zinsdifferenzen und Wechselkurse: Kapitalströme zwischen Währungsräumen",
    "Rohstoffpreise und Währungen von Rohstoffexporteuren",
    "Kapitalflucht aus Schwellenländern: Abwertung, Inflation und Zinserhöhungen",
    "Dollar-Knappheit in Krisen und Swap-Linien der Zentralbanken",
    # Commodities and inflation
    "Ölpreisschock: Energiepreise, Inflation, Zinsen und Konsum",
    "Ölpreis, Inflationserwartungen und Anleiherenditen",
    "Lieferkettenstörungen, Erzeugerpreise und Verbraucherpreise",
    "Goldpreis, Realzinsen und Opportunitätskosten",
    "Goldpreis und US-Dollar: der Wechselkurseffekt",
    "Stagflation: Angebotsschock, Inflation, schwaches Wachstum und das Dilemma der Zentralbank",
    "Schuldendeflation: fallende Preise, steigende reale Schuldenlast und Investitionszurückhaltung",
    "Energiepreisschock in Europa: Industrieproduktion, Inflation und Euro",
    "Zölle und Handelskonflikte: Preise, Unternehmensgewinne, Wechselkurse und Lieferketten",
    # Equities and valuation
    "Inflation und Value- gegenüber Wachstumsaktien",
    "Gewinnrevisionen, Bewertungsniveaus und Kursreaktionen",
    "Rezessionsangst: Gewinnerwartungen, Risikoprämie, zyklische und defensive Aktien",
    "Konjunkturzyklus und Sektorrotation",
    "Vermögenseffekt: Aktienkurse, Konsum und Konjunktur",
    "Aktienrückkäufe, Verschuldung und Zinsniveau",
    "Hohe Indexkonzentration und passive Fondsströme",
    "Korrelation zwischen Aktien und Anleihen bei hoher Inflation",
    "Politische Unsicherheit, Risikoprämien und Kapitalflüsse",
    # Bubbles and crises
    "Kreditfinanzierte Spekulationsblase: Entstehung, Höhepunkt und Platzen",
    "Immobilienblase, Verbriefung und Bankenkrise",
    "Bewertungsexzesse, Kapitalzuflüsse und Korrektur bei Technologieaktien",
    "Herabstufung der Bonität: Zwangsverkäufe, Finanzierungskosten und Ausfallrisiko",
    # Behaviour and data surprises
    "Herdenverhalten und Momentum: wie Trends sich selbst verstärken",
    "Verlustaversion, Panikverkäufe und Kapitulation am Tiefpunkt",
    "Anlegerstimmung als Kontraindikator und Trendwenden",
    "Einkaufsmanagerindex, Gewinnerwartungen und Aktienmärkte",
    "Arbeitsmarktdaten, Zinserwartungen und Reaktionen von Anleihen und Aktien",
    "Überraschende Inflationsdaten und Marktreaktionen",
    "Konjunkturabschwung, Steuereinnahmen, Staatsdefizit und Anleihemarkt",
    "Kryptowährungen, Liquidität und Risikoappetit",
    "Stablecoins, Reserven und Ansteckung am Kryptomarkt",
    "Saisonale Effekte und ihre möglichen Ursachen",
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
    "Systemisches Risiko": "Systemrisiko",
    "Finanzielle Ansteckung": "Ansteckungseffekt",
    "Flight to Quality": None,
    "Prozyklizität": None,
    "Vermögenseffekt": None,
    "Deleveraging": None,
    "Margin Call": None,
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
    for topic in TOPICS + WIKI_MECHANISMS:
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


class ChainText(Meta):
    content: str = Field(
        ...,
        description="German text, 1800-3000 characters: 'Auslöser:' paragraph, then 'Wirkungskette:' with numbered "
        "steps (one cause -> effect per step, each with its mechanism), then 'Bedingungen:' and "
        "'Gegeneffekte und Grenzen:' paragraphs, separated by blank lines",
    )


CHAIN_PROMPT = """Explain in German, for private investors, how this capital-markets chain works: "{topic}".
- Start from the trigger and follow the propagation step by step through markets, institutions and investor
  behaviour to the final effects. 4-7 numbered steps; each step is one cause -> effect with its mechanism
  (who acts, why, what it does to prices, liquidity or financing).
- Then the conditions under which the chain runs (and when it breaks), and counter-effects or limits.
- Timeless and general: use historical episodes only as examples, no current market assessment, forecast,
  price target or recommendation.
- Write in your own words from general financial knowledge. Do not reproduce text from any book, article,
  course or video, and do not name any author, course, company offering or website.
Metadata: one relation per numbered step, in step order, so the relations form a chain (the target of one step
is the source of the next where the text says so). Concepts and relation ends: short English Title Case names
of the economic quantity or actor (e.g. "Stock Prices", "Margin Calls", "Gold Price"), the same name every time
the same thing is meant. Questions: 8 German questions a private investor would ask about the whole chain or
its steps, several of them "Warum ... wenn ..." / "Was passiert mit ... wenn ..." questions."""


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


# Control characters / replacement characters: gpt-4.1 occasionally emits umlauts as "\x1f" in long
# structured outputs ("f\x1fhren"). Such a text can never be quoted correctly by the answer check.
BROKEN_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f�]")


def is_broken(value) -> bool:
    if isinstance(value, str):
        return bool(BROKEN_CHARS.search(value))
    if isinstance(value, list):
        return any(is_broken(v) for v in value)
    if isinstance(value, dict):
        return any(is_broken(v) for v in value.values())
    if isinstance(value, BaseModel):
        return is_broken(value.model_dump())
    return False


def load_previous() -> dict:
    # Chunks of the previous build by source label, to take over unchanged ones without LLM calls.
    # Broken chunks are left out, so they are generated again.
    path = f"{OUT_DIR}/extracted_v2_all.json"
    if not os.path.exists(path):
        return {}
    return {c["sources"][0]: c for c in json.load(open(path, encoding="utf-8")) if not is_broken(c)}


def generate(llm, label: str, items: List[tuple], previous: dict) -> List[dict]:
    # items: (source label, prompt, fixed content or None for generated texts). A previous chunk is
    # reused when its source matches and, for fixed content (Wikipedia), the content is unchanged.
    todo = [(src, prompt) for src, prompt, content in items
            if not (src in previous and (content is None or previous[src]["content"] == content))]
    print(f"{label}: {len(items)} ({len(items) - len(todo)} from previous build, {len(todo)} new) ...")
    fresh = dict(zip([src for src, _ in todo], run_batch(llm, [p for _, p in todo]))) if todo else {}
    prompts = dict(todo)
    for _ in range(2):
        # Outputs with broken characters are generated again (twice at most), then dropped.
        broken = [src for src, r in fresh.items() if not isinstance(r, Exception) and is_broken(r)]
        if not broken:
            break
        print(f"  {len(broken)} outputs with broken characters, generating again ...")
        fresh.update(zip(broken, run_batch(llm, [prompts[src] for src in broken])))
    for src, r in fresh.items():
        if not isinstance(r, Exception) and is_broken(r):
            fresh[src] = ValueError("broken characters in output")
    out = []
    for src, _, content in items:
        if src not in fresh:
            out.append(dict(previous[src]))
        elif isinstance(fresh[src], Exception):
            print(f"  skipped {src}: {fresh[src]}")
        else:
            result = fresh[src]
            out.append(to_chunk(result, content if content is not None else result.content, src, 0))
    return out


def short_name(topic: str) -> str:
    return re.sub(r"\s*\(.*\)$", "", topic)


def stage_build(model: str, chain_model: str):
    articles = json.load(open(f"{OUT_DIR}/wiki_raw.json", encoding="utf-8"))
    previous = load_previous()
    llm_meta = ChatOpenAI(model=model, temperature=0, max_retries=10, timeout=120).with_structured_output(Meta)
    llm_own = ChatOpenAI(model=model, temperature=0.3, max_retries=10, timeout=120).with_structured_output(OwnText)
    # gpt-4.1 has a low tokens-per-minute limit on lower tiers: many retries, long timeout.
    llm_chain = ChatOpenAI(model=chain_model, temperature=0.3, max_retries=30, timeout=180).with_structured_output(
        ChainText)

    wiki_items = []
    for a in articles:
        for i, c in enumerate(a["chunks"]):
            content = f'Aus dem Wikipedia-Artikel „{a["title"]}“ ({", ".join(c["sections"])}):\n\n{c["text"]}'
            prompt = META_PROMPT.format(article=a["title"], sections=", ".join(c["sections"]), text=c["text"])
            wiki_items.append((f"Wikipedia: {a['title']} #{i}", prompt, content))
    own_items = [(f"Mjoelnir-Erklärtext: {short_name(t)} #0", OWN_PROMPT.format(topic=short_name(t)), None)
                 for t in TOPICS]
    chain_items = [(f"Mjoelnir-Wirkungskette: {short_name(t)} #0", CHAIN_PROMPT.format(topic=t), None)
                   for t in CHAIN_TOPICS]

    chunks = []
    for label, llm, items in (("Wikipedia metadata", llm_meta, wiki_items), ("Own texts", llm_own, own_items),
                              ("Cause -> effect chains", llm_chain, chain_items)):
        for chunk in generate(llm, label, items, previous):
            chunk["id"] = len(chunks)
            chunks.append(chunk)
    own_topics = [c for c in chunks if c["sources"][0].startswith("Mjoelnir-")]

    json.dump(chunks, open(f"{OUT_DIR}/extracted_v2_all.json", "w", encoding="utf-8"), ensure_ascii=False)
    attribution = [{k: a[k] for k in ("title", "url", "revid")} for a in articles]
    json.dump(attribution, open(f"{OUT_DIR}/attribution.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(f"{OUT_DIR}/lizenzen.html", "w", encoding="utf-8").write(lizenzen_html(articles, own_topics))
    kinds = Counter(c["sources"][0].split(":")[0] for c in chunks)
    print(f"{len(chunks)} chunks ({dict(kinds)}) -> {OUT_DIR}")


# ---------------------------------------------------------------------------
# Stage 3: concept normalization
# ---------------------------------------------------------------------------


class Canonical(BaseModel):
    names: List[str] = Field(..., description="Canonical name for each input name, same order and count")


class SynonymGroups(BaseModel):
    groups: List[List[str]] = Field(..., description="Input names grouped; names in a group mean the same thing")
    canonical: List[str] = Field(..., description="One canonical name per group, same order as groups")


class AnchorChoice(BaseModel):
    item: int = Field(..., description="Number of the concept in the list")
    choice: int = Field(..., description="Number of the fitting candidate, 0 if none fits")


class AnchorChoices(BaseModel):
    choices: List[AnchorChoice] = Field(..., description="One choice per listed concept")


CANON_PROMPT = """Map each capital-markets concept name below to a short canonical English name in Title Case:
singular unless the plural is the usual term ("Interest Rates", "Stock Prices"), no articles, no explanations.
Keep the meaning exactly: a change is part of the name only if the input names one ("Zinserhöhung" ->
"Interest Rate Hike", but "Zinsen" -> "Interest Rates").
Return exactly {n} names in the same order.

{names}"""

GROUP_PROMPT = """These capital-markets concept names look similar. Group the ones that mean exactly the same
economic quantity, actor or event. Never group opposites or different directions ("Interest Rate Hike" vs
"Interest Rate Cut"), a quantity with its change ("Inflation" vs "Rising Inflation"), or a part with the whole.
Every name must be in exactly one group (single-name groups are fine). Give one canonical English Title Case
name per group, preferably one of its members.

{names}"""

ANCHOR_PROMPT = """Each numbered capital-markets concept below is used only rarely. It comes with numbered candidate
core concepts that are used often. For each concept, pick the candidate that names the same economic quantity,
actor, instrument or event, or its general form when the concept is only a specific case of it (a region,
market, period, example or wording of it). Examples:
- "Anstieg des Ölpreises" -> "Oil Price Increase"
- "Stock Market Crash US Canada" -> "Stock Market Crash"
- "US Technology Stocks" -> "Technology Stocks"
- "Arbeitslosigkeit USA" -> "Unemployment"
Keep a change as a change and a level as a level: "Oil Price Increase" may not become "Oil Price", "Oil Price"
may not become "Oil Price Increase", "Falling VIX" may not become "Volatility". A specific case may become its
general form, never the other way round ("Risk Return Ratio" is not "Sharpe Ratio"). Never pick an opposite or
another direction ("Interest Rate Cut" is not "Interest Rate Hike"), a different quantity, a related measure or
policy ("Fixed Exchange Rates" is not "Central Bank Intervention"), a cause or an effect of the concept, or only
a part of it. When in doubt, choose 0: a wrong link is worse than none. Answer for every concept.

{items}"""


def relation_key(chunk_id, rel: dict) -> str:
    # Same as relation_hash() in ingest_capital_chunks.py, for re-keying its cache.
    key = json.dumps([chunk_id, rel], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def graph_stats(chunks: List[dict]) -> str:
    # How well relations connect: relation ends used only once link nothing, and the chain search can
    # only follow paths inside one connected component.
    ends = Counter(r[k] for c in chunks for r in c["relations"] for k in ("source", "target"))
    neighbors = {}
    for c in chunks:
        for r in c["relations"]:
            neighbors.setdefault(r["source"], set()).add(r["target"])
            neighbors.setdefault(r["target"], set()).add(r["source"])
    seen, largest = set(), 0
    for start in neighbors:
        if start in seen:
            continue
        stack, size = [start], 0
        seen.add(start)
        while stack:
            size += 1
            for other in neighbors[stack.pop()]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        largest = max(largest, size)
    once = sum(n == 1 for n in ends.values())
    return (f"{len(ends)} relation ends, {once} used only once, "
            f"largest connected part {largest} ({largest / max(len(ends), 1):.0%})")


def embed_names(names: List[str]) -> np.ndarray:
    vectors = np.array(OpenAIEmbeddings(model="text-embedding-3-large", timeout=120).embed_documents(names))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def canonical_names(llm, todo: List[str], mapping: dict) -> int:
    # Name -> canonical English name. Only successful batches are cached; a failed batch keeps its names for
    # the next run instead of caching them unchanged (which left German names in the graph).
    batches = [todo[i:i + 40] for i in range(0, len(todo), 40)]
    results = run_batch(llm.with_structured_output(Canonical),
                        [CANON_PROMPT.format(n=len(b), names="\n".join(b)) for b in batches])
    failed = 0
    for batch, result in zip(batches, results):
        if isinstance(result, Exception) or len(result.names) != len(batch):
            failed += len(batch)
            continue
        mapping.update(zip(batch, (n.strip() or old for n, old in zip(result.names, batch))))
    return failed


def anchor_names(llm, chunks: List[dict], final: dict, anchors_path: str, min_uses: int,
                 min_similarity: float) -> dict:
    # Rare relation ends -> a core concept (an end used at least min_uses times, or an end of a chain text) that
    # means the same or is its general form, as confirmed by the LLM. Links the long tail of one-off names
    # ("Anstieg des Ölpreises") to the concepts the chains run through ("Oil Price Increase").
    # concept_anchors.json caches name -> core concept ("" = none fits) across builds.
    cache = json.load(open(anchors_path, encoding="utf-8")) if os.path.exists(anchors_path) else {}
    uses = Counter(final[r[k]] for c in chunks for r in c["relations"] for k in ("source", "target"))
    core = {n for n, k in uses.items() if k >= min_uses}
    core |= {final[r[k]] for c in chunks if any("Wirkungskette" in s for s in c["sources"])
             for r in c["relations"] for k in ("source", "target")}
    core = sorted(core)
    rare = sorted(n for n in uses if n not in core)
    todo = [n for n in rare if n not in cache]
    print(f"Anchoring: {len(rare)} rare relation ends ({len(todo)} new), {len(core)} core concepts ...")
    if todo:
        core_vectors = embed_names(core)
        similar = embed_names(todo) @ core_vectors.T
        items = []
        for name, row in zip(todo, similar):
            best = [i for i in np.argsort(row)[::-1][:8] if row[i] >= min_similarity]
            if best:
                items.append((name, [core[i] for i in best]))
            else:
                cache[name] = ""
        batches = [items[i:i + 20] for i in range(0, len(items), 20)]
        prompts = [
            ANCHOR_PROMPT.format(items="\n\n".join(
                f"{n}. {name}\n" + "\n".join(f"   {j}) {cand}" for j, cand in enumerate(cands, start=1))
                for n, (name, cands) in enumerate(batch, start=1)
            ))
            for batch in batches
        ]
        results = run_batch(llm.with_structured_output(AnchorChoices), prompts)
        for batch, result in zip(batches, results):
            if isinstance(result, Exception):
                continue
            for choice in result.choices:
                if 1 <= choice.item <= len(batch):
                    name, cands = batch[choice.item - 1]
                    cache[name] = cands[choice.choice - 1] if 1 <= choice.choice <= len(cands) else ""
        json.dump(cache, open(anchors_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    return {n: cache[n] for n in rare if cache.get(n)}


def stage_normalize(model: str, similarity: float, out_dir: str, retranslate: bool, anchor_min_uses: int,
                    anchor_similarity: float, anchor_model: str):
    path = f"{out_dir}/extracted_v2_all.json"
    chunks = json.load(open(path, encoding="utf-8"))
    print(f"Before: {graph_stats(chunks)}")
    names = sorted({n for c in chunks for n in c["concepts"]}
                   | {r[k] for c in chunks for r in c["relations"] for k in ("source", "target")})
    # concept_map.json caches name -> canonical name across builds.
    map_path = f"{out_dir}/concept_map.json"
    mapping = json.load(open(map_path, encoding="utf-8")) if os.path.exists(map_path) else {}
    llm = ChatOpenAI(model=model, temperature=0, max_retries=10, timeout=120)

    # 1. Every name -> canonical English name (translates German names, unifies spelling). --retranslate sends
    # names cached as their own canonical name again (repairs names a failed batch left untranslated).
    todo = [n for n in names if n not in mapping or (retranslate and mapping[n] == n)]
    print(f"Canonical names: {len(names)} names ({len(todo)} to map) ...")
    failed = canonical_names(llm, todo, mapping)
    if failed:
        print(f"  {failed} names not mapped (failed batches), kept unchanged for this run")
    canonical = {n: mapping.get(n, n) for n in names}

    # 2. Near-duplicate canonical names (embedding clusters) are merged where the LLM confirms synonyms.
    canon = sorted(set(canonical.values()))
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=1 - similarity).fit_predict(embed_names(canon))
    clusters = [c for c in ([canon[i] for i in np.where(labels == k)[0]] for k in set(labels)) if len(c) > 1]
    print(f"Merging: {len(canon)} canonical names, {len(clusters)} candidate clusters ...")
    merge = {}
    results = run_batch(llm.with_structured_output(SynonymGroups),
                        [GROUP_PROMPT.format(names="\n".join(c)) for c in clusters])
    for cluster, result in zip(clusters, results):
        if isinstance(result, Exception) or len(result.groups) != len(result.canonical):
            continue
        for group, name in zip(result.groups, result.canonical):
            for member in group:
                if member in cluster and len(group) > 1:
                    merge[member] = name.strip()
    final = {n: merge.get(canonical[n], canonical[n]) for n in names}

    # 3. Rare relation ends -> core concepts.
    if anchor_min_uses:
        # The stronger model: a wrong link joins unrelated chains, which a missed link does not.
        anchor_llm = ChatOpenAI(model=anchor_model, temperature=0, max_retries=10, timeout=180)
        anchored = anchor_names(anchor_llm, chunks, final, f"{out_dir}/concept_anchors.json", anchor_min_uses,
                                anchor_similarity)
        final = {n: anchored.get(f, f) for n, f in final.items()}
        print(f"  {len(anchored)} rare relation ends linked to a core concept")

    # 4. Apply to the chunks. Re-key the relation cache of ingest_capital_chunks.py, so relations whose
    # only change is a renamed end are not classified again.
    cache_path = f"{out_dir}/relation_classes.json"
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
    rekeyed = 0
    for c in chunks:
        c["concepts"] = list(dict.fromkeys(final[n] for n in c["concepts"]))
        for rel in c["relations"]:
            old = relation_key(c["id"], rel)
            rel["source"], rel["target"] = final[rel["source"]], final[rel["target"]]
            new = relation_key(c["id"], rel)
            if old in cache and new not in cache:
                cache[new] = cache[old]
                rekeyed += 1
    json.dump(chunks, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(mapping, open(map_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    if cache:
        json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{len(names)} names -> {len(set(final.values()))} concepts, {rekeyed} cached relation classes re-keyed")
    print(f"After: {graph_stats(chunks)}")


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
			Zusätzlich enthält die Wissensbasis {n_own} eigene Texte zu allgemeinen Kapitalmarktthemen und zu
			Wirkungsketten zwischen Märkten. Sie wurden für diese Website mit Hilfe eines KI-Sprachmodells erstellt und
			sind in der Quellenliste des Chats als „Mjoelnir-Erklärtext“ bzw. „Mjoelnir-Wirkungskette“ gekennzeichnet.
		</p>

		<a class="back" href="/">← Zurück zum Chat</a> · <a class="back" href="/impressum">Impressum</a> · <a class="back" href="/datenschutz">Datenschutz</a>
	</main>
</body>
</html>
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", choices=["fetch", "build", "normalize"], required=True)
    # gpt-4o-mini shares its daily request limit with the chat, so the build uses gpt-4.1-mini.
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--chain_model", default="gpt-4.1")
    # Cosine similarity above which canonical concept names become merge candidates.
    parser.add_argument("--similarity", type=float, default=0.85)
    # normalize: directory to work in (a copy of the index directory, to test a new normalization first).
    parser.add_argument("--out_dir", default=OUT_DIR)
    # normalize: map names cached as their own canonical name again (repairs untranslated German names).
    parser.add_argument("--retranslate", action="store_true")
    # normalize: relation ends used fewer times are linked to a core concept (0 = off), and the cosine
    # similarity a core concept needs to be offered as a candidate.
    parser.add_argument("--anchor_min_uses", type=int, default=3)
    parser.add_argument("--anchor_similarity", type=float, default=0.55)
    parser.add_argument("--anchor_model", default="gpt-4.1")
    args = parser.parse_args()
    if args.stage == "fetch":
        stage_fetch()
    elif args.stage == "build":
        stage_build(args.model, args.chain_model)
    else:
        stage_normalize(args.model, args.similarity, args.out_dir, args.retranslate, args.anchor_min_uses,
                        args.anchor_similarity, args.anchor_model)

"""
Renders architecture_diagram.png (repo root): technical overview of the
Capital Markets RAG pipeline - Docker deployment, knowledge-base build and
question answering, one box per component with what it does and how.
Full details (files, every threshold) are in README.md and docs/.

    python docs/architecture_diagram.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "architecture_diagram.png"))

# (fill, edge)
PLATFORM = ("#eef2ff", "#4338ca")
VOLUME = ("#f3f4f6", "#4b5563")
EXTERNAL = ("#fae8ff", "#a21caf")
BROWSER = ("#fef9c3", "#a16207")
DATA = ("#ecfccb", "#4d7c0f")
STORE = ("#d9f99d", "#3f6212")
SEARCH = ("#e0f2fe", "#0369a1")
LLM = ("#dbeafe", "#1d4ed8")
CHECK = ("#ffedd5", "#c2410c")
OK = ("#dcfce7", "#15803d")
NO = ("#fee2e2", "#b91c1c")
INK = "#1f2937"
MUTED = "#4b5563"
MONO = "DejaVu Sans Mono"

fig, ax = plt.subplots(figsize=(17, 25.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 151)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, title, text, color, tech=None, size=9.8, dashed=False):
    # Title on top, optional technical line (monospace) below it, body centered in the rest.
    fill, edge = color
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                facecolor=fill, edgecolor=edge, linewidth=2,
                                linestyle=(0, (5, 3)) if dashed else "solid"))
    ax.text(x + w / 2, y + h - 1.1, title, ha="center", va="top", fontsize=size + 1.6,
            color=edge, fontweight="bold")
    top = y + h - 3.3
    if tech:
        ax.text(x + w / 2, top, tech, ha="center", va="top", fontsize=size - 1.2, color=edge,
                family=MONO, linespacing=1.3)
        top -= 1.45 * (tech.count("\n") + 1) + 0.6
    if text:
        ax.text(x + w / 2, (y + top) / 2, text, ha="center", va="center", fontsize=size,
                color=INK, linespacing=1.35)
    return (x, y, w, h)


def frame(x, y, w, h, label, color="#9ca3af", dashed=False, fill="none", label_bottom=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.8",
                                facecolor=fill, edgecolor=color, linewidth=1.8,
                                linestyle=(0, (6, 3)) if dashed else "solid"))
    if label_bottom:
        ax.text(x + 1.6, y + 1.2, label, fontsize=10.5, fontweight="bold", color=color, va="bottom")
    else:
        ax.text(x + 1.6, y + h - 1.3, label, fontsize=10.5, fontweight="bold", color=color, va="top")


def arrow(start, end, label=None, rad=0.0, pos=0.5, dx=0, dy=0, color=INK, style="solid", both=False, size=9.2):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(
        arrowstyle="<|-|>" if both else "-|>", color=color, lw=1.8, mutation_scale=15,
        linestyle=style, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if label:
        ax.text(start[0] + (end[0] - start[0]) * pos + dx, start[1] + (end[1] - start[1]) * pos + dy,
                label, fontsize=size, color=color if color != INK else MUTED, ha="center", va="center",
                linespacing=1.25, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none"))


def section(y, title, note, color):
    ax.text(2, y, title, fontsize=15, fontweight="bold", color=color, va="center")
    ax.text(98, y, note, fontsize=10.5, color=MUTED, style="italic", va="center", ha="right")
    ax.plot([2, 98], [y - 1.8, y - 1.8], color="#e5e7eb", lw=1.2)


def mid(b, side):
    x, y, w, h = b
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]


ax.text(50, 149.3, "Capital Markets RAG  —  technical architecture", ha="center", fontsize=21,
        fontweight="bold", color=INK)
ax.text(50, 146.4, "Open WebUI + a custom Pipelines server that answers questions about a creator's "
        "capital-markets videos — only with what the videos actually say.", ha="center", fontsize=11.3, color=MUTED)

# ---------------------------------------------------------------------------
# 0. Deployment (Docker)
# ---------------------------------------------------------------------------
section(141.5, "Deployment  (Docker Compose)", "two containers on one compose network  ·  OpenAI as external API", PLATFORM[1])
frame(2, 106, 76, 33.5, "Docker host  (Windows · Docker Desktop)")
frame(20.5, 119, 55.5, 17.5, "compose network  (default bridge)", color="#6b7280", dashed=True)

browser = box(4, 124, 14.5, 10, "Browser", "user's chat", BROWSER, tech="localhost:3000")
webui = box(22.5, 120.5, 21.5, 13.5, "open-webui",
            "Chat UI, accounts, model list,\nbackground tasks (title, follow-ups)", PLATFORM,
            tech="image open-webui:v0.11.4\nport 3000 → 8080")
engine = box(51, 120.5, 23.5, 13.5, "pipelines-capital  (RAG engine)",
             "runs capital_rag_pipeline.py;\nindex, BM25 and graph held in RAM", PLATFORM,
             tech="image pipelines (pinned digest)\nport 9099 (internal) · env OPENAI_API_KEY", size=9.4)
v_data = box(22.5, 108, 15.5, 8.5, "open-webui-data/", "DB, uploads, settings", VOLUME,
             tech="→ /app/backend/data", size=9, dashed=True)
v_data = box(40, 108, 17.5, 8.5, "data/", "index files + scripts", VOLUME,
             tech="→ /data  (both)", size=9, dashed=True)
v_code = box(59.5, 108, 16.5, 8.5, "pipelines-capital/", "pipeline code + valves", VOLUME,
             tech="→ /app/pipelines", size=9, dashed=True)
v_env = box(4, 108, 14.5, 8.5, ".env", "git-ignored secrets", VOLUME,
            tech="OPENAI_API_KEY → engine", size=9, dashed=True)
openai = box(81, 120.5, 17, 13.5, "OpenAI API",
             "embeddings and\nchat completions", EXTERNAL,
             tech="text-embedding-3-large\ngpt-4o-mini · gpt-4o", size=9.4)

arrow(mid(browser, "r"), mid(webui, "l"), "HTTP", dy=1.8)
arrow(mid(webui, "r"), mid(engine, "l"), "OpenAI\nAPI\n(SSE)", both=True, dy=4.6, size=8.6)
arrow(mid(engine, "r"), mid(openai, "l"), "HTTPS", both=True, dy=1.8)
arrow(mid(v_data, "t"), (30.25, 120.5), style="dashed", color=VOLUME[1])
arrow((45, 116.5), (42, 120.5), style="dashed", color=VOLUME[1])
arrow((52.5, 116.5), (56, 120.5), style="dashed", color=VOLUME[1])
arrow(mid(v_code, "t"), (67.75, 120.5), style="dashed", color=VOLUME[1])
ax.text(81, 116.4, "Connection set in Open WebUI:\nAdmin › Settings › Connections\n"
        "http://pipelines-capital:9099\n+ Pipelines API key", fontsize=8.8, color=MUTED, va="top",
        linespacing=1.35, family=MONO)

# ---------------------------------------------------------------------------
# 1. Building the knowledge base
# ---------------------------------------------------------------------------
section(102, "1   Building the knowledge base", "offline, run once per corpus change:  ingest_capital_chunks.py", DATA[1])
W5, RY, RH = 18, 79.5, 19
xs = [2, 21.5, 41, 60.5, 80]
k_src = box(xs[0], RY, W5, RH, "Video transcripts",
            "pre-cut passages with key\nconcepts, cause → effect\nrelations and example\nquestions", DATA,
            tech="extracted_v2_all.json")
k_cls = box(xs[1], RY, W5, RH, "Content classifier",
            "labels passages as timeless\nconcept or time-bound\nopinion; relations as\ntimeless / time-bound", DATA,
            tech="gpt-4o-mini · structured\ncached by content hash")
k_vec = box(xs[2], RY, W5, RH, "Multi-vector indexing",
            "each passage stored under\nits text AND under the\nquestions it answers", DATA,
            tech="HyPE · ~10 vectors/chunk\ntext-embedding-3-large")
k_sum = box(xs[3], RY, W5, RH, "Concept summaries",
            "same idea from several\nvideos merged into one\nsummary, sources only;\nopinions never merged", DATA,
            tech="RAPTOR-style · agglomerative\ncosine ≥ 0.73 · ≥ 2 videos")
k_graph = box(xs[4], RY, W5, RH, "Knowledge graph",
              "concepts linked by their\ncause → effect relations,\nwith the passages that\nstate each link", DATA,
              tech="nodes: concepts\nedges: source → target")
for a, b in zip([k_src, k_cls, k_vec, k_sum], [k_cls, k_vec, k_sum, k_graph]):
    arrow(mid(a, "r"), mid(b, "l"))

frame(2, 60, 96, 16, "Knowledge base   (files in data/faiss_capital_index/, loaded into the RAG engine at startup)",
      color=STORE[1], dashed=True, fill="#f7fee7", label_bottom=True)
SW, SY, SH = 21, 63.6, 10.2
s_sem = box(4, SY, SW, SH, "Vector index", "finds passages by meaning", STORE, tech="FAISS · cosine similarity", size=9.3)
s_kw = box(27.5, SY, SW, SH, "Keyword index", "exact terms, names, tickers", STORE, tech="BM25 · DE + EN stopwords", size=9.3)
s_sum = box(51, SY, SW, SH, "Summaries", "cross-video explanations", STORE, tech="in FAISS + summaries.json", size=9.3)
s_graph = box(74.5, SY, SW, SH, "Concept graph", "cause → effect links", STORE, tech="graph.json", size=9.3)
arrow(mid(k_vec, "b"), (20, SY + SH))
arrow(mid(k_vec, "b"), (40, SY + SH))
arrow(mid(k_sum, "b"), mid(s_sum, "t"))
arrow(mid(k_graph, "b"), mid(s_graph, "t"))

# ---------------------------------------------------------------------------
# 2. Answering a question
# ---------------------------------------------------------------------------
section(56.5, "2   Answering a question", "every chat message · shown only after the fact check", SEARCH[1])
AY, AH = 37.5, 15.5
a_q = box(xs[0], AY, W5, AH, "Question", "asked in the chat;\nOpen WebUI background\ntasks are routed apart", PLATFORM,
          tech="Pipeline.pipe()")
a_search = box(xs[1], AY, W5, AH, "Hybrid search",
               "meaning + keyword search\nmerged into one ranking", SEARCH,
               tech="FAISS 40 + BM25 40\n0.6·dense + 0.4·bm25 → 12")
a_graph = box(xs[2], AY, W5, AH, "Graph expansion",
              "adds passages stating the\nsame cause → effect,\noften from other videos", SEARCH,
              tech="relations of top 3 → +≤ 4")
a_judge = box(xs[3], AY, W5, AH, "Relevance judge",
              "keeps only passages that\nreally answer THIS question", CHECK,
              tech="gpt-4o-mini · structured\nrelevance ≥ 6/10 → top 5")
a_write = box(xs[4], AY, W5, AH, "Answer writer",
              "context only, cites [n],\nnames the video for\nevery opinion", LLM,
              tech="gpt-4o-mini · temp 0\nnot streamed yet")
for a, b in zip([a_q, a_search, a_graph, a_judge], [a_search, a_graph, a_judge, a_write]):
    arrow(mid(a, "r"), mid(b, "l"))
arrow((30.5, 60), mid(a_search, "t"), "reads", style="dashed", color=STORE[1], dx=3.5)
arrow((50, 60), mid(a_graph, "t"), "reads", style="dashed", color=STORE[1], dx=3.5)

BY, BH = 18.5, 15.5
a_check = box(80, BY, 18, BH, "Fact checker",
              "single claims, each needs a\nword-for-word quote with\nthe same cause & certainty", CHECK,
              tech="gpt-4o + Python rules\nquote · cause · video", size=9.5)
a_fix = box(58, BY, 17, BH, "Fixer",
            "removes or corrects\nunproven claims,\nthen re-checks", CHECK,
            tech="gpt-4o · ≤ 2 rounds\nthen trim sentences")
a_no = box(35.5, BY, 18, BH, "Honest \"not covered\"",
           "no guessing; suggests\nrelated topics that ARE\ncovered (verified)", NO,
           tech="refusal + explorer")
arrow(mid(a_write, "b"), mid(a_check, "t"))
arrow((80, BY + 10), (75, BY + 10), "unproven", dy=2.0)
arrow((75, BY + 4.5), (80, BY + 4.5), "re-check", dy=-2.0)
arrow(mid(a_fix, "l"), mid(a_no, "r"), "still\nunproven", dy=3.4)
arrow((62.5, AY), (47, BY + BH), "nothing relevant / check fails", pos=0.5, dy=0.3, color=NO[1], size=8.8)

CY = 1
a_ok = box(58, CY, 40, 14, "Verified answer  ✓",
           "shown with only the cited sources (concept / opinion + why used)\n"
           "and follow-up questions that were checked to be answerable", OK,
           tech="status events while working · follow-up chips")
arrow(mid(a_check, "b"), (89, CY + 14), "all claims proven ✓", dx=-6.5)

# Design principles
ax.add_patch(FancyBboxPatch((2, 1), 31, 33, boxstyle="round,pad=0,rounding_size=1.8",
                            facecolor="#f9fafb", edgecolor="#9ca3af", linewidth=1.6))
ax.text(4, 31.4, "Design principles", fontsize=12.5, fontweight="bold", color=INK, va="center")
principles = [
    "Answer only from the videos —\nno general textbook knowledge",
    "Every claim is proven by a\nword-for-word quote (checked in code)",
    "Keep the source's cause and certainty\n(\"could\" never becomes \"always\")",
    "Opinions are time-bound: always\nname the video they come from",
    "Conflicting views are shown side\nby side, never blended",
    "Fail closed: no proof, no answer —\nsuggest related topics instead",
]
for i, p in enumerate(principles):
    y = 27.4 - i * 4.4
    ax.text(4.2, y, "✓", fontsize=12, color=OK[1], fontweight="bold", va="center")
    ax.text(6.6, y, p, fontsize=9.6, color=INK, va="center", linespacing=1.25)

fig.savefig(OUT, dpi=130, bbox_inches="tight", pad_inches=0.35, facecolor="white")
print(f"Saved {OUT}")

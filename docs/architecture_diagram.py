"""
Renders architecture_diagram.png (repo root): Docker architecture, offline
ingestion and query-time flow of the Capital Markets RAG pipeline.

    python docs/architecture_diagram.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "architecture_diagram.png")

# (fill, edge)
YELLOW = ("#fff8c4", "#a16207")
GREEN = ("#ecfccb", "#4d7c0f")
GREEN_DARK = ("#d9f99d", "#4d7c0f")
BLUE = ("#dbeafe", "#1d4ed8")
RED = ("#ffe4e6", "#be123c")
RED_DARK = ("#fecaca", "#991b1b")
GRAY = ("#f3f4f6", "#6b7280")
PURPLE = ("#fae8ff", "#a21caf")
OK = ("#dcfce7", "#15803d")
ORANGE = ("#ffedd5", "#c2410c")
INK = "#1f2937"
MUTED = "#4b5563"

fig, ax = plt.subplots(figsize=(17, 24))
ax.set_xlim(0, 100)
ax.set_ylim(0, 141)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, text, color, size=8.6, bold=False, title=None):
    fill, edge = color
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=1.6",
            facecolor=fill, edgecolor=edge, linewidth=1.8,
        )
    )
    cy = y + h / 2
    if title:
        # Bold title on its own line at the top; body centered in the space below.
        ax.text(x + w / 2, y + h - 1.0, title, ha="center", va="top", fontsize=size + 0.6,
                color=INK, fontweight="bold")
        cy = y + (h - 2.2) / 2
    ax.text(
        x + w / 2, cy, text, ha="center", va="center", fontsize=size,
        color=INK, fontweight="bold" if bold else "normal", linespacing=1.35,
    )
    return (x, y, w, h)


def frame(x, y, w, h, label, dashed=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=2",
            facecolor="none", edgecolor="#9ca3af", linewidth=1.6,
            linestyle=(0, (5, 4)) if dashed else "solid",
        )
    )
    ax.text(x + 1.5, y + h - 1.4, label, fontsize=9.5, color=MUTED,
            fontweight="normal" if dashed else "bold", style="italic" if dashed else "normal", va="top")


def arrow(start, end, label=None, rad=0.0, label_pos=0.5, color=INK, dx=0, dy=0):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, mutation_scale=14,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2),
    )
    if label:
        lx = start[0] + (end[0] - start[0]) * label_pos + dx
        ly = start[1] + (end[1] - start[1]) * label_pos + dy
        ax.text(lx, ly, label, fontsize=8.2, color=MUTED, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.95))


def section(y, text, color):
    ax.text(2, y, text, fontsize=13, fontweight="bold", color=color, va="center")


def mid(b, side):
    x, y, w, h = b
    return {
        "l": (x, y + h / 2), "r": (x + w, y + h / 2),
        "t": (x + w / 2, y + h), "b": (x + w / 2, y),
    }[side]


ax.text(50, 139, "Capital Markets RAG  —  architecture", ha="center", fontsize=19, fontweight="bold", color=INK)

# ---------------------------------------------------------------------------
# 1. Docker architecture
# ---------------------------------------------------------------------------
section(134.5, "1  Docker architecture", "#111827")
frame(1.5, 96, 97, 36, "Host: Windows machine (Docker Desktop)")
frame(32, 110.5, 64.5, 18.5, "Docker network (compose default)", dashed=True)

browser = box(4, 118, 18, 8, "Browser\nlocalhost:3000", YELLOW, size=10)
host = box(
    4, 98.5, 24, 16,
    "Host folder  pdfs/   →  /data\n\n"
    "faiss_capital_index/   (git-ignored)\n"
    "  extracted_v2_all.json  (source)\n"
    "  index.faiss · index.pkl\n"
    "  graph.json · summaries.json\n"
    "ingest_capital_chunks.py\n"
    "generate / apply_start_suggestions.py",
    GREEN, size=8.3,
)
webui = box(
    34.5, 112, 25, 13,
    "port 3000 → 8080\nvolumes: pdfs → /data\nopen-webui-data → /app/backend/data\n"
    "model picker: Capital Markets RAG",
    BLUE, title="container: open-webui",
)
capital = box(
    64, 112, 30.5, 13,
    "port 9098 → 9099\nvolumes: pdfs → /data\npipelines-capital/ → /app/pipelines\n"
    "env: OPENAI_API_KEY  (from .env)\nloads capital_rag_pipeline.py at startup",
    RED, title="container: open-webui-pipelines-capital",
)
conn = box(
    34.5, 98.5, 25, 9.5,
    "Admin › Settings › Connections\nURL  http://pipelines-capital:9099\n"
    "Auth: Pipelines API key  (added manually)",
    GRAY,
)
openai = box(
    64, 98.5, 30.5, 9.5,
    "OpenAI API  (external)\ntext-embedding-3-large · gpt-4o-mini · gpt-4o",
    PURPLE, size=9,
)

arrow(mid(browser, "r"), (34.5, 121), "HTTP :3000")
arrow(mid(webui, "r"), mid(capital, "l"), "OpenAI-compatible\nchat (streamed)", dy=0)
arrow(mid(webui, "b"), mid(conn, "t"), "reads connection")
arrow(mid(capital, "b"), mid(openai, "t"), "embeddings / LLM calls")
arrow((28, 110), (34.5, 114), "bind mount /data\n(both containers)", label_pos=0.45, dy=-2.6)

# ---------------------------------------------------------------------------
# 2. Offline ingestion
# ---------------------------------------------------------------------------
section(92, "2  Offline ingestion  (manual:  docker exec … ingest_capital_chunks.py,  then restart the container)", "#3f6212")

W, H, Y = 14.6, 12.5, 76.5
xs = [1.5, 18, 34.5, 51, 67.5, 84]
ing = [
    box(xs[0], Y, W, H, "chunks with content,\nembedding_text,\nhypothetical_questions,\nconcepts, relations,\nsources (video labels)", GREEN, title="extracted_v2_all.json"),
    box(xs[1], Y, W, H, "concept  vs  opinion\nmixed  →  opinion\ngpt-4o-mini, cached by\ncontent hash\nchunk_classes.json", GREEN, title="Classify chunks"),
    box(xs[2], Y, W, H, "timeless  vs  time-bound\none relation per request\n(batching unreliable)\nrelation_classes.json", GREEN, title="Classify relations"),
    box(xs[3], Y, W, H, "embedding_text  +\neach hypothetical question\n≈ 10 vectors / chunk,\nall → the full chunk\ntext-embedding-3-large", GREEN, title="HyPE embedding"),
    box(xs[4], Y, W, H, "cluster concept chunks\ncosine ≥ 0.73, ≥ 2 videos\nopinions never merged\nsummarize from passages\nsummaries.json", GREEN, title="RAPTOR summaries"),
    box(xs[5], Y, W, H, "MAX_INNER_PRODUCT\n+ L2 normalized\n=  cosine similarity\nchunks + summaries\nindex.faiss / index.pkl", GREEN_DARK, title="FAISS index"),
]
for a, b in zip(ing, ing[1:]):
    arrow(mid(a, "r"), mid(b, "l"))

graph = box(
    1.5, 66, 31.1, 7.5,
    "nodes = concepts  ·  edges = source → target (direction)\nwith the chunks that assert each relation",
    GREEN, size=8.4, title="Concept graph  graph.json",
)
starts = box(
    51, 66, 47.6, 7.5,
    "generate_start_suggestions.py:  best-supported summaries → starter questions,\n"
    "kept only if the grader confirms them  →  apply_start_suggestions.py (Open WebUI model)",
    GREEN, size=8.4, title="Start-page suggestions",
)
arrow((xs[0] + W / 2, Y), (xs[0] + W / 2, 73.5), "relations", label_pos=0.5)
arrow((xs[4] + W / 2, Y), (xs[4] + W / 2, 73.5), "summaries.json", label_pos=0.5)

# ---------------------------------------------------------------------------
# 3. Query time
# ---------------------------------------------------------------------------
section(61.5, "3  Query time  (every chat message,  Pipeline.pipe())", "#9f1239")

QY, QH = 47, 11.5
q_user = box(1.5, QY, 11, QH, "User question\n(Open WebUI chat)", RED, size=9)
q_route = box(15, QY, 14.5, QH, "'### Task:' ?\nfollow-ups → explorer\ntitle / tags → plain LLM\nelse: answer pipeline", RED, title="Routing")
q_ret = box(32, QY, 17, QH, "dense: 40 hits, cos ≥ 0.35\n+ BM25: top 40 (stopwords)\nmin-max, 0.6·dense + 0.4·bm25\n→ top 12 candidates", RED, title="Hybrid retrieval")
q_graph = box(51.5, QY, 14, QH, "chunks asserting the\nsame source → target\nrelation as the top 3\n+ ≤ 4 candidates", RED, title="Graph expansion")
q_grade = box(68, QY, 14.5, QH, "gpt-4o-mini, structured\nsupports_answer  AND\nrelevance ≥ 6 / 10\n→ top 5", ORANGE, title="LLM relevance gate")
q_ctx = box(84.5, QY, 14, QH, "[n] CONCEPT  /  OPINION\n+ video labels\nRelation TIMELESS /\nTIME-BOUND", RED, title="Context blocks")
for a, b in [(q_user, q_route), (q_route, q_ret), (q_ret, q_graph), (q_graph, q_grade), (q_grade, q_ctx)]:
    arrow(mid(a, "r"), mid(b, "l"))

BY = 29.5
q_gen = box(84.5, BY, 14, QH, "gpt-4o-mini, temp 0\ncontext only, cite [n]\nno textbook knowledge\nkeep certainty, attribute\nopinions to their video", BLUE, size=8.2, title="Generate answer")
q_check = box(64, BY, 17.5, QH, "gpt-4o splits into claims\n+ CODE checks:\nquote really in context\n(1 sentence, ≥ 90 % match)\nsame cause · video named", ORANGE, size=8.2, title="Claim check")
q_rev = box(44.5, BY, 15.5, QH, "gpt-4o, targeted fixes:\ndelete unsupported\nadd video title\nsplit merged statements\nmax 2 rounds", ORANGE, size=8.2, title="Revise")
q_trim = box(25.5, BY, 15, QH, "drop flagged sentences\ncheck again,\nsame bar\nmust keep ≥ 1 fact", ORANGE, size=8.4, title="Trim (last resort)")
q_refuse = box(1.5, BY, 20.5, QH, "\"Dazu habe ich in den Quellen\nleider keine belegte Antwort …\"\n+ verified related topics\n(explorer, clickable chips)", RED_DARK, size=8.3, title="Refuse  (fail closed)")

arrow(mid(q_ctx, "b"), mid(q_gen, "t"))
arrow(mid(q_gen, "l"), mid(q_check, "r"))
arrow((64, BY + 8), (60, BY + 8), "problems", dy=1.6)
arrow((60, BY + 3.5), (64, BY + 3.5), "re-check", dy=-1.6)
arrow(mid(q_rev, "l"), mid(q_trim, "r"), "still failing", dy=1.7)
arrow(mid(q_trim, "l"), mid(q_refuse, "r"), "not clean", dy=1.7)
arrow((75.25, QY), (11.75, BY + QH), "no chunk passes the gate", label_pos=0.5, dy=1.5, color="#991b1b")

CY = 11.5
q_ans = box(44.5, CY, 25, QH, "shown only after the check\n(never streamed unchecked)\nQuellen: only cited blocks,\nKonzept / Meinung + relevance reason", OK, size=8.4, title="Answer + sources  ✓")
q_fu = box(76, CY, 22.5, QH, "Open WebUI follow-up task\n→ explorer, after the answer\nquestions verified against\ntheir excerpt (same gate)", OK, size=8.4, title="Follow-up chips")
arrow(mid(q_check, "b"), (64, CY + QH), "all supported ✓", label_pos=0.45, dx=3)
arrow(mid(q_trim, "b"), (44.5, CY + 6), "clean", rad=0.25, label_pos=0.5, dx=-2)
arrow(mid(q_ans, "r"), mid(q_fu, "l"))

# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------
ax.text(
    1.5, 5.5,
    "Fallbacks:  missing index → server starts, replies with the ingestion command  ·  missing graph.json → no graph expansion  ·  "
    "failed grading call → only that chunk skipped\n"
    "check raises an exception → refuse (an unchecked answer is never shown)  ·  CHECK_ANSWER_GROUNDING = False → "
    "streams an unchecked answer marked \"(ungeprüft)\"  ·  details: README.md, docs/",
    fontsize=8.6, color=MUTED, va="center", style="italic", linespacing=1.5,
)
for i, (color, label) in enumerate(
    [(GREEN, "offline / data"), (RED, "retrieval"), (ORANGE, "verification gate"), (BLUE, "generation"), (OK, "shown to user"), (RED_DARK, "refusal")]
):
    x = 1.5 + i * 16
    ax.add_patch(FancyBboxPatch((x, 0.8), 2.4, 1.8, boxstyle="round,pad=0,rounding_size=0.4",
                                facecolor=color[0], edgecolor=color[1], linewidth=1.4))
    ax.text(x + 3.2, 1.7, label, fontsize=8.8, color=MUTED, va="center")

fig.savefig(OUT, dpi=140, bbox_inches="tight", pad_inches=0.3, facecolor="white")
print(f"Saved {os.path.abspath(OUT)}")

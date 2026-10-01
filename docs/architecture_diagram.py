"""
Renders architecture_diagram.png (repo root): a component-level overview of the
Capital Markets RAG pipeline for readers who know RAG concepts but not the code.
Implementation details (files, ports, thresholds) are in README.md and docs/.

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
DATA = ("#ecfccb", "#4d7c0f")
STORE = ("#d9f99d", "#3f6212")
SEARCH = ("#e0f2fe", "#0369a1")
LLM = ("#dbeafe", "#1d4ed8")
CHECK = ("#ffedd5", "#c2410c")
OK = ("#dcfce7", "#15803d")
NO = ("#fee2e2", "#b91c1c")
INK = "#1f2937"
MUTED = "#4b5563"

fig, ax = plt.subplots(figsize=(17, 20.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 122)
ax.axis("off")
fig.patch.set_facecolor("white")


def box(x, y, w, h, title, text, color, size=10):
    fill, edge = color
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                facecolor=fill, edgecolor=edge, linewidth=2))
    ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top", fontsize=size + 1.6,
            color=edge, fontweight="bold")
    ax.text(x + w / 2, y + (h - 2.6) / 2, text, ha="center", va="center", fontsize=size,
            color=INK, linespacing=1.4)
    return (x, y, w, h)


def arrow(start, end, label=None, rad=0.0, pos=0.5, dx=0, dy=0, color=INK, style="solid", both=False):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(
        arrowstyle="<|-|>" if both else "-|>", color=color, lw=1.8, mutation_scale=16,
        linestyle=style, connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if label:
        ax.text(start[0] + (end[0] - start[0]) * pos + dx, start[1] + (end[1] - start[1]) * pos + dy,
                label, fontsize=9.5, color=color if color != INK else MUTED, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none"))


def section(y, title, note, color):
    ax.text(2, y, title, fontsize=15, fontweight="bold", color=color, va="center")
    ax.text(98, y, note, fontsize=10.5, color=MUTED, style="italic", va="center", ha="right")
    ax.plot([2, 98], [y - 1.8, y - 1.8], color="#e5e7eb", lw=1.2)


def mid(b, side):
    x, y, w, h = b
    return {"l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (x + w / 2, y + h), "b": (x + w / 2, y)}[side]


ax.text(50, 120, "Capital Markets RAG  —  how it works", ha="center", fontsize=21, fontweight="bold", color=INK)
ax.text(50, 116.8, "A chat assistant that answers questions about a creator's capital-markets videos "
        "— only with what the videos actually say.", ha="center", fontsize=11.5, color=MUTED)

# ---------------------------------------------------------------------------
# The platform
# ---------------------------------------------------------------------------
section(111.5, "The platform", "three components, each in its own Docker container or service", PLATFORM[1])
W3, PY, PH = 28, 98, 10.5
p_chat = box(2, PY, W3, PH, "Chat interface",
             "Open WebUI: where users ask questions\nand see answers, sources and\nsuggested follow-up questions", PLATFORM)
p_engine = box(36, PY, W3, PH, "RAG engine",
               "Our custom pipeline: searches, judges,\nwrites and fact-checks — the logic\nshown in sections 1 and 2", PLATFORM)
p_llm = box(70, PY, W3, PH, "Language models",
            "OpenAI: embeddings for search, a fast\nmodel for most steps, a stronger\nmodel for fact-checking", PLATFORM)
arrow(mid(p_chat, "r"), mid(p_engine, "l"), "question / answer", both=True, dy=2.2)
arrow(mid(p_engine, "r"), mid(p_llm, "l"), "AI calls", both=True, dy=2.2)

# ---------------------------------------------------------------------------
# 1. Building the knowledge base
# ---------------------------------------------------------------------------
section(92, "1   Building the knowledge base", "done once, offline — before anyone asks a question", DATA[1])
W5, RY, RH = 18, 70.5, 17
xs = [2, 21.5, 41, 60.5, 80]
k_src = box(xs[0], RY, W5, RH, "Video transcripts",
            "Passages from the videos,\npre-tagged with their key\nconcepts, cause → effect\nstatements and example\nquestions they answer", DATA)
k_cls = box(xs[1], RY, W5, RH, "Content classifier",
            "An LLM labels every passage:\ntimeless concept  or\ntime-bound opinion\n(forecasts, market views).\nWhen in doubt: opinion.", DATA)
k_vec = box(xs[2], RY, W5, RH, "Multi-vector indexing",
            "Each passage is stored under\nits own text AND under the\nquestions it answers, so a\nuser's question can match\na similar question", DATA)
k_sum = box(xs[3], RY, W5, RH, "Concept summaries",
            "When the same idea is\nexplained in several videos,\nan LLM merges it into one\ncomplete summary — using\nonly what the videos say", DATA)
k_graph = box(xs[4], RY, W5, RH, "Knowledge graph",
              "Links concepts through their\ncause → effect relations,\ne.g. interest rates → bond\nprices, and remembers which\npassages state each link", DATA)
for a, b in zip([k_src, k_cls, k_vec, k_sum], [k_cls, k_vec, k_sum, k_graph]):
    arrow(mid(a, "r"), mid(b, "l"))

# Knowledge base (the stores the engine reads at question time)
ax.add_patch(FancyBboxPatch((2, 54.5), 96, 12, boxstyle="round,pad=0,rounding_size=1.8",
                            facecolor="#f7fee7", edgecolor=STORE[1], linewidth=2, linestyle=(0, (6, 3))))
ax.text(4, 65.2, "Knowledge base", fontsize=12, fontweight="bold", color=STORE[1], va="center")
SW, SY, SH = 21, 55.8, 7.6
s_sem = box(4, SY, SW, SH, "Semantic index", "finds passages by meaning", STORE, size=9.5)
s_kw = box(27.5, SY, SW, SH, "Keyword index", "finds exact terms, names, tickers", STORE, size=9.5)
s_sum = box(51, SY, SW, SH, "Summaries", "complete cross-video explanations", STORE, size=9.5)
s_graph = box(74.5, SY, SW, SH, "Graph", "concepts and their cause → effect links", STORE, size=9.5)
arrow(mid(k_vec, "b"), (20, SY + SH))
arrow(mid(k_vec, "b"), (40, SY + SH))
arrow(mid(k_sum, "b"), mid(s_sum, "t"))
arrow(mid(k_graph, "b"), mid(s_graph, "t"))

# ---------------------------------------------------------------------------
# 2. Answering a question
# ---------------------------------------------------------------------------
section(50, "2   Answering a question", "every chat message — shown only after the fact check", SEARCH[1])
AY, AH = 33, 13.5
a_q = box(xs[0], AY, W5, AH, "Question", "asked in the chat", PLATFORM)
a_search = box(xs[1], AY, W5, AH, "Hybrid search",
               "meaning-based search +\nkeyword search, merged\ninto one ranking", SEARCH)
a_graph = box(xs[2], AY, W5, AH, "Graph expansion",
              "adds passages that state\nthe same cause → effect,\noften from other videos", SEARCH)
a_judge = box(xs[3], AY, W5, AH, "Relevance judge",
              "an LLM keeps only the\nfew passages that really\nanswer THIS question", CHECK)
a_write = box(xs[4], AY, W5, AH, "Answer writer",
              "an LLM writes only from\nthose passages, cites each\nstatement, names the video\nfor every opinion", LLM)
for a, b in zip([a_q, a_search, a_graph, a_judge], [a_search, a_graph, a_judge, a_write]):
    arrow(mid(a, "r"), mid(b, "l"))
arrow((30.5, 54.5), mid(a_search, "t"), "looks up", style="dashed", color=STORE[1], dx=4.5)
arrow((50, 54.5), mid(a_graph, "t"), "looks up", style="dashed", color=STORE[1], dx=4.5)

# Checking loop: not covered <- fixer <-> fact checker, verified answer below.
BY = 16.5
a_check = box(80, BY, 18, AH, "Fact checker",
              "a stronger LLM splits the\nanswer into single claims;\nrules verify each one has a\nword-for-word quote with\nthe same cause & certainty", CHECK, size=9.6)
a_fix = box(58, BY, 17, AH, "Fixer",
            "removes or corrects\nthe unproven claims,\nthen checks again\n(up to 2 rounds)", CHECK)
a_no = box(35.5, BY, 18, AH, "Honest \"not covered\"",
           "no guessing: says the\nvideos don't answer this\nand suggests related\ntopics that ARE covered", NO)
arrow(mid(a_write, "b"), mid(a_check, "t"))
arrow((80, BY + 9), (75, BY + 9), "unproven", dy=2.0)
arrow((75, BY + 4), (80, BY + 4), "re-check", dy=-2.0)
arrow(mid(a_fix, "l"), mid(a_no, "r"), "still\nunproven", dy=3.2)
arrow((62.5, AY), (47, BY + AH), "nothing relevant found", pos=0.5, dy=0.2, color=NO[1])

CY = 1
a_ok = box(58, CY, 40, AH, "Verified answer  ✓",
           "shown with its sources and why each source was used,\n"
           "plus suggested next questions — each checked to be answerable", OK)
arrow(mid(a_check, "b"), (89, CY + AH), "all claims proven ✓", dx=-6.5)

# Design principles
ax.add_patch(FancyBboxPatch((2, 1), 31, 29, boxstyle="round,pad=0,rounding_size=1.8",
                            facecolor="#f9fafb", edgecolor="#9ca3af", linewidth=1.6))
ax.text(4, 27.6, "Design principles", fontsize=12.5, fontweight="bold", color=INK, va="center")
principles = [
    "Answer only from the videos —\nno general textbook knowledge",
    "Every claim is proven by a\nword-for-word quote",
    "Keep the source's cause and certainty\n(\"could\" never becomes \"always\")",
    "Opinions are time-bound: always\nname the video they come from",
    "Conflicting views are shown side\nby side, never blended",
    "No proof, no answer — suggest\nrelated topics instead of guessing",
]
for i, p in enumerate(principles):
    y = 23.9 - i * 3.85
    ax.text(4.2, y, "✓", fontsize=12, color=OK[1], fontweight="bold", va="center")
    ax.text(6.6, y, p, fontsize=9.6, color=INK, va="center", linespacing=1.25)

fig.savefig(OUT, dpi=130, bbox_inches="tight", pad_inches=0.35, facecolor="white")
print(f"Saved {os.path.abspath(OUT)}")

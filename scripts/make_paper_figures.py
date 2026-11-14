"""Paper figures for the KisaanDost research paper -> docs/figures/figNN_*.png (300 dpi) and .pdf (vector).

  pip install matplotlib
  python -m scripts.make_paper_figures

Every number is taken from EXPERIMENT_LOG.md (P1-P13) unless a figure reads a local data file (marked DATA). Nothing is estimated:
where a value is derived (figure 13b) the axis says so. Colour: grey for baselines and context, one blue accent for the system the
figure is about, hatching for numbers that are optimistic (same-source test). No titles inside the images: captions are in
docs/figures/CAPTIONS.md so the paper's own caption style can be used.
"""
import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "figures"
GREY, LIGHT, ACCENT, GOOD = "#7d838b", "#c9ccd1", "#1f5fbf", "#2e8b57"
plt.rcParams.update({"font.family": ["Arial", "Nirmala UI"], "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#444", "axes.linewidth": .8, "pdf.fonttype": 42, "axes.titlesize": 9, "axes.labelsize": 9})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  ", name)


def vlabels(ax, bars, fmt="{:.3f}", pad=.008, **kw):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + pad, fmt.format(b.get_height()), ha="center", va="bottom", fontsize=8, zorder=5,
                bbox=dict(fc="white", ec="none", pad=.6), **kw)


def hlabels(ax, bars, fmt="{:.3f}", pad=.006):
    for b in bars:
        ax.text(b.get_width() + pad, b.get_y() + b.get_height() / 2, fmt.format(b.get_width()), va="center", fontsize=8)


def fig_architecture():
    src = next(iter((Path.home() / ".claude" / "projects").glob("*/*/tool-results/mcp-claude_ai_Claude_Docs-blob-*.png")), None)
    if src:
        shutil.copy(src, OUT / "fig01_architecture.png")
        print("   fig01_architecture (copied from the rendered diagram in the paper document)")
    else:
        print("   fig01_architecture SKIPPED: render of the diagram not found")


def fig_scripture():  # P3d, P9, P10, P13: 206 held-out scripture verses, WER
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    x = np.arange(4)
    raw = [0.396, 0.338, 0.384]
    full = [0.332, 0.327, 0.329, 0.315]
    b1 = ax.bar(x[:3] - .19, raw, .36, color=LIGHT, label="raw model output")
    b2 = ax.bar(list(x[:3] + .19) + [x[3]], full, .36, color=[GREY] * 3 + [ACCENT], label="full layer (guards, filter, rules)")
    vlabels(ax, b1), vlabels(ax, b2)
    for y, t in ((.539, "Marathi left unchanged  0.539"), (.401, "word rules only  0.401")):
        ax.axhline(y, color="#444", lw=.8, ls=(0, (4, 3)))
        ax.text(3.55, y + .006, t, ha="right", va="bottom", fontsize=8, color="#444")
    ax.set_xticks(x, ["v3", "v4", "v5", "v5 + word lexicon\n(deployed)"])
    ax.set_ylabel("Word error rate (206 held-out verses)")
    ax.set_ylim(0, .6)
    ax.legend(frameon=False, loc="lower left", fontsize=8, ncol=2, bbox_to_anchor=(0, -.3))
    save(fig, "fig02_conversion_scripture")


def fig_new_sentences():  # P10, P13: 119 held-out sentences from the user datasets
    rows = [("Marathi left unchanged", .451, LIGHT, ""), ("Word rules only", .451, LIGHT, ""), ("v3 raw", .474, LIGHT, ""),
            ("v3 full layer", .438, GREY, ""), ("v5 full layer, old allowed list", .413, GREY, ""),
            ("v5 full layer, rebuilt allowed list", .345, GREY, ""), ("v5 + word lexicon (deployed)", .333, ACCENT, ""),
            ("v5 length guard only", .167, "white", "////"), ("v5 raw", .167, "white", "////")]
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ys = np.arange(len(rows))[::-1]
    for y, (name, v, c, h) in zip(ys, rows):
        b = ax.barh(y, v, .62, color=c, hatch=h, edgecolor=ACCENT if h else c, lw=.8)
        hlabels(ax, b)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlabel("Word error rate (119 held-out sentences from the user datasets)")
    ax.set_xlim(0, .56)
    ax.text(.5, ys[-1] - 1.05, "hatched: optimistic, the test shares source and templates with the training pairs", fontsize=7.5, ha="right", color="#444")
    ax.set_ylim(ys[-1] - 1.4, ys[0] + .6)
    save(fig, "fig03_conversion_new_sentences")


def fig_safeguards():  # P3a, P3b, P3c
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.25, 1]})
    rows = [("Marathi unchanged", .533, LIGHT), ("Word rules only", .406, LIGHT), ("ByT5 raw", .593, "#b0413e"), ("+ length guard", .333, GREY), ("+ all safeguards", .339, ACCENT)]
    ys = np.arange(len(rows))[::-1]
    for y, (n, v, c) in zip(ys, rows):
        hlabels(a, a.barh(y, v, .6, color=c))
    a.set_yticks(ys, [r[0] for r in rows]), a.set_xlim(0, .72)
    a.set_xlabel("Word error rate (135 held-out verses, first model)")
    a.text(-.02, 1.04, "(a)", transform=a.transAxes, fontweight="bold")
    items = [("Seen in\ntraining pairs", 71, "n = 575", ACCENT), ("Not seen", 21, "n = 152", GREY), ("Not seen, follows\ntraining patterns", 19.5, "n = 91-108", GREY)]
    bars = b.bar(range(3), [i[1] for i in items], .6, color=[i[3] for i in items])
    for bar, i in zip(bars, items):
        b.text(bar.get_x() + .3, bar.get_height() + 1.5, ("19-20%" if i[1] == 19.5 else f"{i[1]}%") + f"\n({i[2]})", ha="center", fontsize=8)
    b.set_xticks(range(3), [i[0] for i in items], fontsize=8), b.set_ylim(0, 92)
    b.set_ylabel("Word changes matching\nthe real Varhadi (%)")
    b.text(-.18, 1.04, "(b)", transform=b.transAxes, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig04_safeguards")


def fig_farm_words():  # P8, P9, P10, P13
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [3, 4]}, sharey=True)
    A = [("v3 full layer,\nno farm rules", 9.7, GREY), ("Rules only", 5.35, LIGHT), ("Rules +\nlexicon", 8.3, ACCENT)]
    B = [("v3", 10.1, GREY), ("v4", 6.4, "#b0413e"), ("v5", 11.9, GREY), ("v5 + lexicon\n(deployed)", 12.9, ACCENT)]
    for ax, data, t, tag in ((a, A, "573 farm sentences", "(a)"), (b, B, "60 sampled farm sentences", "(b)")):
        bars = ax.bar(range(len(data)), [d[1] for d in data], .6, color=[d[2] for d in data])
        for bar in bars:
            ax.text(bar.get_x() + .3, bar.get_height() + .25, f"{bar.get_height():g}%", ha="center", fontsize=8)
        ax.set_xticks(range(len(data)), [d[0] for d in data], fontsize=8)
        ax.set_xlabel(t)
        ax.text(-.02, 1.04, tag, transform=ax.transAxes, fontweight="bold")
    a.set_ylabel("Farm words changed (%)"), a.set_ylim(0, 15)
    fig.tight_layout()
    save(fig, "fig05_farm_words_changed")


def fig_register():  # P13: share of words with that ending that were changed
    labels = ["-ते → -तं", "-ावे → -ावं", "-ले → -लं"]
    scr, usr = [1, 23, 2], [82, 71, 39]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(3)
    b1, b2 = ax.bar(x - .19, scr, .36, color=GREY, label="Scripture pairs (3,221)"), ax.bar(x + .19, usr, .36, color=ACCENT, label="User datasets (993)")
    vlabels(ax, b1, "{:g}%", pad=1.2), vlabels(ax, b2, "{:g}%", pad=1.2)
    ax.set_xticks(x, labels, fontsize=10)
    ax.set_ylabel("Words with that ending that were changed (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "fig06_register_disagreement")


def fig_lexicon_sweep():  # DATA: re-runs the validation sweep of scripts/build_word_lexicon.py
    from scripts import build_word_lexicon as L
    train, val = L.uniq(L.rd(L.V5 / "train.jsonl")), L.rd(L.V5 / "val.jsonl")
    ref, base = [r["target"] for r in val], [L.server.varhadify(r["source"]) for r in val]
    cs, ss = (1, 2, 3), (.3, .5, .7, .9)
    grid, size = np.zeros((3, 4)), np.zeros((3, 4), int)
    for i, c in enumerate(cs):
        for j, s in enumerate(ss):
            lex = L.build(train, c, s)
            grid[i, j] = L.wer(ref, [L.lexicon.apply(t, lex) for t in base])[0]
            size[i, j] = len(lex["spans"])
    none = L.wer(ref, base)[0]
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    im = ax.imshow(grid, cmap="Blues_r", vmin=none - .05, vmax=none)
    for i in range(3):
        for j in range(4):
            ax.text(j, i, f"{grid[i, j]:.3f}\n({size[i, j]:,} entries)", ha="center", va="center", fontsize=8, color="white" if grid[i, j] < none - .03 else "black")
    ax.set_xticks(range(4), [f"{s:g}" for s in ss]), ax.set_yticks(range(3), [str(c) for c in cs])
    ax.set_xlabel("min_share (entry must be a replacement in this share of its occurrences)")
    ax.set_ylabel("min_count")
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, label=f"validation WER (rules only: {none:.3f})", shrink=.85)
    save(fig, "fig07_lexicon_threshold_sweep")


def fig_triage():  # DATA: data/transcripts/gold_transcripts.csv (local, not in git)
    f = ROOT / "data" / "transcripts" / "gold_transcripts.csv"
    if not f.exists():
        return print("   fig08 SKIPPED: gold_transcripts.csv not found")
    rows = [(float(r["confidence_score"]), r["triage_status"]) for r in csv.DictReader(open(f, encoding="utf-8")) if r.get("confidence_score")]
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    bins = np.linspace(.3, 1.0, 36)
    stack = [("REJECT_CANDIDATE", "#b0413e", "reject candidate"), ("NEEDS_REVIEW", LIGHT, "needs review"), ("HIGH_CONFIDENCE_CANDIDATE", ACCENT, "high-confidence candidate")]
    ax.hist([[c for c, s in rows if s == k] for k, _, _ in stack], bins=bins, stacked=True, color=[c for _, c, _ in stack],
            label=[f"{n} ({sum(s == k for _, s in rows):,})" for k, _, n in stack])
    for x in (.5, .9):
        ax.axvline(x, color="#444", lw=.8, ls=(0, (4, 3)))
    ax.text(.895, ax.get_ylim()[1] * .97, "high ≥ 0.90", fontsize=8, va="top", ha="right"), ax.text(.495, ax.get_ylim()[1] * .97, "reject < 0.50", fontsize=8, va="top", ha="right")
    ax.set_xlabel("Triage confidence of an aligned segment"), ax.set_ylabel(f"Segments (n = {len(rows):,})")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=3)
    save(fig, "fig08_triage_confidence")


def fig_datasets():  # P-data: quality of the three user sentence datasets
    names = ["Varhadi sentences\n(402)", "Agriculture queries\n(300)", "General sentences\n(600)"]
    metrics = [("Rows with clear Varhadi marker words", [50, 5, 26], ACCENT), ("Words differing from the Marathi side", [41, 3, 17], GREY), ("Rows identical to the Marathi side", [0, 0, 22], LIGHT)]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(3)
    for k, (n, v, c) in enumerate(metrics):
        vlabels(ax, ax.bar(x + (k - 1) * .26, v, .25, color=c, label=n), "{:g}%", pad=1)
    ax.set_xticks(x, names), ax.set_ylabel("Share (%)"), ax.set_ylim(0, 62)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "fig09_user_dataset_quality")


def fig_disease():  # P4
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(11.6, 3.3), gridspec_kw={"width_ratios": [1.5, 1, 1.25]})
    rows = [("Soybean: Septoria\nbrown spot", 92, "65/71"), ("Soybean: healthy", 53, "38/72"), ("Orange: canker", 96, "77/80"), ("Orange: black spot", 88, "56/64"), ("Orange: healthy", 62, "16/26")]
    ys = np.arange(len(rows))[::-1]
    for y, (n, v, k) in zip(ys, rows):
        bar = a.barh(y, v, .6, color=GREY if "healthy" in n else ACCENT)
        a.text(v + 1.5, y, f"{v}%  ({k})", va="center", fontsize=8)
    a.set_yticks(ys, [r[0] for r in rows], fontsize=8), a.set_xlim(0, 125), a.set_xlabel("Photos classified correctly (%)")
    a.text(-.02, 1.04, "(a)", transform=a.transAxes, fontweight="bold")
    th = [.45, .60, .75]
    b.plot(th, [73, 76, 78], "o-", color=GREY, label="soybean"), b.plot(th, [88, 91, 93], "o-", color=ACCENT, label="orange")
    b.axvline(.60, color="#444", lw=.8, ls=(0, (4, 3))), b.text(.605, 71, "chosen\nbar 0.60", fontsize=8)
    b.set_xticks(th), b.set_ylim(65, 100), b.set_xlabel("Confidence threshold"), b.set_ylabel("Precision on known diseases (%)")
    b.legend(frameon=False, fontsize=8, loc="lower right")
    b.text(-.2, 1.04, "(b)", transform=b.transAxes, fontweight="bold")
    items = [("Soybean, 3 unseen\ndiseases:\nanswered", 64, "97/151"), ("Orange greening:\nanswered", 100, "73/73"), ("Orange greening:\ncalled black spot", 92, "67/73")]
    bars = c.bar(range(3), [i[1] for i in items], .6, color="#b0413e")
    for bar, i in zip(bars, items):
        c.text(bar.get_x() + .3, bar.get_height() + 1.5, f"{i[1]}%\n({i[2]})", ha="center", fontsize=8)
    c.set_xticks(range(3), [i[0] for i in items], fontsize=8), c.set_ylim(0, 125), c.set_ylabel("Photos of classes the model\ndoes not know (%), at ≥ 0.45")
    c.text(-.2, 1.04, "(c)", transform=c.transAxes, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig10_disease_classifier_test")


def fig_voice():  # P7
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    d = [("Real narrator\nrecording", .263, LIGHT), ("Generic MMS\nMarathi voice", .260, GREY), ("Fine-tuned\nVarhadi voice", .360, "#b0413e")]
    vlabels(ax, ax.bar(range(3), [x[1] for x in d], .6, color=[x[2] for x in d]))
    ax.set_xticks(range(3), [x[0] for x in d], fontsize=8), ax.set_ylim(0, .44)
    ax.set_ylabel("Round-trip character error (n = 20)")
    save(fig, "fig11_voice_round_trip")


def fig_rag():  # DATA: data/rag/chunks.jsonl (local, not in git)
    f = ROOT / "data" / "rag" / "chunks.jsonl"
    if not f.exists():
        return print("   fig12 SKIPPED: data/rag/chunks.jsonl not found")
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    per = {}
    for r in rows:
        per[r["source"]] = per.get(r["source"], 0) + 1
    names = {"cicr-ipm-guide": "ICAR-CICR cotton IPM guide", "crida-pigeonpea": "ICAR-CRIDA pigeonpea guide", "pdkv-techinv-iv": "PDKV Technology Inventory IV",
             "crida-amravati": "CRIDA Amravati contingency plan", "pdkv-recom-2025": "PDKV Recommendations 2025", "pdkv-recom-2024": "PDKV Recommendations 2024",
             "vikas-soybean": "Vikaspedia soybean", "vikas-pigeonpea": "Vikaspedia pigeon pea", "vikas-mandarin": "Vikaspedia Nagpur mandarin",
             "mahapocra-crt": "MahaPoCRA climate booklet", "pdkv-contingency": "PDKV kharif contingency"}
    items = sorted(per.items(), key=lambda kv: kv[1])
    crops = [("cotton", sum("cotton" in r["crops"] for r in rows)), ("tur", sum("tur" in r["crops"] for r in rows)), ("soybean", sum("soybean" in r["crops"] for r in rows)),
             ("orange", sum("orange" in r["crops"] for r in rows)), ("no crop named", sum(not r["crops"] for r in rows))]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
    hlabels(a, a.barh(range(len(items)), [v for _, v in items], .65, color=ACCENT), "{:g}", pad=4)
    a.set_yticks(range(len(items)), [names.get(k, k) for k, _ in items], fontsize=8), a.set_xlabel(f"Chunks per document (total {len(rows):,})")
    a.text(-.02, 1.04, "(a)", transform=a.transAxes, fontweight="bold")
    bars = b.bar(range(5), [v for _, v in crops], .6, color=[ACCENT] * 4 + [LIGHT])
    vlabels(b, bars, "{:g}", pad=6)
    b.set_xticks(range(5), [c[0] for c in crops], fontsize=8, rotation=25, ha="right"), b.set_ylabel("Chunks that mention the crop")
    b.text(-.2, 1.04, "(b)", transform=b.transAxes, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig12_retrieval_corpus")


def fig_agent():  # P11 live checks; derived quota
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    d = [("Tur wilt", 23.0, 1), ("Orange fruit drop", 18.0, 2), ("Soybean sowing", 9.8, 1), ("Cotton bollworm", 12.1, 2)]
    bars = a.bar(range(4), [x[1] for x in d], .6, color=ACCENT)
    for bar, x in zip(bars, d):
        a.text(bar.get_x() + .3, bar.get_height() + .5, f"{x[1]} s\n{x[2]} search" + ("es" if x[2] > 1 else ""), ha="center", fontsize=8)
    a.set_xticks(range(4), [x[0] for x in d], fontsize=8, rotation=15), a.set_ylim(0, 30), a.set_ylabel("Seconds per answer (live check)")
    a.text(-.02, 1.04, "(a)", transform=a.transAxes, fontweight="bold")
    s = [1, 2, 3, 4]
    q = [20 // (k + 1) for k in s]
    vb = b.bar(range(4), q, .6, color=GREY)
    for bar in vb:
        b.text(bar.get_x() + .3, bar.get_height() + .2, f"{int(bar.get_height())}", ha="center", fontsize=8)
    b.set_xticks(range(4), [str(k) for k in s]), b.set_xlabel("Searches per question (requests = searches + 1)")
    b.set_ylabel("Answers per day on the free tier\n(20 requests per model; derived)"), b.set_ylim(0, 12)
    b.text(-.2, 1.04, "(b)", transform=b.transAxes, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig13_agent_latency_and_quota")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("figures ->", OUT)
    for fn in (fig_architecture, fig_scripture, fig_new_sentences, fig_safeguards, fig_farm_words, fig_register, fig_lexicon_sweep, fig_triage, fig_datasets,
               fig_disease, fig_voice, fig_rag, fig_agent):
        fn()

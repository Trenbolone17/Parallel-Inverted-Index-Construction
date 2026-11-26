#!/usr/bin/env python3
"""Charts for the presentation (deep-green theme), from inverted-index/results.

  ../inverted-index/.venv/bin/python make_charts.py      # -> charts/*.png

Sized for a 1920x1080 slide: drawn at 2x and shown at half size, so 17-19 pt text
reads as about 24-26 px on the slide. Transparent background, so it sits on any slide.
Colors: D2 green, D3 amber, Phoenix++ blue (validated: lightness, chroma, CVD);
every line is also labelled directly, since amber is light on the pale slides.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "inverted-index" / "results"
OUT = HERE / "charts"
OUT.mkdir(exist_ok=True)

INK, BODY, MUTED, GRID, AXIS = "#10261C", "#3C5247", "#6E7F75", "#D9E2D7", "#A9B8AC"
VARIANT = {"omp-dynamic": ("#1E8657", "s", "D2"), "heavy-dynamic": ("#D99A1E", "D", "D3"),
           "phoenix": ("#3F7DB5", "v", "Phoenix++")}
CORPUS = [("gutenberg", "Gutenberg", "#0E3B2C", "o"), ("wiki", "Wikipedia", "#1E8657", "s"),
          ("zipf_2M", "Synthetic Zipf", "#6FB08A", "^")]

plt.rcParams.update({
    "font.family": "Noto Sans", "font.size": 17, "axes.labelsize": 18, "axes.labelcolor": BODY,
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": BODY, "ytick.labelcolor": BODY,
    "lines.linewidth": 3.2, "lines.markersize": 9, "legend.frameon": False,
})
META = json.loads((RES / "e1_wiki.meta.json").read_text())


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"charts/{name}")


def label_ends(ax, items, gap=22, size=19):
    """Direct labels right of the line ends, nudged apart vertically (in points)."""
    ax.get_ylim()
    to_pt = 72 / ax.figure.dpi
    pts = sorted((ax.transData.transform((x, y))[1] * to_pt, x, y, t, c) for x, y, t, c in items)
    placed = []
    for y_pt, *_ in pts:
        placed.append(y_pt if not placed else max(y_pt, placed[-1] + gap))
    for (y_pt, x, y, text, color), p in zip(pts, placed):
        ax.annotate(text, (x, y), xytext=(12, p - y_pt), textcoords="offset points", va="center",
                    fontsize=size, fontweight="bold", color=INK)


def speedup():
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    p, e = META["p_cores"], META["e_cores"]
    ax.axvspan(p, p + e, color=GRID, alpha=0.45, lw=0, zorder=0)
    ax.axvspan(p + e, 24, color=GRID, alpha=0.95, lw=0, zorder=0)
    for x, text in [((p * (p + e)) ** 0.5, "+ E-cores"), (((p + e) * 24) ** 0.5, "+ HT")]:
        ax.text(x, 0.97, text, transform=ax.get_xaxis_transform(), ha="center", va="top", color=MUTED, fontsize=16)
    items = []
    for key, name, color, marker in CORPUS:
        df = pd.read_json(RES / f"e1_{key}.jsonl", lines=True)
        seq = df[df.variant == "seq"].t_build_total.median()
        m = df[df.variant == "heavy-dynamic"].groupby("threads").t_build_total.median()
        s = seq / m
        ax.plot(s.index, s.values, color=color, marker=marker, zorder=3)
        items.append((s.index[-1], s.values[-1], f"{name} {s.values[-1]:.1f}×", color))
    ax.axhline(1, color=AXIS, lw=1.2)
    ax.set_xscale("log", base=2)
    threads = [1, 2, 4, 8, 12, 16, 24]
    ax.set_xticks(threads)
    ax.set_xticklabels([str(t) for t in threads])
    ax.minorticks_off()
    ax.set_xlabel("threads (P-cores first, then E-cores, then hyperthreads)")
    ax.set_ylabel("speedup vs best sequential (×)")
    ax.set_ylim(0, 12)
    label_ends(ax, items)
    save(fig, "speedup.png")


def vocab():
    vs = [10000, 100000, 1000000, 10000000]
    frames = {v: pd.read_json(RES / f"e4_vocab_{v}.jsonl", lines=True) for v in vs}
    # shown at ~1000 px of ~2300, so larger type than speedup() to stay >= 24 px on the slide
    plt.rcParams.update({"font.size": 20, "axes.labelsize": 21})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4))
    for ax, measure in zip(axes, ["speedup", "memory"]):
        items = []
        for var, (color, marker, name) in VARIANT.items():
            ys = []
            for v in vs:
                df = frames[v]
                seq = df[df.variant == "seq"]
                r = df[(df.variant == var) & (df.threads == 24)]
                ys.append(seq.t_build_total.median() / r.t_build_total.median() if measure == "speedup"
                          else r.peak_rss_kb.median() / 1048576)
            ax.plot(vs, ys, color=color, marker=marker, zorder=3)
            items.append((vs[-1], ys[-1], name, color))
        if measure == "memory":
            seq_rss = [frames[v][frames[v].variant == "seq"].peak_rss_kb.median() / 1048576 for v in vs]
            ax.plot(vs, seq_rss, color=MUTED, ls=(0, (4, 3)), lw=2.2, zorder=2)
            items.append((vs[-1], seq_rss[-1], "sequential", MUTED))
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(vs)
        ax.set_xticklabels(["10k", "100k", "1M", "10M"])
        ax.set_xlabel("distinct words in the vocabulary")
        ax.set_ylim(bottom=0)
        label_ends(ax, items, gap=26, size=22)
    axes[0].set_ylabel("speedup at 24 threads (×)")
    axes[1].set_ylabel("peak memory at 24 threads (GB)")
    fig.tight_layout(w_pad=3)
    save(fig, "vocab.png")


if __name__ == "__main__":
    speedup()
    vocab()

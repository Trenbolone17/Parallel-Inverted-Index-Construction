#!/usr/bin/env python3
"""Figures for the report, sized for the page (170 mm text width), from inverted-index/results.

  ../inverted-index/.venv/bin/python make_figures.py      # -> figures/*.svg

Reuses the style and helpers of inverted-index/bench/plot.py (palette, phase colors, variant
colors), so the report and the per-experiment figures look the same.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "inverted-index"
RES = ROOT / "results"
sys.path.insert(0, str(ROOT / "bench"))
import plot as P  # noqa: E402  (sets the shared rcParams)

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
W = 6.69  # inches = 170 mm
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "svg.fonttype": "path", "lines.markersize": 4})

CORPORA = [("zipf_2M", "Synthetic Zipf"), ("gutenberg", "Gutenberg"), ("wiki", "Wikipedia")]
STYLES = P.variant_styles(P.KNOWN)
META = json.loads((RES / "e1_wiki.meta.json").read_text())


def load(name):
    return pd.read_json(RES / f"{name}.jsonl", lines=True)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"figures/{stem}.svg")


def core_shading(ax, tmax):
    """Thread counts that add E-cores (lighter), then P-core hyperthreads (darker); no text in small panels."""
    p, e = META["p_cores"], META["e_cores"]
    for lo, hi, alpha in [(p, p + e, 0.35), (p + e, tmax, 0.7)]:
        if min(hi, tmax) > lo:
            ax.axvspan(lo, min(hi, tmax), color=P.GRID, alpha=alpha, lw=0, zorder=0)


def threads_axis(ax, threads):
    P.thread_axis(ax, threads)
    ax.set_xlabel("threads")


def fig_legend(fig, handles, labels, n):
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.01, 1.0), ncols=n, frameon=False,
               fontsize=8, handlelength=1.8, columnspacing=1.4, borderaxespad=0)


def scaling(prefix, variants, stem):
    """Practical speedup vs threads, one panel per corpus."""
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.05), sharey=True)
    top = 0
    for ax, (c, title) in zip(axes, CORPORA):
        df = load(f"{prefix}_{c}")
        seq = df[df.variant == "seq"].t_build_total.median()
        med = df.groupby(["variant", "threads"]).t_build_total.median()
        threads = sorted(df[df.variant != "seq"].threads.unique())
        core_shading(ax, max(threads))
        best = (0, "")
        for v in variants:
            s = seq / med[v]
            col, mk = STYLES[v]
            ax.plot(s.index, s.values, color=col, marker=mk, label=v, zorder=3)
            best = max(best, (s.values[-1], v))
            top = max(top, s.max())
        ax.annotate(f"{best[0]:.1f}×", (threads[-1], best[0]), xytext=(-5, 5), textcoords="offset points",
                    ha="right", va="bottom", color=P.TEXT, fontsize=8, fontweight="bold")
        ax.axhline(1, color=P.AXIS, lw=0.8)
        threads_axis(ax, threads)
        ax.set_title(f"{title} (seq {seq:.1f} s)", loc="left")
    for ax in axes:
        ax.set_ylim(0, top * 1.12)
    axes[0].set_ylabel("practical speedup vs seq (×)")
    h, l = axes[0].get_legend_handles_labels()
    fig_legend(fig, h, l, len(l))
    fig.tight_layout(w_pad=1.2)
    save(fig, stem)


def phases():
    """Where the time goes at 24 threads (E6 run): D2, D3 and Phoenix++ per corpus."""
    variants = ["omp-dynamic", "heavy-dynamic", "phoenix"]
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.75))
    present = {}
    for ax, (c, title) in zip(axes, CORPORA):
        df = load(f"e6_{c}")
        rows = df[df.threads == 24].copy()
        for label, cols in P.PHASES:
            have = [k for k in cols if k in rows]
            rows[label] = rows[have].fillna(0).sum(axis=1) if have else 0.0
        med = rows.groupby("variant")[[p for p, _ in P.PHASES]].median()
        ys = list(range(len(variants)))[::-1]
        left = pd.Series(0.0, index=variants)
        for i, (label, _) in enumerate(P.PHASES):
            vals = med.loc[variants, label]
            if vals.max() <= 0.004:
                continue
            bars = ax.barh(ys, vals.values, left=left.values, color=P.SERIES[i], height=0.62,
                           edgecolor=P.SURFACE, linewidth=1.0, zorder=3)
            present[label] = bars
            left += vals
        for y, v in zip(ys, variants):
            ax.annotate(f"{left[v]:.2f} s", (left[v], y), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=P.TEXT_2)
        ax.set_yticks(ys)
        ax.set_yticklabels(variants if ax is axes[0] else [])
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, left.max() * 1.25)
        ax.set_xlabel("median time at 24 threads, s")
        ax.set_title(title, loc="left")
    order = [p for p, _ in P.PHASES if p in present]
    fig_legend(fig, [present[p] for p in order], order, len(order))
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_phases")


def inputs():
    """(a) term skew: D3 relative to D2; (b, c) vocabulary size: speedup and memory."""
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.1))
    # (a) skew
    ax = axes[0]
    rows = []
    for s in ["0", "0.6", "0.8", "1.0", "1.2", "1.4"]:
        df = load(f"e3_zipf_s{s}")
        m = df[df.threads == 24].groupby("variant")[["t_build_total", "peak_rss_kb"]].median()
        rows.append((float(s), *(100 * (m.loc["heavy-dynamic"] / m.loc["omp-dynamic"] - 1)).values))
    sk = pd.DataFrame(rows, columns=["s", "time", "rss"])
    ax.axhline(0, color=P.AXIS, lw=0.8)
    ax.plot(sk.s, sk.time, color=P.SERIES[0], marker="o", label="build time", zorder=3)
    ax.plot(sk.s, sk.rss, color=P.SERIES[2], marker="^", label="peak memory", zorder=3)
    ax.set_xlabel("Zipf exponent s (0 = uniform)")
    ax.set_ylabel("D3 vs D2, 24 threads (%)")
    ax.set_title("(a) Skew: D3 vs D2", loc="left")
    ax.legend(fontsize=7, loc="lower left")
    # (b, c) vocabulary
    vs = [10000, 100000, 1000000, 10000000]
    frames = {v: load(f"e4_vocab_{v}") for v in vs}
    variants = ["omp-dynamic", "heavy-dynamic", "phoenix"]
    for ax, measure in zip(axes[1:], ["speedup", "memory"]):
        for var in variants + (["seq"] if measure == "memory" else []):
            ys = []
            for v in vs:
                df = frames[v]
                seq = df[df.variant == "seq"]
                r = seq if var == "seq" else df[(df.variant == var) & (df.threads == 24)]
                ys.append(seq.t_build_total.median() / r.t_build_total.median() if measure == "speedup"
                          else r.peak_rss_kb.median() / 1048576)
            if var == "seq":
                ax.plot(vs, ys, color=P.MUTED, ls=(0, (4, 3)), lw=1, label="seq", zorder=2)
            else:
                col, mk = STYLES[var]
                ax.plot(vs, ys, color=col, marker=mk, label=var, zorder=3)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(vs)
        ax.set_xticklabels(["10k", "100k", "1M", "10M"])
        ax.set_xlabel("vocabulary size V (Zipf s = 1)")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, loc="upper right" if measure == "speedup" else "upper left")
    axes[1].set_ylabel("speedup vs seq, 24 threads (×)")
    axes[1].set_title("(b) Vocabulary: speedup", loc="left")
    axes[2].set_ylabel("peak RSS, GB")
    axes[2].set_title("(c) Vocabulary: memory", loc="left")
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig_inputs")


def theta():
    """E5: D3's threshold theta on Wikipedia, 24 threads, against D2 (dashed)."""
    df = load("e5_theta")
    r = df[df.threads == 24]
    names = ["theta 1/4", "theta 1/8", "theta 1/32", "theta 1/128", "theta 1/512"]
    m = r.groupby("variant").median(numeric_only=True)
    d2 = m.loc["omp-dynamic"]
    d3 = m.loc[names]
    x = list(range(len(names)))
    ticks = [f"{n.split()[1]}\n{int(h)}" for n, h in zip(names, d3.n_heavy_terms)]
    col, mk = STYLES["heavy-dynamic"]
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.95))
    panels = [("(a) Build time, s", [("D3", d3.t_build_total, col, mk)], d2.t_build_total, "D2"),
              ("(b) Layout and decode time, s", [("D3 layout", d3.t_layout, col, mk),
                                                ("D3 decode", d3.t_heavy_decode, P.SERIES[6], "v")],
               d2.t_layout, "D2 layout"),
              ("(c) Peak RSS, GB", [("D3", d3.peak_rss_kb / 1048576, col, mk)], d2.peak_rss_kb / 1048576, "D2")]
    for ax, (title, lines, ref, ref_label) in zip(axes, panels):
        ax.axhline(ref, color=P.MUTED, ls=(0, (4, 3)), lw=1, label=ref_label, zorder=2)
        for label, ys, c, m_ in lines:
            ax.plot(x, ys.values, color=c, marker=m_, label=label, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=7)
        ax.set_xlabel("θ (and number of heavy terms)")
        ax.set_title(title, loc="left")
        if ax is not axes[1]:
            ax.legend(fontsize=7, loc="lower left" if ax is axes[0] else "upper left")
    ax = axes[1]  # direct labels instead of a legend (three lines, room at the ends)
    kw = dict(textcoords="offset points", fontsize=7, color=P.TEXT_2)
    ax.annotate("D2 layout", (0, d2.t_layout), xytext=(0, 4), va="bottom", **kw)
    ax.annotate("D3 layout", (x[-2], d3.t_layout.iloc[-2]), xytext=(4, 7), ha="left", va="bottom", **kw)
    ax.annotate("D3 decode", (0, d3.t_heavy_decode.iloc[0]), xytext=(0, 5), va="bottom", **kw)
    axes[0].set_ylim(2.0, 2.45)
    axes[1].set_ylim(0, 0.34)
    axes[2].set_ylim(3.8, 5.0)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_theta")


if __name__ == "__main__":
    scaling("e1", ["omp", "omp-dynamic", "heavy", "heavy-dynamic"], "fig_scaling")
    phases()
    inputs()
    theta()
    scaling("e6", ["omp-dynamic", "heavy-dynamic", "phoenix"], "fig_phoenix")

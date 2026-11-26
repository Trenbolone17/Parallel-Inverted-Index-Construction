#!/usr/bin/env python3
"""Figures for the report from a bench/run.py result file.

  .venv/bin/python bench/plot.py results/<name>.jsonl [more.jsonl ...]

Writes results/figures/<name>_{time,speedup,phases,memory}.{png,pdf}:
  time     median build time vs threads (min-max whiskers), seq as a reference line
  speedup  practical speedup (vs median seq) vs threads, with the core-type regions
  phases   where the time goes: stacked phase medians per thread count (one variant)
  memory   peak RSS vs threads
Files with <= 3 thread counts (ablations, sweeps) get per-variant bar charts instead.
Sweeps across files (e3 skew, e4 vocabulary / document length) get
<sweep>_T<threads>_{time,speedup,memory}; corpus sizes (e1s_* with e1_*) get e1_sizes.

Figures are static (for the PDF report): identity is carried by color AND marker
shape AND a direct label at each line's end, so no chart relies on color alone.
Colors: the validated reference categorical palette, fixed order, light surface.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, TEXT, TEXT_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# validated reference categorical palette, fixed order, never cycled
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
# Phases keep one color everywhere; small phases fold into "other"; D3's own phases are "heavy".
# Phoenix++ runs: map / reduce (cross-thread union) / merge (sort of the reduced terms) / layout.
PHASES = [("map", ["t_map"]), ("sort", ["t_local_sort", "t_sort"]), ("merge", ["t_merge"]),
          ("layout", ["t_layout"]), ("heavy (D3)", ["t_sample", "t_heavy_setup", "t_heavy_count", "t_heavy_decode"]),
          ("other", ["t_scan", "t_splitters", "t_free", "t_split", "t_init", "t_teardown"]),
          ("reduce (Phoenix++)", ["t_reduce"])]  # own slot, distinct from merge

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": TEXT_2, "ytick.labelcolor": TEXT_2,
    "font.size": 9, "legend.frameon": False, "legend.labelcolor": TEXT_2,
    "lines.linewidth": 1.6, "lines.markersize": 5,
})


KNOWN = ["omp", "omp-dynamic", "heavy", "heavy-dynamic", "phoenix"]  # fixed slots: same color in every figure
# Corpus sizes are ordered, so they get an ordinal one-hue ramp (validated: light end >= 2:1 on the surface).
SIZE_RAMP = {25: "#84b9f3", 50: "#257de0", 100: "#0f4a96"}


def variant_styles(variants):
    """Color follows the variant, never its position: known variants keep fixed slots,
    others (ablation variants) take the following slots in command-line order."""
    styles, nxt = {}, len(KNOWN)
    for v in variants:
        if v in KNOWN:
            i = KNOWN.index(v)
        else:
            i, nxt = nxt, nxt + 1
        styles[v] = (SERIES[i % len(SERIES)], MARKERS[i % len(MARKERS)])
    return styles


def variant_order(df, meta):
    """Command-line order from the run's metadata (baseline first), else first-seen."""
    order = [v for v in (meta.get("variants") or {}) if v != "seq"]
    seen = list(dict.fromkeys(df.variant[df.variant != "seq"]))
    return [v for v in order if v in seen] + [v for v in seen if v not in order]


def core_regions(ax, meta, tmax):
    """Shade the thread counts that add E-cores, then hyperthreads (pinning order from run.py)."""
    p, e = meta.get("p_cores"), meta.get("e_cores")
    if not p:
        return
    for lo, hi, label, alpha in [(p, p + e, "+E-cores", 0.35), (p + e, tmax, "+HT", 0.7)]:
        hi = min(hi, tmax)
        if hi > lo:
            ax.axvspan(lo, hi, color=GRID, alpha=alpha, lw=0, zorder=0)
            ax.axvline(lo, color=AXIS, lw=0.6, zorder=1)
            ax.text((lo * hi) ** 0.5, 0.985, label, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", color=MUTED, fontsize=7.5)


def end_labels(ax, items, min_gap_pt=11):
    """Direct labels at line ends, nudged apart vertically so they never overlap.
    items: (x, y, text). Call after the axis limits are final."""
    ax.get_ylim()  # finalize autoscaling before converting to display coordinates
    to_pt = 72 / ax.figure.dpi
    pts = sorted((ax.transData.transform((x, y))[1] * to_pt, x, y, t) for x, y, t in items)
    placed = []
    for y_pt, *_ in pts:  # stack upward...
        placed.append(y_pt if not placed else max(y_pt, placed[-1] + min_gap_pt))
    top_pt = ax.transAxes.transform((0, 1))[1] * to_pt - min_gap_pt / 2
    shift = max(0.0, placed[-1] - top_pt) if placed else 0.0  # ...then back inside the plot
    for (y_pt, x, y, text), p in zip(pts, placed):
        ax.annotate(text, (x, y), xytext=(6, p - shift - y_pt), textcoords="offset points", va="center",
                    color=TEXT_2, fontsize=8.5)


def top_legend(ax, n):
    """Legend in its own row between the title and the plot area."""
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=n, borderaxespad=0.2, fontsize=8.5,
              handlelength=1.8, columnspacing=1.2)
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=22)  # titles are left-aligned (rcParams)


def thread_axis(ax, threads):
    ax.set_xscale("log", base=2)
    ax.set_xticks(threads)
    ax.set_xticklabels([str(t) for t in threads])
    ax.minorticks_off()
    ax.set_xlabel("threads (pinned: one per P-core first, then E-cores, then P-core hyperthreads = HT)")


def seq_line(ax, value, text, x):
    ax.axhline(value, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.annotate(text, (x, value), xytext=(0, 4), textcoords="offset points", ha="right",
                color=TEXT_2, fontsize=8.5)


def save(fig, out, stem):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out / stem}.png")


def slug(text):
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def phase_chart(rows, variant, corpus, out, stem):
    """Stacked median phase times per thread count for one variant."""
    sub = rows.copy()
    for label, cols in PHASES:  # a phase missing for this variant (e.g. D3's on D2 runs) counts as 0
        present = [c for c in cols if c in sub]
        sub[label] = sub[present].fillna(0).sum(axis=1) if present else 0.0
    ph = sub.groupby("threads")[[p for p, _ in PHASES]].median()
    shown = [(i, p) for i, (p, _) in enumerate(PHASES) if ph[p].max() > 0]  # skip phases this variant lacks
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    xs = [str(t) for t in ph.index]
    bottom = pd.Series(0.0, index=ph.index)
    for i, label in shown:  # colors stay fixed per phase even when some are skipped
        ax.bar(xs, ph[label], bottom=bottom, color=SERIES[i], width=0.62, edgecolor=SURFACE,
               linewidth=1.2, label=label, zorder=3)
        bottom += ph[label]
    for x, total in zip(xs, bottom):
        ax.annotate(f"{total:.3g}", (x, total), xytext=(0, 3), textcoords="offset points",
                    ha="center", color=TEXT_2, fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("threads")
    ax.set_ylabel("median time, s")
    ax.set_title(f"Where the time goes - {variant}, {corpus}")
    top_legend(ax, len(shown))
    ax.set_ylim(top=bottom.max() * 1.1)
    save(fig, out, stem)


def variant_charts(par, variants, threads, corpus, out, name):
    """Ablation view: one panel per thread count, a bar per variant (time; then memory).
    One measure per chart, one hue: the row label carries identity, the first variant
    (the baseline) is drawn in gray."""
    for measure, col, scale, unit, stem in [("build time", "t_build_total", 1, "s", "variants"),
                                            ("peak RSS", "peak_rss_kb", 1 / 1048576, "GB", "variants_memory")]:
        g = par.groupby(["variant", "threads"])[col]
        med, lo, hi = g.median() * scale, g.min() * scale, g.max() * scale
        fig, axes = plt.subplots(1, len(threads), figsize=(3.3 * len(threads) + 1.2, 0.42 * len(variants) + 1.4),
                                 sharey=True, sharex=True, squeeze=False)
        for ax, t in zip(axes[0], threads):
            ys = range(len(variants))[::-1]
            vals = [med[(v, t)] for v in variants]
            colors = [MUTED if i == 0 else SERIES[0] for i in range(len(variants))]
            ax.barh(list(ys), vals, color=colors, height=0.6, zorder=3,
                    xerr=[[med[(v, t)] - lo[(v, t)] for v in variants], [hi[(v, t)] - med[(v, t)] for v in variants]],
                    error_kw=dict(ecolor=TEXT_2, elinewidth=0.8, capsize=0))
            for y, v, var in zip(ys, vals, variants):  # label just past the whisker, never on it
                ax.annotate(f"{v:.3g}", (hi[(var, t)], y), xytext=(4, 0), textcoords="offset points",
                            va="center", color=TEXT_2, fontsize=8)
            ax.set_yticks(list(ys))
            ax.set_yticklabels(variants)
            ax.grid(axis="y", visible=False)
            ax.set_xlim(0, max(hi) * 1.2)  # shared across panels
            ax.set_title(f"{t} threads", loc="left", fontsize=9.5)
            ax.set_xlabel(f"{measure}, {unit} (median; whiskers min-max)")
        fig.suptitle(f"{name}: {measure} - {corpus}", x=0.02, ha="left", fontsize=11, fontweight="bold", color=TEXT)
        fig.tight_layout(w_pad=2.5)
        save(fig, out, f"{name}_{stem}")


SWEEPS = [  # file-name prefix -> (figure stem, x label, log x, what is swept)
    ("e3_zipf_s", "e3_skew", "Zipf exponent s (0 = uniform vocabulary; natural text ~ 1)", False, "term skew"),
    ("e4_vocab_", "e4_vocab", "vocabulary size V the generator draws from (Zipf s = 1, 200M tokens)", True,
     "vocabulary size"),
    ("e4_len_", "e4_len", "mean document length, tokens (total 200M tokens, V = 1M, Zipf s = 1)", True,
     "document length"),
]


def human(x):
    for div, suffix in ((1e6, "M"), (1e3, "k")):
        if x >= div:
            return f"{x / div:g}{suffix}"
    return f"{x:g}"


def plot_sweep(paths, prefix, stem, xlabel, logx, what):
    """One line per variant across a family of corpora that differ in one parameter:
    build time, practical speedup (vs seq on the same corpus) and peak memory, per thread count."""
    frames = []
    for p in paths:
        df = pd.read_json(p, lines=True)
        df["x"] = float(Path(p).stem[len(prefix):])
        frames.append(df)
    df = pd.concat(frames)
    out = Path(paths[0]).parent / "figures"
    out.mkdir(exist_ok=True)
    meta = json.loads(Path(paths[0]).with_suffix(".meta.json").read_text())
    variants = variant_order(df, meta)
    styles = variant_styles(variants)
    seq = df[df.variant == "seq"].groupby("x")
    seq_t, seq_rss = seq.t_build_total.median(), seq.peak_rss_kb.median() / 1048576
    xs = sorted(df.x.unique())
    for t in sorted(df[df.variant != "seq"].threads.unique()):
        rows = df[(df.variant != "seq") & (df.threads == t)].groupby(["variant", "x"])
        med_t, med_rss = rows.t_build_total.median(), rows.peak_rss_kb.median() / 1048576
        for measure, ylabel, fmt in [("time", "build time, s (median)", "{:.2f} s"),
                                     ("speedup", "practical speedup vs seq on the same corpus (x)", "{:.1f}x"),
                                     ("memory", "peak RSS, GB (median)", "{:.1f} GB")]:
            fig, ax = plt.subplots(figsize=(6.4, 3.8))
            labels = []
            for v in variants:
                m = {"time": med_t[v], "speedup": seq_t / med_t[v], "memory": med_rss[v]}[measure].dropna()
                c, mk = styles[v]
                ax.plot(m.index, m.values, color=c, marker=mk, label=v, zorder=3)
                labels.append((m.index[-1], m.values[-1], f"{v} " + fmt.format(m.values[-1])))
            if measure == "memory":
                ax.plot(seq_rss.index, seq_rss.values, color=MUTED, ls=(0, (4, 3)), lw=1, label="seq", zorder=2)
                labels.append((seq_rss.index[-1], seq_rss.values[-1], "seq " + fmt.format(seq_rss.values[-1])))
            if measure == "speedup":
                ax.axhline(1, color=AXIS, lw=1)
            if logx:
                ax.set_xscale("log")
                ax.minorticks_off()
            ax.set_xticks(xs)
            ax.set_xticklabels([human(x) for x in xs])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=0)
            ax.set_title(f"{what.capitalize()} vs {'build time' if measure == 'time' else measure}"
                         f" - synthetic, {t} threads")
            top_legend(ax, len(labels))
            end_labels(ax, labels)
            save(fig, out, f"{stem}_T{t}_{measure}")


def plot_sizes(paths, variant="heavy-dynamic"):
    """Practical speedup vs threads at 25 / 50 / 100% of each corpus (e1s_<corpus>_<pct> + e1_<corpus>)."""
    by_corpus = {}
    for p in paths:
        corpus, pct = Path(p).stem[len("e1s_"):].rsplit("_", 1)
        by_corpus.setdefault(corpus, {})[int(pct)] = Path(p)
    corpora = [c for c in ("zipf_2M", "gutenberg", "wiki") if c in by_corpus] + \
              [c for c in by_corpus if c not in ("zipf_2M", "gutenberg", "wiki")]
    out = Path(paths[0]).parent / "figures"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, len(corpora), figsize=(4.3 * len(corpora), 3.9), sharey=True, squeeze=False)
    panels, top = [], 0.0
    for ax, corpus in zip(axes[0], corpora):
        files = dict(by_corpus[corpus])
        full = Path(paths[0]).with_name(f"e1_{corpus}.jsonl")
        if full.exists():
            files[100] = full
        labels, threads, meta = [], [], {}
        for i, pct in enumerate(sorted(files)):
            df = pd.read_json(files[pct], lines=True)
            meta = json.loads(files[pct].with_suffix(".meta.json").read_text())
            seq_med = df[df.variant == "seq"].t_build_total.median()
            m = df[df.variant == variant].groupby("threads").t_build_total.median()
            s = seq_med / m
            threads = list(s.index)
            ax.plot(s.index, s.values, color=SIZE_RAMP.get(pct, SERIES[0]), marker=MARKERS[i], label=f"{pct}%",
                    zorder=3)
            labels.append((s.index[-1], s.values[-1], f"{pct}% {s.values[-1]:.1f}x"))
        core_regions(ax, meta, max(threads))
        ax.axhline(1, color=AXIS, lw=1)
        thread_axis(ax, threads)
        ax.set_xlabel("threads")
        ax.set_title(corpus, loc="left", fontsize=9.5)
        panels.append((ax, labels))
        top = max(top, max(y for _, y, _ in labels))
    for ax, labels in panels:  # shared y: set once from all panels, then place labels
        ax.set_ylim(0, top * 1.08)
        end_labels(ax, labels)
    axes[0][0].set_ylabel(f"practical speedup of {variant} vs seq (x)")
    axes[0][0].legend(loc="upper left", title="share of the corpus", title_fontsize=8, fontsize=8.5)
    fig.suptitle("Corpus size vs speedup - every 4th / 2nd / every document", x=0.02, ha="left",
                 fontsize=11, fontweight="bold", color=TEXT)
    fig.tight_layout(w_pad=3)
    save(fig, out, "e1_sizes")


def plot(path):
    path = Path(path)
    name = path.stem
    df = pd.read_json(path, lines=True)
    meta_path = path.with_name(name + ".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    out = path.parent / "figures"
    out.mkdir(exist_ok=True)
    corpus = Path(meta.get("corpus", name)).name
    seq_rows = df[df.variant == "seq"]
    seq_med = seq_rows.t_build_total.median() if len(seq_rows) else None
    par = df[df.variant != "seq"]
    variants = variant_order(df, meta)
    styles = variant_styles(variants)
    threads = sorted(par.threads.unique())
    g = par.groupby(["variant", "threads"]).t_build_total
    med, lo, hi = g.median(), g.min(), g.max()

    for v in variants:  # where the time goes, one chart per parallel variant
        phase_chart(par[par.variant == v], v, corpus, out, f"{name}_phases_{slug(v)}")
    if len(variants) >= 2 and len(threads) <= 3:  # ablation-style file: compare variants directly
        variant_charts(par, variants, threads, corpus, out, name)
    if len(threads) < 3:
        return

    # ---- time
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    core_regions(ax, meta, max(threads))
    labels = []
    for v in variants:
        c, m = styles[v]
        x = med[v].index
        ax.errorbar(x, med[v].values, yerr=[med[v] - lo[v], hi[v] - med[v]], color=c, marker=m,
                    capsize=0, elinewidth=1, label=v, zorder=3)
        labels.append((x[-1], med[v].values[-1], f"{v} {med[v].values[-1]:.3g} s"))
    if seq_med is not None:
        seq_line(ax, seq_med, f"seq (best sequential) {seq_med:.3g} s", max(threads))
    thread_axis(ax, threads)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("build time, s (median; whiskers min-max)")
    ax.set_title(f"Index build time - {corpus}")
    top_legend(ax, len(variants))
    end_labels(ax, labels)
    save(fig, out, f"{name}_time")

    # ---- practical speedup
    if seq_med is not None:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        core_regions(ax, meta, max(threads))
        labels = []
        for v in variants:
            c, m = styles[v]
            s = seq_med / med[v]
            ax.plot(s.index, s.values, color=c, marker=m, label=v, zorder=3)
            labels.append((s.index[-1], s.values[-1], f"{v} {s.values[-1]:.1f}x"))
        ax.axhline(1, color=AXIS, lw=1)
        thread_axis(ax, threads)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("practical speedup vs seq (x)")
        ax.set_title(f"Practical speedup - {corpus}")
        top_legend(ax, len(variants))
        end_labels(ax, labels)
        save(fig, out, f"{name}_speedup")

    # ---- memory
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    core_regions(ax, meta, max(threads))
    rss = par.groupby(["variant", "threads"]).peak_rss_kb.median() / 1048576
    labels = []
    for v in variants:
        c, m = styles[v]
        ax.plot(rss[v].index, rss[v].values, color=c, marker=m, label=v, zorder=3)
        labels.append((rss[v].index[-1], rss[v].values[-1], f"{v} {rss[v].values[-1]:.2f} GB"))
    if len(seq_rows):
        s_rss = seq_rows.peak_rss_kb.median() / 1048576
        seq_line(ax, s_rss, f"seq {s_rss:.2f} GB", max(threads))
    input_gb = meta.get("corpus_bytes", 0) / 1e9
    thread_axis(ax, threads)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("peak RSS, GB")
    ax.set_title(f"Peak memory - {corpus} ({input_gb:.2f} GB input)")
    top_legend(ax, len(variants))
    end_labels(ax, labels)
    save(fig, out, f"{name}_memory")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for prefix, stem, xlabel, logx, what in SWEEPS:
        files = [p for p in sys.argv[1:] if Path(p).name.startswith(prefix)]
        if len(files) >= 2:
            print(f"{stem} sweep")
            plot_sweep(files, prefix, stem, xlabel, logx, what)
    sizes = [p for p in sys.argv[1:] if Path(p).name.startswith("e1s_")]
    if sizes:
        print("corpus sizes")
        plot_sizes(sizes)
    for p in sys.argv[1:]:
        print(p)
        plot(p)

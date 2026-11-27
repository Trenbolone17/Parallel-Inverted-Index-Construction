#!/usr/bin/env python3
"""Benchmark harness: pinned, repeated, shuffled runs of bin/invidx.

Measurement protocol (README.md, section Benchmarks):
  - Each run is a fresh process, so peak RSS is clean per configuration.
  - One untimed warm-up run per corpus loads it into the page cache; load I/O is
    timed separately by invidx anyway.
  - Every repetition runs all configurations in a new random order, so slow drift
    (heat, background load) spreads across variants instead of biasing one.
  - Threads are pinned with OMP_PLACES / OMP_PROC_BIND=close in this order:
    one hyperthread per P-core, then E-cores, then the P-cores' second hyperthreads.
    T <= #P-cores therefore uses only P-cores, one per physical core.
  - Every run of the same corpus must produce the same index checksum; a mismatch
    is reported and makes the script exit non-zero.

Raw results: results/<name>.jsonl (one JSON object per run, appended as it goes).
Summary:     results/<name>_summary.csv (medians, min/max, speedups) + table on stdout.
Machine:     results/<name>.meta.json (CPU, kernel, compiler, governor, corpus info).

Example:
  .venv/bin/python bench/run.py --name wiki_scaling --corpus data/wiki.txt \\
      --variants seq,omp,omp-dynamic --threads 1,2,4,8,12,16,24 --reps 5
"""
import argparse
import datetime
import json
import os
import random
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INVIDX = ROOT / "bin" / "invidx"

# name -> invidx arguments. Add more with --variant NAME="ARGS".
VARIANTS = {
    "seq": "--algo seq",
    "omp": "--algo omp",
    "omp-dynamic": "--algo omp --schedule dynamic",
    "heavy": "--algo heavy",
    "heavy-dynamic": "--algo heavy --schedule dynamic",
    "phoenix": "bin/phoenix_index",  # Phoenix++ reference (make phoenix); a leading bin/ path picks the binary
}
SEQUENTIAL = {"seq", "naive"}  # run once, at T = 1


def cpu_list(text):
    cpus = []
    for part in text.strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def read(path, default=""):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def placement_order():
    """CPU IDs in pinning order: first hyperthread of each P-core, E-cores, other hyperthreads."""
    all_cpus = cpu_list(read("/sys/devices/system/cpu/online", f"0-{os.cpu_count() - 1}"))
    p_cores = set(cpu_list(read("/sys/devices/cpu_core/cpus"))) or set(all_cpus)  # non-hybrid: all "P"
    firsts, seconds, e_cores = [], [], []
    for c in all_cpus:
        siblings = cpu_list(read(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list", str(c)))
        if c not in p_cores:
            e_cores.append(c)
        elif c == min(siblings):
            firsts.append(c)
        else:
            seconds.append(c)
    return firsts + e_cores + seconds, len(firsts), len(e_cores)


def machine_info():
    def cmd(*args):
        try:
            return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    model = next((l.split(":", 1)[1].strip() for l in read("/proc/cpuinfo").splitlines() if l.startswith("model name")), "")
    ac = [read(p / "online") for p in Path("/sys/class/power_supply").glob("A*") if (p / "online").exists()]
    return {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "cpu_model": model,
        "kernel": cmd("uname", "-r"),
        "compiler": cmd("g++", "--version").splitlines()[0] if cmd("g++", "--version") else "",
        "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "power_profile": cmd("powerprofilesctl", "get"),  # empty if its python3 lacks `gi` (e.g. venv on PATH)
        "platform_profile": read("/sys/firmware/acpi/platform_profile"),
        "energy_perf_pref": read("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"),
        "no_turbo": read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
        "on_ac_power": ac[0] == "1" if ac else None,
        "loadavg_at_start": os.getloadavg()[0],
        "invidx_mtime": datetime.datetime.fromtimestamp(INVIDX.stat().st_mtime).isoformat(timespec="seconds"),
    }


def run_once(variant, args, threads, corpus, order, stats_path):
    env = dict(os.environ)
    env.pop("OMP_NUM_THREADS", None)
    places = ",".join(f"{{{c}}}" for c in order[:threads])
    env.update(OMP_PLACES=places, OMP_PROC_BIND="close")
    argv = shlex.split(args)
    exe = str(ROOT / argv.pop(0)) if argv and argv[0].startswith("bin/") else str(INVIDX)
    cmd = [exe, *argv, "--threads", str(threads), "--input", corpus, "--stats", stats_path, "--quiet"]
    subprocess.run(cmd, env=env, check=True)
    s = json.loads(Path(stats_path).read_text())
    row = {"variant": variant, "args": args, "threads": threads, "places": places,
           "peak_rss_kb": s["peak_rss_kb"], "checksum": s["info"].get("checksum", "")}
    row.update({f"t_{k}": v for k, v in s["phases_s"].items()})
    row.update({f"n_{k}": v for k, v in s["counters"].items()})
    for k, per_thread in s["thread_phases_s"].items():  # load imbalance per phase
        mean = sum(per_thread) / len(per_thread) if per_thread else 0
        row[f"imb_{k}"] = max(per_thread) / mean if mean > 0 else 1.0
    return row


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["variant"], r["threads"]), []).append(r["t_build_total"])
    med = {k: statistics.median(v) for k, v in groups.items()}
    seq = next((m for (v, t), m in med.items() if v in SEQUENTIAL), None)
    out = []
    for (v, t) in sorted(groups, key=lambda k: (k[0] not in SEQUENTIAL, k[0], k[1])):
        times = groups[(v, t)]
        one = med.get((v, 1))
        out.append({
            "variant": v, "threads": t, "runs": len(times),
            "median_s": round(med[(v, t)], 4), "min_s": round(min(times), 4), "max_s": round(max(times), 4),
            "scaling_speedup": round(one / med[(v, t)], 3) if one else "",
            "practical_speedup": round(seq / med[(v, t)], 3) if seq else "",
        })
    return out


def main():
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up line by line even through `| tee`
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="results/<name>.jsonl etc.")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--variants", default="seq,omp", help=f"comma list of {sorted(VARIANTS)}")
    ap.add_argument("--variant", action="append", default=[], metavar='NAME="ARGS"',
                    help='extra variant, e.g. omp-r8="--algo omp --ranges-per-thread 8"')
    ap.add_argument("--threads", default="1,2,4,8,12,16,24")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--cooldown", type=float, default=0.0, help="seconds to sleep between runs")
    ap.add_argument("--seed", type=int, default=12345, help="shuffle seed (recorded)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    variants = {}
    for v in filter(None, args.variants.split(",")):
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v}; known: {sorted(VARIANTS)} (or use --variant)")
        variants[v] = VARIANTS[v]
    for spec in args.variant:
        name, _, a = spec.partition("=")
        variants[name] = a
    threads = [int(t) for t in args.threads.split(",")]
    order, n_p, n_e = placement_order()
    if max(threads) > len(order):
        raise SystemExit(f"{max(threads)} threads requested, {len(order)} CPUs online")

    configs = [(v, a, 1 if v in SEQUENTIAL else t) for v, a in variants.items()
               for t in ([1] if v in SEQUENTIAL else threads)]
    print(f"{len(configs)} configurations x {args.reps} reps = {len(configs) * args.reps} runs on {args.corpus}")
    print(f"pinning order: {n_p} P-cores (1 thread each), {n_e} E-cores, then hyperthreads: {order}")
    if args.dry_run:
        for c in configs:
            print("  ", c)
        return

    load = os.getloadavg()[0]
    if load > 1.5:
        print(f"WARNING: load average is {load:.1f}; something else is running and will distort timings",
              file=sys.stderr)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    raw_path = results / f"{args.name}.jsonl"
    meta = machine_info()
    if meta["on_ac_power"] is False:
        print("WARNING: running on battery; plug in for stable clocks", file=sys.stderr)
    corpus_meta = Path(args.corpus + ".meta.json")
    meta.update(name=args.name, corpus=args.corpus, corpus_bytes=Path(args.corpus).stat().st_size,
                corpus_meta=json.loads(corpus_meta.read_text()) if corpus_meta.exists() else None,
                variants=variants, threads=threads, reps=args.reps, seed=args.seed,
                cooldown_s=args.cooldown, pinning_order=order, p_cores=n_p, e_cores=n_e)
    (results / f"{args.name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    rng = random.Random(args.seed)
    rows, checksums = [], set()
    with tempfile.TemporaryDirectory() as tmp, open(raw_path, "w") as raw:
        stats_path = str(Path(tmp) / "stats.json")
        print("warm-up run (page cache)...")
        run_once("warmup", "--algo seq", 1, args.corpus, order, stats_path)
        total, done = len(configs) * args.reps, 0
        t_start = time.time()
        for rep in range(1, args.reps + 1):
            for v, a, t in rng.sample(configs, len(configs)):
                row = run_once(v, a, t, args.corpus, order, stats_path)
                row.update(name=args.name, corpus=args.corpus, rep=rep)
                raw.write(json.dumps(row) + "\n")
                raw.flush()
                rows.append(row)
                checksums.add(row["checksum"])
                done += 1
                elapsed = time.time() - t_start
                eta = elapsed / done * (total - done)
                print(f"  [{done}/{total}] {args.name} rep {rep} {v:<14} T={t:<3} {row['t_build_total']:8.3f}s  "
                      f"rss {row['peak_rss_kb'] / 1048576:5.2f} GB  ({elapsed / 60:.0f} min elapsed, "
                      f"~{eta / 60:.0f} min left in this experiment)")
                if args.cooldown:
                    time.sleep(args.cooldown)

    summary = summarize(rows)
    cols = list(summary[0].keys())
    with open(results / f"{args.name}_summary.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for s in summary:
            f.write(",".join(str(s[c]) for c in cols) + "\n")
    print(f"\n{'variant':<16}{'T':>4}{'median s':>10}{'min':>9}{'max':>9}{'scaling':>9}{'practical':>10}")
    for s in summary:
        sc = f"{s['scaling_speedup']:.2f}x" if s["scaling_speedup"] != "" else ""
        pr = f"{s['practical_speedup']:.2f}x" if s["practical_speedup"] != "" else ""
        print(f"{s['variant']:<16}{s['threads']:>4}{s['median_s']:>10.3f}{s['min_s']:>9.3f}{s['max_s']:>9.3f}{sc:>9}{pr:>10}")
    print(f"\nraw: {raw_path}\nsummary: {results / (args.name + '_summary.csv')}")
    if len(checksums) != 1:
        print(f"ERROR: runs produced {len(checksums)} different index checksums: {sorted(checksums)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

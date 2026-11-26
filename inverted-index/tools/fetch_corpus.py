#!/usr/bin/env python3
"""Download a Hugging Face parquet corpus shard by shard and write it pretokenized.

Presets:
  gutenberg  emozilla/pg19 (PG-19: Project Gutenberg books published before 1919,
             license header/footer already stripped); one book = one document.
  wiki       wikimedia/wikipedia 20231101.en; the first paragraph of each article
             = one document (empty documents are kept so IDs map 1:1 to articles).

Each shard is downloaded into --raw-dir (with retries and resume), converted, then
deleted unless --keep-raw, so at most one shard (~300 MB) is on disk at a time.
Tokenization is identical to tools/pretokenize.py (same function). A sidecar
<out>.meta.json records the source, shards, counts and time for reproducibility.

Examples (run with the project venv, which has pyarrow):
  .venv/bin/python -I tools/fetch_corpus.py gutenberg --split test --out data/gutenberg_test.txt
  .venv/bin/python -I tools/fetch_corpus.py gutenberg --shards 4 --out data/gutenberg.txt
  .venv/bin/python -I tools/fetch_corpus.py wiki --shards 41 --out data/wiki.txt
"""
import argparse
import datetime
import http.client
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # our own tools/ dir (needed under -I)
from pretokenize import tokenize  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402

PRESETS = {
    "gutenberg": dict(dataset="emozilla/pg19", config="default", split="train", first_paragraph=False, batch=32),
    "wiki": dict(dataset="wikimedia/wikipedia", config="20231101.en", split="train", first_paragraph=True, batch=8192),
}
API = "https://datasets-server.huggingface.co/parquet?dataset={}"
UA = {"User-Agent": "corpus-fetch/1.0"}


def get_json(url, retries=8):
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as e:
            if attempt == retries:
                raise SystemExit(f"request failed after {retries} attempts: {url}: {e}")
            print(f"    request retry {attempt}/{retries}: {e}", file=sys.stderr)
            time.sleep(min(30, 2 ** attempt))


def list_shards(dataset, config, split):
    files = get_json(API.format(dataset))["parquet_files"]
    shards = [f for f in files if f["config"] == config and f["split"] == split]
    if not shards:
        raise SystemExit(f"no parquet files for {dataset} config={config} split={split}")
    return sorted(shards, key=lambda f: f["filename"])


def download(url, dest, size, retries=40):
    """Download with resume (HTTP Range) and retries. The HF CDN sometimes resets every
    connection for minutes at a time, so back off up to a minute per attempt."""
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if size and have == size:
            return
        headers = dict(UA)
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                with open(dest, mode) as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
            if not size or dest.stat().st_size == size:
                return
        except (OSError, http.client.HTTPException) as e:
            print(f"    download retry {attempt}/{retries}: {e}", file=sys.stderr)
            time.sleep(min(60, 2 ** attempt) + random.random())
    raise SystemExit(f"download failed after {retries} attempts: {url}\n"
                     "Re-run the same command to resume after the last completed shard.")


def first_paragraph(text):
    text = text.lstrip()
    cut = text.find("\n\n")
    return text if cut < 0 else text[:cut]


def texts(path, batch, lead_only):
    """Yields lists of document texts, one parquet batch at a time (bounded memory)."""
    for b in pq.ParquetFile(path).iter_batches(batch_size=batch, columns=["text"]):
        docs = b.column(0).to_pylist()
        yield [first_paragraph(t or "") for t in docs] if lead_only else [t or "" for t in docs]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", choices=sorted(PRESETS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", type=int, help="use only the first N shards (default: all)")
    ap.add_argument("--split", help="override the preset's split (e.g. test, validation)")
    ap.add_argument("--max-docs", type=int, help="stop after N documents")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()

    p = PRESETS[args.preset]
    split = args.split or p["split"]
    shards = list_shards(p["dataset"], p["config"], split)[: args.shards]
    raw_dir = Path(args.raw_dir) / args.preset
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_mb = sum(s["size"] for s in shards) / 1e6
    print(f"{p['dataset']} {p['config']}/{split}: {len(shards)} shard(s), {total_mb:.0f} MB to download", file=sys.stderr)

    # Resume support: after every completed shard, <out>.progress.json records the shards
    # done and the output size. A re-run with the same arguments truncates the output to
    # that size (dropping any half-written shard) and continues with the next shard.
    progress_path = Path(str(out_path) + ".progress.json")
    run_key = {"preset": args.preset, "dataset": p["dataset"], "config": p["config"], "split": split,
               "max_docs": args.max_docs}
    state = {**run_key, "shards_done": [], "docs": 0, "tokens": 0, "bytes": 0, "tokenize_seconds": 0.0}
    if progress_path.exists():
        saved = json.loads(progress_path.read_text())
        if ({k: saved.get(k) for k in run_key} == run_key and out_path.exists()
                and out_path.stat().st_size >= saved["bytes"]):
            state = saved
            print(f"resuming after {len(state['shards_done'])} completed shard(s), {state['docs']} docs",
                  file=sys.stderr)
        else:
            print("progress file does not match this run; starting over", file=sys.stderr)

    def save_progress():
        tmp = progress_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        os.replace(tmp, progress_path)

    def limit_reached():
        return args.max_docs is not None and state["docs"] >= args.max_docs

    t_start = time.time()
    with open(out_path, "r+b" if state["shards_done"] else "wb") as out, Pool(args.workers) as pool:
        out.truncate(state["bytes"])
        out.seek(state["bytes"])
        for i, s in enumerate(shards, 1):
            if s["filename"] in state["shards_done"]:
                continue
            if limit_reached():
                break
            dest = raw_dir / f"{split}-{s['filename']}"
            t0 = time.time()
            download(s["url"], dest, s["size"])
            t1 = time.time()
            shard_docs = shard_tokens = 0
            for docs in texts(dest, p["batch"], p["first_paragraph"]):
                if args.max_docs is not None:
                    docs = docs[: max(0, args.max_docs - state["docs"] - shard_docs)]
                chunk = max(1, len(docs) // (4 * (pool._processes or 1)))
                for toks in pool.imap(tokenize, docs, chunksize=chunk):
                    out.write(toks.encode() + b"\n")
                    shard_tokens += toks.count(" ") + 1 if toks else 0
                shard_docs += len(docs)
                if args.max_docs is not None and state["docs"] + shard_docs >= args.max_docs:
                    break
            out.flush()
            os.fsync(out.fileno())
            t2 = time.time()
            state["shards_done"].append(s["filename"])
            state["docs"] += shard_docs
            state["tokens"] += shard_tokens
            state["bytes"] = out.tell()
            state["tokenize_seconds"] = round(state["tokenize_seconds"] + t2 - t1, 1)
            save_progress()
            if not args.keep_raw:
                dest.unlink()
            print(f"  shard {i}/{len(shards)} {s['filename']}: {shard_docs} docs "
                  f"(download {t1 - t0:.0f}s, tokenize {t2 - t1:.0f}s); total {state['docs']} docs, "
                  f"{state['tokens'] / 1e6:.1f}M tokens, {state['bytes'] / 1e9:.2f} GB", file=sys.stderr)

    meta = {
        **run_key,
        "document": "first paragraph of each article" if p["first_paragraph"] else "whole book",
        "shards": state["shards_done"], "docs": state["docs"], "tokens": state["tokens"],
        "bytes": state["bytes"], "tokenize_seconds": state["tokenize_seconds"],
        "last_run_seconds": round(time.time() - t_start, 1), "created": datetime.date.today().isoformat(),
    }
    Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    progress_path.unlink(missing_ok=True)
    print(f"wrote {out_path}: {state['docs']} docs, {state['tokens'] / 1e6:.1f}M tokens, "
          f"{state['bytes'] / 1e9:.2f} GB", file=sys.stderr)


if __name__ == "__main__":
    main()

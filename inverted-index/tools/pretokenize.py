#!/usr/bin/env python3
"""Convert raw text into the pretokenized corpus format used by invidx.

Output: one document per line; tokens are lowercase runs of Unicode letters/digits
joined by single spaces. Punctuation and underscores are dropped. This runs once,
offline; its time is reported separately from index construction.

Inputs (choose one):
  --dir DIR [--glob '*.txt']   each file is one document (sorted by path)
  --jsonl FILE [--field text]  each JSON line is one document
  --lines FILE                 each line is one document

(Named pretokenize.py, not tokenize.py, so it cannot shadow the stdlib tokenize module.)
"""
import argparse
import functools
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

TOKEN = re.compile(r"[^\W_]+")


def tokenize(text, first_paragraph=False):
    if first_paragraph:
        text = text.strip().split("\n\n", 1)[0]
    return " ".join(TOKEN.findall(text.lower()))


def docs_from_dir(path, pattern):
    for f in sorted(Path(path).rglob(pattern)):
        yield f.read_text(encoding="utf-8", errors="replace")


def docs_from_jsonl(path, field):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)[field]


def docs_from_lines(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            yield line


def limited(it, n):
    for i, x in enumerate(it):
        if n is not None and i >= n:
            return
        yield x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir")
    src.add_argument("--jsonl")
    src.add_argument("--lines")
    ap.add_argument("--glob", default="*.txt")
    ap.add_argument("--field", default="text")
    ap.add_argument("--first-paragraph", action="store_true", help="keep only text before the first blank line")
    ap.add_argument("--max-docs", type=int)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.dir:
        docs = docs_from_dir(args.dir, args.glob)
    elif args.jsonl:
        docs = docs_from_jsonl(args.jsonl, args.field)
    else:
        docs = docs_from_lines(args.lines)
    docs = limited(docs, args.max_docs)

    fn = functools.partial(tokenize, first_paragraph=args.first_paragraph)
    n_docs = n_tokens = 0
    with open(args.out, "w", encoding="utf-8", newline="\n") as out, Pool(args.workers) as pool:
        for toks in pool.imap(fn, docs, chunksize=64):
            out.write(toks)
            out.write("\n")
            n_docs += 1
            n_tokens += toks.count(" ") + 1 if toks else 0
    print(f"pretokenize: docs={n_docs} tokens={n_tokens} -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

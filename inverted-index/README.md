# Parallel inverted index construction (CS F422, Project 23)

This project builds word → document posting lists from a pretokenized corpus, in C++ with OpenMP. The full plan, schedule and experiment design are in [../PLAN.md](../PLAN.md).

## Build and test

```sh
make            # bin/invidx, bin/idxcheck, bin/gen_zipf  (g++ -O3 -march=native -fopenmp)
make test       # hand-checked cases + 200 randomized differential tests vs the naive oracle
make asan       # same tests under AddressSanitizer + UBSan
make tsan       # ThreadSanitizer via clang; first: sudo pacman -S openmp
make phoenix    # Phoenix++ reference: clones the authors' repo (pinned) into third_party/, builds bin/phoenix_index
ALGOS=phoenix tests/run_tests.sh   # the same tests for the Phoenix++ build
```

To change the hand-checked cases, edit `tests/make_tiny.py`, then run `python3 tests/make_tiny.py`. The expected outputs in that file are written by hand, not computed.

## Usage

```sh
# Synthetic corpus: 1M docs, vocabulary 100k, Zipf s = 1.0, mean length 100 (lognormal)
bin/gen_zipf --docs 1000000 --vocab 100000 --zipf 1.0 --len 100 --seed 1 --out data/zipf.txt

# Build an index: naive = test oracle, seq = D1 best sequential, omp = D2
bin/invidx --algo seq --input data/zipf.txt --out data/zipf.bin --stats results/seq.json
OMP_PLACES=cores OMP_PROC_BIND=close bin/invidx --algo omp --threads 8 --input data/zipf.txt --out data/zipf_omp.bin
bin/invidx --algo omp --schedule dynamic --threads 24 --input data/zipf.txt   # dynamic doc chunks (hybrid CPU)

# D3 = omp + heavy-term bitmaps; all omp options apply too
bin/invidx --algo heavy --schedule dynamic --threads 24 --input data/wiki.txt \
    --theta 0.03125 --sample 0.01 --heavy-mem-mb 1024    # (these are the defaults)

# Phoenix++ reference (the starting paper's runtime) building the same index; same options and stats
OMP_PLACES="{0},{2},{4},{6}" bin/phoenix_index --threads 4 --input data/wiki.txt --out data/wiki_phx.bin

# Check results
bin/idxcheck a.bin b.bin                          # identical? (exit 0 / 1)
bin/idxcheck --validate a.bin --corpus zipf.txt   # invariants + independent recount
bin/idxcheck --dump a.bin | head                  # term<TAB>df<TAB>doc ids
bin/idxcheck --info a.bin                         # sizes + checksum

# Real corpora: download parquet shards from Hugging Face, one at a time, and write them
# pretokenized (needs the venv: uv venv .venv && uv pip install --python .venv/bin/python pyarrow numpy pandas matplotlib)
.venv/bin/python -I tools/fetch_corpus.py gutenberg --split test --out data/gutenberg_test.txt  # 100 books, 25 MB
.venv/bin/python -I tools/fetch_corpus.py gutenberg --shards 4 --out data/gutenberg.txt         # ~5k books
.venv/bin/python -I tools/fetch_corpus.py wiki --shards 41 --out data/wiki.txt                  # ~6.4M lead paragraphs

# Your own raw text -> pretokenized corpus (one doc per file, per JSON line, or per line)
python3 tools/pretokenize.py --dir books/ --out data/books.txt
```

`--stats` writes JSON with the following:
- phase times: `load`, `scan`, `map`, `sort`, `layout`, `free`, `build_total` and `write`;
- work counters: tokens, hash probes, docs, terms and postings;
- peak RSS and the `OMP_PLACES`/`OMP_PROC_BIND` settings.

`build_total` covers everything from the document-boundary scan up to a finished in-memory index. File loading and writing are reported separately.

## Benchmarks

```sh
# Pinned, repeated (default 5), shuffled-order runs -> results/<name>.jsonl, _summary.csv, .meta.json
.venv/bin/python bench/run.py --name wiki_scaling --corpus data/wiki.txt \
    --variants seq,omp,omp-dynamic --threads 1,2,4,8,12,16,24 --reps 5
.venv/bin/python bench/run.py ... --variant omp-r8="--algo omp --ranges-per-thread 8"   # ad-hoc variant
.venv/bin/python bench/plot.py results/wiki_scaling.jsonl    # -> results/figures/*_{time,speedup,phases,memory}.{png,pdf}

# The report's experiments (see the header of the script for each one's contents and run time)
bench/experiments.sh e1 sizes e3 e4 e5 e6 plots      # or: bench/experiments.sh all
```

| experiment | what varies | variants | results |
|---|---|---|---|
| `e1` | threads 1–24, 3 corpora | seq, omp, omp-dynamic, heavy, heavy-dynamic | `e1_<corpus>` |
| `sizes` | 25 / 50% of each corpus (every 4th / 2nd document) | seq, omp-dynamic, heavy-dynamic | `e1s_<corpus>_<pct>`, figure `e1_sizes` |
| `e3` | Zipf exponent s = 0 … 1.4 | seq, omp-dynamic, heavy-dynamic | `e3_zipf_s<s>`, figures `e3_skew_*` |
| `e4` | vocabulary 10⁴–10⁷; mean document length 10–10⁴ tokens | seq, omp-dynamic, heavy-dynamic, phoenix | `e4_vocab_<V>`, `e4_len_<L>`, figures `e4_vocab_*`, `e4_len_*` |
| `e5` | D3's θ, sample rate, memory budget (Wikipedia) | omp-dynamic + D3 settings | `e5_theta`, `e5_sample`, `e5_budget` |
| `e6` | threads 1–24, 3 corpora | seq, omp-dynamic, heavy-dynamic, phoenix | `e6_<corpus>` |


How the harness measures:
- **Pinning:** threads are placed one per P-core first, then on E-cores, then on the P-cores' second hyperthreads. The order is read from sysfs and recorded in `.meta.json`.
- **Clean runs:** every run is a fresh process, after one untimed warm-up run that loads the corpus into the page cache.
- **Checksum check:** all runs on a corpus must produce the same index checksum, or the script exits non-zero.
- **Warnings:** the harness warns if the load average is high or the laptop is on battery.

Run benchmarks only on AC power with nothing else running, including the dataset download. Never mix results from different machines on one curve.

## Input and output contract

**Input** (details in `include/corpus.hpp`):
- One document per line; the document ID is the line number.
- Empty lines are empty documents and still use an ID.
- Tokens are runs of non-whitespace bytes.

**Output** (details in `include/index_format.hpp`):
- A canonical binary index.
- Terms are in byte order, and posting lists are strictly increasing.
- Two builds are correct relative to each other only if their files are byte-identical.

## Layout

```
include/   hash.hpp, hash_table.hpp, local_index.hpp (inner loop shared by seq and omp), corpus.hpp,
           index_format.hpp, pod_vector.hpp, stats.hpp, builders.hpp
src/       invidx.cpp (CLI), build_naive.cpp (oracle), build_seq.cpp (D1), build_omp.cpp (D2 + D3),
           corpus.cpp, index_format.cpp, stats.cpp, idxcheck.cpp
tools/     gen_zipf.cpp, pretokenize.py, fetch_corpus.py
bench/     run.py (benchmark harness), plot.py (report figures), experiments.sh (all report experiments)
baselines/ phoenix_index.cpp (our index as a Phoenix++ application; built by `make phoenix`)
third_party/ Phoenix++ checkout (not committed; `make phoenix` clones it)
tests/     make_tiny.py -> tiny/*.txt + *.expected, run_tests.sh
```

## Status

- [x] Contract, canonical format, `idxcheck`, `naive` oracle, hand-checked and differential tests
- [x] D1: `seq` builder (and `gen_zipf`, `pretokenize.py`)
- [x] Dataset fetch script (Gutenberg, Wikipedia lead paragraphs), tested on the PG-19 test split and 1 Wikipedia shard
- [x] Full dataset download: Gutenberg (4,976 books, 1.73 GB) and Wikipedia (6.4M lead paragraphs, 1.68 GB)
- [x] E1 strong scaling on all three corpora (balanced power profile; rerun with the performance profile for the report)
- [x] D2: `omp` (local dictionaries + sorted range merge; static or dynamic schedule)
- [x] Benchmark harness (`bench/run.py`: pinning, repetitions, shuffled order, checksum guard; `bench/plot.py`: figures)
- [x] D3: `heavy` (heavy-term bitmaps under a memory budget), with tests for all-heavy, budget-capped, none-heavy and multi-segment bitmaps
- [x] Final run of E1, E3 and E5 on a quiet machine with the performance profile (735 runs, all checksums identical)
- [x] Phoenix++ reference (`make phoenix`): same index, passes the hand-written and differential tests
- [x] Corpus sizes, E4 (vocabulary and document-length sweeps) and E6 (Phoenix++ vs D2/D3): 1,060 runs, all checksums identical
- [x] Report draft: `../report/report.pdf` (build with `../report/build.sh`); fill in the highlighted placeholders
- [ ] Presentation, demo

To add a builder:
1. Declare it in `include/builders.hpp`.
2. Add a branch in `src/invidx.cpp`.
3. Add the source file to `BUILDERS` in the Makefile.
4. Add its name to `ALGOS` in `tests/run_tests.sh`; `algo_args` maps test variant names to flags.

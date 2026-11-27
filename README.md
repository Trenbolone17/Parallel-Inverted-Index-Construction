# Parallel inverted index construction (CS F422, Project 23)

This project builds word → document posting lists from a pretokenized corpus, in C++ with OpenMP:
- **D1:** a sequential builder.
- **D2:** an OpenMP builder with per-thread dictionaries and a sorted merge.
- **D3:** heavy-term bitmaps under a memory budget.
- **Baseline:** the same index on the Phoenix++ runtime.

| | |
|---|---|
| **Report** (7 pages) | [report/report.pdf](report/report.pdf) |
| **Presentation** | [slide deck](https://claude.ai/artifact/89aNd8GcLuYT2Xr9fyL2zm#slide-075bc7f8ebe2), downloadable as PowerPoint or PDF |
| **Live demo** | [inverted-index/demo.sh](inverted-index/demo.sh), see [Demo](#demo) |
| **Plan and progress log** | [PLAN.md](PLAN.md) |
| **Project brief** | [Parallel_Computing_25_Project_Catalogue.docx](Parallel_Computing_25_Project_Catalogue.docx), Project 23 |

## Repository layout

```
inverted-index/   the code: builders, tests, tools, benchmark harness and raw results
report/           report.html (source), report.pdf, make_figures.py, build.sh
presentation/     make_charts.py: the slide charts, drawn from inverted-index/results
PLAN.md           plan, design decisions and dated progress log with all results
```

## Requirements

- **Compiler:** a C++20 compiler with OpenMP. Tested with GCC 16.2.1 and libgomp on Linux.
- **Python 3.12** (for datasets and figures), with pyarrow, numpy, pandas and matplotlib. Tested with pyarrow 26.0.1, numpy 2.5.5, pandas 3.1.1 and matplotlib 3.11.2.
- **Optional:** `git`, to fetch Phoenix++ for `make phoenix`; LLVM's OpenMP runtime, for `make tsan`.

## Datasets

**Download: [Google Drive folder](https://drive.google.com/drive/folders/1jMJhfT0rGFCVyRQ_BsZYel7MI44Wak44?usp=drive_link).** Put the files in `inverted-index/data/`, then decompress them with `gunzip *.gz`.

| File | Contents | Size |
|---|---|---|
| `gutenberg.txt` (+ `.meta.json`) | 4,976 Project Gutenberg books (PG-19, shards 0–3), one book per line, tokenized | 1.73 GB |
| `wiki.txt` (+ `.meta.json`) | first paragraph of 6,407,814 English Wikipedia articles (dump 2023-11-01), tokenized | 1.68 GB |
| `zipf_2M.txt` | synthetic: 2M documents of about 100 words, Zipf s = 1, vocabulary 1M | 0.71 GB |
| `gutenberg_test.txt`, `wiki_1shard.txt` (+ `.meta.json`) | small samples for quick runs (100 books; 156k paragraphs) | 38 MB, 47 MB |

To check the files after decompressing, run `sha256sum -c` in `inverted-index/data/` with these lines:

```
09a33f117096592019f700492782dee81c47cdaa3bbaa4763f593fa239f55b04  gutenberg.txt
0bee59009b24f8d07aabcb8d95762342e6844bbe6f411648c952c671baa073ee  wiki.txt
62f7d3a46f94ab036987fdd8de04fd0280c883e6b77995e781728ff56c2bdb39  zipf_2M.txt
384c6536c9fbd7d26d564d3ddd39edc5d02fc339cb736db30567c1ae75a4ddd8  gutenberg_test.txt
dd42c30f0450e52fa91e58f0c090a01f195abb27ab9b4c5b9d1fb5b109c72093  wiki_1shard.txt
```

**Not on the Drive, because the scripts regenerate them:**
- **Synthetic corpora:** `bench/experiments.sh` generates the other synthetic corpora used for the skew, vocabulary and document-length sweeps. `bin/gen_zipf` is deterministic, so `zipf_2M.txt` also comes out byte-for-byte the same: `bin/gen_zipf --docs 2000000 --vocab 1000000 --zipf 1.0 --len 100 --seed 7 --out data/zipf_2M.txt`.
- **25% and 50% subsets:** `bench/experiments.sh sizes` takes every 4th or 2nd line of the full corpora.

`tools/fetch_corpus.py` can rebuild the two real corpora from Hugging Face (see Usage), but the hosted datasets may change. The Drive copies are the exact files the results were measured on.

## Build and test

All commands from here on run in `inverted-index/`.

```sh
cd inverted-index
make            # bin/invidx, bin/idxcheck, bin/gen_zipf  (g++ -O3 -march=native -fopenmp)
make test       # hand-checked cases + 200 randomized differential tests vs the naive oracle
make asan       # same tests under AddressSanitizer + UBSan
make tsan       # ThreadSanitizer via clang; first: sudo pacman -S openmp
make phoenix    # Phoenix++ reference: clones the authors' repo (pinned) into third_party/, builds bin/phoenix_index
ALGOS=phoenix tests/run_tests.sh   # the same tests for the Phoenix++ build
```

To change the hand-checked cases, edit [inverted-index/tests/make_tiny.py](inverted-index/tests/make_tiny.py), then run `python3 tests/make_tiny.py`. The expected outputs in that file are written by hand, not computed.

## Usage

```sh
# Python environment for the dataset and figure scripts
uv venv .venv && uv pip install --python .venv/bin/python pyarrow numpy pandas matplotlib

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
bin/idxcheck --lookup a.bin parallel index        # per word: document count and first IDs (binary search)

# Real corpora: download parquet shards from Hugging Face, one at a time, and write them pretokenized
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

## Demo

```sh
./demo.sh                  # five steps, Enter between them; about a minute of building on Wikipedia
./demo.sh --no-pause       # a rehearsal straight through
./demo.sh --help           # --corpus FILE, --words "…", --threads N, --skip-tests
```

1. **The slide-2 example:** three tiny documents and their index.
2. **Correctness:** a short run of the test suite, about 300 checks across every builder at 1, 7 and 24 threads.
3. **Full-size builds:** the sequential builder, D2, D3 (8 P-cores, then all threads) and Phoenix++, pinned like the benchmarks.
4. **Same answer:** byte-for-byte comparison plus a table of times and speedups.
5. **Using the index:** word lookups, and the start of a document that contains the first word.

Without `data/wiki.txt` the demo generates a synthetic corpus. For the slides' timings, plug in and run `powerprofilesctl set performance` first; the script warns otherwise.

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

The raw results of the runs in the report are committed in [inverted-index/results/](inverted-index/results/): one JSON line per run, plus summaries and machine metadata. The figures are not committed; `bench/experiments.sh plots` redraws them.

How the harness measures:
- **Pinning:** threads are placed one per P-core first, then on E-cores, then on the P-cores' second hyperthreads. The order is read from sysfs and recorded in `.meta.json`.
- **Clean runs:** every run is a fresh process, after one untimed warm-up run that loads the corpus into the page cache.
- **Checksum check:** all runs on a corpus must produce the same index checksum, or the script exits non-zero.
- **Warnings:** the harness warns if the load average is high or the laptop is on battery.

Run benchmarks only on AC power with nothing else running, including the dataset download. Never mix results from different machines on one curve.

## Input and output contract

**Input** (details in [inverted-index/include/corpus.hpp](inverted-index/include/corpus.hpp)):
- One document per line; the document ID is the line number.
- Empty lines are empty documents and still use an ID.
- Tokens are runs of non-whitespace bytes.

**Output** (details in [inverted-index/include/index_format.hpp](inverted-index/include/index_format.hpp)):
- A canonical binary index.
- Terms are in byte order, and posting lists are strictly increasing.
- Two builds are correct relative to each other only if their files are byte-identical.

## Code layout (`inverted-index/`)

```
include/   hash.hpp, hash_table.hpp, local_index.hpp (inner loop shared by seq and omp), corpus.hpp,
           index_format.hpp, pod_vector.hpp, stats.hpp, builders.hpp
src/       invidx.cpp (CLI), build_naive.cpp (oracle), build_seq.cpp (D1), build_omp.cpp (D2 + D3),
           corpus.cpp, index_format.cpp, stats.cpp, idxcheck.cpp
tools/     gen_zipf.cpp, pretokenize.py, fetch_corpus.py
bench/     run.py (benchmark harness), plot.py (report figures), experiments.sh (all report experiments)
baselines/ phoenix_index.cpp (our index as a Phoenix++ application; built by `make phoenix`)
results/   raw benchmark results (JSONL, CSV summaries, machine metadata, run logs)
demo.sh    the live demo for the presentation
tests/     make_tiny.py -> tiny/*.txt + *.expected, run_tests.sh
third_party/ Phoenix++ checkout (not committed; `make phoenix` clones it)
```

To add a builder (paths under `inverted-index/`):
1. Declare it in `include/builders.hpp`.
2. Add a branch in `src/invidx.cpp`.
3. Add the source file to `BUILDERS` in the Makefile.
4. Add its name to `ALGOS` in `tests/run_tests.sh`; `algo_args` maps test variant names to flags.

## Attribution

- **Phoenix++** (J. Talbot, R. M. Yoo, C. Kozyrakis; Stanford University, BSD license) is used unmodified as the baseline. `make phoenix` fetches it at a pinned commit, and it is not redistributed. Our application on top of it is `baselines/phoenix_index.cpp`.
- **Datasets:** PG-19 (Rae et al., ICLR 2020; Project Gutenberg books published before 1919), from Hugging Face `emozilla/pg19`, and English Wikipedia (Wikimedia Foundation, CC BY-SA 4.0), from Hugging Face `wikimedia/wikipedia`, dump 20231101.en.
- **Hash finalizer:** `fmix64` in `include/hash.hpp` and `tools/gen_zipf.cpp` is MurmurHash3's 64-bit finalizer (Austin Appleby, public domain).
- Everything else (index format, builders, tests, corpus generator, benchmark scripts, figures) was written for this project. The use of AI tools and each member's contribution are stated in the report, section "Attribution and use of AI tools".

## Report and slide sources

The report is `report/report.html`, printed to PDF by `report/build.sh` (needs a Chromium-based browser). The slide charts are drawn by `presentation/make_charts.py`. Both read `inverted-index/results/`. From the repository root:

```sh
report/build.sh                                              # figures + report/report.pdf
inverted-index/.venv/bin/python presentation/make_charts.py  # presentation/charts/*.png
```

`./make_submission.sh` assembles the submission package (`submission/CS_F422_Project23.zip`) from the last commit.

## Status

- [x] Contract, canonical format, `idxcheck`, `naive` oracle, hand-checked and differential tests
- [x] D1: `seq` builder (and `gen_zipf`, `pretokenize.py`)
- [x] Dataset fetch script (Gutenberg, Wikipedia lead paragraphs), tested on the PG-19 test split and 1 Wikipedia shard
- [x] Full dataset download: Gutenberg (4,976 books, 1.73 GB) and Wikipedia (6.4M lead paragraphs, 1.68 GB)
- [x] D2: `omp` (local dictionaries + sorted range merge; static or dynamic schedule)
- [x] Benchmark harness (`bench/run.py`: pinning, repetitions, shuffled order, checksum guard; `bench/plot.py`: figures)
- [x] D3: `heavy` (heavy-term bitmaps under a memory budget), with tests for all-heavy, budget-capped, none-heavy and multi-segment bitmaps
- [x] Final run of E1, E3 and E5 on a quiet machine with the performance profile (735 runs, all checksums identical)
- [x] Phoenix++ reference (`make phoenix`): same index, passes the hand-written and differential tests
- [x] Corpus sizes, E4 (vocabulary and document-length sweeps) and E6 (Phoenix++ vs D2/D3): 1,060 runs, all checksums identical
- [x] Report: [report/report.pdf](report/report.pdf) (source `report/report.html`, rebuilt by `report/build.sh`)
- [x] Presentation (slide charts: [presentation/make_charts.py](presentation/make_charts.py))
- [x] Demo: [inverted-index/demo.sh](inverted-index/demo.sh)

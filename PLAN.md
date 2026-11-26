# Project 23: Parallel Inverted Index Construction (CS F422)

## Context

This plan covers Project 23 from `Parallel_Computing_25_Project_Catalogue.docx`. Its parameters:
- **Category, length, risk:** non-graph, 8–10 weeks, moderate risk.
- **Starting paper:** Ranger et al., *Evaluating MapReduce for Multi-core and Multiprocessor Systems*, HPCA 2007 (Phoenix). It supplies the map → local aggregation → reduce model.
- **Scope:** build word → document posting lists from pretokenized text. No crawler, ranking or web UI.

**Required deliverables:**
- **D1:** a sequential index builder and tests that check posting lists for equality.
- **D2:** an OpenMP version that builds local dictionaries and merges them in sorted order.
- **D3:** one improvement, either merge scheduling or frequent-term handling, under a memory budget.

**Common requirements:**
- At least 3 input families, with multiple sizes for each.
- Thread counts 1, 2, 4, … up to the core count.
- Repeated timings.
- Report:
  - runtime, peak memory and algorithmic work;
  - preprocessing time separately from kernel time;
  - at least one ablation and one unfavorable case;
  - scaling speedup separately from practical speedup.
- A 6–8 page report, reproducibility instructions, a presentation and a demo.

**Decisions made:**
- 2 students.
- D3 is **frequent-term (heavy-term) handling**.
- Inputs are a **synthetic Zipf corpus, Project Gutenberg books (long documents) and Wikipedia lead paragraphs (short documents)**.

**Machine (also the only benchmark machine):**
- **CPU:** i7-13700HX, a hybrid chip. P-cores are CPU IDs 0–15 (8 cores × 2 hyperthreads, up to 5.0 GHz). E-cores are CPU IDs 16–23 (8 cores, up to 3.7 GHz).
- **Caches:** 30 MB L3, 1 NUMA node.
- **Memory and disk:** 31 GB RAM, about 19 GB free disk.
- **Toolchain:**
  - g++ 16 (libgomp, OpenMP 5.2) and clang 23.
  - No `libomp`, so clang can't run OpenMP with the thread sanitizer until `sudo pacman -S openmp`.
  - No cmake, so the build uses Make.
  - `uv` is available for a Python virtual environment; numpy and matplotlib are not installed yet.
  - `perf` can be installed with pacman.

---

## 1. Input/output contract (fix this in week 2)

**Input format: pretokenized text.**
- One document per line. The document ID is the 0-based line number.
- An empty line is a valid empty document and still uses an ID.
- Tokens are maximal runs of bytes other than ASCII whitespace (space, `\t`, `\r`, `\v`, `\f`). The builder treats every other byte as opaque, so UTF-8 passes through unchanged.
- A final line without a trailing newline is still a document.
- Fewer than 2³² − 1 documents, so document IDs fit in `uint32` (`UINT32_MAX` is reserved).
- Lowercasing and splitting happen offline in `tools/pretokenize.py`. That time is reported separately, not as part of the build. (It isn't named `tokenize.py` because that would shadow Python's standard `tokenize` module.)

**Output: a canonical binary index**, so two builds are equal exactly when their files are identical byte for byte.
```
magic "INVIDX01" | u64 n_docs, n_terms, n_postings
u64 term_off[n_terms+1] | char term_bytes[]      // terms in strict memcmp order
u64 post_off[n_terms+1] | u32  postings[]         // each list strictly increasing, deduplicated
```
- A term that occurs several times in one document gets a single posting.
- Positions and term frequencies are an optional extension, not part of the minimum scope.
- `--dump` prints `term<TAB>df<TAB>doc doc …` for small tests.

---

## 2. Algorithms

All builders share one open-addressing hash table. It is keyed by `string_view` into the loaded input buffer, uses the same hash function in every builder, and adds no per-token string allocations. Because every builder uses the same table, speedups measure parallelism rather than differences between data structures.

**Shared phase 0: load and document boundaries.**
- Read the whole file into memory. This I/O is timed separately.
- Find document boundaries in parallel: each thread counts newlines in its chunk, a prefix sum gives offsets, and threads fill `doc_start[N+1]`.

### D1: Sequential builder (`seq`) and oracle (`naive`)
- **`naive`:** `std::map<std::string, std::set<uint32_t>>`. It is deliberately simple and independent of the other builders, and serves as the test oracle on small inputs only.
- **`seq` (best sequential baseline):**
  - Scan documents in order. For each token, look it up in the hash table and append the document ID unless it equals the list's `last_doc`. Scanning in order makes deduplication free.
  - Then sort the terms and emit the canonical layout.

### D2: OpenMP local dictionaries with sorted merging (`omp`)
1. **Map.** Split documents into T contiguous blocks with equal byte counts (binary search over `doc_start`).
   - Each thread builds a local hash table mapping each term to a posting vector.
   - Each local list is already sorted.
   - Every document ID in block t is smaller than every ID in block t+1. So when the threads' lists for one term are merged, **concatenating them in thread order is enough**; no sort is needed.
2. **Local sort.** Each thread sorts its own (term, list) entries by term.
3. **Splitters.** Use regular sampling from each sorted local dictionary, as in sample sort, to choose R = k·T term ranges, with oversplitting k ≈ 4.
4. **Two-pass range merge.**
   - **Count pass:** run `omp for schedule(dynamic)` over the ranges. For each range, binary-search its bounds in all T runs, then do a T-way merge by term to count distinct terms and postings.
   - Prefix sums over the range counts give global offsets.
   - **Write pass:** repeat the merge, writing term bytes and concatenating posting lists into place. Doing it in two passes avoids reallocation and gives deterministic output.
5. **Schedule variant (small ablation).** Use many dynamic chunks of documents instead of T static blocks.
   - **Why:** on this hybrid CPU, equal static blocks leave P-core threads idle while E-core threads catch up.
   - **Cost:** each thread's lists are still sorted, but they interleave across threads. The postings merge becomes a T-way merge instead of `memcpy`.

**Where D2 suffers** (this is the reason for D3):
- Under Zipf skew a few terms ("the", "of", …) hold a large share of all postings. Ranges that contain them dominate the merge.
- Their per-thread vectors are reallocated repeatedly as they grow, with up to 2× capacity overhead.
- Light terms are replicated in every thread's local dictionary, so memory grows with T.

### D3: Heavy-term separation (`heavy`), the evaluated improvement
1. **Detection.** Count document frequency (df) on an evenly spaced (stratified) sample of documents, about 1% (`--sample`), with a time cap.
   - Terms whose estimated df/N ≥ θ are marked heavy.
   - The default θ = 1/32 is the break-even point: from that frequency on, a bitmap of N bits is no larger than a list of `uint32` document IDs.
   - The number of heavy terms H is capped so that H·N/8 bytes ≤ `--heavy-mem` (**the memory budget**).
   - Classification affects only performance, never output. A wrongly classified term simply takes the light path.
2. **Map.** Insert heavy terms into each local table in advance with a `heavy_id`.
   - When a token hits a heavy term, set bit `doc` in a shared bitmap `bm[heavy_id]`. There's no push, no dedup check and no reallocation.
   - Document chunks are aligned to multiples of 512 documents (one cache line of bitmap), so no two threads ever touch the same word or cache line. This needs no atomics and avoids false sharing.
   - Light terms take the D2 path.
3. **Merge.**
   - Light terms use the D2 range merge. Splitters are weighted by posting volume, which is cheap now that the giant lists are gone.
   - Heavy terms need no merge, only decoding. Range threads reserve the term slots during the write pass.
   - A separate parallel loop over (heavy term × bitmap segment) tasks does `popcount`, then a prefix sum, then decodes with `ctz` into the reserved slots. The tasks are fine-grained and balanced.
4. **Ablations.**
   - Heavy terms separated but kept as lists, versus bitmaps.
   - A sweep over θ.
   - A sweep over sample rate, including what sampling costs.
   - Splitters weighted by count versus by volume.
   - A sweep over `--heavy-mem`.
5. **Expected unfavorable cases:**
   - uniform vocabulary (Zipf s = 0): no heavy terms, so sampling is pure overhead;
   - θ set too low: sparse bitmaps waste memory and decode time;
   - the Gutenberg corpus: few documents and huge per-document vocabularies, so postings ≪ tokens and the map phase dominates.

---

## 3. Code layout

Everything goes in `Parallel Project/inverted-index/`. Build with `g++ -O3 -march=native -fopenmp`.

```
Makefile                      targets: all, test, asan, tsan (clang+libomp), bench-smoke
include/index_format.hpp      canonical index struct, binary writer/reader, 64-bit checksum, --dump
include/corpus.hpp            mmap/read, parallel doc-boundary scan, token iterator
include/hash_table.hpp        open-addressing string_view → entry table (shared by all builders)
include/stats.hpp             phase timers, per-thread timers, imbalance (max/mean), peak RSS, work counters → JSON
src/build_naive.cpp  src/build_seq.cpp  src/build_omp.cpp  src/build_heavy.cpp
src/invidx.cpp                CLI: --algo naive|seq|omp|heavy --threads --schedule --theta --sample --heavy-mem --stats out.json
src/idxcheck.cpp              compare two index files byte-for-byte; --validate invariants
tools/gen_zipf.cpp            synthetic corpus: N docs, length distribution, vocabulary V, Zipf s, seed
tools/fetch_gutenberg.py  tools/fetch_wiki.py  tools/pretokenize.py (multiprocessing)
tests/tiny/                   hand-written inputs + expected dumps
tests/run_tests.sh            hand tests + randomized differential tests
bench/run.py                  sweeps with pinning, warm-up + 5 reps, fresh process per run → results/*.csv
bench/plot.py                 scaling, phase breakdown, skew sweep, memory, ablation figures
```

**Work counters reported for every run:**
- tokens, distinct terms, total postings (distinct term–document pairs), heavy-term count H and their share of postings;
- hash probes;
- the replication factor: the sum of distinct terms across threads' local dictionaries divided by the number of distinct terms overall;
- bytes moved in the merge;
- per-thread time for each phase.

---

## 4. Correctness (keep all of it even with 2 people)

**Hand-checkable inputs in `tests/tiny/`:**
- an empty corpus, empty documents and a single document;
- a term repeated within one document;
- a term present in every document;
- identical documents;
- terms that are prefixes of each other (`a`, `ab`, `abc`);
- a very long token;
- non-ASCII UTF-8 bytes.

**Randomized differential tests:**
- Run `gen_zipf` with 200 or more random seeds, small N, and s in {0, 1, 1.5}.
- Compare `naive` with `seq`, `omp` and `heavy`.
- Use thread counts {1, 2, 3, 7, 16, 24}. The odd counts catch block and range boundary bugs.
- Run both schedules, and θ values that make everything heavy or nothing heavy.

**Invariant checker (`idxcheck --validate`):**
- terms strictly increasing;
- postings strictly increasing and all IDs < N;
- Σdf equals the number of distinct term–document pairs, computed independently.

**Heavy-term edge cases:**
- heavy terms whose bits fall on chunk and word boundaries;
- N not a multiple of 64 or 512;
- a sample that misses a truly heavy term.

**Sanitizers:**
- `make asan` (g++ ASan + UBSan) runs on the tiny and differential tests.
- `make tsan` uses clang + libomp, after `sudo pacman -S openmp`.

---

## 5. Experiments and measurement protocol

**Pinning.** The hybrid CPU makes this essential. Set `OMP_PROC_BIND=close` and give `OMP_PLACES` explicitly, recording it with every result.
- **Physical-core curve:** 1, 2, 4, 8 threads with one per P-core (CPUs 0, 2, …, 14), then 12 (8P+4E) and 16 (8P+8E).
- **All hardware threads:** 24.
- Report the P-core-only and P+E regions separately, and expect a bend in the curve at 8.

**Noise control:**
- Run on AC power with the performance power profile.
- Do one warm-up run, then 5 repetitions, and report the median with min and max.
- Randomize the order of variants across repetitions.
- Keep the input in page cache, but time I/O separately.
- Run each configuration in a fresh process so peak RSS is clean.

**Speedups:**
- Scaling speedup = T_variant(1) / T_variant(p).
- Practical speedup = T_best-sequential / T_variant(p). If `heavy` at T=1 beats `seq`, it becomes the best sequential baseline, and the report says so.
- End-to-end times include document-boundary scanning and sampling. Tokenization and load I/O are listed separately.

**Datasets** (about 19 GB of free disk, so download, extract and delete raw files as you go; keep tokenized data ≤ about 6 GB):

| Family | Source | Shape | Sizes |
|---|---|---|---|
| Synthetic | `gen_zipf` | controlled V, document length, s | sweeps (below) |
| Gutenberg (long docs) | PG-19 on Hugging Face, or the PG mirror/harvest endpoint (not a site crawl) | ~2–4k books, 1 book = 1 doc | 25 / 50 / 100% of ~1.5 GB |
| Wikipedia (short docs) | `wikimedia/wikipedia` parquet shards (or abstracts dump if available); keep the lead paragraph only | ~5M docs × ~60 tokens | 25 / 50 / 100% of ~2.5 GB |

**Experiment matrix:**
- **E1, strong scaling:** 3 families × 3 sizes × the thread counts above × {`seq`, `omp`, `heavy`}.
- **E2, phase breakdown and load imbalance:** at 8 and 24 threads, show stacked phase times and per-thread max/mean.
- **E3, skew sweep (synthetic):** s ∈ {0, 0.6, 0.8, 1.0, 1.2, 1.4}, comparing `omp` and `heavy` on time and peak RSS.
- **E4, vocabulary and document-length sweeps (synthetic):** V ∈ {10⁴…10⁷} and document length ∈ {10, 100, 1k, 10k}.
- **E5, D3 ablations:** the list in §2, plus the static versus dynamic schedule.
- **E6, Phoenix reference:**
  - Build Phoenix 2 or Phoenix++ and time `word_count` on our tokenized corpus, comparing against our map/aggregation phase only. Their `reverse_index` indexes HTML links, not words.
  - If Phoenix won't build on GCC 16, document the attempt.
  - Optional stronger baseline: PISA's `invert` tool.
- **Unfavorable case:** the uniform corpus (s = 0) and the low-θ case, shown honestly in the report.

---

## 6. Schedule and roles

- **A** owns baseline, correctness, datasets and the evaluation pipeline.
- **B** owns the parallel implementation and profiling.
- A and B review each other's code: A reviews B's merge code, and B reviews A's checker.

| Week | Student A | Student B | Milestone |
|---|---|---|---|
| 1 | Read the Phoenix paper; set up repo, Makefile and `uv` venv (numpy, pandas, matplotlib, pyarrow); write fetch and tokenize scripts; write `gen_zipf` | Read the paper; write `index_format`, the corpus loader and the parallel boundary scan | Contract draft |
| 2 | Write `naive`, `idxcheck` and the tiny tests; download and tokenize corpora; try building Phoenix | Write the shared hash table and D1 `seq`; time it on the full corpora (checks memory fits) | **Feasibility review; freeze the contract** |
| 3 | Write the differential test harness and the ASan/UBSan targets | D2 map phase and local sort | |
| 4 | Write the bench harness (pinning, repetitions, CSV, RSS) and first plots | D2 splitters and two-pass range merge | **D1 + correct D2, all tests passing at 1–24 threads** |
| 5 | Run the synthetic skew, vocabulary and document-length sweeps | Profile (timers + `perf`); dynamic-schedule variant; confirm the heavy-term bottleneck | Bottleneck identified |
| 6 | Write the D3 edge-case tests | D3: sampling, heavy table, bitmaps, decode tasks | **D3 correct** |
| 7 | Run the full E1–E6 matrix | Run ablations, tune θ and the sample rate | |
| 8 | Report draft (6–8 pages) and README reproduction steps | Figures; check the analysis of the unfavorable case | **Viable class submission** |
| 9–10 | Generalization check: tune θ on synthetic + Gutenberg, test on Wikipedia | Memory-budget sweep; optional PISA comparison | Final report, presentation, demo |

**Optional GPU extension (only after the week 8 milestone).** Add a `cuda` mode to `invidx` for the RTX 4060 Laptop GPU (8 GB). The GPU version uses a sort-based design rather than hash tables:
1. Hash each token to a 64-bit ID, and detect collisions.
2. Radix-sort the (term ID, doc ID) pairs with CUB.
3. Drop duplicate pairs and cut the sorted array into posting lists.
4. Map the IDs back to strings on the CPU.

The full corpora exceed 8 GB of VRAM, so they're processed in batches. Times are reported end to end, including PCIe transfers. GPU and CPU benchmarks never run at the same time, because they share power and cooling. The same output format and `idxcheck` verify the results.

**Reading:**
- Ranger et al., HPCA 2007 (required).
- Zobel & Moffat, *Inverted Files for Text Search Engines*, ACM Computing Surveys 2006 (§ on index construction).
- Heinz & Zobel, *Efficient Single-Pass Index Construction*, JASIST 2003.
- Talbot et al., *Phoenix++*, MapReduce workshop 2011 (combiner containers).

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| E-cores distort the scaling curves | Explicit pinning, P-core-only curve reported separately, dynamic-schedule variant |
| Laptop thermal throttling | AC power, performance profile, randomized run order, cool-down between sweeps, record clocks |
| Memory grows with T (vocabulary replicated in each thread's dictionary) | Measure it as a result; size inputs so peak RSS stays ≤ ~20 GB |
| Off-by-one bugs at merge boundaries | Odd thread counts, prefix-term tests, randomized seeds in CI-style `make test` |
| Phoenix doesn't build | It's a conceptual baseline in the catalogue; document the attempt and keep the `seq` comparison |
| Too little disk | Stream shards: download, extract, delete |

---

## 8. Progress

**2026-11-25: weeks 1–2 foundation and D1 done.** See `README.md`.
- **Done:** index format, corpus loader, shared hash table, `naive` oracle, `seq` builder, `idxcheck`, `gen_zipf` and `pretokenize.py`.
- **Tests:** 13 hand-checked cases plus 200 randomized differential tests, all passing. The ASan/UBSan build is clean.
- **First measurement:** `seq` on 2M synthetic documents (200M tokens, V = 1M, s = 1.0) builds in 22.2 s. 97% of that is the map phase, about 107 ns per token, because each token waits on memory (cache misses). Peak RSS is 2.25 GB.
- **Next:** dataset fetch scripts, then D2 (`omp`).

**2026-11-25: D2 and dataset tooling done.**
- **Dataset tooling:** `tools/fetch_corpus.py` streams Hugging Face parquet shards with retries and resume, converting each one and then deleting it.
  - PG-19 (`emozilla/pg19`): 23 shards, 7.0 GB.
  - Wikipedia `20231101.en`: 41 shards, 11.6 GB.
  - Tested on the PG-19 test split (100 books) and 1 Wikipedia shard (156k lead paragraphs, 51 tokens each on average).
  - **Full corpora downloaded the same day.** The CDN kept resetting connections, so the script now resumes after the last completed shard and retries up to 40 times.

    | corpus | docs | tokens | distinct terms | postings | size |
    |---|---|---|---|---|---|
    | `data/gutenberg.txt` (PG-19 shards 0–3) | 4,976 books | 326M | 781k | 29.9M | 1.73 GB |
    | `data/wiki.txt` (all 41 shards) | 6,407,814 lead paragraphs | 285M | 2.78M | 214M | 1.68 GB |
    | `data/zipf_2M.txt` (`gen_zipf`, s = 1.0, seed 7) | 2,000,000 | 200M | 1.00M | 163M | 0.68 GB |

  - `seq` and `omp` produce identical, valid indexes on both real corpora.
- **D2 (`omp`):** passes 4,060 tests (static, dynamic and single-range variants at 1–24 threads) and the ASan run. Three planted merge bugs: 2 caught; the third was equivalent to correct code, not a bug.
- **First scaling numbers** (2M synthetic docs, 200M tokens, V = 1M, s = 1.0; pinned P-cores first, then E-cores, then hyperthreads; best of 2):

  | threads | 1 | 2 | 4 | 8 | 12 | 16 | 24 |
  |---|---|---|---|---|---|---|---|
  | `omp` static, s | 26.0 | 14.2 | 8.5 | 5.3 | 4.8 | 4.1 | 3.9 |
  | practical speedup vs `seq` (25.4 s) | 0.98× | 1.79× | 2.98× | 4.79× | 5.24× | 6.21× | 6.51× |

- **Findings so far:**
  - **The map phase scales well:** 10.6× at 24 threads. It is limited by memory latency, so E-cores are nearly as fast as P-cores here. Static-block imbalance at 16 threads is only 1.12 (max/mean).
  - **Dynamic schedule:** reduces that imbalance to 1.02, but the k-way posting merge it requires costs more in the layout phase. Overall it ties with static at 24 threads and loses at 2–8.
  - **Main bottleneck — vocabulary replication** *(corrected by E1 below: this is mostly an artifact of the synthetic corpus)*: at 16 threads, each thread's local dictionary holds about 80% of the vocabulary (13.1× replication).
    - Local sort and merge therefore don't shrink as T grows: about 1.1 s out of 3.9 s at 24 threads, an Amdahl limit.
    - Peak memory grows from 2.2 GB to 3.9 GB.
  - **D3 check:** measure the same breakdown on the real corpora before finalizing D3. Heavy-term handling targets posting volume and merge cost, not replication.

**2026-11-25: E1 strong scaling on all three corpora** (`bench/experiments.sh e1`, 5 runs each, shuffled order, all checksums identical). Results are in `inverted-index/results/e1_*` and figures in `results/figures/`. This run used the *balanced* power profile with a browser open: good for design decisions, but rerun with the performance profile for the report.

| corpus | `seq` | best `omp` (threads) | practical speedup | `omp` static at 8 / 12 threads |
|---|---|---|---|---|
| Gutenberg (long docs) | 11.36 s | 1.18 s (dynamic, 24) | **9.6×** | 1.88 s / 2.09 s |
| Wikipedia (short docs) | 16.54 s | 2.37 s (static, 24) | **7.0×** | 3.26 s / 3.47 s |
| synthetic Zipf | 21.32 s | 3.51 s (dynamic, 24) | **6.1×** | 4.55 s / 4.80 s |

Findings:
1. **The map phase is 89–97% of sequential time** and scales about 9–10× at 24 threads.
2. **E-cores matter after all.** Static blocks are slower at 12 threads than at 8 on every corpus. Each thread gets equal bytes, but E-core threads run slower, so map imbalance (max/mean) reaches 1.18–1.37.
   - The dynamic schedule removes that imbalance (map max/mean 1.02–1.04) and wins from 12 threads on.
   - It loses on P-cores only, at 2–8 threads, because of the extra posting-list merge.
   - This corrects the earlier synthetic-only finding; at 16 threads the gap is larger on real text.
3. **Vocabulary replication is a synthetic artifact.** At 24 threads it is 17× on Zipf, but only 3.0× on Wikipedia and 3.7× on Gutenberg.
   - Real rare words cluster in a few documents ("burstiness"), whereas `gen_zipf` draws every token independently.
   - Report this as a limitation of the synthetic family. On Wikipedia, everything outside the map phase is about 0.77 s of 2.37 s at 24 threads (local sort 0.39 s).
4. **The dynamic schedule's merge is where heavy terms hurt.**
   - On Wikipedia at 24 threads, `layout` costs 0.28 s with the dynamic schedule against 0.10 s with static.
   - Its per-thread imbalance is 1.26–1.41, because each very frequent term's k-way merge falls to a single thread.
   - This is exactly what D3's bitmaps remove: bits are positional, so heavy terms need no merge under either schedule.
5. **Gutenberg confirms the expected unfavorable case for D3.** Only 0.09 postings per token, and merge plus layout take about 0.07 s.
6. **Not yet addressed: local sort.** It takes 1.28 s of `seq`'s 16.5 s on Wikipedia and scales only 3.3×. Comparisons chase string pointers into the corpus.
   - Sorting on an 8-byte key prefix stored inline would cut this.
   - It would have to be applied to `seq` as well, for fairness.

**D3 decision: keep heavy-term handling.** The real data sharpens the motivation:
- The dynamic schedule is needed on this hybrid CPU, but it makes heavy-term merges expensive and unbalanced.
- Bitmaps should give the dynamic schedule's map balance together with the static schedule's cheap layout.
- The measurement to beat is the best of static and dynamic at 12–24 threads on Wikipedia and Zipf, with Gutenberg as the unfavorable case.

**2026-11-26: D3 implemented** as `--algo heavy`, an option of the D2 builder, so D2 and D3 share every line except the heavy-term parts.
- **How it works:**
  - **Sample:** each thread counts document frequency on a share of a 1% stratified document sample. Per-thread candidates are filtered with a pigeonhole bound, then exact sample counts are summed.
  - **Classify:** a term is heavy if its sampled df/m ≥ θ (default 1/32, the break-even point between bitmap and list size).
  - **Budget:** heavy terms are capped by the bitmap memory budget (`--heavy-mem-mb`), keeping the most frequent.
  - **Map:** heavy tokens set a bit in a shared N-bit bitmap, once per document thanks to the same last-document check as D2.
  - **Merge:** light terms are merged exactly as in D2.
  - **Layout and decode:** heavy terms are slotted into term order during the write pass, then decoded by balanced (term × 262k-document segment) tasks.
- **Correctness:**
  - 10,462 tests pass: five heavy variants (default, all heavy, dynamic, budget-capped, none heavy) at 1–24 threads, plus a 600k-document corpus with 3 decode segments per bitmap. The ASan/UBSan build is clean.
  - Three planted D3 bugs were all caught: wrong segment offsets, wrong heavy/light order, and a dropped bitmap word.
  - Race-freedom of the boundary atomics holds by construction; ThreadSanitizer still needs `sudo pacman -S openmp`.
- **Two design changes forced by measurement:**
  1. **Dropped:** the planned 512-document chunk alignment (meant to avoid races and false sharing). With 4,976 Gutenberg books it collapsed 24 blocks into about 10 and made D3 2× slower. Chunks now start anywhere; only each chunk's first and last bitmap word use an atomic OR.
  2. **Added:** the heavy path now checks the last document before writing a bit. Without it, every repeat of a word in a long book rewrote a scattered bitmap word, and D3 was 23% slower on Gutenberg.
- **Preliminary results** (median of 3 runs, 24 threads, busy machine; differences of about 3% are within noise):

  | corpus | `omp` | `omp-dynamic` | `heavy` | `heavy-dynamic` |
  |---|---|---|---|---|
  | Wikipedia | 2.13 s | 2.33 s | 2.16 s | 2.25 s |
  | Gutenberg | 1.15 s | 1.15 s | 1.13 s | **1.06 s** |

  - **Overheads are small:** about 0.045 s on Wikipedia (sampling 0.012–0.02 s, decode 0.02 s); on Gutenberg sampling alone is 0.048 s, about 4%.
  - **D3 removes what it targeted:** the dynamic schedule's heavy merge. On Wikipedia, layout falls from 0.30 to 0.18–0.20 s, and its imbalance from 1.38 to 1.19.
  - **Map time barely changes on Wikipedia,** because heavy tokens (43% of tokens) are cache-hot lookups either way. On Gutenberg, skipping vector growth makes the map phase about 7% faster.
  - **Memory:** peak RSS is about 5% lower (Wikipedia 4.43 → 4.19 GB at 24 threads).
  - **θ preview:** lower θ cuts layout time (0.22 → 0.12 s) but raises decode time and memory once bitmaps outgrow the lists they replace. The total is flat to U-shaped around θ = 1/32 to 1/128.
- **Honest summary so far:**
  - D3 works as designed and costs little, but the end-to-end gain is a few percent, because the map phase (rare-word lookups) dominates.
  - Gutenberg was *not* the unfavorable case predicted in §2 (D3 helps there). The likely unfavorable case is a uniform vocabulary (s = 0), to be shown by E3.
  - The catalogue accepts a well-explained modest or negative result.
- **E5 ablation preview** (Wikipedia, dynamic schedule, 3 runs, shuffled; the load average reached 11–14 during the run, so these are preliminary). Results are in `results/e5_*` and figures in `results/figures/e5_*`. Every D3 setting is compared against `omp-dynamic` at 24 threads (2.48–2.53 s).

  - **θ (threshold; budget not binding except at 1/512):**

    | θ | heavy terms | bitmaps | time | layout | decode | peak RSS |
    |---|---|---|---|---|---|---|
    | 1/4 | 11 | 8 MB | 2.46 s | 0.25 s | 0.013 s | 4.31 GB |
    | 1/32 (default) | 104 | 79 MB | 2.43 s | 0.20 s | 0.024 s | 4.23 GB |
    | 1/128 | 557 | 426 MB | **2.40 s** | 0.15 s | 0.039 s | 4.36 GB |
    | 1/512 (capped at 1 GB) | 1,340 | 1,024 MB | 2.47 s | 0.12 s | 0.053 s | **4.83 GB** (more than D2) |

    Layout time and its imbalance fall steadily as θ drops (imbalance 1.34 → 1.07), while decode time and memory rise. The result is a U-shaped total. θ = 1/32 gives the lowest memory, about 5% below D2 (4.46 GB); θ = 1/128 gives the fastest build, about 3% faster than D2.
  - **Sample rate:** a 0.1% sample finds almost the same heavy set (101 against 104 terms) in 0.003 s. A 10% sample costs 0.12–0.21 s and makes D3 slower than D2, so this is an unfavorable setting.
  - **Memory budget at θ = 1/128:**

    | budget | heavy terms | time |
    |---|---|---|
    | 16 MB | 20 | 2.47 s |
    | 64 MB | 83 | 2.45 s |
    | 256 MB | 335 | **2.41 s** |
    | 1 GB (unconstrained, 426 MB) | 557 | 2.44 s |

    The budget caps memory cleanly (the 64 MB budget matches θ = 1/32's memory) at a cost of 1–2%.
- **Still to do** (done below): the final `bench/experiments.sh all` run on a quiet machine with the performance profile.

**2026-11-26: final run of E1, E3 and E5** (`bench/experiments.sh all`, performance profile, AC power, 5 runs each in shuffled order, 735 runs). These numbers replace every preview above; use them in the report.
- **Run quality:**
  - Every corpus produced one index checksum across all builders and thread counts.
  - Repeated runs of one configuration differ by a median of 0.5–2.5% (max/min). One outlier: Zipf `heavy` at 4 threads, 18%.
  - The "load average" warnings in `results/final_run.log` were caused by the previous experiment's own 24-thread runs. The first experiment started at a load of 1.4.
  - The power profile is blank in these `.meta.json` files: `powerprofilesctl` failed in that shell. `run.py` now also records the profile from sysfs.
- **E1 strong scaling (24 threads):**

  | corpus | `seq` | best (all `heavy-dynamic`) | practical speedup | peak RSS, D2 dynamic → D3 dynamic |
  |---|---|---|---|---|
  | Gutenberg | 11.80 s | 1.11 s | **10.6×** | 2.21 → 2.16 GB |
  | Wikipedia | 16.95 s | 2.28 s | **7.4×** | 4.46 → 4.22 GB |
  | Zipf | 20.65 s | 3.42 s | **6.0×** | 3.95 → 3.79 GB |

  - **Parallel overhead is small:** at 1 thread every parallel variant is within 5% of `seq` (0.97–1.05×).
  - **The map phase runs at what the hardware allows.**
    - Sampled clocks: 4.75 GHz with 1 thread busy, 3.94 GHz with all 8 P-cores busy. This caps 8 P-cores at about 6.6× over one core.
    - Map speedup is 5.3–6.8× at 8 threads and 10.3–13.2× at 24.
    - Map CPU-seconds at 8 threads are 1.14–1.42× those of `seq`. On real text the clock drop explains this; Zipf has another ~17% from memory traffic (1M terms drawn independently).
  - **The phases after map are the gap.** At 24 threads, local sort, merge, layout and free take 23% (Gutenberg), 38% (Wikipedia) and 43% (Zipf) of build time.
    - On Zipf, local sort takes 0.78 s at 24 threads against 0.40 s for `seq`'s single sort, because each thread sorts its own copy of most of the vocabulary (replication).
    - This cost is built into the D2 design that the catalogue requires (local dictionaries with sorted merge).
  - **Correction to the preview's finding 2:** with the performance profile, static blocks are no longer slower at 12 threads than at 8 (Wikipedia 3.43 → 3.01 s).
    - Map imbalance (max/mean) is 1.07–1.14 for static and 1.02–1.03 for dynamic, at both 8 and 12 threads.
    - Dynamic scheduling wins clearly only on Gutenberg (1.96 against 2.10 s at 8 threads). On Wikipedia and Zipf, static is equal or slightly better below 24 threads.
- **E3 skew sweep** (2M docs, V = 1M, 200M tokens; `omp-dynamic` against `heavy-dynamic`):

  | s | heavy terms | time at 8 threads | time at 24 threads | peak RSS at 24 threads |
  |---|---|---|---|---|
  | 0 | 0 | 9.77 → 9.77 s | 6.20 → 6.21 s | 5.32 → 5.32 GB |
  | 0.6 | 13 | 9.04 → 9.07 s | 5.89 → 5.91 s | 5.13 → 5.13 GB |
  | 0.8 | 104 | 7.56 → 7.58 s | 5.16 → 5.19 s | 4.85 → 4.79 GB |
  | 1.0 | 212 | 4.76 → 4.74 s | 3.44 → 3.43 s | 3.95 → 3.79 GB |
  | 1.2 | 203 | 2.18 → 2.14 s | 1.46 → 1.42 s | 2.33 → 2.13 GB |
  | 1.4 | 137 | 0.98 → 0.93 s | 0.56 → 0.55 s | 1.27 → 1.11 GB |

  - D3's benefit grows with skew: up to 5% time and 13% memory at s = 1.4.
  - **A uniform vocabulary (s = 0) is not an unfavorable case after all.** The sample finds no heavy terms and costs nothing measurable. This corrects the prediction in §2.
- **E5 ablations** (Wikipedia, dynamic schedule; `omp-dynamic` = 2.32 s, 4.46 GB at 24 threads):

  | setting | heavy terms | bitmaps | time at 24 threads | layout (imbalance) | decode | peak RSS |
  |---|---|---|---|---|---|---|
  | D2 (`omp-dynamic`) | – | – | 2.32 s | 0.28 s (1.33) | – | 4.46 GB |
  | θ = 1/4 | 11 | 8 MB | 2.30 s | 0.25 s (1.32) | 0.012 s | 4.31 GB |
  | θ = 1/8 | 20 | 15 MB | 2.28 s | 0.22 s (1.27) | 0.019 s | 4.28 GB |
  | θ = 1/32 (default) | 104 | 79 MB | 2.25 s | 0.18 s (1.16) | 0.023 s | **4.23 GB** |
  | θ = 1/128 | 557 | 426 MB | **2.25 s** | 0.13 s (1.07) | 0.036 s | 4.37 GB |
  | θ = 1/512 (1 GB cap) | 1,340 | 1,024 MB | 2.29 s | 0.12 s (1.05) | 0.050 s | **4.83 GB (> D2)** |
  | sample 0.1% | 101 | 77 MB | 2.26 s | | | 4.23 GB |
  | sample 10% (sampling 0.11 s) | 102 | 78 MB | 2.34 s (> D2) | | | 4.22 GB |
  | budget 16 / 64 / 256 MB (θ = 1/128) | 20 / 83 / 335 | | 2.28 / 2.25 / 2.26 s | | | 4.28 / 4.23 / 4.27 GB |

  - D3 halves the dynamic schedule's layout time and its imbalance, as designed. End to end that is 3% faster and 5% less memory, because layout is a small share of the total.
  - **Unfavorable settings, for the report:**
    - θ too low (1/512): bitmaps outgrow the lists they replace, so memory exceeds D2.
    - Sampling 10%: sampling costs more than D3 saves.
  - The memory budget caps bitmap memory as intended. At 64 MB it gives θ = 1/32's memory with θ = 1/128's speed.

**2026-11-26: Phoenix++ reference built (E6).** `make phoenix` fetches the authors' repository (`kozyraki/phoenix`, pinned commit `1276c8d`) into `third_party/` and builds `bin/phoenix_index`. It is our index written as a Phoenix++ application (`baselines/phoenix_index.cpp`):
- **Ours:** the map callback (tokens → term, document), a combiner (posting list with last-document dedup), the reduce callback (sort each term's document IDs) and the canonical layout.
- **Phoenix++ code:** the scheduler, per-thread hash tables, hash partitioning into T reduce buckets, the cross-thread union in reduce, and the sort/merge of the results.
- **Build note:** Phoenix++ predates C++11 two-phase lookup, so it needs `-fpermissive` with GCC 16. Its source is unchanged.
- **Same index:** it passes the hand-written and 360 randomized differential tests (`ALGOS=phoenix tests/run_tests.sh`). On full Wikipedia it gives the same checksum as `seq`, so `bench/run.py` treats it as one more variant.
- **Pinning:** threads are pinned to the CPUs in `OMP_PLACES`. Phoenix++'s own default would put threads 0–7 on 4 P-cores with both hyperthreads each.
- **First look** (single runs, Wikipedia):
  - 8 threads: 4.10 s, 5.0 GB.
  - 24 threads: 2.80 s, 5.5 GB, against 2.28 s and 4.2 GB for `heavy-dynamic`.
  - Its map phase is as fast as ours (2.58 s at 8 threads). The difference is its reduce (1.02 s) and merge (0.33 s), against our local sort, merge and layout (about 0.6 s).
  - On 10-token documents (`zipf_len10`) Phoenix++ is faster than us: 3.45 s against 3.74 s at 24 threads.
  - Teardown is nearly free because Phoenix++ never frees combiner state; we do not add that cost.

**2026-11-26: corpus sizes, E4 and E6 done** (`bench/experiments.sh sizes e4 e6`, 1,060 runs, 5 each in shuffled order, performance profile). Every file has one index checksum, including the Phoenix++ runs. Figures are in `results/figures/`: `e1_sizes`, `e4_vocab_T{8,24}_{time,speedup,memory}`, `e4_len_*`, `e6_*`. All experiments in the catalogue's list are now complete.
- **Corpus sizes** (25% and 50% are every 4th and every 2nd document, so only the size changes; figure `e1_sizes`). Practical speedup of `heavy-dynamic` at 24 threads, 25% → 50% → 100%:
  - Gutenberg: 9.8 → 10.2 → 10.6×
  - Wikipedia: 7.0 → 7.3 → 7.4×
  - Zipf: 5.4 → 5.8 → 6.0×

  Speedup rises slightly with size (fixed costs amortize) and the curves keep the same shape. The full-size results are not a lucky size.
- **E4 vocabulary sweep** (2M docs, 200M tokens, s = 1; 24 threads):

  | V | distinct terms | `seq` | `omp-dynamic` | `heavy-dynamic` | Phoenix++ | peak RSS: D3 / Phoenix++ / `seq` |
  |---|---|---|---|---|---|---|
  | 10⁴ | 10k | 7.59 s | 0.71 s | **0.66 s (11.6×)** | 0.88 s | 1.6 / 2.3 / 1.7 GB |
  | 10⁵ | 100k | 10.70 s | 1.71 s | **1.67 s (6.4×)** | 1.71 s | 2.0 / 2.9 / 1.9 GB |
  | 10⁶ | 1.0M | 20.43 s | 3.45 s | 3.44 s | **2.94 s (6.9×)** | 3.8 / 4.7 / 2.2 GB |
  | 10⁷ | 8.9M | 33.75 s | 5.82 s | **5.77 s (5.9×)** | 6.28 s | 6.6 / 8.7 / 3.3 GB |

  - A small vocabulary stays in cache, so it scales best (11.6×).
  - From 10⁵ on, speedup falls and parallel memory grows faster than `seq`'s (6.6 against 3.3 GB at 10⁷), because each thread's dictionary holds its own copy of most of the vocabulary.
- **E4 document-length sweep** (V = 1M, 200M tokens; mean length 10 to 10⁴ tokens): length barely matters. Our time is 3.49 → 3.26 s and Phoenix++ is 3.19 → 2.78 s; practical speedup is 6.5–7.0×. Per-document overhead is negligible even at 10 tokens.
- **E6 Phoenix++ against D2 and D3** (all thread counts; 24 threads shown):

  | corpus | replication at 24 threads | `omp-dynamic` | `heavy-dynamic` | Phoenix++ | time after map, D3 / Phoenix++ | peak RSS, D3 / Phoenix++ |
  |---|---|---|---|---|---|---|
  | Zipf | 17.1× | 3.44 s | 3.45 s | **2.94 s** | 1.47 / 0.75 s | 3.8 / 4.7 GB |
  | Gutenberg | 3.8× | 1.17 s | **1.08 s** | 1.15 s | 0.21 / 0.22 s | 2.2 / 2.5 GB |
  | Wikipedia | 3.2× | 2.33 s | **2.26 s** | 2.68 s | 0.81 / 1.12 s | 4.2 / 5.6 GB |

  (Replication is the number of entries in all threads' local dictionaries divided by the number of distinct terms.)
  - **The map phases are equal:** both are per-thread hash tables. What decides the result is the step after map.
  - **Phoenix++ hash-partitions terms into reduce buckets**, so each term is combined once. Our D2 sorts every thread's local dictionary, then merges the sorted runs by key range, so its cost grows with replication.
    - On the synthetic corpus, where tokens are drawn independently, replication is 17×. There Phoenix++ is 15% faster: its post-map work is half of ours (0.75 against 1.47 s).
    - On real text (3–4×, because rare words cluster in a few documents) our sorted merge wins. Wikipedia is 16% faster and uses 24% less memory (4.2 against 5.6 GB); Gutenberg ties.
  - Phoenix++ always uses more memory: 17–31% more in E6 and up to 44% more in E4. It keeps posting lists, partitioned copies of the tables and the reduced copy alive at once.
  - Phoenix++ never frees its combiner state, so its teardown is about 0 s. Ours frees 0.01–0.24 s inside `build_total`, which gives Phoenix++ a small advantage that this comparison does not correct.
  - **For the report:** our design is the right one for real text, and hash partitioning is the right one when replication is high. A natural next step is a hybrid that partitions light terms by hash and leaves heavy terms to D3's bitmaps; it is beyond the class scope.

**2026-11-27: report drafted** in `report/` (7 pages, A4). `report/report.html` is the source; `report/build.sh` regenerates the figures (`make_figures.py`, from `inverted-index/results`) and prints the PDF with headless Brave/Chromium, because no LaTeX is installed. Fill in the highlighted placeholders before submitting: the authors and IDs, and the AI-use and per-member contribution statement required by the course policy. Still to do: the presentation and the demo.

**2026-11-27: presentation drafted** as an 18-slide deck (a claude.ai Slides artifact, downloadable as PowerPoint or PDF). It uses a deep-green theme, and sage slides tie each design choice to a lecture in `Slides/`:
- Decomposition Techniques: input, then output, data decomposition.
- Process Mapping & Load Balancing: dynamic mapping on hybrid cores.
- Brent's Theorem & Parallel Merge Sort: splitter merge vs merge tree.
- PRAM and Multi-Core Architectures: concurrent writes, atomics.
- Analytical Modeling: T_o = pT_p − T_s split into T_idle and T_excess; true vs relative speedup.

Its charts come from `presentation/make_charts.py`. Still to do: the demo.

## 9. How to check the finished project works end to end

- `make && make test`: hand tests plus randomized differential tests. All variants must match `naive` byte for byte at every thread count.
- `make asan` is clean, and `make tsan` is clean too once `openmp` is installed.
- `idxcheck --validate` passes on every full-size output, and `idxcheck a.bin b.bin` reports `seq`, `omp` and `heavy` as identical on all three corpora.
- `bench/experiments.sh all` reruns every experiment (about 4.5 hours) and the figures; `make phoenix && ALGOS=phoenix tests/run_tests.sh` checks the Phoenix++ build.
- Speedup figures quote both scaling speedup and practical speedup.

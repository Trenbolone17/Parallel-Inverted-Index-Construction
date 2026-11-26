# Viva preparation: Parallel Inverted Index Construction (CS F422, Project 23)

This guide explains the project from the ground up: the problem, the background ideas, every design decision, how we tested and measured, what the results mean, and how it all connects to the lectures. It ends with likely viva questions with model answers, a one-page sheet of numbers and a glossary.

**How to use it:**
- Read sections 1–3 for the big picture.
- Read sections 4–9 for each part of the design, with file names, so you can open the code while reading.
- Read sections 10–13 for testing, results and theory.
- Practise answering section 15 out loud, without looking.

The examiner will most likely ask *why* we did things and *what the numbers mean*, not just *what* we did.

---

## 1. The project in one minute

We built a program that takes a large collection of text documents and produces an **inverted index**: for every word, the sorted list of documents that contain it. We wrote it three ways:

| | What it is | Where |
|---|---|---|
| **D1** | The best sequential builder we could make: the baseline. | `src/build_seq.cpp` |
| **D2** | A parallel builder in OpenMP. Every thread indexes its share of the documents into a private dictionary, and the dictionaries are merged in sorted order, in parallel. | `src/build_omp.cpp` |
| **D3** | Our improvement to D2: very frequent words ("the", "of") are stored as bitmaps (one bit per document) instead of lists, with a memory budget. | `src/build_omp.cpp`, `--algo heavy` |
| **Baseline** | The same index built on **Phoenix++**, the MapReduce runtime from the authors of our starting paper. | `baselines/phoenix_index.cpp` |

**Headline results** on a 24-thread laptop CPU:
- **Speedup over the sequential builder:** 6.0× (synthetic text), 10.6× (Project Gutenberg books) and 7.4× (Wikipedia).
- **The map phase scales about as far as the hardware allows.** The limit is the merge, because every thread's dictionary repeats much of the vocabulary. We call that *replication*.
- **D3** removes the imbalance it targets but gains only 1–8% end to end.
- **Against Phoenix++,** our design wins on real text: 16% faster with 24% less memory on Wikipedia. It loses by 15% on synthetic text, where replication is extreme.
- **Correctness:** every builder produces a byte-identical file, checked by more than 10,000 tests and in all 1,795 benchmark runs.

---

## 2. Background, from the basics

### 2.1 What an inverted index is

A normal ("forward") view of documents is: *document → the words in it*. An **inverted** index flips this: *word → the documents containing it*. It is the core data structure of every search engine.

Example with three documents (numbered from 0):

```
doc 0: the cat sat          word   documents  (the "posting list")
doc 1: the dog sat    →     a      2
doc 2: a cat ran            cat    0, 2
                            dog    1
                            ran    2
                            sat    0, 1
                            the    0, 1
```

**Vocabulary:**
- **Term** means a word.
- **Posting** means one (term, document) pair.
- **Posting list** is a term's sorted list of document IDs.
- **df** (document frequency) is the length of a term's posting list: how many documents contain the term.

**Why it is useful:** to answer the query "cat AND sat", intersect two sorted lists, {0, 2} ∩ {0, 1} = {0}, without reading any document. Sorted lists make the intersection a linear merge. Ranking, positions and compression are outside our scope (the catalogue excludes them).

### 2.2 What "building" an index involves

1. **Tokenize:** split text into words, lowercase them, drop punctuation. We do this **once, offline** (`tools/pretokenize.py`), and time it separately, as the catalogue says preprocessing must be reported separately.
2. **Group:** for every token, find its term's list and append the current document's ID, once per document even if the word repeats.
3. **Sort** the terms, so the output has a fixed order.
4. **Lay out** the result as compact arrays.

It is a **group-by**. Our corpora have 200–330 million tokens, which collapse into 30–215 million distinct (term, document) pairs.

### 2.3 Why it is hard to make fast

- **Memory-bound.** Every token is a hash-table lookup at an unpredictable address, so most lookups miss the CPU caches and wait for main memory. Our sequential builder spends 35–100 ns per token, and 88–97% of its time, in this "map" step. Speed is limited by memory latency, not arithmetic.
- **Skew (Zipf's law).** In natural language a word's frequency is roughly proportional to 1/rank. A handful of words ("the", "of", "and") appear in almost every document, and millions of rare words appear once or twice. Long lists and short lists coexist, so equal-sized pieces of work are hard to make.
- **Merging across threads.** If threads work separately, their partial results for the same word must be combined: the parallel version has work the sequential one doesn't.

### 2.4 Parallel computing ideas we rely on

- **Shared memory and threads.** All threads of one process see the same memory. That is cheap communication, but writes to the same location need care (data races).
- **OpenMP.** A C/C++ extension: `#pragma omp parallel` starts a team of threads, and `#pragma omp for` splits a loop's iterations among them.
  - `schedule(static)` assigns fixed blocks in advance.
  - `schedule(dynamic, k)` hands out chunks of k iterations to whichever thread is free.
  - `atomic` makes a read-modify-write indivisible.
  - `nowait` drops the barrier at the end of a loop.
- **Speedup.** S = T_seq / T_par(p). The lecture distinguishes **true (absolute) speedup**, against the *best sequential* program, from **relative speedup**, against the parallel program on 1 thread. We report both: "practical" = true, "scaling" = relative.
- **Efficiency.** E = S / p, the fraction of the processors' time spent on useful work.
- **Overhead.** T_o = p·T_p − T_s: everything the p processors do beyond the sequential work. The lecture splits it into communication/interaction, idling (load imbalance, waiting) and excess computation (work the sequential program never does).
- **Load balance.** The parallel time is set by the slowest thread. Imbalance is measured as max/mean of per-thread times (1.00 is perfect).
- **Cache coherence and false sharing.** Caches work in 64-byte lines. If two cores write to the same line, even to different bytes, the line bounces between them (false sharing). If two threads read-modify-write the same 64-bit word without atomics, one can overwrite the other's update.
- **Amdahl's law.** If part of the work does not speed up, it bounds the total speedup.
- **Brent's theorem.** A computation with total work W and depth (critical path) D runs on p processors in T_p ≤ W/p + D.

### 2.5 The machine: a hybrid CPU

Intel Core i7-13700HX laptop:
- **8 P-cores** (performance), each with 2 hardware threads (hyperthreading, SMT), up to 5.0 GHz.
- **8 E-cores** (efficiency), slower: up to 3.7 GHz, with simpler cores.
- **24 hardware threads** in total, a 30 MB shared L3 cache and one memory node (UMA, no NUMA).

Three consequences matter for us:
1. **Cores are not equal.** A thread on an E-core does less work per second, so equal-sized static work leaves P-core threads waiting.
2. **Clock speed drops under load (turbo).** We measured 4.75 GHz with one busy core and 3.94 GHz with all 8 P-cores busy. Eight cores therefore cannot deliver 8× one core: at most about 8 × 3.94 / 4.75 ≈ 6.6×.
3. **Hyperthreads share a core.** Two threads on one P-core share its execution units, so the second thread adds only a fraction of a core.

**Thread pinning.** We fix every thread to a specific CPU (`OMP_PLACES`, `OMP_PROC_BIND=close`) in a deliberate order:
- 1–8 threads: one per P-core, first hyperthread only.
- 12 and 16 threads: add E-cores.
- 24 threads: add the P-cores' second hyperthreads.

The order is read from Linux's `/sys` files (`bench/run.py: placement_order`). Pinning makes results repeatable and lets us read the scaling curve in regions: P-cores, then E-cores, then hyperthreads.

### 2.6 MapReduce and Phoenix (the starting paper)

**MapReduce** organizes a computation as:
1. **map:** turn input pieces into (key, value) pairs;
2. **combine / local aggregation:** pre-aggregate pairs with the same key inside each worker;
3. **reduce:** combine each key's values across workers;
4. **merge:** sort the final output.

Ranger et al. (HPCA 2007) built **Phoenix**, MapReduce for shared-memory multicores, and showed it performs well on word count and similar tasks. **Phoenix++** (Talbot et al., 2011) is the authors' rewrite, with pluggable *containers* (hash tables, arrays) and *combiners*.

An inverted index is a MapReduce job: map emits (word, document), combine collects each word's documents per worker, and reduce unites them across workers. Our D2 follows the same shape, but with a different reduce: a sorted range merge instead of hash partitioning. That difference turns out to matter (§12.9).

---

## 3. What the catalogue asked for, and where we did it

**Project 23 itself:**

| Requirement | Where |
|---|---|
| D1: sequential builder + posting-list equality tests | `build_seq.cpp`, `build_naive.cpp` (oracle), `tests/run_tests.sh`, `idxcheck` |
| D2: OpenMP local dictionaries with sorted merging | `build_omp.cpp` (`--algo omp`) |
| D3: frequent-term handling under a memory budget | `build_omp.cpp` + `local_index.hpp` (`--algo heavy`) |
| Baseline: Phoenix | `baselines/phoenix_index.cpp`, `make phoenix`, experiment E6 |
| Vary vocabulary, document length, term skew; include sorting, deduplication and final merge | E4 (vocabulary, length), E3 (skew); build time includes all of them |

**Common requirements:**

| Requirement | Where |
|---|---|
| Hand-checkable inputs, randomized differential tests, clear output semantics | `tests/`, the canonical format (§4) |
| ≥ 3 input families, multiple sizes | synthetic Zipf, Gutenberg, Wikipedia; 25/50/100% sizes |
| Threads 1, 2, 4, … up to all cores; repeated timings | 1, 2, 4, 8, 12, 16, 24 threads; 5 runs each, shuffled |
| Runtime, peak memory, algorithmic work | `--stats` JSON: phase times, peak RSS, counters |
| Preprocessing separate from kernel time | tokenization, load and write reported separately |
| Ablation + unfavorable case | E5 (θ, sample, budget), static vs dynamic; 4 unfavorable cases |
| Scaling vs practical speedup, same machine | both reported; one laptop for everything |
| 6–8 page report, reproducibility, presentation, demo, attribution | `report/report.pdf` (7 pages), README, slide deck, `demo.sh` |

---

## 4. The input/output contract (why correctness is easy to check)

### Input (`include/corpus.hpp`)

- **Documents:** one document per line, and the **document ID is the line number** (from 0). An empty line is an empty document but still takes an ID.
- **Tokens:** maximal runs of non-whitespace bytes. The bytes are opaque, so UTF-8 passes through.
- **Limit:** fewer than 2³² − 1 documents, so an ID fits in 32 bits; the value `UINT32_MAX` is reserved as "no document".

### Output: the canonical binary index (`include/index_format.hpp`)

```
"INVIDX01" | n_docs, n_terms, n_postings (u64)
term_off[n_terms+1] (u64) | term_bytes[]          ← all terms concatenated, in strict byte order
post_off[n_terms+1] (u64) | postings[] (u32)      ← each term's list, strictly increasing
```

Term i is `term_bytes[term_off[i] .. term_off[i+1])`. Its list is `postings[post_off[i] .. post_off[i+1])`.

**Why "canonical" matters.** For a given input there is exactly **one** correct file: terms in byte order, lists sorted and without duplicates. So:

> A builder is correct exactly when its output file is **byte-identical** to the reference builder's, whatever the thread count, schedule or heuristic.

This turns correctness into `cmp`-style equality, which is easy to automate and impossible to fool. It is the key idea behind our testing.

### Tools

`bin/idxcheck` handles index files:
- `A B` compares two indexes;
- `--validate` checks the invariants, and with `--corpus` independently recounts the postings from the text;
- `--dump` prints the index as text;
- `--info` prints sizes and the checksum;
- `--lookup` finds a word's list by binary search.

---

## 5. Code tour

```
inverted-index/
  include/hash.hpp          hash_bytes: one hash function for every builder
  include/hash_table.hpp    StringTable: open-addressing table, shared by every builder
  include/local_index.hpp   the inner indexing loop (add_docs, add_docs_heavy)
  include/corpus.hpp        input contract, loader, document boundaries, tokenizer loop
  include/index_format.hpp  canonical output, read/write, checksum, compare, validate
  include/pod_vector.hpp    a vector that doesn't zero memory on resize (speed)
  include/stats.hpp         phase timers, counters, peak RSS → JSON
  src/build_naive.cpp       the test oracle: std::map<string, set<uint32>>
  src/build_seq.cpp         D1
  src/build_omp.cpp         D2 and D3
  src/invidx.cpp            command line: --algo naive|seq|omp|heavy, options, --stats
  src/idxcheck.cpp          compare / validate / dump / info / lookup
  tools/gen_zipf.cpp        synthetic corpus generator
  tools/pretokenize.py      raw text → one-document-per-line tokens
  tools/fetch_corpus.py     downloads Gutenberg / Wikipedia from Hugging Face
  baselines/phoenix_index.cpp   our index as a Phoenix++ application
  tests/                    hand-written cases + run_tests.sh
  bench/run.py, plot.py, experiments.sh   benchmark harness, figures, all experiments
  results/                  raw results of every benchmark run
  demo.sh                   the live demo
```

Build with `make`, test with `make test`, run the demo with `./demo.sh`, all inside `inverted-index/`.

---

## 6. Shared machinery (used by every builder)

**One hash table for everyone** (`StringTable`, `hash_table.hpp`):
- **Open addressing with linear probing:** all entries live in one array of slots. On a collision we try the next slot.
- **Load factor ≤ 0.5:** the table doubles when half full, which keeps probe chains short.
- **Small slots:** each slot is 8 bytes, a 32-bit *tag* (the top half of the hash) plus the entry's index. A probe compares tags first and only touches the full entry when the tag matches, so most probes stay in one cache line.
- **No copied strings:** keys are pointers into the loaded input buffer, so there is no allocation per token.
- **Entries in insertion order** in a dense vector, which makes sorting them later simple.

**One hash function** (`hash_bytes`): it reads the word 8 bytes at a time, mixes each block, and ends with MurmurHash3's finalizer. The word's length is folded into the seed.

**Why share them?** If D2 used a better hash table than D1, part of the "speedup" would come from the data structure, not parallelism. Sharing them means our speedups measure parallelization only.

**The inner loop** (`add_docs` in `local_index.hpp`) is also shared by D1 and D2. For each document in order, and each token in it:
1. Find or insert the term.
2. If the term's list doesn't already end with this document, append the document's ID.

This **last-document check** makes deduplication free, because documents are visited in increasing order, so a repeat within a document always meets its own ID at the end of the list.

**Document boundaries** (`Corpus::index_docs`) are found in parallel:
1. Each thread counts the newlines in its slice of the buffer.
2. A prefix sum gives each slice its first document number.
3. Each thread fills in `doc_start[]` for its slice.

---

## 7. D1: the sequential builder (`build_seq.cpp`)

Four phases, timed separately:
1. **map:** `add_docs` over all documents into one table.
2. **sort:** sort the distinct terms by bytes (an index array sorted by `string_view` comparison).
3. **layout:** compute the array sizes, then copy every term's bytes and list into the canonical arrays.
4. **free:** release the table. Freeing millions of vectors is a real cost, so we time it.

**Costs:**
- **Time:** about O(tokens) hash operations + O(V log V) string comparisons for the sort + O(postings) copying (V = distinct terms).
- **Memory:** the input buffer, the table (entries plus slots) and every posting list.

**Why it is the "best sequential" baseline.** The catalogue's practical speedup must be measured against the best sequential program, not the parallel program on 1 thread. D1 has no parallel overhead at all, and our parallel builders at 1 thread are within 2% of it. That shows both that D1 is a fair baseline and that parallelization costs little.

The **naive** builder (`std::map<std::string, std::set<uint32_t>>`) is deliberately simple and slow, with its own input reader. It is the **test oracle** and is never benchmarked.

---

## 8. D2: local dictionaries with sorted merging (`build_omp.cpp`, `--algo omp`)

```
 scan ──► map ──► local sort ──► splitters ──► count pass ──► prefix sums ──► write pass ──► free
(parallel (private  (each thread   (sample runs,   (T-way merge    (offsets per    (write terms and
newlines)  table per sorts its own  4T ranges,      per range,      range)          lists in place)
           thread)   dictionary)    binary-search   count terms
                                    cuts)           and postings)
```

### 8.1 Map: each thread builds a private dictionary

Each of the T threads has its own `StringTable` and runs the same `add_docs` loop on its documents. There are no locks and no shared writes. The question is *which* documents each thread gets.

**Static schedule** (default):
- **How:** the documents are cut into T contiguous blocks of equal *bytes* (`byte_balanced_bounds`, a binary search on `doc_start`). We balance bytes, not document counts, because Gutenberg books vary wildly in length.
- **Benefit:** every document ID in thread t's block is smaller than every ID in block t+1. So a word's lists from different threads, **concatenated in thread order, are already sorted**, and merging them is a copy.

**Dynamic schedule** (`--schedule dynamic`):
- **How:** the documents are cut into 16 × T byte-balanced chunks, handed out at run time with `schedule(monotonic:dynamic, 1)`. A thread that finishes early takes another chunk, so fast P-cores do more chunks than slow E-cores. *Monotonic* means each thread receives its chunks in increasing order, so each thread's own lists stay sorted.
- **Cost:** a word's lists from different threads **interleave**, so merging them needs a **k-way merge** (a min-heap over the lists' heads, `merge_lists`).

### 8.2 Local sort

Each thread sorts its own distinct terms, producing a sorted **run**: T sorted runs in total.

### 8.3 Splitters: dividing the merge into ranges

We want to merge T sorted runs in parallel. We cut the space of words into **R = 4T ranges** (*oversplitting*: more ranges than threads, so a dynamic schedule over ranges can balance uneven ranges).

1. **Regular sampling** (as in parallel sorting by regular sampling): take up to 64·R evenly spaced sample words, from each run in proportion to its size; sort the samples; pick every (S/R)-th sample as a **splitter**.
2. **Cuts:** in each run, binary-search (`lower_bound`) each splitter. Every run is cut **at the same words**, so a word never ends up in two ranges.

### 8.4 Count pass (`merge` phase)

In parallel over ranges, with `schedule(dynamic, 1)`, each range merges its T run-pieces with a min-heap on (word, thread). Equal words come out together, in thread order. The pass counts the range's distinct words, bytes and postings, and records the merged order compactly. Each record is one 64-bit number:
- bit 63: "a new word starts here";
- bits 32–39: which thread;
- bits 0–31: which entry in that thread's table.

### 8.5 Prefix sums, then the write pass (`layout` phase)

A prefix sum over the ranges' counts gives every range its exact output position. The output arrays are allocated once at their final size. Then, in parallel over ranges, the **write pass** replays the recorded order and writes each word's bytes and posting list in place:
- **static schedule:** concatenation;
- **dynamic schedule:** k-way merge.

**Why two passes?** Without knowing sizes in advance, ranges couldn't write into one shared output array without locks or reallocation. Counting first makes every write go to a precomputed, disjoint place (the lecture's *owner computes*). The result is also deterministic.

A consistency check compares the written totals with the counted ones and throws an error if they differ. It is a guard against merge bugs.

### 8.6 Free

Each thread frees its own table in parallel. Memory allocators keep per-thread areas, so freeing on the allocating thread is fastest.

### 8.7 What it costs, and where the time goes

- **map:** O(tokens / T) per thread, the bulk of the work.
- **local sort:** each thread sorts **its own** distinct words. If every thread sees most of the vocabulary, the threads together sort many copies of it: that is **replication**.
- **merge + write:** proportional to the total entries across all threads (replicated) plus the postings.

**Replication** = (entries in all local dictionaries) / (distinct words). At 24 threads:

| Corpus | Replication |
|---|---|
| Synthetic Zipf | 17.1× |
| Gutenberg | 3.8× |
| Wikipedia | 3.2× |

Real text is lower because rare words are *bursty*: they cluster in a few documents, so they appear in few threads. The synthetic generator draws every word independently, so even rare words appear everywhere. Replication is the main reason the phases after the map scale poorly (§12.3).

---

## 9. D3: heavy-term bitmaps under a memory budget (`--algo heavy`)

### 9.1 Why

On Wikipedia, the 104 words that appear in at least 1/32 of the documents are **43% of all tokens** and **35% of all postings**. In D2 these words:
- grow huge per-thread vectors by repeated doubling (copying);
- under the dynamic schedule, each need a k-way merge done by **one thread**, so the write pass is unbalanced: its imbalance (max/mean) is 1.35 at 24 threads on Wikipedia.

### 9.2 The idea

For a word that appears in a large fraction of documents, a **bitmap** (one bit per document, bit d set if document d contains the word) is no bigger than the list, and it needs **no merging**:
- every thread sets bits for its own documents directly in one shared bitmap;
- bit positions *are* document IDs, so the result is automatically in order.

### 9.3 Step by step

1. **Sample.** Look at an evenly spaced 1% of the documents (`--sample 0.01`). Each thread counts, for its share of the sample, how many sampled documents contain each word.
   - **Pigeonhole shortcut:** a word whose total count is ≥ θ·m (m = sampled documents) must reach θ·mₜ in at least one thread's share. So each thread nominates only its own frequent words, and exact counts are then summed for those candidates only.
2. **Classify.** A word is **heavy** if (sampled df) / m ≥ θ. The default **θ = 1/32** is the break-even point:
   - an N-bit bitmap takes N/8 bytes;
   - a list of df 32-bit IDs takes 4·df bytes;
   - they are equal when df = N/32.
3. **Budget.** Keep at most H heavy words, so that H bitmaps of N/8 bytes (rounded up to whole 64-byte cache lines) fit in `--heavy-mem-mb` (default 1 GB). The most frequent words are kept first.
   - Misclassifying a word only affects speed, never correctness: a "missed" heavy word just takes the normal list path.
4. **Set up.** Allocate the bitmaps 64-byte aligned and zeroed, and pre-insert every heavy word into every thread's table with its heavy ID.
5. **Map** (`add_docs_heavy`). When a token is a heavy word and this is its first occurrence in the current document (the same last-document check as D2), set bit d in that word's bitmap instead of appending to a list.
6. **Counting** (`heavy_count`). In parallel tasks of (heavy word × segment of 4,096 words = 262,144 documents), count the set bits with `popcount`. That gives each heavy word's df and each segment's starting offset.
7. **Merge.** Light words are merged exactly as in D2. Heavy words are not merged at all: the write pass **reserves** each heavy word's slot in word order and the space for its list.
8. **Decode** (`heavy_decode`). Balanced parallel tasks per (word × segment) walk the bitmap words and write the document IDs: for each set bit, `countr_zero` (count trailing zeros) gives its position, then `x &= x − 1` clears it.

### 9.4 The concurrency question: why only two atomic writes per chunk

Several threads write into the **same** bitmap. Hardware doesn't write single bits; it reads, modifies and writes whole 64-bit words. If two threads update different bits of the **same word** at once, one update can be lost.

- **PRAM view:** in the lecture's Common-CRCW model, concurrent writes of the same value to one location are allowed. Real hardware is word-granular, so we can't rely on that.
- **Our solution:** a thread's documents form contiguous chunks, so the only bitmap words it can share with another thread are the **first and last word of each chunk**. Those two words use an atomic OR (`std::atomic_ref<uint64_t>::fetch_or`, relaxed ordering). Every word strictly inside a chunk belongs to one thread alone and gets a plain OR.
- **Why relaxed ordering is enough:** OR operations commute, and nothing reads the bitmaps until the parallel region's barrier, which orders all writes.
- **False sharing:** it can only occur at chunk boundaries, a negligible amount.

### 9.5 Two design changes forced by measurements

1. **We dropped chunk alignment.** We first aligned chunk boundaries to multiples of 512 documents (one 64-byte cache line of bitmap), so no word would ever be shared. On Gutenberg, with only 4,976 very long books, alignment collapsed 24 chunks into about 10 and made D3 **2× slower**. Boundary atomics replaced alignment.
2. **We added the last-document check for heavy words.** Without it, every repeat of a common word inside a long book rewrote a bitmap word in memory: D3 was **23% slower** on Gutenberg. With the check, repeats only touch the table entry, which is already cached.

---

## 10. The Phoenix++ baseline (`baselines/phoenix_index.cpp`)

The catalogue names Phoenix as the baseline, but its example apps don't build a word index (its "reverse index" indexes HTML links). So we wrote **our index as a Phoenix++ application**.

**Our code supplies:**
- **split:** about 1 MB chunks of whole documents, like Phoenix++'s word count;
- **map:** emit (word, document) for each token;
- **combiner:** a per-thread posting list with the same last-document deduplication;
- **reduce:** sort a word's document IDs;
- the canonical layout.

**Phoenix++'s own, unmodified code does:**
- task scheduling with work-stealing queues;
- per-thread hash tables;
- **hash partitioning** of each table into T reduce buckets;
- the cross-thread union in reduce;
- **sorting and merging** the reduced words (`MapReduceSort`: a pairwise merge tree).

**Setup:**
- **Fairness:** same input, same pinning (threads go to the CPUs listed in `OMP_PLACES`; Phoenix++'s default would put threads 0–7 on 4 P-cores, both hyperthreads each), and the same output, checked by checksum.
- **Build:** `make phoenix` clones the authors' repository at a fixed commit. GCC 16 needs `-fpermissive`, because Phoenix++ predates C++11 name-lookup rules. We didn't change their code.
- **Tests:** it passes the same tests (`ALGOS=phoenix tests/run_tests.sh`: 451 checks).

---

## 11. Correctness and testing

| Layer | What it checks |
|---|---|
| **13 hand-written cases** (`tests/make_tiny.py`) | Expected outputs **written by hand**: empty corpus, empty documents, a word repeated in one document, a word in every document, identical documents, prefix words (`a`, `ab`, `abc`), unusual whitespace, missing final newline, UTF-8, uppercase vs lowercase, punctuation, a single document, a 10,000-byte token |
| **200 random corpora** | `gen_zipf` with random size, vocabulary, skew and length distribution; every builder variant must be byte-identical to the naive oracle at **1, 2, 3, 7, 16 and 24 threads**. Odd thread counts catch off-by-one bugs at block and range boundaries. |
| **9 builder variants** | D1; D2 static, dynamic, and with one range per thread; D3 default, all words heavy, dynamic, budget-capped (only a few heavy), budget 0 (must equal D2) |
| **A 600,001-document corpus** | each bitmap spans several 262,144-document decode segments |
| **Independent recount** (`idxcheck --validate --corpus`) | counts documents and distinct (word, document) pairs straight from the text with different code |
| **Sanitizers** | AddressSanitizer + UndefinedBehaviorSanitizer: 2,622 checks clean |
| **Planted bugs** (mutation testing) | we inserted bugs on purpose: an inconsistent range cut, wrong segment offsets, a wrong heavy/light order, a dropped bitmap word. All were caught. One planted "bug" turned out to be equivalent to correct code (switching every cut consistently gives the same output), which is a nice illustration of why the cuts work. |
| **Benchmark checksum guard** | every one of the 1,795 benchmark runs records the index checksum; all runs on a corpus must agree, or the harness fails |

**Total:** 10,462 checks pass in `make test`, plus 451 for Phoenix++.

**Not done:** ThreadSanitizer. It needs LLVM's OpenMP runtime (`sudo pacman -S openmp`), which isn't installed, and GCC's libgomp gives false positives. The bitmap atomics are race-free by the argument in §9.4. Everything else writes only thread-private data or disjoint, precomputed output ranges.

---

## 12. Experiments and results

### 12.1 Method

**Corpora (three families):**

| | Synthetic Zipf | Gutenberg books | Wikipedia leads |
|---|---|---|---|
| Source | `gen_zipf` (Zipf s = 1, 10⁶-word vocabulary, seed 7) | PG-19 dataset, 4 shards | first paragraph of every English article |
| Documents | 2,000,000 | 4,976 | 6,407,814 |
| Words per document | 100 | 65,438 | 44.5 |
| Tokens | 200 M | 326 M | 285 M |
| Distinct words | 1.00 M | 0.78 M | 2.78 M |
| Postings | 163 M | 30 M | 214 M |
| Input / index size | 0.71 / 0.67 GB | 1.73 / 0.14 GB | 1.68 / 0.93 GB |
| Preprocessing | generation < 1 s | tokenizing 13.3 s | tokenizing 53.9 s |
| Sequential build | 20.65 s | 11.80 s | 16.95 s |

The families differ on purpose. Gutenberg has few, enormous documents: 0.09 postings per token, because words repeat within books. Wikipedia has millions of short documents: 0.75 postings per token. The synthetic corpus lets us control vocabulary, length and skew.

**Protocol:**
- Every run is a **fresh process** (so its peak memory is clean), after one untimed warm-up that loads the file into the OS page cache.
- **5 runs** per configuration, in a **new random order** each repetition, so slow drifts such as heat spread evenly. We report the **median**; repeated runs differed by a median of 0.5–2.5%.
- AC power, "performance" power profile, nothing else running.

**Build time** covers everything from the document-boundary scan to a finished in-memory index, including freeing temporaries. Loading the file (0.1–0.3 s), writing the index (0.05–0.19 s) and tokenizing are reported separately.

**Experiments:**
- **E1:** strong scaling, 3 corpora × 1–24 threads × 5 variants.
- **Sizes:** 25% and 50% of each corpus.
- **E3:** skew, Zipf s from 0 to 1.4.
- **E4:** vocabulary 10⁴–10⁷, and document length 10–10,000 words.
- **E5:** D3 ablations (θ, sample rate, budget).
- **E6:** Phoenix++.

### 12.2 Strong scaling (E1)

Median build time in seconds:

| | Zipf, 8 threads | Zipf, 24 | Gutenberg, 8 | Gutenberg, 24 | Wikipedia, 8 | Wikipedia, 24 |
|---|---|---|---|---|---|---|
| seq (1 thread) | 20.65 | | 11.80 | | 16.95 | |
| D2 static | **4.63** | 3.48 | 2.10 | 1.31 | **3.43** | 2.34 |
| D2 dynamic | 4.75 | 3.45 | 1.96 | 1.20 | 3.61 | 2.35 |
| D3 static | 4.66 | 3.50 | 2.13 | 1.28 | 3.51 | 2.36 |
| D3 dynamic | 4.74 | **3.42** | **1.92** | **1.11** | 3.49 | **2.28** |
| **true speedup of the best** | 4.5× | **6.0×** | 6.1× | **10.6×** | 4.9× | **7.4×** |

- **Relative speedups** (D3 dynamic vs itself at 1 thread): 6.0×, 10.5×, 7.9×.
- **At 1 thread** every D2 variant is within 2.1% of seq.
- **8 → 24 threads:** E-cores and hyperthreads add 1.35–1.73× on top of the 8 P-cores.

**Static vs dynamic.** With static blocks, E-core threads lag: map imbalance is 1.07–1.19, against 1.02–1.03 with dynamic chunks.
- **Gutenberg:** dynamic clearly wins (1.20 vs 1.31 s at 24 threads).
- **Wikipedia at 8 threads:** static wins (3.43 vs 3.61 s), because the dynamic schedule's k-way merge costs 0.46 s against 0.08 s for concatenation. That merge cost is exactly what D3 attacks.

### 12.3 Where the time goes

**The map phase is close to the hardware limit.**
- **Clock ceiling:** with all 8 P-cores busy the clock drops from 4.75 to 3.94 GHz, so 8 P-cores can give at most about 6.6× one core.
- **Measured map speedup:** 5.3–6.8× at 8 threads, and 10.3–13.2× at 24 (D3 dynamic).
- **Per-core slowdown:** the total CPU time spent in map at 8 threads is 1.14× (Gutenberg), 1.28× (Wikipedia) and 1.42× (Zipf) the sequential map. The clock drop (1.21×) explains the real-text numbers. On Zipf, independent random draws from a million words add extra memory traffic.

**The phases after the map are the gap.** At 24 threads, local sort + merge + write + free take **23%** (Gutenberg), **38%** (Wikipedia) and **43%** (Zipf) of the build time. They grow with replication (§8.7). On Zipf, the 24 threads' local sorts take 0.78 s, **twice** the sequential builder's single sort of the whole vocabulary (0.40 s).
- **Amdahl estimate:** if the post-map work scaled like the map, Zipf would build in about 2.0 s (10×) instead of 3.4 s.

**Overhead breakdown** (lecture formula, Wikipedia, D2 static on 8 P-cores):
- T_s = 16.95 s and T_p = 3.43 s, so p·T_p = 27.4 core-seconds.
- **T_o = 10.5 s**, and efficiency **E = 0.62**.

| Component | Core-seconds | Lecture category |
|---|---|---|
| Useful work | 16.9 | T_s |
| Slower cores: the clock drop plus shared cache and memory contention | 4.2 | interaction; the clock drop is outside the lecture's model, which assumes identical processors |
| Idling in the map (threads finish unevenly, max/mean 1.12) | 2.3 | T_idle |
| Phases after the map: replicated sorting and merging, plus their idling | 4.0 | mostly T_excess |

### 12.4 Corpus size

Practical speedup of D3 dynamic at 8 / 24 threads:

| Corpus | 25% | 50% | 100% |
|---|---|---|---|
| Zipf | 4.0 / 5.4 | 4.3 / 5.8 | 4.4 / 6.0 |
| Gutenberg | 5.9 / 9.8 | 6.0 / 10.2 | 6.1 / 10.6 |
| Wikipedia | 4.7 / 7.0 | 4.8 / 7.3 | 4.9 / 7.4 |

Speedup rises slightly with size, because fixed costs (starting threads, allocations, the scan) are spread over more work. The curves keep their shape, so the result isn't an artifact of one size. (This is the Gustafson / isoefficiency intuition: bigger problems keep processors more efficient.)

### 12.5 Skew (E3)

2M documents, 10⁶-word vocabulary, Zipf exponent s from 0 (uniform) to 1.4:
- **Build time** at 24 threads falls from 6.2 s to 0.56 s. Skewed text has fewer distinct (word, document) pairs, and its frequent words stay in cache.
- **D3's benefit grows with skew:** up to 3% faster and 13% less memory at s ≥ 1.2.
- **At s = 0** the sample finds no heavy words, and D3 costs nothing measurable. We had predicted uniform text would be D3's bad case; it isn't.

### 12.6 Vocabulary and document length (E4)

**Vocabulary,** 10⁴ → 10⁷ distinct words, 200M tokens:
- **10k words:** the tables fit in cache, and the build is **11.6×** faster at 24 threads.
- **From 100k words:** about 6×, and parallel memory grows faster than sequential memory: **6.6 vs 3.3 GB** at 10⁷, which is replication again.
- **Phoenix++:** uses 8.7 GB at 10⁷.

**Document length,** 10 → 10,000 words per document: hardly matters, 6.5–7.0× throughout. Per-document overhead is negligible.

### 12.7 D3 ablations (E5; Wikipedia, dynamic schedule, 24 threads)

| Setting | Heavy words | Bitmaps | Time | Write pass (imbalance) | Peak memory |
|---|---|---|---|---|---|
| D2 (no D3) | – | – | 2.32 s | 0.28 s (1.33) | 4.46 GB |
| θ = 1/4 | 11 | 8 MB | 2.30 s | 0.25 s (1.32) | 4.31 GB |
| θ = 1/8 | 20 | 15 MB | 2.28 s | 0.22 s (1.27) | 4.28 GB |
| **θ = 1/32 (default)** | 104 | 79 MB | 2.25 s | 0.18 s (1.16) | **4.23 GB** |
| θ = 1/128 | 557 | 426 MB | **2.25 s** | 0.13 s (1.07) | 4.37 GB |
| θ = 1/512 (1 GB cap) | 1,340 | 1,024 MB | 2.29 s | 0.12 s (1.05) | **4.83 GB (> D2)** |

- **θ trades merge time against bitmap cost.** Lower θ makes more words heavy: the write pass gets cheaper and better balanced, but decoding and bitmap memory grow. Build time is U-shaped; memory is lowest at the break-even θ = 1/32.
- **Sample rate:** a 0.1% sample finds 101 of the 104 heavy words in 3 ms. A 10% sample costs 0.11 s and makes D3 slower than D2 (2.34 s).
- **Budget:** at θ = 1/128, budgets of 16 / 64 / 256 MB keep 20 / 83 / 335 heavy words. 64 MB gives θ = 1/32's memory at θ = 1/128's speed.

**Overall:** D3 achieves its goal (the write pass is 36% faster and much better balanced), but that phase is a small share of the total. End to end it gains 3% on Wikipedia, 1% on Zipf and 8% on Gutenberg, where 96% of tokens hit one of 27,110 heavy words, and uses about 5% less memory.

### 12.8 The unfavorable cases (the catalogue asks for at least one)

1. **θ too low (1/512):** the bitmaps outgrow the lists they replace. Memory is 4.83 GB, 8% more than D2.
2. **Oversampling (10%):** the sampling costs more than D3 saves.
3. **One thread, Wikipedia:** D3 is 6% slower than D2 (18.07 vs 17.07 s). The 79 MB of bitmaps don't fit in the 30 MB L3, so the bit writes miss the cache (+0.77 s in the map), and sampling (0.13 s) and decoding (0.10 s) add more. From 8 threads on, each thread writes only a slice of each bitmap and the penalty disappears.
4. **Static schedule:** concatenation is already cheap (0.08 s), so D3 has nothing to remove: within ±2% of D2.

### 12.9 Phoenix++ vs ours (E6, 24 threads)

| Corpus | Replication | D2 | D3 | Phoenix++ | Time after the map, D3 / Phoenix++ | Peak memory, D3 / Phoenix++ |
|---|---|---|---|---|---|---|
| Zipf | 17.1× | 3.44 s | 3.45 s | **2.94 s** | 1.47 / 0.75 s | 3.8 / 4.7 GB |
| Gutenberg | 3.8× | 1.17 s | **1.08 s** | 1.15 s | 0.21 / 0.22 s | 2.2 / 2.5 GB |
| Wikipedia | 3.2× | 2.33 s | **2.26 s** | 2.68 s | 0.81 / 1.12 s | 4.2 / 5.6 GB |

**The map phases are about equal**, since both use per-thread hash tables. The difference is what happens after:
- **Phoenix++** hash-partitions words into reduce buckets, so each word is combined **once**, whatever the replication. On the replication-heavy synthetic corpus it does half our post-map work and takes 15% less time overall.
- **Ours** sorts every thread's copy, so the work grows with replication. On real text (3–4× replication) the sorted range merge is cheaper: 16% faster on Wikipedia, with 24% less memory. Phoenix++ always uses 17–31% more memory, because per-thread tables, partitioned copies and reduced lists all exist at once.
- **Phoenix++'s final merge tree** ends on one thread: its merge phase on Wikipedia goes 1.01 s → 0.31 s (8 threads) → 0.39 s (24), with no gain after 8. Our two merge passes keep shrinking: 0.42 s → 0.30 s from 8 to 24 threads.
- **A caveat in Phoenix++'s favour:** it never frees its combiner state, while our time includes up to 0.24 s of freeing.

---

## 13. Connections to the lectures (theory → our project)

| Lecture | Concept | In our project |
|---|---|---|
| **Decomposition Techniques** | input, output and intermediate data decomposition; owner computes | The map splits the **input** (documents), and each thread's dictionary is **intermediate data**. The merge splits the **output** (the word space) into ranges, and each task owns and writes one range. The lecture's frequent-itemset counting example is the same problem (documents = transactions, words = items), with the same trade-off: splitting the input replicates the output in every task. |
| **Decomposition & Concurrency** | granularity, degree of concurrency, critical path | 16 chunks per thread and 4 ranges per thread are granularity choices: more tasks balance better but cost more scheduling and merging. Phoenix++'s merge tree has a long critical path (the last merge). |
| **Process Mapping & Load Balancing** | static vs dynamic mapping; chunking; work stealing | Static blocks fail on heterogeneous cores (imbalance up to 1.19). The lecture prescribes dynamic mapping "when the system is heterogeneous", and chunks reduce scheduling overhead. Phoenix++ uses per-thread task queues with stealing. |
| **Brent's Theorem & Parallel Merge Sort** | T_p ≤ W/p + D; the naive merge sort's sequential final merge; parallel merge by pivot + binary search + rank | Phoenix++'s pairwise merge tree is the naive version (last round on one thread, no gain after 8 threads). Our splitter merge is the parallel merge generalized to T lists: sampled pivots, binary-search cuts, prefix-sum ranks, disjoint writes. Its depth holds one range (≈ n/4T), not all n. |
| **PRAM Algorithms** | EREW / CREW / CRCW; prefix sums (scan) | Prefix sums give document numbers (scan), range offsets (merge) and segment offsets (D3). Common-CRCW concurrent bit writes don't exist in hardware, which is why D3 needs atomics at chunk boundaries. |
| **Analytical Modeling** | speedup (true vs relative), efficiency, T_o = T_comm + T_idle + T_excess, superlinear speedup from caches | We report true and relative speedup, compute T_o and E for 8 threads (§12.3), identify replication as excess computation, and see cache effects (a small vocabulary fits in cache: 11.6×). |
| **Multi-Core Architectures** | SMT/hyperthreading, shared caches, coherence (MESI), false sharing, affinity, frequency limits | Pinning order (P-cores, then E-cores, then HT); hyperthreads add only part of a core; the clock drop under load; atomic boundary words and no false sharing inside chunks; the 1-thread D3 case missing the L3. |
| **OpenMP** | `parallel for`, `schedule(static / dynamic / guided)`, `atomic`, barriers, `nowait`, `OMP_PLACES` / `OMP_PROC_BIND` | All of these appear in `build_omp.cpp`; the harness sets the places and binding. |

---

## 14. Limitations and what we would do next

**Limitations:**
- **One machine, warm page cache:** we did not measure cold disk reads.
- **The synthetic generator overstates replication** (17× vs 3–4× on real text), because it ignores burstiness. Synthetic results are pessimistic for D2.
- **ThreadSanitizer not run** (§11).
- **D3's gains are small:** the map phase dominates, and frequent words are already cache-friendly lookups.

**Next steps:**
- **Speed up the local sort,** the biggest phase after the map, by storing each word's first 8 bytes inline as an integer, so most comparisons don't follow a pointer.
- **A hybrid** that hash-partitions *light* words at map time (Phoenix++'s strength against replication) and keeps D3's bitmaps for heavy words.
- **A GPU version** (CUDA): a sort-based design rather than hash tables. We deferred it as optional.

---

## 15. Likely viva questions, with model answers

### Basics

**1. What is an inverted index and why "inverted"?**
A map from each word to the sorted list of documents containing it. "Inverted" because documents naturally map to their words; we flip that to word → documents, which is what a search needs.

**2. Why must posting lists be sorted?**
Queries intersect or union lists. With sorted lists that is a linear merge, and a canonical sorted output also lets us check correctness by byte comparison.

**3. What does your program take as input and produce?**
Pretokenized text, one document per line (the line number is the ID). It produces a binary file with the words in byte order, and for each word a strictly increasing list of 32-bit document IDs.

**4. Why is tokenization outside the build?**
The catalogue scopes the project to pretokenized input and wants preprocessing reported separately. We tokenize once offline (13–54 s in Python) and time it separately.

**5. What is the most expensive part of building an index, and why?**
The map phase: one hash-table lookup per token at an unpredictable address, so it is bound by memory latency (35–100 ns per token, 88–97% of sequential time).

### Design

**6. How does D1 deduplicate within a document for free?**
It processes documents in increasing order, so a repeated word in the same document finds its own document ID at the end of its list. One comparison with the last element is enough.

**7. Why open addressing, and why a load factor of 0.5?**
Open addressing keeps everything in one array (no pointer chasing per bucket). A load factor ≤ 0.5 keeps linear-probe chains short. Storing a 32-bit tag in each 8-byte slot means most probes don't touch the full entry.

**8. Why don't threads share one global hash table?**
Every insert would need locks or atomics on a structure that all 24 threads hit hundreds of millions of times, with heavy contention on frequent words and cache lines bouncing between cores. Private tables need no synchronization during the map. The cost moves to the merge.

**9. Explain static vs dynamic scheduling in your map.**
- **Static:** one contiguous byte-balanced block per thread. Merging is concatenation, but E-cores finish late.
- **Dynamic:** 16 chunks per thread handed out at run time, so fast cores take more. It balances well, but a word's lists from different threads interleave and need a k-way merge.

**10. Why `monotonic:dynamic`?**
It guarantees each thread receives its chunks in increasing order, so each thread's own lists stay sorted, which the k-way merge requires.

**11. How do you split the merge across threads?**
1. Regular sampling of the sorted runs picks R − 1 = 4T − 1 splitter words.
2. Each run is cut at those words by binary search (`lower_bound`), so every word falls in exactly one range.
3. Ranges are merged in parallel with a dynamic schedule.

**12. Why 4T ranges rather than T?**
Oversplitting: ranges differ in work (frequent words make some ranges heavy). With more ranges than threads, a dynamic schedule lets fast threads take more of them.

**13. Why two passes over the ranges?**
To write into one shared output array without locks: the first pass counts each range's words and postings, prefix sums turn the counts into exact offsets, and the second pass writes each range into its own disjoint region. No reallocation, and the output is deterministic.

**14. What is replication and why does it matter?**
The number of entries in all threads' dictionaries divided by the number of distinct words: 17.1× on Zipf, 3.2–3.8× on real text at 24 threads. Every replicated entry is sorted and merged, which is extra work the sequential builder never does. It is the main reason the phases after the map scale poorly.

**15. What exactly is D3?**
Words found (by a 1% sample) in at least 1/32 of the documents become N-bit shared bitmaps, up to a memory budget. Their tokens set bits instead of appending to lists, so they need no list growth and no merge. They are decoded at the end with popcount, prefix sums and count-trailing-zeros.

**16. Why θ = 1/32?**
A bitmap costs N/8 bytes, and a list of df 32-bit IDs costs 4·df bytes. They are equal at df = N/32, so above that density the bitmap is no bigger than the list.

**17. Several threads write to one bitmap: how is that safe?**
Each thread's documents are contiguous chunks, so only the first and last 64-bit word of a chunk can be shared with another thread. Those use an atomic OR (relaxed ordering is enough: OR commutes, and a barrier comes before anyone reads). All other words are private and get plain stores.

**18. What happens if the sample misclassifies a word?**
Only speed changes. A truly heavy word missed by the sample takes the normal list path; a light word wrongly marked heavy gets a bitmap. The output is identical either way, and the tests include all-heavy, none-heavy and budget-capped variants to prove it.

**19. What is the memory budget for?**
To cap bitmap memory (H words × N/8 bytes). Too many bitmaps on many documents can exceed the memory of the lists they replace. The budget keeps the most frequent words.

**20. Why does D3 help Gutenberg the most (8%)?**
Gutenberg books are huge, so most common words are in most books: 27,110 heavy words cover 96% of tokens. D3 skips most posting-vector growth in the map, not just the merge.

### Correctness

**21. How do you know your parallel code is correct?**
The output format is canonical, so correct means byte-identical to the reference. We check that across:
- 13 hand-written cases and 200 random corpora;
- 9 builder variants at 1, 2, 3, 7, 16 and 24 threads;
- a 600,001-document case;
- an independent recount from the text;
- sanitizers and planted bugs;
- the checksum of every benchmark run.

**22. Why odd thread counts like 3 and 7?**
They make block and range boundaries fall at awkward places, which is where off-by-one errors hide.

**23. Is there any data race?**
- **Map:** only thread-private tables.
- **Merge and write:** disjoint precomputed output ranges.
- **D3 bitmaps:** the only shared words are chunk boundaries, written atomically.

ThreadSanitizer wasn't run (no LLVM OpenMP runtime installed); that is a stated limitation.

### Performance and theory

**24. Why isn't the speedup 24× on 24 threads?**
1. The threads aren't 24 equal cores: 8 E-cores are slower, and 8 are hyperthreads sharing P-cores.
2. The clock drops under load (4.75 → 3.94 GHz on 8 P-cores), capping 8 P-cores at about 6.6×.
3. Memory bandwidth and the shared L3 are shared resources.
4. The merge phases scale poorly because of replication.

The map phase itself reaches 10–13×, about what the hardware allows.

**25. What's the difference between your "practical" and "scaling" speedup?**
Practical (true) speedup divides the best sequential time by the parallel time. Scaling (relative) speedup divides the parallel program's 1-thread time by its p-thread time. The lecture warns that relative speedup can hide a weak baseline. Ours are close because the parallel code at 1 thread is within 2% of D1.

**26. Compute the efficiency at 8 threads on Wikipedia.**
E = T_s / (p·T_p) = 16.95 / (8 × 3.43) = 0.62. The overhead T_o = 27.4 − 16.95 = 10.5 core-seconds:
- about 4.2 s from slower cores (clock drop plus memory contention);
- 2.3 s idling in the map;
- 4.0 s in the phases after the map, mostly excess work from replication.

**27. Where is Amdahl's law in your results?**
Here the post-map phases are parallel but scale poorly, rather than fully serial, so they act like Amdahl's slow fraction. If they scaled like the map, Zipf would take 2.0 s instead of 3.4 s. Phoenix++'s merge tree, with its single-threaded last round, is a literal Amdahl bottleneck: its merge stops improving after 8 threads.

**28. How does Brent's theorem apply?**
T_p ≤ W/p + D. For a merge, the tree's depth D contains one merge of all n words, while our splitter merge's depth contains only one range (about n/4T). With the same work W, the splitter merge scales further.

**29. Why do E-cores and hyperthreads still help?**
The map is memory-latency-bound: while one thread waits for memory, another can run. Extra hardware threads hide latency. From 8 to 24 threads we gain 1.35–1.73× more.

**30. Did you see superlinear speedup?**
Not overall. But cache effects push the same way: a 10,000-word vocabulary fits in cache and scales 11.6×, and on Gutenberg the per-thread map work is slightly less than the clock drop predicts (smaller per-thread tables).

**31. Why is Phoenix++ faster on the synthetic corpus but slower on Wikipedia?**
Phoenix++ hash-partitions words, so each word is combined once regardless of replication. We sort and merge every thread's copy. At 17× replication (synthetic) that costs us; at 3× (real text) our sorted merge is cheaper and uses 17–31% less memory than Phoenix++'s structures.

**32. Why is the synthetic corpus's replication so high?**
`gen_zipf` draws every word independently, so a rare word is equally likely in any document and therefore appears in every thread's share. Real rare words are bursty (they cluster in a few documents), so each appears in few threads.

### Methodology

**33. How do you make timings trustworthy?**
- **Isolation:** pinned threads, a fresh process per run after a warm-up, AC power, the performance profile.
- **Repetition:** 5 runs in shuffled order, reporting the median; the spread was 0.5–2.5%.
- **Safety nets:** a checksum guard, and warnings when the load is high or the laptop is on battery.

**34. How did you measure memory?**
Peak resident set size (`getrusage` max RSS) of each run's fresh process. The input buffer (about 1.7 GB) is included in every number.

**35. What counts in "build time"?**
From the document-boundary scan to a finished in-memory canonical index, including freeing temporaries. File loading, writing and tokenization are reported separately.

**36. What's your unfavorable case?**
We have four:
- θ too low (memory exceeds D2);
- 10% sampling (slower than D2);
- D3 on 1 thread on Wikipedia (bitmaps miss the L3, 6% slower);
- the static schedule (nothing to gain).

The uniform vocabulary we predicted would hurt doesn't: D3 finds no heavy words and costs nothing.

### "Why not…?"

**37. Why not `std::unordered_map`?**
It is node-based (an allocation per insert, pointer chasing) and slower for millions of string keys. We wanted one fast table shared by all builders, so comparisons are fair.

**38. Why not sort all (word, document) pairs instead of hashing?**
Sorting 200–330 M pairs costs O(n log n) with heavy memory traffic. Hashing groups in one pass, and only the distinct words get sorted. A sort-based design would suit a GPU better; that is our future work.

**39. Why not compress the posting lists?**
Out of scope for the catalogue. Compression changes the output size, not the parallel construction we were asked to study.

**40. What would you change first?**
Speed up the local sort with inline 8-byte key prefixes. It is the largest phase after the map, and it would help D2 and D3 alike. After that, a hybrid with hash-partitioned light words to remove most replication.

### About the team and process

**41. Who did what?**
Answer honestly with your own split. The plan's suggestion: student A did the baseline, correctness, datasets and evaluation; student B did the parallel builders and profiling; each reviewed the other's code.

**42. Did you use AI tools?**
Answer honestly, as the course policy and your report's attribution section say. Whatever was used, you must be able to explain every part of the code and every number yourselves, which is what this document is for.

---

## 16. Numbers to remember (one page)

| What | Value |
|---|---|
| **Machine** | |
| Cores | 8 P-cores (16 threads) + 8 E-cores = 24 threads, 30 MB L3, 31 GB RAM |
| Clock, 1 busy core → 8 busy P-cores | 4.75 → 3.94 GHz (8 P-cores ≤ about 6.6×) |
| **Corpora** | |
| Documents | Zipf 2 M · Gutenberg 4,976 · Wikipedia 6.4 M |
| Tokens | 200 / 326 / 285 M |
| Distinct words | 1.0 / 0.78 / 2.78 M |
| Sequential build | 20.65 / 11.80 / 16.95 s |
| **Results** | |
| Best at 24 threads (D3 dynamic) | 3.42 / 1.11 / 2.28 s |
| True speedup | **6.0× / 10.6× / 7.4×** |
| Map speedup at 24 threads | 10–13× |
| Share of time after the map | 43% / 23% / 38% |
| Replication at 24 threads | 17.1× / 3.8× / 3.2× |
| Map imbalance, static → dynamic | up to 1.19 → 1.02–1.03 |
| Overhead at 8 threads, Wikipedia | T_o = 10.5 s; E = 0.62 |
| **D3 on Wikipedia** | |
| Heavy words | 104 = 43% of tokens, 79 MB of bitmaps |
| Write pass | 0.28 → 0.18 s, imbalance 1.33 → 1.16 |
| End to end | −3% time, −5% memory (Gutenberg −8% time) |
| **Phoenix++ (24 threads)** | |
| Wikipedia | 2.68 s vs our 2.26 s; 5.6 vs 4.2 GB |
| Zipf | 2.94 s vs our 3.45 s |
| **Tests** | |
| `make test` | 10,462 checks (+2,622 under ASan, +451 Phoenix++) |
| Benchmark runs | 1,795, all checksums identical |

---

## 17. Glossary

**General**
- **Posting:** a (term, document) pair. **Posting list:** a term's sorted document IDs. **df:** a term's number of documents.
- **Canonical output:** the one correct byte sequence for an input, so correctness becomes equality.
- **Oracle:** a deliberately simple reference implementation used to check others.
- **Zipf's law:** frequency ∝ 1/rank^s; s ≈ 1 for natural text, s = 0 is uniform.
- **Burstiness:** rare words cluster in a few documents rather than spreading evenly.

**Our design**
- **Heavy term:** a word with sampled df/N ≥ θ, stored as a bitmap in D3.
- **Replication:** total local dictionary entries / distinct words.
- **Run:** one thread's dictionary after its local sort.
- **Splitter:** a sampled word that marks a range boundary in the merge.
- **Regular sampling:** evenly spaced samples from each sorted run, used to choose splitters.
- **Oversplitting:** more tasks (ranges, chunks) than threads, for balance.
- **Owner computes:** each task writes only the output it owns.
- **Prefix sum (scan):** running totals; turns counts into write offsets.
- **popcount / countr_zero:** CPU instructions counting the set bits / trailing zero bits of a word.

**Performance**
- **Pinning (affinity):** fixing each thread to a CPU.
- **P-core / E-core / SMT (hyperthreading):** performance core / efficiency core / two hardware threads sharing one core.
- **Peak RSS:** the most physical memory a process used at once.
- **True vs relative speedup:** against the best sequential program vs against the same parallel program on 1 thread.
- **Efficiency:** S/p. **Overhead T_o:** p·T_p − T_s.
- **Load imbalance:** slowest thread time / mean thread time.
- **False sharing:** different threads writing different data in the same cache line, causing it to bounce between cores.

**Tools**
- **MapReduce / Phoenix / Phoenix++:** map → combine → reduce → merge; Phoenix is its shared-memory version from the starting paper; Phoenix++ is the authors' rewrite.

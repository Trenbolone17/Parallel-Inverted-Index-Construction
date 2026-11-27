# Presentation script: Parallel inverted index construction

10 slides, about 760 words: 5 minutes at a normal speaking pace (150 words a minute), or 4:45 with the two optional sentences cut.
Albert presents the design (slides 1–5), Johns the results (6–10); swap freely.
Lines in *italics* are cues, not words to say.

| # | Slide | Speaker | Time | Ends at |
|---|---|---|---|---|
| 1 | Cover | Albert | 15 s | 0:15 |
| 2 | The problem | Albert | 25 s | 0:40 |
| 3 | What we built | Albert | 30 s | 1:10 |
| 4 | D2 pipeline | Albert | 40 s | 1:50 |
| 5 | D3 heavy-term bitmaps | Albert | 40 s | 2:30 |
| 6 | Correctness and setup | Johns | 25 s | 2:55 |
| 7 | Strong scaling | Johns | 40 s | 3:35 |
| 8 | D3 results | Johns | 25 s | 4:00 |
| 9 | Phoenix++ baseline | Johns | 30 s | 4:30 |
| 10 | Takeaways | Johns | 20 s | 4:50 |

If you run long, slide 4 and slide 7 are where time goes: drop the sentences marked *(cut if short on time)*.

---

## 1 · Cover · Albert · 15 s

Good morning. I'm Albert, and this is Johns. Our project is building an inverted index in parallel with OpenMP. We started from the Phoenix paper's model, map, local aggregation and reduce, and we'll link each design choice to a lecture as we go.

## 2 · The problem · Albert · 30 s

An inverted index maps every word to the sorted list of documents that contain it. Here, three tiny documents give six lists: "cat" is in documents 0 and 2.

At full scale this is a huge group-by: 200 to 330 million words collapse into 30 to 215 million word–document pairs. Almost every word is a hash-table lookup that misses the cache. That is the work we parallelise.

*Point at the "cat" row of the table.*

## 3 · What we built · Albert · 30 s

We built three builders and ran one baseline. D1 is the best sequential builder, and every speedup is measured against it. D2 is the OpenMP version: each thread builds its own dictionary, and then we merge them by word range. D3 turns the most frequent words into shared bitmaps, under a memory budget. And Phoenix++, the paper's own runtime, runs our index as its application.

All four write the same canonical file, so "correct" simply means byte-identical.

## 4 · D2 pipeline · Albert · 40 s

D2 runs in six steps, all of them parallel. Each thread indexes its documents into a private hash table and sorts it. Sampled splitters cut the words into ranges, and each range is merged on its own; prefix sums give every range its place in the output.

*Point at the three amber boxes.*

Three lecture ideas are in here. Decomposition: we split the input, the documents, and then the output, the words. Parallel merge: every range at once, instead of a merge tree that ends on one thread. *(cut if short on time)* And dynamic mapping: on our hybrid CPU, small chunks let the fast cores take more work, so the slowest thread's lag drops from 19% to 3%.

## 5 · D3 heavy-term bitmaps · Albert · 40 s

D3 targets the most frequent words. A 1% sample finds every word in at least 1 of 32 documents; from there, a bit per document is smaller than a list. We keep as many bitmaps as fit the memory budget.

In the map, a heavy word just sets one bit: nothing to grow, nothing to merge.

*Point at the amber box.*

The PRAM link: in theory, threads can write the same bit at once. Real cores write whole 64-bit words, so neighbours could erase each other's bits. Only a chunk's first and last word are shared, so only those use an atomic OR. On Wikipedia, 104 words are heavy: 43% of everything we read.

*Hand over: "Johns will show what this gives us."*

## 6 · Correctness and setup · Johns · 25 s

Thanks. Correctness first: 10,462 checks pass, from hand-written cases and 200 random corpora at 1 to 24 threads, and all 1,795 benchmark runs agree on the index checksum.

We used three very different inputs: synthetic text, whole Gutenberg books and short Wikipedia paragraphs. Threads are pinned, P-cores first, and every setting runs five times; we report the median.

## 7 · Strong scaling · Johns · 35 s

At 24 threads, our best builder is 10.6 times faster than the best sequential builder on Gutenberg, 7.4 times on Wikipedia and 6 times on synthetic text. That is true speedup in the lecture's sense: against the best serial code.

*(cut if short on time)* The shaded bands are where E-cores and then hyperthreads join; both still add speed.

The map scales 10 to 13 times. What doesn't scale is the work after it. Each thread holds its own copy of much of the vocabulary, 3 to 17 times over, and sorting and merging those copies is work the sequential builder never does: the lecture's excess computation.

## 8 · D3 results · Johns · 30 s

D3 does what it was designed for. On Wikipedia, the write pass drops from 0.28 to 0.18 seconds, and its imbalance from 1.33 to 1.16. But that phase is small, so end to end it is 3% faster with 5% less memory.

We also report where it loses. For example, on one thread its 79 MB of bitmaps don't fit the 30 MB cache, so it is 6% slower.

## 9 · Phoenix++ baseline · Johns · 30 s

Against Phoenix++, the paper's own runtime: on real text ours is faster, by 16% on Wikipedia and 6% on Gutenberg, and uses 17 to 31% less memory. On the synthetic corpus, Phoenix++ wins by 15%.

The reason is replication. Phoenix++ sends each word to one reduce bucket by hash, so it combines each word once; we sort every thread's copy. At 17 times replication that costs us; at 3 to 4 times, on real text, our balanced merge wins.

## 10 · Takeaways · Johns · 20 s

To sum up. Splitting documents, then words, lets the map scale close to what the hardware allows. The limit is replication. Hybrid cores need dynamic mapping. And on real text, our design beats the paper's own runtime.

Thank you. Now we'll show it running live.

---

## Into the demo

*Before the talk:* plugged in, `powerprofilesctl set performance`, `cd inverted-index`, `cat data/wiki.txt > /dev/null`, terminal font large. Then `./demo.sh`; it waits for Enter between steps.

| Step | What to say |
|---|---|
| 1 · Small example | "This is slide 2's example: word, how many documents, which documents." |
| 2 · Tests | "About 300 checks, every builder against a simple reference, byte for byte. The odd thread count, 7, catches boundary bugs." |
| 3 · Full builds | Fill the 17-second sequential build with slide 2: lookups that miss the cache. Then: "Same checksum from every builder." |
| 4 · Same answer | "Identical files, and here is the speedup from slide 7, measured just now." |
| 5 · Lookups | "The finished index answers a lookup instantly." |

For questions, see `viva-prep.md` and the answers doc.

#pragma once
// The per-document indexing loop shared by seq (D1) and omp (D2), so both run the
// identical inner loop and speedups reflect parallelism only.

#include <atomic>
#include <cstdint>
#include <vector>

#include "corpus.hpp"
#include "hash.hpp"
#include "hash_table.hpp"

namespace invidx {

struct Postings {
    uint32_t last = kNoDoc;  // last doc appended; makes in-order deduplication free
    int32_t heavy = -1;      // D3: heavy-term id (>= 0) or -1. Fills padding: struct stays 32 bytes
    std::vector<uint32_t> docs;
};

using LocalTable = StringTable<Postings>;

// Indexes documents [lo, hi) into `table` in increasing doc order; returns tokens seen.
// Every posting list stays sorted and duplicate-free as long as callers add document
// ranges to a table in increasing order.
inline uint64_t add_docs(LocalTable& table, const Corpus& corpus, uint64_t lo, uint64_t hi) {
    uint64_t tokens = 0;
    for (uint64_t d = lo; d < hi; ++d) {
        const uint32_t doc = static_cast<uint32_t>(d);
        for_each_token(corpus.doc(d), [&](const char* p, uint32_t len) {
            ++tokens;
            Postings& pl = table.get_or_insert(p, len, hash_bytes(p, len));
            if (pl.last != doc) {
                pl.last = doc;
                pl.docs.push_back(doc);
            }
        });
    }
    return tokens;
}

// D3 version of add_docs. Heavy terms are pre-inserted with heavy >= 0; their tokens set
// bit d of that term's shared bitmap (bits + heavy * stride_words) instead of appending to
// a list: no dedup check, no vector growth.
// Chunks [lo, hi) may start and end anywhere, so the first and last 64-doc word of a chunk
// can be shared with the neighbouring chunk (another thread): those two words use an
// atomic OR. Every other word lies wholly inside this chunk and gets a plain store.
inline uint64_t add_docs_heavy(LocalTable& table, const Corpus& corpus, uint64_t lo, uint64_t hi,
                               uint64_t* bits, uint64_t stride_words, uint64_t& heavy_tokens) {
    uint64_t tokens = 0, heavy = 0;
    const uint64_t first_word = lo >> 6, last_word = hi > lo ? (hi - 1) >> 6 : first_word;
    for (uint64_t d = lo; d < hi; ++d) {
        const uint32_t doc = static_cast<uint32_t>(d);
        const uint64_t w = d >> 6, bit = 1ull << (d & 63);
        const bool shared = w == first_word || w == last_word;
        for_each_token(corpus.doc(d), [&](const char* p, uint32_t len) {
            ++tokens;
            Postings& pl = table.get_or_insert(p, len, hash_bytes(p, len));
            heavy += pl.heavy >= 0;
            if (pl.last == doc) return;  // repeat within this doc: nothing to do (cheap: entry is hot)
            pl.last = doc;
            if (pl.heavy < 0) {
                pl.docs.push_back(doc);
            } else {  // first occurrence in this doc: one bitmap write, not one per repeat
                uint64_t& word = bits[static_cast<uint64_t>(pl.heavy) * stride_words + w];
                if (shared)
                    std::atomic_ref<uint64_t>(word).fetch_or(bit, std::memory_order_relaxed);
                else
                    word |= bit;
            }
        });
    }
    heavy_tokens += heavy;
    return tokens;
}

}  // namespace invidx

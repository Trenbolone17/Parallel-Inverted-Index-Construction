#pragma once

#include <string>

#include "corpus.hpp"
#include "index_format.hpp"
#include "stats.hpp"

namespace invidx {

// Test oracle: reads the file itself (std::getline + std::istringstream) and uses
// std::map<std::string, std::set<uint32_t>>. Deliberately shares no code with the
// other builders. Small inputs only.
Index build_naive(const std::string& path);

// D1: best sequential builder. Expects corpus.index_docs() to have been called.
Index build_seq(const Corpus& corpus, Stats& st);

struct OmpOptions {
    bool dynamic = false;        // false: one contiguous doc block per thread; true: dynamic chunks
    int ranges_per_thread = 4;   // term ranges in the merge = ranges_per_thread * threads
    int chunks_per_thread = 16;  // dynamic schedule only: doc chunks = chunks_per_thread * threads

    // D3: heavy-term handling (off = plain D2)
    bool heavy = false;
    double theta = 1.0 / 32;     // heavy if sampled df / sampled docs >= theta (1/32: bitmap <= list size)
    double sample = 0.01;        // fraction of documents sampled to find heavy terms
    double heavy_mem_mb = 1024;  // memory budget for heavy bitmaps (H * N/8 bytes); caps H
};

// D2: OpenMP local dictionaries + sorted range merge; with opt.heavy, D3 heavy-term
// bitmaps on top. Expects index_docs() to have run.
Index build_omp(const Corpus& corpus, int threads, const OmpOptions& opt, Stats& st);

}  // namespace invidx

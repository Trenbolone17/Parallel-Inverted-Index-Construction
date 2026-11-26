#pragma once
// Canonical inverted index. Two builds are equal exactly when their files are
// byte-for-byte identical.
//
// File layout (little-endian):
//   char magic[8] = "INVIDX01"
//   u64  n_docs, n_terms, n_postings
//   u64  term_off[n_terms + 1]      offsets into term_bytes; term_off[0] = 0
//   char term_bytes[term_off[n_terms]]
//   u64  post_off[n_terms + 1]      offsets into postings;   post_off[0] = 0
//   u32  postings[n_postings]
//
// Invariants: terms are non-empty, contain no whitespace and are in strictly
// increasing memcmp order; every posting list is non-empty, strictly increasing
// (deduplicated) and contains only IDs < n_docs.

#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>

#include "pod_vector.hpp"

namespace invidx {

// Arrays are pod_vectors: resize() does not zero-fill, so builders must write every
// element they allocate.
struct Index {
    uint64_t n_docs = 0;
    pod_vector<uint64_t> term_off{0};
    pod_vector<char> term_bytes;
    pod_vector<uint64_t> post_off{0};
    pod_vector<uint32_t> postings;

    uint64_t n_terms() const { return term_off.size() - 1; }
    uint64_t n_postings() const { return postings.size(); }
    std::string_view term(uint64_t i) const {
        return {term_bytes.data() + term_off[i], term_off[i + 1] - term_off[i]};
    }
    std::span<const uint32_t> list(uint64_t i) const {
        return {postings.data() + post_off[i], post_off[i + 1] - post_off[i]};
    }
};

void write_index(const Index& ix, const std::string& path);
Index read_index(const std::string& path);

// 64-bit checksum over every array, for quick equality spot checks in logs.
uint64_t checksum(const Index& ix);

// Text dump: "#docs\t<n>" then one line per term: "term\tdf\tdoc doc ...".
void dump_index(const Index& ix, std::FILE* out);

// Empty string if identical; otherwise a description of the first difference.
std::string compare_index(const Index& a, const Index& b);

// Empty string if all invariants hold; otherwise the first violation.
std::string validate_index(const Index& ix);

}  // namespace invidx

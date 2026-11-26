#pragma once
// Hash function shared by every builder, so speedups compare parallel strategies,
// not hash functions.

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace invidx {

// MurmurHash3 64-bit finalizer.
inline uint64_t fmix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

// Word-at-a-time hash for short byte strings (tokens are typically < 16 bytes).
// The length is folded into the seed, so the zero-padded tail cannot collide with
// a shorter string.
inline uint64_t hash_bytes(const char* p, size_t n) {
    uint64_t h = 0x9E3779B97F4A7C15ULL ^ (static_cast<uint64_t>(n) * 0xbf58476d1ce4e5b9ULL);
    while (n >= 8) {
        uint64_t w;
        std::memcpy(&w, p, 8);
        h = (h ^ w) * 0x94d049bb133111ebULL;
        h ^= h >> 29;
        p += 8;
        n -= 8;
    }
    if (n) {
        uint64_t w = 0;
        std::memcpy(&w, p, n);
        h = (h ^ w) * 0x94d049bb133111ebULL;
        h ^= h >> 29;
    }
    return fmix64(h);
}

}  // namespace invidx

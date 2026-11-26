#pragma once
// Open-addressing (linear probing) map from byte strings to V, shared by all builders.
//
// Keys are NOT copied: they point into the corpus buffer, which must outlive the table.
// Entries live in a dense vector in insertion order; each slot holds a 32-bit hash tag
// plus (entry index + 1), so a probe touches 8 bytes until the tag matches.

#include <cstdint>
#include <cstring>
#include <string_view>
#include <utility>
#include <vector>

namespace invidx {

template <class V>
class StringTable {
public:
    struct Entry {
        const char* key;
        uint32_t len;
        uint64_t hash;
        V val;
        std::string_view key_view() const { return {key, len}; }
    };

    explicit StringTable(size_t initial_slots = 1024) {
        size_t cap = 16;
        while (cap < initial_slots) cap <<= 1;
        slots_.assign(cap, Slot{0, 0});
        mask_ = cap - 1;
    }

    // Returns the value for `key`, inserting a default-constructed V if absent.
    V& get_or_insert(const char* key, uint32_t len, uint64_t hash) {
        if ((entries_.size() + 1) * 2 > slots_.size()) grow();  // load factor <= 0.5
        const uint32_t tag = static_cast<uint32_t>(hash >> 32);
        size_t i = hash & mask_;
        for (;; i = (i + 1) & mask_) {
            ++probes_;
            const Slot s = slots_[i];
            if (s.idx == 0) break;
            if (s.tag == tag) {
                Entry& e = entries_[s.idx - 1];
                if (e.len == len && std::memcmp(e.key, key, len) == 0) return e.val;
            }
        }
        entries_.push_back(Entry{key, len, hash, V{}});
        slots_[i] = Slot{tag, static_cast<uint32_t>(entries_.size())};
        return entries_.back().val;
    }

    // Returns nullptr if `key` is absent.
    V* find(const char* key, uint32_t len, uint64_t hash) {
        const uint32_t tag = static_cast<uint32_t>(hash >> 32);
        for (size_t i = hash & mask_;; i = (i + 1) & mask_) {
            const Slot s = slots_[i];
            if (s.idx == 0) return nullptr;
            if (s.tag == tag) {
                Entry& e = entries_[s.idx - 1];
                if (e.len == len && std::memcmp(e.key, key, len) == 0) return &e.val;
            }
        }
    }

    size_t size() const { return entries_.size(); }
    std::vector<Entry>& entries() { return entries_; }
    const std::vector<Entry>& entries() const { return entries_; }
    uint64_t probes() const { return probes_; }

private:
    struct Slot {
        uint32_t tag;
        uint32_t idx;  // entry index + 1; 0 = empty
    };

    void grow() {
        std::vector<Slot> bigger(slots_.size() * 2, Slot{0, 0});
        const size_t mask = bigger.size() - 1;
        for (size_t k = 0; k < entries_.size(); ++k) {
            const uint64_t h = entries_[k].hash;
            size_t i = h & mask;
            while (bigger[i].idx != 0) i = (i + 1) & mask;
            bigger[i] = Slot{static_cast<uint32_t>(h >> 32), static_cast<uint32_t>(k + 1)};
        }
        slots_ = std::move(bigger);
        mask_ = mask;
    }

    std::vector<Slot> slots_;
    std::vector<Entry> entries_;
    size_t mask_ = 0;
    uint64_t probes_ = 0;
};

}  // namespace invidx

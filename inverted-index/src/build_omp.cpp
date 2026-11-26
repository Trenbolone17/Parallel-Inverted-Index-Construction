// D2: OpenMP local dictionaries with sorted merging (Phoenix-style map / local
// aggregation / reduce), and D3: heavy-term bitmaps on top (opt.heavy).
//
//   map        Each thread indexes its documents into a private hash table.
//              static : thread t gets one contiguous, byte-balanced block of documents.
//                       Every doc ID in block t is smaller than every ID in block t+1,
//                       so a term's lists from different threads are merged by plain
//                       concatenation in thread order.
//              dynamic: many byte-balanced chunks under schedule(monotonic:dynamic).
//                       Load-balances P-cores against E-cores, but lists from different
//                       threads interleave and need a k-way merge.
//   local_sort Each thread sorts its own distinct terms (a sorted "run").
//   splitters  Sample the runs (weighted by run size) to cut the term space into
//              R = ranges_per_thread * T ranges; binary-search each range in each run.
//   merge      Count pass, per range (dynamic schedule): T-way heap merge of the runs by
//              term, recording the merged order and the range's term/byte/posting totals.
//   layout     Prefix sums give every range its output offsets; the write pass replays
//              the merged order and writes terms and posting lists in place.
//   free       Each thread frees its own table (malloc arenas are per thread).
//
// D3 (heavy terms) adds:
//   sample       Count document frequency on an evenly spaced sample of documents, in
//                parallel. Terms with sampled df >= theta * sampled docs are heavy, capped
//                at the number whose N-bit bitmaps fit heavy_mem_mb (most frequent kept).
//                Classification affects speed only: a misclassified term takes the light path.
//   heavy_setup  Allocate and zero one N-bit bitmap per heavy term.
//   map          Heavy terms are pre-inserted in every table with their heavy id; their
//                tokens set a bit instead of appending to a list. Only the first and last
//                bitmap word of a doc block/chunk can be shared with a neighbouring chunk;
//                those use an atomic OR, all other words a plain store (add_docs_heavy).
//                (Aligning chunks to 64-doc words instead was tried and rejected: with few,
//                long documents, as in Gutenberg, it collapses chunks and idles threads.)
//   heavy_count  popcount per (heavy term, segment): each term's df and segment offsets.
//   merge        Light terms only, exactly as in D2. Heavy terms need no merge.
//   layout       The write pass slots heavy terms into term order and reserves their
//                posting ranges; heavy_decode expands the bitmaps into those ranges as
//                balanced (term x segment) tasks.

#include <omp.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <numeric>
#include <span>
#include <stdexcept>

#include "builders.hpp"
#include "local_index.hpp"

namespace invidx {
namespace {

// Merged-order record: bit 63 = first entry of a new term, bits 32..39 = thread,
// bits 0..31 = entry index in that thread's table.
constexpr uint64_t kGroupStart = 1ull << 63;
inline uint64_t pack(uint32_t t, uint32_t idx, bool start) {
    return (start ? kGroupStart : 0) | (static_cast<uint64_t>(t) << 32) | idx;
}
inline uint32_t unpack_thread(uint64_t x) { return static_cast<uint32_t>(x >> 32) & 0xFF; }
inline uint32_t unpack_index(uint64_t x) { return static_cast<uint32_t>(x); }

constexpr uint64_t kSegWords = 4096;  // heavy_count / heavy_decode task = 4096 words = 262,144 docs

// Splits documents into `parts` contiguous ranges of roughly equal bytes:
// part p = documents [b[p], b[p+1]).
std::vector<uint64_t> byte_balanced_bounds(const Corpus& c, uint64_t parts) {
    const auto& ds = c.doc_starts();
    std::vector<uint64_t> b(parts + 1);
    for (uint64_t p = 0; p < parts; ++p) {
        const uint64_t target = static_cast<uint64_t>(static_cast<unsigned __int128>(c.size()) * p / parts);
        b[p] = static_cast<uint64_t>(std::lower_bound(ds.begin(), ds.end() - 1, target) - ds.begin());
    }
    b[parts] = c.n_docs();
    return b;
}

// k-way merge of sorted, pairwise-disjoint doc lists into out; returns elements written.
uint64_t merge_lists(const std::vector<std::span<const uint32_t>>& lists, uint32_t* out) {
    if (lists.size() == 1) {
        std::memcpy(out, lists[0].data(), lists[0].size() * sizeof(uint32_t));
        return lists[0].size();
    }
    struct Head {
        uint32_t value;
        uint32_t list;
        size_t pos;
    };
    auto after = [](const Head& a, const Head& b) { return a.value > b.value; };
    std::vector<Head> heap;
    heap.reserve(lists.size());
    for (uint32_t l = 0; l < lists.size(); ++l)
        if (!lists[l].empty()) heap.push_back({lists[l][0], l, 0});
    std::make_heap(heap.begin(), heap.end(), after);
    uint64_t n = 0;
    while (!heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), after);
        Head& h = heap.back();
        out[n++] = h.value;
        if (++h.pos < lists[h.list].size()) {
            h.value = lists[h.list][h.pos];
            std::push_heap(heap.begin(), heap.end(), after);
        } else {
            heap.pop_back();
        }
    }
    return n;
}

// ---- D3: heavy-term detection ---------------------------------------------------------

struct SampleCount {
    uint32_t last = kNoDoc;
    uint32_t count = 0;  // sampled documents containing the term
};

// Returns the heavy terms in increasing term order. Each thread counts df on its share of
// an evenly spaced sample of m documents. A term with total count >= theta*m must reach
// theta*m_t in at least one thread (pigeonhole), so only those per-thread candidates are
// collected; their exact sample counts are then summed over all threads' tables.
std::vector<std::string_view> detect_heavy(const Corpus& corpus, int T, double theta, double sample,
                                           uint64_t max_heavy, uint64_t& sample_docs) {
    const uint64_t n = corpus.n_docs();
    sample_docs = 0;
    if (n == 0 || sample <= 0 || max_heavy == 0) return {};
    const uint64_t m = std::clamp<uint64_t>(static_cast<uint64_t>(std::llround(sample * n)), 1, n);
    sample_docs = m;

    std::vector<std::unique_ptr<StringTable<SampleCount>>> tables(T);
    std::vector<std::vector<std::string_view>> cand(T);
#pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        const uint64_t lo = m * t / T, hi = m * (t + 1) / T;
        auto table = std::make_unique<StringTable<SampleCount>>(1 << 12);
        for (uint64_t k = lo; k < hi; ++k) {
            const uint64_t d = static_cast<uint64_t>(static_cast<unsigned __int128>(2 * k + 1) * n / (2 * m));
            const uint32_t sk = static_cast<uint32_t>(k);
            for_each_token(corpus.doc(d), [&](const char* p, uint32_t len) {
                SampleCount& c = table->get_or_insert(p, len, hash_bytes(p, len));
                if (c.last != sk) {
                    c.last = sk;
                    ++c.count;
                }
            });
        }
        const double need = theta * static_cast<double>(hi - lo);
        if (hi > lo)
            for (const auto& e : table->entries())
                if (e.val.count >= need) cand[t].push_back(e.key_view());
        tables[t] = std::move(table);
    }

    std::vector<std::string_view> all;
    for (const auto& c : cand) all.insert(all.end(), c.begin(), c.end());
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());

    std::vector<uint64_t> count(all.size());
#pragma omp parallel for num_threads(T) schedule(dynamic, 64)
    for (int64_t i = 0; i < static_cast<int64_t>(all.size()); ++i) {
        const std::string_view k = all[i];
        const uint64_t h = hash_bytes(k.data(), k.size());
        uint64_t c = 0;
        for (const auto& table : tables)
            if (const SampleCount* v = table->find(k.data(), static_cast<uint32_t>(k.size()), h)) c += v->count;
        count[i] = c;
    }

    std::vector<uint32_t> keep;
    for (uint32_t i = 0; i < all.size(); ++i)
        if (static_cast<double>(count[i]) >= theta * static_cast<double>(m)) keep.push_back(i);
    if (keep.size() > max_heavy) {  // memory budget: keep the most frequent
        std::sort(keep.begin(), keep.end(), [&](uint32_t a, uint32_t b) {
            return count[a] != count[b] ? count[a] > count[b] : all[a] < all[b];
        });
        keep.resize(max_heavy);
    }
    std::vector<std::string_view> heavy;
    heavy.reserve(keep.size());
    for (uint32_t i : keep) heavy.push_back(all[i]);
    std::sort(heavy.begin(), heavy.end());
    return heavy;
}

struct AlignedFree {
    void operator()(uint64_t* p) const { std::free(p); }
};

}  // namespace

Index build_omp(const Corpus& corpus, int threads, const OmpOptions& opt, Stats& st) {
    const int T = threads;
    if (T < 1 || T > 255) throw std::runtime_error("omp builder supports 1..255 threads");
    if (opt.ranges_per_thread < 1 || opt.chunks_per_thread < 1)
        throw std::runtime_error("ranges/chunks per thread must be >= 1");
    const bool dynamic = opt.dynamic;
    const uint64_t n_docs = corpus.n_docs();

    omp_set_dynamic(0);
    int team = 0;
#pragma omp parallel num_threads(T)
    {
#pragma omp single
        team = omp_get_num_threads();
    }
    if (team != T)
        throw std::runtime_error("requested " + std::to_string(T) + " threads, OpenMP gave " + std::to_string(team));

    std::vector<std::unique_ptr<LocalTable>> tables(T);
    std::vector<std::vector<uint32_t>> runs(T);  // per-thread entry indices sorted by term (light only)
    std::vector<uint64_t> tokens(T), heavy_tokens(T);
    std::vector<double> t_map(T), t_sort(T), t_merge(T), t_layout(T), t_decode(T);
    double t0;

    // ---- D3: find heavy terms, allocate their bitmaps ----------------------------------
    const uint64_t words = (n_docs + 63) / 64;
    const uint64_t stride = (words + 7) / 8 * 8;  // whole cache lines per bitmap
    std::vector<std::string_view> heavy;
    std::unique_ptr<uint64_t[], AlignedFree> bits;
    if (opt.heavy) {
        t0 = now();
        const uint64_t max_heavy =
            stride == 0 ? 0 : static_cast<uint64_t>(opt.heavy_mem_mb * 1048576.0 / (stride * 8.0));
        uint64_t sample_docs = 0;
        heavy = detect_heavy(corpus, T, opt.theta, opt.sample, std::min<uint64_t>(max_heavy, INT32_MAX), sample_docs);
        st.counter("sample_docs", sample_docs);
        st.phase("sample", now() - t0);

        t0 = now();
        if (!heavy.empty()) {
            const uint64_t total = heavy.size() * stride;
            bits.reset(static_cast<uint64_t*>(std::aligned_alloc(64, total * sizeof(uint64_t))));
            if (!bits) throw std::bad_alloc();
#pragma omp parallel for num_threads(T) schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(total); ++i) bits[i] = 0;
        }
        st.phase("heavy_setup", now() - t0);
    }
    const uint64_t H = heavy.size();

    // ---- map ------------------------------------------------------------------
    t0 = now();
    const uint64_t n_parts = dynamic ? static_cast<uint64_t>(T) * opt.chunks_per_thread : T;
    const std::vector<uint64_t> bounds = byte_balanced_bounds(corpus, n_parts);
#pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        const double ts = now();
        auto table = std::make_unique<LocalTable>(1 << 16);
        for (uint64_t h = 0; h < H; ++h)
            table->get_or_insert(heavy[h].data(), static_cast<uint32_t>(heavy[h].size()),
                                 hash_bytes(heavy[h].data(), heavy[h].size()))
                .heavy = static_cast<int32_t>(h);
        uint64_t tok = 0, ht = 0;
        auto index_range = [&](uint64_t lo, uint64_t hi) {
            return H ? add_docs_heavy(*table, corpus, lo, hi, bits.get(), stride, ht)
                     : add_docs(*table, corpus, lo, hi);
        };
        if (!dynamic) {
            tok = index_range(bounds[t], bounds[t + 1]);
        } else {
            // monotonic: each thread receives chunks in increasing order, so its lists stay sorted
#pragma omp for schedule(monotonic : dynamic, 1) nowait
            for (int64_t c = 0; c < static_cast<int64_t>(n_parts); ++c) tok += index_range(bounds[c], bounds[c + 1]);
        }
        tables[t] = std::move(table);
        tokens[t] = tok;
        heavy_tokens[t] = ht;
        t_map[t] = now() - ts;
    }
    st.phase("map", now() - t0);

    // ---- D3: heavy document frequencies and per-segment offsets --------------------
    const uint64_t S = H ? (words + kSegWords - 1) / kSegWords : 0;
    std::vector<uint64_t> seg_off(H * S), df(H);
    if (H) {
        t0 = now();
#pragma omp parallel for num_threads(T) schedule(dynamic, 1)
        for (int64_t task = 0; task < static_cast<int64_t>(H * S); ++task) {
            const uint64_t h = task / S, s = task % S;
            const uint64_t* w = bits.get() + h * stride;
            uint64_t c = 0;
            for (uint64_t i = s * kSegWords, e = std::min(words, (s + 1) * kSegWords); i < e; ++i) c += std::popcount(w[i]);
            seg_off[task] = c;
        }
        for (uint64_t h = 0; h < H; ++h) {  // counts -> exclusive offsets within each term
            uint64_t run = 0;
            for (uint64_t s = 0; s < S; ++s) {
                const uint64_t c = seg_off[h * S + s];
                seg_off[h * S + s] = run;
                run += c;
            }
            df[h] = run;
        }
        st.phase("heavy_count", now() - t0);
    }

    // ---- local sort (light terms only) ------------------------------------------
    t0 = now();
#pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        const double ts = now();
        const auto& E = tables[t]->entries();
        auto& run = runs[t];
        if (H) {
            run.reserve(E.size());
            for (uint32_t i = 0; i < E.size(); ++i)
                if (E[i].val.heavy < 0) run.push_back(i);
        } else {
            run.resize(E.size());
            std::iota(run.begin(), run.end(), 0u);
        }
        std::sort(run.begin(), run.end(), [&](uint32_t a, uint32_t b) { return E[a].key_view() < E[b].key_view(); });
        t_sort[t] = now() - ts;
    }
    st.phase("local_sort", now() - t0);

    // ---- splitters and range cuts -----------------------------------------------
    t0 = now();
    auto key = [&](int t, uint64_t pos) { return tables[t]->entries()[runs[t][pos]].key_view(); };
    const uint64_t R = static_cast<uint64_t>(T) * opt.ranges_per_thread;
    uint64_t local_entries = 0;
    for (const auto& run : runs) local_entries += run.size();

    std::vector<std::string_view> samples;
    if (local_entries > 0) {
        const uint64_t S_ = std::min<uint64_t>(local_entries, 64 * R);
        for (int t = 0; t < T; ++t) {
            const uint64_t n = runs[t].size();
            const uint64_t s_t = (S_ * n + local_entries - 1) / local_entries;  // proportional to run size
            for (uint64_t j = 0; j < s_t; ++j) samples.push_back(key(t, (2 * j + 1) * n / (2 * s_t)));
        }
        std::sort(samples.begin(), samples.end());
    }
    std::vector<std::string_view> splitters(R > 0 ? R - 1 : 0);
    for (uint64_t r = 1; r < R; ++r) splitters[r - 1] = samples.empty() ? std::string_view{} : samples[r * samples.size() / R];

    // cut[r * T + t] = first position in run t whose term is >= the start of range r
    std::vector<uint64_t> cut((R + 1) * T);
#pragma omp parallel for num_threads(T) schedule(static)
    for (int t = 0; t < T; ++t) {
        const auto& E = tables[t]->entries();
        const auto& run = runs[t];
        cut[t] = 0;
        cut[R * T + t] = run.size();
        for (uint64_t r = 1; r < R; ++r)
            cut[r * T + t] = static_cast<uint64_t>(
                std::lower_bound(run.begin(), run.end(), splitters[r - 1],
                                 [&](uint32_t idx, std::string_view k) { return E[idx].key_view() < k; }) -
                run.begin());
    }
    // heavy terms [hcut[r], hcut[r+1]) fall in range r (same lower_bound rule)
    std::vector<uint64_t> hcut(R + 1);
    hcut[R] = H;
    for (uint64_t r = 1; r < R; ++r)
        hcut[r] = static_cast<uint64_t>(std::lower_bound(heavy.begin(), heavy.end(), splitters[r - 1]) - heavy.begin());
    st.phase("splitters", now() - t0);

    // ---- merge: count pass (light terms) --------------------------------------------
    t0 = now();
    struct Range {
        std::vector<uint64_t> order;  // merged order, see pack()
        uint64_t terms = 0, bytes = 0, postings = 0;
    };
    std::vector<Range> ranges(R);
#pragma omp parallel num_threads(T)
    {
        const double ts = now();
        struct Head {
            std::string_view key;
            uint32_t t;
            uint64_t pos, end;
        };
        // min-heap on (term, thread): equal terms leave in thread order
        auto after = [](const Head& a, const Head& b) {
            const int c = a.key.compare(b.key);
            return c > 0 || (c == 0 && a.t > b.t);
        };
        std::vector<Head> heap;
        heap.reserve(T);
#pragma omp for schedule(dynamic, 1) nowait
        for (int64_t r = 0; r < static_cast<int64_t>(R); ++r) {
            Range& out = ranges[r];
            heap.clear();
            uint64_t expected = 0;
            for (int t = 0; t < T; ++t) {
                const uint64_t lo = cut[r * T + t], hi = cut[(r + 1) * T + t];
                if (lo < hi) heap.push_back({key(t, lo), static_cast<uint32_t>(t), lo, hi});
                expected += hi - lo;
            }
            out.order.reserve(expected);
            std::make_heap(heap.begin(), heap.end(), after);
            std::string_view last;
            bool have_last = false;
            while (!heap.empty()) {
                std::pop_heap(heap.begin(), heap.end(), after);
                Head& h = heap.back();
                const uint32_t idx = runs[h.t][h.pos];
                const bool start = !have_last || h.key != last;
                if (start) {
                    ++out.terms;
                    out.bytes += h.key.size();
                    last = h.key;
                    have_last = true;
                }
                out.postings += tables[h.t]->entries()[idx].val.docs.size();
                out.order.push_back(pack(h.t, idx, start));
                if (++h.pos < h.end) {
                    h.key = key(h.t, h.pos);
                    std::push_heap(heap.begin(), heap.end(), after);
                } else {
                    heap.pop_back();
                }
            }
        }
        t_merge[omp_get_thread_num()] = now() - ts;
    }
    st.phase("merge", now() - t0);

    // ---- layout: prefix sums + write pass -----------------------------------------
    t0 = now();
    std::vector<uint64_t> base_term(R + 1), base_byte(R + 1), base_post(R + 1);
    for (uint64_t r = 0; r < R; ++r) {
        uint64_t terms = ranges[r].terms, bytes = ranges[r].bytes, postings = ranges[r].postings;
        for (uint64_t h = hcut[r]; h < hcut[r + 1]; ++h)
            if (df[h]) {
                ++terms;
                bytes += heavy[h].size();
                postings += df[h];
            }
        base_term[r + 1] = base_term[r] + terms;
        base_byte[r + 1] = base_byte[r] + bytes;
        base_post[r + 1] = base_post[r] + postings;
    }
    Index ix;
    ix.n_docs = n_docs;
    ix.term_off.resize(base_term[R] + 1);  // not zero-filled: the write pass touches every element
    ix.post_off.resize(base_term[R] + 1);
    ix.term_bytes.resize(base_byte[R]);
    ix.postings.resize(base_post[R]);
    std::vector<uint64_t> heavy_base(H);  // where each heavy term's postings start
    bool inconsistent = false;

#pragma omp parallel num_threads(T)
    {
        const double ts = now();
        std::vector<std::span<const uint32_t>> group;
#pragma omp for schedule(dynamic, 1) nowait
        for (int64_t r = 0; r < static_cast<int64_t>(R); ++r) {
            uint64_t ti = base_term[r], bi = base_byte[r], pi = base_post[r], group_base = pi;
            uint64_t h = hcut[r];
            const uint64_t h_end = hcut[r + 1];
            group.clear();
            auto flush = [&] {  // dynamic schedule: k-way merge the finished light group
                if (dynamic && !group.empty()) {
                    pi = group_base + merge_lists(group, ix.postings.data() + group_base);
                    group.clear();
                }
            };
            auto new_term = [&](std::string_view k) {
                ix.term_off[ti] = bi;
                ix.post_off[ti] = pi;
                ++ti;
                std::memcpy(ix.term_bytes.data() + bi, k.data(), k.size());
                bi += k.size();
            };
            auto heavy_before = [&](const std::string_view* k) {  // emit heavy terms < k (all if null)
                for (; h < h_end && (!k || heavy[h] < *k); ++h)
                    if (df[h]) {
                        new_term(heavy[h]);
                        heavy_base[h] = pi;  // reserved; filled by heavy_decode
                        pi += df[h];
                    }
            };
            for (const uint64_t x : ranges[r].order) {
                const auto& e = tables[unpack_thread(x)]->entries()[unpack_index(x)];
                if (x & kGroupStart) {
                    flush();
                    const std::string_view k = e.key_view();
                    heavy_before(&k);
                    new_term(k);
                    group_base = pi;
                }
                if (dynamic) {
                    group.emplace_back(e.val.docs.data(), e.val.docs.size());
                } else {  // static blocks: thread order is doc order, so concatenate
                    std::memcpy(ix.postings.data() + pi, e.val.docs.data(), e.val.docs.size() * sizeof(uint32_t));
                    pi += e.val.docs.size();
                }
            }
            flush();
            heavy_before(nullptr);
            if (ti != base_term[r + 1] || bi != base_byte[r + 1] || pi != base_post[r + 1]) {
#pragma omp atomic write
                inconsistent = true;
            }
        }
        t_layout[omp_get_thread_num()] = now() - ts;
    }
    ix.term_off[base_term[R]] = base_byte[R];
    ix.post_off[base_term[R]] = base_post[R];
    if (inconsistent) throw std::logic_error("omp builder: write pass disagrees with count pass");
    st.phase("layout", now() - t0);

    // ---- D3: expand heavy bitmaps into their reserved posting ranges ---------------
    if (H) {
        t0 = now();
#pragma omp parallel num_threads(T)
        {
            const double ts = now();
#pragma omp for schedule(dynamic, 1) nowait
            for (int64_t task = 0; task < static_cast<int64_t>(H * S); ++task) {
                const uint64_t h = task / S, s = task % S;
                if (!df[h]) continue;
                uint32_t* out = ix.postings.data() + heavy_base[h] + seg_off[task];
                const uint64_t* w = bits.get() + h * stride;
                for (uint64_t i = s * kSegWords, e = std::min(words, (s + 1) * kSegWords); i < e; ++i)
                    for (uint64_t x = w[i]; x; x &= x - 1) *out++ = static_cast<uint32_t>(i * 64 + std::countr_zero(x));
            }
            t_decode[omp_get_thread_num()] = now() - ts;
        }
        st.phase("heavy_decode", now() - t0);
    }

    // ---- counters, then free --------------------------------------------------------
    uint64_t total_tokens = 0, total_heavy_tokens = 0, probes = 0, heavy_postings = 0;
    for (int t = 0; t < T; ++t) {
        total_tokens += tokens[t];
        total_heavy_tokens += heavy_tokens[t];
        probes += tables[t]->probes();
    }
    for (uint64_t d : df) heavy_postings += d;
    st.counter("tokens", total_tokens);
    st.counter("hash_probes", probes);
    st.counter("local_entries", local_entries);  // / n_terms = vocabulary replication factor
    st.counter("ranges", R);
    st.thread_phase("map", t_map);
    st.thread_phase("local_sort", t_sort);
    st.thread_phase("merge", t_merge);
    st.thread_phase("layout", t_layout);
    if (opt.heavy) {
        st.counter("heavy_terms", H);
        st.counter("heavy_tokens", total_heavy_tokens);
        st.counter("heavy_postings", heavy_postings);
        st.counter("heavy_bitmap_bytes", H * stride * sizeof(uint64_t));
        if (H) st.thread_phase("heavy_decode", t_decode);
    }

    t0 = now();
#pragma omp parallel num_threads(T)
    {
        const int t = omp_get_thread_num();
        tables[t].reset();
        std::vector<uint32_t>().swap(runs[t]);
#pragma omp for schedule(static) nowait
        for (int64_t r = 0; r < static_cast<int64_t>(R); ++r) std::vector<uint64_t>().swap(ranges[r].order);
    }
    bits.reset();
    st.phase("free", now() - t0);
    return ix;
}

}  // namespace invidx

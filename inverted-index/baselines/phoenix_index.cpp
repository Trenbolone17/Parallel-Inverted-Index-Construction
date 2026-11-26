// phoenix_index: the same inverted index, built with the Phoenix++ MapReduce runtime.
//
// Reference point for the starting paper (Ranger et al., HPCA 2007): Phoenix++ (Talbot et
// al., MAPREDUCE 2011) is the authors' successor runtime. Only the application callbacks
// below are ours; scheduling, intermediate storage, partitioning, reduce and merge are
// Phoenix++ code (third_party/phoenix/phoenix++-1.0, `make phoenix` fetches it).
//
//   split   documents grouped into ~1 MB chunks (like Phoenix++ word_count's 1 MB splits)
//   map     per-thread Phoenix++ hash_table: term -> combiner; the combiner is our posting
//           list with last-document dedup (Phoenix++'s extension point for this)
//   reduce  Phoenix++ hash-partitions each thread's table into T buckets; a reduce task
//           unions the threads' lists of a term (tr1::unordered_map, Phoenix++ code) and we
//           sort the document IDs
//   merge   MapReduceSort: each thread's output sorted by term, then a pairwise merge tree
//   layout  sorted (term, list) pairs copied into the canonical index (parallel, OpenMP)
//
// Same command line and --stats JSON as invidx, so bench/run.py runs it as a variant, and
// its checksum guard checks that the index is identical to ours. Threads are pinned to the
// CPUs listed in OMP_PLACES ("{0},{2},..."), the same placement the OpenMP builders get.
#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "corpus.hpp"
#include "hash.hpp"
#include "index_format.hpp"
#include "stats.hpp"

#include "map_reduce.h"  // Phoenix++

using namespace invidx;

namespace {

// Key: a term inside the corpus buffer. Ordered like the canonical index (memcmp, then length).
struct Term {
    const char* p;
    uint32_t n;
    bool operator==(Term const& o) const { return n == o.n && std::memcmp(p, o.p, n) == 0; }
    bool operator<(Term const& o) const {
        const int c = std::memcmp(p, o.p, std::min(n, o.n));
        return c < 0 || (c == 0 && n < o.n);
    }
};
struct TermHash {
    size_t operator()(Term const& t) const { return hash_bytes(t.p, t.n); }
};

// Combiner = one map thread's posting list for a term. Documents are mapped one at a time,
// so a repeated term in a document hits the same combiner consecutively: last-doc dedup is
// exact. Phoenix++ copies combiners freely, so they hold a pointer (like buffer_combiner);
// unlike buffer_combiner, the list is allocated on first use, not for every empty slot.
struct PostingList {
    std::vector<uint32_t> docs;
    uint32_t last = kNoDoc;
};

template <typename V, template <class> class Allocator = std::allocator>
class posting_combiner {
    PostingList* l = nullptr;

public:
    void add(V const& v) {
        if (!l) l = new PostingList;
        const uint32_t d = static_cast<uint32_t>(v);
        if (d != l->last) {
            l->docs.push_back(d);
            l->last = d;
        }
    }
    bool empty() const { return !l || l->docs.empty(); }

    // All threads' lists for one term, handed to reduce.
    class combined {
        std::vector<PostingList const*> items;
        mutable size_t list = 0, pos = 0;

    public:
        void add(posting_combiner const* c) { items.push_back(c->l); }
        int size() const { return static_cast<int>(items.size()); }
        std::vector<PostingList const*> const& lists() const { return items; }
        bool next(V& v) const {
            while (list < items.size() && pos >= items[list]->docs.size()) ++list, pos = 0;
            if (list == items.size()) return false;
            v = items[list]->docs[pos++];
            return true;
        }
        void reset() { list = pos = 0; }
        void clear() { items.clear(), reset(); }
    };
    void combineinto(combined& m) const { m.add(this); }
};

// Reduced lists live in big blocks: [length, doc, doc, ...]; the keyval value points at one.
// Blocks are owned globally so they outlive Phoenix++'s worker threads.
class Arena {
    static constexpr size_t kBlock = size_t(1) << 24;  // uint32 words (64 MB)
    static inline std::mutex mu;
    static inline std::vector<std::unique_ptr<uint32_t[]>> blocks;
    static inline thread_local uint32_t* cur = nullptr;
    static inline thread_local size_t left = 0;

public:
    static uint32_t* alloc(size_t words) {
        if (words > left) {
            const size_t size = std::max(words, kBlock);
            std::unique_ptr<uint32_t[]> b(new uint32_t[size]);  // uninitialized
            cur = b.get();
            left = size;
            std::lock_guard<std::mutex> g(mu);
            blocks.push_back(std::move(b));
        }
        uint32_t* r = cur;
        cur += words;
        left -= words;
        return r;
    }
};

struct Chunk {
    uint32_t lo, hi;  // documents [lo, hi)
};

// Thread i runs on the i-th CPU of OMP_PLACES (single-CPU places), else Phoenix++'s default.
class PlacesPolicy : public sched_policy {
    std::vector<int> cpus;

public:
    PlacesPolicy() {
        const char* s = std::getenv("OMP_PLACES");
        for (const char* p = s; p && *p; ++p)
            if (*p >= '0' && *p <= '9') {
                cpus.push_back(std::atoi(p));
                while (p[1] >= '0' && p[1] <= '9') ++p;
            }
    }
    int thr_to_cpu(int thr) const override {
        return cpus.empty() ? thr % num_cpus : cpus[thr % cpus.size()];
    }
};

using Container = hash_container<Term, uint64_t, posting_combiner, TermHash>;

class IndexMR : public MapReduceSort<IndexMR, Chunk, Term, uint64_t, Container> {
    using Base = MapReduceSort<IndexMR, Chunk, Term, uint64_t, Container>;
    const Corpus& corpus;
    Stats& st;
    std::vector<double> map_t, reduce_t;

protected:
    // Phase timers around Phoenix++'s own phases (qualified calls = Phoenix++ code).
    void run_map(data_type* data, uint64_t count) override {
        const double t = now();
        Base::run_map(data, count);
        st.phase("map", now() - t);
    }
    void run_reduce() override {
        const double t = now();
        Base::run_reduce();
        st.phase("reduce", now() - t);
    }
    void run_merge() override {
        const double t = now();
        Base::run_merge();
        st.phase("merge", now() - t);
    }
    void map_worker(thread_loc const& loc, double& time, double& user, int& tasks) override {
        const double t = now();
        Base::map_worker(loc, time, user, tasks);
        map_t[loc.thread] = now() - t;
    }
    void reduce_worker(thread_loc const& loc, double& time, double& user, int& tasks) override {
        const double t = now();
        Base::reduce_worker(loc, time, user, tasks);
        reduce_t[loc.thread] = now() - t;
    }

public:
    IndexMR(const Corpus& c, int threads, Stats& s)
        : corpus(c), st(s), map_t(threads), reduce_t(threads) {
        PlacesPolicy policy;
        setThreads(threads, &policy);
    }
    void record_thread_phases() {
        st.thread_phase("map", map_t);
        st.thread_phase("reduce", reduce_t);
    }

    void map(Chunk const& c, map_container& out) const {
        for (uint32_t d = c.lo; d < c.hi; ++d)
            for_each_token(corpus.doc(d), [&](const char* p, uint32_t n) {
                emit_intermediate(out, Term{p, n}, d);
            });
    }

    void reduce(Term const& key, reduce_iterator const& values, std::vector<keyval>& out) const {
        size_t n = 0;
        for (const PostingList* l : values.lists()) n += l->docs.size();
        uint32_t* r = Arena::alloc(n + 1);
        r[0] = static_cast<uint32_t>(n);
        uint32_t* w = r + 1;
        for (const PostingList* l : values.lists()) w = std::copy(l->docs.begin(), l->docs.end(), w);
        if (!std::is_sorted(r + 1, w)) std::sort(r + 1, w);  // map tasks interleave per thread
        out.push_back(keyval{key, reinterpret_cast<uint64_t>(r)});
    }
};

Index build_phoenix(const Corpus& corpus, int threads, Stats& st) {
    double t = now();
    std::vector<Chunk> chunks;  // ~1 MB of text each, whole documents
    const auto& starts = corpus.doc_starts();
    const uint64_t n_docs = corpus.n_docs();
    for (uint64_t lo = 0; lo < n_docs;) {
        uint64_t hi = std::upper_bound(starts.begin() + lo + 1, starts.end() - 1, starts[lo] + (1u << 20)) -
                      starts.begin();
        hi = std::max(hi, lo + 1);
        chunks.push_back({static_cast<uint32_t>(lo), static_cast<uint32_t>(hi)});
        lo = hi;
    }
    st.phase("split", now() - t);
    st.counter("map_chunks", chunks.size());

    std::vector<IndexMR::keyval> result;
    {
        t = now();
        IndexMR mr(corpus, threads, st);
        st.phase("init", now() - t);
        mr.run(chunks.data(), chunks.size(), result);
        mr.record_thread_phases();
        t = now();
    }  // Phoenix++ teardown: thread pool, partitioned intermediate tables
    st.phase("teardown", now() - t);

    t = now();
    Index ix;
    ix.n_docs = n_docs;
    const size_t n_terms = result.size();
    ix.term_off.resize(n_terms + 1);
    ix.post_off.resize(n_terms + 1);
    ix.term_off[0] = ix.post_off[0] = 0;
    for (size_t i = 0; i < n_terms; ++i) {
        ix.term_off[i + 1] = ix.term_off[i] + result[i].key.n;
        ix.post_off[i + 1] = ix.post_off[i] + reinterpret_cast<const uint32_t*>(result[i].val)[0];
    }
    ix.term_bytes.resize(ix.term_off[n_terms]);
    ix.postings.resize(ix.post_off[n_terms]);
#pragma omp parallel for num_threads(threads) schedule(static, 4096)
    for (size_t i = 0; i < n_terms; ++i) {
        const auto* list = reinterpret_cast<const uint32_t*>(result[i].val);
        std::memcpy(ix.term_bytes.data() + ix.term_off[i], result[i].key.p, result[i].key.n);
        std::memcpy(ix.postings.data() + ix.post_off[i], list + 1, size_t(list[0]) * 4);
    }
    st.phase("layout", now() - t);
    return ix;
}

}  // namespace

int main(int argc, char** argv) {
    std::string input, out, stats_path;
    int threads = omp_get_max_threads();
    bool quiet = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "phoenix_index: missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--input") input = value();
        else if (a == "--out") out = value();
        else if (a == "--threads") threads = std::atoi(value().c_str());
        else if (a == "--stats") stats_path = value();
        else if (a == "--quiet") quiet = true;
        else if (a == "--algo") value();  // accepted for symmetry with invidx
        else {
            std::fprintf(stderr,
                         "usage: phoenix_index --input CORPUS [--threads N] [--out FILE] [--stats FILE] [--quiet]\n");
            return 2;
        }
    }
    if (input.empty() || threads < 1) {
        std::fprintf(stderr, "phoenix_index: --input required, --threads >= 1\n");
        return 2;
    }
    try {
        Stats st;
        st.threads = threads;
        st.info("algo", "phoenix++");
        st.info("input", input);
        const char* places = std::getenv("OMP_PLACES");
        st.info("omp_places", places ? places : "");
        setenv("MR_NUMTHREADS", std::to_string(threads).c_str(), 1);  // Phoenix++'s first pool size

        double t = now();
        Corpus corpus = Corpus::load(input);
        st.phase("load", now() - t);
        st.counter("input_bytes", corpus.size());
        const double build_start = now();
        corpus.index_docs(threads);
        st.phase("scan", now() - build_start);
        Index ix = build_phoenix(corpus, threads, st);
        st.phase("build_total", now() - build_start);

        st.counter("n_docs", ix.n_docs);
        st.counter("n_terms", ix.n_terms());
        st.counter("n_postings", ix.n_postings());
        char sum[24];
        std::snprintf(sum, sizeof sum, "%016llx", static_cast<unsigned long long>(checksum(ix)));
        st.info("checksum", sum);
        if (!out.empty()) {
            t = now();
            write_index(ix, out);
            st.phase("write", now() - t);
        }
        if (!stats_path.empty()) st.write_json(stats_path);
        if (!quiet) st.print_summary(stderr);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "phoenix_index: %s\n", e.what());
        return 1;
    }
    return 0;
}

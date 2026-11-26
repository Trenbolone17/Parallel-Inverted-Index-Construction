// D1: best sequential builder.
//   map    - scan documents in order; append doc ID to the term's list unless it equals
//            the list's last doc (in-order scan makes deduplication free)
//   sort   - sort distinct terms (memcmp order)
//   layout - copy terms and posting lists into the canonical arrays
//   free   - release the hash table and per-term vectors (a real cost of this design)

#include <algorithm>
#include <cstring>
#include <numeric>

#include "builders.hpp"
#include "local_index.hpp"

namespace invidx {

Index build_seq(const Corpus& corpus, Stats& st) {
    const uint64_t n_docs = corpus.n_docs();

    double t = now();
    LocalTable table(1 << 16);
    const uint64_t tokens = add_docs(table, corpus, 0, n_docs);
    st.phase("map", now() - t);

    t = now();
    const auto& entries = table.entries();
    std::vector<uint32_t> order(entries.size());
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b) { return entries[a].key_view() < entries[b].key_view(); });
    st.phase("sort", now() - t);

    t = now();
    uint64_t total_term_bytes = 0, total_postings = 0;
    for (const auto& e : entries) {
        total_term_bytes += e.len;
        total_postings += e.val.docs.size();
    }
    Index ix;
    ix.n_docs = n_docs;
    ix.term_off.resize(entries.size() + 1);
    ix.post_off.resize(entries.size() + 1);
    ix.term_bytes.resize(total_term_bytes);
    ix.postings.resize(total_postings);
    uint64_t to = 0, po = 0;
    for (size_t i = 0; i < order.size(); ++i) {
        const auto& e = entries[order[i]];
        ix.term_off[i] = to;
        ix.post_off[i] = po;
        std::memcpy(ix.term_bytes.data() + to, e.key, e.len);
        std::memcpy(ix.postings.data() + po, e.val.docs.data(), e.val.docs.size() * sizeof(uint32_t));
        to += e.len;
        po += e.val.docs.size();
    }
    ix.term_off.back() = to;
    ix.post_off.back() = po;
    st.phase("layout", now() - t);

    st.counter("tokens", tokens);
    st.counter("hash_probes", table.probes());

    t = now();
    { LocalTable discard = std::move(table); }
    st.phase("free", now() - t);
    return ix;
}

}  // namespace invidx

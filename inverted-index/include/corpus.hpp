#pragma once
// Input contract (pretokenized corpus):
//   - one document per line; document ID = 0-based line number
//   - an empty line is a valid empty document and still uses an ID
//   - a final line without a trailing '\n' is still a document
//   - tokens are maximal runs of non-whitespace bytes; ASCII whitespace
//     (space, \t, \r, \v, \f) separates tokens; all other bytes are opaque
//   - fewer than 2^32 - 1 documents (UINT32_MAX is reserved as "no document")

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace invidx {

inline constexpr uint64_t kMaxDocs = 0xFFFFFFFEull;
inline constexpr uint32_t kNoDoc = 0xFFFFFFFFu;

struct SepTable {
    bool sep[256];
    constexpr SepTable() : sep{} {
        sep[static_cast<unsigned char>(' ')] = true;
        sep[static_cast<unsigned char>('\t')] = true;
        sep[static_cast<unsigned char>('\n')] = true;
        sep[static_cast<unsigned char>('\r')] = true;
        sep[static_cast<unsigned char>('\v')] = true;
        sep[static_cast<unsigned char>('\f')] = true;
    }
};
inline constexpr SepTable kSep{};

inline bool is_sep(char c) { return kSep.sep[static_cast<unsigned char>(c)]; }

// Calls f(const char* token, uint32_t length) for each token of `doc`, in order.
template <class F>
inline void for_each_token(std::string_view doc, F&& f) {
    const char* p = doc.data();
    const char* const end = p + doc.size();
    while (p < end) {
        while (p < end && is_sep(*p)) ++p;
        const char* const start = p;
        while (p < end && !is_sep(*p)) ++p;
        if (p > start) f(start, static_cast<uint32_t>(p - start));
    }
}

class Corpus {
public:
    // Reads the whole file into memory and guarantees the buffer ends with '\n'.
    static Corpus load(const std::string& path);

    // Finds document boundaries with a parallel newline scan + prefix sum.
    void index_docs(int threads);

    uint64_t n_docs() const { return doc_start_.empty() ? 0 : doc_start_.size() - 1; }
    // Document text without its terminating '\n'.
    std::string_view doc(uint64_t i) const {
        return {buf_.get() + doc_start_[i], doc_start_[i + 1] - doc_start_[i] - 1};
    }
    // doc_starts()[i] = byte offset of document i; doc_starts()[n_docs()] = size().
    const std::vector<uint64_t>& doc_starts() const { return doc_start_; }
    const char* data() const { return buf_.get(); }
    uint64_t size() const { return size_; }

private:
    std::unique_ptr<char[]> buf_;
    uint64_t size_ = 0;
    std::vector<uint64_t> doc_start_;
};

}  // namespace invidx

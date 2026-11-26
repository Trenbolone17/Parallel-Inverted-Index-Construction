// idxcheck: compare, validate, dump or summarize canonical index files.
//
//   idxcheck A.bin B.bin                          exit 0 if identical, 1 if not
//   idxcheck --validate A.bin [--corpus C.txt]    check invariants; with --corpus also
//                                                 recount docs and distinct (term, doc)
//                                                 pairs independently of any builder
//   idxcheck --dump A.bin                         text dump to stdout
//   idxcheck --info A.bin                         sizes and checksum
//   idxcheck --lookup A.bin WORD...               each word's document count and first IDs
//                                                 (binary search over the sorted terms)

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "index_format.hpp"

using namespace invidx;

static void usage(std::FILE* f) {
    std::fputs(
        "usage: idxcheck A.bin B.bin\n"
        "       idxcheck --validate A.bin [--corpus CORPUS]\n"
        "       idxcheck --dump A.bin\n"
        "       idxcheck --info A.bin\n"
        "       idxcheck --lookup A.bin WORD...\n",
        f);
}

// Recounts documents and distinct (term, doc) pairs by sorting each document's tokens.
// Shares no code with the builders or the oracle's std::map.
static std::string check_against_corpus(const Index& ix, const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return "cannot open corpus " + path;
    std::string line, tok;
    std::vector<std::string> toks;
    uint64_t docs = 0, pairs = 0;
    while (std::getline(in, line)) {
        toks.clear();
        std::istringstream words(line);
        while (words >> tok) toks.push_back(tok);
        std::sort(toks.begin(), toks.end());
        pairs += static_cast<uint64_t>(std::unique(toks.begin(), toks.end()) - toks.begin());
        ++docs;
    }
    if (docs != ix.n_docs)
        return "corpus has " + std::to_string(docs) + " docs, index says " + std::to_string(ix.n_docs);
    if (pairs != ix.n_postings())
        return "corpus has " + std::to_string(pairs) + " distinct (term, doc) pairs, index has " +
               std::to_string(ix.n_postings()) + " postings";
    return "";
}

// term<TAB>df<TAB>first IDs (" ..." if more), like --dump; "term<TAB>0" if absent.
static void lookup(const Index& ix, int argc, char** argv, int first) {
    constexpr size_t kShown = 12;
    for (int a = first; a < argc; ++a) {
        const std::string_view w = argv[a];
        uint64_t lo = 0, hi = ix.n_terms();  // terms are sorted like string_view compares
        while (lo < hi) {
            const uint64_t mid = lo + (hi - lo) / 2;
            if (ix.term(mid) < w) lo = mid + 1;
            else hi = mid;
        }
        if (lo == ix.n_terms() || ix.term(lo) != w) {
            std::printf("%s\t0\n", argv[a]);
            continue;
        }
        const auto list = ix.list(lo);
        std::printf("%s\t%zu\t", argv[a], list.size());
        for (size_t i = 0; i < std::min(list.size(), kShown); ++i) std::printf(i ? " %u" : "%u", list[i]);
        std::printf(list.size() > kShown ? " ...\n" : "\n");
    }
}

int main(int argc, char** argv) {
    try {
        if (argc >= 3 && std::strcmp(argv[1], "--validate") == 0) {
            const Index ix = read_index(argv[2]);
            std::string err = validate_index(ix);
            if (err.empty() && argc == 5 && std::strcmp(argv[3], "--corpus") == 0)
                err = check_against_corpus(ix, argv[4]);
            else if (err.empty() && argc != 3) {
                usage(stderr);
                return 2;
            }
            if (!err.empty()) {
                std::fprintf(stderr, "INVALID %s: %s\n", argv[2], err.c_str());
                return 1;
            }
            std::printf("valid: %s\n", argv[2]);
            return 0;
        }
        if (argc == 3 && std::strcmp(argv[1], "--dump") == 0) {
            dump_index(read_index(argv[2]), stdout);
            return 0;
        }
        if (argc == 3 && std::strcmp(argv[1], "--info") == 0) {
            const Index ix = read_index(argv[2]);
            std::printf("docs=%llu terms=%llu postings=%llu term_bytes=%zu checksum=%016llx\n",
                        static_cast<unsigned long long>(ix.n_docs), static_cast<unsigned long long>(ix.n_terms()),
                        static_cast<unsigned long long>(ix.n_postings()), ix.term_bytes.size(),
                        static_cast<unsigned long long>(checksum(ix)));
            return 0;
        }
        if (argc >= 4 && std::strcmp(argv[1], "--lookup") == 0) {
            lookup(read_index(argv[2]), argc, argv, 3);
            return 0;
        }
        if (argc == 3 && argv[1][0] != '-') {
            const std::string diff = compare_index(read_index(argv[1]), read_index(argv[2]));
            if (!diff.empty()) {
                std::fprintf(stderr, "DIFFERENT: %s\n", diff.c_str());
                return 1;
            }
            std::printf("identical\n");
            return 0;
        }
        usage(stderr);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "idxcheck: %s\n", e.what());
        return 2;
    }
}

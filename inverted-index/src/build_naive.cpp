// Test oracle. Deliberately simple and independent of corpus.hpp / hash_table.hpp:
// it reads lines with std::getline and splits with operator>>, which in the classic
// "C" locale separates on exactly the contract's whitespace set. Small inputs only.

#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

#include "builders.hpp"

namespace invidx {

Index build_naive(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);

    std::map<std::string, std::set<uint32_t>> terms;
    std::string line, token;
    uint64_t doc = 0;
    while (std::getline(in, line)) {
        std::istringstream words(line);
        while (words >> token) terms[token].insert(static_cast<uint32_t>(doc));
        ++doc;
    }

    Index ix;
    ix.n_docs = doc;
    for (const auto& [term, docs] : terms) {
        ix.term_bytes.insert(ix.term_bytes.end(), term.begin(), term.end());
        ix.term_off.push_back(ix.term_bytes.size());
        ix.postings.insert(ix.postings.end(), docs.begin(), docs.end());
        ix.post_off.push_back(ix.postings.size());
    }
    return ix;
}

}  // namespace invidx

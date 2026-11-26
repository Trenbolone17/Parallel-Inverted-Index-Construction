#include "index_format.hpp"

#include <cerrno>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "corpus.hpp"
#include "hash.hpp"

namespace invidx {
namespace {

constexpr char kMagic[8] = {'I', 'N', 'V', 'I', 'D', 'X', '0', '1'};

using File = std::unique_ptr<std::FILE, int (*)(std::FILE*)>;

File open_file(const std::string& path, const char* mode) {
    std::FILE* f = std::fopen(path.c_str(), mode);
    if (!f) throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
    return File(f, &std::fclose);
}

// Printable, truncated rendering of a term for error messages.
std::string show(std::string_view t) {
    std::string s;
    for (unsigned char c : t.substr(0, 60)) {
        if (c >= 0x20 && c < 0x7f) {
            s += static_cast<char>(c);
        } else {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\x%02x", c);
            s += buf;
        }
    }
    if (t.size() > 60) s += "...";
    return "\"" + s + "\"";
}

}  // namespace

void write_index(const Index& ix, const std::string& path) {
    File f = open_file(path, "wb");
    std::setvbuf(f.get(), nullptr, _IOFBF, 1 << 20);
    auto put = [&](const void* p, size_t bytes) {
        if (bytes && std::fwrite(p, 1, bytes, f.get()) != bytes)
            throw std::runtime_error("write failed: " + path);
    };
    const uint64_t header[3] = {ix.n_docs, ix.n_terms(), ix.n_postings()};
    put(kMagic, sizeof kMagic);
    put(header, sizeof header);
    put(ix.term_off.data(), ix.term_off.size() * sizeof(uint64_t));
    put(ix.term_bytes.data(), ix.term_bytes.size());
    put(ix.post_off.data(), ix.post_off.size() * sizeof(uint64_t));
    put(ix.postings.data(), ix.postings.size() * sizeof(uint32_t));
    std::FILE* raw = f.release();
    if (std::fclose(raw) != 0) throw std::runtime_error("close failed: " + path);
}

Index read_index(const std::string& path) {
    File f = open_file(path, "rb");
    if (std::fseek(f.get(), 0, SEEK_END) != 0) throw std::runtime_error("cannot seek " + path);
    const uint64_t file_size = static_cast<uint64_t>(std::ftell(f.get()));
    std::rewind(f.get());

    uint64_t remaining = file_size;
    auto get = [&](void* p, uint64_t bytes) {
        if (bytes > remaining) throw std::runtime_error(path + ": truncated or corrupt index");
        if (bytes && std::fread(p, 1, bytes, f.get()) != bytes)
            throw std::runtime_error(path + ": read failed");
        remaining -= bytes;
    };

    char magic[8];
    uint64_t header[3];
    get(magic, sizeof magic);
    if (std::memcmp(magic, kMagic, sizeof kMagic) != 0)
        throw std::runtime_error(path + ": not an INVIDX01 file");
    get(header, sizeof header);
    const uint64_t n_terms = header[1], n_postings = header[2];
    // Size checks before allocating, so a corrupt header cannot trigger a huge allocation.
    if (n_terms + 1 > remaining / 16 || n_postings > remaining / 4)
        throw std::runtime_error(path + ": header sizes exceed file size");

    Index ix;
    ix.n_docs = header[0];
    ix.term_off.resize(n_terms + 1);
    get(ix.term_off.data(), ix.term_off.size() * sizeof(uint64_t));
    if (ix.term_off.front() != 0 || ix.term_off.back() > remaining)
        throw std::runtime_error(path + ": bad term offsets");
    ix.term_bytes.resize(ix.term_off.back());
    get(ix.term_bytes.data(), ix.term_bytes.size());
    ix.post_off.resize(n_terms + 1);
    get(ix.post_off.data(), ix.post_off.size() * sizeof(uint64_t));
    if (ix.post_off.front() != 0 || ix.post_off.back() != n_postings)
        throw std::runtime_error(path + ": bad posting offsets");
    ix.postings.resize(n_postings);
    get(ix.postings.data(), ix.postings.size() * sizeof(uint32_t));
    if (remaining != 0) throw std::runtime_error(path + ": trailing bytes after index");
    return ix;
}

uint64_t checksum(const Index& ix) {
    auto mix = [](uint64_t h, const void* p, size_t bytes) {
        return fmix64(h ^ hash_bytes(static_cast<const char*>(p), bytes)) + 0x9E3779B97F4A7C15ULL;
    };
    uint64_t h = fmix64(ix.n_docs + 1);
    h = mix(h, ix.term_off.data(), ix.term_off.size() * sizeof(uint64_t));
    h = mix(h, ix.term_bytes.data(), ix.term_bytes.size());
    h = mix(h, ix.post_off.data(), ix.post_off.size() * sizeof(uint64_t));
    h = mix(h, ix.postings.data(), ix.postings.size() * sizeof(uint32_t));
    return h;
}

void dump_index(const Index& ix, std::FILE* out) {
    std::fprintf(out, "#docs\t%llu\n", static_cast<unsigned long long>(ix.n_docs));
    for (uint64_t i = 0; i < ix.n_terms(); ++i) {
        const std::string_view t = ix.term(i);
        const auto docs = ix.list(i);
        std::fwrite(t.data(), 1, t.size(), out);
        std::fprintf(out, "\t%zu\t", docs.size());
        for (size_t k = 0; k < docs.size(); ++k)
            std::fprintf(out, k ? " %u" : "%u", docs[k]);
        std::fputc('\n', out);
    }
}

std::string compare_index(const Index& a, const Index& b) {
    if (a.n_docs != b.n_docs)
        return "n_docs differs: " + std::to_string(a.n_docs) + " vs " + std::to_string(b.n_docs);
    const uint64_t n = std::min(a.n_terms(), b.n_terms());
    for (uint64_t i = 0; i < n; ++i) {
        if (a.term(i) != b.term(i))
            return "term #" + std::to_string(i) + " differs: " + show(a.term(i)) + " vs " + show(b.term(i));
        const auto la = a.list(i), lb = b.list(i);
        if (la.size() != lb.size())
            return "df of " + show(a.term(i)) + " differs: " + std::to_string(la.size()) + " vs " +
                   std::to_string(lb.size());
        for (size_t k = 0; k < la.size(); ++k)
            if (la[k] != lb[k])
                return "posting " + std::to_string(k) + " of " + show(a.term(i)) + " differs: " +
                       std::to_string(la[k]) + " vs " + std::to_string(lb[k]);
    }
    if (a.n_terms() != b.n_terms()) {
        const Index& longer = a.n_terms() > b.n_terms() ? a : b;
        return "n_terms differs: " + std::to_string(a.n_terms()) + " vs " + std::to_string(b.n_terms()) +
               " (first extra term " + show(longer.term(n)) + ")";
    }
    return "";
}

std::string validate_index(const Index& ix) {
    const uint64_t nt = ix.n_terms();
    if (ix.term_off.empty() || ix.post_off.size() != ix.term_off.size()) return "offset arrays have wrong sizes";
    if (ix.term_off[0] != 0 || ix.term_off[nt] != ix.term_bytes.size()) return "term_off does not span term_bytes";
    if (ix.post_off[0] != 0 || ix.post_off[nt] != ix.postings.size()) return "post_off does not span postings";
    for (uint64_t i = 0; i < nt; ++i) {
        if (ix.term_off[i + 1] <= ix.term_off[i]) return "empty or negative-length term at #" + std::to_string(i);
        if (ix.post_off[i + 1] <= ix.post_off[i]) return "empty posting list at term #" + std::to_string(i);
    }
    for (uint64_t i = 0; i < nt; ++i) {
        const std::string_view t = ix.term(i);
        for (char c : t)
            if (is_sep(c)) return "term " + show(t) + " contains whitespace";
        if (i > 0 && !(ix.term(i - 1) < t))
            return "terms out of order: " + show(ix.term(i - 1)) + " then " + show(t);
        const auto docs = ix.list(i);
        for (size_t k = 0; k < docs.size(); ++k) {
            if (docs[k] >= ix.n_docs)
                return "doc id " + std::to_string(docs[k]) + " >= n_docs in list of " + show(t);
            if (k > 0 && docs[k] <= docs[k - 1])
                return "posting list of " + show(t) + " not strictly increasing at position " + std::to_string(k);
        }
    }
    return "";
}

}  // namespace invidx

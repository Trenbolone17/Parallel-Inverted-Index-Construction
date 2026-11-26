// gen_zipf: synthetic pretokenized corpus with controllable vocabulary, term skew and
// document length. Output is deterministic for a given parameter set, independent of
// the thread count (each fixed-size chunk of documents has its own RNG stream).
//
// Term rank r (0 = most frequent) is drawn with P(r) ~ 1 / (r + 1)^s via an alias
// table; s = 0 gives a uniform vocabulary. Rank r is written as the bijective base-26
// numeral of r, so frequent terms are short words, as in natural language.

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <string>
#include <vector>

namespace {

uint64_t fmix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

struct Rng {  // splitmix64
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }  // [0, 1)
    uint64_t below(uint64_t n) { return static_cast<uint64_t>((static_cast<unsigned __int128>(next()) * n) >> 64); }
    double normal() {
        const double u1 = uniform(), u2 = uniform();
        return std::sqrt(-2.0 * std::log(1.0 - u1)) * std::cos(2.0 * std::numbers::pi * u2);
    }
};

// Vose alias table for O(1) sampling from a Zipf distribution over V ranks.
struct ZipfAlias {
    std::vector<float> prob;
    std::vector<uint32_t> alias;

    ZipfAlias(uint32_t V, double s) : prob(V), alias(V) {
        std::vector<double> w(V);
        double sum = 0;
        for (uint32_t r = 0; r < V; ++r) sum += (w[r] = std::pow(r + 1.0, -s));
        std::vector<uint32_t> small, large;
        for (uint32_t r = 0; r < V; ++r) {
            w[r] = w[r] * V / sum;
            (w[r] < 1.0 ? small : large).push_back(r);
        }
        while (!small.empty() && !large.empty()) {
            const uint32_t l = small.back();
            small.pop_back();
            const uint32_t g = large.back();
            prob[l] = static_cast<float>(w[l]);
            alias[l] = g;
            w[g] = (w[g] + w[l]) - 1.0;
            if (w[g] < 1.0) {
                large.pop_back();
                small.push_back(g);
            }
        }
        for (uint32_t g : large) prob[g] = 1.0f, alias[g] = g;
        for (uint32_t l : small) prob[l] = 1.0f, alias[l] = l;
    }

    uint32_t sample(Rng& rng) const {
        const auto i = static_cast<uint32_t>(rng.below(prob.size()));
        return rng.uniform() < prob[i] ? i : alias[i];
    }
};

void append_word(std::string& out, uint64_t rank) {  // bijective base-26: a..z, aa..zz, ...
    char tmp[16];
    int n = 0;
    for (uint64_t x = rank + 1; x > 0; x = (x - 1) / 26) tmp[n++] = static_cast<char>('a' + (x - 1) % 26);
    out.append(tmp, n);
}

enum class LenDist { Fixed, Uniform, Lognormal };

void usage(std::FILE* f) {
    std::fputs(
        "usage: gen_zipf --docs N [options]\n"
        "  --vocab V         vocabulary size (default 100000)\n"
        "  --zipf S          Zipf exponent, 0 = uniform (default 1.0)\n"
        "  --len L           mean document length in tokens (default 100)\n"
        "  --len-dist D      fixed | uniform (1..2L-1) | lognormal (default lognormal)\n"
        "  --len-sigma X     lognormal sigma (default 1.0)\n"
        "  --seed K          RNG seed (default 1)\n"
        "  --threads T       generator threads (output does not depend on T)\n"
        "  --out FILE        output file (default stdout)\n",
        f);
}

}  // namespace

int main(int argc, char** argv) {
    uint64_t n_docs = 0, seed = 1;
    uint32_t vocab = 100000;
    double s = 1.0, mean_len = 100, sigma = 1.0;
    LenDist dist = LenDist::Lognormal;
    int threads = omp_get_max_threads();
    std::string out_path = "-";

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            usage(stdout);
            return 0;
        }
        if (i + 1 >= argc) {
            std::fprintf(stderr, "gen_zipf: missing value for %s\n", a.c_str());
            return 2;
        }
        const std::string v = argv[++i];
        if (a == "--docs") n_docs = std::strtoull(v.c_str(), nullptr, 10);
        else if (a == "--vocab") vocab = static_cast<uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
        else if (a == "--zipf") s = std::atof(v.c_str());
        else if (a == "--len") mean_len = std::atof(v.c_str());
        else if (a == "--len-sigma") sigma = std::atof(v.c_str());
        else if (a == "--seed") seed = std::strtoull(v.c_str(), nullptr, 10);
        else if (a == "--threads") threads = std::atoi(v.c_str());
        else if (a == "--out") out_path = v;
        else if (a == "--len-dist") {
            if (v == "fixed") dist = LenDist::Fixed;
            else if (v == "uniform") dist = LenDist::Uniform;
            else if (v == "lognormal") dist = LenDist::Lognormal;
            else {
                std::fprintf(stderr, "gen_zipf: unknown --len-dist %s\n", v.c_str());
                return 2;
            }
        } else {
            std::fprintf(stderr, "gen_zipf: unknown option %s\n", a.c_str());
            usage(stderr);
            return 2;
        }
    }
    if (n_docs == 0 || vocab == 0 || s < 0 || mean_len < 1 || threads < 1 || n_docs > 0xFFFFFFFEull) {
        usage(stderr);
        return 2;
    }

    std::FILE* out = out_path == "-" ? stdout : std::fopen(out_path.c_str(), "wb");
    if (!out) {
        std::perror(out_path.c_str());
        return 1;
    }

    const double t0 = omp_get_wtime();
    const ZipfAlias zipf(vocab, s);
    const auto L = static_cast<uint64_t>(std::llround(mean_len));
    const double mu = std::log(mean_len) - sigma * sigma / 2;  // lognormal with mean = mean_len
    const uint64_t max_len = 1000 * L + 1000;
    auto doc_len = [&](Rng& rng) -> uint64_t {
        switch (dist) {
            case LenDist::Fixed: return L;
            case LenDist::Uniform: return 1 + rng.below(2 * L - 1);
            case LenDist::Lognormal:
            default: return std::min<uint64_t>(max_len, std::llround(std::exp(mu + sigma * rng.normal())));
        }
    };

    // Fixed chunking (depends only on --len) keeps the output independent of --threads.
    const uint64_t docs_per_chunk = std::max<uint64_t>(1, (1u << 16) / L);
    const uint64_t n_chunks = (n_docs + docs_per_chunk - 1) / docs_per_chunk;
    const uint64_t batch = 4ull * threads;
    std::vector<std::string> bufs(batch);
    uint64_t tokens = 0, bytes = 0;

    for (uint64_t base = 0; base < n_chunks; base += batch) {
        const uint64_t end = std::min(n_chunks, base + batch);
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1) reduction(+ : tokens)
        for (uint64_t c = base; c < end; ++c) {
            std::string& buf = bufs[c - base];
            buf.clear();
            Rng rng(fmix64(seed * 0x9E3779B97F4A7C15ULL + 1) ^ fmix64(c + 1));
            const uint64_t lo = c * docs_per_chunk, hi = std::min(n_docs, lo + docs_per_chunk);
            for (uint64_t d = lo; d < hi; ++d) {
                const uint64_t len = doc_len(rng);
                for (uint64_t j = 0; j < len; ++j) {
                    if (j) buf.push_back(' ');
                    append_word(buf, zipf.sample(rng));
                }
                buf.push_back('\n');
                tokens += len;
            }
        }
        for (uint64_t c = base; c < end; ++c) {
            const std::string& buf = bufs[c - base];
            if (std::fwrite(buf.data(), 1, buf.size(), out) != buf.size()) {
                std::perror("gen_zipf: write");
                return 1;
            }
            bytes += buf.size();
        }
    }
    if (out != stdout && std::fclose(out) != 0) {
        std::perror("gen_zipf: close");
        return 1;
    }
    std::fprintf(stderr, "gen_zipf: docs=%llu tokens=%llu bytes=%llu vocab=%u zipf=%.3g time=%.2fs\n",
                 static_cast<unsigned long long>(n_docs), static_cast<unsigned long long>(tokens),
                 static_cast<unsigned long long>(bytes), vocab, s, omp_get_wtime() - t0);
    return 0;
}

// invidx: build an inverted index from a pretokenized corpus.
//
// Timed phases: load (file I/O, reported separately), scan (document boundaries),
// builder-specific phases, build_total (scan + builder), write (optional output).

#include <omp.h>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "builders.hpp"

using namespace invidx;

static void usage(std::FILE* f) {
    std::fputs(
        "usage: invidx --algo naive|seq|omp|heavy --input CORPUS [options]\n"
        "  --out FILE                write the canonical binary index\n"
        "  --threads N               worker threads (parallel algorithms; default: OMP default)\n"
        "  --stats FILE              write phase times, counters and peak RSS as JSON\n"
        "  --quiet                   no summary on stderr\n"
        "omp / heavy options:\n"
        "  --schedule static|dynamic one contiguous doc block per thread (default) or dynamic chunks\n"
        "  --ranges-per-thread K     merge term ranges = K * threads (default 4)\n"
        "  --chunks-per-thread C     dynamic schedule: doc chunks = C * threads (default 16)\n"
        "heavy (D3 = omp + heavy-term bitmaps) options:\n"
        "  --theta X                 heavy if sampled df / sampled docs >= X (default 1/32)\n"
        "  --sample F                fraction of documents sampled to find heavy terms (default 0.01)\n"
        "  --heavy-mem-mb M          memory budget for heavy bitmaps, caps their number (default 1024)\n"
        "corpus: one document per line (doc ID = line number), tokens separated by ASCII whitespace\n",
        f);
}

int main(int argc, char** argv) {
    std::string algo, input, out, stats_path, schedule = "static";
    int threads = omp_get_max_threads();
    bool quiet = false;
    OmpOptions omp_opt;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "invidx: missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--algo") algo = value();
        else if (a == "--input") input = value();
        else if (a == "--out") out = value();
        else if (a == "--threads") threads = std::atoi(value().c_str());
        else if (a == "--stats") stats_path = value();
        else if (a == "--quiet") quiet = true;
        else if (a == "--schedule") schedule = value();
        else if (a == "--ranges-per-thread") omp_opt.ranges_per_thread = std::atoi(value().c_str());
        else if (a == "--chunks-per-thread") omp_opt.chunks_per_thread = std::atoi(value().c_str());
        else if (a == "--theta") omp_opt.theta = std::atof(value().c_str());
        else if (a == "--sample") omp_opt.sample = std::atof(value().c_str());
        else if (a == "--heavy-mem-mb") omp_opt.heavy_mem_mb = std::atof(value().c_str());
        else if (a == "-h" || a == "--help") {
            usage(stdout);
            return 0;
        } else {
            std::fprintf(stderr, "invidx: unknown option %s\n", a.c_str());
            usage(stderr);
            return 2;
        }
    }
    if (algo.empty() || input.empty() || threads < 1 || (schedule != "static" && schedule != "dynamic")) {
        usage(stderr);
        return 2;
    }
    omp_opt.dynamic = schedule == "dynamic";

    try {
        Stats st;
        st.info("algo", algo);
        st.info("input", input);
        const char* places = std::getenv("OMP_PLACES");
        const char* bind = std::getenv("OMP_PROC_BIND");
        st.info("omp_places", places ? places : "");
        st.info("omp_proc_bind", bind ? bind : "");

        Index ix;
        if (algo == "naive") {
            threads = 1;
            const double t = now();
            ix = build_naive(input);
            st.phase("build_total", now() - t);
        } else if (algo == "seq") {
            threads = 1;  // best sequential baseline: every phase single-threaded
            double t = now();
            Corpus corpus = Corpus::load(input);
            st.phase("load", now() - t);
            st.counter("input_bytes", corpus.size());

            const double build_start = now();
            corpus.index_docs(threads);
            st.phase("scan", now() - build_start);
            ix = build_seq(corpus, st);
            st.phase("build_total", now() - build_start);
        } else if (algo == "omp" || algo == "heavy") {
            omp_opt.heavy = algo == "heavy";
            st.info("schedule", schedule);
            if (omp_opt.heavy) {
                st.info("theta", std::to_string(omp_opt.theta));
                st.info("sample", std::to_string(omp_opt.sample));
                st.info("heavy_mem_mb", std::to_string(omp_opt.heavy_mem_mb));
            }
            double t = now();
            Corpus corpus = Corpus::load(input);
            st.phase("load", now() - t);
            st.counter("input_bytes", corpus.size());

            const double build_start = now();
            corpus.index_docs(threads);
            st.phase("scan", now() - build_start);
            ix = build_omp(corpus, threads, omp_opt, st);
            st.phase("build_total", now() - build_start);
        } else {
            std::fprintf(stderr, "invidx: unknown --algo '%s' (implemented: naive, seq, omp, heavy)\n", algo.c_str());
            return 2;
        }

        st.threads = threads;
        st.counter("n_docs", ix.n_docs);
        st.counter("n_terms", ix.n_terms());
        st.counter("n_postings", ix.n_postings());
        char sum[24];
        std::snprintf(sum, sizeof sum, "%016llx", static_cast<unsigned long long>(checksum(ix)));
        st.info("checksum", sum);

        if (!out.empty()) {
            const double t = now();
            write_index(ix, out);
            st.phase("write", now() - t);
        }
        if (!stats_path.empty()) st.write_json(stats_path);
        if (!quiet) st.print_summary(stderr);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "invidx: %s\n", e.what());
        return 1;
    }
    return 0;
}

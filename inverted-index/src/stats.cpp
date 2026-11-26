#include "stats.hpp"

#include <omp.h>
#include <sys/resource.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace invidx {

double now() { return omp_get_wtime(); }

uint64_t peak_rss_kb() {
    struct rusage ru {};
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<uint64_t>(ru.ru_maxrss);  // kilobytes on Linux
}

namespace {
template <class T>
void set_value(std::vector<std::pair<std::string, T>>& v, const std::string& key, T value) {
    for (auto& [k, x] : v)
        if (k == key) {
            x = std::move(value);
            return;
        }
    v.emplace_back(key, std::move(value));
}

void json_string(std::FILE* f, const std::string& s) {
    std::fputc('"', f);
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') {
            std::fputc('\\', f);
            std::fputc(c, f);
        } else if (c < 0x20) {
            std::fprintf(f, "\\u%04x", c);
        } else {
            std::fputc(c, f);
        }
    }
    std::fputc('"', f);
}
}  // namespace

void Stats::info(const std::string& key, const std::string& value) { set_value(infos, key, value); }
void Stats::counter(const std::string& name, uint64_t value) { set_value(counters, name, value); }
void Stats::thread_phase(const std::string& name, std::vector<double> per_thread) {
    set_value(thread_phases, name, std::move(per_thread));
}
void Stats::phase(const std::string& name, double seconds) {  // accumulates repeated phases
    for (auto& [k, x] : phases)
        if (k == name) {
            x += seconds;
            return;
        }
    phases.emplace_back(name, seconds);
}

void Stats::write_json(const std::string& path) const {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));

    std::fprintf(f, "{\n  \"threads\": %d,\n  \"peak_rss_kb\": %llu,\n  \"info\": {", threads,
                 static_cast<unsigned long long>(peak_rss_kb()));
    for (size_t i = 0; i < infos.size(); ++i) {
        std::fputs(i ? ", " : "", f);
        json_string(f, infos[i].first);
        std::fputs(": ", f);
        json_string(f, infos[i].second);
    }
    std::fputs("},\n  \"phases_s\": {", f);
    for (size_t i = 0; i < phases.size(); ++i) {
        std::fputs(i ? ", " : "", f);
        json_string(f, phases[i].first);
        std::fprintf(f, ": %.9f", phases[i].second);
    }
    std::fputs("},\n  \"counters\": {", f);
    for (size_t i = 0; i < counters.size(); ++i) {
        std::fputs(i ? ", " : "", f);
        json_string(f, counters[i].first);
        std::fprintf(f, ": %llu", static_cast<unsigned long long>(counters[i].second));
    }
    std::fputs("},\n  \"thread_phases_s\": {", f);
    for (size_t i = 0; i < thread_phases.size(); ++i) {
        std::fputs(i ? ", " : "", f);
        json_string(f, thread_phases[i].first);
        std::fputs(": [", f);
        const auto& v = thread_phases[i].second;
        for (size_t k = 0; k < v.size(); ++k) std::fprintf(f, k ? ", %.9f" : "%.9f", v[k]);
        std::fputc(']', f);
    }
    std::fputs("}\n}\n", f);
    if (std::fclose(f) != 0) throw std::runtime_error("write failed: " + path);
}

void Stats::print_summary(std::FILE* out) const {
    for (const auto& [k, v] : infos)
        if (k == "algo") std::fprintf(out, "[%s] ", v.c_str());
    std::fprintf(out, "threads=%d", threads);
    for (const auto& [k, v] : counters)
        if (k == "n_docs" || k == "n_terms" || k == "n_postings" || k == "tokens")
            std::fprintf(out, " %s=%llu", k.c_str(), static_cast<unsigned long long>(v));
    for (const auto& [k, v] : infos)
        if (k == "checksum") std::fprintf(out, " checksum=%s", v.c_str());
    std::fputs("\n ", out);
    for (const auto& [k, v] : phases) std::fprintf(out, " %s=%.4fs", k.c_str(), v);
    std::fprintf(out, "  peak_rss=%.1fMB\n", peak_rss_kb() / 1024.0);
}

}  // namespace invidx

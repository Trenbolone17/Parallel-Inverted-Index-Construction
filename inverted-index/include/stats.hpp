#pragma once
// Phase timers, per-thread timers, work counters and peak memory, written as JSON
// for the benchmark scripts.

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace invidx {

double now();              // wall-clock seconds (omp_get_wtime)
uint64_t peak_rss_kb();    // peak resident set size of this process

struct Stats {
    int threads = 1;
    std::vector<std::pair<std::string, std::string>> infos;
    std::vector<std::pair<std::string, double>> phases;  // seconds, in recording order
    std::vector<std::pair<std::string, uint64_t>> counters;
    std::vector<std::pair<std::string, std::vector<double>>> thread_phases;  // seconds per thread

    void info(const std::string& key, const std::string& value);
    void phase(const std::string& name, double seconds);  // accumulates if repeated
    void counter(const std::string& name, uint64_t value);
    void thread_phase(const std::string& name, std::vector<double> per_thread);

    void write_json(const std::string& path) const;
    void print_summary(std::FILE* out) const;
};

}  // namespace invidx

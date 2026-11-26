#include "corpus.hpp"

#include <fcntl.h>
#include <omp.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace invidx {

Corpus Corpus::load(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        throw std::runtime_error("cannot stat " + path + ": " + std::strerror(errno));
    }
    uint64_t n = static_cast<uint64_t>(st.st_size);

    Corpus c;
    c.buf_.reset(new char[n + 1]);  // +1 for a possibly missing final '\n'; not zero-filled
    uint64_t got = 0;
    while (got < n) {
        const ssize_t r = ::read(fd, c.buf_.get() + got, std::min<uint64_t>(n - got, 1ull << 30));
        if (r < 0) {
            if (errno == EINTR) continue;
            const int err = errno;
            ::close(fd);
            throw std::runtime_error("cannot read " + path + ": " + std::strerror(err));
        }
        if (r == 0) break;
        got += static_cast<uint64_t>(r);
    }
    ::close(fd);
    if (got != n) throw std::runtime_error("short read on " + path);

    if (n > 0 && c.buf_[n - 1] != '\n') c.buf_[n++] = '\n';
    c.size_ = n;
    return c;
}

void Corpus::index_docs(int threads) {
    const char* const b = buf_.get();
    const uint64_t n = size_;
    if (threads < 1) threads = 1;

    // count[t + 1] = newlines in thread t's byte range; prefix-summed into start indices.
    std::vector<uint64_t> count(static_cast<size_t>(threads) + 1, 0);
    int used = 1;

#pragma omp parallel num_threads(threads)
    {
        const int t = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const char* const lo = b + n * t / nt;
        const char* const hi = b + n * (t + 1) / nt;

        uint64_t c = 0;
        for (const char* p = lo; (p = static_cast<const char*>(std::memchr(p, '\n', hi - p))); ++p) ++c;
        count[t + 1] = c;

#pragma omp barrier
#pragma omp single
        {
            used = nt;
            for (int i = 0; i < nt; ++i) count[i + 1] += count[i];
            doc_start_.assign(count[nt] + 1, 0);
        }  // implicit barrier

        uint64_t k = count[t];
        for (const char* p = lo; (p = static_cast<const char*>(std::memchr(p, '\n', hi - p))); ++p)
            doc_start_[++k] = static_cast<uint64_t>(p - b) + 1;
    }

    if (count[used] > kMaxDocs)
        throw std::runtime_error("too many documents for 32-bit document IDs");
}

}  // namespace invidx

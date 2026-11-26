CXX      := g++
CXXFLAGS := -std=c++20 -O3 -march=native -fopenmp -Wall -Wextra -Iinclude
LDFLAGS  := -fopenmp
BIN      := bin

HEADERS  := $(wildcard include/*.hpp)
LIB_SRC  := src/corpus.cpp src/index_format.cpp src/stats.cpp
BUILDERS := src/build_naive.cpp src/build_seq.cpp src/build_omp.cpp

SAN_FLAGS := -std=c++20 -O1 -g -fno-omit-frame-pointer -fopenmp -Wall -Wextra -Iinclude

.PHONY: all test asan tsan phoenix clean

all: $(BIN)/invidx $(BIN)/idxcheck $(BIN)/gen_zipf

$(BIN):
	mkdir -p $@

$(BIN)/invidx: src/invidx.cpp $(BUILDERS) $(LIB_SRC) $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ src/invidx.cpp $(BUILDERS) $(LIB_SRC) $(LDFLAGS)

$(BIN)/idxcheck: src/idxcheck.cpp $(LIB_SRC) $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ src/idxcheck.cpp $(LIB_SRC) $(LDFLAGS)

$(BIN)/gen_zipf: tools/gen_zipf.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ tools/gen_zipf.cpp $(LDFLAGS)

# Phoenix++ reference (authors' runtime, fetched at a pinned commit) running our index as an app.
# Phoenix++ predates C++11 two-phase lookup, so its headers need -fpermissive with current GCC.
PHOENIX_REPO   := https://github.com/kozyraki/phoenix.git
PHOENIX_COMMIT := 1276c8d8f3b82050071d0a7a4b8a352a05d1faab
PHOENIX        := third_party/phoenix/phoenix++-1.0
PHOENIX_SRC    := $(PHOENIX)/src/task_queue.cpp $(PHOENIX)/src/thread_pool.cpp

phoenix: $(BIN)/phoenix_index

$(PHOENIX_SRC):
	git clone $(PHOENIX_REPO) third_party/phoenix
	git -C third_party/phoenix checkout --quiet $(PHOENIX_COMMIT)

$(BIN)/phoenix_index: baselines/phoenix_index.cpp $(PHOENIX_SRC) $(LIB_SRC) $(HEADERS) | $(BIN)
	$(CXX) $(CXXFLAGS) -fpermissive -w -D_LINUX_ -isystem $(PHOENIX)/include -o $@ \
	    baselines/phoenix_index.cpp $(PHOENIX_SRC) $(LIB_SRC) $(LDFLAGS) -lpthread

# Hand-checked tests + randomized differential tests against the naive oracle.
test: all
	BIN_DIR=$(BIN) tests/run_tests.sh

# AddressSanitizer + UndefinedBehaviorSanitizer build, then the same tests.
asan:
	$(MAKE) BIN=bin-asan CXXFLAGS="$(SAN_FLAGS) -fsanitize=address,undefined" \
	        LDFLAGS="-fopenmp -fsanitize=address,undefined" all
	BIN_DIR=bin-asan SEEDS=40 tests/run_tests.sh

# ThreadSanitizer needs LLVM's OpenMP runtime (libomp); GCC's libgomp gives false positives.
tsan:
	@test -e /usr/lib/libomp.so || { echo "tsan needs LLVM OpenMP: sudo pacman -S openmp"; exit 1; }
	$(MAKE) BIN=bin-tsan CXX=clang++ CXXFLAGS="$(SAN_FLAGS) -fsanitize=thread" \
	        LDFLAGS="-fopenmp -fsanitize=thread" all
	BIN_DIR=bin-tsan SEEDS=40 tests/run_tests.sh

clean:
	rm -rf bin bin-asan bin-tsan

#!/usr/bin/env bash
# Correctness tests.
#   1. Hand-checked inputs (tests/tiny): every builder's dump must equal the
#      hand-written .expected file, and the index must pass idxcheck --validate.
#   2. Randomized differential tests: gen_zipf corpora with varied size, vocabulary,
#      skew and length distribution; every builder must be byte-identical to the
#      naive oracle at every thread count.
#
# Env: BIN_DIR (bin), ALGOS (builders under test besides the oracle),
#      THREADS (thread counts for parallel builders), SEEDS (number of random corpora).
set -uo pipefail
cd "$(dirname "$0")/.."

BIN_DIR=${BIN_DIR:-bin}
ALGOS=${ALGOS:-"seq omp omp-dynamic omp-r1 heavy heavy-all heavy-dynamic heavy-budget heavy-none"}
THREADS=${THREADS:-"1 2 3 7 16 24"}
SEEDS=${SEEDS:-200}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
pass=0
fail=0
bad() { fail=$((fail + 1)); echo "FAIL: $*"; }

threads_for() {  # sequential builders ignore --threads; run them once
    case "$1" in naive | seq) echo 1 ;; *) echo "$THREADS" ;; esac
}

algo_args() {  # test variant name -> invidx arguments
    case "$1" in
        omp-dynamic) echo "--algo omp --schedule dynamic --chunks-per-thread 3" ;;
        omp-r1) echo "--algo omp --ranges-per-thread 1" ;;
        # D3. Small corpora put many chunk boundaries inside shared bitmap words (atomic path).
        heavy) echo "--algo heavy" ;;
        heavy-all) echo "--algo heavy --theta 0 --sample 1" ;;  # every term heavy
        heavy-dynamic) echo "--algo heavy --schedule dynamic --chunks-per-thread 3 --theta 0.02 --sample 0.3" ;;
        heavy-budget) echo "--algo heavy --theta 0 --sample 1 --heavy-mem-mb 0.0005" ;;  # a few heavy
        heavy-none) echo "--algo heavy --heavy-mem-mb 0" ;;  # budget 0: must behave exactly like omp
        *) echo "--algo $1" ;;
    esac
}

build() {  # build <variant> <threads> <corpus> <out>
    if [ "$1" = phoenix ]; then  # Phoenix++ baseline (make phoenix; test with ALGOS=phoenix)
        "$BIN_DIR/phoenix_index" --threads "$2" --input "$3" --out "$4" --quiet
        return
    fi
    # shellcheck disable=SC2046
    "$BIN_DIR/invidx" $(algo_args "$1") --threads "$2" --input "$3" --out "$4" --quiet
}

# ---- 1. hand-checked inputs -------------------------------------------------
for input in tests/tiny/*.txt; do
    expected="${input%.txt}.expected"
    for algo in naive $ALGOS; do
        for t in $(threads_for "$algo"); do
            label="$algo t=$t $(basename "$input")"
            if ! build "$algo" "$t" "$input" "$TMP/out.bin"; then bad "$label: invidx failed"; continue; fi
            "$BIN_DIR/idxcheck" --dump "$TMP/out.bin" >"$TMP/out.dump"
            if ! cmp -s "$TMP/out.dump" "$expected"; then
                bad "$label: dump differs from $expected"
                diff "$TMP/out.dump" "$expected" | head -6 | cut -c1-120
                continue
            fi
            if ! "$BIN_DIR/idxcheck" --validate "$TMP/out.bin" --corpus "$input" >/dev/null; then
                bad "$label: validation failed"
                continue
            fi
            pass=$((pass + 1))
        done
    done
done
echo "hand-checked: $pass passed, $fail failed"

# ---- 2. randomized differential tests vs the naive oracle -------------------
zipfs=(0 1.0 1.5)
dists=(lognormal uniform fixed)
hand_pass=$pass
for seed in $(seq 1 "$SEEDS"); do
    docs=$(((seed * 7919) % 600 + 1))
    vocab=$(((seed * 104729) % 3000 + 1))
    len=$(((seed * 31) % 60 + 1))
    zipf=${zipfs[$((seed % 3))]}
    dist=${dists[$(((seed / 3) % 3))]}
    params="seed=$seed docs=$docs vocab=$vocab zipf=$zipf len=$len dist=$dist"

    corpus="$TMP/corpus.txt"
    if ! "$BIN_DIR/gen_zipf" --docs "$docs" --vocab "$vocab" --zipf "$zipf" --len "$len" \
        --len-dist "$dist" --seed "$seed" --threads 2 --out "$corpus" 2>/dev/null; then
        bad "gen_zipf $params"
        continue
    fi
    if ! build naive 1 "$corpus" "$TMP/ref.bin"; then bad "naive $params"; continue; fi
    if ! "$BIN_DIR/idxcheck" --validate "$TMP/ref.bin" --corpus "$corpus" >/dev/null; then
        bad "oracle output invalid: $params"
        continue
    fi
    for algo in $ALGOS; do
        for t in $(threads_for "$algo"); do
            if build "$algo" "$t" "$corpus" "$TMP/x.bin" && "$BIN_DIR/idxcheck" "$TMP/ref.bin" "$TMP/x.bin" >/dev/null 2>"$TMP/err"; then
                pass=$((pass + 1))
            else
                bad "$algo t=$t $params: $(cat "$TMP/err")"
            fi
        done
    done
done
echo "differential: $((pass - hand_pass)) passed"

# ---- 3. one large corpus: > 262,144 docs, so each heavy bitmap spans several decode
# segments (4096 words), with many doc blocks/chunks.
diff_pass=$pass
if [ "${LARGE:-1}" = 1 ]; then
    big="$TMP/big.txt"
    "$BIN_DIR/gen_zipf" --docs 600001 --vocab 400 --zipf 1.1 --len 4 --len-dist uniform --seed 99 \
        --out "$big" 2>/dev/null
    build naive 1 "$big" "$TMP/big_ref.bin" || bad "naive on large corpus"
    for variant in "--algo heavy" "--algo heavy --schedule dynamic" "--algo heavy --theta 0 --sample 0.05" "--algo omp"; do
        for t in 1 5 24; do
            # shellcheck disable=SC2086
            if "$BIN_DIR/invidx" $variant --threads "$t" --input "$big" --out "$TMP/big.bin" --quiet &&
                "$BIN_DIR/idxcheck" "$TMP/big_ref.bin" "$TMP/big.bin" >/dev/null 2>"$TMP/err"; then
                pass=$((pass + 1))
            else
                bad "large corpus: $variant t=$t: $(cat "$TMP/err")"
            fi
        done
    done
    echo "large corpus: $((pass - diff_pass)) passed"
fi

echo "TOTAL: $pass passed, $fail failed"
[ "$fail" -eq 0 ]

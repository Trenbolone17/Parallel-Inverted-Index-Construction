#!/usr/bin/env bash
# Live demo for the presentation: the same index from every builder.
#
#   ./demo.sh                     Wikipedia (data/wiki.txt); waits for Enter between steps
#   ./demo.sh --no-pause          run straight through (a rehearsal)
#   ./demo.sh --corpus FILE       any pretokenized corpus (without data/wiki.txt: a synthetic one)
#   ./demo.sh --words "a b c"     words to look up at the end
#   ./demo.sh --threads N         threads for the parallel builders (default: every CPU)
#   ./demo.sh --skip-tests        leave out step 2
#
# 1. the three-document example from slide 2   2. a quick correctness check
# 3. full-size builds: sequential, D2, D3 (8 P-cores and all threads), Phoenix++ if built
# 4. byte-for-byte comparison   5. word lookups in the finished index
# Threads are pinned like the benchmarks (bench/run.py): one per P-core, then E-cores, then
# hyperthreads. Index files go to a temporary folder that is removed at the end.
set -euo pipefail
cd "$(dirname "$0")"

PAUSE=1 TESTS=1 CORPUS="" WORDS="" THREADS=$(nproc)
while [ $# -gt 0 ]; do
    case "$1" in
        --no-pause) PAUSE=0 ;;
        --skip-tests) TESTS=0 ;;
        --corpus) CORPUS=$2; shift ;;
        --words) WORDS=$2; shift ;;
        --threads) THREADS=$2; shift ;;
        -h | --help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "demo.sh: unknown option $1 (try --help)" >&2; exit 2 ;;
    esac
    shift
done
[ -t 0 ] || PAUSE=0

if [ -t 1 ]; then G=$'\e[1;32m' Y=$'\e[1;33m' B=$'\e[1m' D=$'\e[2m' R=$'\e[0m'; else G="" Y="" B="" D="" R=""; fi
step() { printf '\n%s━━ %s ━━%s\n' "$G" "$1" "$R"; }
say() { printf '%s%s%s\n' "$D" "$1" "$R"; }
show() { printf '%s$ %s%s\n' "$B" "$1" "$R"; }  # the command as the audience should read it
pause() { if [ "$PAUSE" = 1 ]; then read -r -p "${D}[Enter]${R} " _; fi; }
group() { python3 -c 'import sys; print(f"{int(sys.argv[1]):,}")' "$1"; }  # 6,407,814 whatever the locale

make -s all
OUT=$(mktemp -d "${TMPDIR:-/tmp}/invidx-demo.XXXXXX")
trap 'rm -rf -- "$OUT"' EXIT

# Same CPU order as the benchmark harness.
read -r -a ORDER <<<"$(python3 -c 'import sys; sys.path.insert(0, "bench"); import run; print(*run.placement_order()[0])')"
places() {
    local s="" i
    for ((i = 0; i < $1; i++)); do s+="{${ORDER[i]}},"; done
    printf '%s' "${s%,}"
}

if [ -z "$CORPUS" ]; then
    if [ -s data/wiki.txt ]; then
        CORPUS=data/wiki.txt
    else
        CORPUS=data/demo_zipf.txt
        mkdir -p data
        [ -s "$CORPUS" ] || bin/gen_zipf --docs 2000000 --vocab 1000000 --zipf 1.0 --len 100 --seed 7 --out "$CORPUS"
    fi
fi
[ -s "$CORPUS" ] || { echo "demo.sh: no corpus at $CORPUS" >&2; exit 1; }
if [ -z "$WORDS" ]; then
    case "$CORPUS" in *wiki* | *gutenberg*) WORDS="parallel inverted index" ;; *) WORDS="a b abc" ;; esac
fi

# build <label> <threads> <index file or -> <program and its options...>
ROWS=()
build() {
    local label=$1 t=$2 out=$3
    shift 3
    local stats="$OUT/stats-${#ROWS[@]}.json" extra=()
    [ "$out" = - ] || extra=(--out "$OUT/$out")
    show "$* --threads $t --input $CORPUS$([ "$out" = - ] || printf ' --out %s' "$out")"
    OMP_PLACES=$(places "$t") OMP_PROC_BIND=close "$@" --threads "$t" --input "$CORPUS" "${extra[@]}" \
        --stats "$stats" --quiet
    local secs gb sum
    read -r secs gb sum < <(python3 -c 'import json, sys; s = json.load(open(sys.argv[1]))
print("%.2f %.1f %s" % (s["phases_s"]["build_total"], s["peak_rss_kb"] / 1048576, s["info"]["checksum"]))' "$stats")
    printf '  → %s%s s%s to build, peak memory %s GB, index checksum %s\n' "$B" "$secs" "$R" "$gb" "$sum"
    ROWS+=("$label|$t|$secs|$gb|$sum")
}

printf '%sParallel inverted index construction: live demo%s\n' "$G" "$R"
say "$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'), $(nproc) hardware threads"
profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null || true)
if [ -n "$profile" ] && [ "$profile" != performance ]; then
    printf '%snote: power profile is "%s"; the slides were measured with: powerprofilesctl set performance%s\n' \
        "$Y" "$profile" "$R"
fi
if grep -qx 0 /sys/class/power_supply/A*/online 2>/dev/null; then
    printf '%snote: on battery; plug in for the timings in the slides%s\n' "$Y" "$R"
fi

step "1 · The problem in miniature"
printf 'the cat sat\nthe dog sat\na cat ran\n' >"$OUT/tiny.txt"
show "cat tiny.txt                      # one document per line: documents 0, 1, 2"
cat "$OUT/tiny.txt"
show "bin/invidx --algo seq --input tiny.txt --out tiny.bin"
bin/invidx --algo seq --input "$OUT/tiny.txt" --out "$OUT/tiny.bin" --quiet
show "bin/idxcheck --dump tiny.bin       # word, number of documents, document IDs"
bin/idxcheck --dump "$OUT/tiny.bin" | tail -n +2
pause

if [ "$TESTS" = 1 ]; then
    step "2 · Correctness: every builder must equal the reference byte for byte"
    algos="seq omp omp-dynamic heavy heavy-dynamic"
    [ -x bin/phoenix_index ] && algos+=" phoenix"
    show "tests/run_tests.sh     # 13 hand-written cases + 5 random corpora; $algos; threads 1, 7, $THREADS"
    SEEDS=5 LARGE=0 THREADS="1 7 $THREADS" ALGOS="$algos" tests/run_tests.sh | tail -n 3
    pause
fi

step "3 · Full-size builds: $(basename "$CORPUS"), $(awk -v b="$(stat -c %s "$CORPUS")" 'BEGIN { printf "%.2f", b / 1e9 }') GB, $(group "$(wc -l <"$CORPUS")") documents"
say "D1, the best sequential builder: the baseline for every speedup"
build "sequential (D1)" 1 seq.bin bin/invidx --algo seq
pause
say "D2: per-thread dictionaries, dynamic chunks, splitter merge"
build "D2 omp-dynamic" "$THREADS" - bin/invidx --algo omp --schedule dynamic
if [ "$THREADS" -gt 8 ]; then
    say "D3 on the 8 P-cores only, then on all $THREADS threads (E-cores and hyperthreads join)"
    build "D3 heavy-dynamic" 8 - bin/invidx --algo heavy --schedule dynamic
fi
build "D3 heavy-dynamic" "$THREADS" d3.bin bin/invidx --algo heavy --schedule dynamic
if [ -x bin/phoenix_index ]; then
    say "Phoenix++: the starting paper's runtime, with our index as its application"
    build "Phoenix++" "$THREADS" phx.bin bin/phoenix_index
fi
pause

step "4 · Same answer from every builder"
show "bin/idxcheck seq.bin d3.bin"
bin/idxcheck "$OUT/seq.bin" "$OUT/d3.bin"
if [ -f "$OUT/phx.bin" ]; then
    show "bin/idxcheck seq.bin phx.bin"
    bin/idxcheck "$OUT/seq.bin" "$OUT/phx.bin"
fi
show "bin/idxcheck --validate d3.bin    # words strictly sorted, every list strictly increasing"
bin/idxcheck --validate "$OUT/d3.bin" | sed "s#$OUT/##"
echo
seq_secs=${ROWS[0]#*|*|}
seq_secs=${seq_secs%%|*}
printf '%s%-20s %8s %10s %12s   %-16s%s\n' "$B" builder threads "build (s)" speedup checksum "$R"
for row in "${ROWS[@]}"; do
    IFS='|' read -r label t secs gb sum <<<"$row"
    printf '%-20s %8s %10s %11s×   %-16s\n' "$label" "$t" "$secs" "$(awk -v a="$seq_secs" -v b="$secs" 'BEGIN { printf "%.1f", a / b }')" "$sum"
done
pause

step "5 · Using the index: look up words"
show "bin/idxcheck --lookup d3.bin $WORDS"
bin/idxcheck --lookup "$OUT/d3.bin" $WORDS | while IFS=$'\t' read -r word df ids; do
    if [ "$df" = 0 ]; then
        printf '  %s%-12s%s not in the index\n' "$Y" "$word" "$R"
    else
        printf '  %s%-12s%s in %s documents: %s\n' "$Y" "$word" "$R" "$(group "$df")" "$ids"
    fi
done
first=$(bin/idxcheck --lookup "$OUT/d3.bin" $WORDS | awk -F'\t' '$2 > 0 { split($3, a, " "); print $1, a[1]; exit }')
if [ -n "$first" ]; then
    read -r word doc <<<"$first"
    show "sed -n $((doc + 1))p $CORPUS | cut -c1-200      # document $doc, which contains \"$word\""
    sed -n "$((doc + 1))p" "$CORPUS" | cut -c1-200
fi
printf '\n%sDone.%s\n' "$G" "$R"

#!/usr/bin/env bash
# Reproduces the report's experiments. Run from inverted-index/ on AC power with
# nothing else running (not even the dataset download). See PLAN.md section 5.
#
#   bench/experiments.sh e1      # E1/E2 strong scaling + phases, 3 corpora, D2 and D3 (~45 min)
#   bench/experiments.sh sizes   # E1 at 25% and 50% of each corpus (~20 min)
#   bench/experiments.sh e3      # E3 Zipf skew sweep on synthetic corpora (D2 vs D3)
#   bench/experiments.sh e4      # E4 vocabulary and document-length sweeps (D2, D3, Phoenix++)
#   bench/experiments.sh e5      # E5 D3 ablations: theta, sample rate, memory budget
#   bench/experiments.sh e6      # E6 Phoenix++ reference vs D2/D3, 3 corpora (~35 min)
#   bench/experiments.sh all     # all of the above, then plots
#   bench/experiments.sh plots   # figures for every results/e*.jsonl
# Several at once: bench/experiments.sh sizes e4 e6 plots
#
# Inputs:
#   data/zipf_2M.txt   synthetic, generated here (deterministic)
#   data/gutenberg.txt tools/fetch_corpus.py gutenberg --shards 4 --out data/gutenberg.txt
#   data/wiki.txt      tools/fetch_corpus.py wiki --out data/wiki.txt
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
THREADS=${THREADS:-1,2,4,8,12,16,24}
REPS=${REPS:-5}

synthetic() {  # synthetic <name> <zipf s>
    [ -s "data/$1.txt" ] || bin/gen_zipf --docs 2000000 --vocab 1000000 --zipf "$2" --len 100 --seed 7 \
        --out "data/$1.txt"
}

gen() {  # gen <name> <gen_zipf args...>: a synthetic corpus, generated once (deterministic)
    local name=$1
    shift
    [ -s "data/$name.txt" ] || bin/gen_zipf "$@" --seed 7 --out "data/$name.txt"
}

phoenix() { [ -x bin/phoenix_index ] || make phoenix; }

e1() {  # strong scaling: D1, D2 (static/dynamic), D3 (static/dynamic)
    synthetic zipf_2M 1.0
    for corpus in zipf_2M gutenberg wiki; do
        $PY bench/run.py --name "e1_$corpus" --corpus "data/$corpus.txt" \
            --variants seq,omp,omp-dynamic,heavy,heavy-dynamic --threads "$THREADS" --reps "$REPS"
    done
}

sizes() {  # E1 on 25% and 50% of each corpus: every 4th / 2nd document, so only the size changes
    synthetic zipf_2M 1.0
    for corpus in zipf_2M gutenberg wiki; do
        for k in 4 2; do
            local pct=$((100 / k)) f="data/${corpus}_$((100 / k))pct.txt"
            [ -s "$f" ] || awk -v k="$k" 'NR % k == 1' "data/$corpus.txt" >"$f"
            $PY bench/run.py --name "e1s_${corpus}_$pct" --corpus "$f" \
                --variants seq,omp-dynamic,heavy-dynamic --threads "$THREADS" --reps "$REPS"
        done
    done
}

e3() {  # skew sweep: same size and vocabulary, Zipf exponent 0 (uniform) .. 1.4
    for s in 0 0.6 0.8 1.0 1.2 1.4; do
        synthetic "zipf_s$s" "$s"
        $PY bench/run.py --name "e3_zipf_s$s" --corpus "data/zipf_s$s.txt" \
            --variants seq,omp-dynamic,heavy-dynamic --threads 8,24 --reps "$REPS"
    done
}

e4() {  # Zipf s = 1.0, ~200M tokens each; vocabulary 10^4..10^7, then mean doc length 10..10^4 tokens
    phoenix
    synthetic zipf_2M 1.0  # = vocabulary 10^6, length 100
    for v in 10000 100000 1000000 10000000; do
        local name="zipf_v$v"
        if [ "$v" = 1000000 ]; then name=zipf_2M; else gen "$name" --docs 2000000 --vocab "$v" --zipf 1.0 --len 100; fi
        $PY bench/run.py --name "e4_vocab_$v" --corpus "data/$name.txt" \
            --variants seq,omp-dynamic,heavy-dynamic,phoenix --threads 8,24 --reps "$REPS"
    done
    for len in 10 100 1000 10000; do
        local name="zipf_len$len"
        if [ "$len" = 100 ]; then name=zipf_2M; else gen "$name" --docs $((200000000 / len)) --vocab 1000000 --zipf 1.0 --len "$len"; fi
        $PY bench/run.py --name "e4_len_$len" --corpus "data/$name.txt" \
            --variants seq,omp-dynamic,heavy-dynamic,phoenix --threads 8,24 --reps "$REPS"
    done
}

e5() {  # D3 ablations on the short-document corpus (dynamic schedule, where D3 targets the merge)
    local base="--algo heavy --schedule dynamic"
    $PY bench/run.py --name e5_theta --corpus data/wiki.txt --variants omp-dynamic \
        --variant "theta 1/4=$base --theta 0.25" --variant "theta 1/8=$base --theta 0.125" \
        --variant "theta 1/32=$base --theta 0.03125" --variant "theta 1/128=$base --theta 0.0078125" \
        --variant "theta 1/512=$base --theta 0.001953125" --threads 8,24 --reps "$REPS"
    $PY bench/run.py --name e5_sample --corpus data/wiki.txt --variants omp-dynamic \
        --variant "sample 0.1%=$base --sample 0.001" --variant "sample 1%=$base --sample 0.01" \
        --variant "sample 10%=$base --sample 0.1" --threads 8,24 --reps "$REPS"
    # theta 1/128 wants ~430 MB of bitmaps on wiki, so the budget is what binds
    $PY bench/run.py --name e5_budget --corpus data/wiki.txt --variants omp-dynamic \
        --variant "budget 16MB=$base --theta 0.0078125 --heavy-mem-mb 16" \
        --variant "budget 64MB=$base --theta 0.0078125 --heavy-mem-mb 64" \
        --variant "budget 256MB=$base --theta 0.0078125 --heavy-mem-mb 256" \
        --variant "budget 1GB=$base --theta 0.0078125 --heavy-mem-mb 1024" --threads 8,24 --reps "$REPS"
}

e6() {  # Phoenix++ (authors' MapReduce runtime, our index as its application) vs D2 and D3
    phoenix
    synthetic zipf_2M 1.0
    for corpus in zipf_2M gutenberg wiki; do
        $PY bench/run.py --name "e6_$corpus" --corpus "data/$corpus.txt" \
            --variants seq,omp-dynamic,heavy-dynamic,phoenix --threads "$THREADS" --reps "$REPS"
    done
}

[ $# -gt 0 ] || { sed -n '2,20p' "$0"; exit 2; }
for cmd in "$@"; do
    case "$cmd" in
        e1 | sizes | e3 | e4 | e5 | e6) "$cmd" ;;
        all) e1; sizes; e3; e4; e5; e6; $PY bench/plot.py results/e*.jsonl ;;
        plots) $PY bench/plot.py results/e*.jsonl ;;
        *)
            sed -n '2,20p' "$0"
            exit 2
            ;;
    esac
done

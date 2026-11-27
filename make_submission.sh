#!/usr/bin/env bash
# Assembles the course submission from the last commit, with only what the brief asks for:
#   submission/CS_F422_Project23/
#     README.md          how to build, run, demo and reproduce; dataset link (from ./README.md)
#     report.pdf         the report
#     presentation.pdf   the slides (or presentation.pptx if there is no PDF)
#     inverted-index/    the code, tests, benchmark harness and raw results; empty data/ for the datasets
#   submission/CS_F422_Project23.zip     the same folder, zipped for upload
#
#   ./make_submission.sh
#
# Left out: PLAN.md, viva-prep.md, the project catalogue, the report and slide sources, .gitignore
# files, and everything git ignores (binaries, datasets, figures, Phoenix++ checkout).
# The presentation is exported by hand from the slide deck (Share > Export > PDF) and saved as
# presentation/slides.pdf. The code and README come from the last commit (REV=<commit> to pick
# another), so commit first. Run again after any change; the folder and zip are rebuilt from scratch.
set -euo pipefail
cd "$(dirname "$0")"
NAME=CS_F422_Project23
OUT="submission/$NAME"
REV=${REV:-HEAD}
readme=$(git show "$REV:README.md")

slides=""
for f in presentation/slides.pdf presentation/slides.pptx; do [ -f "$f" ] && { slides=$f; break; }; done

warn=()
grep -q '\[Student A name' report/report.html && warn+=("report.pdf still has the author placeholders: edit report/report.html, run report/build.sh, commit")
grep -q 'Describe AI assistance' report/report.html && warn+=("report.pdf still has the AI-use / contribution placeholder: same steps")
[ -z "$(git status --porcelain --untracked-files=no)" ] || warn+=("uncommitted changes: the package holds the last commit, not your working copy")
[ report/report.pdf -nt report/report.html ] || warn+=("report/report.pdf is older than report/report.html: run report/build.sh")
if [ -z "$slides" ]; then
    warn+=("no presentation file: export the deck to presentation/slides.pdf")
elif [ "${slides##*.}" = pdf ] && command -v pdfinfo >/dev/null; then
    pages=$(pdfinfo "$slides" | awk '/^Pages:/ { print $2 }')
    [ "$pages" -le 10 ] || warn+=("$slides has $pages slides; the brief asks for 8-10 (for a 5-minute talk)")
fi
grep -q 'PASTE-GOOGLE-DRIVE-LINK' <<<"$readme" && warn+=("README.md: paste the Google Drive link of the datasets (section Datasets)")
drive=$(grep -o 'https://drive\.google\.com/[^)]*' <<<"$readme" | head -n 1 || true)
[ -n "$drive" ] || warn+=("README.md (committed) has no Google Drive link")

rm -rf "$OUT" "submission/$NAME.zip"
mkdir -p "$OUT"
git archive --format=tar "$REV" inverted-index | tar -x -C "$OUT"
find "$OUT" -name .gitignore -delete
mkdir -p "$OUT/inverted-index/data"
cp report/report.pdf "$OUT/report.pdf"
[ -z "$slides" ] || cp "$slides" "$OUT/presentation.${slides##*.}"
commit=$(git rev-parse --short "$REV")
deck=$([ -n "$slides" ] && printf 'presentation.%s' "${slides##*.}" || printf 'presentation.pdf')

{
    cat <<EOF
# Parallel inverted index construction (CS F422, Project 23)

Word → document posting lists built from a pretokenized corpus, in C++20 with OpenMP. Authors: see \`report.pdf\`. Code snapshot: commit \`$commit\`.

\`\`\`
README.md          this file: how to build, run, demo and reproduce
report.pdf         the report (7 pages)
$(printf '%-18s' "$deck") the slides for the 5-minute presentation
inverted-index/    the code, tests, benchmark harness, raw results and demo.sh
inverted-index/data/   empty: put the datasets here (see Datasets)
\`\`\`

| Deliverable | Where |
|---|---|
| **D1:** sequential index builder + posting-list equality tests | \`inverted-index/src/build_seq.cpp\`; \`inverted-index/tests/\` checks every builder byte for byte against the oracle \`src/build_naive.cpp\` |
| **D2:** OpenMP local dictionaries with sorted merging | \`inverted-index/src/build_omp.cpp\` (\`--algo omp\`) |
| **D3:** frequent-term handling under a memory budget | \`inverted-index/src/build_omp.cpp\` + \`include/local_index.hpp\` (\`--algo heavy\`, \`--heavy-mem-mb\`) |
| **Baseline:** Phoenix | \`inverted-index/baselines/phoenix_index.cpp\`, our index on Phoenix++ (\`make phoenix\`) |
| **Three input families, several sizes** | synthetic Zipf, Project Gutenberg, Wikipedia: [Datasets](#datasets) |
| **Experiments: repeated, pinned timings and raw results** | \`inverted-index/bench/\`, \`inverted-index/results/\`: [Benchmarks](#benchmarks) |
| **Report** (6–8 pages) | \`report.pdf\` |
| **Presentation** | \`$deck\` |
| **Demo** (10–15 minutes) | \`inverted-index/demo.sh\`: [Demo](#demo) |
| **Reproducibility instructions, dependency versions** | this file |
| **Attribution** | [Attribution](#attribution) and the report |

## Quick start

\`\`\`sh
cd inverted-index
make && make test     # builds bin/; 10,462 correctness checks across all builders and thread counts
./demo.sh             # the live demo (generates a synthetic corpus if data/wiki.txt is missing)
\`\`\`

**Datasets:** [Google Drive folder]($drive). Download every file into \`inverted-index/data/\` and run \`gunzip *.gz\` there; details and checksums are in [Datasets](#datasets).

To rerun every experiment in the report (about 4.5 hours, on AC power with nothing else running):

\`\`\`sh
uv venv .venv && uv pip install --python .venv/bin/python pyarrow numpy pandas matplotlib
bench/experiments.sh all   # -> results/*.jsonl and results/figures/
\`\`\`

EOF
    # The rest of ./README.md, without its repository-only sections.
    awk '/^## / { started = 1; skip = /^## (Repository layout|Report and slide sources|Status)$/ }
         started && !skip' <<<"$readme"
} >"$OUT/README.md"

(cd submission && bsdtar -a -cf "$NAME.zip" "$NAME")

echo "built $OUT and submission/$NAME.zip ($(du -h "submission/$NAME.zip" | cut -f1)) from commit $commit"
find "$OUT" -maxdepth 2 -mindepth 1 -not -path "*/inverted-index/*/*" -printf '  %P\n' | sort
if [ ${#warn[@]} -gt 0 ]; then
    echo "NOT READY TO SUBMIT YET:"
    printf '  - %s\n' "${warn[@]}"
    exit 1
fi
echo "ready to submit"

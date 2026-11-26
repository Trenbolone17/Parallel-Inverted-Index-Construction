#!/usr/bin/env bash
# Builds report.pdf: figures from ../inverted-index/results, then report.html printed to PDF
# by a headless Chromium-based browser (Brave, Chromium or Chrome; override with BROWSER=...).
set -euo pipefail
cd "$(dirname "$0")"
../inverted-index/.venv/bin/python make_figures.py
BROWSER=${BROWSER:-$(command -v brave || command -v chromium || command -v google-chrome || true)}
[ -n "$BROWSER" ] || { echo "no Chromium-based browser found; set BROWSER=/path/to/chrome" >&2; exit 1; }
# The browser profile must live at a path without spaces (headless Chromium hangs otherwise),
# and a brand-new profile can hang on its first launch, hence one retry.
PROFILE=${XDG_CACHE_HOME:-$HOME/.cache}/report-print-profile
for attempt in 1 2; do
    if timeout 90 "$BROWSER" --headless=new --disable-gpu --no-first-run --no-pdf-header-footer \
        --user-data-dir="$PROFILE" --print-to-pdf="$PWD/report.pdf" "file://$PWD/report.html" 2>/dev/null; then
        break
    fi
    echo "browser attempt $attempt timed out" >&2
done
command -v pdfinfo >/dev/null && pdfinfo report.pdf | grep Pages

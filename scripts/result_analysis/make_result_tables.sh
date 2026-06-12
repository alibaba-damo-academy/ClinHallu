#!/bin/bash
# ============================================================
# Build markdown comparison tables from results summaries.
#
# Usage:
#   bash scripts/result_analysis/make_result_tables.sh [results_root] [dataset] [extra args...]
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

RESULTS_ROOT="${1:-/home/ysc/v8_medhalu/results}"
DATASET="${2:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

EXTRA_ARGS=()
if [ "$#" -ge 3 ]; then
    EXTRA_ARGS=("${@:3}")
fi

CMD=(
    "$PYTHON_BIN"
    src/result_analysis/05_make_result_tables.py
    --results-root "$RESULTS_ROOT"
)

if [ -n "$DATASET" ]; then
    CMD+=(--dataset "$DATASET")
fi

"${CMD[@]}" "${EXTRA_ARGS[@]}"

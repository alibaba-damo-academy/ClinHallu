#!/bin/bash
# ============================================================
# Run model CoT + replace generation for all *-test datasets.
#
# Usage:
#   bash scripts/model_evaluation/run_all_test_cot_replace.sh <config.yaml> <gold_model> <test_model> [replace_mode]
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

CONFIG="${1:-configs/config.yaml}"
GOLD_MODEL="${2:-}"
TEST_MODEL="${3:-}"
REPLACE_MODE="${4:-replace_all}"

if [ -z "$GOLD_MODEL" ] || [ -z "$TEST_MODEL" ]; then
    echo "Usage: bash scripts/model_evaluation/run_all_test_cot_replace.sh <config.yaml> <gold_model> <test_model> [replace_mode]"
    exit 1
fi

mapfile -t TEST_DATASETS < <(
    awk '
        /^datasets:/ { in_datasets = 1; next }
        in_datasets && /^[^[:space:]]/ { in_datasets = 0 }
        in_datasets && /^  [^[:space:]][^:]*:/ {
            key = $1
            sub(/:$/, "", key)
            if (key ~ /-test$/) {
                print key
            }
        }
    ' "$CONFIG"
)

if [ "${#TEST_DATASETS[@]}" -eq 0 ]; then
    echo "No *-test datasets found in $CONFIG"
    exit 1
fi

echo "=========================================="
echo " Batch Run: All Test CoT + Replace"
echo " Config:       $CONFIG"
echo " Gold model:   $GOLD_MODEL"
echo " Test model:   $TEST_MODEL"
echo " Replace mode: $REPLACE_MODE"
echo " Datasets:     ${TEST_DATASETS[*]}"
echo "=========================================="

for DATASET_KEY in "${TEST_DATASETS[@]}"; do
    echo ""
    echo "========== Dataset: $DATASET_KEY =========="
    echo "[1/2] Generate model CoT"
    bash scripts/model_evaluation/01_generate_cot.sh "$CONFIG" model "$DATASET_KEY" "$GOLD_MODEL" "$TEST_MODEL"

    echo ""
    echo "[2/2] Generate replace results"
    bash scripts/model_evaluation/03_replace_experiment.sh "$CONFIG" generate "$REPLACE_MODE" "$DATASET_KEY" "$GOLD_MODEL" "$TEST_MODEL"
done

echo ""
echo "All test datasets completed successfully!"

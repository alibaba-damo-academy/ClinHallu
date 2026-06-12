#!/bin/bash
# ============================================================
# Run step-level hallucination judging for all *-test datasets.
#
# Usage:
#   bash scripts/model_evaluation/run_all_test_step_hallucination.sh <config.yaml> <gold_model> <test_model> [all|model|replace_all|replace_vr|replace_kr|replace_vr_kr]
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

CONFIG="${1:-configs/config.yaml}"
GOLD_MODEL="${2:-}"
TEST_MODEL="${3:-}"
TARGET_MODE="${4:-all}"

if [ -z "$GOLD_MODEL" ] || [ -z "$TEST_MODEL" ]; then
    echo "Usage: bash scripts/model_evaluation/run_all_test_step_hallucination.sh <config.yaml> <gold_model> <test_model> [all|model|replace_all|replace_vr|replace_kr|replace_vr_kr]"
    exit 1
fi

case "$TARGET_MODE" in
    all|model|replace_all|replace_vr|replace_kr|replace_vr_kr)
        ;;
    *)
        echo "Invalid target mode: $TARGET_MODE"
        echo "Expected one of: all, model, replace_all, replace_vr, replace_kr, replace_vr_kr"
        exit 1
        ;;
esac

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

echo "=========================================================="
echo " Batch Run: All Test Step Hallucination Judge"
echo " Config:      $CONFIG"
echo " Gold model:  $GOLD_MODEL"
echo " Test model:  $TEST_MODEL"
echo " Target mode: $TARGET_MODE"
echo " Datasets:    ${TEST_DATASETS[*]}"
echo "=========================================================="

for DATASET_KEY in "${TEST_DATASETS[@]}"; do
    echo ""
    echo "========== Dataset: $DATASET_KEY =========="

    if [ "$TARGET_MODE" = "all" ]; then
        # echo "[1/2] Judge model step hallucination"
        # bash scripts/model_evaluation/04_judge_step_hallucination.sh \
        #     "$CONFIG" model "$DATASET_KEY" "$GOLD_MODEL" "$TEST_MODEL"

        echo ""
        echo "[2/2] Judge replace step hallucination"
        bash scripts/model_evaluation/04_judge_step_hallucination.sh \
            "$CONFIG" replace_all "$DATASET_KEY" "$GOLD_MODEL" "$TEST_MODEL"
    else
        echo "[1/1] Judge target: $TARGET_MODE"
        bash scripts/model_evaluation/04_judge_step_hallucination.sh \
            "$CONFIG" "$TARGET_MODE" "$DATASET_KEY" "$GOLD_MODEL" "$TEST_MODEL"
    fi
done

echo ""
echo "All test dataset step hallucination judging completed successfully!"

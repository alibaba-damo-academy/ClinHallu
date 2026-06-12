#!/bin/bash
# ============================================================
# Batch local model generation without judge
#
# Runs:
#   1. Step 1(model): model_cot generation
#   2. Step 4(generate): replace_cot generation
#
# Default scope:
#   gold model: qwen3.5-plus
#   datasets:   vqa-rad-test, pathvqa-test, medframeqa-test, medxpert-test
#   models:     qwen25vl-local, qwen3-vl-8b-instruct-local, lingshu-7b-local
#
# Usage:
#   bash scripts/model_evaluation/run_local_generate_batch.sh
#   bash scripts/model_evaluation/run_local_generate_batch.sh configs/config.yaml
#   bash scripts/model_evaluation/run_local_generate_batch.sh configs/config.yaml qwen3.5-plus
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

CONFIG="${1:-configs/config.yaml}"
GOLD_MODEL="${2:-qwen3.5-plus}"

DATASETS=(
    "vqa-rad-test"
    "pathvqa-test"
    "medframeqa-test"
    "medxpert-test"
)

TEST_MODELS=(
    "qwen25vl-local"
    "qwen3-vl-8b-instruct-local"
    "lingshu-7b-local"
)

TOTAL=$(( ${#DATASETS[@]} * ${#TEST_MODELS[@]} ))
INDEX=0

echo "============================================================"
echo " Batch local generation without judge"
echo " Config:     $CONFIG"
echo " Gold model: $GOLD_MODEL"
echo " Datasets:   ${DATASETS[*]}"
echo " Models:     ${TEST_MODELS[*]}"
echo " Total runs: $TOTAL"
echo "============================================================"

for DATASET_KEY in "${DATASETS[@]}"; do
    for TEST_MODEL in "${TEST_MODELS[@]}"; do
        INDEX=$((INDEX + 1))
        echo ""
        echo "############################################################"
        echo " Run [$INDEX/$TOTAL]"
        echo " Dataset: $DATASET_KEY"
        echo " Gold:    $GOLD_MODEL"
        echo " Test:    $TEST_MODEL"
        echo "############################################################"

        echo ""
        echo "========== Step 1(model): Generate Model CoT =========="
        bash scripts/model_evaluation/01_generate_cot.sh \
            "$CONFIG" \
            model \
            "$DATASET_KEY" \
            "$GOLD_MODEL" \
            "$TEST_MODEL"
        echo "========== Step 1(model) Done =========="

        echo ""
        echo "========== Step 4(generate): Generate Replace CoT =========="
        bash scripts/model_evaluation/03_replace_experiment.sh \
            "$CONFIG" \
            generate \
            replace_all \
            "$DATASET_KEY" \
            "$GOLD_MODEL" \
            "$TEST_MODEL"
        echo "========== Step 4(generate) Done =========="
    done
done

echo ""
echo "All local generation runs completed successfully!"

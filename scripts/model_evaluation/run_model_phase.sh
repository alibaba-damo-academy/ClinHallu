#!/bin/bash
# ============================================================
# Model Phase Only
#
# Runs Step 1(model) / Step 3(model judge) / Step 4(replace).
#
# Usage:
#   bash scripts/model_evaluation/run_model_phase.sh <config.yaml> <dataset_key> <gold_model> <test_model>
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

CONFIG="${1:-configs/config.yaml}"
DATASET_KEY="${2:-}"
GOLD_MODEL="${3:-}"
TEST_MODEL="${4:-}"

if [ -z "$DATASET_KEY" ] || [ -z "$GOLD_MODEL" ] || [ -z "$TEST_MODEL" ]; then
    echo "Usage: bash scripts/model_evaluation/run_model_phase.sh <config.yaml> <dataset_key> <gold_model> <test_model>"
    exit 1
fi

echo "========== Step 1(model): Generate Model CoT =========="
bash scripts/model_evaluation/01_generate_cot.sh "$CONFIG" model "$DATASET_KEY" "$GOLD_MODEL" "$TEST_MODEL"
echo "========== Step 1(model) Done =========="

echo ""
echo "========== Step 3(model): Judge Final Answers =========="
bash scripts/model_evaluation/02_judge_answers.sh "$CONFIG" model "$DATASET_KEY" "$GOLD_MODEL" "$TEST_MODEL"
echo "========== Step 3(model) Done =========="

echo ""
echo "========== Step 4(replace): Replace Experiment =========="
bash scripts/model_evaluation/03_replace_experiment.sh "$CONFIG" all replace_all "$DATASET_KEY" "$GOLD_MODEL" "$TEST_MODEL"
echo "========== Step 4(replace) Done =========="

echo ""
echo "Model phase completed successfully!"

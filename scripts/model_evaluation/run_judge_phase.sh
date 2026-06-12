#!/bin/bash
# ============================================================
# Judge Phase Only
#
# Runs Step 3(model judge) and Step 4(replace judge only).
#
# Usage:
#   bash scripts/model_evaluation/run_judge_phase.sh <config.yaml> <dataset_key> <gold_model> <test_model> [replace_mode]
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

CONFIG="${1:-configs/config.yaml}"
DATASET_KEY="${2:-}"
GOLD_MODEL="${3:-}"
TEST_MODEL="${4:-}"
REPLACE_MODE="${5:-replace_all}"

if [ -z "$DATASET_KEY" ] || [ -z "$GOLD_MODEL" ] || [ -z "$TEST_MODEL" ]; then
    echo "Usage: bash scripts/model_evaluation/run_judge_phase.sh <config.yaml> <dataset_key> <gold_model> <test_model> [replace_mode]"
    exit 1
fi

echo "========== Step 3(model): Judge Final Answers =========="
bash scripts/model_evaluation/02_judge_answers.sh "$CONFIG" model "$DATASET_KEY" "$GOLD_MODEL" "$TEST_MODEL"
echo "========== Step 3(model) Done =========="

echo ""
echo "========== Step 4(replace judge): Judge Replace Results =========="
bash scripts/model_evaluation/03_replace_experiment.sh "$CONFIG" judge "$REPLACE_MODE" "$DATASET_KEY" "$GOLD_MODEL" "$TEST_MODEL"
echo "========== Step 4(replace judge) Done =========="

echo ""
echo "Judge phase completed successfully!"

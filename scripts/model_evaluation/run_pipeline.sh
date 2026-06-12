#!/bin/bash
# ============================================================
# Full Pipeline: Medical VQA Hallucination Analysis
#
# Runs all steps in order. Each step supports checkpoint/resume,
# so it's safe to re-run after interruption.
#
# Usage:
#   bash scripts/model_evaluation/run_pipeline.sh <config.yaml> <dataset_key> <gold_model> <test_model>
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

CONFIG="${1:-configs/config.yaml}"
DATASET_KEY="${2:-}"
GOLD_MODEL="${3:-}"
TEST_MODEL="${4:-}"

if [ -z "$DATASET_KEY" ] || [ -z "$GOLD_MODEL" ] || [ -z "$TEST_MODEL" ]; then
    echo "Usage: bash scripts/model_evaluation/run_pipeline.sh <config.yaml> <dataset_key> <gold_model> <test_model>"
    exit 1
fi

bash scripts/model_evaluation/run_gold_phase.sh "$CONFIG" "$DATASET_KEY" "$GOLD_MODEL" "$TEST_MODEL"
echo ""
bash scripts/model_evaluation/run_model_phase.sh "$CONFIG" "$DATASET_KEY" "$GOLD_MODEL" "$TEST_MODEL"

echo ""
echo "All steps completed successfully!"

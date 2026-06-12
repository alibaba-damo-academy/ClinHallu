#!/bin/bash
# ============================================================
# Gold Phase Only: Medical VQA Hallucination Analysis
#
# Runs Step 0/1/2 only:
#   0. Prepare image reports
#   1. Generate Gold CoT candidates
#   2. Filter Gold CoT with LLM Judge
#
# Usage:
#   bash scripts/model_evaluation/run_gold_phase.sh <config.yaml> <dataset_key> <gold_model> [test_model]
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

CONFIG="${1:-configs/config.yaml}"
DATASET_KEY="${2:-}"
GOLD_MODEL="${3:-}"
TEST_MODEL="${4:-}"

if [ -z "$DATASET_KEY" ] || [ -z "$GOLD_MODEL" ]; then
    echo "Usage: bash scripts/model_evaluation/run_gold_phase.sh <config.yaml> <dataset_key> <gold_model> [test_model]"
    exit 1
fi

echo "========== Step 0: Prepare Image Reports (optional) =========="
bash scripts/data_preparation/01_prepare_reports.sh "$CONFIG" "$DATASET_KEY" "$GOLD_MODEL" "$TEST_MODEL"
echo "========== Step 0 Done =========="

echo ""
echo "========== Step 1: Generate Gold CoT Candidates =========="
bash scripts/model_evaluation/01_generate_cot.sh "$CONFIG" gold "$DATASET_KEY" "$GOLD_MODEL" "$TEST_MODEL"
echo "========== Step 1 Done =========="

echo ""
echo "========== Step 2: Filter Gold CoT =========="
bash scripts/data_preparation/02_filter_gold.sh "$CONFIG" "$DATASET_KEY" "$GOLD_MODEL" "$TEST_MODEL"
echo "========== Step 2 Done =========="

echo ""
echo "Gold phase completed successfully!"

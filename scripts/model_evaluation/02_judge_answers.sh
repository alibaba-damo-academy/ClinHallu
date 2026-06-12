#!/bin/bash
# ============================================================
# Usage:
#   bash scripts/model_evaluation/02_judge_answers.sh [config.yaml] [model|replace_all|replace_vr|replace_kr|replace_vr_kr]
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

CONFIG="${1:-configs/config.yaml}"
TARGET="${2:-model}"
DATASET_KEY="${3:-}"
GOLD_MODEL="${4:-}"
TEST_MODEL="${5:-}"

EXTRA_ARGS=()
if [ -n "$DATASET_KEY" ]; then
    EXTRA_ARGS+=(--dataset-key "$DATASET_KEY")
fi
if [ -n "$GOLD_MODEL" ]; then
    EXTRA_ARGS+=(--gold-model "$GOLD_MODEL")
fi
if [ -n "$TEST_MODEL" ]; then
    EXTRA_ARGS+=(--test-model "$TEST_MODEL")
fi

echo "=========================================="
echo " Step 3: LLM Judge Final-Answer Evaluation"
echo " Config: $CONFIG"
echo " Target: $TARGET"
if [ -n "$DATASET_KEY" ]; then echo " Dataset: $DATASET_KEY"; fi
if [ -n "$GOLD_MODEL" ]; then echo " Gold:    $GOLD_MODEL"; fi
if [ -n "$TEST_MODEL" ]; then echo " Test:    $TEST_MODEL"; fi
echo "=========================================="

python src/model_evaluation/02_judge_answers.py --config "$CONFIG" --target "$TARGET" "${EXTRA_ARGS[@]}"

echo ""
echo "Done."

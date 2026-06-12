#!/bin/bash
# ============================================================
# Usage:
#   bash scripts/model_evaluation/01_generate_cot.sh [config.yaml] [gold|model|replace_all|replace_vr|replace_kr|replace_vr_kr]
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

CONFIG="${1:-configs/config.yaml}"
MODE="${2:-gold}"
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
echo " Step 1: Generate CoT"
echo " Config: $CONFIG"
echo " Mode:   $MODE"
if [ -n "$DATASET_KEY" ]; then echo " Dataset: $DATASET_KEY"; fi
if [ -n "$GOLD_MODEL" ]; then echo " Gold:    $GOLD_MODEL"; fi
if [ -n "$TEST_MODEL" ]; then echo " Test:    $TEST_MODEL"; fi
echo "=========================================="

python src/model_evaluation/01_generate_cot.py --config "$CONFIG" --mode "$MODE" "${EXTRA_ARGS[@]}"

echo ""
echo "Done."

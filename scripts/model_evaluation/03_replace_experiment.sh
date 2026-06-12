#!/bin/bash
# ============================================================
# Usage:
#   bash scripts/model_evaluation/03_replace_experiment.sh [config.yaml] [all|generate|judge] [replace_all|replace_vr|replace_kr|replace_vr_kr]
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

CONFIG="${1:-configs/config.yaml}"
STAGE="${2:-all}"
MODE="${3:-replace_all}"
DATASET_KEY="${4:-}"
GOLD_MODEL="${5:-}"
TEST_MODEL="${6:-}"

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
echo " Step 4: Replacement Experiment"
echo " Config: $CONFIG"
echo " Stage:  $STAGE"
echo " Mode:   $MODE"
if [ -n "$DATASET_KEY" ]; then echo " Dataset: $DATASET_KEY"; fi
if [ -n "$GOLD_MODEL" ]; then echo " Gold:    $GOLD_MODEL"; fi
if [ -n "$TEST_MODEL" ]; then echo " Test:    $TEST_MODEL"; fi
echo "=========================================="

python src/model_evaluation/03_replace_experiment.py --config "$CONFIG" --stage "$STAGE" --mode "$MODE" "${EXTRA_ARGS[@]}"

echo ""
echo "Done. Output: results/{dataset}/eval/{model}/replace/"

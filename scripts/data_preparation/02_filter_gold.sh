set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

CONFIG="${1:-configs/config.yaml}"
DATASET_KEY="${2:-}"
GOLD_MODEL="${3:-}"
TEST_MODEL="${4:-}"

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
echo " Step 2: Filter Gold CoT (LLM Judge)"
echo " Config: $CONFIG"
if [ -n "$DATASET_KEY" ]; then echo " Dataset: $DATASET_KEY"; fi
if [ -n "$GOLD_MODEL" ]; then echo " Gold:    $GOLD_MODEL"; fi
if [ -n "$TEST_MODEL" ]; then echo " Test:    $TEST_MODEL"; fi
echo "=========================================="

python src/data_preparation/02_filter_gold.py --config "$CONFIG" "${EXTRA_ARGS[@]}"

echo ""
echo "Done. Output: results/{dataset}/gold/{model}/gold_cot.jsonl"

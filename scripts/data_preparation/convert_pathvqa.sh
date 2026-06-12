#!/bin/bash
# ============================================================
# Data Prep: Convert PathVQA to medhalu pipeline format
#
# PathVQA is a pathology visual question answering dataset.
# This script downloads from HuggingFace, extracts images,
# and converts to the unified JSONL + images format.
#
# Usage:
#   bash scripts/data_preparation/convert_pathvqa.sh                           # Download & convert (test split)
#   bash scripts/data_preparation/convert_pathvqa.sh --split test validation   # Multiple splits
#   bash scripts/data_preparation/convert_pathvqa.sh --split all               # All splits
#   bash scripts/data_preparation/convert_pathvqa.sh --from-parquet data/path-vqa/input/data  # From local Parquet
#
# Output:
#   data/pathvqa/input/converted.jsonl
#   data/pathvqa/input/images/
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=========================================="
echo " Data Prep: Convert PathVQA"
echo "=========================================="

python src/data_preparation/convert_pathvqa.py "$@"

echo ""
echo "Done. To run the pipeline, update configs/config.yaml:"
echo '  dataset: "pathvqa"'
echo '  input_file: "data/pathvqa/input/test/converted.jsonl"'
echo '  image_dir: "data/pathvqa/input/test/images"'
echo '  split: "test"'

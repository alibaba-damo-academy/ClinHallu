#!/bin/bash
# ============================================================
# Data Prep: Convert VQA-RAD to medhalu pipeline format
#
# VQA-RAD is a radiology visual question answering dataset.
# This script downloads from HuggingFace, extracts images,
# and converts to the unified JSONL + images format.
#
# Usage:
#   bash scripts/data_preparation/convert_vqarad.sh                           # Download & convert (test split)
#   bash scripts/data_preparation/convert_vqarad.sh --split test train        # Multiple splits
#   bash scripts/data_preparation/convert_vqarad.sh --split all               # All splits
#   bash scripts/data_preparation/convert_vqarad.sh --from-parquet data/vqa-rad/input/data  # From local Parquet
#
# Output:
#   data/vqa-rad/input/converted.jsonl
#   data/vqa-rad/input/images/
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=========================================="
echo " Data Prep: Convert VQA-RAD"
echo "=========================================="

python src/data_preparation/convert_vqarad.py "$@"

echo ""
echo "Done. To run the pipeline, update configs/config.yaml:"
echo '  dataset: "vqa-rad"'
echo '  input_file: "data/vqa-rad/input/test/converted.jsonl"'
echo '  image_dir: "data/vqa-rad/input/test/images"'
echo '  split: "test"'

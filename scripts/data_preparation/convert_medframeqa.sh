#!/bin/bash
# ============================================================
# Data Prep: Convert MedFrameQA to medhalu pipeline format
#
# MedFrameQA is a multi-image medical VQA benchmark (2,851 samples).
# This script downloads from HuggingFace and converts to pipeline format.
#
# Usage:
#   bash scripts/data_preparation/convert_medframeqa.sh                  # Download & convert
#   bash scripts/data_preparation/convert_medframeqa.sh --download-timeout 1800 --download-attempts 8
#   bash scripts/data_preparation/convert_medframeqa.sh --input file.json  # From local JSON
#   bash scripts/data_preparation/convert_medframeqa.sh --input dir/ --from-parquet  # From Parquet
#
# Output:
#   data/medframeqa/input/converted.jsonl
#   data/medframeqa/input/images/
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=========================================="
echo " Data Prep: Convert MedFrameQA"
echo "=========================================="

python src/data_preparation/convert_medframeqa.py "$@"

echo ""
echo "Done. To run the pipeline, update configs/config.yaml:"
echo '  dataset: "medframeqa"'
echo '  input_file: "data/medframeqa/input/test/converted.jsonl"'
echo '  image_dir: "data/medframeqa/input/test/images"'
echo '  split: "test"'

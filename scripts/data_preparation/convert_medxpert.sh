#!/bin/bash
# ============================================================
# Data Prep: Convert MedXpertQA (MM) to medhalu pipeline format
#
# MedXpertQA is a medical expert-level multi-modal QA benchmark.
# This script downloads from HuggingFace and converts to pipeline format.
#
# Usage:
#   bash scripts/data_preparation/convert_medxpert.sh                     # Download & convert
#   bash scripts/data_preparation/convert_medxpert.sh --input file.jsonl  # From local JSONL
#
# Output:
#   data/medxpert/input/converted.jsonl
#   data/medxpert/input/images/
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=========================================="
echo " Data Prep: Convert MedXpertQA (MM)"
echo "=========================================="

python src/data_preparation/convert_medxpert.py "$@"

echo ""
echo "Done. To run the pipeline, update configs/config.yaml:"
echo '  dataset: "medxpert"'
echo '  input_file: "data/medxpert/input/test/converted.jsonl"'
echo '  image_dir: "data/medxpert/input/test/images"'
echo '  split: "test"'

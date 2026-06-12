#!/bin/bash
# ============================================================
# Model Download Runner
#
# Downloads model caches according to configs/config.yaml.
#
# Usage:
#   bash scripts/model_download/run_model_downloads.sh [config.yaml]
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

CONFIG="${1:-configs/config.yaml}"
if [ "$#" -gt 0 ]; then
    shift
fi

python src/model_download/01_download_models.py --config "$CONFIG" "$@"

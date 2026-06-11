#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
python3 ml/prepare_dataset.py \
  --source "$ROOT_DIR" \
  --out "$ROOT_DIR/ml/datasets/sku_live" \
  --label-studio-base-url http://localhost:8099

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
python3 "$ROOT_DIR/ml/serve_dataset.py" --root "$ROOT_DIR/ml/datasets/sku_live" --port 8099

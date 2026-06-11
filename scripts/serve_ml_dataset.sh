#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
python3 -m http.server 8099 -d "$ROOT_DIR/ml/datasets/sku_live"

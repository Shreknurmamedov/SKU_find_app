#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR/backend"
PYTHONPATH=src python3 -m sku_audit.cli create-job --input "$ROOT_DIR" --var-dir "$ROOT_DIR/var"

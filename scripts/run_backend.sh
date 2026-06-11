#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../backend"
PYTHONPATH=src python3 -m uvicorn sku_audit.app:app --host 0.0.0.0 --port 8088

#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
for v in "ТТ Пэкстрой/IMG_8886.MOV" "ООО ВРЕМЕНА ГОДА/IMG_8942.MOV" "ЕВРОМИКС/IMG_8916.MOV"; do
  echo "[rerun] $v"
  python3 -m ml.audit_video --video "$v" --weights weights/product_det.pt \
    --device mps --reuse-tracks || echo "[rerun] FAILED $v"
done
echo "[rerun] done"

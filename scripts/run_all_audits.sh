#!/usr/bin/env bash
# Wait for the FastSAM-detector training, then audit all 7 videos and combine.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "[all] waiting for product_det_fs training..."
while pgrep -f "train_detector.py.*product_det_fs" >/dev/null; do sleep 30; done
echo "[all] training finished."

SRC="runs/detect/ml/runs/product_det_fs/weights/best.pt"
if [[ ! -f "$SRC" ]]; then echo "[all] ERROR: $SRC missing"; exit 1; fi
mkdir -p weights
cp "$SRC" weights/product_det_fs.pt
echo "[all] weights -> weights/product_det_fs.pt"

VIDEOS=(
  "ТТ Пэкстрой/IMG_8886.MOV"
  "ТТ Пэкстрой/IMG_8882.MOV"
  "ТТ Пэкстрой/IMG_8883.MOV"
  "ТТ Пэкстрой/IMG_8884.MOV"
  "ООО ВРЕМЕНА ГОДА/IMG_8942.MOV"
  "ЕВРОМИКС/IMG_8916.MOV"
  "ИП Маргарян/IMG_8967.MOV"
)

for v in "${VIDEOS[@]}"; do
  echo "[all] auditing: $v"
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 -m ml.audit_video \
    --video "$v" --weights weights/product_det_fs.pt \
    --device mps --vid-stride 3 || echo "[all] FAILED on $v"
done

echo "[all] combining reports..."
python3 -m ml.combine_reports
echo "[all] done."

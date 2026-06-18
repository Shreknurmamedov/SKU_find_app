#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:-ml/datasets/sku_products_v8/data.yaml}"
NAME="${2:-product_det_v8}"

if [[ ! -f "$DATASET" ]]; then
  echo "[v8] missing dataset: $DATASET"
  echo "[v8] build it after Label Studio review:"
  echo "     python3 ml/build_product_dataset.py --src ml/datasets/sku_products_v8_review --dst ml/datasets/sku_products_v8"
  exit 1
fi

echo "[v8] training $NAME on $DATASET..."
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 ml/train_detector.py \
  --data "$DATASET" \
  --imgsz 1024 \
  --batch 4 \
  --epochs 40 \
  --patience 10 \
  --name "$NAME" \
  || { echo "[v8] train FAILED"; exit 1; }

SRC="runs/detect/ml/runs/$NAME/weights/best.pt"
if [[ ! -f "$SRC" ]]; then
  echo "[v8] missing trained weights: $SRC"
  exit 1
fi
cp "$SRC" weights/product_det_v2.pt
echo "[v8] DONE -> weights/product_det_v2.pt updated"

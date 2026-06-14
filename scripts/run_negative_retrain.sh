#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[neg] 1/3 extracting background negatives from videos..."
python3 ml/add_negatives.py || { echo "[neg] negatives FAILED"; exit 1; }
echo "[neg] 2/3 training product_det_v3..."
rm -rf "runs/detect/ml/runs/product_det_v3"* 2>/dev/null || true
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 ml/train_detector.py \
  --data ml/datasets/sku_products_v3/data.yaml --imgsz 1024 --batch 4 \
  --epochs 30 --patience 8 --name product_det_v3 || { echo "[neg] train FAILED"; exit 1; }
cp "runs/detect/ml/runs/product_det_v3/weights/best.pt" weights/product_det_v2.pt
echo "[neg] DONE -> weights/product_det_v2.pt updated"

#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[v5] training product_det_v5 (v4 + indoor/furniture negatives)..."
rm -rf "runs/detect/ml/runs/product_det_v5"* 2>/dev/null || true
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 ml/train_detector.py \
  --data ml/datasets/sku_products_v5/data.yaml --imgsz 1024 --batch 4 \
  --epochs 30 --patience 8 --name product_det_v5 || { echo "[v5] train FAILED"; exit 1; }
cp "runs/detect/ml/runs/product_det_v5/weights/best.pt" weights/product_det_v2.pt
echo "[v5] DONE -> weights/product_det_v2.pt updated"

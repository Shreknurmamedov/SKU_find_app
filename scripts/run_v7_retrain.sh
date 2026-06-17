#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[v7] 1/2 device hard-negatives -> dataset v7..."
python3 -m ml.add_device_negatives || { echo "[v7] data FAILED"; exit 1; }
echo "[v7] 2/2 training product_det_v7..."
rm -rf "runs/detect/ml/runs/product_det_v7"* 2>/dev/null || true
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 ml/train_detector.py \
  --data ml/datasets/sku_products_v7/data.yaml --imgsz 1024 --batch 4 \
  --epochs 30 --patience 8 --name product_det_v7 || { echo "[v7] train FAILED"; exit 1; }
cp "runs/detect/ml/runs/product_det_v7/weights/best.pt" weights/product_det_v2.pt
echo "[v7] DONE -> weights/product_det_v2.pt updated"

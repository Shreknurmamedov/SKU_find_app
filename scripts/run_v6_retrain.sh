#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[v6] 1/2 pulling retail positives from SUN397 -> dataset v6..."
python3 -m ml.hf_retail_positives || { echo "[v6] data FAILED"; exit 1; }
echo "[v6] 2/2 training product_det_v6..."
rm -rf "runs/detect/ml/runs/product_det_v6"* 2>/dev/null || true
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 ml/train_detector.py \
  --data ml/datasets/sku_products_v6/data.yaml --imgsz 1024 --batch 4 \
  --epochs 30 --patience 8 --name product_det_v6 || { echo "[v6] train FAILED"; exit 1; }
cp "runs/detect/ml/runs/product_det_v6/weights/best.pt" weights/product_det_v2.pt
echo "[v6] DONE -> weights/product_det_v2.pt updated"

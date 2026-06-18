#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

echo "[v8-hardneg] 1/2 build balanced hard-negative dataset..."
python3 -m ml.add_v8_hard_negatives \
  --base ml/datasets/sku_products_v8_auto \
  --out ml/datasets/sku_products_v8_hardneg \
  --weights weights/product_det_v8_auto.pt \
  --target-device 60 \
  --target-mined 80 \
  || { echo "[v8-hardneg] dataset FAILED"; exit 1; }

echo "[v8-hardneg] 2/2 train from v8_auto candidate..."
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 ml/train_detector.py \
  --data ml/datasets/sku_products_v8_hardneg/data.yaml \
  --model weights/product_det_v8_auto.pt \
  --imgsz 1024 \
  --batch 4 \
  --epochs 12 \
  --patience 5 \
  --name product_det_v8_hardneg \
  || { echo "[v8-hardneg] train FAILED"; exit 1; }

cp runs/detect/ml/runs/product_det_v8_hardneg/weights/best.pt weights/product_det_v8_hardneg.pt
echo "[v8-hardneg] DONE -> weights/product_det_v8_hardneg.pt"

#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[v4] 1/3 FastSAM-labeling 'Обучение ИИ' photos -> dataset v4..."
python3 -m ml.add_training_photos || { echo "[v4] label FAILED"; exit 1; }
echo "[v4] 2/3 training product_det_v4..."
rm -rf "runs/detect/ml/runs/product_det_v4"* 2>/dev/null || true
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 ml/train_detector.py \
  --data ml/datasets/sku_products_v4/data.yaml --imgsz 1024 --batch 4 \
  --epochs 30 --patience 8 --name product_det_v4 || { echo "[v4] train FAILED"; exit 1; }
cp "runs/detect/ml/runs/product_det_v4/weights/best.pt" weights/product_det_v2.pt
echo "[v4] DONE -> weights/product_det_v2.pt updated"

#!/usr/bin/env bash
# Full detector-tuning cycle: stricter FastSAM labels -> retrain -> audit all 7
# videos with stronger inference -> combined report. Produces weights/product_det_v2.pt
# (baseline weights/product_det.pt is left untouched).
set -uo pipefail
cd "$(dirname "$0")/.."

echo "[tune] 1/4 strict FastSAM relabel (whole products, not fragments)..."
python3 -m ml.fastsam_label --min-area 0.005 --contain-thr 0.7 --nms-iou 0.6 \
  || { echo "[tune] relabel FAILED"; exit 1; }

echo "[tune] 2/4 train detector (product_det_v2)..."
rm -rf "runs/detect/ml/runs/product_det_v2"* 2>/dev/null || true
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 ml/train_detector.py \
  --data ml/datasets/sku_products_fs/data.yaml --imgsz 1024 --batch 4 \
  --epochs 30 --patience 8 --name product_det_v2 \
  || { echo "[tune] train FAILED"; exit 1; }
mkdir -p weights
cp "runs/detect/ml/runs/product_det_v2/weights/best.pt" weights/product_det_v2.pt
echo "[tune] weights -> weights/product_det_v2.pt"

echo "[tune] 3/4 audit all 7 videos (conf 0.4, min-frames 4)..."
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
  echo "[tune] audit: $v"
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 -m ml.audit_video \
    --video "$v" --weights weights/product_det_v2.pt \
    --device mps --vid-stride 3 --conf 0.4 --min-frames 4 \
    || echo "[tune] FAILED on $v"
done

echo "[tune] 4/4 combine..."
python3 -m ml.combine_reports
echo "[tune] done."

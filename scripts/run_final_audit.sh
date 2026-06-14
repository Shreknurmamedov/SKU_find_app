#!/usr/bin/env bash
# Wait for detector training to finish, then run a full SKU audit on a few
# representative videos with good inference settings on MPS.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "[final] waiting for detector training to finish..."
while pgrep -f train_detector.py >/dev/null; do sleep 30; done
echo "[final] training finished."

SRC="runs/detect/ml/runs/product_det/weights/best.pt"
mkdir -p weights
cp "$SRC" weights/product_det.pt
echo "[final] weights -> weights/product_det.pt"

VIDEOS=(
  "ТТ Пэкстрой/IMG_8886.MOV"
  "ООО ВРЕМЕНА ГОДА/IMG_8942.MOV"
  "ЕВРОМИКС/IMG_8916.MOV"
)

for v in "${VIDEOS[@]}"; do
  echo "[final] auditing: $v"
  python3 -m ml.audit_video \
    --video "$v" \
    --weights weights/product_det.pt \
    --device mps \
    --vid-stride 3 \
    --review-conf 0.75 \
    || echo "[final] FAILED on $v"
done

echo "[final] all done."

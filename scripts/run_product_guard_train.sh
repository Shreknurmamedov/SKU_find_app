#!/usr/bin/env bash
# Build and train the tablet product-vs-interior guard classifier.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DATA="${OUT_DATA:-ml/datasets/product_guard_cls}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-64}"
DEVICE="${SKU_AUDIT_DEVICE:-mps}"

python3 -m ml.build_product_guard_dataset --out "$OUT_DATA"
python3 -m ml.train_product_guard \
  --data "$OUT_DATA" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --device "$DEVICE"

echo "Done: weights/product_guard_cls.pt"
echo "For Android CI, force-add and commit weights/product_guard_cls.pt."

"""Auto-label the user-provided 'Обучение ИИ' photos with FastSAM and merge them
into the detector dataset (v4 = v3 + these).

The folder holds real shop photos including our own SKUs (Ресанта / Huter boxes)
plus dense generic hardware. We FastSAM-label every product (strict params, whole
products not fragments); near-empty floor/corner shots naturally get few/no boxes
and act as soft negatives. Output: ml/datasets/sku_products_v4.

    python3 ml/add_training_photos.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2

from ml.fastsam_label import boxes_from_fastsam, to_yolo_lines

SRC = Path("Обучение ИИ")
BASE = Path("ml/datasets/sku_products_v3")
OUT = Path("ml/datasets/sku_products_v4")
VAL_EVERY = 6


def main() -> None:
    from ultralytics import FastSAM

    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in (BASE / "images" / split).glob("*"):
            shutil.copy2(img, OUT / "images" / split / img.name)
        for lbl in (BASE / "labels" / split).glob("*"):
            shutil.copy2(lbl, OUT / "labels" / split / lbl.name)

    model = FastSAM("FastSAM-s.pt")
    photos = sorted(SRC.glob("*.jpg"))
    added = boxes_total = empties = 0
    for i, ip in enumerate(photos):
        res = model.predict(str(ip), imgsz=1024, conf=0.4, iou=0.9,
                            retina_masks=False, device="mps", verbose=False)[0]
        h, w = res.orig_shape
        kept = boxes_from_fastsam(res, w, h, min_area=0.005, max_area=0.25,
                                  max_side=0.6, nms_iou=0.7, contain_thr=0.7)
        lines = to_yolo_lines(kept, w, h)
        split = "val" if i % VAL_EVERY == 0 else "train"
        stem = f"train_{ip.stem}"
        cv2.imwrite(str(OUT / "images" / split / f"{stem}.jpg"), res.orig_img)
        (OUT / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""))
        added += 1
        boxes_total += len(lines)
        if not lines:
            empties += 1
        if added % 16 == 0:
            print(f"[{added}/{len(photos)}] {ip.name}: {len(lines)} boxes")

    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: product\n")
    n_train = len(list((OUT / "images/train").glob("*")))
    n_val = len(list((OUT / "images/val").glob("*")))
    print(f"\nadded {added} photos ({boxes_total} boxes, {empties} empty); "
          f"dataset v4: train={n_train} val={n_val}")
    print(f"-> {OUT/'data.yaml'}")


if __name__ == "__main__":
    main()

"""Build a dataset that adds background negatives to teach the detector that
not everything is a product.

Negatives are frames from the shop videos where the current detector finds
almost nothing (floor, ceiling, aisles, transitions) — i.e. genuine non-shelf
views. They are added as label-free images, which YOLO treats as backgrounds and
learns to keep empty. This lowers the false-positive rate on furniture/walls.

    python3 ml/add_negatives.py
Produces ml/datasets/sku_products_v3 (= sku_products_fs + negatives).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2

BASE = Path("ml/datasets/sku_products_fs")
OUT = Path("ml/datasets/sku_products_v3")
WEIGHTS = "weights/product_det_v2.pt"
VIDEOS = [
    "ТТ Пэкстрой/IMG_8886.MOV", "ТТ Пэкстрой/IMG_8882.MOV",
    "ТТ Пэкстрой/IMG_8883.MOV", "ТТ Пэкстрой/IMG_8884.MOV",
    "ООО ВРЕМЕНА ГОДА/IMG_8942.MOV", "ЕВРОМИКС/IMG_8916.MOV",
    "ИП Маргарян/IMG_8967.MOV",
]
FRAMES_PER_VIDEO = 28          # candidates sampled per video
MAX_DETS_FOR_NEGATIVE = 2      # frame counts as background if <= this many products
MAX_NEGATIVES = 110            # cap so negatives stay a minority of the dataset
VAL_EVERY = 6


def main() -> None:
    from ultralytics import YOLO

    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in (BASE / "images" / split).glob("*"):
            shutil.copy2(img, OUT / "images" / split / img.name)
        for lbl in (BASE / "labels" / split).glob("*"):
            shutil.copy2(lbl, OUT / "labels" / split / lbl.name)

    model = YOLO(WEIGHTS)
    added = 0
    fi = 0
    import numpy as np
    for vid in VIDEOS:
        vp = Path(vid)
        if not vp.exists() or added >= MAX_NEGATIVES:
            continue
        cap = cv2.VideoCapture(str(vp))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        idxs = np.linspace(0, total - 1, FRAMES_PER_VIDEO).astype(int)
        kept_here = 0
        for idx in idxs:
            if added >= MAX_NEGATIVES or kept_here >= 18:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, fr = cap.read()
            if not ok:
                continue
            res = model.predict(fr, imgsz=320, conf=0.5, device="mps", verbose=False)[0]
            ndet = 0 if res.boxes is None else len(res.boxes)
            if ndet > MAX_DETS_FOR_NEGATIVE:
                continue  # has products -> not a background negative
            split = "val" if fi % VAL_EVERY == 0 else "train"
            stem = f"neg_{vp.stem}_{int(idx):05d}"
            cv2.imwrite(str(OUT / "images" / split / f"{stem}.jpg"), fr)
            (OUT / "labels" / split / f"{stem}.txt").write_text("")  # empty = background
            added += 1
            kept_here += 1
            fi += 1
        cap.release()
        print(f"{vid}: +{kept_here} negatives")

    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: product\n")
    n_train = len(list((OUT / "images/train").glob("*")))
    n_val = len(list((OUT / "images/val").glob("*")))
    print(f"\nadded {added} background negatives; dataset v3: train={n_train} val={n_val}")
    print(f"-> {OUT/'data.yaml'}")


if __name__ == "__main__":
    main()

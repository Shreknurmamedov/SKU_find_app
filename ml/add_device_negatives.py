"""Hard negatives from a video the user recorded of empty zones (their own
walls / floor / cabinets / yard — no products). Frames are added as label-free
backgrounds so the detector learns THIS environment's non-products aren't SKUs.

    python3 -m ml.add_device_negatives
Produces ml/datasets/sku_products_v7 (= v6 + device hard negatives).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

SRC = Path("var/neg_videos")
BASE = Path("ml/datasets/sku_products_v6")
OUT = Path("ml/datasets/sku_products_v7")
TARGET = 160
VAL_EVERY = 6


def main() -> None:
    vids = sorted(SRC.rglob("*.mp4"))
    if not vids:
        raise SystemExit("no negative videos in var/neg_videos")

    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in (BASE / "images" / split).glob("*"):
            shutil.copy2(img, OUT / "images" / split / img.name)
        for lbl in (BASE / "labels" / split).glob("*"):
            shutil.copy2(lbl, OUT / "labels" / split / lbl.name)

    per_video = max(1, TARGET // len(vids))
    saved = 0
    for vi, v in enumerate(vids):
        cap = cv2.VideoCapture(str(v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        idxs = np.linspace(n * 0.02, n * 0.98, per_video).astype(int)
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, fr = cap.read()
            if not ok:
                continue
            split = "val" if saved % VAL_EVERY == 0 else "train"
            stem = f"devneg_{vi}_{int(idx):06d}"
            cv2.imwrite(str(OUT / "images" / split / f"{stem}.jpg"), fr)
            (OUT / "labels" / split / f"{stem}.txt").write_text("")  # empty = background
            saved += 1
        cap.release()

    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: product\n")
    n_train = len(list((OUT / "images/train").glob("*")))
    n_val = len(list((OUT / "images/val").glob("*")))
    print(f"added {saved} device hard-negatives; dataset v7: train={n_train} val={n_val}")


if __name__ == "__main__":
    main()

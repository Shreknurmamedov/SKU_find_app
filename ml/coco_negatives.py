"""Add furniture/indoor negatives from COCO so the detector stops boxing
furniture (cabinets, shelving units, tables) as products.

Selects COCO val2017 images that contain furniture (chair/couch/bed/dining
table) and NONE of the product-shelf-like categories (bottle/cup/bowl/book/
vase/wine glass), downloads them, and adds them as label-free negatives to a new
dataset (v5 = v4 + COCO negatives). COCO is openly licensed (CC BY 4.0 images).

    python3 -m ml.coco_negatives
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ANN = Path("var/coco/annotations/instances_val2017.json")
BASE = Path("ml/datasets/sku_products_v4")
OUT = Path("ml/datasets/sku_products_v5")
IMG_DIR = Path("var/coco/neg_images")

FURNITURE = {62, 63, 65, 67}          # chair, couch, bed, dining table
AVOID = {44, 46, 47, 51, 84, 86}      # bottle, wine glass, cup, bowl, book, vase
MAX_NEG = 150
VAL_EVERY = 6


def main() -> None:
    data = json.loads(ANN.read_text())
    img_cats: dict[int, set[int]] = {}
    for a in data["annotations"]:
        img_cats.setdefault(a["image_id"], set()).add(a["category_id"])
    id2file = {im["id"]: im["file_name"] for im in data["images"]}

    selected = []
    for iid, cats in img_cats.items():
        if (cats & FURNITURE) and not (cats & AVOID):
            selected.append(iid)
        if len(selected) >= MAX_NEG:
            break
    print(f"selected {len(selected)} furniture-only images")

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    got = []
    for iid in selected:
        fn = id2file[iid]
        dst = IMG_DIR / fn
        if not (dst.exists() and dst.stat().st_size > 2000):
            url = f"https://images.cocodataset.org/val2017/{fn}"
            subprocess.run(["curl", "-sL", "-o", str(dst), url], check=False)
        if dst.exists() and dst.stat().st_size > 2000:
            got.append(dst)
    print(f"downloaded {len(got)} images")

    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in (BASE / "images" / split).glob("*"):
            shutil.copy2(img, OUT / "images" / split / img.name)
        for lbl in (BASE / "labels" / split).glob("*"):
            shutil.copy2(lbl, OUT / "labels" / split / lbl.name)

    for i, src in enumerate(got):
        split = "val" if i % VAL_EVERY == 0 else "train"
        stem = f"coco_neg_{src.stem}"
        shutil.copy2(src, OUT / "images" / split / f"{stem}.jpg")
        (OUT / "labels" / split / f"{stem}.txt").write_text("")  # empty = background

    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: product\n")
    n_train = len(list((OUT / "images/train").glob("*")))
    n_val = len(list((OUT / "images/val").glob("*")))
    print(f"dataset v5: train={n_train} val={n_val} (+{len(got)} COCO negatives)")


if __name__ == "__main__":
    main()

"""Add retail-shop POSITIVES from SUN397 to diversify the product detector.

Streams SUN397 store/shop/market scenes (shelves of goods), FastSAM-labels the
products, and merges them into a new dataset (v6 = v5 + retail positives) so the
detector generalizes better to varied real shelves. COCO is blocked from this
env; SUN397 via HuggingFace is reachable.

    python3 -m ml.hf_retail_positives
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
from PIL import Image

from ml.fastsam_label import boxes_from_fastsam, to_yolo_lines

POS_DIR = Path("var/hf_pos")
BASE = Path("ml/datasets/sku_products_v5")
OUT = Path("ml/datasets/sku_products_v6")
TARGET = 120
MAX_SCAN = 16000
VAL_EVERY = 6
RETAIL_KW = ("shop", "store", "market", "bookstore", "supermarket",
             "bakery", "florist", "drugstore", "delicatessen")


def _first_image(example):
    for v in example.values():
        if isinstance(v, Image.Image):
            return v
    return None


def grab_retail() -> int:
    from datasets import load_dataset
    if POS_DIR.exists():
        shutil.rmtree(POS_DIR)
    POS_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("tanganke/sun397", split="test", streaming=True)
    try:
        names = ds.features["label"].names
    except Exception:  # noqa: BLE001
        names = None
    n = scanned = 0
    for ex in ds:
        scanned += 1
        if scanned > MAX_SCAN or n >= TARGET:
            break
        if names is not None and "label" in ex:
            cls = names[ex["label"]] if isinstance(ex["label"], int) else str(ex["label"])
            if not any(k in cls for k in RETAIL_KW):
                continue
        img = _first_image(ex)
        if img is None:
            continue
        try:
            img.convert("RGB").save(POS_DIR / f"hfpos_{n:04d}.jpg", quality=88)
        except Exception:
            continue
        n += 1
    print(f"scanned {scanned}, saved {n} retail images")
    return n


def build_v6() -> None:
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
    imgs = sorted(POS_DIR.glob("*.jpg"))
    boxes_total = 0
    for i, ip in enumerate(imgs):
        res = model.predict(str(ip), imgsz=1024, conf=0.4, iou=0.9,
                            retina_masks=False, device="mps", verbose=False)[0]
        h, w = res.orig_shape
        kept = boxes_from_fastsam(res, w, h, min_area=0.005, max_area=0.25,
                                  max_side=0.6, nms_iou=0.7, contain_thr=0.7)
        lines = to_yolo_lines(kept, w, h)
        split = "val" if i % VAL_EVERY == 0 else "train"
        stem = f"retail_{ip.stem}"
        cv2.imwrite(str(OUT / "images" / split / f"{stem}.jpg"), res.orig_img)
        (OUT / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""))
        boxes_total += len(lines)
    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: product\n")
    print(f"dataset v6: train={len(list((OUT/'images/train').glob('*')))} "
          f"val={len(list((OUT/'images/val').glob('*')))} (+{len(imgs)} retail pos, {boxes_total} boxes)")


def main() -> None:
    n = grab_retail()
    if n < 20:
        raise SystemExit(f"only {n} retail images; aborting")
    build_v6()


if __name__ == "__main__":
    main()

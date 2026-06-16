"""Pull indoor/furniture images from a HuggingFace dataset (streaming) and add
them as label-free negatives so the detector stops boxing furniture as products.

COCO's CDN is blocked from this environment, but huggingface.co is reachable, so
we stream an indoor-scenes dataset and save images. Output: ml/datasets/sku_products_v5
(= v4 + indoor negatives).

    python3 -m ml.hf_negatives
"""
from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

NEG_DIR = Path("var/hf_neg")
BASE = Path("ml/datasets/sku_products_v4")
OUT = Path("ml/datasets/sku_products_v5")
TARGET = 150
VAL_EVERY = 6

# SUN397 scene classes that are furniture-rich indoor rooms (substring match).
INDOOR_KEYWORDS = (
    "bedroom", "living_room", "dining_room", "kitchen", "office", "closet",
    "hotel_room", "childs_room", "nursery", "waiting_room", "conference_room",
    "basement", "corridor", "dorm_room", "playroom", "recreation_room",
    "bathroom", "pantry", "attic", "home_office", "reading_room", "lobby",
    "restaurant", "bar", "shoe_shop", "toyshop", "bookstore", "clothing_store",
)
MAX_SCAN = 12000  # streamed samples to scan while collecting indoor ones


def _first_image(example):
    for v in example.values():
        if isinstance(v, Image.Image):
            return v
    return None


def grab_images() -> int:
    from datasets import load_dataset
    if NEG_DIR.exists():
        shutil.rmtree(NEG_DIR)
    NEG_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("tanganke/sun397", split="test", streaming=True)
    names = None
    try:
        names = ds.features["label"].names
    except Exception:  # noqa: BLE001
        names = None

    n = scanned = 0
    for ex in ds:
        scanned += 1
        if scanned > MAX_SCAN:
            break
        keep = True
        if names is not None and "label" in ex:
            cls = names[ex["label"]] if isinstance(ex["label"], int) else str(ex["label"])
            keep = any(k in cls for k in INDOOR_KEYWORDS)
        if not keep:
            continue
        img = _first_image(ex)
        if img is None:
            continue
        try:
            img.convert("RGB").save(NEG_DIR / f"hfneg_{n:04d}.jpg", quality=85)
        except Exception:
            continue
        n += 1
        if n >= TARGET:
            break
    print(f"scanned {scanned}, saved {n} indoor negatives")
    return n


def build_v5() -> None:
    imgs = sorted(NEG_DIR.glob("*.jpg"))
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in (BASE / "images" / split).glob("*"):
            shutil.copy2(img, OUT / "images" / split / img.name)
        for lbl in (BASE / "labels" / split).glob("*"):
            shutil.copy2(lbl, OUT / "labels" / split / lbl.name)
    for i, src in enumerate(imgs):
        split = "val" if i % VAL_EVERY == 0 else "train"
        stem = f"hfneg_{src.stem}"
        shutil.copy2(src, OUT / "images" / split / f"{stem}.jpg")
        (OUT / "labels" / split / f"{stem}.txt").write_text("")
    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: product\n")
    print(f"dataset v5: train={len(list((OUT/'images/train').glob('*')))} "
          f"val={len(list((OUT/'images/val').glob('*')))} (+{len(imgs)} indoor negatives)")


def main() -> None:
    n = grab_images()
    if n < 20:
        raise SystemExit(f"only {n} negatives fetched; aborting")
    build_v5()


if __name__ == "__main__":
    main()

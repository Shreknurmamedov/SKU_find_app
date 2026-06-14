"""Collapse the 2-class seg labels into a single-class detection dataset.

Stage-1 detector only needs to localize every visible product. Brand /
own-vs-competitor / model / category are decided in stage 2 from the crop
(OCR + catalog matching), which is far more reliable than guessing a brand
from a tiny box. So here we:

  * map every instance to class 0 ("product");
  * convert the rectangle polygons into YOLO detection boxes (cx cy w h);
  * link images and write a fresh data.yaml.

Run:
    python3 ml/build_product_dataset.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

SRC = Path("ml/datasets/sku_live")
DST = Path("ml/datasets/sku_products")
SPLITS = ("train", "val")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def poly_to_box(coords: list[float]) -> tuple[float, float, float, float]:
    xs = coords[0::2]
    ys = coords[1::2]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = x1 - x0, y1 - y0
    return cx, cy, w, h


def find_image(stem: str, split: str) -> Path | None:
    for ext in IMG_EXTS:
        p = SRC / "images" / split / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main() -> None:
    n_imgs = n_boxes = n_skipped = 0
    for split in SPLITS:
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)
        for lbl in sorted((SRC / "labels" / split).glob("*.txt")):
            img = find_image(lbl.stem, split)
            if img is None:
                n_skipped += 1
                continue
            out_lines: list[str] = []
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 7:  # need class + >=3 points
                    continue
                coords = [float(v) for v in parts[1:]]
                cx, cy, w, h = poly_to_box(coords)
                if w <= 0 or h <= 0:
                    continue
                out_lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            # keep empty-label images too (valid negatives)
            (DST / "labels" / split / f"{lbl.stem}.txt").write_text(
                "\n".join(out_lines) + ("\n" if out_lines else "")
            )
            dst_img = DST / "images" / split / img.name
            if not dst_img.exists():
                shutil.copy2(img, dst_img)
            n_imgs += 1
            n_boxes += len(out_lines)

    data_yaml = DST / "data.yaml"
    data_yaml.write_text(
        f"path: {DST.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: product\n"
    )
    print(f"images={n_imgs} boxes={n_boxes} skipped(no image)={n_skipped}")
    print(f"wrote {data_yaml}")


if __name__ == "__main__":
    main()

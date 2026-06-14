"""Re-label the stage-1 dataset with FastSAM class-agnostic proposals.

The prompt-based auto-labels miss most unboxed products on dense shelves, which
caps detector recall. FastSAM segments *everything*, so it proposes far more of
the real products. We convert masks to boxes and filter to product-like regions:

  * area within [min_area, max_area] of the frame (drop specks and whole-shelf);
  * drop boxes that are mostly nested inside a bigger kept box (sub-parts like a
    handle inside a tool) so we label whole products, not pieces;
  * class-agnostic NMS to dedup overlaps.

Rebuilds a fresh single-class dataset from the existing image files (base photos
+ extracted video frames), preserving the train/val split.

    python3 ml/fastsam_label.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

BASE = Path("ml/datasets/sku_products")          # 293 base photos (images only reused)
VID = Path("ml/datasets/sku_products_v2")         # 70 extracted video frames
OUT = Path("ml/datasets/sku_products_fs")


def boxes_from_fastsam(result, W: int, H: int, *, min_area, max_area,
                       max_side, nms_iou, contain_thr) -> list[tuple]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.cpu().numpy()
    # clip to image bounds (FastSAM boxes can spill a few px past the edge)
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, W)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, H)
    # area filter
    cand = []
    for x1, y1, x2, y2 in xyxy:
        bw, bh = (x2 - x1) / W, (y2 - y1) / H
        area = bw * bh
        if bw <= 0 or bh <= 0:
            continue
        if area < min_area or area > max_area:
            continue
        if bw > max_side or bh > max_side:
            continue
        cand.append((x1, y1, x2, y2, area))
    if not cand:
        return []
    cand.sort(key=lambda c: -c[4])  # largest first

    kept: list[tuple] = []
    for x1, y1, x2, y2, area in cand:
        drop = False
        for kx1, ky1, kx2, ky2, karea in kept:
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                continue
            self_area = (x2 - x1) * (y2 - y1)
            union = self_area + (kx2 - kx1) * (ky2 - ky1) - inter
            iou = inter / (union + 1e-6)
            contained = inter / (self_area + 1e-6)
            if contained > contain_thr or iou > nms_iou:
                drop = True
                break
        if not drop:
            kept.append((x1, y1, x2, y2, area))
    return kept


def to_yolo_lines(kept, W, H) -> list[str]:
    lines = []
    for x1, y1, x2, y2, _ in kept:
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        bw, bh = (x2 - x1) / W, (y2 - y1) / H
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def iter_source_images():
    for split in ("train", "val"):
        for img in sorted((BASE / "images" / split).glob("*.jpg")):
            yield split, img
    for split in ("train", "val"):
        for img in sorted((VID / "images" / split).glob("vid_*.jpg")):
            yield split, img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.9)
    ap.add_argument("--min-area", type=float, default=0.0008)
    ap.add_argument("--max-area", type=float, default=0.25)
    ap.add_argument("--max-side", type=float, default=0.6)
    ap.add_argument("--nms-iou", type=float, default=0.7)
    ap.add_argument("--contain-thr", type=float, default=0.8)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    import cv2
    from ultralytics import FastSAM
    import shutil

    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    model = FastSAM("FastSAM-s.pt")
    n_img = n_box = 0
    empties = 0
    for split, img in iter_source_images():
        # retina_masks=False: we only need boxes, and it avoids the 4K mask OOM.
        res = model.predict(str(img), imgsz=args.imgsz, conf=args.conf,
                            iou=args.iou, retina_masks=False, device=args.device,
                            verbose=False)[0]
        # normalize by the exact size FastSAM used, not PIL's (EXIF can swap W/H)
        H, W = res.orig_shape
        kept = boxes_from_fastsam(res, W, H, min_area=args.min_area,
                                  max_area=args.max_area, max_side=args.max_side,
                                  nms_iou=args.nms_iou, contain_thr=args.contain_thr)
        # save the exact pixels FastSAM saw (EXIF baked in, no metadata) so the
        # labels (normalized by res.orig_shape) match what YOLO loads in training
        cv2.imwrite(str(OUT / "images" / split / img.name), res.orig_img)
        lines = to_yolo_lines(kept, W, H)
        (OUT / "labels" / split / f"{img.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""))
        n_img += 1
        n_box += len(lines)
        if not lines:
            empties += 1
        if n_img % 50 == 0:
            print(f"[{n_img}] {img.name}: {len(lines)} boxes")

    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: product\n")
    print(f"\nimages={n_img} boxes={n_box} avg={n_box/max(1,n_img):.1f}/img empty={empties}")
    print(f"-> {OUT/'data.yaml'}")


if __name__ == "__main__":
    main()

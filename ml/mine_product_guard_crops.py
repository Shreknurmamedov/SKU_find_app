"""Mine product/interior crops for the tablet product guard.

The tablet pipeline is detector-first: YOLO proposes boxes, then a small
classifier rejects room/interior false positives. This script mines crops from
real store photos using the same idea and stores them as review candidates:

    var/tablet/guard_mining_m9/
      positive_candidates/*.jpg
      hard_negative_candidates/*.jpg
      uncertain_review/*.jpg
      manifest.json

Use the candidate sheets for a quick human pass before copying examples into
``var/tablet/product_guard_positive`` or
``var/tablet/product_guard_negative``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class CropItem:
    file: str
    bucket: str
    source: str
    det_conf: float
    guard_product_prob: float
    xyxy: list[int]


def iter_images(paths: list[Path]) -> list[Path]:
    images: list[Path] = []
    for raw in paths:
        path = raw.expanduser()
        if path.is_dir():
            images.extend(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.append(path)
    return sorted(set(images))


def clean_out(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    for bucket in ("positive_candidates", "hard_negative_candidates", "uncertain_review"):
        (out / bucket).mkdir(parents=True, exist_ok=True)


def crop_box(image: np.ndarray, xyxy: np.ndarray, pad: float) -> tuple[np.ndarray, list[int]] | None:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    bw, bh = x2 - x1, y2 - y1
    if bw < 24 or bh < 24:
        return None
    area_frac = (bw * bh) / max(1, w * h)
    if area_frac < 0.0008 or area_frac > 0.65:
        return None
    x1 = max(0, int(x1 - bw * pad))
    y1 = max(0, int(y1 - bh * pad))
    x2 = min(w, int(x2 + bw * pad))
    y2 = min(h, int(y2 + bh * pad))
    if x2 - x1 < 24 or y2 - y1 < 24:
        return None
    return image[y1:y2, x1:x2].copy(), [x1, y1, x2, y2]


def product_prob(result) -> float:
    names = result.names or {}
    product_idx = None
    for idx, name in names.items():
        if str(name).lower() == "product":
            product_idx = int(idx)
            break
    if product_idx is None:
        product_idx = 1
    probs = result.probs.data.detach().cpu().numpy()
    if product_idx >= len(probs):
        return 0.0
    return float(probs[product_idx])


def bucket_for(det_conf: float, guard_prob: float, positive_min: float,
               negative_max: float, hard_det_min: float) -> str:
    if guard_prob >= positive_min:
        return "positive_candidates"
    if det_conf >= hard_det_min and guard_prob <= negative_max:
        return "hard_negative_candidates"
    return "uncertain_review"


def safe_stem(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    cleaned = "".join(c if c.isalnum() else "_" for c in path.stem)[:48]
    return f"{cleaned}_{digest}"


def save_crop(out: Path, bucket: str, index: int, source: Path,
              det_conf: float, guard_prob: float, crop: np.ndarray) -> Path:
    stem = safe_stem(source)
    name = f"{bucket}_{index:05d}_{stem}_det{det_conf:.2f}_guard{guard_prob:.2f}.jpg"
    dst = out / bucket / name
    cv2.imwrite(str(dst), crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
    return dst


def make_sheet(bucket_dir: Path, out_file: Path, max_items: int = 120) -> None:
    files = sorted(bucket_dir.glob("*.jpg"))[:max_items]
    if not files:
        return
    font = ImageFont.load_default()
    tiles = []
    for src in files:
        im = Image.open(src).convert("RGB")
        im = ImageOps.exif_transpose(im)
        im.thumbnail((180, 180), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (210, 230), "white")
        tile.paste(im, ((210 - im.width) // 2, 6))
        ImageDraw.Draw(tile).text((6, 190), src.stem[-32:], fill="black", font=font)
        tiles.append(tile)
    cols = 6
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * 210, rows * 230 + 30), "white")
    ImageDraw.Draw(sheet).text((8, 8), f"{bucket_dir.name}: {len(files)} shown", fill="black", font=font)
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % cols) * 210, 30 + (idx // cols) * 230))
    sheet.save(out_file, "JPEG", quality=90)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sources", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("var/tablet/guard_mining"))
    ap.add_argument("--detector", default="weights/product_det_v8_hardneg.pt")
    ap.add_argument("--guard", default="weights/product_guard_cls.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.18)
    ap.add_argument("--iou", type=float, default=0.55)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-det", type=int, default=120)
    ap.add_argument("--pad", type=float, default=0.06)
    ap.add_argument("--positive-min", type=float, default=0.90)
    ap.add_argument("--negative-max", type=float, default=0.18)
    ap.add_argument("--hard-det-min", type=float, default=0.30)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    images = iter_images(args.sources)
    if not images:
        raise SystemExit("no images found")
    clean_out(args.out)

    from ultralytics import YOLO

    detector = YOLO(args.detector)
    guard = YOLO(args.guard)

    counters = {"positive_candidates": 0, "hard_negative_candidates": 0, "uncertain_review": 0}
    items: list[CropItem] = []

    for start in range(0, len(images), args.batch):
        batch_paths = images[start:start + args.batch]
        results = detector.predict(
            [str(p) for p in batch_paths],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            max_det=args.max_det,
            verbose=False,
        )
        for source, result in zip(batch_paths, results):
            image = cv2.imread(str(source))
            if image is None or result.boxes is None:
                continue
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            crop_records: list[tuple[np.ndarray, list[int], float]] = []
            for xyxy, det_conf in zip(boxes, confs):
                cropped = crop_box(image, xyxy, args.pad)
                if cropped is None:
                    continue
                crop, adjusted_xyxy = cropped
                crop_records.append((crop, adjusted_xyxy, float(det_conf)))
            if not crop_records:
                continue
            guard_results = guard.predict(
                [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop, _, _ in crop_records],
                imgsz=224,
                device=args.device,
                verbose=False,
            )
            for (crop, adjusted_xyxy, det_conf), guard_result in zip(crop_records, guard_results):
                gp = product_prob(guard_result)
                bucket = bucket_for(
                    det_conf, gp, args.positive_min, args.negative_max, args.hard_det_min)
                index = counters[bucket]
                saved = save_crop(args.out, bucket, index, source, det_conf, gp, crop)
                counters[bucket] += 1
                items.append(CropItem(
                    file=str(saved),
                    bucket=bucket,
                    source=str(source),
                    det_conf=det_conf,
                    guard_product_prob=gp,
                    xyxy=adjusted_xyxy,
                ))
        print(f"processed {min(start + args.batch, len(images))}/{len(images)}", flush=True)

    manifest = {
        "sources": [str(p) for p in args.sources],
        "images": len(images),
        "detector": args.detector,
        "guard": args.guard,
        "counts": counters,
        "items": [item.__dict__ for item in items],
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for bucket in counters:
        make_sheet(args.out / bucket, args.out / f"{bucket}_sheet.jpg")
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

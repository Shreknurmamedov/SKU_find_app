"""Build a binary product-vs-interior dataset for the tablet guard model.

The on-device YOLO detector is intentionally small and recall-oriented, so it
sometimes proposes chairs, doors, coolers, walls and furniture as "product".
This builder creates an ImageFolder classification dataset:

    ml/datasets/product_guard_cls/
      train/interior/*.jpg
      train/product/*.jpg
      val/interior/*.jpg
      val/product/*.jpg

Positive examples come from the reference product photos plus curated real
camera crops. Negative examples are random crops from tablet room/screenrecord
videos and curated hard-negative crops where YOLO saw interior as product. The
resulting classifier is a second on-device guard: YOLO proposes a box, the guard
decides whether the crop looks like retail product or room/interior.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2


REPO = Path(__file__).resolve().parent.parent
DEFAULT_METADATA = Path("data/catalog/reference_dataset_all/training_images.csv")
DEFAULT_OUT = Path("ml/datasets/product_guard_cls")
DEFAULT_NEGATIVE_VIDEOS = [
    "var/tablet/sku_live_debug.mp4:16-24",
    "var/tablet/cap4.mp4",
    "var/tablet/cap_live.mp4:6-9",
]
DEFAULT_NEGATIVE_EXTRA_DIRS = [
    Path("var/tablet/product_guard_negative"),
]


@dataclass(frozen=True)
class VideoSpec:
    path: Path
    start: float | None = None
    end: float | None = None


def parse_video_spec(raw: str) -> VideoSpec:
    if ":" not in raw:
        return VideoSpec(Path(raw))
    path, span = raw.rsplit(":", 1)
    if "-" not in span:
        return VideoSpec(Path(raw))
    start_s, end_s = span.split("-", 1)
    return VideoSpec(Path(path), float(start_s or 0), float(end_s or 0))


def clean_out(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        for cls in ("interior", "product"):
            (out / split / cls).mkdir(parents=True, exist_ok=True)


def product_paths(metadata: Path, limit: int, seed: int,
                  extra_dirs: list[Path] | None = None) -> list[Path]:
    rows: list[Path] = []
    with metadata.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            p = Path(row.get("image_path") or "")
            if p.exists():
                rows.append(p)
    rng = random.Random(seed)
    rng.shuffle(rows)
    selected = rows[:limit] if limit else rows
    for extra_dir in extra_dirs or []:
        root = extra_dir if extra_dir.is_absolute() else REPO / extra_dir
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                selected.append(p)
    return selected


def image_paths(extra_dirs: list[Path] | None, limit: int, seed: int) -> list[Path]:
    rows: list[Path] = []
    for extra_dir in extra_dirs or []:
        root = extra_dir if extra_dir.is_absolute() else REPO / extra_dir
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                rows.append(p)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit] if limit else rows


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def write_products(paths: list[Path], out: Path, val_frac: float, copy: bool, seed: int) -> dict:
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    n_val = max(1, int(len(paths) * val_frac)) if paths else 0
    summary = {"train": 0, "val": 0}
    for split, items in (("val", paths[:n_val]), ("train", paths[n_val:])):
        for i, src in enumerate(items):
            ext = src.suffix.lower() if src.suffix else ".jpg"
            dst = out / split / "product" / f"product_{i:05d}{ext}"
            link_or_copy(src, dst, copy)
            summary[split] += 1
    return summary


def existing_count(out: Path, split: str, cls: str) -> int:
    root = out / split / cls
    return len([p for p in root.glob("*") if p.is_file() or p.is_symlink()])


def write_negative_images(paths: list[Path], out: Path, val_frac: float,
                          copy: bool, seed: int) -> dict:
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    n_val = max(1, int(len(paths) * val_frac)) if paths else 0
    summary = {"train": 0, "val": 0}
    for split, items in (("val", paths[:n_val]), ("train", paths[n_val:])):
        offset = existing_count(out, split, "interior")
        for i, src in enumerate(items):
            ext = src.suffix.lower() if src.suffix else ".jpg"
            dst = out / split / "interior" / f"interior_extra_{offset + i:05d}{ext}"
            link_or_copy(src, dst, copy)
            summary[split] += 1
    return summary


def camera_roi(width: int, height: int) -> tuple[int, int, int, int]:
    """Approximate the camera preview region in tablet screenrecords."""
    if height > width * 1.15:
        return 0, int(height * 0.18), width, int(height * 0.60)
    return 0, int(height * 0.14), width, int(height * 0.50)


def random_crop(frame, roi, rng: random.Random):
    x0, y0, x1, y1 = roi
    rw, rh = x1 - x0, y1 - y0
    if rw < 64 or rh < 64:
        return None
    area = rng.uniform(0.10, 0.42) * rw * rh
    aspect = rng.uniform(0.55, 1.9)
    cw = int((area * aspect) ** 0.5)
    ch = int((area / aspect) ** 0.5)
    cw = max(48, min(cw, rw))
    ch = max(48, min(ch, rh))
    cx = rng.randint(x0 + cw // 2, x1 - cw // 2)
    cy = rng.randint(y0 + ch // 2, y1 - ch // 2)
    crop = frame[cy - ch // 2:cy + ch // 2, cx - cw // 2:cx + cw // 2]
    if crop.size == 0:
        return None
    return crop


def write_negatives(video_specs: list[VideoSpec], out: Path, limit: int,
                    val_frac: float, seed: int, crops_per_frame: int) -> dict:
    rng = random.Random(seed)
    crops = []
    per_video = {}
    for spec in video_specs:
        path = spec.path if spec.path.is_absolute() else REPO / spec.path
        if not path.exists():
            per_video[str(spec.path)] = {"missing": True, "crops": 0}
            continue
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            per_video[str(spec.path)] = {"opened": False, "crops": 0}
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frames / fps if fps else 0
        start = spec.start if spec.start is not None else 0.0
        end = spec.end if spec.end is not None and spec.end > 0 else duration
        if end <= start:
            end = duration
        frame_count = max(1, min(180, int((end - start) * 5)))
        made = 0
        for idx in range(frame_count):
            if len(crops) >= limit:
                break
            t = start + (idx + 0.5) * (end - start) / frame_count
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            roi = camera_roi(w, h)
            for _ in range(crops_per_frame):
                if len(crops) >= limit:
                    break
                crop = random_crop(frame, roi, rng)
                if crop is None:
                    continue
                crops.append(crop)
                made += 1
        cap.release()
        per_video[str(spec.path)] = {"crops": made, "span": [start, end]}
        if len(crops) >= limit:
            break

    rng.shuffle(crops)
    n_val = max(1, int(len(crops) * val_frac)) if crops else 0
    summary = {"train": 0, "val": 0, "videos": per_video}
    for split, items in (("val", crops[:n_val]), ("train", crops[n_val:])):
        for i, crop in enumerate(items):
            dst = out / split / "interior" / f"interior_{i:05d}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            summary[split] += 1
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--positive-limit", type=int, default=2400)
    ap.add_argument("--negative-limit", type=int, default=2400)
    ap.add_argument("--positive-extra-dir", action="append", type=Path,
                    default=[Path("var/tablet/product_guard_positive")],
                    help="extra real-camera product crops to include")
    ap.add_argument("--negative-video", action="append", default=None,
                    help="video[:start-end] used for interior crops")
    ap.add_argument("--negative-extra-dir", action="append", type=Path,
                    default=DEFAULT_NEGATIVE_EXTRA_DIRS,
                    help="extra curated interior/hard-negative crops to include")
    ap.add_argument("--negative-extra-limit", type=int, default=2400)
    ap.add_argument("--crops-per-frame", type=int, default=12)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--copy-products", action="store_true",
                    help="copy product photos instead of symlinking")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    clean_out(args.out)
    positives = product_paths(
        args.metadata, args.positive_limit, args.seed, args.positive_extra_dir)
    product_summary = write_products(
        positives, args.out, args.val_frac, args.copy_products, args.seed)
    video_specs = [parse_video_spec(v) for v in (args.negative_video or DEFAULT_NEGATIVE_VIDEOS)]
    interior_summary = write_negatives(
        video_specs, args.out, args.negative_limit, args.val_frac,
        args.seed, args.crops_per_frame)
    negative_extra = image_paths(
        args.negative_extra_dir, args.negative_extra_limit, args.seed)
    negative_extra_summary = write_negative_images(
        negative_extra, args.out, args.val_frac, args.copy_products, args.seed)
    summary = {
        "classes": ["interior", "product"],
        "product": product_summary,
        "interior": interior_summary,
        "interior_extra": negative_extra_summary,
        "metadata": str(args.metadata),
        "negative_videos": [str(v) for v in (args.negative_video or DEFAULT_NEGATIVE_VIDEOS)],
        "negative_extra_dirs": [str(v) for v in args.negative_extra_dir],
    }
    (args.out / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Build v8 hard-negative dataset from v8_auto.

v7 added too many device negatives and suppressed real products. This script is
more conservative:

* starts from ``ml/datasets/sku_products_v8_auto``;
* adds a moderate number of empty frames from the user's negative videos;
* mines obvious false-positive background crops from real shelf videos using the
  current v8 candidate, but only when they look like plain floor/wall/cabinet.

All added negatives are empty-label images. The output is safe to train as a
candidate; do not replace production weights until it beats v6/v8_auto on a
hand-count validation set.

Example:
    python3 -m ml.add_v8_hard_negatives \
      --base ml/datasets/sku_products_v8_auto \
      --out ml/datasets/sku_products_v8_hardneg \
      --weights weights/product_det_v8_auto.pt
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import cv2
import numpy as np


DEFAULT_VIDEOS = [
    "ТТ Пэкстрой/IMG_8886.MOV",
    "ТТ Пэкстрой/IMG_8882.MOV",
    "ТТ Пэкстрой/IMG_8883.MOV",
    "ТТ Пэкстрой/IMG_8884.MOV",
    "ООО ВРЕМЕНА ГОДА/IMG_8942.MOV",
    "ЕВРОМИКС/IMG_8916.MOV",
    "ИП Маргарян/IMG_8967.MOV",
]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=Path("ml/datasets/sku_products_v8_auto"))
    ap.add_argument("--out", type=Path, default=Path("ml/datasets/sku_products_v8_hardneg"))
    ap.add_argument("--weights", default="weights/product_det_v8_auto.pt")
    ap.add_argument("--neg-videos", type=Path, default=Path("var/neg_videos"))
    ap.add_argument("--target-device", type=int, default=60)
    ap.add_argument("--target-mined", type=int, default=80)
    ap.add_argument("--mine-frames-per-video", type=int, default=14)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--val-every", type=int, default=7)
    ap.add_argument("--sat", type=float, default=50.0)
    ap.add_argument("--lap", type=float, default=90.0)
    ap.add_argument("--gray-std", type=float, default=38.0)
    ap.add_argument("--low-detail-gray-std", type=float, default=18.0)
    ap.add_argument("--edge-frac", type=float, default=0.03)
    args = ap.parse_args()

    copy_dataset(args.base, args.out)
    saved_device = add_device_negatives(args)
    saved_mined = mine_plain_false_positive_crops(args)
    write_data_yaml(args.out)

    print(f"base={args.base}")
    print(f"added device negatives={saved_device}")
    print(f"added mined negatives={saved_mined}")
    print_dataset_summary(args.out)


def copy_dataset(base: Path, out: Path) -> None:
    if not base.exists():
        raise SystemExit(f"base dataset missing: {base}")
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in iter_images(base / "images" / split):
            shutil.copy2(img, out / "images" / split / img.name)
        for lbl in sorted((base / "labels" / split).glob("*.txt")):
            shutil.copy2(lbl, out / "labels" / split / lbl.name)


def iter_images(root: Path):
    for p in sorted(root.glob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def add_device_negatives(args: argparse.Namespace) -> int:
    vids = sorted(args.neg_videos.rglob("*.mp4"))
    if not vids or args.target_device <= 0:
        return 0
    per_video = max(1, args.target_device // len(vids))
    saved = 0
    for vi, video in enumerate(vids):
        cap = cv2.VideoCapture(str(video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        idxs = np.linspace(total * 0.04, total * 0.96, per_video).astype(int)
        for idx in idxs:
            if saved >= args.target_device:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            split = split_for(saved, args.val_every)
            stem = f"v8devneg_{vi}_{int(idx):06d}"
            save_empty(args.out, split, stem, frame)
            saved += 1
        cap.release()
    return saved


def mine_plain_false_positive_crops(args: argparse.Namespace) -> int:
    if args.target_mined <= 0:
        return 0
    from ultralytics import YOLO

    model = YOLO(args.weights)
    saved = 0
    seen: set[str] = set()
    for video in [Path(v) for v in DEFAULT_VIDEOS if Path(v).exists()]:
        for frame_idx, frame in sample_video_frames(video, args.mine_frames_per_video):
            result = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                   device=args.device, verbose=False)[0]
            if result.boxes is None:
                continue
            h, w = frame.shape[:2]
            for box, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                if saved >= args.target_mined:
                    return saved
                crop = crop_box(frame, box, pad=0.04)
                if crop is None:
                    continue
                if not looks_like_plain_background(
                    crop, sat_thr=args.sat, lap_thr=args.lap,
                    gray_std_thr=args.gray_std,
                    low_detail_gray_std_thr=args.low_detail_gray_std,
                    edge_frac_thr=args.edge_frac,
                ):
                    continue
                # avoid saving near-duplicates from adjacent frames
                digest = image_digest(crop)
                if digest in seen:
                    continue
                seen.add(digest)
                split = split_for(saved, args.val_every)
                stem = f"v8mine_{video.stem}_{frame_idx:06d}_{saved:04d}_{float(score):.2f}"
                save_empty(args.out, split, stem, crop)
                saved += 1
    return saved


def sample_video_frames(video: Path, count: int) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(total * 0.08, total * 0.92, count).astype(int)
    frames = []
    prev_small = None
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        small = cv2.resize(frame, (64, 64)).astype(np.float32)
        if prev_small is not None and np.abs(small - prev_small).mean() < 4.0:
            continue
        prev_small = small
        frames.append((int(idx), frame))
    cap.release()
    return frames


def crop_box(frame: np.ndarray, box, *, pad: float) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    if bw < 32 or bh < 32:
        return None
    area_frac = (bw * bh) / max(1, w * h)
    if area_frac < 0.004 or area_frac > 0.45:
        return None
    x1 = max(0, int(x1 - bw * pad))
    y1 = max(0, int(y1 - bh * pad))
    x2 = min(w, int(x2 + bw * pad))
    y2 = min(h, int(y2 + bh * pad))
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    return frame[y1:y2, x1:x2].copy()


def looks_like_plain_background(crop: np.ndarray, *, sat_thr: float,
                                lap_thr: float, gray_std_thr: float,
                                low_detail_gray_std_thr: float,
                                edge_frac_thr: float) -> bool:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    sat = float(hsv[:, :, 1].mean())
    lap = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    gray_std = float(gray.std())
    edge_frac = float((edges > 0).mean())
    if sat >= sat_thr:
        return False
    plain_large = lap < lap_thr and gray_std < gray_std_thr
    seam_only = gray_std < low_detail_gray_std_thr and edge_frac < edge_frac_thr
    return plain_large or seam_only


def image_digest(crop: np.ndarray) -> str:
    small = cv2.resize(crop, (24, 24), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return hashlib.sha1(gray.tobytes()).hexdigest()[:12]


def split_for(index: int, val_every: int) -> str:
    return "val" if val_every > 0 and index % val_every == 0 else "train"


def save_empty(out: Path, split: str, stem: str, image: np.ndarray) -> None:
    cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), image)
    (out / "labels" / split / f"{stem}.txt").write_text("")


def write_data_yaml(out: Path) -> None:
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: product\n",
        encoding="utf-8",
    )


def print_dataset_summary(root: Path) -> None:
    total_images = 0
    total_labels = 0
    empty = 0
    boxes = 0
    for split in ("train", "val"):
        images = list(iter_images(root / "images" / split))
        labels = list((root / "labels" / split).glob("*.txt"))
        split_boxes = 0
        split_empty = 0
        for label in labels:
            lines = [line for line in label.read_text().splitlines() if line.strip()]
            split_boxes += len(lines)
            split_empty += int(not lines)
        total_images += len(images)
        total_labels += len(labels)
        empty += split_empty
        boxes += split_boxes
        print(f"{split}: images={len(images)} labels={len(labels)} boxes={split_boxes} empty={split_empty}")
    print(f"total: images={total_images} labels={total_labels} boxes={boxes} empty={empty}")


if __name__ == "__main__":
    main()

"""Prepare a product-detector review dataset for Label Studio.

This is the v8 path: instead of asking a person to draw every box from zero, we
sample real shop photos/video frames, pre-fill product boxes with several weak
signals, and send the result to Label Studio for correction. The corrected
export can then be converted with ``convert_labelstudio_to_yolo.py`` and
collapsed to a single-class detector dataset with ``build_product_dataset.py``.

Example:
    python3 -m ml.prepare_product_review_dataset \
      --source . \
      --out ml/datasets/sku_products_v8_review \
      --frames-per-video 30 \
      --max-images 250
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
EXCLUDED_TOP_LEVEL = {
    ".git",
    ".github",
    ".claude",
    "backend",
    "data",
    "dist",
    "docs",
    "ml",
    "mobile",
    "reports",
    "runs",
    "scripts",
    "sku_exact_areas",
    "sku_uncertain_areas",
    "var",
    "weights",
}
BOX_CLASSES = ["product box", "cardboard box", "tool package", "retail product"]
TOOL_CLASSES = [
    "power tool",
    "electric tool",
    "garden tool",
    "chainsaw",
    "trimmer",
    "lawn mower",
    "generator",
    "pump",
    "welding machine",
    "sprayer",
    "pressure washer",
]


@dataclass(frozen=True)
class Proposal:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    source: str

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("ml/datasets/sku_products_v8_review"))
    ap.add_argument("--label-studio-base-url", default="http://localhost:8099")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-images", type=int, default=250)
    ap.add_argument("--frames-per-video", type=int, default=25)
    ap.add_argument("--val-ratio", type=float, default=0.18)
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--weights", default="weights/product_det_v2.pt")
    ap.add_argument("--product-conf", type=float, default=0.18)
    ap.add_argument("--world-conf", type=float, default=0.06)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-proposals", type=int, default=80)
    ap.add_argument("--checkpoint-every", type=int, default=20)
    ap.add_argument("--no-yoloworld", action="store_true")
    ap.add_argument("--use-fastsam", action="store_true",
                    help="add FastSAM proposals; slower and noisier, useful only for recall audits")
    ap.add_argument("--no-fastsam", dest="use_fastsam", action="store_false",
                    help="kept for older commands; FastSAM is disabled by default")
    args = ap.parse_args()

    source = args.source.expanduser().resolve()
    out = args.out.expanduser().resolve()
    rng = random.Random(args.seed)

    images = discover_files(source, IMAGE_SUFFIXES)
    videos = discover_files(source, VIDEO_SUFFIXES)
    rng.shuffle(images)
    selected_images = images[: args.max_images]

    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out / "tasks").mkdir(parents=True, exist_ok=True)

    product_model, world_models, fastsam = load_models(args)

    rows: list[dict[str, str]] = []
    tasks: list[dict] = []
    item_index = 0

    for image_path in selected_images:
        item_index += 1
        split = split_for(item_index, args.val_ratio)
        image_id = build_image_id(source, image_path)
        dst = out / "images" / split / f"{image_id}.jpg"
        convert_image(image_path, dst, max_side=args.max_side)
        add_item(
            out=out,
            rows=rows,
            tasks=tasks,
            index=item_index,
            split=split,
            image_id=image_id,
            image_path=dst,
            source_path=image_path,
            source_root=source,
            base_url=args.label_studio_base_url,
            proposals=predict_proposals(dst, product_model, world_models, fastsam, args),
        )
        if item_index % 10 == 0:
            print(f"[review] prepared {item_index} items", flush=True)
        maybe_checkpoint(out, rows, tasks, args.checkpoint_every)

    for video_path in videos:
        frames = sample_video_frames(video_path, args.frames_per_video)
        for frame_idx, frame in frames:
            item_index += 1
            split = split_for(item_index, args.val_ratio)
            image_id = build_video_frame_id(source, video_path, frame_idx)
            dst = out / "images" / split / f"{image_id}.jpg"
            cv2.imwrite(str(dst), resize_frame(frame, args.max_side))
            add_item(
                out=out,
                rows=rows,
                tasks=tasks,
                index=item_index,
                split=split,
                image_id=image_id,
                image_path=dst,
                source_path=video_path,
                source_root=source,
                base_url=args.label_studio_base_url,
                proposals=predict_proposals(dst, product_model, world_models, fastsam, args),
                frame_idx=frame_idx,
            )
            if item_index % 10 == 0:
                print(f"[review] prepared {item_index} items", flush=True)
            maybe_checkpoint(out, rows, tasks, args.checkpoint_every)

    write_checkpoint(out, rows, tasks)
    print(f"images={len(rows)} videos={len(videos)}", flush=True)
    print(f"tasks -> {out / 'tasks' / 'label_studio_tasks.json'}", flush=True)
    print(f"config -> ml/label_studio_product_config.xml", flush=True)


def discover_files(root: Path, suffixes: set[str]) -> list[Path]:
    result = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if relative.parts and relative.parts[0] in EXCLUDED_TOP_LEVEL:
            continue
        result.append(path)
    return sorted(result)


def load_models(args):
    from ultralytics import YOLO

    product_model = YOLO(args.weights) if Path(args.weights).exists() else None
    world_models = []
    if not args.no_yoloworld and Path("yolov8s-worldv2.pt").exists():
        from ultralytics import YOLOWorld

        box_world = YOLOWorld("yolov8s-worldv2.pt")
        box_world.set_classes(BOX_CLASSES)
        tool_world = YOLOWorld("yolov8s-worldv2.pt")
        tool_world.set_classes(TOOL_CLASSES)
        world_models = [("world-box", box_world), ("world-tool", tool_world)]
    fastsam = None
    if args.use_fastsam and Path("FastSAM-s.pt").exists():
        from ultralytics import FastSAM

        fastsam = FastSAM("FastSAM-s.pt")
    return product_model, world_models, fastsam


def predict_proposals(image_path: Path, product_model, world_models, fastsam, args) -> list[Proposal]:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    width, height = image.size
    proposals: list[Proposal] = []

    if product_model is not None:
        for r in product_model.predict(
            str(image_path), imgsz=args.imgsz, conf=args.product_conf,
            device=args.device, verbose=False,
        ):
            proposals.extend(result_to_proposals(r, width, height, "product-det"))

    for name, model in world_models:
        for r in model.predict(
            str(image_path), imgsz=args.imgsz, conf=args.world_conf,
            device=args.device, verbose=False,
        ):
            proposals.extend(result_to_proposals(r, width, height, name))

    if fastsam is not None:
        try:
            r = fastsam.predict(
                str(image_path), imgsz=1024, conf=0.4, iou=0.9,
                retina_masks=False, device=args.device, verbose=False,
            )[0]
            proposals.extend(result_to_proposals(r, width, height, "fastsam", score=0.45))
        except Exception as exc:  # noqa: BLE001 - keep dataset prep moving
            print(f"FastSAM skipped {image_path.name}: {type(exc).__name__}")

    proposals = filter_proposals(proposals, width, height)
    return nms(proposals, threshold=0.58)[: args.max_proposals]


def result_to_proposals(result, width: int, height: int, source: str,
                        score: float | None = None) -> list[Proposal]:
    if result.boxes is None:
        return []
    out = []
    confs = result.boxes.conf.tolist() if result.boxes.conf is not None else []
    for idx, xyxy in enumerate(result.boxes.xyxy.tolist()):
        x1, y1, x2, y2 = xyxy
        out.append(
            Proposal(
                x1=clip(x1, 0, width),
                y1=clip(y1, 0, height),
                x2=clip(x2, 0, width),
                y2=clip(y2, 0, height),
                score=float(score if score is not None else confs[idx]),
                source=source,
            )
        )
    return out


def filter_proposals(proposals: list[Proposal], width: int, height: int) -> list[Proposal]:
    image_area = width * height
    kept = []
    for p in proposals:
        if p.width < 24 or p.height < 24:
            continue
        area_frac = p.area / max(1, image_area)
        if area_frac < 0.0006 or area_frac > 0.42:
            continue
        aspect = p.width / max(1.0, p.height)
        if aspect < 0.08 or aspect > 8.5:
            continue
        kept.append(p)
    return kept


def nms(proposals: list[Proposal], threshold: float) -> list[Proposal]:
    remaining = sorted(proposals, key=lambda p: p.score, reverse=True)
    kept: list[Proposal] = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        remaining = [candidate for candidate in remaining if iou(current, candidate) < threshold]
    return sorted(kept, key=lambda p: (p.y1, p.x1))


def iou(a: Proposal, b: Proposal) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def add_item(
    *,
    out: Path,
    rows: list[dict[str, str]],
    tasks: list[dict],
    index: int,
    split: str,
    image_id: str,
    image_path: Path,
    source_path: Path,
    source_root: Path,
    base_url: str,
    proposals: list[Proposal],
    frame_idx: int | None = None,
) -> None:
    relative_image = image_path.relative_to(out).as_posix()
    relative_label = f"labels/{split}/{image_id}.txt"
    label_path = out / relative_label
    label_path.write_text(yolo_lines(image_path, proposals), encoding="utf-8")
    rows.append(
        {
            "image_id": image_id,
            "split": split,
            "store_name": infer_store_name(source_root, source_path) or "",
            "source_path": str(source_path),
            "frame_idx": "" if frame_idx is None else str(frame_idx),
            "image_path": relative_image,
            "label_path": relative_label,
        }
    )
    tasks.append(label_studio_task(index, image_id, image_path, out, relative_image,
                                   source_path, base_url, proposals, frame_idx))


def yolo_lines(image_path: Path, proposals: list[Proposal]) -> str:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    width, height = image.size
    lines = []
    for p in proposals:
        cx = (p.x1 + p.x2) / 2 / width
        cy = (p.y1 + p.y2) / 2 / height
        bw = p.width / width
        bh = p.height / height
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def label_studio_task(index: int, image_id: str, image_path: Path, out: Path,
                      relative_image: str, source_path: Path, base_url: str,
                      proposals: list[Proposal], frame_idx: int | None) -> dict:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    width, height = image.size
    result = []
    for pidx, p in enumerate(proposals):
        result.append(
            {
                "id": f"{image_id}_{pidx}",
                "from_name": "bbox",
                "to_name": "image",
                "type": "rectanglelabels",
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "value": {
                    "x": 100.0 * p.x1 / width,
                    "y": 100.0 * p.y1 / height,
                    "width": 100.0 * p.width / width,
                    "height": 100.0 * p.height / height,
                    "rotation": 0,
                    "rectanglelabels": ["product"],
                },
                "score": p.score,
            }
        )
    return {
        "id": index,
        "data": {
            "image": f"{base_url.rstrip('/')}/{relative_image}",
            "image_id": image_id,
            "source_path": str(source_path),
            "frame_idx": frame_idx,
        },
        "predictions": [
            {
                "model_version": "product-review-v8-seed",
                "score": float(sum(p.score for p in proposals) / len(proposals)) if proposals else 0.0,
                "result": result,
            }
        ],
    }


def sample_video_frames(video_path: Path, count: int) -> list[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(total * 0.05, total * 0.95, count).astype(int)
    frames = []
    prev_small = None
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        small = cv2.resize(frame, (64, 64)).astype(np.float32)
        if prev_small is not None and np.abs(small - prev_small).mean() < 5.0:
            continue
        prev_small = small
        frames.append((int(idx), frame))
    cap.release()
    return frames


def resize_frame(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(1.0, max_side / max(w, h))
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def convert_image(source: Path, target: Path, *, max_side: int) -> None:
    if source.suffix.lower() == ".heic":
        run(["sips", "-Z", str(max_side), "-s", "format", "jpeg", str(source), "--out", str(target)])
        return
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        image.thumbnail((max_side, max_side))
        image.convert("RGB").save(target, format="JPEG", quality=92)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["image_id", "split", "store_name", "source_path", "frame_idx",
                  "image_path", "label_path"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_checkpoint(out: Path, rows: list[dict[str, str]], tasks: list[dict],
                     checkpoint_every: int) -> None:
    if checkpoint_every <= 0:
        return
    if rows and len(rows) % checkpoint_every == 0:
        write_checkpoint(out, rows, tasks)


def write_checkpoint(out: Path, rows: list[dict[str, str]], tasks: list[dict]) -> None:
    write_manifest(out / "manifest.csv", rows)
    write_data_yaml(out / "data.yaml", out)
    (out / "tasks").mkdir(parents=True, exist_ok=True)
    (out / "tasks" / "label_studio_tasks.json").write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_data_yaml(path: Path, output: Path) -> None:
    path.write_text(
        f"path: {output.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: product\n",
        encoding="utf-8",
    )


def build_image_id(root: Path, path: Path) -> str:
    relative = path.relative_to(root).as_posix()
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:10]
    return f"{sanitize(path.stem)}_{digest}"


def build_video_frame_id(root: Path, path: Path, frame_idx: int) -> str:
    relative = f"{path.relative_to(root).as_posix()}#{frame_idx}"
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:10]
    return f"vid_{sanitize(path.stem)}_{frame_idx:06d}_{digest}"


def infer_store_name(root: Path, path: Path) -> str | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    if len(relative.parts) < 2:
        return None
    top = relative.parts[0]
    return None if top in EXCLUDED_TOP_LEVEL else top


def split_for(index: int, val_ratio: float) -> str:
    if val_ratio <= 0:
        return "train"
    every = max(2, round(1.0 / val_ratio))
    return "val" if index % every == 0 else "train"


def sanitize(value: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")
    return safe or "item"


def clip(value: float, low: float, high: float) -> float:
    if math.isnan(float(value)):
        return low
    return max(low, min(high, float(value)))


def run(command: list[str]) -> None:
    if shutil.which(command[0]) is None:
        raise RuntimeError(f"Missing command: {command[0]}")
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


if __name__ == "__main__":
    main()

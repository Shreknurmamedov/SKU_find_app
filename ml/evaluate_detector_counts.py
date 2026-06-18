"""Evaluate product-detector object-count quality on a YOLO dataset split.

This is a detector diagnostic, not the final business metric. The product now
cares about SKU presence (unique models/articles), so use
``ml.evaluate_sku_coverage`` for end-to-end validation.

This measures the detector itself, before tracking/OCR:

  * how many product boxes are predicted vs labeled;
  * TP/FP/FN by IoU matching;
  * count error, precision and recall.

Use it on the reviewed v8 validation split. This does not measure SKU coverage
or OCR quality.

Example:
    python3 -m ml.evaluate_detector_counts \
      --data ml/datasets/sku_products_v8/data.yaml \
      --weights weights/product_det_v2.pt \
      --split val \
      --conf 0.35 \
      --device mps
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class ImageEval:
    image: str
    gt: int
    pred: int
    tp: int
    fp: int
    fn: int
    count_error: float


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou-match", type=float, default=0.5)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", type=Path, default=Path("reports/detector_eval.json"))
    args = ap.parse_args()

    dataset = read_data_yaml(args.data)
    images_dir = dataset / "images" / args.split
    labels_dir = dataset / "labels" / args.split
    images = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"No images found: {images_dir}")

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    rows: list[ImageEval] = []
    for image_path in images:
        gt = read_gt_boxes(labels_dir / f"{image_path.stem}.txt")
        pred = predict_boxes(model, image_path, imgsz=args.imgsz,
                             conf=args.conf, device=args.device)
        tp, fp, fn = match_counts(gt, pred, args.iou_match)
        true_count = len(gt)
        pred_count = len(pred)
        count_error = abs(pred_count - true_count) / true_count if true_count else float(pred_count > 0)
        rows.append(ImageEval(
            image=image_path.name,
            gt=true_count,
            pred=pred_count,
            tp=tp,
            fp=fp,
            fn=fn,
            count_error=count_error,
        ))

    summary = summarize(rows, args)
    payload = {"summary": summary, "images": [asdict(row) for row in rows]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, args.out.with_suffix(".md"))
    print(f"images={summary['images']} gt={summary['gt']} pred={summary['pred']} "
          f"count_error={summary['count_error']:.3f} precision={summary['precision']:.3f} "
          f"recall={summary['recall']:.3f}")
    print(f"report -> {args.out}")


def read_data_yaml(path: Path) -> Path:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(data["path"])
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    return root


def read_gt_boxes(path: Path) -> np.ndarray:
    boxes = []
    if not path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            cx, cy, w, h = [float(v) for v in parts[1:]]
            boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        elif len(parts) >= 7:
            values = [float(v) for v in parts[1:]]
            xs = values[0::2]
            ys = values[1::2]
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
    return np.asarray(boxes, dtype=np.float32)


def predict_boxes(model, image_path: Path, *, imgsz: int, conf: float, device: str) -> np.ndarray:
    result = model.predict(str(image_path), imgsz=imgsz, conf=conf,
                           device=device, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    h, w = result.orig_shape
    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    boxes[:, [0, 2]] /= max(1, w)
    boxes[:, [1, 3]] /= max(1, h)
    boxes = np.clip(boxes, 0.0, 1.0)
    return boxes


def match_counts(gt: np.ndarray, pred: np.ndarray, iou_thr: float) -> tuple[int, int, int]:
    if len(gt) == 0:
        return 0, len(pred), 0
    if len(pred) == 0:
        return 0, 0, len(gt)
    ious = iou_matrix(pred, gt)
    order = np.dstack(np.unravel_index(np.argsort(ious.ravel())[::-1], ious.shape))[0]
    matched_pred = set()
    matched_gt = set()
    for pi, gi in order:
        if ious[pi, gi] < iou_thr:
            break
        if int(pi) in matched_pred or int(gi) in matched_gt:
            continue
        matched_pred.add(int(pi))
        matched_gt.add(int(gi))
    tp = len(matched_gt)
    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros((len(a), len(b)), dtype=np.float32)
    for i, pa in enumerate(a):
        ax1, ay1, ax2, ay2 = pa
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        for j, gb in enumerate(b):
            bx1, by1, bx2, by2 = gb
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            out[i, j] = inter / (area_a + area_b - inter + 1e-6)
    return out


def summarize(rows: list[ImageEval], args: argparse.Namespace) -> dict:
    gt = sum(r.gt for r in rows)
    pred = sum(r.pred for r in rows)
    tp = sum(r.tp for r in rows)
    fp = sum(r.fp for r in rows)
    fn = sum(r.fn for r in rows)
    return {
        "data": str(args.data),
        "weights": str(args.weights),
        "split": args.split,
        "images": len(rows),
        "gt": gt,
        "pred": pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "count_error": abs(pred - gt) / gt if gt else float(pred > 0),
        "mean_image_count_error": sum(r.count_error for r in rows) / len(rows),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def write_markdown(payload: dict, path: Path) -> None:
    s = payload["summary"]
    lines = [
        "# Detector Count Evaluation",
        "",
        f"- Dataset: `{s['data']}`",
        f"- Weights: `{s['weights']}`",
        f"- Split: `{s['split']}`",
        f"- Images: **{s['images']}**",
        f"- GT boxes: **{s['gt']}**",
        f"- Pred boxes: **{s['pred']}**",
        f"- Count error: **{s['count_error']:.1%}**",
        f"- Mean image count error: **{s['mean_image_count_error']:.1%}**",
        f"- Precision: **{s['precision']:.1%}**",
        f"- Recall: **{s['recall']:.1%}**",
        "",
        "## Worst Images",
        "",
        "| Image | GT | Pred | TP | FP | FN | Count error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    worst = sorted(payload["images"], key=lambda r: (r["count_error"], r["fp"] + r["fn"]), reverse=True)[:30]
    for r in worst:
        lines.append(f"| {r['image']} | {r['gt']} | {r['pred']} | {r['tp']} | "
                     f"{r['fp']} | {r['fn']} | {r['count_error']:.1%} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Densify the stage-1 dataset with auto-labeled frames from the shelf videos.

Recall is limited by the training labels, so we add in-domain video frames and
pseudo-label each as a UNION of two complementary detectors:

  * our trained product detector (knows product appearance, low conf for recall);
  * YOLO-World on generic box/tool prompts (catches packages the detector misses).

Boxes are merged with class-agnostic NMS and written as single-class `product`
labels. The result is copied alongside the existing sku_products dataset into a
fresh dataset dir for retraining.

    python3 ml/expand_dataset_with_video.py --frames-per-video 10
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import cv2
import numpy as np

from ml.auto_label_dataset import BOX_CLASSES, UNBOXED_PRODUCT_CLASSES

BASE = Path("ml/datasets/sku_products")
OUT = Path("ml/datasets/sku_products_v2")
VIDEOS = [
    "ТТ Пэкстрой/IMG_8886.MOV", "ТТ Пэкстрой/IMG_8882.MOV",
    "ТТ Пэкстрой/IMG_8883.MOV", "ТТ Пэкстрой/IMG_8884.MOV",
    "ООО ВРЕМЕНА ГОДА/IMG_8942.MOV", "ЕВРОМИКС/IMG_8916.MOV",
    "ИП Маргарян/IMG_8967.MOV",
]


def iou_matrix(boxes: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    n = len(boxes)
    iou = np.zeros((n, n), np.float32)
    for i in range(n):
        xx1 = np.maximum(x1[i], x1); yy1 = np.maximum(y1[i], y1)
        xx2 = np.minimum(x2[i], x2); yy2 = np.minimum(y2[i], y2)
        w = np.clip(xx2 - xx1, 0, None); h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        iou[i] = inter / (area[i] + area - inter + 1e-6)
    return iou


def nms_agnostic(boxes: np.ndarray, scores: np.ndarray, thr=0.55) -> list[int]:
    order = scores.argsort()[::-1]
    keep = []
    suppressed = np.zeros(len(boxes), bool)
    iou = iou_matrix(boxes)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= iou[i] > thr
        suppressed[i] = True
    return keep


def extract_frames(video: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(total * 0.1, total * 0.9, n).astype(int)
    frames, prev = [], None
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = cap.read()
        if not ok:
            continue
        small = cv2.resize(fr, (64, 64)).astype(np.float32)
        if prev is not None and np.abs(small - prev).mean() < 6:  # near-duplicate
            continue
        prev = small
        frames.append(fr)
    cap.release()
    return frames


def pseudo_label(frame, det, world_box, world_tool, *, imgsz, det_conf, world_conf):
    H, W = frame.shape[:2]
    boxes, scores = [], []
    for r in det.predict(frame, imgsz=imgsz, conf=det_conf, verbose=False, device="mps"):
        if r.boxes is not None:
            boxes.append(r.boxes.xyxy.cpu().numpy()); scores.append(r.boxes.conf.cpu().numpy())
    for model in (world_box, world_tool):
        for r in model.predict(frame, imgsz=imgsz, conf=world_conf, verbose=False, device="mps"):
            if r.boxes is not None and len(r.boxes):
                boxes.append(r.boxes.xyxy.cpu().numpy()); scores.append(r.boxes.conf.cpu().numpy())
    if not boxes:
        return []
    boxes = np.vstack(boxes); scores = np.concatenate(scores)
    keep = nms_agnostic(boxes, scores)
    out = []
    for i in keep:
        x1, y1, x2, y2 = boxes[i]
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        bw, bh = (x2 - x1) / W, (y2 - y1) / H
        if bw <= 0 or bh <= 0:
            continue
        out.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-per-video", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--det-conf", type=float, default=0.15)
    ap.add_argument("--world-conf", type=float, default=0.05)
    ap.add_argument("--weights", default="weights/product_det.pt")
    ap.add_argument("--val-every", type=int, default=6, help="put every Nth frame in val")
    args = ap.parse_args()

    from ultralytics import YOLO, YOLOWorld
    det = YOLO(args.weights)
    world_box = YOLOWorld("yolov8s-worldv2.pt"); world_box.set_classes(BOX_CLASSES)
    world_tool = YOLOWorld("yolov8s-worldv2.pt"); world_tool.set_classes(UNBOXED_PRODUCT_CLASSES)

    # start from a copy of the base dataset
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in (BASE / "images" / split).glob("*"):
            shutil.copy2(img, OUT / "images" / split / img.name)
        for lbl in (BASE / "labels" / split).glob("*"):
            shutil.copy2(lbl, OUT / "labels" / split / lbl.name)

    added = labeled = 0
    fi = 0
    for vid in VIDEOS:
        vp = Path(vid)
        if not vp.exists():
            print(f"skip missing {vid}"); continue
        frames = extract_frames(vp, args.frames_per_video)
        for fr in frames:
            labels = pseudo_label(fr, det, world_box, world_tool,
                                  imgsz=args.imgsz, det_conf=args.det_conf,
                                  world_conf=args.world_conf)
            split = "val" if fi % args.val_every == 0 else "train"
            stem = f"vid_{vp.stem}_{hashlib.md5(fr.tobytes()).hexdigest()[:8]}"
            cv2.imwrite(str(OUT / "images" / split / f"{stem}.jpg"), fr)
            (OUT / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(labels) + ("\n" if labels else ""))
            added += 1
            labeled += len(labels)
            fi += 1
        print(f"{vid}: +{len(frames)} frames")

    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: product\n")
    n_train = len(list((OUT / "images/train").glob("*")))
    n_val = len(list((OUT / "images/val").glob("*")))
    print(f"\nadded {added} video frames ({labeled} pseudo-boxes)")
    print(f"dataset v2: train={n_train} val={n_val} -> {OUT/'data.yaml'}")


if __name__ == "__main__":
    main()

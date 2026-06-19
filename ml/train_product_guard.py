"""Train the tablet product-vs-interior guard classifier.

This trains a small YOLO classification model on the dataset produced by
``ml.build_product_guard_dataset`` and writes ``weights/product_guard_cls.pt``.
The Android app exports that weight to TFLite and runs it on every YOLO crop.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("ml/datasets/product_guard_cls"))
    ap.add_argument("--base", default="yolo11n-cls.pt")
    ap.add_argument("--out", type=Path, default=Path("weights/product_guard_cls.pt"))
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--project", default="runs/classify")
    ap.add_argument("--name", default="product_guard")
    args = ap.parse_args()

    if not (args.data / "train" / "product").exists():
        raise SystemExit(f"dataset not found: {args.data}; run ml.build_product_guard_dataset first")

    model = YOLO(args.base)
    res = model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=4,
        plots=True,
        verbose=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    best = Path(res.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"training finished but best.pt was not found: {best}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.out)
    print(f"GUARD_WEIGHTS: {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

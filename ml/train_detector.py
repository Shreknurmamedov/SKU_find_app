"""Train the single-class product detector (stage 1).

    python3 ml/train_detector.py --data ml/datasets/sku_products/data.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("ml/datasets/sku_products/data.yaml"))
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--project", default="ml/runs")
    ap.add_argument("--name", default="product_det")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        task="detect",
        patience=args.patience,
        # chaotic shelves: lots of small overlapping boxes
        mosaic=1.0,
        close_mosaic=10,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0, fliplr=0.5,
        verbose=True,
    )


if __name__ == "__main__":
    main()

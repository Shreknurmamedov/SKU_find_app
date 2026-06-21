"""Export the product detector to TFLite and drop it into the Android assets."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

DEFAULT_WEIGHTS = Path("weights/product_det_v2.pt")
DEFAULT_ASSET = Path("mobile/android/app/src/main/assets/models/product_det_v2_float32.tflite")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"detector weights not found: {args.weights}")

    model = YOLO(str(args.weights))
    exported = Path(model.export(
        format="tflite", imgsz=args.imgsz, nms=False, half=False, device=args.device))

    tflite = exported if exported.suffix == ".tflite" and exported.exists() else None
    if tflite is None:
        export_root = exported if exported.exists() and exported.is_dir() else args.weights.parent
        candidates = (list(export_root.rglob("*float32.tflite"))
                      or list(export_root.rglob("*.tflite")))
        tflite = candidates[0] if candidates else None
    if tflite is None or not tflite.exists():
        raise SystemExit("TFLite export did not produce a .tflite file")

    args.asset.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tflite, args.asset)
    print(f"ASSET: {args.asset} ({args.asset.stat().st_size} bytes) from {tflite}")


if __name__ == "__main__":
    main()

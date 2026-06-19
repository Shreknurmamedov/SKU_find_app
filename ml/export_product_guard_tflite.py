"""Export the binary product guard classifier to Android TFLite assets."""
from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO


WEIGHTS = Path("weights/product_guard_cls.pt")
ASSET = Path("mobile/android/app/src/main/assets/models/product_guard_cls_float32.tflite")


def main() -> None:
    if not WEIGHTS.exists():
        raise SystemExit(f"guard weights not found: {WEIGHTS}")
    model = YOLO(str(WEIGHTS))
    exported = Path(model.export(format="tflite", imgsz=224, half=False))
    tflite = exported if exported.suffix == ".tflite" and exported.exists() else None
    if tflite is None:
        candidates = (list(Path("weights").rglob("*guard*float32.tflite"))
                      or list(Path("weights").rglob("*guard*.tflite"))
                      or list(Path("weights").rglob("*.tflite")))
        tflite = candidates[0] if candidates else None
    if tflite is None or not tflite.exists():
        raise SystemExit("TFLite export did not produce a .tflite file")
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tflite, ASSET)
    print(f"GUARD_ASSET: {ASSET} ({ASSET.stat().st_size} bytes) from {tflite}")


if __name__ == "__main__":
    main()

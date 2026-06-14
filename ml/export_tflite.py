"""Export the product detector to TFLite and drop it into the Android assets.

Runs in CI (Ubuntu x86_64) where ultralytics' TF export toolchain works cleanly.
Produces a float32 TFLite at 320x320 and copies it to the app assets so it gets
bundled into the APK for on-device live detection.
"""
import shutil
from pathlib import Path

from ultralytics import YOLO

WEIGHTS = "weights/product_det_v2.pt"
ASSET = Path("mobile/android/app/src/main/assets/models/product_det_v2_float32.tflite")


def main() -> None:
    model = YOLO(WEIGHTS)
    exported = Path(model.export(format="tflite", imgsz=320, nms=False, half=False))

    tflite = exported if exported.suffix == ".tflite" and exported.exists() else None
    if tflite is None:
        candidates = (list(Path("weights").rglob("*float32.tflite"))
                      or list(Path("weights").rglob("*.tflite")))
        tflite = candidates[0] if candidates else None
    if tflite is None or not tflite.exists():
        raise SystemExit("TFLite export did not produce a .tflite file")

    ASSET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tflite, ASSET)
    print(f"ASSET: {ASSET} ({ASSET.stat().st_size} bytes) from {tflite}")


if __name__ == "__main__":
    main()

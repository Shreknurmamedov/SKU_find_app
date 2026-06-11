from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--format", default="tflite", choices=["tflite", "onnx"])
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--android-assets", type=Path, default=Path("mobile/android/app/src/main/assets/models"))
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install ML deps first: python3 -m pip install -r ml/requirements.txt") from exc

    model = YOLO(str(args.weights))
    exported = Path(model.export(format=args.format, imgsz=args.imgsz))
    args.android_assets.mkdir(parents=True, exist_ok=True)
    target = args.android_assets / exported.name
    shutil.copy2(exported, target)
    print(target)


if __name__ == "__main__":
    main()

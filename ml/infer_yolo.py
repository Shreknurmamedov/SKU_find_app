from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--project", default="ml/runs")
    parser.add_argument("--name", default="predict")
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install ML deps first: python3 -m pip install -r ml/requirements.txt") from exc

    model = YOLO(str(args.weights))
    model.predict(
        source=str(args.source),
        project=args.project,
        name=args.name,
        conf=args.conf,
        save=True,
        task="segment",
    )


if __name__ == "__main__":
    main()

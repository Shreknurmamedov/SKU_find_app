from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CLASS_TO_ID = {"own_product": 0, "competitor_or_unknown": 1, "product": 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", required=True, type=Path, help="Label Studio JSON export")
    parser.add_argument("--dataset", required=True, type=Path, help="Prepared dataset folder")
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    manifest = load_manifest(dataset / "manifest.csv")
    payload = json.loads(args.export.read_text(encoding="utf-8"))

    written = 0
    missing = 0
    for task in payload:
        image_id = (task.get("data") or {}).get("image_id")
        if not image_id or image_id not in manifest:
            missing += 1
            continue
        label_path = dataset / manifest[image_id]["label_path"]
        lines = extract_yolo_lines(task)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1

    print(f"Updated labels for {written} tasks")
    if missing:
        print(f"Skipped {missing} tasks without manifest match")


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = csv.DictReader(file)
        return {row["image_id"]: row for row in rows}


def extract_yolo_lines(task: dict) -> list[str]:
    results = []
    for annotation in task.get("annotations", []):
        results.extend(annotation.get("result", []))

    lines = []
    for result in results:
        value = result.get("value") or {}
        if "polygonlabels" in value:
            label = value["polygonlabels"][0]
            cls = CLASS_TO_ID[label]
            points = value.get("points") or []
            if len(points) < 3:
                continue
            coords = []
            for x, y in points:
                coords.extend([x / 100.0, y / 100.0])
            lines.append(format_yolo_line(cls, coords))
        elif "rectanglelabels" in value:
            label = value["rectanglelabels"][0]
            cls = CLASS_TO_ID[label]
            x = value["x"] / 100.0
            y = value["y"] / 100.0
            width = value["width"] / 100.0
            height = value["height"] / 100.0
            lines.append(
                format_yolo_line(
                    cls,
                    [
                        x,
                        y,
                        x + width,
                        y,
                        x + width,
                        y + height,
                        x,
                        y + height,
                    ],
                )
            )
    return lines


def format_yolo_line(cls: int, values: list[float]) -> str:
    return " ".join([str(cls), *[f"{value:.6f}" for value in values]])


if __name__ == "__main__":
    main()

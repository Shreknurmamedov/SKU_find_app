from __future__ import annotations

import argparse
from pathlib import Path

from auto_label_dataset import (
    apply_manual_overrides,
    load_manifest,
    load_manual_overrides,
    read_yolo_labels,
    write_yolo_labels,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply manual label overrides to an existing prepared dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("ml/datasets/sku_live"))
    parser.add_argument("--manual-overrides", type=Path, default=Path("ml/manual_overrides.json"))
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    rows_by_id = {row["image_id"]: row for row in load_manifest(dataset / "manifest.csv")}
    overrides = load_manual_overrides(args.manual_overrides)

    applied_total = 0
    for image_id in sorted(overrides):
        row = rows_by_id.get(image_id)
        if not row:
            print(f"Skipped unknown image_id: {image_id}")
            continue

        image_path = dataset / row["image_path"]
        label_path = dataset / row["label_path"]
        labels = read_yolo_labels(label_path, image_path)
        labels, applied = apply_manual_overrides(image_id, labels, image_path, overrides)
        write_yolo_labels(label_path, labels, image_path)
        applied_total += applied
        print(f"{image_id}: applied {applied} override(s)")

    print(f"Applied {applied_total} manual override(s)")


if __name__ == "__main__":
    main()

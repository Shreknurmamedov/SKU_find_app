from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageOps
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic"}
EXCLUDED_TOP_LEVEL = {
    ".git",
    ".github",
    "backend",
    "data",
    "dist",
    "docs",
    "ml",
    "mobile",
    "reports",
    "scripts",
    "var",
}
CLASSES = {0: "own_product", 1: "competitor_or_unknown"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--label-studio-base-url", default="http://localhost:8099")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.out.expanduser().resolve()
    paths = discover_images(source)
    random.Random(args.seed).shuffle(paths)

    val_count = max(1, int(len(paths) * args.val_ratio)) if paths else 0
    val_paths = set(paths[:val_count])

    rows = []
    tasks = []
    for index, image_path in enumerate(paths, start=1):
        split = "val" if image_path in val_paths else "train"
        image_id = build_image_id(source, image_path)
        target_image = output / "images" / split / f"{image_id}.jpg"
        target_label = output / "labels" / split / f"{image_id}.txt"
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)

        convert_to_jpeg(image_path, target_image, max_side=args.max_side)
        target_label.touch(exist_ok=True)

        relative_image = target_image.relative_to(output).as_posix()
        relative_label = target_label.relative_to(output).as_posix()
        store_name = infer_store_name(source, image_path)
        rows.append(
            {
                "image_id": image_id,
                "split": split,
                "store_name": store_name or "",
                "source_path": str(image_path),
                "image_path": relative_image,
                "label_path": relative_label,
            }
        )
        tasks.append(
            {
                "id": index,
                "data": {
                    "image": f"{args.label_studio_base_url.rstrip('/')}/{relative_image}",
                    "image_id": image_id,
                    "split": split,
                    "store_name": store_name,
                    "source_path": str(image_path),
                    "label_path": relative_label,
                },
            }
        )

    write_manifest(output / "manifest.csv", rows)
    write_data_yaml(output / "data.yaml", output)
    (output / "tasks").mkdir(parents=True, exist_ok=True)
    (output / "tasks" / "label_studio_tasks.json").write_text(
        json.dumps(tasks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Prepared {len(paths)} images")
    print(output)
    print(output / "data.yaml")
    print(output / "tasks" / "label_studio_tasks.json")


def discover_images(source: Path) -> list[Path]:
    result = []
    for path in source.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            relative = path.relative_to(source)
        except ValueError:
            continue
        if relative.parts and relative.parts[0] in EXCLUDED_TOP_LEVEL:
            continue
        result.append(path)
    return sorted(result)


def convert_to_jpeg(source: Path, target: Path, *, max_side: int) -> None:
    if target.exists():
        return
    if source.suffix.lower() == ".heic":
        run(["sips", "-Z", str(max_side), "-s", "format", "jpeg", str(source), "--out", str(target)])
        return

    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        image.thumbnail((max_side, max_side))
        image.convert("RGB").save(target, format="JPEG", quality=92)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "split", "store_name", "source_path", "image_path", "label_path"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_data_yaml(path: Path, output: Path) -> None:
    payload = {
        "path": str(output),
        "train": "images/train",
        "val": "images/val",
        "names": CLASSES,
    }
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def build_image_id(root: Path, path: Path) -> str:
    relative = path.relative_to(root).as_posix()
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:10]
    stem = sanitize(path.stem)
    return f"{stem}_{digest}"


def infer_store_name(root: Path, path: Path) -> str | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    if len(relative.parts) < 2:
        return None
    top = relative.parts[0]
    return None if top in EXCLUDED_TOP_LEVEL else top


def sanitize(value: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")
    return safe or "image"


def run(command: list[str]) -> None:
    if shutil.which(command[0]) is None:
        raise RuntimeError(f"Missing command: {command[0]}")
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


if __name__ == "__main__":
    main()

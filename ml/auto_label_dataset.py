from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageOps


OWN_PRODUCT = 0
COMPETITOR_OR_UNKNOWN = 1

BOX_CLASSES = [
    "product box",
    "cardboard box",
    "tool package",
]

UNBOXED_PRODUCT_CLASSES = [
    "pressure washer",
    "high pressure washer",
    "portable pressure washer",
    "power washer",
    "cleaning machine",
    "power drill",
    "chainsaw",
    "electric tool",
    "power tool",
    "garden trimmer",
    "garden tool",
    "lawn mower",
    "generator",
    "pump",
    "welding machine",
    "sprayer",
    "battery charger",
    "air compressor",
    "hose",
    "hose reel",
]
WORLD_CLASSES = BOX_CLASSES

CONFUSABLES = str.maketrans(
    {
        "А": "A",
        "В": "B",
        "Е": "E",
        "К": "K",
        "М": "M",
        "Н": "H",
        "О": "O",
        "Р": "P",
        "С": "C",
        "Т": "T",
        "У": "Y",
        "Х": "X",
        "а": "A",
        "в": "B",
        "е": "E",
        "к": "K",
        "м": "M",
        "н": "H",
        "о": "O",
        "р": "P",
        "с": "C",
        "т": "T",
        "у": "Y",
        "х": "X",
    }
)

OWN_TEXT_NEEDLES = {
    "HUTER",
    "HYTER",
    "HATEK",
    "HUTEP",
    "RESANTA",
    "PECANTA",
    "PECAHTA",
    "ВИХ",
    "BИX",
    "ВИХР",
    "VIHR",
    "VIXR",
    "EUROLUX",
    "ЕВРОЛЮКС",
}

SHORT_OWN_WORDS = {"TEK", "ТЕК"}


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    prompt: str

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True)
class OcrText:
    box: tuple[float, float, float, float]
    text: str
    confidence: float


@dataclass(frozen=True)
class Label:
    cls: int
    box: Box
    reason: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate weak YOLO-seg labels for SKU shelf photos.")
    parser.add_argument("--dataset", type=Path, default=Path("ml/datasets/sku_live"))
    parser.add_argument("--conf", type=float, default=0.06, help="Confidence for boxed/package detections.")
    parser.add_argument("--unboxed-conf", type=float, default=0.02, help="Confidence for visible unpacked products.")
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preview-count", type=int, default=24)
    parser.add_argument("--preview-dir", type=Path, default=Path("reports/auto_label_preview"))
    parser.add_argument("--label-studio-output", type=Path, default=Path("ml/datasets/sku_live/tasks/auto_predictions.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/auto_label_summary.json"))
    parser.add_argument("--manual-overrides", type=Path, default=Path("ml/manual_overrides.json"))
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--no-prune-stale", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    rows = load_manifest(dataset / "manifest.csv")
    if args.limit:
        rows = rows[: args.limit]

    from ultralytics import YOLOWorld

    box_world = YOLOWorld("yolov8s-worldv2.pt")
    box_world.set_classes(BOX_CLASSES)
    unboxed_world = YOLOWorld("yolov8s-worldv2.pt")
    unboxed_world.set_classes(UNBOXED_PRODUCT_CLASSES)

    reader = None
    if not args.skip_ocr:
        import easyocr

        reader = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)

    if not args.no_prune_stale and not args.limit:
        prune_stale_dataset_files(dataset, rows)
    clear_label_files(dataset, rows)

    tasks_by_image_id = load_tasks(dataset / "tasks" / "label_studio_tasks.json")
    manual_overrides = load_manual_overrides(args.manual_overrides)
    all_predictions = []
    summary = {
        "images": 0,
        "images_with_labels": 0,
        "own_product": 0,
        "competitor_or_unknown": 0,
        "total_labels": 0,
        "manual_overrides": 0,
        "ocr_images": 0,
        "ocr_own_hits": 0,
        "empty_images": 0,
    }
    preview_images: list[tuple[Path, list[Label]]] = []

    for index, row in enumerate(rows, start=1):
        image_path = dataset / row["image_path"]
        label_path = dataset / row["label_path"]
        labels = label_image(
            image_path=image_path,
            box_world=box_world,
            unboxed_world=unboxed_world,
            reader=reader,
            conf=args.conf,
            unboxed_conf=args.unboxed_conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
        )
        labels, manual_count = apply_manual_overrides(row["image_id"], labels, image_path, manual_overrides)
        write_yolo_labels(label_path, labels, image_path)

        task = tasks_by_image_id.get(row["image_id"])
        if task:
            all_predictions.append(build_label_studio_prediction(task, labels, image_path))

        summary["images"] += 1
        summary["total_labels"] += len(labels)
        summary["own_product"] += sum(label.cls == OWN_PRODUCT for label in labels)
        summary["competitor_or_unknown"] += sum(label.cls == COMPETITOR_OR_UNKNOWN for label in labels)
        summary["manual_overrides"] += manual_count
        summary["ocr_images"] += int(reader is not None and len(labels) > 0)
        summary["ocr_own_hits"] += sum(label.cls == OWN_PRODUCT for label in labels)
        if labels:
            summary["images_with_labels"] += 1
            if len(preview_images) < args.preview_count:
                preview_images.append((image_path, labels))
        else:
            summary["empty_images"] += 1

        print(
            f"[{index:03d}/{len(rows):03d}] {row['image_id']}: "
            f"{len(labels)} labels ({sum(label.cls == OWN_PRODUCT for label in labels)} own)"
        )

    args.label_studio_output.parent.mkdir(parents=True, exist_ok=True)
    args.label_studio_output.write_text(
        json.dumps(all_predictions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_previews(args.preview_dir, preview_images)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_tasks(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    tasks = json.loads(path.read_text(encoding="utf-8"))
    return {task["data"]["image_id"]: task for task in tasks if task.get("data", {}).get("image_id")}


def load_manual_overrides(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manual overrides must be a JSON object: {path}")
    return payload


def clear_label_files(dataset: Path, rows: list[dict[str, str]]) -> None:
    wanted = {row["label_path"] for row in rows}
    for label_path in (dataset / "labels").glob("*/*.txt"):
        relative = label_path.relative_to(dataset).as_posix()
        if relative in wanted:
            label_path.write_text("", encoding="utf-8")


def prune_stale_dataset_files(dataset: Path, rows: list[dict[str, str]]) -> None:
    wanted_images = {row["image_path"] for row in rows}
    wanted_labels = {row["label_path"] for row in rows}
    removed = 0
    for folder, wanted in ((dataset / "images", wanted_images), (dataset / "labels", wanted_labels)):
        for path in folder.glob("*/*"):
            if not path.is_file():
                continue
            relative = path.relative_to(dataset).as_posix()
            if relative not in wanted:
                path.unlink()
                removed += 1
    if removed:
        print(f"Pruned {removed} stale dataset files")


def label_image(
    *,
    image_path: Path,
    box_world,
    unboxed_world,
    reader,
    conf: float,
    unboxed_conf: float,
    iou: float,
    imgsz: int,
    device: str,
) -> list[Label]:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    width, height = image.size
    box_result = box_world.predict(str(image_path), imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)[0]
    package_boxes = extract_boxes(box_result, width, height, BOX_CLASSES)
    package_boxes = filter_boxes(package_boxes, width, height)
    package_boxes = nms(package_boxes, threshold=0.65)

    unboxed_result = unboxed_world.predict(
        str(image_path),
        imgsz=imgsz,
        conf=unboxed_conf,
        iou=iou,
        device=device,
        verbose=False,
    )[0]
    unboxed_boxes = extract_boxes(unboxed_result, width, height, UNBOXED_PRODUCT_CLASSES)
    unboxed_boxes = filter_unboxed_boxes(unboxed_boxes, width, height)
    unboxed_boxes = remove_unboxed_inside_packages(unboxed_boxes, package_boxes)
    unboxed_boxes = nms(unboxed_boxes, threshold=0.55)

    boxes = nms(package_boxes + unboxed_boxes, threshold=0.75)
    if not boxes:
        return []

    ocr_texts = read_ocr(reader, np.array(image)) if reader is not None else []
    labels = []
    for box in boxes:
        texts = texts_inside_box(ocr_texts, box)
        is_own, reason = has_own_brand(texts)
        cls = OWN_PRODUCT if is_own else COMPETITOR_OR_UNKNOWN
        labels.append(Label(cls=cls, box=box, reason=reason if is_own else f"world:{box.prompt}"))
    labels.extend(build_own_text_anchor_labels(ocr_texts, labels, width, height))
    return labels


def extract_boxes(result, width: int, height: int, prompts: list[str]) -> list[Box]:
    if result.boxes is None:
        return []
    boxes = []
    for xyxy, confidence, cls in zip(
        result.boxes.xyxy.tolist(),
        result.boxes.conf.tolist(),
        result.boxes.cls.tolist(),
    ):
        x1, y1, x2, y2 = xyxy
        boxes.append(
            Box(
                x1=clip(x1, 0, width),
                y1=clip(y1, 0, height),
                x2=clip(x2, 0, width),
                y2=clip(y2, 0, height),
                confidence=float(confidence),
                prompt=prompts[int(cls)],
            )
        )
    return boxes


def filter_boxes(boxes: Iterable[Box], width: int, height: int) -> list[Box]:
    image_area = width * height
    result = []
    for box in boxes:
        if box.width < 35 or box.height < 35:
            continue
        area_ratio = box.area / image_area
        if area_ratio < 0.0015 or area_ratio > 0.45:
            continue
        aspect = box.width / max(1.0, box.height)
        if aspect < 0.12 or aspect > 8.0:
            continue
        result.append(box)
    return result


def filter_unboxed_boxes(boxes: Iterable[Box], width: int, height: int) -> list[Box]:
    image_area = width * height
    result = []
    for box in boxes:
        if box.width < 45 or box.height < 45:
            continue
        area_ratio = box.area / image_area
        if area_ratio < 0.002 or area_ratio > 0.5:
            continue
        aspect = box.width / max(1.0, box.height)
        if aspect < 0.08 or aspect > 7.5:
            continue
        result.append(box)
    return result


def remove_unboxed_inside_packages(unboxed_boxes: list[Box], package_boxes: list[Box]) -> list[Box]:
    result = []
    for unboxed in unboxed_boxes:
        if any(is_small_box_inside(unboxed, package) for package in package_boxes):
            continue
        result.append(unboxed)
    return result


def is_small_box_inside(inner: Box, outer: Box) -> bool:
    center_x = (inner.x1 + inner.x2) / 2
    center_y = (inner.y1 + inner.y2) / 2
    center_inside = outer.x1 <= center_x <= outer.x2 and outer.y1 <= center_y <= outer.y2
    return center_inside and inner.area < outer.area * 0.55


def nms(boxes: list[Box], threshold: float) -> list[Box]:
    remaining = sorted(boxes, key=lambda box: box.confidence, reverse=True)
    kept: list[Box] = []
    while remaining:
        current = remaining.pop(0)
        kept.append(current)
        remaining = [candidate for candidate in remaining if box_iou(current, candidate) < threshold]
    return sorted(kept, key=lambda box: (box.y1, box.x1))


def box_iou(a: Box, b: Box) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def read_ocr(reader, image: np.ndarray) -> list[OcrText]:
    if reader is None:
        return []
    try:
        horizontal, free = reader.detect(image, min_size=5, text_threshold=0.4, low_text=0.25)
    except Exception:
        return []

    height, width = image.shape[:2]
    horizontal_boxes = []
    for x1, x2, y1, y2 in horizontal[0] if horizontal else []:
        x1 = int(clip(x1, 0, width - 1))
        x2 = int(clip(x2, 0, width))
        y1 = int(clip(y1, 0, height - 1))
        y2 = int(clip(y2, 0, height))
        if x2 - x1 >= 4 and y2 - y1 >= 4:
            horizontal_boxes.append([x1, x2, y1, y2])

    gray = np.asarray(Image.fromarray(image).convert("L"))
    try:
        results = reader.recognize(
            gray,
            horizontal_list=horizontal_boxes,
            free_list=[],
            detail=1,
            paragraph=False,
            batch_size=8,
            reformat=False,
        )
    except Exception:
        return []

    texts = []
    for polygon, text, confidence in results:
        if not text or confidence < 0.08:
            continue
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        texts.append(OcrText(box=(min(xs), min(ys), max(xs), max(ys)), text=str(text), confidence=float(confidence)))
    return texts


def texts_inside_box(texts: list[OcrText], box: Box) -> list[OcrText]:
    expanded = expand_box(box, margin=0.08)
    result = []
    for text in texts:
        x1, y1, x2, y2 = text.box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        if expanded.x1 <= cx <= expanded.x2 and expanded.y1 <= cy <= expanded.y2:
            result.append(text)
    return result


def expand_box(box: Box, margin: float) -> Box:
    dx = box.width * margin
    dy = box.height * margin
    return Box(box.x1 - dx, box.y1 - dy, box.x2 + dx, box.y2 + dy, box.confidence, box.prompt)


def has_own_brand(texts: list[OcrText]) -> tuple[bool, str]:
    for text in texts:
        normalized = normalize_text(text.text)
        if any(needle in normalized for needle in OWN_TEXT_NEEDLES):
            return True, f"ocr:{text.text}"
        words = set(re.findall(r"[A-ZА-ЯЁ]{2,}", normalized))
        if words & SHORT_OWN_WORDS and text.confidence >= 0.45:
            return True, f"ocr:{text.text}"
    return False, ""


def build_own_text_anchor_labels(
    texts: list[OcrText],
    existing: list[Label],
    width: int,
    height: int,
) -> list[Label]:
    anchors = []
    for text in texts:
        is_own, reason = has_own_brand([text])
        if not is_own:
            continue
        anchor = ocr_anchor_box(text, width, height)
        if not anchor:
            continue
        if any(box_iou(anchor, label.box) > 0.35 for label in existing):
            continue
        if any(box_iou(anchor, label.box) > 0.35 for label in anchors):
            continue
        anchors.append(Label(cls=OWN_PRODUCT, box=anchor, reason=reason))
    return anchors


def ocr_anchor_box(text: OcrText, width: int, height: int) -> Box | None:
    x1, y1, x2, y2 = text.box
    text_width = x2 - x1
    text_height = y2 - y1
    if text_width < 18 or text_height < 10:
        return None

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    box_width = clip(text_width * 1.8, 70, 220)
    box_height = clip(text_height * 3.4, 60, 180)
    anchor = Box(
        x1=clip(center_x - box_width / 2, 0, width),
        y1=clip(center_y - box_height * 0.45, 0, height),
        x2=clip(center_x + box_width / 2, 0, width),
        y2=clip(center_y + box_height * 0.55, 0, height),
        confidence=text.confidence,
        prompt="ocr-own-anchor",
    )
    if anchor.width < 50 or anchor.height < 50:
        return None
    return anchor


def normalize_text(text: str) -> str:
    upper = text.upper()
    mapped = upper.translate(CONFUSABLES)
    compact = re.sub(r"[^A-ZА-ЯЁ0-9]+", "", mapped)
    return compact


def write_yolo_labels(path: Path, labels: list[Label], image_path: Path) -> None:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    width, height = image.size
    lines = []
    for label in labels:
        coords = box_to_normalized_polygon(label.box, width, height)
        lines.append(" ".join([str(label.cls), *[f"{value:.6f}" for value in coords]]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_yolo_labels(path: Path, image_path: Path) -> list[Label]:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    width, height = image.size
    labels = []
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 9:
            continue
        cls = int(parts[0])
        values = [float(value) for value in parts[1:]]
        xs = values[0::2]
        ys = values[1::2]
        labels.append(
            Label(
                cls=cls,
                box=Box(
                    x1=min(xs) * width,
                    y1=min(ys) * height,
                    x2=max(xs) * width,
                    y2=max(ys) * height,
                    confidence=1.0,
                    prompt="existing-label",
                ),
                reason="existing-label",
            )
        )
    return labels


def apply_manual_overrides(
    image_id: str,
    labels: list[Label],
    image_path: Path,
    overrides: dict[str, list[dict]],
) -> tuple[list[Label], int]:
    entries = overrides.get(image_id) or []
    if not entries:
        return labels, 0

    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    width, height = image.size
    result = list(labels)
    applied = 0
    for entry in entries:
        manual_label = manual_entry_to_label(entry, width, height)
        duplicate = any(label.cls == manual_label.cls and box_iou(label.box, manual_label.box) > 0.65 for label in result)
        if duplicate:
            continue
        result = [
            label
            for label in result
            if not (label.cls != manual_label.cls and box_iou(label.box, manual_label.box) > 0.35)
        ]
        result.append(manual_label)
        applied += 1
    return sorted(result, key=lambda label: (label.box.y1, label.box.x1)), applied


def manual_entry_to_label(entry: dict, width: int, height: int) -> Label:
    cls = parse_class_id(entry.get("class"))
    box = entry.get("box")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"Manual override requires box [x1, y1, x2, y2]: {entry}")

    x1, y1, x2, y2 = [float(value) for value in box]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height

    x1, x2 = sorted((clip(x1, 0, width), clip(x2, 0, width)))
    y1, y2 = sorted((clip(y1, 0, height), clip(y2, 0, height)))
    if x2 - x1 < 5 or y2 - y1 < 5:
        raise ValueError(f"Manual override box is too small: {entry}")

    return Label(
        cls=cls,
        box=Box(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            confidence=float(entry.get("confidence", 1.0)),
            prompt="manual-override",
        ),
        reason=str(entry.get("reason", "manual-override")),
    )


def parse_class_id(value) -> int:
    if value in (OWN_PRODUCT, "own_product", "own", "green"):
        return OWN_PRODUCT
    if value in (COMPETITOR_OR_UNKNOWN, "competitor_or_unknown", "unknown", "red"):
        return COMPETITOR_OR_UNKNOWN
    raise ValueError(f"Unknown class in manual override: {value}")


def box_to_normalized_polygon(box: Box, width: int, height: int) -> list[float]:
    points = [
        (box.x1, box.y1),
        (box.x2, box.y1),
        (box.x2, box.y2),
        (box.x1, box.y2),
    ]
    coords = []
    for x, y in points:
        coords.extend([clip(x / width, 0.0, 1.0), clip(y / height, 0.0, 1.0)])
    return coords


def build_label_studio_prediction(task: dict, labels: list[Label], image_path: Path) -> dict:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    width, height = image.size
    results = []
    for label in labels:
        box = label.box
        results.append(
            {
                "from_name": "bbox",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": 100.0 * box.x1 / width,
                    "y": 100.0 * box.y1 / height,
                    "width": 100.0 * box.width / width,
                    "height": 100.0 * box.height / height,
                    "rotation": 0,
                    "rectanglelabels": ["own_product" if label.cls == OWN_PRODUCT else "competitor_or_unknown"],
                },
                "score": label.box.confidence,
            }
        )
    return {
        "data": task["data"],
        "predictions": [
            {
                "model_version": "yoloworld-easyocr-weak-v1",
                "score": average([label.box.confidence for label in labels]),
                "result": results,
            }
        ],
    }


def write_previews(output_dir: Path, images: list[tuple[Path, list[Label]]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (image_path, labels) in enumerate(images, start=1):
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        for label in labels:
            color = (36, 182, 103, 60) if label.cls == OWN_PRODUCT else (230, 58, 58, 55)
            outline = (36, 182, 103, 255) if label.cls == OWN_PRODUCT else (230, 58, 58, 255)
            box = label.box
            draw.rectangle([box.x1, box.y1, box.x2, box.y2], fill=color, outline=outline, width=4)
            title = "own" if label.cls == OWN_PRODUCT else "unknown"
            draw.text((box.x1 + 4, box.y1 + 4), f"{title} {box.confidence:.2f}", fill=outline)
        image.thumbnail((1000, 1000))
        image.save(output_dir / f"{index:02d}_{image_path.stem}.jpg", quality=92)


def average(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def clip(value: float, low: float, high: float) -> float:
    if math.isnan(float(value)):
        return low
    return max(low, min(high, float(value)))


if __name__ == "__main__":
    main()

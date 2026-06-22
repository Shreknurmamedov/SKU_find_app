"""Run the real SKU-presence ML pipeline on a job's uploaded videos.

The heavy model (YOLO detector + ByteTrack + OCR/catalog match) lives in the
repo-root ``ml`` package and pulls in torch/ultralytics/easyocr. To keep the
FastAPI process light and avoid import-path coupling, we invoke it as a
subprocess (``python -m ml.audit_video``) per video, then read back the JSON
report it writes and aggregate unique SKU/model/article presence across videos.

Recognition runs in a background thread so the upload request returns immediately;
the job's ``sku_status`` moves pending -> processing -> done/failed and the
``sku_report`` is filled in when finished. The Android client polls the job.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
DEFAULT_WEIGHTS = "weights/product_det_v2.pt"


def repo_root() -> Path:
    # backend/src/sku_audit/sku_count.py -> repo root
    return Path(__file__).resolve().parents[3]


def _device() -> str:
    return os.environ.get("SKU_AUDIT_DEVICE", "mps")


def _weights() -> str:
    return os.environ.get("SKU_AUDIT_WEIGHTS", DEFAULT_WEIGHTS)


def _read_report_json(path: Path) -> dict[str, Any]:
    for encoding in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _feedback_label(path: Path) -> str | None:
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            label = str(payload.get("label") or "").strip().lower()
            if label:
                return label
        except Exception:
            pass
    name = path.name.lower()
    if "hardneg" in name or "hard_negative" in name:
        return "hard_negative"
    if "product" in name:
        return "product"
    return None


def _feedback_images(feedback_dir: Path, label: str) -> list[Path]:
    if not feedback_dir.exists():
        return []
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return [
        path for path in feedback_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in suffixes
        and _feedback_label(path) == label
    ]


def _image_signature(path: Path) -> tuple[int, Any] | None:
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as image:
            rgb = image.convert("RGB")
            gray = np.asarray(rgb.resize((9, 8)).convert("L"), dtype=np.int16)
            diff = gray[:, 1:] > gray[:, :-1]
            dhash = 0
            for bit in diff.reshape(-1):
                dhash = (dhash << 1) | int(bool(bit))

            arr = np.asarray(rgb.resize((64, 64)), dtype=np.float32).reshape(-1, 3)
            hist, _ = np.histogramdd(
                arr,
                bins=(8, 8, 8),
                range=((0, 256), (0, 256), (0, 256)),
            )
            hist = hist.astype(np.float32).reshape(-1)
            norm = float(np.linalg.norm(hist))
            if norm > 0:
                hist /= norm
            return dhash, hist
    except Exception:
        return None


def _resolve_item_crop(item: dict[str, Any], job_dir: Path) -> Path | None:
    crop = item.get("crop")
    if not crop:
        return None
    path = Path(str(crop).replace("\\", "/"))
    candidates = [path] if path.is_absolute() else [
        job_dir / "sku" / path,
        job_dir / path,
        repo_root() / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _best_feedback_match(
    signature: tuple[int, Any],
    negatives: list[tuple[Path, tuple[int, Any]]],
) -> dict[str, Any] | None:
    import numpy as np

    dhash, hist = signature
    best: dict[str, Any] | None = None
    for feedback_path, (neg_dhash, neg_hist) in negatives:
        dhash_distance = int((dhash ^ neg_dhash).bit_count())
        hist_similarity = float(np.dot(hist, neg_hist))
        matched = (
            dhash_distance <= 8
            or hist_similarity >= 0.97
            or (dhash_distance <= 18 and hist_similarity >= 0.90)
        )
        if not matched:
            continue
        score = hist_similarity - dhash_distance / 64.0
        if best is None or score > best["score"]:
            best = {
                "feedback": feedback_path,
                "dhash_distance": dhash_distance,
                "hist_similarity": round(hist_similarity, 4),
                "score": score,
            }
    return best


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _collapse_sku_presence(items: list[dict[str, Any]], review_conf: float) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for item in items:
        if item.get("status") != "matched_sku" or _float(item.get("ocr_conf")) < review_conf:
            continue
        key = item.get("sku_id") or f"{item.get('brand') or ''}|{item.get('model') or ''}"
        row = by_key.setdefault(key, {
            "sku_key": key,
            "sku_id": item.get("sku_id"),
            "brand": item.get("brand"),
            "model": item.get("model"),
            "article_codes": item.get("article_codes"),
            "category": item.get("category"),
            "is_own": bool(item.get("is_own")),
            "evidence_objects": 0,
            "object_ids": [],
            "best_ocr_conf": 0.0,
            "best_crop": None,
        })
        row["evidence_objects"] += 1
        row["object_ids"].append(item.get("object_id"))
        if _float(item.get("ocr_conf")) > _float(row.get("best_ocr_conf")):
            row["best_ocr_conf"] = item.get("ocr_conf")
            row["best_crop"] = item.get("crop")
    return sorted(by_key.values(), key=lambda row: (
        row.get("brand") or "", row.get("model") or "", row.get("article_codes") or "",
    ))


def _collapse_brand_category_presence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for item in items:
        brand = item.get("brand")
        category = item.get("category")
        if item.get("status") == "matched_sku" or not brand or not category:
            continue
        key = f"{brand}|{category}"
        row = by_key.setdefault(key, {
            "brand": brand,
            "category": category,
            "evidence_objects": 0,
            "object_ids": [],
            "best_ocr_conf": 0.0,
            "best_crop": None,
        })
        row["evidence_objects"] += 1
        row["object_ids"].append(item.get("object_id"))
        if _float(item.get("ocr_conf")) > _float(row.get("best_ocr_conf")):
            row["best_ocr_conf"] = item.get("ocr_conf")
            row["best_crop"] = item.get("crop")
    return sorted(by_key.values(), key=lambda row: (
        row.get("brand") or "", row.get("category") or "",
    ))


def _recompute_report_from_items(report: dict[str, Any], *, excluded_count: int) -> None:
    review_conf = _float(report.get("review_conf"), 0.75)
    items = list(report.get("items") or [])
    review = [
        item for item in items
        if item.get("status") != "matched_sku" or _float(item.get("ocr_conf")) < review_conf
    ]
    confident = [
        item for item in items
        if item.get("status") == "matched_sku" and _float(item.get("ocr_conf")) >= review_conf
    ]
    sku_presence = _collapse_sku_presence(items, review_conf)
    brand_category_presence = _collapse_brand_category_presence(items)

    report["sku_presence"] = sku_presence
    report["brand_category_presence"] = brand_category_presence
    report["needs_review"] = review
    report["by_brand"] = dict(Counter(sku.get("brand") or "-" for sku in sku_presence).most_common())
    report["by_category"] = dict(Counter(
        sku.get("category") for sku in sku_presence if sku.get("category")
    ).most_common())
    report["by_model"] = dict(Counter(_model_label(sku) for sku in sku_presence).most_common())
    report["by_brand_objects"] = dict(Counter(item.get("brand") or "-" for item in items).most_common())
    report["by_category_objects"] = dict(Counter(
        item.get("category") for item in items if item.get("category")
    ).most_common())
    report["by_model_objects"] = dict(Counter(
        item.get("model") for item in items if item.get("model")
    ).most_common())
    report["totals"] = {
        **(report.get("totals") or {}),
        "unique_skus": len(sku_presence),
        "unique_own_skus": sum(1 for sku in sku_presence if sku.get("is_own")),
        "candidate_objects": len(items),
        "physical_objects": len(items),
        "own_brand_objects": sum(1 for item in items if item.get("is_own")),
        "competitor_or_unknown": sum(1 for item in items if not item.get("is_own")),
        "sku_evidence_objects": len(confident),
        "confident_sku": len(sku_presence),
        "needs_review_objects": len(review),
        "needs_review": len(review),
        "brand_category_partial": len(brand_category_presence),
        "brand_not_visible_objects": sum(1 for item in items if item.get("status") == "unknown"),
        "brand_not_visible": sum(1 for item in items if item.get("status") == "unknown"),
        "excluded_by_feedback": excluded_count,
    }


def apply_feedback_exclusions(
    report: dict[str, Any],
    *,
    job_dir: Path,
    upload_dir: Path,
) -> dict[str, Any]:
    feedback_dir = upload_dir / "feedback"
    negatives = [
        (path, signature)
        for path in _feedback_images(feedback_dir, "hard_negative")
        if (signature := _image_signature(path)) is not None
    ]
    if not negatives or not report.get("items"):
        return report

    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for item in report.get("items") or []:
        crop_path = _resolve_item_crop(item, job_dir)
        signature = _image_signature(crop_path) if crop_path is not None else None
        match = _best_feedback_match(signature, negatives) if signature is not None else None
        if match is None:
            kept.append(item)
            continue
        feedback_path = match["feedback"]
        try:
            feedback_label = str(feedback_path.relative_to(feedback_dir))
        except ValueError:
            feedback_label = str(feedback_path)
        excluded.append({
            "object_id": item.get("object_id"),
            "crop": item.get("crop"),
            "matched_feedback": feedback_label,
            "dhash_distance": match["dhash_distance"],
            "hist_similarity": match["hist_similarity"],
        })

    if not excluded:
        return report
    filtered = {**report, "items": kept}
    filtered["feedback_exclusions"] = list(report.get("feedback_exclusions") or []) + excluded
    _recompute_report_from_items(filtered, excluded_count=len(excluded))
    return filtered


def run_video_audit(video_path: Path, job_dir: Path, *, timeout: int = 3600) -> dict[str, Any]:
    """Audit a single video, returning the parsed report JSON."""
    root = repo_root()
    weights = _weights()
    if not (root / weights).exists():
        raise FileNotFoundError(
            f"Detector weights not found: {weights}. Train/copy product_det_v2.pt first."
        )
    reports_dir = job_dir / "sku"
    work_dir = job_dir / "sku" / "track"
    reports_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "ml.audit_video",
        "--video", str(video_path),
        "--weights", weights,
        "--reports-dir", str(reports_dir),
        "--work-dir", str(work_dir),
        "--device", _device(),
    ]
    subprocess.run(cmd, cwd=str(root), check=True, capture_output=True, timeout=timeout)
    report_path = reports_dir / f"audit_{video_path.stem}.json"
    return _read_report_json(report_path)


def aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    by_brand_objects = Counter()
    by_category_objects = Counter()
    by_model_objects = Counter()
    sku_by_key: dict[str, dict[str, Any]] = {}
    brand_cat_by_key: dict[str, dict[str, Any]] = {}
    per_video = []
    for r in reports:
        t = r.get("totals", {})
        for k, v in t.items():
            totals[k] += v
        by_brand_objects.update(r.get("by_brand_objects", r.get("by_brand", {})))
        by_category_objects.update(r.get("by_category_objects", r.get("by_category", {})))
        by_model_objects.update(r.get("by_model_objects", {}))
        for sku in r.get("sku_presence", []):
            key = sku.get("sku_key") or sku.get("sku_id") or f"{sku.get('brand')}|{sku.get('model')}"
            current = sku_by_key.get(key)
            if current is None:
                current = {**sku, "videos": [], "evidence_objects": 0, "object_ids": []}
                sku_by_key[key] = current
            current["videos"].append(Path(r.get("video", "")).name)
            current["evidence_objects"] += int(sku.get("evidence_objects", 0))
            current["object_ids"].extend(sku.get("object_ids", []))
            if float(sku.get("best_ocr_conf", 0.0)) > float(current.get("best_ocr_conf", 0.0)):
                current["best_ocr_conf"] = sku.get("best_ocr_conf", 0.0)
                current["best_crop"] = sku.get("best_crop")
        for bc in r.get("brand_category_presence", []):
            key = f"{bc.get('brand') or ''}|{bc.get('category') or ''}"
            current = brand_cat_by_key.get(key)
            if current is None:
                current = {**bc, "videos": [], "evidence_objects": 0, "object_ids": []}
                brand_cat_by_key[key] = current
            current["videos"].append(Path(r.get("video", "")).name)
            current["evidence_objects"] += int(bc.get("evidence_objects", 0))
            current["object_ids"].extend(bc.get("object_ids", []))
            if float(bc.get("best_ocr_conf", 0.0)) > float(current.get("best_ocr_conf", 0.0)):
                current["best_ocr_conf"] = bc.get("best_ocr_conf", 0.0)
                current["best_crop"] = bc.get("best_crop")
        per_video.append({"video": Path(r.get("video", "")).name, "totals": t})

    sku_presence = sorted(sku_by_key.values(), key=lambda s: (
        s.get("brand") or "", s.get("model") or "", s.get("article_codes") or "",
    ))
    brand_category_presence = sorted(brand_cat_by_key.values(), key=lambda s: (
        s.get("brand") or "￿", s.get("category") or "",
    ))
    by_brand = Counter(sku.get("brand") or "—" for sku in sku_presence)
    by_category = Counter(sku.get("category") for sku in sku_presence if sku.get("category"))
    by_model = Counter(
        _model_label(sku) for sku in sku_presence
    )
    totals["unique_skus"] = len(sku_presence)
    totals["unique_own_skus"] = sum(1 for sku in sku_presence if sku.get("is_own"))
    totals["confident_sku"] = len(sku_presence)
    # Deduped across videos, overriding the naive per-video sum.
    totals["brand_category_partial"] = len(brand_category_presence)

    return {
        "videos": len(reports),
        "totals": dict(totals),
        "by_brand": dict(by_brand.most_common()),
        "by_category": dict(by_category.most_common()),
        "by_model": dict(by_model.most_common()),
        "by_brand_objects": dict(by_brand_objects.most_common()),
        "by_category_objects": dict(by_category_objects.most_common()),
        "by_model_objects": dict(by_model_objects.most_common()),
        "sku_presence": sku_presence,
        "brand_category_presence": brand_category_presence,
        "per_video": per_video,
    }


def _model_label(sku: dict[str, Any]) -> str:
    brand = sku.get("brand") or "—"
    model = sku.get("model") or "—"
    article = sku.get("article_codes") or sku.get("sku_id") or ""
    return f"{brand} {model}" + (f" ({article})" if article else "")


def process_job(job_store, job_id: str) -> None:
    """Background entry point: count SKUs for every video in the job."""
    upload_dir = job_store.uploads_dir / job_id
    job_dir = job_store.jobs_dir / job_id
    videos = sorted(
        p for p in upload_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    ) if upload_dir.exists() else []

    if not videos:
        job_store.update_sku(job_id, "skipped", None,
                             note="Нет видео для распознавания SKU (загружены только фото).")
        return

    job_store.update_sku(job_id, "processing", None,
                         note=f"Распознавание SKU: {len(videos)} видео...")
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    for video in videos:
        try:
            report = run_video_audit(video, job_dir)
            reports.append(apply_feedback_exclusions(
                report,
                job_dir=job_dir,
                upload_dir=upload_dir,
            ))
        except subprocess.CalledProcessError as exc:
            tail = (exc.stderr or b"").decode("utf-8", "replace")[-500:]
            errors.append(f"{video.name}: ошибка модели ({tail.strip()[:200]})")
        except Exception as exc:  # noqa: BLE001 - surface any failure to the client
            errors.append(f"{video.name}: {exc}")

    if not reports:
        job_store.update_sku(job_id, "failed", None,
                             note="; ".join(errors) or "Подсчёт не дал результата.")
        return

    report = aggregate(reports)
    if errors:
        report["errors"] = errors
    job_store.update_sku(job_id, "done", report,
                         note="Распознавание SKU завершено.")

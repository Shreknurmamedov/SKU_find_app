"""Run the real SKU-counting ML pipeline on a job's uploaded videos.

The heavy model (YOLO detector + ByteTrack + OCR/catalog match) lives in the
repo-root ``ml`` package and pulls in torch/ultralytics/easyocr. To keep the
FastAPI process light and avoid import-path coupling, we invoke it as a
subprocess (``python -m ml.audit_video``) per video, then read back the JSON
report it writes and aggregate across videos.

Counting runs in a background thread so the upload request returns immediately;
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
    return json.loads(report_path.read_text(encoding="utf-8"))


def aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    by_brand = Counter()
    by_category = Counter()
    by_model = Counter()
    per_video = []
    for r in reports:
        t = r.get("totals", {})
        for k, v in t.items():
            totals[k] += v
        by_brand.update(r.get("by_brand", {}))
        by_category.update(r.get("by_category", {}))
        by_model.update(r.get("by_model", {}))
        per_video.append({"video": Path(r.get("video", "")).name, "totals": t})
    return {
        "videos": len(reports),
        "totals": dict(totals),
        "by_brand": dict(by_brand.most_common()),
        "by_category": dict(by_category.most_common()),
        "by_model": dict(by_model.most_common()),
        "per_video": per_video,
    }


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
                             note="Нет видео для подсчёта SKU (загружены только фото).")
        return

    job_store.update_sku(job_id, "processing", None,
                         note=f"Подсчёт SKU: {len(videos)} видео...")
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    for video in videos:
        try:
            reports.append(run_video_audit(video, job_dir))
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
                         note="Подсчёт SKU завершён.")

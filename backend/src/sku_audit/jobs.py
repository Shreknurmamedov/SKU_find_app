from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from sku_audit.media import MediaInspection, discover_media_paths, inspect_media_file
from sku_audit.models import MediaKind, QualityStatus

INTERNAL_TOP_LEVEL_DIRS = {
    "backend",
    "data",
    "docs",
    "mobile",
    "reports",
    "sku_exact_areas",
    "sku_uncertain_areas",
    "var",
}


@dataclass(frozen=True)
class ProcessingJob:
    job_id: str
    status: str
    created_at: str
    store_name: str | None
    source: str
    media: list[MediaInspection]
    summary: dict[str, Any]
    report_markdown: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "media": [item.to_dict() for item in self.media],
        }


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.uploads_dir = self.root / "uploads"
        self.previews_dir = self.root / "previews"
        self.jobs_dir = self.root / "jobs"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.previews_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def create_from_local_folder(self, input_root: Path, *, store_name: str | None = None) -> ProcessingJob:
        input_root = input_root.expanduser().resolve()
        media_paths = discover_media_paths(input_root)
        return self._create_job(
            media_paths,
            root=input_root,
            source=f"local_folder:{input_root}",
            store_name=store_name,
        )

    def create_from_uploaded_paths(
        self, uploaded_paths: Iterable[Path], *, store_name: str | None = None
    ) -> ProcessingJob:
        job_id = _new_job_id()
        upload_dir = self.uploads_dir / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        copied_paths = []
        for path in uploaded_paths:
            destination = upload_dir / _safe_filename(path.name)
            shutil.copy2(path, destination)
            copied_paths.append(destination)
        return self._create_job(
            copied_paths,
            root=upload_dir,
            source="upload",
            store_name=store_name,
            job_id=job_id,
        )

    def save_uploaded_bytes(
        self, files: Iterable[tuple[str, bytes]], *, store_name: str | None = None
    ) -> ProcessingJob:
        job_id = _new_job_id()
        upload_dir = self.uploads_dir / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for filename, content in files:
            destination = upload_dir / _safe_filename(filename)
            destination.write_bytes(content)
            paths.append(destination)
        return self._create_job(
            paths,
            root=upload_dir,
            source="upload",
            store_name=store_name,
            job_id=job_id,
        )

    def get(self, job_id: str) -> ProcessingJob:
        path = self.jobs_dir / job_id / "job.json"
        if not path.exists():
            raise FileNotFoundError(job_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        media = [_media_from_dict(item) for item in payload["media"]]
        return ProcessingJob(
            job_id=payload["job_id"],
            status=payload["status"],
            created_at=payload["created_at"],
            store_name=payload.get("store_name"),
            source=payload["source"],
            media=media,
            summary=payload["summary"],
            report_markdown=payload["report_markdown"],
            notes=payload.get("notes", []),
        )

    def list_jobs(self) -> list[dict[str, Any]]:
        jobs = []
        for path in sorted(self.jobs_dir.glob("*/job.json"), reverse=True):
            payload = json.loads(path.read_text(encoding="utf-8"))
            jobs.append(
                {
                    "job_id": payload["job_id"],
                    "status": payload["status"],
                    "created_at": payload["created_at"],
                    "store_name": payload.get("store_name"),
                    "source": payload["source"],
                    "summary": payload["summary"],
                }
            )
        return jobs

    def _create_job(
        self,
        media_paths: Iterable[Path],
        *,
        root: Path,
        source: str,
        store_name: str | None,
        job_id: str | None = None,
    ) -> ProcessingJob:
        job_id = job_id or _new_job_id()
        job_dir = self.jobs_dir / job_id
        preview_root = self.previews_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        inspections = []
        for path in media_paths:
            inferred_store = store_name or _infer_store_name(root, path)
            inspections.append(
                inspect_media_file(path, root=root, preview_root=preview_root, store_name=inferred_store)
            )
        summary = summarize_media(inspections)
        notes = _job_notes(summary)
        report_markdown = render_job_markdown(job_id, source, store_name, inspections, summary, notes)
        job = ProcessingJob(
            job_id=job_id,
            status="completed",
            created_at=datetime.now(timezone.utc).isoformat(),
            store_name=store_name,
            source=source,
            media=inspections,
            summary=summary,
            report_markdown=report_markdown,
            notes=notes,
        )
        (job_dir / "job.json").write_text(
            json.dumps(job.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (job_dir / "report.md").write_text(report_markdown, encoding="utf-8")
        return job


def summarize_media(media: Iterable[MediaInspection]) -> dict[str, Any]:
    items = list(media)
    images = [item for item in items if item.media_kind == MediaKind.IMAGE]
    videos = [item for item in items if item.media_kind == MediaKind.VIDEO]
    retake = [item for item in items if item.quality_status == QualityStatus.RETAKE]
    warnings = [item for item in items if item.quality_status == QualityStatus.WARNING]
    ok = [item for item in items if item.quality_status == QualityStatus.OK]
    pending = [item for item in items if item.quality_status == QualityStatus.PENDING]
    stores = sorted({item.store_name for item in items if item.store_name})
    return {
        "total_files": len(items),
        "image_files": len(images),
        "video_files": len(videos),
        "stores": stores,
        "store_count": len(stores),
        "quality_ok": len(ok),
        "quality_warning": len(warnings),
        "quality_retake": len(retake),
        "quality_pending": len(pending),
        "ready_for_ml": len(ok) + len(warnings) + len(pending),
    }


def render_job_markdown(
    job_id: str,
    source: str,
    store_name: str | None,
    media: list[MediaInspection],
    summary: dict[str, Any],
    notes: list[str],
) -> str:
    lines = [
        "# SKU Processing Job",
        "",
        f"- Job ID: `{job_id}`",
        f"- Source: `{source}`",
        f"- Store: `{store_name or '-'}`",
        f"- Files: `{summary['total_files']}`",
        f"- Images: `{summary['image_files']}`",
        f"- Videos: `{summary['video_files']}`",
        f"- Quality OK: `{summary['quality_ok']}`",
        f"- Quality warnings: `{summary['quality_warning']}`",
        f"- Retake needed: `{summary['quality_retake']}`",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in notes)
    lines.extend(
        [
            "",
            "## Files",
            "",
            "| File | Kind | Size | Resolution | Quality | Issues |",
            "| --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for item in media:
        size_mb = (item.file_size_bytes or 0) / 1024 / 1024
        resolution = f"{item.width}x{item.height}" if item.width and item.height else "-"
        issues = "<br>".join(item.issues)
        lines.append(
            f"| `{item.source_path}` | `{item.media_kind.value}` | {size_mb:.1f} MB | "
            f"{resolution} | `{item.quality_status.value}` | {issues} |"
        )
    lines.append("")
    return "\n".join(lines)


def _job_notes(summary: dict[str, Any]) -> list[str]:
    notes = [
        "MVP completed ingestion and quality analysis. SKU recognition is the next ML layer.",
        "Competitor catalog is not connected yet; non-own products will be treated as competitor_or_unknown.",
    ]
    if summary["quality_retake"]:
        notes.append("Some files need retake before reliable SKU recognition.")
    if summary["video_files"]:
        notes.append("Videos are accepted; keyframe extraction/tracking must be enabled for production counting.")
    return notes


def _new_job_id() -> str:
    return f"job_{uuid.uuid4().hex[:12]}"


def _infer_store_name(root: Path, path: Path) -> str | None:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return None
    if len(parts) >= 2:
        top = parts[0]
        return None if top in INTERNAL_TOP_LEVEL_DIRS else top
    if root.name not in INTERNAL_TOP_LEVEL_DIRS:
        return root.name
    return None


def _media_from_dict(payload: dict[str, Any]) -> MediaInspection:
    return MediaInspection(
        **{
            **payload,
            "media_kind": MediaKind(payload["media_kind"]),
            "quality_status": QualityStatus(payload["quality_status"]),
        }
    )


def _safe_filename(filename: str) -> str:
    safe = "".join(char if char.isalnum() or char in ".-_" else "_" for char in filename)
    return safe or "upload.bin"

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from sku_audit.models import MediaKind, QualityStatus

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}


@dataclass(frozen=True)
class MediaInspection:
    source_path: str
    media_kind: MediaKind
    store_name: str | None
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    file_size_bytes: int | None = None
    brightness: float | None = None
    contrast: float | None = None
    sharpness: float | None = None
    quality_status: QualityStatus = QualityStatus.PENDING
    issues: list[str] = field(default_factory=list)
    preview_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["media_kind"] = self.media_kind.value
        data["quality_status"] = self.quality_status.value
        return data


def discover_media_paths(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and _media_kind(path) != MediaKind.OTHER and not _is_runtime_path(root, path)
    )


def inspect_media_file(
    path: Path,
    *,
    root: Path,
    preview_root: Path | None = None,
    store_name: str | None = None,
) -> MediaInspection:
    path = path.expanduser().resolve()
    root = root.expanduser().resolve()
    relative = _relative_or_absolute(path, root)
    kind = _media_kind(path)
    size = path.stat().st_size if path.exists() else None

    if kind is MediaKind.IMAGE:
        return _inspect_image(
            path,
            relative=relative,
            preview_root=preview_root,
            store_name=store_name,
            file_size_bytes=size,
        )
    if kind is MediaKind.VIDEO:
        return _inspect_video(path, relative=relative, store_name=store_name, file_size_bytes=size)

    return MediaInspection(
        source_path=relative,
        media_kind=MediaKind.OTHER,
        store_name=store_name,
        file_size_bytes=size,
        issues=["Unsupported media type."],
    )


def copy_upload_file(source: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / _safe_filename(source.name)
    shutil.copy2(source, destination)
    return destination


def _inspect_image(
    path: Path,
    *,
    relative: str,
    preview_root: Path | None,
    store_name: str | None,
    file_size_bytes: int | None,
) -> MediaInspection:
    width, height = _image_size(path)
    preview_path = _build_preview(path, preview_root) if preview_root else None
    analysis_path = preview_path or path

    brightness: float | None = None
    contrast: float | None = None
    sharpness: float | None = None
    issues: list[str] = []

    try:
        brightness, contrast, sharpness = _image_quality_metrics(analysis_path)
    except Exception as exc:
        issues.append(f"Could not calculate image quality: {type(exc).__name__}")

    status = QualityStatus.OK
    if width and height and min(width, height) < 900:
        status = QualityStatus.WARNING
        issues.append("Image resolution is low for reliable SKU reading.")
    if brightness is not None and brightness < 45:
        status = QualityStatus.RETAKE
        issues.append("Image is too dark.")
    elif brightness is not None and brightness > 225:
        status = QualityStatus.WARNING
        issues.append("Image is very bright; glare may hide labels.")
    if contrast is not None and contrast < 22:
        status = max_quality_status(status, QualityStatus.WARNING)
        issues.append("Image has low contrast.")
    if sharpness is not None and sharpness < 5.0:
        status = QualityStatus.RETAKE
        issues.append("Image appears blurred.")

    if not issues:
        issues.append("Frame quality is acceptable for the MVP queue.")

    return MediaInspection(
        source_path=relative,
        media_kind=MediaKind.IMAGE,
        store_name=store_name,
        width=width,
        height=height,
        file_size_bytes=file_size_bytes,
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        quality_status=status,
        issues=issues,
        preview_path=str(preview_path) if preview_path else None,
    )


def _inspect_video(
    path: Path,
    *,
    relative: str,
    store_name: str | None,
    file_size_bytes: int | None,
) -> MediaInspection:
    metadata = _video_metadata(path)
    issues = ["Video accepted. Keyframe extraction and tracking are queued for the ML layer."]
    status = QualityStatus.PENDING
    duration = metadata.get("duration_seconds")
    if duration is not None and duration > 90:
        issues.append("Long video: backend should extract sparse keyframes before ML processing.")
    return MediaInspection(
        source_path=relative,
        media_kind=MediaKind.VIDEO,
        store_name=store_name,
        width=metadata.get("width"),
        height=metadata.get("height"),
        duration_seconds=duration,
        file_size_bytes=file_size_bytes,
        quality_status=status,
        issues=issues,
    )


def max_quality_status(left: QualityStatus, right: QualityStatus) -> QualityStatus:
    order = {
        QualityStatus.OK: 0,
        QualityStatus.PENDING: 1,
        QualityStatus.WARNING: 2,
        QualityStatus.RETAKE: 3,
    }
    return left if order[left] >= order[right] else right


def _image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        pass

    result = _run(["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(path)])
    width: int | None = None
    height: int | None = None
    for line in result.splitlines():
        if "pixelWidth:" in line:
            width = int(line.rsplit(":", 1)[1].strip())
        elif "pixelHeight:" in line:
            height = int(line.rsplit(":", 1)[1].strip())
    return width, height


def _build_preview(path: Path, preview_root: Path) -> Path:
    preview_root.mkdir(parents=True, exist_ok=True)
    preview_path = preview_root / f"{_file_hash(path)}.jpg"
    if preview_path.exists():
        return preview_path

    if path.suffix.lower() == ".heic":
        _run(["sips", "-Z", "1400", "-s", "format", "jpeg", str(path), "--out", str(preview_path)])
        return preview_path

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image.thumbnail((1400, 1400))
        image.convert("RGB").save(preview_path, format="JPEG", quality=88)
    return preview_path


def _image_quality_metrics(path: Path) -> tuple[float, float, float]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        image.thumbnail((720, 720))
        arr = np.asarray(image, dtype=np.float32)

    brightness = float(arr.mean())
    contrast = float(arr.std())
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return brightness, contrast, 0.0
    gx = np.diff(arr, axis=1)
    gy = np.diff(arr, axis=0)
    sharpness = float((np.mean(np.abs(gx)) + np.mean(np.abs(gy))) / 2.0)
    return brightness, contrast, sharpness


def _video_metadata(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
    if not Path(ffprobe).exists():
        return {}
    try:
        raw = _run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,duration:format=duration",
                "-of",
                "json",
                str(path),
            ],
            timeout=20,
        )
        payload = json.loads(raw)
    except Exception:
        return {}

    stream = (payload.get("streams") or [{}])[0]
    duration = stream.get("duration") or (payload.get("format") or {}).get("duration")
    return {
        "width": _safe_int(stream.get("width")),
        "height": _safe_int(stream.get("height")),
        "duration_seconds": _safe_float(duration),
    }


def _media_kind(path: Path) -> MediaKind:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return MediaKind.IMAGE
    if suffix in VIDEO_SUFFIXES:
        return MediaKind.VIDEO
    return MediaKind.OTHER


def _is_runtime_path(root: Path, path: Path) -> bool:
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        return False
    return bool(relative_parts and relative_parts[0] == "var")


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(str(path).encode("utf-8"))
    digest.update(str(path.stat().st_mtime_ns).encode("ascii"))
    digest.update(str(path.stat().st_size).encode("ascii"))
    return digest.hexdigest()[:20]


def _safe_filename(filename: str) -> str:
    safe = "".join(char if char.isalnum() or char in ".-_" else "_" for char in filename)
    return safe or "upload.bin"


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run(command: list[str], *, timeout: int = 30) -> str:
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    return completed.stdout + completed.stderr

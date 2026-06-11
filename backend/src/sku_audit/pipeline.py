from __future__ import annotations

import hashlib
from pathlib import Path

from sku_audit.models import (
    AuditObservation,
    AuditReport,
    AuditSummary,
    EvidenceFile,
    RecognitionStatus,
    StoreSummary,
    ZoneType,
    normalize_path,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
INTERNAL_TOP_LEVEL_DIRS = {
    "backend",
    "data",
    "docs",
    "mobile",
    "reports",
    "sku_exact_areas",
    "sku_uncertain_areas",
}


def run_image_folder_audit(input_root: Path) -> AuditReport:
    root = input_root.expanduser().resolve()
    image_paths = _find_image_paths(root)
    evidence = [_build_evidence(root, path) for path in image_paths]
    observations = [_build_observation(index, item) for index, item in enumerate(evidence, start=1)]
    store_summaries = _build_store_summaries(evidence)
    summary = _build_summary(evidence, observations)

    return AuditReport(
        report_id=_report_id(root, image_paths),
        input_root=normalize_path(root),
        summary=summary,
        store_summaries=store_summaries,
        evidence=evidence,
        observations=observations,
    )


def _find_image_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and not _is_generated_report(path)
    )


def _is_generated_report(path: Path) -> bool:
    return "reports" in path.parts


def _build_evidence(root: Path, path: Path) -> EvidenceFile:
    relative = path.relative_to(root)
    return EvidenceFile(
        path=str(relative),
        role=_infer_role(path),
        zone_type=_infer_zone_type(path),
        store_name=_infer_store_name(relative),
    )


def _infer_role(path: Path) -> str:
    name = path.stem.lower()
    if "contact_sheet" in name:
        return "contact_sheet"
    if "overview" in name:
        return "overview"
    if "zone" in name:
        return "zone_crop"
    return "source_image"


def _infer_zone_type(path: Path) -> ZoneType:
    parts = {part.lower() for part in path.parts}
    if "sku_exact_areas" in parts:
        return ZoneType.EXACT
    if "sku_uncertain_areas" in parts:
        return ZoneType.UNCERTAIN
    return ZoneType.UNKNOWN


def _infer_store_name(relative_path: Path) -> str | None:
    if len(relative_path.parts) < 2:
        return None
    top_level = relative_path.parts[0]
    if top_level in INTERNAL_TOP_LEVEL_DIRS:
        return None
    return top_level


def _build_observation(index: int, evidence: EvidenceFile) -> AuditObservation:
    observation_id = f"obs_{index:06d}"

    if evidence.role in {"contact_sheet", "overview"}:
        return AuditObservation(
            observation_id=observation_id,
            source_path=evidence.path,
            zone_type=evidence.zone_type,
            status=RecognitionStatus.NOT_PRODUCT,
            confidence=1.0,
            store_name=evidence.store_name,
            notes=[f"{evidence.role} is useful for review but should not be counted as a product."],
        )

    if evidence.zone_type == ZoneType.EXACT:
        return AuditObservation(
            observation_id=observation_id,
            source_path=evidence.path,
            zone_type=evidence.zone_type,
            status=RecognitionStatus.BRAND_ONLY,
            confidence=0.35,
            store_name=evidence.store_name,
            notes=[
                "Baseline ingestion only: image is marked as an exact-count area, but ML recognition is not connected yet."
            ],
        )

    if evidence.zone_type == ZoneType.UNCERTAIN:
        return AuditObservation(
            observation_id=observation_id,
            source_path=evidence.path,
            zone_type=evidence.zone_type,
            status=RecognitionStatus.NEEDS_RETAKE,
            confidence=0.9,
            store_name=evidence.store_name,
            notes=["Source folder marks this as an uncertain area. Send to retake or manual review."],
        )

    if evidence.store_name:
        return AuditObservation(
            observation_id=observation_id,
            source_path=evidence.path,
            zone_type=evidence.zone_type,
            status=RecognitionStatus.PENDING_ANALYSIS,
            confidence=0.0,
            store_name=evidence.store_name,
            notes=["Raw market photo from a trading point. Queued for ML detection and SKU recognition."],
        )

    return AuditObservation(
        observation_id=observation_id,
        source_path=evidence.path,
        zone_type=evidence.zone_type,
        status=RecognitionStatus.UNKNOWN,
        confidence=0.0,
        notes=["Unknown folder convention. Needs manual classification."],
    )


def _build_store_summaries(evidence: list[EvidenceFile]) -> list[StoreSummary]:
    counts: dict[str, int] = {}
    for item in evidence:
        if item.store_name:
            counts[item.store_name] = counts.get(item.store_name, 0) + 1
    return [
        StoreSummary(store_name=store_name, image_count=count)
        for store_name, count in sorted(counts.items())
    ]


def _build_summary(
    evidence: list[EvidenceFile], observations: list[AuditObservation]
) -> AuditSummary:
    return AuditSummary(
        total_evidence_files=len(evidence),
        store_count=len({item.store_name for item in evidence if item.store_name}),
        market_photo_files=sum(1 for item in evidence if item.store_name),
        exact_zone_files=sum(1 for item in evidence if item.zone_type == ZoneType.EXACT),
        uncertain_zone_files=sum(1 for item in evidence if item.zone_type == ZoneType.UNCERTAIN),
        countable_zone_files=sum(1 for item in evidence if item.role == "zone_crop"),
        review_artifact_files=sum(
            1 for item in evidence if item.role in {"contact_sheet", "overview"}
        ),
        recognized_observations=sum(
            1
            for item in observations
            if item.status in {RecognitionStatus.RECOGNIZED, RecognitionStatus.BRAND_ONLY}
        ),
        pending_analysis_observations=sum(
            1 for item in observations if item.status == RecognitionStatus.PENDING_ANALYSIS
        ),
        needs_review_observations=sum(
            1 for item in observations if item.status == RecognitionStatus.NEEDS_RETAKE
        ),
        unknown_observations=sum(1 for item in observations if item.status == RecognitionStatus.UNKNOWN),
    )


def _report_id(root: Path, image_paths: list[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(str(root).encode("utf-8"))
    for path in image_paths:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
    return f"report_{digest.hexdigest()[:12]}"

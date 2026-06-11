from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class RecognitionStatus(StrEnum):
    RECOGNIZED = "recognized"
    BRAND_ONLY = "brand_only"
    PENDING_ANALYSIS = "pending_analysis"
    UNKNOWN = "unknown"
    NEEDS_RETAKE = "needs_retake"
    NOT_PRODUCT = "not_product"


class ZoneType(StrEnum):
    EXACT = "exact"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SkuCatalogEntry:
    sku_id: str
    brand_id: str
    brand_name: str
    is_own_brand: bool
    category: str
    model_name: str
    article_codes: list[str] = field(default_factory=list)
    barcodes: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    reference_images: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceFile:
    path: str
    role: str
    zone_type: ZoneType
    store_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["zone_type"] = self.zone_type.value
        return data


@dataclass(frozen=True)
class AuditObservation:
    observation_id: str
    source_path: str
    zone_type: ZoneType
    status: RecognitionStatus
    confidence: float
    store_name: str | None = None
    brand_id: str | None = None
    brand_name: str | None = None
    sku_id: str | None = None
    is_own_brand: bool | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["zone_type"] = self.zone_type.value
        data["status"] = self.status.value
        return data


@dataclass(frozen=True)
class AuditSummary:
    total_evidence_files: int
    store_count: int
    market_photo_files: int
    exact_zone_files: int
    uncertain_zone_files: int
    countable_zone_files: int
    review_artifact_files: int
    recognized_observations: int
    pending_analysis_observations: int
    needs_review_observations: int
    unknown_observations: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StoreSummary:
    store_name: str
    image_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuditReport:
    report_id: str
    input_root: str
    summary: AuditSummary
    store_summaries: list[StoreSummary]
    evidence: list[EvidenceFile]
    observations: list[AuditObservation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "input_root": self.input_root,
            "summary": self.summary.to_dict(),
            "store_summaries": [item.to_dict() for item in self.store_summaries],
            "evidence": [item.to_dict() for item in self.evidence],
            "observations": [item.to_dict() for item in self.observations],
        }


def normalize_path(path: Path) -> str:
    return str(path.expanduser().resolve())

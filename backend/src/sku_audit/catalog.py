from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from sku_audit.models import SkuCatalogEntry


def _split_multi_value(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split("|") if part.strip()]


def load_catalog(path: Path) -> list[SkuCatalogEntry]:
    if not path.exists():
        raise FileNotFoundError(f"Catalog file does not exist: {path}")
    if path.suffix.lower() == ".json":
        return _load_catalog_json(path)
    if path.suffix.lower() == ".csv":
        return _load_catalog_csv(path)
    raise ValueError(f"Unsupported catalog format: {path.suffix}")


def _load_catalog_json(path: Path) -> list[SkuCatalogEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["items"] if isinstance(payload, dict) else payload
    return [SkuCatalogEntry(**row) for row in rows]


def _load_catalog_csv(path: Path) -> list[SkuCatalogEntry]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = csv.DictReader(file)
        return [
            SkuCatalogEntry(
                sku_id=row["sku_id"],
                brand_id=row["brand_id"],
                brand_name=row["brand_name"],
                is_own_brand=row["is_own_brand"].strip().lower() in {"1", "true", "yes", "y"},
                category=row["category"],
                model_name=row["model_name"],
                article_codes=_split_multi_value(row.get("article_codes")),
                barcodes=_split_multi_value(row.get("barcodes")),
                aliases=_split_multi_value(row.get("aliases")),
                reference_images=_split_multi_value(row.get("reference_images")),
            )
            for row in rows
        ]


def index_catalog_by_sku(entries: Iterable[SkuCatalogEntry]) -> dict[str, SkuCatalogEntry]:
    return {entry.sku_id: entry for entry in entries}

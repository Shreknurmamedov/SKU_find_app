"""Stage 2: recognize the SKU on a single product crop.

Pipeline: OCR the crop -> normalize text -> match against the Utake catalog
(data/catalog/own_products.csv) by model code, then by brand. The catalog
already carries brand, model and category, so a match yields all three plus
the own-brand flag and a confidence. No per-SKU training required; adding a
new product only means adding a catalog row.

Public API:
    from ml.sku_recognize import Recognizer
    rec = Recognizer()
    result = rec.recognize(bgr_or_pil_crop)   # -> RecognitionResult

Standalone:
    python3 ml/sku_recognize.py path/to/crop.jpg [more.jpg ...]
"""
from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np

CATALOG_CSV = Path("data/catalog/own_products.csv")

# Cyrillic letters that share a glyph with a Latin one. We canonicalize BOTH
# the catalog and the OCR text through this map so "АВР" (Cyrillic) and an OCR
# reading of "ABP" (Latin) collapse to the same string.
CONFUSABLES = str.maketrans(
    {
        "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
        "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
    }
)
TOKEN_RE = re.compile(r"[0-9A-ZА-ЯЁ]+")


def canon(text: str) -> str:
    return text.upper().translate(CONFUSABLES)


def compact(text: str) -> str:
    """Uppercase, canonicalized, alphanumerics only (drops spaces/punct)."""
    return "".join(TOKEN_RE.findall(canon(text)))


def model_keys(model: str) -> list[str]:
    """Discriminative compact keys for a model name.

    'DY5000LX/DY6500LX' -> ['DY5000LX', 'DY6500LX']
    'АВР-40I'           -> ['ABP40I']
    Keeps only keys that contain a digit and are >= 4 chars, so generic words
    don't produce false hits.
    """
    keys: list[str] = []
    for variant in re.split(r"[\\/,]", model):
        k = compact(variant)
        if len(k) >= 4 and any(c.isdigit() for c in k):
            keys.append(k)
    return keys


@dataclass
class CatalogEntry:
    sku_id: str
    brand: str
    is_own: bool
    category: str
    model: str
    keys: tuple[str, ...]
    brand_canon: str


@dataclass
class RecognitionResult:
    status: str            # "matched_sku" | "brand_only" | "unknown"
    is_own: bool
    brand: str | None
    model: str | None
    category: str | None
    sku_id: str | None
    confidence: float
    method: str
    matched_key: str | None
    text: str
    rotation: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


class Recognizer:
    def __init__(self, catalog_csv: Path = CATALOG_CSV, languages=("ru", "en")):
        self.entries = self._load_catalog(catalog_csv)
        # brand canon -> display name (longest catalog spelling wins)
        self.brand_by_canon: dict[str, str] = {}
        for e in self.entries:
            self.brand_by_canon.setdefault(e.brand_canon, e.brand)
        # key -> entry, longest keys first so the most specific match wins
        self.key_index: list[tuple[str, CatalogEntry]] = sorted(
            ((k, e) for e in self.entries for k in e.keys),
            key=lambda kv: -len(kv[0]),
        )
        # category keyword -> coarse type label, for the "category but no model"
        # tier. Keep product-type NOUNS (drop adjectives by Russian suffix) and
        # show the noun itself ("Опрыскиватель", "Триммер") rather than the exact
        # over-specific catalog sub-category, which one OCR word can't pin down.
        adj_suffix = ("ЫЙ", "ИЙ", "ОЙ", "ЫЕ", "ИЕ", "АЯ", "ЯЯ", "ОЕ", "ЕЕ",
                      "ОГО", "ЕГО", "ЫХ", "ИХ", "УЮ", "ЮЮ", "ЫМ", "ИМ", "ОМУ")
        cat_kw: dict[str, str] = {}
        for e in self.entries:
            if not e.category:
                continue
            for raw in re.findall(r"[A-ZА-ЯЁ0-9]+", e.category.upper()):
                if len(raw) >= 5 and not raw.endswith(adj_suffix):
                    cat_kw.setdefault(canon(raw), raw.capitalize())
        self.category_index: list[tuple[str, str]] = sorted(
            cat_kw.items(), key=lambda kv: -len(kv[0]))
        self._reader = None
        self._languages = list(languages)

    @staticmethod
    def _load_catalog(path: Path) -> list[CatalogEntry]:
        entries: list[CatalogEntry] = []
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                brand = (row.get("brand_name") or "").strip()
                if not brand or brand == "-":
                    continue
                keys = set(model_keys(row.get("model_name", "")))
                for alias in (row.get("aliases") or "").split("|"):
                    keys.update(model_keys(alias))
                entries.append(
                    CatalogEntry(
                        sku_id=row.get("sku_id", ""),
                        brand=brand,
                        is_own=(row.get("is_own_brand", "").strip().lower() == "true"),
                        category=(row.get("category") or "").strip(),
                        model=(row.get("model_name") or "").strip(),
                        keys=tuple(sorted(keys)),
                        brand_canon=canon(brand),
                    )
                )
        return entries

    # ---- OCR ---------------------------------------------------------------
    def _ensure_reader(self):
        if self._reader is None:
            import easyocr  # lazy: heavy import / model download on first use
            self._reader = easyocr.Reader(self._languages, gpu=False, verbose=False)
        return self._reader

    def _ocr_array(self, arr: np.ndarray) -> str:
        try:
            lines = self._ensure_reader().readtext(arr, detail=0, paragraph=True)
        except Exception:
            return ""
        return " ".join(lines)

    @staticmethod
    def _to_array(image) -> np.ndarray:
        if isinstance(image, (str, Path)):
            from PIL import Image
            return np.array(Image.open(image).convert("RGB"))
        if isinstance(image, np.ndarray):
            return image
        # assume PIL.Image
        return np.array(image.convert("RGB"))

    # ---- matching ----------------------------------------------------------
    def match_text(self, text: str) -> RecognitionResult:
        comp = compact(text)
        # token set guards brand matching against substrings inside real words
        # (e.g. "ТЕКСТ" -> canon "TEKCT" must NOT match brand "TEK").
        tokens = set(TOKEN_RE.findall(canon(text)))

        # 1) model-code hit -> specific SKU (strongest signal)
        if comp:
            for key, entry in self.key_index:
                if key in comp:
                    conf = 0.9 if len(key) >= 6 else 0.78
                    return RecognitionResult(
                        status="matched_sku", is_own=entry.is_own,
                        brand=entry.brand, model=entry.model,
                        category=entry.category, sku_id=entry.sku_id,
                        confidence=conf, method="model_code",
                        matched_key=key, text=text,
                    )

        # Graceful fallback: brand and/or category from the printed words.
        brand_found = brand_canon = None
        for bcanon, bname in self.brand_by_canon.items():
            if bcanon in tokens:
                brand_found, brand_canon = bname, bcanon
                break
        cat_found = cat_kw = None
        for kw, cat in self.category_index:
            if kw in tokens:
                cat_found, cat_kw = cat, kw
                break

        # 2) brand known (model unresolved), category attached if also seen
        if brand_found:
            return RecognitionResult(
                status="brand_only", is_own=True, brand=brand_found, model=None,
                category=cat_found, sku_id=None,
                confidence=0.62 if cat_found else 0.6,
                method="brand+category" if cat_found else "brand_text",
                matched_key=brand_canon, text=text,
            )

        # 3) only the product type (category) is readable, brand not seen
        if cat_found:
            return RecognitionResult(
                status="category_only", is_own=False, brand=None, model=None,
                category=cat_found, sku_id=None, confidence=0.45,
                method="category_text", matched_key=cat_kw, text=text,
            )

        # 4) nothing readable -> brand not visible, manager must re-shoot
        return RecognitionResult(
            status="unknown", is_own=False, brand=None, model=None,
            category=None, sku_id=None, confidence=0.3 if tokens else 0.15,
            method="no_match", matched_key=None, text=text,
        )

    _STATUS_RANK = {"matched_sku": 4, "brand_only": 3, "category_only": 2, "unknown": 1}

    def recognize(self, image, rotations=(0, 270, 90, 180), min_side=80) -> RecognitionResult:
        """OCR + match a crop, trying several rotations.

        Shelf photos/video frames often arrive rotated, and easyocr only reads
        roughly-horizontal text, so we OCR each rotation and keep the strongest
        catalog match. Tiny crops (< min_side px) are upscaled first.
        """
        import numpy as _np
        from PIL import Image as _Image

        arr = self._to_array(image)
        pil = _Image.fromarray(arr) if isinstance(arr, _np.ndarray) else arr
        w, h = pil.size
        if min(w, h) < min_side and min(w, h) > 0:
            scale = min_side / min(w, h)
            pil = pil.resize((int(w * scale), int(h * scale)), _Image.LANCZOS)

        best: RecognitionResult | None = None
        for rot in rotations:
            rimg = pil if rot == 0 else pil.rotate(rot, expand=True)
            res = self.match_text(self._ocr_array(_np.array(rimg)))
            res.rotation = rot
            if best is None or self._score(res) > self._score(best):
                best = res
            if best.status == "matched_sku":  # strongest signal, stop early
                break
        return best  # type: ignore[return-value]

    def _score(self, r: RecognitionResult) -> float:
        return self._STATUS_RANK.get(r.status, 0) * 10 + r.confidence + len(r.text) * 1e-4


def main(argv: Iterable[str]) -> None:
    import json
    paths = list(argv)
    if not paths:
        print("usage: python3 ml/sku_recognize.py crop.jpg [...]")
        return
    rec = Recognizer()
    for p in paths:
        r = rec.recognize(p)
        print(p)
        print(json.dumps(r.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])

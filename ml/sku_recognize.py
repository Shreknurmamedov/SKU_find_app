"""Stage 2: recognize the SKU on a single product crop.

Pipeline: OCR the crop -> normalize text -> match against the product catalog
(data/catalog/reference_dataset_all/training_images.csv, with own_products.csv
as a fallback) by model code, article, then brand. The catalog carries own and
competitor brands, model and category, so a match yields all three plus the
own-brand flag and a confidence.

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
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import numpy as np

# NOTE: ml.visual_reference is imported lazily inside Recognizer._reference() so
# this module still runs as a standalone script (python3 ml/sku_recognize.py),
# where the repo root is not on sys.path and `import ml...` would fail. We only
# need the default index paths at import time, so keep them defined locally.
DEFAULT_INDEX = Path("data/catalog/reference_index_yolo11n.npz")
DEFAULT_METADATA = Path("data/catalog/reference_index_yolo11n.jsonl")

REFERENCE_CATALOG_CSV = Path("data/catalog/reference_dataset_all/training_images.csv")
OWN_CATALOG_CSV = Path("data/catalog/own_products.csv")
CATALOG_CSV = REFERENCE_CATALOG_CSV

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
ARTICLE_RE = re.compile(r"\d+(?:\s*[/\\-]\s*\d+){2,}")
CODE_TRANSLIT = {
    "Ё": "E", "Э": "E", "Л": "L", "И": "I", "Й": "I", "З": "3",
}
# A literal model/article code at least this long is discriminative enough to
# commit to a specific SKU. Shorter or OCR-guessed codes are treated as a weak
# hint that only fills in brand/category (business rule: never report a guessed
# model when the article is not reliably readable -- brand + category instead).
STRONG_KEY_LEN = 5


def canon(text: str) -> str:
    return text.upper().translate(CONFUSABLES)


def compact(text: str) -> str:
    """Uppercase, canonicalized, alphanumerics only (drops spaces/punct)."""
    return "".join(TOKEN_RE.findall(canon(text)))


def code_key_variants(key: str) -> set[str]:
    """Variants for mixed Cyrillic/Latin model codes seen in OCR.

    This is intentionally for model codes only, not article codes: it lets
    catalog "ЭТ-20-2ЛИ" meet OCR "ET-20-2Li", and Huter "GET-20M-2Li" meet
    a crop where the leading G was missed ("ET-20M-2Li").
    """
    variants = {key}
    translit = "".join(CODE_TRANSLIT.get(c, c) for c in key)
    variants.add(translit)
    for item in list(variants):
        if item.startswith("GET") and len(item) >= 7:
            variants.add(item[1:])
    return {v for v in variants if v}


def model_keys(model: str) -> list[str]:
    """Discriminative compact keys for a model name.

    'DY5000LX/DY6500LX' -> ['DY5000LX', 'DY6500LX']
    'АВР-40I'           -> ['ABP40I']
    'SP-3,7 Lite'       -> ['SP37LITE']
    Keeps only keys that contain a digit and are >= 4 chars, so generic words
    don't produce false hits.
    """
    def is_variant_key(key: str) -> bool:
        letters = sum(1 for c in key if c.isalpha())
        return len(key) >= 4 and any(c.isdigit() for c in key) and letters >= 2

    keys: set[str] = set()
    variants = [model]
    # Slash often separates model variants. Comma often means a decimal number
    # in Russian model names (SP-3,7), so keep it inside the key.
    if "/" in model or "\\" in model:
        split_variants = re.split(r"[\\/]", model)
        compact_variants = [compact(v) for v in split_variants]
        if len(compact_variants) > 1 and all(is_variant_key(v) for v in compact_variants):
            variants.extend(split_variants)
    for variant in variants:
        k = compact(variant)
        if len(k) >= 4 and any(c.isdigit() for c in k):
            keys.update(code_key_variants(k))
    return sorted(keys, key=len, reverse=True)


def article_keys(article_codes: str) -> list[str]:
    keys: list[str] = []
    for variant in re.split(r"[|,;\s]+", article_codes or ""):
        k = compact(variant)
        if len(k) >= 4 and any(c.isdigit() for c in k):
            keys.append(k)
    return keys


def looks_like_article(text: str) -> bool:
    return bool(ARTICLE_RE.search(text))


@dataclass
class CatalogEntry:
    sku_id: str
    brand: str
    is_own: bool
    category: str
    model: str
    article_codes: str
    model_keys: tuple[str, ...]
    article_keys: tuple[str, ...]
    keys: tuple[str, ...]
    brand_canon: str


@dataclass
class RecognitionResult:
    status: str            # "matched_sku" | "brand_only" | "category_only" | "unknown"
    is_own: bool
    brand: str | None
    model: str | None
    category: str | None
    sku_id: str | None
    article_codes: str | None
    confidence: float
    method: str
    matched_key: str | None
    text: str
    rotation: int = 0
    visual_score: float | None = None
    visual_margin: float | None = None
    reference_image: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class Recognizer:
    def __init__(
        self,
        catalog_csv: Path = CATALOG_CSV,
        languages=("ru", "en"),
        *,
        reference_index: Path = DEFAULT_INDEX,
        reference_metadata: Path = DEFAULT_METADATA,
        visual_threshold: float = 0.78,
        visual_margin: float = 0.035,
    ):
        self.entries = self._load_catalog(catalog_csv)
        # brand canon -> display name (longest catalog spelling wins)
        self.brand_by_canon: dict[str, str] = {}
        self.brand_own_by_canon: dict[str, bool] = {}
        for e in self.entries:
            self.brand_by_canon.setdefault(e.brand_canon, e.brand)
            self.brand_own_by_canon[e.brand_canon] = (
                self.brand_own_by_canon.get(e.brand_canon, False) or e.is_own
            )
        self._brand_canons = sorted(self.brand_by_canon.items(), key=lambda kv: -len(kv[0]))
        # model key -> entry, longest keys first so the most specific match wins.
        # Model codes are safe to find as substrings in compact OCR text
        # (e.g. "SP-3,7 Lite" -> SP37LITE).
        self.model_index: list[tuple[str, CatalogEntry]] = sorted(
            ((k, e) for e in self.entries for k in e.model_keys),
            key=lambda kv: -len(kv[0]),
        )
        # Article codes are more collision-prone: 73/7/2/6 must not match inside
        # 73/7/2/26. We therefore match them only against explicit OCR article
        # candidates/tokens, not as arbitrary substrings.
        self.article_index: list[tuple[str, CatalogEntry]] = sorted(
            ((k, e) for e in self.entries for k in e.article_keys),
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
        self._reference_index_path = reference_index
        self._reference_metadata_path = reference_metadata
        self._reference_index: VisualReferenceIndex | None = None
        self._visual_threshold = visual_threshold
        self._visual_margin = visual_margin

    @staticmethod
    def _load_catalog(path: Path) -> list[CatalogEntry]:
        if not path.exists() and path == REFERENCE_CATALOG_CSV:
            path = OWN_CATALOG_CSV
        entries: list[CatalogEntry] = []
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            if reader.fieldnames and "image_path" in reader.fieldnames:
                rows = Recognizer._dedupe_reference_rows(rows)
            for row in rows:
                brand = (row.get("brand_name") or "").strip()
                if not brand or brand == "-":
                    continue
                mkeys = set(model_keys(row.get("model_name", "")))
                article_codes = (row.get("article_codes") or row.get("sku") or "").strip()
                akeys = set(article_keys(article_codes))
                aliases = "|".join(
                    str(row.get(name) or "")
                    for name in ("aliases", "source_name", "catalog_sku_id", "product_id", "sku")
                )
                for alias in aliases.split("|"):
                    if looks_like_article(alias):
                        akeys.update(article_keys(alias))
                    else:
                        mkeys.update(model_keys(alias))
                keys = mkeys | akeys
                dataset_role = (row.get("dataset_role") or "").strip().lower()
                is_own = (
                    (row.get("is_own_brand", "").strip().lower() == "true")
                    or dataset_role == "own_target"
                )
                entries.append(
                    CatalogEntry(
                        sku_id=(
                            row.get("sku_id")
                            or row.get("catalog_sku_id")
                            or row.get("product_id")
                            or row.get("sku")
                            or ""
                        ),
                        brand=brand,
                        is_own=is_own,
                        category=(row.get("category") or "").strip(),
                        model=(row.get("model_name") or "").strip(),
                        article_codes=article_codes,
                        model_keys=tuple(sorted(mkeys)),
                        article_keys=tuple(sorted(akeys)),
                        keys=tuple(sorted(keys)),
                        brand_canon=canon(brand),
                    )
                )
        return entries

    @staticmethod
    def _dedupe_reference_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
        by_product: dict[str, dict[str, str]] = {}
        for row in rows:
            key = (
                row.get("catalog_sku_id")
                or row.get("product_id")
                or row.get("sku")
                or f"{row.get('brand_id', '')}|{row.get('model_name', '')}"
            )
            current = by_product.get(key)
            if current is None:
                by_product[key] = row
                continue
            # Prefer the row with more catalog text; image rows for the same
            # product can differ only by source image metadata.
            current_score = sum(bool(current.get(name)) for name in ("model_name", "category", "sku"))
            new_score = sum(bool(row.get(name)) for name in ("model_name", "category", "sku"))
            if new_score > current_score:
                by_product[key] = row
        return list(by_product.values())

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
    @staticmethod
    def _article_candidates(text: str) -> tuple[set[str], set[str]]:
        explicit = {compact(m.group(0)) for m in ARTICLE_RE.finditer(canon(text))}
        token = {compact(tok) for tok in TOKEN_RE.findall(canon(text))}
        return {c for c in explicit if c}, {c for c in token if c}

    @staticmethod
    def _model_text_variants(comp: str) -> set[str]:
        variants = set(code_key_variants(comp))
        # Common OCR confusions in compact model codes:
        # S can become 5, and Cyrillic/Latin С often becomes C while catalog SP
        # stays Latin S. Keep this narrow to avoid making article matching fuzzy.
        for candidate in list(variants):
            if "5P" in candidate:
                variants.add(candidate.replace("5P", "SP"))
            if "CP" in candidate:
                variants.add(candidate.replace("CP", "SP"))
            if "3T" in candidate:
                variants.add(candidate.replace("3T", "ET"))
            if len(candidate) >= 5 and candidate.startswith("T") and candidate[1].isdigit():
                variants.add("E" + candidate)
        return variants

    def _match_brand(self, tokens: set[str]) -> tuple[str | None, str | None]:
        for bcanon, bname in self._brand_canons:
            if bcanon in tokens:
                return bname, bcanon
        for token in tokens:
            if len(token) < 5:
                continue
            for bcanon, bname in self._brand_canons:
                if len(bcanon) < 5:
                    continue
                ratio = SequenceMatcher(None, token, bcanon).ratio()
                if ratio >= 0.78:
                    return bname, bcanon
        return None, None

    def _match_category(self, tokens: set[str]) -> tuple[str | None, str | None]:
        for kw, cat in self.category_index:
            if kw in tokens:
                return cat, kw
        return None, None

    def _strong_specific_match(self, comp: str, explicit_articles: set[str],
                               token_articles: set[str]):
        """A specific SKU we can trust: an explicitly-read article, or a literal
        model code long enough to be discriminative.

        Returns (entry, key, method, confidence) or None. OCR-confusable guesses
        and very short codes are deliberately excluded here -- they go through
        the weak-hint path and only enrich brand/category.
        """
        # Explicit article pattern (e.g. "64/1/20"): low collision, trust it even
        # when short. Must outrank model text, since several SKUs share a model.
        for key, entry in self.article_index:
            if key in explicit_articles:
                return entry, key, "article_code", 0.9 if len(key) >= 6 else 0.78
        # Literal / transliterated model code, long enough to be specific.
        literal_texts = code_key_variants(comp) if comp else set()
        for key, entry in self.model_index:
            if len(key) >= STRONG_KEY_LEN and any(key in t for t in literal_texts):
                return entry, key, "model_code", 0.9 if len(key) >= 6 else 0.8
        # A long article read as a bare token (OCR dropped the separators).
        for key, entry in self.article_index:
            if len(key) >= 6 and key in token_articles:
                return entry, key, "article_code", 0.9
        return None

    def _weak_specific_hint(self, comp: str, token_articles: set[str]):
        """A plausible but collision-prone code match: a short code or an
        OCR-confusable variant. Used only to fill in brand/category, never to
        emit a guessed SKU. Returns (entry, key) or None."""
        model_texts = self._model_text_variants(comp) if comp else set()
        for key, entry in self.model_index:
            if any(key in t for t in model_texts):
                return entry, key
        for key, entry in self.article_index:
            if key in token_articles:
                return entry, key
        return None

    def match_text(self, text: str) -> RecognitionResult:
        comp = compact(text)
        # token set guards brand matching against substrings inside real words
        # (e.g. "ТЕКСТ" -> canon "TEKCT" must NOT match brand "TEK").
        tokens = set(TOKEN_RE.findall(canon(text)))
        explicit_articles, token_articles = self._article_candidates(text)

        # 1) strong, trustworthy evidence -> commit to a specific SKU
        if comp or explicit_articles:
            strong = self._strong_specific_match(comp, explicit_articles, token_articles)
            if strong is not None:
                entry, key, method, conf = strong
                return RecognitionResult(
                    status="matched_sku", is_own=entry.is_own,
                    brand=entry.brand, model=entry.model,
                    category=entry.category, sku_id=entry.sku_id,
                    article_codes=entry.article_codes,
                    confidence=conf, method=method, matched_key=key, text=text,
                )

        # Graceful fallback: brand and/or category from the printed words.
        brand_found, brand_canon = self._match_brand(tokens)
        cat_found, cat_kw = self._match_category(tokens)

        # 2) weak code hint: the article is not reliably readable, so DON'T guess
        # a model. Use the hint only to report brand + product category, which is
        # far more robust than the exact SKU on a blurry/partial code.
        method_brand = "brand_text"
        is_own_brand = self.brand_own_by_canon.get(brand_canon or "", False)
        weak = self._weak_specific_hint(comp, token_articles) if comp else None
        if weak is not None:
            entry, key = weak
            if brand_found is None and len(key) >= STRONG_KEY_LEN:
                brand_found, brand_canon = entry.brand, entry.brand_canon
                method_brand = "brand_from_code"
                is_own_brand = entry.is_own
            if cat_found is None and brand_canon == entry.brand_canon:
                cat_found = entry.category or None

        # 3) brand known (model unresolved), category attached when known
        if brand_found:
            return RecognitionResult(
                status="brand_only", is_own=is_own_brand, brand=brand_found,
                model=None, category=cat_found, sku_id=None, article_codes=None,
                confidence=0.62 if cat_found else 0.6,
                method="brand+category" if cat_found else method_brand,
                matched_key=brand_canon, text=text,
            )

        # 4) only the product type (category) is readable, brand not seen
        if cat_found:
            return RecognitionResult(
                status="category_only", is_own=False, brand=None, model=None,
                category=cat_found, sku_id=None, article_codes=None, confidence=0.45,
                method="category_text", matched_key=cat_kw, text=text,
            )

        # 5) nothing readable -> brand not visible, manager must re-shoot
        return RecognitionResult(
            status="unknown", is_own=False, brand=None, model=None,
            category=None, sku_id=None, article_codes=None,
            confidence=0.3 if tokens else 0.15,
            method="no_match", matched_key=None, text=text,
        )

    _STATUS_RANK = {"matched_sku": 4, "brand_only": 3, "category_only": 2, "unknown": 1}

    def recognize(self, image, rotations=(0, 270, 90, 180), min_side=80,
                  enhance=True, use_visual=True) -> RecognitionResult:
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
        for variant in self._variants(pil, enhance=enhance):
            for rot in rotations:
                rimg = variant if rot == 0 else variant.rotate(rot, expand=True)
                res = self.match_text(self._ocr_array(_np.array(rimg)))
                res.rotation = rot
                if best is None or self._score(res) > self._score(best):
                    best = res
                if best.status == "matched_sku":  # strongest signal, stop early
                    break
            if best is not None and best.status == "matched_sku":
                break
        if best is not None and best.status != "matched_sku" and use_visual:
            visual = self._recognize_visual(pil, best.text)
            if visual is not None and self._score(visual) > self._score(best):
                best = visual
        return best  # type: ignore[return-value]

    def recognize_visual(self, image, text: str = "") -> RecognitionResult | None:
        import numpy as _np
        from PIL import Image as _Image

        arr = self._to_array(image)
        pil = _Image.fromarray(arr) if isinstance(arr, _np.ndarray) else arr
        return self._recognize_visual(pil, text)

    def _score(self, r: RecognitionResult) -> float:
        return self._STATUS_RANK.get(r.status, 0) * 10 + r.confidence + len(r.text) * 1e-4

    @staticmethod
    def _variants(pil, *, enhance: bool):
        yield pil
        if not enhance:
            return
        try:
            from PIL import ImageEnhance, ImageFilter

            img = pil.convert("RGB")
            img = ImageEnhance.Contrast(img).enhance(1.35)
            img = ImageEnhance.Sharpness(img).enhance(1.8)
            yield img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=140, threshold=3))
        except Exception:
            return

    def _reference(self):
        if self._reference_index is None:
            try:
                from ml.visual_reference import VisualReferenceIndex
            except ImportError:
                # Running as a standalone script without the repo root on
                # sys.path: visual fallback is simply unavailable, OCR still works.
                return None
            idx = VisualReferenceIndex(
                self._reference_index_path,
                self._reference_metadata_path,
            )
            if not idx.available():
                return None
            self._reference_index = idx
        return self._reference_index

    def _recognize_visual(self, pil, text: str) -> RecognitionResult | None:
        ref = self._reference()
        if ref is None:
            return None
        matches = ref.search(pil, topk_products=3)
        if not matches:
            return None
        best = matches[0]
        if best.score < self._visual_threshold or best.margin < self._visual_margin:
            return None
        meta = best.metadata
        is_own = str(meta.get("is_in_own_catalog", "")).lower() == "true" or meta.get("dataset_role") == "own_target"
        brand = meta.get("brand_name") or None
        category = meta.get("category") or None
        # Visual retrieval over reference photos is reliable for the coarse
        # brand/category cluster, but similar packages give near-identical
        # embeddings, so it must NOT commit to a specific model/SKU. Report
        # brand + category only (business rule), never a guessed sku_id.
        if not brand and not category:
            return None
        return RecognitionResult(
            status="brand_only" if brand else "category_only",
            is_own=is_own if brand else False,
            brand=brand,
            model=None,
            category=category,
            sku_id=None,
            article_codes=None,
            confidence=0.62 if (brand and category) else (0.6 if brand else 0.45),
            method="visual_brand" if brand else "visual_category",
            matched_key=best.product_key,
            text=text,
            rotation=0,
            visual_score=round(best.score, 4),
            visual_margin=round(best.margin, 4),
            reference_image=best.reference_image,
        )


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

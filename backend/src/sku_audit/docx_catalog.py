from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

try:
    from docx import Document
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph
except ImportError as exc:  # pragma: no cover - depends on optional local environment
    raise RuntimeError(
        "python-docx is required to import DOCX catalogs. Install backend dependencies first."
    ) from exc


PRODUCT_TABLE_HEADER = ["Бренд", "Модель", "Артикул", "Характеристики"]
CSV_FIELDS = [
    "sku_id",
    "brand_id",
    "brand_name",
    "is_own_brand",
    "category",
    "model_name",
    "article_codes",
    "barcodes",
    "aliases",
    "reference_images",
]


@dataclass(frozen=True)
class ImportedProduct:
    sku_id: str
    brand_id: str
    brand_name: str
    is_own_brand: bool
    category: str
    model_name: str
    article_codes: list[str]
    barcodes: list[str]
    aliases: list[str]
    reference_images: list[str]
    characteristics: str

    def to_catalog_csv_row(self) -> dict[str, str]:
        return {
            "sku_id": self.sku_id,
            "brand_id": self.brand_id,
            "brand_name": self.brand_name,
            "is_own_brand": "true" if self.is_own_brand else "false",
            "category": self.category,
            "model_name": self.model_name,
            "article_codes": "|".join(self.article_codes),
            "barcodes": "|".join(self.barcodes),
            "aliases": "|".join(self.aliases),
            "reference_images": "|".join(self.reference_images),
        }

    def to_raw_dict(self) -> dict[str, object]:
        return asdict(self)


def import_docx_catalog(path: Path, *, own_brand: bool = True) -> list[ImportedProduct]:
    document = Document(path)
    products: list[ImportedProduct] = []
    current_category: str | None = None

    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            text = Paragraph(child, document).text.strip()
            if text and _is_category_heading(text):
                current_category = _strip_category_count(text)
        elif isinstance(child, CT_Tbl):
            table = Table(child, document)
            if not _is_product_table(table):
                continue
            category = current_category or "Без категории"
            products.extend(_extract_product_rows(table, category, own_brand=own_brand))

    return _ensure_unique_sku_ids(products)


def write_catalog_csv(products: Iterable[ImportedProduct], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for product in products:
            writer.writerow(product.to_catalog_csv_row())


def write_catalog_raw_json(products: Iterable[ImportedProduct], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([product.to_raw_dict() for product in products], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _is_product_table(table: Table) -> bool:
    if not table.rows:
        return False
    header = [_clean_cell(cell.text) for cell in table.rows[0].cells]
    return header == PRODUCT_TABLE_HEADER


def _extract_product_rows(
    table: Table, category: str, *, own_brand: bool
) -> list[ImportedProduct]:
    products: list[ImportedProduct] = []
    for row in table.rows[1:]:
        cells = [_clean_cell(cell.text) for cell in row.cells]
        if len(cells) < 4:
            continue
        brand_name, model_name, article_code, characteristics = cells[:4]
        if not brand_name or not model_name or not article_code:
            continue
        products.append(
            ImportedProduct(
                sku_id=_build_sku_id(brand_name, article_code, model_name),
                brand_id=_slugify(brand_name),
                brand_name=brand_name,
                is_own_brand=own_brand,
                category=category,
                model_name=model_name,
                article_codes=[article_code],
                barcodes=[],
                aliases=_build_aliases(brand_name, model_name, article_code),
                reference_images=[],
                characteristics=characteristics,
            )
        )
    return products


def _ensure_unique_sku_ids(products: list[ImportedProduct]) -> list[ImportedProduct]:
    seen: dict[str, int] = {}
    unique_products: list[ImportedProduct] = []
    for product in products:
        count = seen.get(product.sku_id, 0) + 1
        seen[product.sku_id] = count
        if count == 1:
            unique_products.append(product)
            continue
        unique_products.append(
            ImportedProduct(
                **{
                    **product.to_raw_dict(),
                    "sku_id": f"{product.sku_id}_{count}",
                }
            )
        )
    return unique_products


def _is_category_heading(text: str) -> bool:
    if text in {"Каталог продукции Utake", "Сводка по группам"}:
        return False
    if text.startswith("Источник:"):
        return False
    return bool(re.search(r"\(\d+\)$", text))


def _strip_category_count(text: str) -> str:
    return re.sub(r"\s*\(\d+\)$", "", text).strip()


def _clean_cell(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _build_sku_id(brand_name: str, article_code: str, model_name: str) -> str:
    base = "_".join(
        part for part in [_slugify(brand_name), _slugify(article_code), _slugify(model_name)] if part
    )
    return base.upper()[:120]


def _build_aliases(brand_name: str, model_name: str, article_code: str) -> list[str]:
    aliases = [
        f"{brand_name} {model_name}",
        f"{brand_name} {article_code}",
        model_name,
        article_code,
    ]
    unique_aliases: list[str] = []
    for alias in aliases:
        if alias and alias not in unique_aliases:
            unique_aliases.append(alias)
    return unique_aliases


def _slugify(value: str) -> str:
    transliterated = "".join(_TRANSLITERATION.get(char, char) for char in value.lower())
    slug = re.sub(r"[^a-z0-9]+", "_", transliterated).strip("_")
    return slug or "unknown"


_TRANSLITERATION = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "i",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "c",
    "ч": "ch",
    "ш": "sh",
    "щ": "sch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}

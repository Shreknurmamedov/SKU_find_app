#!/usr/bin/env python3
"""Collect official product reference photos into ML-friendly manifests.

The collector uses sitemap URLs instead of search pages. For every product page
it can parse, it stores one row per product and one row per reference image.

Default run:
    python3 scripts/collect_reference_catalog.py --download-images

Competitor seed run:
    python3 scripts/collect_reference_catalog.py --download-images --include-competitors
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urljoin, urlparse
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup
from PIL import Image


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
)
JINA_READER_PREFIX = "https://r.jina.ai/http://"
ZUBR_START_CATEGORIES = (
    "https://zubr.ru/mekhanizirovannye-instrumenty/",
    "https://zubr.ru/raskhodnye-instrumenty/",
    "https://zubr.ru/ruchnye-instrumenty/",
    "https://zubr.ru/inzhenernaya-santekhnika-i-instrumenty/",
    "https://zubr.ru/khozyaystvennye-prinadlezhnosti/",
    "https://zubr.ru/khimiya-krepezh-siz/",
    "https://zubr.ru/malyarno-shtukaturnye-instrumenty/",
    "https://zubr.ru/elektrika-i-svet/",
    "https://zubr.ru/sad-i-ogorod/",
    "https://zubr.ru/avtotovary/",
    "https://zubr.ru/krepezh/",
    "https://zubr.ru/stroitelnaya-khimiya-i-prinadlezhnosti/",
)


@dataclass(frozen=True)
class Source:
    brand_id: str
    brand_name: str
    base_url: str
    sitemap_urls: tuple[str, ...]
    dataset_role: str


TARGET_SOURCES = (
    Source("resanta", "Ресанта", "https://resanta.ru", ("https://resanta.ru/sitemap-shop.xml",), "own_target"),
    Source("huter", "Huter", "https://huter.su", ("https://huter.su/sitemap-shop.xml",), "own_target"),
    Source("vihr", "Вихрь", "https://vihr.su", ("https://vihr.su/sitemap-shop.xml",), "own_target"),
)


COMPETITOR_SOURCES = (
    Source("zubr", "ЗУБР", "https://zubr.ru", ("https://zubr.ru/sitemap.xml",), "competitor"),
    Source("interskol", "Интерскол", "https://www.interskol.ru", ("https://www.interskol.ru/sitemap.xml",), "competitor"),
    Source("patriot", "PATRIOT", "https://onlypatriot.com", ("https://onlypatriot.com/sitemap.xml",), "competitor"),
    Source("fubag", "FUBAG", "https://fubag.ru", ("https://fubag.ru/sitemap.xml",), "competitor"),
)


IMAGE_EXT_BY_TYPE = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


PRODUCT_CSV_FIELDS = [
    "product_id",
    "dataset_role",
    "brand_id",
    "brand_name",
    "sku",
    "catalog_sku_id",
    "is_in_own_catalog",
    "category",
    "model_name",
    "source_name",
    "source_url",
    "page_status",
    "image_count",
    "reference_images",
    "reference_image_urls",
]


IMAGE_CSV_FIELDS = [
    "image_id",
    "product_id",
    "dataset_role",
    "brand_id",
    "brand_name",
    "sku",
    "catalog_sku_id",
    "category",
    "model_name",
    "source_name",
    "source_url",
    "image_url",
    "local_path",
    "source_rank",
    "download_status",
    "width",
    "height",
    "bytes",
    "sha256",
]


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    image_dir = args.image_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ru,en;q=0.8"})

    own_rows = load_own_catalog(args.own_catalog)
    own_by_article = build_own_article_index(own_rows)
    requested_brands = set(args.brands)
    sources = [source for source in TARGET_SOURCES if source.brand_id in requested_brands]
    if args.include_competitors:
        requested_competitors = set(args.competitor_brands)
        sources.extend(source for source in COMPETITOR_SOURCES if source.brand_id in requested_competitors)

    products: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    seen_product_keys: set[tuple[str, str]] = set()
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "download_images": args.download_images,
        "max_images_per_product": args.max_images_per_product,
        "sources": {},
    }

    for source in sources:
        print(f"[source] {source.brand_name}: sitemap")
        urls = collect_source_urls(session, source, args.timeout, args.max_products_per_source)
        urls = [url for url in urls if not is_obvious_non_product_url(url, source)]
        if args.max_products_per_source and source.brand_id != "zubr":
            urls = urls[: args.max_products_per_source]
        source_report = {
            "sitemap_urls": list(source.sitemap_urls),
            "candidate_urls": len(urls),
            "parsed_products": 0,
            "matched_own_catalog": 0,
            "images": 0,
            "downloaded": 0,
            "skipped_non_product": 0,
            "errors": 0,
        }

        for index, url in enumerate(urls, start=1):
            if index % args.progress_every == 0:
                print(
                    f"[source] {source.brand_name}: {index}/{len(urls)} pages, "
                    f"products={source_report['parsed_products']}, images={source_report['images']}",
                    flush=True,
                )
                write_outputs(out_dir, products, image_rows)

            try:
                html = fetch_page_text(session, url, source, args.timeout)
                time.sleep(args.request_delay)
            except requests.RequestException:
                source_report["errors"] += 1
                continue
            if not html:
                source_report["skipped_non_product"] += 1
                continue

            parsed = parse_product_page(html, url, source, args.max_images_per_product)
            if parsed is None:
                source_report["skipped_non_product"] += 1
                continue

            own_match = own_by_article.get((source.brand_id, normalize_article(parsed["sku"])))
            if own_match:
                parsed["catalog_sku_id"] = own_match["sku_id"]
                parsed["category"] = own_match["category"]
                parsed["model_name"] = own_match["model_name"]
                parsed["is_in_own_catalog"] = True
                source_report["matched_own_catalog"] += 1
            else:
                parsed["catalog_sku_id"] = ""
                parsed["category"] = parsed.get("category") or ""
                parsed["model_name"] = parsed["name"]
                parsed["is_in_own_catalog"] = False

            product_key = (source.brand_id, normalize_article(parsed["sku"]) or slug_from_url(url))
            if product_key in seen_product_keys:
                continue
            seen_product_keys.add(product_key)

            parsed["product_id"] = build_product_id(source.brand_id, parsed["sku"], parsed["source_url"])
            parsed["page_status"] = "ok"
            parsed["image_count"] = 0
            parsed["reference_images"] = []
            parsed["reference_image_urls"] = []

            for rank, image_url in enumerate(parsed.pop("image_urls"), start=1):
                image_row = build_image_row(parsed, image_url, rank)
                if args.download_images:
                    download_info = download_image(
                        session=session,
                        image_url=image_url,
                        product=parsed,
                        rank=rank,
                        image_dir=image_dir,
                        timeout=args.timeout,
                    )
                    image_row.update(download_info)
                    if image_row["download_status"] == "ok":
                        source_report["downloaded"] += 1
                else:
                    image_row.update(
                        {
                            "local_path": "",
                            "download_status": "not_downloaded",
                            "width": "",
                            "height": "",
                            "bytes": "",
                            "sha256": "",
                        }
                    )
                image_rows.append(image_row)
                parsed["reference_image_urls"].append(image_url)
                if image_row.get("local_path"):
                    parsed["reference_images"].append(image_row["local_path"])

            parsed["image_count"] = len(parsed["reference_image_urls"])
            products.append(parsed)
            source_report["parsed_products"] += 1
            source_report["images"] += parsed["image_count"]

        report["sources"][source.brand_id] = source_report
        write_outputs(out_dir, products, image_rows)

    add_missing_own_catalog_rows(products, own_rows, requested_brands, seen_product_keys)
    write_outputs(out_dir, products, image_rows)
    write_own_products_with_references(out_dir, args.own_catalog, products)
    write_report(out_dir, report, products, image_rows)
    write_readme(out_dir, image_dir)
    print_summary(out_dir, products, image_rows, report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--own-catalog", type=Path, default=Path("data/catalog/own_products.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/catalog/reference_dataset"))
    parser.add_argument("--image-dir", type=Path, default=Path("var/reference_images/official_products"))
    parser.add_argument("--brands", nargs="+", default=["resanta", "vihr", "huter"])
    parser.add_argument("--include-competitors", action="store_true")
    parser.add_argument("--competitor-brands", nargs="+", default=["zubr", "interskol", "patriot", "fubag"])
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--max-images-per-product", type=int, default=5)
    parser.add_argument("--max-products-per-source", type=int, default=0)
    parser.add_argument("--request-delay", type=float, default=0.15)
    parser.add_argument("--timeout", type=float, default=25)
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def collect_source_urls(
    session: requests.Session,
    source: Source,
    timeout: float,
    max_products: int = 0,
) -> list[str]:
    if source.brand_id == "zubr":
        return collect_zubr_product_urls(session, timeout, max_products)
    return collect_sitemap_urls(session, source.sitemap_urls, timeout)


def fetch_page_text(session: requests.Session, url: str, source: Source, timeout: float) -> str:
    if source.brand_id == "zubr":
        text = fetch_text_with_curl(jina_url(url), min(timeout, 20))
        if "Warning: Target URL returned error 404" in text or "# Страница не найдена" in text:
            return ""
        return text
    response = session.get(url, timeout=timeout)
    if response.status_code != 200 or "text/html" not in response.headers.get("content-type", ""):
        return ""
    return response.text


def jina_url(url: str) -> str:
    return JINA_READER_PREFIX + url


def fetch_text_with_curl(url: str, timeout: float) -> str:
    seconds = max(3, int(timeout))
    try:
        result = subprocess.run(
            ["curl", "-L", "--max-time", str(seconds), "-A", USER_AGENT, url],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=seconds + 2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout


def collect_sitemap_urls(session: requests.Session, sitemap_urls: tuple[str, ...], timeout: float) -> list[str]:
    seen_sitemaps: set[str] = set()
    seen_urls: set[str] = set()
    urls: list[str] = []

    def visit(sitemap_url: str) -> None:
        if sitemap_url in seen_sitemaps:
            return
        seen_sitemaps.add(sitemap_url)
        try:
            response = session.get(sitemap_url, timeout=timeout)
        except requests.RequestException:
            return
        if response.status_code != 200:
            return
        try:
            root = ElementTree.fromstring(response.content)
        except ElementTree.ParseError:
            return
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        sitemap_locs = [node.text for node in root.findall(".//sm:sitemap/sm:loc", ns) if node.text]
        if sitemap_locs:
            for loc in sitemap_locs:
                visit(loc)
            return
        for loc_node in root.findall(".//sm:url/sm:loc", ns):
            loc = loc_node.text
            if not loc or loc in seen_urls:
                continue
            seen_urls.add(loc)
            urls.append(loc)

    for sitemap_url in sitemap_urls:
        visit(sitemap_url)
    return urls


def collect_zubr_product_urls(session: requests.Session, timeout: float, max_products: int = 0) -> list[str]:
    request_timeout = (5, min(timeout, 10))
    queue = collect_zubr_category_seeds(session, request_timeout) or list(ZUBR_START_CATEGORIES)
    seen_pages: set[str] = set()
    seen_products: set[str] = set()
    products: list[str] = []
    max_category_pages = 350

    while queue and len(seen_pages) < max_category_pages:
        page_url = queue.pop(0)
        if page_url in seen_pages:
            continue
        seen_pages.add(page_url)
        text = fetch_text_with_curl(jina_url(page_url), request_timeout[1])
        if not text:
            continue
        for link in extract_markdown_links(text):
            normalized = normalize_zubr_url(link)
            if not normalized:
                continue
            if is_zubr_product_url(normalized):
                if normalized not in seen_products:
                    seen_products.add(normalized)
                    products.append(normalized)
                    if max_products and len(products) >= max_products:
                        return products
                continue
            if is_zubr_category_url(normalized) and normalized not in seen_pages and normalized not in queue:
                queue.append(normalized)
    return products


def collect_zubr_category_seeds(
    session: requests.Session,
    timeout: tuple[float, float],
) -> list[str]:
    text = fetch_text_with_curl(jina_url("https://zubr.ru/"), timeout[1])
    if not text:
        return []
    urls = set()
    for value in re.findall(r'\["SECTION_PAGE_URL"\]=>\s+string\(\d+\)\s+"([^"]+/)"', text):
        urls.add(urljoin("https://zubr.ru/", value))
    for link in extract_markdown_links(text):
        normalized = normalize_zubr_url(link)
        if normalized and is_zubr_category_url(normalized):
            urls.add(normalized)
    categories = sorted(urls, key=lambda item: (-urlparse(item).path.count("/"), item))
    return categories


def extract_markdown_links(text: str) -> list[str]:
    links = re.findall(r"\((https?://[^)\s]+)\)", text)
    links.extend(re.findall(r"https?://[^\]\)\s]+", text))
    return links


def normalize_zubr_url(url: str) -> str:
    url = url.split("#", 1)[0].rstrip()
    parsed = urlparse(url)
    if parsed.netloc == "zubr.itech-test.ru":
        parsed = parsed._replace(scheme="https", netloc="zubr.ru")
        url = parsed.geturl()
    if parsed.netloc != "zubr.ru":
        return ""
    if any(part in parsed.path for part in ("/upload/", "/dist/", "/bitrix/")):
        return ""
    return url


def is_zubr_product_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc == "zubr.ru" and "ID=" in parsed.query and not parsed.path.startswith("/guarantee/")


def is_zubr_category_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc != "zubr.ru" or parsed.query or not parsed.path.endswith("/"):
        return False
    blocked = (
        "/about/",
        "/articles/",
        "/career/",
        "/contacts/",
        "/documentation/",
        "/guarantee/",
        "/news/",
        "/promo_products/",
        "/services/",
        "/support/",
        "/where_to_buy/",
    )
    return not any(parsed.path.startswith(prefix) for prefix in blocked)


def is_obvious_non_product_url(url: str, source: Source) -> bool:
    path = urlparse(url).path.strip("/").lower()
    if not path:
        return True
    if source.dataset_role == "competitor":
        segments = [segment for segment in path.split("/") if segment]
        if source.brand_id == "interskol":
            return not path.startswith("product/")
        if source.brand_id == "patriot":
            if not path.startswith("catalog/") or len(segments) < 2:
                return True
            return len(segments) == 2 and "patriot" not in segments[-1]
        if source.brand_id == "fubag":
            return not path.startswith("catalog/") or len(segments) < 3
        if source.brand_id == "zubr":
            return not is_zubr_product_url(url)

    blocked_prefixes = [
        "kompaniya",
        "company",
        "about",
        "news",
        "service",
        "servis",
        "contacts",
        "contact",
        "contac",
        "pay-delivery",
        "dostavka",
        "delivery",
        "video",
        "partners",
        "for-legal-entities",
        "yuridicheskim-litsam",
        "organization_details",
        "politika",
        "privacy",
        "public-oferta",
        "oferta",
        "p-oferta",
        "soglashenie",
        "soglasie",
        "agreement",
        "polzovatelskoe",
        "pravila-polzovaniya",
        "warranty",
        "guarantee",
        "quiz",
        "obratnaya",
        "luchshie-predlozheniya",
        "novinki",
    ]
    if source.dataset_role == "own_target":
        blocked_prefixes.extend(("category/", "catalog/"))
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in blocked_prefixes)


def parse_product_page(
    html: str,
    url: str,
    source: Source,
    max_images: int,
) -> dict[str, Any] | None:
    if source.brand_id == "zubr":
        return parse_zubr_markdown_page(html, url, source, max_images)
    soup = BeautifulSoup(html, "html.parser")
    product_data = find_product_jsonld(soup) or {}
    sku = normalize_text(str(product_data.get("sku") or ""))
    if not sku:
        sku = extract_sku_from_soup(soup)
    if not sku:
        return None
    name = normalize_text(str(product_data.get("name") or ""))
    if not name:
        name = extract_product_name(soup)
    if not name:
        return None
    brand_name = extract_brand_name(product_data) or source.brand_name
    brand_id = source.brand_id
    category = extract_category(soup, name)
    image_urls = extract_image_urls(soup, product_data, url, max_images)
    if not image_urls:
        return None
    return {
        "product_id": "",
        "dataset_role": source.dataset_role,
        "brand_id": brand_id,
        "brand_name": brand_name,
        "sku": sku,
        "catalog_sku_id": "",
        "is_in_own_catalog": False,
        "category": category,
        "model_name": name,
        "name": name,
        "source_name": source.brand_name,
        "source_url": url,
        "image_urls": image_urls,
    }


def parse_zubr_markdown_page(
    text: str,
    url: str,
    source: Source,
    max_images: int,
) -> dict[str, Any] | None:
    if "# Страница не найдена" in text:
        return None
    name = extract_zubr_name(text)
    sku = extract_zubr_sku(text)
    image_urls = extract_zubr_image_urls(text, max_images)
    if not name or not sku or not image_urls:
        return None
    return {
        "product_id": "",
        "dataset_role": source.dataset_role,
        "brand_id": source.brand_id,
        "brand_name": source.brand_name,
        "sku": sku,
        "catalog_sku_id": "",
        "is_in_own_catalog": False,
        "category": extract_zubr_category(text),
        "model_name": name,
        "name": name,
        "source_name": source.brand_name,
        "source_url": url,
        "image_urls": image_urls,
    }


def extract_zubr_name(text: str) -> str:
    for match in re.finditer(r"^#\s+(.+)$", text, re.M):
        name = normalize_text(match.group(1))
        if name and "Страница не найдена" not in name and "Предложение" not in name:
            return name
    title_match = re.search(r"^Title:\s+(.+)$", text, re.M)
    return normalize_text(title_match.group(1)) if title_match else ""


def extract_zubr_sku(text: str) -> str:
    patterns = [
        r"\|\s*Артикул\s*\|\s*(?:\[)?([^|\]\n]+)",
        r"\bАртикул\s+([0-9A-Za-zА-Яа-яЁё./\\_\-\s]+?)(?:\s{2,}|\n|\|)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            value = re.sub(r"\]\([^)]+\)", "", match.group(1))
            return normalize_text(value)
    return ""


def extract_zubr_category(text: str) -> str:
    breadcrumb_match = re.search(r"\[Главная\]\([^)]+\)>(.+?)\s+#", text, re.S)
    if not breadcrumb_match:
        return ""
    names = re.findall(r"\[([^\]]+)\]\(https://zubr\.ru/[^)]+\)", breadcrumb_match.group(1))
    return normalize_text(names[-1]) if names else ""


def extract_zubr_image_urls(text: str, max_images: int) -> list[str]:
    urls = re.findall(r"!\[[^\]]*\]\((https://zubr\.ru/[^)]+)\)", text)
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if "/upload/" not in url:
            continue
        normalized = normalize_image_url(url)
        if not is_product_image_url(normalized):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
        if len(deduped) >= max_images:
            break
    return deduped


def find_product_jsonld(soup: BeautifulSoup) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for script in soup.find_all("script", attrs={"type": re.compile("ld\\+json", re.I)}):
        raw = script.string or script.get_text()
        if not raw:
            continue
        raw = raw.strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        candidates.extend(flatten_jsonld(data))
    for candidate in candidates:
        type_value = candidate.get("@type")
        types = type_value if isinstance(type_value, list) else [type_value]
        if any(str(item).lower() == "product" for item in types):
            return candidate
    return None


def flatten_jsonld(data: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if isinstance(data, dict):
        result.append(data)
        graph = data.get("@graph")
        if isinstance(graph, list):
            for item in graph:
                result.extend(flatten_jsonld(item))
    elif isinstance(data, list):
        for item in data:
            result.extend(flatten_jsonld(item))
    return result


def extract_brand_name(product_data: dict[str, Any]) -> str:
    brand = product_data.get("brand")
    if isinstance(brand, dict):
        return normalize_text(str(brand.get("name") or ""))
    if isinstance(brand, str):
        return normalize_text(brand)
    return ""


def extract_product_name(soup: BeautifulSoup) -> str:
    for meta in soup.select("[itemtype*='Product'] meta[itemprop='name'], .product-info meta[itemprop='name']"):
        value = normalize_text(meta.get("content") or "")
        if value:
            return value
    for h1 in soup.find_all("h1"):
        value = normalize_text(h1.get_text(" ", strip=True))
        if not value:
            continue
        value = re.split(r"\s+Сравнение товаров\b|\s+Добавить в список сравнения\b", value)[0]
        if value and "Корзина" not in value:
            return value
    meta_title = soup.select_one("meta[property='og:title'], meta[name='og:title']")
    return normalize_text(meta_title.get("content") if meta_title else "")


def extract_sku_from_soup(soup: BeautifulSoup) -> str:
    preferred_selectors = [
        ".article__value",
        ".vendor-code",
        ".product-info-headnote__article",
        "[itemprop='additionalProperty']",
    ]
    for selector in preferred_selectors:
        for node in soup.select(selector):
            value = normalize_text(node.get("content") or node.get_text(" ", strip=True))
            sku = extract_sku_from_text(value) or extract_article_like_value(value)
            if sku:
                return sku
    meta_sku = soup.select_one("[itemprop='sku']")
    if meta_sku:
        value = normalize_text(meta_sku.get("content") or meta_sku.get_text(" ", strip=True))
        sku = extract_sku_from_text(value) or extract_article_like_value(value)
        if sku:
            return sku
    return extract_sku_from_text(soup.get_text(" ", strip=True))


def extract_image_urls(
    soup: BeautifulSoup,
    product_data: dict[str, Any],
    page_url: str,
    max_images: int,
) -> list[str]:
    urls: list[str] = []

    for node in soup.select("[data-type='image'][data-big], [data-type='image'][data-img]"):
        for attr in ("data-big", "data-img", "data-preview", "data-thumbnail"):
            value = node.get(attr)
            if value:
                urls.append(urljoin(page_url, value))
                break

    for node in soup.select(
        ".detail-slider a[href], "
        ".detail-slider img[src], "
        ".detail-slider img[data-src], "
        ".product-detail-gallery a[href], "
        ".product-detail-gallery img[src], "
        ".product-detail-gallery img[data-src], "
        "a[data-fancybox='gallery'][href], "
        "a[data-fancybox='gal'][href], "
        "link[itemprop='image'][href]"
    ):
        urls.extend(extract_urls_from_node(node, page_url))

    images = product_data.get("image")
    if isinstance(images, str):
        urls.append(urljoin(page_url, images))
    elif isinstance(images, list):
        for image in images:
            if isinstance(image, str):
                urls.append(urljoin(page_url, image))

    for meta_selector in ("meta[property='og:image']", "meta[name='og:image']"):
        for meta in soup.select(meta_selector):
            value = meta.get("content")
            if value:
                urls.append(urljoin(page_url, value))

    for img in soup.select("article img[src], .product img[src], [itemscope][itemtype*='Product'] img[src]"):
        urls.extend(extract_urls_from_node(img, page_url))

    for img in soup.select("img[src], img[data-src], img[srcset]"):
        raw_values = " ".join(filter(None, [img.get("src"), img.get("data-src"), img.get("srcset")])).lower()
        if "interskol-develop%2fpublic" in raw_values or "interskol-develop/public" in raw_values:
            urls.extend(extract_urls_from_node(img, page_url))

    product_name = normalize_text(str(product_data.get("name") or "")) or extract_product_name(soup)
    for img in soup.select("img[src], img[data-src], img[srcset]"):
        alt_text = normalize_text(" ".join(filter(None, [img.get("alt"), img.get("title")])))
        if product_name and not text_matches_product_name(alt_text, product_name):
            continue
        urls.extend(extract_urls_from_node(img, page_url))

    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        normalized = normalize_image_url(url)
        if not is_product_image_url(normalized):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
        if len(deduped) >= max_images:
            break
    return deduped


def extract_category(soup: BeautifulSoup, product_name: str) -> str:
    selectors = [
        ".breadcrumbs a span",
        ".breadcrumbs a",
        ".breadcrumb a",
        ".product--parent--link",
    ]
    parts: list[str] = []
    for selector in selectors:
        parts = [normalize_text(node.get_text(" ", strip=True)) for node in soup.select(selector)]
        parts = [part for part in parts if part and part.lower() not in {"главная", "каталог"}]
        if parts:
            break
    if parts:
        return parts[-1]
    return ""


def extract_sku_from_text(text: str) -> str:
    text = normalize_text(text)
    for pattern in (
        r"Артикул\s*[:№#]?\s*([0-9A-Za-zА-Яа-яЁё./\\_-]+)",
        r"Код\s*[:№#]?\s*([0-9A-Za-zА-Яа-яЁё./\\_-]+)",
    ):
        match = re.search(pattern, text, re.I)
        if match:
            return normalize_text(match.group(1))
    return ""


def extract_article_like_value(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    text = re.sub(r"^(Артикул|Код)\s*[:№#]?\s*", "", text, flags=re.I).strip()
    if re.fullmatch(r"[0-9A-Za-zА-Яа-яЁё./\\_-]{3,}", text):
        return text
    return ""


def extract_urls_from_node(node: Any, page_url: str) -> list[str]:
    urls: list[str] = []
    for attr in ("href", "data-big", "data-img", "data-src", "data-original", "data-lazy", "src"):
        value = node.get(attr)
        if value:
            urls.append(urljoin(page_url, value))
    srcset = node.get("srcset")
    if srcset:
        urls.extend(urljoin(page_url, item) for item in parse_srcset(srcset))
    style = node.get("style") or ""
    for match in re.finditer(r"url\((['\"]?)(.*?)\1\)", style):
        urls.append(urljoin(page_url, match.group(2)))
    return urls


def parse_srcset(value: str) -> list[str]:
    urls: list[str] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        urls.append(item.split()[0])
    return urls


def text_matches_product_name(text: str, product_name: str) -> bool:
    text = normalize_text(text).lower()
    product_name = normalize_text(product_name).lower()
    if not text or not product_name:
        return False
    if text in product_name or product_name in text:
        return True
    tokens = [token for token in re.split(r"[^0-9a-zа-яё]+", product_name) if len(token) >= 3]
    if not tokens:
        return False
    hits = sum(1 for token in tokens if token in text)
    return hits >= min(3, len(tokens))


def normalize_image_url(url: str) -> str:
    url = url.split("#", 1)[0]
    parsed = urlparse(url)
    if parsed.path == "/_next/image":
        query = parse_qs(parsed.query)
        source_urls = query.get("url")
        if source_urls:
            return unquote(source_urls[0])
    if parsed.netloc.endswith("fubag.ru") and parsed.path.startswith("/upload/resize_cache/iblock/"):
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 7:
            path = "/upload/iblock/" + "/".join([parts[3], parts[4], parts[-1]])
            return parsed._replace(path=path, query="").geturl()
    if parsed.netloc.endswith("zubr.ru") and parsed.path.startswith("/upload/resize_cache/iblock/"):
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 6:
            path = "/upload/iblock/" + "/".join([parts[3], parts[-1]])
            return parsed._replace(path=path, query="").geturl()
    if not parsed.scheme.startswith("http"):
        return url
    return url


def is_product_image_url(url: str) -> bool:
    lower = url.lower()
    parsed = urlparse(url)
    if parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        return False
    if lower.startswith("data:") or lower.endswith(".svg"):
        return False
    bad_parts = ("sprite", "logo", "icon", "captcha", "loader", "loading", "blank")
    if any(part in lower for part in bad_parts):
        return False
    good_ext = (".jpg", ".jpeg", ".png", ".webp")
    if any(ext in lower for ext in good_ext):
        return True
    return "/images/" in lower or "/upload/" in lower or "/wa-data/public/shop/products/" in lower


def download_image(
    session: requests.Session,
    image_url: str,
    product: dict[str, Any],
    rank: int,
    image_dir: Path,
    timeout: float,
) -> dict[str, Any]:
    target_dir = image_dir / product["brand_id"] / safe_filename(product["product_id"])
    target_dir.mkdir(parents=True, exist_ok=True)
    headers = {"Referer": product["source_url"]}
    download_url = image_download_url(image_url, product)
    try:
        response = session.get(download_url, headers=headers, timeout=timeout)
    except requests.RequestException:
        return empty_download_info("request_error")
    if response.status_code != 200 or not response.content:
        return empty_download_info(f"http_{response.status_code}")

    content_type = response.headers.get("content-type", "").split(";")[0].lower()
    ext = IMAGE_EXT_BY_TYPE.get(content_type) or Path(urlparse(image_url).path).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        ext = ".jpg"

    digest = hashlib.sha256(response.content).hexdigest()
    path = target_dir / f"image_{rank:02d}_{digest[:10]}{ext}"
    if not path.exists():
        path.write_bytes(response.content)

    try:
        with Image.open(path) as image:
            width, height = image.size
    except Exception:
        path.unlink(missing_ok=True)
        return empty_download_info("invalid_image")

    if width < 120 or height < 120:
        return empty_download_info("too_small")
    return {
        "local_path": str(path),
        "download_status": "ok",
        "width": width,
        "height": height,
        "bytes": len(response.content),
        "sha256": digest,
    }


def image_download_url(image_url: str, product: dict[str, Any]) -> str:
    if product.get("brand_id") == "zubr" and urlparse(image_url).netloc.endswith("zubr.ru"):
        return "https://wsrv.nl/?url=" + image_url.replace("https://", "").replace("http://", "")
    return image_url


def empty_download_info(status: str) -> dict[str, Any]:
    return {
        "local_path": "",
        "download_status": status,
        "width": "",
        "height": "",
        "bytes": "",
        "sha256": "",
    }


def build_image_row(product: dict[str, Any], image_url: str, rank: int) -> dict[str, Any]:
    image_id = f"{product['product_id']}_IMG_{rank:02d}"
    return {
        "image_id": image_id,
        "product_id": product["product_id"],
        "dataset_role": product["dataset_role"],
        "brand_id": product["brand_id"],
        "brand_name": product["brand_name"],
        "sku": product["sku"],
        "catalog_sku_id": product["catalog_sku_id"],
        "category": product["category"],
        "model_name": product["model_name"],
        "source_name": product["source_name"],
        "source_url": product["source_url"],
        "image_url": image_url,
        "local_path": "",
        "source_rank": rank,
        "download_status": "",
        "width": "",
        "height": "",
        "bytes": "",
        "sha256": "",
    }


def add_missing_own_catalog_rows(
    products: list[dict[str, Any]],
    own_rows: list[dict[str, str]],
    requested_brands: set[str],
    seen_product_keys: set[tuple[str, str]],
) -> None:
    for row in own_rows:
        brand_id = row.get("brand_id", "")
        if brand_id not in requested_brands:
            continue
        article_codes = split_pipe(row.get("article_codes", ""))
        key_articles = [normalize_article(article) for article in article_codes if normalize_article(article)]
        if key_articles and any((brand_id, article) in seen_product_keys for article in key_articles):
            continue
        sku = article_codes[0] if article_codes else ""
        product_id = row.get("sku_id") or build_product_id(brand_id, sku, row.get("model_name", ""))
        products.append(
            {
                "product_id": product_id,
                "dataset_role": "own_target",
                "brand_id": brand_id,
                "brand_name": row.get("brand_name", ""),
                "sku": sku,
                "catalog_sku_id": row.get("sku_id", ""),
                "is_in_own_catalog": True,
                "category": row.get("category", ""),
                "model_name": row.get("model_name", ""),
                "name": row.get("model_name", ""),
                "source_name": "",
                "source_url": "",
                "page_status": "missing_official_reference",
                "image_count": 0,
                "reference_images": [],
                "reference_image_urls": [],
            }
        )


def load_own_catalog(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_own_article_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    index: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        brand_id = row.get("brand_id", "")
        for article in split_pipe(row.get("article_codes", "")):
            normalized = normalize_article(article)
            if normalized:
                index.setdefault((brand_id, normalized), row)
    return index


def split_pipe(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split("|") if item.strip()]


def normalize_article(value: str) -> str:
    value = normalize_text(value).replace("\\", "/")
    value = re.sub(r"\s+", "", value)
    return value.upper()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    return re.sub(r"\s+", " ", value).strip()


def build_product_id(brand_id: str, sku: str, source_url: str) -> str:
    key = normalize_article(sku) or slug_from_url(source_url)
    return f"{brand_id.upper()}_{safe_filename(key).upper()}"


def slug_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    return path.split("/")[-1] if path else "product"


def safe_filename(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = re.sub(r"[^0-9A-Za-zА-Яа-яЁё]+", "_", value)
    value = value.strip("_")
    return value or "item"


def write_outputs(out_dir: Path, products: list[dict[str, Any]], image_rows: list[dict[str, Any]]) -> None:
    write_products(out_dir / "products.csv", out_dir / "products.jsonl", products)
    write_images(out_dir / "images.csv", out_dir / "images.jsonl", image_rows)


def write_products(csv_path: Path, jsonl_path: Path, products: list[dict[str, Any]]) -> None:
    rows = []
    for product in products:
        row = {field: product.get(field, "") for field in PRODUCT_CSV_FIELDS}
        row["is_in_own_catalog"] = str(bool(product.get("is_in_own_catalog"))).lower()
        row["reference_images"] = "|".join(product.get("reference_images", []))
        row["reference_image_urls"] = "|".join(product.get("reference_image_urls", []))
        rows.append(row)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRODUCT_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for product in products:
            handle.write(json.dumps(product, ensure_ascii=False) + "\n")


def write_images(csv_path: Path, jsonl_path: Path, image_rows: list[dict[str, Any]]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=IMAGE_CSV_FIELDS)
        writer.writeheader()
        for row in image_rows:
            writer.writerow({field: row.get(field, "") for field in IMAGE_CSV_FIELDS})
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in image_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_own_products_with_references(out_dir: Path, own_catalog_path: Path, products: list[dict[str, Any]]) -> None:
    own_rows = load_own_catalog(own_catalog_path)
    refs_by_sku_id: dict[str, list[str]] = {}
    for product in products:
        sku_id = product.get("catalog_sku_id")
        refs = product.get("reference_images") or product.get("reference_image_urls") or []
        if sku_id and refs:
            refs_by_sku_id.setdefault(sku_id, []).extend(refs)
    if not own_rows:
        return
    fieldnames = list(own_rows[0].keys())
    if "reference_images" not in fieldnames:
        fieldnames.append("reference_images")
    out_path = out_dir / "own_products_with_references.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in own_rows:
            refs = refs_by_sku_id.get(row.get("sku_id", ""), [])
            if refs:
                row = dict(row)
                row["reference_images"] = "|".join(dict.fromkeys(refs))
            writer.writerow(row)


def write_report(out_dir: Path, report: dict[str, Any], products: list[dict[str, Any]], image_rows: list[dict[str, Any]]) -> None:
    report = dict(report)
    report["totals"] = {
        "products": len(products),
        "products_with_images": sum(1 for product in products if int(product.get("image_count") or 0) > 0),
        "image_rows": len(image_rows),
        "downloaded_images": sum(1 for row in image_rows if row.get("download_status") == "ok"),
        "own_products": sum(1 for product in products if product.get("dataset_role") == "own_target"),
        "competitor_products": sum(1 for product in products if product.get("dataset_role") == "competitor"),
    }
    with (out_dir / "run_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def write_readme(out_dir: Path, image_dir: Path) -> None:
    text = f"""# Reference Product Dataset

Generated by `scripts/collect_reference_catalog.py`.

Files:

- `products.csv` / `products.jsonl`: one row per product/model.
- `images.csv` / `images.jsonl`: one row per reference image.
- `own_products_with_references.csv`: copy of `data/catalog/own_products.csv` with `reference_images` filled where matched.
- `run_report.json`: source and coverage statistics.

Local images are stored outside the catalog manifest folder:

`{image_dir}`

`dataset_role` values:

- `own_target`: Ресанта, Вихрь, Huter from the official target catalogs and the existing own catalog.
- `competitor`: competitor seed brands collected from official public catalogs.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def print_summary(out_dir: Path, products: list[dict[str, Any]], image_rows: list[dict[str, Any]], report: dict[str, Any]) -> None:
    downloaded = sum(1 for row in image_rows if row.get("download_status") == "ok")
    print("\n[done]")
    print(f"products: {len(products)}")
    print(f"products_with_images: {sum(1 for item in products if int(item.get('image_count') or 0) > 0)}")
    print(f"image_rows: {len(image_rows)}")
    print(f"downloaded_images: {downloaded}")
    print(f"out: {out_dir}")


if __name__ == "__main__":
    main()

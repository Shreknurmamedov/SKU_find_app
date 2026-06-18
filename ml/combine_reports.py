"""Combine per-video audit JSONs into one SKU-presence summary.

    python3 ml/combine_reports.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

REPORTS = Path("reports")


def main() -> None:
    files = sorted(REPORTS.glob("audit_*.json"))
    if not files:
        print("no audit_*.json in reports/")
        return

    rows = []
    sku_by_key = {}
    object_brand_total = Counter()
    object_cat_total = Counter()
    object_model_total = Counter()
    tot = Counter()
    for f in files:
        r = json.loads(f.read_text())
        t = r["totals"]
        rows.append((Path(r["video"]).name, t))
        for k, v in t.items():
            tot[k] += v
        object_brand_total.update(r.get("by_brand_objects", {}))
        object_cat_total.update(r.get("by_category_objects", {}))
        object_model_total.update(r.get("by_model_objects", {}))
        for sku in r.get("sku_presence", []):
            key = sku.get("sku_key") or sku.get("sku_id") or f"{sku.get('brand')}|{sku.get('model')}"
            current = sku_by_key.get(key)
            if current is None:
                current = {**sku, "videos": [], "evidence_objects": 0}
                sku_by_key[key] = current
            current["videos"].append(Path(r["video"]).name)
            current["evidence_objects"] += int(sku.get("evidence_objects", 0))
            if float(sku.get("best_ocr_conf", 0.0)) > float(current.get("best_ocr_conf", 0.0)):
                current["best_ocr_conf"] = sku.get("best_ocr_conf", 0.0)
                current["best_crop"] = sku.get("best_crop")

    sku_presence = sorted(sku_by_key.values(), key=lambda s: (
        s.get("brand") or "", s.get("model") or "", s.get("article_codes") or "",
    ))
    brand_total = Counter(s.get("brand") or "—" for s in sku_presence)
    cat_total = Counter(s.get("category") for s in sku_presence if s.get("category"))
    model_total = Counter(_model_label(s) for s in sku_presence)
    tot["unique_skus"] = len(sku_presence)
    tot["unique_own_skus"] = sum(1 for s in sku_presence if s.get("is_own"))

    L = ["# Сводный SKU-аудит по всем видео", ""]
    L += ["| Видео | Уникальных SKU | Наших SKU | Кропов-доказательств | Кандидатных объектов | На проверку |",
          "|---|---|---|---|---|---|"]
    for name, t in rows:
        L.append(f"| {name} | {t.get('unique_skus', t.get('confident_sku', 0))} | "
                 f"{t.get('unique_own_skus', 0)} | {t.get('sku_evidence_objects', 0)} | "
                 f"{t.get('candidate_objects', t.get('physical_objects', 0))} | "
                 f"{t.get('needs_review_objects', t.get('needs_review', 0))} |")
    L.append(f"| **ИТОГО** | **{tot['unique_skus']}** | **{tot['unique_own_skus']}** | "
             f"**{tot.get('sku_evidence_objects', 0)}** | "
             f"**{tot.get('candidate_objects', tot.get('physical_objects', 0))}** | "
             f"**{tot.get('needs_review_objects', tot.get('needs_review', 0))}** |")

    L += ["", "## Найденные SKU/артикулы", "",
          "| Бренд | Модель | Артикул | Группа | Видео | Доказательств |",
          "|---|---|---|---|---|---:|"]
    L += [
        f"| {s.get('brand') or '—'} | {s.get('model') or '—'} | "
        f"{s.get('article_codes') or s.get('sku_id') or '—'} | "
        f"{s.get('category') or '—'} | {', '.join(sorted(set(s.get('videos', []))))} | "
        f"{s.get('evidence_objects', 0)} |"
        for s in sku_presence
    ] or ["| — | — | — | — | — | 0 |"]

    L += ["", "## Наши бренды (уникальные SKU)", "", "| Бренд | SKU |", "|---|---|"]
    L += [f"| {b} | {n} |" for b, n in brand_total.most_common() if b and b != "—"] or ["| — | 0 |"]

    L += ["", "## Группы товара (уникальные SKU)", "", "| Группа | SKU |", "|---|---|"]
    L += [f"| {c} | {n} |" for c, n in cat_total.most_common()] or ["| — | 0 |"]

    L += ["", "## Модели/артикулы", "", "| Модель / артикул | SKU |", "|---|---|"]
    L += [f"| {m} | {n} |" for m, n in model_total.most_common()] or ["| — | 0 |"]

    if object_brand_total:
        L += ["", "## Диагностика: объектные кропы по брендам", "",
              "| Бренд | Объектов |", "|---|---:|"]
        L += [f"| {b} | {n} |" for b, n in object_brand_total.most_common()]

    out = REPORTS / "audit_summary.md"
    out.write_text("\n".join(L) + "\n")
    print(f"combined {len(files)} reports -> {out}")
    print(f"TOTAL unique_skus={tot['unique_skus']} candidate_objects="
          f"{tot.get('candidate_objects', tot.get('physical_objects', 0))} "
          f"review_objects={tot.get('needs_review_objects', tot.get('needs_review', 0))}")


def _model_label(sku: dict) -> str:
    brand = sku.get("brand") or "—"
    model = sku.get("model") or "—"
    article = sku.get("article_codes") or sku.get("sku_id") or ""
    return f"{brand} {model}" + (f" ({article})" if article else "")


if __name__ == "__main__":
    main()

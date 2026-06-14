"""Combine per-video audit JSONs into one cross-store summary.

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
    brand_total = Counter()
    cat_total = Counter()
    model_total = Counter()
    tot = Counter()
    for f in files:
        r = json.loads(f.read_text())
        t = r["totals"]
        rows.append((Path(r["video"]).name, t))
        for k, v in t.items():
            tot[k] += v
        brand_total.update(r.get("by_brand", {}))
        cat_total.update(r.get("by_category", {}))
        model_total.update(r.get("by_model", {}))

    L = ["# Сводный SKU-аудит по всем видео", ""]
    L += ["| Видео | Объектов | Наши | Конкур./н.о. | Уверенный SKU | На проверку |",
          "|---|---|---|---|---|---|"]
    for name, t in rows:
        L.append(f"| {name} | {t['physical_objects']} | {t['own_brand_objects']} | "
                 f"{t['competitor_or_unknown']} | {t['confident_sku']} | {t['needs_review']} |")
    L.append(f"| **ИТОГО** | **{tot['physical_objects']}** | **{tot['own_brand_objects']}** | "
             f"**{tot['competitor_or_unknown']}** | **{tot['confident_sku']}** | **{tot['needs_review']}** |")

    L += ["", "## Наши бренды (все ТТ)", "", "| Бренд | Объектов |", "|---|---|"]
    L += [f"| {b} | {n} |" for b, n in brand_total.most_common() if b and b != "—"] or ["| — | 0 |"]

    L += ["", "## Группы товара наших брендов (все ТТ)", "", "| Группа | Объектов |", "|---|---|"]
    L += [f"| {c} | {n} |" for c, n in cat_total.most_common()] or ["| — | 0 |"]

    L += ["", "## Модели, определенные уверенно (все ТТ)", "", "| Модель | Объектов |", "|---|---|"]
    L += [f"| {m} | {n} |" for m, n in model_total.most_common()] or ["| — | 0 |"]

    out = REPORTS / "audit_summary.md"
    out.write_text("\n".join(L) + "\n")
    print(f"combined {len(files)} reports -> {out}")
    print(f"TOTAL objects={tot['physical_objects']} own={tot['own_brand_objects']} "
          f"confident_sku={tot['confident_sku']} review={tot['needs_review']}")


if __name__ == "__main__":
    main()

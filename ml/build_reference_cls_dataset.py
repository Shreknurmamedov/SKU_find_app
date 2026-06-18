"""Turn reference product photos into an ultralytics classification dataset.

The reference index metadata (``data/catalog/reference_index_yolo11n.jsonl``)
carries a brand for every photo. This builds an ImageFolder layout

    <out>/train/<brand_slug>/*.png
    <out>/val/<brand_slug>/*.png

so an overnight ``yolo classify train`` can learn a visual brand recognizer.

Brand (not SKU/category) is the target on purpose: there are 9 well-populated
brands but ~679 sparse categories and thousands of SKUs, so brand is the only
class space with enough photos per class to train reliably.

    python3 -m ml.build_reference_cls_dataset \
        --metadata data/catalog/reference_index_yolo11n.jsonl \
        --out ml/datasets/ref_brand_cls
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

# Cyrillic brand names -> ascii folder slug (folder names must be path-safe).
BRAND_SLUGS = {
    "Вихрь": "vihr",
    "Ресанта": "resanta",
    "Huter": "huter",
    "ЗУБР": "zubr",
    "FUBAG": "fubag",
    "Интерскол": "interskol",
    "PATRIOT": "patriot",
    "TEK": "tek",
}
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "",
    "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def slugify(name: str) -> str:
    if name in BRAND_SLUGS:
        return BRAND_SLUGS[name]
    out = []
    for ch in name.lower():
        if ch in _TRANSLIT:
            out.append(_TRANSLIT[ch])
        elif ch.isalnum() and ch.isascii():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "unknown"


def build(out: Path, *, metadata: Path = Path("data/catalog/reference_index_yolo11n.jsonl"),
          field: str = "brand_name", val_frac: float = 0.15, min_count: int = 50,
          copy: bool = False, limit: int = 0, seed: int = 0) -> dict:
    """Build the ImageFolder dataset; return the per-class summary dict."""
    rows: dict[str, list[str]] = {}
    n = 0
    for line in metadata.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        label = (d.get(field) or "").strip()
        path = d.get("image_path") or d.get("local_path") or ""
        if not label or label == "-" or not path or not os.path.exists(path):
            continue
        rows.setdefault(label, []).append(path)
        n += 1
        if limit and n >= limit:
            break

    classes = {lbl: paths for lbl, paths in rows.items() if len(paths) >= min_count}
    if not classes:
        raise SystemExit("no classes met --min-count; check metadata/--field")

    if out.exists():
        shutil.rmtree(out)
    rng = random.Random(seed)
    summary = {}
    for label, paths in sorted(classes.items()):
        slug = slugify(label)
        paths = list(paths)
        rng.shuffle(paths)
        n_val = max(1, int(len(paths) * val_frac))
        splits = {"val": paths[:n_val], "train": paths[n_val:]}
        for split, items in splits.items():
            dst_dir = out / split / slug
            dst_dir.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(items):
                ext = Path(src).suffix or ".png"
                dst = dst_dir / f"{slug}_{i:05d}{ext}"
                if copy:
                    shutil.copy2(src, dst)
                else:
                    os.symlink(os.path.abspath(src), dst)
        summary[slug] = {"label": label, "train": len(splits["train"]),
                         "val": len(splits["val"])}

    (out / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", type=Path,
                    default=Path("data/catalog/reference_index_yolo11n.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("ml/datasets/ref_brand_cls"))
    ap.add_argument("--field", default="brand_name",
                    help="metadata field used as the class label")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--min-count", type=int, default=50,
                    help="drop classes with fewer than this many photos")
    ap.add_argument("--copy", action="store_true",
                    help="copy files instead of symlinking (slower, portable)")
    ap.add_argument("--limit", type=int, default=0, help="cap rows (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    summary = build(args.out, metadata=args.metadata, field=args.field,
                    val_frac=args.val_frac, min_count=args.min_count,
                    copy=args.copy, limit=args.limit, seed=args.seed)
    total_tr = sum(s["train"] for s in summary.values())
    total_val = sum(s["val"] for s in summary.values())
    print(f"classes={len(summary)} train={total_tr} val={total_val} -> {args.out}")
    for slug, s in sorted(summary.items()):
        print(f"  {slug:12} {s['label']:16} train={s['train']:5} val={s['val']:4}")


if __name__ == "__main__":
    main()

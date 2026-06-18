"""End-to-end SKU presence audit of a shelf video.

    video -> detect+track product candidates
          -> re-id merge (collapse re-passes of the camera)
          -> OCR + catalog match each good crop (brand / model / article)
          -> aggregate unique SKU presence + uncertain bucket for review

    python3 ml/audit_video.py --video "ТТ Пэкстрой/IMG_8886.MOV" \
        --weights ml/runs/product_det/weights/best.pt

Outputs JSON + Markdown into reports/.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from ml import track_video, reid_merge
from ml.sku_recognize import Recognizer


STATUS_RANK = {"matched_sku": 4, "brand_only": 3, "category_only": 2, "unknown": 1}


def _sku_key(item: dict) -> str:
    return item.get("sku_id") or f"{item.get('brand') or ''}|{item.get('model') or ''}"


def _model_label(item: dict) -> str:
    brand = item.get("brand") or "—"
    model = item.get("model") or "—"
    article = item.get("article_codes") or item.get("sku_id") or ""
    return f"{brand} {model}" + (f" ({article})" if article else "")


def _unique_sku_presence(items: list[dict], review_conf: float) -> list[dict]:
    """Collapse repeated physical objects to one row per recognized SKU."""
    by_key: dict[str, dict] = {}
    for item in items:
        if item["status"] != "matched_sku" or item["ocr_conf"] < review_conf:
            continue
        key = _sku_key(item)
        row = by_key.get(key)
        if row is None:
            row = {
                "sku_key": key,
                "sku_id": item.get("sku_id"),
                "article_codes": item.get("article_codes"),
                "brand": item.get("brand"),
                "model": item.get("model"),
                "category": item.get("category"),
                "is_own": item.get("is_own", False),
                "evidence_objects": 0,
                "object_ids": [],
                "best_crop": item.get("crop"),
                "best_ocr_conf": item.get("ocr_conf", 0.0),
                "method": item.get("method"),
                "visual_score": item.get("visual_score"),
                "reference_image": item.get("reference_image"),
            }
            by_key[key] = row
        row["evidence_objects"] += 1
        row["object_ids"].append(item["object_id"])
        if item["ocr_conf"] > row["best_ocr_conf"]:
            row["best_ocr_conf"] = item["ocr_conf"]
            row["best_crop"] = item.get("crop")
            row["method"] = item.get("method")
            row["visual_score"] = item.get("visual_score")
            row["reference_image"] = item.get("reference_image")
    return sorted(by_key.values(), key=lambda r: (
        r.get("brand") or "", r.get("model") or "", r.get("article_codes") or "",
    ))


def _brand_category_presence(items: list[dict]) -> list[dict]:
    """Unique brand+category recognized when the exact SKU was NOT resolved.

    Business rule: if the article/model is not reliably readable, we report the
    brand and product category instead of guessing a SKU. These partial reads are
    genuine findings (not "re-shoot"), so they are aggregated separately from
    sku_presence and not gated by the SKU confidence floor.
    """
    by_key: dict[str, dict] = {}
    for item in items:
        if item["status"] not in ("brand_only", "category_only"):
            continue
        brand = item.get("brand")
        category = item.get("category")
        if not brand and not category:
            continue
        key = f"{brand or ''}|{category or ''}"
        row = by_key.get(key)
        if row is None:
            row = {
                "brand": brand,
                "category": category,
                "is_own": item.get("is_own", False),
                "evidence_objects": 0,
                "object_ids": [],
                "best_crop": item.get("crop"),
                "best_ocr_conf": item.get("ocr_conf", 0.0),
                "method": item.get("method"),
            }
            by_key[key] = row
        row["evidence_objects"] += 1
        row["object_ids"].append(item["object_id"])
        if item["ocr_conf"] > row["best_ocr_conf"]:
            row["best_ocr_conf"] = item["ocr_conf"]
            row["best_crop"] = item.get("crop")
            row["method"] = item.get("method")
    return sorted(by_key.values(), key=lambda r: (
        r.get("brand") or "￿", r.get("category") or "",
    ))


def _recognition_score(result) -> float:
    return STATUS_RANK.get(result.status, 0) * 10 + result.confidence + len(result.text) * 1e-4


_CROP_SCORE_CACHE: dict[str, float] = {}


def _ocr_candidate_score(path: Path) -> float:
    cached = _CROP_SCORE_CACHE.get(str(path))
    if cached is not None:
        return cached
    try:
        import cv2

        img = cv2.imread(str(path))
        if img is None or img.size == 0:
            return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        lap = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        edge_frac = float((edges > 0).mean())
        small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) \
            if min(gray.shape[:2]) > 80 else gray
        thr = cv2.adaptiveThreshold(
            small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 7,
        )
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(thr, 8)
        h, w = small.shape[:2]
        components = 0
        for i in range(1, n_labels):
            _, _, bw, bh, area = stats[i]
            if 3 <= bw <= w * 0.35 and 3 <= bh <= h * 0.25 and 4 <= area <= w * h * 0.08:
                components += 1
        score = lap + edge_frac * 3000.0 + components * 4.0
    except Exception:
        score = 0.0
    _CROP_SCORE_CACHE[str(path)] = score
    return score


def _crop_candidates(obj: dict, limit: int, *, out: Path | None = None,
                     prefer_ocr: bool = False) -> list[str]:
    rows = obj.get("crop_candidates") or [{"crop": obj["rep_crop"]}]
    if prefer_ocr and out is not None and len(rows) > 1:
        rows = sorted(
            rows,
            key=lambda row: (
                _ocr_candidate_score(out / row.get("crop", "")),
                float(row.get("score", 0.0)),
            ),
            reverse=True,
        )
    seen = set()
    crops = []
    for row in rows:
        crop = row.get("crop")
        if crop and crop not in seen:
            seen.add(crop)
            crops.append(crop)
        if len(crops) >= limit:
            break
    return crops or [obj["rep_crop"]]


def _recognize_object(rec: Recognizer, out: Path, obj: dict, *,
                      rotations, max_crops: int, allow_retry: bool):
    best = None
    votes: dict[str, dict] = {}
    used_crops = []
    crop_limit = max_crops if allow_retry else 1
    crops = _crop_candidates(obj, crop_limit, out=out, prefer_ocr=True)
    for idx, rel_crop in enumerate(crops):
        used_crops.append(rel_crop)
        result = rec.recognize(
            out / rel_crop,
            rotations=rotations,
            enhance=False,
            use_visual=False,
        )
        if best is None or _recognition_score(result) > _recognition_score(best):
            best = result
        if result.status == "matched_sku" and result.sku_id:
            bucket = votes.setdefault(result.sku_id, {"count": 0, "best": result})
            bucket["count"] += 1
            if _recognition_score(result) > _recognition_score(bucket["best"]):
                bucket["best"] = result
        if result.status == "matched_sku" and result.confidence >= 0.86:
            break
        if idx == 0 and result.status == "matched_sku":
            break
    if not allow_retry:
        if votes:
            best_vote = max(votes.values(), key=lambda v: (v["count"], _recognition_score(v["best"])))
            return best_vote["best"], used_crops, {sku: v["count"] for sku, v in votes.items()}
        return best, used_crops, {}
    if not votes and best is not None and crops:
        enhanced = rec.recognize(
            out / crops[0],
            rotations=rotations,
            enhance=True,
            use_visual=False,
        )
        if _recognition_score(enhanced) > _recognition_score(best):
            best = enhanced
        if enhanced.status == "matched_sku" and enhanced.sku_id:
            votes[enhanced.sku_id] = {"count": 1, "best": enhanced}
    if not votes and best is not None and crops:
        visual = rec.recognize_visual(out / crops[0], text=best.text)
        if visual is not None and _recognition_score(visual) > _recognition_score(best):
            best = visual
            if visual.sku_id:
                votes[visual.sku_id] = {"count": 1, "best": visual}
    if votes:
        best_vote = max(votes.values(), key=lambda v: (v["count"], _recognition_score(v["best"])))
        return best_vote["best"], used_crops, {sku: v["count"] for sku, v in votes.items()}
    return best, used_crops, {}


def audit(video: Path, weights: Path, *, work_dir: Path, reports_dir: Path,
          embed_weights="yolo11n.pt", sim=0.96, review_conf=0.75,
          vid_stride=6, device="mps", reuse_tracks=False,
          conf=0.35, min_frames=4, ocr_crops_per_object=2,
          ocr_retry_objects=10) -> dict:
    stem = video.stem
    out = work_dir / stem

    # 1) detect + track -> unique-ish physical objects
    if reuse_tracks and (out / "tracks.json").exists():
        print(f"[audit] reusing existing tracks: {out/'tracks.json'}")
    else:
        track_video.run(video, weights, out, vid_stride=vid_stride, device=device,
                        conf=conf, min_frames=min_frames,
                        crops_per_track=max(1, ocr_crops_per_object + 2))
    # 2) re-id merge of camera re-passes
    data = reid_merge.merge(out, sim_threshold=sim, weights=embed_weights)
    objects = data["reid"].get("objects", [])

    # 3) recognize each unique object.
    # Camera orientation is constant within a video, so instead of OCR-ing every
    # crop at 4 rotations, vote the dominant rotation on the most confident
    # objects, then read the rest at that single rotation (~4x fewer OCR calls).
    rec = Recognizer()
    sample_size = min(8, max(0, len(objects)))
    sample = sorted(objects, key=lambda o: -o["peak_conf"])[:sample_size]
    cached: dict[int, object] = {}
    votes = Counter()
    for obj in sample:
        r, _, _ = _recognize_object(
            rec, out, obj, rotations=(0, 270, 90, 180),
            max_crops=min(2, max(1, ocr_crops_per_object)),
            allow_retry=False,
        )
        cached[obj["object_id"]] = r
        # only a real catalog match reliably reveals orientation; OCR emits
        # noise "text" at every rotation, so text length is not trustworthy
        if r.status != "unknown":
            votes[r.rotation] += 1

    if votes and votes.most_common(1)[0][1] >= 2:
        dom = votes.most_common(1)[0][0]
        fast_rotations = (dom,)  # orientation is clear -> fast single rotation
        retry_rotations = (dom,)
    else:
        dom = None
        # Inconclusive orientation should not explode into 4x OCR for every
        # low-confidence object. Keep the expensive full search for the top-N
        # retry set; ordinary objects get a fast upright read and go to review
        # if text is not readable.
        fast_rotations = (0,)
        retry_rotations = (0, 270, 90, 180)
    print(f"[audit] dominant rotation = {dom} (votes={dict(votes)}); "
          f"fast uses {fast_rotations}; retry uses {retry_rotations}")

    items = []
    retry_ids = {
        obj["object_id"]
        for obj in sorted(objects, key=lambda o: -o["peak_conf"])[:max(0, ocr_retry_objects)]
    }
    for obj in objects:
        r = cached.get(obj["object_id"])
        used_crops = _crop_candidates(obj, 1, out=out, prefer_ocr=True)
        sku_votes = {}
        if r is None or (r.status != "matched_sku" and obj["object_id"] in retry_ids):
            is_retry = obj["object_id"] in retry_ids
            r, used_crops, sku_votes = _recognize_object(
                rec, out, obj, rotations=retry_rotations if is_retry else fast_rotations,
                max_crops=ocr_crops_per_object,
                allow_retry=is_retry,
            )
        crop_path = out / used_crops[0]
        items.append({
            "object_id": obj["object_id"],
            "track_ids": obj["track_ids"],
            "crop": str(crop_path.relative_to(work_dir.parent)) if work_dir.parent in crop_path.parents else str(crop_path),
            "ocr_crops": used_crops,
            "sku_votes": sku_votes,
            "status": r.status,
            "is_own": r.is_own,
            "brand": r.brand,
            "model": r.model,
            "category": r.category,
            "sku_id": r.sku_id,
            "article_codes": r.article_codes,
            "ocr_conf": round(r.confidence, 3),
            "method": r.method,
            "text": r.text[:160],
            "visual_score": r.visual_score,
            "visual_margin": r.visual_margin,
            "reference_image": r.reference_image,
        })

    # 4) aggregate
    total = len(items)
    own = [it for it in items if it["is_own"]]
    review = [it for it in items
              if it["status"] != "matched_sku" or it["ocr_conf"] < review_conf]
    confident = [it for it in items
                 if it["status"] == "matched_sku" and it["ocr_conf"] >= review_conf]
    sku_presence = _unique_sku_presence(items, review_conf)
    own_skus = [sku for sku in sku_presence if sku["is_own"]]
    # Partial reads: brand + category recognized but the exact SKU was not.
    brand_category_presence = _brand_category_presence(items)

    by_brand = Counter(sku["brand"] or "—" for sku in sku_presence)
    by_category = Counter(sku["category"] for sku in sku_presence if sku["category"])
    by_model = Counter(_model_label(sku) for sku in sku_presence)
    by_brand_objects = Counter(it["brand"] or "—" for it in items)
    by_category_objects = Counter(it["category"] for it in items if it["category"])
    by_model_objects = Counter(
        _model_label(it) for it in items if it["status"] == "matched_sku"
    )

    report = {
        "video": str(video),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "weights": str(weights),
        "totals": {
            # Main business metric: presence of unique SKU/model/article.
            "unique_skus": len(sku_presence),
            "unique_own_skus": len(own_skus),
            # Diagnostic object-level metrics: useful for capture quality/OCR,
            # but not the business answer.
            "candidate_objects": total,
            "physical_objects": total,
            "own_brand_objects": len(own),
            "competitor_or_unknown": total - len(own),
            "sku_evidence_objects": len(confident),
            "confident_sku": len(sku_presence),
            "needs_review_objects": len(review),
            "needs_review": len(review),
            # распознано до бренда+категории, но артикул не прочитан
            "brand_category_partial": len(brand_category_presence),
            # объект найден, но бренд/тип не читаются -> переснять этот участок
            "brand_not_visible_objects": sum(1 for it in items if it["status"] == "unknown"),
            "brand_not_visible": sum(1 for it in items if it["status"] == "unknown"),
        },
        "by_brand": dict(by_brand.most_common()),
        "by_category": dict(by_category.most_common()),
        "by_model": dict(by_model.most_common()),
        "by_brand_objects": dict(by_brand_objects.most_common()),
        "by_category_objects": dict(by_category_objects.most_common()),
        "by_model_objects": dict(by_model_objects.most_common()),
        "sku_presence": sku_presence,
        "brand_category_presence": brand_category_presence,
        "needs_review": review,
        "items": items,
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"audit_{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    _write_markdown(report, reports_dir / f"audit_{stem}.md")
    print(f"\nreport -> {reports_dir / f'audit_{stem}.md'}")
    print(f"unique_skus={len(sku_presence)} evidence_objects={len(confident)} "
          f"candidate_objects={total} review_objects={len(review)}")
    return report


def _write_markdown(r: dict, path: Path) -> None:
    t = r["totals"]
    L = [
        f"# SKU-аудит видео: {Path(r['video']).name}",
        "",
        f"_Сгенерировано: {r['generated_at']}_  ",
        f"_Веса детектора: `{r['weights']}`_",
        "",
        "## Итоги",
        "",
        f"- Уникальных SKU/артикулов найдено: **{t['unique_skus']}**",
        f"- Уникальных SKU наших брендов: **{t['unique_own_skus']}**",
        f"- Распознано до бренда+категории (артикул не прочитан): **{t.get('brand_category_partial', 0)}**",
        f"- Кропов-доказательств для SKU: **{t['sku_evidence_objects']}**",
        f"- Кандидатных объектов после трекинга (диагностика): **{t['candidate_objects']}**",
        f"- Бренд не виден у объектов (переснять): **{t.get('brand_not_visible_objects', 0)}**",
        f"- Объектов требует ручной проверки: **{t['needs_review_objects']}**",
        "",
        "## Найденные SKU/артикулы",
        "",
        "| Бренд | Модель | Артикул | Группа | Доказательств | Лучший crop |",
        "|---|---|---|---|---:|---|",
    ]
    L += [
        f"| {sku['brand'] or '—'} | {sku['model'] or '—'} | "
        f"{sku.get('article_codes') or sku.get('sku_id') or '—'} | "
        f"{sku['category'] or '—'} | {sku['evidence_objects']} | "
        f"`{sku['best_crop']}` |"
        for sku in r.get("sku_presence", [])
    ] or ["| — | — | — | — | 0 | — |"]
    L += [
        "",
        "## Распознано до бренда + категории (артикул не прочитан)",
        "",
        "| Бренд | Группа | Объектов | Лучший crop |",
        "|---|---|---:|---|",
    ]
    L += [
        f"| {bc['brand'] or '—'} | {bc['category'] or '—'} | "
        f"{bc['evidence_objects']} | `{bc['best_crop']}` |"
        for bc in r.get("brand_category_presence", [])
    ] or ["| — | — | 0 | — |"]
    L += [
        "",
        "## По брендам (уникальные SKU)",
        "",
        "| Бренд | SKU |",
        "|---|---|",
    ]
    L += [f"| {b} | {n} |" for b, n in r["by_brand"].items()]
    L += ["", "## По группам товара (уникальные SKU)", "", "| Группа | SKU |", "|---|---|"]
    L += [f"| {c} | {n} |" for c, n in r["by_category"].items()] or ["| — | 0 |"]
    L += ["", "## По моделям/артикулам", "", "| Модель / артикул | SKU |", "|---|---|"]
    L += [f"| {m} | {n} |" for m, n in r["by_model"].items()] or ["| — | 0 |"]
    L += ["", "## Спорные зоны (нужна проверка человеком)", ""]
    if r["needs_review"]:
        L += ["| object_id | статус | бренд | модель | OCR conf | crop |",
              "|---|---|---|---|---|---|"]
        L += [f"| {it['object_id']} | {it['status']} | {it['brand'] or '—'} | "
              f"{it['model'] or '—'} | {it['ocr_conf']} | `{it['crop']}` |"
              for it in r["needs_review"]]
    else:
        L += ["_Нет — все объекты определены уверенно._"]
    path.write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--weights", type=Path,
                    default=Path("ml/runs/product_det/weights/best.pt"))
    ap.add_argument("--work-dir", type=Path, default=Path("var/track"))
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--embed-weights", default="yolo11n.pt")
    ap.add_argument("--sim", type=float, default=0.96)
    ap.add_argument("--review-conf", type=float, default=0.75)
    ap.add_argument("--vid-stride", type=int, default=6)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--min-frames", type=int, default=4)
    ap.add_argument("--ocr-crops-per-object", type=int, default=2)
    ap.add_argument("--ocr-retry-objects", type=int, default=10,
                    help="run enhanced OCR + visual reference fallback only for the top N detections")
    ap.add_argument("--reuse-tracks", action="store_true",
                    help="skip detection+tracking if tracks.json already exists")
    args = ap.parse_args()
    audit(args.video, args.weights, work_dir=args.work_dir,
          reports_dir=args.reports_dir, embed_weights=args.embed_weights,
          sim=args.sim, review_conf=args.review_conf,
          vid_stride=args.vid_stride, device=args.device,
          reuse_tracks=args.reuse_tracks, conf=args.conf, min_frames=args.min_frames,
          ocr_crops_per_object=args.ocr_crops_per_object,
          ocr_retry_objects=args.ocr_retry_objects)


if __name__ == "__main__":
    main()

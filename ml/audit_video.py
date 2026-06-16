"""End-to-end SKU audit of a shelf video.

    video -> detect+track (count physical objects, no double count)
          -> re-id merge (collapse re-passes of the camera)
          -> OCR + catalog match each unique object (brand / model / category)
          -> aggregate report + uncertain bucket for human review

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


def audit(video: Path, weights: Path, *, work_dir: Path, reports_dir: Path,
          embed_weights="yolo11n.pt", sim=0.96, review_conf=0.75,
          vid_stride=6, device="mps", reuse_tracks=False,
          conf=0.35, min_frames=4) -> dict:
    stem = video.stem
    out = work_dir / stem

    # 1) detect + track -> unique-ish physical objects
    if reuse_tracks and (out / "tracks.json").exists():
        print(f"[audit] reusing existing tracks: {out/'tracks.json'}")
    else:
        track_video.run(video, weights, out, vid_stride=vid_stride, device=device,
                        conf=conf, min_frames=min_frames)
    # 2) re-id merge of camera re-passes
    data = reid_merge.merge(out, sim_threshold=sim, weights=embed_weights)
    objects = data["reid"].get("objects", [])

    # 3) recognize each unique object.
    # Camera orientation is constant within a video, so instead of OCR-ing every
    # crop at 4 rotations, vote the dominant rotation on the most confident
    # objects, then read the rest at that single rotation (~4x fewer OCR calls).
    rec = Recognizer()
    sample = sorted(objects, key=lambda o: -o["peak_conf"])[:15]
    cached: dict[int, object] = {}
    votes = Counter()
    for obj in sample:
        r = rec.recognize(out / obj["rep_crop"])  # full 4-rotation search
        cached[obj["object_id"]] = r
        # only a real catalog match reliably reveals orientation; OCR emits
        # noise "text" at every rotation, so text length is not trustworthy
        if r.status != "unknown":
            votes[r.rotation] += 1

    if votes and votes.most_common(1)[0][1] >= 2:
        dom = votes.most_common(1)[0][0]
        rest_rotations = (dom,)  # orientation is clear -> fast single rotation
    else:
        dom = None  # inconclusive -> safe full 4-rotation search for every crop
        rest_rotations = (0, 270, 90, 180)
    print(f"[audit] dominant rotation = {dom} (votes={dict(votes)}); "
          f"rest uses {rest_rotations}")

    items = []
    for obj in objects:
        crop_path = out / obj["rep_crop"]
        r = cached.get(obj["object_id"])
        if r is None:
            r = rec.recognize(crop_path, rotations=rest_rotations)
        items.append({
            "object_id": obj["object_id"],
            "track_ids": obj["track_ids"],
            "crop": str(crop_path.relative_to(work_dir.parent)) if work_dir.parent in crop_path.parents else str(crop_path),
            "status": r.status,
            "is_own": r.is_own,
            "brand": r.brand,
            "model": r.model,
            "category": r.category,
            "sku_id": r.sku_id,
            "ocr_conf": round(r.confidence, 3),
            "method": r.method,
            "text": r.text[:160],
        })

    # 4) aggregate
    total = len(items)
    own = [it for it in items if it["is_own"]]
    review = [it for it in items
              if it["status"] != "matched_sku" or it["ocr_conf"] < review_conf]
    confident = [it for it in items
                 if it["status"] == "matched_sku" and it["ocr_conf"] >= review_conf]

    by_brand = Counter(it["brand"] or "—" for it in items)
    by_category = Counter(it["category"] for it in items if it["category"])
    by_model = Counter(
        f'{it["brand"]} {it["model"]}' for it in items
        if it["status"] == "matched_sku"
    )

    report = {
        "video": str(video),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "weights": str(weights),
        "totals": {
            "physical_objects": total,
            "own_brand_objects": len(own),
            "competitor_or_unknown": total - len(own),
            "confident_sku": len(confident),
            "needs_review": len(review),
            # объект найден, но бренд/тип не читаются -> переснять этот участок
            "brand_not_visible": sum(1 for it in items if it["status"] == "unknown"),
        },
        "by_brand": dict(by_brand.most_common()),
        "by_category": dict(by_category.most_common()),
        "by_model": dict(by_model.most_common()),
        "needs_review": review,
        "items": items,
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"audit_{stem}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))
    _write_markdown(report, reports_dir / f"audit_{stem}.md")
    print(f"\nreport -> {reports_dir / f'audit_{stem}.md'}")
    print(f"objects={total} own={len(own)} confident_sku={len(confident)} review={len(review)}")
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
        f"- Физических объектов (после дедупликации): **{t['physical_objects']}**",
        f"- Наши бренды: **{t['own_brand_objects']}**",
        f"- Конкуренты / не распознано: **{t['competitor_or_unknown']}**",
        f"- Уверенно определен SKU: **{t['confident_sku']}**",
        f"- Бренд не виден (переснять): **{t.get('brand_not_visible', 0)}**",
        f"- Требует ручной проверки: **{t['needs_review']}**",
        "",
        "## По брендам",
        "",
        "| Бренд | Объектов |",
        "|---|---|",
    ]
    L += [f"| {b} | {n} |" for b, n in r["by_brand"].items()]
    L += ["", "## По группам товара", "", "| Группа | Объектов |", "|---|---|"]
    L += [f"| {c} | {n} |" for c, n in r["by_category"].items()] or ["| — | 0 |"]
    L += ["", "## По моделям (уверенно)", "", "| Модель | Объектов |", "|---|---|"]
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
    ap.add_argument("--reuse-tracks", action="store_true",
                    help="skip detection+tracking if tracks.json already exists")
    args = ap.parse_args()
    audit(args.video, args.weights, work_dir=args.work_dir,
          reports_dir=args.reports_dir, embed_weights=args.embed_weights,
          sim=args.sim, review_conf=args.review_conf,
          vid_stride=args.vid_stride, device=args.device,
          reuse_tracks=args.reuse_tracks, conf=args.conf, min_frames=args.min_frames)


if __name__ == "__main__":
    main()

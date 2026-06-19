"""Overnight self-improvement run: measure, sweep, train, report.

Set it running before bed; read ``reports/overnight/SUMMARY.md`` in the morning.

It needs a hand-labeled ground truth at ``data/eval/sku_presence.csv``
(``video,expected_sku_ids`` with ``|``-separated catalog ``sku_id``s). Without
it the sweep has nothing to score against, so the run refuses to start.

Stages (each isolated; a failure is logged and the rest still run):
  1. Detector/conf SWEEP — audit every labeled video under a few detector
     configs, score SKU precision/recall/F1 against the CSV, pick the best.
  2. Brand CLASSIFIER — build an ImageFolder dataset from the reference photos
     and train a visual brand recognizer (``yolo classify``); report val top-1.
  3. REPORT — write a before/after summary and an actionable best-config file.

    python3 -m ml.overnight_improve --truth data/eval/sku_presence.csv
    # quick plumbing check (reuses existing tracks, 1 epoch):
    python3 -m ml.overnight_improve --truth data/eval/sku_presence.csv --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

from ml import audit_video, evaluate_sku_coverage as evalcov

REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = Path("reports/overnight")
WORK_ROOT = Path("var/overnight")

# (name, weights, conf). Weights that are missing on disk are skipped with a note.
DEFAULT_CONFIGS = [
    ("v6_c40", "weights/product_det_v2.pt", 0.40),
    ("v8hardneg_c40", "weights/product_det_v8_hardneg.pt", 0.40),
    ("v8hardneg_c35", "weights/product_det_v8_hardneg.pt", 0.35),
]


def log(msg: str, logfile: Path | None = None) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logfile is not None:
        with logfile.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def resolve_videos(truth: dict[str, set[str]]) -> tuple[dict[str, Path], list[str]]:
    """Map each truth video name to a real file on disk (search the repo)."""
    by_name: dict[str, Path] = {}
    for path in REPO.rglob("*"):
        if path.suffix.lower() in (".mov", ".mp4") and "var/" not in str(path.relative_to(REPO)):
            by_name.setdefault(path.name, path)
    resolved, missing = {}, []
    for name in truth:
        p = by_name.get(name) or by_name.get(Path(name).name)
        if p is not None:
            resolved[name] = p
        else:
            missing.append(name)
    return resolved, missing


def run_sweep(truth, videos, *, device, configs, reuse_tracks, max_videos,
              logfile) -> list[dict]:
    items = list(videos.items())
    if max_videos:
        items = items[:max_videos]
    results = []
    for name, weights_rel, conf in configs:
        weights = REPO / weights_rel
        if not weights.exists():
            log(f"[sweep] skip {name}: weights missing ({weights_rel})", logfile)
            results.append({"config": name, "weights": weights_rel, "conf": conf,
                            "skipped": "weights missing"})
            continue
        reports_dir = OUT_ROOT / name
        work_dir = WORK_ROOT / name
        log(f"[sweep] {name}: weights={weights_rel} conf={conf} on {len(items)} videos", logfile)
        for vname, vpath in items:
            try:
                t0 = time.time()
                audit_video.audit(
                    vpath, weights, work_dir=work_dir, reports_dir=reports_dir,
                    device=device, conf=conf, reuse_tracks=reuse_tracks,
                )
                log(f"    {vname} done in {time.time() - t0:.0f}s", logfile)
            except Exception as exc:  # noqa: BLE001
                log(f"    {vname} FAILED: {exc}", logfile)
                traceback.print_exc()
        summary = score_config(truth, reports_dir)
        summary.update({"config": name, "weights": weights_rel, "conf": conf})
        results.append(summary)
        log(f"[sweep] {name}: F1={summary.get('f1', 0):.3f} "
            f"P={summary.get('precision', 0):.3f} R={summary.get('recall', 0):.3f}", logfile)
    return results


def score_config(truth, reports_dir: Path) -> dict:
    reports = evalcov.read_reports(reports_dir)
    per_video = []
    for video, expected in truth.items():
        report = reports.get(video)
        predicted = evalcov.predicted_skus(report) if report else set()
        per_video.append(evalcov.row_metrics(video, expected, predicted))
    m = evalcov.aggregate_rows(per_video)
    m["per_video"] = per_video
    return m


def train_brand_classifier(*, epochs, imgsz, device, batch, logfile,
                           dataset=Path("ml/datasets/ref_brand_cls"),
                           out_weights=Path("weights/ref_brand_cls.pt")) -> dict:
    from ml import build_reference_cls_dataset as builder

    log(f"[cls] building dataset -> {dataset}", logfile)
    summary = builder.build(dataset)  # symlinks; fast
    if not (dataset / "train").exists():
        return {"ok": False, "error": "dataset build produced no train split"}

    from ultralytics import YOLO

    log(f"[cls] training yolo11n-cls epochs={epochs} imgsz={imgsz} device={device}", logfile)
    model = YOLO("yolo11n-cls.pt")
    res = model.train(data=str(dataset), epochs=epochs, imgsz=imgsz, batch=batch,
                      device=device, workers=4, plots=False, verbose=False,
                      project=str(OUT_ROOT / "cls_run"), name="brand", exist_ok=True)
    top1 = float(getattr(res, "top1", 0.0) or 0.0)
    top5 = float(getattr(res, "top5", 0.0) or 0.0)
    best = Path(res.save_dir) / "weights" / "best.pt"
    saved = None
    if best.exists():
        out_weights.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(best, out_weights)
        saved = str(out_weights)
    return {"ok": True, "top1": top1, "top5": top5, "weights": saved,
            "classes": list(summary.keys()), "epochs": epochs}


def write_report(sweep, cls_result, *, baseline_name, started_at, missing_videos):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    scored = [s for s in sweep if "f1" in s]
    best = max(scored, key=lambda s: (s["f1"], s["recall"]), default=None)
    baseline = next((s for s in scored if s["config"] == baseline_name), None)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_min": round((time.time() - started_at) / 60, 1),
        "sweep": sweep,
        "best_config": best,
        "baseline_config": baseline_name,
        "classifier": cls_result,
        "missing_videos": missing_videos,
    }
    (OUT_ROOT / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if best:
        (OUT_ROOT / "best_config.json").write_text(json.dumps(
            {"config": best["config"], "weights": best["weights"], "conf": best["conf"],
             "f1": best["f1"], "precision": best["precision"], "recall": best["recall"]},
            ensure_ascii=False, indent=2), encoding="utf-8")

    L = [
        f"# Ночной прогон — итог ({payload['generated_at']})",
        "",
        f"_Длительность: {payload['elapsed_min']} мин_",
        "",
        "## Перебор детектора (метрика — уникальные SKU против ручной разметки)",
        "",
        "| Конфиг | Веса | conf | SKU ожид. | найдено | совпало | лишних | Recall | Precision | **F1** |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in sweep:
        if "f1" not in s:
            L.append(f"| {s['config']} | {s.get('weights','—')} | {s.get('conf','—')} | "
                     f"— | — | — | — | — | — | _{s.get('skipped','—')}_ |")
            continue
        mark = " 🏆" if best and s["config"] == best["config"] else ""
        L.append(f"| {s['config']}{mark} | {s['weights']} | {s['conf']} | "
                 f"{s['expected_skus']} | {s['predicted_skus']} | {s['matched_skus']} | "
                 f"{s['extra_skus']} | {s['recall']:.1%} | {s['precision']:.1%} | "
                 f"**{s['f1']:.1%}** |")

    if best and baseline:
        d = best["f1"] - baseline["f1"]
        L += ["", f"**Лучший конфиг:** `{best['config']}` "
              f"(F1 {best['f1']:.1%}). Базовый `{baseline_name}` = {baseline['f1']:.1%}. "
              f"Δ = {d:+.1%}.",
              "", "Применить лучший конфиг к проду: см. `reports/overnight/best_config.json`."]
    elif best:
        L += ["", f"**Лучший конфиг:** `{best['config']}` (F1 {best['f1']:.1%})."]

    L += ["", "## Визуальный классификатор бренда (этап 2 fallback)", ""]
    if cls_result.get("ok"):
        L += [f"- val top-1 точность по бренду: **{cls_result['top1']:.1%}** "
              f"(top-5 {cls_result.get('top5', 0):.1%})",
              f"- классы: {', '.join(cls_result.get('classes', []))}",
              f"- веса: `{cls_result.get('weights')}`",
              "",
              "Если top-1 высокая (>~85%), стоит подключить классификатор как "
              "визуальный fallback бренд+категория (следующий шаг, по согласованию)."]
    else:
        L += [f"- не обучен: {cls_result.get('error', 'см. лог')}"]

    if missing_videos:
        L += ["", "## Видео из разметки не найдены на диске", ""]
        L += [f"- `{v}`" for v in missing_videos]

    (OUT_ROOT / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"[report] -> {OUT_ROOT / 'SUMMARY.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", type=Path, default=Path("data/eval/sku_presence.csv"))
    ap.add_argument("--device", default=os.environ.get("SKU_AUDIT_DEVICE", "mps"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--baseline", default="v6_c40")
    ap.add_argument("--max-videos", type=int, default=0, help="cap videos (smoke)")
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--reuse-tracks", action="store_true",
                    help="reuse existing tracks.json (fast; for smoke)")
    ap.add_argument("--smoke", action="store_true",
                    help="fast plumbing check: 1 config, reuse tracks, 1 epoch, 1 video")
    args = ap.parse_args()

    if not args.truth.exists():
        raise SystemExit(
            f"ground truth not found: {args.truth}\n"
            "Fill it first (video,expected_sku_ids). Example: data/eval/sku_presence.example.csv")
    truth = evalcov.read_truth(args.truth)
    if not truth:
        raise SystemExit(f"no rows in {args.truth}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    logfile = OUT_ROOT / "run.log"
    started = time.time()
    log(f"=== overnight run start: {len(truth)} labeled videos, device={args.device} ===", logfile)

    videos, missing = resolve_videos(truth)
    log(f"resolved {len(videos)} videos, missing {len(missing)}: {missing}", logfile)

    configs = DEFAULT_CONFIGS
    reuse = args.reuse_tracks
    max_videos = args.max_videos
    epochs = args.epochs
    if args.smoke:
        configs = DEFAULT_CONFIGS[:1]
        reuse, max_videos, epochs = True, 1, 1
        log("SMOKE mode: 1 config, reuse tracks, 1 video, 1 epoch", logfile)

    sweep = []
    if not args.skip_sweep:
        try:
            sweep = run_sweep(truth, videos, device=args.device, configs=configs,
                              reuse_tracks=reuse, max_videos=max_videos, logfile=logfile)
        except Exception as exc:  # noqa: BLE001
            log(f"[sweep] STAGE FAILED: {exc}", logfile)
            traceback.print_exc()

    cls_result = {"ok": False, "error": "skipped"}
    if not args.skip_train:
        try:
            cls_result = train_brand_classifier(
                epochs=epochs, imgsz=args.imgsz, device=args.device,
                batch=args.batch, logfile=logfile)
        except Exception as exc:  # noqa: BLE001
            log(f"[cls] STAGE FAILED: {exc}", logfile)
            traceback.print_exc()
            cls_result = {"ok": False, "error": str(exc)}

    write_report(sweep, cls_result, baseline_name=args.baseline,
                 started_at=started, missing_videos=missing)
    log(f"=== done in {(time.time() - started) / 60:.1f} min ===", logfile)


if __name__ == "__main__":
    main()

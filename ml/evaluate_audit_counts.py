"""Diagnostic object-count evaluation for the full video-audit pipeline.

This is NOT the business acceptance metric anymore. The product goal is SKU
presence (unique models/articles), so use ``ml.evaluate_sku_coverage`` for final
validation. Keep this script only for debugging detector/tracker over/under-count.

Truth CSV format:

    video,true_objects
    IMG_8886.MOV,84
    IMG_8916.MOV,120

The script reads ``reports/audit_*.json`` and compares each report's
``totals.physical_objects`` to the manually counted number. This is the metric
that includes tracking/re-ID duplicates, unlike detector-only validation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--out", type=Path, default=Path("reports/audit_count_eval.json"))
    args = ap.parse_args()

    truth = read_truth(args.truth)
    reports = read_reports(args.reports_dir)
    rows = []
    total_true = 0
    total_pred = 0
    missing = []
    for video, true_count in truth.items():
        report = reports.get(video)
        if report is None:
            missing.append(video)
            continue
        pred = int((report.get("totals") or {}).get("physical_objects", 0))
        total_true += true_count
        total_pred += pred
        rows.append({
            "video": video,
            "true_objects": true_count,
            "predicted_objects": pred,
            "delta": pred - true_count,
            "count_error": abs(pred - true_count) / true_count if true_count else float(pred > 0),
        })

    summary = {
        "videos": len(rows),
        "missing_reports": missing,
        "true_objects": total_true,
        "predicted_objects": total_pred,
        "delta": total_pred - total_true,
        "count_error": abs(total_pred - total_true) / total_true if total_true else float(total_pred > 0),
        "mean_video_count_error": (
            sum(r["count_error"] for r in rows) / len(rows) if rows else 0.0
        ),
    }
    payload = {"summary": summary, "videos": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, args.out.with_suffix(".md"))
    print(f"videos={summary['videos']} true={summary['true_objects']} "
          f"pred={summary['predicted_objects']} count_error={summary['count_error']:.3f}")
    if missing:
        print("missing reports:", ", ".join(missing))
    print(f"report -> {args.out}")


def read_truth(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = csv.DictReader(file)
        result = {}
        for row in rows:
            video = (row.get("video") or "").strip()
            if not video:
                continue
            result[Path(video).name] = int(row["true_objects"])
        return result


def read_reports(reports_dir: Path) -> dict[str, dict]:
    reports = {}
    for path in sorted(reports_dir.glob("audit_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        video = Path(payload.get("video", path.stem.replace("audit_", ""))).name
        reports[video] = payload
    return reports


def write_markdown(payload: dict, path: Path) -> None:
    s = payload["summary"]
    lines = [
        "# Video Audit Count Evaluation",
        "",
        f"- Videos: **{s['videos']}**",
        f"- True objects: **{s['true_objects']}**",
        f"- Predicted objects: **{s['predicted_objects']}**",
        f"- Delta: **{s['delta']}**",
        f"- Count error: **{s['count_error']:.1%}**",
        f"- Mean video count error: **{s['mean_video_count_error']:.1%}**",
        "",
        "| Video | True | Predicted | Delta | Count error |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["videos"]:
        lines.append(f"| {row['video']} | {row['true_objects']} | "
                     f"{row['predicted_objects']} | {row['delta']} | "
                     f"{row['count_error']:.1%} |")
    if s["missing_reports"]:
        lines += ["", "## Missing Reports", ""]
        lines += [f"- `{video}`" for video in s["missing_reports"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

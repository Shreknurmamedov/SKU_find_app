"""Evaluate full video-audit SKU presence against a hand-labeled CSV.

Truth CSV format:

    video,expected_sku_ids
    IMG_8886.MOV,HUTER_64_1_20_DY5000LX_DY6500LX|RESANTA_64_1_82_AVR_40

Use catalog ``sku_id`` values when possible. The script compares them with
``sku_presence`` in ``reports/audit_*.json`` and reports SKU-level precision,
recall and F1. Object counts are intentionally ignored here: five copies of one
model still count as one expected SKU.
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
    ap.add_argument("--out", type=Path, default=Path("reports/sku_coverage_eval.json"))
    args = ap.parse_args()

    truth = read_truth(args.truth)
    reports = read_reports(args.reports_dir)
    rows = []
    missing_reports = []

    for video, expected in truth.items():
        report = reports.get(video)
        if report is None:
            missing_reports.append(video)
            predicted = set()
        else:
            predicted = predicted_skus(report)
        rows.append(row_metrics(video, expected, predicted))

    summary = aggregate_rows(rows)
    summary["videos"] = len(rows)
    summary["missing_reports"] = missing_reports
    payload = {"summary": summary, "videos": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, args.out.with_suffix(".md"))
    print(f"videos={summary['videos']} expected_skus={summary['expected_skus']} "
          f"predicted_skus={summary['predicted_skus']} recall={summary['recall']:.3f} "
          f"precision={summary['precision']:.3f} f1={summary['f1']:.3f}")
    if missing_reports:
        print("missing reports:", ", ".join(missing_reports))
    print(f"report -> {args.out}")


def read_truth(path: Path) -> dict[str, set[str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = csv.DictReader(file)
        result = {}
        for row in rows:
            video = (row.get("video") or "").strip()
            if not video:
                continue
            raw = row.get("expected_sku_ids") or row.get("expected_skus") or ""
            result[Path(video).name] = split_skus(raw)
        return result


def split_skus(raw: str) -> set[str]:
    return {part.strip() for part in raw.replace(";", "|").split("|") if part.strip()}


def read_reports(reports_dir: Path) -> dict[str, dict]:
    reports = {}
    for path in sorted(reports_dir.glob("audit_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        video = Path(payload.get("video", path.stem.replace("audit_", ""))).name
        reports[video] = payload
    return reports


def predicted_skus(report: dict) -> set[str]:
    result = set()
    for sku in report.get("sku_presence", []):
        sku_id = (sku.get("sku_id") or "").strip()
        if sku_id:
            result.add(sku_id)
    # Backward compatibility with older reports that only had item-level matches.
    if not result:
        for item in report.get("items", []):
            if item.get("status") == "matched_sku" and item.get("sku_id"):
                result.add(item["sku_id"])
    return result


def row_metrics(video: str, expected: set[str], predicted: set[str]) -> dict:
    m = metrics(expected, predicted)
    m["video"] = video
    m["missing_sku_ids"] = sorted(expected - predicted)
    m["extra_sku_ids"] = sorted(predicted - expected)
    return m


def aggregate_rows(rows: list[dict]) -> dict:
    """Micro-average SKU-presence metrics over videos.

    Do not union SKU ids across videos: finding SKU_A in the wrong video must
    remain one false negative plus one false positive, not a true positive.
    """
    expected = sum(int(row["expected_skus"]) for row in rows)
    predicted = sum(int(row["predicted_skus"]) for row in rows)
    matched = sum(int(row["matched_skus"]) for row in rows)
    extra = sum(int(row["extra_skus"]) for row in rows)
    missing = sum(int(row["missing_skus"]) for row in rows)
    if matched + extra:
        precision = matched / (matched + extra)
    else:
        precision = 1.0 if expected == 0 else 0.0
    recall = matched / (matched + missing) if matched + missing else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "expected_skus": expected,
        "predicted_skus": predicted,
        "matched_skus": matched,
        "extra_skus": extra,
        "missing_skus": missing,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def metrics(expected: set[str], predicted: set[str]) -> dict:
    tp = len(expected & predicted)
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    precision = tp / (tp + fp) if tp + fp else 1.0 if not expected else 0.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "expected_skus": len(expected),
        "predicted_skus": len(predicted),
        "matched_skus": tp,
        "extra_skus": fp,
        "missing_skus": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def write_markdown(payload: dict, path: Path) -> None:
    s = payload["summary"]
    lines = [
        "# SKU Coverage Evaluation",
        "",
        f"- Videos: **{s['videos']}**",
        f"- Expected SKU: **{s['expected_skus']}**",
        f"- Predicted SKU: **{s['predicted_skus']}**",
        f"- Matched SKU: **{s['matched_skus']}**",
        f"- Missing SKU: **{s['missing_skus']}**",
        f"- Extra SKU: **{s['extra_skus']}**",
        f"- Recall: **{s['recall']:.1%}**",
        f"- Precision: **{s['precision']:.1%}**",
        f"- F1: **{s['f1']:.1%}**",
        "",
        "| Video | Expected | Predicted | Matched | Missing | Extra | Recall | Precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["videos"]:
        lines.append(f"| {row['video']} | {row['expected_skus']} | "
                     f"{row['predicted_skus']} | {row['matched_skus']} | "
                     f"{row['missing_skus']} | {row['extra_skus']} | "
                     f"{row['recall']:.1%} | {row['precision']:.1%} |")
    for row in payload["videos"]:
        if row["missing_sku_ids"] or row["extra_sku_ids"]:
            lines += ["", f"## {row['video']}", ""]
            if row["missing_sku_ids"]:
                lines += ["Missing:"] + [f"- `{sku}`" for sku in row["missing_sku_ids"]]
            if row["extra_sku_ids"]:
                lines += ["Extra:"] + [f"- `{sku}`" for sku in row["extra_sku_ids"]]
    if s["missing_reports"]:
        lines += ["", "## Missing Reports", ""]
        lines += [f"- `{video}`" for video in s["missing_reports"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

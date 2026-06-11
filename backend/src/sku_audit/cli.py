from __future__ import annotations

import argparse
from pathlib import Path

from sku_audit.docx_catalog import import_docx_catalog, write_catalog_csv, write_catalog_raw_json
from sku_audit.jobs import JobStore
from sku_audit.pipeline import run_image_folder_audit
from sku_audit.reporting import write_json_report, write_markdown_report


def main() -> None:
    parser = argparse.ArgumentParser(prog="sku-audit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit-images", help="Build a baseline audit report")
    audit_parser.add_argument("--input", required=True, type=Path, help="Input folder with images")
    audit_parser.add_argument("--output", required=True, type=Path, help="Output JSON report path")
    audit_parser.add_argument("--markdown", type=Path, help="Optional Markdown report path")

    catalog_parser = subparsers.add_parser(
        "import-docx-catalog", help="Extract own-product catalog rows from a DOCX file"
    )
    catalog_parser.add_argument("--input", required=True, type=Path, help="Input DOCX catalog")
    catalog_parser.add_argument("--output", required=True, type=Path, help="Output catalog CSV")
    catalog_parser.add_argument("--raw-json", type=Path, help="Optional raw JSON with characteristics")
    catalog_parser.add_argument(
        "--competitor",
        action="store_true",
        help="Mark imported rows as competitor products instead of own products",
    )

    job_parser = subparsers.add_parser("create-job", help="Create a processing job from local media")
    job_parser.add_argument("--input", required=True, type=Path, help="Input folder with media")
    job_parser.add_argument("--store-name", help="Optional trading point name override")
    job_parser.add_argument("--var-dir", type=Path, default=Path("var"), help="Runtime storage dir")

    args = parser.parse_args()

    if args.command == "audit-images":
        report = run_image_folder_audit(args.input)
        write_json_report(report, args.output)
        if args.markdown:
            write_markdown_report(report, args.markdown)
        print(f"Wrote {args.output}")
        if args.markdown:
            print(f"Wrote {args.markdown}")
    elif args.command == "import-docx-catalog":
        products = import_docx_catalog(args.input, own_brand=not args.competitor)
        write_catalog_csv(products, args.output)
        if args.raw_json:
            write_catalog_raw_json(products, args.raw_json)
        print(f"Imported {len(products)} products")
        print(f"Wrote {args.output}")
        if args.raw_json:
            print(f"Wrote {args.raw_json}")
    elif args.command == "create-job":
        store = JobStore(args.var_dir)
        job = store.create_from_local_folder(args.input, store_name=args.store_name)
        print(f"Created {job.job_id}")
        print(f"Files: {job.summary['total_files']}")
        print(f"Retake: {job.summary['quality_retake']}")
        print(store.jobs_dir / job.job_id / "report.md")


if __name__ == "__main__":
    main()

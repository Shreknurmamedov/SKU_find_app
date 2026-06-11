from __future__ import annotations

import argparse
from pathlib import Path

from sku_audit.docx_catalog import import_docx_catalog, write_catalog_csv, write_catalog_raw_json
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


if __name__ == "__main__":
    main()

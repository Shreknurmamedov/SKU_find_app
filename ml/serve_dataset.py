"""Serve the prepared ML dataset with browser-friendly CORS headers."""

from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class DatasetRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("ml/datasets/sku_live"),
        help="Dataset root directory to serve.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface to bind.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8099,
        help="Port to listen on.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"Dataset root does not exist: {root}")

    handler = partial(DatasetRequestHandler, directory=str(root))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {root} at http://localhost:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

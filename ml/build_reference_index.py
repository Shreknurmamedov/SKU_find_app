"""Build a visual retrieval index from reference product photos.

Example:
    python3 -m ml.build_reference_index \
      --manifest data/catalog/reference_dataset_all/training_images.csv \
      --out data/catalog/reference_index_yolo11n.npz \
      --metadata data/catalog/reference_index_yolo11n.jsonl \
      --device mps
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/catalog/reference_dataset_all/training_images.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/catalog/reference_index_yolo11n.npz"))
    ap.add_argument("--metadata", type=Path, default=Path("data/catalog/reference_index_yolo11n.jsonl"))
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = read_rows(args.manifest, args.limit)
    if not rows:
        raise SystemExit(f"no usable rows in {args.manifest}")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    vectors = []
    kept_rows = []
    total = len(rows)
    for start in range(0, total, args.batch):
        batch_rows = rows[start:start + args.batch]
        paths = [row["image_path"] for row in batch_rows]
        try:
            embeddings = model.embed(paths, imgsz=args.imgsz, device=args.device, verbose=False)
        except Exception as exc:  # noqa: BLE001 - keep building around bad files
            print(f"[warn] batch {start}-{start + len(paths)} failed: {exc}")
            continue
        for row, emb in zip(batch_rows, embeddings):
            vec = emb.detach().cpu().numpy().astype(np.float32).ravel()
            norm = np.linalg.norm(vec)
            if norm <= 0:
                continue
            vectors.append(vec / norm)
            kept_rows.append(row)
        if (start // args.batch) % 10 == 0 or start + args.batch >= total:
            print(f"embedded {min(start + args.batch, total)}/{total} images", flush=True)

    if not vectors:
        raise SystemExit("no embeddings produced")
    matrix = np.vstack(vectors).astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, embeddings=matrix)
    args.metadata.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in kept_rows) + "\n",
        encoding="utf-8",
    )
    print(f"index images={len(kept_rows)} dim={matrix.shape[1]} -> {args.out}")
    print(f"metadata -> {args.metadata}")


def read_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_path = Path(row.get("image_path") or row.get("local_path") or "")
            if not image_path.exists():
                continue
            row = dict(row)
            row["image_path"] = str(image_path)
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


if __name__ == "__main__":
    main()

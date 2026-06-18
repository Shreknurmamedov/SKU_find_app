"""Visual retrieval over product reference images.

The index is built from ``data/catalog/reference_dataset_all/training_images.csv``
by ``ml.build_reference_index``. At recognition time this module embeds a crop
with the same YOLO backbone and returns the closest product reference images.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INDEX = Path("data/catalog/reference_index_yolo11n.npz")
DEFAULT_METADATA = Path("data/catalog/reference_index_yolo11n.jsonl")


@dataclass(frozen=True)
class VisualMatch:
    product_key: str
    score: float
    margin: float
    metadata: dict[str, Any]
    reference_image: str


class VisualReferenceIndex:
    def __init__(
        self,
        index_path: Path = DEFAULT_INDEX,
        metadata_path: Path = DEFAULT_METADATA,
        *,
        weights: str = "yolo11n.pt",
        imgsz: int = 224,
    ) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.weights = weights
        self.imgsz = imgsz
        self._embeddings: np.ndarray | None = None
        self._metadata: list[dict[str, Any]] | None = None
        self._model = None

    def available(self) -> bool:
        return self.index_path.exists() and self.metadata_path.exists()

    def search(self, image, *, topk_products: int = 5) -> list[VisualMatch]:
        if not self.available():
            return []
        embeddings, metadata = self._load()
        query = self._embed(image)
        if query.size == 0:
            return []
        sims = embeddings @ query
        top_idx = np.argsort(-sims)[: max(topk_products * 8, topk_products)]
        best_by_product: dict[str, tuple[float, int]] = {}
        for idx in top_idx:
            meta = metadata[int(idx)]
            key = product_key(meta)
            score = float(sims[int(idx)])
            prev = best_by_product.get(key)
            if prev is None or score > prev[0]:
                best_by_product[key] = (score, int(idx))
        ranked = sorted(best_by_product.items(), key=lambda kv: -kv[1][0])
        matches: list[VisualMatch] = []
        for rank, (key, (score, idx)) in enumerate(ranked[:topk_products]):
            next_score = ranked[rank + 1][1][0] if rank + 1 < len(ranked) else 0.0
            meta = metadata[idx]
            matches.append(
                VisualMatch(
                    product_key=key,
                    score=score,
                    margin=score - float(next_score),
                    metadata=meta,
                    reference_image=meta.get("image_path") or meta.get("local_path") or "",
                )
            )
        return matches

    def _load(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if self._embeddings is None or self._metadata is None:
            data = np.load(self.index_path)
            embeddings = data["embeddings"].astype(np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            self._embeddings = embeddings / np.maximum(norms, 1e-8)
            self._metadata = [
                json.loads(line)
                for line in self.metadata_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return self._embeddings, self._metadata

    def _embed(self, image) -> np.ndarray:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.weights)
        emb = self._model.embed(image, imgsz=self.imgsz, verbose=False)
        if not emb:
            return np.zeros((0,), dtype=np.float32)
        vec = emb[0].detach().cpu().numpy().astype(np.float32).ravel()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


def product_key(meta: dict[str, Any]) -> str:
    return (
        meta.get("catalog_sku_id")
        or meta.get("product_id")
        or f"{meta.get('brand_id', '')}:{meta.get('sku', '')}:{meta.get('model_name', '')}"
    )

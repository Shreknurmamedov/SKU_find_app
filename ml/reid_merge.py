"""Merge tracks that are the same physical object seen on a second camera pass.

ByteTrack gives a fresh id when the camera leaves an object and returns, which
would double-count it. We merge two tracks only when BOTH hold:

  * their frame spans do NOT overlap (one ended before the other started) -- a
    re-pass, not two items sitting side by side at the same time;
  * their crop embeddings are cosine-similar above a threshold.

This deliberately keeps two identical SKUs that are visible simultaneously as
two separate objects (correct), while collapsing re-appearances of one object.

Embeddings come from a YOLO backbone (offline, no extra downloads).

    python3 ml/reid_merge.py --track-dir var/track/IMG_8886 --sim 0.86
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def embed_crops(crop_paths: list[Path], weights="yolo11n.pt", imgsz=224) -> np.ndarray:
    from ultralytics import YOLO
    model = YOLO(weights)
    vecs = []
    for p in crop_paths:
        emb = model.embed(str(p), imgsz=imgsz, verbose=False)
        v = emb[0].detach().cpu().numpy().astype(np.float32).ravel()
        n = np.linalg.norm(v)
        vecs.append(v / n if n > 0 else v)
    return np.vstack(vecs) if vecs else np.zeros((0, 1), np.float32)


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def repass_gap(a: dict, b: dict) -> int | None:
    """Frames between the end of the earlier track and start of the later one.

    Returns None if the two tracks overlap in time (co-visible -> two distinct
    physical objects, never a re-pass). Otherwise the (non-negative) gap.
    """
    if a["last_frame"] < b["first_frame"]:
        return b["first_frame"] - a["last_frame"]
    if b["last_frame"] < a["first_frame"]:
        return a["first_frame"] - b["last_frame"]
    return None  # overlapping spans


def merge(track_dir: Path, sim_threshold=0.96, min_gap_frames=12,
          weights="yolo11n.pt") -> dict:
    data = json.loads((track_dir / "tracks.json").read_text())
    tracks = data["tracks"]
    n = len(tracks)
    if n == 0:
        data["reid"] = {"unique_objects": 0, "merges": []}
        (track_dir / "tracks_reid.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
        return data

    embs = embed_crops([track_dir / t["crop"] for t in tracks], weights=weights)
    sims = embs @ embs.T

    # A merge requires a genuine re-pass: the camera left the object (a real
    # time gap) and came back to a near-identical crop. The high threshold +
    # gap keep single-pan videos at ~0 merges (every track = a distinct item)
    # and stop union-find from chaining loosely-similar items into blobs.
    uf = UnionFind(n)
    merges = []
    for i in range(n):
        for j in range(i + 1, n):
            gap = repass_gap(tracks[i], tracks[j])
            if gap is None or gap < min_gap_frames:
                continue
            s = float(sims[i, j])
            if s >= sim_threshold:
                uf.union(i, j)
                merges.append({"a": tracks[i]["track_id"], "b": tracks[j]["track_id"],
                               "sim": round(s, 3), "gap": gap})

    groups: dict[int, list[int]] = {}
    for idx in range(n):
        groups.setdefault(uf.find(idx), []).append(idx)

    # representative crop per group = highest peak_conf
    objects = []
    for root, members in sorted(groups.items()):
        members.sort(key=lambda k: -tracks[k]["peak_conf"])
        rep = tracks[members[0]]
        crop_candidates = []
        for member in members:
            track = tracks[member]
            crop_candidates.extend(
                track.get("crop_candidates")
                or [{"crop": track["crop"], "score": track.get("peak_conf", 0.0)}]
            )
        crop_candidates.sort(key=lambda c: -float(c.get("score", 0.0)))
        objects.append({
            "object_id": len(objects) + 1,
            "rep_track_id": rep["track_id"],
            "rep_crop": crop_candidates[0]["crop"] if crop_candidates else rep["crop"],
            "crop_candidates": crop_candidates[:8],
            "track_ids": [tracks[k]["track_id"] for k in members],
            "peak_conf": rep["peak_conf"],
        })

    data["reid"] = {
        "unique_objects": len(objects),
        "sim_threshold": sim_threshold,
        "merges": merges,
        "objects": objects,
    }
    out = track_dir / "tracks_reid.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"tracks={n} -> unique objects={len(objects)} (merged {len(merges)} pairs) -> {out}")
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-dir", required=True, type=Path)
    ap.add_argument("--sim", type=float, default=0.86)
    ap.add_argument("--weights", default="yolo11n.pt")
    args = ap.parse_args()
    merge(args.track_dir, sim_threshold=args.sim, weights=args.weights)


if __name__ == "__main__":
    main()

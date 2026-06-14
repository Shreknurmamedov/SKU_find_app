"""Stage 1 over video: detect + track every product, count physical objects.

The double-count problem (the same item seen across many frames) is solved by
a tracker: ByteTrack assigns a stable id to each object, so one track == one
physical product no matter how many frames it spans. We keep the single best
crop per track to hand to stage-2 recognition, and drop flickering tracks that
appear in too few frames or never reach a confidence floor.

Re-appearances (camera pans away and back -> new track id) are merged later by
ml/reid_merge.py using visual embeddings.

    python3 ml/track_video.py --video "ТТ Пэкстрой/IMG_8886.MOV" \
        --weights ml/runs/product_det/weights/best.pt --out var/track/IMG_8886
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Track:
    track_id: int
    n_frames: int = 0
    peak_conf: float = 0.0
    best_score: float = 0.0
    best_frame: int = -1
    best_box: tuple[int, int, int, int] = (0, 0, 0, 0)
    best_crop: np.ndarray | None = field(default=None, repr=False)
    first_frame: int = -1
    last_frame: int = -1


def crop_quality(conf: float, box, W: int, H: int) -> float:
    """Prefer confident, large, non-edge-clipped crops for OCR."""
    x1, y1, x2, y2 = box
    area = max(0, x2 - x1) * max(0, y2 - y1)
    edge_pen = 1.0
    margin = 4
    if x1 <= margin or y1 <= margin or x2 >= W - margin or y2 >= H - margin:
        edge_pen = 0.7
    return conf * (area ** 0.5) * edge_pen


def run(video: Path, weights: Path, out: Path, *, imgsz=1280, conf=0.25,
        vid_stride=6, tracker="bytetrack.yaml", min_frames=3, min_peak_conf=0.35,
        pad=0.06, device="mps") -> dict:
    from ultralytics import YOLO

    out.mkdir(parents=True, exist_ok=True)
    crops_dir = out / "crops"
    crops_dir.mkdir(exist_ok=True)

    model = YOLO(str(weights))
    tracks: dict[int, Track] = {}

    results = model.track(
        source=str(video), stream=True, persist=True, tracker=tracker,
        imgsz=imgsz, conf=conf, vid_stride=vid_stride, device=device,
        verbose=False,
    )

    for fi, res in enumerate(results):
        if res.boxes is None or res.boxes.id is None:
            continue
        frame = res.orig_img  # BGR HxWx3
        H, W = frame.shape[:2]
        ids = res.boxes.id.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        xyxy = res.boxes.xyxy.cpu().numpy()
        for tid, cf, box in zip(ids, confs, xyxy):
            t = tracks.get(int(tid))
            if t is None:
                t = Track(track_id=int(tid), first_frame=fi)
                tracks[int(tid)] = t
            t.n_frames += 1
            t.last_frame = fi
            t.peak_conf = max(t.peak_conf, float(cf))
            score = crop_quality(float(cf), box, W, H)
            if score > t.best_score:
                px = (box[2] - box[0]) * pad
                py = (box[3] - box[1]) * pad
                x1 = max(0, int(box[0] - px)); y1 = max(0, int(box[1] - py))
                x2 = min(W, int(box[2] + px)); y2 = min(H, int(box[3] + py))
                t.best_score = score
                t.best_frame = fi
                t.best_box = (x1, y1, x2, y2)
                t.best_crop = frame[y1:y2, x1:x2].copy()

    # filter flickering / low-confidence tracks
    kept = [t for t in tracks.values()
            if t.n_frames >= min_frames and t.peak_conf >= min_peak_conf
            and t.best_crop is not None and t.best_crop.size > 0]

    import cv2
    manifest = []
    for t in sorted(kept, key=lambda x: x.track_id):
        name = f"track_{t.track_id:04d}.jpg"
        cv2.imwrite(str(crops_dir / name), t.best_crop)
        manifest.append({
            "track_id": t.track_id,
            "crop": f"crops/{name}",
            "n_frames": t.n_frames,
            "peak_conf": round(t.peak_conf, 3),
            "best_frame": t.best_frame,
            "first_frame": t.first_frame,
            "last_frame": t.last_frame,
            "box": t.best_box,
        })

    summary = {
        "video": str(video),
        "weights": str(weights),
        "raw_tracks": len(tracks),
        "kept_tracks": len(kept),
        "params": {"imgsz": imgsz, "conf": conf, "vid_stride": vid_stride,
                   "min_frames": min_frames, "min_peak_conf": min_peak_conf,
                   "tracker": tracker},
        "tracks": manifest,
    }
    (out / "tracks.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"raw tracks={len(tracks)} kept(unique objects)={len(kept)} -> {out/'tracks.json'}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--weights", type=Path, default=Path("ml/runs/product_det/weights/best.pt"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--vid-stride", type=int, default=6)
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--min-peak-conf", type=float, default=0.35)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    run(args.video, args.weights, args.out, imgsz=args.imgsz, conf=args.conf,
        vid_stride=args.vid_stride, min_frames=args.min_frames,
        min_peak_conf=args.min_peak_conf, device=args.device)


if __name__ == "__main__":
    main()

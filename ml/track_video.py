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
class CropCandidate:
    score: float
    frame: int
    box: tuple[int, int, int, int]
    crop: np.ndarray = field(repr=False)


@dataclass
class Track:
    track_id: int
    n_frames: int = 0
    peak_conf: float = 0.0
    best_score: float = 0.0
    best_frame: int = -1
    best_box: tuple[int, int, int, int] = (0, 0, 0, 0)
    best_crop: np.ndarray | None = field(default=None, repr=False)
    crop_candidates: list[CropCandidate] = field(default_factory=list, repr=False)
    first_frame: int = -1
    last_frame: int = -1

    def add_crop_candidate(self, candidate: CropCandidate, limit: int) -> None:
        self.crop_candidates.append(candidate)
        self.crop_candidates.sort(key=lambda c: -c.score)
        del self.crop_candidates[limit:]
        best = self.crop_candidates[0]
        self.best_score = best.score
        self.best_frame = best.frame
        self.best_box = best.box
        self.best_crop = best.crop


def crop_quality(conf: float, box, W: int, H: int) -> float:
    """Prefer confident, large, non-edge-clipped crops for OCR."""
    x1, y1, x2, y2 = box
    area = max(0, x2 - x1) * max(0, y2 - y1)
    edge_pen = 1.0
    margin = 4
    if x1 <= margin or y1 <= margin or x2 >= W - margin or y2 >= H - margin:
        edge_pen = 0.7
    return conf * (area ** 0.5) * edge_pen


def is_plain_background(crop: np.ndarray, *, sat_thr: float = 45.0,
                        lap_thr: float = 55.0,
                        gray_std_thr: float = 30.0) -> bool:
    """Reject large plain floor/wall/cabinet crops masquerading as products.

    Real packages can be white or grey, so saturation alone is unsafe. We only
    reject low-colour crops that also have little edge texture. Text, handles,
    printed labels and product contours usually push the Laplacian variance up.
    """
    if crop is None or crop.size == 0:
        return True
    import cv2

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sat_mean = float(hsv[:, :, 1].mean())
    gray_std = float(gray.std())
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    if sat_mean >= sat_thr:
        return False
    return lap_var < lap_thr or gray_std < gray_std_thr


def is_low_detail_background(crop: np.ndarray, *, sat_thr: float = 35.0,
                             gray_std_thr: float = 18.0,
                             edge_frac_thr: float = 0.03) -> bool:
    """Reject small light floor/wall crops that contain only a seam or shadow."""
    if crop is None or crop.size == 0:
        return True
    import cv2

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    sat_mean = float(hsv[:, :, 1].mean())
    gray_std = float(gray.std())
    edge_frac = float((edges > 0).mean())
    return sat_mean < sat_thr and gray_std < gray_std_thr and edge_frac < edge_frac_thr


def run(video: Path, weights: Path, out: Path, *, imgsz=1280, conf=0.25,
        vid_stride=6, tracker="bytetrack.yaml", min_frames=3, min_peak_conf=0.35,
        pad=0.06, device="mps", min_area_frac=0.0005, max_area_frac=0.55,
        max_aspect=8.0, reject_plain_bg=True, plain_min_area=0.015,
        plain_sat=45.0, plain_lap=55.0, plain_gray_std=30.0,
        plain_small_min_area=0.008, low_detail_sat=35.0,
        low_detail_gray_std=18.0, low_detail_edge_frac=0.03,
        crops_per_track=5) -> dict:
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
            bw = max(0.0, float(box[2] - box[0]))
            bh = max(0.0, float(box[3] - box[1]))
            area_frac = (bw * bh) / max(1, W * H)
            aspect = bw / max(1.0, bh)
            if area_frac < min_area_frac or area_frac > max_area_frac:
                continue
            if aspect > max_aspect or aspect < 1.0 / max_aspect:
                continue
            if reject_plain_bg and area_frac >= plain_small_min_area:
                x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
                x2 = min(W, int(box[2])); y2 = min(H, int(box[3]))
                bg_crop = frame[y1:y2, x1:x2]
                if area_frac >= plain_min_area and is_plain_background(
                    bg_crop, sat_thr=plain_sat, lap_thr=plain_lap,
                    gray_std_thr=plain_gray_std,
                ):
                    continue
                if is_low_detail_background(
                    bg_crop, sat_thr=low_detail_sat,
                    gray_std_thr=low_detail_gray_std,
                    edge_frac_thr=low_detail_edge_frac,
                ):
                    continue
            t = tracks.get(int(tid))
            if t is None:
                t = Track(track_id=int(tid), first_frame=fi)
                tracks[int(tid)] = t
            t.n_frames += 1
            t.last_frame = fi
            t.peak_conf = max(t.peak_conf, float(cf))
            score = crop_quality(float(cf), box, W, H)
            should_keep_crop = (
                len(t.crop_candidates) < crops_per_track
                or score > t.crop_candidates[-1].score
            )
            if should_keep_crop:
                px = (box[2] - box[0]) * pad
                py = (box[3] - box[1]) * pad
                x1 = max(0, int(box[0] - px)); y1 = max(0, int(box[1] - py))
                x2 = min(W, int(box[2] + px)); y2 = min(H, int(box[3] + py))
                t.add_crop_candidate(
                    CropCandidate(
                        score=score,
                        frame=fi,
                        box=(x1, y1, x2, y2),
                        crop=frame[y1:y2, x1:x2].copy(),
                    ),
                    max(1, crops_per_track),
                )

    # filter flickering / low-confidence tracks
    kept = [t for t in tracks.values()
            if t.n_frames >= min_frames and t.peak_conf >= min_peak_conf
            and t.best_crop is not None and t.best_crop.size > 0]

    import cv2
    manifest = []
    for t in sorted(kept, key=lambda x: x.track_id):
        candidate_rows = []
        candidates = t.crop_candidates or [
            CropCandidate(score=t.best_score, frame=t.best_frame, box=t.best_box, crop=t.best_crop)
        ]
        for idx, candidate in enumerate(candidates):
            name = f"track_{t.track_id:04d}.jpg" if idx == 0 else f"track_{t.track_id:04d}_alt{idx}.jpg"
            cv2.imwrite(str(crops_dir / name), candidate.crop)
            candidate_rows.append({
                "crop": f"crops/{name}",
                "score": round(float(candidate.score), 3),
                "frame": int(candidate.frame),
                "box": [int(v) for v in candidate.box],
            })
        name = Path(candidate_rows[0]["crop"]).name
        manifest.append({
            "track_id": t.track_id,
            "crop": f"crops/{name}",
            "crop_candidates": candidate_rows,
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
                   "tracker": tracker, "min_area_frac": min_area_frac,
                   "max_area_frac": max_area_frac, "max_aspect": max_aspect,
                   "reject_plain_bg": reject_plain_bg,
                   "plain_min_area": plain_min_area,
                   "plain_sat": plain_sat, "plain_lap": plain_lap,
                   "plain_gray_std": plain_gray_std,
                   "plain_small_min_area": plain_small_min_area,
                   "low_detail_sat": low_detail_sat,
                   "low_detail_gray_std": low_detail_gray_std,
                   "low_detail_edge_frac": low_detail_edge_frac,
                   "crops_per_track": crops_per_track},
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
    ap.add_argument("--min-area-frac", type=float, default=0.0005)
    ap.add_argument("--max-area-frac", type=float, default=0.55)
    ap.add_argument("--max-aspect", type=float, default=8.0)
    ap.add_argument("--no-reject-plain-bg", action="store_true")
    ap.add_argument("--plain-min-area", type=float, default=0.015)
    ap.add_argument("--plain-sat", type=float, default=45.0)
    ap.add_argument("--plain-lap", type=float, default=55.0)
    ap.add_argument("--plain-gray-std", type=float, default=30.0)
    ap.add_argument("--plain-small-min-area", type=float, default=0.008)
    ap.add_argument("--low-detail-sat", type=float, default=35.0)
    ap.add_argument("--low-detail-gray-std", type=float, default=18.0)
    ap.add_argument("--low-detail-edge-frac", type=float, default=0.03)
    ap.add_argument("--crops-per-track", type=int, default=5)
    args = ap.parse_args()
    run(args.video, args.weights, args.out, imgsz=args.imgsz, conf=args.conf,
        vid_stride=args.vid_stride, min_frames=args.min_frames,
        min_peak_conf=args.min_peak_conf, device=args.device,
        min_area_frac=args.min_area_frac, max_area_frac=args.max_area_frac,
        max_aspect=args.max_aspect,
        reject_plain_bg=not args.no_reject_plain_bg,
        plain_min_area=args.plain_min_area, plain_sat=args.plain_sat,
        plain_lap=args.plain_lap, plain_gray_std=args.plain_gray_std,
        plain_small_min_area=args.plain_small_min_area,
        low_detail_sat=args.low_detail_sat,
        low_detail_gray_std=args.low_detail_gray_std,
        low_detail_edge_frac=args.low_detail_edge_frac,
        crops_per_track=args.crops_per_track)


if __name__ == "__main__":
    main()

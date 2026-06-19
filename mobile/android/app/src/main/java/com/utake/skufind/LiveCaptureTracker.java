package com.utake.skufind;

import android.graphics.RectF;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Lightweight live-session memory for the tablet.
 *
 * The TFLite model only detects product boxes frame-by-frame; it does not read
 * the SKU (OCR is server-side). To still bind feedback to a specific physical
 * product, this tracker gives every product a stable identity from two signals:
 *
 *   1. position — IoU with the previous frame (cheap frame-to-frame continuity);
 *   2. appearance — a tiny mean-removed luma fingerprint of the crop, so the same
 *      product is re-identified after the camera pans away and comes back, even
 *      though its box is now somewhere else on screen.
 *
 * Once a product has been seen sharp/large/confident for a few frames it is
 * marked captured: its box is no longer drawn (the manager only sees what still
 * needs shooting), and appearance re-id keeps it captured if it reappears, so it
 * is never outlined again.
 */
public class LiveCaptureTracker {
    private static final float MATCH_IOU = 0.28f;        // same product, frame-to-frame
    private static final float APPEARANCE_MATCH = 0.62f; // re-id after the box moved/left
    private static final int GOOD_FRAMES_TO_CAPTURE = 3;
    private static final int DROP_MISSED_AFTER = 40;
    private static final int KEEP_CAPTURED_MISSED = 600; // remember captured items much longer
    private static final int MAX_TRACKS = 240;           // bound memory on long scans
    private static final float EMA = 0.35f;              // box smoothing
    private static final float SIG_EMA = 0.25f;          // fingerprint smoothing (pre-capture)

    private final List<Track> tracks = new ArrayList<>();
    private int nextId = 1;

    public Result update(List<ProductDetection> detections) {
        for (Track t : tracks) {
            t.updated = false;
            t.missedFrames++;
        }

        // Greedy one-to-one assignment: a detection binds to an existing track if
        // it overlaps it (IoU) OR looks like it (appearance), strongest match first.
        // One-to-one keeps two identical side-by-side products from collapsing.
        int n = detections.size();
        Track[] owner = new Track[n];
        List<float[]> pairs = new ArrayList<>(); // {detIdx, trackIdx, score}
        for (int di = 0; di < n; di++) {
            ProductDetection det = detections.get(di);
            for (int ti = 0; ti < tracks.size(); ti++) {
                Track t = tracks.get(ti);
                float iou = iou(t.box, det.normalizedBounds);
                float app = sim(t.signature, det.signature);
                boolean valid = iou >= MATCH_IOU || app >= APPEARANCE_MATCH;
                if (valid) {
                    // Spatial continuity outranks pure re-id (+1 bias), so a product
                    // still in view binds to itself rather than to a look-alike.
                    float score = iou >= MATCH_IOU ? (1f + iou) : app;
                    pairs.add(new float[]{di, ti, score});
                }
            }
        }
        pairs.sort((a, b) -> Float.compare(b[2], a[2]));
        boolean[] trackUsed = new boolean[tracks.size()];
        for (float[] p : pairs) {
            int di = (int) p[0];
            int ti = (int) p[1];
            if (owner[di] != null || trackUsed[ti]) {
                continue;
            }
            owner[di] = tracks.get(ti);
            trackUsed[ti] = true;
        }

        for (int di = 0; di < n; di++) {
            if (owner[di] == null) {
                Track t = new Track(nextId++, detections.get(di).normalizedBounds);
                tracks.add(t);
                owner[di] = t;
            }
            owner[di].update(detections.get(di));
        }

        Iterator<Track> it = tracks.iterator();
        while (it.hasNext()) {
            Track t = it.next();
            int limit = t.captured ? KEEP_CAPTURED_MISSED : DROP_MISSED_AFTER;
            if (t.missedFrames > limit) {
                it.remove();
            }
        }
        if (tracks.size() > MAX_TRACKS) {
            // Drop the longest-unseen non-captured tracks first.
            tracks.sort((a, b) -> Integer.compare(b.missedFrames, a.missedFrames));
            for (Iterator<Track> i = tracks.iterator(); i.hasNext() && tracks.size() > MAX_TRACKS; ) {
                Track t = i.next();
                if (!t.captured) {
                    i.remove();
                }
            }
        }

        List<ProductDetection> visible = new ArrayList<>();
        List<RectF> capturedBoxes = new ArrayList<>();
        for (int di = 0; di < n; di++) {
            if (!owner[di].captured) {
                visible.add(detections.get(di));
            }
        }
        int captured = 0;
        int needsRetake = 0;
        for (Track t : tracks) {
            if (t.captured) {
                captured++;
                if (t.updated) {
                    capturedBoxes.add(new RectF(t.box));
                }
            } else if (t.updated) {
                needsRetake++;
            }
        }
        return new Result(visible, capturedBoxes, captured, needsRetake, tracks.size());
    }

    public void reset() {
        tracks.clear();
        nextId = 1;
    }

    private static float iou(RectF a, RectF b) {
        float left = Math.max(a.left, b.left);
        float top = Math.max(a.top, b.top);
        float right = Math.min(a.right, b.right);
        float bottom = Math.min(a.bottom, b.bottom);
        float inter = Math.max(0f, right - left) * Math.max(0f, bottom - top);
        float areaA = Math.max(0f, a.width()) * Math.max(0f, a.height());
        float areaB = Math.max(0f, b.width()) * Math.max(0f, b.height());
        return inter / (areaA + areaB - inter + 1e-6f);
    }

    /** Cosine similarity of two unit-length fingerprints; 0 if either is absent. */
    private static float sim(float[] a, float[] b) {
        if (a == null || b == null || a.length != b.length) {
            return 0f;
        }
        float dot = 0f;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
        }
        return dot;
    }

    public static class Result {
        public final List<ProductDetection> visibleDetections;
        public final List<RectF> capturedBoxes;
        public final int capturedCount;
        public final int needsRetakeCount;
        public final int trackedCount;

        Result(List<ProductDetection> visibleDetections, List<RectF> capturedBoxes,
               int capturedCount, int needsRetakeCount, int trackedCount) {
            this.visibleDetections = visibleDetections;
            this.capturedBoxes = capturedBoxes;
            this.capturedCount = capturedCount;
            this.needsRetakeCount = needsRetakeCount;
            this.trackedCount = trackedCount;
        }
    }

    private static class Track {
        final int id;
        RectF box;
        float[] signature;
        int goodFrames = 0;
        int missedFrames = 0;
        boolean captured = false;
        boolean updated = false;
        float bestConfidence = 0f;

        Track(int id, RectF box) {
            this.id = id;
            this.box = new RectF(box);
        }

        void update(ProductDetection detection) {
            // Continuous frame: smooth the box. Re-identified after the box moved
            // away (low overlap): snap to the new position so the ✓ doesn't fly
            // across the screen.
            if (iou(box, detection.normalizedBounds) >= MATCH_IOU) {
                smoothTo(detection.normalizedBounds);
            } else {
                box.set(detection.normalizedBounds);
            }
            updated = true;
            missedFrames = 0;
            bestConfidence = Math.max(bestConfidence, detection.confidence);
            // Freeze the fingerprint once captured so re-id stays consistent; before
            // that, smooth it toward the latest crop.
            if (!captured && detection.signature != null) {
                signature = blend(signature, detection.signature);
            }
            if (detection.recognized) {
                goodFrames++;
            } else {
                goodFrames = Math.max(0, goodFrames - 1);
            }
            if (goodFrames >= GOOD_FRAMES_TO_CAPTURE) {
                captured = true;
            }
        }

        private static float[] blend(float[] prev, float[] next) {
            if (prev == null) {
                return next.clone();
            }
            float[] out = new float[prev.length];
            double sumSq = 0;
            for (int i = 0; i < out.length; i++) {
                out[i] = prev[i] * (1f - SIG_EMA) + next[i] * SIG_EMA;
                sumSq += (double) out[i] * out[i];
            }
            float norm = (float) Math.sqrt(sumSq);
            if (norm > 1e-6f) {
                for (int i = 0; i < out.length; i++) {
                    out[i] /= norm;
                }
            }
            return out;
        }

        private void smoothTo(RectF next) {
            box.left = box.left * (1f - EMA) + next.left * EMA;
            box.top = box.top * (1f - EMA) + next.top * EMA;
            box.right = box.right * (1f - EMA) + next.right * EMA;
            box.bottom = box.bottom * (1f - EMA) + next.bottom * EMA;
        }
    }
}

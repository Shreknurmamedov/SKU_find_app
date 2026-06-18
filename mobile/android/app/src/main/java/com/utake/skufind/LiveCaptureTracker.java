package com.utake.skufind;

import android.graphics.RectF;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Lightweight live-session memory for the tablet.
 *
 * The TFLite model detects product boxes frame-by-frame. This tracker binds
 * nearby boxes into short-lived physical-object tracks, promotes a track to
 * captured after several sharp/confident frames, and then suppresses its box so
 * the manager only sees products that still need attention.
 */
public class LiveCaptureTracker {
    private static final float MATCH_IOU = 0.28f;
    private static final int GOOD_FRAMES_TO_CAPTURE = 3;
    private static final int DROP_MISSED_AFTER = 40;
    private static final int KEEP_CAPTURED_MISSED = 180;
    private static final float EMA = 0.35f;

    private final List<Track> tracks = new ArrayList<>();
    private int nextId = 1;

    public Result update(List<ProductDetection> detections) {
        for (Track t : tracks) {
            t.updated = false;
            t.missedFrames++;
        }

        List<ProductDetection> visible = new ArrayList<>();
        for (ProductDetection detection : detections) {
            Track track = bestMatch(detection.normalizedBounds);
            if (track == null) {
                track = new Track(nextId++, detection.normalizedBounds);
                tracks.add(track);
            }
            track.update(detection);
            if (!track.captured) {
                visible.add(detection);
            }
        }

        Iterator<Track> it = tracks.iterator();
        while (it.hasNext()) {
            Track t = it.next();
            int limit = t.captured ? KEEP_CAPTURED_MISSED : DROP_MISSED_AFTER;
            if (t.missedFrames > limit) {
                it.remove();
            }
        }

        int captured = 0;
        int needsRetake = 0;
        for (Track t : tracks) {
            if (t.captured) {
                captured++;
            } else if (t.updated) {
                needsRetake++;
            }
        }
        return new Result(visible, captured, needsRetake, tracks.size());
    }

    public void reset() {
        tracks.clear();
        nextId = 1;
    }

    private Track bestMatch(RectF box) {
        Track best = null;
        float bestScore = 0f;
        for (Track t : tracks) {
            float score = iou(t.box, box);
            if (score > bestScore) {
                bestScore = score;
                best = t;
            }
        }
        return bestScore >= MATCH_IOU ? best : null;
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

    public static class Result {
        public final List<ProductDetection> visibleDetections;
        public final int capturedCount;
        public final int needsRetakeCount;
        public final int trackedCount;

        Result(List<ProductDetection> visibleDetections, int capturedCount,
               int needsRetakeCount, int trackedCount) {
            this.visibleDetections = visibleDetections;
            this.capturedCount = capturedCount;
            this.needsRetakeCount = needsRetakeCount;
            this.trackedCount = trackedCount;
        }
    }

    private static class Track {
        final int id;
        RectF box;
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
            smoothTo(detection.normalizedBounds);
            updated = true;
            missedFrames = 0;
            bestConfidence = Math.max(bestConfidence, detection.confidence);
            if (detection.recognized) {
                goodFrames++;
            } else {
                goodFrames = Math.max(0, goodFrames - 1);
            }
            if (goodFrames >= GOOD_FRAMES_TO_CAPTURE) {
                captured = true;
            }
        }

        private void smoothTo(RectF next) {
            box.left = box.left * (1f - EMA) + next.left * EMA;
            box.top = box.top * (1f - EMA) + next.top * EMA;
            box.right = box.right * (1f - EMA) + next.right * EMA;
            box.bottom = box.bottom * (1f - EMA) + next.bottom * EMA;
        }
    }
}

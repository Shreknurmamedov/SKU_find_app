package com.utake.skufind;

/**
 * Estimates how far the camera view has moved across the scene, in frame units
 * (1.0 == one full frame width/height), accumulating into (cx, cy).
 *
 * <p>Primary signal: coarse block matching between consecutive luma frames —
 * this captures apparent motion from BOTH rotating in place and walking along a
 * shelf. When the scene has too little texture (low match confidence) it falls
 * back to gyroscope deltas supplied by the caller.
 */
public class MotionTracker {

    /** Half search range in downscaled pixels (max motion per frame). */
    private static final int SEARCH = 10;
    /** Subsample step when scoring, for speed. */
    private static final int STEP = 2;
    private static final float MIN_CONFIDENCE = 0.25f;
    /** Rough camera field of view (radians) for gyro fallback conversion. */
    private static final float HFOV = 1.10f; // ~63 deg
    private static final float VFOV = 0.85f; // ~49 deg

    private byte[] prev;
    private int prevW;
    private int prevH;

    private float cx;
    private float cy;
    private float speed;       // last step magnitude in frame units
    private float confidence;  // 0..1 of last block match

    public float cx() { return cx; }
    public float cy() { return cy; }
    public float speed() { return speed; }
    public float confidence() { return confidence; }

    public void reset() {
        prev = null;
        cx = cy = 0f;
        speed = 0f;
        confidence = 0f;
    }

    /**
     * Update with the latest frame.
     *
     * @param frame           downscaled luma frame (sensor orientation)
     * @param rotationDegrees ImageProxy rotation (0/90/180/270)
     * @param dYaw            gyro yaw delta since last frame (rad), fallback only
     * @param dPitch          gyro pitch delta since last frame (rad), fallback only
     */
    public void update(LumaFrame frame, int rotationDegrees, float dYaw, float dPitch) {
        if (frame == null) {
            return;
        }
        byte[] curr = frame.data;
        int w = frame.width;
        int h = frame.height;

        float dcx;
        float dcy;

        if (prev != null && prevW == w && prevH == h) {
            float[] m = blockMatch(curr, prev, w, h);
            int ox = (int) m[0];
            int oy = (int) m[1];
            confidence = m[2];

            if (confidence >= MIN_CONFIDENCE) {
                // content displacement (ox,oy) -> world window moves opposite
                float sx = -ox / (float) w;
                float sy = -oy / (float) h;
                float[] rot = rotateToDisplay(sx, sy, rotationDegrees);
                dcx = rot[0];
                dcy = rot[1];
            } else {
                // low texture -> trust gyro
                dcx = dYaw / HFOV;
                dcy = -dPitch / VFOV;
            }
        } else {
            dcx = 0f;
            dcy = 0f;
            confidence = 0f;
        }

        // Horizontal was reversed on the test device (pan left moved the grid
        // right), so the coverage window follows the camera correctly.
        cx -= dcx;
        cy += dcy;
        speed = (float) Math.hypot(dcx, dcy);

        if (prev == null || prevW != w || prevH != h) {
            prev = new byte[curr.length];
            prevW = w;
            prevH = h;
        }
        System.arraycopy(curr, 0, prev, 0, curr.length);
    }

    /** Returns {ox, oy, confidence}: content displacement prev->curr. */
    private float[] blockMatch(byte[] curr, byte[] prev, int w, int h) {
        long bestCost = Long.MAX_VALUE;
        long worstCost = 0;
        int bestOx = 0;
        int bestOy = 0;
        int margin = SEARCH + 1;

        for (int oy = -SEARCH; oy <= SEARCH; oy++) {
            for (int ox = -SEARCH; ox <= SEARCH; ox++) {
                long cost = 0;
                int count = 0;
                for (int y = margin; y < h - margin; y += STEP) {
                    int py = y - oy;
                    int currRow = y * w;
                    int prevRow = py * w;
                    for (int x = margin; x < w - margin; x += STEP) {
                        int px = x - ox;
                        int a = curr[currRow + x] & 0xff;
                        int b = prev[prevRow + px] & 0xff;
                        cost += Math.abs(a - b);
                        count++;
                    }
                }
                if (count == 0) {
                    continue;
                }
                long avg = cost / count; // mean abs diff per pixel (0..255)
                if (avg < bestCost) {
                    bestCost = avg;
                    bestOx = ox;
                    bestOy = oy;
                }
                if (avg > worstCost) {
                    worstCost = avg;
                }
            }
        }

        // confidence: clear minimum + not pinned at the search boundary
        float conf = worstCost > 0 ? (float) (worstCost - bestCost) / (float) worstCost : 0f;
        if (Math.abs(bestOx) >= SEARCH || Math.abs(bestOy) >= SEARCH) {
            conf *= 0.4f; // motion likely exceeded search window (too fast)
        }
        return new float[]{bestOx, bestOy, conf};
    }

    /** Rotate a sensor-space delta into display space. Signs may need a per-device flip. */
    private float[] rotateToDisplay(float x, float y, int deg) {
        switch (((deg % 360) + 360) % 360) {
            case 90:
                return new float[]{y, -x};
            case 180:
                return new float[]{-x, -y};
            case 270:
                return new float[]{-y, x};
            default:
                return new float[]{x, y};
        }
    }
}

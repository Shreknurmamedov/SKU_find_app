package com.utake.skufind;

/**
 * Per-frame sharpness from the variance of the Laplacian on the luma frame.
 * A blurry frame has a low-variance Laplacian; a sharp one has high variance.
 * The raw variance is squashed into a 0..1 score so thresholds are stable.
 */
public final class FrameQuality {

    /** Soft-knee constant: variance == K maps to score 0.5. Tune per device. */
    private static final float K = 550f;

    private FrameQuality() {}

    /** Sharpness score in [0,1]; ~0 blurry, ~1 crisp. */
    public static float sharpScore(LumaFrame f) {
        if (f == null || f.width < 3 || f.height < 3) {
            return 0f;
        }
        int w = f.width;
        int h = f.height;
        byte[] d = f.data;
        double sum = 0.0;
        double sumSq = 0.0;
        int n = 0;
        for (int y = 1; y < h - 1; y++) {
            int row = y * w;
            for (int x = 1; x < w - 1; x++) {
                int i = row + x;
                int c = d[i] & 0xff;
                int lap = 4 * c
                        - (d[i - 1] & 0xff)
                        - (d[i + 1] & 0xff)
                        - (d[i - w] & 0xff)
                        - (d[i + w] & 0xff);
                sum += lap;
                sumSq += (double) lap * lap;
                n++;
            }
        }
        if (n == 0) {
            return 0f;
        }
        double mean = sum / n;
        double var = sumSq / n - mean * mean;
        if (var < 0) {
            var = 0;
        }
        return (float) (var / (var + K));
    }
}

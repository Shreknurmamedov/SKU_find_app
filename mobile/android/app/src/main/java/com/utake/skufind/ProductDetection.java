package com.utake.skufind;

import android.graphics.RectF;

public class ProductDetection {
    public static final int STATE_GOOD = 0;
    public static final int STATE_BLUR = 1;
    public static final int STATE_FAR = 2;
    public static final int STATE_UNCERTAIN = 3;

    public final RectF normalizedBounds;
    public final boolean recognized;
    public final String label;
    public final float confidence;
    public final int qualityState;
    public final float sharpness;
    public final float areaFraction;
    /**
     * Lightweight on-device appearance fingerprint of the crop (mean-removed,
     * L2-normalized 8x8 luma, 64 floats), or null. Lets {@link LiveCaptureTracker}
     * re-identify the same physical product after the camera pans away and back,
     * so an already-captured item is not outlined again. Full SKU recognition is
     * still server-side; this only binds the live box to a stable product identity.
     */
    public float[] signature;
    public float productness = 1f;
    public boolean stale = false;

    public ProductDetection(RectF normalizedBounds, boolean recognized, String label, float confidence) {
        this(normalizedBounds, recognized, label, confidence,
                recognized ? STATE_GOOD : STATE_UNCERTAIN, 0f, 0f);
    }

    public ProductDetection(RectF normalizedBounds, boolean recognized, String label,
                            float confidence, int qualityState,
                            float sharpness, float areaFraction) {
        this.normalizedBounds = normalizedBounds;
        this.recognized = recognized;
        this.label = label;
        this.confidence = confidence;
        this.qualityState = qualityState;
        this.sharpness = sharpness;
        this.areaFraction = areaFraction;
    }
}

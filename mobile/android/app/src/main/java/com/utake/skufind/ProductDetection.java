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

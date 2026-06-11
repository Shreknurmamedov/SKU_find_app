package com.utake.skufind;

import android.graphics.RectF;

public class ProductDetection {
    public final RectF normalizedBounds;
    public final boolean recognized;
    public final String label;
    public final float confidence;

    public ProductDetection(RectF normalizedBounds, boolean recognized, String label, float confidence) {
        this.normalizedBounds = normalizedBounds;
        this.recognized = recognized;
        this.label = label;
        this.confidence = confidence;
    }
}

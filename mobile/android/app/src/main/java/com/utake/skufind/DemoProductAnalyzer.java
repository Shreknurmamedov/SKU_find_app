package com.utake.skufind;

import android.graphics.RectF;

import androidx.camera.core.ImageProxy;

import java.util.ArrayList;
import java.util.List;

public class DemoProductAnalyzer implements ProductAnalyzer {
    private long frameIndex = 0;

    @Override
    public List<ProductDetection> analyze(ImageProxy image) {
        frameIndex++;
        float drift = (float) (Math.sin(frameIndex / 12.0) * 0.025);
        List<ProductDetection> detections = new ArrayList<>();
        detections.add(new ProductDetection(
                clamp(new RectF(0.08f + drift, 0.16f, 0.34f + drift, 0.44f)),
                true,
                "держите",
                0.91f,
                ProductDetection.STATE_GOOD,
                0.72f,
                0.07f
        ));
        detections.add(new ProductDetection(
                clamp(new RectF(0.46f - drift, 0.20f, 0.76f - drift, 0.50f)),
                false,
                "медленнее",
                0.64f,
                ProductDetection.STATE_BLUR,
                0.24f,
                0.09f
        ));
        detections.add(new ProductDetection(
                clamp(new RectF(0.24f, 0.58f + drift, 0.62f, 0.86f + drift)),
                false,
                "ближе",
                0.78f,
                ProductDetection.STATE_FAR,
                0.65f,
                0.025f
        ));
        return detections;
    }

    private RectF clamp(RectF rect) {
        return new RectF(
                Math.max(0f, Math.min(1f, rect.left)),
                Math.max(0f, Math.min(1f, rect.top)),
                Math.max(0f, Math.min(1f, rect.right)),
                Math.max(0f, Math.min(1f, rect.bottom))
        );
    }
}

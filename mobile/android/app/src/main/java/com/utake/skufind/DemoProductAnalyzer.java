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
                "recognized",
                0.91f
        ));
        detections.add(new ProductDetection(
                clamp(new RectF(0.46f - drift, 0.20f, 0.76f - drift, 0.50f)),
                false,
                "unknown",
                0.64f
        ));
        detections.add(new ProductDetection(
                clamp(new RectF(0.24f, 0.58f + drift, 0.62f, 0.86f + drift)),
                true,
                "own SKU",
                0.78f
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

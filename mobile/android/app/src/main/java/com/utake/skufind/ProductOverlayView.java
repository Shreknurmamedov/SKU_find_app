package com.utake.skufind;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class ProductOverlayView extends View {
    private final Paint fillPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint strokePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textBackgroundPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final List<ProductDetection> detections = new ArrayList<>();

    public ProductOverlayView(Context context) {
        super(context);
        init();
    }

    public ProductOverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        setWillNotDraw(false);
        strokePaint.setStyle(Paint.Style.STROKE);
        strokePaint.setStrokeWidth(5f);
        fillPaint.setStyle(Paint.Style.FILL);
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(32f);
        textPaint.setFakeBoldText(true);
        textBackgroundPaint.setStyle(Paint.Style.FILL);
    }

    public void setDetections(List<ProductDetection> newDetections) {
        detections.clear();
        detections.addAll(newDetections);
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        int width = getWidth();
        int height = getHeight();

        for (ProductDetection detection : detections) {
            int color = detection.recognized ? Color.rgb(36, 182, 103) : Color.rgb(230, 58, 58);
            strokePaint.setColor(color);
            fillPaint.setColor(withAlpha(color, 52));
            textBackgroundPaint.setColor(withAlpha(color, 210));

            RectF bounds = new RectF(
                    detection.normalizedBounds.left * width,
                    detection.normalizedBounds.top * height,
                    detection.normalizedBounds.right * width,
                    detection.normalizedBounds.bottom * height
            );

            Path path = segmentedPath(bounds);
            canvas.drawPath(path, fillPaint);
            canvas.drawPath(path, strokePaint);

            String label = detection.label + " "
                    + String.format(Locale.US, "%.0f%%", detection.confidence * 100f);
            float textWidth = textPaint.measureText(label);
            RectF labelBackground = new RectF(
                    bounds.left,
                    Math.max(0, bounds.top - 42f),
                    Math.min(width, bounds.left + textWidth + 24f),
                    bounds.top
            );
            canvas.drawRoundRect(labelBackground, 8f, 8f, textBackgroundPaint);
            canvas.drawText(label, labelBackground.left + 12f, labelBackground.bottom - 11f, textPaint);
        }
    }

    private Path segmentedPath(RectF bounds) {
        float notch = Math.min(bounds.width(), bounds.height()) * 0.12f;
        Path path = new Path();
        path.moveTo(bounds.left + notch, bounds.top);
        path.lineTo(bounds.right - notch, bounds.top);
        path.lineTo(bounds.right, bounds.top + notch);
        path.lineTo(bounds.right, bounds.bottom - notch);
        path.lineTo(bounds.right - notch, bounds.bottom);
        path.lineTo(bounds.left + notch, bounds.bottom);
        path.lineTo(bounds.left, bounds.bottom - notch);
        path.lineTo(bounds.left, bounds.top + notch);
        path.close();
        return path;
    }

    private int withAlpha(int color, int alpha) {
        return Color.argb(alpha, Color.red(color), Color.green(color), Color.blue(color));
    }
}

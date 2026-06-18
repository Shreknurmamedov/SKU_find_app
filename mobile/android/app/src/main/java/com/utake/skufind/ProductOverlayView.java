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
    private final Paint smallTextPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textBackgroundPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint bannerPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final List<ProductDetection> detections = new ArrayList<>();
    private int todoCount = 0;
    private int capturedCount = 0;
    private int retakeCount = 0;
    private int closerCount = 0;
    private int blurCount = 0;
    private int aimCount = 0;
    private String primaryHint = "Наведите камеру на полку";

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
        smallTextPaint.setColor(Color.WHITE);
        smallTextPaint.setTextSize(28f);
        smallTextPaint.setFakeBoldText(true);
        textBackgroundPaint.setStyle(Paint.Style.FILL);
        bannerPaint.setStyle(Paint.Style.FILL);
    }

    public void setDetections(List<ProductDetection> newDetections) {
        detections.clear();
        detections.addAll(newDetections);
        invalidate();
    }

    public void setLiveSummary(int todoCount, int capturedCount, int retakeCount) {
        setLiveSummary(todoCount, capturedCount, retakeCount, 0, 0, retakeCount,
                "Ведите камеру по полке");
    }

    public void setLiveSummary(int todoCount, int capturedCount, int retakeCount,
                               int closerCount, int blurCount, int aimCount,
                               String primaryHint) {
        this.todoCount = Math.max(0, todoCount);
        this.capturedCount = Math.max(0, capturedCount);
        this.retakeCount = Math.max(0, retakeCount);
        this.closerCount = Math.max(0, closerCount);
        this.blurCount = Math.max(0, blurCount);
        this.aimCount = Math.max(0, aimCount);
        this.primaryHint = primaryHint == null ? "" : primaryHint;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        int width = getWidth();
        int height = getHeight();

        drawSummary(canvas, width);
        for (ProductDetection detection : detections) {
            int color = colorFor(detection);
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
            if (detection.sharpness > 0f && detection.qualityState == ProductDetection.STATE_BLUR) {
                label = detection.label + " "
                        + String.format(Locale.US, "%.0f%%", detection.sharpness * 100f);
            }
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
        drawHintBanner(canvas, width, height);
    }

    private void drawSummary(Canvas canvas, int width) {
        String left = "Осталось " + todoCount;
        String middle = "Готово " + capturedCount;
        String right = "Проблемы " + retakeCount;
        float pad = 16f;
        float gap = 18f;
        float h = 54f;
        smallTextPaint.setTextSize(28f);
        float leftW = smallTextPaint.measureText(left) + pad * 2;
        float midW = smallTextPaint.measureText(middle) + pad * 2;
        float rightW = smallTextPaint.measureText(right) + pad * 2;
        float total = leftW + midW + rightW + gap * 2;
        float available = Math.max(1f, width - 32f);
        if (total > available) {
            float scale = Math.max(0.68f, available / total);
            smallTextPaint.setTextSize(28f * scale);
            h = Math.max(44f, 54f * scale);
            gap = Math.max(8f, 18f * scale);
            leftW = smallTextPaint.measureText(left) + pad * 2;
            midW = smallTextPaint.measureText(middle) + pad * 2;
            rightW = smallTextPaint.measureText(right) + pad * 2;
            total = leftW + midW + rightW + gap * 2;
        }
        float x = Math.max(16f, (width - total) / 2f);
        drawPill(canvas, x, 18f, leftW, h, Color.rgb(45, 120, 220), left);
        x += leftW + gap;
        drawPill(canvas, x, 18f, midW, h, Color.rgb(36, 182, 103), middle);
        x += midW + gap;
        drawPill(canvas, x, 18f, rightW, h, Color.rgb(230, 58, 58), right);
    }

    private void drawPill(Canvas canvas, float x, float y, float w, float h, int color, String label) {
        textBackgroundPaint.setColor(withAlpha(color, 215));
        RectF r = new RectF(x, y, x + w, y + h);
        canvas.drawRoundRect(r, 12f, 12f, textBackgroundPaint);
        Paint.FontMetrics fm = smallTextPaint.getFontMetrics();
        float baseline = y + (h - fm.ascent - fm.descent) / 2f;
        canvas.drawText(label, x + 16f, baseline, smallTextPaint);
    }

    private void drawHintBanner(Canvas canvas, int width, int height) {
        if (primaryHint == null || primaryHint.isEmpty()) {
            return;
        }
        String detail = detailText();
        String label = detail.isEmpty() ? primaryHint : primaryHint + " · " + detail;
        float padX = 18f;
        float bannerH = 58f;
        float y = Math.max(0f, height - bannerH - 20f);
        smallTextPaint.setTextSize(28f);
        float maxText = width - padX * 4;
        if (smallTextPaint.measureText(label) > maxText) {
            while (label.length() > 12 && smallTextPaint.measureText(label + "...") > maxText) {
                label = label.substring(0, label.length() - 1);
            }
            label = label + "...";
        }
        int color = dominantHintColor();
        bannerPaint.setColor(withAlpha(color, 225));
        RectF r = new RectF(padX, y, width - padX, y + bannerH);
        canvas.drawRoundRect(r, 10f, 10f, bannerPaint);
        Paint.FontMetrics fm = smallTextPaint.getFontMetrics();
        float baseline = y + (bannerH - fm.ascent - fm.descent) / 2f;
        canvas.drawText(label, r.left + 18f, baseline, smallTextPaint);
    }

    private String detailText() {
        if (closerCount + blurCount + aimCount == 0) {
            return "";
        }
        List<String> parts = new ArrayList<>();
        if (closerCount > 0) {
            parts.add("ближе " + closerCount);
        }
        if (blurCount > 0) {
            parts.add("резче " + blurCount);
        }
        if (aimCount > 0) {
            parts.add("навести " + aimCount);
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < parts.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(parts.get(i));
        }
        return sb.toString();
    }

    private int dominantHintColor() {
        if (closerCount > 0) {
            return Color.rgb(240, 137, 35);
        }
        if (blurCount > 0) {
            return Color.rgb(220, 55, 55);
        }
        if (aimCount > 0) {
            return Color.rgb(45, 120, 220);
        }
        return Color.rgb(36, 182, 103);
    }

    private int colorFor(ProductDetection detection) {
        if (detection.qualityState == ProductDetection.STATE_GOOD) {
            return Color.rgb(36, 182, 103);
        }
        if (detection.qualityState == ProductDetection.STATE_FAR) {
            return Color.rgb(240, 137, 35);
        }
        if (detection.qualityState == ProductDetection.STATE_BLUR) {
            return Color.rgb(220, 55, 55);
        }
        return Color.rgb(45, 120, 220);
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

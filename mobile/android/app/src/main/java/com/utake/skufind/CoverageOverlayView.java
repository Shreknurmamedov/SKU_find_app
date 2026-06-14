package com.utake.skufind;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import java.util.List;

/**
 * Draws the live coverage feedback over the camera preview:
 * a scrolling grid of cells (grey = not scanned, amber = re-shoot, green = ok),
 * a minimap of the whole scanned area with the current viewport, and a hint
 * line with an optional arrow toward the nearest spot that needs re-shooting.
 */
public class CoverageOverlayView extends View {

    private static final int COL_UNSEEN = Color.argb(70, 150, 150, 150);
    private static final int COL_POOR = Color.argb(120, 240, 176, 32);
    private static final int COL_GOOD = Color.argb(110, 36, 182, 103);
    private static final int COL_GRID = Color.argb(90, 255, 255, 255);

    private final Paint fill = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint grid = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint mapBg = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint mapCell = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint mapView = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint text = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textBg = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint arrow = new Paint(Paint.ANTI_ALIAS_FLAG);

    // Snapshots built on the camera thread so onDraw never touches the live grid.
    private List<ScanGrid.Cell> viewCells;
    private List<ScanGrid.Cell> mapCells;
    private int minCol, maxCol, minRow, maxRow;
    private boolean hasBounds;
    private float cx;
    private float cy;
    private String hint = "";
    private float[] arrowDir; // world-units direction to nearest poor cell, or null
    private int good;
    private int poor;
    private boolean blurNow;

    public CoverageOverlayView(Context context) {
        super(context);
        init();
    }

    public CoverageOverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    private void init() {
        setWillNotDraw(false);
        fill.setStyle(Paint.Style.FILL);
        grid.setStyle(Paint.Style.STROKE);
        grid.setStrokeWidth(2f);
        grid.setColor(COL_GRID);
        mapBg.setStyle(Paint.Style.FILL);
        mapBg.setColor(Color.argb(150, 20, 20, 20));
        mapCell.setStyle(Paint.Style.FILL);
        mapView.setStyle(Paint.Style.STROKE);
        mapView.setStrokeWidth(3f);
        mapView.setColor(Color.WHITE);
        text.setColor(Color.WHITE);
        text.setTextSize(38f);
        text.setFakeBoldText(true);
        textBg.setStyle(Paint.Style.FILL);
        textBg.setColor(Color.argb(160, 0, 0, 0));
        arrow.setColor(Color.rgb(240, 176, 32));
        arrow.setStyle(Paint.Style.FILL);
    }

    public void setState(List<ScanGrid.Cell> viewCells, List<ScanGrid.Cell> mapCells,
                         int minCol, int maxCol, int minRow, int maxRow, boolean hasBounds,
                         float cx, float cy, String hint, float[] arrowDir,
                         int good, int poor, boolean blurNow) {
        this.viewCells = viewCells;
        this.mapCells = mapCells;
        this.minCol = minCol;
        this.maxCol = maxCol;
        this.minRow = minRow;
        this.maxRow = maxRow;
        this.hasBounds = hasBounds;
        this.cx = cx;
        this.cy = cy;
        this.hint = hint == null ? "" : hint;
        this.arrowDir = arrowDir;
        this.good = good;
        this.poor = poor;
        this.blurNow = blurNow;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (viewCells == null) {
            return;
        }
        int w = getWidth();
        int h = getHeight();

        drawGrid(canvas, w, h);
        drawArrow(canvas, w, h);
        drawMinimap(canvas, w, h);
        drawStatus(canvas, w, h);
    }

    private void drawGrid(Canvas canvas, int w, int h) {
        for (ScanGrid.Cell c : viewCells) {
            float left = (c.col * ScanGrid.CELL - cx) * w;
            float top = (c.row * ScanGrid.CELL - cy) * h;
            float right = ((c.col + 1) * ScanGrid.CELL - cx) * w;
            float bottom = ((c.row + 1) * ScanGrid.CELL - cy) * h;
            RectF r = new RectF(left, top, right, bottom);
            int state = c.state();
            fill.setColor(state == ScanGrid.GOOD ? COL_GOOD
                    : state == ScanGrid.POOR ? COL_POOR : COL_UNSEEN);
            canvas.drawRect(r, fill);
            canvas.drawRect(r, grid);
        }
    }

    private void drawArrow(Canvas canvas, int w, int h) {
        if (arrowDir == null) {
            return;
        }
        float dx = arrowDir[0];
        float dy = arrowDir[1];
        float len = (float) Math.hypot(dx, dy);
        if (len < 0.25f) {
            return; // poor cell already near the centre
        }
        float nx = dx / len;
        float ny = dy / len;
        float cxp = w * 0.5f;
        float cyp = h * 0.5f;
        float tipX = cxp + nx * (w * 0.18f);
        float tipY = cyp + ny * (w * 0.18f);
        float backX = cxp - nx * (w * 0.02f);
        float backY = cyp - ny * (w * 0.02f);
        float perpX = -ny;
        float perpY = nx;
        float s = w * 0.035f;
        android.graphics.Path p = new android.graphics.Path();
        p.moveTo(tipX, tipY);
        p.lineTo(backX + perpX * s, backY + perpY * s);
        p.lineTo(backX - perpX * s, backY - perpY * s);
        p.close();
        canvas.drawPath(p, arrow);
    }

    private void drawMinimap(Canvas canvas, int w, int h) {
        if (!hasBounds || mapCells == null) {
            return;
        }
        float mapW = w * 0.26f;
        // include current viewport in bounds so it's always shown
        float minX = Math.min(minCol * ScanGrid.CELL, cx);
        float maxX = Math.max((maxCol + 1) * ScanGrid.CELL, cx + 1f);
        float minY = Math.min(minRow * ScanGrid.CELL, cy);
        float maxY = Math.max((maxRow + 1) * ScanGrid.CELL, cy + 1f);
        float worldW = Math.max(1e-3f, maxX - minX);
        float worldH = Math.max(1e-3f, maxY - minY);
        float scale = mapW / worldW;
        float mapH = worldH * scale;
        float pad = w * 0.03f;
        float ox = w - mapW - pad;
        float oy = pad;

        canvas.drawRect(ox - 6, oy - 6, ox + mapW + 6, oy + mapH + 6, mapBg);
        for (ScanGrid.Cell c : mapCells) {
            int state = c.state();
            mapCell.setColor(state == ScanGrid.GOOD ? Color.rgb(36, 182, 103)
                    : state == ScanGrid.POOR ? Color.rgb(240, 176, 32)
                    : Color.rgb(110, 110, 110));
            float l = ox + (c.col * ScanGrid.CELL - minX) * scale;
            float t = oy + (c.row * ScanGrid.CELL - minY) * scale;
            canvas.drawRect(l, t, l + ScanGrid.CELL * scale, t + ScanGrid.CELL * scale, mapCell);
        }
        float vl = ox + (cx - minX) * scale;
        float vt = oy + (cy - minY) * scale;
        canvas.drawRect(vl, vt, vl + scale, vt + scale, mapView);
    }

    private void drawStatus(Canvas canvas, int w, int h) {
        String status = "✓ " + good + "   ⟲ переснять: " + poor;
        float tw = text.measureText(status);
        canvas.drawRoundRect(20, 20, 40 + tw, 80, 12, 12, textBg);
        canvas.drawText(status, 30, 64, text);

        String line = blurNow ? "Размыто — двигайтесь медленнее" : hint;
        if (line != null && !line.isEmpty()) {
            float lw = text.measureText(line);
            float left = (w - lw) / 2f - 18;
            float top = h - 96;
            textBg.setColor(blurNow ? Color.argb(190, 150, 60, 0) : Color.argb(160, 0, 0, 0));
            canvas.drawRoundRect(left, top, left + lw + 36, top + 64, 14, 14, textBg);
            canvas.drawText(line, left + 18, top + 44, text);
            textBg.setColor(Color.argb(160, 0, 0, 0));
        }
    }
}

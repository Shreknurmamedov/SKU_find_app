package com.utake.skufind;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Virtual coverage grid for in-store scanning.
 *
 * <p>The world is measured in "frame units": the live camera frame is a 1x1
 * window whose top-left corner sits at the accumulated offset (cx, cy) produced
 * by {@link MotionTracker}. As the manager pans or walks, that window slides
 * over an unbounded grid of cells. Each cell remembers the best frame sharpness
 * seen while it was visible, which yields three states:
 *
 * <ul>
 *   <li>UNSEEN  - never in view (grey);</li>
 *   <li>POOR    - seen, but only blurry/low-quality frames (amber, re-shoot);</li>
 *   <li>GOOD    - seen with at least one sharp frame (green).</li>
 * </ul>
 */
public class ScanGrid {

    public static final int UNSEEN = 0;
    public static final int POOR = 1;
    public static final int GOOD = 2;

    /** Cell edge length in frame units (0.25 => 4 cells across one camera view). */
    public static final float CELL = 0.25f;
    /** Min sharpness score (0..1) for a cell to count as well captured. */
    public static final float GOOD_SHARPNESS = 0.55f;

    public static final class Cell {
        public final int col;
        public final int row;
        public boolean covered;
        public float bestSharpness;
        public int frames;

        Cell(int col, int row) {
            this.col = col;
            this.row = row;
        }

        public int state() {
            if (!covered) {
                return UNSEEN;
            }
            return bestSharpness >= GOOD_SHARPNESS ? GOOD : POOR;
        }
    }

    private final Map<Long, Cell> cells = new HashMap<>();
    private int minCol, maxCol, minRow, maxRow;
    private boolean hasBounds;

    private static long key(int col, int row) {
        return (((long) col) << 32) ^ (row & 0xffffffffL);
    }

    private Cell cell(int col, int row) {
        long k = key(col, row);
        Cell c = cells.get(k);
        if (c == null) {
            c = new Cell(col, row);
            cells.put(k, c);
            if (!hasBounds) {
                minCol = maxCol = col;
                minRow = maxRow = row;
                hasBounds = true;
            } else {
                minCol = Math.min(minCol, col);
                maxCol = Math.max(maxCol, col);
                minRow = Math.min(minRow, row);
                maxRow = Math.max(maxRow, row);
            }
        }
        return c;
    }

    /**
     * Mark every cell overlapping the current camera window with the frame's
     * sharpness score (0..1). The window spans world rect [cx, cx+1] x [cy, cy+1].
     */
    public void markView(float cx, float cy, float sharpness) {
        int c0 = (int) Math.floor(cx / CELL);
        int c1 = (int) Math.floor((cx + 1f - 1e-4f) / CELL);
        int r0 = (int) Math.floor(cy / CELL);
        int r1 = (int) Math.floor((cy + 1f - 1e-4f) / CELL);
        for (int col = c0; col <= c1; col++) {
            for (int row = r0; row <= r1; row++) {
                Cell cell = cell(col, row);
                cell.covered = true;
                cell.frames++;
                if (sharpness > cell.bestSharpness) {
                    cell.bestSharpness = sharpness;
                }
            }
        }
    }

    /** Cells overlapping the view window, for drawing the on-screen grid. */
    public List<Cell> cellsInView(float cx, float cy) {
        int c0 = (int) Math.floor(cx / CELL);
        int c1 = (int) Math.floor((cx + 1f) / CELL);
        int r0 = (int) Math.floor(cy / CELL);
        int r1 = (int) Math.floor((cy + 1f) / CELL);
        List<Cell> out = new ArrayList<>();
        for (int col = c0; col <= c1; col++) {
            for (int row = r0; row <= r1; row++) {
                long k = key(col, row);
                Cell c = cells.get(k);
                out.add(c != null ? c : new Cell(col, row));
            }
        }
        return out;
    }

    /** Snapshot of all known cells, for the minimap. */
    public List<Cell> allCells() {
        return new ArrayList<>(cells.values());
    }

    public int minCol() { return minCol; }
    public int maxCol() { return maxCol; }
    public int minRow() { return minRow; }
    public int maxRow() { return maxRow; }
    public boolean hasBounds() { return hasBounds; }

    public int countGood() { return count(GOOD); }
    public int countPoor() { return count(POOR); }

    private int count(int state) {
        int n = 0;
        for (Cell c : cells.values()) {
            if (c.state() == state) {
                n++;
            }
        }
        return n;
    }

    /**
     * Direction (dx, dy in frame units) from the view centre to the nearest
     * POOR cell that needs re-shooting, or null if none. Used for hint arrows.
     */
    public float[] nearestPoorDirection(float cx, float cy) {
        float vcx = cx + 0.5f;
        float vcy = cy + 0.5f;
        Cell best = null;
        float bestDist = Float.MAX_VALUE;
        for (Cell c : cells.values()) {
            if (c.state() != POOR) {
                continue;
            }
            float ccx = (c.col + 0.5f) * CELL;
            float ccy = (c.row + 0.5f) * CELL;
            float d = (ccx - vcx) * (ccx - vcx) + (ccy - vcy) * (ccy - vcy);
            if (d < bestDist) {
                bestDist = d;
                best = c;
            }
        }
        if (best == null) {
            return null;
        }
        return new float[]{(best.col + 0.5f) * CELL - vcx, (best.row + 0.5f) * CELL - vcy};
    }

    public void reset() {
        cells.clear();
        hasBounds = false;
        minCol = maxCol = minRow = maxRow = 0;
    }
}

package com.utake.skufind;

import androidx.camera.core.ImageProxy;

import java.nio.ByteBuffer;

/**
 * A camera frame as a packed ARGB int array. We run CameraX ImageAnalysis in
 * RGBA_8888 output mode, so the first (only) plane is already RGBA — no YUV
 * conversion needed. Both the coverage tracker (via {@link LumaFrame#fromArgb})
 * and the TFLite detector consume this.
 */
public final class RgbFrame {
    public final int[] argb;
    public final int width;
    public final int height;

    private RgbFrame(int[] argb, int width, int height) {
        this.argb = argb;
        this.width = width;
        this.height = height;
    }

    public static RgbFrame from(ImageProxy proxy) {
        ImageProxy.PlaneProxy plane = proxy.getPlanes()[0];
        ByteBuffer buffer = plane.getBuffer();
        int rowStride = plane.getRowStride();
        int pixelStride = plane.getPixelStride(); // 4 for RGBA_8888
        int w = proxy.getWidth();
        int h = proxy.getHeight();
        if (w <= 0 || h <= 0) {
            return null;
        }
        int[] argb = new int[w * h];
        byte[] row = new byte[rowStride];
        for (int y = 0; y < h; y++) {
            int pos = y * rowStride;
            buffer.position(pos);
            int toRead = Math.min(w * pixelStride, buffer.remaining());
            buffer.get(row, 0, toRead);
            int base = y * w;
            for (int x = 0; x < w; x++) {
                int i = x * pixelStride;
                int r = row[i] & 0xff;
                int g = row[i + 1] & 0xff;
                int b = row[i + 2] & 0xff;
                argb[base + x] = 0xff000000 | (r << 16) | (g << 8) | b;
            }
        }
        return new RgbFrame(argb, w, h);
    }

    public int get(int x, int y) {
        return argb[y * width + x];
    }
}

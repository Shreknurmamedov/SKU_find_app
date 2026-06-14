package com.utake.skufind;

import android.graphics.ImageFormat;
import android.media.Image;

import androidx.annotation.OptIn;
import androidx.camera.core.ExperimentalGetImage;
import androidx.camera.core.ImageProxy;

import java.nio.ByteBuffer;

/**
 * Small downscaled greyscale copy of a camera frame's luminance (Y) plane.
 *
 * <p>CameraX delivers YUV_420_888, whose first plane is luminance, so we get a
 * grey image for free (no colour conversion). We downscale to a tiny buffer so
 * the per-frame sharpness and motion estimation stay cheap. Orientation is left
 * in sensor space; callers rotate motion vectors afterwards.
 */
public final class LumaFrame {
    public final int width;
    public final int height;
    public final byte[] data; // width*height, unsigned via & 0xff

    private LumaFrame(int width, int height, byte[] data) {
        this.width = width;
        this.height = height;
        this.data = data;
    }

    public int get(int x, int y) {
        return data[y * width + x] & 0xff;
    }

    /** Build a downscaled luma frame whose longer side is about targetLongSide. */
    @OptIn(markerClass = ExperimentalGetImage.class)
    public static LumaFrame from(ImageProxy proxy, int targetLongSide) {
        Image image = proxy.getImage();
        if (image == null || image.getFormat() != ImageFormat.YUV_420_888) {
            return null;
        }
        Image.Plane yPlane = image.getPlanes()[0];
        ByteBuffer buffer = yPlane.getBuffer();
        int rowStride = yPlane.getRowStride();
        int pixelStride = yPlane.getPixelStride();
        int srcW = proxy.getWidth();
        int srcH = proxy.getHeight();
        if (srcW <= 0 || srcH <= 0) {
            return null;
        }

        int dstW, dstH;
        if (srcW >= srcH) {
            dstW = Math.max(1, targetLongSide);
            dstH = Math.max(1, Math.round(srcH * (dstW / (float) srcW)));
        } else {
            dstH = Math.max(1, targetLongSide);
            dstW = Math.max(1, Math.round(srcW * (dstH / (float) srcH)));
        }

        byte[] out = new byte[dstW * dstH];
        for (int y = 0; y < dstH; y++) {
            int srcY = Math.min(srcH - 1, y * srcH / dstH);
            int rowBase = srcY * rowStride;
            for (int x = 0; x < dstW; x++) {
                int srcX = Math.min(srcW - 1, x * srcW / dstW);
                out[y * dstW + x] = buffer.get(rowBase + srcX * pixelStride);
            }
        }
        return new LumaFrame(dstW, dstH, out);
    }
}

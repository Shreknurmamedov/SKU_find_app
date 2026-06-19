package com.utake.skufind;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * Tiny binary classifier that filters YOLO live detections on the tablet.
 *
 * Class order is fixed by the training ImageFolder layout:
 *   0 = interior, 1 = product
 *
 * If the asset is missing, the guard stays disabled and MainActivity falls back
 * to detector+heuristics. Full SKU recognition still happens server-side.
 */
public class TFLiteProductGuard {
    private static final String TAG = "SKUProductGuard";
    private static final String MODEL = "models/product_guard_cls_float32.tflite";
    private static final int PRODUCT_CLASS_INDEX = 1;

    private Interpreter interpreter;
    private int inputSize = 224;
    private boolean channelsLast = true;
    private int outputClasses = 2;
    private String error;

    public TFLiteProductGuard(Context context) {
        try {
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(2);
            interpreter = new Interpreter(loadModel(context), options);
            int[] in = interpreter.getInputTensor(0).shape();
            if (in.length == 4) {
                channelsLast = in[3] == 3;
                inputSize = channelsLast ? in[1] : in[2];
            }
            int[] out = interpreter.getOutputTensor(0).shape();
            outputClasses = out[out.length - 1];
            Log.i(TAG, "Loaded " + MODEL + " input=" + inputSize
                    + " channelsLast=" + channelsLast + " classes=" + outputClasses);
        } catch (Exception e) {
            interpreter = null;
            error = e.getMessage();
            Log.e(TAG, "Failed to load " + MODEL + ": " + error, e);
        }
    }

    public boolean isReady() {
        return interpreter != null;
    }

    public String error() {
        return error;
    }

    public String statusLabel() {
        if (isReady()) {
            return "guard:on";
        }
        return "guard:off" + (error == null ? "" : " (" + error + ")");
    }

    public float productProbability(RgbFrame frame, int left, int top, int right, int bottom) {
        if (interpreter == null || frame == null) {
            return 1f;
        }
        left = Math.max(0, Math.min(frame.width - 1, left));
        top = Math.max(0, Math.min(frame.height - 1, top));
        right = Math.max(left + 1, Math.min(frame.width, right));
        bottom = Math.max(top + 1, Math.min(frame.height, bottom));
        try {
            float[] raw = run(frame, left, top, right, bottom);
            if (raw == null || raw.length <= PRODUCT_CLASS_INDEX) {
                return 0f;
            }
            return probabilityAt(raw, PRODUCT_CLASS_INDEX);
        } catch (Exception e) {
            error = e.getMessage();
            Log.e(TAG, "Guard inference failed: " + error, e);
            return 0f;
        }
    }

    private MappedByteBuffer loadModel(Context ctx) throws Exception {
        try (AssetFileDescriptor fd = ctx.getAssets().openFd(MODEL);
             FileInputStream is = new FileInputStream(fd.getFileDescriptor())) {
            FileChannel channel = is.getChannel();
            return channel.map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getDeclaredLength());
        }
    }

    private float[] run(RgbFrame frame, int left, int top, int right, int bottom) {
        if (channelsLast) {
            float[][][][] input = new float[1][inputSize][inputSize][3];
            fillChannelsLast(input, frame, left, top, right, bottom);
            float[][] output = new float[1][outputClasses];
            interpreter.run(input, output);
            return output[0];
        }
        float[][][][] input = new float[1][3][inputSize][inputSize];
        fillChannelsFirst(input, frame, left, top, right, bottom);
        float[][] output = new float[1][outputClasses];
        interpreter.run(input, output);
        return output[0];
    }

    private void fillChannelsLast(float[][][][] input, RgbFrame frame,
                                  int left, int top, int right, int bottom) {
        int rw = Math.max(1, right - left);
        int rh = Math.max(1, bottom - top);
        for (int y = 0; y < inputSize; y++) {
            int sy = top + Math.min(rh - 1, y * rh / inputSize);
            int rowBase = sy * frame.width;
            for (int x = 0; x < inputSize; x++) {
                int sx = left + Math.min(rw - 1, x * rw / inputSize);
                int p = frame.argb[rowBase + sx];
                input[0][y][x][0] = ((p >> 16) & 0xff) / 255f;
                input[0][y][x][1] = ((p >> 8) & 0xff) / 255f;
                input[0][y][x][2] = (p & 0xff) / 255f;
            }
        }
    }

    private void fillChannelsFirst(float[][][][] input, RgbFrame frame,
                                   int left, int top, int right, int bottom) {
        int rw = Math.max(1, right - left);
        int rh = Math.max(1, bottom - top);
        for (int y = 0; y < inputSize; y++) {
            int sy = top + Math.min(rh - 1, y * rh / inputSize);
            int rowBase = sy * frame.width;
            for (int x = 0; x < inputSize; x++) {
                int sx = left + Math.min(rw - 1, x * rw / inputSize);
                int p = frame.argb[rowBase + sx];
                input[0][0][y][x] = ((p >> 16) & 0xff) / 255f;
                input[0][1][y][x] = ((p >> 8) & 0xff) / 255f;
                input[0][2][y][x] = (p & 0xff) / 255f;
            }
        }
    }

    private static float probabilityAt(float[] raw, int index) {
        float sum = 0f;
        boolean looksProb = true;
        for (float v : raw) {
            if (v < -0.001f || v > 1.001f) {
                looksProb = false;
                break;
            }
            sum += v;
        }
        if (looksProb && sum > 0.8f && sum < 1.2f) {
            return raw[index];
        }
        float max = raw[0];
        for (float v : raw) {
            max = Math.max(max, v);
        }
        double expSum = 0.0;
        for (float v : raw) {
            expSum += Math.exp(v - max);
        }
        return (float) (Math.exp(raw[index] - max) / Math.max(expSum, 1e-9));
    }
}

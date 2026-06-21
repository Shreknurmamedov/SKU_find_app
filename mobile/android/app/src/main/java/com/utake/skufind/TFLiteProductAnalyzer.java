package com.utake.skufind;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.RectF;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

/**
 * On-device product detector (TFLite) for the live screen. It finds every
 * product box per frame and colours it by capture quality: a box that is sharp
 * enough is "OK" (green), a blurry one is "переснять" (red), so the manager
 * knows which items to re-shoot. This is detection + quality only — full SKU
 * recognition (OCR/catalog) still happens server-side after upload.
 */
public class TFLiteProductAnalyzer {

    private static final String MODEL = "models/product_det_v2_float32.tflite";
    private static final int INPUT = 320;
    // Reverted to the v6 model (v7 hard-negatives over-suppressed real products:
    // a clear Huter box scored only ~0.23-0.45 on v7 vs ~0.56-0.63 on v6).
    // The backend can stay recall-first; live feedback must be conservative
    // because false boxes become "captured" noise for the manager.
    private static final float CONF_THRESHOLD = 0.60f;
    private static final float GREEN_CONF = 0.70f;
    private static final float IOU_THRESHOLD = 0.5f;
    private static final int MAX_DETECTIONS = 100;
    // Drop only near-full-frame boxes (walls/cabinets) and specks. 0.6 keeps big
    // close-up product boxes (e.g. a Ресанта carton) that 0.35 was wrongly cutting.
    private static final float MAX_AREA_FRAC = 0.60f;
    private static final float MIN_AREA_FRAC = 0.004f;

    private Interpreter interpreter;
    private int outD1, outD2;     // output shape [1, outD1, outD2]
    private boolean featFirst;    // layout [1, feat, anchors] vs [1, anchors, feat]
    private String error;

    public TFLiteProductAnalyzer(Context context) {
        try {
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(2);
            interpreter = new Interpreter(loadModel(context), options);
            int[] shape = interpreter.getOutputTensor(0).shape(); // [1, d1, d2]
            outD1 = shape[1];
            outD2 = shape[2];
            featFirst = outD1 < outD2; // 5 features < ~2100 anchors
        } catch (Exception e) {
            interpreter = null;
            error = e.getMessage();
        }
    }

    public boolean isReady() {
        return interpreter != null;
    }

    public String error() {
        return error;
    }

    private MappedByteBuffer loadModel(Context ctx) throws Exception {
        AssetFileDescriptor fd = ctx.getAssets().openFd(MODEL);
        try (FileInputStream is = new FileInputStream(fd.getFileDescriptor())) {
            FileChannel channel = is.getChannel();
            return channel.map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getDeclaredLength());
        }
    }

    public List<ProductDetection> analyze(RgbFrame frame) {
        List<ProductDetection> result = new ArrayList<>();
        if (interpreter == null || frame == null) {
            return result;
        }
        int w = frame.width;
        int h = frame.height;
        float scale = Math.min(INPUT / (float) w, INPUT / (float) h);
        int newW = Math.round(w * scale);
        int newH = Math.round(h * scale);
        int padX = (INPUT - newW) / 2;
        int padY = (INPUT - newH) / 2;

        float[][][][] input = new float[1][INPUT][INPUT][3];
        for (int y = 0; y < INPUT; y++) {
            int sy = (int) ((y - padY) / scale);
            for (int x = 0; x < INPUT; x++) {
                int sx = (int) ((x - padX) / scale);
                if (sx >= 0 && sx < w && sy >= 0 && sy < h) {
                    int p = frame.argb[sy * w + sx];
                    input[0][y][x][0] = ((p >> 16) & 0xff) / 255f;
                    input[0][y][x][1] = ((p >> 8) & 0xff) / 255f;
                    input[0][y][x][2] = (p & 0xff) / 255f;
                } else {
                    input[0][y][x][0] = 0.5f;
                    input[0][y][x][1] = 0.5f;
                    input[0][y][x][2] = 0.5f;
                }
            }
        }

        float[][][] output = new float[1][outD1][outD2];
        interpreter.run(input, output);

        int anchors = featFirst ? outD2 : outD1;
        List<float[]> boxes = new ArrayList<>();
        for (int a = 0; a < anchors; a++) {
            float cx, cy, bw, bh, conf;
            if (featFirst) {
                cx = output[0][0][a]; cy = output[0][1][a];
                bw = output[0][2][a]; bh = output[0][3][a]; conf = output[0][4][a];
            } else {
                cx = output[0][a][0]; cy = output[0][a][1];
                bw = output[0][a][2]; bh = output[0][a][3]; conf = output[0][a][4];
            }
            if (conf < CONF_THRESHOLD) {
                continue;
            }
            // ultralytics TFLite boxes are normalized 0..1 of the input; guard for pixel output
            float pcx, pcy, pw, ph;
            if (cx <= 1.5f) {
                pcx = cx * INPUT; pcy = cy * INPUT; pw = bw * INPUT; ph = bh * INPUT;
            } else {
                pcx = cx; pcy = cy; pw = bw; ph = bh;
            }
            float x1 = (pcx - pw / 2 - padX) / scale;
            float y1 = (pcy - ph / 2 - padY) / scale;
            float x2 = (pcx + pw / 2 - padX) / scale;
            float y2 = (pcy + ph / 2 - padY) / scale;
            float nx1 = clamp01(x1 / w), ny1 = clamp01(y1 / h);
            float nx2 = clamp01(x2 / w), ny2 = clamp01(y2 / h);
            float bwN = nx2 - nx1, bhN = ny2 - ny1;
            if (bwN < 0.01f || bhN < 0.01f) {
                continue;
            }
            float areaFrac = bwN * bhN;
            if (areaFrac > MAX_AREA_FRAC || areaFrac < MIN_AREA_FRAC) {
                continue; // furniture/wall (too big) or noise (too small)
            }
            boxes.add(new float[]{nx1, ny1, nx2, ny2, conf});
        }

        List<float[]> kept = nms(boxes, IOU_THRESHOLD);
        int count = 0;
        for (float[] b : kept) {
            if (count++ >= MAX_DETECTIONS) {
                break;
            }
            // green = confidently a SKU, red = detected but uncertain / poorly recognized
            boolean confident = b[4] >= GREEN_CONF;
            result.add(new ProductDetection(
                    new RectF(b[0], b[1], b[2], b[3]),
                    confident,
                    confident ? "Товар" : "проверить",
                    b[4]));
        }
        return result;
    }

    private static float clamp01(float v) {
        return v < 0 ? 0 : (v > 1 ? 1 : v);
    }

    private List<float[]> nms(List<float[]> boxes, float iouThr) {
        boxes.sort((a, b) -> Float.compare(b[4], a[4]));
        List<float[]> kept = new ArrayList<>();
        boolean[] removed = new boolean[boxes.size()];
        for (int i = 0; i < boxes.size(); i++) {
            if (removed[i]) {
                continue;
            }
            float[] bi = boxes.get(i);
            kept.add(bi);
            for (int j = i + 1; j < boxes.size(); j++) {
                if (!removed[j] && iou(bi, boxes.get(j)) > iouThr) {
                    removed[j] = true;
                }
            }
        }
        return kept;
    }

    private float iou(float[] a, float[] b) {
        float xx1 = Math.max(a[0], b[0]);
        float yy1 = Math.max(a[1], b[1]);
        float xx2 = Math.min(a[2], b[2]);
        float yy2 = Math.min(a[3], b[3]);
        float iw = Math.max(0, xx2 - xx1);
        float ih = Math.max(0, yy2 - yy1);
        float inter = iw * ih;
        float areaA = (a[2] - a[0]) * (a[3] - a[1]);
        float areaB = (b[2] - b[0]) * (b[3] - b[1]);
        return inter / (areaA + areaB - inter + 1e-6f);
    }
}

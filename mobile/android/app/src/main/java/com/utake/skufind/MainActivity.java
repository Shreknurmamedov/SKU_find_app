package com.utake.skufind;

import android.Manifest;
import android.app.Activity;
import android.content.ClipData;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.OpenableColumns;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import androidx.activity.ComponentActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.video.FileOutputOptions;
import androidx.camera.video.Quality;
import androidx.camera.video.QualitySelector;
import androidx.camera.video.Recorder;
import androidx.camera.video.Recording;
import androidx.camera.video.VideoCapture;
import androidx.camera.video.VideoRecordEvent;
import androidx.camera.view.PreviewView;

import java.io.File;
import java.io.FileInputStream;

import com.google.common.util.concurrent.ListenableFuture;

import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends ComponentActivity {
    private static final int PICK_MEDIA_REQUEST = 1001;
    private static final int CAMERA_PERMISSION_REQUEST = 1002;
    private static final String PREFS = "sku_find_prefs";
    private static final String DEFAULT_SERVER_URL = "http://10.0.2.2:8088";

    private EditText serverUrlInput;
    private EditText storeNameInput;
    private TextView selectedFilesText;
    private TextView resultText;
    private TextView liveStatusText;
    private TextView feedbackText;
    private Button uploadButton;
    private Button scanButton;
    private Button recordButton;
    private Button markNotProductButton;
    private Button markProductButton;
    private PreviewView previewView;
    private ProductOverlayView overlayView;
    private CoverageOverlayView coverageView;

    private final List<Uri> selectedUris = new ArrayList<>();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final Executor mainExecutor = command -> mainHandler.post(command);
    private final DemoProductAnalyzer productAnalyzer = new DemoProductAnalyzer();
    private final LiveCaptureTracker liveCaptureTracker = new LiveCaptureTracker();
    private final List<FeedbackCandidate> feedbackBuffer = new ArrayList<>();
    private final List<File> pendingFeedbackFiles = new ArrayList<>();
    private TFLiteProductAnalyzer tfliteAnalyzer;
    private TFLiteProductGuard productGuard;
    private ExecutorService cameraExecutor;
    private ProcessCameraProvider cameraProvider;

    // video recording (Live button records, then upload to backend)
    private VideoCapture<Recorder> videoCapture;
    private Recording activeRecording;
    private File recordedFile;
    private volatile boolean isRecording = false;
    private static final float LIVE_SHARP_GOOD = 0.44f;
    private static final float LIVE_SHARP_BLUR = 0.34f;
    private static final float LIVE_MIN_AREA_READABLE = 0.026f;
    private static final float LIVE_MIN_SIDE_READABLE = 0.115f;
    private static final float LIVE_CAPTURE_CONF = 0.70f;
    // On-device wall/floor/ceiling rejection: a box that is BOTH near-colourless
    // (low chroma) AND near-flat (low luma contrast) is a blank surface, not a
    // product. Both must be low (AND) so a white/grey package with text/edges
    // still survives via its contrast. Conservative on purpose — tune per device.
    private static final float LIVE_BG_CHROMA_MAX = 22f; // mean(max-min) over RGB, 0..255
    private static final float LIVE_BG_STD_MAX = 15f;    // luma std, 0..255
    private static final float LIVE_MIN_EDGE_DENSITY = 0.035f;
    private static final float LIVE_LARGE_BOX_AREA = 0.12f;
    private static final float LIVE_LARGE_BOX_MIN_EDGE_DENSITY = 0.055f;
    private static final float LIVE_LOW_DETAIL_STD_MAX = 24f;
    private static final float LIVE_GUARD_DISPLAY_PRODUCT = 0.55f;
    private static final float LIVE_GUARD_CAPTURE_PRODUCT = 0.75f;
    private static final long FEEDBACK_BUFFER_MS = 8000L;
    private static final int FEEDBACK_MAX_CANDIDATES = 24;
    private static final int FEEDBACK_MAX_CROP_SIDE = 512;
    private int savedFeedbackNegative = 0;
    private int savedFeedbackPositive = 0;
    private long lastFeedbackUiRefreshMs = 0L;

    // ---- coverage scan mode ----
    private final ScanGrid scanGrid = new ScanGrid();
    private final MotionTracker motionTracker = new MotionTracker();
    private volatile boolean scanMode = false;
    // gyro orientation, updated on the sensor thread, sampled on the camera thread
    private SensorManager sensorManager;
    private Sensor rotationSensor;
    private volatile float gyroYaw = 0f;
    private volatile float gyroPitch = 0f;
    private volatile boolean gyroReady = false;
    private float prevYaw = 0f;
    private float prevPitch = 0f;
    private boolean havePrevAngles = false;
    private final float[] rotMatrix = new float[9];
    private final float[] orientation = new float[3];

    private final SensorEventListener sensorListener = new SensorEventListener() {
        @Override
        public void onSensorChanged(SensorEvent event) {
            SensorManager.getRotationMatrixFromVector(rotMatrix, event.values);
            SensorManager.getOrientation(rotMatrix, orientation);
            gyroYaw = orientation[0];   // azimuth
            gyroPitch = orientation[1]; // pitch
            gyroReady = true;
        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) { }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        cameraExecutor = Executors.newSingleThreadExecutor();
        tfliteAnalyzer = new TFLiteProductAnalyzer(this);
        productGuard = new TFLiteProductGuard(this);
        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        if (sensorManager != null) {
            rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);
        }
        setContentView(buildUi());
        ensureCamera();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (sensorManager != null && rotationSensor != null) {
            sensorManager.registerListener(sensorListener, rotationSensor,
                    SensorManager.SENSOR_DELAY_GAME);
        }
    }

    @Override
    protected void onPause() {
        if (sensorManager != null) {
            sensorManager.unregisterListener(sensorListener);
        }
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
        }
        if (cameraExecutor != null) {
            cameraExecutor.shutdown();
        }
        super.onDestroy();
    }

    private View buildUi() {
        SharedPreferences prefs = getSharedPreferences(PREFS, MODE_PRIVATE);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(0xFFFFFFFF);

        LinearLayout header = new LinearLayout(this);
        header.setOrientation(LinearLayout.VERTICAL);
        header.setPadding(dp(18), dp(14), dp(18), dp(10));
        root.addView(header);

        TextView title = new TextView(this);
        title.setText("SKU Find Live");
        title.setTextSize(24);
        title.setGravity(Gravity.START);
        title.setTextColor(0xFF111111);
        header.addView(title);

        liveStatusText = new TextView(this);
        liveStatusText.setText("Live camera: запуск...");
        liveStatusText.setTextSize(14);
        liveStatusText.setTextColor(0xFF555555);
        header.addView(liveStatusText);

        FrameLayout cameraFrame = new FrameLayout(this);
        cameraFrame.setBackgroundColor(0xFF111111);
        // Camera is the main surface: give it the bulk of the screen (weight 2)
        // so the manager actually sees what is highlighted; the control panel
        // below scrolls within the remaining third (weight 1). Without this the
        // WRAP_CONTENT controls ate all the height and squeezed the preview to a
        // thin strip in landscape.
        root.addView(cameraFrame, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                0,
                2f
        ));

        previewView = new PreviewView(this);
        previewView.setScaleType(PreviewView.ScaleType.FILL_CENTER);
        cameraFrame.addView(previewView, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
        ));

        overlayView = new ProductOverlayView(this);
        cameraFrame.addView(overlayView, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
        ));

        coverageView = new CoverageOverlayView(this);
        coverageView.setVisibility(View.GONE);
        cameraFrame.addView(coverageView, new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
        ));

        ScrollView controlsScroll = new ScrollView(this);
        root.addView(controlsScroll, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                0,
                1f
        ));

        LinearLayout controls = new LinearLayout(this);
        controls.setOrientation(LinearLayout.VERTICAL);
        controls.setPadding(dp(18), dp(12), dp(18), dp(18));
        controlsScroll.addView(controls);

        LinearLayout cameraButtons = new LinearLayout(this);
        cameraButtons.setOrientation(LinearLayout.HORIZONTAL);
        controls.addView(cameraButtons, matchWrap());

        recordButton = new Button(this);
        recordButton.setText("● Запись");
        recordButton.setOnClickListener(view -> toggleRecording());
        cameraButtons.addView(recordButton, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));

        Button stopButton = new Button(this);
        stopButton.setText("Stop камеру");
        stopButton.setOnClickListener(view -> stopCamera());
        cameraButtons.addView(stopButton, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));

        Button resetLiveButton = new Button(this);
        resetLiveButton.setText("Сброс снятого");
        resetLiveButton.setOnClickListener(view -> resetLiveCapture());
        cameraButtons.addView(resetLiveButton, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));

        LinearLayout scanButtons = new LinearLayout(this);
        scanButtons.setOrientation(LinearLayout.HORIZONTAL);
        controls.addView(scanButtons, matchWrap());

        scanButton = new Button(this);
        scanButton.setText("Сканировать покрытие");
        scanButton.setOnClickListener(view -> toggleScanMode());
        scanButtons.addView(scanButton, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 2f));

        Button resetButton = new Button(this);
        resetButton.setText("Сброс сетки");
        resetButton.setOnClickListener(view -> resetScan());
        scanButtons.addView(resetButton, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));

        LinearLayout feedbackButtons = new LinearLayout(this);
        feedbackButtons.setOrientation(LinearLayout.HORIZONTAL);
        controls.addView(feedbackButtons, matchWrap());

        markNotProductButton = new Button(this);
        markNotProductButton.setText("НЕ ТОВАР");
        markNotProductButton.setEnabled(false);
        markNotProductButton.setOnClickListener(view -> saveBufferedFeedback(false));
        feedbackButtons.addView(markNotProductButton,
                new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));

        markProductButton = new Button(this);
        markProductButton.setText("ЭТО ТОВАР");
        markProductButton.setEnabled(false);
        markProductButton.setOnClickListener(view -> saveBufferedFeedback(true));
        feedbackButtons.addView(markProductButton,
                new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));

        feedbackText = new TextView(this);
        feedbackText.setText("Буфер разметки пуст");
        feedbackText.setTextSize(13);
        feedbackText.setTextColor(0xFF333333);
        feedbackText.setPadding(0, 0, 0, dp(8));
        controls.addView(feedbackText);

        serverUrlInput = new EditText(this);
        serverUrlInput.setSingleLine(true);
        serverUrlInput.setHint("Backend URL");
        serverUrlInput.setText(prefs.getString("server_url", DEFAULT_SERVER_URL));
        controls.addView(serverUrlInput, matchWrap());

        storeNameInput = new EditText(this);
        storeNameInput.setSingleLine(true);
        storeNameInput.setHint("Название ТТ");
        storeNameInput.setText(prefs.getString("store_name", ""));
        controls.addView(storeNameInput, matchWrap());

        Button pickButton = new Button(this);
        pickButton.setText("Выбрать фото/видео");
        pickButton.setOnClickListener(view -> openPicker());
        controls.addView(pickButton, matchWrap());

        uploadButton = new Button(this);
        uploadButton.setText("Отправить на backend");
        uploadButton.setEnabled(false);
        uploadButton.setOnClickListener(view -> uploadSelectedFiles());
        controls.addView(uploadButton, matchWrap());

        selectedFilesText = new TextView(this);
        selectedFilesText.setText("Файлы не выбраны");
        selectedFilesText.setTextSize(14);
        selectedFilesText.setTextColor(0xFF333333);
        selectedFilesText.setPadding(0, dp(10), 0, dp(8));
        controls.addView(selectedFilesText);

        resultText = new TextView(this);
        resultText.setText("Live: зелёный = уверенно товар, красный = неуверенно (проверить), "
                + "слабые скрыты. Бренд/модель (Ресанта, Huter...) — после отправки видео на backend.");
        resultText.setTextSize(14);
        resultText.setTextColor(0xFF111111);
        resultText.setPadding(0, dp(8), 0, 0);
        controls.addView(resultText);

        return root;
    }

    private void ensureCamera() {
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST
                && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            liveStatusText.setText("Camera permission is required for live recognition.");
        }
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                bindCameraUseCases();
            } catch (Exception exception) {
                liveStatusText.setText("Camera error: " + exception.getMessage());
            }
        }, mainExecutor);
    }

    private void bindCameraUseCases() {
        if (cameraProvider == null) {
            return;
        }

        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        ImageAnalysis analysis = new ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build();
        analysis.setAnalyzer(cameraExecutor, image -> {
            RgbFrame frame = RgbFrame.from(image);
            if (frame != null) {
                int rotation = image.getImageInfo().getRotationDegrees();
                RgbFrame displayFrame = frame.rotated(rotation);
                if (scanMode) {
                    processScanFrame(displayFrame, 0);
                } else {
                    runDetection(displayFrame);
                }
            }
            image.close();
        });

        Recorder recorder = new Recorder.Builder()
                .setQualitySelector(QualitySelector.from(Quality.HD))
                .build();
        videoCapture = VideoCapture.withOutput(recorder);

        cameraProvider.unbindAll();
        try {
            cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA,
                    preview, analysis, videoCapture);
        } catch (Exception bindAll) {
            // Some devices can't run preview+analysis+video together; drop analysis.
            cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA,
                    preview, videoCapture);
        }
    }

    private void stopCamera() {
        if (isRecording && activeRecording != null) {
            activeRecording.stop();
            activeRecording = null;
            isRecording = false;
        }
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
        }
        overlayView.setCapturedMarks(new ArrayList<>());
        overlayView.setDetections(new ArrayList<>());
        overlayView.setLiveSummary(0, 0, 0, 0, 0, 0, "Камера остановлена");
        liveCaptureTracker.reset();
        clearFeedbackBuffer();
        liveStatusText.setText("Камера остановлена.");
    }

    /** Live button: start recording a video, stop to finish, then it's ready to upload. */
    private void toggleRecording() {
        if (videoCapture == null) {
            liveStatusText.setText("Камера ещё не готова, подождите");
            return;
        }
        if (isRecording) {
            if (activeRecording != null) {
                activeRecording.stop();
                activeRecording = null;
            }
            return;
        }
        liveCaptureTracker.reset();
        File dir = new File(getExternalFilesDir(null), "videos");
        if (!dir.exists()) {
            dir.mkdirs();
        }
        recordedFile = new File(dir, "SKU_" + System.currentTimeMillis() + ".mp4");
        FileOutputOptions options = new FileOutputOptions.Builder(recordedFile).build();
        activeRecording = videoCapture.getOutput()
                .prepareRecording(this, options)
                .start(mainExecutor, event -> {
                    if (event instanceof VideoRecordEvent.Start) {
                        isRecording = true;
                        recordButton.setText("■ Стоп запись");
                        liveStatusText.setText("● Идёт запись... ведите камеру по полкам");
                    } else if (event instanceof VideoRecordEvent.Finalize) {
                        isRecording = false;
                        recordButton.setText("● Запись");
                        VideoRecordEvent.Finalize fin = (VideoRecordEvent.Finalize) event;
                        if (fin.hasError()) {
                            liveStatusText.setText("Ошибка записи: " + fin.getError());
                        } else {
                            selectedUris.clear();
                            selectedUris.add(Uri.fromFile(recordedFile));
                            selectedFilesText.setText("Записано видео: " + recordedFile.getName());
                            uploadButton.setEnabled(true);
                            liveStatusText.setText("Видео записано — нажмите «Отправить на backend»");
                        }
                    }
                });
    }

    // ---- coverage scan mode ----

    private void toggleScanMode() {
        scanMode = !scanMode;
        if (scanMode) {
            resetScan();
            overlayView.setVisibility(View.GONE);
            coverageView.setVisibility(View.VISIBLE);
            scanButton.setText("Стоп сканирование");
            liveStatusText.setText("Сканирование: ведите камеру по полкам");
        } else {
            coverageView.setVisibility(View.GONE);
            overlayView.setVisibility(View.VISIBLE);
            scanButton.setText("Сканировать покрытие");
            liveStatusText.setText("Сканирование остановлено");
        }
    }

    private void resetScan() {
        scanGrid.reset();
        motionTracker.reset();
        liveCaptureTracker.reset();
        havePrevAngles = false;
        coverageView.setState(new ArrayList<>(), new ArrayList<>(), 0, 0, 0, 0,
                false, 0f, 0f, "", null, 0, 0, false);
    }

    private void runDetection(RgbFrame frame) {
        final List<ProductDetection> detections;
        final List<android.graphics.RectF> capturedMarks;
        final String statusText;
        if (tfliteAnalyzer != null && tfliteAnalyzer.isReady()) {
            List<ProductDetection> raw = tfliteAnalyzer.analyze(frame);
            List<ProductDetection> scored = scoreLiveQuality(frame, raw);
            LiveCaptureTracker.Result tracked = liveCaptureTracker.update(scored);
            detections = tracked.visibleDetections;
            bufferFeedbackCandidates(frame, tracked.currentDetections);
            capturedMarks = tracked.capturedBoxes;
            int blur = 0;
            int closer = 0;
            int aim = 0;
            int hold = 0;
            for (ProductDetection d : detections) {
                if (d.qualityState == ProductDetection.STATE_BLUR) {
                    blur++;
                } else if (d.qualityState == ProductDetection.STATE_FAR) {
                    closer++;
                } else if (d.qualityState == ProductDetection.STATE_UNCERTAIN) {
                    aim++;
                } else if (d.qualityState == ProductDetection.STATE_GOOD) {
                    hold++;
                }
            }
            String hint = liveHint(detections.size(), closer, blur, aim, hold,
                    tracked.capturedCount);
            statusText = "Осталось: " + detections.size()
                    + " · " + hint
                    + " · снято: " + tracked.capturedCount
                    + " · " + guardStatus();
            final int todo = detections.size();
            final int captured = tracked.capturedCount;
            final int retake = closer + blur + aim;
            final int closerCount = closer;
            final int blurCount = blur;
            final int aimCount = aim;
            final String hintFinal = hint;
            runOnUiThread(() -> overlayView.setLiveSummary(
                    todo, captured, retake, closerCount, blurCount, aimCount, hintFinal));
        } else {
            detections = productAnalyzer.analyze(null);
            capturedMarks = new ArrayList<>();
            statusText = "Демо-режим (модель не загрузилась)";
            clearFeedbackBuffer();
            runOnUiThread(() -> overlayView.setLiveSummary(
                    detections.size(), 0, detections.size(), 0, 0, detections.size(),
                    "Демо-режим: проверьте модель"));
        }
        runOnUiThread(() -> {
            overlayView.setCapturedMarks(capturedMarks);
            overlayView.setDetections(detections);
            liveStatusText.setText(statusText);
        });
    }

    private void bufferFeedbackCandidates(RgbFrame frame, List<ProductDetection> detections) {
        long now = System.currentTimeMillis();
        int count;
        long ageMs;
        synchronized (feedbackBuffer) {
            pruneFeedbackBufferLocked(now);
            for (ProductDetection detection : detections) {
                FeedbackCandidate candidate = makeFeedbackCandidate(frame, detection, now);
                if (candidate != null) {
                    feedbackBuffer.add(candidate);
                }
            }
            while (feedbackBuffer.size() > FEEDBACK_MAX_CANDIDATES) {
                feedbackBuffer.remove(0);
            }
            count = feedbackBuffer.size();
            ageMs = count == 0 ? 0 : now - feedbackBuffer.get(count - 1).timeMs;
        }
        if (now - lastFeedbackUiRefreshMs > 500L) {
            lastFeedbackUiRefreshMs = now;
            refreshFeedbackUi(count, ageMs);
        }
    }

    private FeedbackCandidate makeFeedbackCandidate(RgbFrame frame, ProductDetection detection,
                                                    long timeMs) {
        RectPixels px = toPixels(detection.normalizedBounds, frame.width, frame.height);
        int width = px.right - px.left;
        int height = px.bottom - px.top;
        if (width < 8 || height < 8) {
            return null;
        }
        int outWidth = width;
        int outHeight = height;
        int maxSide = Math.max(width, height);
        if (maxSide > FEEDBACK_MAX_CROP_SIDE) {
            float scale = FEEDBACK_MAX_CROP_SIDE / (float) maxSide;
            outWidth = Math.max(1, Math.round(width * scale));
            outHeight = Math.max(1, Math.round(height * scale));
        }
        int[] crop = new int[outWidth * outHeight];
        for (int y = 0; y < outHeight; y++) {
            int srcY = px.top + Math.min(height - 1, y * height / outHeight);
            int srcRow = srcY * frame.width;
            int dstRow = y * outWidth;
            for (int x = 0; x < outWidth; x++) {
                int srcX = px.left + Math.min(width - 1, x * width / outWidth);
                crop[dstRow + x] = frame.argb[srcRow + srcX];
            }
        }
        return new FeedbackCandidate(crop, outWidth, outHeight, new RectF(detection.normalizedBounds),
                detection.confidence, detection.productness, detection.sharpness,
                detection.areaFraction, detection.qualityState, timeMs);
    }

    private void pruneFeedbackBufferLocked(long now) {
        for (int i = feedbackBuffer.size() - 1; i >= 0; i--) {
            if (now - feedbackBuffer.get(i).timeMs > FEEDBACK_BUFFER_MS) {
                feedbackBuffer.remove(i);
            }
        }
    }

    private void clearFeedbackBuffer() {
        boolean hadCandidates;
        synchronized (feedbackBuffer) {
            hadCandidates = !feedbackBuffer.isEmpty();
            feedbackBuffer.clear();
        }
        if (hadCandidates) {
            refreshFeedbackUi(0, 0);
        }
    }

    private FeedbackCandidate latestFeedbackCandidate() {
        long now = System.currentTimeMillis();
        synchronized (feedbackBuffer) {
            pruneFeedbackBufferLocked(now);
            if (feedbackBuffer.isEmpty()) {
                return null;
            }
            long newest = feedbackBuffer.get(feedbackBuffer.size() - 1).timeMs;
            FeedbackCandidate best = null;
            float bestScore = -1f;
            for (FeedbackCandidate candidate : feedbackBuffer) {
                if (newest - candidate.timeMs > 1400L) {
                    continue;
                }
                float score = candidate.rankScore();
                if (best == null || score > bestScore) {
                    best = candidate;
                    bestScore = score;
                }
            }
            return best == null ? feedbackBuffer.get(feedbackBuffer.size() - 1) : best;
        }
    }

    private void saveBufferedFeedback(boolean product) {
        FeedbackCandidate candidate = latestFeedbackCandidate();
        if (candidate == null) {
            refreshFeedbackUi(0, 0);
            return;
        }
        try {
            SavedFeedback saved = writeFeedbackCandidate(candidate, product);
            synchronized (pendingFeedbackFiles) {
                pendingFeedbackFiles.add(saved.image);
                pendingFeedbackFiles.add(saved.sidecar);
            }
            if (product) {
                savedFeedbackPositive++;
            } else {
                savedFeedbackNegative++;
            }
            synchronized (feedbackBuffer) {
                feedbackBuffer.clear();
            }
            markNotProductButton.setEnabled(false);
            markProductButton.setEnabled(false);
            String label = product ? "товар" : "не товар";
            feedbackText.setText("Сохранено: " + label
                    + "\nВсего: не товар " + savedFeedbackNegative
                    + ", товар " + savedFeedbackPositive);
            resultText.setText("Feedback сохранён: " + saved.image.getParentFile().getName()
                    + "/" + saved.image.getName());
        } catch (Exception exception) {
            feedbackText.setText("Не удалось сохранить feedback: " + exception.getMessage());
        }
    }

    private SavedFeedback writeFeedbackCandidate(FeedbackCandidate candidate, boolean product)
            throws Exception {
        File baseDir = getExternalFilesDir(null);
        if (baseDir == null) {
            baseDir = getFilesDir();
        }
        File dir = new File(baseDir, product ? "feedback/product" : "feedback/hard_negative");
        if (!dir.exists() && !dir.mkdirs()) {
            throw new IllegalStateException("cannot create " + dir.getAbsolutePath());
        }

        String stem = String.format(Locale.US, "%s_%d_c%.2f_g%.2f",
                product ? "product" : "hardneg",
                candidate.timeMs,
                candidate.confidence,
                candidate.productness);
        File image = new File(dir, stem + ".jpg");
        Bitmap bitmap = Bitmap.createBitmap(candidate.argb, candidate.width, candidate.height,
                Bitmap.Config.ARGB_8888);
        try (FileOutputStream output = new FileOutputStream(image)) {
            if (!bitmap.compress(Bitmap.CompressFormat.JPEG, 92, output)) {
                throw new IllegalStateException("jpeg encode failed");
            }
        } finally {
            bitmap.recycle();
        }

        JSONObject meta = new JSONObject();
        meta.put("label", product ? "product" : "hard_negative");
        meta.put("timestamp_ms", candidate.timeMs);
        meta.put("detector_confidence", candidate.confidence);
        meta.put("guard_productness", candidate.productness);
        meta.put("sharpness", candidate.sharpness);
        meta.put("area_fraction", candidate.areaFraction);
        meta.put("quality_state", candidate.qualityState);
        meta.put("bbox_left", candidate.bounds.left);
        meta.put("bbox_top", candidate.bounds.top);
        meta.put("bbox_right", candidate.bounds.right);
        meta.put("bbox_bottom", candidate.bounds.bottom);
        meta.put("store_name", storeNameInput == null ? "" : storeNameInput.getText().toString());
        meta.put("guard_status", guardStatus());
        File sidecar = new File(dir, stem + ".json");
        try (FileOutputStream output = new FileOutputStream(sidecar)) {
            output.write(meta.toString(2).getBytes(StandardCharsets.UTF_8));
        }
        return new SavedFeedback(image, sidecar, product);
    }

    private void refreshFeedbackUi(int count, long newestAgeMs) {
        runOnUiThread(() -> {
            boolean hasCandidate = count > 0;
            if (markNotProductButton != null) {
                markNotProductButton.setEnabled(hasCandidate);
            }
            if (markProductButton != null) {
                markProductButton.setEnabled(hasCandidate);
            }
            if (feedbackText != null) {
                if (hasCandidate) {
                    feedbackText.setText(String.format(Locale.US,
                            "Последняя рамка: %.1fс · буфер %d · сохранено %d/%d",
                            newestAgeMs / 1000f,
                            count,
                            savedFeedbackNegative,
                            savedFeedbackPositive));
                } else {
                    feedbackText.setText("Буфер разметки пуст"
                            + " · сохранено " + savedFeedbackNegative + "/" + savedFeedbackPositive);
                }
            }
        });
    }

    private String guardStatus() {
        return productGuard == null ? "guard:missing" : productGuard.statusLabel();
    }

    private List<ProductDetection> scoreLiveQuality(RgbFrame frame, List<ProductDetection> raw) {
        List<ProductDetection> result = new ArrayList<>();
        for (ProductDetection d : raw) {
            RectPixels px = toPixels(d.normalizedBounds, frame.width, frame.height);
            // One luma read of the crop feeds both the sharpness score and the
            // re-id fingerprint, so the extra work per box stays cheap.
            LumaFrame luma = LumaFrame.fromArgbRegion(frame, px.left, px.top, px.right, px.bottom, 96);
            float boxW = Math.max(0f, d.normalizedBounds.width());
            float boxH = Math.max(0f, d.normalizedBounds.height());
            float area = boxW * boxH;
            float minSide = Math.min(boxW, boxH);
            float chroma = meanChroma(frame, px);
            float std = lumaStd(luma);
            float edgeDensity = edgeDensity(luma);
            // Drop blank wall/floor/ceiling boxes before they reach the manager:
            // colourless AND flat -> not a product.
            if (isLiveBackgroundLike(chroma, std, edgeDensity, area)) {
                continue;
            }
            float productness = 1f;
            if (productGuard != null && productGuard.isReady()) {
                productness = productGuard.productProbability(
                        frame, px.left, px.top, px.right, px.bottom);
                if (productness < LIVE_GUARD_DISPLAY_PRODUCT) {
                    continue;
                }
            }
            float sharp = FrameQuality.sharpScore(luma);
            boolean tooFar = area < LIVE_MIN_AREA_READABLE || minSide < LIVE_MIN_SIDE_READABLE;
            int state;
            boolean good;
            String label;
            if (tooFar) {
                state = ProductDetection.STATE_FAR;
                good = false;
                label = "ближе";
            } else if (sharp < LIVE_SHARP_BLUR) {
                state = ProductDetection.STATE_BLUR;
                good = false;
                label = "медленнее";
            } else if (!d.recognized
                    || d.confidence < LIVE_CAPTURE_CONF
                    || productness < LIVE_GUARD_CAPTURE_PRODUCT
                    || sharp < LIVE_SHARP_GOOD) {
                state = ProductDetection.STATE_UNCERTAIN;
                good = false;
                label = "наведите";
            } else {
                state = ProductDetection.STATE_GOOD;
                good = true;
                label = "держите";
            }
            ProductDetection scored = new ProductDetection(d.normalizedBounds, good, label,
                    d.confidence, state, sharp, area);
            scored.signature = signatureFromLuma(luma, 8);
            scored.productness = productness;
            result.add(scored);
        }
        return result;
    }

    /**
     * Mean-removed, L2-normalized grid×grid luma fingerprint of a crop, or null.
     * Mean-removal makes it brightness-invariant and L2-normalization
     * contrast-invariant, so the same product matches across lighting changes
     * while staying cheap (64 floats). Used by {@link LiveCaptureTracker} for re-id.
     */
    private static float[] signatureFromLuma(LumaFrame luma, int grid) {
        if (luma == null || luma.width < grid || luma.height < grid) {
            return null;
        }
        float[] sig = new float[grid * grid];
        for (int gy = 0; gy < grid; gy++) {
            int y0 = gy * luma.height / grid;
            int y1 = Math.max(y0 + 1, (gy + 1) * luma.height / grid);
            for (int gx = 0; gx < grid; gx++) {
                int x0 = gx * luma.width / grid;
                int x1 = Math.max(x0 + 1, (gx + 1) * luma.width / grid);
                int sum = 0;
                int cnt = 0;
                for (int y = y0; y < y1; y++) {
                    int row = y * luma.width;
                    for (int x = x0; x < x1; x++) {
                        sum += luma.data[row + x] & 0xff;
                        cnt++;
                    }
                }
                sig[gy * grid + gx] = cnt > 0 ? (float) sum / cnt : 0f;
            }
        }
        float mean = 0f;
        for (float v : sig) {
            mean += v;
        }
        mean /= sig.length;
        double sumSq = 0;
        for (int i = 0; i < sig.length; i++) {
            sig[i] -= mean;
            sumSq += (double) sig[i] * sig[i];
        }
        float norm = (float) Math.sqrt(sumSq);
        if (norm < 1e-6f) {
            return null; // flat patch (wall/shadow) -> no usable fingerprint
        }
        for (int i = 0; i < sig.length; i++) {
            sig[i] /= norm;
        }
        return sig;
    }

    private String liveHint(int todo, int closer, int blur, int aim, int hold, int captured) {
        if (todo == 0) {
            if (captured > 0) {
                return "Эта зона снята, ведите дальше";
            }
            return "Наведите камеру на полку";
        }
        if (closer > 0) {
            return "Подойдите ближе: товар мелкий для артикула";
        }
        if (blur > 0) {
            return "Медленнее: кадр смазан";
        }
        if (aim > 0) {
            return "Держите товар в центре";
        }
        if (hold > 0) {
            return "Держите 1 секунду";
        }
        return "Ведите камеру по полке";
    }

    private static boolean isLiveBackgroundLike(float chroma, float lumaStd,
                                                float edgeDensity, float area) {
        if (chroma < LIVE_BG_CHROMA_MAX && lumaStd < LIVE_BG_STD_MAX) {
            return true;
        }
        if (edgeDensity < LIVE_MIN_EDGE_DENSITY && lumaStd < LIVE_LOW_DETAIL_STD_MAX) {
            return true;
        }
        return area > LIVE_LARGE_BOX_AREA
                && edgeDensity < LIVE_LARGE_BOX_MIN_EDGE_DENSITY
                && lumaStd < LIVE_LOW_DETAIL_STD_MAX * 1.45f;
    }

    /** Mean colourfulness (max-min over RGB, 0..255) of a box region, subsampled. */
    private static float meanChroma(RgbFrame frame, RectPixels px) {
        int rw = px.right - px.left;
        int rh = px.bottom - px.top;
        if (rw < 2 || rh < 2) {
            return 255f; // too small to judge -> don't reject as background
        }
        int stepX = Math.max(1, rw / 24);
        int stepY = Math.max(1, rh / 24);
        long sum = 0;
        int n = 0;
        for (int y = px.top; y < px.bottom; y += stepY) {
            int row = y * frame.width;
            for (int x = px.left; x < px.right; x += stepX) {
                int p = frame.argb[row + x];
                int r = (p >> 16) & 0xff;
                int g = (p >> 8) & 0xff;
                int b = p & 0xff;
                int mx = Math.max(r, Math.max(g, b));
                int mn = Math.min(r, Math.min(g, b));
                sum += (mx - mn);
                n++;
            }
        }
        return n > 0 ? (float) sum / n : 255f;
    }

    /** Standard deviation of luma (0..255) — low means a flat, texture-less patch. */
    private static float lumaStd(LumaFrame luma) {
        if (luma == null || luma.data.length == 0) {
            return 255f;
        }
        byte[] d = luma.data;
        long sum = 0;
        long sumSq = 0;
        for (byte value : d) {
            int v = value & 0xff;
            sum += v;
            sumSq += (long) v * v;
        }
        double mean = (double) sum / d.length;
        double var = (double) sumSq / d.length - mean * mean;
        return (float) Math.sqrt(Math.max(0, var));
    }

    /** Fraction of crop pixels with a meaningful local edge. Low = smooth furniture/wall. */
    private static float edgeDensity(LumaFrame luma) {
        if (luma == null || luma.width < 3 || luma.height < 3) {
            return 0f;
        }
        int strong = 0;
        int total = 0;
        int w = luma.width;
        int h = luma.height;
        byte[] d = luma.data;
        for (int y = 1; y < h - 1; y++) {
            int row = y * w;
            for (int x = 1; x < w - 1; x++) {
                int i = row + x;
                int gx = Math.abs((d[i + 1] & 0xff) - (d[i - 1] & 0xff));
                int gy = Math.abs((d[i + w] & 0xff) - (d[i - w] & 0xff));
                if (gx + gy > 42) {
                    strong++;
                }
                total++;
            }
        }
        return total > 0 ? strong / (float) total : 0f;
    }

    private RectPixels toPixels(android.graphics.RectF r, int width, int height) {
        return new RectPixels(
                Math.max(0, Math.min(width - 1, Math.round(r.left * width))),
                Math.max(0, Math.min(height - 1, Math.round(r.top * height))),
                Math.max(0, Math.min(width, Math.round(r.right * width))),
                Math.max(0, Math.min(height, Math.round(r.bottom * height)))
        );
    }

    private void resetLiveCapture() {
        liveCaptureTracker.reset();
        clearFeedbackBuffer();
        overlayView.setCapturedMarks(new ArrayList<>());
        overlayView.setDetections(new ArrayList<>());
        overlayView.setLiveSummary(0, 0, 0, 0, 0, 0, "Наведите камеру на полку");
        liveStatusText.setText("Память live-съёмки сброшена.");
    }

    private static class FeedbackCandidate {
        final int[] argb;
        final int width;
        final int height;
        final RectF bounds;
        final float confidence;
        final float productness;
        final float sharpness;
        final float areaFraction;
        final int qualityState;
        final long timeMs;

        FeedbackCandidate(int[] argb, int width, int height, RectF bounds,
                          float confidence, float productness, float sharpness,
                          float areaFraction, int qualityState, long timeMs) {
            this.argb = argb;
            this.width = width;
            this.height = height;
            this.bounds = bounds;
            this.confidence = confidence;
            this.productness = productness;
            this.sharpness = sharpness;
            this.areaFraction = areaFraction;
            this.qualityState = qualityState;
            this.timeMs = timeMs;
        }

        float rankScore() {
            return confidence * 0.40f
                    + productness * 0.35f
                    + Math.min(1f, areaFraction * 5f) * 0.20f
                    + Math.min(1f, sharpness) * 0.05f;
        }
    }

    private static class SavedFeedback {
        final File image;
        final File sidecar;
        final boolean product;

        SavedFeedback(File image, File sidecar, boolean product) {
            this.image = image;
            this.sidecar = sidecar;
            this.product = product;
        }
    }

    private static class RectPixels {
        final int left;
        final int top;
        final int right;
        final int bottom;

        RectPixels(int left, int top, int right, int bottom) {
            this.left = left;
            this.top = top;
            this.right = right;
            this.bottom = bottom;
        }
    }

    private void processScanFrame(RgbFrame frame, int rotation) {
        LumaFrame luma = LumaFrame.fromArgb(frame, 80);
        if (luma == null) {
            return;
        }

        float dYaw = 0f;
        float dPitch = 0f;
        if (gyroReady) {
            if (havePrevAngles) {
                dYaw = angleDelta(prevYaw, gyroYaw);
                dPitch = angleDelta(prevPitch, gyroPitch);
            }
            prevYaw = gyroYaw;
            prevPitch = gyroPitch;
            havePrevAngles = true;
        }

        float sharp = FrameQuality.sharpScore(luma);
        motionTracker.update(luma, rotation, dYaw, dPitch);
        float cx = motionTracker.cx();
        float cy = motionTracker.cy();
        float speed = motionTracker.speed();

        boolean tooFast = speed > 0.18f;
        // Mark coverage; a cell only turns green once a sharp frame covers it.
        scanGrid.markView(cx, cy, sharp);

        int good = scanGrid.countGood();
        int poor = scanGrid.countPoor();
        float[] arrow = scanGrid.nearestPoorDirection(cx, cy);
        boolean blurNow = sharp < 0.40f || tooFast;

        String hint;
        if (tooFast) {
            hint = "Слишком быстро — медленнее";
        } else if (poor > 0) {
            hint = "Есть зоны для пересъёмки";
        } else {
            hint = "Ведите камеру по полкам";
        }

        // Build draw snapshots on this (camera) thread so the UI thread never
        // iterates the live grid map concurrently.
        final List<ScanGrid.Cell> viewCells = scanGrid.cellsInView(cx, cy);
        final List<ScanGrid.Cell> mapCells = scanGrid.allCells();
        final int minCol = scanGrid.minCol();
        final int maxCol = scanGrid.maxCol();
        final int minRow = scanGrid.minRow();
        final int maxRow = scanGrid.maxRow();
        final boolean hasBounds = scanGrid.hasBounds();
        final String hintFinal = hint;

        runOnUiThread(() -> {
            coverageView.setState(viewCells, mapCells, minCol, maxCol, minRow, maxRow,
                    hasBounds, cx, cy, hintFinal, arrow, good, poor, blurNow);
            liveStatusText.setText(String.format(Locale.US,
                    "Покрытие: OK %d, переснять %d, резкость %.0f%%", good, poor, sharp * 100f));
        });
    }

    /** Shortest signed angle from a to b, handling the -PI..PI wrap. */
    private float angleDelta(float a, float b) {
        float d = b - a;
        while (d > Math.PI) {
            d -= (float) (2 * Math.PI);
        }
        while (d < -Math.PI) {
            d += (float) (2 * Math.PI);
        }
        return d;
    }

    private void openPicker() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        intent.putExtra(Intent.EXTRA_MIME_TYPES, new String[]{"image/*", "video/*"});
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        intent.addFlags(Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
        startActivityForResult(intent, PICK_MEDIA_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != PICK_MEDIA_REQUEST || resultCode != Activity.RESULT_OK || data == null) {
            return;
        }

        selectedUris.clear();
        int flags = data.getFlags() & Intent.FLAG_GRANT_READ_URI_PERMISSION;
        ClipData clipData = data.getClipData();
        if (clipData != null) {
            for (int index = 0; index < clipData.getItemCount(); index++) {
                Uri uri = clipData.getItemAt(index).getUri();
                getContentResolver().takePersistableUriPermission(uri, flags);
                selectedUris.add(uri);
            }
        } else if (data.getData() != null) {
            Uri uri = data.getData();
            getContentResolver().takePersistableUriPermission(uri, flags);
            selectedUris.add(uri);
        }

        selectedFilesText.setText("Выбрано файлов: " + selectedUris.size());
        uploadButton.setEnabled(!selectedUris.isEmpty());
    }

    private void uploadSelectedFiles() {
        String serverUrl = cleanServerUrl(serverUrlInput.getText().toString());
        String storeName = storeNameInput.getText().toString().trim();
        getSharedPreferences(PREFS, MODE_PRIVATE)
                .edit()
                .putString("server_url", serverUrl)
                .putString("store_name", storeName)
                .apply();

        uploadButton.setEnabled(false);
        resultText.setText("Загрузка...");

        List<Uri> uploadUris = new ArrayList<>(selectedUris);
        List<File> uploadFeedbackFiles = pendingFeedbackSnapshot();
        new Thread(() -> {
            try {
                String response = uploadMultipart(
                        serverUrl + "/jobs/upload",
                        storeName,
                        uploadUris,
                        uploadFeedbackFiles);
                markFeedbackUploaded(uploadFeedbackFiles);
                String formatted = formatJobResponse(response);
                String jobId = new JSONObject(response).optString("job_id", "");
                runOnUiThread(() -> {
                    resultText.setText(formatted);
                    uploadButton.setEnabled(true);
                });
                if (!jobId.isEmpty()) {
                    pollSkuResult(serverUrl, jobId, formatted);
                }
            } catch (Exception exception) {
                runOnUiThread(() -> {
                    resultText.setText("Ошибка: " + exception.getMessage());
                    uploadButton.setEnabled(true);
                });
            }
        }).start();
    }

    /** Poll the job until SKU counting finishes, updating the result text. */
    private void pollSkuResult(String serverUrl, String jobId, String header) {
        new Thread(() -> {
            for (int attempt = 0; attempt < 240; attempt++) {
                try {
                    String body = httpGet(serverUrl + "/jobs/" + jobId);
                    JSONObject job = new JSONObject(body);
                    String status = job.optString("sku_status", "pending");
                    String note = job.optString("sku_note", "");
                    if ("done".equals(status)) {
                        JSONObject report = job.optJSONObject("sku_report");
                        String text = header + "\n\n=== Подсчёт SKU ===\n" + formatSkuReport(report);
                        runOnUiThread(() -> resultText.setText(text));
                        return;
                    }
                    if ("failed".equals(status) || "skipped".equals(status)) {
                        String text = header + "\n\nПодсчёт SKU: " + status
                                + (note.isEmpty() ? "" : "\n" + note);
                        runOnUiThread(() -> resultText.setText(text));
                        return;
                    }
                    String progress = "processing".equals(status)
                            ? "Идёт подсчёт SKU на сервере..." : "Подсчёт SKU в очереди...";
                    String shown = header + "\n\n" + progress
                            + (note.isEmpty() ? "" : "\n" + note);
                    runOnUiThread(() -> resultText.setText(shown));
                } catch (Exception ignored) {
                    // transient network error: keep polling
                }
                try {
                    Thread.sleep(3000);
                } catch (InterruptedException interrupted) {
                    return;
                }
            }
            runOnUiThread(() -> resultText.setText(header
                    + "\n\nПодсчёт SKU занимает дольше обычного. Проверьте отчёт на сервере позже."));
        }).start();
    }

    private String formatSkuReport(JSONObject report) {
        if (report == null) {
            return "Отчёт пуст.";
        }
        StringBuilder b = new StringBuilder();
        JSONObject totals = report.optJSONObject("totals");
        if (totals != null) {
            b.append("Объектов: ").append(totals.optInt("physical_objects")).append("\n");
            b.append("Наши бренды: ").append(totals.optInt("own_brand_objects")).append("\n");
            b.append("Уверенно SKU: ").append(totals.optInt("confident_sku")).append("\n");
            b.append("Бренд не виден (переснять): ")
                    .append(totals.optInt("brand_not_visible")).append("\n");
            b.append("На проверку: ").append(totals.optInt("needs_review")).append("\n");
        }
        appendCounts(b, "\nПо брендам:", report.optJSONObject("by_brand"));
        appendCounts(b, "\nПо группам:", report.optJSONObject("by_category"));
        appendCounts(b, "\nПо моделям:", report.optJSONObject("by_model"));
        return b.toString();
    }

    private void appendCounts(StringBuilder b, String title, JSONObject obj) {
        if (obj == null || obj.length() == 0) {
            return;
        }
        b.append(title).append("\n");
        java.util.Iterator<String> keys = obj.keys();
        while (keys.hasNext()) {
            String k = keys.next();
            b.append("  ").append(k).append(": ").append(obj.optInt(k)).append("\n");
        }
    }

    private List<File> pendingFeedbackSnapshot() {
        List<File> result = new ArrayList<>();
        synchronized (pendingFeedbackFiles) {
            for (File file : pendingFeedbackFiles) {
                if (file.exists() && file.isFile()) {
                    result.add(file);
                }
            }
        }
        return result;
    }

    private void markFeedbackUploaded(List<File> files) {
        if (files.isEmpty()) {
            return;
        }
        synchronized (pendingFeedbackFiles) {
            pendingFeedbackFiles.removeAll(files);
        }
    }

    private String httpGet(String urlString) throws Exception {
        HttpURLConnection connection = (HttpURLConnection) new URL(urlString).openConnection();
        connection.setRequestMethod("GET");
        connection.setConnectTimeout(10000);
        connection.setReadTimeout(20000);
        int status = connection.getResponseCode();
        InputStream input = status >= 200 && status < 300
                ? connection.getInputStream()
                : connection.getErrorStream();
        String body = readAll(input);
        if (status < 200 || status >= 300) {
            throw new IllegalStateException("HTTP " + status + ": " + body);
        }
        return body;
    }

    private String uploadMultipart(String targetUrl, String storeName, List<Uri> uris,
                                   List<File> feedbackFiles) throws Exception {
        String boundary = "SkuFindBoundary" + System.currentTimeMillis();
        HttpURLConnection connection = (HttpURLConnection) new URL(targetUrl).openConnection();
        connection.setRequestMethod("POST");
        connection.setDoOutput(true);
        connection.setConnectTimeout(30000);
        connection.setReadTimeout(600000); // large videos over Wi-Fi can take minutes
        connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);
        connection.setChunkedStreamingMode(1 << 20);

        try (OutputStream output = new BufferedOutputStream(connection.getOutputStream())) {
            writeFormField(output, boundary, "store_name", storeName);
            for (Uri uri : uris) {
                writeFileField(output, boundary, "files", uri);
            }
            for (File file : feedbackFiles) {
                writeFileField(output, boundary, "feedback_files", file);
            }
            writeString(output, "--" + boundary + "--\r\n");
        }

        int status = connection.getResponseCode();
        InputStream input = status >= 200 && status < 300
                ? connection.getInputStream()
                : connection.getErrorStream();
        String body = readAll(input);
        if (status < 200 || status >= 300) {
            throw new IllegalStateException("HTTP " + status + ": " + body);
        }
        return body;
    }

    private void writeFormField(OutputStream output, String boundary, String name, String value)
            throws Exception {
        writeString(output, "--" + boundary + "\r\n");
        writeString(output, "Content-Disposition: form-data; name=\"" + name + "\"\r\n\r\n");
        writeString(output, value == null ? "" : value);
        writeString(output, "\r\n");
    }

    private void writeFileField(OutputStream output, String boundary, String fieldName, Uri uri)
            throws Exception {
        String filename = getDisplayName(uri);
        String mimeType = getContentResolver().getType(uri);
        if (mimeType == null) {
            mimeType = "application/octet-stream";
        }

        writeString(output, "--" + boundary + "\r\n");
        writeString(
                output,
                "Content-Disposition: form-data; name=\"" + fieldName + "\"; filename=\""
                        + filename.replace("\"", "_") + "\"\r\n"
        );
        writeString(output, "Content-Type: " + mimeType + "\r\n\r\n");
        try (InputStream input = new BufferedInputStream(getContentResolver().openInputStream(uri))) {
            if (input == null) {
                throw new IllegalStateException("Cannot open " + filename);
            }
            byte[] buffer = new byte[64 * 1024];
            int read;
            while ((read = input.read(buffer)) != -1) {
                output.write(buffer, 0, read);
            }
        }
        writeString(output, "\r\n");
    }

    private void writeFileField(OutputStream output, String boundary, String fieldName, File file)
            throws Exception {
        String filename = file.getName();
        String lower = filename.toLowerCase(Locale.US);
        String mimeType;
        if (lower.endsWith(".json")) {
            mimeType = "application/json";
        } else if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) {
            mimeType = "image/jpeg";
        } else if (lower.endsWith(".png")) {
            mimeType = "image/png";
        } else {
            mimeType = "application/octet-stream";
        }

        writeString(output, "--" + boundary + "\r\n");
        writeString(
                output,
                "Content-Disposition: form-data; name=\"" + fieldName + "\"; filename=\""
                        + filename.replace("\"", "_") + "\"\r\n"
        );
        writeString(output, "Content-Type: " + mimeType + "\r\n\r\n");
        try (InputStream input = new BufferedInputStream(new FileInputStream(file))) {
            byte[] buffer = new byte[64 * 1024];
            int read;
            while ((read = input.read(buffer)) != -1) {
                output.write(buffer, 0, read);
            }
        }
        writeString(output, "\r\n");
    }

    private String getDisplayName(Uri uri) {
        ContentResolver resolver = getContentResolver();
        try (Cursor cursor = resolver.query(uri, null, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                int index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (index >= 0) {
                    return cursor.getString(index);
                }
            }
        }
        String fallback = uri.getLastPathSegment();
        return fallback == null ? "upload.bin" : fallback;
    }

    private String formatJobResponse(String response) throws Exception {
        JSONObject payload = new JSONObject(response);
        JSONObject summary = payload.getJSONObject("summary");
        StringBuilder builder = new StringBuilder();
        builder.append("Job: ").append(payload.getString("job_id")).append("\n");
        builder.append("Статус: ").append(payload.getString("status")).append("\n");
        builder.append("Файлов: ").append(summary.getInt("total_files")).append("\n");
        builder.append("Фото: ").append(summary.getInt("image_files")).append("\n");
        builder.append("Видео: ").append(summary.getInt("video_files")).append("\n");
        builder.append("OK: ").append(summary.getInt("quality_ok")).append("\n");
        builder.append("Предупреждения: ").append(summary.getInt("quality_warning")).append("\n");
        builder.append("Переснять: ").append(summary.getInt("quality_retake")).append("\n");
        builder.append("\n");
        builder.append("Backend принял файлы и проверил качество. Live overlay работает отдельно на CameraX.");
        return builder.toString();
    }

    private String readAll(InputStream input) throws Exception {
        if (input == null) {
            return "";
        }
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte[] buffer = new byte[8192];
        int read;
        while ((read = input.read(buffer)) != -1) {
            output.write(buffer, 0, read);
        }
        return output.toString(StandardCharsets.UTF_8.name());
    }

    private void writeString(OutputStream output, String value) throws Exception {
        output.write(value.getBytes(StandardCharsets.UTF_8));
    }

    private String cleanServerUrl(String value) {
        String cleaned = value == null ? "" : value.trim();
        while (cleaned.endsWith("/")) {
            cleaned = cleaned.substring(0, cleaned.length() - 1);
        }
        return cleaned.isEmpty() ? DEFAULT_SERVER_URL : cleaned;
    }

    private LinearLayout.LayoutParams matchWrap() {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        params.setMargins(0, dp(8), 0, 0);
        return params;
    }

    private int dp(int value) {
        return (int) (value * getResources().getDisplayMetrics().density + 0.5f);
    }
}

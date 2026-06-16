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

import com.google.common.util.concurrent.ListenableFuture;

import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
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
    private Button uploadButton;
    private Button scanButton;
    private Button recordButton;
    private PreviewView previewView;
    private ProductOverlayView overlayView;
    private CoverageOverlayView coverageView;

    private final List<Uri> selectedUris = new ArrayList<>();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final Executor mainExecutor = command -> mainHandler.post(command);
    private final DemoProductAnalyzer productAnalyzer = new DemoProductAnalyzer();
    private TFLiteProductAnalyzer tfliteAnalyzer;
    private ExecutorService cameraExecutor;
    private ProcessCameraProvider cameraProvider;

    // video recording (Live button records, then upload to backend)
    private VideoCapture<Recorder> videoCapture;
    private Recording activeRecording;
    private File recordedFile;
    private volatile boolean isRecording = false;

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
        root.addView(cameraFrame, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                0,
                1f
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
                LinearLayout.LayoutParams.WRAP_CONTENT
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
        resultText.setText("Live: зелёный = уверенно SKU, красный = распознан плохо (проверить), "
                + "слабые детекции скрыты. Подсчёт SKU по брендам — после отправки видео на backend.");
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
                if (scanMode) {
                    processScanFrame(frame, image.getImageInfo().getRotationDegrees());
                } else {
                    runDetection(frame);
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
        overlayView.setDetections(new ArrayList<>());
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
        havePrevAngles = false;
        coverageView.setState(new ArrayList<>(), new ArrayList<>(), 0, 0, 0, 0,
                false, 0f, 0f, "", null, 0, 0, false);
    }

    private void runDetection(RgbFrame frame) {
        final List<ProductDetection> detections;
        final String statusText;
        if (tfliteAnalyzer != null && tfliteAnalyzer.isReady()) {
            detections = tfliteAnalyzer.analyze(frame);
            int uncertain = 0;
            for (ProductDetection d : detections) {
                if (!d.recognized) {
                    uncertain++;
                }
            }
            statusText = "SKU: " + detections.size() + ", на проверку (красные): " + uncertain;
        } else {
            detections = productAnalyzer.analyze(null);
            statusText = "Демо-режим (модель не загрузилась)";
        }
        runOnUiThread(() -> {
            overlayView.setDetections(detections);
            liveStatusText.setText(statusText);
        });
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

        new Thread(() -> {
            try {
                String response = uploadMultipart(serverUrl + "/jobs/upload", storeName, selectedUris);
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
            b.append("На проверку: ").append(totals.optInt("needs_review")).append("\n");
        }
        appendCounts(b, "\nПо брендам:", report.optJSONObject("by_brand"));
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

    private String uploadMultipart(String targetUrl, String storeName, List<Uri> uris) throws Exception {
        String boundary = "SkuFindBoundary" + System.currentTimeMillis();
        HttpURLConnection connection = (HttpURLConnection) new URL(targetUrl).openConnection();
        connection.setRequestMethod("POST");
        connection.setDoOutput(true);
        connection.setConnectTimeout(15000);
        connection.setReadTimeout(120000);
        connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);
        connection.setChunkedStreamingMode(0);

        try (OutputStream output = new BufferedOutputStream(connection.getOutputStream())) {
            writeFormField(output, boundary, "store_name", storeName);
            for (Uri uri : uris) {
                writeFileField(output, boundary, "files", uri);
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

package com.utake.skufind;

import android.Manifest;
import android.app.Activity;
import android.content.ClipData;
import android.content.ContentResolver;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.database.Cursor;
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
import androidx.camera.view.PreviewView;

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
    private PreviewView previewView;
    private ProductOverlayView overlayView;

    private final List<Uri> selectedUris = new ArrayList<>();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final Executor mainExecutor = command -> mainHandler.post(command);
    private final DemoProductAnalyzer productAnalyzer = new DemoProductAnalyzer();
    private ExecutorService cameraExecutor;
    private ProcessCameraProvider cameraProvider;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        cameraExecutor = Executors.newSingleThreadExecutor();
        setContentView(buildUi());
        ensureCamera();
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

        Button startButton = new Button(this);
        startButton.setText("Live");
        startButton.setOnClickListener(view -> ensureCamera());
        cameraButtons.addView(startButton, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));

        Button stopButton = new Button(this);
        stopButton.setText("Stop");
        stopButton.setOnClickListener(view -> stopCamera());
        cameraButtons.addView(stopButton, new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f));

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
        resultText.setText("Live overlay: зеленый = распознано, красный = не распознано. Сейчас подключен demo analyzer.");
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
                .build();
        analysis.setAnalyzer(cameraExecutor, image -> {
            List<ProductDetection> detections = productAnalyzer.analyze(image);
            runOnUiThread(() -> {
                overlayView.setDetections(detections);
                liveStatusText.setText("Live: " + detections.size() + " objects, demo analyzer");
            });
            image.close();
        });

        cameraProvider.unbindAll();
        cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis);
    }

    private void stopCamera() {
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
        }
        overlayView.setDetections(new ArrayList<>());
        liveStatusText.setText("Live camera stopped.");
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
                runOnUiThread(() -> {
                    resultText.setText(formatted);
                    uploadButton.setEnabled(true);
                });
            } catch (Exception exception) {
                runOnUiThread(() -> {
                    resultText.setText("Ошибка: " + exception.getMessage());
                    uploadButton.setEnabled(true);
                });
            }
        }).start();
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

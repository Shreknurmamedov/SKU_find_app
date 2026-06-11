package com.utake.skufind;

import android.app.Activity;
import android.content.ClipData;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.provider.OpenableColumns;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

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

public class MainActivity extends Activity {
    private static final int PICK_MEDIA_REQUEST = 1001;
    private static final String PREFS = "sku_find_prefs";
    private static final String DEFAULT_SERVER_URL = "http://10.0.2.2:8088";

    private EditText serverUrlInput;
    private EditText storeNameInput;
    private TextView selectedFilesText;
    private TextView resultText;
    private Button uploadButton;

    private final List<Uri> selectedUris = new ArrayList<>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(buildUi());
    }

    private View buildUi() {
        SharedPreferences prefs = getSharedPreferences(PREFS, MODE_PRIVATE);

        ScrollView scrollView = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(20), dp(18), dp(20), dp(24));
        scrollView.addView(root);

        TextView title = new TextView(this);
        title.setText("SKU Find");
        title.setTextSize(26);
        title.setGravity(Gravity.START);
        title.setTextColor(0xFF111111);
        root.addView(title);

        TextView subtitle = new TextView(this);
        subtitle.setText("MVP upload client");
        subtitle.setTextSize(14);
        subtitle.setTextColor(0xFF666666);
        subtitle.setPadding(0, 0, 0, dp(18));
        root.addView(subtitle);

        serverUrlInput = new EditText(this);
        serverUrlInput.setSingleLine(true);
        serverUrlInput.setHint("Backend URL");
        serverUrlInput.setText(prefs.getString("server_url", DEFAULT_SERVER_URL));
        root.addView(serverUrlInput, matchWrap());

        storeNameInput = new EditText(this);
        storeNameInput.setSingleLine(true);
        storeNameInput.setHint("Название ТТ");
        storeNameInput.setText(prefs.getString("store_name", ""));
        root.addView(storeNameInput, matchWrap());

        Button pickButton = new Button(this);
        pickButton.setText("Выбрать фото/видео");
        pickButton.setOnClickListener(view -> openPicker());
        root.addView(pickButton, matchWrap());

        uploadButton = new Button(this);
        uploadButton.setText("Отправить на backend");
        uploadButton.setEnabled(false);
        uploadButton.setOnClickListener(view -> uploadSelectedFiles());
        root.addView(uploadButton, matchWrap());

        selectedFilesText = new TextView(this);
        selectedFilesText.setText("Файлы не выбраны");
        selectedFilesText.setTextSize(14);
        selectedFilesText.setTextColor(0xFF333333);
        selectedFilesText.setPadding(0, dp(12), 0, dp(12));
        root.addView(selectedFilesText);

        resultText = new TextView(this);
        resultText.setText("Результат появится после загрузки.");
        resultText.setTextSize(14);
        resultText.setTextColor(0xFF111111);
        resultText.setPadding(0, dp(12), 0, 0);
        root.addView(resultText);

        return scrollView;
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
        if (requestCode != PICK_MEDIA_REQUEST || resultCode != RESULT_OK || data == null) {
            return;
        }

        selectedUris.clear();
        ClipData clipData = data.getClipData();
        if (clipData != null) {
            for (int index = 0; index < clipData.getItemCount(); index++) {
                Uri uri = clipData.getItemAt(index).getUri();
                getContentResolver().takePersistableUriPermission(
                        uri,
                        Intent.FLAG_GRANT_READ_URI_PERMISSION
                );
                selectedUris.add(uri);
            }
        } else if (data.getData() != null) {
            Uri uri = data.getData();
            getContentResolver().takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION
            );
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
        builder.append("SKU-распознавание будет следующим ML-слоем. Сейчас backend принял файлы и проверил качество.");
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

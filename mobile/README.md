# Android MVP

Android-приложение находится в `mobile/android`.

## Что умеет текущий MVP

- выбрать фото/видео через системный file picker;
- указать название торговой точки;
- указать адрес backend;
- отправить файлы на `POST /jobs/upload`;
- показать `job_id`, количество файлов, качество кадров и подсказки по retake.

## Запуск backend для планшета

На Mac в корне проекта:

```bash
cd backend
PYTHONPATH=src python3 -m uvicorn sku_audit.app:app --host 0.0.0.0 --port 8088
```

Узнать IP Mac в Wi-Fi:

```bash
ipconfig getifaddr en0
```

В Android-приложении укажите:

```text
http://<IP-MAC>:8088
```

Планшет и Mac должны быть в одной Wi-Fi сети.

## Сборка

Откройте папку `mobile/android` в Android Studio и нажмите Run.

Текущая машина не показывает Android SDK/Gradle в терминале, поэтому локальная CLI-сборка APK здесь не проверялась.

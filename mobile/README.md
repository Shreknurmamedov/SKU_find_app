# Android MVP

Android-приложение находится в `mobile/android`.

## Что умеет текущий MVP

- открыть live camera preview;
- рисовать live overlay поверх видеопотока;
- показывать зеленые сегменты для `recognized`;
- показывать красные сегменты для `unknown`;
- выбрать фото/видео через системный file picker;
- указать название торговой точки;
- указать адрес backend;
- отправить файлы на `POST /jobs/upload`;
- показать `job_id`, количество файлов, качество кадров и подсказки по retake.

Сейчас live overlay подключен к `DemoProductAnalyzer`: это рабочий контур камеры/оверлея, но не настоящая SKU-модель. Реальная модель должна заменить этот класс и вернуть такие же `ProductDetection` с `recognized=true/false`.

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

Или соберите APK через GitHub Actions:

1. Откройте репозиторий на GitHub.
2. Перейдите во вкладку `Actions`.
3. Выберите workflow `Android APK`.
4. Откройте последний успешный run.
5. Скачайте artifact `sku-find-debug-apk`.

Текущая машина не показывает Android SDK/Gradle в терминале, поэтому локальная CLI-сборка APK здесь не проверялась.

## Для реального live SKU recognition

Нужны:

- модель сегментации товаров, например TFLite instance segmentation;
- модель или retrieval-слой SKU/бренда;
- `labels.json` с mapping SKU/бренд/own-brand;
- пороги уверенности: выше порога рисуем зеленым, ниже порога красным;
- тестовый набор видео с ручной разметкой.

# Передача проекта (HANDOFF) — SKU Find / Lidar app

Документ для продолжения работы (в т.ч. в Codex). Это резюме всего, что сделано,
почему, текущее состояние и что дальше. Дата последнего обновления: 2026-06-18.

---

## 1. Цель проекта
Android-приложение для пересчёта SKU по видео в торговых точках:
- считать **все видимые SKU** (в т.ч. на хаотичной выкладке),
- определять **бренд / группу / модель**, отдельно — наши бренды,
- **не считать один товар дважды** на видео,
- давать менеджеру **обратную связь при съёмке** (что снято, что плохо — переснять),
- цель — удобство + высокая точность. В будущем: распознавать **все** бренды.

Наши бренды (каталог `data/catalog/own_products.csv`, 2470 SKU):
**Вихрь (996), Ресанта (705), Huter (633), Eurolux (114), TEK (18)**. Колонки:
brand_name, category, model_name, article_codes, aliases, reference_images (ПУСТО).

---

## 2. Текущее состояние (TL;DR)
- **Архитектура двухэтапная:** детектор находит ВЕСЬ товар одним классом `product`;
  бренд/модель/группа определяются по кропу из каталога (OCR), НЕ на этапе детекции.
- **Боевая модель детектора:** `weights/product_det_v2.pt` = **v6** (откатились с v7).
  on-device в APK — её TFLite-экспорт (генерится в CI).
- **Подсчёт SKU работает на сервере** (backend) по загруженному видео; в приложении
  есть запись видео и отправка на backend, опрос статуса и показ отчёта.
- **Live на устройстве** = детекция товара + уверенность (зелёный/красный), НЕ бренд.
- **Главная нерешённая проблема:** детектор качается между «обводит стены» и «не видит
  товар», потому что обучен на АВТО-разметке (FastSAM), без проверенных человеком
  меток. Durable-фикс = ручная разметка (см. §9–10).
- Репозиторий: `git@github.com:Shreknurmamedov/SKU_find_app.git` (public, ветка main).
  APK собирается **только в GitHub Actions** (локального Android SDK нет).

---

## 3. Архитектура (3 части)

### A. Live на устройстве (TFLite, без сервера)
- `MainActivity` (CameraX, RGBA_8888) → `RgbFrame` → `TFLiteProductAnalyzer`
  (модель `assets/models/product_det_v2_float32.tflite`, экспорт product_det_v2,
  imgsz 320) → боксы товара, цвет по уверенности: зелёный=уверенно (Товар),
  красный=неуверенно («проверить»), ниже порога не показываются.
- Пороги в `TFLiteProductAnalyzer`: `CONF_THRESHOLD=0.45` (показ), `GREEN_CONF=0.60`,
  `MAX_AREA_FRAC=0.60` (отсечь стену/шкаф во весь кадр), `MIN_AREA_FRAC=0.004`.
- **on-device НЕ читает бренд** (OCR тяжёлый) — только детекция+уверенность.

### B. Режим «Сканирование покрытия» (on-device)
- Кнопка «Сканировать покрытие». `ScanGrid` (виртуальная сетка), `MotionTracker`
  (блок-матчинг по яркости + гиро), `FrameQuality` (резкость по Лапласиану),
  `CoverageOverlayView` (сетка/мини-карта/подсказки). Ячейка: серый=не снято,
  жёлтый=снято плохо (переснять), зелёный=ок. Реверс направления исправлен
  (`MotionTracker`: `cx -= dcx`).

### C. Подсчёт SKU на сервере (реальная цифра)
- Приложение: «● Запись» (CameraX VideoCapture, без звука) → видео в app storage →
  «Отправить на backend» (`/jobs/upload`) → опрос `/jobs/{id}` → показ отчёта.
- Backend `backend/src/sku_audit/`: `app.py` (FastAPI), при загрузке видео в фоне
  запускает `sku_count.process_job` → `python -m ml.audit_video` по каждому видео
  (subprocess, cwd=repo root) → агрегирует в job `sku_status`/`sku_report`.
- `ml/audit_video.py`: видео → `ml/track_video.py` (ByteTrack, 1 трек=1 объект,
  лучший кроп) → `ml/reid_merge.py` (склейка повторных проходов; консервативно) →
  `ml/sku_recognize.py` (OCR easyocr ru+en, 4 поворота, матч с каталогом) →
  отчёт JSON+MD (totals, by_brand, by_category, by_model, needs_review,
  brand_not_visible).

---

## 4. Цепочка моделей детектора (важно)
Все под одним именем `weights/product_det_v2.pt` (боевое имя). Датасеты в
`ml/datasets/` (gitignored). Обучение: `ml/train_detector.py` (yolo11n, imgsz 1024,
MPS). Веса для CI закоммичены (force-add).

- **baseline `weights/product_det.pt`** — на старых 2-классовых авто-метках,
  недосчёт (~45/видео), чисто. (отдельный файл, цел)
- **product_det_fs** — FastSAM «обводи всё», переобнаружение (1000+/видео). Плохо.
- **v2** — строгий FastSAM (`ml/fastsam_label.py --min-area 0.005`), ~35 целых
  товаров/фото, mAP50 0.72. Хорошая полнота.
- **v4** — + 64 фото пользователя «Обучение ИИ» (вкл. коробки Ресанта/Huter),
  FastSAM-размечены.
- **v5** — + 150 интерьерных негативов из **SUN397** (HuggingFace; `ml/hf_negatives.py`).
  COCO заблокирован из окружения (CDN `images.cocodataset.org` недоступен).
- **v6** — + 120 розничных позитивов SUN397 (`ml/hf_retail_positives.py`). mAP50 0.62.
  **ТЕКУЩАЯ боевая.** Даёт коробке Huter уверенность ~0.56-0.63.
- **v7** — + 160 «жёстких негативов» из видео пользователя (его шкафы/стены/двор,
  `ml/add_device_negatives.py`). **ПЕРЕЖАЛ**: товар стал ~0.23-0.45 → откатили на v6.

**Вывод:** добавление негативов убирает ложные срабатывания на стенах, но роняет
уверенность на товаре (FN). Авто-разметка достигла потолка — модель осциллирует.

---

## 5. Распознавание (этап 2) и иерархия — `ml/sku_recognize.py`
OCR кропа + матч с каталогом. Иерархия (graceful fallback):
1. **matched_sku** — прочитан код модели → бренд + модель + точная категория;
2. **brand_only** — прочитан бренд (+ грубый тип-категория, если есть);
3. **category_only** — прочитан только тип товара (существительное; прилагательные
   отсеяны), напр. «Опрыскиватель», «Триммер»;
4. **unknown** — ничего не прочитано → «бренд не виден, переснять».
Бренд матчится по токенам (защита от подстрок: «ТЕКСТ»→canon «TEKCT» ≠ бренд TEK).
Подтверждено на реальном видео: читает Huter SP-3,7 Lite, Вихрь, Ресанта АСПТ-63.

---

## 6. Android-приложение
- Пакет `com.utake.skufind`, `mobile/android/`, Java, программный UI (без XML).
  minSdk 26, target 35. CameraX 1.4.1 (+ camera-video), TFLite 2.16.1.
- Классы: `MainActivity`, `TFLiteProductAnalyzer`, `RgbFrame`, `LumaFrame`,
  `FrameQuality`, `MotionTracker`, `ScanGrid`, `CoverageOverlayView`,
  `ProductOverlayView`, `ProductDetection`, `DemoProductAnalyzer` (фейк, не нужен).
- **Сборка APK:** только через CI `.github/workflows/android-apk.yml` (push в main или
  workflow_dispatch). Шаг экспорта TFLite: `actions/setup-python` + `pip install
  "numpy<2" ultralytics` + `python ml/export_tflite.py` (кладёт .tflite в assets) →
  `gradle :app:assembleDebug` → артефакт `sku-find-debug-apk` (~19 МБ).
- Проверка сборки без gh (не установлен): `curl` к public API; парсить
  `json.loads(..., strict=False)` (в сообщениях коммитов переводы строк). Скачивание
  артефакта требует авторизации — качает пользователь со страницы run.

### Тест на реальном устройстве через adb
- `adb` поставлен (`brew install --cask android-platform-tools`). Планшет:
  **Samsung Galaxy Tab A9 (SM-X115), Android 14**.
- `adb exec-out screencap` показывает камеру **ЧЁРНОЙ** (аппаратный слой) — для
  скриншота камеры+оверлея использовать `adb shell screenrecord` (захватывает слой),
  потом `adb pull` + извлечь кадр (cv2).
- Записанные приложением видео: `/sdcard/Android/data/com.utake.skufind/files/videos/`
  (adb pull работает). Запуск приложения: `adb shell am start -n
  com.utake.skufind/.MainActivity`; разрешение камеры: `adb shell pm grant ...
  android.permission.CAMERA`.
- Установка нового APK: пользователь качает zip со страницы run → `adb install -r <apk>`.

---

## 7. Backend (запуск и сеть)
Запуск на Маке (тот же Python с ultralytics/torch/easyocr; нужны веса
`weights/product_det_v2.pt`):
```bash
cd backend
PYTHONPATH=src SKU_AUDIT_DEVICE=mps python3 -m uvicorn sku_audit.app:app --host 0.0.0.0 --port 8088
```
- В приложении адрес backend = `http://<IP-Мака>:8088` (НЕ `10.0.2.2` — это эмулятор).
  IP Мака меняется по сети: `ipconfig getifaddr en0`. Планшет и Мак — одна Wi-Fi.
- Загрузка больших видео была по таймауту — увеличены таймауты клиента (read 10 мин,
  1MB-чанки). Совет: снимать короткие ролики (15-30 с). Локально (adb pull + curl)
  загрузка мгновенная — можно отлаживать, минуя Wi-Fi.
- Env: `SKU_AUDIT_DEVICE` (mps/cpu/cuda), `SKU_AUDIT_WEIGHTS`, `SKU_AUDIT_VAR_DIR`.

---

## 8. Ключевые находки и решения (честно)
- **Не делать класс-на-SKU** (2470 классов нереально) — один класс `product` + этап 2.
- **Авто-метки (FastSAM)** дали полноту, но без человеческой проверки модель
  осциллирует FP↔FN. Это потолок подхода.
- **COCO CDN недоступен** из окружения → негативы берём из **SUN397 (HuggingFace)**.
- **`mobileclip_blt.ts` битый** (обрезан) → re-ID на эмбеддингах YOLO backbone.
- **`reference_images` в каталоге пусты** → визуального поиска SKU нет, работает только
  OCR. Это главный ограничитель распознавания брендов/моделей.
- **Смаз сильно роняет уверенность** (чёткий товар ~0.6-0.74, смазанный ~0.45).
- **TFLite-экспорт нельзя на macOS+py3.13** (блок ultralytics) и капризен в Docker
  arm64 (numpy/cmake) → делаем экспорт в CI (Ubuntu x86_64).

---

## 9. Известные проблемы / ограничения
1. **Детектор: FP на стенах/мебели ↔ FN на товаре.** Порогом не разделить чисто
   (стена ~0.52, товар ~0.56). Нужна ручная разметка.
2. **Live не распознаёт бренд** (только детекция); бренд/модель — на сервере.
3. **Распознавание брендов ограничено OCR** (нет reference_images, нет каталога
   конкурентов) → «все бренды» пока невозможно.
4. **Загрузка больших видео по Wi-Fi медленная/хрупкая** (таймаут увеличен; нужны
   короткие ролики или загрузка кадров вместо видео).
5. Сборка APK медленная (экспорт модели в CI на каждый билд).

---

## 10. Следующие шаги (приоритет)
1. **Ручная разметка (durable fix точности детектора):** нарезать ~200-300 кадров из
   видео/фото пользователя, обвести реальные товары + отметить чистые негативы
   (стены/мебель), обучить supervised-детектор. В проекте есть конфиги Label Studio
   (`ml/label_studio_config.xml`, `ml/serve_dataset.py`). Это уберёт осцилляцию.
2. **Эталонные фото SKU → визуальный поиск** (для распознавания брендов/моделей, в т.ч.
   там, где OCR не читает): снять 3-5 фото на модель, заполнить `reference_images`,
   подключить эмбеддинг-ретривал вторым сигналом к OCR. Путь к «всем брендам».
3. **Сбалансированный детектор v8** (если не делать ручную разметку): v6 + умеренное
   число (40-60, не 160) хард-негативов пользователя → меньше FP без потери товара.
4. **Облачный backend** (чтобы работало вне локальной сети; сейчас backend на Маке).
5. **Загружать кадры вместо видео** (или сжимать) — надёжнее по сети.
6. Подсчёт по одиночному фото (сейчас только видео).

---

## 11. Карта файлов и команды
ML (`ml/`): `fastsam_label.py` (разметка), `build_product_dataset.py`,
`train_detector.py`, `export_tflite.py`, `track_video.py`, `reid_merge.py`,
`sku_recognize.py`, `audit_video.py`, `combine_reports.py`, `add_negatives.py`,
`add_training_photos.py`, `hf_negatives.py`, `hf_retail_positives.py`,
`coco_negatives.py`, `add_device_negatives.py`, `trackers/botsort_sku.yaml`.
Оркестраторы (`scripts/`): `run_*_retrain.sh`, `run_all_audits.sh`,
`run_tuning_cycle.sh`, `rerun_audit_reid.sh`.
Backend (`backend/src/sku_audit/`): `app.py`, `jobs.py`, `sku_count.py`,
`pipeline.py`, `media.py`, `catalog.py`.
Датасеты/веса/`var/`/`runs/` — gitignored (на диске). Отчёты — `reports/`.
Полезное: `docs/video-audit-pipeline.md`, `mobile/README.md`.

Частые команды:
```bash
# обучить детектор
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 ml/train_detector.py --data ml/datasets/<ds>/data.yaml --imgsz 1024 --batch 4 --epochs 30 --patience 8 --name <run>
# аудит видео локально
python3 -m ml.audit_video --video "<path>" --weights weights/product_det_v2.pt --device mps
# backend
cd backend && PYTHONPATH=src SKU_AUDIT_DEVICE=mps python3 -m uvicorn sku_audit.app:app --host 0.0.0.0 --port 8088
# проверить последнюю CI-сборку
curl -s "https://api.github.com/repos/Shreknurmamedov/SKU_find_app/actions/runs?per_page=1"
```

---

## 12. Среда
- Mac (Apple Silicon), Homebrew есть; Python 3.13 (ultralytics/torch/easyocr/cv2
  установлены; pytest нет — тесты backend через `python -m unittest`). MPS доступен.
- НЕТ: Android SDK/Studio, gh CLI, Android-эмулятор. ЕСТЬ: adb, Docker.
- Последние коммиты вели от Codex/Claude; боевая модель = v6 (коммит ~3fd7153),
  APK с ней собирается в CI.

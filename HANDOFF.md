# Передача проекта (HANDOFF) — SKU Find / Lidar app

Документ для продолжения работы (в т.ч. в Codex). Это резюме всего, что сделано,
почему, текущее состояние и что дальше. Дата последнего обновления: 2026-06-18.

---

## 1. Цель проекта
Android-приложение для аудита SKU по видео в торговых точках:
- находить **все видимые уникальные SKU/артикулы** (в т.ч. на хаотичной выкладке),
- определять **бренд / группу / модель / артикул**, отдельно — наши бренды,
- **не считать один SKU дважды**; количество одинаковых экземпляров одной модели
  вторично и нужно только как диагностический сигнал,
- давать менеджеру **обратную связь при съёмке** (что снято, что плохо — переснять),
- цель — удобство + высокая точность. В будущем: распознавать **все** бренды.

Наши бренды (каталог `data/catalog/own_products.csv`, 2470 SKU):
**Вихрь (996), Ресанта (705), Huter (633), Eurolux (114), TEK (18)**. Колонки:
brand_name, category, model_name, article_codes, aliases, reference_images (ПУСТО).

---

## 2. Текущее состояние (TL;DR)
- **Архитектура двухэтапная:** детектор находит ВЕСЬ товар одним классом `product`;
  бренд/модель/группа определяются по кропу через OCR, каталог и эталонные фото,
  НЕ на этапе детекции.
- **Боевая модель детектора:** `weights/product_det_v2.pt` = **v6** (откатились с v7).
  on-device в APK — её TFLite-экспорт (генерится в CI).
- **Новый кандидат, НЕ production:** `weights/product_det_v8_hardneg.pt`. Это v8_auto
  + сбалансированные hard negatives; object count error на auto-val при `conf=0.35`
  ~7.4%, но это только диагностика. Перед заменой v6 нужна ручная SKU-presence
  проверка: какие `sku_id` реально есть на полке.
- **Распознавание SKU работает на сервере** (backend) по загруженному видео; в приложении
  есть запись видео и отправка на backend, опрос статуса и показ отчёта.
- **Эталонные фото подключены:** `data/catalog/reference_index_yolo11n.npz` +
  `.jsonl` собраны из 14 441 фото (`reference_dataset_all`). Visual fallback
  включается консервативно, только когда OCR не дал точный SKU.
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

### C. SKU presence на сервере (реальный бизнес-ответ)
- Приложение: «● Запись» (CameraX VideoCapture, без звука) → видео в app storage →
  «Отправить на backend» (`/jobs/upload`) → опрос `/jobs/{id}` → показ отчёта.
- Backend `backend/src/sku_audit/`: `app.py` (FastAPI), при загрузке видео в фоне
  запускает `sku_count.process_job` → `python -m ml.audit_video` по каждому видео
  (subprocess, cwd=repo root) → агрегирует в job `sku_status`/`sku_report`.
- `ml/audit_video.py`: видео → `ml/track_video.py` (ByteTrack собирает кандидатные
  товары, лучший кроп и несколько запасных кропов) → `ml/reid_merge.py` (склейка
  повторных проходов; консервативно) → `ml/sku_recognize.py` (OCR easyocr ru+en,
  4 поворота, матч с каталогом + осторожный visual fallback по эталонным фото)
  → `sku_presence` (уникальные `sku_id`/модели/артикулы) + `needs_review`.
  Object-level поля (`candidate_objects`, `sku_evidence_objects`) остаются
  диагностикой, не финальной целью.

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
- **v8_auto** — review-набор из реальных фото/видео, предзаполненный `product_det_v2`
  + YOLO-World, без ручной проверки; `weights/product_det_v8_auto.pt`. На v8_auto val:
  `conf=0.40` дал count error ~1.7%, но precision низкий, т.к. это auto-labels.
- **v8_hardneg** — v8_auto + 60 device negatives + 80 замайненных background crops
  (`ml/add_v8_hard_negatives.py`), обучен коротко от v8_auto. Дальше первой эпохи
  метрики ухудшались, поэтому взят `best.pt` epoch 1 → `weights/product_det_v8_hardneg.pt`.
  На `sku_products_v8_hardneg` val: `conf=0.35` count error 7.4%, precision 60.4%,
  recall 64.9%; `conf=0.40` count error 8.1%, precision 64.4%, recall 59.2%.

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

> **Обновлено (§14):** конкретный SKU выдаётся только при НАДЁЖНОМ чтении кода
> (явный артикул или литеральный длинный код). Слабые/OCR-подменённые коды и
> визуальный поиск теперь дают только **бренд + категорию**, а не угаданную модель.

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
- **Эталонные фото SKU уже индексированы** из `data/catalog/reference_dataset_all`:
  14 441 фото по Вихрь/Ресанта/Huter/Зубр/Fubag/Интерскол/Patriot. Visual retrieval
  полезен там, где OCR не читает текст, но похожие коробки дают близкие эмбеддинги,
  поэтому включен порог по score+margin и OCR остается главным сигналом.
- **Смаз сильно роняет уверенность** (чёткий товар ~0.6-0.74, смазанный ~0.45).
- Для backend tracking добавлен runtime-фильтр плоских background-кропов и отдельный
  low-detail фильтр для светлой плитки/стены с одним швом. Он снижает FP пола, не
  заменяя нормальное обучение.
- **TFLite-экспорт нельзя на macOS+py3.13** (блок ultralytics) и капризен в Docker
  arm64 (numpy/cmake) → делаем экспорт в CI (Ubuntu x86_64).

---

## 9. Известные проблемы / ограничения
1. **Детектор: FP на стенах/мебели ↔ FN на товаре.** Порогом не разделить чисто
   (стена ~0.52, товар ~0.56). Нужна ручная разметка.
2. **Live не распознаёт бренд** (только детекция); бренд/модель — на сервере.
3. **Визуальный fallback не является абсолютной истиной.** Он уже видит часть
   конкурентов из эталонной базы, но похожие упаковки могут путаться; нужны
   ручной `sku_presence.csv` и проверка спорных зон.
4. **Загрузка больших видео по Wi-Fi медленная/хрупкая** (таймаут увеличен; нужны
   короткие ролики или загрузка кадров вместо видео).
5. Сборка APK медленная (экспорт модели в CI на каждый билд).

---

## 10. Следующие шаги (приоритет)
1. **Ручная разметка (durable fix точности детектора):** нарезать ~200-300 кадров из
   видео/фото пользователя, обвести реальные товары + отметить чистые негативы
   (стены/мебель), обучить supervised-детектор. В проекте есть конфиги Label Studio
   (`ml/label_studio_config.xml`, `ml/serve_dataset.py`). Это уберёт осцилляцию.
2. **Расширять эталонные фото SKU:** визуальный поиск уже подключен вторым сигналом
   к OCR. Дальше лучше добавлять 3-5 фото на проблемную модель с разных ракурсов и
   перепроверять на holdout-видео, не подмешивая его в обучение.
3. **Проверить v8_hardneg на SKU-presence CSV**: `weights/product_det_v8_hardneg.pt`
   уже собран, но production `weights/product_det_v2.pt` не заменять, пока на
   20-30 ручных видео/кадрах не будет стабильного SKU recall/precision по
   `expected_sku_ids`.
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
# SKU-аудит видео локально
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

---

## 13. Обновление Codex: v8 review workflow + live-память
Добавлены файлы:
- `ml/prepare_product_review_dataset.py` — готовит v8 review-набор: реальные фото
  и кадры видео + предзаполненные product-боксы от `product_det_v2`, YOLO-World и
  FastSAM.
- `ml/label_studio_product_config.xml` — Label Studio config для одного класса
  `product`.
- `docs/detector-v8-workflow.md` — команды v8: подготовка, ревью, экспорт,
  сборка `sku_products_v8`, обучение; финальная метрика теперь SKU coverage,
  object count error только диагностический ориентир.
- `ml/evaluate_detector_counts.py` — метрики детектора на YOLO val split:
  TP/FP/FN, precision/recall, count error.
- `ml/evaluate_sku_coverage.py` — метрики полного video-audit против CSV
  `video,expected_sku_ids`.
- `ml/evaluate_audit_counts.py` — старая диагностическая object-count метрика;
  не использовать как бизнес-критерий.
- `ml/build_reference_index.py` — строит visual-index из эталонных фото
  (`training_images.csv` → `.npz` + `.jsonl`).
- `ml/visual_reference.py` — поиск ближайших эталонных фото по YOLO-эмбеддингам,
  с группировкой по SKU/product key и margin-защитой от похожих коробок.
- `scripts/run_v8_retrain.sh` — wrapper обучения v8 и замены
  `weights/product_det_v2.pt`.
- `ml/add_v8_hard_negatives.py` — сборка сбалансированного hard-negative
  датасета: умеренное число пустых device-frames + obvious background crops,
  замайненных v8_auto. В генератор добавлен строгий low-detail/edge фильтр, чтобы
  не превращать малонасыщенные реальные товары в негативы.
- `scripts/run_v8_hardneg_retrain.sh` — wrapper для `sku_products_v8_hardneg`
  и обучения кандидата `weights/product_det_v8_hardneg.pt`.
- `mobile/android/.../LiveCaptureTracker.java` — on-device память live-сессии:
  товар, снятый хорошо несколько кадров подряд, больше не рисуется рамкой.
- `docs/tablet-live-feedback.md` — логика обратной связи на планшете.

Изменено:
- `ml/build_product_dataset.py` теперь принимает `--src/--dst` и понимает как
  Label Studio polygon/rectangle labels, так и готовый YOLO detection формат.
- `ml/convert_labelstudio_to_yolo.py` понимает label `product`.
- `ml/track_video.py` сохраняет несколько лучших кропов на трек
  (`crop_candidates`, `--crops-per-track`), чтобы OCR мог выбрать читаемый ракурс.
- `ml/reid_merge.py` переносит эти `crop_candidates` после склейки re-ID.
- `ml/audit_video.py` агрегирует результат по уникальным SKU и запускает enhanced
  OCR + visual fallback только для топ-N объектов (`--ocr-retry-objects`, default 10).
  Среди `crop_candidates` теперь выбирает OCR-friendly crop по резкости/edge-density/
  текстоподобным компонентам, чтобы не брать самый уверенный, но нечитаемый бокс.
- `ml/sku_recognize.py` теперь матчится не только по модели/alias, но и по артикулу
  (`article_codes`), а при неудаче OCR может использовать visual reference index.
  Артикулы матчятся строго (без подстроки внутри более длинного артикула);
  модельные коды имеют узкую OCR-нормализацию `5P`/`CP` → `SP`, `ЭТ/ЛИ` ↔
  `ET/LI`, `GET...` ↔ OCR-вариант без начальной `G`, и `T20...` → `ET20...`.
- `MainActivity` считает резкость кропа, подписывает live-боксы `снято`,
  `медленнее`, `ближе`, `наведите`, `держите`, показывает `Осталось` / `Готово` /
  `Проблемы` и нижнюю подсказку главного действия для менеджера. Слишком мелкие
  bbox теперь не считаются снятыми хорошо: менеджеру показывается `ближе`, чтобы
  артикул был читаем на backend. Добавлена кнопка `Сброс снятого`.

Важно: v8 workflow сокращает ручную работу, но не отменяет human verification.

Фактический прогон 2026-06-18:
- `sku_products_v8_review`: 380 изображений/кадров, 13 934 auto-boxes.
- `sku_products_v8_hardneg`: 520 изображений, 143 empty negatives (~27%).
- `weights/product_det_v8_hardneg.pt` создан из best epoch 1; training остановлен,
  потому что epoch 3 просел до mAP50 0.519.
- `IMG_8886.MOV`, hardneg + runtime filters: `conf=0.35` → 88 kept tracks,
  `conf=0.40` → 81 kept tracks, `conf=0.45` → 71 kept tracks. Визуально лучший
  компромисс сейчас `conf=0.40`; `conf=0.35` ловит больше товара, но оставляет
  остаточный FP пола.
Object count error можно использовать только как внутреннюю метрику детектора.
Финальную точность нужно подтверждать на отдельном проверочном наборе
`data/eval/sku_presence.csv`, где для каждого видео перечислены ожидаемые
`sku_id`.

Фактический прогон эталонных фото 2026-06-18:
- `data/catalog/reference_dataset_all/training_images.csv`: 14 441 фото.
- Бренды: Вихрь 4708, Ресанта 3501, Huter 2879, Зубр 1240, Fubag 793,
  Интерскол 698, Patriot 622.
- Роли: `own_target` 11 088, `competitor` 3 353.
- Построены `data/catalog/reference_index_yolo11n.npz` и
  `data/catalog/reference_index_yolo11n.jsonl`.
- Проверка на crop Huter из `IMG_8886`: OCR корректно читает Huter SP-3,7 Lite,
  visual top-1 без margin-защиты путал с Вихрь; поэтому visual fallback включен
  только при `score>=0.78` и `margin>=0.035`.
- Smoke после исправления article/model matcher на старом `IMG_8886.MOV`
  (`--reuse-tracks`, `weights/product_det_v8_hardneg.pt`, `conf=0.40`):
  `unique_skus=1`, `sku_evidence_objects=3`, найден только
  `HUTER_70_13_190_SP_3_7_LITE`, SKU coverage на ручной строке truth = recall 1.0,
  precision 1.0. Ложный матч Вихрь `73/7/2/6` из crop с видимым `73/7/2/26`
  убран.
- Усиление второго этапа на holdout `17732221864542.mp4`:
  - до усиления: v8_hardneg `conf=0.60` → `unique_skus=0`, v6 `conf=0.40` → 0;
  - после OCR/model-code нормализации + OCR-friendly crop selection:
    v8_hardneg `conf=0.60` → `unique_skus=1`, найден
    `HUTER_70_1_67_GET_20M_2LI` по crop `track_0162_alt3.jpg`
    (`[т-20м-2Li Аккумупятопный триммер`);
    v6 `conf=0.40` → `unique_skus=1`, найден близкий crop
    `HUTER_70_1_66_GET_20M_LI`.
  - Старый smoke `IMG_8886.MOV` после усиления не сломался:
    `unique_skus=1`, evidence=3, без ложного Вихрь.
- Важно: `17732221864542.mp4` теперь использовалось для оценки и настройки
  второго этапа (OCR/model-code/crop selection), поэтому больше не является
  чистым независимым holdout. Для финальной проверки после следующих изменений
  нужен новый untouched video/CSV truth.

### v8_auto фактический прогон
- Подготовлен `ml/datasets/sku_products_v8_review`: 380 элементов, из них 200
  кадров видео; split train=317/val=63; авто-предложений 13 934.
- Собран черновой `ml/datasets/sku_products_v8_auto` из этих авто-боксов.
- Обучен кандидат `product_det_v8_auto` 5 эпох, веса сохранены:
  `weights/product_det_v8_auto.pt`.
- Боевой `weights/product_det_v2.pt` НЕ заменён.
- Detector eval на auto-val:
  - v6 (`weights/product_det_v2.pt`, conf=0.35): count error 56.1%,
    precision 93.0%, recall 40.8%.
  - v8_auto (`weights/product_det_v8_auto.pt`, conf=0.35): count error 18.2%,
    precision 56.7%, recall 67.0%.
  - v8_auto с подбором порога: `conf=0.40` дал лучший count error на auto-val
    около 1.7%, но это НЕ ручная истина.
- На реальном `IMG_8886.MOV` v8_auto ловит заметно больше, но также ловит плитку
  пола. Добавлен plain-background фильтр в `ml/track_video.py`; на `IMG_8886`:
  - v8_auto conf=0.50 до фильтра: 86 kept tracks;
  - после фильтра: 64 kept tracks;
  - старый v6: 79 kept tracks.
- Вывод: v8_auto полезен как кандидат/учитель для review, но не готов заменить v6
  без исправления worst-кадров в Label Studio и ручного SKU-presence validation.

---

## 14. Обновление Claude (2026-06-18): bugfix + политика «бренд + категория»

### Исправленные баги
- **Регресс импорта** в `ml/sku_recognize.py`: верхнеуровневый `from
  ml.visual_reference import ...` ломал документированный standalone-запуск
  `python3 ml/sku_recognize.py crop.jpg` (`ModuleNotFoundError: No module named
  'ml'`). Сделан ленивый импорт `VisualReferenceIndex` внутри `_reference()`,
  пути индекса определены локально. Standalone и модульный запуск оба работают.
- **Гигиена git** (`.gitignore`): build-артефакты `reference_index_*.npz/.jsonl`
  (~23 МБ) и генерируемые манифесты `data/catalog/reference_dataset*/` (~100 МБ)
  + папки `reports/holdout_eval_*/`, `reports/model_smoke*/` добавлены в ignore.
  `git add -A` уронил с 89 до 34 файлов, ничего тяжелее 1 МБ (только исходники).

### Политика распознавания: не угадывать модель
Бизнес-правило пользователя: **если артикул/код не читается надёжно — пишем бренд
и категорию, а не выдуманную модель.** Реализация в `ml/sku_recognize.py`:
- `match_text` разделён на `_strong_specific_match` (явный артикул `64/1/20` ИЛИ
  литеральный модельный код длиной ≥ `STRONG_KEY_LEN`=5 типа `DY5000LX`) → SKU,
  и `_weak_specific_hint` (короткие коды + OCR-подмены `5P→SP`, `3T→ET`, `T→ET`)
  → используется ТОЛЬКО для бренда/категории, `sku_id` не присваивается.
- `_recognize_visual` теперь возвращает `brand_only`/`category_only`, а не
  конкретный SKU (близкие коробки дают близкие эмбеддинги — это ненадёжно).
- Проверено: `Huter SP37LITE`→SKU; `Huter 5P37LITE`→Huter/Опрыскиватель (без SKU).

### Отчёт: партиалы как видимый результат
- `ml/audit_video.py`: добавлен `brand_category_presence` +
  `totals.brand_category_partial` + секция markdown «Распознано до бренда +
  категории». Партиалы больше не валятся в «спорные зоны» вместе с «не виден».
- `backend/src/sku_audit/sku_count.py`: то же поле агрегируется по видео с
  дедупом (без двойного счёта одного бренд+категория на разных видео).

### Что НЕ трогалось (намеренно, нужно мерить на данных)
Пороги фоновой фильтрации детектора, замена визуального embedder (`yolo11n` слаб
для fine-grained retrieval), производительность live на планшете. Менять вслепую
= риск регресса. Сначала ручная разметка → измерение F1 (`ml/evaluate_sku_coverage.py`),
потом тюнинг.

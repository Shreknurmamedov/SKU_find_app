# Контекст диалога для продолжения проекта в другой нейросети

Дата: 2026-06-18  
Проект: `SKU Find / Lidar app`  
Путь на диске: `/Users/shreknurmamedov/Documents/Claude/Projects/ИИ СКЮ/Lidar app`

Этот файл нужен, чтобы открыть проект в другой сессии/нейросети и быстро понять,
что уже обсуждали и делали. Это не дословная стенограмма каждого ответа, а полный
рабочий конспект диалога, решений, проверок и текущего состояния.

Перед продолжением обязательно прочитать также `HANDOFF.md` - там техническая
карта проекта, команды, файлы и метрики.

---

## 1. Исходный запрос пользователя

Пользователь попросил продолжить проект после Claude. В корне уже был файл
`HANDOFF.md`, где описаны шаги, сделанные Claude. Нужно было проанализировать
проект и продолжить работу над Android-приложением/ML-пайплайном для аудита SKU.

Главная бизнес-цель уточнена пользователем:

- нужно определять именно **SKU / модель / артикул**, а не количество одинаковых
  физических экземпляров товара;
- если на полке стоит несколько одинаковых инструментов одной модели, это не
  главный результат, важнее уникальные модели/артикулы;
- YOLO должна находить товар без дублей и без мебели/пола/стен;
- на планшете должна быть слабая on-device модель, которая подсказывает
  менеджеру, какие товары сняты плохо и какие уже сняты хорошо;
- уже хорошо снятые товары не должны дальше выделяться рамкой;
- интерфейс должен быть понятным для менеджера, не только техническим.

---

## 2. Важные обсуждения и принятые решения

### 2.1. Класс-на-SKU не делаем

Обсуждали, что 2470+ SKU обучать как отдельные классы YOLO неправильно:

- слишком много классов;
- нужна большая ручная разметка на каждый SKU;
- похожие коробки и ракурсы будут путаться;
- добавление нового артикула потребует переобучения.

Решение: оставить YOLO как детектор одного класса `product`, а SKU определять
вторым этапом по кропу: OCR + каталог + осторожный visual retrieval по эталонным
фото.

### 2.2. Нужно считать SKU presence, а не object count

Пользователь дважды уточнил: нужен не счет физического количества товара, а
количество уникальных SKU/моделей/артикулов. После этого пайплайн был переведен
на SKU-centric отчетность:

- `sku_presence` - основной результат;
- `totals.unique_skus` - основная цифра;
- `candidate_objects`, `sku_evidence_objects` - только диагностика;
- object count error больше не считать бизнес-метрикой.

### 2.3. Сегментация пикселей обсуждалась, но не выбрана как главный путь

Пользователь спросил, улучшит ли качество обучение, если размечать не bbox, а
каждый пиксель объекта. Вывод:

- segmentation masks могут помочь отделять товар от фона;
- но для текущей цели важнее стабильный detection crop для OCR/SKU recognition;
- ручная масочная разметка дороже, чем bbox;
- главная проблема сейчас не форма маски, а авто-разметка без проверки человеком.

Решение: основной путь - bbox `product` + хороший второй этап. Сегментацию можно
рассмотреть позже, если появится ресурс на качественную ручную разметку.

### 2.4. Негативное обучение

Пользователь спросил, делалось ли негативное обучение, как у Claude, чтобы модель
не принимала пол, стулья и мебель за товар.

Ответ и действия:

- Claude уже делал негативы: v5/v7 с SUN397 и device negatives;
- v7 с 160 жесткими негативами пережал модель: стало меньше FP, но товар стал
  ловиться хуже;
- Codex добавил отдельный сбалансированный hard-negative workflow:
  `ml/add_v8_hard_negatives.py`;
- создан кандидат `weights/product_det_v8_hardneg.pt`;
- production пока оставлен на `weights/product_det_v2.pt` (v6), потому что v8
  нужно проверять по SKU-presence, а не только по object count.

### 2.5. Ручная разметка все равно нужна

Пользователь спрашивал, нельзя ли автоматически увидеть весь товар на полках и
разметить его. Ответ: можно предзаполнить разметку автоинструментами, что и было
сделано, но полностью заменить человека нельзя, потому что:

- похожие коробки/полки/мебель дают ложные боксы;
- авто-метки закрепляют ошибки модели;
- без human verification модель качается между FP и FN.

Решение: подготовлен v8 review workflow, который уменьшает ручную работу:
авто-боксы уже предзаполнены, человеку нужно проверить/исправить.

---

## 3. Что было сделано в ML/backend

### 3.1. SKU-centric audit

Обновлены:

- `ml/audit_video.py`
- `backend/src/sku_audit/sku_count.py`
- `ml/combine_reports.py`
- `ml/evaluate_sku_coverage.py`

Теперь итог строится по уникальным SKU, а не по числу физических объектов.

### 3.2. Tracking и кропы

Обновлены:

- `ml/track_video.py`
- `ml/reid_merge.py`

Теперь трек хранит несколько `crop_candidates`, а не только один лучший кроп.
Это важно, потому что самый уверенный bbox не всегда самый читаемый для OCR.

### 3.3. OCR-friendly crop selection

В `ml/audit_video.py` добавлен выбор кропа по признакам:

- резкость;
- edge-density;
- текстоподобные connected components.

Цель: выбирать ракурс, на котором лучше читается артикул/модель.

### 3.4. Усиление второго этапа распознавания

Обновлен `ml/sku_recognize.py`:

- разделены model keys и article keys;
- артикулы матчятся строго, чтобы `73/7/2/6` не совпадал внутри `73/7/2/26`;
- добавлены OCR-варианты:
  - `ЭТ/ЛИ` ↔ `ET/LI`;
  - Huter `GET...` индексируется и как OCR-вариант без первой `G`;
  - `5P/CP -> SP`;
  - `3T -> ET`;
  - `T20... -> ET20...`;
- добавлен fuzzy fallback брендов через `SequenceMatcher`, например:
  - `[РЕНТА` -> `Ресанта`;
  - `HWTER` -> `Huter`.

Реальный проблемный кроп `track_0162_alt3.jpg` теперь мапится на
`HUTER_70_1_67_GET_20M_2LI`.

### 3.5. Эталонные фото брендов

Пользователь добавил эталонные фото нескольких брендов в архив/папку. Было
сделано:

- собран датасет `data/catalog/reference_dataset_all/training_images.csv`;
- всего найдено 14 441 фото;
- бренды:
  - Вихрь: 4708;
  - Ресанта: 3501;
  - Huter: 2879;
  - Зубр: 1240;
  - Fubag: 793;
  - Интерскол: 698;
  - Patriot: 622;
- построены:
  - `data/catalog/reference_index_yolo11n.npz`;
  - `data/catalog/reference_index_yolo11n.jsonl`;
- добавлены:
  - `ml/build_reference_index.py`;
  - `ml/visual_reference.py`;
  - `scripts/collect_reference_catalog.py`.

Важно: visual retrieval полезен, но пока консервативный. Похожие коробки дают
близкие embedding scores, поэтому нельзя просто снижать пороги.

### 3.6. v8/hard-negative workflow

Добавлены:

- `ml/prepare_product_review_dataset.py`;
- `ml/label_studio_product_config.xml`;
- `docs/detector-v8-workflow.md`;
- `ml/evaluate_detector_counts.py`;
- `ml/evaluate_audit_counts.py`;
- `ml/evaluate_sku_coverage.py`;
- `ml/add_v8_hard_negatives.py`;
- `scripts/run_v8_retrain.sh`;
- `scripts/run_v8_hardneg_retrain.sh`.

Создан кандидат:

- `weights/product_det_v8_hardneg.pt`.

Но production остается:

- `weights/product_det_v2.pt` (v6).

Причина: v8_hardneg еще нужно валидировать вручную по SKU presence.

---

## 4. Тесты и результаты, которые уже были

### 4.1. Старое видео `ТТ Пэкстрой/IMG_8886.MOV`

На `weights/product_det_v8_hardneg.pt`, `conf=0.40`:

- `unique_skus=1`;
- `sku_evidence_objects=3`;
- найден `HUTER_70_13_190_SP_3_7_LITE`;
- ложного `Вихрь` не было.

### 4.2. Видео `17732221864542.mp4`

Важно: это видео пользователь сначала хотел держать как тестовое. Но во время
работы оно было использовано для оценки/тюнинга второго этапа. Поэтому оно больше
**не является чистым holdout**.

До усиления второго этапа:

- v8 `conf=0.40`: 0 SKU / 229 candidates;
- v8 `conf=0.60`: 0 SKU / 128 candidates;
- v6 `conf=0.40`: 0 SKU / 104 candidates.

После усиления второго этапа:

- v8 `conf=0.60`: 1 SKU (`HUTER_70_1_67_GET_20M_2LI`) / 128 candidates /
  127 review;
- v8 `conf=0.40`: 1 SKU (`HUTER_70_1_67_GET_20M_2LI`) / 229 candidates /
  228 review;
- v6 `conf=0.40`: 1 SKU (`HUTER_70_1_66_GET_20M_LI`) / 104 candidates.

Вывод: второй этап улучшился с 0 до 1 SKU, но recall все еще низкий. Нужны:

- больше качественных кропов/ракурсов;
- ручная `expected_sku_ids` разметка по тестовым видео;
- улучшение live feedback, чтобы менеджер снимал товар ближе/резче.

---

## 5. Что было сделано в Android/live feedback

Последний запрос пользователя: "давай улучшим обратную связь".

Сделано:

### 5.1. `ProductDetection.java`

Добавлены состояния качества:

- `STATE_GOOD`;
- `STATE_BLUR`;
- `STATE_FAR`;
- `STATE_UNCERTAIN`.

Добавлены поля:

- `qualityState`;
- `sharpness`;
- `areaFraction`.

### 5.2. `MainActivity.java`

Добавлена live-оценка качества:

- резкость crop-а через `FrameQuality.sharpScore`;
- размер bbox в кадре;
- confidence TFLite-детектора;
- главный hint для менеджера.

Пороги:

- `LIVE_SHARP_GOOD = 0.44f`;
- `LIVE_SHARP_BLUR = 0.34f`;
- `LIVE_MIN_AREA_READABLE = 0.026f`;
- `LIVE_MIN_SIDE_READABLE = 0.115f`.

Логика:

- если товар слишком мелкий -> `ближе`;
- если crop смазан -> `медленнее`;
- если неуверенно/не в фокусе -> `наведите`;
- если хорошо -> `держите`.

Приоритет специально сделан так: сначала `ближе`, потом `медленнее`, потому что
для слишком маленького crop-а резкость может оцениваться некорректно.

Добавлена кнопка:

- `Сброс снятого`.

### 5.3. `LiveCaptureTracker.java`

Добавлен новый файл:

- `mobile/android/app/src/main/java/com/utake/skufind/LiveCaptureTracker.java`.

Он:

- связывает соседние detections по IoU;
- ведет короткие tracks физических объектов;
- если объект несколько кадров подряд снят хорошо, считает его `captured`;
- после этого объект больше не показывается рамкой;
- позволяет менеджеру видеть только то, что еще надо доснять.

Важно: это привязка к физическому объекту внутри live-сессии, не к распознанному
SKU. Привязка именно к SKU возможна после backend-распознавания или после легкого
on-device OCR/retrieval.

### 5.4. `ProductOverlayView.java`

Оверлей теперь показывает:

- верхние счетчики:
  - `Осталось`;
  - `Готово`;
  - `Проблемы`;
- нижнюю подсказку главного действия;
- детализацию проблем:
  - `ближе N`;
  - `резче M`;
  - `навести K`.

Цвета:

- зеленый `держите`;
- оранжевый `ближе`;
- красный `медленнее`;
- синий `наведите`.

### 5.5. Документация

Добавлен:

- `docs/tablet-live-feedback.md`.

Обновлен:

- `HANDOFF.md`.

---

## 6. Проверки после последних изменений

Выполнено:

- `python3 -m py_compile ...` по измененным Python-файлам - успешно;
- `git diff --check` - успешно;
- легкая `javac`-проверка измененных Java-классов обратной связи через временные
  Android-стабы - успешно.

Не выполнено:

- полноценная Android Gradle-сборка локально.

Причина:

- в проекте нет `gradlew`;
- системный `gradle` не установлен;
- локального Android SDK/Android Studio в этой среде нет.

APK нужно собирать через GitHub Actions или Android Studio/Gradle на машине, где
есть Android SDK.

---

## 7. Текущее состояние git/worktree

Есть много измененных и новых файлов. Не откатывать чужие/предыдущие изменения.

Ключевые измененные tracked файлы:

- `HANDOFF.md`;
- `backend/src/sku_audit/sku_count.py`;
- `docs/video-audit-pipeline.md`;
- `ml/audit_video.py`;
- `ml/build_product_dataset.py`;
- `ml/combine_reports.py`;
- `ml/convert_labelstudio_to_yolo.py`;
- `ml/reid_merge.py`;
- `ml/sku_recognize.py`;
- `ml/track_video.py`;
- `mobile/android/app/src/main/java/com/utake/skufind/DemoProductAnalyzer.java`;
- `mobile/android/app/src/main/java/com/utake/skufind/MainActivity.java`;
- `mobile/android/app/src/main/java/com/utake/skufind/ProductDetection.java`;
- `mobile/android/app/src/main/java/com/utake/skufind/ProductOverlayView.java`.

Ключевые untracked файлы:

- `mobile/android/app/src/main/java/com/utake/skufind/LiveCaptureTracker.java`;
- `docs/tablet-live-feedback.md`;
- `docs/detector-v8-workflow.md`;
- `ml/add_v8_hard_negatives.py`;
- `ml/build_reference_index.py`;
- `ml/evaluate_audit_counts.py`;
- `ml/evaluate_detector_counts.py`;
- `ml/evaluate_sku_coverage.py`;
- `ml/label_studio_product_config.xml`;
- `ml/prepare_product_review_dataset.py`;
- `ml/visual_reference.py`;
- `scripts/collect_reference_catalog.py`;
- `scripts/run_v8_hardneg_retrain.sh`;
- `scripts/run_v8_retrain.sh`;
- reference datasets/indexes в `data/catalog/`;
- reports в `reports/`.

---

## 8. Что делать дальше

### Ближайший практический шаг

Собрать APK через CI/Android Studio и проверить live feedback на Samsung Galaxy
Tab A9:

- исчезают ли рамки с хорошо снятых товаров;
- не слишком ли быстро товар считается `Готово`;
- корректны ли подсказки `ближе`, `медленнее`, `наведите`;
- не мешает ли нижний баннер реальной съемке;
- нужно ли менять пороги `LIVE_SHARP_*` и `LIVE_MIN_AREA_*`.

### ML-следующий шаг

Сделать ручной `sku_presence.csv` для 20-30 видео/кадров:

```csv
video,expected_sku_ids
path/to/video1.mp4,"SKU_1;SKU_2;SKU_3"
```

Потом сравнивать:

- v6 production;
- v8_hardneg;
- разные `conf`;
- разные настройки OCR/retry/visual fallback.

Главная метрика - SKU recall/precision, не object count.

### Durable fix точности детектора

Подготовить 200-300 кадров с реальной полки и вручную проверить bbox `product`.
Авторазметка может быть предзаполнением, но финальные метки должен проверить
человек. Это самый надежный путь к цели "YOLO идеально определяет товар без
дублирования и без пола/мебели".

---

## 9. Короткое сообщение для новой нейросети

Если нужно быстро продолжить:

1. Прочитай `HANDOFF.md` и этот файл.
2. Не считай `17732221864542.mp4` чистым holdout - оно уже использовалось для
   оценки/тюнинга.
3. Не заменяй production `weights/product_det_v2.pt` на v8 без SKU-presence
   проверки.
4. Помни, что бизнес-цель - уникальные SKU/артикулы, а не количество одинаковых
   товаров.
5. Последняя завершенная работа - улучшение обратной связи на планшете:
   live-состояния, счетчики, подсказки, память снятых объектов.
6. Следующий хороший шаг - собрать APK и проверить UX на реальном планшете, затем
   подкрутить пороги.


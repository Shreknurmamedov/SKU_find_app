# Detector v8: review-first workflow

Цель v8: уйти от чистой авторазметки к проверенному supervised-детектору
одного класса `product`. Авторазметка остается, но только как черновик:
модель предлагает боксы, человек быстро удаляет лишнее и дорисовывает пропуски.

## Почему не только авторазметка

Текущие v6/v7 уже показали потолок: модель качается между FP на стенах/мебели и
FN на реальном товаре. Погрешность около 7% можно честно подтвердить только на
отдельном эталонном наборе, где известно, сколько реальных товаров в кадрах.

## Подготовить review-набор

```bash
python3 -m ml.prepare_product_review_dataset \
  --source . \
  --out ml/datasets/sku_products_v8_review \
  --frames-per-video 30 \
  --max-images 250 \
  --weights weights/product_det_v2.pt \
  --max-proposals 80 \
  --device mps
```

Скрипт:

- берет реальные фото и кадры видео;
- предзаполняет боксы через `product_det_v2` и YOLO-World;
- создает `manifest.csv`, `data.yaml`, `labels/*` и `tasks/label_studio_tasks.json`;
- использует один класс `product`.

FastSAM можно включить флагом `--use-fastsam`, но первый v8-pass лучше делать
без него: он дает больше recall, но заметно больше фрагментов и лишних рамок.

Если нужно быстро проверить без тяжелых моделей:

```bash
python3 -m ml.prepare_product_review_dataset \
  --source . \
  --out /tmp/sku_products_v8_review \
  --max-images 10 \
  --frames-per-video 0 \
  --weights missing.pt \
  --no-yoloworld \
  --no-fastsam
```

## Проверить в Label Studio

```bash
python3 ml/serve_dataset.py --root ml/datasets/sku_products_v8_review --port 8099
```

В Label Studio:

1. создать проект;
2. вставить config из `ml/label_studio_product_config.xml`;
3. импортировать `ml/datasets/sku_products_v8_review/tasks/label_studio_tasks.json`;
4. проверить каждый кадр: оставить только реальные товары, удалить стены/полки,
   добавить пропущенные товары.

## Экспорт и обучение

```bash
python3 ml/convert_labelstudio_to_yolo.py \
  --export labelstudio_export.json \
  --dataset ml/datasets/sku_products_v8_review

python3 ml/build_product_dataset.py \
  --src ml/datasets/sku_products_v8_review \
  --dst ml/datasets/sku_products_v8

PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 ml/train_detector.py \
  --data ml/datasets/sku_products_v8/data.yaml \
  --imgsz 1024 \
  --batch 4 \
  --epochs 40 \
  --patience 10 \
  --name product_det_v8
```

Или через готовый wrapper:

```bash
./scripts/run_v8_retrain.sh
```

После обучения:

```bash
cp runs/detect/ml/runs/product_det_v8/weights/best.pt weights/product_det_v2.pt
python3 -m ml.audit_video --video "ТТ Пэкстрой/IMG_8886.MOV" \
  --weights weights/product_det_v2.pt --device mps --vid-stride 3 --conf 0.4 --min-frames 4
```

## Как мерить качество

Для детектора можно продолжать смотреть object-level метрики, но это только
внутренняя диагностика. Бизнес-цель проекта — SKU coverage: какие модели/артикулы
присутствуют на полке, а не сколько физических экземпляров каждой модели лежит.

Финальный validation-набор должен быть не `true_objects`, а список ожидаемых
`sku_id` по каждому видео/полке:

- SKU recall: сколько ожидаемых моделей/артикулов найдено;
- SKU precision: сколько найденных SKU действительно есть на полке;
- unresolved objects: сколько товаров видны, но бренд/артикул не читается;
- retake rate: сколько зон менеджеру нужно переснять.

Детектор считается достаточно хорошим, когда backend стабильно получает хотя бы
один читаемый кроп на каждый SKU, а лишние FP не засоряют `needs_review`.
Object count error около 7% может быть полезным техническим ориентиром, но он
не должен быть главным критерием приемки.

Проверка самого детектора на размеченном validation split:

```bash
python3 -m ml.evaluate_detector_counts \
  --data ml/datasets/sku_products_v8/data.yaml \
  --weights weights/product_det_v2.pt \
  --split val \
  --conf 0.35 \
  --device mps
```

Проверка полного video-audit по ручному списку SKU:

```bash
cp data/eval/sku_presence.example.csv data/eval/sku_presence.csv
# заполнить expected_sku_ids вручную для каждого видео
python3 -m ml.evaluate_sku_coverage \
  --truth data/eval/sku_presence.csv \
  --reports-dir reports
```

## Сбалансированные hard negatives

v7 показал, что много негативов могут переучить модель и задавить реальные
товары. Для v8 используем более мягкий вариант:

```bash
python3 -m ml.add_v8_hard_negatives \
  --base ml/datasets/sku_products_v8_auto \
  --out ml/datasets/sku_products_v8_hardneg \
  --weights weights/product_det_v8_auto.pt \
  --target-device 60 \
  --target-mined 80

PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python3 ml/train_detector.py \
  --data ml/datasets/sku_products_v8_hardneg/data.yaml \
  --model weights/product_det_v8_auto.pt \
  --imgsz 1024 \
  --batch 4 \
  --epochs 12 \
  --patience 5 \
  --name product_det_v8_hardneg
```

Или одной командой:

```bash
./scripts/run_v8_hardneg_retrain.sh
```

Боевые веса `weights/product_det_v2.pt` менять только после сравнения
`v6 / v8_auto / v8_hardneg` на одинаковых видео и hand-count CSV.

## Результаты текущего v8_hardneg кандидата

Фактический прогон 2026-06-18:

- review-набор: `ml/datasets/sku_products_v8_review`, 380 изображений/кадров,
  13 934 auto-boxes;
- auto-набор: `ml/datasets/sku_products_v8_auto`;
- hard-negative набор: `ml/datasets/sku_products_v8_hardneg`, 520 изображений,
  143 empty labels (~27%);
- веса-кандидат: `weights/product_det_v8_hardneg.pt`;
- production `weights/product_det_v2.pt` не заменялся.

Обучение hardneg остановлено после ухудшения метрик: best был на epoch 1
(`mAP50=0.632`), epoch 3 просел до `mAP50=0.519`. Это признак, что негативы уже
начинали давить товар, поэтому брать нужно только `best.pt`.

Detector count eval на `sku_products_v8_hardneg` val:

| weights | conf | gt | pred | count error | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v8_hardneg | 0.35 | 2242 | 2407 | 7.4% | 60.4% | 64.9% |
| v8_hardneg | 0.40 | 2242 | 2061 | 8.1% | 64.4% | 59.2% |
| v8_auto | 0.40 | 2242 | 2333 | 4.1% | 59.9% | 62.4% |

Важно: это auto-label validation, а не ручной SKU truth. Хороший object count error
тут может быть компенсацией FP/FN, поэтому решающая проверка должна быть по
`sku_presence.csv`.

Реальное видео `ТТ Пэкстрой/IMG_8886.MOV` с runtime-фильтрами:

| weights | conf | kept tracks | визуальный вывод |
| --- | ---: | ---: | --- |
| v8_hardneg | 0.35 | 88 | больше recall, но остается один FP пола |
| v8_hardneg | 0.40 | 81 | лучший текущий компромисс по мусору/полноте |
| v8_hardneg | 0.45 | 71 | чище, но уже заметно теряет товар |

В `ml/track_video.py` добавлены фильтры plain/low-detail background. Они нужны
как runtime safety net против плитки/стен, но не заменяют ручную разметку.

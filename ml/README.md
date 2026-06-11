# ML Pipeline

Цель первого ML-этапа: live instance segmentation для Android overlay.

## Классы

Первый baseline:

- `own_product` — наша продукция из каталога Utake: Huter, Ресанта, Вихрь, Eurolux, TEK.
- `competitor_or_unknown` — конкурент или товар, который модель не может уверенно сопоставить с нашим каталогом.

Такой baseline уже решает live-логику:

- зеленый overlay = `own_product` / recognized;
- красный overlay = `competitor_or_unknown` / unknown.

SKU-level распознавание добавляется вторым этапом на crop товара: OCR + visual retrieval + ranker по `data/catalog/own_products.csv`.

## Подготовка датасета

```bash
python3 ml/prepare_dataset.py \
  --source . \
  --out ml/datasets/sku_live \
  --label-studio-base-url http://localhost:8099
```

Скрипт:

- найдет HEIC/JPG/PNG фото;
- сконвертирует HEIC в JPG preview;
- разложит изображения на train/val;
- создаст пустые YOLO labels;
- создаст `data.yaml`;
- создаст `tasks/label_studio_tasks.json`.

## Разметка

Запустить static server для картинок:

```bash
python3 -m http.server 8099 -d ml/datasets/sku_live
```

В Label Studio:

1. Создать проект.
2. Вставить config из `ml/label_studio_config.xml`.
3. Import -> загрузить `ml/datasets/sku_live/tasks/label_studio_tasks.json`.
4. Размечать полигонами каждый видимый товар:
   - `own_product`, если это наш товар;
   - `competitor_or_unknown`, если не наш или не удалось определить.

Для настоящей segmentation-модели лучше polygon-разметка. Rectangle-разметка годится только для быстрого detection-baseline.

## Экспорт разметки в YOLO

В Label Studio: Export -> JSON.

Потом:

```bash
python3 ml/convert_labelstudio_to_yolo.py \
  --export labelstudio_export.json \
  --dataset ml/datasets/sku_live
```

## Обучение

Поставить зависимости:

```bash
python3 -m pip install -r ml/requirements.txt
```

Обучение segmentation baseline:

```bash
python3 ml/train_yolo.py \
  --data ml/datasets/sku_live/data.yaml \
  --model yolo11n-seg.pt \
  --epochs 80 \
  --imgsz 960
```

Экспорт для Android:

```bash
python3 ml/export_yolo.py \
  --weights ml/runs/sku_live/weights/best.pt \
  --format tflite
```

После экспорта модель кладется в `mobile/android/app/src/main/assets/models/`.

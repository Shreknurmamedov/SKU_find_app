# Lidar SKU Audit

MVP-проект для пересчета SKU по видео и изображениям торговых точек.

Цель системы: не просто распознать товар на одном кадре, а собрать доказательный отчет по видимым физическим объектам: бренд, SKU, признак собственного бренда, количество, уверенность, спорные зоны и кадры-доказательства.

## Текущий статус

В репозитории пока есть:

- стартовый backend-прототип пайплайна аудита;
- схемы каталога SKU и разметки;
- примеры данных;
- CLI, который уже умеет собрать первичный отчет по папкам `sku_exact_areas`, `sku_uncertain_areas` и папкам торговых точек;
- импортер DOCX-каталога продукции Utake в CSV/JSON.

Это не финальная ML-модель. Сейчас это каркас, в который последовательно подключаются detection, OCR, visual embeddings, tracking, scene mapping и human review.

## Быстрый запуск

Запуск backend API:

```bash
./scripts/run_backend.sh
```

После запуска:

- health: `http://127.0.0.1:8088/health`
- каталог: `http://127.0.0.1:8088/catalog/summary`
- список jobs: `http://127.0.0.1:8088/jobs`

Создать job по локальным фото/видео:

```bash
./scripts/create_local_job.sh
```

Старый baseline-аудит папок:

```bash
cd backend
PYTHONPATH=src python3 -m sku_audit.cli audit-images \
  --input .. \
  --output ../reports/market_photo_audit.json \
  --markdown ../reports/market_photo_audit.md
```

Импорт каталога наших товаров из DOCX:

```bash
cd backend
PYTHONPATH=src python3 -m sku_audit.cli import-docx-catalog \
  --input ../utake_products_catalog.docx \
  --output ../data/catalog/own_products.csv \
  --raw-json ../data/catalog/own_products_raw.json
```

Проверка:

```bash
cd backend
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Структура

```text
backend/                 Python backend and CLI
data/catalog/            SKU catalog examples
data/labeling/           Annotation schema and examples
docs/                    Product and engineering documentation
reports/                 Generated local reports
sku_exact_areas/         Current exact-count image examples
sku_uncertain_areas/     Current uncertain-zone image examples
var/                     Runtime uploads, previews, jobs (ignored by git)
```

## Следующий технический слой

1. Разметить фото в `ml/datasets/sku_live`.
2. Обучить segmentation baseline: `own_product` / `competitor_or_unknown`.
3. Экспортировать модель в TFLite и подключить вместо `DemoProductAnalyzer`.
4. Добавить OCR и распознавание логотипов.
5. Связать распознавание с `data/catalog/own_products.csv`.
6. Реализовать visual retrieval по embeddings.
7. Добавить tracking и дедупликацию физических объектов.

## Текущие данные

- Фото торговых точек: 115 HEIC-файлов в 7 папках.
- Видео торговых точек: 7 MOV-файлов.
- Каталог наших товаров: 2470 SKU из `utake_products_catalog.docx`.
- Каталог конкурентов: пока отсутствует; в MVP такие товары будут помечаться как `competitor_or_unknown`.

## Android MVP

Android-проект находится в `mobile/android`.

Текущий MVP на планшете:

1. Введите URL backend, например `http://192.168.1.23:8088`.
2. Введите название ТТ.
3. Выберите фото/видео.
4. Нажмите отправку.
5. Получите `job_id` и summary качества.

Подробности: `mobile/README.md`.

## ML Pipeline

Подготовить фото для разметки:

```bash
python3 ml/prepare_dataset.py \
  --source . \
  --out ml/datasets/sku_live \
  --label-studio-base-url http://localhost:8099
```

Запустить static server для Label Studio:

```bash
python3 -m http.server 8099 -d ml/datasets/sku_live
```

Разметка и обучение описаны в [docs/ml-training.md](docs/ml-training.md) и [ml/README.md](ml/README.md).

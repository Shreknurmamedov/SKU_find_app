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
```

## Следующий технический слой

1. Конвертировать/нормализовать HEIC-фото торговых точек в рабочий формат для ML.
2. Подключить реальный детектор/сегментатор товара.
3. Добавить OCR и распознавание логотипов.
4. Связать распознавание с `data/catalog/own_products.csv`.
5. Реализовать visual retrieval по embeddings.
6. Добавить tracking и дедупликацию физических объектов.
7. Собрать Android-приложение на CameraX для записи, контроля качества и загрузки материалов.

## Текущие данные

- Фото торговых точек: 115 HEIC-файлов в 7 папках.
- Каталог наших товаров: 2470 SKU из `utake_products_catalog.docx`.
- Каталог конкурентов: пока отсутствует; в MVP такие товары будут помечаться как `competitor_or_unknown`.

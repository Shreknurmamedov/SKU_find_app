# MVP: пересчет SKU по видео

## Главный принцип

Нельзя считать detections по кадрам. Система должна считать физические объекты после объединения наблюдений:

```text
video/images
  -> frames/keyframes
  -> product detections
  -> OCR/logo/visual SKU candidates
  -> tracking
  -> deduplication
  -> report with confidence and evidence
```

## Что считаем

- `sku_presence`: SKU найден хотя бы один раз.
- `item_count`: количество видимых физических товаров/упаковок.
- `brand_count`: агрегация по брендам.
- `own_brand_count`: агрегация по собственным брендам.
- `unknown`: видимый объект или зона, где бренд/SKU не подтвержден.
- `needs_retake`: область, которую надо переснять крупнее или четче.

## MVP-ограничения

- Обработка на backend, не полностью offline.
- Одна товарная категория на первом этапе.
- Ограниченный каталог: 5-10 брендов, 100-300 SKU.
- Отчет с evidence frames обязателен.
- Спорные объекты не попадают в уверенный автоматический счет.
- Пока нет каталога конкурентов, распознавание строится как `own product` / `competitor_or_unknown`.

## Первый релиз

1. Android снимает видео и отправляет его на backend.
2. Backend извлекает ключевые кадры.
3. Модель находит зоны/товары.
4. OCR и visual matching возвращают кандидатов SKU из каталога наших товаров.
5. Пайплайн формирует отчет:
   - бренд;
   - SKU;
   - наш/не наш бренд;
   - количество;
   - confidence;
   - evidence frames;
   - unknown/retake зоны.

## Метрики

- `object_recall`
- `object_precision`
- `brand_accuracy`
- `sku_top1_accuracy`
- `sku_top3_accuracy`
- `duplicate_rate`
- `unknown_rate`
- `own_brand_recall`
- `own_brand_precision`

Одна общая accuracy запрещена: она скрывает реальные ошибки.

# Backend pipeline

## Слои

1. `ingestion`: принимает видео, изображения, keyframes и метаданные съемки.
2. `quality`: оценивает смаз, освещение, размер товара в кадре, покрытие сцены.
3. `detection`: находит видимые товары или упаковки.
4. `recognition`: собирает сигналы SKU:
   - OCR;
   - barcode;
   - logo detection;
   - visual embedding retrieval;
   - category classifier.
5. `tracking`: связывает наблюдения одного объекта в соседних кадрах.
6. `deduplication`: объединяет tracklets в физические товары.
7. `reporting`: формирует итоговый отчет с evidence.

## Текущий прототип

Сейчас реализован MVP ingestion/reporting:

- читает изображения из папок;
- принимает upload файлов через `POST /jobs/upload`;
- создает processing jobs в `var/jobs`;
- конвертирует HEIC preview через системный `sips`;
- читает видео-метаданные через `ffprobe`;
- оценивает яркость, контраст и резкость кадров;
- классифицирует текущие зоны как `recognized_zone` или `needs_review_zone` по имени папки;
- формирует JSON и Markdown отчет;
- сохраняет ссылки на evidence-файлы.

Этот слой нужен, чтобы API и формат отчета были готовы до подключения ML.

## HTTP endpoints

- `GET /health`
- `GET /catalog/summary`
- `POST /audit/images`
- `POST /jobs/from-local`
- `POST /jobs/upload`
- `GET /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/report.md`

## Будущие интерфейсы моделей

```json
{
  "observation_id": "obs_000001",
  "frame_id": "frame_00183",
  "bbox": [120, 44, 380, 260],
  "mask_ref": "masks/frame_00183_obs_000001.png",
  "brand_candidates": [
    {"brand_id": "huter", "probability": 0.91}
  ],
  "sku_candidates": [
    {"sku_id": "HUTER_EXAMPLE_001", "probability": 0.74}
  ],
  "world_position": [1.2, 0.5, 2.8],
  "confidence": 0.82
}
```

## Правило дедупликации

Один товар считается один раз, если совпадает хотя бы один сильный сигнал или несколько слабых:

- тот же `track_id`;
- близкая позиция в карте сцены;
- высокая visual similarity;
- совместимые SKU/brand candidates;
- перекрытие масок в scene map;
- временная непрерывность.

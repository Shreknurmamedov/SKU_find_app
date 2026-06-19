# Product Guard Classifier

Цель: убрать ложные live-рамки на планшете, когда маленькая YOLO-модель принимает
стул, кулер, дверь, стену или мебель за товар.

## Архитектура

Live pipeline на планшете теперь двухступенчатый:

1. `TFLiteProductAnalyzer` находит кандидаты `product`.
2. `TFLiteProductGuard` классифицирует каждый crop:
   - `interior` = комната/мебель/фон;
   - `product` = товарная упаковка/инструмент.

Только crop с `product_prob >= LIVE_GUARD_MIN_PRODUCT` попадает в оверлей и
`LiveCaptureTracker`. Для попадания в `Готово` нужен более высокий порог:
`LIVE_GUARD_CAPTURE_PRODUCT`.

Если asset `product_guard_cls_float32.tflite` отсутствует, приложение работает
по старой схеме: YOLO + эвристики.

## Данные

Позитивы:

- эталонные фото из `data/catalog/reference_dataset_all/training_images.csv`;
- реальные crop-ы Huter/инструментов из планшетных и ТТ-фото:
  `var/tablet/product_guard_positive`.

Негативы:

- crop-ы комнаты из записей планшета:
  `var/tablet/sku_live_debug.mp4`, `var/tablet/cap4.mp4`, `var/tablet/cap_live.mp4`.
- curated hard-negative crop-ы из реального офиса/ТТ:
  `var/tablet/product_guard_negative`.

Свежие кропы майнятся так:

```bash
python3 -m ml.mine_product_guard_crops \
  var/photo_review_20260619/converted/м9 \
  --out var/tablet/guard_mining_m9_20260619 \
  --detector weights/product_det_v8_hardneg.pt \
  --guard weights/product_guard_cls.pt \
  --imgsz 960 \
  --conf 0.18 \
  --device mps
```

После майнинга нужно смотреть contact sheets и переносить только чистые crop-ы в
`product_guard_positive` / `product_guard_negative`; сырые buckets нельзя
автоматически считать истиной.

Датасет создаётся в формате ImageFolder:

```text
ml/datasets/product_guard_cls/
  train/interior
  train/product
  val/interior
  val/product
```

## Команды

```bash
python3 -m ml.build_product_guard_dataset \
  --out ml/datasets/product_guard_cls \
  --positive-limit 1600 \
  --negative-limit 1600 \
  --crops-per-frame 16

python3 -m ml.train_product_guard \
  --data ml/datasets/product_guard_cls \
  --epochs 12 \
  --batch 64 \
  --device mps \
  --name product_guard_hardneg_20260619
```

Результат:

```text
weights/product_guard_cls.pt
```

Локальный экспорт в TFLite на macOS + Python 3.13 не работает из-за ограничения
Ultralytics/TensorFlow Lite. Экспорт делается в GitHub Actions через:

```bash
python ml/export_product_guard_tflite.py
```

CI положит asset сюда:

```text
mobile/android/app/src/main/assets/models/product_guard_cls_float32.tflite
```

Важно: `weights/product_guard_cls.pt` должен быть закоммичен, иначе CI соберёт APK
без guard asset.

## Проверка 2026-06-19

Собраны новые данные:

- `м9`: 420 curated product crop-ов и 355 curated interior/hard-negative;
- офисная планшетная raw-запись: 63 curated Huter product crop-а и 180
  interior/hard-negative;
- итоговые extra-папки: 487 positive и 535 negative crop-ов.

Сборка датасета:

- train: 3372 изображений;
- val: 594 изображения;
- `interior_extra`: 455 train / 80 val.

Обучение `product_guard_hardneg_20260619`:

- best validation top1: `0.995`;
- итоговый вес: `weights/product_guard_cls.pt` (~3.2 MB).

Probe старой и новой модели на today's curated crop-ах:

| set | old pass @0.75 | new pass @0.75 |
| --- | ---: | ---: |
| `m9_product` | 21.7% | 98.1% |
| `m9_interior` false accept | 6.5% | 0.6% |
| `office_product` | 93.7% | 98.4% |
| `office_interior` false accept | 6.7% | 0.0% |

Отчёт: `reports/product_guard_hardneg_probe_20260619.json`.

Это не финальная гарантия качества: следующий обязательный шаг — собрать APK в CI,
установить на Samsung Tab A9 и повторить live-тест в офисе/ТТ. Локальный TFLite
export на macOS + Python 3.13 ожидаемо падает из-за ограничения
Ultralytics/TensorFlow Lite; CI экспортирует на Linux/Python 3.11.

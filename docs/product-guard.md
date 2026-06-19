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
- реальные crop-ы Huter из планшетной записи:
  `var/tablet/product_guard_positive`.

Негативы:

- crop-ы комнаты из записей планшета:
  `var/tablet/sku_live_debug.mp4`, `var/tablet/cap4.mp4`, `var/tablet/cap_live.mp4`.

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
  --positive-limit 1200 \
  --negative-limit 1200 \
  --crops-per-frame 24

python3 -m ml.train_product_guard \
  --data ml/datasets/product_guard_cls \
  --epochs 10 \
  --batch 32 \
  --device mps
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

Локальный probe на живых crop-ах:

- Huter дальний/крупный: `product_prob ~0.999`;
- кулер, корпус кулера, шкаф, дверь, ручка двери, стул: `product_prob ~0.000`.

Это не финальная гарантия качества: следующий обязательный шаг — собрать APK в CI,
установить на Samsung Tab A9 и повторить 25-секундный комнатный тест.


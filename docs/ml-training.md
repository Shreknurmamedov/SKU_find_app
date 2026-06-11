# ML Training Plan

## Почему сначала не SKU-level

На текущем датасете еще нет разметки каждого физического товара. Для live overlay на планшете первый полезный слой:

1. найти каждый видимый товар;
2. сегментировать его;
3. присвоить статус:
   - `own_product` -> зеленый;
   - `competitor_or_unknown` -> красный.

Конкретный SKU распознается вторым этапом по crop товара: OCR, logo detection, visual retrieval и сверка с `data/catalog/own_products.csv`.

## Что размечать

Каждый видимый товар/упаковку, даже если он частично перекрыт.

Классы:

- `own_product`: Huter, Ресанта, Вихрь, Eurolux, TEK и другие строки из каталога Utake.
- `competitor_or_unknown`: конкурент, непонятная упаковка, товар с нечитаемым брендом.

Для segmentation лучше рисовать polygon по границе товара. Если нужно быстрее получить baseline, можно разметить прямоугольниками, но качество сегментации будет хуже.

## Сколько нужно фото

Для первого baseline:

- минимум 200-300 фото;
- лучше 1000+ объектов в разметке;
- обязательно разные ТТ, бликующие упаковки, перекрытия, хаотичная выкладка.

Ваши дополнительные 156 фото стоит добавить сразу. После этого можно разметить первую партию и обучить baseline.

## Команды

```bash
python3 ml/prepare_dataset.py --source . --out ml/datasets/sku_live
python3 -m http.server 8099 -d ml/datasets/sku_live
```

После разметки:

```bash
python3 ml/convert_labelstudio_to_yolo.py \
  --export labelstudio_export.json \
  --dataset ml/datasets/sku_live

python3 -m pip install -r ml/requirements.txt
python3 ml/train_yolo.py --data ml/datasets/sku_live/data.yaml --model yolo11n-seg.pt
```

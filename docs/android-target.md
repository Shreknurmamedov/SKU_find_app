# Android target

## Текущее целевое устройство

- Модель: Samsung Galaxy Tab A9 SM-X115
- Память: 8 / 128 ГБ
- Android: 13

## Решение для MVP

Для первого релиза приложение можно делать под Android 13 с запасом вниз до Android 10-11, если не появятся более старые устройства.

Рекомендуемый стек:

- Kotlin;
- CameraX для записи видео и анализа кадров;
- online backend processing как основной режим;
- on-device quality checks: смаз, темнота, скорость движения, примерный размер товара в кадре;
- загрузка HEIC/JPEG/video на backend;
- отчет с зонами пересъемки.

## Live overlay MVP

Текущий Android MVP использует CameraX preview + ImageAnalysis + overlay:

```text
CameraX Preview
  + ImageAnalysis frames
  -> ProductAnalyzer
  -> ProductDetection[]
  -> ProductOverlayView
```

Цвета:

- зеленый: `recognized`;
- красный: `unknown`.

Пока подключен `DemoProductAnalyzer`, который проверяет live overlay без настоящей ML-модели. Реальная модель должна возвращать те же `ProductDetection` с нормализованными координатами сегмента/объекта, label и confidence.

## Почему не весь ML на планшете сразу

Galaxy Tab A9 подходит для съемки, подсказок оператору и легких проверок качества. Для надежного SKU-распознавания, OCR, embeddings и дедупликации лучше начинать с backend. После стабилизации моделей часть логики можно перенести на устройство.

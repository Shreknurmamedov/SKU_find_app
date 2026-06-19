# Model Assets

Generated Android TFLite assets are copied here by CI before APK build.

Expected live files:

- `product_det_v2_float32.tflite` — one-class YOLO detector, finds product
  candidates on-device.
- `product_guard_cls_float32.tflite` — optional binary guard classifier:
  `interior` vs `product`. If absent, the app falls back to detector +
  heuristic filtering.

Full SKU recognition (OCR/catalog/reference search) still runs on the backend.

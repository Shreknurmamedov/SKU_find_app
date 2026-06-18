#!/usr/bin/env bash
# Запустить ночной прогон самоулучшения и уйти спать.
#
#   1) Заполни разметку: data/eval/sku_presence.csv  (video,expected_sku_ids)
#   2) ./scripts/run_overnight.sh
#   3) Утром открой: reports/overnight/SUMMARY.md
#
# Скрипт переживает закрытие терминала (nohup) и пишет полный лог.
set -euo pipefail
cd "$(dirname "$0")/.."

TRUTH="${TRUTH:-data/eval/sku_presence.csv}"
DEVICE="${SKU_AUDIT_DEVICE:-mps}"
EPOCHS="${EPOCHS:-40}"
LOG="reports/overnight/run.console.log"

if [[ ! -f "$TRUTH" ]]; then
  echo "НЕТ разметки: $TRUTH"
  echo "Заполни её (пример: data/eval/sku_presence.example.csv), потом запусти снова."
  exit 1
fi

mkdir -p reports/overnight
echo "Старт ночного прогона. Лог: $LOG"
echo "Утром: reports/overnight/SUMMARY.md"

# MPS любит этот флаг, чтобы не упереться в память на длинном обучении.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"

nohup python3 -m ml.overnight_improve \
  --truth "$TRUTH" --device "$DEVICE" --epochs "$EPOCHS" \
  >> "$LOG" 2>&1 &

echo "PID: $!  (можно закрыть терминал; прогон продолжится)"

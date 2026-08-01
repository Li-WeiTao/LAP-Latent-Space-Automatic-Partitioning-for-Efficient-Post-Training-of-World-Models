#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/sicong/weitao/LAP-Latent-Space-Auto-Partitioned-Fine-Tuning-for-World-Models"
cd "$ROOT"
PYTHON="${PYTHON:-python}"

echo "[formal-smoke] unit tests"
$PYTHON -m pytest tests/test_region_conditional_risk.py tests/test_formal_region_risk_pipeline.py -q

WORK="experiments/control_matrix/assets/formal_region_risk/smoke/pusht"
echo "[formal-smoke] split + dry-run for PushT"
$PYTHON experiments/control_matrix/formal_region_risk_pipeline.py \
  --task pusht \
  --work-root "$WORK" \
  --phase split \
  --max-train-starts 512 \
  --max-eval-starts 128

$PYTHON experiments/control_matrix/formal_region_risk_pipeline.py \
  --task pusht \
  --work-root "$WORK" \
  --phase dry-run \
  --train-seeds 0 \
  --bootstrap-reps 100 \
  --max-anchors 32 \
  --max-episodes 4

echo "[formal-smoke] done -> $WORK"

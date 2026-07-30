#!/usr/bin/env bash
# Train full-dataset region predictors with fixed TwoRoom task geometry.
# Does NOT start automatically — run manually when ready.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

OUT_DIR="experiments/tworoom/results/tworoom_geometry_trajectory_predictors"
CKPT="${LAP_LEWM_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt}"

mkdir -p "${OUT_DIR}"

echo "==== geometry region predictors (full dataset) started at $(date) ===="
/usr/bin/time -p python experiments/tworoom/trajectory.py \
  --region-split-mode geometry \
  --checkpoint "${CKPT}" \
  --out-dir "${OUT_DIR}" \
  --history-size 3 \
  --num-preds 1 \
  --frameskip 0 \
  --img-size 224 \
  --train-fraction 0.9 \
  --split-seed 3072 \
  --seed 42 \
  --batch-size 128 \
  --epochs 30 \
  --lr 5e-05 \
  --weight-decay 0.001 \
  --force-reencode \
  --device cuda \
  > "${OUT_DIR}/train.log" 2>&1

echo "==== geometry region predictors finished at $(date) ===="

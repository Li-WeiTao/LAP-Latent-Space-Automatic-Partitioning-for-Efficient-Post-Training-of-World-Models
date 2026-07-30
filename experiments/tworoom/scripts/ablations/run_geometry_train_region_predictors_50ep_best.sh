#!/usr/bin/env bash
# Retrain doorway / left / right region predictors for 50 epochs.
# Saves checkpoints at epochs 20,30,40,50; final P_train_* uses min eval_loss epoch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

OUT_DIR="experiments/tworoom/results/tworoom_geometry_train_region_predictors"
CKPT="${LAP_LEWM_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt}"
LOG="${OUT_DIR}/train_50ep_doorway_left_right.log"

mkdir -p "${OUT_DIR}"

echo "==== 50-epoch retrain (doorway, left, right) started at $(date) ===="
/usr/bin/time -p python experiments/tworoom/trajectory.py \
  --region-split-mode geometry \
  --restrict-to-train-split \
  --checkpoint "${CKPT}" \
  --out-dir "${OUT_DIR}" \
  --regions doorway_corridor left_room right_room \
  --history-size 3 \
  --num-preds 1 \
  --frameskip 0 \
  --img-size 224 \
  --train-fraction 0.9 \
  --split-seed 3072 \
  --seed 42 \
  --batch-size 128 \
  --epochs 50 \
  --lr 5e-05 \
  --weight-decay 0.001 \
  --save-epochs 20,30,40,50 \
  --select-best-by-eval \
  --device cuda \
  > "${LOG}" 2>&1

echo "==== 50-epoch retrain finished at $(date) ===="

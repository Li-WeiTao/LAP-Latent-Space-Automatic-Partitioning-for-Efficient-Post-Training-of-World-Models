#!/usr/bin/env bash
# Retrain doorway_corridor for 80 epochs; select best by eval_loss.
# Saves checkpoints at epochs 20,30,40,50,60,70,80.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

OUT_DIR="experiments/tworoom/results/tworoom_geometry_train_region_predictors_doorway80ep"
CKPT="${LAP_LEWM_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt}"
LOG="${OUT_DIR}/train_80ep_doorway.log"

# Copy existing embeddings cache so we don't re-encode
SRC="experiments/tworoom/results/tworoom_geometry_train_region_predictors"
mkdir -p "${OUT_DIR}"
cp -n "${SRC}/P_train_doorway_corridor_embeddings.npz" "${OUT_DIR}/" 2>/dev/null || true
cp -n "${SRC}/train_global_reference_starts.npy" "${OUT_DIR}/" 2>/dev/null || true
cp -n "${SRC}/geometry_region_thresholds.npy" "${OUT_DIR}/" 2>/dev/null || true

mkdir -p "${OUT_DIR}"

echo "==== doorway_corridor 80-epoch retrain started at $(date) ===="
/usr/bin/time -p python experiments/tworoom/trajectory.py \
  --region-split-mode geometry \
  --restrict-to-train-split \
  --checkpoint "${CKPT}" \
  --out-dir "${OUT_DIR}" \
  --regions doorway_corridor \
  --history-size 3 \
  --num-preds 1 \
  --frameskip 0 \
  --img-size 224 \
  --train-fraction 0.9 \
  --split-seed 3072 \
  --seed 42 \
  --batch-size 128 \
  --epochs 80 \
  --lr 5e-05 \
  --weight-decay 0.001 \
  --select-best-by-eval \
  --device cuda \
  > "${LOG}" 2>&1

echo "==== doorway_corridor 80-epoch retrain finished at $(date) ===="

#!/usr/bin/env bash
# Fine-tune doorway_corridor in doorway80ep output dir (single job).
# Usage: GPU=3 EPOCHS=80 bash run_geometry_train_doorway_one_region.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

ROOT="experiments/tworoom"
GPU="${GPU:?set GPU=0..N}"
EPOCHS="${EPOCHS:-80}"
SEED="${SEED:-42}"
SRC="${ROOT}/results/tworoom_geometry_train_region_predictors"
OUT_DIR="${OUT_DIR:-${ROOT}/results/tworoom_geometry_train_region_predictors_doorway80ep}"
CKPT="${LAP_LEWM_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt}"
LOG="${OUT_DIR}/train_${EPOCHS}ep_doorway.log"

mkdir -p "${OUT_DIR}"
cp --update=none "${SRC}/P_train_doorway_corridor_embeddings.npz" "${OUT_DIR}/" 2>/dev/null || true
cp --update=none "${SRC}/train_global_reference_starts.npy" "${OUT_DIR}/" 2>/dev/null || true
cp --update=none "${SRC}/geometry_region_thresholds.npy" "${OUT_DIR}/" 2>/dev/null || true

export CUDA_VISIBLE_DEVICES="${GPU}"
echo "==== GPU=${GPU} doorway80ep EPOCHS=${EPOCHS} SEED=${SEED} started at $(date) ====" | tee "${LOG}"

/usr/bin/time -p python "${ROOT}/trajectory.py" \
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
  --seed "${SEED}" \
  --batch-size 128 \
  --epochs "${EPOCHS}" \
  --lr 5e-05 \
  --weight-decay 0.001 \
  --select-best-by-eval \
  --device cuda \
  >> "${LOG}" 2>&1

echo "==== GPU=${GPU} doorway80ep finished at $(date) ====" | tee -a "${LOG}"

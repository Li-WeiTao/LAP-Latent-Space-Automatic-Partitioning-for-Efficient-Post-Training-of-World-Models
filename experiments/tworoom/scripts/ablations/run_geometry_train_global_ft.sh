#!/usr/bin/env bash
# Global-FT compute-matched predictor (single job).
# Usage: GPU=4 EPOCHS=65 bash run_geometry_train_global_ft.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

ROOT="experiments/tworoom"
GPU="${GPU:?set GPU=0..N}"
EPOCHS="${EPOCHS:-65}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-${ROOT}/results/tworoom_geometry_train_global_ft_65ep}"
EMBED_DIR="${EMBED_DIR:-${ROOT}/results/tworoom_geometry_train_region_predictors}"
CKPT="${LAP_LEWM_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt}"
LOG="${OUT_DIR}/train_global_ft_${EPOCHS}ep.log"

mkdir -p "${OUT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU}"

echo "==== GPU=${GPU} global_ft EPOCHS=${EPOCHS} SEED=${SEED} started at $(date) ====" | tee "${LOG}"

/usr/bin/time -p python "${ROOT}/trajectory.py" \
  --region-split-mode geometry \
  --restrict-to-train-split \
  --train-global-predictor \
  --global-predictor-name global_ft \
  --checkpoint "${CKPT}" \
  --out-dir "${OUT_DIR}" \
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
  --embedding-source-dir "${EMBED_DIR}" \
  --select-best-by-eval \
  --device cuda \
  >> "${LOG}" 2>&1

echo "==== GPU=${GPU} global_ft finished at $(date) ====" | tee -a "${LOG}"

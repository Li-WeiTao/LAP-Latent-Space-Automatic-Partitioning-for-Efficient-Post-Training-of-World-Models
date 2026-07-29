#!/usr/bin/env bash
# Experiment 8: Global-FT compute-matched predictor (65 epochs on full train split).
# Equivalent to priority5 (50ep) regional FT budget (~64.2 weighted global epochs).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

GPU="${CUDA_VISIBLE_DEVICES:-4}"
export CUDA_VISIBLE_DEVICES="${GPU}"

OUT_DIR="experiments/tworoom/results/tworoom_geometry_train_global_ft_65ep"
CKPT="${LAP_LEWM_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt}"
LOG="${OUT_DIR}/train_global_ft_65ep.log"

mkdir -p "${OUT_DIR}"

echo "==== Global-FT 65ep (compute-matched) started at $(date) ===="
/usr/bin/time -p python experiments/tworoom/trajectory.py \
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
  --seed 42 \
  --batch-size 128 \
  --epochs 65 \
  --lr 5e-05 \
  --weight-decay 0.001 \
  --embedding-source-dir experiments/tworoom/results/tworoom_geometry_train_region_predictors \
  --select-best-by-eval \
  --device cuda \
  > "${LOG}" 2>&1

echo "==== Global-FT 65ep finished at $(date) ===="

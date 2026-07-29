#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

GPU="${GPU:?set GPU to one available physical GPU id}"
TRAIN_SEED="${TRAIN_SEED:?set TRAIN_SEED}"
export CUDA_VISIBLE_DEVICES="${GPU}"

ROOT=experiments/tworoom
OUT_DIR="${ROOT}/results/tworoom_joint_continue_3ep_trainseed${TRAIN_SEED}"
LOG="${ROOT}/results/tworoom_joint_continue_3ep_trainseed${TRAIN_SEED}_train.log"

if [[ -e "${OUT_DIR}" || -e "${LOG}" ]]; then
  echo "Refusing to overwrite ${OUT_DIR} or ${LOG}" >&2
  exit 2
fi
mkdir -p "${OUT_DIR}"

/usr/bin/time -p nice -n 10 python "${ROOT}/joint_continue_tworoom.py" \
  --out-dir "${OUT_DIR}" \
  --seed "${TRAIN_SEED}" \
  --split-seed 3072 \
  --epochs 3 \
  --num-workers 4 \
  --cpu-threads 4 \
  --precision fp32 \
  --device cuda \
  >> "${LOG}" 2>&1


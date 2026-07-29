#!/usr/bin/env bash
# Fine-tune ONE geometry-region predictor (seed=42, reuse shared embedding cache).
#
# Parallel-safe: writes to per-region WORK_DIR, then consolidates ckpts to MAIN_DIR.
#
# Usage:
#   GPU=0 REGION=common EPOCHS=30 bash run_geometry_train_one_region.sh
#   GPU=1 REGION=left_room EPOCHS=50 SELECT_BEST=1 SAVE_EPOCHS=20,30,40,50 bash ...
#
# Env:
#   GPU              CUDA device id (required)
#   REGION           common | doorway_corridor | left_room | near_wall | right_room
#   EPOCHS           default 30
#   MAIN_DIR         shared cache + final ckpts (default tworoom_geometry_train_region_predictors)
#   WORK_DIR         override per-job work dir (default MAIN_DIR/_work/${REGION}_${EPOCHS}ep_seed${SEED})
#   SEED             default 42
#   SELECT_BEST      1 → --select-best-by-eval
#   SAVE_EPOCHS      e.g. 20,30,40,50 (optional)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

ROOT="experiments/tworoom"
GPU="${GPU:?set GPU=0..N}"
REGION="${REGION:?set REGION=common|doorway_corridor|left_room|near_wall|right_room}"
EPOCHS="${EPOCHS:-30}"
SEED="${SEED:-42}"
MAIN_DIR="${MAIN_DIR:-${ROOT}/results/tworoom_geometry_train_region_predictors}"
WORK_DIR="${WORK_DIR:-${MAIN_DIR}/_work/${REGION}_${EPOCHS}ep_seed${SEED}}"
CKPT="${LAP_LEWM_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt}"
LOG="${MAIN_DIR}/train_${EPOCHS}ep_${REGION}.log"

mkdir -p "${MAIN_DIR}" "${WORK_DIR}"

link_shared_caches() {
  local f base
  for f in "${MAIN_DIR}"/P_train_*_embeddings.npz; do
    [[ -e "${f}" ]] || continue
    base="$(basename "${f}")"
    ln -sfn "$(realpath "${f}")" "${WORK_DIR}/${base}"
  done
  # Do NOT symlink train_global_reference_starts.npy — trajectory.py writes it;
  # each job must own a private copy under WORK_DIR.
}

consolidate_checkpoints() {
  local f pat
  shopt -s nullglob
  local -a patterns=(
    "P_train_${REGION}_object.ckpt"
    "P_train_${REGION}_object.json"
    "P_train_${REGION}_starts.npy"
  )
  for pat in "${patterns[@]}"; do
    if [[ -f "${WORK_DIR}/${pat}" ]]; then
      cp -a "${WORK_DIR}/${pat}" "${MAIN_DIR}/"
      echo "[consolidate] ${WORK_DIR}/${pat} -> ${MAIN_DIR}/${pat}"
    fi
  done
  for f in \
    "${WORK_DIR}/P_train_${REGION}_epoch"*_object.ckpt \
    "${WORK_DIR}/P_train_${REGION}_epoch"*_object.json; do
    [[ -f "${f}" ]] || continue
    cp -a "${f}" "${MAIN_DIR}/"
    echo "[consolidate] ${f} -> ${MAIN_DIR}/$(basename "${f}")"
  done
}

link_shared_caches

EXTRA=()
if [[ "${SELECT_BEST:-0}" == "1" ]]; then
  EXTRA+=(--select-best-by-eval)
fi
if [[ -n "${SAVE_EPOCHS:-}" ]]; then
  EXTRA+=(--save-epochs "${SAVE_EPOCHS}")
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
echo "==== GPU=${GPU} REGION=${REGION} EPOCHS=${EPOCHS} SEED=${SEED} WORK_DIR=${WORK_DIR} started at $(date) ====" | tee "${LOG}"

/usr/bin/time -p python "${ROOT}/trajectory.py" \
  --region-split-mode geometry \
  --restrict-to-train-split \
  --checkpoint "${CKPT}" \
  --out-dir "${WORK_DIR}" \
  --regions "${REGION}" \
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
  --device cuda \
  "${EXTRA[@]}" \
  >> "${LOG}" 2>&1

consolidate_checkpoints

echo "==== GPU=${GPU} REGION=${REGION} finished at $(date) ====" | tee -a "${LOG}"

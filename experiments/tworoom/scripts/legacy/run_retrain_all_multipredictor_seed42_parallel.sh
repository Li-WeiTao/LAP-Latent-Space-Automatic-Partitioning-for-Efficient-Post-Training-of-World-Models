#!/usr/bin/env bash
# Parallel multipredictor retrain: one GPU per independent FT job (seed=42, reuse embeddings).
# Usage: SKIP_ARCHIVE=1 bash run_retrain_all_multipredictor_seed42_parallel.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

ROOT="experiments/tworoom"
STAMP="pre_seed42_20260713"
GEOM_DIR="${ROOT}/results/tworoom_geometry_train_region_predictors"
GEOM_ARCHIVE="${ROOT}/results/tworoom_geometry_train_region_predictors_${STAMP}"
MASTER="${ROOT}/results/retrain_multipredictor_seed42_parallel_master.log"
SKIP_ARCHIVE="${SKIP_ARCHIVE:-0}"
RUN_EVAL="${RUN_EVAL:-0}"

# Default GPU pool (override: GPUS="0 1 2 3 4")
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4}"

log() { echo "$*" | tee -a "${MASTER}"; }

wait_all() {
  local label="$1"
  shift
  local pids=("$@")
  local fail=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      log "[fail] ${label} pid=${pid}"
      fail=1
    fi
  done
  if [[ "${fail}" -ne 0 ]]; then
    log "==== ABORT: ${label} had failures ===="
    exit 1
  fi
  log "[done] ${label}"
}

archive_if_exists() {
  local name="$1"
  local src="${ROOT}/results/${name}"
  local dst="${ROOT}/results/${name}_${STAMP}"
  if [[ -d "${src}" && ! -e "${dst}" ]]; then
    log "[archive] ${src} -> ${dst}"
    mv "${src}" "${dst}"
  fi
}

restore_geometry_embedding_caches() {
  mkdir -p "${GEOM_DIR}"
  if [[ ! -d "${GEOM_ARCHIVE}" ]]; then
    log "[restore] no archive; encode only if caches missing"
    return 0
  fi
  log "[restore] copy embedding caches from ${GEOM_ARCHIVE}"
  cp -f "${GEOM_ARCHIVE}"/P_train_*_embeddings.npz "${GEOM_DIR}/"
  cp -f "${GEOM_ARCHIVE}/train_global_reference_starts.npy" "${GEOM_DIR}/"
  cp -f "${GEOM_ARCHIVE}/geometry_region_thresholds.npy" "${GEOM_DIR}/"
}

train_one_region() {
  local gpu="$1"
  local region="$2"
  local epochs="$3"
  local logfile="$4"
  shift 4
  mkdir -p "$(dirname "${logfile}")"
  log "[launch] GPU=${gpu} region=${region} epochs=${epochs} -> ${logfile}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    /usr/bin/time -p python "${ROOT}/trajectory.py" \
      --region-split-mode geometry \
      --restrict-to-train-split \
      --checkpoint "${LAP_LEWM_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt}" \
      --out-dir "${GEOM_DIR}" \
      --regions "${region}" \
      --history-size 3 \
      --num-preds 1 \
      --frameskip 0 \
      --img-size 224 \
      --train-fraction 0.9 \
      --split-seed 3072 \
      --seed 42 \
      --batch-size 128 \
      --epochs "${epochs}" \
      --lr 5e-05 \
      --weight-decay 0.001 \
      --device cuda \
      "$@"
  ) > "${logfile}" 2>&1 &
  echo $!
}

mkdir -p "${ROOT}/results"
log "==== parallel seed42 RETRAIN started at $(date) GPUs=${GPUS[*]} ===="

if [[ "${SKIP_ARCHIVE}" != "1" ]]; then
  archive_if_exists "tworoom_geometry_train_region_predictors"
  archive_if_exists "tworoom_geometry_train_region_predictors_doorway80ep"
  archive_if_exists "tworoom_geometry_train_global_ft_65ep"
  archive_if_exists "tworoom_latent_kmeanspp_kmeanspp_R50_outer0"
fi
restore_geometry_embedding_caches

# ---- Phase 1a: 30ep × 5 regions (5 GPUs) ----
log "---- Phase 1a: 30ep × 5 regions in parallel ----"
REGIONS_30=(common doorway_corridor left_room near_wall right_room)
PIDS_1A=()
for i in "${!REGIONS_30[@]}"; do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  r="${REGIONS_30[$i]}"
  pid=$(train_one_region "${gpu}" "${r}" 30 \
    "${GEOM_DIR}/train_30ep_${r}.log")
  PIDS_1A+=("${pid}")
done
wait_all "Phase 1a" "${PIDS_1A[@]}"

# ---- Phase 1b: 50ep best × 3 regions (GPUs 0-2) ----
log "---- Phase 1b: 50ep best (doorway, left, right) in parallel ----"
PIDS_1B=()
REGIONS_50=(doorway_corridor left_room right_room)
for i in "${!REGIONS_50[@]}"; do
  gpu="${GPUS[$i]}"
  r="${REGIONS_50[$i]}"
  pid=$(train_one_region "${gpu}" "${r}" 50 \
    "${GEOM_DIR}/train_50ep_${r}.log" \
    --save-epochs 20,30,40,50 \
    --select-best-by-eval)
  PIDS_1B+=("${pid}")
done
wait_all "Phase 1b" "${PIDS_1B[@]}"

# ---- Phase 1c + 1d in parallel (doorway 80ep + global 65ep) ----
log "---- Phase 1c+1d: doorway 80ep + global FT 65ep in parallel ----"
DOOR80_DIR="${ROOT}/results/tworoom_geometry_train_region_predictors_doorway80ep"
GLOBAL_DIR="${ROOT}/results/tworoom_geometry_train_global_ft_65ep"
mkdir -p "${DOOR80_DIR}" "${GLOBAL_DIR}"
cp -f "${GEOM_DIR}/P_train_doorway_corridor_embeddings.npz" "${DOOR80_DIR}/"
cp -f "${GEOM_DIR}/train_global_reference_starts.npy" "${DOOR80_DIR}/"
cp -f "${GEOM_DIR}/geometry_region_thresholds.npy" "${DOOR80_DIR}/"

log "[launch] GPU=${GPUS[3]:-3} doorway 80ep"
(
  export CUDA_VISIBLE_DEVICES="${GPUS[3]:-3}"
  /usr/bin/time -p python "${ROOT}/trajectory.py" \
    --region-split-mode geometry \
    --restrict-to-train-split \
    --checkpoint "${LAP_LEWM_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt}" \
    --out-dir "${DOOR80_DIR}" \
    --regions doorway_corridor \
    --split-seed 3072 --seed 42 \
    --epochs 80 --batch-size 128 --lr 5e-05 --weight-decay 0.001 \
    --history-size 3 --num-preds 1 --frameskip 0 --img-size 224 \
    --train-fraction 0.9 --select-best-by-eval --device cuda \
    > "${DOOR80_DIR}/train_80ep_doorway.log" 2>&1
) &
PID_1C=$!

log "[launch] GPU=${GPUS[4]:-4} global FT 65ep"
(
  export CUDA_VISIBLE_DEVICES="${GPUS[4]:-4}"
  /usr/bin/time -p python "${ROOT}/trajectory.py" \
    --region-split-mode geometry \
    --restrict-to-train-split \
    --train-global-predictor \
    --global-predictor-name global_ft \
    --checkpoint "${LAP_LEWM_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt}" \
    --out-dir "${GLOBAL_DIR}" \
    --split-seed 3072 --seed 42 \
    --epochs 65 --batch-size 128 --lr 5e-05 --weight-decay 0.001 \
    --history-size 3 --num-preds 1 --frameskip 0 --img-size 224 \
    --train-fraction 0.9 \
    --embedding-source-dir "${GEOM_DIR}" \
    --select-best-by-eval --device cuda \
    > "${GLOBAL_DIR}/train_global_ft_65ep.log" 2>&1
) &
PID_1D=$!

wait_all "Phase 1c+1d" "${PID_1C}" "${PID_1D}"

# ---- Phase 1e: kmeanspp outer 0/1/2 (GPUs 0-2) ----
log "---- Phase 1e: latent kmeanspp FT outer 0/1/2 in parallel ----"
PIDS_1E=()
for i in 0 1 2; do
  gpu="${GPUS[$i]}"
  log "[launch] GPU=${gpu} kmeanspp outer=${i}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    OUTER_SEED="${i}" bash "${ROOT}/scripts/internal/run_latent_kmeanspp_train_predictors_50ep.sh"
  ) > "${ROOT}/results/retrain_kmeanspp_outer${i}.log" 2>&1 &
  PIDS_1E+=($!)
done
wait_all "Phase 1e" "${PIDS_1E[@]}"

if [[ "${RUN_EVAL}" == "1" ]]; then
  log "---- Phase 2 eval (sequential; mostly CPU/env bound) ----"
  bash "${ROOT}/scripts/ablations/run_success_rate_5seed.sh" 2>&1 | tee -a "${MASTER}"
  bash "${ROOT}/scripts/ablations/run_success_rate_5seed_50ep_rooms3_priority5.sh" 2>&1 | tee -a "${MASTER}"
  bash "${ROOT}/scripts/ablations/run_success_rate_5seed_exp5_priority5.sh" 2>&1 | tee -a "${MASTER}"
  bash "${ROOT}/scripts/ablations/run_success_rate_5seed_exp6_longrange.sh" 2>&1 | tee -a "${MASTER}"
  bash "${ROOT}/scripts/ablations/run_success_rate_5seed_exp6_longrange_30ep_50ep.sh" 2>&1 | tee -a "${MASTER}"
  bash "${ROOT}/scripts/ablations/run_success_rate_5seed_exp8_global_ft.sh" 2>&1 | tee -a "${MASTER}"
  bash "${ROOT}/scripts/ablations/run_success_rate_5seed_exp8_global_ft_longrange.sh" 2>&1 | tee -a "${MASTER}"
  for i in 0 1 2; do
    OUTER_SEED="${i}" bash "${ROOT}/scripts/internal/run_success_rate_5seed_latent_kmeanspp_longrange.sh" 2>&1 | tee -a "${MASTER}"
  done
fi

log "==== parallel seed42 RETRAIN finished at $(date) ===="

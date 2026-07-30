#!/usr/bin/env bash
# Retrain ALL multipredictor-jepa predictors with reproducible seed=42 protocol.
# Archives pre-fix checkpoints, then retrains 30ep → 50ep best → 80ep doorway → global 65ep.
# Phase 2 (optional): re-run all success-rate evals — set RUN_EVAL=1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

ROOT="experiments/tworoom"
STAMP="pre_seed42_20260713"
GEOM_DIR="${ROOT}/results/tworoom_geometry_train_region_predictors"
GEOM_ARCHIVE="${ROOT}/results/tworoom_geometry_train_region_predictors_${STAMP}"
MASTER="${ROOT}/results/retrain_multipredictor_seed42_master.log"
RUN_EVAL="${RUN_EVAL:-0}"
SKIP_ARCHIVE="${SKIP_ARCHIVE:-0}"

log() { echo "$*" | tee -a "${MASTER}"; }

mkdir -p "${ROOT}/results"
log "==== multipredictor seed42 FULL RETRAIN started at $(date) GPU=${CUDA_VISIBLE_DEVICES} ===="
log "protocol: split_seed=3072, seed=42, cudnn deterministic; reuse frozen-encoder embedding caches (no re-encode)"

archive_if_exists() {
  local name="$1"
  local src="${ROOT}/results/${name}"
  local dst="${ROOT}/results/${name}_${STAMP}"
  if [[ -d "${src}" && ! -e "${dst}" ]]; then
    log "[archive] ${src} -> ${dst}"
    mv "${src}" "${dst}"
  elif [[ -d "${src}" ]]; then
    log "[archive] skip ${src} (already archived as ${dst})"
  fi
}

restore_geometry_embedding_caches() {
  mkdir -p "${GEOM_DIR}"
  if [[ ! -d "${GEOM_ARCHIVE}" ]]; then
    log "[restore] no archive at ${GEOM_ARCHIVE}; will encode only if caches missing"
    return 0
  fi
  log "[restore] copy embedding caches from ${GEOM_ARCHIVE} (encoder frozen, independent of FT seed)"
  cp -f "${GEOM_ARCHIVE}"/P_train_*_embeddings.npz "${GEOM_DIR}/"
  cp -f "${GEOM_ARCHIVE}/train_global_reference_starts.npy" "${GEOM_DIR}/"
  cp -f "${GEOM_ARCHIVE}/geometry_region_thresholds.npy" "${GEOM_DIR}/"
}

if [[ "${SKIP_ARCHIVE}" != "1" ]]; then
  archive_if_exists "tworoom_geometry_train_region_predictors"
  archive_if_exists "tworoom_geometry_train_region_predictors_doorway80ep"
  archive_if_exists "tworoom_geometry_train_global_ft_65ep"
  archive_if_exists "tworoom_latent_kmeanspp_kmeanspp_R50_outer0"
else
  log "[archive] SKIP_ARCHIVE=1"
fi

restore_geometry_embedding_caches

log "---- Phase 1a: 30ep geometry train∩region (5 regions, predictor FT only) ----"
bash "${ROOT}/scripts/ablations/run_geometry_train_region_predictors.sh" 2>&1 | tee -a "${MASTER}"

log "---- Phase 1b: 50ep best (doorway, left, right) ----"
bash "${ROOT}/scripts/ablations/run_geometry_train_region_predictors_50ep_best.sh" 2>&1 | tee -a "${MASTER}"

log "---- Phase 1c: 80ep doorway best ----"
bash "${ROOT}/scripts/ablations/run_geometry_train_doorway_80ep_best.sh" 2>&1 | tee -a "${MASTER}"

log "---- Phase 1d: global FT 65ep (compute-matched) ----"
bash "${ROOT}/scripts/ablations/run_geometry_train_global_ft_65ep.sh" 2>&1 | tee -a "${MASTER}"

log "---- Phase 1e: latent kmeanspp predictor FT (outer 0,1,2) ----"
for OUTER_SEED in 0 1 2; do
  log "  outer seed ${OUTER_SEED}"
  OUTER_SEED="${OUTER_SEED}" bash "${ROOT}/scripts/internal/run_latent_kmeanspp_train_predictors_50ep.sh" 2>&1 | tee -a "${MASTER}"
done

if [[ "${RUN_EVAL}" == "1" ]]; then
  log "---- Phase 2: success-rate eval sweeps ----"
  bash "${ROOT}/scripts/ablations/run_success_rate_5seed.sh" 2>&1 | tee -a "${MASTER}"
  bash "${ROOT}/scripts/ablations/run_success_rate_5seed_50ep_rooms3_priority5.sh" 2>&1 | tee -a "${MASTER}"
  bash "${ROOT}/scripts/ablations/run_success_rate_5seed_exp5_priority5.sh" 2>&1 | tee -a "${MASTER}"
  bash "${ROOT}/scripts/ablations/run_success_rate_5seed_exp6_longrange.sh" 2>&1 | tee -a "${MASTER}"
  bash "${ROOT}/scripts/ablations/run_success_rate_5seed_exp6_longrange_30ep_50ep.sh" 2>&1 | tee -a "${MASTER}"
  bash "${ROOT}/scripts/ablations/run_success_rate_5seed_exp8_global_ft.sh" 2>&1 | tee -a "${MASTER}"
  bash "${ROOT}/scripts/ablations/run_success_rate_5seed_exp8_global_ft_longrange.sh" 2>&1 | tee -a "${MASTER}"
  for OUTER_SEED in 0 1 2; do
    OUTER_SEED="${OUTER_SEED}" bash "${ROOT}/scripts/internal/run_success_rate_5seed_latent_kmeanspp_longrange.sh" 2>&1 | tee -a "${MASTER}"
  done
  log "---- Phase 2 eval complete ----"
else
  log "---- Phase 2 eval SKIPPED (set RUN_EVAL=1 to run) ----"
fi

log "==== multipredictor seed42 FULL RETRAIN finished at $(date) ===="

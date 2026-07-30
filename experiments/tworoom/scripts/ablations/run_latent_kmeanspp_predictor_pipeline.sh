#!/usr/bin/env bash
# Full downstream pipeline: 3 outer seeds predictor FT (50ep, seed=42) + 5-seed long-range eval.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

ROOT="experiments/tworoom"
MASTER="${ROOT}/results/latent_kmeanspp_predictor_pipeline_master.log"

mkdir -p "${ROOT}/results"
echo "==== latent kmeanspp predictor pipeline started at $(date) GPU=${CUDA_VISIBLE_DEVICES} ====" | tee "${MASTER}"

for OUTER_SEED in 0 1 2; do
  echo "---- Step 2: predictor FT outer=${OUTER_SEED} ----" | tee -a "${MASTER}"
  OUTER_SEED="${OUTER_SEED}" bash "${ROOT}/scripts/internal/run_latent_kmeanspp_train_predictors_50ep.sh" 2>&1 | tee -a "${MASTER}"
done

for OUTER_SEED in 0 1 2; do
  echo "---- Step 3: long-range eval outer=${OUTER_SEED} ----" | tee -a "${MASTER}"
  OUTER_SEED="${OUTER_SEED}" bash "${ROOT}/scripts/internal/run_success_rate_5seed_latent_kmeanspp_longrange.sh" 2>&1 | tee -a "${MASTER}"
done

echo "==== latent kmeanspp predictor pipeline finished at $(date) ====" | tee -a "${MASTER}"

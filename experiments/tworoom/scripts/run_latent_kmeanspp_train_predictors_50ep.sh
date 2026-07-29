#!/usr/bin/env bash
# Main experiment step 2 (paper config): per-cluster predictor FT for one K-means++ outer seed.
# Usage: OUTER_SEED=0 bash run_latent_kmeanspp_train_predictors_50ep.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

ROOT="experiments/tworoom"
GPU="${GPU:?set GPU=0..N for parallel dispatch}"
export CUDA_VISIBLE_DEVICES="${GPU}"
OUTER_SEED="${OUTER_SEED:?set OUTER_SEED=0|1|2}"
INNER_BUDGET="${INNER_BUDGET:-50}"
ONLY_CLUSTERS="${ONLY_CLUSTERS:-}"
ROUTE_LABEL_OFFSET_STEPS="${ROUTE_LABEL_OFFSET_STEPS:-}"
KMEANS_DIR="${ROOT}/results/latent_kmeanspp_multirestart_k3"
LABEL_NPZ="${KMEANS_DIR}/labels/kmeanspp_R${INNER_BUDGET}_outer${OUTER_SEED}.npz"
ZSCORE_PARAMS="${KMEANS_DIR}/zscore_params.npz"
EMBED_DIR="${ROOT}/results/tworoom_geometry_train_region_predictors"
CKPT="${LAP_LEWM_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt}"
DEFAULT_OUT_DIR="${ROOT}/results/tworoom_latent_kmeanspp_kmeanspp_R${INNER_BUDGET}_outer${OUTER_SEED}"
if [[ -n "${ROUTE_LABEL_OFFSET_STEPS}" ]]; then
  DEFAULT_OUT_DIR="${DEFAULT_OUT_DIR}_routeoffset${ROUTE_LABEL_OFFSET_STEPS}"
fi
OUT_DIR="${OUT_DIR:-${DEFAULT_OUT_DIR}}"
LOG="${OUT_DIR}/train_50ep.log"

if [[ ! -f "${LABEL_NPZ}" ]]; then
  echo "Missing K-means++ labels: ${LABEL_NPZ}" >&2
  exit 1
fi
if [[ ! -f "${ZSCORE_PARAMS}" ]]; then
  echo "Missing zscore params: ${ZSCORE_PARAMS}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "==== latent kmeanspp predictor FT outer=${OUTER_SEED} R=${INNER_BUDGET} started at $(date) ===="
extra_args=()
if [[ -n "${ONLY_CLUSTERS}" ]]; then
  extra_args+=(--only-clusters "${ONLY_CLUSTERS}")
fi
if [[ -n "${ROUTE_LABEL_OFFSET_STEPS}" ]]; then
  extra_args+=(--route-label-offset-steps "${ROUTE_LABEL_OFFSET_STEPS}")
fi
/usr/bin/time -p python "${ROOT}/latent_cluster_train_predictors.py" \
  --kmeanspp-label-npz "${LABEL_NPZ}" \
  --zscore-params "${ZSCORE_PARAMS}" \
  --embedding-source-dir "${EMBED_DIR}" \
  --train-starts "${EMBED_DIR}/train_global_reference_starts.npy" \
  --checkpoint "${CKPT}" \
  --out-dir "${OUT_DIR}" \
  --epochs 50 \
  --seed 42 \
  --device cuda \
  "${extra_args[@]}" \
  >> "${LOG}" 2>&1

echo "==== latent kmeanspp predictor FT outer=${OUTER_SEED} R=${INNER_BUDGET} finished at $(date) ===="

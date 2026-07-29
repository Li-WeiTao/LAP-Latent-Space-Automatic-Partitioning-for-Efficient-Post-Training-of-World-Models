#!/usr/bin/env bash
# Main experiment step 2: per-cluster predictor FT (50ep, best-by-eval) for one clustering seed.
# Usage: CLUSTER_SEED=0 bash run_latent_cluster_train_predictors_50ep.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

ROOT="experiments/tworoom"
CLUSTER_SEED="${CLUSTER_SEED:-0}"
METHOD="faiss_spherical_kmeans"
ARTIFACT="${ROOT}/results/latent_unsup_cluster/${METHOD}_k3_seed${CLUSTER_SEED}"
EMBED_DIR="${ROOT}/results/tworoom_geometry_train_region_predictors"
CKPT="${LAP_LEWM_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt}"
OUT_DIR="${ROOT}/results/tworoom_latent_cluster3_${METHOD}_k3_seed${CLUSTER_SEED}"
LOG="${OUT_DIR}/train_50ep.log"

if [[ ! -d "${ARTIFACT}" ]]; then
  echo "Missing cluster artifact: ${ARTIFACT}" >&2
  echo "Run scripts/run_latent_unsup_cluster_faiss_3seed.sh first." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "==== latent cluster predictor FT seed=${CLUSTER_SEED} started at $(date) ===="
/usr/bin/time -p python "${ROOT}/latent_cluster_train_predictors.py" \
  --cluster-artifact-dir "${ARTIFACT}" \
  --embedding-source-dir "${EMBED_DIR}" \
  --train-starts "${EMBED_DIR}/train_global_reference_starts.npy" \
  --checkpoint "${CKPT}" \
  --out-dir "${OUT_DIR}" \
  --epochs 50 \
  --seed 42 \
  --device cuda \
  > "${LOG}" 2>&1

echo "==== latent cluster predictor FT seed=${CLUSTER_SEED} finished at $(date) ===="

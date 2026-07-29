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
RANDOM_SEED="${RANDOM_SEED:-0}"
TRAIN_SEED="${TRAIN_SEED:-42}"
export CUDA_VISIBLE_DEVICES="${GPU}"

ROOT=experiments/tworoom
ARTIFACT_DIR="${ROOT}/results/latent_random_voronoi_k3/random_voronoi_k3_seed${RANDOM_SEED}"
EMBED_DIR="${ROOT}/results/tworoom_geometry_train_region_predictors"
OUT_DIR="${ROOT}/results/tworoom_latent_random_voronoi_k3_seed${RANDOM_SEED}_trainseed${TRAIN_SEED}"
LOG="${OUT_DIR}/train_50ep.log"
SHARED_MERGED_CACHE="${SHARED_MERGED_CACHE:-${ROOT}/results/tworoom_latent_spectral_spectral_M20000_k30_P16_seed0_trainseed42/P_train_global_merged_embeddings.npz}"

mkdir -p "${OUT_DIR}"
if [[ ! -e "${OUT_DIR}/P_train_global_merged_embeddings.npz" ]]; then
  [[ -f "${SHARED_MERGED_CACHE}" ]] || {
    echo "Missing shared merged embedding cache: ${SHARED_MERGED_CACHE}" >&2
    exit 1
  }
  ln -s "$(realpath "${SHARED_MERGED_CACHE}")" \
    "${OUT_DIR}/P_train_global_merged_embeddings.npz"
fi
STARTUP_LOCK="${STARTUP_LOCK:-${ROOT}/results/.latent_predictor_cache_load.lock}"
exec 9>"${STARTUP_LOCK}"
flock 9
/usr/bin/time -p nice -n 10 python "${ROOT}/latent_cluster_train_predictors.py" \
  --cluster-artifact-dir "${ARTIFACT_DIR}" \
  --embedding-source-dir "${EMBED_DIR}" \
  --train-starts "${EMBED_DIR}/train_global_reference_starts.npy" \
  --checkpoint "${LAP_LEWM_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt}" \
  --out-dir "${OUT_DIR}" \
  --epochs 50 \
  --seed "${TRAIN_SEED}" \
  --device cuda \
  >> "${LOG}" 2>&1 &
train_pid=$!
for _ in $(seq 1 300); do
  if grep -q '\[cache\] loaded merged embeddings' "${LOG}"; then
    break
  fi
  if ! kill -0 "${train_pid}" 2>/dev/null; then
    break
  fi
  sleep 2
done
flock -u 9
wait "${train_pid}"

#!/usr/bin/env bash
# Lightweight automatic partition.  Use SEEDS=0,1,2 for the stability run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

GPU="${GPU:?set GPU to one available physical GPU id}"
export CUDA_VISIBLE_DEVICES="${GPU}"

ROOT="experiments/tworoom"
DATA_ROOT="${LAP_DATA_ROOT:?set LAP_DATA_ROOT}"
DATA_FILE="${LAP_TWOROOM_DATA:-${DATA_ROOT}/tworoom.h5}"
EMBED_DIR="${EMBED_DIR:-${ROOT}/results/tworoom_geometry_train_region_predictors}"
SEEDS="${SEEDS:-0}"
NUM_CLUSTERS="${NUM_CLUSTERS:-3}"
AUTO_K="${AUTO_K:-0}"
AUTO_K_MIN="${AUTO_K_MIN:-2}"
AUTO_K_MAX="${AUTO_K_MAX:-6}"
INPUT_CACHE_HASH_MODE="${INPUT_CACHE_HASH_MODE:-sha256}"
DETERMINISTIC_ALGORITHMS="${DETERMINISTIC_ALGORITHMS:-0}"
OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-0}"

if [[ -z "${OUT_DIR:-}" ]]; then
  if [[ "${AUTO_K}" == "1" ]]; then
    OUT_DIR="${ROOT}/results/latent_landmark_spectral_auto_k"
  else
    OUT_DIR="${ROOT}/results/latent_landmark_spectral_k${NUM_CLUSTERS}"
  fi
fi

extra_args=()
if [[ "${AUTO_K}" == "1" ]]; then
  extra_args+=(--auto-k --auto-k-min "${AUTO_K_MIN}" --auto-k-max "${AUTO_K_MAX}")
elif [[ "${AUTO_K}" != "0" ]]; then
  echo "AUTO_K must be 0 or 1" >&2
  exit 1
fi
if [[ "${DETERMINISTIC_ALGORITHMS}" == "1" ]]; then
  extra_args+=(--deterministic-algorithms)
elif [[ "${DETERMINISTIC_ALGORITHMS}" != "0" ]]; then
  echo "DETERMINISTIC_ALGORITHMS must be 0 or 1" >&2
  exit 1
fi
if [[ "${OVERWRITE_EXISTING}" == "1" ]]; then
  extra_args+=(--overwrite-existing)
elif [[ "${OVERWRITE_EXISTING}" != "0" ]]; then
  echo "OVERWRITE_EXISTING must be 0 or 1" >&2
  exit 1
fi

echo "==== landmark spectral partition started at $(date) GPU=${GPU} seeds=${SEEDS} ===="
/usr/bin/time -p nice -n 10 python "${ROOT}/latent_landmark_spectral.py" \
  --embed-dir "${EMBED_DIR}" \
  --data-root "${DATA_ROOT}" \
  --data-file "${DATA_FILE}" \
  --seeds "${SEEDS}" \
  --out-dir "${OUT_DIR}" \
  --num-clusters "${NUM_CLUSTERS}" \
  --num-landmarks 20000 \
  --knn 30 \
  --knn-fallback 50 \
  --knn-backend torch_exact \
  --gpu-id 0 \
  --query-chunk 2048 \
  --cpu-threads 4 \
  --spectral-n-init 20 \
  --prototypes-per-cluster 16 \
  --max-prototypes-per-cluster 32 \
  --route-anchor transition_start \
  --input-cache-hash-mode "${INPUT_CACHE_HASH_MODE}" \
  "${extra_args[@]}"
echo "==== landmark spectral partition finished at $(date) ===="

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

ROOT=experiments/tworoom
nice -n 10 python "${ROOT}/latent_random_voronoi.py" \
  --embedding-source-dir "${ROOT}/results/tworoom_geometry_train_region_predictors" \
  --zscore-params "${ROOT}/results/latent_kmeanspp_multirestart_k3/zscore_params.npz" \
  --out-root "${ROOT}/results/latent_random_voronoi_k3" \
  --num-clusters 3 \
  --seeds 0 1 2

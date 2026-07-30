#!/usr/bin/env bash
# Main experiment step 1: unsupervised 3-class clustering (FAISS spherical K-means, 3 seeds).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

ROOT="experiments/tworoom"
EMBED_DIR="${ROOT}/results/tworoom_geometry_train_region_predictors"
OUT_DIR="${ROOT}/results/latent_unsup_cluster"
LOG="${OUT_DIR}/run_faiss_spherical_kmeans_3seed.log"

mkdir -p "${OUT_DIR}"

echo "==== latent unsup cluster (faiss spherical kmeans, seeds 0 1 2) started at $(date) ===="
/usr/bin/time -p python "${ROOT}/latent_unsup_cluster.py" \
  --method faiss_spherical_kmeans \
  --num-clusters 3 \
  --seeds 0 1 2 \
  --device cpu \
  --embed-dir "${EMBED_DIR}" \
  --out-dir "${OUT_DIR}" \
  > "${LOG}" 2>&1

echo "==== latent unsup cluster finished at $(date) ===="

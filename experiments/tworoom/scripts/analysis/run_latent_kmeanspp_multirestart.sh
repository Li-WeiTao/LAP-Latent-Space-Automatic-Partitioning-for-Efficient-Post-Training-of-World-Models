#!/usr/bin/env bash
# Z-score + K-means++ multi-restart (20 outer × 50 inner).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

GPU="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"

ROOT="experiments/tworoom"
OUT="${ROOT}/results/latent_kmeanspp_multirestart_k3"
LOG="${OUT}/run.log"

mkdir -p "${OUT}"

echo "==== kmeans++ multirestart started at $(date) ===="
/usr/bin/time -p python "${ROOT}/latent_kmeanspp_multirestart.py" \
  --device cuda \
  --gpu-id 0 \
  --out-dir "${OUT}" \
  2>&1 | tee "${LOG}"

echo "==== kmeans++ multirestart finished at $(date) ===="

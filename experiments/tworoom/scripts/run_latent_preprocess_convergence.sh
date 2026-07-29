#!/usr/bin/env bash
# Converged K-means on raw/center/zscore (3 preprocess × 20 seeds).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

GPU="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"

ROOT="experiments/tworoom"
OUT="${ROOT}/results/latent_preprocess_convergence_k3"
LOG="${OUT}/run.log"

mkdir -p "${OUT}"

echo "==== latent preprocess convergence started at $(date) ===="
/usr/bin/time -p python "${ROOT}/latent_preprocess_convergence.py" \
  --device cuda \
  --gpu-id 0 \
  --out-dir "${OUT}" \
  2>&1 | tee "${LOG}"

echo "==== latent preprocess convergence finished at $(date) ===="

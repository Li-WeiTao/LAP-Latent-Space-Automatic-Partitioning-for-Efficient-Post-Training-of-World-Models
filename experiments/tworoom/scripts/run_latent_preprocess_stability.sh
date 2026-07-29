#!/usr/bin/env bash
# Latent preprocessing stability: 7 preprocess × 20 clustering seeds (140 runs).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate

GPU="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"

ROOT="experiments/tworoom"
OUT="${ROOT}/results/latent_preprocess_stability_k3"
LOG="${OUT}/run.log"

mkdir -p "${OUT}"

echo "==== latent preprocess stability started at $(date) ===="
/usr/bin/time -p python "${ROOT}/latent_preprocess_stability.py" \
  --device cuda \
  --gpu-id 0 \
  --out-dir "${OUT}" \
  2>&1 | tee "${LOG}"

echo "==== latent preprocess stability finished at $(date) ===="

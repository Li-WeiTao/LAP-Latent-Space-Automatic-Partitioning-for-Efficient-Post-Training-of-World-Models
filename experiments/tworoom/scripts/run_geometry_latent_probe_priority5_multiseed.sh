#!/usr/bin/env bash
# priority5 geometry latent probes: 5-seed stability check (episode split + probe + RFF).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

GPU="${CUDA_VISIBLE_DEVICES:-4}"
export CUDA_VISIBLE_DEVICES="${GPU}"

ROOT="experiments/tworoom"
OUT="${ROOT}/results/geometry_latent_probe_priority5_multiseed"
LOG="${OUT}/run_multiseed.log"
mkdir -p "${OUT}"

echo "==== priority5 multiseed latent probes started at $(date) ====" | tee "${LOG}"

/usr/bin/time -p python "${ROOT}/geometry_latent_probe_multiseed.py" \
  --partition priority5 \
  --seeds 0 1 2 3 4 \
  --embedding-dir "${ROOT}/results/tworoom_geometry_train_region_predictors" \
  --data-root "${LAP_DATA_ROOT:-/data/sicong/weitao/datasets/lewm}" \
  --out-dir "${OUT}" \
  --device cuda \
  --torch-epochs 5 \
  --torch-batch-size 16384 \
  --rbf-n-components 8192 \
  >> "${LOG}" 2>&1

echo "==== finished at $(date) ====" | tee -a "${LOG}"

#!/usr/bin/env bash
# Geometry priority5 (mutually exclusive) latent linear-decodability probes (episode split).
# Does NOT run automatically — invoke manually when ready.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

GPU="${CUDA_VISIBLE_DEVICES:-4}"
export CUDA_VISIBLE_DEVICES="${GPU}"

ROOT="experiments/tworoom"
OUT="${ROOT}/results/geometry_latent_probe_priority5"
LOG="${OUT}/run_full_cached.log"
mkdir -p "${OUT}"

echo "==== geometry priority5 latent probes (episode split) started at $(date) ====" | tee "${LOG}"

/usr/bin/time -p python "${ROOT}/geometry_latent_svm_rooms3.py" \
  --partition priority5 \
  --embedding-dir "${ROOT}/results/tworoom_geometry_train_region_predictors" \
  --data-root "${LAP_DATA_ROOT:-/data/sicong/weitao/datasets/lewm}" \
  --out-dir "${OUT}" \
  --episode-split-seed 20260711 \
  --episode-train-fraction 0.7 \
  --episode-val-fraction 0.15 \
  --episode-test-fraction 0.15 \
  --model-seed 20260711 \
  --device cuda \
  --torch-epochs 5 \
  --torch-batch-size 16384 \
  --rbf-n-components 8192 \
  >> "${LOG}" 2>&1

echo "==== finished at $(date) ====" | tee -a "${LOG}"

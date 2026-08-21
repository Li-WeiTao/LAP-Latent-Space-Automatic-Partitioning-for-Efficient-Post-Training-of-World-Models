#!/usr/bin/env bash
# Gate-only OAT sensitivity audit (no predictor training or planning evaluation).
set -euo pipefail

REPO="/data/sicong/weitao/LAP-Latent-Space-Auto-Partitioned-Fine-Tuning-for-World-Models"
PYTHON="${PYTHON:-/data/sicong/weitao/le-wm/.venv/bin/python}"
OUT="${REPO}/experiments/control_matrix/assets/gate_sensitivity"

cd "${REPO}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

"${PYTHON}" experiments/control_matrix/gate_sensitivity_audit.py \
  --repo-root "${REPO}" \
  --output-dir "${OUT}" \
  --gpu-id "${GPU_ID:-0}" \
  "$@"

#!/usr/bin/env bash
# Gate-only OAT sensitivity audit (no predictor training or planning evaluation).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/experiments/control_matrix/assets/gate_sensitivity}"
STAGING_DIR="${STAGING_DIR:-${REPO_ROOT}/experiments/control_matrix/assets/gate_sensitivity_next}"
PYTHON="${PYTHON:-python}"
GPU_ID="${GPU_ID:--1}"

cd "${REPO_ROOT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

ARGS=(--repo-root "${REPO_ROOT}" --staging-dir "${STAGING_DIR}" --output-dir "${OUT_DIR}" --promote)
if [[ "${GPU_ID}" -ge 0 ]]; then
  ARGS+=(--gpu-id "${GPU_ID}")
fi

"${PYTHON}" experiments/control_matrix/gate_sensitivity_audit.py "${ARGS[@]}" "$@"

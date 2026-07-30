#!/usr/bin/env bash
# Archive aborted / partial training artifacts (do not delete; for postmortem).
# Usage: LABEL=aborted_20260713 bash archive_aborted_training.sh
set -euo pipefail

ROOT="experiments/tworoom/results"
STAMP="${LABEL:-aborted_$(date +%Y%m%d_%H%M%S)}"

archive_path() {
  local src="$1"
  local dst="${ROOT}/$(basename "${src}")_${STAMP}"
  if [[ -e "${src}" ]]; then
    echo "[archive] ${src} -> ${dst}"
    mv "${src}" "${dst}"
  fi
}

GEOM="${ROOT}/tworoom_geometry_train_region_predictors"
ABORT_CKPT="${GEOM}/_aborted_checkpoints_${STAMP}"

# Per-region work dirs
if [[ -d "${GEOM}/_work" ]]; then
  archive_path "${GEOM}/_work"
fi

# Partial ckpts already consolidated into MAIN_DIR (post-consolidation abort)
if [[ -d "${GEOM}" ]]; then
  mkdir -p "${ABORT_CKPT}"
  shopt -s nullglob
  moved=0
  for f in \
    "${GEOM}"/P_train_*_object.ckpt \
    "${GEOM}"/P_train_*_object.json \
    "${GEOM}"/P_train_*_starts.npy \
    "${GEOM}"/P_train_*_epoch*_object.ckpt \
    "${GEOM}"/P_train_*_epoch*_object.json \
    "${GEOM}"/manifest.json; do
    [[ -f "${f}" ]] || continue
    mv "${f}" "${ABORT_CKPT}/"
    moved=1
    echo "[archive] ${f} -> ${ABORT_CKPT}/$(basename "${f}")"
  done
  if [[ "${moved}" -eq 0 ]]; then
    rmdir "${ABORT_CKPT}" 2>/dev/null || true
  fi
fi

archive_path "${ROOT}/tworoom_geometry_train_global_ft_65ep"
for d in "${ROOT}"/tworoom_latent_kmeanspp_kmeanspp_R50_outer*; do
  [[ -e "${d}" ]] || continue
  [[ "${d}" == *"_${STAMP}" ]] && continue
  archive_path "${d}"
done

echo "==== archive done: ${STAMP} ===="

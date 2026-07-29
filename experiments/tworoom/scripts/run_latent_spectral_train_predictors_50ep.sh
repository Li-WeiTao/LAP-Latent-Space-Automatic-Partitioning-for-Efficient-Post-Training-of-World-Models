#!/usr/bin/env bash
# Fine-tune one predictor per spectral cluster.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

GPU="${GPU:?set GPU to one available physical GPU id}"
export CUDA_VISIBLE_DEVICES="${GPU}"

ROOT="experiments/tworoom"
SPECTRAL_SEED="${SPECTRAL_SEED:-0}"
TRAIN_SEED="${TRAIN_SEED:-42}"
ONLY_CLUSTERS="${ONLY_CLUSTERS:-}"
OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-0}"
SPECTRAL_ROOT="${SPECTRAL_ROOT:-${ROOT}/results/latent_landmark_spectral_k3}"
if [[ -z "${ARTIFACT_DIR:-}" ]]; then
  SUMMARY="${SPECTRAL_ROOT}/stability_summary.json"
  if [[ -f "${SUMMARY}" ]]; then
    ARTIFACT_DIR="$(python - "${SUMMARY}" "${SPECTRAL_SEED}" <<'PY'
import json
import sys
from pathlib import Path
summary = json.load(open(sys.argv[1]))
seed = sys.argv[2]
by_seed = summary.get("artifacts_by_seed", {})
if seed in by_seed:
    print(by_seed[seed])
else:
    matches = [p for p in summary.get("artifacts", []) if Path(p).name.endswith(f"seed{seed}")]
    if len(matches) != 1:
        raise SystemExit(f"summary has no unique artifact for seed {seed}")
    print(matches[0])
PY
)"
  else
    shopt -s nullglob
    candidates=(
      "${SPECTRAL_ROOT}"/spectral_cfg*_seed"${SPECTRAL_SEED}"
      "${SPECTRAL_ROOT}"/spectral_M*_seed"${SPECTRAL_SEED}"
    )
    shopt -u nullglob
    if (( ${#candidates[@]} != 1 )); then
      echo "Expected exactly one spectral artifact for seed ${SPECTRAL_SEED}; found ${#candidates[@]}. Set ARTIFACT_DIR explicitly." >&2
      printf '  %s\n' "${candidates[@]}" >&2
      exit 1
    fi
    ARTIFACT_DIR="${candidates[0]}"
  fi
fi
EMBED_DIR="${EMBED_DIR:-${ROOT}/results/tworoom_geometry_train_region_predictors}"
CKPT="${CKPT:-/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt}"
artifact_tag="$(basename "${ARTIFACT_DIR}")"
OUT_DIR="${OUT_DIR:-${ROOT}/results/tworoom_latent_spectral_${artifact_tag}_trainseed${TRAIN_SEED}}"
LOG="${OUT_DIR}/train_50ep.log"

for file in cluster_labels.npz routing_prototypes.npy prototype_cluster_ids.npy zscore_params.npz cluster_meta.json; do
  [[ -f "${ARTIFACT_DIR}/${file}" ]] || { echo "Missing ${ARTIFACT_DIR}/${file}" >&2; exit 1; }
done
mkdir -p "${OUT_DIR}"

extra_args=()
if [[ -n "${ONLY_CLUSTERS}" ]]; then
  extra_args+=(--only-clusters "${ONLY_CLUSTERS}")
fi
if [[ "${OVERWRITE_EXISTING}" == "1" ]]; then
  extra_args+=(--overwrite-existing)
elif [[ "${OVERWRITE_EXISTING}" != "0" ]]; then
  echo "OVERWRITE_EXISTING must be 0 or 1" >&2
  exit 1
fi

echo "==== spectral predictor FT seed=${SPECTRAL_SEED} train_seed=${TRAIN_SEED} started at $(date) ===="
/usr/bin/time -p nice -n 10 python "${ROOT}/latent_cluster_train_predictors.py" \
  --cluster-artifact-dir "${ARTIFACT_DIR}" \
  --embedding-source-dir "${EMBED_DIR}" \
  --train-starts "${EMBED_DIR}/train_global_reference_starts.npy" \
  --checkpoint "${CKPT}" \
  --out-dir "${OUT_DIR}" \
  --epochs 50 \
  --seed "${TRAIN_SEED}" \
  --device cuda \
  "${extra_args[@]}" \
  >> "${LOG}" 2>&1
echo "==== spectral predictor FT finished at $(date) ===="

#!/usr/bin/env bash
# Short-horizon rollout MSE for one spectral artifact/predictor set.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

GPU="${GPU:?set GPU to one available physical GPU id}"
export CUDA_VISIBLE_DEVICES="${GPU}"

ROOT="experiments/tworoom"
SPECTRAL_SEED="${SPECTRAL_SEED:-0}"
TRAIN_SEED="${TRAIN_SEED:-42}"
LATENT_ROUTING="${LATENT_ROUTING:-mpc}"
ROOMS3_ROUTING="${ROOMS3_ROUTING:-mpc}"
BATCH_SIZE="${BATCH_SIZE:-64}"
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

artifact_tag="$(basename "${ARTIFACT_DIR}")"
PRED_DIR="${PRED_DIR:-${ROOT}/results/tworoom_latent_spectral_${artifact_tag}_trainseed${TRAIN_SEED}}"
OUT_DIR="${OUT_DIR:-${ROOT}/results/tworoom_trajectory_switch_rollout_${artifact_tag}_trainseed${TRAIN_SEED}_${LATENT_ROUTING}_rooms3${ROOMS3_ROUTING}}"

for file in cluster_labels.npz centroids.npy routing_prototypes.npy prototype_cluster_ids.npy zscore_params.npz cluster_meta.json; do
  [[ -f "${ARTIFACT_DIR}/${file}" ]] || {
    echo "Missing ${ARTIFACT_DIR}/${file}" >&2
    exit 1
  }
done

num_clusters="$(python - "${ARTIFACT_DIR}/cluster_meta.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    print(int(json.load(f)["num_clusters"]))
PY
)"
for ((k = 0; k < num_clusters; k++)); do
  [[ -f "${PRED_DIR}/P_train_cluster${k}_object.ckpt" ]] || {
    echo "Missing cluster${k} predictor under ${PRED_DIR}" >&2
    exit 1
  }
done

echo "==== spectral short-horizon rollout started at $(date) ===="
echo "artifact=${ARTIFACT_DIR} predictors=${PRED_DIR} K=${num_clusters} latent_routing=${LATENT_ROUTING}"
/usr/bin/time -p nice -n 10 python "${ROOT}/trajectory_switch_rollout.py" \
  --cluster-artifact-dir "${ARTIFACT_DIR}" \
  --cluster-predictor-dir "${PRED_DIR}" \
  --latent-routing "${LATENT_ROUTING}" \
  --rooms3-routing "${ROOMS3_ROUTING}" \
  --batch-size "${BATCH_SIZE}" \
  --device cuda \
  --out-dir "${OUT_DIR}"
echo "==== spectral short-horizon rollout finished at $(date) ===="

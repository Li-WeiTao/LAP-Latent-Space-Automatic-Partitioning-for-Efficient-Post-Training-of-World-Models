#!/usr/bin/env bash
# Build the five dense latent caches consumed by the automatic partitioner.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

GPU="${GPU:?set GPU to one available physical GPU id}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export STABLEWM_HOME="${STABLEWM_HOME:-${LAP_STABLEWM_HOME:-/data/sicong/weitao/.stable_worldmodel}}"

ROOT="experiments/tworoom"
DATA_FILE="${LAP_TWOROOM_DATA:-${LAP_DATA_ROOT:-/data/sicong/weitao/datasets/lewm}/tworoom.h5}"
CKPT="${CKPT:-${LAP_LEWM_CHECKPOINT:-${STABLEWM_HOME}/tworoom/lewm_object.ckpt}}"
OUT_DIR="${EMBED_DIR:-${ROOT}/results/tworoom_geometry_train_region_predictors}"
OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-0}"
ENCODE_WORKERS="${ENCODE_WORKERS:-4}"
CPU_THREADS="${CPU_THREADS:-4}"
REGIONS=(doorway_corridor near_wall common right_room left_room)

[[ -f "${DATA_FILE}" ]] || { echo "Missing dataset: ${DATA_FILE}" >&2; exit 1; }
[[ -f "${CKPT}" ]] || { echo "Missing checkpoint: ${CKPT}" >&2; exit 1; }
if [[ "${OVERWRITE_EXISTING}" != "0" && "${OVERWRITE_EXISTING}" != "1" ]]; then
  echo "OVERWRITE_EXISTING must be 0 or 1" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
python "${ROOT}/trajectory.py" \
  --dataset tworoom \
  --data-file "${DATA_FILE}" \
  --checkpoint "${CKPT}" \
  --out-dir "${OUT_DIR}" \
  --regions "${REGIONS[@]}" \
  --restrict-to-train-split \
  --region-split-mode geometry \
  --predictor-prefix train_ \
  --prepare-starts-only \
  --device cuda

for region in "${REGIONS[@]}"; do
  starts="${OUT_DIR}/P_train_${region}_starts.npy"
  output="${OUT_DIR}/P_train_${region}_embeddings.npz"
  report="${output}.report.json"
  if [[ -f "${output}" && "${OVERWRITE_EXISTING}" == "0" ]]; then
    echo "[skip] existing cache: ${output}"
    continue
  fi
  if [[ "${OVERWRITE_EXISTING}" == "1" ]]; then
    rm -f -- "${output}" "${report}"
  fi
  python "${ROOT}/unique_timestep_reencode.py" encode \
    --dataset-factory backends.lewm.encoding:make_hdf5_transition_dataset \
    --dataset-arg "data_file=${DATA_FILE}" \
    --dataset-arg "starts=${starts}" \
    --dataset-arg "action_norm_starts=${OUT_DIR}/train_global_reference_starts.npy" \
    --dataset-arg history_size=3 \
    --dataset-arg num_preds=1 \
    --encoder-factory backends.lewm.encoding:make_encoder \
    --encoder-arg img_size=224 \
    --pretrained-model "${CKPT}" \
    --output "${output}" \
    --report "${report}" \
    --num-workers "${ENCODE_WORKERS}" \
    --cpu-threads "${CPU_THREADS}" \
    --device cuda
done

echo "Prepared spectral inputs under ${OUT_DIR}"

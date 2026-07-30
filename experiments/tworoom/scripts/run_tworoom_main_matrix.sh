#!/usr/bin/env bash
# Canonical TwoRoom paper matrix. All task/model paths are explicit parameters.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

DATA_FILE="${LAP_TWOROOM_DATA:?set LAP_TWOROOM_DATA to tworoom.h5}"
CHECKPOINT="${LAP_LEWM_CHECKPOINT:?set LAP_LEWM_CHECKPOINT to the official checkpoint}"
GPU="${GPU:?set GPU to one available physical GPU id}"
CACHE_DIR="${LAP_STABLEWM_HOME:-${STABLEWM_HOME:-$HOME/.stable_worldmodel}}"
WORK_ROOT="${WORK_ROOT:-experiments/tworoom/matrix}"
PHASE="${PHASE:-all}"
PYTHON="${PYTHON:-python}"

run_automatic() {
  env \
    DATASET_NAME=tworoom \
    DATA_FILE="$DATA_FILE" \
    CHECKPOINT="$CHECKPOINT" \
    EVAL_CONFIG=tworoom \
    EVAL_DATASET_NAME=tworoom \
    CACHE_DIR="$CACHE_DIR" \
    WORK_ROOT="$WORK_ROOT" \
    GPU_ID="$GPU" \
    PHASE="$1" \
    TRAIN_SEEDS="${TRAIN_SEEDS:-0,42,625}" \
    PARTITION_SEEDS="${PARTITION_SEEDS:-0,1,2}" \
    EVAL_SEEDS="${EVAL_SEEDS:-0,1,2,3,4}" \
    METHODS="${METHODS:-random_voronoi,kmeanspp,spectral}" \
    PYTHON="$PYTHON" \
    bash experiments/control_matrix/scripts/run_lewm_matrix.sh
}

run_human() {
  env \
    LAP_TWOROOM_DATA="$DATA_FILE" \
    LAP_LEWM_CHECKPOINT="$CHECKPOINT" \
    LAP_STABLEWM_HOME="$CACHE_DIR" \
    WORK_ROOT="$WORK_ROOT" \
    GPU="$GPU" \
    PHASE="$1" \
    TRAIN_SEEDS="${TRAIN_SEEDS:-0,42,625}" \
    EVAL_SEEDS="${EVAL_SEEDS:-0,1,2,3,4}" \
    PYTHON="$PYTHON" \
    bash experiments/tworoom/scripts/run_tworoom_human_rooms3_matrix.sh
}

case "$PHASE" in
  prepare) run_automatic prepare; run_human prepare ;;
  partition) run_automatic partition ;;
  train) run_automatic train; run_human train ;;
  eval) run_automatic eval; run_human eval ;;
  aggregate)
    run_automatic aggregate
    "$PYTHON" experiments/tworoom/aggregate_tworoom_canonical.py --root "$WORK_ROOT"
    ;;
  all)
    run_automatic all
    run_human all
    "$PYTHON" experiments/tworoom/aggregate_tworoom_canonical.py --root "$WORK_ROOT"
    ;;
  *) echo "unknown PHASE=$PHASE" >&2; exit 2 ;;
esac

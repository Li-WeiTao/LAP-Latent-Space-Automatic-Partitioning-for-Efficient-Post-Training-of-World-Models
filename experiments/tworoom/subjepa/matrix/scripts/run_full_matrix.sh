#!/usr/bin/env bash
# Sub-JEPA TwoRoom 50-epoch matrix orchestrator.
# Reuses formal full latent cache, formal spectral partitions, and LeWM paired eval starts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/data/sicong/weitao/le-wm/.venv/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
CPU_THREADS="${CPU_THREADS:-4}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

TASK_SPEC="configs/experiments/tasks/tworoom.json"
DATASET="/data/sicong/weitao/datasets/lewm/tworoom.h5"
CHECKPOINT="/data/sicong/weitao/.stable_worldmodel/tworoom/subjepa_object.ckpt"
WORK_ROOT="experiments/tworoom/subjepa/matrix"
CACHE_DIR="${CACHE_DIR:-/data/sicong/weitao/.stable_worldmodel/subjepa/tworoom}"
PAIR_SHORT="$WORK_ROOT/paired_starts/lewm_short"
PAIR_LONG="$WORK_ROOT/paired_starts/lewm_long"

MATRIX_METHODS="${MATRIX_METHODS:-kmeanspp,spectral}"
DEPLOYMENT_SEED="${DEPLOYMENT_SEED:-0}"

JEPA_BASE=(
  --task-spec "$TASK_SPEC"
  --dataset "$DATASET"
  --checkpoint "$CHECKPOINT"
  --eval-config-name tworoom
  --work-root "$WORK_ROOT"
  --cache-dir "$CACHE_DIR"
  --python "$PYTHON"
  --cpu-threads "$CPU_THREADS"
  --methods "$MATRIX_METHODS"
)

snapshot_eval_horizon() {
  local label=$1
  local dst="$WORK_ROOT/eval_${label}"
  rm -rf "$dst"
  cp -a "$WORK_ROOT/eval" "$dst"
  echo "[snapshot] $WORK_ROOT/eval -> $dst"
}

restore_eval_horizon() {
  local label=$1
  local src="$WORK_ROOT/eval_${label}"
  [[ -d "$src" ]] || { echo "missing eval snapshot: $src" >&2; exit 1; }
  rm -rf "$WORK_ROOT/eval"
  cp -a "$src" "$WORK_ROOT/eval"
}

aggregate_horizon() {
  local label=$1
  restore_eval_horizon "$label"
  "$PYTHON" experiments/control_matrix/aggregate_matrix.py \
    --root "$WORK_ROOT" \
    --dataset-name tworoom \
    --train-seeds 0,42,625 \
    --partition-seeds 0,1,2 \
    --eval-seeds 0,1,2,3,4 \
    --methods "$MATRIX_METHODS" \
    --skip-joint \
    --include-auto-lap \
    --deployment-seed "$DEPLOYMENT_SEED"
  mv "$WORK_ROOT/matrix_summary.json" "$WORK_ROOT/manifests/matrix_summary_${label}.json"
  mv "$WORK_ROOT/matrix_raw.csv" "$WORK_ROOT/manifests/matrix_raw_${label}.csv"
}

run_parallel() {
  env GPU_IDS="$GPU_IDS" CPU_THREADS="$CPU_THREADS" RUN_ID="$RUN_ID" \
    START_STAGE="${START_STAGE:-partition}" END_STAGE="${END_STAGE:-training}" \
    PAIRED_START_ROOT_SHORT="${PAIRED_START_ROOT_SHORT:-}" \
    PAIRED_START_ROOT_LONG="${PAIRED_START_ROOT_LONG:-}" \
    SHORT_GOAL_OFFSET="${SHORT_GOAL_OFFSET:-}" \
    LONG_GOAL_OFFSET="${LONG_GOAL_OFFSET:-}" \
    METHODS="$MATRIX_METHODS" SKIP_JOINT=1 \
    bash experiments/control_matrix/scripts/run_jepa_matrix_parallel.sh \
    "${JEPA_BASE[@]}" "$@"
}

case "${1:-training}" in
  setup)
    bash experiments/tworoom/subjepa/matrix/scripts/setup_matrix.sh
    ;;
  training)
    bash experiments/tworoom/subjepa/matrix/scripts/setup_matrix.sh
    START_STAGE=partition END_STAGE=training run_parallel
    DEPLOYMENT_SEED="$DEPLOYMENT_SEED" \
      bash experiments/tworoom/subjepa/matrix/scripts/link_auto_lap.sh \
      "$WORK_ROOT" training
    ;;
  eval-short)
    bash experiments/tworoom/subjepa/matrix/scripts/setup_matrix.sh
    rm -rf "$WORK_ROOT/eval/official" "$WORK_ROOT/eval/global" \
      "$WORK_ROOT/eval/kmeanspp" "$WORK_ROOT/eval/spectral"
    RUN_ID="${RUN_ID}_short" \
      PAIRED_START_ROOT_SHORT="$(realpath "$PAIR_SHORT")" \
      SHORT_GOAL_OFFSET=25 \
      START_STAGE=eval_short END_STAGE=eval_short \
      run_parallel
    DEPLOYMENT_SEED="$DEPLOYMENT_SEED" \
      bash experiments/tworoom/subjepa/matrix/scripts/link_auto_lap.sh \
      "$WORK_ROOT" eval
    snapshot_eval_horizon short
    ;;
  eval-long)
    bash experiments/tworoom/subjepa/matrix/scripts/setup_matrix.sh
    rm -rf "$WORK_ROOT/eval/official" "$WORK_ROOT/eval/global" \
      "$WORK_ROOT/eval/kmeanspp" "$WORK_ROOT/eval/spectral"
    RUN_ID="${RUN_ID}_long" \
      PAIRED_START_ROOT_LONG="$(realpath "$PAIR_LONG")" \
      LONG_GOAL_OFFSET=50 \
      START_STAGE=eval_long END_STAGE=eval_long \
      run_parallel
    DEPLOYMENT_SEED="$DEPLOYMENT_SEED" \
      bash experiments/tworoom/subjepa/matrix/scripts/link_auto_lap.sh \
      "$WORK_ROOT" eval
    snapshot_eval_horizon long
    ;;
  aggregate)
    aggregate_horizon short
    aggregate_horizon long
    ;;
  audit)
    "$PYTHON" experiments/tworoom/subjepa/matrix/scripts/matrix_frozen_audit.py \
      --work-root "$WORK_ROOT" \
      --checkpoint "$CHECKPOINT" \
      --lock "$WORK_ROOT/manifests/pre_execution_lock.json"
    "$PYTHON" experiments/tworoom/subjepa/matrix/scripts/matrix_one_step_mse.py \
      --work-root "$WORK_ROOT" \
      --checkpoint "$CHECKPOINT" \
      --cache "$WORK_ROOT/preparation/embedding_cache.npz" \
      --lock "$WORK_ROOT/manifests/pre_execution_lock.json"
    ;;
  bootstrap)
    for label in short long; do
      seed=20260803
      [[ "$label" == "long" ]] && seed=20260804
      "$PYTHON" experiments/tworoom/subjepa/matrix/scripts/matrix_paired_bootstrap.py \
        --eval-root "$WORK_ROOT/eval_${label}" \
        --out "$WORK_ROOT/manifests/bootstrap_${label}.json" \
        --reps 200000 \
        --seed "$seed"
    done
    ;;
  all-post-train)
    bash "$0" eval-short
    bash "$0" eval-long
    bash "$0" audit
    bash "$0" aggregate
    bash "$0" bootstrap
    ;;
  *)
    echo "usage: $0 {setup|training|eval-short|eval-long|audit|aggregate|bootstrap|all-post-train}" >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
# Reacher Sub-JEPA 50-epoch matrix — Global-FT50, K-means++ K3-50, Spectral
# K3-50, and Auto-LAP, evaluated with paired short/long official starts.
#
# Mirrors experiments/pusht/subjepa/matrix/scripts/run_full_matrix.sh but
# drops the preflight/passport/audit/bootstrap stages (see task note: no new
# audit workflow for this migration). Stages are independently resumable and
# skip already-complete artifacts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/experiments/reacher/subjepa/env.sh"

K3_MATRIX="${K3_MATRIX:-$MATRIX}"
# shellcheck source=/dev/null
source "$REPO_ROOT/experiments/control_matrix/scripts/subjepa_k_variant.sh"

GPU_IDS="${GPU_IDS:-0}"
GPU_ID="${GPU_ID:-${GPU_IDS%%,*}}"
CPU_THREADS="${CPU_THREADS:-4}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

WORK_ROOT="$MATRIX"
PAIR_SHORT="$WORK_ROOT/paired_starts/canon_short"
PAIR_LONG="$WORK_ROOT/paired_starts/canon_long"

MATRIX_METHODS="${MATRIX_METHODS:-kmeanspp,spectral}"
DEPLOYMENT_SEED="${DEPLOYMENT_SEED:-0}"
TRAIN_SEEDS="${TRAIN_SEEDS:-0,42,625}"
PARTITION_SEEDS="${PARTITION_SEEDS:-0,1,2}"
EVAL_SEEDS="${EVAL_SEEDS:-0,1,2,3,4}"

REACHER_MATRIX_SCRIPTS="$REPO_ROOT/experiments/reacher/subjepa/matrix/scripts"
SETUP="$REACHER_MATRIX_SCRIPTS/setup_matrix.sh"
LINK_AUTO_LAP="$REACHER_MATRIX_SCRIPTS/link_auto_lap.sh"

JEPA_BASE=(
  --task-spec "$TASK_SPEC"
  --dataset "$DATASET"
  --checkpoint "$CHECKPOINT"
  --eval-config-name reacher
  --work-root "$WORK_ROOT"
  --cache-dir "$CACHE_DIR"
  --python "$PYTHON"
  --cpu-threads "$CPU_THREADS"
  --methods "$MATRIX_METHODS"
)

run_partition() {
  require_checkpoint
  env GPU_ID="$GPU_ID" SKIP_JOINT=1 \
    TRAIN_SEEDS="$TRAIN_SEEDS" PARTITION_SEEDS="$PARTITION_SEEDS" EVAL_SEEDS="$EVAL_SEEDS" \
    bash experiments/control_matrix/scripts/run_subjepa_matrix.sh \
    "${JEPA_BASE[@]}" \
    --train-seeds "$TRAIN_SEEDS" --partition-seeds "$PARTITION_SEEDS" --eval-seeds "$EVAL_SEEDS" \
    --phase partition \
    2>&1 | tee "$WORK_ROOT/logs/partition_${RUN_ID}.log"
}

run_parallel() {
  require_checkpoint
  env GPU_IDS="$GPU_IDS" CPU_THREADS="$CPU_THREADS" RUN_ID="$RUN_ID" \
    START_STAGE="${START_STAGE:-training}" END_STAGE="${END_STAGE:-training}" \
    PAIRED_START_ROOT_SHORT="${PAIRED_START_ROOT_SHORT:-}" \
    PAIRED_START_ROOT_LONG="${PAIRED_START_ROOT_LONG:-}" \
    SHORT_GOAL_OFFSET="${SHORT_GOAL_OFFSET:-}" \
    LONG_GOAL_OFFSET="${LONG_GOAL_OFFSET:-}" \
    METHODS="$MATRIX_METHODS" SKIP_JOINT=1 \
    TRAIN_SEEDS="$TRAIN_SEEDS" PARTITION_SEEDS="$PARTITION_SEEDS" EVAL_SEEDS="$EVAL_SEEDS" \
    NUM_CLUSTERS="$NUM_CLUSTERS" \
    SKIP_OFFICIAL="$SKIP_OFFICIAL" \
    SKIP_GLOBAL="$SKIP_GLOBAL" \
    PARTITION_INCLUDE_SPECTRAL="$PARTITION_INCLUDE_SPECTRAL" \
    bash experiments/control_matrix/scripts/run_jepa_matrix_parallel.sh \
    "${JEPA_BASE[@]}" \
    --train-seeds "$TRAIN_SEEDS" --partition-seeds "$PARTITION_SEEDS" --eval-seeds "$EVAL_SEEDS" \
    "$@"
}

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
  aggregate_k_variant_flags
  "$PYTHON" experiments/control_matrix/aggregate_matrix.py \
    --root "$WORK_ROOT" \
    --dataset-name reacher \
    --train-seeds "$TRAIN_SEEDS" \
    --partition-seeds "$PARTITION_SEEDS" \
    --eval-seeds "$EVAL_SEEDS" \
    --methods "$MATRIX_METHODS" \
    --skip-joint \
    "${AGG_EXTRA[@]}"
  mv "$WORK_ROOT/matrix_summary.json" "$WORK_ROOT/manifests/matrix_summary_${label}.json"
  mv "$WORK_ROOT/matrix_raw.csv" "$WORK_ROOT/manifests/matrix_raw_${label}.csv"
  echo "[aggregate] $label -> $WORK_ROOT/manifests/matrix_summary_${label}.json"
}

case "${1:-training}" in
  setup)
    bash "$SETUP"
    ;;
  partition)
    bash "$SETUP"
    run_partition
    ;;
  training)
    bash "$SETUP"
    if [[ "$NUM_CLUSTERS" == "3" ]]; then
      run_partition
      START_STAGE=training END_STAGE=training run_parallel
    else
      START_STAGE=partition END_STAGE=training run_parallel
    fi
    if [[ "$INCLUDE_AUTO_LAP" == "1" ]]; then
      DEPLOYMENT_SEED="$DEPLOYMENT_SEED" bash "$LINK_AUTO_LAP" "$WORK_ROOT" training
    fi
    ;;
  eval-short)
    bash "$SETUP"
    wipe_eval_for_k_variant "$WORK_ROOT"
    RUN_ID="${RUN_ID}_short" \
      PAIRED_START_ROOT_SHORT="$(realpath "$PAIR_SHORT")" \
      SHORT_GOAL_OFFSET=25 \
      START_STAGE=eval_short END_STAGE=eval_short \
      run_parallel
    if [[ "$INCLUDE_AUTO_LAP" == "1" ]]; then
      DEPLOYMENT_SEED="$DEPLOYMENT_SEED" bash "$LINK_AUTO_LAP" "$WORK_ROOT" eval
    fi
    seed_reused_eval_from_k3 short "$WORK_ROOT"
    snapshot_eval_horizon short
    ;;
  eval-long)
    bash "$SETUP"
    wipe_eval_for_k_variant "$WORK_ROOT"
    RUN_ID="${RUN_ID}_long" \
      PAIRED_START_ROOT_LONG="$(realpath "$PAIR_LONG")" \
      LONG_GOAL_OFFSET=50 \
      START_STAGE=eval_long END_STAGE=eval_long \
      run_parallel
    if [[ "$INCLUDE_AUTO_LAP" == "1" ]]; then
      DEPLOYMENT_SEED="$DEPLOYMENT_SEED" bash "$LINK_AUTO_LAP" "$WORK_ROOT" eval
    fi
    seed_reused_eval_from_k3 long "$WORK_ROOT"
    snapshot_eval_horizon long
    ;;
  aggregate)
    aggregate_horizon short
    aggregate_horizon long
    ;;
  all-post-train)
    bash "$0" eval-short
    bash "$0" eval-long
    bash "$0" aggregate
    ;;
  *)
    echo "usage: $0 {setup|partition|training|eval-short|eval-long|aggregate|all-post-train}" >&2
    exit 2
    ;;
esac

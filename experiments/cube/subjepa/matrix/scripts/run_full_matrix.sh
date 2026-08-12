#!/usr/bin/env bash
# OGBench Cube Sub-JEPA 50-epoch matrix — Global-FT50, K-means++ K3-50,
# Spectral K3-50, and Auto-LAP, evaluated with paired short/long official
# starts.
#
# Mirrors experiments/reacher/subjepa/matrix/scripts/run_full_matrix.sh: no
# preflight/passport/audit/bootstrap-only stages beyond what the shared
# statistical protocol requires (the `audit)` branch from PushT's original
# run_full_matrix.sh has been dropped entirely; `all-post-train` only runs
# eval-short, eval-long, and aggregate). Stages are independently resumable
# and skip already-complete artifacts.
#
# If experiments/cube/subjepa/matrix/setup_matrix.sh found a complete
# external canonical paired-start set (CANON_SHORT/CANON_LONG), eval-short /
# eval-long reuse it via PAIRED_START_ROOT_SHORT/LONG. Otherwise those
# variables are left unset and the generic driver's eval_official task
# self-generates the official paired starts once per horizon (fixed by eval
# seed); every method within that horizon and this work root then reuses the
# same starts, so no method resamples independently. Cube never reuses
# PushT's (or any other task's) starts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/experiments/cube/subjepa/env.sh"

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

CUBE_MATRIX_SCRIPTS="$REPO_ROOT/experiments/cube/subjepa/matrix/scripts"
SETUP="$CUBE_MATRIX_SCRIPTS/setup_matrix.sh"
LINK_AUTO_LAP="$CUBE_MATRIX_SCRIPTS/link_auto_lap.sh"

JEPA_BASE=(
  --task-spec "$TASK_SPEC"
  --dataset "$DATASET"
  --checkpoint "$CHECKPOINT"
  --eval-config-name cube
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
  "$PYTHON" experiments/control_matrix/aggregate_matrix.py \
    --root "$WORK_ROOT" \
    --dataset-name cube \
    --train-seeds "$TRAIN_SEEDS" \
    --partition-seeds "$PARTITION_SEEDS" \
    --eval-seeds "$EVAL_SEEDS" \
    --methods "$MATRIX_METHODS" \
    --skip-joint \
    --include-auto-lap \
    --deployment-seed "$DEPLOYMENT_SEED"
  mv "$WORK_ROOT/matrix_summary.json" "$WORK_ROOT/manifests/matrix_summary_${label}.json"
  mv "$WORK_ROOT/matrix_raw.csv" "$WORK_ROOT/manifests/matrix_raw_${label}.csv"
  echo "[aggregate] $label -> $WORK_ROOT/manifests/matrix_summary_${label}.json"
}

# Generate Cube official paired starts at eval time (not during setup/training).
# Only runs when CANON_SHORT/CANON_LONG have no complete 5-seed set to copy.
ensure_official_eval_starts() {
  local goal_offset=$1
  local pair_root=$2
  local out_official="$pair_root/eval/official"
  require_checkpoint
  local num_eval eval_budget
  num_eval="$("$PYTHON" -c "import json; print(json.load(open('$TASK_SPEC')).get('num_eval', 50))")"
  eval_budget="$("$PYTHON" -c "import json; print(json.load(open('$TASK_SPEC')).get('eval_budget', 50))")"
  IFS=, read -r -a eval_seeds_arr <<< "$EVAL_SEEDS"
  for eseed in "${eval_seeds_arr[@]}"; do
    local dst="$out_official/eval${eseed}/results.json"
    [[ -f "$dst" ]] && continue
    echo "[eval] official paired starts: eval${eseed} goal_offset=${goal_offset} -> $dst"
    mkdir -p "$(dirname "$dst")"
    "$PYTHON" experiments/tworoom/tworoom_success_rate_eval.py \
      --mode baseline \
      --seed "$eseed" \
      --checkpoint "$CHECKPOINT" \
      --config-name cube \
      --dataset-tag cube \
      --cache-dir "$CACHE_DIR" \
      --out-dir "$(dirname "$dst")" \
      --num-eval "$num_eval" \
      --eval-budget "$eval_budget" \
      --goal-offset "$goal_offset" \
      --model-family subjepa \
      --sample-eval-starts
    [[ -f "$dst" ]] || { echo "STOP: official eval did not produce $dst" >&2; exit 1; }
  done
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
    run_partition
    START_STAGE=training END_STAGE=training run_parallel
    DEPLOYMENT_SEED="$DEPLOYMENT_SEED" bash "$LINK_AUTO_LAP" "$WORK_ROOT" training
    ;;
  eval-short)
    bash "$SETUP"
    paired_short=""
    [[ -f "$PAIR_SHORT/eval/official/eval0/results.json" ]] && paired_short="$(realpath "$PAIR_SHORT/eval/official")"
    [[ -n "$paired_short" ]] || ensure_official_eval_starts 25 "$PAIR_SHORT"
    [[ -f "$PAIR_SHORT/eval/official/eval0/results.json" ]] && paired_short="$(realpath "$PAIR_SHORT/eval/official")"
    rm -rf "$WORK_ROOT/eval/official" "$WORK_ROOT/eval/global" \
      "$WORK_ROOT/eval/kmeanspp" "$WORK_ROOT/eval/spectral"
    RUN_ID="${RUN_ID}_short" \
      PAIRED_START_ROOT_SHORT="$paired_short" \
      SHORT_GOAL_OFFSET=25 \
      START_STAGE=eval_short END_STAGE=eval_short \
      run_parallel
    DEPLOYMENT_SEED="$DEPLOYMENT_SEED" bash "$LINK_AUTO_LAP" "$WORK_ROOT" eval
    snapshot_eval_horizon short
    ;;
  eval-long)
    bash "$SETUP"
    paired_long=""
    [[ -f "$PAIR_LONG/eval/official/eval0/results.json" ]] && paired_long="$(realpath "$PAIR_LONG/eval/official")"
    [[ -n "$paired_long" ]] || ensure_official_eval_starts 50 "$PAIR_LONG"
    [[ -f "$PAIR_LONG/eval/official/eval0/results.json" ]] && paired_long="$(realpath "$PAIR_LONG/eval/official")"
    rm -rf "$WORK_ROOT/eval/official" "$WORK_ROOT/eval/global" \
      "$WORK_ROOT/eval/kmeanspp" "$WORK_ROOT/eval/spectral"
    RUN_ID="${RUN_ID}_long" \
      PAIRED_START_ROOT_LONG="$paired_long" \
      LONG_GOAL_OFFSET=50 \
      START_STAGE=eval_long END_STAGE=eval_long \
      run_parallel
    DEPLOYMENT_SEED="$DEPLOYMENT_SEED" bash "$LINK_AUTO_LAP" "$WORK_ROOT" eval
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

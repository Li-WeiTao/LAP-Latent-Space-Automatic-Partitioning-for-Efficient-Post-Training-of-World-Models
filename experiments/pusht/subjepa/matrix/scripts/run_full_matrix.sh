#!/usr/bin/env bash
# PushT Sub-JEPA 50-epoch matrix — same stages as tworoom/subjepa/matrix/scripts/run_full_matrix.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/experiments/pusht/subjepa/env.sh"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6}"
CPU_THREADS="${CPU_THREADS:-4}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

WORK_ROOT="$MATRIX"
PAIR_SHORT="$WORK_ROOT/paired_starts/canon_short"
PAIR_LONG="$WORK_ROOT/paired_starts/canon_long"

MATRIX_METHODS="${MATRIX_METHODS:-kmeanspp,spectral}"
DEPLOYMENT_SEED="${DEPLOYMENT_SEED:-0}"

JEPA_BASE=(
  --task-spec "$TASK_SPEC"
  --dataset "$DATASET"
  --checkpoint "$CHECKPOINT"
  --eval-config-name pusht
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
    --dataset-name pusht \
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

PUSH_MATRIX_SCRIPTS="$REPO_ROOT/experiments/pusht/subjepa/matrix/scripts"
SETUP="$PUSH_MATRIX_SCRIPTS/setup_matrix.sh"
PREFLIGHT="$PUSH_MATRIX_SCRIPTS/matrix_preflight.py"
LINK_AUTO_LAP="$PUSH_MATRIX_SCRIPTS/link_auto_lap.sh"

run_preflight() {
  "$PYTHON" "$PREFLIGHT" \
    --formal-root "$FORMAL" \
    --matrix-root "$WORK_ROOT" \
    --smoke-root "$SMOKE_ROOT" \
    --dataset "$DATASET" \
    --checkpoint "$CHECKPOINT" \
    --task-spec "$TASK_SPEC" \
    --canon-short "$CANON_SHORT" \
    --canon-long "$CANON_LONG" \
    --expected-smoke-cache-sha256 "$SMOKE_CACHE_SHA256" \
    --out "$WORK_ROOT/manifests/preflight_report.json"
}

case "${1:-training}" in
  setup)
    bash "$SETUP"
    ;;
  preflight)
    bash "$SETUP"
    run_preflight
    ;;
  training)
    bash "$SETUP"
    run_preflight
    START_STAGE=partition END_STAGE=training run_parallel
    DEPLOYMENT_SEED="$DEPLOYMENT_SEED" bash "$LINK_AUTO_LAP" "$WORK_ROOT" training
    ;;
  eval-short)
    bash "$SETUP"
    rm -rf "$WORK_ROOT/eval/official" "$WORK_ROOT/eval/global" \
      "$WORK_ROOT/eval/kmeanspp" "$WORK_ROOT/eval/spectral"
    RUN_ID="${RUN_ID}_short" \
      PAIRED_START_ROOT_SHORT="$(realpath "$PAIR_SHORT")" \
      SHORT_GOAL_OFFSET=25 \
      START_STAGE=eval_short END_STAGE=eval_short \
      run_parallel
    DEPLOYMENT_SEED="$DEPLOYMENT_SEED" bash "$LINK_AUTO_LAP" "$WORK_ROOT" eval
    snapshot_eval_horizon short
    ;;
  eval-long)
    bash "$SETUP"
    rm -rf "$WORK_ROOT/eval/official" "$WORK_ROOT/eval/global" \
      "$WORK_ROOT/eval/kmeanspp" "$WORK_ROOT/eval/spectral"
    RUN_ID="${RUN_ID}_long" \
      PAIRED_START_ROOT_LONG="$(realpath "$PAIR_LONG")" \
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
  audit)
    "$PYTHON" "$TWOROOM_MATRIX_SCRIPTS/matrix_frozen_audit.py" \
      --work-root "$WORK_ROOT" \
      --checkpoint "$CHECKPOINT" \
      --lock "$WORK_ROOT/manifests/pre_execution_lock.json"
    "$PYTHON" "$TWOROOM_MATRIX_SCRIPTS/matrix_one_step_mse.py" \
      --work-root "$WORK_ROOT" \
      --checkpoint "$CHECKPOINT" \
      --cache "$WORK_ROOT/preparation/embedding_cache.npz" \
      --lock "$WORK_ROOT/manifests/pre_execution_lock.json"
    ;;
  bootstrap)
    for label in short long; do
      seed=20260803
      [[ "$label" == "long" ]] && seed=20260804
      "$PYTHON" "$TWOROOM_MATRIX_SCRIPTS/matrix_paired_bootstrap.py" \
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
    ;;
  *)
    echo "usage: $0 {setup|preflight|training|eval-short|eval-long|audit|aggregate|bootstrap|all-post-train}" >&2
    exit 2
    ;;
esac

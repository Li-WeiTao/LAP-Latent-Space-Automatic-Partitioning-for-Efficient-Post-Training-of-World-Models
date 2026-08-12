#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU controller for Held-out Region-Risk Analysis.
# "formal" in compatibility paths and filenames remains internal provenance.
# Each worker gets one physical GPU via CUDA_VISIBLE_DEVICES; tasks are round-robin
# assigned so no output directory has concurrent writers.

ROOT="${ROOT:-/data/sicong/weitao/LAP-Latent-Space-Auto-Partitioned-Fine-Tuning-for-World-Models}"
source "$ROOT/experiments/control_matrix/scripts/run_formal_region_risk_env.sh"
cd "$ROOT"

TASK="${TASK:-pusht}"
WORK_ROOT="${WORK_ROOT:-experiments/control_matrix/assets/formal_region_risk/$TASK}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
TRAIN_SEEDS="${TRAIN_SEEDS:-0,42,625}"
EPOCHS="${EPOCHS:-50}"
BOOTSTRAP_REPS="${BOOTSTRAP_REPS:-50000}"
MAX_TRAIN_STARTS="${MAX_TRAIN_STARTS:-0}"
MAX_EVAL_STARTS="${MAX_EVAL_STARTS:-0}"
MAX_ANCHORS="${MAX_ANCHORS:-0}"
MAX_EPISODES="${MAX_EPISODES:-0}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
OVERWRITE="${OVERWRITE:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
CPU_THREADS="${CPU_THREADS:-4}"

PIPELINE="$ROOT/experiments/control_matrix/formal_region_risk_pipeline.py"
LOG_ROOT="$WORK_ROOT/logs/parallel_$RUN_ID"
mkdir -p "$LOG_ROOT"

IFS=, read -r -a gpu_ids <<< "$GPU_IDS"
if [[ ${#gpu_ids[@]} -eq 0 ]]; then
  echo "[parallel] GPU_IDS must contain at least one GPU" >&2
  exit 2
fi

if [[ "$SMOKE_ONLY" == "1" ]]; then
  MAX_TRAIN_STARTS=4096
  MAX_EVAL_STARTS=512
  TRAIN_SEEDS=0
  EPOCHS=1
  BOOTSTRAP_REPS=100
  MAX_ANCHORS=32
  MAX_EPISODES=4
  OVERWRITE=1
fi

IFS=, read -r -a train_seeds <<< "$TRAIN_SEEDS"

common_args=(
  --task "$TASK"
  --work-root "$WORK_ROOT"
  --device cuda
  --python "$PYTHON"
  --train-seeds "$TRAIN_SEEDS"
  --epochs "$EPOCHS"
  --bootstrap-reps "$BOOTSTRAP_REPS"
  --encoding-batch-size 128
)
if [[ "$MAX_TRAIN_STARTS" -gt 0 ]]; then common_args+=(--max-train-starts "$MAX_TRAIN_STARTS"); fi
if [[ "$MAX_EVAL_STARTS" -gt 0 ]]; then common_args+=(--max-eval-starts "$MAX_EVAL_STARTS"); fi
if [[ "$MAX_ANCHORS" -gt 0 ]]; then common_args+=(--max-anchors "$MAX_ANCHORS"); fi
if [[ "$MAX_EPISODES" -gt 0 ]]; then common_args+=(--max-episodes "$MAX_EPISODES"); fi
if [[ "$OVERWRITE" == "1" ]]; then common_args+=(--overwrite); fi
if [[ "$SMOKE_ONLY" == "1" ]]; then common_args+=(--smoke-only); fi

declare -a task_names=()
declare -a task_phases=()
declare -a task_train_seeds=()
declare -a task_roles=()

add_task() {
  task_names+=("$1")
  task_phases+=("$2")
  task_train_seeds+=("${3:-}")
  task_roles+=("${4:-}")
}

clear_tasks() {
  task_names=()
  task_phases=()
  task_train_seeds=()
  task_roles=()
}

run_pipeline() {
  local gpu=$1
  shift
  env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    OMP_NUM_THREADS="$CPU_THREADS" \
    MKL_NUM_THREADS="$CPU_THREADS" \
    OPENBLAS_NUM_THREADS="$CPU_THREADS" \
    NUMEXPR_NUM_THREADS="$CPU_THREADS" \
    "$PYTHON" "$PIPELINE" "${common_args[@]}" "$@"
}

run_stage() {
  local stage=$1
  local task_count=${#task_names[@]}
  local worker_count=${#gpu_ids[@]}
  local -a worker_pids=()
  mkdir -p "$LOG_ROOT/$stage"
  echo "[parallel] stage=$stage tasks=$task_count workers=$worker_count"
  for ((worker=0; worker<worker_count; worker++)); do
    (
      gpu=${gpu_ids[$worker]}
      for ((index=worker; index<task_count; index+=worker_count)); do
        name=${task_names[$index]}
        phase=${task_phases[$index]}
        log="$LOG_ROOT/$stage/${name}.log"
        extra=()
        if [[ -n "${task_train_seeds[$index]}" ]]; then
          extra+=(--train-seed "${task_train_seeds[$index]}")
        fi
        if [[ -n "${task_roles[$index]}" ]]; then
          extra+=(--training-role "${task_roles[$index]}")
        fi
        echo "[start] stage=$stage task=$name gpu=$gpu log=$log"
        run_pipeline "$gpu" --phase "$phase" "${extra[@]}" >"$log" 2>&1
        echo "[done] stage=$stage task=$name gpu=$gpu"
      done
    ) &
    worker_pids+=("$!")
  done
  local failed=0
  for pid in "${worker_pids[@]}"; do
    wait "$pid" || failed=1
  done
  if [[ $failed -ne 0 ]]; then
    echo "[parallel] stage failed: $stage; inspect $LOG_ROOT/$stage" >&2
    exit 1
  fi
}

{
  echo "run_id=$RUN_ID"
  echo "task=$TASK"
  echo "work_root=$WORK_ROOT"
  echo "gpu_ids=$GPU_IDS"
  echo "train_seeds=$TRAIN_SEEDS"
  echo "epochs=$EPOCHS"
  echo "bootstrap_reps=$BOOTSTRAP_REPS"
  echo "smoke_only=$SMOKE_ONLY"
  echo "python=$PYTHON"
  echo "git_commit=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
} >"$LOG_ROOT/run.env"

echo "[parallel] split -> $WORK_ROOT"
run_pipeline "${gpu_ids[0]}" --phase split

clear_tasks
add_task encode_train encode-train
add_task encode_eval encode-eval
run_stage encode
run_pipeline "${gpu_ids[0]}" --phase encode-finalize

clear_tasks
add_task partition_auto partition-auto
add_task partition_global partition-global
add_task partition_forced_spectral partition-forced-spectral
run_stage partition
run_pipeline "${gpu_ids[0]}" --phase partition-finalize

clear_tasks
for seed in "${train_seeds[@]}"; do
  add_task "global_train${seed}" train-one "$seed" global
  add_task "regional_train${seed}" train-one "$seed" forced_spectral_negative_control
done
run_stage train

echo "[parallel] evaluate -> $WORK_ROOT/evaluation"
run_pipeline "${gpu_ids[0]}" --phase evaluate

echo "[parallel] finalize manifest"
run_pipeline "${gpu_ids[0]}" --phase finalize

if [[ "$SMOKE_ONLY" == "1" ]]; then
  "$PYTHON" "$ROOT/experiments/control_matrix/verify_formal_region_risk.py" \
    --work-root "$WORK_ROOT" \
    --expect-smoke
else
  "$PYTHON" "$ROOT/experiments/control_matrix/verify_formal_region_risk.py" \
    --work-root "$WORK_ROOT"
fi

echo "[parallel] complete run_id=$RUN_ID logs=$LOG_ROOT work_root=$WORK_ROOT"

#!/usr/bin/env bash
# Multi-GPU controller for the generic JEPA comparison matrix.
# Dispatches leaf jobs to run_jepa_matrix.sh / run_subjepa_matrix.sh with
# disjoint train/partition/eval seed filters.  Does not duplicate training logic.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <jepa-matrix-cli-args...> [--parallel-stage STAGE]" >&2
  echo "  Pass the same CLI args as run_subjepa_matrix.sh (task-spec, dataset, etc.)." >&2
  echo "  Optional env: GPU_IDS, CPU_THREADS, RUN_ID, START_STAGE, END_STAGE, TASK_RETRIES" >&2
  exit 2
fi

# Optional trailing --parallel-stage is stripped; everything else is forwarded.
PARALLEL_STAGE="${PARALLEL_STAGE:-all}"
JEPA_ARGS=("$@")
for ((i = 0; i < ${#JEPA_ARGS[@]}; i++)); do
  if [[ "${JEPA_ARGS[$i]}" == "--parallel-stage" ]]; then
    PARALLEL_STAGE="${JEPA_ARGS[$((i + 1))]}"
    unset 'JEPA_ARGS[i]' "JEPA_ARGS[$((i + 1))]"
    JEPA_ARGS=("${JEPA_ARGS[@]}")
    break
  fi
done

PYTHON="${PYTHON:-python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
CPU_THREADS="${CPU_THREADS:-4}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
START_STAGE="${START_STAGE:-partition}"
END_STAGE="${END_STAGE:-training}"
TASK_RETRIES="${TASK_RETRIES:-2}"

TRAIN_SEEDS="${TRAIN_SEEDS:-0,42,625}"
PARTITION_SEEDS="${PARTITION_SEEDS:-0,1,2}"
EVAL_SEEDS="${EVAL_SEEDS:-0,1,2,3,4}"
METHODS="${METHODS:-kmeanspp,spectral}"
SKIP_JOINT="${SKIP_JOINT:-1}"
NUM_CLUSTERS="${NUM_CLUSTERS:-3}"
SKIP_OFFICIAL="${SKIP_OFFICIAL:-0}"
SKIP_GLOBAL="${SKIP_GLOBAL:-0}"
PARTITION_INCLUDE_SPECTRAL="${PARTITION_INCLUDE_SPECTRAL:-0}"

TMP_RESOLVED="$(mktemp)"
"$PYTHON" experiments/control_matrix/resolve_jepa_matrix_config.py \
  "${JEPA_ARGS[@]}" --output "$TMP_RESOLVED"
WORK_ROOT="$( "$PYTHON" -c "import json; print(json.load(open('$TMP_RESOLVED'))['work_root'])" )"
rm -f "$TMP_RESOLVED"

CANONICAL_RESOLVED="$(mktemp)"
resolve_canonical_args=( "${JEPA_ARGS[@]}" )
[[ "$SKIP_JOINT" == "1" ]] && resolve_canonical_args+=( --skip-joint )
"$PYTHON" experiments/control_matrix/resolve_jepa_matrix_config.py \
  "${resolve_canonical_args[@]}" \
  --methods "$METHODS" \
  --train-seeds "$TRAIN_SEEDS" \
  --partition-seeds "$PARTITION_SEEDS" \
  --eval-seeds "$EVAL_SEEDS" \
  --output "$CANONICAL_RESOLVED"
mkdir -p "$WORK_ROOT/manifests"
cp "$CANONICAL_RESOLVED" "$WORK_ROOT/manifests/resolved_config.json"
rm -f "$CANONICAL_RESOLVED"

BASE_SCRIPT="${JEPA_MATRIX_SCRIPT:-experiments/control_matrix/scripts/run_subjepa_matrix.sh}"
LOG_ROOT="$WORK_ROOT/logs/parallel_$RUN_ID"
mkdir -p "$LOG_ROOT"

IFS=, read -r -a gpu_ids <<< "$GPU_IDS"
IFS=, read -r -a train_seeds <<< "$TRAIN_SEEDS"
IFS=, read -r -a partition_seeds <<< "$PARTITION_SEEDS"
IFS=, read -r -a eval_seeds <<< "$EVAL_SEEDS"
IFS=, read -r -a methods <<< "$METHODS"

declare -a task_names=()
declare -a task_phases=()
declare -a task_train_seeds=()
declare -a task_partition_seeds=()
declare -a task_methods=()
declare -a task_eval_seeds=()
declare -a task_extra_args=()

task_allowed() {
  local name=$1
  if [[ -z "${PARALLEL_TASKS:-}" ]]; then
    return 0
  fi
  local allowed
  IFS=, read -r -a allowed <<< "$PARALLEL_TASKS"
  for candidate in "${allowed[@]}"; do
    [[ "$candidate" == "$name" ]] && return 0
  done
  return 1
}

add_task() {
  if ! task_allowed "$1"; then
    return 0
  fi
  task_names+=("$1")
  task_phases+=("$2")
  task_train_seeds+=("$3")
  task_partition_seeds+=("$4")
  task_methods+=("$5")
  task_eval_seeds+=("$6")
  task_extra_args+=("${7:-}")
}

clear_tasks() {
  task_names=()
  task_phases=()
  task_train_seeds=()
  task_partition_seeds=()
  task_methods=()
  task_eval_seeds=()
  task_extra_args=()
}

run_stage() {
  local stage=$1
  local task_count=${#task_names[@]}
  if [[ "$task_count" -eq 0 ]]; then
    echo "[stage] $stage skipped (no tasks)"
    return 0
  fi
  local worker_count=${#gpu_ids[@]}
  local max_attempts=${TASK_RETRIES}
  local -a worker_pids=()
  mkdir -p "$LOG_ROOT/$stage"
  echo "[stage] $stage tasks=$task_count workers=$worker_count retries=$max_attempts"
  for ((worker = 0; worker < worker_count; worker++)); do
    (
      gpu=${gpu_ids[$worker]}
      for ((index = worker; index < task_count; index += worker_count)); do
        name=${task_names[$index]}
        log="$LOG_ROOT/$stage/${name}.log"
        attempt=1
        while true; do
          echo "[start] stage=$stage task=$name gpu=$gpu attempt=$attempt log=$log"
          extra=${task_extra_args[$index]}
          leaf_methods="${task_methods[$index]}"
          [[ -z "$leaf_methods" ]] && leaf_methods="$METHODS"
          leaf_args=(
            --phase "${task_phases[$index]}"
            --train-seeds "${task_train_seeds[$index]}"
            --partition-seeds "${task_partition_seeds[$index]}"
            --eval-seeds "${task_eval_seeds[$index]}"
            --methods "$leaf_methods"
          )
          # shellcheck disable=SC2086
          if env \
            $extra \
            CUDA_VISIBLE_DEVICES="$gpu" \
            GPU_ID= \
            TRAIN_SEEDS="${task_train_seeds[$index]}" \
            PARTITION_SEEDS="${task_partition_seeds[$index]}" \
            METHODS="$leaf_methods" \
            EVAL_SEEDS="${task_eval_seeds[$index]}" \
            SKIP_JOINT="$SKIP_JOINT" \
            NUM_CLUSTERS="$NUM_CLUSTERS" \
            RESOLVED_CONFIG_SKIP=1 \
            RESOLVED_CONFIG_LEAF="$LOG_ROOT/$stage/${name}.resolved.json" \
            OMP_NUM_THREADS="$CPU_THREADS" \
            MKL_NUM_THREADS="$CPU_THREADS" \
            OPENBLAS_NUM_THREADS="$CPU_THREADS" \
            NUMEXPR_NUM_THREADS="$CPU_THREADS" \
            bash "$BASE_SCRIPT" \
            "${JEPA_ARGS[@]}" \
            "${leaf_args[@]}" \
            >"$log" 2>&1; then
            echo "[done] stage=$stage task=$name gpu=$gpu"
            break
          fi
          if [[ "$attempt" -ge "$max_attempts" ]]; then
            echo "[failed] stage=$stage task=$name gpu=$gpu attempts=$attempt" >&2
            exit 1
          fi
          echo "[retry] stage=$stage task=$name gpu=$gpu attempt=$attempt" >>"$log"
          attempt=$((attempt + 1))
          sleep 10
        done
      done
    ) &
    worker_pids+=("$!")
  done
  failed=0
  for pid in "${worker_pids[@]}"; do
    wait "$pid" || failed=1
  done
  if [[ $failed -ne 0 ]]; then
    echo "stage failed: $stage; inspect $LOG_ROOT/$stage" >&2
    exit 1
  fi
}

STAGE_ORDER=(partition training eval_official eval_short eval_long aggregate)

stage_enabled() {
  local stage=$1
  local in_range=0
  for candidate in "${STAGE_ORDER[@]}"; do
    [[ "$candidate" == "$START_STAGE" ]] && in_range=1
    if [[ "$candidate" == "$stage" ]]; then
      [[ "$in_range" -eq 1 ]] && return 0
      return 1
    fi
    [[ "$candidate" == "$END_STAGE" ]] && in_range=0
  done
  return 1
}

{
  echo "run_id=$RUN_ID"
  echo "work_root=$WORK_ROOT"
  echo "gpu_ids=$GPU_IDS"
  echo "cpu_threads_per_job=$CPU_THREADS"
  echo "train_seeds=$TRAIN_SEEDS"
  echo "partition_seeds=$PARTITION_SEEDS"
  echo "eval_seeds=$EVAL_SEEDS"
  echo "methods=$METHODS"
  echo "num_clusters=$NUM_CLUSTERS"
  echo "skip_joint=$SKIP_JOINT"
  echo "skip_official=$SKIP_OFFICIAL"
  echo "skip_global=$SKIP_GLOBAL"
  echo "partition_include_spectral=$PARTITION_INCLUDE_SPECTRAL"
  echo "start_stage=$START_STAGE"
  echo "end_stage=$END_STAGE"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "jepa_args=${JEPA_ARGS[*]}"
} >"$LOG_ROOT/run.env"
echo "$$" >"$LOG_ROOT/controller.pid"

if stage_enabled partition; then
  clear_tasks
  if [[ "$SKIP_GLOBAL" != "1" ]]; then
    add_task global partition_global "" "" "" "" ""
  fi
  for method in "${methods[@]}"; do
    if [[ "$method" == "spectral" && "$PARTITION_INCLUDE_SPECTRAL" != "1" ]]; then
      continue
    fi
    for pseed in "${partition_seeds[@]}"; do
      add_task "${method}_p${pseed}" partition_regions "" "$pseed" "$method" "" ""
    done
  done
  run_stage partition
fi

if stage_enabled training; then
  clear_tasks
  for tseed in "${train_seeds[@]}"; do
    if [[ "$SKIP_JOINT" != "1" ]]; then
      add_task "joint_t${tseed}" train_joint "$tseed" "" "" "" ""
    fi
    if [[ "$SKIP_GLOBAL" != "1" ]]; then
      add_task "global_t${tseed}" train_global "$tseed" "" "" "" ""
    fi
    for method in "${methods[@]}"; do
      for pseed in "${partition_seeds[@]}"; do
        add_task "${method}_p${pseed}_t${tseed}" train_regions \
          "$tseed" "$pseed" "$method" "" ""
      done
    done
  done
  run_stage training
fi

if stage_enabled eval_short; then
  clear_tasks
  short_goal="${SHORT_GOAL_OFFSET:-25}"
  short_paired="${PAIRED_START_ROOT_SHORT:-${PAIRED_START_ROOT:-}}"
  # EVAL_GOAL_OFFSET drives MPC horizon; avoid GOAL_OFFSET here because config
  # resolution treats GOAL_OFFSET as short_goal_offset and conflicts with task spec.
  short_env="EVAL_GOAL_OFFSET=${short_goal}"
  [[ -n "$short_paired" ]] && short_env+=" PAIRED_START_ROOT=${short_paired}"
  if [[ "$SKIP_OFFICIAL" != "1" ]]; then
    add_task "official" eval_official "" "" "" "$EVAL_SEEDS" "$short_env"
  fi
  for tseed in "${train_seeds[@]}"; do
    if [[ "$SKIP_GLOBAL" != "1" ]]; then
      add_task "global_t${tseed}" eval_global "$tseed" "" "" "$EVAL_SEEDS" "$short_env"
    fi
    for method in "${methods[@]}"; do
      for pseed in "${partition_seeds[@]}"; do
        add_task "${method}_p${pseed}_t${tseed}" eval_regions \
          "$tseed" "$pseed" "$method" "$EVAL_SEEDS" "$short_env"
      done
    done
  done
  run_stage eval_short
fi

if stage_enabled eval_long; then
  clear_tasks
  long_goal="${LONG_GOAL_OFFSET:-50}"
  long_paired="${PAIRED_START_ROOT_LONG:-${PAIRED_START_ROOT:-}}"
  long_env="EVAL_GOAL_OFFSET=${long_goal}"
  [[ -n "$long_paired" ]] && long_env+=" PAIRED_START_ROOT=${long_paired}"
  if [[ "$SKIP_OFFICIAL" != "1" ]]; then
    add_task "official" eval_official "" "" "" "$EVAL_SEEDS" "$long_env"
  fi
  for tseed in "${train_seeds[@]}"; do
    if [[ "$SKIP_GLOBAL" != "1" ]]; then
      add_task "global_t${tseed}" eval_global "$tseed" "" "" "$EVAL_SEEDS" "$long_env"
    fi
    for method in "${methods[@]}"; do
      for pseed in "${partition_seeds[@]}"; do
        add_task "${method}_p${pseed}_t${tseed}" eval_regions \
          "$tseed" "$pseed" "$method" "$EVAL_SEEDS" "$long_env"
      done
    done
  done
  run_stage eval_long
fi

if stage_enabled aggregate; then
  env \
    CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" \
    GPU_ID= \
    TRAIN_SEEDS="$TRAIN_SEEDS" \
    PARTITION_SEEDS="$PARTITION_SEEDS" \
    METHODS="$METHODS" \
    EVAL_SEEDS="$EVAL_SEEDS" \
    SKIP_JOINT="$SKIP_JOINT" \
    NUM_CLUSTERS="$NUM_CLUSTERS" \
    bash "$BASE_SCRIPT" \
    "${JEPA_ARGS[@]}" \
    --phase aggregate \
    --train-seeds "$TRAIN_SEEDS" \
    --partition-seeds "$PARTITION_SEEDS" \
    --methods "$METHODS" \
    --eval-seeds "$EVAL_SEEDS" \
    >"$LOG_ROOT/aggregate.log" 2>&1
fi

echo "[complete] run_id=$RUN_ID logs=$LOG_ROOT"

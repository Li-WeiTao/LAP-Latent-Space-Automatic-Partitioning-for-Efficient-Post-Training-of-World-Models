#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU controller for the dataset/model-parameterized LeWM comparison
# matrix.  One worker is created per GPU and each worker executes a disjoint
# round-robin task list, so no output directory has multiple writers.
DATASET_NAME=${DATASET_NAME:-pusht}
DATA_FILE=${DATA_FILE:?set DATA_FILE to the task HDF5 file}
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to the official LeWM object checkpoint}
EVAL_CONFIG=${EVAL_CONFIG:-pusht}
EVAL_DATASET_NAME=${EVAL_DATASET_NAME:-pusht_expert_train}
CACHE_DIR=${CACHE_DIR:-$HOME/.stable_worldmodel}
WORK_ROOT=${WORK_ROOT:-experiments/${DATASET_NAME}/matrix}
TRAIN_SEEDS=${TRAIN_SEEDS:-0,42,625}
PARTITION_SEEDS=${PARTITION_SEEDS:-0,1,2}
EVAL_SEEDS=${EVAL_SEEDS:-0,1,2,3,4}
METHODS=${METHODS:-random_voronoi,kmeanspp,spectral}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
CPU_THREADS=${CPU_THREADS:-4}
PYTHON=${PYTHON:-python}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}

BASE_SCRIPT=experiments/control_matrix/scripts/run_lewm_matrix.sh
LOG_ROOT="$WORK_ROOT/logs/parallel_$RUN_ID"
mkdir -p "$LOG_ROOT"

IFS=, read -r -a gpu_ids <<< "$GPU_IDS"
IFS=, read -r -a train_seeds <<< "$TRAIN_SEEDS"
IFS=, read -r -a partition_seeds <<< "$PARTITION_SEEDS"
IFS=, read -r -a eval_seeds <<< "$EVAL_SEEDS"
IFS=, read -r -a methods <<< "$METHODS"
if [[ ${#gpu_ids[@]} -eq 0 ]]; then
  echo "GPU_IDS must contain at least one GPU" >&2
  exit 2
fi

common_env=(
  DATASET_NAME="$DATASET_NAME"
  DATA_FILE="$DATA_FILE"
  CHECKPOINT="$CHECKPOINT"
  EVAL_CONFIG="$EVAL_CONFIG"
  EVAL_DATASET_NAME="$EVAL_DATASET_NAME"
  CACHE_DIR="$CACHE_DIR"
  WORK_ROOT="$WORK_ROOT"
  CPU_THREADS="$CPU_THREADS"
  PYTHON="$PYTHON"
  OMP_NUM_THREADS="$CPU_THREADS"
  MKL_NUM_THREADS="$CPU_THREADS"
  OPENBLAS_NUM_THREADS="$CPU_THREADS"
  NUMEXPR_NUM_THREADS="$CPU_THREADS"
)

declare -a task_names=()
declare -a task_phases=()
declare -a task_train_seeds=()
declare -a task_partition_seeds=()
declare -a task_methods=()
declare -a task_eval_seeds=()

add_task() {
  task_names+=("$1")
  task_phases+=("$2")
  task_train_seeds+=("$3")
  task_partition_seeds+=("$4")
  task_methods+=("$5")
  task_eval_seeds+=("$6")
}

clear_tasks() {
  task_names=()
  task_phases=()
  task_train_seeds=()
  task_partition_seeds=()
  task_methods=()
  task_eval_seeds=()
}

run_stage() {
  local stage=$1
  local task_count=${#task_names[@]}
  local worker_count=${#gpu_ids[@]}
  local -a worker_pids=()
  mkdir -p "$LOG_ROOT/$stage"
  echo "[stage] $stage tasks=$task_count workers=$worker_count"
  for ((worker=0; worker<worker_count; worker++)); do
    (
      gpu=${gpu_ids[$worker]}
      for ((index=worker; index<task_count; index+=worker_count)); do
        name=${task_names[$index]}
        log="$LOG_ROOT/$stage/$name.log"
        echo "[start] stage=$stage task=$name gpu=$gpu log=$log"
        env "${common_env[@]}" \
          CUDA_VISIBLE_DEVICES="$gpu" GPU_ID= \
          PHASE="${task_phases[$index]}" \
          TRAIN_SEEDS="${task_train_seeds[$index]}" \
          PARTITION_SEEDS="${task_partition_seeds[$index]}" \
          METHODS="${task_methods[$index]}" \
          EVAL_SEEDS="${task_eval_seeds[$index]}" \
          bash "$BASE_SCRIPT" >"$log" 2>&1
        echo "[done] stage=$stage task=$name gpu=$gpu"
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

{
  echo "run_id=$RUN_ID"
  echo "dataset=$DATASET_NAME"
  echo "data_file=$DATA_FILE"
  echo "checkpoint=$CHECKPOINT"
  echo "work_root=$WORK_ROOT"
  echo "gpu_ids=$GPU_IDS"
  echo "cpu_threads_per_job=$CPU_THREADS"
  echo "train_seeds=$TRAIN_SEEDS"
  echo "partition_seeds=$PARTITION_SEEDS"
  echo "eval_seeds=$EVAL_SEEDS"
  echo "methods=$METHODS"
  echo "git_commit=$(git rev-parse HEAD)"
} >"$LOG_ROOT/run.env"
echo "$$" >"$LOG_ROOT/controller.pid"

env "${common_env[@]}" CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" GPU_ID= \
  PHASE=prepare bash "$BASE_SCRIPT" >"$LOG_ROOT/prepare.log" 2>&1

clear_tasks
add_task global partition_global "" "" "" ""
for method in "${methods[@]}"; do
  for pseed in "${partition_seeds[@]}"; do
    add_task "${method}_p${pseed}" partition_regions "" "$pseed" "$method" ""
  done
done
run_stage partition

clear_tasks
for tseed in "${train_seeds[@]}"; do
  add_task "joint_t${tseed}" train_joint "$tseed" "" "" ""
  add_task "global_t${tseed}" train_global "$tseed" "" "" ""
  for method in "${methods[@]}"; do
    for pseed in "${partition_seeds[@]}"; do
      add_task "${method}_p${pseed}_t${tseed}" train_regions \
        "$tseed" "$pseed" "$method" ""
    done
  done
done
run_stage training

clear_tasks
for eseed in "${eval_seeds[@]}"; do
  add_task "official_e${eseed}" eval_official "" "" "" "$eseed"
done
run_stage official_eval

clear_tasks
for tseed in "${train_seeds[@]}"; do
  add_task "joint_t${tseed}" eval_joint "$tseed" "" "" "$EVAL_SEEDS"
  add_task "global_t${tseed}" eval_global "$tseed" "" "" "$EVAL_SEEDS"
  for method in "${methods[@]}"; do
    for pseed in "${partition_seeds[@]}"; do
      add_task "${method}_p${pseed}_t${tseed}" eval_regions \
        "$tseed" "$pseed" "$method" "$EVAL_SEEDS"
    done
  done
done
run_stage model_eval

env "${common_env[@]}" CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" GPU_ID= \
  PHASE=aggregate TRAIN_SEEDS="$TRAIN_SEEDS" \
  PARTITION_SEEDS="$PARTITION_SEEDS" METHODS="$METHODS" \
  EVAL_SEEDS="$EVAL_SEEDS" bash "$BASE_SCRIPT" \
  >"$LOG_ROOT/aggregate.log" 2>&1

echo "[complete] run_id=$RUN_ID logs=$LOG_ROOT"

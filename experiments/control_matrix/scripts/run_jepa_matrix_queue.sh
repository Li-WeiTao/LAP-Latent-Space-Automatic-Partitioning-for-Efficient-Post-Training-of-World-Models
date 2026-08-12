#!/usr/bin/env bash
# Work-queue dispatcher for an explicit list of JEPA matrix training tasks.
#
# run_jepa_matrix_parallel.sh enumerates the full seed matrix and stripes it
# across GPUs statically.  This script instead takes the exact task names that
# still need to run and hands each one to whichever GPU slot frees up first, so
# a partially finished matrix can be resumed on a different GPU set without
# disturbing leaf jobs that are already running elsewhere.
#
# Task names use the same convention as the parallel controller:
#   global_t<train_seed>
#   <method>_p<partition_seed>_t<train_seed>
#
# Leaf invocation, skip-if-complete behaviour and log layout match
# run_jepa_matrix_parallel.sh; the canonical resolved_config.json is left alone
# because this runs a subset of the matrix.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

TASKS=""
JEPA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)
      TASKS="$2"
      shift 2
      ;;
    *)
      JEPA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$TASKS" || ${#JEPA_ARGS[@]} -eq 0 ]]; then
  echo "usage: $0 <jepa-matrix-cli-args...> --tasks name1,name2,..." >&2
  echo "  Optional env: GPU_IDS, JOBS_PER_GPU, CPU_THREADS, RUN_ID, TASK_RETRIES" >&2
  exit 2
fi

PYTHON="${PYTHON:-python}"
GPU_IDS="${GPU_IDS:-0}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
CPU_THREADS="${CPU_THREADS:-4}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
TASK_RETRIES="${TASK_RETRIES:-2}"
SKIP_JOINT="${SKIP_JOINT:-1}"
BASE_SCRIPT="${JEPA_MATRIX_SCRIPT:-experiments/control_matrix/scripts/run_subjepa_matrix.sh}"

TMP_RESOLVED="$(mktemp)"
"$PYTHON" experiments/control_matrix/resolve_jepa_matrix_config.py \
  "${JEPA_ARGS[@]}" --output "$TMP_RESOLVED"
WORK_ROOT="$( "$PYTHON" -c "import json; print(json.load(open('$TMP_RESOLVED'))['work_root'])" )"
rm -f "$TMP_RESOLVED"

LOG_ROOT="$WORK_ROOT/logs/queue_$RUN_ID"
mkdir -p "$LOG_ROOT"
QUEUE_FILE="$LOG_ROOT/queue.txt"
LOCK_FILE="$LOG_ROOT/queue.lock"
: >"$LOCK_FILE"

IFS=, read -r -a gpu_ids <<< "$GPU_IDS"
IFS=, read -r -a task_names <<< "$TASKS"

# Reject unparseable names up front so a typo cannot silently drop a job.
for name in "${task_names[@]}"; do
  if ! [[ "$name" =~ ^global_t[0-9]+$ || "$name" =~ ^[A-Za-z0-9]+_p[0-9]+_t[0-9]+$ ]]; then
    echo "unrecognised task name: $name" >&2
    exit 2
  fi
done

printf '%s\n' "${task_names[@]}" >"$QUEUE_FILE"

take_next() {
  local task=""
  exec 9>"$LOCK_FILE"
  flock 9
  if [[ -s "$QUEUE_FILE" ]]; then
    task="$(head -n 1 "$QUEUE_FILE")"
    tail -n +2 "$QUEUE_FILE" >"$QUEUE_FILE.tmp"
    mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
  fi
  exec 9>&-
  printf '%s' "$task"
}

run_slot() {
  local gpu=$1
  local name phase tseed pseed method log attempt
  while :; do
    name="$(take_next)"
    [[ -n "$name" ]] || break

    if [[ "$name" =~ ^global_t([0-9]+)$ ]]; then
      phase=train_global
      tseed="${BASH_REMATCH[1]}"
      pseed=""
      method=""
    else
      [[ "$name" =~ ^([A-Za-z0-9]+)_p([0-9]+)_t([0-9]+)$ ]]
      phase=train_regions
      method="${BASH_REMATCH[1]}"
      pseed="${BASH_REMATCH[2]}"
      tseed="${BASH_REMATCH[3]}"
    fi
    [[ -n "$method" ]] || method="${METHODS:-kmeanspp,spectral}"

    log="$LOG_ROOT/${name}.log"
    attempt=1
    while true; do
      echo "[start] task=$name gpu=$gpu attempt=$attempt log=$log"
      if env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        GPU_ID= \
        TRAIN_SEEDS="$tseed" \
        PARTITION_SEEDS="$pseed" \
        METHODS="$method" \
        EVAL_SEEDS= \
        SKIP_JOINT="$SKIP_JOINT" \
        RESOLVED_CONFIG_SKIP=1 \
        RESOLVED_CONFIG_LEAF="$LOG_ROOT/${name}.resolved.json" \
        OMP_NUM_THREADS="$CPU_THREADS" \
        MKL_NUM_THREADS="$CPU_THREADS" \
        OPENBLAS_NUM_THREADS="$CPU_THREADS" \
        NUMEXPR_NUM_THREADS="$CPU_THREADS" \
        bash "$BASE_SCRIPT" \
        "${JEPA_ARGS[@]}" \
        --phase "$phase" \
        --train-seeds "$tseed" \
        --partition-seeds "$pseed" \
        --eval-seeds "" \
        --methods "$method" \
        >"$log" 2>&1; then
        echo "[done] task=$name gpu=$gpu"
        break
      fi
      if [[ "$attempt" -ge "$TASK_RETRIES" ]]; then
        echo "[failed] task=$name gpu=$gpu attempts=$attempt" >&2
        return 1
      fi
      echo "[retry] task=$name gpu=$gpu attempt=$attempt" >>"$log"
      attempt=$((attempt + 1))
      sleep 10
    done
  done
}

{
  echo "run_id=$RUN_ID"
  echo "work_root=$WORK_ROOT"
  echo "gpu_ids=$GPU_IDS"
  echo "jobs_per_gpu=$JOBS_PER_GPU"
  echo "cpu_threads_per_job=$CPU_THREADS"
  echo "skip_joint=$SKIP_JOINT"
  echo "tasks=$TASKS"
  echo "jepa_args=${JEPA_ARGS[*]}"
} >"$LOG_ROOT/run.env"
echo "$$" >"$LOG_ROOT/controller.pid"

echo "[queue] tasks=${#task_names[@]} gpus=${#gpu_ids[@]} jobs_per_gpu=$JOBS_PER_GPU logs=$LOG_ROOT"

slot_pids=()
for gpu in "${gpu_ids[@]}"; do
  for ((slot = 0; slot < JOBS_PER_GPU; slot++)); do
    run_slot "$gpu" &
    slot_pids+=("$!")
  done
done

failed=0
for pid in "${slot_pids[@]}"; do
  wait "$pid" || failed=1
done
if [[ $failed -ne 0 ]]; then
  echo "queue finished with failures; inspect $LOG_ROOT" >&2
  exit 1
fi

echo "[complete] run_id=$RUN_ID logs=$LOG_ROOT"

#!/usr/bin/env bash
# Method-transfer extension of the matched LeWM K-variant protocol.
# No new representation, Global predictor, threshold fit, or evaluation starts.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$REPO_ROOT"
PYTHON=${PYTHON:-$REPO_ROOT/.venv/bin/python}
NUM_CLUSTERS=${NUM_CLUSTERS:-4}
METHODS=${METHODS:-kmeanspp}
TASKS=${TASKS:-tworoom,pusht,reacher,cube}
TRAIN_SEEDS=${TRAIN_SEEDS:-0,42,625}
PARTITION_SEEDS=${PARTITION_SEEDS:-0,1,2}
EVAL_SEEDS=${EVAL_SEEDS:-0,1,2,3,4}
GPU_IDS=${GPU_IDS:-0,1,2,3,4}
EVAL_WORKERS=${EVAL_WORKERS:-2}
CPU_THREADS=${CPU_THREADS:-4}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/experiments/control_matrix/assets/lewm_k${NUM_CLUSTERS}_${METHODS}}
THRESHOLD_POLICY=${THRESHOLD_POLICY:-$REPO_ROOT/experiments/control_matrix/assets/lewm_k4_geometry_screen/frozen_bures_gate_policy.json}
GPU_CANDIDATES=${GPU_CANDIDATES:-0,1,2,3,4,5,6,7}
MAX_USED_MIB=${MAX_USED_MIB:-2048}
MAX_UTIL_PERCENT=${MAX_UTIL_PERCENT:-10}
STABLE_POLLS=${STABLE_POLLS:-3}
POLL_SECONDS=${POLL_SECONDS:-60}
[[ "$METHODS" != *,* ]] || { echo 'Use one partition method per variant run' >&2; exit 2; }
mkdir -p "$OUTPUT_DIR/logs/$RUN_ID"
LOG_ROOT=$OUTPUT_DIR/logs/$RUN_ID
exec 9>"$OUTPUT_DIR/coordinator.lock"
flock -n 9 || { echo 'Another coordinator is running for this variant' >&2; exit 2; }
echo $$ >"$LOG_ROOT/coordinator.pid"
status() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$1" | tee "$LOG_ROOT/status.txt"; }
trap 'rc=$?; if ((rc)); then status "FAILED exit=$rc; no automatic retry"; fi' EXIT
export PYTHON NUM_CLUSTERS METHODS TASKS TRAIN_SEEDS PARTITION_SEEDS EVAL_SEEDS GPU_IDS CPU_THREADS RUN_ID
export GPU_CANDIDATES MAX_USED_MIB MAX_UTIL_PERCENT STABLE_POLLS POLL_SECONDS
analysis=("$PYTHON" experiments/control_matrix/analyze_partition_variant.py --repo "$REPO_ROOT" --tasks "$TASKS"
  --num-clusters "$NUM_CLUSTERS" --method "$METHODS" --partition-seeds "$PARTITION_SEEDS"
  --training-seeds "$TRAIN_SEEDS" --evaluation-seeds "$EVAL_SEEDS" --cpu-threads "$CPU_THREADS"
  --threshold-policy "$THRESHOLD_POLICY" --output-dir "$OUTPUT_DIR")
status PREFLIGHT
"${analysis[@]}" --phase preflight
status PARTITION_AND_TRAINING
# Reuse the existing task/cache/checkpoint resolver and multi-GPU controller.
BACKGROUND=0 timeout --signal=TERM --kill-after=30s 72h bash experiments/control_matrix/scripts/run_lewm_k2.sh training
status COMPUTING_FROZEN_BURES
env OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS" \
  timeout --signal=TERM --kill-after=30s 24h "${analysis[@]}" --phase score >"$LOG_ROOT/bures_score.log" 2>&1
IFS=, read -r -a tasks <<<"$TASKS"
IFS=, read -r -a partitions <<<"$PARTITION_SEEDS"
IFS=, read -r -a trains <<<"$TRAIN_SEEDS"
task_names=(); partition_ids=(); train_ids=()
for task in "${tasks[@]}"; do
  for part in "${partitions[@]}"; do
    for train in "${trains[@]}"; do
      task_names+=("$task"); partition_ids+=("$part"); train_ids+=("$train")
    done
  done
done
eval_job() {
  local task=$1 part=$2 train=$3 data eval_name
  case "$task" in
    tworoom) data=${LAP_TWOROOM_DATA:-/data/sicong/weitao/datasets/lewm/tworoom.h5}; eval_name=tworoom ;;
    pusht) data=${LAP_PUSHT_DATA:-/data/sicong/weitao/datasets/lewm/pusht_expert_train.h5}; eval_name=pusht_expert_train ;;
    reacher) data=${LAP_REACHER_DATA:-/data/sicong/weitao/datasets/lewm/reacher.h5}; eval_name=reacher ;;
    cube) data=${LAP_CUBE_DATA:-/data/sicong/weitao/datasets/lewm/cube_single_expert.h5}; eval_name=ogbench/cube_single_expert ;;
    *) return 2 ;;
  esac
  # The immutable Spectral training manifest declares the exact matched checkpoint.
  local checkpoint
  checkpoint=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["pretrained_model"])' \
    "experiments/$task/matrix_k$NUM_CLUSTERS/training/spectral/partition${part}_train${train}/manifest.json")
  env DATASET_NAME="$task" DATA_FILE="$data" CHECKPOINT="$checkpoint" EVAL_CONFIG="$task" \
    EVAL_DATASET_NAME="$eval_name" CACHE_DIR="${CACHE_DIR:-/data/sicong/weitao/.stable_worldmodel}" \
    WORK_ROOT="experiments/$task/matrix_k${NUM_CLUSTERS}_long" PHASE=eval_regions \
    METHODS="$METHODS" PARTITION_SEEDS="$part" TRAIN_SEEDS="$train" EVAL_SEEDS="$EVAL_SEEDS" \
    GOAL_OFFSET=50 EVAL_BUDGET=50 SKIP_GLOBAL=1 SKIP_OFFICIAL=1 SKIP_JOINT=1 GPU_ID= \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" \
    OPENBLAS_NUM_THREADS="$CPU_THREADS" NUMEXPR_NUM_THREADS="$CPU_THREADS" \
    bash experiments/control_matrix/scripts/wait_for_free_gpu.sh -- \
      timeout --signal=TERM --kill-after=30s 12h bash experiments/control_matrix/scripts/run_lewm_matrix.sh
}
status WAITING_OR_RUNNING_LONG_EVAL
pids=()
for ((worker=0; worker<EVAL_WORKERS; worker++)); do
  (
    for ((i=worker; i<${#task_names[@]}; i+=EVAL_WORKERS)); do
      task=${task_names[$i]}; part=${partition_ids[$i]}; train=${train_ids[$i]}
      log="$LOG_ROOT/eval_${task}_p${part}_t${train}.log"
      printf '[eval-queue] task=%s partition=%s train=%s log=%s\n' "$task" "$part" "$train" "$log"
      eval_job "$task" "$part" "$train" >"$log" 2>&1
      printf '[eval-done] task=%s partition=%s train=%s\n' "$task" "$part" "$train"
    done
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed == 0)) || exit 1
status FINALIZING
"${analysis[@]}" --phase finalize >"$LOG_ROOT/finalize.log" 2>&1
status COMPLETE

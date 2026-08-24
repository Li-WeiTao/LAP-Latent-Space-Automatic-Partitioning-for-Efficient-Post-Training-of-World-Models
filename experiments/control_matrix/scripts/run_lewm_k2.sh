#!/usr/bin/env bash
# LeWM spectral-only partitioning and post-training for all paper tasks.
# K=2 remains the default; set NUM_CLUSTERS to reuse the identical protocol
# for a predeclared resolution sensitivity run such as K=4.
# Global is unchanged from K=3 and is intentionally not retrained here.
# Evaluation is intentionally excluded; launch it separately after every
# spectral predictor manifest has been verified.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

STAGE="${1:-training}"
if [[ "$STAGE" != "training" ]]; then
  echo "usage: $0 training" >&2
  exit 2
fi

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
CPU_THREADS="${CPU_THREADS:-4}"
NUM_CLUSTERS="${NUM_CLUSTERS:-2}"
TRAIN_SEEDS="${TRAIN_SEEDS:-0,42,625}"
PARTITION_SEEDS="${PARTITION_SEEDS:-0,1,2}"
EVAL_SEEDS="${EVAL_SEEDS:-0,1,2,3,4}"
TASKS="${TASKS:-tworoom,pusht,reacher,cube}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "$NUM_CLUSTERS" =~ ^[2-9][0-9]*$ ]] || {
  echo "NUM_CLUSTERS must be an integer of at least 2" >&2
  exit 2
}
K_TAG="k${NUM_CLUSTERS}"
LOG_ROOT="$REPO_ROOT/experiments/control_matrix/assets/lewm_${K_TAG}_logs"
ORCH_LOG="$LOG_ROOT/orchestrator_${RUN_ID}.log"
ORCH_PID="$LOG_ROOT/orchestrator_${RUN_ID}.pid"
mkdir -p "$LOG_ROOT"
[[ -x "$PYTHON" ]] || {
  echo "missing executable Python environment: $PYTHON" >&2
  exit 1
}

task_spec() {
  local task=$1
  case "$task" in
    tworoom)
      DATA_FILE="${LAP_TWOROOM_DATA:-/data/sicong/weitao/datasets/lewm/tworoom.h5}"
      CHECKPOINT="${LAP_TWOROOM_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt}"
      EVAL_CONFIG=tworoom
      EVAL_DATASET_NAME=tworoom
      SOURCE_CACHE="${LAP_TWOROOM_K_CACHE:-${LAP_TWOROOM_K2_CACHE:-/data/sicong/weitao/le-wm/experiments/real_gauge_drift/results/tworoom_latent_spectral_spectral_M20000_k30_P16_seed0_trainseed0/P_train_global_merged_embeddings.npz}}"
      ;;
    pusht)
      DATA_FILE="${LAP_PUSHT_DATA:-/data/sicong/weitao/datasets/lewm/pusht_expert_train.h5}"
      CHECKPOINT="${LAP_PUSHT_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/pusht/lewm_object.ckpt}"
      EVAL_CONFIG=pusht
      EVAL_DATASET_NAME=pusht_expert_train
      SOURCE_CACHE="${LAP_PUSHT_K_CACHE:-${LAP_PUSHT_K2_CACHE:-experiments/pusht/matrix/pusht_lewm_train_latent_cache.npz}}"
      ;;
    reacher)
      DATA_FILE="${LAP_REACHER_DATA:-/data/sicong/weitao/datasets/lewm/reacher.h5}"
      CHECKPOINT="${LAP_REACHER_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/reacher/lewm_object.ckpt}"
      EVAL_CONFIG=reacher
      EVAL_DATASET_NAME=reacher
      SOURCE_CACHE="${LAP_REACHER_K_CACHE:-${LAP_REACHER_K2_CACHE:-experiments/reacher/results/auto_gate_complete_k3/preparation/embedding_cache.npz}}"
      ;;
    cube)
      DATA_FILE="${LAP_CUBE_DATA:-/data/sicong/weitao/datasets/lewm/cube_single_expert.h5}"
      CHECKPOINT="${LAP_CUBE_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/cube/lewm_object.ckpt}"
      EVAL_CONFIG=cube
      EVAL_DATASET_NAME=ogbench/cube_single_expert
      SOURCE_CACHE="${LAP_CUBE_K_CACHE:-${LAP_CUBE_K2_CACHE:-experiments/cube/matrix/preparation/embedding_cache.npz}}"
      ;;
    *)
      echo "unknown task: $task" >&2
      return 2
      ;;
  esac
  WORK_ROOT="experiments/${task}/matrix_${K_TAG}"
}

prepare_work_root() {
  local source_abs source_sha target
  source_abs="$(realpath "$SOURCE_CACHE")"
  target="$WORK_ROOT/preparation/embedding_cache.npz"
  [[ -f "$DATA_FILE" ]] || { echo "missing dataset: $DATA_FILE" >&2; return 1; }
  [[ -f "$CHECKPOINT" ]] || { echo "missing checkpoint: $CHECKPOINT" >&2; return 1; }
  [[ -f "$source_abs" ]] || { echo "missing latent cache: $SOURCE_CACHE" >&2; return 1; }
  "$PYTHON" - "$source_abs" <<'PY'
import sys
import zipfile
from numpy.lib import format

path = sys.argv[1]
required = {"emb.npy", "act_emb.npy", "region_starts.npy"}
with zipfile.ZipFile(path) as archive:
    names = set(archive.namelist())
    missing = sorted(required - names)
    if missing:
        raise SystemExit(f"latent cache is not training-compatible; missing {missing}: {path}")
    headers = {}
    for name in sorted(required):
        with archive.open(name) as handle:
            version = format.read_magic(handle)
            reader = {
                (1, 0): format.read_array_header_1_0,
                (2, 0): format.read_array_header_2_0,
                (3, 0): format.read_array_header_2_0,
            }[version]
            shape, _, dtype = reader(handle)
            headers[name] = (shape, str(dtype))
emb_shape, emb_dtype = headers["emb.npy"]
act_shape, act_dtype = headers["act_emb.npy"]
starts_shape, starts_dtype = headers["region_starts.npy"]
if len(emb_shape) != 3 or len(act_shape) != 3 or emb_shape[0] != act_shape[0]:
    raise SystemExit(f"incompatible emb/act_emb shapes: {emb_shape} vs {act_shape}")
if starts_shape != (emb_shape[0],):
    raise SystemExit(f"incompatible region_starts shape: {starts_shape} for {emb_shape}")
print(
    f"[lewm-k] cache preflight: emb={emb_shape}/{emb_dtype} "
    f"act_emb={act_shape}/{act_dtype} region_starts={starts_shape}/{starts_dtype}"
)
PY
  source_sha="$(sha256sum "$source_abs" | awk '{print $1}')"
  if compgen -G "$WORK_ROOT/partitions/*/seed*/manifest.json" >/dev/null; then
    "$PYTHON" - "$WORK_ROOT" "$source_sha" <<'PY'
import glob
import json
import sys

work_root, expected_sha = sys.argv[1:]
mismatches = []
for path in sorted(glob.glob(f"{work_root}/partitions/*/seed*/manifest.json")):
    with open(path, encoding="utf-8") as handle:
        actual_sha = json.load(handle).get("latent_cache_sha256")
    if actual_sha != expected_sha:
        mismatches.append((path, actual_sha))
if mismatches:
    detail = "\n".join(f"  {path}: {sha}" for path, sha in mismatches)
    raise SystemExit(
        "existing partitions were fitted from a different latent cache; "
        "use a fresh work root or archive the failed attempt:\n" + detail
    )
PY
  fi
  mkdir -p "$WORK_ROOT/preparation" "$WORK_ROOT/manifests"
  if [[ -e "$target" && ! -L "$target" ]]; then
    echo "refusing to replace non-symlink cache: $target" >&2
    return 1
  fi
  ln -sfn "$source_abs" "$target"
  {
    echo "task=$task"
    echo "num_clusters=$NUM_CLUSTERS"
    echo "methods=spectral"
    echo "train_seeds=$TRAIN_SEEDS"
    echo "partition_seeds=$PARTITION_SEEDS"
    echo "source_cache=$source_abs"
    echo "source_cache_sha256=$source_sha"
    echo "global_policy=reuse_k3"
    echo "evaluation_policy=not_launched"
    echo "git_commit=$(git rev-parse HEAD)"
  } >"$WORK_ROOT/manifests/${K_TAG}_training.env"
}

run_task() {
  task_spec "$task"
  prepare_work_root
  local task_run_id="${RUN_ID}_${task}"
  echo "[lewm-k] K=$NUM_CLUSTERS task=$task work_root=$WORK_ROOT source_cache=$SOURCE_CACHE"
  env \
    DATASET_NAME="$task" DATA_FILE="$DATA_FILE" CHECKPOINT="$CHECKPOINT" \
    EVAL_CONFIG="$EVAL_CONFIG" EVAL_DATASET_NAME="$EVAL_DATASET_NAME" \
    CACHE_DIR=/data/sicong/weitao/.stable_worldmodel WORK_ROOT="$WORK_ROOT" \
    PYTHON="$PYTHON" GPU_IDS="$GPU_IDS" CPU_THREADS="$CPU_THREADS" \
    TRAIN_SEEDS="$TRAIN_SEEDS" PARTITION_SEEDS="$PARTITION_SEEDS" \
    EVAL_SEEDS="$EVAL_SEEDS" METHODS=spectral NUM_CLUSTERS="$NUM_CLUSTERS" \
    SKIP_JOINT=1 SKIP_GLOBAL=1 SKIP_OFFICIAL=1 TASK_RETRIES=1 \
    START_STAGE=partition END_STAGE=training RUN_ID="$task_run_id" \
    bash experiments/control_matrix/scripts/run_lewm_matrix_parallel.sh
  echo "[lewm-k] done K=$NUM_CLUSTERS task=$task"
}

{
  echo "[lewm-k] repo=$REPO_ROOT"
  echo "[lewm-k] commit=$(git rev-parse HEAD)"
  echo "[lewm-k] K=$NUM_CLUSTERS gpus=$GPU_IDS tasks=$TASKS"
  IFS=, read -r -a task_list <<<"$TASKS"
  for task in "${task_list[@]}"; do
    run_task "$task"
  done
  echo "[lewm-k] complete K=$NUM_CLUSTERS"
} >>"$ORCH_LOG" 2>&1 &

pid=$!
echo "$pid" >"$ORCH_PID"
echo "[lewm-k] orchestrator pid=$pid"
echo "[lewm-k] log=$ORCH_LOG"

#!/usr/bin/env bash
# OGBench Cube Le-WM multi-GPU control-matrix — detached, thin wrapper around
# experiments/control_matrix/scripts/run_lewm_matrix_parallel.sh. One worker
# per listed GPU; disjoint output directories; safe to re-run to resume.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

DATASET_NAME="${DATASET_NAME:-cube}"
DATA_FILE="${DATA_FILE:-/home/sicong/weitao/datasets/lewm/cube_single_expert.h5}"
: "${CHECKPOINT:?set CHECKPOINT to the official Le-WM Cube object checkpoint. None was found under /home/sicong/weitao or /data/sicong/weitao as of 2026-08-10 -- see experiments/cube/README.md 'Known blockers'.}"
EVAL_CONFIG="${EVAL_CONFIG:-cube}"
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-ogbench/cube_single_expert}"
WORK_ROOT="${WORK_ROOT:-experiments/cube/matrix}"
CACHE_DIR="${CACHE_DIR:-/data/sicong/weitao/.stable_worldmodel}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
GPU_IDS="${GPU_IDS:-0}"
CPU_THREADS="${CPU_THREADS:-4}"
TRAIN_SEEDS="${TRAIN_SEEDS:-0,42,625}"
PARTITION_SEEDS="${PARTITION_SEEDS:-0,1,2}"
EVAL_SEEDS="${EVAL_SEEDS:-0,1,2,3,4}"
METHODS="${METHODS:-random_voronoi,kmeanspp,spectral}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

LOG_DIR="$WORK_ROOT/logs"
LOG="$LOG_DIR/parallel_${RUN_ID}.detached.log"
PID_FILE="$LOG_DIR/parallel_${RUN_ID}.pid"
mkdir -p "$LOG_DIR"

launch_cmd=(
  env
  DATASET_NAME="$DATASET_NAME" DATA_FILE="$DATA_FILE" CHECKPOINT="$CHECKPOINT"
  EVAL_CONFIG="$EVAL_CONFIG" EVAL_DATASET_NAME="$EVAL_DATASET_NAME"
  WORK_ROOT="$WORK_ROOT" CACHE_DIR="$CACHE_DIR" PYTHON="$PYTHON"
  GPU_IDS="$GPU_IDS" CPU_THREADS="$CPU_THREADS"
  TRAIN_SEEDS="$TRAIN_SEEDS" PARTITION_SEEDS="$PARTITION_SEEDS" EVAL_SEEDS="$EVAL_SEEDS"
  METHODS="$METHODS" RUN_ID="$RUN_ID"
  bash experiments/control_matrix/scripts/run_lewm_matrix_parallel.sh
)

setsid nohup "${launch_cmd[@]}" >>"$LOG" 2>&1 &
pid=$!
echo "$pid" >"$PID_FILE"
echo "[cube-lewm-matrix] pid=$pid run_id=$RUN_ID"
echo "[cube-lewm-matrix] log=$LOG"
echo "[cube-lewm-matrix] tail: tail -f $LOG"
echo "[cube-lewm-matrix] stop: kill $pid"

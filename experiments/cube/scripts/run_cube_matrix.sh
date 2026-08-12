#!/usr/bin/env bash
# OGBench Cube Le-WM control-matrix — thin wrapper around the generic,
# task-agnostic control_matrix LeWM driver. Mirrors the "Eight-GPU matrix
# command" pattern documented in experiments/pusht/README.md. No Python
# logic is duplicated here; this only sets Cube-specific defaults
# (overridable from the environment) and execs the shared driver.
#
# PHASE defaults to "all" (prepare; partition; train; evaluate; aggregate).
# Every phase skips already-complete artifacts, so this script is safe to
# re-run to resume a partially finished matrix. See
# experiments/control_matrix/scripts/run_lewm_matrix.sh for the full phase
# list (prepare, partition_global, partition_regions, partition, partition_auto,
# train_joint, train_global, train_regions, train, train_auto, eval_official,
# eval_joint, eval_global, eval_regions, eval, eval_auto, auto_posttrain,
# aggregate, all).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export DATASET_NAME="${DATASET_NAME:-cube}"
export DATA_FILE="${DATA_FILE:-/home/sicong/weitao/datasets/lewm/cube_single_expert.h5}"
: "${CHECKPOINT:?set CHECKPOINT to the official Le-WM Cube object checkpoint (e.g. /data/sicong/weitao/.stable_worldmodel/cube/lewm_object.ckpt). None was found under /home/sicong/weitao or /data/sicong/weitao as of 2026-08-10 -- see experiments/cube/README.md 'Known blockers'.}"
export CHECKPOINT
export EVAL_CONFIG="${EVAL_CONFIG:-cube}"
export EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-ogbench/cube_single_expert}"
export WORK_ROOT="${WORK_ROOT:-experiments/cube/matrix}"
export CACHE_DIR="${CACHE_DIR:-/data/sicong/weitao/.stable_worldmodel}"
export PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
export GPU_ID="${GPU_ID:-0}"
export CPU_THREADS="${CPU_THREADS:-4}"
export TRAIN_SEEDS="${TRAIN_SEEDS:-0,42,625}"
export PARTITION_SEEDS="${PARTITION_SEEDS:-0,1,2}"
export EVAL_SEEDS="${EVAL_SEEDS:-0,1,2,3,4}"
export METHODS="${METHODS:-random_voronoi,kmeanspp,spectral}"
export PHASE="${PHASE:-all}"

exec bash experiments/control_matrix/scripts/run_lewm_matrix.sh

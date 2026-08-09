#!/usr/bin/env bash
# Shared Reacher Sub-JEPA paths (source from task-local scripts).
#
# CHECKPOINT has no default: the Sub-JEPA Reacher checkpoint path is not yet
# finalized (download in progress). Every script that actually needs the
# checkpoint must call require_checkpoint (defined below) before using it;
# merely sourcing this file must never fail and must never guess a path.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

export PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
export DATASET="${DATASET:-/home/sicong/weitao/datasets/lewm/reacher.h5}"
export CHECKPOINT="${CHECKPOINT:-}"
export TASK_SPEC="${TASK_SPEC:-configs/experiments/tasks/reacher.json}"
export CACHE_DIR="${CACHE_DIR:-/data/sicong/weitao/.stableworldmodel/subjepa/reacher}"

export WORK_ROOT="${WORK_ROOT:-experiments/reacher/subjepa}"
export SMOKE_ROOT="${SMOKE_ROOT:-experiments/reacher/subjepa}"
export FORMAL="${FORMAL:-experiments/reacher/subjepa/formal}"
export MATRIX="${MATRIX:-experiments/reacher/subjepa/matrix}"

# Canonical paired eval starts, reused from the existing Reacher LeWM matrix
# so Sub-JEPA and LeWM evaluate from identical initial qpos/qvel/goal_qpos
# states. Override if the local LeWM Reacher layout differs.
export CANON_SHORT="${CANON_SHORT:-experiments/reacher/matrix/eval/official}"
export CANON_LONG="${CANON_LONG:-experiments/reacher/matrix_long/eval/official}"

# require_checkpoint: call this from any script right before the checkpoint
# is actually used (training / eval / prepare). Never call it merely to
# source this file.
require_checkpoint() {
  if [[ -z "${CHECKPOINT:-}" ]]; then
    echo "ERROR: CHECKPOINT is not set." >&2
    echo "  The Sub-JEPA Reacher checkpoint path is not finalized yet (download in progress)." >&2
    echo "  Set it explicitly, e.g.:" >&2
    echo "    export CHECKPOINT=/path/to/subjepa_reacher_object.ckpt" >&2
    echo "  This script will not guess, download, or default a checkpoint path." >&2
    exit 1
  fi
}

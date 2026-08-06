#!/usr/bin/env bash
# Shared PushT Sub-JEPA paths (source from task-local scripts).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

export PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
export DATASET="${DATASET:-/home/sicong/weitao/datasets/lewm/pusht_expert_train.h5}"
export CHECKPOINT="${CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/pusht/subjepa_object.ckpt}"
export TASK_SPEC="${TASK_SPEC:-configs/experiments/tasks/pusht.json}"
export CACHE_DIR="${CACHE_DIR:-/data/sicong/weitao/.stableworldmodel/subjepa/pusht}"

export SMOKE_ROOT="${SMOKE_ROOT:-experiments/pusht/subjepa}"
export FORMAL="${FORMAL:-experiments/pusht/subjepa/formal}"
export MATRIX="${MATRIX:-experiments/pusht/subjepa/matrix}"

# Preserved smoke cache (see manifests/verification_status.json).
export SMOKE_CACHE_SHA256="${SMOKE_CACHE_SHA256:-3d2e75d1e347c1826b94ab47a474d8e97af0eb92a7a9f6f63a733dcb3177ec3e}"

# Reuse TwoRoom Sub-JEPA audit/materialization scripts (task-agnostic via --work-root).
export TWOROOM_FORMAL_SCRIPTS="${TWOROOM_FORMAL_SCRIPTS:-experiments/tworoom/subjepa/formal/scripts}"
export TWOROOM_MATRIX_SCRIPTS="${TWOROOM_MATRIX_SCRIPTS:-experiments/tworoom/subjepa/matrix/scripts}"

# Canonical paired eval starts (PushT matrix official baselines).
export CANON_SHORT="${CANON_SHORT:-experiments/pusht/matrix/eval/official}"
export CANON_LONG="${CANON_LONG:-experiments/pusht/matrix_long/eval/official}"

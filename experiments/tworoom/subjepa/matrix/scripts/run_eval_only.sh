#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"

export PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6}"
export CPU_THREADS="${CPU_THREADS:-4}"
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

bash experiments/tworoom/subjepa/matrix/scripts/run_full_matrix.sh eval-short
bash experiments/tworoom/subjepa/matrix/scripts/run_full_matrix.sh eval-long

echo "[eval-only] short and long complete"

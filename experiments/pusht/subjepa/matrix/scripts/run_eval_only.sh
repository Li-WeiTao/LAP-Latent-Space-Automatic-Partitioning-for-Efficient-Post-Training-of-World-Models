#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/experiments/pusht/subjepa/env.sh"

export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6}"
export CPU_THREADS="${CPU_THREADS:-4}"
export RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

RUN="$REPO_ROOT/experiments/pusht/subjepa/matrix/scripts/run_full_matrix.sh"
bash "$RUN" eval-short
bash "$RUN" eval-long

echo "[pusht-eval-only] short and long complete"

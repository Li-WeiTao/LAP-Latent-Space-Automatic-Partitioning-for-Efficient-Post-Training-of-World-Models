#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

source "$REPO_ROOT/.venv/bin/activate"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

export STABLEWM_HOME="${LAP_STABLEWM_HOME:-$REPO_ROOT/.stable_worldmodel}"
export PATH="$HOME/.local/bin:$PATH"

echo "LAP environment activated"
echo "PWD=$PWD"
echo "PYTHON=$(which python)"
echo "STABLEWM_HOME=$STABLEWM_HOME"
echo "GPU=${GPU:-0}"

#!/usr/bin/env bash
# PushT Auto-LAP symlinks — branch follows gate passport (global or spectral).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/experiments/pusht/subjepa/env.sh"

WORK_ROOT="${1:-$MATRIX}"
MODE="${2:-all}"
LOCK="$WORK_ROOT/manifests/pre_execution_lock.json"

if [[ ! -f "$LOCK" ]]; then
  echo "[auto-lap] missing pre_execution_lock: $LOCK" >&2
  exit 1
fi

read -r BRANCH DEPLOYMENT_SEED <<EOF
$("$PYTHON" -c "
import json
lock = json.load(open('$LOCK'))
print(lock['gate_selected_branch'], lock.get('deployment_seed', 0))
")
EOF

link_spectral_training() {
  for tseed in 0 42 625; do
    local src="$WORK_ROOT/training/spectral/partition${DEPLOYMENT_SEED}_train${tseed}"
    local dst="$WORK_ROOT/auto/training/train${tseed}"
    if [[ ! -f "$src/manifest.json" ]]; then
      echo "[auto-lap] skip spectral training train${tseed}: missing $src/manifest.json" >&2
      continue
    fi
    mkdir -p "$(dirname "$dst")"
    rm -rf "$dst"
    ln -sfn "$(realpath "$src")" "$dst"
    echo "[auto-lap] spectral train${tseed} -> $src"
  done
}

link_spectral_eval() {
  for tseed in 0 42 625; do
    for eseed in 0 1 2 3 4; do
      local src="$WORK_ROOT/eval/spectral/partition${DEPLOYMENT_SEED}_train${tseed}/eval${eseed}"
      local dst="$WORK_ROOT/auto/eval/train${tseed}/eval${eseed}"
      if [[ ! -f "$src/results.json" ]]; then
        echo "[auto-lap] skip spectral eval train${tseed}/eval${eseed}: missing $src/results.json" >&2
        continue
      fi
      mkdir -p "$(dirname "$dst")"
      rm -rf "$dst"
      ln -sfn "$(realpath "$src")" "$dst"
      echo "[auto-lap] spectral eval train${tseed}/eval${eseed} -> $src"
    done
  done
}

link_global_training() {
  for tseed in 0 42 625; do
    local src="$WORK_ROOT/training/global/train${tseed}"
    local dst="$WORK_ROOT/auto/training/train${tseed}"
    if [[ ! -f "$src/manifest.json" ]]; then
      echo "[auto-lap] skip global training train${tseed}: missing $src/manifest.json" >&2
      continue
    fi
    mkdir -p "$(dirname "$dst")"
    rm -rf "$dst"
    ln -sfn "$(realpath "$src")" "$dst"
    echo "[auto-lap] global train${tseed} -> $src"
  done
}

link_global_eval() {
  for tseed in 0 42 625; do
    for eseed in 0 1 2 3 4; do
      local src="$WORK_ROOT/eval/global/train${tseed}/eval${eseed}"
      local dst="$WORK_ROOT/auto/eval/train${tseed}/eval${eseed}"
      if [[ ! -f "$src/results.json" ]]; then
        echo "[auto-lap] skip global eval train${tseed}/eval${eseed}: missing $src/results.json" >&2
        continue
      fi
      mkdir -p "$(dirname "$dst")"
      rm -rf "$dst"
      ln -sfn "$(realpath "$src")" "$dst"
      echo "[auto-lap] global eval train${tseed}/eval${eseed} -> $src"
    done
  done
}

echo "[auto-lap] gate_selected_branch=$BRANCH deployment_seed=$DEPLOYMENT_SEED mode=$MODE"

case "$MODE" in
  training)
    if [[ "$BRANCH" == "spectral" ]]; then
      link_spectral_training
    else
      link_global_training
    fi
    ;;
  eval)
    if [[ "$BRANCH" == "spectral" ]]; then
      link_spectral_eval
    else
      link_global_eval
    fi
    ;;
  all)
    if [[ "$BRANCH" == "spectral" ]]; then
      link_spectral_training
      link_spectral_eval
    else
      link_global_training
      link_global_eval
    fi
    ;;
  *)
    echo "usage: $0 [work_root] {training|eval|all}" >&2
    exit 2
    ;;
esac

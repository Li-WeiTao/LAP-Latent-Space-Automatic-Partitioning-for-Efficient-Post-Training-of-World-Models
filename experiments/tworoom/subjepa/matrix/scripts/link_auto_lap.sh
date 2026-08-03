#!/usr/bin/env bash
# Symlink Auto-LAP training/eval artifacts to deployed Spectral (partition seed 0).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"

WORK_ROOT="${1:-experiments/tworoom/subjepa/matrix}"
DEPLOYMENT_SEED="${DEPLOYMENT_SEED:-0}"
MODE="${2:-all}"

link_training() {
  for tseed in 0 42 625; do
    local src="$WORK_ROOT/training/spectral/partition${DEPLOYMENT_SEED}_train${tseed}"
    local dst="$WORK_ROOT/auto/training/train${tseed}"
    if [[ ! -f "$src/manifest.json" ]]; then
      echo "[auto-lap] skip training train${tseed}: missing $src/manifest.json" >&2
      continue
    fi
    mkdir -p "$(dirname "$dst")"
    rm -rf "$dst"
    ln -sfn "$(realpath "$src")" "$dst"
    echo "[auto-lap] training train${tseed} -> $src"
  done
}

link_eval() {
  for tseed in 0 42 625; do
    for eseed in 0 1 2 3 4; do
      local src="$WORK_ROOT/eval/spectral/partition${DEPLOYMENT_SEED}_train${tseed}/eval${eseed}"
      local dst="$WORK_ROOT/auto/eval/train${tseed}/eval${eseed}"
      if [[ ! -f "$src/results.json" ]]; then
        echo "[auto-lap] skip eval train${tseed} eval${eseed}: missing $src/results.json" >&2
        continue
      fi
      mkdir -p "$(dirname "$dst")"
      rm -rf "$dst"
      ln -sfn "$(realpath "$src")" "$dst"
      echo "[auto-lap] eval train${tseed}/eval${eseed} -> $src"
    done
  done
}

case "$MODE" in
  training) link_training ;;
  eval) link_eval ;;
  all)
    link_training
    link_eval
    ;;
  *)
    echo "usage: $0 [work_root] {training|eval|all}" >&2
    exit 2
    ;;
esac

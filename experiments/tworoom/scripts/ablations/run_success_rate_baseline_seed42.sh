#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

mkdir -p experiments/tworoom/results/tworoom_success_rate_baseline_seed42

echo "==== baseline LeWM success rate (seed=42) started at $(date) ===="
/usr/bin/time -p python experiments/tworoom/tworoom_success_rate_eval.py \
  --mode baseline \
  --seed 42 \
  --checkpoint "${LAP_LEWM_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt}" \
  --out-dir experiments/tworoom/results/tworoom_success_rate_baseline_seed42 \
  > experiments/tworoom/results/tworoom_success_rate_baseline_seed42/run.log 2>&1

echo "==== baseline finished at $(date) ===="

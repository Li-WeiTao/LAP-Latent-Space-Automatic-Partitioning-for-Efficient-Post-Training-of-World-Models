#!/usr/bin/env bash
# Success-rate eval for rooms3 + priority5 with 50-epoch best region predictors.
# Reuses eval_start_indices from prior baseline runs (seeds 0-4).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

GPU="${CUDA_VISIBLE_DEVICES:-4}"
export CUDA_VISIBLE_DEVICES="${GPU}"

SEEDS=(0 1 2 3 4)
MODES=(rooms3 priority5)
ROOT="experiments/tworoom"
TAG="50ep"
SUMMARY="${ROOT}/results/tworoom_success_rate_${TAG}_5seed_summary.csv"
MASTER="${ROOT}/results/tworoom_success_rate_${TAG}_5seed_master.log"

echo "seed,mode,success_rate,successes,num_eval,out_dir" > "${SUMMARY}"

for seed in "${SEEDS[@]}"; do
  baseline_starts="${ROOT}/results/tworoom_success_rate_baseline_seed${seed}/results.json"
  if [[ ! -f "${baseline_starts}" ]]; then
    echo "Missing baseline starts: ${baseline_starts}" >&2
    exit 1
  fi

  for mode in "${MODES[@]}"; do
    out_dir="${ROOT}/results/tworoom_success_rate_${mode}_${TAG}_seed${seed}"
    mkdir -p "${out_dir}"
    log="${out_dir}/run.log"

    echo "==== ${TAG} ${mode} seed=${seed} started at $(date) ====" | tee -a "${MASTER}"

    /usr/bin/time -p python "${ROOT}/tworoom_success_rate_eval.py" \
      --mode "${mode}" \
      --seed "${seed}" \
      --out-dir "${out_dir}" \
      --eval-start-indices "${baseline_starts}" \
      > "${log}" 2>&1

    python - <<PY
import json
from pathlib import Path
p = Path("${out_dir}/results.json")
d = json.loads(p.read_text())
m = d["metrics"]
print(f"seed=${seed} mode=${mode} success_rate={m['success_rate']}% ({sum(m['episode_successes'])}/{d['num_eval']})")
with open("${SUMMARY}", "a") as f:
    f.write(f"${seed},${mode},{m['success_rate']},{sum(m['episode_successes'])},{d['num_eval']},${out_dir}\n")
PY

    echo "==== ${TAG} ${mode} seed=${seed} finished at $(date) ====" | tee -a "${MASTER}"
  done
done

python "${ROOT}/aggregate_success_rate_5seed.py" \
  --summary "${SUMMARY}" \
  --out-json "${ROOT}/results/tworoom_success_rate_${TAG}_5seed_summary.json"

echo "==== ${TAG} rooms3/priority5 5-seed eval finished at $(date) ===="

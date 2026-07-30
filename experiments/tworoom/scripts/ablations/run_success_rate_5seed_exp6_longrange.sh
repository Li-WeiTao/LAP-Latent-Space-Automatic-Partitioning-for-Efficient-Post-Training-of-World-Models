#!/usr/bin/env bash
# Experiment 6: long-range eval (goal_offset=50, eval_budget=50).
# Re-runs baseline (fresh starts sampled with goal_offset=50) and exp5 setup
# (rooms3 + priority5 with 80ep doorway, 50ep left/right, 30ep common/near_wall),
# reusing the exp6 baseline starts per seed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

GPU="${CUDA_VISIBLE_DEVICES:-4}"
export CUDA_VISIBLE_DEVICES="${GPU}"

SEEDS=(0 1 2 3 4)
ROOT="experiments/tworoom"
TAG="exp6"
GOAL_OFFSET=50
SUMMARY="${ROOT}/results/tworoom_success_rate_${TAG}_5seed_summary.csv"
MASTER="${ROOT}/results/tworoom_success_rate_${TAG}_5seed_master.log"

DOORWAY80_CKPT="${ROOT}/results/tworoom_geometry_train_region_predictors_doorway80ep/P_train_doorway_corridor_object.ckpt"

echo "seed,mode,success_rate,successes,num_eval,out_dir" > "${SUMMARY}"

run_one() {
  local seed="$1" mode="$2" extra_args=("${@:3}")
  local out_dir="${ROOT}/results/tworoom_success_rate_${mode}_${TAG}_seed${seed}"
  mkdir -p "${out_dir}"
  local log="${out_dir}/run.log"

  echo "==== ${TAG} ${mode} seed=${seed} started at $(date) ====" | tee -a "${MASTER}"

  /usr/bin/time -p python "${ROOT}/tworoom_success_rate_eval.py" \
    --mode "${mode}" \
    --seed "${seed}" \
    --out-dir "${out_dir}" \
    --goal-offset "${GOAL_OFFSET}" \
    "${extra_args[@]}" \
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
}

for seed in "${SEEDS[@]}"; do
  baseline_starts="${ROOT}/results/tworoom_success_rate_baseline_${TAG}_seed${seed}/results.json"

  run_one "${seed}" baseline --sample-eval-starts

  for mode in rooms3 priority5; do
    run_one "${seed}" "${mode}" \
      --eval-start-indices "${baseline_starts}" \
      --region-ckpt "doorway_corridor=${DOORWAY80_CKPT}"
  done
done

python "${ROOT}/aggregate_success_rate_5seed.py" \
  --summary "${SUMMARY}" \
  --out-json "${ROOT}/results/tworoom_success_rate_${TAG}_5seed_summary.json"

echo "==== ${TAG} long-range (goal_offset=${GOAL_OFFSET}) 5-seed eval finished at $(date) ===="

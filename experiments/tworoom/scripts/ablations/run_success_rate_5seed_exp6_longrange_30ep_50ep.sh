#!/usr/bin/env bash
# Exp6 long-range: exp2 (30ep) and exp4 (50ep) rooms3/priority5.
# Reuses existing exp6 baseline eval_start_indices per seed.
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
PRED_DIR="${ROOT}/results/tworoom_geometry_train_region_predictors"
SUMMARY="${ROOT}/results/tworoom_success_rate_${TAG}_30ep_50ep_5seed_summary.csv"
MASTER="${ROOT}/results/tworoom_success_rate_${TAG}_30ep_50ep_5seed_master.log"

echo "seed,mode,success_rate,successes,num_eval,out_dir" > "${SUMMARY}"

run_one() {
  local seed="$1" label="$2" eval_mode="$3" out_suffix="$4"
  shift 4
  local extra_args=("$@")
  local out_dir="${ROOT}/results/tworoom_success_rate_${out_suffix}_seed${seed}"
  local baseline_starts="${ROOT}/results/tworoom_success_rate_baseline_${TAG}_seed${seed}/results.json"
  mkdir -p "${out_dir}"
  local log="${out_dir}/run.log"

  if [[ ! -f "${baseline_starts}" ]]; then
    echo "Missing ${baseline_starts}" >&2
    exit 1
  fi

  echo "==== ${TAG} ${label} seed=${seed} started at $(date) ====" | tee -a "${MASTER}"

  /usr/bin/time -p python "${ROOT}/tworoom_success_rate_eval.py" \
    --mode "${eval_mode}" \
    --seed "${seed}" \
    --out-dir "${out_dir}" \
    --goal-offset "${GOAL_OFFSET}" \
    --eval-start-indices "${baseline_starts}" \
    "${extra_args[@]}" \
    > "${log}" 2>&1

  python - <<PY
import json
from pathlib import Path
p = Path("${out_dir}/results.json")
d = json.loads(p.read_text())
m = d["metrics"]
print(f"seed=${seed} ${label} success_rate={m['success_rate']}% ({sum(m['episode_successes'])}/{d['num_eval']})")
with open("${SUMMARY}", "a") as f:
    f.write(f"${seed},${label},{m['success_rate']},{sum(m['episode_successes'])},{d['num_eval']},${out_dir}\n")
PY

  echo "==== ${TAG} ${label} seed=${seed} finished at $(date) ====" | tee -a "${MASTER}"
}

# Exp2: 30ep geometry train∩region (epoch-30 ckpt for retrained regions)
CKPT30=(
  "doorway_corridor=${PRED_DIR}/P_train_doorway_corridor_epoch30_object.ckpt"
  "left_room=${PRED_DIR}/P_train_left_room_epoch30_object.ckpt"
  "right_room=${PRED_DIR}/P_train_right_room_epoch30_object.ckpt"
)

for seed in "${SEEDS[@]}"; do
  for eval_mode in rooms3 priority5; do
    args30=()
    for item in "${CKPT30[@]}"; do
      args30+=(--region-ckpt "${item}")
    done
    run_one "${seed}" "${eval_mode}_30ep" "${eval_mode}" "${eval_mode}_${TAG}_30ep" "${args30[@]}"

    # Exp4: 50ep best in geometry_train dir (default paths, no overrides)
    run_one "${seed}" "${eval_mode}_50ep" "${eval_mode}" "${eval_mode}_${TAG}_50ep"
  done
done

python "${ROOT}/aggregate_success_rate_5seed.py" \
  --summary "${SUMMARY}" \
  --out-json "${ROOT}/results/tworoom_success_rate_${TAG}_30ep_50ep_5seed_summary.json"

echo "==== exp6 30ep/50ep long-range finished at $(date) ===="

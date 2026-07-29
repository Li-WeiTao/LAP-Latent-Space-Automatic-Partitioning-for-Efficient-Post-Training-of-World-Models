#!/usr/bin/env bash
# Short-range (goal_offset=25) rooms3 + priority5 with explicit epoch-30 ckpts
# for doorway/left/right. Required because top-level P_train_*_object.ckpt were
# overwritten by 50ep best models after exp4 training.
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
TAG="30ep"
PRED_DIR="${ROOT}/results/tworoom_geometry_train_region_predictors"
SUMMARY="${ROOT}/results/tworoom_success_rate_${TAG}_5seed_summary.csv"
MASTER="${ROOT}/results/tworoom_success_rate_${TAG}_5seed_master.log"

CKPT30=(
  "doorway_corridor=${PRED_DIR}/P_train_doorway_corridor_epoch30_object.ckpt"
  "left_room=${PRED_DIR}/P_train_left_room_epoch30_object.ckpt"
  "right_room=${PRED_DIR}/P_train_right_room_epoch30_object.ckpt"
)

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

    args30=()
    for item in "${CKPT30[@]}"; do
      args30+=(--region-ckpt "${item}")
    done

    echo "==== ${TAG} ${mode} seed=${seed} started at $(date) ====" | tee -a "${MASTER}"

    /usr/bin/time -p python "${ROOT}/tworoom_success_rate_eval.py" \
      --mode "${mode}" \
      --seed "${seed}" \
      --out-dir "${out_dir}" \
      --eval-start-indices "${baseline_starts}" \
      "${args30[@]}" \
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

echo "==== ${TAG} short-range rooms3/priority5 5-seed eval finished at $(date) ===="

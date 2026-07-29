#!/usr/bin/env bash
# Run success-rate experiments 1–3 across 5 seeds (0–4).
# Per seed: baseline samples eval starts; rooms3/priority5 reuse that seed's baseline starts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

GPU="${CUDA_VISIBLE_DEVICES:-4}"
export CUDA_VISIBLE_DEVICES="${GPU}"

SEEDS=(0 1 2 3 4)
MODES=(baseline rooms3 priority5)
ROOT="experiments/tworoom"
PRED_DIR="${ROOT}/results/tworoom_geometry_train_region_predictors"
SUMMARY="${ROOT}/results/tworoom_success_rate_5seed_summary.csv"

# Top-level P_train_{doorway,left,right}_object.ckpt are 50ep best; use epoch30 for exp2/3.
CKPT30=(
  "doorway_corridor=${PRED_DIR}/P_train_doorway_corridor_epoch30_object.ckpt"
  "left_room=${PRED_DIR}/P_train_left_room_epoch30_object.ckpt"
  "right_room=${PRED_DIR}/P_train_right_room_epoch30_object.ckpt"
)

echo "seed,mode,success_rate,successes,num_eval,out_dir" > "${SUMMARY}"

for seed in "${SEEDS[@]}"; do
  for mode in "${MODES[@]}"; do
    out_dir="${ROOT}/results/tworoom_success_rate_${mode}_seed${seed}"
    mkdir -p "${out_dir}"
    log="${out_dir}/run.log"

    echo "==== ${mode} seed=${seed} started at $(date) ====" | tee -a "${ROOT}/results/tworoom_success_rate_5seed_master.log"

    extra_args=()
    if [[ "${mode}" == "baseline" ]]; then
      extra_args+=(--sample-eval-starts)
    else
      for item in "${CKPT30[@]}"; do
        extra_args+=(--region-ckpt "${item}")
      done
    fi

    /usr/bin/time -p python "${ROOT}/tworoom_success_rate_eval.py" \
      --mode "${mode}" \
      --seed "${seed}" \
      --out-dir "${out_dir}" \
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

    echo "==== ${mode} seed=${seed} finished at $(date) ====" | tee -a "${ROOT}/results/tworoom_success_rate_5seed_master.log"
  done
done

python "${ROOT}/aggregate_success_rate_5seed.py" --summary "${SUMMARY}"

echo "==== all 5-seed success-rate runs finished at $(date) ===="

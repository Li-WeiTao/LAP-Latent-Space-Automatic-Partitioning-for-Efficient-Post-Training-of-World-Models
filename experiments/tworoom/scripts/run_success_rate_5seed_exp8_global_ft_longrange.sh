#!/usr/bin/env bash
# Experiment 8 long-range: Global-FT compute-matched (goal_offset=50, 5 seeds).
# Reuses exp6 baseline eval_start_indices per seed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

GPU="${CUDA_VISIBLE_DEVICES:-4}"
export CUDA_VISIBLE_DEVICES="${GPU}"

SEEDS=(0 1 2 3 4)
ROOT="experiments/tworoom"
TAG="exp8_exp6"
GOAL_OFFSET=50
GLOBAL_FT_CKPT="${ROOT}/results/tworoom_geometry_train_global_ft_65ep/P_train_global_ft_object.ckpt"
SUMMARY="${ROOT}/results/tworoom_success_rate_exp8_global_ft_exp6_5seed_summary.csv"
MASTER="${ROOT}/results/tworoom_success_rate_exp8_global_ft_exp6_5seed_master.log"

if [[ ! -f "${GLOBAL_FT_CKPT}" ]]; then
  echo "Missing Global-FT checkpoint: ${GLOBAL_FT_CKPT}" >&2
  exit 1
fi

echo "seed,mode,success_rate,successes,num_eval,out_dir" > "${SUMMARY}"

for seed in "${SEEDS[@]}"; do
  baseline_starts="${ROOT}/results/tworoom_success_rate_baseline_exp6_seed${seed}/results.json"
  if [[ ! -f "${baseline_starts}" ]]; then
    echo "Missing exp6 baseline starts: ${baseline_starts}" >&2
    exit 1
  fi

  out_dir="${ROOT}/results/tworoom_success_rate_global_ft_65ep_${TAG}_seed${seed}"
  mkdir -p "${out_dir}"
  log="${out_dir}/run.log"

  echo "==== exp8 global_ft longrange seed=${seed} started at $(date) ====" | tee -a "${MASTER}"

  /usr/bin/time -p python "${ROOT}/tworoom_success_rate_eval.py" \
    --mode baseline \
    --checkpoint "${GLOBAL_FT_CKPT}" \
    --seed "${seed}" \
    --out-dir "${out_dir}" \
    --goal-offset "${GOAL_OFFSET}" \
    --eval-start-indices "${baseline_starts}" \
    > "${log}" 2>&1

  python - <<PY
import json
from pathlib import Path
p = Path("${out_dir}/results.json")
d = json.loads(p.read_text())
m = d["metrics"]
print(f"seed=${seed} global_ft longrange success_rate={m['success_rate']}% ({sum(m['episode_successes'])}/{d['num_eval']})")
with open("${SUMMARY}", "a") as f:
    f.write(f"${seed},global_ft,{m['success_rate']},{sum(m['episode_successes'])},{d['num_eval']},${out_dir}\n")
PY

  echo "==== exp8 global_ft longrange seed=${seed} finished at $(date) ====" | tee -a "${MASTER}"
done

python "${ROOT}/aggregate_success_rate_5seed.py" \
  --summary "${SUMMARY}" \
  --out-json "${ROOT}/results/tworoom_success_rate_exp8_global_ft_exp6_5seed_summary.json"

echo "==== exp8 global_ft long-range 5-seed eval finished at $(date) ===="

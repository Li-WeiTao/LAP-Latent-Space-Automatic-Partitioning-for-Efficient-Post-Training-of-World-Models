#!/usr/bin/env bash
# Experiment 7: goal_offset=100 (task length), eval_budget=50 (unchanged).
# Usage: GPU=0 MODEL=baseline bash run_success_rate_exp7_goal100_model.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

GPU="${GPU:?set GPU=0..N}"
export CUDA_VISIBLE_DEVICES="${GPU}"
MODEL="${MODEL:?set MODEL=baseline|rooms3_50ep|global_ft_50ep|latent_kmeanspp}"

SEEDS=(0 1 2 3 4)
ROOT="experiments/tworoom"
TAG="exp7"
GOAL_OFFSET=100
EVAL_BUDGET=50
GLOBAL_CKPT="${LAP_LEWM_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt}"
GLOBAL_FT50="${ROOT}/results/tworoom_geometry_train_global_ft_50ep/P_train_global_ft_object.ckpt"
KMEANS_DIR="${ROOT}/results/latent_kmeanspp_multirestart_k3"
ZSCORE_PARAMS="${KMEANS_DIR}/zscore_params.npz"
MASTER="${ROOT}/results/tworoom_success_rate_${TAG}_${MODEL}_master.log"

wait_for_baseline_starts() {
  local seed="$1"
  local starts="${ROOT}/results/tworoom_success_rate_baseline_${TAG}_seed${seed}/results.json"
  while [[ ! -f "${starts}" ]]; do
    echo "waiting for baseline ${TAG} seed=${seed} starts..." | tee -a "${MASTER}"
    sleep 30
  done
}

run_seed() {
  local seed="$1"
  local out_dir="$2"
  shift 2
  local extra_args=("$@")

  mkdir -p "${out_dir}"
  local log="${out_dir}/run.log"
  echo "==== ${TAG} ${MODEL} seed=${seed} GPU=${GPU} started at $(date) ====" | tee -a "${MASTER}"

  /usr/bin/time -p python "${ROOT}/tworoom_success_rate_eval.py" \
    --seed "${seed}" \
    --out-dir "${out_dir}" \
    --goal-offset "${GOAL_OFFSET}" \
    --eval-budget "${EVAL_BUDGET}" \
    "${extra_args[@]}" \
    > "${log}" 2>&1

  python - <<PY
import json
from pathlib import Path
p = Path("${out_dir}/results.json")
d = json.loads(p.read_text())
m = d["metrics"]
print(f"${MODEL} seed=${seed} success_rate={m['success_rate']}% ({sum(m['episode_successes'])}/{d['num_eval']}) goal_offset=${GOAL_OFFSET} budget=${EVAL_BUDGET}")
PY

  echo "==== ${TAG} ${MODEL} seed=${seed} finished at $(date) ====" | tee -a "${MASTER}"
}

case "${MODEL}" in
  baseline)
    SUMMARY="${ROOT}/results/tworoom_success_rate_${TAG}_baseline_5seed_summary.csv"
    echo "seed,mode,success_rate,successes,num_eval,out_dir" > "${SUMMARY}"
    for seed in "${SEEDS[@]}"; do
      out_dir="${ROOT}/results/tworoom_success_rate_baseline_${TAG}_seed${seed}"
      run_seed "${seed}" "${out_dir}" \
        --mode baseline \
        --checkpoint "${GLOBAL_CKPT}" \
        --sample-eval-starts
      python - <<PY
import json
from pathlib import Path
d = json.loads(Path("${out_dir}/results.json").read_text())
m = d["metrics"]
with open("${SUMMARY}", "a") as f:
    f.write(f"${seed},baseline,{m['success_rate']},{sum(m['episode_successes'])},{d['num_eval']},${out_dir}\n")
PY
    done
    python "${ROOT}/aggregate_success_rate_5seed.py" \
      --summary "${SUMMARY}" \
      --out-json "${ROOT}/results/tworoom_success_rate_${TAG}_baseline_5seed_summary.json"
    ;;

  rooms3_50ep)
    SUMMARY="${ROOT}/results/tworoom_success_rate_${TAG}_rooms3_50ep_5seed_summary.csv"
    echo "seed,mode,success_rate,successes,num_eval,out_dir" > "${SUMMARY}"
    for seed in "${SEEDS[@]}"; do
      wait_for_baseline_starts "${seed}"
      out_dir="${ROOT}/results/tworoom_success_rate_rooms3_${TAG}_seed${seed}"
      baseline_starts="${ROOT}/results/tworoom_success_rate_baseline_${TAG}_seed${seed}/results.json"
      run_seed "${seed}" "${out_dir}" \
        --mode rooms3 \
        --eval-start-indices "${baseline_starts}"
      python - <<PY
import json
from pathlib import Path
d = json.loads(Path("${out_dir}/results.json").read_text())
m = d["metrics"]
with open("${SUMMARY}", "a") as f:
    f.write(f"${seed},rooms3,{m['success_rate']},{sum(m['episode_successes'])},{d['num_eval']},${out_dir}\n")
PY
    done
    python "${ROOT}/aggregate_success_rate_5seed.py" \
      --summary "${SUMMARY}" \
      --out-json "${ROOT}/results/tworoom_success_rate_${TAG}_rooms3_50ep_5seed_summary.json"
    ;;

  global_ft_50ep)
    SUMMARY="${ROOT}/results/tworoom_success_rate_${TAG}_global_ft_50ep_5seed_summary.csv"
    echo "seed,mode,success_rate,successes,num_eval,out_dir" > "${SUMMARY}"
    for seed in "${SEEDS[@]}"; do
      wait_for_baseline_starts "${seed}"
      out_dir="${ROOT}/results/tworoom_success_rate_global_ft_50ep_${TAG}_seed${seed}"
      baseline_starts="${ROOT}/results/tworoom_success_rate_baseline_${TAG}_seed${seed}/results.json"
      run_seed "${seed}" "${out_dir}" \
        --mode baseline \
        --checkpoint "${GLOBAL_FT50}" \
        --eval-start-indices "${baseline_starts}"
      python - <<PY
import json
from pathlib import Path
d = json.loads(Path("${out_dir}/results.json").read_text())
m = d["metrics"]
with open("${SUMMARY}", "a") as f:
    f.write(f"${seed},global_ft_50ep,{m['success_rate']},{sum(m['episode_successes'])},{d['num_eval']},${out_dir}\n")
PY
    done
    python "${ROOT}/aggregate_success_rate_5seed.py" \
      --summary "${SUMMARY}" \
      --out-json "${ROOT}/results/tworoom_success_rate_${TAG}_global_ft_50ep_5seed_summary.json"
    ;;

  latent_kmeanspp)
    for OUTER_SEED in 0 1 2; do
      LABEL_NPZ="${KMEANS_DIR}/labels/kmeanspp_R50_outer${OUTER_SEED}.npz"
      PRED_DIR="${ROOT}/results/tworoom_latent_kmeanspp_kmeanspp_R50_outer${OUTER_SEED}"
      SUMMARY="${ROOT}/results/tworoom_success_rate_latent_kmeanspp_R50_outer${OUTER_SEED}_${TAG}_5seed_summary.csv"
      echo "seed,mode,success_rate,successes,num_eval,classify_ms_per_step,out_dir" > "${SUMMARY}"
      for seed in "${SEEDS[@]}"; do
        wait_for_baseline_starts "${seed}"
        out_dir="${ROOT}/results/tworoom_success_rate_latent_kmeanspp_R50_outer${OUTER_SEED}_${TAG}_seed${seed}"
        baseline_starts="${ROOT}/results/tworoom_success_rate_baseline_${TAG}_seed${seed}/results.json"
        run_seed "${seed}" "${out_dir}" \
          --mode latent_cluster3 \
          --checkpoint "${GLOBAL_CKPT}" \
          --eval-start-indices "${baseline_starts}" \
          --kmeanspp-label-npz "${LABEL_NPZ}" \
          --zscore-params "${ZSCORE_PARAMS}" \
          --cluster-predictor-dir "${PRED_DIR}"
        python - <<PY
import json
from pathlib import Path
d = json.loads(Path("${out_dir}/results.json").read_text())
m = d["metrics"]
cls_ms = d.get("inference_classify_per_step_ms", float("nan"))
with open("${SUMMARY}", "a") as f:
    f.write(f"${seed},latent_cluster3,{m['success_rate']},{sum(m['episode_successes'])},{d['num_eval']},{cls_ms},${out_dir}\n")
PY
      done
      python "${ROOT}/aggregate_success_rate_5seed.py" \
        --summary "${SUMMARY}" \
        --out-json "${ROOT}/results/tworoom_success_rate_latent_kmeanspp_R50_outer${OUTER_SEED}_${TAG}_5seed_summary.json"
    done
    ;;

  *)
    echo "Unknown MODEL=${MODEL}" >&2
    exit 1
    ;;
esac

echo "==== ${TAG} ${MODEL} DONE at $(date) ====" | tee -a "${MASTER}"

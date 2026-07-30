#!/usr/bin/env bash
# Long-range success rate for one K-means++ outer seed with per-imagined-step routing.
# Usage: OUTER_SEED=0 CUDA_VISIBLE_DEVICES=4 bash run_success_rate_5seed_latent_kmeanspp_step_routing_longrange.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

GPU="${CUDA_VISIBLE_DEVICES:-4}"
export CUDA_VISIBLE_DEVICES="${GPU}"

ROOT="experiments/tworoom"
OUTER_SEED="${OUTER_SEED:-0}"
INNER_BUDGET="${INNER_BUDGET:-50}"
GOAL_OFFSET="${GOAL_OFFSET:-50}"
EVAL_BUDGET="${EVAL_BUDGET:-50}"
KMEANS_DIR="${ROOT}/results/latent_kmeanspp_multirestart_k3"
LABEL_NPZ="${KMEANS_DIR}/labels/kmeanspp_R${INNER_BUDGET}_outer${OUTER_SEED}.npz"
ZSCORE_PARAMS="${KMEANS_DIR}/zscore_params.npz"
PRED_DIR="${ROOT}/results/tworoom_latent_kmeanspp_kmeanspp_R${INNER_BUDGET}_outer${OUTER_SEED}"
TAG="kmeanspp_R${INNER_BUDGET}_outer${OUTER_SEED}_step_routing"
CKPT="${LAP_LEWM_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt}"
SEEDS=(0 1 2 3 4)
SUMMARY="${ROOT}/results/tworoom_success_rate_latent_kmeanspp_${TAG}_5seed_summary.csv"
MASTER="${ROOT}/results/tworoom_success_rate_latent_kmeanspp_${TAG}_5seed_master.log"

for cluster_id in 0 1 2; do
  if [[ ! -f "${PRED_DIR}/P_train_cluster${cluster_id}_object.ckpt" ]]; then
    echo "Missing cluster predictor: ${PRED_DIR}/P_train_cluster${cluster_id}_object.ckpt" >&2
    exit 1
  fi
done

echo "seed,mode,success_rate,successes,num_eval,route_switch_rate,cluster0_fraction,cluster1_fraction,cluster2_fraction,evaluation_time_sec,out_dir" > "${SUMMARY}"

for seed in "${SEEDS[@]}"; do
  baseline_starts="${ROOT}/results/tworoom_success_rate_baseline_exp6_seed${seed}/results.json"
  if [[ ! -f "${baseline_starts}" ]]; then
    echo "Missing exp6 baseline starts: ${baseline_starts}" >&2
    exit 1
  fi

  out_dir="${ROOT}/results/tworoom_success_rate_latent_kmeanspp_${TAG}_seed${seed}"
  mkdir -p "${out_dir}"
  log="${out_dir}/run.log"

  echo "==== latent_kmeanspp step routing outer=${OUTER_SEED} eval_seed=${seed} started at $(date) ====" | tee -a "${MASTER}"

  if [[ ! -f "${out_dir}/results.json" ]]; then
    /usr/bin/time -p python "${ROOT}/tworoom_success_rate_eval.py" \
      --mode latent_cluster3 \
      --latent-routing step \
      --checkpoint "${CKPT}" \
      --seed "${seed}" \
      --out-dir "${out_dir}" \
      --goal-offset "${GOAL_OFFSET}" \
      --eval-budget "${EVAL_BUDGET}" \
      --eval-start-indices "${baseline_starts}" \
      --kmeanspp-label-npz "${LABEL_NPZ}" \
      --zscore-params "${ZSCORE_PARAMS}" \
      --cluster-predictor-dir "${PRED_DIR}" \
      > "${log}" 2>&1
  else
    echo "Reusing completed ${out_dir}/results.json" | tee -a "${MASTER}"
  fi

  python - <<PY
import json
from pathlib import Path

p = Path("${out_dir}/results.json")
d = json.loads(p.read_text())
if d.get("latent_routing") != "step":
    raise RuntimeError(f"Expected latent_routing=step in {p}, got {d.get('latent_routing')!r}")
m = d["metrics"]
frac = d["inference_route_fraction"]
switch_rate = d["inference_route_switch_rate"]
print(
    f"outer=${OUTER_SEED} eval_seed=${seed} success_rate={m['success_rate']}% "
    f"route_switch_rate={switch_rate:.6f} route_fraction={frac}"
)
with open("${SUMMARY}", "a") as f:
    f.write(
        f"${seed},latent_cluster3_step,{m['success_rate']},"
        f"{sum(m['episode_successes'])},{d['num_eval']},{switch_rate},"
        f"{frac['cluster0']},{frac['cluster1']},{frac['cluster2']},"
        f"{d['evaluation_time_sec']},${out_dir}\n"
    )
PY

  echo "==== latent_kmeanspp step routing outer=${OUTER_SEED} eval_seed=${seed} finished at $(date) ====" | tee -a "${MASTER}"
done

python "${ROOT}/aggregate_success_rate_5seed.py" \
  --summary "${SUMMARY}" \
  --out-json "${ROOT}/results/tworoom_success_rate_latent_kmeanspp_${TAG}_5seed_summary.json"

echo "==== latent_kmeanspp per-step long-range 5-seed eval finished at $(date) ===="

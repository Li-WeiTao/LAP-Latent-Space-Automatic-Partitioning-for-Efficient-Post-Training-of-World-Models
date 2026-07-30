#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"

GPU="${GPU:?set GPU to one available physical GPU id}"
RANDOM_SEED="${RANDOM_SEED:-0}"
TRAIN_SEED="${TRAIN_SEED:-42}"
HORIZON="${HORIZON:-long}"
export CUDA_VISIBLE_DEVICES="${GPU}"

ROOT=experiments/tworoom
ARTIFACT_DIR="${ROOT}/results/latent_random_voronoi_k3/random_voronoi_k3_seed${RANDOM_SEED}"
PRED_DIR="${ROOT}/results/tworoom_latent_random_voronoi_k3_seed${RANDOM_SEED}_trainseed${TRAIN_SEED}"
if [[ "${HORIZON}" == "long" ]]; then
  GOAL_OFFSET=50
  BASELINE_PREFIX="tworoom_success_rate_baseline_exp6_seed"
elif [[ "${HORIZON}" == "short" ]]; then
  GOAL_OFFSET=25
  BASELINE_PREFIX="tworoom_success_rate_baseline_seed"
else
  echo "HORIZON must be short or long" >&2
  exit 1
fi

TAG="random_voronoi_k3_seed${RANDOM_SEED}_trainseed${TRAIN_SEED}_${HORIZON}"
SUMMARY="${ROOT}/results/tworoom_success_rate_${TAG}_5eval_summary.csv"
MASTER="${ROOT}/results/tworoom_success_rate_${TAG}_5eval_master.log"
echo "seed,mode,success_rate,successes,num_eval,out_dir" > "${SUMMARY}"

for eval_seed in 0 1 2 3 4; do
  starts="${ROOT}/results/${BASELINE_PREFIX}${eval_seed}/results.json"
  out_dir="${ROOT}/results/tworoom_success_rate_${TAG}_evalseed${eval_seed}"
  mkdir -p "${out_dir}"
  /usr/bin/time -p nice -n 10 python "${ROOT}/tworoom_success_rate_eval.py" \
    --mode latent_cluster3 \
    --checkpoint "${LAP_LEWM_CHECKPOINT:-/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt}" \
    --seed "${eval_seed}" \
    --out-dir "${out_dir}" \
    --goal-offset "${GOAL_OFFSET}" \
    --eval-start-indices "${starts}" \
    --cluster-artifact-dir "${ARTIFACT_DIR}" \
    --cluster-predictor-dir "${PRED_DIR}" \
    --latent-routing mpc \
    > "${out_dir}/run.log" 2>&1
  python - "${out_dir}/results.json" "${SUMMARY}" "${eval_seed}" "${out_dir}" <<'PY'
import json
import sys

d = json.load(open(sys.argv[1]))
m = d["metrics"]
with open(sys.argv[2], "a") as handle:
    handle.write(
        f"{sys.argv[3]},random_voronoi_k3,{m['success_rate']},"
        f"{sum(m['episode_successes'])},{d['num_eval']},{sys.argv[4]}\n"
    )
print(f"eval_seed={sys.argv[3]} success_rate={m['success_rate']}%")
PY
done

python "${ROOT}/aggregate_success_rate_5seed.py" \
  --summary "${SUMMARY}" \
  --out-json "${ROOT}/results/tworoom_success_rate_${TAG}_5eval_summary.json" \
  >> "${MASTER}" 2>&1

#!/usr/bin/env bash
# Long-horizon evaluation for one spectral partition and one predictor-train seed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export STABLEWM_HOME="${STABLEWM_HOME:-${LAP_STABLEWM_HOME:-/data/sicong/weitao/.stable_worldmodel}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

GPU="${GPU:?set GPU to one available physical GPU id}"
export CUDA_VISIBLE_DEVICES="${GPU}"

ROOT="experiments/tworoom"
SPECTRAL_SEED="${SPECTRAL_SEED:-0}"
TRAIN_SEED="${TRAIN_SEED:-42}"
LATENT_ROUTING="${LATENT_ROUTING:-mpc}"
SPECTRAL_ROOT="${SPECTRAL_ROOT:-${ROOT}/results/latent_landmark_spectral_k3}"
if [[ -z "${ARTIFACT_DIR:-}" ]]; then
  SUMMARY="${SPECTRAL_ROOT}/stability_summary.json"
  if [[ -f "${SUMMARY}" ]]; then
    ARTIFACT_DIR="$(python - "${SUMMARY}" "${SPECTRAL_SEED}" <<'PY'
import json
import sys
from pathlib import Path
summary = json.load(open(sys.argv[1]))
seed = sys.argv[2]
by_seed = summary.get("artifacts_by_seed", {})
if seed in by_seed:
    print(by_seed[seed])
else:
    matches = [p for p in summary.get("artifacts", []) if Path(p).name.endswith(f"seed{seed}")]
    if len(matches) != 1:
        raise SystemExit(f"summary has no unique artifact for seed {seed}")
    print(matches[0])
PY
)"
  else
    shopt -s nullglob
    candidates=(
      "${SPECTRAL_ROOT}"/spectral_cfg*_seed"${SPECTRAL_SEED}"
      "${SPECTRAL_ROOT}"/spectral_M*_seed"${SPECTRAL_SEED}"
    )
    shopt -u nullglob
    if (( ${#candidates[@]} != 1 )); then
      echo "Expected exactly one spectral artifact for seed ${SPECTRAL_SEED}; found ${#candidates[@]}. Set ARTIFACT_DIR explicitly." >&2
      printf '  %s\n' "${candidates[@]}" >&2
      exit 1
    fi
    ARTIFACT_DIR="${candidates[0]}"
  fi
fi
artifact_tag="$(basename "${ARTIFACT_DIR}")"
PRED_DIR="${PRED_DIR:-${ROOT}/results/tworoom_latent_spectral_${artifact_tag}_trainseed${TRAIN_SEED}}"
TAG="spectral_${artifact_tag}_trainseed${TRAIN_SEED}_${LATENT_ROUTING}"
CKPT="${CKPT:-${LAP_LEWM_CHECKPOINT:-${STABLEWM_HOME}/tworoom/lewm_object.ckpt}}"
GOAL_OFFSET=50
SEEDS=(0 1 2 3 4)
SUMMARY="${ROOT}/results/tworoom_success_rate_latent_${TAG}_5seed_summary.csv"
MASTER="${ROOT}/results/tworoom_success_rate_latent_${TAG}_5seed_master.log"

num_clusters="$(python - "${ARTIFACT_DIR}/cluster_meta.json" <<'PY'
import json
import sys
print(int(json.load(open(sys.argv[1]))["num_clusters"]))
PY
)"
for ((k = 0; k < num_clusters; k++)); do
  [[ -f "${PRED_DIR}/P_train_cluster${k}_object.ckpt" ]] || {
    echo "Missing cluster${k} predictor under ${PRED_DIR}" >&2
    exit 1
  }
done

echo "seed,mode,success_rate,successes,num_eval,classify_ms_per_route_call,out_dir" > "${SUMMARY}"
for seed in "${SEEDS[@]}"; do
  baseline_starts="${ROOT}/results/tworoom_success_rate_baseline_exp6_seed${seed}/results.json"
  [[ -f "${baseline_starts}" ]] || { echo "Missing ${baseline_starts}" >&2; exit 1; }
  out_dir="${ROOT}/results/tworoom_success_rate_latent_${TAG}_evalseed${seed}"
  mkdir -p "${out_dir}"
  log="${out_dir}/run.log"
  echo "==== ${TAG} eval_seed=${seed} started at $(date) ====" | tee -a "${MASTER}"
  /usr/bin/time -p nice -n 10 python "${ROOT}/tworoom_success_rate_eval.py" \
    --mode latent_cluster3 \
    --checkpoint "${CKPT}" \
    --seed "${seed}" \
    --out-dir "${out_dir}" \
    --goal-offset "${GOAL_OFFSET}" \
    --eval-start-indices "${baseline_starts}" \
    --cluster-artifact-dir "${ARTIFACT_DIR}" \
    --cluster-predictor-dir "${PRED_DIR}" \
    --latent-routing "${LATENT_ROUTING}" \
    > "${log}" 2>&1
  python - <<PY
import json
from pathlib import Path
d = json.loads(Path("${out_dir}/results.json").read_text())
m = d["metrics"]
cls_ms = d.get("inference_classify_per_call_ms", d.get("inference_classify_per_step_ms", float("nan")))
print(f"spectral_seed=${SPECTRAL_SEED} train_seed=${TRAIN_SEED} eval_seed=${seed} success_rate={m['success_rate']}%")
with open("${SUMMARY}", "a") as f:
    f.write(f"${seed},latent_cluster${num_clusters}_${LATENT_ROUTING},{m['success_rate']},{sum(m['episode_successes'])},{d['num_eval']},{cls_ms},${out_dir}\n")
PY
  echo "==== ${TAG} eval_seed=${seed} finished at $(date) ====" | tee -a "${MASTER}"
done

python "${ROOT}/aggregate_success_rate_5seed.py" \
  --summary "${SUMMARY}" \
  --out-json "${ROOT}/results/tworoom_success_rate_latent_${TAG}_5seed_summary.json"
echo "==== spectral long-range 5-seed eval finished at $(date) ===="

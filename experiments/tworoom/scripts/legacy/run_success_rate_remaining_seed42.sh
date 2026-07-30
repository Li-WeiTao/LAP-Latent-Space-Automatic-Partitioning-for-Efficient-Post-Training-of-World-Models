#!/usr/bin/env bash
# Resume seed=42 geometry success-rate eval (NO latent/cluster).
# Skips completed short 30ep + 50ep runs; finishes exp5, exp8, exp6 long-range.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
source .venv/bin/activate
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"
ROOT="experiments/tworoom"
LOG="${ROOT}/results/success_rate_remaining_seed42.log"
SEEDS=(0 1 2 3 4)
DOORWAY80="${ROOT}/results/tworoom_geometry_train_region_predictors_doorway80ep/P_train_doorway_corridor_object.ckpt"
GLOBAL_FT="${ROOT}/results/tworoom_geometry_train_global_ft_65ep/P_train_global_ft_object.ckpt"

log() { echo "$*" | tee -a "${LOG}"; }

run_eval() {
  local label="$1"; shift
  log "==== ${label} started at $(date) ===="
  /usr/bin/time -p python "${ROOT}/tworoom_success_rate_eval.py" "$@" >> "${LOG}" 2>&1
  log "==== ${label} finished at $(date) ===="
}

log "==== remaining seed=42 success-rate eval GPU=${GPU} at $(date) ===="

# --- exp5 short: finish seeds 1-4 (seed 0 done 2026-07-13) ---
EXP5_SUMMARY="${ROOT}/results/tworoom_success_rate_exp5_5seed_summary.csv"
for seed in 1 2 3 4; do
  baseline="${ROOT}/results/tworoom_success_rate_baseline_seed${seed}/results.json"
  [[ -f "${baseline}" ]] || { log "missing ${baseline}"; exit 1; }
  for mode in rooms3 priority5; do
    out="${ROOT}/results/tworoom_success_rate_${mode}_exp5_seed${seed}"
    run_eval "exp5 ${mode} seed=${seed}" \
      --mode "${mode}" --seed "${seed}" --out-dir "${out}" \
      --eval-start-indices "${baseline}" \
      --region-ckpt "doorway_corridor=${DOORWAY80}"
  done
done

python - <<'PY' | tee -a "${LOG}"
import csv, json
from pathlib import Path
root = Path("experiments/tworoom/results")
summary = root / "tworoom_success_rate_exp5_5seed_summary.csv"
rows = []
for seed in range(5):
    for mode in ("rooms3", "priority5"):
        p = root / f"tworoom_success_rate_{mode}_exp5_seed{seed}" / "results.json"
        d = json.loads(p.read_text())
        m = d["metrics"]
        rows.append({
            "seed": seed, "mode": mode,
            "success_rate": m["success_rate"],
            "successes": sum(m["episode_successes"]),
            "num_eval": d["num_eval"],
            "out_dir": str(p.parent),
        })
with summary.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"rebuilt {summary}")
PY

python "${ROOT}/aggregate_success_rate_5seed.py" \
  --summary "${EXP5_SUMMARY}" \
  --out-json "${ROOT}/results/tworoom_success_rate_exp5_5seed_summary.json"

# --- exp8 short (seed=42 global_ft ckpt) ---
EXP8_SUMMARY="${ROOT}/results/tworoom_success_rate_exp8_global_ft_5seed_summary.csv"
echo "seed,mode,success_rate,successes,num_eval,out_dir" > "${EXP8_SUMMARY}"
for seed in "${SEEDS[@]}"; do
  baseline="${ROOT}/results/tworoom_success_rate_baseline_seed${seed}/results.json"
  out="${ROOT}/results/tworoom_success_rate_global_ft_65ep_seed${seed}"
  run_eval "exp8 short seed=${seed}" \
    --mode baseline --checkpoint "${GLOBAL_FT}" --seed "${seed}" \
    --out-dir "${out}" --eval-start-indices "${baseline}"
  python - <<PY >> "${EXP8_SUMMARY}"
import json
from pathlib import Path
p = Path("${out}/results.json")
d = json.loads(p.read_text())
m = d["metrics"]
print(f"${seed},global_ft,{m['success_rate']},{sum(m['episode_successes'])},{d['num_eval']},${out}")
PY
done
python "${ROOT}/aggregate_success_rate_5seed.py" \
  --summary "${EXP8_SUMMARY}" \
  --out-json "${ROOT}/results/tworoom_success_rate_exp8_global_ft_5seed_summary.json"

# --- exp6 long-range (baseline + exp5 doorway80) ---
bash "${ROOT}/scripts/ablations/run_success_rate_5seed_exp6_longrange.sh"

# --- exp6 long-range 30ep / 50ep ---
bash "${ROOT}/scripts/ablations/run_success_rate_5seed_exp6_longrange_30ep_50ep.sh"

# --- exp8 long-range ---
bash "${ROOT}/scripts/ablations/run_success_rate_5seed_exp8_global_ft_longrange.sh"

log "==== remaining seed=42 success-rate eval DONE at $(date) ===="

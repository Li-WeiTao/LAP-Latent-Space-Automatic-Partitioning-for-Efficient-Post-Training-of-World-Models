#!/usr/bin/env bash
# PushT Sub-JEPA matrix setup — mirrors tworoom/subjepa/matrix/scripts/setup_matrix.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=/dev/null
source "$REPO_ROOT/experiments/pusht/subjepa/env.sh"

mkdir -p "$MATRIX/manifests" "$MATRIX/logs" "$MATRIX/partitions" "$MATRIX/training"

"$PYTHON" - <<PY
import json
import hashlib
import sys
from pathlib import Path

repo = Path(".")
formal = repo / "$FORMAL"
passport_path = formal / "manifests/material_passport.json"
if not passport_path.is_file():
    print(f"PRE-LOCK FAILED: missing passport: {passport_path}", file=sys.stderr)
    sys.exit(1)
passport = json.loads(passport_path.read_text(encoding="utf-8"))

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

errors = []
if passport.get("verification_status") != "VERIFIED":
    errors.append(f"gate not VERIFIED: {passport.get('verification_status')}")
if passport.get("selected_branch") != "spectral":
    errors.append(f"expected spectral branch, got {passport.get('selected_branch')}")
gate = passport.get("gate_task_summary", {})
if gate.get("deployment_seed") != 0:
    errors.append(f"expected deployment_seed=0, got {gate.get('deployment_seed')}")
if not passport.get("replay_audit", {}).get("all_passed"):
    errors.append("replay audit not all_passed")

cache_path = formal / "preparation/embedding_cache.npz"
if not cache_path.exists():
    errors.append(f"missing full cache: {cache_path}")
cache_sha = sha256_file(cache_path) if cache_path.exists() else None
if cache_sha != passport.get("full_cache_sha256"):
    errors.append(f"cache sha mismatch: {cache_sha} != {passport.get('full_cache_sha256')}")

if errors:
    print("PRE-LOCK FAILED:", file=sys.stderr)
    for item in errors:
        print(f"  - {item}", file=sys.stderr)
    sys.exit(1)
print("[pre-lock] gate VERIFIED, spectral seed=0, cache hash ok")
PY

# --- read-only cache link ---
PREP="$MATRIX/preparation"
rm -rf "$PREP"
ln -sfn "$(realpath "$FORMAL/preparation")" "$PREP"

# --- reuse formal spectral partitions (remap cluster_labels to global IDs) ---
"$PYTHON" "$TWOROOM_MATRIX_SCRIPTS/materialize_spectral_partitions.py" \
  --formal-root "$FORMAL/partitions/spectral" \
  --matrix-root "$MATRIX/partitions/spectral" \
  --latent-cache "$FORMAL/preparation/embedding_cache.npz" \
  --seeds 0,1,2

# --- paired evaluation starts from canonical PushT matrix official eval ---
PAIR_SHORT="$MATRIX/paired_starts/canon_short/eval/official"
PAIR_LONG="$MATRIX/paired_starts/canon_long/eval/official"
mkdir -p "$PAIR_SHORT" "$PAIR_LONG"
for seed in 0 1 2 3 4; do
  short_src="$CANON_SHORT/eval${seed}/results.json"
  long_src="$CANON_LONG/eval${seed}/results.json"
  [[ -f "$short_src" ]] || { echo "missing canonical short eval seed $seed: $short_src" >&2; exit 1; }
  [[ -f "$long_src" ]] || { echo "missing canonical long eval seed $seed: $long_src" >&2; exit 1; }
  mkdir -p "$PAIR_SHORT/eval${seed}" "$PAIR_LONG/eval${seed}"
  cp "$short_src" "$PAIR_SHORT/eval${seed}/results.json"
  cp "$long_src" "$PAIR_LONG/eval${seed}/results.json"
done

LOCK="$MATRIX/manifests/pre_execution_lock.json"
"$PYTHON" - <<PY
import json, hashlib, subprocess
from pathlib import Path

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8<<20), b""):
            h.update(c)
    return h.hexdigest()

repo = Path("$REPO_ROOT")
lock = {
    "schema_version": 1,
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "gate_passport": str(repo / "$FORMAL/manifests/material_passport.json"),
    "gate_status": "VERIFIED",
    "selected_branch": "spectral",
    "deployment_seed": 0,
    "sha256": {
        "dataset": sha256_file(Path("$DATASET")),
        "checkpoint": sha256_file(Path("$CHECKPOINT")),
        "full_cache": sha256_file(repo / "$FORMAL/preparation/embedding_cache.npz"),
        "task_spec": sha256_file(repo / "$TASK_SPEC"),
    },
    "task_spec_path": str(repo / "$TASK_SPEC"),
    "eval_config_name": "pusht",
    "paired_start_sources": {
        "short": str(repo / "$CANON_SHORT"),
        "long": str(repo / "$CANON_LONG"),
    },
}
out = repo / "$LOCK"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(lock, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(lock, indent=2))
PY

# --- global partition on full cache (once) ---
if [[ ! -f "$MATRIX/partitions/global/seed0/manifest.json" ]]; then
  echo "[setup] fitting global partition on full cache"
  CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" "$PYTHON" experiments/control_matrix/fit_partition.py \
    --method global \
    --dataset-name pusht \
    --latent-cache "$FORMAL/preparation/embedding_cache.npz" \
    --frameskip 5 \
    --cpu-threads 4 \
    --out-dir "$MATRIX/partitions/global/seed0"
fi

# --- auto-lap deployment partition symlink ---
AUTO="$MATRIX/auto/partition"
rm -rf "$AUTO"
mkdir -p "$(dirname "$AUTO")"
ln -sfn "$(realpath "$FORMAL/gate/partition")" "$AUTO"

echo "[setup] matrix ready under $MATRIX"
echo "[setup] pre_execution_lock=$LOCK"

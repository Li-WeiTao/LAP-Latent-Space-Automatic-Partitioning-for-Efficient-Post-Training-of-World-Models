#!/usr/bin/env bash
# Shared Sub-JEPA matrix K-variant. Source after the task sets K3_MATRIX to the
# canonical K=3 matrix root (experiments/<task>/subjepa/matrix).
#
# NUM_CLUSTERS=3 (default): unchanged K=3 protocol.
# NUM_CLUSTERS=2: write to ${K3_MATRIX}_k2, methods=spectral only, reuse the
# K=3 global partition/training/eval and paired starts, and refit+train+eval
# spectral with K=2. Official / k-means++ / Auto-LAP are skipped.

apply_subjepa_k_variant() {
  if [[ -z "${K3_MATRIX:-}" ]]; then
    echo "K3_MATRIX must be set before apply_subjepa_k_variant" >&2
    return 1
  fi
  NUM_CLUSTERS="${NUM_CLUSTERS:-3}"
  MATRIX="${MATRIX:-$K3_MATRIX}"
  if [[ "$NUM_CLUSTERS" == "3" ]]; then
    MATRIX_METHODS="${MATRIX_METHODS:-kmeanspp,spectral}"
    SKIP_OFFICIAL="${SKIP_OFFICIAL:-0}"
    SKIP_GLOBAL="${SKIP_GLOBAL:-0}"
    PARTITION_INCLUDE_SPECTRAL="${PARTITION_INCLUDE_SPECTRAL:-0}"
    INCLUDE_AUTO_LAP="${INCLUDE_AUTO_LAP:-1}"
    REUSE_K3_GLOBAL="${REUSE_K3_GLOBAL:-0}"
    SKIP_SPECTRAL_MATERIALIZE="${SKIP_SPECTRAL_MATERIALIZE:-0}"
    SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
  else
    if [[ "$MATRIX" == "$K3_MATRIX" ]]; then
      MATRIX="${K3_MATRIX}_k${NUM_CLUSTERS}"
    fi
    MATRIX_METHODS="${MATRIX_METHODS:-spectral}"
    SKIP_OFFICIAL="${SKIP_OFFICIAL:-1}"
    SKIP_GLOBAL="${SKIP_GLOBAL:-1}"
    PARTITION_INCLUDE_SPECTRAL="${PARTITION_INCLUDE_SPECTRAL:-1}"
    INCLUDE_AUTO_LAP="${INCLUDE_AUTO_LAP:-0}"
    REUSE_K3_GLOBAL="${REUSE_K3_GLOBAL:-1}"
    SKIP_SPECTRAL_MATERIALIZE="${SKIP_SPECTRAL_MATERIALIZE:-1}"
    SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-1}"
  fi
  export NUM_CLUSTERS MATRIX K3_MATRIX MATRIX_METHODS
  export SKIP_OFFICIAL SKIP_GLOBAL PARTITION_INCLUDE_SPECTRAL
  export INCLUDE_AUTO_LAP REUSE_K3_GLOBAL SKIP_SPECTRAL_MATERIALIZE SKIP_PREFLIGHT
}

reuse_k3_global_artifacts() {
  local k3="${1:-$K3_MATRIX}"
  local dst="${2:-$MATRIX}"
  if [[ "$REUSE_K3_GLOBAL" != "1" ]]; then
    return 1
  fi
  if [[ ! -f "$k3/partitions/global/seed0/manifest.json" ]]; then
    echo "[k-variant] missing K=3 global partition: $k3/partitions/global/seed0/manifest.json" >&2
    return 1
  fi
  if [[ ! -f "$k3/training/global/train0/manifest.json" ]]; then
    echo "[k-variant] missing K=3 global training: $k3/training/global/train0/manifest.json" >&2
    return 1
  fi
  mkdir -p "$dst/partitions" "$dst/training"
  ln -sfn "$(realpath "$k3/partitions/global")" "$dst/partitions/global"
  ln -sfn "$(realpath "$k3/training/global")" "$dst/training/global"
  echo "[k-variant] reused global partition+training from $k3"
}

reuse_k3_paired_starts() {
  local k3="${1:-$K3_MATRIX}"
  local dst="${2:-$MATRIX}"
  if [[ "$NUM_CLUSTERS" == "3" ]]; then
    return 1
  fi
  if [[ ! -d "$k3/paired_starts" ]]; then
    echo "[k-variant] missing K=3 paired starts: $k3/paired_starts" >&2
    return 1
  fi
  mkdir -p "$dst"
  if [[ -L "$dst/paired_starts" || ! -e "$dst/paired_starts" ]]; then
    ln -sfn "$(realpath "$k3/paired_starts")" "$dst/paired_starts"
  fi
  echo "[k-variant] reused paired starts from $k3/paired_starts"
  return 0
}

wipe_eval_for_k_variant() {
  local root="${1:-$MATRIX}"
  if [[ "$NUM_CLUSTERS" == "3" ]]; then
    rm -rf "$root/eval/official" "$root/eval/global" \
      "$root/eval/kmeanspp" "$root/eval/spectral"
    return 0
  fi
  rm -rf "$root/eval/spectral"
  if [[ "$SKIP_GLOBAL" != "1" ]]; then
    rm -rf "$root/eval/global"
  fi
  if [[ "$SKIP_OFFICIAL" != "1" ]]; then
    rm -rf "$root/eval/official"
  fi
}

seed_reused_eval_from_k3() {
  local label=$1
  local root="${2:-$MATRIX}"
  local src="$K3_MATRIX/eval_${label}"
  if [[ "$SKIP_GLOBAL" != "1" ]]; then
    return 0
  fi
  [[ -d "$src/global" ]] || {
    echo "[k-variant] missing K=3 global eval: $src/global" >&2
    return 1
  }
  mkdir -p "$root/eval"
  rm -rf "$root/eval/global"
  cp -a "$src/global" "$root/eval/global"
  echo "[k-variant] reused $src/global -> $root/eval/global"
}

write_k_variant_lock() {
  local root="${1:-$MATRIX}"
  local python_bin="${PYTHON:-python}"
  mkdir -p "$root/manifests"
  "$python_bin" - "$root" "$K3_MATRIX" "$NUM_CLUSTERS" "$MATRIX_METHODS" <<'PY'
import json, subprocess, sys
from pathlib import Path

root = Path(sys.argv[1])
k3 = Path(sys.argv[2])
lock = {
    "schema_version": 1,
    "num_clusters": int(sys.argv[3]),
    "methods": sys.argv[4],
    "k3_matrix": str(k3),
    "work_root": str(root),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
    "reused": {
        "global_partition": True,
        "global_training": True,
        "global_eval": True,
        "paired_starts": True,
        "latent_cache": True,
    },
    "skipped": ["official", "kmeanspp", "auto_lap", "gate"],
}
out = root / "manifests" / "k_variant_lock.json"
out.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
print(f"[k-variant] wrote {out}")
PY
}

aggregate_k_variant_flags() {
  AGG_EXTRA=()
  if [[ "${INCLUDE_AUTO_LAP:-0}" == "1" ]]; then
    AGG_EXTRA+=(--include-auto-lap --deployment-seed "${DEPLOYMENT_SEED:-0}")
  fi
  if [[ "${SKIP_OFFICIAL:-0}" == "1" ]]; then
    AGG_EXTRA+=(--skip-official)
  fi
}

build_repeated_gpu_ids() {
  local gpu=$1
  local workers=$2
  local i out=""
  for ((i = 0; i < workers; i++)); do
    out+="${gpu},"
  done
  echo "${out%,}"
}

resolve_train_gpu_ids() {
  if [[ -n "${TRAIN_GPU_IDS:-}" ]]; then
    echo "$TRAIN_GPU_IDS"
    return 0
  fi
  local gpu="${TRAIN_GPU:-${GPU_IDS%%,*}}"
  local workers="${TRAIN_WORKERS:-1}"
  build_repeated_gpu_ids "$gpu" "$workers"
}

resolve_partition_gpu_ids() {
  if [[ -n "${PARTITION_GPU_IDS:-}" ]]; then
    echo "$PARTITION_GPU_IDS"
    return 0
  fi
  echo "${EVAL_GPU_IDS:-${EVAL_GPU:-${GPU_IDS%%,*}}}"
}

resolve_eval_gpu_ids() {
  if [[ -n "${EVAL_GPU_IDS:-}" ]]; then
    echo "$EVAL_GPU_IDS"
    return 0
  fi
  echo "${EVAL_GPU:-${GPU_IDS%%,*}}"
}

# Pick GPU_IDS for run_jepa_matrix_parallel.sh from START_STAGE and env.
matrix_parallel_gpu_ids() {
  case "${START_STAGE:-training}" in
    partition)
      resolve_partition_gpu_ids
      ;;
    eval_short|eval_long)
      resolve_eval_gpu_ids
      ;;
    *)
      resolve_train_gpu_ids
      ;;
  esac
}

# Partition fits are VRAM-heavy; training packs multiple low-VRAM jobs per GPU.
matrix_train_needs_split() {
  [[ -n "${TRAIN_GPU_IDS:-}" || -n "${TRAIN_WORKERS:-}" ]] \
    && [[ "$(resolve_train_gpu_ids)" != "$(resolve_partition_gpu_ids)" ]]
}

apply_subjepa_k_variant

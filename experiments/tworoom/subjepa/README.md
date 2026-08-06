# Sub-JEPA Experiments (TwoRoom)

This directory holds Sub-JEPA outputs isolated from historical LeWM results.

**Verification status: `VERIFIED` (smoke, 2026-08-03)**  
See `manifests/verification_status.json`.

The initial cache-equivalence failure (`max_abs_diff ≈ 0.013`) was a **validator false
positive**: the old smoke check used direct per-window `encode_frames`, while production
cache uses unique-frame encoding with `exact_batch_shapes`. The preserved cache SHA-256
is `6828c6b5b7f87df33878ed43684821e975b4e5aa9e859a1ce00e1bf6f40ab3a7` (not rebuilt).

Reduced smoke eval at 100% for official/global/spectral proves pipeline wiring only,
not method comparison. Use full paired short/long evaluation for formal conclusions.

## Reproduction (smoke)

```bash
export PYTHON="${PYTHON:-python}"
export TASK_SPEC="configs/experiments/tasks/tworoom.json"
export DATASET="${DATASET:?path to tworoom.h5}"
export CHECKPOINT="${CHECKPOINT:?path to subjepa_object.ckpt}"

bash experiments/control_matrix/scripts/run_subjepa_matrix.sh \
  --task-spec "$TASK_SPEC" \
  --dataset "$DATASET" \
  --checkpoint "$CHECKPOINT" \
  --eval-config-name tworoom \
  --work-root experiments/tworoom/subjepa \
  --cache-dir "${CACHE_DIR:-$HOME/.stable_worldmodel}/subjepa/tworoom" \
  --max-train-starts 4096 \
  --phase smoke
```

Do not write into `experiments/tworoom/matrix` or other LeWM result directories.

## Full 50-epoch matrix

Formal gate, protocol parity, detached launch, training, paired short/long eval, audit,
and bootstrap are documented in **`matrix/README.md`**.

Quick start (detached, 8 GPUs):

```bash
bash experiments/tworoom/subjepa/matrix/scripts/launch_matrix_detached.sh
```

## Server runbook (sicong, Aug 2026)

End-to-end notes from bringing up Sub-JEPA TwoRoom on the sicong machine: environment,
re-encoding, matrix training, and MPC evaluation. Gate (`spectral` branch) was run on
another server with the same protocol; this server reused committed `formal/` artifacts
and rebuilt the full latent cache locally.

### Paths (important)

`/home/sicong/weitao` and `/data/sicong/weitao` are **different directories** on this
host. Scripts default to `/data/sicong/weitao/...`; use symlinks or explicit CLI paths.

| Artifact | Path on this server |
|----------|---------------------|
| Dataset | `/home/sicong/weitao/datasets/lewm/tworoom.h5` |
| Dataset symlink (for scripts) | `/data/sicong/weitao/datasets/lewm/tworoom.h5` → home path |
| Sub-JEPA checkpoint | `/data/sicong/weitao/.stable_worldmodel/tworoom/subjepa_object.ckpt` |
| Repo | `/data/sicong/weitao/LAP-Latent-Space-Automatic-Partitioning-for-Efficient-Post-Training-of-World-Models` |
| Full latent cache (local re-encode) | `preparation/embedding_cache.npz` (693,728 transitions, SHA `b2fab2bfc3127e7cefcad0d1dca409b9e39d156f55d518945bc9ab4ee417a900`) |
| Formal gate | `formal/` — branch `spectral`, deployment seed `0` |

Link the local cache into formal preparation before matrix setup:

```bash
ln -sfn "$(realpath preparation/embedding_cache.npz)" formal/preparation/embedding_cache.npz
ln -sfn "$(realpath preparation/spectral_embedding_cache.npz)" formal/preparation/spectral_embedding_cache.npz
```

Update `formal/manifests/material_passport.json` → `full_cache_sha256` to match the
local cache hash if `setup_matrix.sh` pre-lock fails.

### Python environment

From repository root:

```bash
uv venv .venv --python python3.10
source .venv/bin/activate
uv pip install -r requirements/tworoom.txt
uv pip install -e ".[test]" --no-deps
uv pip install pytest hdf5plugin 'imageio[ffmpeg]'
```

- **`hdf5plugin`**: required to read compressed `pixels` in `tworoom.h5`.
- **`imageio[ffmpeg]`**: required for MPC eval video export (`stable_worldmodel`).

Activate helper:

```bash
source activate_lap.sh
export STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel
```

Hardware: **7× RTX 3090** (GPU 0–6).

### 1. Smoke re-encode (4096 starts)

```bash
source activate_lap.sh
export GPU_ID=0 CUDA_VISIBLE_DEVICES=0 STABLEWM_HOME=/data/sicong/weitao/.stable_worldmodel
export PREPARE_OVERWRITE=1

bash experiments/control_matrix/scripts/run_subjepa_matrix.sh \
  --task-spec configs/experiments/tasks/tworoom.json \
  --dataset /home/sicong/weitao/datasets/lewm/tworoom.h5 \
  --checkpoint /data/sicong/weitao/.stable_worldmodel/tworoom/subjepa_object.ckpt \
  --eval-config-name tworoom \
  --work-root experiments/tworoom/subjepa \
  --cache-dir /data/sicong/weitao/.stable_worldmodel/subjepa/tworoom \
  --max-train-starts 4096 \
  --gpu-id 0 \
  --phase smoke
```

Remove stale `preparation/embedding_cache.npz.report.json` (without a matching `.npz`)
before the first run if `FastLatentCacheEncoder` refuses to overwrite.

### 2. Full re-encode (693,728 transitions)

```bash
export PREPARE_OVERWRITE=1
bash experiments/control_matrix/scripts/run_subjepa_matrix.sh \
  --task-spec configs/experiments/tasks/tworoom.json \
  --dataset /home/sicong/weitao/datasets/lewm/tworoom.h5 \
  --checkpoint /data/sicong/weitao/.stable_worldmodel/tworoom/subjepa_object.ckpt \
  --eval-config-name tworoom \
  --work-root experiments/tworoom/subjepa \
  --cache-dir /data/sicong/weitao/.stable_worldmodel/subjepa/tworoom \
  --gpu-id 0 \
  --phase prepare
```

Wall time on this server: ~30 minutes (909,609 unique frames).

### 3. Matrix training (Global-FT + K-means++ + Spectral only)

Gate conclusion elsewhere: **spectral** branch. Training scope: **21 jobs** (3 global +
9 kmeanspp + 9 spectral), 50 epochs each, train seeds `0,42,625`, partition seeds
`0,1,2`. Skip Random-Voronoi, Joint-Continue, Human.

```bash
PYTHON=$PWD/.venv/bin/python \
GPU_IDS=0,1,2,3,4,5,6 \
bash experiments/tworoom/subjepa/matrix/scripts/launch_matrix_detached.sh
```

Completed run (`RUN_ID=20260805T123145Z`): **21/21** manifests, ~10.5 h on 7×3090.

Monitor:

```bash
tail -f experiments/tworoom/subjepa/matrix/logs/detached_<RUN_ID>.log
find experiments/tworoom/subjepa/matrix/training -name manifest.json | wc -l
```

### 4. MPC evaluation (short + long)

Official Sub-JEPA checkpoint + all trained predictors; paired LeWM eval starts;
goal_offset **25** (short) and **50** (long); **22 parallel tasks per horizon** on
7 GPUs.

Completed on sicong (`RUN_ID=20260806T024200Z` short, `20260806T034626Z` long):
**110/110** `results.json` per horizon; aggregate in
`matrix/manifests/matrix_summary_{short,long}.json`.

```bash
# Short + long (sequential)
GPU_IDS=0,1,2,3,4,5,6 \
bash experiments/tworoom/subjepa/matrix/scripts/launch_eval_detached.sh

# Long only (after short is done)
bash experiments/tworoom/subjepa/matrix/scripts/launch_eval_long_detached.sh
```

Results: `matrix/eval_short/`, `matrix/eval_long/`.

Long eval requires `EVAL_GOAL_OFFSET` (not bare `GOAL_OFFSET`) in
`run_jepa_matrix_parallel.sh` so config resolution does not conflict with
`short_goal_offset=25` in the task spec.

Requires `imageio[ffmpeg]`.

See **`matrix/README.md`** for gate + matrix results and stage definitions.

## Results summary (sicong, Aug 2026)

Full tables and interpretation: **`matrix/README.md` → Results**.

### Formal gate (`formal/`, 2026-08-03)

| Field | Value |
|-------|-------|
| Status | `VERIFIED` |
| Selected branch | **spectral** (`spectrally_nondegenerate`) |
| Deployment seed | **0** |
| Artifacts | `formal/manifests/material_passport.json`, `formal/gate/partition/manifest.json` |

The gate **passed narrowly on the background check**: `R_K − T_bg ≈ 0.008`
(~1.7% above threshold). Safety retention `S_task ≈ 0.989` was comfortable.
Spectral eigengap after K=3 was small (`≈ 0.00069`), indicating weak but
non-degenerate partition signal in Sub-JEPA TwoRoom latents.

### 50-epoch matrix MPC (aggregate)

| Method | Short (offset 25) | Long (offset 50) |
|--------|-------------------|------------------|
| Official | 94.0% | 57.2% |
| Global-FT50 | **94.8% ± 0.0%** | **58.5% ± 0.6%** |
| K-means++ | 93.9% ± 0.1% | 57.9% ± 0.8% |
| Spectral | 94.0% ± 0.3% | 58.3% ± 0.5% |
| Auto-LAP | 93.9% ± 0.2% | 58.7% ± 1.4% |

Error bars: sample SD across fine-tuning seeds (after averaging partition and
eval seeds). Post-training helps on both horizons; regional methods do not
clearly beat Global-FT, consistent with the narrow gate margin and weak latent
partition structure on this task.

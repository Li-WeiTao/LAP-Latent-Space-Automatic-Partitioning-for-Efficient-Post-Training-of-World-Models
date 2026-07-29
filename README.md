# LAP: Latent-Space Auto-Partitioned Fine-Tuning for World Models

LAP is a low-cost post-training framework for latent world models. It freezes a
pretrained encoder, partitions the planning latent space without action labels,
fine-tunes one predictor per region, and routes among the regional predictors at
inference time.

The repository contains the complete TwoRoom experiment suite migrated from the
original LeWM development tree, including trajectory-deviation experiments and
all later geometry, clustering, spectral-partition, fine-tuning, routing,
planning, stability, and efficiency experiments. Compact result files and all
paper figures are versioned; datasets, embedding caches, videos, and model
checkpoints are regenerated or downloaded separately.

## Method

LAP consists of four stages:

1. Freeze the pretrained world-model encoder and encode the training trajectories.
2. Automatically partition the latent space. The main method uses lightweight
   landmark spectral partitioning.
3. Fine-tune a separate dynamics predictor on each partition while keeping the
   encoder fixed.
4. At each MPC replanning step, route the current latent state to one regional
   predictor and use that predictor within the current candidate rollout.

The online spectral router is deployed as a small spherical Voronoi lookup:
nearest prototype followed by prototype-to-region ownership. It uses only the
current latent representation; candidate actions, goals, task IDs, and future
observations are not router inputs.

## TwoRoom result snapshot

Long-horizon task success rates are aggregated across predictor fine-tuning
seeds 0, 42, and 625. The official pretrained checkpoint is reported without a
fine-tuning error bar.

| Method | Long-horizon success rate |
|---|---:|
| Official baseline | 49.2% |
| Joint-Continue FP32, 3 epochs | 53.1 +/- 1.2% |
| Global-FT, 50 epochs | 58.13 +/- 1.22% |
| Random-Voronoi K3, 50 epochs | 59.33 +/- 0.58% |
| K-means++ K3, 50 epochs | 58.44 +/- 1.08% |
| **LAP (Spectral K3), 50 epochs** | **61.38 +/- 0.63%** |
| Human rooms3 partition, 50 epochs | 59.87 +/- 1.51% |

![TwoRoom long-horizon results](experiments/tworoom/assets/long_horizon_metrics/tworoom_long_horizon_main.png)

The checked-in numbers are descriptive summaries of the completed runs. See
[`experiments/tworoom/LEGACY_EXPERIMENTS.md`](experiments/tworoom/LEGACY_EXPERIMENTS.md)
for the full experiment log and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for
the reproduction contract.

## Repository layout

```text
lap/
  interfaces/       architecture-neutral world-model protocol
  partition/        partition artifacts and method-facing entry points
  routing/          deployable Voronoi routing
  finetuning/       regional fine-tuning interfaces
backends/
  lewm/             LeWM adapter and MIT-licensed compatibility backend
experiments/
  tworoom/          exact migrated experiment programs, launchers, results, figures
requirements/       reproducible environment specifications
scripts/            repository-level validation utilities
```

## Installation

The completed TwoRoom experiments used Python 3.10 and CUDA. Create an isolated
environment and install the pinned experiment dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements/tworoom.txt
```

The dataset and pretrained checkpoint are intentionally not committed. Set:

```bash
export LAP_DATA_ROOT=/path/to/dataset-directory
export LAP_TWOROOM_DATA=/path/to/dataset-directory/tworoom.h5
export LAP_LEWM_CHECKPOINT=/path/to/lewm_object.ckpt
```

On the original experiment server these correspond to:

```text
LAP_DATA_ROOT=/data/sicong/weitao/datasets/lewm
LAP_TWOROOM_DATA=/data/sicong/weitao/datasets/lewm/tworoom.h5
/data/sicong/weitao/.stable_worldmodel/tworoom/lewm_object.ckpt
```

## Main reproduction path

Run commands from the repository root. Existing launchers accept `GPU`,
`TRAIN_SEED`, and output-path overrides.

```bash
# 1. Lossless frozen-encoder re-encoding (unique timestep backend)
python experiments/tworoom/unique_timestep_reencode.py --help

# 2. Automatic spectral partition
bash experiments/tworoom/scripts/run_latent_landmark_spectral.sh

# 3. Three regional predictors, 50 epochs
TRAIN_SEED=42 GPU_LIST="0 1 2" \
  bash experiments/tworoom/scripts/run_latent_spectral_train_predictors_50ep.sh

# 4. Long-horizon five-seed evaluation
TRAIN_SEED=42 LATENT_ROUTING=mpc \
  bash experiments/tworoom/scripts/run_success_rate_5seed_latent_spectral_longrange.sh
```

The complete commands for every migrated result remain in the TwoRoom experiment
document. The repository-level validation command is:

```bash
python scripts/validate_repository.py
```

## Reproducibility and provenance

- Source development tree commit: `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`.
- The source tree contained untracked experiment files; migration therefore uses
  a file-level manifest rather than claiming that all experiments existed in that
  commit.
- Result summaries preserve their original commands and paths as provenance.
- Large artifacts are excluded from Git and listed in
  [`ARTIFACTS.md`](ARTIFACTS.md).
- LeWM compatibility files retain the upstream MIT license in
  [`LICENSES/LEWM-LICENSE`](LICENSES/LEWM-LICENSE).

## License

LAP-specific code is released under Apache-2.0. Vendored or adapted LeWM files
remain under the upstream MIT license identified above.

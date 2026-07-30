# TwoRoom experiment inventory

This document lists the experiment families represented by the committed
TwoRoom code, compact result artifacts, figures, and raw text logs. Individual
seed-level run directories are enumerated in
[`EXPERIMENT_INVENTORY.csv`](EXPERIMENT_INVENTORY.csv); every migrated log is
listed with its byte size and SHA-256 digest in
[`LOG_MANIFEST.csv`](LOG_MANIFEST.csv).

## 1. Predictor trajectory-deviation analysis

- Global predictor versus region-specific predictor rollout deviation.
- Doorway-corridor, near-wall, common, left-room, and right-room regions.
- Per-step latent MSE, final-step MSE, cosine similarity, and inference error.
- Four-way dynamically switched predictor rollout compared with Global-FT.

## 2. Encoder and latent-space analysis

- Encoder gauge-drift controls and encoder seed sweeps.
- Train-global encoder/predictor joint analysis and resampling checks.
- Episode-held-out, unique-timestep latent probes for `rooms3` and `priority5`.
- Linear and nonlinear probes, five probe seeds, UMAP projections, and class F1.
- Predictor-rule train/test and episode-bootstrap analyses supporting
  action-free routing from the current latent state.

## 3. Unsupervised partition diagnostics

- Raw spherical K-means stability across seeds.
- `raw_l2`, `center_l2`, and `zscore_l2` preprocessing comparison.
- Fully converged preprocessing study.
- K-means++ restart-budget and outer-seed stability study.
- Random-Voronoi K=3 partitions.
- Landmark spectral K=3 partitioning with 20,000 landmarks, kNN `k=30`, and
  16 deployable Voronoi prototypes per region.
- Offline spectral-label versus online Voronoi-router fidelity checks.

## 4. Post-training baselines

- Official released LeWM checkpoint without post-training.
- Joint-Continue with the encoder and predictor jointly updated: archived
  BF16 1-epoch control and canonical FP32 1/3-epoch runs.
- Frozen-encoder Global-FT for 45, 50, and 65 predictor epochs.

## 5. Manual-partition predictor post-training

- `rooms3` regional predictors.
- `priority5` regional predictors.
- 30-, 50-, and 80-epoch sweeps.
- Doorway-corridor 80-epoch variant.
- Predictor fine-tuning seeds 0, 42, and 625 where available.

## 6. Automatically partitioned predictor post-training

- Random-Voronoi K3, 50 predictor epochs.
- K-means++ K3, 50 predictor epochs and 50 initialization restarts.
- Spectral K3, 50 predictor epochs.
- Partition seeds 0, 1, and 2 and predictor fine-tuning seeds 0, 42, and 625.

## 7. Task-success evaluation

- Official baseline, Joint-Continue, Global-FT, Random-Voronoi, K-means++,
  spectral LAP, `rooms3`, and `priority5` comparisons.
- Short- and long-horizon evaluations.
- Five paired evaluation seeds with identical episode start indices per seed.
- MPC-level latent routing as the main deployable protocol.
- Step-wise predicted-latent routing experiments as a routing ablation.
- Additional epoch, doorway, and evaluation-budget ablations.

## 8. Efficiency and implementation validation

- Predictor-routing and end-to-end inference latency benchmarks.
- Unique-timestep lossless re-encoding equivalence and throughput validation.
- Dense-cache loader, checkpoint migration, and routing-contract smoke tests.

## 9. Derived figures and aggregates

- Main TwoRoom long-horizon comparison figure.
- Short-horizon and manual-partition figures.
- Latent UMAP separability figures.
- Multi-seed probe figure and numerical source tables.
- Seed-level aggregation and paired-start audit artifacts.

The historical commands and detailed interpretation remain in
[`LEGACY_EXPERIMENTS.md`](LEGACY_EXPERIMENTS.md). Dense latent caches, datasets,
videos, and model checkpoints remain external artifacts as documented in
[`ARTIFACTS.md`](ARTIFACTS.md); text logs and compact numerical outputs are
versioned in this repository.

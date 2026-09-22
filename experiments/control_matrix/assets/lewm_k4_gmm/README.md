# LeWM K=4 Gaussian Mixture partition transfer

Supplementary four-task partition-method experiment on the Layer-1 resolution.
It does not recalibrate the Bures threshold or change original validation totals.

## Predeclared configuration

- scikit-learn `GaussianMixture`, 4 components, full covariance, init_params=kmeans;
  n_init=1, max_iter=1000, tol=1e-3, reg_covar=1e-6. No planning-based parameter selection.
  The initial default-100 iteration run failed convergence on all three TwoRoom seeds;
  the user authorized a restart with only the iteration cap increased to 1000.
  Original failure logs are retained. This does not relax the convergence tolerance.
- Same Z-score/L2 transformation and unique training latent cache as existing experiments.
- Uniform without-replacement 20,000 training latents per partition seed 0,1,2 for CPU fit;
  serialized full-covariance MAP posterior router assigns all training/evaluation latents.
  This subset is a computational configuration, not full-data EM. No PCA/dimension reduction.
- Convergence failure or empty region stops the run; no automatic retry/parameter alteration.
- Same predictor training: seeds 0,42,625, 50 epochs, batch128, lr5e-5, wd1e-3,
  FP32, history3, one-step targets. Global and encoder are reused unchanged.
- Long evaluation only: eval seeds0..4, 50 episodes, goal offset50, budget50, MPC routing.
  Existing Official paired starts and Global results are reused, including TwoRoom symlinks.
- Frozen pure weakest-pair Bures threshold is read from the existing policy, never refitted;
  ridge1e-8, weakest pair per partition seed, then mean over seeds. No Check1.

## Reproduction

```bash
METHODS=gmm NUM_CLUSTERS=4 GMM_MAX_ITER=1000 GPU_IDS=5,6,7 EVAL_WORKERS=2 GUARD_LIGHT_TRAINING=1 \
  bash experiments/control_matrix/scripts/run_lewm_partition_variant.sh
```

Partition fitting is CPU-only (4 BLAS threads per fit); the GPU set applies to
predictor training. Evaluation acquires idle GPUs via cooperative locks and
used-memory/utilization checks after training and Bures scoring finish.
Training also holds the same per-GPU cooperative lock and requires >=6144MiB
free before each predictor job; this is a starting guard, not a VRAM reservation
against unrelated processes. GPU_IDS and TRAIN_MIN_FREE_MIB are configurable.
No source/raw result is overwritten. Per-phase hard timeouts are recorded in the runner.

Outputs: bures_by_seed.csv, bures_pair_metrics.csv, long_raw.csv,
long_comparison.csv and experiment_manifest.json. Training and partition artifacts
live under each task's matrix_k4/{training,partitions}/gmm, long results under
matrix_k4_long/eval/gmm. Coordinator log/status/PID live in this directory's logs.

Source: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html

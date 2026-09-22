# LeWM K=4 K-means++ partition-method transfer

This is a supplementary partition-method transfer experiment at the original
development resolution K=4. It is not a new threshold-calibration run, and its
four cells are not added to the original Layer-2/Layer-3 validation totals.

Protocol: TwoRoom, PushT, Reacher and Cube; spherical K-means++ with 50 restarts;
partition seeds 0/1/2; predictor post-training seeds 0/42/625; unchanged
predictor-only 50-epoch post-training; batch 128, learning rate 5e-5, weight decay
1e-3, FP32. The encoder, datasets and frozen latent caches are reused by reference.
No Global, Official or Joint-Continue models are retrained.

Only long-horizon evaluation is requested for this extension: five evaluation
seeds 0/1/2/3/4, 50 episodes each, goal offset 50 and evaluation budget 50, using
the exact existing K=4 Official/Global/Spectral paired starts. Existing K=3 and
K=4 spectral results are preserved. K-means++ results are written to the added
`kmeanspp` subdirectories of `matrix_k4` and `matrix_k4_long`.

The weakest-pair Jacobian-Bures score reuses the original sufficient-statistics
and distance functions: ridge 1e-8, transition stride 1, mean weakest-pair score
over partition seeds 0/1/2. The existing frozen threshold is read from the policy
file, not estimated from the new outcomes; the pure Bures rule does not apply
the legacy Check-1 condition in that file. Scores are recorded before evaluation.

Run from the main repository, without a new branch or worktree:

```bash
METHODS=kmeanspp NUM_CLUSTERS=4 GPU_IDS=0,1,2,3,4 EVAL_WORKERS=2 \
  bash experiments/control_matrix/scripts/run_lewm_partition_variant.sh
```

The pipeline automatically runs preflight, multi-GPU partitioning/post-training,
fixed-threshold Bures scoring, resource-queued long-horizon evaluation, and final
paired result validation. Long-horizon jobs start only after all training is
complete and a GPU is idle for three polls: used memory <=2048 MiB and utilization
<=10%. One queue lock is held per GPU. Crashes are reported without automatic
retry. Training has a 72-hour timeout; scoring 24 hours; each evaluation job
(one partition/training seed and five evaluation seeds) 12 hours. GPU wait time
is excluded from the per-job evaluation timeout. Overrides are environment
variables; data and checkpoints remain configurable in the reused launcher.

Outputs under this directory:

- `preflight.json`: exact seeds, protocol, frozen policy and implementation hashes.
- `bures_pair_metrics.csv`, `bures_by_seed.csv`, `score_manifest.json`: fixed scores.
- `long_raw.csv`: all 180 new evaluation blocks with paired Global sources/hashes.
- `long_comparison.csv`, `experiment_manifest.json`: four task comparisons,
  point-estimate winner, frozen-rule predictions and accuracy. Near-zero differences
  are flagged, not automatically relabeled as Global.
- `logs/<RUN_ID>/`: coordinator PID, stage status and per-job logs.

No accuracy or completion is claimed until the final manifest is generated.

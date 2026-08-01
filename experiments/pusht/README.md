# PushT experiment record

This directory records the PushT reproduction of the LAP control matrix. The
matrix uses the same method families as the TwoRoom main experiment, except
that it does not include a manually designed partition:

- official pretrained baseline;
- Joint-Continue for 3 epochs;
- Global-FT for 50 epochs;
- Random-Voronoi;
- K-means++;
- spectral partitioning.

## Frozen-encoder cache

The completed PushT LeWM cache is stored at
`experiments/pusht/matrix/pusht_lewm_train_latent_cache.npz`; its encoder report
is the adjacent `.report.json` file. The measured run used FP32 inference and
the legacy transition batch size of 128.

| Quantity | Value |
|---|---:|
| Transition windows | 1,850,815 |
| Latent frames per window | 4 |
| Reconstructed frame slots | 7,403,260 |
| Unique timesteps | 2,315,671 |
| Encoded `(timestep, batch shape)` keys | 2,315,686 |
| Canonical partition-input latents | 2,315,671 |
| Cross-shape timesteps | 15 |
| Total cache-build time | 3,861.662 s (64.36 min) |
| Unique-frame encoding time | 3,600.669 s |
| NPZ save time | 242.735 s |
| Measured reduction relative to repeated window encoding | 3.1965x |

The final NPZ reconstructs the original overlapping window layout. Therefore,
the same timestep can appear in multiple output slots even though its encoding
was computed only once per valid encoding key. Repeated output slots are not
repeated encoder calls.

## Canonical-first deduplication contract

The cache audit shows that GPU batch shape can introduce small FP32 numerical
differences. These differences must not turn one observation into multiple
latent states. LAP therefore uses the following partition-input rule:

1. Flatten cached windows in their original order.
2. Stable-sort occurrences by timestep.
3. Retain the first occurrence of each timestep as its canonical latent.
4. Discard every later occurrence, including the 15 cases encoded under a
   different visual batch shape.

The first occurrence is representative because almost all later occurrences
were encoded with the same batch shape and are bitwise identical. For the 15
cross-shape cases, the observed difference is numerical noise rather than a
new environment state.

### PushT measurement

The audit reconstructed the original frame indices and the legacy visual batch
assignment, then compared every embedding for the 15 timesteps that occur in
both the 252-frame final batch and a 512-frame full batch.

| Check | Result |
|---|---:|
| Cross-shape timesteps tested | 15 |
| Bitwise-equal 252-vs-512 pairs | 0 / 15 |
| Pairs passing absolute tolerance `1e-7` | 0 / 15 |
| Pairs passing absolute tolerance `1e-6` | 0 / 15 |
| Pairs passing relative tolerance `1e-5`, absolute tolerance `1e-7` | 0 / 15 |
| Maximum absolute component difference | 0.0010030866 |
| Maximum L2 difference | 0.0040290930 |
| Maximum relative L2 difference | 0.0003500833 |
| Minimum cosine similarity | 0.9999999420 |
| Same-shape repeated embeddings that were bitwise equal | 15 / 15 per shape |

The differences are directionally tiny even though they are not bitwise equal:
the minimum cosine similarity is above 0.99999994. The completed cache does not
need to be rebuilt. Its windows preserve the legacy encoder output, while the
partition loader deterministically canonicalizes them to one latent per
timestep using the first occurrence.

The stopped partition run was caused by the old loader requiring all repeated
slots to be bitwise equal. The corrected loader no longer interprets
batch-shape noise as a second state or as a fatal cache inconsistency; it keeps
the stable first occurrence and records this policy in the partition manifest.

## Eight-GPU matrix command

The parallel controller uses one bounded worker per listed GPU, four CPU
threads per worker, disjoint output directories, and a dependency barrier
between partitioning, training, official-start generation, model evaluation,
and aggregation:

```bash
DATASET_NAME=pusht \
DATA_FILE=/path/to/pusht_expert_train.h5 \
CHECKPOINT=/path/to/lewm_object.ckpt \
EVAL_CONFIG=pusht \
EVAL_DATASET_NAME=pusht_expert_train \
CACHE_DIR=/path/to/stable_worldmodel_cache \
WORK_ROOT=experiments/pusht/matrix \
PYTHON=/path/to/lewm/python \
GPU_IDS=0,1,2,3,4,5,6,7 \
CPU_THREADS=4 \
bash experiments/control_matrix/scripts/run_lewm_matrix_parallel.sh
```

Each run records its resolved settings, Git commit, controller PID, and
per-task logs under `experiments/pusht/matrix/logs/parallel_<RUN_ID>/`.

## Current PushT control results

The short-horizon protocol uses `goal_offset=25` and `eval_budget=50`; the
long-horizon protocol uses `goal_offset=50` and the same `eval_budget=50`.
Every method is evaluated on the same five evaluation seeds and paired start
states. For trained methods, the reported uncertainty is the sample standard
deviation across the three fine-tuning seeds (0, 42, and 625). For partitioned
methods, partition-seed results are first averaged within each fine-tuning
seed. The official pretrained baseline has no fine-tuning-seed error bar.

| Method | Short horizon | Delta vs. baseline | Long horizon | Delta vs. baseline |
|---|---:|---:|---:|---:|
| Official baseline | 86.00% | -- | 37.20% | -- |
| Joint-Continue 3ep | 89.07 +/- 1.01% | +3.07 pp | 36.53 +/- 0.46% | -0.67 pp |
| Global-FT50 | **93.73 +/- 0.46%** | **+7.73 pp** | **40.40 +/- 0.40%** | **+3.20 pp** |
| Random-Voronoi K3-50 | 92.80 +/- 0.13% | +6.80 pp | 39.60 +/- 0.35% | +2.40 pp |
| K-means++ K3-50 | 93.47 +/- 0.13% | +7.47 pp | 39.42 +/- 0.73% | +2.22 pp |
| Spectral K3-50 | 93.47 +/- 0.53% | +7.47 pp | 38.58 +/- 1.15% | +1.38 pp |

The source summaries are `matrix/matrix_summary.csv` and
`matrix_long/matrix_summary.csv`.

The corresponding ggplot2 figures are generated by
`assets/control_metrics/plot_pusht_control_matrix.R`:

![PushT short-horizon control matrix](assets/control_metrics/pusht_short_horizon_main.png)

![PushT long-horizon control matrix](assets/control_metrics/pusht_long_horizon_main.png)

## Interpretation and current limitation

### What the results establish

1. Frozen-encoder predictor post-training is effective on PushT. Global-FT50
   improves the short-horizon success rate by 7.73 percentage points and the
   long-horizon success rate by 3.20 percentage points over the current
   official-checkpoint baseline.
2. The improvement is much smaller on the long-horizon protocol. Its baseline
   success rate is only 37.20%, compared with 86.00% on the short-horizon
   protocol, even though the evaluation budget remains 50 while the goal
   offset doubles from 25 to 50. This is consistent with the longer task
   exceeding the reliable planning range of the pretrained world model and
   planner: model error compounds over a longer rollout and PushT contact
   decisions become less recoverable.
3. Global-FT50 is the best tested PushT method at both horizons. None of the
   tested hard K=3 partitions improves over unpartitioned fine-tuning. The
   short-horizon gap between Global-FT50 and K-means++/Spectral is small
   (0.27 percentage points), whereas the long-horizon gaps are 0.80, 0.98,
   and 1.82 percentage points relative to Random-Voronoi, K-means++, and
   Spectral, respectively.
4. Changing the spectral router from once-per-MPC routing to step-wise routing
   does not materially change this conclusion. Step-wise Spectral obtains
   38.98 +/- 0.66%, versus 38.58 +/- 1.15% for once-per-MPC routing; the paired
   change is only +0.40 +/- 1.29 percentage points, and the imagined-step route
   switch rate is 4.68%.

### What the results do not yet prove

These results do not prove that PushT has no dynamics partition, nor do they
isolate an insufficient encoder as the cause. They support the narrower claim
that the current pretrained representation does not expose a sufficiently
useful **hard, state-only K=3 partition** for the tested LAP algorithms to beat
Global-FT50 under this planning protocol.

Several explanations remain compatible with the evidence:

- PushT may have continuous or overlapping contact regimes rather than a small
  number of well-separated latent-space regions.
- Relevant dynamics may depend on velocity, contact history, or other state
  information that is not cleanly recoverable from the current latent alone.
- The pretrained encoder may be insufficiently dynamics-aware to expose those
  regimes geometrically, even if they exist in the environment.
- Hard partitioning reduces the amount and diversity of data seen by each
  predictor; on PushT, this statistical cost may exceed the benefit of local
  specialization.
- A single predictor may already have enough capacity to model PushT's useful
  local variation, making Global-FT the better bias-data trade-off.

The spectral geometry is consistent with this caution: across the three
partition seeds, the eigengap after K=3 is only 0.00137--0.00166. This is weak
evidence for a privileged three-way decomposition, not evidence that every
possible partition is absent.

### Working conclusion for the paper

PushT should currently be reported as a limitation/negative case for hard
automatic partitioning, while still being positive evidence for efficient
frozen-encoder predictor post-training. The defensible conclusion is:

> When the frozen latent space contains a stable and useful regional geometry,
> LAP can exploit it; when that geometry is weak, as in the current PushT
> checkpoint, unpartitioned Global-FT can be preferable.

Distinguishing "encoder limitation" from "task does not admit a useful hard
partition" requires additional diagnostics rather than another interpretation
of the same success-rate table: latent probes for task state/contact variables,
held-out multi-step prediction by region, partition stability/eigenspectrum
analysis, and the same PushT experiment with a stronger pretrained encoder.

## Normalized eigengap and the LAP fallback

An absolute eigengap is not comparable across datasets because its scale
depends on the graph spectrum. For a candidate K-way partition, LAP therefore
uses the scale-free normalized eigengap

$$
g_K = \frac{\lambda_{K+1} - \lambda_K}{\lambda_{K+1} + \varepsilon},
$$

where the normalized-Laplacian eigenvalues are ordered as
$0 = \lambda_1 \leq \lambda_2 \leq \cdots$. For the currently tested K=3
configuration, the observed relation is:

| Dataset | Raw K=3 eigengap | Normalized K=3 eigengap | Spectral vs. Global-FT (long) | Current LAP decision |
|---|---:|---:|---:|---|
| TwoRoom | 0.000829--0.000831 | **57.36%--57.51%** | **+3.24 pp** | Retain Spectral K3-50 |
| PushT | 0.001375--0.001657 | **14.22%--16.20%** | **-1.82 pp** | Fall back to Global-FT (LAP) |

The raw gap would give the wrong cross-dataset ordering: PushT has the larger
absolute gap, while TwoRoom has the substantially stronger relative spectral
separation and is the dataset on which spectral regional fine-tuning improves
long-horizon control. This motivates the following dataset-level gate:

$$
\operatorname{LAP}(D) =
\begin{cases}
\text{Spectral K-way regional fine-tuning},
& g_K \geq \tau_g \text{ and the partition is stable}, \\
\text{Global-FT (LAP)},
& g_K < \tau_g \text{ or the partition is unstable}.
\end{cases}
$$

Here, "fall back" means that LAP keeps the encoder frozen and performs the
same efficient predictor post-training, but does not force a regional split.
The eigengap is an unsupervised gate computed before downstream control
evaluation; task success must not be used to select the branch.

The two current datasets establish the observed correlation but are not enough
to fix a universal threshold $\tau_g$. The final gate should combine the
normalized eigengap with cross-seed partition stability and a minimum cluster
mass, and its thresholds must be fixed on development datasets before final
test evaluation.

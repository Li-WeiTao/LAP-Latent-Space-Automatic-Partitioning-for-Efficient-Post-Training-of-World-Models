# LeWM K=4 Partition-Assumption Audit

## Scope

This audit examines why the frozen weakest-pair Jacobian-Bures rule fails on
two LeWM K=4 K-means++ configurations. It reuses existing latent caches,
partition assignments, predictor training logs, and long-horizon results. It
does not retrain predictors, rerun planning evaluation, or modify original
artifacts.

- Repository commit: `17414055d96f2378cdae09265fc6592a646bfff5`
- Tasks: TwoRoom, PushT, Reacher, Cube
- Partition methods: spectral, GMM, K-means++
- Region count: K=4
- Partition seeds: 0, 1, 2
- Frozen Bures threshold: `0.508947854338762`
- Corresponding q-space reference threshold: `0.254473927169381`

## Assumption 2 audit: regionwise first-order response approximation

For each task, partition method, and partition seed, regionwise affine action
responses were fitted on approximately 80% of episodes and evaluated on a
disjoint 20% episode fold. Action normalization was fitted on the training
fold only. The response target was next-latent minus current-latent.

The table reports the mean held-out MSE reduction of the regionwise affine
model relative to the Global affine model. Positive values favor the
regionwise affine approximation.

| Task | Spectral | GMM | K-means++ |
|---|---:|---:|---:|
| TwoRoom | 0.63% | 0.95% | **1.86%** |
| PushT | 2.23% | **2.42%** | 1.59% |
| Reacher | 0.59% | 0.94% | **1.24%** |
| Cube | 2.94% | **3.07%** | 1.94% |

K-means++ does not show a distinctive failure of the observable Assumption 2
proxy. In particular, its Reacher regions improve held-out affine prediction
more than the spectral and GMM regions. The K-means++ error therefore cannot
be explained simply by an inability to expose first-order dynamics
differences.

The residual uses observed responses and therefore includes irreducible
conditional variation. It is a comparative diagnostic, not a direct estimate
of the population conditional-mean approximation bound in Assumption 2.

## Exact affine-reference decomposition

The affine-reference identity was recomputed using the region probabilities,
response energies, pairwise Bures terms, fixed-action factors, and intercepts.
The maximum absolute reconstruction error was
`1.6653345369377348e-16`.

### Cube K-means++: pair aggregation explains the false Global decision

- weakest-pair q: `0.23328709075832565`
- q minus the reference threshold: `-0.021186836411055343`
- exact pair-aggregation remainder: `0.07602748326831937`
- q plus pair aggregation minus the reference threshold:
  `0.05484064685726403`

The exact positive pair-aggregation remainder alone reverses the sign of the
weakest-pair reference margin. Thus Cube is direct evidence that the small
pair-aggregation budget required by Assumption 4 is not satisfied for this
K-means++ configuration. One relatively weak region pair makes the minimum
overly conservative even though the complete Regional branch remains useful.

### Reacher K-means++: the break occurs after affine response geometry

- weakest-pair q: `0.3025505850130707`
- q minus the reference threshold: `0.048076657843689696`
- pair-aggregation remainder: `0.05214176675466844`
- intercept contribution: `0.013335101861429485`
- fixed-action/orthogonal-alignment contribution: `0.28943652091122324`
- complete normalized affine-reference gain: `0.6574639745403917`

The affine reference strongly favors Regional modeling. Nevertheless, the
weighted same-cache loss of the trained K-means++ Regional predictors is
approximately 5.70% higher than that of the trained Global predictor:

- Global loss: approximately `0.019778`
- Regional loss: approximately `0.020906`

The same approximately 5.8% disadvantage occurs for spectral and GMM
Regional predictors on Reacher. This is therefore not specific to the
K-means++ partition.

## Reacher diagnosis

The principal failure is most consistent with **Assumption 3**, specifically
the requirement that the actual neural candidate comparison remain close to
the affine reference comparison. The theoretical Global affine class uses a
shared response law, whereas the actual Global neural predictor observes the
full latent history and may represent state-dependent or piecewise local
dynamics internally without an explicit router. Local response differences
can therefore be real while providing little additional approximation power
to separately trained Regional neural predictors. This produces a large
actual-model discrepancy term.

**Assumption 4** is a likely accompanying failure. Each Reacher Regional
predictor uses approximately one quarter of the samples available to the
Global predictor. Separate estimation and optimization can therefore impose
a configuration-dependent specialization cost larger than the common
reference cost represented by the frozen threshold.

Training-curve evidence is consistent with a real implementation-level cost:

- all three Global runs selected epoch 50;
- 35 of 36 K-means++ Regional models selected epoch 50;
- from epoch 40 to 50, Global loss decreased by approximately 6.2%, whereas
  K-means++ Regional loss decreased by approximately 2.7% on average.

This does not isolate optimization error from statistical estimation cost,
but it does not support the simpler claim that only the Regional models were
under-trained.

The current evidence does **not** require an Assumption 5 failure to explain
Reacher: the realized one-step trained-model comparison already favors
Global before planning is applied. PushT is the cleaner candidate for a
dedicated prediction-to-planning audit because its Regional predictors have
lower same-cache prediction loss while GMM and K-means++ still favor Global
in long-horizon planning.

## Assumption-level conclusion

| Assumption | Reacher assessment | Evidence |
|---|---|---|
| Assumption 2: regionwise first-order response approximation | No distinctive violation observed | K-means++ improves held-out affine prediction by 1.24% relative to Global |
| Assumption 3: controlled actual-model discrepancy | **Primary likely violation** | Affine geometry strongly favors Regional, but the actual trained Global predictor has lower loss |
| Assumption 4: controlled pair aggregation and specialization cost | **Likely accompanying violation** | Regional splitting incurs a task/configuration-dependent realized cost; Cube directly violates the small pair-aggregation budget |
| Assumption 5: prediction-to-planning connection | Not needed to explain Reacher | The trained one-step comparison already favors Global |

Accordingly, the Reacher failure should be described as a breakdown of the
controlled affine-to-neural bridge and common specialization-cost budget, not
as evidence that K-means++ failed to expose any dynamics information.

## Files

- `affine_response_residual_by_region.csv`
- `affine_response_residual_by_seed.csv`
- `affine_response_residual_summary.csv`
- `affine_reference_pair_decomposition.csv`
- `affine_reference_decomposition_by_seed.csv`
- `affine_reference_decomposition_summary.csv`
- `prediction_to_planning_alignment.csv`
- `trained_predictor_same_cache_loss_by_run.csv`
- `trained_predictor_same_cache_loss_summary.csv`
- `assumption_diagnosis.json`
- `audit_manifest.json`

The trained-predictor loss files use the trainer's same-cache evaluation,
which is also used for checkpoint selection. These values diagnose realized
model and optimization cost but are not independent held-out generalization
estimates.

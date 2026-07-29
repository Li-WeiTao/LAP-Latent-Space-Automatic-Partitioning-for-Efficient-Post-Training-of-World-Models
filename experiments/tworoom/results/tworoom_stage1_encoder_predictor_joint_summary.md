## Stage 1 TwoRoom Summary: Encoder Drift and Predictor Rule Drift

This section summarizes the current Stage 1 TwoRoom evidence after correcting the predictor-side reference. The predictor diagnostic now uses a train-global reference instead of the earlier common-region reference.

### Predictor-Side Design

- Train reference: global train split reconstructed from the LeWM config (`train_split=0.9`, `seed=3072`).
- Test reports: all metrics are reported on held-out test transitions and natural test regions.
- Seeds: five sampling seeds (`0,1,2,3,4`) with fixed `split_seed=3072`.
- Per run: `train_max_samples=5000`, `test_max_samples=5000`, `jacobian_samples=512`, `iid_bootstrap_trials=20`.
- Caveat: this approximates the LeWM train/test split from config; it is not the exact saved training index set from the checkpoint.

### Predictor Rule Drift Across 5 Seeds

| test split | rule drift to train | excess vs test IID | z vs test IID | rollout h10 MSE | one-step MSE |
|---|---:|---:|---:|---:|---:|
| test_all | 0.423 +- 0.133 | 0.044 +- 0.121 | 0.389 +- 1.005 | 1.178 +- 0.049 | 0.086 +- 0.003 |
| doorway_corridor | 0.667 +- 0.123 | 0.287 +- 0.111 | 2.477 +- 1.099 | 0.935 +- 0.039 | 0.102 +- 0.003 |
| goal_other_side | 0.448 +- 0.157 | 0.069 +- 0.157 | 0.707 +- 1.326 | 1.204 +- 0.052 | 0.085 +- 0.002 |
| left_room | 0.768 +- 0.195 | 0.389 +- 0.193 | 3.343 +- 1.650 | 1.226 +- 0.059 | 0.086 +- 0.002 |
| near_wall | 0.945 +- 0.320 | 0.566 +- 0.351 | 4.922 +- 3.645 | 1.111 +- 0.064 | 0.073 +- 0.001 |
| right_room | 0.735 +- 0.094 | 0.356 +- 0.139 | 3.212 +- 1.715 | 1.270 +- 0.045 | 0.075 +- 0.002 |

Interpretation: `excess vs test IID` subtracts the test-IID bootstrap baseline from a region's rule drift. `z vs test IID` expresses that excess in units of the bootstrap standard deviation. Positive, large z means the region's predictor dynamics rule differs from train-global dynamics more than ordinary held-out test sampling noise.

### Joint Encoder/Predictor Stage 1 View

| split | encoder frame drift / IID | encoder frame residual / IID | predictor rule z | predictor excess | rollout h10 MSE |
|---|---:|---:|---:|---:|---:|
| common | 1.000 | 1.000 | 3.208 | 0.377 | 1.170 |
| doorway_corridor | 9.578 | 1.031 | 2.477 | 0.287 | 0.935 |
| goal_other_side | 10.460 | 1.428 | 0.707 | 0.069 | 1.204 |
| left_room | 10.516 | 1.473 | 3.343 | 0.389 | 1.226 |
| near_wall | 11.083 | 1.469 | 4.922 | 0.566 | 1.111 |
| right_room | 10.486 | 1.446 | 3.212 | 0.356 | 1.270 |

Stage 1 reading:

- Encoder side: natural TwoRoom regions show much larger state-aligned frame drift than IID. Doorway/corridor is the cleanest encoder-side case because its frame residual stays close to IID while frame drift is high. Right room, left room, near wall, and goal-other-side also drift strongly, but their residual ratios are higher, so they mix coordinate drift with representation distortion or state-proxy mismatch.
- Predictor side: with the corrected train-global reference, physical/dynamics rule inconsistency is strongest in `near_wall`, `right_room`, and `left_room`. These regions have large positive excess and z scores across five seeds, meaning their local predictor dynamics differ from train-global dynamics beyond ordinary test-IID variation.
- Doorway/corridor remains important for encoder gauge drift, but it is not currently the strongest predictor-rule-drift region. Its predictor-side z is positive but weaker than wall/room-side regions. This means the cleanest encoder gauge drift does not automatically imply the largest predictor dynamics mismatch.
- The current Stage 1 evidence therefore supports a more careful claim: natural regions can show both encoder-side state-aligned coordinate drift and predictor-side dynamics-rule drift, but the two effects are region-dependent and not one-to-one.
- For the project's main hypothesis, the most useful next test is to check whether regions with high predictor rule drift also show higher rollout/planning failure after controlling for ordinary test-IID drift, and then test whether gauge-aware predictor interventions reduce that excess.

Generated files:

- `results/tworoom_predictor_rule_train_test_5seed_summary.csv`
- `results/tworoom_stage1_encoder_predictor_joint_summary.csv`


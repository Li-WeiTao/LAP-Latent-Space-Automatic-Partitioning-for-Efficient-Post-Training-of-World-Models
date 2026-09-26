# Sub-JEPA finite-sample theory diagnostics

This directory estimates two quantities conditional on each already-fixed
candidate partition.

1. `epsilon_q_95` is the 95th percentile absolute deviation of the weakest-pair
   score `q` under a nonparametric bootstrap of episode blocks. The associated
   `epsilon_score_95` is on the paper's Bures-score scale and equals twice this
   quantity.
2. `heldout_affine_l2_upper95` is a one-sided, episode-cluster-robust upper
   limit for the response prediction residual of an affine model fitted on
   disjoint episodes. Region-level values are in
   `heldout_affine_error_by_region.csv`.

The affine residual is conservative for conditional-mean approximation under
the usual zero-mean regression-noise decomposition, because it also contains
conditional noise. It is not a complete certificate for `epsilon(c)`, and no
unjustified numerical pass/fail threshold is applied. The partition itself is
treated as fixed: the analysis evaluates score estimation and affine fitting,
not uncertainty from re-estimating the partition.

## Limitation

当前实验制品无法给出完整且可信的 \(\epsilon(c)\) 数值上界。这里报告的结果只覆盖当前数据能够识别的组成项，不能被解释为完整的理论条件证书。

## Aggregate results

Values below average point estimates across the three predeclared partition
seeds; uncertainty columns use the maximum estimate across those seeds.

| Task | K | mean q | max epsilon_q (95%) | held-out affine L2 upper (95%) | normalized upper |
|---|---:|---:|---:|---:|---:|
| Cube | 2 | 0.222733 | 0.005293 | 0.684391 | 3.118412 |
| Cube | 3 | 0.175666 | 0.003508 | 0.669172 | 2.216772 |
| Cube | 4 | 0.108410 | 0.004165 | 0.663863 | 1.977781 |
| PushT | 2 | 0.232283 | 0.002060 | 1.372143 | 5.533491 |
| PushT | 3 | 0.244789 | 0.002576 | 1.368048 | 4.472063 |
| PushT | 4 | 0.239362 | 0.003099 | 1.363385 | 3.891474 |
| Reacher | 2 | 0.126359 | 0.006199 | 3.307782 | 15.105992 |
| Reacher | 3 | 0.198439 | 0.009178 | 3.304242 | 10.788329 |
| Reacher | 4 | 0.236592 | 0.009375 | 3.303572 | 9.731191 |
| TwoRoom | 2 | 0.252905 | 0.004556 | 4.160047 | 4.428839 |
| TwoRoom | 3 | 0.281019 | 0.006890 | 4.053445 | 2.672140 |
| TwoRoom | 4 | 0.269667 | 0.003250 | 3.986997 | 2.159512 |

Using the frozen Bures threshold 0.508947854338762 (hence q threshold
0.254473927169381), the conservative comparison `absolute q margin > max
epsilon_q` resolves the threshold side in 11 of 12 configurations. The only
unresolved configuration is TwoRoom K=2. This statement isolates score
estimation uncertainty; it does not include calibration, planning, or the
other structural remainder terms in the paper.

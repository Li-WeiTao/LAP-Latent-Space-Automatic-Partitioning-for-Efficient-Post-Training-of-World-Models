# LAP branch-selection uncertainty and decision-risk diagnostics

This appendix-level sensitivity analysis supplements the descriptive 7-of-8 point-estimate agreement reported in the main text. It does **not** redefine the empirically better-performing branch, alter gate decisions, or constitute a predeclared or confirmatory analysis.

## Interpretation

- The **7/8** headline counts pairs where Auto-LAP's selected dynamics-modeling branch has the higher **observed mean** success rate at long horizon.
- **Paired bootstrap CIs** quantify evaluation uncertainty; a CI that crosses zero does not overturn the observed-mean ordering.
- **p_harm** and **EOL** are supplementary decision-risk diagnostics for the selected-vs-rejected dynamics-modeling contrast.
- **margin = 2.0 pp** is a **post-hoc practical-sensitivity margin** (one episode success ≈ 2 pp within a 50-start evaluation block).

## Long-horizon diagnostics

| Model | Task | Pilot | Selected | Rejected | Δ mean (pp) | 95% CI | Point favors selected | p_harm | EOL (pp) | Practical EOL (pp) | Classification |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---|
| lewm | cube | non-pilot | global | spectral | -1.54 | [-4.84, 1.47] | false | 0.3682 | 1.6657 | 0.4332 | statistically_unresolved |
| lewm | pusht | pilot | global | spectral | 1.83 | [-0.80, 3.87] | true | 0.0001 | 0.0553 | 0.0000 | practically_noninferior |
| lewm | reacher | non-pilot | global | spectral | 0.75 | [-4.09, 4.67] | true | 0.1093 | 0.5847 | 0.1588 | statistically_unresolved |
| lewm | tworoom | pilot | spectral | global | 3.46 | [-0.40, 7.07] | true | 0.0024 | 0.0285 | 0.0013 | practically_noninferior |
| subjepa | cube | non-pilot | global | spectral | 1.08 | [-2.13, 4.22] | true | 0.0302 | 0.2440 | 0.0195 | statistically_unresolved |
| subjepa | pusht | non-pilot | global | spectral | 2.30 | [0.18, 4.84] | true | 0.0000 | 0.0056 | 0.0000 | practically_noninferior |
| subjepa | reacher | non-pilot | global | spectral | 1.74 | [-1.64, 5.60] | true | 0.0162 | 0.1377 | 0.0131 | practically_noninferior |
| subjepa | tworoom | non-pilot | spectral | global | 0.13 | [-2.40, 2.80] | true | 0.0416 | 0.4222 | 0.0281 | statistically_unresolved |

## Aggregates

- Overall point-estimate agreement: 7/8
- Pilot pairs: 2/2
- Non-pilot pairs: 5/6
- Non-pilot mean EOL: 0.5100 pp
- Non-pilot mean practical EOL: 0.1088 pp
- Max p_harm: 0.3682 (lewm/cube)
- Max EOL: 1.6657 pp (lewm/cube)

## LaTeX appendix snippet

```latex
Because the empirically better-performing branch is defined by the higher observed mean, the 7-of-8 result is a point-estimate comparison. We supplement it with paired-bootstrap decision-risk diagnostics. Using a post-hoc practical sensitivity margin of 2 percentage points, we report the probability that the selected branch is materially worse and its expected opportunity loss. These diagnostics quantify uncertainty but do not alter the gate or the reported branch decisions.

\caption{Paired-bootstrap uncertainty and decision-risk diagnostics for LAP's long-horizon dynamics-modeling choices.}
```

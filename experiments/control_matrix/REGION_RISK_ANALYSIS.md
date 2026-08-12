# Held-out Region-Risk Analysis

The full public name is **Held-out Region-Conditional Prediction-Risk
Analysis**. This is a held-out mechanistic analysis, not the main
planning-performance experiment.

Evaluation episodes are not used for LAP partition fitting and are not used for
predictor post-training. They are held out at those two stages only; the
analysis does not guarantee that these episodes were absent from base
world-model pretraining.

`formal` is an internal audit/provenance term and is not the public experiment
name. Compatibility identifiers such as `formal_region_risk_pipeline.py`,
`--formal`, `formal_region_risk/`, and the `formal` field in `audit.json` remain
unchanged.

## Staged evaluation

Run the stages in this order:

```bash
python experiments/control_matrix/formal_region_risk_pipeline.py \
  --task tworoom \
  --work-root /path/to/tworoom_region_risk \
  --data-file /path/to/tworoom.h5 \
  --pretrained-model /path/to/lewm_object.ckpt \
  --phase evaluate-rollout \
  --resume

python experiments/control_matrix/formal_region_risk_pipeline.py \
  --task tworoom \
  --work-root /path/to/tworoom_region_risk \
  --data-file /path/to/tworoom.h5 \
  --pretrained-model /path/to/lewm_object.ckpt \
  --phase evaluate-bootstrap \
  --bootstrap-chunk-size 1000 \
  --bootstrap-workers 8 \
  --resume

python experiments/control_matrix/formal_region_risk_pipeline.py \
  --task tworoom \
  --work-root /path/to/tworoom_region_risk \
  --data-file /path/to/tworoom.h5 \
  --pretrained-model /path/to/lewm_object.ckpt \
  --phase evaluate-finalize \
  --resume
```

`evaluate-rollout` loads predictors and uses a GPU. It atomically writes one
raw NPZ per train seed and horizon-valid anchor support. The H10-support raw
file contains H1, H5, and H10 losses from one H10 rollout, so the common-H10
analysis never invokes the predictors a second time.

`evaluate-bootstrap` reads only `evaluation/raw/`, does not load predictors,
and does not require a GPU. It pre-aggregates loss sums and anchor counts by
train seed, episode, and metric, then writes deterministic chunks under
`evaluation/bootstrap_chunks/`. `--bootstrap-workers` controls CPU workers.

`evaluate-finalize` reads the raw files and bootstrap chunks and writes the
existing CSV, NPZ, PDF, audit, and manifest outputs. It performs neither
rollout nor bootstrap.

The compatibility entry point runs all three stages in order:

```bash
python experiments/control_matrix/formal_region_risk_pipeline.py \
  --task tworoom \
  --work-root /path/to/tworoom_region_risk \
  --data-file /path/to/tworoom.h5 \
  --pretrained-model /path/to/lewm_object.ckpt \
  --phase evaluate \
  --resume
```

Resume is accepted only when the recorded task, train seed, requested
horizons, anchors, checkpoint hashes, cache hash, partition hash, history size,
frameskip, and route index match exactly. A mismatch is an error and is never
silently overwritten.

The completed PushT artifacts under
`experiments/control_matrix/assets/formal_region_risk/pusht_formal_v1/` are
immutable historical outputs. They do not need to be regenerated and these
staged interfaces do not rewrite them.

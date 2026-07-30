# TwoRoom executable scripts

The supported publication entrypoint is:

```bash
export LAP_TWOROOM_DATA=/absolute/path/to/tworoom.h5
export LAP_LEWM_CHECKPOINT=/absolute/path/to/lewm_object.ckpt
export GPU=0
python experiments/tworoom/reproduce.py check --profile main
python experiments/tworoom/reproduce.py run --profile main
```

`run_tworoom_main_matrix.sh` executes the seven-method comparison used by the
TwoRoom main result: Official baseline, Joint-Continue 3ep, Global-FT50,
Random-Voronoi K3-50, K-means++ K3-50, Spectral K3-50, and human rooms3-50.
Training seeds are `0,42,625`, automatic partition seeds are `0,1,2`, and all
methods reuse the same five official evaluation-start files.

The machine-readable registry is
[`../reproduction_manifest.json`](../reproduction_manifest.json). It is the
authority for whether a command is canonical, analysis, ablation, or
validation. Commands not referenced by that registry are implementation
helpers or historical launchers and must not be cited as independent
experiments.

The historical one-off queue, `nohup`, recovery, and partial-rerun launchers
are preserved under `../legacy/orchestration/` for provenance only. They are
not supported reproduction entrypoints and are never called by the canonical
runner.

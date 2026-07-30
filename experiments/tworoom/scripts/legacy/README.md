# Historical orchestration launchers

These files are preserved only to document how completed server jobs were
queued, detached, resumed, or repaired during development. They are not
publication reproduction entrypoints and are never called by
`experiments/tworoom/reproduce.py`.

The launchers may contain historical GPU assumptions, polling loops, PID-file
logic, or references to partially completed output directories. Do not run
them for a clean reproduction. Equivalent supported operations are exposed by:

```bash
python experiments/tworoom/reproduce.py list --profile main
python experiments/tworoom/reproduce.py check --profile main
python experiments/tworoom/reproduce.py run --profile main
```

Their location in Git preserves provenance without presenting them as current
experiment scripts.

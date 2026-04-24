# CSMF Incremental Experiment Tree -- EXP-SCAFFOLD v2.2

Each of the 11 plan stages is a runnable Python package under `experiments/`.
Shared model code lives under `models/`; shared utilities under `common/`.
**Reproducibility:** every run is parameterised by `--seed` and its output is scoped
by `(seed, cfg_hash)` so reruns never collide.

## Install
```
pip install -r requirements-project.txt   # full CSMF ML stack
pip install -r requirements.txt           # dashboard + scaffolder only
```

## Quick start
```
python -m CSMF2.experiments.step_1_1.run --seed 0
python scripts/experiment_dashboard.py --root CSMF2/
```

Drop the `EXP-DASH v1.0` `experiment_dashboard.py` into `scripts/`.

**CONVENTION:** NLL = LOSS (lower = better).

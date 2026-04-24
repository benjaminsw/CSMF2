# Step 1.1 -- Single conditional expert (no mixture)

**Workpackage:** WP0

## Run

```
python -m CSMF2.experiments.step_1_1.run --seed 0
python -m CSMF2.experiments.step_1_1.run --seed 1
python -m CSMF2.experiments.step_1_1.run --seed 2
```

## Output layout

```
results/seed<N>_cfg<HASH>/   one sub-dir per (seed, cfg_hash)
checkpoints/                  per-run checkpoints
plots/                        per-run diagnostic PNGs
logs/                         stdout/stderr tee logs
configs/                      snapshot of the config used
status.json                   step progress (shared, atomic)
metrics.csv                   append-only per-epoch metrics
```

## Exit criterion (go / no-go)

All WP0 tests pass; no NaN; conditioning active; >= 2 experts competent.

## Move-forward gate (applies to every step)

- no NaNs
- no invertibility failure
- gate not collapsed
- Neff monitored
- residual improves when expected
- NLL not materially worse unless justified
- stable across 3 seeds

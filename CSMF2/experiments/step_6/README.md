# Step 6 -- Final sweep (lambda_cons, T, tau)

**Workpackage:** WP5

## Run

```
python -m CSMF2.experiments.step_6.run --seed 0
python -m CSMF2.experiments.step_6.run --seed 1
python -m CSMF2.experiments.step_6.run --seed 2
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

Final output pack: curves, Pareto, PIT/ES, sample grids, gate histograms.

## Move-forward gate (applies to every step)

- no NaNs
- no invertibility failure
- gate not collapsed
- Neff monitored
- residual improves when expected
- NLL not materially worse unless justified
- stable across 3 seeds

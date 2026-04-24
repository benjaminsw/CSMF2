# Step 5.1 -- Port to optical SR (DIV2K / BSD68)

**Workpackage:** WP4

## Run

```
python -m CSMF2.experiments.step_5_1.run --seed 0
python -m CSMF2.experiments.step_5_1.run --seed 1
python -m CSMF2.experiments.step_5_1.run --seed 2
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

Residual/geometry gain without NLL collapse; visible recon improvement.

## Move-forward gate (applies to every step)

- no NaNs
- no invertibility failure
- gate not collapsed
- Neff monitored
- residual improves when expected
- NLL not materially worse unless justified
- stable across 3 seeds

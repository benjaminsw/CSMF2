# Step 3.2 -- Stage B full hybrid (experts frozen)

**Workpackage:** WP2

## Run

```
python -m CSMF2.experiments.step_3_2.run --seed 0
python -m CSMF2.experiments.step_3_2.run --seed 1
python -m CSMF2.experiments.step_3_2.run --seed 2
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

Hybrid beats pure NLL on residual/geometry without hurting NLL (M2).

## Move-forward gate (applies to every step)

- no NaNs
- no invertibility failure
- gate not collapsed
- Neff monitored
- residual improves when expected
- NLL not materially worse unless justified
- stable across 3 seeds

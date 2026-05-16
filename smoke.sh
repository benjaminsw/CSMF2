#!/bin/bash
# NOTE: no set -e -- smoke runs at 1 epoch ALWAYS fail v2 gates,
# but exit code 2 means "gates failed", not "code broken".
# We accept exit 0 and 2, halt only on exit 1 (crash).

run_one() {
  local cmd="$@"
  echo "+++ $cmd"
  $cmd
  local rc=$?
  if [ $rc -eq 1 ]; then
    echo "!!! CRASHED (exit 1) -- halting smoke"
    exit 1
  fi
  echo "--- exit $rc ($([ $rc -eq 0 ] && echo 'gates passed' || echo 'gates failed but training ok'))"
}

for e in nice realnvp glow; do
  echo "=== smoke: $e (v2) ==="
  run_one python -m CSMF2.experiments.step_1_1.run \
    --expert $e --scale 2 --noise-sigma 0.0 --seed 0 --epochs 1
done

echo "=== smoke: nsf (legacy v1 cond) ==="
run_one python -m CSMF2.experiments.step_1_1.run \
  --expert nsf --scale 2 --noise-sigma 0.0 --seed 0 --epochs 1 \
  --no-use-v2-conditioner \
  --cond-width 64 --h-dim 128 --film-depth 1 --film-hidden 64 --no-film-use-gelu

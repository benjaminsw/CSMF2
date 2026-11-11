# SEQREF-MRI-DIAGJSHF2PROBE v0.1 -- F2 sandbox probe: DIAG-I3 analysis, plotting and output bytes at the real shape (SYNTHETIC values)
# LIFETIME: DIAGNOSTIC
# =============================================================================
# Plan SEQREF-MRI-DIAGJSHF2 v0.1, reviewer edit 1 (217 J17): measure/bound
# plotting time and output storage before the main-run limit is registered.
# SYNTHETIC numbers only (random volume matrices, random 96x96 images); no
# real data. Real shape from manifest d3e8176e...: 3 seeds x 97 volumes,
# H-D / H-F thirds 33/32/32, 4 retained slices. Runs the G start() (matplotlib
# cache inside OUT, guard, budget, log) and then exactly the I3 calls that G
# makes (volume_table, analyse, reliability, figures, write_outputs).
# Prints one JSON line: t_analysis_s, t_plot_s, bytes per file and in total.
# Usage: python -B diag_jsh_f2_probe.py EMPTY_OUT_DIR
# Changelog (v0.1, 2026-11-11): first probe.
# =============================================================================
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))


def main(out: Path) -> int:
    import diag_jsh_adapter as A
    import diag_jsh_core as K
    import diag_jsh_g_run as G
    import diag_jsh_f2_resources as F2R
    budget, pre = G.start(out, A, K, 0, dict(F2R.RESOURCES))
    import diag_jsh_i3 as I3
    rng = np.random.default_rng(1)
    V = 97
    vols = [f"file{1000000 + i}.h5" for i in range(V)]
    M = {m: {a: rng.standard_normal((3, V)) * 0.1 + (30.0 if m == "psnr"
                                                    else 1.0)
             for a in I3.ARMS} for m in I3.ALL}
    for a in I3.ARMS:                      # positive cells, ratio > 0
        for m in I3.ALL:
            M[m][a] = np.abs(M[m][a]) + 0.01
    hd = {v: float(rng.uniform(20, 40)) for v in vols}
    hf = {v: float(rng.uniform(0.0, 0.1)) for v in vols}
    strat = {"H-D": {"values": hd, "thirds": K.thirds(hd, descending=False)},
             "H-F": {"values": hf, "thirds": K.thirds(hf, descending=True)}}
    ident = {(s, a): {"epoch": 10, "model_state_sha256": "0" * 64}
             for s in K.SEEDS for a in I3.ARMS}
    nsl = {v: 36 for v in vols}
    ret = [{"volume_id": vols[i], "slice_index": 18,
            "x_true": rng.standard_normal((96, 96)) +
            1j * rng.standard_normal((96, 96)),
            "lam0": rng.standard_normal((96, 96)) +
            1j * rng.standard_normal((96, 96)),
            "lam2": rng.standard_normal((96, 96)) +
            1j * rng.standard_normal((96, 96))} for i in (3, 30, 60, 90)]
    t0 = time.monotonic()
    res = I3.analyse(vols, M, strat)
    t_an = time.monotonic() - t0
    table = I3.volume_table(vols, M, ident, nsl, strat)
    rel = I3.reliability({}, 0.0, 0.0, res["D-JAC"])
    t0 = time.monotonic()
    figs = I3.figures(res, M, vols, strat, ret)
    t_plot = time.monotonic() - t0
    I3.write_outputs(budget, out, res, table, figs, rel)
    sizes = {str(p.relative_to(out)): p.stat().st_size
             for p in sorted(out.rglob("*")) if p.is_file()}
    print(json.dumps({"t_analysis_s": t_an, "t_plot_s": t_plot,
                      "pre_guard_files": pre, "bytes": sizes,
                      "total_bytes": sum(sizes.values()),
                      "budget_used": budget.used}))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))

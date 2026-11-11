# SEQREF-MRI-DIAGJSHG v0.1 -- DIAG-JSH stage G: single real-data main run (frozen mini-spec v1.0 §1-§8)
# LIFETIME: KEEP
# =============================================================================
# Plan SEQREF-MRI-DIAGJSHF2 v0.1 (F2b), reviewer edits 1-5 (217 J17).
# One run, bound to the F1 manifest (d3e8176e...) by hash and to the F2
# resource record (diag_jsh_f2_resources.py) by code hash:
#   1. binding through the pinned V05LT code (F1 bind_production, reused --
#      not the F1 preparation run); every binding field must equal the
#      manifest (pin, launch chain, launch inputs, completion evidence,
#      backend, bound inputs, ZF record, core/adapter/F1 code hashes);
#   2. selected checkpoints re-derived by the pinned load_run and compared
#      field by field with the manifest; population and case order equal;
#   3. each of the six checkpoints decoded ONCE (adapter replay); the
#      stored rows' NLL identity (§3) and the §1.3 reproduction check
#      (exact identities, frozen limits) per checkpoint; regenerated masks
#      equal the manifest; spectra (§4) per slice from the same decode;
#   4. stratifiers recomputed and required to equal the manifest's thirds,
#      boundaries and hashes; retained slices equal the manifest (the
#      recorded membership and slices are used, never re-chosen);
#   5. DIAG-I3 v0.3 (unchanged): volume matrices, analysis (§2-§5), table,
#      figures, write-once outputs with sha256 sidecars; then a run record
#      g_run.json (+ sidecar) with identities, limits, timings, output hashes.
# Limits: Python deadline = start + min(T, MAX_WALL_S - 120 s), T from the
#   F2 resource record (never extended); write budget and log allowance
#   from the same record; the reserve is checked before every write.
# Writes: matplotlib is initialised BEFORE the write guard with
#   MPLCONFIGDIR=OUT/mplconfig and TMPDIR=OUT (its font cache is the only
#   pre-guard write, inside OUT, after a free-space check); the bytes are then
#   charged to the budget. From the guard on, every write is registered and
#   budgeted (strict guard). At the end the bytes under OUT must equal the
#   budgeted bytes (DIAG_UNBUDGETED_BYTES otherwise; catches native writes
#   inside OUT, not outside).
# Exits: 0 PASS; 1 typed STOP; 2 usage / unexpected (logged with traceback);
#   77 write guard; 78 log allowance. Partial outputs are kept (write-once).
# No fallback, placeholder, mock or dummy value in this file.
# Changelog (v0.1, 2026-11-11): first G runner; manifest/resource binding,
#   single decode per checkpoint, recorded membership, budgeted plotting
#   cache, final byte reconciliation, run record.
# =============================================================================
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

__abbr__ = "SEQREF-MRI-DIAGJSHG"
__version__ = "0.1"
DIAG_ROOT = Path("/home/benjamin/CSMFII/v06lt_evidence/DIAG_JSH")
V05LT = Path("/home/benjamin/CSMFII")
CODE_FILES = ("diag_jsh_core.py", "diag_jsh_adapter.py",
              "diag_jsh_f1_prepare.py", "diag_jsh_i3.py", "diag_jsh_g_run.py",
              "diag_jsh_f2_resources.py")
WALL_MARGIN_S = 120             # Python deadline before MAX_WALL_S (as F1)
MIN_WALL_S = 300
MPL_DIR = "mplconfig"
logger = logging.getLogger("seqref_diag.jsh_g")


def _args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--chain", required=True)
    p.add_argument("--chain-sha256", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reserve-bytes", type=int, required=True)
    p.add_argument("--max-wall-s", type=float, required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--manifest-sha256", required=True)
    for f in CODE_FILES:
        p.add_argument(f"--sha-{f.split('.')[0]}", required=True)
    return p.parse_args(argv)


def start(out: Path, A, K, reserve: int, R: dict):
    """Matplotlib cache (inside OUT, pre-guard) -> guard -> budget -> log."""
    budget_bytes = R["g_budget_bytes"]
    free = shutil.disk_usage(out).free
    K.require(free >= reserve + budget_bytes, "DIAG_DISK",
              f"start: free {free} < reserve {reserve} + budget "
              f"{budget_bytes}")
    os.environ["MPLCONFIGDIR"] = str(out / MPL_DIR)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["TMPDIR"] = str(out)
    os.mkdir(out / MPL_DIR)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot  # noqa: F401 -- initialise before the guard
    from matplotlib import font_manager
    font_manager.fontManager.findfont("DejaVu Sans")
    pre = [(str(p.relative_to(out)), p.stat().st_size)
           for p in sorted(out.rglob("*")) if p.is_file()]
    A.install_write_guard(out, git_repo=V05LT)
    budget = A.WriteBudget(out, reserve, budget_bytes)
    budget.take(sum(n for _, n in pre), "matplotlib cache (pre-guard, OUT)")
    h = A.BudgetLogHandler(budget, out / "g.log", R["log_allowance_bytes"])
    h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[h], force=True)
    logger.info("%s v%s start; pre-guard matplotlib files %s", __abbr__,
                __version__, pre)
    return budget, pre


def load_manifest(path: str, want: str, A, K) -> dict:
    """Manifest bytes = registered sha256 = its sidecar; strict JSON."""
    p = Path(path)
    K.require(p.is_file() and not p.is_symlink(), "DIAG_MANIFEST",
              f"{p} missing or a symlink")
    body = p.read_bytes()
    sha = A.sha256_bytes(body)
    K.require(sha == want, "DIAG_MANIFEST", f"manifest sha256 {sha} != {want}")
    side = Path(str(p) + ".sha256")
    K.require(side.is_file() and side.read_text().split()[:1] == [sha],
              "DIAG_MANIFEST", "sidecar missing or different")

    def no_const(x):
        K.fail("DIAG_MANIFEST", f"non-finite constant {x}")
    man = json.loads(body, parse_constant=no_const)
    K.require(man.get("schema") == "SEQREF-MRI-DIAG-JSH-MAN/0.2" and
              str(man.get("spec", "")).endswith(
                  "8665d790f4ab96d1cb526c442826da06b81a672c387a170a3c08adc2f"
                  "4018336"), "DIAG_MANIFEST", "schema or spec")
    return man


def check_resources(R: dict, man_sha: str, a, K) -> dict:
    """F2 resource record bound to this manifest; T never extended."""
    K.require(R["manifest_sha256"] == man_sha, "DIAG_RESOURCES",
              "resource record is bound to another manifest")
    T, C = R["T_s"], R["max_wall_s"]
    K.require(isinstance(T, float) and math.isfinite(T) and T > 0 and
              a.max_wall_s == C and C - WALL_MARGIN_S >= T,
              "DIAG_RESOURCES", f"T {T}, registered cap {C}, "
              f"given {a.max_wall_s}")
    return {"T_s": T, "max_wall_s": C}


def same(K, what: str, got, want) -> None:
    K.require(got == want, "DIAG_MANIFEST_MISMATCH",
              f"{what}: run {got!r} != manifest {want!r}")


def run(ctx: dict, man: dict, man_sha: str, R: dict, out: Path, lim: dict,
        code: dict, A, K, I3, budget, pre: list) -> str:
    from seqref_mri.src import free_flow_runtime as ffr
    t_all = time.monotonic()
    hard = lim["deadline"]
    b = ctx["binding"]
    for k in ("pin", "launch_chain", "launch_inputs_sha256",
              "completion_evidence_sha256", "backend", "inputs"):
        same(K, k, b[k], man[k])
    same(K, "zf_record_sha256", ctx["zf_record_sha256"],
         man["zf_record_sha256"])
    for f, h in man["code_sha256"].items():
        same(K, f"code {f}", code.get(f), h)
    logger.info("STAGE G1 binding equals the manifest")

    data = ctx["data"]
    vv, elig, p4, s_ref = (data["vv"], data["eligibility"], data["p4"],
                           data["s_ref"])
    sels = {k: ctx["load_selected"](k) for k in A.SEED_ARMS}
    for k, s in sels.items():
        rid = f"{k[0]}_{k[1]}"
        for x in ("epoch", "model_state_sha256", "weights", "weights_sha256",
                  "rows_file", "rows_sha256", "selection_evidence"):
            same(K, f"{rid}.{x}", s[x], man["checkpoints"][rid][x])
    docs = {k: s["rows"] for k, s in sels.items()}
    entries = A.check_population(docs, ctx["zf_rows"])
    n = len(entries)
    K.require(n == len(vv), "DIAG_POPULATION_MISMATCH",
              f"{n} rows vs {len(vv)} view slices")
    pop = man["population"]
    same(K, "slices", n, pop["slices"])
    same(K, "case_order_sha256", A.canonical_sha(entries),
         pop["case_order_sha256"])
    same(K, "slice_ids_sha256", docs[A.SEED_ARMS[0]]["slice_ids_sha256"],
         pop["slice_ids_sha256"])
    retained_keys = {(r["volume_id"], r["file"], r["slice_index"])
                     for r in man["retained_slices"]}
    A.check_deadline(hard, "inputs")
    logger.info("STAGE G2 checkpoints and population equal the manifest "
                "(%d slices)", n)

    per_slice, repro, hfs, imgs = {}, {}, [], {}
    t_loads, t_reps, disk = [], [], []
    max_decomp = max_spec = 0.0
    for k in A.SEED_ARMS:
        rid = f"{k[0]}_{k[1]}"
        stored = {(r["volume_id"], r["file"], r["slice_index"]): r
                  for r in docs[k]["rows"]}
        vals = {}

        def on_slice(key, row, spec, xh, xt, k=k, stored=stored, vals=vals):
            nonlocal max_spec
            vals[key] = I3.slice_values(stored[key], spec)
            for c, f in (("sc_cells", "sc_full"), ("sm_cells", "sm_full")):
                full = spec[f]
                r = abs(float(spec[c].sum()) - full) / full if full else 0.0
                max_spec = max(max_spec, r)
            if k[0] == 2026 and key in retained_keys:
                d = imgs.setdefault(key, {"volume_id": key[0],
                                          "slice_index": key[2]})
                d[k[1]] = xh
                if "x_true" in d:
                    K.require(bool((d["x_true"] == xt).all()),
                              "DIAG_REFERENCE_MISMATCH", f"{key} reference")
                d["x_true"] = xt
        t0 = time.monotonic()
        model, _vc = A.load_model(sels[k])
        t_loads.append(time.monotonic() - t0)
        t0 = time.monotonic()
        rep = A.replay(model, vv, elig, p4, s_ref, on_slice=on_slice,
                       deadline=hard)
        t_reps.append(time.monotonic() - t0)
        del model
        repro[rid] = A.reproduce(docs[k], rep, n)     # also §3 identity
        same(K, f"{rid} regenerated masks",
             A.canonical_sha(sorted(rep["mask_regen"].items())),
             pop["masks_regenerated_sha256"])
        K.require(list(vals) == list(entries), "DIAG_POPULATION_MISMATCH",
                  f"{rid}: decoded case order")
        per_slice[k] = vals
        hfs.append(rep["hf"])
        max_decomp = max(max_decomp, max(r["nll_decomp_abs_dev"]
                                         for r in docs[k]["rows"]))
        disk.append(budget.check(0, f"after {rid}"))
        logger.info("STAGE G3 %s decoded once; reproduction %s", rid,
                    repro[rid])
    _ = ffr.FLOW_DIM_REAL                             # pinned d_free in use
    hf = A.check_reference_equal(hfs)

    strat = A.stratifiers(entries, ctx["zf_rows"], hf)
    for s in ("H-D", "H-F"):
        ms = man["subgroups"][s]
        same(K, f"{s} thirds", strat[s]["thirds"], ms["thirds"])
        same(K, f"{s} boundaries", [[strat[s]["values"][g[0]],
                                     strat[s]["values"][g[-1]]]
                                    for g in strat[s]["thirds"]],
             ms["boundaries"])
        same(K, f"{s} sha256", strat[s]["sha256"], ms["sha256"])
    retained = A.retained_slices(entries, elig)
    same(K, "retained slices", retained, man["retained_slices"])
    ret_imgs = []
    for r in man["retained_slices"]:
        key = (r["volume_id"], r["file"], r["slice_index"])
        K.require(key in imgs and set(imgs[key]) >= {"lam0", "lam2", "x_true"},
                  "DIAG_RETAINED", f"{key}: images incomplete")
        ret_imgs.append(imgs[key])
    A.check_deadline(hard, "analysis")
    logger.info("STAGE G4 recorded subgroups and retained slices reproduced")

    t0 = time.monotonic()
    vols, M = I3.volume_matrices(
        {(s, a): per_slice[(s, a)] for s, a in A.SEED_ARMS}, entries)
    res = I3.analyse(vols, M, strat)
    t_boot = time.monotonic() - t0
    A.check_deadline(hard, "outputs")
    ident = {k: {"epoch": sels[k]["epoch"],
                 "model_state_sha256": sels[k]["model_state_sha256"]}
             for k in A.SEED_ARMS}
    nsl = {v: sum(1 for e in entries if e[0] == v) for v in vols}
    table = I3.volume_table(vols, M, ident, nsl, strat)
    rel = I3.reliability(repro, max_decomp, max_spec, res["D-JAC"])
    t0 = time.monotonic()
    figs = I3.figures(res, M, vols, strat, ret_imgs)
    t_plot = time.monotonic() - t0
    A.check_deadline(hard, "write")
    shas = I3.write_outputs(budget, out, res, table, figs, rel)
    logger.info("STAGE G5 outputs written: %s", sorted(shas))

    rec = {"schema": "SEQREF-MRI-DIAG-JSH-GRUN/0.1",
           "g": {"abbr": __abbr__, "version": __version__,
                 "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime())},
           "manifest_sha256": man_sha, "resources": R, "code_sha256": code,
           "limits": {k: lim[k] for k in ("reserve", "T_s", "max_wall_s")},
           "timings_s": {"load": t_loads, "replay": t_reps,
                         "analysis": t_boot, "plot": t_plot,
                         "total_before_record": time.monotonic() - t_all},
           "disk_checks": disk, "pre_guard_files": pre,
           "outputs_sha256": shas}
    body = (json.dumps(rec, sort_keys=True, indent=1, allow_nan=False) +
            "\n").encode()
    sha = A.sha256_bytes(body)
    side = f"{sha}  g_run.json\n".encode()
    budget.check(len(body) + len(side), "before run record")
    budget.write_new(out / "g_run.json", body, "run record")
    budget.write_new(out / "g_run.json.sha256", side, "run record sidecar")
    logger.info("STAGE G6 run record written; reconciling bytes")
    for hd in logging.getLogger().handlers:
        hd.flush()
    on_disk = A.tree_bytes(out)
    K.require(on_disk == budget.used, "DIAG_UNBUDGETED_BYTES",
              f"{on_disk} bytes under OUT != budgeted {budget.used}")
    return sha


def main(argv=None) -> int:
    a = _args(argv)
    code_dir = Path(__file__).resolve().parent
    if code_dir != DIAG_ROOT / "code":
        print("STOP usage: run from DIAG_JSH/code", file=sys.stderr)
        return 2
    sys.path.insert(0, str(code_dir))
    import diag_jsh_adapter as A
    import diag_jsh_core as K
    import diag_jsh_f1_prepare as F1
    import diag_jsh_f2_resources as F2R
    out = Path(a.out)
    try:
        F1.check_out(out, DIAG_ROOT / "out")
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2
    if a.max_wall_s < MIN_WALL_S or a.reserve_bytes < 0:
        print(f"STOP usage: MAX_WALL_S >= {MIN_WALL_S}, reserve >= 0",
              file=sys.stderr)
        return 2
    t0 = time.monotonic()
    try:
        R = dict(F2R.RESOURCES)
        budget, pre = start(out, A, K, a.reserve_bytes, R)
        lr = check_resources(R, a.manifest_sha256, a, K)
        lim = {"reserve": a.reserve_bytes, **lr,
               "deadline": t0 + min(lr["T_s"], a.max_wall_s - WALL_MARGIN_S)}
        want = {f: getattr(a, f"sha_{f.split('.')[0]}") for f in CODE_FILES}
        code = A.check_code(code_dir, want)
        man = load_manifest(a.manifest, a.manifest_sha256, A, K)
        import diag_jsh_i3 as I3
        ctx = F1.bind_production(a, A, K)
        sha = run(ctx, man, a.manifest_sha256, R, out, lim, code, A, K, I3,
                  budget, pre)
    except K.DiagError as exc:
        logger.error("STOP %s", exc)
        print(f"RESULT DIAG_G STOP {exc.code}")
        return 1
    except Exception as exc:  # noqa: BLE001 -- logged, never swallowed
        logger.error("UNEXPECTED %s:\n%s", type(exc).__name__,
                     traceback.format_exc())
        print(f"RESULT DIAG_G FAILED UNEXPECTED {type(exc).__name__}")
        return 2
    print(f"RESULT DIAG_G PASS run_record_sha256={sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SEQREF-MRI-DIAGJSHGE2E v0.1 -- synthetic end-to-end test of the complete DIAG-JSH G pipeline (F1 manifest -> G outputs)
# LIFETIME: KEEP
# =============================================================================
# Plan SEQREF-MRI-DIAGJSHF2 v0.1, addition A (reviewer edit 3, 217 J17).
# SYNTHETIC DATA ONLY: pinned v05lt_fixtures volumes, untrained production
# architecture in the pinned weights format, rows from pinned validate(),
# synthetic ZF values (labelled). The parent builds the fixture, runs F1
# prepare() in a child (real code dict) to obtain a fixture manifest, then
# runs G children under the actual guard and budget: G.start (matplotlib
# cache inside OUT), check_resources, load_manifest, check_code, G.run.
# DECLARED TEST DOUBLES: (1) the fixture binding context replaces
#   F1.bind_production (no campaign in the sandbox; host-only); (2) the
#   resource record is the registered one with manifest_sha256 set to the
#   fixture manifest and T_s / max_wall_s set for the fixture; (3) the
#   child reproduces main()'s call sequence (main's DIAG_ROOT/code and OUT
#   location checks are host-only). Everything else is the production code.
# Usage: python -B test_diag_jsh_g_e2e.py PINNED_REPO_ROOT
# Fixture models: untrained architecture with seeded weight perturbation
#   (1e-2 x N(0,1)), so that R0 decodes differ between arms (an untrained R0
#   decode is arm-identical and the Spearman point estimate is undefined).
# Changelog (v0.1, 2026-11-11): first G end-to-end test.
# =============================================================================
from __future__ import annotations

import ctypes
import json
import logging
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PIN = Path(sys.argv[1]).resolve()
CODE = Path(__file__).resolve().parent
sys.path[:0] = [str(PIN), str(CODE)]
logging.basicConfig(level=logging.CRITICAL)
SEED_ARMS = [(s, a) for s in (2026, 2027, 2028) for a in ("lam0", "lam2")]
G_FILES = ("diag_jsh_core.py", "diag_jsh_adapter.py", "diag_jsh_f1_prepare.py",
           "diag_jsh_i3.py", "diag_jsh_g_run.py", "diag_jsh_f2_resources.py")


def fixture_ctx(fx: Path, mode: str = "pass"):
    import diag_jsh_adapter as A
    import diag_jsh_core as K
    import diag_jsh_f1_prepare as F1
    from seqref_mri.scripts import v05lt_common as C
    from seqref_mri.scripts import v05lt_metrics as M
    from seqref_mri.scripts import v05lt_prodstages as PS
    from seqref_mri.scripts import v05lt_fixtures as FX
    meta = pickle.loads((fx / "meta.pkl").read_bytes())
    sdoc = C.load_split_manifest(meta["mp"], meta["msha"], meta["expect"])
    _, vv = C.build_split_views(meta["root"], sdoc, FX.SYN_HOLDOUT,
                                meta["expect"])
    elig = M.eligibility_index(json.loads((fx / "elig.json").read_text()))
    paths = C.Paths(repo=PIN)
    stages = PS.ProductionStages(paths, paths.data_root,
                                 fixture={"root": str(fx / "fxroot")})
    pin_code = PS.source_identity(PIN)

    def load_selected(k):
        sel = json.loads((fx / f"sel_{k[0]}_{k[1]}.json").read_text())
        sel["rows"] = json.loads(Path(sel["rows_file"]).read_text())
        if mode == "tamper" and k == (2027, "lam2"):
            sel["rows"]["rows"][3]["psnr"] += 0.01
        return sel
    raw = {k: {"path": fx / f"{k}.input", "sha256": "0" * 64}
           for k in F1.INPUT_KEYS}
    return {"binding": {"pin": "SYNTHETIC", "launch_chain": "SYNTHETIC",
                        "launch_inputs_sha256": "SYNTHETIC",
                        "completion_evidence_sha256": "SYNTHETIC",
                        "backend": "SYNTHETIC (1 thread)",
                        "inputs": F1.record_inputs(raw, K)},
            "data": {"vv": vv, "eligibility": elig, "p4": FX._unit_p4(),
                     "s_ref": 1.0},
            "zf_rows": json.loads((fx / "zf.json").read_text()),
            "zf_record_sha256": "SYNTHETIC",
            "load_selected": load_selected,
            "loaded_modules": lambda: A.check_loaded(stages, pin_code,
                                                     root=PIN)}, A, K


def f1child(fx: Path, out: Path) -> int:
    import torch
    torch.set_num_threads(1)
    ctx, A, K = fixture_ctx(fx)
    import diag_jsh_f1_prepare as F1
    budget = F1.start(out, A, 0)
    code = A.check_code(CODE, {f: A.sha256_file(CODE / f)
                               for f in F1.CODE_FILES})
    t0 = time.monotonic()
    lim = {"reserve": 0, "budget": F1.F1_BUDGET, "max_wall_s": 3600.0,
           "deadline": t0 + 3480}
    try:
        sha = F1.prepare(ctx, out, lim, code, A, K, budget)
    except K.DiagError as exc:
        print(f"RESULT DIAG_F1 STOP {exc.code}")
        return 1
    print(f"RESULT DIAG_F1 PASS manifest_sha256={sha}")
    return 0


class Args:
    def __init__(self, cap):
        self.max_wall_s = cap


def gchild(fx: Path, out: Path, mf: Path, mode: str) -> int:
    import torch
    torch.set_num_threads(1)
    import diag_jsh_g_run as G
    import diag_jsh_f2_resources as F2R
    ctx, A, K = fixture_ctx(fx, mode)
    mf_sha = A.sha256_file(mf)
    R = dict(F2R.RESOURCES)
    R.update(manifest_sha256=mf_sha, T_s=3000.0, max_wall_s=3200.0)  # double 2
    if mode == "res_other":
        R["manifest_sha256"] = "f" * 64
    if mode == "budget":
        R["g_budget_bytes"] = 400_000      # cache + log fit, all outputs do not
    cap = 3100.0 if mode == "res_cap" else 3200.0
    want_sha = ("0" * 64) if mode == "man_sha" else mf_sha
    if mode == "man_sha":                    # record bound to the given sha:
        R["manifest_sha256"] = want_sha      # the manifest check must refuse
    t0 = time.monotonic()
    try:
        budget, pre = G.start(out, A, K, 0, R)
        if mode == "unbudgeted":
            open(out / "extra.txt", "w").close()            # must halt (77)
        if mode == "outside":
            open(CODE / "x_outside.txt", "w").close()       # must halt (77)
        lr = G.check_resources(R, want_sha, Args(cap), K)
        lim = {"reserve": 0, **lr,
               "deadline": (t0 - 1) if mode == "deadline"
               else t0 + min(lr["T_s"], cap - G.WALL_MARGIN_S)}
        code = A.check_code(CODE, {f: A.sha256_file(CODE / f)
                                   for f in G_FILES})
        man = G.load_manifest(str(mf), want_sha, A, K)
        if mode == "man_epoch":
            man["checkpoints"]["2028_lam2"]["epoch"] += 1
        if mode == "man_thirds":
            t = man["subgroups"]["H-D"]["thirds"]
            t[0][0], t[2][0] = t[2][0], t[0][0]
        if mode == "man_retained":
            man["retained_slices"] = man["retained_slices"][::-1]
        import diag_jsh_i3 as I3
        if mode == "plotfail":
            def boom(*a, **k):
                raise RuntimeError("synthetic plotting failure")
            I3.figures = boom
        if mode == "native":
            real = I3.write_outputs

            def native_then_write(budget, out_, *a):
                libc = ctypes.CDLL(None)
                fd = libc.open(str(out_ / "native.bin").encode(),
                               os.O_WRONLY | os.O_CREAT, 0o644)
                libc.write(fd, b"x" * 100, 100)
                libc.close(fd)
                return real(budget, out_, *a)
            I3.write_outputs = native_then_write
        sha = G.run(ctx, man, want_sha, R, out, lim, code, A, K, I3, budget,
                    pre)
    except K.DiagError as exc:
        logging.getLogger("seqref_diag.jsh_g").error("STOP %s", exc)
        print(f"RESULT DIAG_G STOP {exc.code}")
        return 1
    except Exception as exc:  # noqa: BLE001 -- as main()
        logging.getLogger("seqref_diag.jsh_g").error(
            "UNEXPECTED %s", type(exc).__name__, exc_info=True)
        print(f"RESULT DIAG_G FAILED UNEXPECTED {type(exc).__name__}")
        return 2
    print(f"RESULT DIAG_G PASS run_record_sha256={sha}")
    return 0


def build_fixture(fx: Path) -> None:
    import torch
    torch.set_num_threads(1)
    import diag_jsh_adapter as A
    from seqref_mri.scripts import v05lt_common as C
    from seqref_mri.scripts import v05lt_eligibility as E
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts import v05lt_fixtures as FX
    from seqref_mri.scripts import v05lt_metrics as M
    from seqref_mri.scripts import v05lt_train as T
    from seqref_mri.src import free_flow_runtime as ffr
    vols = {"file9101.h5": 2, "file9201.h5": 2, "file9202.h5": 1,
            "file9203.h5": 3, "file9204.h5": 2}
    val = ["file9201.h5", "file9202.h5", "file9203.h5", "file9204.h5"]
    root, mp, msha, ex = FX.synth_split(fx, vols, ["file9101.h5"], val)
    sdoc = C.load_split_manifest(mp, msha, ex)
    _, vv = C.build_split_views(root, sdoc, FX.SYN_HOLDOUT, ex)
    edoc = E.build_manifest(E.compute_records(vv, 1.0),
                            {"validation": sdoc["validation"]}, {"cases": []},
                            {"p": 1})
    (fx / "elig.json").write_text(json.dumps(edoc))
    elig = M.eligibility_index(edoc)
    (fx / "meta.pkl").write_bytes(pickle.dumps(
        {"root": root, "mp": mp, "msha": msha, "expect": ex}))

    class Cfg:
        def __init__(self, seed, arm):
            self.seed, self.arm = seed, arm
    for i, (seed, arm) in enumerate(SEED_ARMS):
        model = ffr.build_model(init_seed=seed + 10 * i)
        g = torch.Generator().manual_seed(seed + 10 * i)
        with torch.no_grad():       # untrained R0 decodes are arm-identical
            for prm in model.parameters():   # (constant dPSNR): perturb so
                prm.add_(1e-2 * torch.randn(prm.shape, generator=g))  # they differ
        ms = C.model_state_sha256(model)
        ck = fx / f"run_{seed}_{arm}" / "ckpt"
        ck.mkdir(parents=True)
        wp = ck / "weights_e001.pt"
        torch.save({"schema": EV.WEIGHTS_SCHEMA,
                    "model": {k: v.detach() for k, v in
                              model.state_dict().items()},
                    "meta": {"seed": seed, "arm": arm, "lambda": 0.0,
                             "epoch": 1, "step": 1, "model_state_sha256": ms,
                             "roles": ["best_psnr"]}}, wp)
        rows = M.validate(model, vv, elig, FX._unit_p4(), 1.0)
        doc = T.val_rows_doc(Cfg(seed, arm), 1, ms, rows)
        rp = ck / "val_e001.json"
        rp.write_text(json.dumps(doc))
        run = {"seed": seed, "arm": arm, "ckpt_dir": str(ck),
               "checkpoints": {"selected": {"epoch": 1,
                                            "model_state_sha256": ms}},
               "retained": {"best_psnr": {"name": wp.name,
                                          "sha256": A.sha256_file(wp)},
                            "val_rows_e001": {"name": rp.name,
                                              "sha256": A.sha256_file(rp)}},
               "rows": {1: doc},
               "epochs": {1: {"validation": M.summarise(rows)}}}
        sel = A.selected_checkpoint(run)
        sel.pop("rows")
        (fx / f"sel_{seed}_{arm}.json").write_text(json.dumps(sel))
        if i == 0:
            zf = [{"volume_id": r["volume_id"], "file": r["file"],
                   "slice_index": r["slice_index"],
                   "psnr": 20.0 + 0.5 * j}            # SYNTHETIC ZF values
                  for j, r in enumerate(rows)]
            (fx / "zf.json").write_text(json.dumps(zf))


def parent() -> int:
    import hashlib
    n_ok = n_fail = 0

    def ok(cond, label, detail=""):
        nonlocal n_ok, n_fail
        n_ok, n_fail = n_ok + bool(cond), n_fail + (not cond)
        print(("ok   " if cond else "FAIL ") + label +
              (f"  [{detail}]" if detail else ""), flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="diaggit_"))
    fx = tmp / "fixture"
    fx.mkdir()
    build_fixture(fx)
    ok(True, "E0 synthetic fixture built (6 checkpoints, 8 slices, 4 volumes)")
    f1out = tmp / "f1out"
    f1out.mkdir()
    p = subprocess.run([sys.executable, "-B", __file__, str(PIN), "f1child",
                        str(fx), str(f1out)], capture_output=True, text=True,
                       timeout=3000)
    mf = f1out / "diag_jsh_manifest.json"
    ok(p.returncode == 0 and mf.is_file(), "E1 fixture F1 manifest produced "
       "by F1 prepare() (real code dict)", p.stdout.strip()[-80:] +
       p.stderr.strip()[-300:])
    if not mf.is_file():
        print(f"RESULT {n_ok}/{n_ok + n_fail} checks passed")
        return 1
    mf_bytes = mf.read_bytes()
    home_cache = Path.home() / ".cache" / "matplotlib"
    before_tmp = set(os.listdir(tempfile.gettempdir()))
    before_home = (sorted(os.listdir(home_cache)) if home_cache.is_dir()
                   else None)

    def run_g(mode):
        out = tmp / f"g_{mode}"
        out.mkdir()
        env = {k: v for k, v in os.environ.items()
               if k not in ("MPLCONFIGDIR", "TMPDIR")}
        q = subprocess.run([sys.executable, "-B", __file__, str(PIN), "gchild",
                            str(fx), str(out), str(mf), mode],
                           capture_output=True, text=True, timeout=3000,
                           env=env)
        return q, out

    q, out = run_g("pass")
    ok(q.returncode == 0 and "RESULT DIAG_G PASS" in q.stdout,
       "E2 complete G run under the guard and budget: PASS",
       q.stdout.strip()[-90:] + q.stderr.strip()[-400:])
    names = sorted(p.name for p in out.iterdir())
    want = ["fig5a_likelihood_components.png", "fig5b_spectral_cells.png",
            "fig6a_heterogeneity.png", "figS1_associations.png",
            "figS2_registered_slices.png", "g_run.json", "limitations.md",
            "per_volume.csv", "results.json"]
    exp = sorted(want + [w + ".sha256" for w in want] + ["g.log", "mplconfig"])
    ok(names == exp, "E3 exactly the registered outputs, each with a sidecar, "
       "+ g.log + mplconfig", str(sorted(set(names) ^ set(exp))))
    bad = [w for w in want if (out / w).is_file() and
           (out / (w + ".sha256")).read_text().split()[0] !=
           hashlib.sha256((out / w).read_bytes()).hexdigest()]
    ok(not bad and all((out / w).is_file() for w in want),
       "E4 every sidecar equals its file's sha256", str(bad))

    def strict(path):
        return json.loads(Path(path).read_text(),
                          parse_constant=lambda c: (_ for _ in ()).throw(
                              ValueError(c)))
    try:
        res, rec = strict(out / "results.json"), strict(out / "g_run.json")
        okj = True
    except Exception as exc:  # noqa: BLE001
        okj, res, rec = False, {}, {"err": str(exc)}
    ok(okj and res.get("schema") == "SEQREF-MRI-DIAG-JSH-RESULTS/0.1" and
       set(res) >= {"reconstruction", "D-JAC", "D-SPEC", "D-HET",
                    "reliability"} and
       rec.get("manifest_sha256") == hashlib.sha256(mf_bytes).hexdigest(),
       "E5 results and run record are strict JSON (no NaN/Infinity), bound "
       "to the manifest", str(rec)[:120])
    used = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    ok(okj and used <= rec["resources"]["g_budget_bytes"] and
       rec["pre_guard_files"] and
       all(f.startswith("mplconfig/") for f, _ in rec["pre_guard_files"]),
       "E6 bytes within the budget; the only pre-guard files are the "
       "matplotlib cache inside OUT", f"{used} B, {rec.get('pre_guard_files')}")
    het = res.get("D-HET", {})
    ok(okj and all(het[s]["n_per_third"] ==
                   [len(g) for g in json.loads(mf_bytes)["subgroups"][s]
                    ["thirds"]] for s in ("H-D", "H-F")) and
       len(res["reliability"]["replay_deviations"]) == 6,
       "E7 recorded subgroup sizes used; six reproduction records")
    import csv as _csv
    rows = (list(_csv.reader(open(out / "per_volume.csv")))
            if (out / "per_volume.csv").is_file() else [])
    ok(len(rows) == 1 + 3 * 4 * 3, "E8 per-volume table: 3 seeds x 4 volumes "
       "x {lam0, lam2, lam2-lam0}", str(len(rows)))
    glog = (out / "g.log").read_text() if (out / "g.log").is_file() else ""
    ok(all(f"STAGE G{i}" in glog for i in range(1, 7)) and
       glog.count("decoded once") == 6,
       "E9 log: stages G1-G6; each checkpoint decoded once")
    after_tmp = set(os.listdir(tempfile.gettempdir())) - before_tmp
    after_home = (sorted(os.listdir(home_cache)) if home_cache.is_dir()
                  else None)
    ok(after_home == before_home and not [x for x in after_tmp
                                          if "matplotlib" in x or "mpl" in x],
       "E10 no matplotlib cache or temp folder created outside OUT",
       f"{after_tmp} {after_home}")
    for mode, code, label in (
            ("man_sha", "DIAG_MANIFEST", "E11 wrong manifest sha256"),
            ("res_other", "DIAG_RESOURCES", "E12 resource record bound to "
             "another manifest"),
            ("res_cap", "DIAG_RESOURCES", "E13 command cap != registered cap"),
            ("man_epoch", "DIAG_MANIFEST_MISMATCH", "E14 selected epoch differs "
             "from the manifest"),
            ("man_thirds", "DIAG_MANIFEST_MISMATCH", "E15 recomputed thirds "
             "differ from the manifest"),
            ("man_retained", "DIAG_MANIFEST_MISMATCH", "E16 retained slices "
             "differ from the manifest"),
            ("deadline", "DIAG_DEADLINE", "E17 Python deadline enforced"),
            ("tamper", "DIAG_REPRODUCTION_MISMATCH", "E18 stored-row deviation "
             "in checkpoint 4 of 6 stops G"),
            ("budget", "DIAG_DISK", "E19 outputs larger than the budget "
             "refused before writing"),
            ("native", "DIAG_UNBUDGETED_BYTES", "E20 native (unaudited) write "
             "inside OUT caught by the final byte reconciliation")):
        q, out = run_g(mode)
        ok(q.returncode == 1 and f"STOP {code}" in q.stdout and
           not (out / "g_run.json").exists() or
           (mode == "native" and q.returncode == 1 and
            f"STOP {code}" in q.stdout),
           label + " (no run record)" if mode != "native" else label,
           q.stdout.strip()[-70:] + q.stderr.strip()[-200:])
        if mode == "budget":
            gl = (out / "g.log").read_text() if (out / "g.log").exists() else ""
            part = [x.name for x in out.iterdir() if x.suffix == ".png"]
            ok("DIAG_DISK" in gl and not (out / "results.json").exists() and
               all((out / (x + ".sha256")).exists() for x in part),
               "E19b budget refusal logged; any written output keeps its "
               "sidecar (write-once, partial outputs preserved)")
    q, out = run_g("plotfail")
    gl = (out / "g.log").read_text() if (out / "g.log").exists() else ""
    ok(q.returncode == 2 and "FAILED UNEXPECTED RuntimeError" in q.stdout and
       "synthetic plotting failure" in gl and
       not (out / "results.json").exists(),
       "E21 plotting failure: logged with traceback, exit 2, no outputs")
    q, out = run_g("unbudgeted")
    ok(q.returncode == 77 and "unbudgeted" in q.stderr and
       not (out / "extra.txt").exists(),
       "E22 unregistered write inside OUT halts (77)")
    q, out = run_g("outside")
    ok(q.returncode == 77 and "DIAG_WRITE_OUTSIDE" in q.stderr and
       not (CODE / "x_outside.txt").exists(),
       "E23 write outside OUT halts (77)")
    import shutil
    shutil.rmtree(tmp)
    ok(not tmp.exists(), "E24 fixture folder removed")
    print(f"RESULT {n_ok}/{n_ok + n_fail} checks passed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "f1child":
        sys.exit(f1child(Path(sys.argv[3]), Path(sys.argv[4])))
    if len(sys.argv) > 2 and sys.argv[2] == "gchild":
        sys.exit(gchild(Path(sys.argv[3]), Path(sys.argv[4]),
                        Path(sys.argv[5]), sys.argv[6]))
    sys.exit(parent())

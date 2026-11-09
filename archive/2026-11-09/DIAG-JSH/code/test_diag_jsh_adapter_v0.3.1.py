# SEQREF-MRI-DIAGJSHADPTEST v0.3.1 -- synthetic tests for the DIAG-JSH adapter (mini-spec v1.0 §7, DIAG-I2 list)
# LIFETIME: KEEP
# =============================================================================
# SYNTHETIC DATA ONLY (pinned v05lt_fixtures volumes, untrained production
# architecture, unit P4 stand-in, s_ref = 1.0). Every model, loader, decode
# and metric call is the pinned V05LT code; nothing is mocked. One small
# test double (_RunDirOnly) supplies only run_dir() to reach the adapter's
# missing-run refusal; J10/J11 replace shutil.disk_usage by fixed values to
# test the reserve arithmetic. Neither replaces pinned logic.
# Usage: python -B test_diag_jsh_adapter.py PINNED_REPO_ROOT
# Groups: A source identity · B FFT parity · C metric/label parity ·
#   D masks · E replay + reproduction · F checkpoint loading/selection ·
#   G population/stratifiers/retained · H write guard · I F1 pieces ·
#   J enforced write budget (v0.3).
# Exit 0 only if every check passes.
# Changelog (v0.3.1, 2026-11-09): J10/J11 reserve-arithmetic regression
#   (free-space test double with fixed values).
# Changelog (v0.3, 2026-11-09): group J (budget reserved before writes,
#   strict guard, RLIMIT backstop, log allowance); F1 v0.3 wall minimum.
# Changelog (v0.2, 2026-11-09): guard v0.2 refusals (dir_fd, relative,
#   process creation, symlinked root), allowed read-only git/uname, replay
#   deadline, required selection fields, F1 v0.2 arguments.
# Changelog (v0.1, 2026-11-09): first adapter test set.
# =============================================================================
from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import sys
import tempfile
import types
from fractions import Fraction
from pathlib import Path

PIN = Path(sys.argv[1]).resolve()
CODE = Path(__file__).resolve().parent
sys.path[:0] = [str(PIN), str(CODE)]
logging.basicConfig(level=logging.CRITICAL)

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

import diag_jsh_core as K                                     # noqa: E402
import diag_jsh_adapter as A                                  # noqa: E402
from seqref_mri.scripts import v05lt_common as C              # noqa: E402
from seqref_mri.scripts import v05lt_eligibility as E         # noqa: E402
from seqref_mri.scripts import v05lt_eval as EV               # noqa: E402
from seqref_mri.scripts import v05lt_fixtures as FX           # noqa: E402
from seqref_mri.scripts import v05lt_metrics as M             # noqa: E402
from seqref_mri.scripts import v05lt_prodstages as PS         # noqa: E402
from seqref_mri.scripts import v05lt_stats as ST              # noqa: E402
from seqref_mri.scripts import v05lt_train as T               # noqa: E402
from seqref_mri.src import fastmri_data as FD                 # noqa: E402
from seqref_mri.src import free_flow_runtime as ffr           # noqa: E402

torch.set_num_threads(1)
N_OK = N_FAIL = 0


def ok(cond, label, detail=""):
    global N_OK, N_FAIL
    N_OK, N_FAIL = N_OK + bool(cond), N_FAIL + (not cond)
    print(("ok   " if cond else "FAIL ") + label + (f"  [{detail}]" if detail
                                                  else ""))


def raises(code, fn, label, needle=None):
    try:
        fn()
    except K.DiagError as e:
        good = e.code == code and (needle is None or needle in str(e))
        ok(good, label, f"got {e.code}: {str(e)[:90]}")
        return
    except Exception as e:  # noqa: BLE001 -- reported as a failed check
        ok(False, label, f"untyped {type(e).__name__}: {e}")
        return
    ok(False, label, "no error")


TMP = Path(tempfile.mkdtemp(prefix="diagadp_"))

# A. source identity ----------------------------------------------------------
paths = C.Paths(repo=PIN)
stages = PS.ProductionStages(paths, paths.data_root,
                             fixture={"root": str(TMP / "fxroot")})
pin_code = PS.source_identity(PIN)
rec = A.check_loaded(stages, pin_code, root=PIN)
ok(len(rec) > 10 and all(r["path"].startswith(str(PIN)) for r in rec),
   "A1 loaded seqref_mri modules all pinned (pinned _check_loaded)",
   f"{len(rec)} modules")
bad = dict(pin_code)
rel = "seqref_mri/scripts/v05lt_metrics.py"
bad[rel] = "0" * 64
raises("DIAG_PIN_MISMATCH", lambda: A.check_loaded(stages, bad, root=PIN),
       "A2 changed pinned hash of a loaded module refused")
fake = TMP / "outside" / "fake.py"
fake.parent.mkdir()
fake.write_text("x = 1\n")
mods = dict(sys.modules)
mods["seqref_mri.fake_outside"] = types.SimpleNamespace(__file__=str(fake))
raises("DIAG_PIN_MISMATCH", lambda: A.check_loaded(stages, pin_code,
                                                    root=PIN, modules=mods),
       "A3 seqref_mri module loaded from outside the tree refused")
exp = {"diag_jsh_core.py": A.sha256_file(CODE / "diag_jsh_core.py"),
       "diag_jsh_adapter.py": A.sha256_file(CODE / "diag_jsh_adapter.py")}
ok(A.check_code(CODE, exp) == exp, "A4 diagnostic code = reviewed hashes")
raises("DIAG_CODE_MISMATCH", lambda: A.check_code(
    CODE, {**exp, "diag_jsh_core.py": "1" * 64}),
    "A5 diagnostic code hash mismatch refused")

# B. FFT parity -----------------------------------------------------------------
rng = np.random.default_rng(7)
x = rng.standard_normal((96, 96)) + 1j * rng.standard_normal((96, 96))
t = FD.fft2c(torch.from_numpy(x)).numpy()
ok(np.max(np.abs(t - K.fft2c(x))) <= 1e-12 * np.max(np.abs(t)),
   "B1 core fft2c = imported pinned fastmri_data.fft2c (complex128)")

# C. metric and label parity --------------------------------------------------------
ent = [("v1", "f1", 0), ("v1", "f1", 1), ("v2", "f2", 0), ("v3", "f3", 4),
       ("v3", "f3", 5), ("v3", "f3", 6)]
vals = {(f, s): float(rng.standard_normal()) for _, f, s in ent}
ok(K.volume_means(vals, ent) == E.volume_means(vals, ent),
   "C1 core volume_means = pinned v05lt_eligibility.volume_means (exact)")
Am, Rm = rng.standard_normal((3, 9)) + 30, rng.standard_normal((3, 9)) + 29
v = np.random.default_rng(3).integers(0, 9, size=(50, 9), dtype=np.int64)
d = K.delta_estimate(Am, Rm, v)
ok(d["estimate"] == ST._stat(Am, Rm, "delta"),
   "C2 delta point = pinned _stat('delta') (exact)")
boot = Am[:, v].mean(axis=2).mean(axis=0) - Rm[:, v].mean(axis=2).mean(axis=0)
pin_b = [ST._stat(Am[:, v[b]], Rm[:, v[b]], "delta") for b in range(50)]
ok(np.max(np.abs(boot - pin_b)) <= 1e-12,
   "C3 per-draw delta = pinned _stat on the same draws (1e-12)")
r = K.ratio_estimate(Am, Rm, v)
ok(abs(r["estimate"] - ST._stat(Am, Rm, "ratio")) <= 1e-15,
   "C4 ratio point = pinned _stat('ratio')")
pin_r = [ST._stat(Am[:, v[b]], Rm[:, v[b]], "ratio") for b in range(50)]
lo, hi = np.percentile(pin_r, K.LEVELS, method="linear")
ok(abs(r["ci"][0] - lo) <= 1e-12 and abs(r["ci"][1] - hi) <= 1e-12,
   "C5 ratio interval = percentiles of pinned per-draw ratios")
ci_pin = ST.ci(pin_b)
ok(abs(d["ci"][0] - ci_pin[0]) <= 1e-12 and abs(d["ci"][1] - ci_pin[1]) <= 1e-12
   and ST.LEVELS == (Fraction(1, 40), Fraction(39, 40)),
   "C6 percentile (linear) = pinned ci/percentile_linear")
grid = [-1.0, -0.5, 0.0, 0.005, 0.01, 0.2, 0.5, 0.7]
mapd = {"supported improvement": "improvement supported",
        "supported deterioration": "deterioration supported",
        "not established": "direction unresolved"}
mapp = {"above reference": "0.5 dB improvement supported",
        "below reference": "0.5 dB improvement ruled out at this interval level",
        "straddles reference": "0.5 dB improvement unresolved"}
pairs = [(a, b) for a in grid for b in grid if a <= b]
ok(all(mapd[ST.support_label(a, b)] == K.direction_label(a, b) and
       mapp[ST.practical_label(a, b)] == K.practical_label(a, b) and
       ST.density_label(a, b) == K.density_label(a, b) for a, b in pairs),
   "C7 direction/practical/density labels = pinned (boundaries incl.)",
   f"{len(pairs)} intervals")
ok(K.PRACTICAL_DB == ST.PRACTICAL_DB and K.DENSITY_BOUND == ST.DENSITY_BOUND
   and K.NLL_TOL_REL == M.NLL_DECOMP_TOL_REL and K.N == FD.CELL_HW,
   "C8 frozen constants = pinned values")

# synthetic data (pinned fixture volumes) --------------------------------------------
VOLS = {"file9101.h5": 2, "file9201.h5": 2, "file9202.h5": 1,
        "file9203.h5": 3, "file9204.h5": 2}
VAL = ["file9201.h5", "file9202.h5", "file9203.h5", "file9204.h5"]
root, mp, msha, ex = FX.synth_split(TMP, VOLS, ["file9101.h5"], VAL)
sdoc = C.load_split_manifest(mp, msha, ex)
_, vv = C.build_split_views(root, sdoc, FX.SYN_HOLDOUT, ex)
edoc = E.build_manifest(E.compute_records(vv, 1.0),
                        {"validation": sdoc["validation"]}, {"cases": []},
                        {"p": 1})
elig = M.eligibility_index(edoc)
p4, s_ref = FX._unit_p4(), 1.0
NS = len(vv)

# D. masks ---------------------------------------------------------------------------
regs = [A.mask_identity(vv[i], *vv.entry(i)) for i in range(NS)]
ok(all(a == b for a, b in regs), "D1 regenerated masks = loaded masks (sha256)",
   f"{NS} slices")
f0, s0 = vv.entry(0)
seed0 = FD.canonical_mask_seed(FD.EVAL_BASE_SEED, f0, int(s0))
m0 = FD.make_cartesian_mask(int(vv[0]["y"].shape[-1]), seed0).astype(bool)
ok(regs[0][0] == hashlib.sha256(m0.tobytes(order="C")).hexdigest(),
   "D2 regeneration = pinned canonical_mask_seed + make_cartesian_mask")
it = dict(vv[0])
it["meta"] = {**it["meta"], "mask_seed": seed0 + 1}
raises("DIAG_MASK_SEED", lambda: A.mask_identity(it, f0, s0),
       "D3 wrong loaded mask seed refused")
it2 = dict(vv[0])
mm = np.asarray(it2["mask"]).copy().reshape(-1)
mm[0] = 1 - mm[0]
it2["mask"] = torch.from_numpy(mm)
a2, b2 = A.mask_identity(it2, f0, s0)
ok(a2 != b2, "D4 altered loaded mask detected by identity comparison")


# E. replay and reproduction ---------------------------------------------------------
class Cfg:
    def __init__(self, seed, arm):
        self.seed, self.arm = seed, arm


def stored_doc(model, seed, arm, epoch=5):
    rows = M.validate(model, vv, elig, p4, s_ref)
    return T.val_rows_doc(Cfg(seed, arm), epoch, C.model_state_sha256(model),
                          rows)


model = ffr.build_model(init_seed=2026)
doc = stored_doc(model, 2026, "lam0")
seen = []
rep = A.replay(model, vv, elig, p4, s_ref,
               on_slice=lambda k, row, sp, xh, xt: seen.append((k, sp, xh, xt)))
out = A.reproduce(doc, rep, NS)
ok(out["slices"] == NS and out["bitwise_equal_slices"] == NS,
   "E1 full replay reproduces pinned validate() rows (single thread)",
   f"{out}")
ok([k for k, *_ in seen] == [(r["volume_id"], r["file"], r["slice_index"])
                             for r in doc["rows"]],
   "E2 replay order = stored validation order")
ok(all(abs(sp["sc_cells"].sum() - sp["sc_full"]) <= 1e-9 * sp["sc_full"] +
           1e-24 for _, sp, _, _ in seen),
   "E3 spectral cell sums = full error on real decodes")
pm = [float(np.mean((np.abs(xh) - np.abs(xt)) ** 2)) for _, _, xh, xt in seen]
ok(all(abs(sp["sm_pixel_mse"] - p) <= 1e-12 * max(p, 1e-300)
       for (_, sp, _, _), p in zip(seen, pm)),
   "E4 S-M cell sum / 96^2 = magnitude pixel MSE (Parseval)")
ok(rep["sm_rel_dev"] <= 1e-6, "E5 S-M MSE vs pinned psnr_record MSE "
   "(recorded deviation)", f"max rel {rep['sm_rel_dev']:.3e}")
ok(all(r["nll_decomp_abs_dev"] == s["nll_decomp_abs_dev"]
       for r, s in zip(rep["rows"].values(), doc["rows"])),
   "E6 replay NLL components = stored (bitwise)")
rep_b = A.replay(model, vv, elig, p4, s_ref)
ok(A.check_reference_equal([rep["hf"], rep_b["hf"]]) == rep["hf"],
   "E7 reference HF fractions identical across replays")
raises("DIAG_REFERENCE_MISMATCH", lambda: A.check_reference_equal(
    [rep["hf"], {**rep["hf"], vv.entry(0): 0.5}]),
    "E8 differing reference HF refused")
model2 = ffr.build_model(init_seed=2027)
rep2 = A.replay(model2, vv, elig, p4, s_ref)
raises("DIAG_REPRODUCTION_MISMATCH", lambda: A.reproduce(doc, rep2, NS),
       "E9 other checkpoint vs stored rows refused (identity)")
k0 = next(iter(rep["rows"]))


def shifted(field, dv):
    return {**rep, "rows": {**rep["rows"], k0: {**rep["rows"][k0],
                                                 field: rep["rows"][k0][field]
                                                 + dv}}}


o_in = A.reproduce(doc, shifted("psnr", 0.0009), NS)
ok(o_in["bitwise_equal_slices"] == NS - 1 and
   abs(o_in["max_abs_psnr_dev"] - 0.0009) < 1e-9,
   "E10a PSNR deviation within 0.001 dB accepted and recorded")
raises("DIAG_REPRODUCTION_MISMATCH", lambda: A.reproduce(
    doc, shifted("psnr", 0.0011), NS),
    "E10b PSNR deviation > 0.001 dB refused", "|dPSNR|")
o_n = A.reproduce(doc, shifted("nll_per_dim", 9e-5), NS)
ok(abs(o_n["max_abs_nll_dev"] - 9e-5) < 1e-12,
   "E10c NLL deviation within 1e-4 nats/dim accepted and recorded")
raises("DIAG_REPRODUCTION_MISMATCH", lambda: A.reproduce(
    doc, shifted("nll_per_dim", 1.1e-4), NS),
    "E10d NLL deviation > 1e-4 nats/dim refused", "|dNLL|")
repm = {**rep, "rows": {**rep["rows"], k0: {**rep["rows"][k0],
                                             "mask": "0" * 64}}}
raises("DIAG_REPRODUCTION_MISMATCH", lambda: A.reproduce(doc, repm, NS),
       "E11 replay mask != regenerated mask refused", "mask")
raises("DIAG_POPULATION_MISMATCH", lambda: A.reproduce(doc, rep, NS + 1),
       "E12 miscounted population refused")
rev = {**rep, "rows": dict(reversed(list(rep["rows"].items())))}
raises("DIAG_POPULATION_MISMATCH", lambda: A.reproduce(doc, rev, NS),
       "E13 reordered replay refused")
raises("DIAG_POPULATION_EMPTY", lambda: A.replay(model, vv, elig, p4, s_ref,
                                                 positions=[]),
       "E14 empty replay refused")
docb = {**doc, "rows": [dict(doc["rows"][0], nll_prior_per_dim=doc["rows"][0][
    "nll_prior_per_dim"] + 1e-3)] + doc["rows"][1:]}
raises("DIAG_NLL_IDENTITY", lambda: A.reproduce(docb, rep, NS),
       "E15 stored NLL decomposition breach refused before comparison")
el_bad = {**elig, vv.entry(0): {**elig[vv.entry(0)],
                                "eligible": not elig[vv.entry(0)]["eligible"]}}
raises("DIAG_POPULATION_MISMATCH", lambda: A.replay(model, vv, el_bad, p4,
                                                    s_ref, positions=[0]),
       "E16 eligibility flag mismatch refused")

# F. checkpoint loading and selection ---------------------------------------------------
ck = TMP / "run" / "ckpt"
ck.mkdir(parents=True)
ms = C.model_state_sha256(model)
wp = ck / "weights_e005.pt"
torch.save({"schema": EV.WEIGHTS_SCHEMA,
            "model": {k_: v_.detach() for k_, v_ in model.state_dict().items()},
            "meta": {"seed": 2026, "arm": "lam0", "lambda": 0.0, "epoch": 5,
                     "step": 10, "model_state_sha256": ms,
                     "roles": ["best_psnr"]}}, wp)
wsha = A.sha256_file(wp)
run = {"seed": 2026, "arm": "lam0", "ckpt_dir": str(ck),
       "checkpoints": {"selected": {"epoch": 5, "model_state_sha256": ms}},
       "retained": {"best_psnr": {"name": "weights_e005.pt", "sha256": wsha},
                    "val_rows_e005": {"name": "val_e005.json",
                                      "sha256": "a" * 64}},
       "rows": {5: doc},
       "epochs": {e_: {"validation": {"all_slices": {"psnr": p_}}}
                  for e_, p_ in ((4, 20.0), (5, 21.0), (6, 21.0))}}
sel = A.selected_checkpoint(run)
ok(sel["weights"] == str(wp) and sel["weights_sha256"] == wsha and
   sel["epoch"] == 5 and sel["rows"] is doc,
   "F1 selected checkpoint = registered selected role (best_psnr)")
m_l, vc = A.load_model(sel)
ok(C.model_state_sha256(m_l) == ms and vc.file_sha256 == wsha,
   "F2 pinned load_trained_checkpoint: state and file hash verified")
calls = []
orig = torch.load
torch.load = lambda *a, **k: calls.append(1) or orig(*a, **k)
raises("DIAG_CHECKPOINT_MISMATCH", lambda: A.load_model(
    {**sel, "weights_sha256": "b" * 64}),
    "F3 weights file hash mismatch refused", "POSTEVAL_FILE_MISMATCH")
ok(not calls, "F4 identity verified BEFORE torch.load (not called)")
torch.load = orig
raises("DIAG_CHECKPOINT_MISMATCH", lambda: A.load_model(
    {**sel, "model_state_sha256": "c" * 64}),
    "F5 model-state mismatch refused", "POSTEVAL_META_MISMATCH")
raises("DIAG_INPUT_MISSING", lambda: A.load_model(
    {**sel, "weights": str(ck / "weights_e099.pt")}),
    "F6 missing weights file refused")
raises("DIAG_SELECTION_MISMATCH", lambda: A.selected_checkpoint(
    {**run, "retained": {**run["retained"], "best_psnr":
                         {"name": "weights_e004.pt", "sha256": wsha}}}),
    "F7 retained best_psnr != selected epoch refused")
raises("DIAG_SELECTION_MISMATCH", lambda: A.selected_checkpoint(
    {**run, "rows": {5: {**doc, "model_state_sha256": "d" * 64}}}),
    "F8 rows not bound to the selected state refused")
raises("DIAG_SELECTION_MISMATCH", lambda: A.selected_checkpoint(
    {**run, "retained": {"best_psnr": run["retained"]["best_psnr"]}}),
    "F8b missing selected row-file record refused")
raises("DIAG_SELECTION_MISMATCH", lambda: A.selected_checkpoint(
    {**run, "epochs": {**run["epochs"], 7: {"validation": {"all_slices":
                                                         {"psnr": 22.0}}}}}),
    "F8c selected epoch != first best-PSNR epoch refused")
ok(sel["rows_file"] == str(ck / "val_e005.json") and
   len(sel["selection_evidence"]["per_epoch_psnr_sha256"]) == 64,
   "F8d resolvable row path and hashed selection evidence (tie -> first)")
import time as _t                                             # noqa: E402
raises("DIAG_DEADLINE", lambda: A.replay(model, vv, elig, p4, s_ref,
                                         deadline=_t.monotonic() - 1),
       "F8e replay stops at the deadline")


class _RunDirOnly:          # test double: run_dir only (see header)
    def run_dir(self, seed, arm):
        return TMP / "no_such_run"


raises("DIAG_INPUT_MISSING", lambda: A.load_selected(_RunDirOnly(), 2026,
                                                     "lam0"),
       "F9 missing run directory refused")
raises("DIAG_INPUT_INVALID", lambda: A.load_zf(stages),
       "F10 unbound inputs refused (pinned binding required)")

# G. population, stratifiers, retained slices ------------------------------------------------
docs = {(s_, a_): {**doc, "seed": s_, "arm": a_} for s_, a_ in A.SEED_ARMS}
zf = [{"volume_id": r["volume_id"], "file": r["file"],
       "slice_index": r["slice_index"], "psnr": 20.0 + i * 0.5}
      for i, r in enumerate(doc["rows"])]
ents = A.check_population(docs, zf)
ok(ents == A.entries_of(doc), "G1 six identical populations + ZF accepted")
rows_alt = list(reversed(doc["rows"]))
d_alt = T.val_rows_doc(Cfg(2027, "lam2"), 5, ms, rows_alt)
raises("DIAG_POPULATION_MISMATCH", lambda: A.check_population(
    {**docs, (2027, "lam2"): d_alt}, zf),
    "G2 slice_ids_sha256 differs between runs refused")
raises("DIAG_POPULATION_MISMATCH", lambda: A.check_population(docs, zf[1:]),
       "G3 ZF missing a slice refused")
zf_v = [dict(zf[0], volume_id="other")] + zf[1:]
raises("DIAG_POPULATION_MISMATCH", lambda: A.check_population(docs, zf_v),
       "G4 ZF volume ID mismatch refused")
raises("DIAG_POPULATION_MISMATCH", lambda: A.check_population(
    {k_: v_ for k_, v_ in docs.items() if k_ != (2028, "lam2")}, zf),
    "G5 missing run refused")
stt = A.stratifiers(ents, zf, rep["hf"])
vols = sorted({e[0] for e in ents})
hd = stt["H-D"]
ok([len(g) for g in hd["thirds"]] == [2, 1, 1] and
   hd["thirds"][0][0] == min(hd["values"], key=lambda k_: (hd["values"][k_], k_)),
   "G6 H-D thirds: hardest (lowest ZF PSNR) first, sizes floor+remainder")
ok(stt == A.stratifiers(ents, zf, dict(rep["hf"])) and
   len(stt["H-F"]["sha256"]) == 64,
   "G7 stratifiers deterministic, membership hashed")
ret = A.retained_slices(ents, elig)
pick = np.random.default_rng(K.DIAG_SEED + 1).choice(len(vols), 4,
                                                     replace=False)
ok([r_["volume_id"] for r_ in ret] == [vols[i] for i in pick] and
   all(elig[(r_["file"], r_["slice_index"])]["midpoint"] for r_ in ret),
   "G8 retained slices = registered draw, midpoint slices")
el_m = {k_: {**v_, "midpoint": True} for k_, v_ in elig.items()}
raises("DIAG_RETAINED", lambda: A.retained_slices(ents, el_m),
       "G9 non-unique midpoint refused")

# H. write guard (subprocesses) ------------------------------------------------------------
gro = TMP / "guard_root"
gro.mkdir()
outside = TMP / "guard_outside"
outside.mkdir()
(outside / "readme.txt").write_text("read only\n")
PRE = (f"import sys; sys.path[:0]=[{str(CODE)!r}]\n"
       "import diag_jsh_adapter as A\n"
       f"A.install_write_guard({str(gro)!r})\n")
cases = {
    "H1 write inside root allowed":
        (f"open({str(gro / 'a.txt')!r}, 'w').write('x')\n"
         f"import os; os.rename({str(gro / 'a.txt')!r}, {str(gro / 'b.txt')!r})\n"
         f"print(open({str(outside / 'readme.txt')!r}).read().strip())", 0),
    "H2 open-for-write outside refused":
        (f"open({str(outside / 'x.txt')!r}, 'w')", A.WRITE_EXIT),
    "H3 os.open(O_CREAT) outside refused":
        (f"import os; os.open({str(outside / 'y.txt')!r}, os.O_CREAT|os.O_WRONLY)",
         A.WRITE_EXIT),
    "H4 rename into outside refused":
        (f"open({str(gro / 'c.txt')!r}, 'w').close()\nimport os; "
         f"os.rename({str(gro / 'c.txt')!r}, {str(outside / 'c.txt')!r})",
         A.WRITE_EXIT),
    "H5 mkdir outside refused":
        (f"import os; os.mkdir({str(outside / 'd')!r})", A.WRITE_EXIT),
    "H6b os.open via dir_fd to outside refused (review finding)":
        (f"import os; fd = os.open({str(outside)!r}, os.O_RDONLY); "
         "os.open('dirfd.txt', os.O_CREAT|os.O_WRONLY, dir_fd=fd)",
         A.WRITE_EXIT),
    "H6c os.mkdir via dir_fd refused":
        (f"import os; fd = os.open({str(outside)!r}, os.O_RDONLY); "
         "os.mkdir('dfd', dir_fd=fd)", A.WRITE_EXIT),
    "H6d relative write path refused (even inside root)":
        (f"import os; os.chdir({str(gro)!r}); open('rel.txt', 'w')",
         A.WRITE_EXIT),
    "H6e subprocess writer refused":
        (f"import subprocess; subprocess.run(['touch', {str(outside / 'p.txt')!r}])",
         A.WRITE_EXIT),
    "H6f os.system refused":
        (f"import os; os.system('touch {outside / 's.txt'}')", A.WRITE_EXIT),
    "H6g read-only git rev-parse and uname -p allowed":
        (f"import subprocess as s; s.run(['git', '-C', {str(PIN)!r}, "
         "'rev-parse', 'HEAD'], check=True, capture_output=True); "
         "import platform; platform.processor()", 0),
    "H6h git subcommand other than rev-parse/ls-tree refused":
        (f"import subprocess as s; s.run(['git', '-C', {str(PIN)!r}, "
         "'status'], capture_output=True)", A.WRITE_EXIT),
    "H6 append via symlink to outside refused":
        (f"import os; os.symlink({str(outside / 'readme.txt')!r}, "
         f"{str(gro / 'ln')!r}); open({str(gro / 'ln')!r}, 'a')",
         A.WRITE_EXIT),
}
for label, (body, want) in cases.items():
    pre = PRE.replace("A.install_write_guard(", "A.install_write_guard(") \
        .replace(f"{str(gro)!r})", f"{str(gro)!r}, git_repo={str(PIN)!r})")
    p = subprocess.run([sys.executable, "-B", "-c", pre + body],
                       capture_output=True, text=True, timeout=120)
    good = p.returncode == want and (want == 0 or "DIAG_WRITE_OUTSIDE" in
                                     p.stderr)
    ok(good, label, f"rc {p.returncode}")
lnk = TMP / "guard_link"
lnk.symlink_to(gro)
p = subprocess.run([sys.executable, "-B", "-c",
                    f"import sys; sys.path[:0]=[{str(CODE)!r}]\n"
                    "import diag_jsh_adapter as A, diag_jsh_core as K\n"
                    f"try:\n A.install_write_guard({str(lnk)!r})\n"
                    "except K.DiagError as e:\n print(e.code)"],
                   capture_output=True, text=True, timeout=120)
ok("DIAG_GUARD" in p.stdout, "H6i guard root reached through a symlink refused")
ok(not any((outside / n).exists() for n in ("x.txt", "y.txt", "c.txt", "d",
                                            "dirfd.txt", "dfd", "p.txt",
                                            "s.txt"))
   and (outside / "readme.txt").read_text() == "read only\n",
   "H7 nothing created or changed outside the root")

# I. F1 preparation pieces (sandbox: no campaign; refusal and pieces only) ----------
pre2 = {**rep, "rows": dict(list(rep["rows"].items())[:2])}
ok(A.reproduce_prefix(doc, pre2)["slices"] == 2,
   "I1 early 2-slice check (prefix) passes on the true checkpoint")
raises("DIAG_REPRODUCTION_MISMATCH", lambda: A.reproduce_prefix(
    {**doc, "model_state_sha256": "e" * 64}, pre2),
    "I2 early check refuses a wrong checkpoint identity")
sys.argv = [sys.argv[0]]
import diag_jsh_f1_prepare as F1                              # noqa: E402
tb = F1.bootstrap_probe(K, vols, stt)
ok(tb > 0.0, "I3 counted bootstrap probe runs on the real thirds shape",
   f"{tb:.2f} s for V={len(vols)}")
p = subprocess.run([sys.executable, "-B", str(CODE / "diag_jsh_f1_prepare.py"),
                    "--chain", "x", "--chain-sha256", "0" * 64, "--out",
                    str(TMP / "f1out"), "--reserve-bytes", "0",
                    "--max-wall-s", "600", "--sha-diag_jsh_core", "0" * 64,
                    "--sha-diag_jsh_adapter", "0" * 64,
                    "--sha-diag_jsh_f1_prepare", "0" * 64],
                   capture_output=True, text=True, timeout=120)
ok(p.returncode == 2 and not (TMP / "f1out").exists(),
   "I4 F1 refuses to run outside DIAG_JSH/code (no output created)")
p = subprocess.run(["bash", "-n", str(CODE / "diag_jsh_f1_cmd.sh")],
                   capture_output=True, text=True)
p2 = subprocess.run(["bash", str(CODE / "diag_jsh_f1_cmd.sh")],
                    capture_output=True, text=True)
ok(p.returncode == 0 and p2.returncode == 2 and "usage" in p2.stderr,
   "I5 F1 command: syntax OK; refuses without arguments")

# J. enforced write budget (subprocesses; the guard is process-wide) ---------------
def jrun(body, budget=16, reserve=0):
    root = Path(tempfile.mkdtemp(dir=TMP, prefix="j_"))
    pre = (f"import sys, os, errno; sys.path[:0]=[{str(CODE)!r}]\n"
           "import diag_jsh_adapter as A, diag_jsh_core as K\n"
           f"R={str(root)!r}\nA.install_write_guard(R)\n"
           f"B=A.WriteBudget(R, {reserve}, {budget})\n")
    p = subprocess.run([sys.executable, "-B", "-c", pre + body],
                       capture_output=True, text=True, timeout=120)
    return p, root


TRY = ("try:\n {}\nexcept K.DiagError as e:\n print('CODE', e.code)\n")
p, r = jrun(TRY.format("B.write_new(R + '/a.bin', b'x' * 32, 'w')"), 16)
ok("CODE DIAG_DISK" in p.stdout and not (r / "a.bin").exists(),
   "J1 32-byte write after a 16-byte budget refused BEFORE writing "
   "(review repro)")
p, r = jrun("B.write_new(R + '/a.bin', b'x' * 16, 'w')\n" +
            TRY.format("B.write_new(R + '/b.bin', b'y', 'w')"), 16)
ok("CODE DIAG_DISK" in p.stdout and (r / "a.bin").stat().st_size == 16 and
   not (r / "b.bin").exists(), "J2 exact budget used; next byte refused")
p, r = jrun("pass", 16, reserve=10 ** 18)
ok(p.returncode != 0 and "DIAG_DISK" in p.stderr,
   "J3 reserve above live free space refused at attachment")
p, r = jrun("open(R + '/free.txt', 'w')")
ok(p.returncode == A.WRITE_EXIT and "unbudgeted" in p.stderr and
   not (r / "free.txt").exists(),
   "J4 strict: unregistered write-open inside the root halts")
p, r = jrun("os.mkdir(R + '/d')")
ok(p.returncode == A.WRITE_EXIT and not (r / "d").exists(),
   "J5 strict: mkdir inside the root halts")
p, r = jrun("q = B.register(R + '/big.bin')\nfh = open(q, 'wb', buffering=0)\n"
            "print('N', fh.write(b'z' * 4096))\n"
            "try:\n fh.write(b'z')\nexcept OSError as e:\n"
            " print('ERRNO', e.errno == errno.EFBIG)", 1024)
ok("N 1024" in p.stdout and "ERRNO True" in p.stdout and
   (r / "big.bin").stat().st_size == 1024,
   "J6 RLIMIT_FSIZE backstop: file capped at the limit (partial write, "
   "then EFBIG), no kill")
p, r = jrun("import logging\nh = A.BudgetLogHandler(B, R + '/l.log', 200)\n"
            "lg = logging.getLogger('t'); lg.addHandler(h); lg.propagate = False\n"
            "for i in range(50): lg.warning('line %d' % i)", 10000)
ok(p.returncode == A.LOG_EXIT and "DIAG_LOG_BUDGET" in p.stderr and
   (r / "l.log").stat().st_size <= 200,
   "J7 log allowance exhausted halts (78), file within allowance")
p, r = jrun("import logging\nh = A.BudgetLogHandler(B, R + '/l.log', 4000)\n"
            "lg = logging.getLogger('t'); lg.addHandler(h); lg.propagate = False\n"
            "for i in range(5): lg.warning('x' * 50)\nprint('USED', B.used)\n" +
            TRY.format("B.write_new(R + '/m.bin', b'm' * 400, 'w')"), 600)
ok("CODE DIAG_DISK" in p.stdout and not (r / "m.bin").exists(),
   "J8 log bytes count in the shared budget", p.stdout.strip()[:40])
p, r = jrun("import tempfile; os.environ['TMPDIR'] = R\n"
            "tempfile.tempdir = None; tempfile.mkstemp()")
ok(p.returncode == A.WRITE_EXIT and not any(r.iterdir()),
   "J9 temporary-file attempt halts (no tmp folder supported)")

FREE = ("import shutil, collections\nU = collections.namedtuple('U', "
        "'total used free')\nFREE = [116]\n"
        "shutil.disk_usage = lambda p: U(0, 0, FREE[0])\n")

def jrun_free(body):          # free space test double (fixed values only)
    root = Path(tempfile.mkdtemp(dir=TMP, prefix="j_"))
    pre = (f"import sys, os; sys.path[:0]=[{str(CODE)!r}]\n" + FREE +
           "import diag_jsh_adapter as A, diag_jsh_core as K\n"
           f"R={str(root)!r}\nA.install_write_guard(R)\n")
    q = subprocess.run([sys.executable, "-B", "-c", pre + body],
                       capture_output=True, text=True, timeout=120)
    return q, root


p, r = jrun_free("B = A.WriteBudget(R, 100, 16)\nFREE[0] = 100\n" +
                 TRY.format("B.write_new(R + '/r.bin', b'x' * 16, 'w')"))
ok("CODE DIAG_DISK" in p.stdout and not (r / "r.bin").exists(),
   "J10 regression (review repro): free 100 = reserve 100, 16-byte write "
   "refused (would leave 84)")
p, r = jrun_free("B = A.WriteBudget(R, 100, 16)\n" +
                 "B.write_new(R + '/r.bin', b'x' * 16, 'w')\nprint('WROTE')")
ok("WROTE" in p.stdout and (r / "r.bin").stat().st_size == 16,
   "J11 free 116 = reserve 100 + budget 16: the full budget may be written")

import shutil                                                 # noqa: E402
shutil.rmtree(TMP)
ok(not TMP.exists(), "I6 synthetic fixture folder removed")
print(f"RESULT {N_OK}/{N_OK + N_FAIL} checks passed")
sys.exit(0 if N_FAIL == 0 else 1)

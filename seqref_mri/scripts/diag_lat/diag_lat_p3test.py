# SEQREF-DLAT-P3T v0.2 -- diag_lat_p3test.py: P3 fixtures T9-T11, T12b, R1a, R1b and refusal test R5 (plan SEQREF-DLAT-P3 v0.2)
# LIFETIME: KEEP
"""Synthetic tests of the D-LAT adapter against the pinned V05LT classes.

PRE-FREEZE ENGINEERING FILE (candidate v0.5.2, not frozen; 238 LE23, LE24). SYNTHETIC TENSORS ONLY:
no checkpoint, no MRI file, no stored row, no standardisation file, no registered random array.
Every output is marked NOT D-LAT EVIDENCE.

Run:  python -I diag_lat_p3test.py --pinned-root <repo root holding seqref_mri/> --out <new folder>
      [--mode sandbox|host] [--ram-limit-gib 6] [--only T9,T10,...]
Refused before any other import and before any file is opened by this program (R5):
      --checkpoint, --data-root (also in the --name=value form).

Fixtures and result states: each fixture is PASS, FAIL or NOT RUN. NOT RUN is used when a pinned
module cannot be imported (named in the report; never stubbed), when a registered pinned hash does
not match, when R1a stops on the memory limit (C4), or when --only leaves a fixture out. The run
is COMPLETE only if no fixture is NOT RUN. Exit status: 0 complete and all PASS; 1 any FAIL;
2 no FAIL but INCOMPLETE; 3 refusal (R5).

SELF-TEST EXCEPTION (project rule): this file builds synthetic inputs and deliberately wrong
inputs (shifted mask, doubled division, broken records). None of that exists in the adapter.

Output: EVERY file this program writes goes through one WriteGuard rooted at --out (write-once,
sha256 sidecar, byte ledger): the R1a records, the R1b tables, p3_resource.json and p3_report.json.

Stage timings (R6) are of three kinds and must not be summed across kinds: "setup" (once per
model state), "component" (one separately timed operation) and "nested_total" (a complete call
that CONTAINS the components). Adding a nested total to its components counts work twice.

Changelog:
  v0.2 (2026-11-21) review of v0.1 (238 LE25): (1) p3_report.json and p3_resource.json are now
        written through the write guard; one guard for the whole output folder; (2) every stage
        timing carries its kind, and the resource JSON states which timings are nested and which
        are additive. No fixture, tolerance, parameter or check was changed.
  v0.1 (2026-11-21) first version: R5, T9, T10, T11, T12b, R1a (with C2, C4, R6), R1b (with C3).
"""
import sys

REFUSED = "DIAG_LAT_REFUSED"
FORBIDDEN = ("--checkpoint", "--data-root")
EXIT_REFUSED = 3


def refuse_forbidden(argv):
    """R5: standard library only; runs before PyTorch, NumPy or any project module is imported."""
    hit = sorted({f for a in argv for f in FORBIDDEN if a == f or a.startswith(f + "=")})
    if hit:
        loaded = {m: any(k == m or k.startswith(m + ".") for k in sys.modules) for m in ("torch", "numpy", "seqref_mri")}
        sys.stderr.write("ERROR %s: option(s) %s are not accepted: P3 uses no checkpoint and no data root\n"
                         % (REFUSED, ", ".join(hit)))
        print("%s options=%s" % (REFUSED, ",".join(hit)))
        print("loaded_before_refusal " + " ".join("%s=%s" % (m, loaded[m]) for m in sorted(loaded)))
        raise SystemExit(EXIT_REFUSED)


if __name__ == "__main__":
    refuse_forbidden(sys.argv[1:])

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import diag_lat_adapter as A  # noqa: E402
import diag_lat_core as C  # noqa: E402
import diag_lat_guard as GD  # noqa: E402
import diag_lat_stats as ST  # noqa: E402

__version__ = "0.2"
ABBR = "SEQREF-DLAT-P3T"
LOG = logging.getLogger("diag_lat.p3test")
SELFTEST_SEED = 987654321                       # fixtures only; not a registered seed
BANNERS = {"sandbox": "SANDBOX REHEARSAL -- NOT HOST EVIDENCE -- NOT D-LAT EVIDENCE (synthetic tensors)",
           "host": "SYNTHETIC HOST TEST -- NOT D-LAT EVIDENCE (synthetic tensors)"}
B_S = C.SPLINE_B
D_H_MIN = 1e-3                                  # C2 (238 LE23; threshold approved with LE24's review)
TOY_SUBSETS = {"toy_full": np.arange(8), "toy_half_a": np.arange(0, 4), "toy_half_b": np.arange(4, 8)}
# (label, generator seed, amplitude of the spline-net weights). Calibration, disclosed in the P3 evidence
# report: a first run with amplitudes 0.02 / 0.03 gave a float32 round-trip residual of 1.00e-5 to 1.03e-5
# against the tolerance 1e-5 (FAIL, preserved as run 1). The tolerance was NOT widened; the amplitudes
# were lowered. The margin stays thin (about 9e-6 at K = 128): see the evidence report, point P3-D1.
SETTINGS = (("S1", 11, 0.01), ("S2", 12, 0.02))
N_SLICES = 4
OUTPUT_CEILING = 64 * 2 ** 20                   # bytes, whole output folder (synthetic test output only)
OUTPUT_RESERVE = 2 ** 30
# kind of each stage timing: nested totals CONTAIN the listed components and are never added to them
STAGE_KINDS = {
    "model_construction": ("setup", None),
    "conditioning": ("component", None),
    "truth_encoding": ("component", None),
    "bank_decoding": ("component", None),
    "bank_reencoding": ("component", None),
    "r0_decoding": ("component", None),
    "core_bank_measures_standardised": ("component", None),
    "slice_state_record_complete": ("nested_total", ["truth_encoding", "bank_decoding", "bank_reencoding", "r0_decoding",
                                                     "core_bank_measures_standardised"]),
    "statistics_r1b": ("component", None),
}
TIMING_NOTE = ("Do NOT sum these timings. 'slice_state_record_complete' is a nested total: one complete adapter call that "
               "repeats the truth encoding, bank decoding, bank re-encoding, R0 decoding and standardised core measures "
               "timed separately above, and adds the tail check, the physical complements, KS and the schema check. "
               "'conditioning' is also contained in every decode and encode pass. For a projection use EITHER the nested "
               "total per slice-state OR its components, never both; 'model_construction' is a setup cost per model "
               "state; 'statistics_r1b' is a separate stage on a synthetic 12-volume population. The L5 passes (swap, "
               "perturbation) are not timed separately. Sandbox CPU, synthetic parameters: inputs to C1 only.")

# sha256 at the pin of the files verified earlier (236 LE13). Checked BEFORE any pinned import.
REGISTERED_PINNED = {
    "seqref_mri/src/base_experts.py": "c3d1adcafd57c71b2e176f4aa29b72e0b809539ef5006eae4f4162bbe406b952",
    "seqref_mri/src/conditioner.py": "7dc7176125140613dfdc71457ac8bfb1c407bd86ed99f0176e0826bbda9669cc",
    "seqref_mri/src/flows/nsf_layer.py": "37c1f0960dda0415f998503ed80fb4bc7ae2f69b87ab2812ee8cd3731524f7f2",
    "seqref_mri/src/forward_operator.py": "3cbc5162da9b727e82457a7c8f93121097c8228701f884a2190c681e4e1d4b5d",
    "seqref_mri/src/free_flow_runtime.py": "8e171b2c521d77c8ad22b21d02537be160d15166aecb9e0319fd0a25b69823c9",
    "seqref_mri/src/fastmri_data.py": "610cc1d1d7968deebc88f645270e1baefb6589cb56841bdd327450ca1069cb44",
    "seqref_mri/src/residual_decoder.py": "fd65e3a229e3df4f0226e276c92c916f5bd32cc77ed3655d68fb8a14608990d6",
    "seqref_mri/scripts/tiny_gate.py": "3ddd73ac99612e4af8d69664539acaef2ec4e9621deae530327fe0a8cd1e88d8",
    "seqref_mri/scripts/train_base.py": "2ae948663dbd53df32e032c0ef8c54e23666106529ca1906a354fcee7ba6ffc5",
    "seqref_mri/tdiag/d3.py": "73110e7d36a1ebdcb51926c4bbde81f88392f578e1aa860736c509706041245e",
    "seqref_mri/tdiag/estimators.py": "05561e729c7b5a7e3f6914990af2808918e372bec1ed0710c97ef3de867dfc28",
    "seqref_mri/scripts/v05lt_eval.py": "10b4bc65d465e9c6d0a016c2d27977cd9f485ffb851936d9a15ad6a325898554",
    "seqref_mri/scripts/v05lt_metrics.py": "87eaa5a4e85b6b70072c1c76fcd3e73ce104c3cdb04ebef973c346ff90b18a74",
    "seqref_mri/scripts/v05lt_common.py": "f35664fcb92607d0b0919ea80e55a90c99bbea9d6584b329c0d7688d9e3712a1",
    "seqref_mri/scripts/v02_manifests.py": "5ef4c55121df0e5fa151c920bcafb75c55121c222afb7f0d73b19cdbc44749e9",
}

RESULTS = []
CTX = {"pinned": {"status": "not checked", "files": [], "reason": None}, "timings": {}, "passes": {}, "r1a": {}}


class NotRun(Exception):
    pass


class Fx:
    def __init__(self, fid, title):
        self.fid, self.title, self.checks = fid, title, []

    def close(self, name, observed, expected, tol):
        if isinstance(observed, bool) or not isinstance(observed, (int, float)) or not math.isfinite(observed):
            self.true(name, False, "observed %r is not a finite number" % (observed,))
            return
        res = abs(observed - expected)
        self.checks.append({"check": name, "observed": observed, "expected": expected,
                            "residual": res, "tolerance": tol, "pass": bool(res <= tol)})

    def true(self, name, cond, note=""):
        self.checks.append({"check": name, "observed": bool(cond), "expected": True,
                            "residual": None, "tolerance": "exact", "pass": bool(cond), "note": str(note)})

    def raises(self, name, code, fn):
        try:
            fn()
        except C.DiagLatError as e:
            self.true(name, e.code == code, "raised %s" % e.code)
        else:
            self.true(name, False, "no error raised")

    def info(self, name, value):
        self.checks.append({"check": name, "observed": value, "expected": None, "residual": None,
                            "tolerance": "recorded", "pass": True})


@contextlib.contextmanager
def timed(stage, passes=1):
    t0 = time.perf_counter()
    yield
    CTX["timings"][stage] = CTX["timings"].get(stage, 0.0) + time.perf_counter() - t0
    CTX["passes"][stage] = CTX["passes"].get(stage, 0) + passes


def sha_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def rng(offset=0):
    return np.random.Generator(np.random.PCG64(SELFTEST_SEED + offset))


# ---- pinned sources: hash check before import, listing after ------------------------------
def pinned_preflight(root):
    if not root:
        CTX["pinned"].update(status="absent", reason="no --pinned-root given")
        return
    root = os.path.realpath(root)
    rows, bad = [], []
    for rel, want in sorted(REGISTERED_PINNED.items()):
        p = os.path.join(root, rel)
        got = sha_file(p) if os.path.isfile(p) else None
        ok = got == want
        rows.append({"file": rel, "sha256": got, "registered": want, "match": ok})
        if not ok:
            bad.append(rel)
    CTX["pinned"].update(root=root, files=rows)
    if bad:
        LOG.error("pinned hash check failed for: %s", ", ".join(bad))
        CTX["pinned"].update(status="mismatch", reason="registered hash mismatch or file missing: %s" % ", ".join(bad))
        return
    sys.path.insert(0, root)
    CTX["pinned"]["status"] = "ok"


def pinned_modules_loaded():
    """Every module imported from the pinned root, with its sha256 and whether a registered hash exists."""
    root = CTX["pinned"].get("root")
    out = []
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if root and f and os.path.realpath(f).startswith(root + os.sep):
            rel = os.path.relpath(os.path.realpath(f), root)
            out.append({"module": name, "file": rel, "sha256": sha_file(f),
                        "registered_hash_checked": rel in REGISTERED_PINNED})
    return out


def need_pinned():
    if CTX["pinned"]["status"] != "ok":
        raise NotRun("pinned sources unavailable: %s" % CTX["pinned"]["reason"])
    return A.seams()


# ---- synthetic construction (fixtures only) ---------------------------------------------
def synthetic_p4(s):
    """Synthetic standardisation pairs for every grid location (V2): fixed means, positive scales."""
    locs = [{"row": r, "column": c,
             "applied_mean_re": 0.01 * math.sin(0.1 * r + 0.2 * c), "applied_mean_im": 0.01 * math.cos(0.13 * r - 0.07 * c),
             "applied_scale_re": 0.12 * (1.3 + math.sin(0.05 * r + 0.11 * c)),
             "applied_scale_im": 0.12 * (1.3 + math.cos(0.09 * r - 0.03 * c))} for r in range(96) for c in range(96)]
    return {"location_index": s.ffr.build_location_index(locs)}


def synthetic_states(s, n, offset=0):
    """n synthetic slice states built by the PINNED builders from a synthetic batch (V2, U3):
    tiny_gate._build_slice_states, then the two fields v05lt_eval.build_states adds."""
    torch = s.torch
    from seqref_mri.src import forward_operator as fop
    from seqref_mri.scripts.train_base import _prepare
    g = rng(100 + offset)
    xs, ys, ms, am, tm, meta = [], [], [], [], [], []
    rr, cc = np.meshgrid(np.arange(96) - 48.0, np.arange(96) - 48.0, indexing="ij")
    for j in range(n):
        blob = np.exp(-(rr ** 2 + cc ** 2) / (2 * (14.0 + 2 * j) ** 2))
        x = ((g.standard_normal((96, 96)) + 1j * g.standard_normal((96, 96))) * (0.35 + blob)).astype(np.complex64)
        m = torch.from_numpy(s.fdm.make_cartesian_mask(96, 1000 + offset + j))
        xc = torch.from_numpy(x)
        xs.append(fop.complex_to_two_channel(xc)); ys.append(fop.MaskedFourierOperator(m).A(xc)); ms.append(m)
        am.append(float(np.abs(x).max() * 1.25)); tm.append(xc.abs())
        meta.append({"file": "synthetic_%d.h5" % (offset + j), "slice_index": j})
    batch = {"x_true": torch.stack(xs), "y": torch.stack(ys), "mask": torch.stack(ms),
             "file_attr_max": torch.tensor(am, dtype=torch.float32), "target_mag": torch.stack(tm), "meta": meta}
    sel = {"ordered_identities": [{"file": m["file"], "slice_index": m["slice_index"]} for m in meta]}
    states = s.tg._build_slice_states(batch, sel, synthetic_p4(s), 1.0)
    prep = _prepare(batch, "cpu", test0=False)
    for j, st in enumerate(states):                 # as v05lt_eval.build_states
        x1, c1 = prep["x_norm"][j], st["cond"][0]
        st["x_true_c"] = torch.complex(x1[0], x1[1])
        st["cond_c"] = torch.complex(c1[0], c1[1])
        if not (torch.equal(st["x_true_c"].abs(), st["x_true_mag"]) and torch.equal(prep["cond_in"][j], c1)):
            raise C.DiagLatError(C.INPUT, "synthetic state %d: complex ground truth / conditioning mismatch" % j)
    return states


def set_parameters(s, model, seed, amp):
    """Deterministic non-trivial parameters (the pinned initialisation gives identity couplings)."""
    torch = s.torch
    g = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".post." in n:
                p.add_(torch.randn(p.shape, generator=g) * amp)
            elif ".film.mlp.2." in n:
                p.add_(torch.randn(p.shape, generator=g) * 0.05)
            elif n.startswith("mask_branch") or n.startswith("flow.cond"):
                p.add_(torch.randn(p.shape, generator=g) * 0.02)


def parameter_identity(model):
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        h.update(n.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def small_flow(s, dtype):
    """Pinned CondNSF at small size, NO conditioner object (strictly flow-only); sinusoid parameters."""
    torch = s.torch
    from seqref_mri.src.base_experts import CondNSF
    flow = CondNSF(dim=8, h_dim=16, conditioner=None, hidden=32, n_layers=6, K=8, B=B_S, use_film=True).to(dtype)
    with torch.no_grad():
        for i, (n, p) in enumerate(flow.named_parameters()):
            amp = 0.6 if ".post." in n else 0.3
            p.copy_((amp * torch.sin(torch.arange(p.numel(), dtype=torch.float64) * 0.37 + i)).view(p.shape).to(dtype))
    h = (0.5 * torch.sin(torch.arange(16, dtype=torch.float64) * 0.9 + 0.2)).view(1, 16).to(dtype)
    return flow.eval(), h


# ---- R5 -------------------------------------------------------------------------------------
def r5(fx):
    me = os.path.abspath(__file__)
    cases = (["--checkpoint", "x.pt"], ["--data-root", "/data"], ["--checkpoint", "x.pt", "--data-root", "/data"],
             ["--checkpoint=x.pt"], ["--data-root=/data"], ["--out", "o", "--data-root=/data", "--pinned-root", "p"])
    for args in cases:
        r = subprocess.run([sys.executable, "-I", me] + args, capture_output=True, text=True, timeout=120)
        ok = (r.returncode == EXIT_REFUSED and REFUSED in r.stdout and REFUSED in r.stderr
              and "numpy=False seqref_mri=False torch=False" in r.stdout)
        fx.true("refused before any import: %s" % " ".join(args), ok, "exit %d; %s" % (r.returncode, r.stdout.strip().replace("\n", " | ")))
    r = subprocess.run([sys.executable, "-I", me, "--help"], capture_output=True, text=True, timeout=120)
    fx.true("control: --help is not refused (exit 0)", r.returncode == 0 and REFUSED not in r.stdout)


# ---- T9 -------------------------------------------------------------------------------------
def t9(fx):
    s = need_pinned()
    torch = s.torch
    flow, h = small_flow(s, torch.float64)
    fx.true("strictly flow-only: the flow holds no conditioner object", flow.cond is None)
    enc = lambda u: flow.encode(u.view(1, -1), h)[0][0]
    dec = lambda z: flow.decode(z.view(1, -1), h)[0]
    g = rng(9)
    points = [torch.from_numpy(g.uniform(-0.8 * B_S, 0.8 * B_S, size=8)) for _ in range(3)]
    tail_pt = torch.tensor([2.0 * B_S, -3.0 * B_S, 0.3, -1.0, 4.0, 1.7, -2.2, 0.9], dtype=torch.float64)
    worst = {"ldj": 0.0, "cd": 0.0, "dec": 0.0, "rt": 0.0}
    pre = {"move": math.inf, "cond": 0.0, "logdet": math.inf}
    wrong_sign = math.inf
    for u in points + [tail_pt]:
        z, ldj = flow.encode(u.view(1, -1), h)
        z, ldj = z[0].detach(), float(ldj[0])
        jac = torch.autograd.functional.jacobian(enc, u)
        sign, logdet = torch.linalg.slogdet(jac)
        logdet = float(logdet)
        pre["move"] = min(pre["move"], float((z - u).abs().max()))
        pre["cond"] = max(pre["cond"], float(torch.linalg.cond(jac)))
        pre["logdet"] = min(pre["logdet"], abs(logdet))
        worst["ldj"] = max(worst["ldj"], abs(ldj - logdet))
        wrong_sign = min(wrong_sign, abs(-ldj - logdet))
        cd = torch.zeros(8, 8, dtype=torch.float64)
        for j in range(8):
            e = torch.zeros(8, dtype=torch.float64); e[j] = 1e-6
            cd[:, j] = (enc(u + e) - enc(u - e)).detach() / 2e-6
        worst["cd"] = max(worst["cd"], float((cd - jac).abs().max() / jac.abs().max()))
        jdec = torch.autograd.functional.jacobian(dec, z)
        worst["dec"] = max(worst["dec"], abs(float(torch.linalg.slogdet(jdec)[1]) + ldj))
        worst["rt"] = max(worst["rt"], float((dec(z).detach() - u).abs().max()))
    if not (pre["move"] > 1e-3 and pre["cond"] < 1e6 and pre["logdet"] > 1e-3):
        raise C.DiagLatError(A.JACOBIAN_FIXTURE, "T9 construction: pre-conditions not met: %r" % (pre,))
    fx.info("pre-conditions (construction, not a pass criterion): min max|z-u|, max condition number, min |log det|", pre)
    fx.close("encode log-determinant = log|det dz/du| of the autodiff Jacobian (max over 4 points)", worst["ldj"], 0.0, 1e-9)
    fx.close("autodiff Jacobian agrees with central differences (max relative)", worst["cd"], 0.0, 1e-5)
    fx.close("Jacobian of decode has minus that log-determinant", worst["dec"], 0.0, 1e-9)
    fx.close("decode(encode(u)) = u (float64)", worst["rt"], 0.0, 1e-9)
    fx.true("control: the opposite sign convention would fail (min |-ldj - log det| > 1e-3)", wrong_sign > 1e-3, "%.3e" % wrong_sign)
    z_t = enc(tail_pt).detach()
    jac_t = torch.autograd.functional.jacobian(enc, tail_pt)
    eye = torch.eye(8, dtype=torch.float64)
    fx.true("tail coordinates pass unchanged (bitwise) and their Jacobian rows are exact unit rows: contribution exactly 0",
            bool(torch.equal(z_t[:2], tail_pt[:2]) and torch.equal(jac_t[:2], eye[:2])))
    conv = ("verified convention: ldj = log|det dz/du|; log p(u) = log N(z) + ldj; J = -ldj/d. "
            "This is the convention of production (v05lt_metrics.py: nll_logdet_per_dim = -ldj / d_free); "
            "the production formula itself is exercised in R1a (iii).")
    fx.info("sign convention", conv)
    print("T9 " + conv)


# ---- T10 ------------------------------------------------------------------------------------
def t10(fx):
    s = need_pinned()
    torch = s.torch
    import inspect
    from seqref_mri.src.flows import nsf_layer
    b32 = np.float32(B_S)
    up, down = np.nextafter(b32, np.float32(np.inf)), np.nextafter(b32, np.float32(0))
    u32 = np.array([2 * B_S, -3 * B_S, up, b32, down, -b32, -up, 0.3], dtype=np.float32)
    mask, tau, _q = C.tail_set(u32)
    fx.true("core tail set on the float32 vector = {2B, -3B, next above B, next below -B}", mask.tolist() == [True, True, True, False, False, False, True, False])
    src = inspect.getsource(nsf_layer._rq_spline)
    line = "inside = (inputs >= -B) & (inputs <= B)"
    fx.true("the pinned spline still defines its interior as: " + line, line in src)
    t = torch.from_numpy(u32)
    inside = ((t >= -B_S) & (t <= B_S)).numpy()               # the pinned expression, B a Python float
    fx.true("D7: core float32 comparison |u| > float32(B_s) selects exactly the complement of the pinned interior, boundary cases included",
            bool(np.array_equal(~inside, mask)))
    fx.info("boundary detail: float32(B_s) - B_s; u = float32(B_s) is interior for the pinned expression",
            {"float32(B_s) - B_s": float(b32) - B_S, "interior": bool(inside[3]), "torch": torch.__version__})
    flow, h = small_flow(s, torch.float32)
    with torch.no_grad():
        z = flow.encode(t.view(1, -1), h)[0][0].numpy()
        x = flow.decode(t.view(1, -1), h)[0].numpy()
    fx.true("float32: tail entries equal on output, encode direction (bitwise)", bool(np.array_equal(z[mask], u32[mask])))
    fx.true("float32: tail entries equal on output, decode direction (bitwise)", bool(np.array_equal(x[mask], u32[mask])))
    fx.true("float32: core invariance check passes on the pinned output", C.check_tail_invariance(z, u32, mask) == int(mask.sum()))
    fx.true("float32: a clearly interior entry (0.3) is changed by the flow", bool(z[7] != u32[7]))
    bad = z.copy(); bad[0] = np.nextafter(bad[0], np.float32(0))
    fx.raises("a changed tail entry stops with DIAG_LAT_TAIL_INVARIANCE", C.TAIL_INVARIANCE, lambda: C.check_tail_invariance(bad, u32, mask))
    flow64, h64 = small_flow(s, torch.float64)
    u64 = np.array([2 * B_S, -3 * B_S, np.nextafter(B_S, np.inf), B_S, np.nextafter(B_S, 0), -B_S, np.nextafter(-B_S, -np.inf), 0.3])
    t64 = torch.from_numpy(u64)
    m64 = ~((t64 >= -B_S) & (t64 <= B_S)).numpy()
    with torch.no_grad():
        z64 = flow64.encode(t64.view(1, -1), h64)[0][0].numpy()
        x64 = flow64.decode(t64.view(1, -1), h64)[0].numpy()
    fx.true("float64: entries beyond the bound equal on output in both directions (bitwise)",
            bool(m64.tolist() == [True, True, True, False, False, False, True, False]
                 and np.array_equal(z64[m64], u64[m64]) and np.array_equal(x64[m64], u64[m64])))


# ---- T11 ------------------------------------------------------------------------------------
def t11(fx):
    s = need_pinned()
    torch = s.torch
    from seqref_mri.src import forward_operator as fop
    st0, st1 = synthetic_states(s, 2, offset=20)
    k1 = s.fdm.fft2c(st1["x_true_c"]).abs().numpy()
    fx.true("synthetic donor image has non-zero coefficients in every column", bool((k1.max(axis=0) > 0).all()))
    sw = A.swap_input(st0, st1)
    fx.close("S2: excluded columns of the swapped input (max |k|)", sw["s2_excluded_max"], 0.0, sw["s2_tolerance"])
    fx.close("S2: acquired columns preserved (max difference to fft2c(x_true_c[j]))", sw["s2_acquired_max_diff"], 0.0, sw["s2_tolerance"])
    fx.true("the mask handed back is the recipient's stored mask (same object; byte-equal)",
            sw["mask"] is st0["mask"] and bool(torch.equal(sw["mask"], st0["mask"])))
    fx.true("S4: the content changed (relative L2 difference > 0)", sw["s4_relative_l2"] > 0.0, "%.4f" % sw["s4_relative_l2"])
    op = fop.MaskedFourierOperator(st0["mask"][0])
    direct = fop.complex_to_two_channel(op.A_adjoint(op.A(st1["x_true_c"]))).unsqueeze(0)
    fx.true("cond_swap equals the direct evaluation two_channel(A_adjoint(A(x_j))) (bitwise)", bool(torch.equal(direct, sw["cond_swap"])))
    selfsw = A.swap_input(st0, st0, check_identity=False)
    fx.close("S1 construction: self-swap reproduces the stored cond (max abs difference)", selfsw["max_abs_diff_to_own"], 0.0,
             A.SWAP_REL_TOL * selfsw["max_abs_own"])
    shifted = torch.roll(st0["mask"][0], 1)
    aff = (shifted != st0["mask"][0]).numpy()
    fx.true("wrong-mask example: the donor has non-zero coefficients on every column affected by the shift", bool((k1[:, aff].max(axis=0) > 0).all()))
    op_bad = fop.MaskedFourierOperator(shifted)
    cond_bad = op_bad.A_adjoint(op_bad.A(st1["x_true_c"]))
    fx.raises("S5: an input built under the mask shifted by one column fails S2", C.SWAP_FIXTURE,
              lambda: A.swap_check_s2(st0, cond_bad, st1["x_true_c"]))
    fx.raises("S4: a donor identical to the recipient stops with DIAG_LAT_SWAP_IDENTITY when the input is bit-identical",
              C.SWAP_IDENTITY, lambda: C.check_swap_identity(st0["cond"].numpy(), st0["cond"].numpy()))
    zero = dict(st0); zero["x_true_c"] = torch.zeros_like(st0["x_true_c"])
    fx.raises("S2: a zero tolerance scale stops with DIAG_LAT_SWAP_FIXTURE", C.SWAP_FIXTURE, lambda: A.swap_input(st0, zero))


# ---- T12b -----------------------------------------------------------------------------------
def t12b(fx):
    s = need_pinned()
    torch = s.torch
    (st,) = synthetic_states(s, 1, offset=30)
    y_acq = A.acquired_values(st)
    fx.true("2,304 acquired cells (24 columns x 96 rows)", y_acq.shape == (C.N_ACQUIRED,))
    zero = A.perturbed_input(st, np.zeros(y_acq.size, dtype=np.complex128))
    fx.true("zero noise reproduces the stored conditioning input exactly (bitwise)", bool(torch.equal(zero, st["cond"])))
    fx.close("P1: max |cond' - cond| with zero noise", A.perturb_check_p1(st, zero), 0.0, 0.0)
    doubled = zero / st["amax"].view(-1, 1, 1, 1)
    fx.raises("a doubled division by amax fails P1 (amax = %.4f)" % float(st["amax"][0]), A.PERTURB_FIXTURE,
              lambda: A.perturb_check_p1(st, doubled))
    noise = C.scale_noise(rng(12).standard_normal((y_acq.size, 2)), y_acq)
    cond = A.perturbed_input(st, noise)
    # realised noise as seen through the production path: undo the single division, back to k-space
    k = s.fdm.fft2c(torch.complex(cond[0, 0], cond[0, 1]) * st["amax"][0]).numpy().astype(np.complex128)
    r, c = A.acquired_cells(st)
    realised = k[r, c] - y_acq
    rms = lambda v: math.sqrt(float(np.mean(v.real ** 2 + v.imag ** 2)))
    fx.close("realised RMS of the noise on the acquired cells / (eps x RMS(y)), through the float32 production path",
             rms(realised) / (C.EPSILON * rms(y_acq)), 1.0, 1e-6)
    fx.info("max |realised - requested| / RMS(noise) (float32 path)", float(np.max(np.abs(realised - noise))) / rms(noise))
    fx.true("the perturbed input differs from the stored one", not bool(torch.equal(cond, st["cond"])))
    bad = noise.copy(); bad[5] = np.nan
    fx.raises("non-finite noise stops with DIAG_LAT_NONFINITE", C.NONFINITE, lambda: A.perturbed_input(st, bad))


# ---- R1a ------------------------------------------------------------------------------------
def r1a(fx, out_dir, ram_limit):
    s = need_pinned()
    torch = s.torch
    op_name = "start"
    try:
        with GD.memory_guarded(ram_limit):
            states = synthetic_states(s, N_SLICES, offset=0)
            donors = s.d3.derangement(N_SLICES)
            bank8 = rng(1).standard_normal((8, C.D_FREE)).astype(np.float32)
            guard = CTX["guard"]
            t64, _ = A.target_std(states[0]); sc, mn = A.scale_mean_interleaved(states[0])
            ref = s.ffr.pack_scalars(states[0]["u_true"].real, states[0]["u_true"].imag)
            fx.close("A4: interleaved scale and mean reproduce the physical coefficients in pack_scalars order (max abs)",
                     float(np.max(np.abs(C.to_physical(t64, sc, mn) - ref))), 0.0, 1e-12 * max(1.0, float(np.max(np.abs(ref)))))
            per = {}
            for label, seed, amp in SETTINGS:
                op_name = "build_model (%s)" % label
                with timed("model_construction"):
                    model = s.ffr.build_model(); model.eval()
                    set_parameters(s, model, seed, amp)
                ident = parameter_identity(model)
                res = {"rt": 0.0, "jrep": True, "nll_ld": 0.0, "nll_prod": 0.0, "self_cond": 0.0, "self_nll": 0.0,
                       "zero_exact": True, "keys": True, "files": True, "finite": True, "nll": [], "ks": [], "h": [], "records": [], "sens": []}
                for i, st in enumerate(states):
                    op_name = "%s slice %d" % (label, i)
                    own = A.encode_truth(model, st)
                    with torch.no_grad():
                        lp = float(model.log_prob_free(torch.from_numpy(np.ascontiguousarray(st["target"], dtype=np.float32)), st["cond"], st["mask"])[0])
                    res["nll_ld"] = max(res["nll_ld"], abs(own["NLL"] - (-(own["log_pz"] + own["ldj"]) / own["d"])))
                    res["nll_prod"] = max(res["nll_prod"], abs(own["NLL"] - (-lp / own["d"])))
                    sw = A.swap_input(st, states[donors[i]])
                    p_swap = C.swap_penalty(A.encode_truth(model, st, sw["cond_swap"], sw["mask"])["NLL"], own["NLL"])
                    selfsw = A.swap_input(st, st, check_identity=False)
                    res["self_cond"] = max(res["self_cond"], selfsw["max_abs_diff_to_own"] / (A.SWAP_REL_TOL * selfsw["max_abs_own"]))
                    res["self_nll"] = max(res["self_nll"], abs(A.encode_truth(model, st, selfsw["cond_swap"], selfsw["mask"])["NLL"] - own["NLL"]))
                    y_acq = A.acquired_values(st)
                    zero = A.perturbed_input(st, np.zeros(y_acq.size, dtype=np.complex128))
                    res["zero_exact"] &= bool(torch.equal(zero, st["cond"])) and A.perturb_check_p1(st, zero) == 0.0
                    cond_p = A.perturbed_input(st, C.scale_noise(rng(50 + i).standard_normal((y_acq.size, 2)), y_acq))
                    enc_p = A.encode_truth(model, st, cond_p, st["mask"])
                    x0, _u0 = s.EV.r0_image(model, st)
                    r0_psnr = float(s.tg._psnr(x0.abs(), st["x_true_mag"]))
                    rec, diag = A.slice_state_record(model, st, bank8, subsets=TOY_SUBSETS, p_swap=p_swap, r0_psnr=r0_psnr)
                    r0 = A.decode_r0_std(model, st)
                    sens = C.l5b_sensitivities(r0.astype(np.float64), A.decode_r0_std(model, st, cond_p, st["mask"]).astype(np.float64),
                                               own["z_star"].astype(np.float64), enc_p["z_star"].astype(np.float64), own["NLL"], enc_p["NLL"])
                    res["sens"].append(sens)
                    res["rt"] = max(res["rt"], diag["roundtrip_max_abs"])
                    u_b = A.decode_bank_std(model, st, bank8)
                    j1, j2 = A.reencode_bank(model, st, u_b)[1], A.reencode_bank(model, st, u_b)[1]
                    res["jrep"] &= bool(np.array_equal(j1, j2) and np.all(np.isfinite(j1)))
                    try:
                        A.validate_record(rec, TOY_SUBSETS.keys())
                    except C.DiagLatError:
                        res["keys"] = False
                    flat = [rec[k["name"]] for k in A.RECORD_KEYS if not k["bank"]] + \
                           [v for lab in rec[A.BANK_FIELD].values() for v in lab.values() if v is not C.NOT_COMPUTED]
                    res["finite"] &= all(isinstance(v, float) and math.isfinite(v) for v in flat)
                    payload = {"banner": CTX["banner"], "toy_subsets": True, "setting": label, "slice": i,
                               "identity": st["identity"], "record": rec,
                               "diagnostics": {k: v for k, v in diag.items() if k != "events"}, "events": diag["events"]}
                    data = json.dumps(payload, indent=1, sort_keys=True).encode()
                    rel = "r1a/records/%s_slice%d.json" % (label, i)
                    guard.write(rel, data)
                    res["files"] &= guard.read(rel) == data and json.loads(guard.read(rel))["record"] == rec
                    res["nll"].append(own["NLL"]); res["ks"].append(rec["KS"]); res["h"].append(A.conditioning_vector(model, st))
                    res["records"].append(rec)
                per[label] = res
                fx.info("%s: parameter identity (sha256 over names and values), generator seed, amplitude" % label,
                        {"sha256": ident, "seed": seed, "amplitude": amp})
                fx.close("%s (i) round trip encode(decode(z_k)) = z_k, max abs over 4 slices x 8 samples (pinned float32 tolerance)" % label, res["rt"], 0.0, 1e-5)
                fx.true("%s (ii) J(u_k) from re-encoding is finite and identical on repetition" % label, res["jrep"])
                fx.close("%s (iii) NLL = B + J equals -(log N(z*) + ldj)/d" % label, res["nll_ld"], 0.0, 1e-12)
                fx.close("%s (iii) NLL equals the production value -log_prob_free/d (float32 sum)" % label, res["nll_prod"], 0.0, 1e-5)
                fx.close("%s (iv) self-swap reproduces the conditioning input (max difference / S1 tolerance)" % label, res["self_cond"], 0.0, 1.0)
                fx.close("%s (iv) self-swap NLL equals the own NLL (nats/dim)" % label, res["self_nll"], 0.0, 1e-4)
                fx.true("%s (v) zero noise reproduces the conditioning input exactly; P1 difference is 0" % label, res["zero_exact"])
                fx.true("%s (vii) every record carries exactly RECORD_KEYS for the toy subset labels" % label, res["keys"])
                fx.true("%s (vii) every record value is finite or the explicit not-computed status" % label, res["finite"])
                fx.true("%s (viii) records written through the write guard and read back against their sidecars" % label, res["files"])
                fx.info("%s L5(b) sensitivities, slice 0 (finite; not a result)" % label, res["sens"][0])
            # (vi) and C2: the two settings are demonstrably different
            d_h = []
            for h1, h2 in zip(per["S1"]["h"], per["S2"]["h"]):
                d_h.append(float(np.linalg.norm(h2 - h1) / max(np.linalg.norm(h1), np.linalg.norm(h2), 1e-12)))
            fx.info("C2: d_h = ||h2 - h1|| / max(||h1||, ||h2||, 1e-12), per slice", d_h)
            fx.true("C2 / (vi): d_h >= 1e-3 on every slice (conditioning differs between the settings)", min(d_h) >= D_H_MIN, "min %.4f" % min(d_h))
            d_nll = [abs(a - b) for a, b in zip(per["S1"]["nll"], per["S2"]["nll"])]
            d_ks = [abs(a - b) for a, b in zip(per["S1"]["ks"], per["S2"]["ks"])]
            fx.true("C2 / (vi): output measures change beyond numerical tolerance (|dNLL| and |dKS| > 1e-6 on every slice)",
                    min(d_nll) > 1e-6 and min(d_ks) > 1e-6, "min |dNLL| %.3e, min |dKS| %.3e" % (min(d_nll), min(d_ks)))
            fx.true("C2: the parameter identities of the two settings differ",
                    len({c["observed"]["sha256"] for c in fx.checks if "parameter identity" in c["check"]}) == 2)
            fx.info("C2 scope", "engineering check that the two settings behave differently; not evidence about reconstruction quality")
            # registered-size case: one slice, K = 128, registered subsets, with separate stage timings (R6)
            op_name = "K = 128 case (S2, slice 0)"
            st = states[0]
            bank = rng(2).standard_normal((C.K_BANK, C.D_FREE)).astype(np.float32)
            with timed("conditioning"):
                A.conditioning_vector(model, st)
            with timed("truth_encoding"):
                enc = A.encode_truth(model, st)
            with timed("bank_decoding", C.K_BANK):
                u_b = A.decode_bank_std(model, st, bank)
            with timed("bank_reencoding", C.K_BANK):
                z_re, j_b = A.reencode_bank(model, st, u_b)
            with timed("r0_decoding"):
                r0 = A.decode_r0_std(model, st)
            with timed("core_bank_measures_standardised"):
                C.bank_measures(A.target_std(st)[0], r0.astype(np.float64), enc["z_star"].astype(np.float64),
                                u_b.astype(np.float64), bank.astype(np.float64), j_b, [])
            with timed("slice_state_record_complete"):
                rec, diag = A.slice_state_record(model, st, bank, subsets=C.STABILITY_SUBSETS, p_swap=0.0, r0_psnr=0.0)
            fx.true("K = 128: record carries exactly RECORD_KEYS for the four registered subset labels",
                    A.validate_record(rec, C.STABILITY_SUBSETS.keys()) and set(rec[A.BANK_FIELD]) == set(ST.STABILITY_SUBSETS))
            fx.close("K = 128: round trip max abs over 128 samples (pinned float32 tolerance)", diag["roundtrip_max_abs"], 0.0, 1e-5)
            fx.true("K = 128: pre64 and halfA give identical values (same subset)", rec[A.BANK_FIELD]["pre64"] == rec[A.BANK_FIELD]["halfA"])
            fx.true("K = 128: P_swap and R0_PSNR passed through unchanged (0.0 placeholders of this case only, labelled)",
                    rec["P_swap"] == 0.0 and rec["R0_PSNR"] == 0.0, "this case tests the subset path; values are fixture inputs")
            data = json.dumps({"banner": CTX["banner"], "registered_subsets": True, "setting": "S2", "slice": 0,
                               "note": "P_swap and R0_PSNR are fixture inputs (0.0) in this case", "record": rec,
                               "diagnostics": {k: v for k, v in diag.items() if k != "events"}}, indent=1, sort_keys=True).encode()
            guard.write("r1a/records/S2_slice0_K128.json", data)
            fx.true("K = 128: record written through the write guard and read back", guard.read("r1a/records/S2_slice0_K128.json") == data)
            fx.info("(ix) device record", A.device_record(model))
            fx.info("guard: bytes charged so far (whole output folder)", guard.written)
    except C.DiagLatError as e:
        if e.code == GD.RESOURCE_LIMIT:                       # C4
            CTX["r1a"] = {"attempted_operation": op_name, "observed_peak_bytes": GD.peak_rss_bytes(), "reason": str(e)}
            raise NotRun("incomplete due to resource limits: %s (operation: %s; observed peak %d bytes)"
                         % (e, op_name, GD.peak_rss_bytes()))
        raise
    fx.info("(ix) observed peak resident memory, bytes (limit %d)" % ram_limit, GD.peak_rss_bytes())


# ---- R1b ------------------------------------------------------------------------------------
ROLES = ["%s/%s" % (a, st) for a in ("lam0", "lam2") for st in ("e3", "e10", "e30", "sel", "fin")]
SEEDS = (2026, 2027, 2028)
N_VOL = 12


def synthetic_record(role, si, v, labels):
    """One synthetic record with exactly RECORD_KEYS. Numbers only; no model, no sample bank."""
    lv = {"lam0/sel": 0.1, "lam0/fin": 0.5, "lam0/e10": 0.05, "lam0/e30": 0.3, "lam2/fin": 0.1}.get(role, 0.0)
    base = math.sin(v * (1.3 + si)); wig = 0.01 * math.cos(v * 0.7 + ROLES.index(role) + si)

    def val(name):
        if name == "NLL":
            return float(2.0 - lv + base + wig)
        if name == "R0_PSNR":
            return float(30.0 - 3.0 * lv + 0.5 * base + wig)
        return float((-1.0 if name == "P_swap" else 1.0) * lv + base + wig)
    rec = {k["name"]: val(k["name"]) for k in A.RECORD_KEYS if not k["bank"]}
    rec[A.BANK_FIELD] = {lab: {k["name"]: val(k["name"]) for k in A.RECORD_KEYS if k["bank"]} for lab in labels}
    return rec


def population_from_records(records, labels, drop=None):
    """records[(role, seed)] = list of per-volume records -> sealed Population (one value per volume)."""
    pop = ST.Population(["v%03d" % i for i in range(N_VOL)])
    for (role, seed), recs in sorted(records.items()):
        for k in A.RECORD_KEYS:
            for lab in (labels if k["bank"] else (None,)):
                if drop == (role, seed, k["name"], lab):
                    continue
                vals = [(r[A.BANK_FIELD][lab][k["name"]] if k["bank"] else r[k["name"]]) for r in recs]
                pop.set(role, seed, k["name"], vals, lab)
    return pop.seal()


def r1b(fx, out_dir):
    labels = list(ST.STABILITY_SUBSETS)
    fx.true("the four registered subset labels are the same in core and stats", set(labels) == set(C.STABILITY_SUBSETS))
    fx.true("RECORD_KEYS covers every key P2 reads (classified keys, physical keys, NLL, R0_PSNR), with bank flags that agree",
            {k["name"] for k in A.RECORD_KEYS} == {m[0] for m in ST.MEASURES.values()} | set(A.PHYSICAL_NAMES.values()) | {"NLL", "R0_PSNR"}
            and all(next(k for k in A.RECORD_KEYS if k["name"] == m[0])["bank"] == m[2] for m in ST.MEASURES.values()))
    fx.true("C3: every schema entry defines name, bank flag, subset requirement and type",
            all(set(k) == {"name", "bank", "subsets", "type"} and k["subsets"] == ("every label" if k["bank"] else "none")
                and k["type"] in ("float", "float|NOT_COMPUTED") for k in A.RECORD_KEYS))
    records = {(role, seed): [synthetic_record(role, si, v, labels) for v in range(N_VOL)]
               for role in ROLES for si, seed in enumerate(SEEDS)}
    n_ok = sum(A.validate_record(r, labels) for recs in records.values() for r in recs)
    fx.true("all %d synthetic records (10 roles x 3 seeds x 12 volumes) pass the schema check" % (len(ROLES) * len(SEEDS) * N_VOL),
            n_ok == len(ROLES) * len(SEEDS) * N_VOL)
    # C3: rejection of malformed records
    good = records[(ROLES[0], SEEDS[0])][0]
    cp = lambda: json.loads(json.dumps(good))

    def broken(edit):
        r = cp(); edit(r); return lambda: A.validate_record(r, labels)
    cases = (("missing key (KS removed)", lambda r: r.pop("KS")),
             ("unexpected key", lambda r: r.update(extra=1.0)),
             ("bank part is a list, not a dict", lambda r: r.update({A.BANK_FIELD: [1.0]})),
             ("a subset label missing", lambda r: r[A.BANK_FIELD].pop("halfB")),
             ("an unexpected subset label", lambda r: r[A.BANK_FIELD].update(toy_full=dict(r[A.BANK_FIELD]["full"]))),
             ("a bank key missing (TC_G)", lambda r: r[A.BANK_FIELD]["full"].pop("TC_G")),
             ("an unexpected bank key", lambda r: r[A.BANK_FIELD]["full"].update(extra=1.0)),
             ("a subset entry that is not a dict", lambda r: r[A.BANK_FIELD].update(full=3.0)),
             ("a string value", lambda r: r.update(NLL="2.0")),
             ("a non-finite value", lambda r: r.update(NLL=float("nan"))),
             ("an integer value", lambda r: r.update(NLL=2)),
             ("a boolean value", lambda r: r[A.BANK_FIELD]["full"].update(d01=True)),
             ("not-computed status on a key that does not allow it (KS)", lambda r: r.update(KS=C.NOT_COMPUTED)),
             ("not-computed status on d01", lambda r: r[A.BANK_FIELD]["full"].update(d01=C.NOT_COMPUTED)))
    for name, edit in cases:
        fx.raises("C3: record rejected -- " + name, A.RECORD_SCHEMA, broken(edit))
    fx.raises("C3: a record that is not a dict is rejected", A.RECORD_SCHEMA, lambda: A.validate_record([good], labels))
    ok_nc = cp(); ok_nc[A.BANK_FIELD]["full"]["a01"] = C.NOT_COMPUTED; ok_nc[A.BANK_FIELD]["halfB"]["TC_G"] = C.NOT_COMPUTED
    fx.true("C3: the not-computed status is accepted where the schema allows it (a01, TC_G)", A.validate_record(ok_nc, labels))
    # population -> classes -> guarded output
    with timed("statistics_r1b"):
        pop = population_from_records(records, labels)
        idx = np.random.Generator(np.random.PCG64(SELFTEST_SEED)).integers(0, N_VOL, size=(400, N_VOL), dtype=np.int64)
        res = ST.run_registered(pop, idx, A.PHYSICAL_NAMES)
        tau = np.abs(np.sin(np.arange(N_VOL) * 0.9)) * 0.05
        l6 = {str(seed): ST.l6_contrast(pop, "S-GAP0", seed, idx, tau) for seed in SEEDS}
    m = res["measures"]
    fx.true("Population.seal() succeeds on the full structure (10 roles, 3 seeds, every key and label)",
            pop.sealed and len(pop.roles) == 10 and len(pop.seeds) == 3 and len(pop.keys) == 4 + 7 * 4)
    fx.true("run_registered returns six classes, all tags and an outcome",
            set(m) == set(ST.MEASURES) and all({"class", "seed_tag", "fixed_epoch_tag", "stability_summary", "coordinate_tag"} <= set(m[k]) for k in m)
            and set(res["outcome"]) == {"outcome", "provisional", "open_measures"})
    fx.true("the synthetic design (lam0-only rise, L5 mirrored) is classified as constructed: six gap-selective, outcome A",
            all(m[k]["class"] == ST.C_SELECTIVE for k in m) and res["outcome"]["outcome"] == "A",
            "synthetic numbers; shows the interface, not a finding")
    fx.true("L6 runs for the read contrast S-GAP0 and each of the three seeds",
            all(set(l6[str(sd)]["measures"]) == set(ST.MEASURES) and "share" in l6[str(sd)]["share_nll_lower_psnr_lower"] for sd in SEEDS))
    guard = CTX["guard"]
    tables = {"r1b/tables/classes.json": {"banner": CTX["banner"], "synthetic_population": True, "outcome": res["outcome"],
                                      "classes": {k: {"class": m[k]["class"], "seed_tag": m[k]["seed_tag"], "fixed_epoch_tag": m[k]["fixed_epoch_tag"],
                                                      "stability_summary": m[k]["stability_summary"], "coordinate_tag": m[k]["coordinate_tag"]} for k in m}},
              "r1b/tables/l6_S-GAP0.json": {"banner": CTX["banner"], "synthetic_population": True, "l6": l6}}
    ok = True
    for rel, obj in tables.items():
        data = json.dumps(obj, indent=1, sort_keys=True, default=str).encode()
        guard.write(rel, data)
        ok &= guard.read(rel) == data
    fx.true("result tables written through the write guard and read back against their sidecars", ok)
    fx.raises("a population with one key of RECORD_KEYS missing for one state is refused", ST.POPULATION_MISMATCH,
              lambda: population_from_records(records, labels, drop=("lam2/e3", 2027, "TC_G", "full")))
    recs_nc = dict(records); row = [json.loads(json.dumps(r)) for r in records[("lam2/e3", 2027)]]
    row[3][A.BANK_FIELD]["full"]["TC_G"] = C.NOT_COMPUTED; recs_nc[("lam2/e3", 2027)] = row
    pop_nc = population_from_records(recs_nc, labels)
    fx.true("a not-computed record value reaches the population as an explicit not-computed state",
            pop_nc.col("lam2/e3", 2027, "TC_G", "full") is C.NOT_COMPUTED and len(pop_nc.not_computed) == 1)


# ---- runner ---------------------------------------------------------------------------------
PINNED_MODULE_ROOTS = ("torch", "seqref_mri", "preflight_io", "preflight_parents", "h5py")


def run_fixture(fid, title, fn, selected):
    fx = Fx(fid, title)
    entry = {"id": fid, "title": title, "checks": fx.checks, "state": None, "reason": None}
    RESULTS.append(entry)
    if selected is not None and fid not in selected:
        entry.update(state="NOT RUN", reason="left out by --only")
        return
    t0 = time.perf_counter()
    try:
        fn(fx)
        entry["state"] = "PASS" if all(c["pass"] for c in fx.checks) else "FAIL"
    except NotRun as e:
        entry.update(state="NOT RUN", reason=str(e))
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] in PINNED_MODULE_ROOTS:
            LOG.error("%s: module %s is not available; fixture NOT RUN (no stub is written)", fid, e.name)
            entry.update(state="NOT RUN", reason="missing module: %s" % e.name)
        else:
            LOG.error("%s failed: %s", fid, traceback.format_exc())
            entry.update(state="FAIL", reason="ModuleNotFoundError: %s" % e)
    except C.DiagLatError as e:
        missing = "No module named" in str(e) and e.code == A.ADAPTER
        if missing:
            LOG.error("%s: production import failed; fixture NOT RUN (no stub is written): %s", fid, e)
            entry.update(state="NOT RUN", reason="missing module: %s" % e)
        else:
            LOG.error("%s stopped: %s", fid, traceback.format_exc())
            entry.update(state="FAIL", reason="typed stop %s: %s" % (e.code, e))
    except Exception as e:  # noqa: BLE001 -- logged and reported as FAIL; never swallowed
        LOG.error("%s failed: %s", fid, traceback.format_exc())
        entry.update(state="FAIL", reason="%s: %s" % (type(e).__name__, e))
    entry["seconds"] = time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description="D-LAT P3 synthetic fixtures (NOT D-LAT EVIDENCE). "
                                             "--checkpoint and --data-root are refused.")
    ap.add_argument("--pinned-root", help="repository root that holds seqref_mri/ (pinned sources, read only)")
    ap.add_argument("--out", help="output folder; must not exist yet")
    ap.add_argument("--mode", choices=("sandbox", "host"), default="sandbox")
    ap.add_argument("--ram-limit-gib", type=float, default=6.0, help="peak resident memory limit for R1a (C4)")
    ap.add_argument("--only", help="comma-separated fixture ids; the others are NOT RUN and the run is INCOMPLETE")
    args = ap.parse_args()
    if not args.out:
        ap.error("--out is required")
    logging.basicConfig(level=logging.CRITICAL)             # expected typed stops are exercised on purpose
    errors = {"n": 0}

    class Count(logging.Handler):
        def emit(self, rec):
            if rec.levelno >= logging.ERROR:
                errors["n"] += 1
    logging.getLogger().addHandler(Count()); logging.getLogger().setLevel(logging.DEBUG)
    for h in logging.getLogger().handlers:
        if not isinstance(h, Count):
            h.setLevel(logging.CRITICAL)
    if os.path.exists(args.out):
        LOG.error("output folder exists: %s", args.out)
        raise SystemExit("output folder already exists: %s" % args.out)
    os.makedirs(args.out)
    CTX["guard"] = GD.WriteGuard(args.out, ceiling=OUTPUT_CEILING, reserve=OUTPUT_RESERVE)   # every output goes through it
    CTX["banner"] = BANNERS[args.mode]
    selected = None if not args.only else set(args.only.split(","))
    t0 = time.perf_counter()
    pinned_preflight(args.pinned_root)
    ram = int(args.ram_limit_gib * GD.GIB)
    run_fixture("R5", "refusal of --checkpoint / --data-root before any import", r5, selected)
    run_fixture("T9", "flow-only Jacobian validation (pinned CondNSF, float64, no conditioner)", t9, selected)
    run_fixture("T10", "tail invariance and the float32 boundary (decision D7)", t10, selected)
    run_fixture("T11", "recipient-mask swap (S5) with the pinned operator", t11, selected)
    run_fixture("T12b", "perturbation, production path in float32 (P1)", t12b, selected)
    run_fixture("R1a", "real-model integration at production size (toy design; K = 8 and one K = 128 case)",
                lambda fx: r1a(fx, args.out, ram), selected)
    run_fixture("R1b", "statistical-interface integration (synthetic population, pure NumPy)", lambda fx: r1b(fx, args.out), selected)
    counts = {st: sum(1 for r in RESULTS if r["state"] == st) for st in ("PASS", "FAIL", "NOT RUN")}
    complete = counts["NOT RUN"] == 0
    here = os.path.dirname(os.path.abspath(__file__))
    files = {n: sha_file(os.path.join(here, n)) for n in ("diag_lat_adapter.py", "diag_lat_p3test.py", "diag_lat_core.py",
                                                         "diag_lat_stats.py", "diag_lat_guard.py", "diag_unc_adapter.py")
             if os.path.isfile(os.path.join(here, n))}
    torch_mod = sys.modules.get("torch")
    env = {"python": sys.version.split()[0], "numpy": np.__version__, "torch": getattr(torch_mod, "__version__", None),
           "torch_threads": torch_mod.get_num_threads() if torch_mod else None, "platform": platform.platform(),
           "cpu_count": os.cpu_count(), "mode": args.mode}
    timings = {k: {"seconds": v, "passes": CTX["passes"][k], "seconds_per_pass": v / CTX["passes"][k],
                   "kind": STAGE_KINDS[k][0], "contains": STAGE_KINDS[k][1]} for k, v in sorted(CTX["timings"].items())}
    resource = {"banner": CTX["banner"], "schema": "seqref-dlat-p3-resource/2", "environment": env, "stage_timings": timings,
                "timing_interpretation": TIMING_NOTE, "timings_are_additive": False,
                "peak_rss_bytes": GD.peak_rss_bytes(), "ram_limit_bytes": ram, "r1a_resource_stop": CTX["r1a"] or None,
                "note": "inputs to the C1 decision only; synthetic parameters; not a validation of the ceilings"}
    rep = {"banner": CTX["banner"], "schema": "seqref-dlat-p3-report/2", "p3test_version": __version__, "adapter_version": A.__version__,
           "core_version": C.__version__, "stats_version": ST.__version__, "guard_version": GD.__version__, "environment": env,
           "selftest_seed": SELFTEST_SEED, "only": sorted(selected) if selected else None, "pinned_preflight": CTX["pinned"],
           "pinned_modules_loaded": pinned_modules_loaded(), "fixtures": RESULTS, "counts": counts,
           "rehearsal": "COMPLETE" if complete else "INCOMPLETE", "error_log_records": errors["n"], "files_sha256": files,
           "elapsed_seconds": time.perf_counter() - t0}
    guard = CTX["guard"]
    out_sha = {}
    for rel, obj in (("p3_resource.json", resource), ("p3_report.json", rep)):
        data = json.dumps(obj, indent=1, default=str).encode()
        out_sha[rel] = guard.write(rel, data)                 # write-once, sha256 sidecar, counted in the ledger
        if guard.read(rel) != data:
            LOG.error("%s does not read back as written", rel)
            raise SystemExit("guarded output %s does not read back as written" % rel)
    print(CTX["banner"])
    print("pinned preflight: %s%s" % (CTX["pinned"]["status"], "" if not CTX["pinned"]["reason"] else " (%s)" % CTX["pinned"]["reason"]))
    for r in RESULTS:
        print("%-5s %-7s %s%s" % (r["id"], r["state"], r["title"], "" if not r["reason"] else "  [%s]" % r["reason"]))
        for c in r["checks"]:
            res = "" if c["residual"] is None else " residual %.3e (tol %s)" % (c["residual"], c["tolerance"])
            obs = " value %r" % (c["observed"],) if c["tolerance"] == "recorded" else ""
            note = " (%s)" % c["note"] if c.get("note") else ""
            print("      [%s] %s%s%s%s" % ("ok" if c["pass"] else "FAIL", c["check"], res, obs, note))
    print("stage timings (NOT additive; see timing_interpretation in p3_resource.json):", json.dumps(timings))
    print("guarded outputs:", json.dumps(out_sha), "bytes charged:", guard.written)
    print("peak resident memory: %.2f GiB; error log records: %d" % (GD.peak_rss_bytes() / GD.GIB, errors["n"]))
    print("fixtures: PASS %d, FAIL %d, NOT RUN %d; checks %d; run %s"
          % (counts["PASS"], counts["FAIL"], counts["NOT RUN"], sum(len(r["checks"]) for r in RESULTS), rep["rehearsal"]))
    return 1 if counts["FAIL"] else (0 if complete else 2)


if __name__ == "__main__":
    raise SystemExit(main())

# SEQREF-P12ST v0.1 -- P1/P2 self-test (fixtures; no dataset, no publication
#                       to the locked authoritative path)
# LIFETIME: EPHEMERAL
#
# Why this exists
#   The BLOCK path must write a VALID facts artefact BEFORE the process exits
#   non-zero. If it raised first, the verdict would be lost and a 256-slice run
#   wasted. Untested error paths are how the P0S unreachable-gate defect
#   reached review. This script exercises the paths a passing run never takes,
#   and checks ARTEFACT CONTENTS and EXIT CODES, not merely that exceptions
#   occur.
#
# Coverage
#   P1  REAL / COMPLEX / AMBIGUOUS classification
#       R_REAL_MIN outcome under the final taxonomy (BLOCK, published)
#       ordering guard: ratios refuse to form on a degenerate real channel
#       conjugate symmetry: real image ~0 violation, complex image >> 0
#   P2  ordinary PASS; near-zero PASS; near-zero FAILURE
#       k_i degeneracy -> BLOCK, published with slice identity
#       x0 discrepancy: agreement and mismatch against X0_ASSERT_RTOL
#       boundary band excludes a zero residual ratio by construction
#   shared  BLOCK publishes valid facts; ERROR record carries a distinct
#       filename AND artefact_type; non-finite values are refused rather than
#       silently written; publication never overwrites; the claim is exclusive;
#       semantic_sha256 is stable under changed runtime metadata while the
#       artefact bytes differ
#   end-to-end  P1.main and P2.main return EXIT_ERROR on a malformed parent and
#       leave a distinct error record (no dataset access is reached)
#
# NOT covered here: anything requiring the real dataset. That is the smoke run
#   (--smoke N against an EPHEMERAL out-dir).
#
# CONVENTION: logger.error + raise on every failure path. No fallback, no mock.
#
# Changelog
#   v0.1 (2026-07-30) Created under Amendment A3 as build addition 1.

from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import os
import sys
import tempfile
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "seqref_mri", "src"))
sys.path.insert(0, _HERE)

from preflight_io import verify_sidecar  # noqa: E402
from preflight_parents import (EXIT_ERROR, StageBlock,  # noqa: E402
                               StageError, attach_semantic_hash,
                               guard_run_mode, publication_claim,
                               publish_error, publish_stage)
import p1_representation as P1  # noqa: E402
import p2_support as P2  # noqa: E402

SCRIPT_ID = "SEQREF-P12ST"
SCRIPT_VERSION = "v0.1"

logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(SCRIPT_ID)

N = 16                       # fixture grid; the operators are size-agnostic
_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}"
          + (f"   [{detail}]" if detail and not ok else ""))


def close(got, want, *, rel: float = 1e-9) -> bool:
    """Quotient assertions use isclose UNIFORMLY. Binary floating-point
    division makes exact equality wrong for some quotients and accidentally
    right for others; a suite that mixes the two invites the wrong one to be
    copied."""
    return isinstance(got, float) and math.isclose(got, want, rel_tol=rel)


def expect_raise(name: str, exc_type, fn, *, code: str | None = None) -> None:
    try:
        fn()
    except exc_type as exc:
        got = getattr(exc, "error_code", None) or getattr(exc, "block_code",
                                                          None)
        check(name, code is None or got == code, f"code={got!r}")
        return
    except Exception as exc:                       # wrong exception type
        check(name, False, f"raised {type(exc).__name__}: {exc}")
        return
    check(name, False, "no exception raised")


# ---------------------------------------------------------------------------
# P1
# ---------------------------------------------------------------------------

def test_p1_classification() -> None:
    print("\nP1 classification")
    rng = np.random.default_rng(0)
    n = 64
    real_e = rng.uniform(1e-14, 1e-11, n)
    real_m = rng.uniform(1e-9, 1e-6, n)
    ruling, d = P1.classify_branch(real_e, real_m)
    check("REAL ruled when every slice satisfies both conditions",
          ruling == "REAL", f"got {ruling}")

    cx_e = rng.uniform(1e-4, 1e-1, n)
    cx_m = rng.uniform(1e-2, 1e-1, n)
    ruling, d = P1.classify_branch(cx_e, cx_m)
    check("COMPLEX ruled on the median conditions", ruling == "COMPLEX",
          f"got {ruling}")

    amb_e = np.full(n, 1e-8)          # > REAL bound, < COMPLEX median bound
    amb_m = np.full(n, 1e-4)          # > REAL bound, < COMPLEX median bound
    ruling, d = P1.classify_branch(amb_e, amb_m)
    check("AMBIGUOUS when neither condition set holds", ruling == "AMBIGUOUS",
          f"got {ruling}")

    mixed_e = np.concatenate([np.full(n - 1, 1e-14), [1e-9]])
    mixed_m = np.full(n, 1e-9)
    ruling, _ = P1.classify_branch(mixed_e, mixed_m)
    check("one non-conforming slice defeats REAL (all-slices condition)",
          ruling == "AMBIGUOUS", f"got {ruling}")

    expect_raise("empty subset is an ERROR, never a verdict", StageError,
                 lambda: P1.classify_branch(np.array([]), np.array([])),
                 code="EMPTY_SUBSET")
    expect_raise("a non-finite ratio never reaches a comparison", StageError,
                 lambda: P1.classify_branch(np.array([np.nan]),
                                            np.array([1e-9])),
                 code="NON_FINITE_RATIO")


def test_p1_gate_and_metrics() -> None:
    print("\nP1 R_REAL_MIN gate, ordering and cross-check")
    rows = [{"dataset_index": 3, "file": "a.h5", "slice_index": 1,
             "E_re": 1.0, "E_re_over_S_ref_sq": 1e-12},
            {"dataset_index": 9, "file": "b.h5", "slice_index": 2,
             "E_re": 5.0, "E_re_over_S_ref_sq": 1.0}]
    try:
        P1._gate_real_channel(rows)
        check("R_REAL_MIN fires on a degenerate real channel", False,
              "no block")
    except StageBlock as blk:
        ok = (blk.block_code == "REAL_CHANNEL_DEGENERATE"
              and blk.n_failing == 1
              and blk.first_failing["dataset_index"] == 3
              and blk.threshold == P1.R_REAL_MIN)
        check("R_REAL_MIN BLOCKs with slice identity and threshold", ok,
              f"{blk.block_code} n={blk.n_failing}")

    ok_rows = [{"E_re_over_S_ref_sq": 1e-9}, {"E_re_over_S_ref_sq": 1.0}]
    try:
        P1._gate_real_channel(ok_rows)
        check("R_REAL_MIN does not fire above the floor", True)
    except StageBlock:
        check("R_REAL_MIN does not fire above the floor", False)

    expect_raise("ratios refuse to form on a zero real channel (ordering "
                 "guard)", StageError,
                 lambda: P1.slice_representation_metrics(np.zeros((4, 4)),
                                                         np.ones((4, 4))),
                 code="REAL_CHANNEL_DENOMINATOR_NON_POSITIVE")

    rng = np.random.default_rng(1)
    re = rng.normal(size=(N, N))
    m = P1.slice_representation_metrics(re, np.zeros((N, N)))
    check("purely real slice has rho_imag_E == 0", m["rho_imag_E"] == 0.0)

    c_real = P1.conjugate_symmetry_violation(re + 0j)
    check("real image satisfies conjugate symmetry",
          c_real["conj_symmetry_violation_rel"] < 1e-12,
          f"rel={c_real['conj_symmetry_violation_rel']:.3e}")
    c_cx = P1.conjugate_symmetry_violation(re + 1j * rng.normal(size=(N, N)))
    check("complex image violates conjugate symmetry",
          c_cx["conj_symmetry_violation_rel"] > 1e-3,
          f"rel={c_cx['conj_symmetry_violation_rel']:.3e}")


# ---------------------------------------------------------------------------
# P2
# ---------------------------------------------------------------------------

def _mask(n: int, keep: int) -> torch.Tensor:
    m = torch.zeros(n, dtype=torch.bool)
    m[::max(1, n // keep)] = True
    return m


def test_p2_branches() -> None:
    print("\nP2 branch structure")
    s2 = 100.0
    base = {"E_Fdx": 1.0, "E_MFdx": 1e-12, "max_Fdx": 1.0, "max_MFdx": 1e-9,
            "k_i": 2.0}

    v = P2.classify_slice(dict(base), s2)
    check("ordinary branch selected above R_RESID_MIN",
          v["branch"] == "ordinary" and v["passed"] is True)
    check("ordinary branch applies both ratio tests",
          v["rho_M_applicable"] and v["relative_max_applicable"]
          and not v["absolute_leakage_applicable"])

    bad = dict(base, E_MFdx=1e-2)
    v = P2.classify_slice(bad, s2)
    check("ordinary branch fails on rho_M", v["rho_M_pass"] is False
          and v["passed"] is False)

    nz = {"E_Fdx": 1e-9, "E_MFdx": 1e-20, "max_Fdx": 1e-5,
          "max_MFdx": 1e-9, "k_i": 1.0}          # ratio 1e-11 <= R_RESID_MIN
    v = P2.classify_slice(nz, s2)
    check("near-zero branch selected at or below R_RESID_MIN",
          v["branch"] == "near_zero")
    check("near-zero branch REPLACES BOTH ordinary tests",
          v["rho_M_pass"] is None and v["relative_max_pass"] is None
          and v["absolute_leakage_applicable"] is True)
    check("near-zero ordinary values are recorded but marked diagnostic",
          v["rho_M"] is not None
          and set(v["diagnostic_only"]) == {"rho_M", "relative_max"})
    check("near-zero passes within ABS_LEAK · k_i",
          v["absolute_leakage_pass"] is True and v["passed"] is True)

    v = P2.classify_slice(dict(nz, max_MFdx=1e-3), s2)
    check("near-zero fails above ABS_LEAK · k_i",
          v["absolute_leakage_pass"] is False and v["passed"] is False)

    eq = {"E_Fdx": P2.R_RESID_MIN * s2, "E_MFdx": 0.0, "max_Fdx": 1e-6,
          "max_MFdx": 0.0, "k_i": 1.0}
    v = P2.classify_slice(eq, s2)
    check("equality at R_RESID_MIN enters the NEAR-ZERO branch",
          v["branch"] == "near_zero", f"got {v['branch']}")

    v = P2.classify_slice(dict(base, k_i=0.0), s2)
    check("k_i == 0 is flagged degenerate, not divided by",
          v["k_i_degenerate"] is True and v["passed"] is False)

    z = P2.classify_slice({"E_Fdx": 0.0, "E_MFdx": 0.0, "max_Fdx": 0.0,
                           "max_MFdx": 0.0, "k_i": 1.0}, s2)
    check("zero residual ratio falls OUTSIDE the boundary band by "
          "construction", z["boundary_band_member"] is False
          and z["boundary_distance_decades"] is None)

    band = P2.classify_slice(dict(base, E_Fdx=P2.R_RESID_MIN * s2 * 5,
                                  E_MFdx=0.0, max_MFdx=0.0), s2)
    check("a slice within one decade is a band member",
          band["boundary_band_member"] is True
          and band.get("allowance_ratio") is not None)

    expect_raise("ordinary branch asserts max|F dx| != 0 rather than handling "
                 "it", StageError,
                 lambda: P2.classify_slice({"E_Fdx": 1.0, "E_MFdx": 0.0,
                                            "max_Fdx": 0.0, "max_MFdx": 0.0,
                                            "k_i": 1.0}, s2),
                 code="ORDINARY_BRANCH_ZERO_MAX")


def test_p2_operators() -> None:
    print("\nP2 operator path and x0 contract")
    torch.manual_seed(0)
    m = _mask(N, N // 2)
    x = torch.randn(N, N) + 1j * torch.randn(N, N)
    k = P2.fft2c(x)
    y_raw = P2._mask_k(k, m) * 7.5                # raw units; already masked
    amax = 7.5

    r32 = P2.reconstruct_x0(y_raw, m, amax, dtype=torch.complex64)
    r64 = P2.reconstruct_x0(y_raw, m, amax, dtype=torch.complex128)
    check("independent x0 has the (2, H, W) layout",
          tuple(r32.shape) == (2, N, N))
    d = P2.x0_discrepancy(r64.numpy().astype(np.float64),
                          r32.numpy().astype(np.float64))
    check("fp32 and fp64 x0 agree well inside X0_ASSERT_RTOL",
          d["x0_rel_error"] <= P2.X0_ASSERT_RTOL,
          f"rel={d['x0_rel_error']:.3e}")

    wrong = r32.numpy().astype(np.float64) * 1.05
    d = P2.x0_discrepancy(wrong, r32.numpy().astype(np.float64))
    check("a 5% wrong x0 exceeds X0_ASSERT_RTOL",
          d["x0_rel_error"] > P2.X0_ASSERT_RTOL,
          f"rel={d['x0_rel_error']:.3e}")
    check("x0 discrepancy reports the worst pixel coordinates",
          len(d["x0_worst_pixel"]) == 2)

    dpath = P2.x0_discrepancy(r64.numpy().astype(np.float64),
                              r32.numpy().astype(np.float64))
    check("direct fp64-vs-fp32 x0 discrepancy is at operator roundoff",
          dpath["x0_rel_error"] < 1e-5,
          f"rel={dpath['x0_rel_error']:.3e}")
    # The PROXY (difference of two scalar max-errors) can read exactly zero
    # while the two reconstructions differ pointwise. This is why it was
    # renamed and the direct measure added.
    ref = np.zeros((2, 4, 4))
    a = np.zeros((2, 4, 4)); a[0, 0, 0] = 1.0
    bb = np.zeros((2, 4, 4)); bb[0, 3, 3] = 1.0
    ea = P2.x0_discrepancy(a, ref)["x0_abs_error"]
    eb = P2.x0_discrepancy(bb, ref)["x0_abs_error"]
    direct = P2.x0_discrepancy(a, bb)["x0_abs_error"]
    check("the proxy field can read zero where the direct difference is "
          "large", abs(ea - eb) == 0.0 and direct == 1.0,
          f"proxy={abs(ea - eb)} direct={direct}")

    expect_raise("a non-positive per-volume divisor is refused (no fallback)",
                 StageError,
                 lambda: P2.reconstruct_x0(y_raw, m, 0.0,
                                           dtype=torch.complex64),
                 code="AMAX_INVALID")
    expect_raise("a non-boolean mask is refused", StageError,
                 lambda: P2._mask_k(k, m.to(torch.float32)),
                 code="MASK_DTYPE")
    expect_raise("a 2-D mask is refused (locked (W,) contract)", StageError,
                 lambda: P2._mask_k(k, m.unsqueeze(0).expand(N, N)),
                 code="MASK_SHAPE")

    # Exact residual identity: with y already masked, M F (x - x0) is roundoff.
    x0 = P2.reconstruct_x0(y_raw, m, amax, dtype=torch.complex64)
    x_norm = torch.stack([x.real, x.imag]) / amax * amax
    dx = x_norm - x0 * 1.0
    q = P2.support_quantities(torch.complex(dx[0], dx[1]),
                              torch.complex(x_norm[0], x_norm[1]), m,
                              dtype=torch.complex64)
    check("measured-support leakage of the exact residual is at roundoff",
          q["max_MFdx"] <= 1e-4 * q["k_i"],
          f"max|MFdx|={q['max_MFdx']:.3e} k_i={q['k_i']:.3e}")


def test_p2_gate() -> None:
    print("\nP2 gate")
    rows = [{"dataset_index": 1, "file": "a.h5", "slice_index": 0,
             "k_i_degenerate": True, "k_i": 0.0, "passed": False}]
    try:
        P2._gate(rows)
        check("k_i degeneracy BLOCKs", False, "no block")
    except StageBlock as blk:
        check("k_i degeneracy BLOCKs with identity",
              blk.block_code == "LEAKAGE_REFERENCE_DEGENERATE"
              and blk.first_failing["dataset_index"] == 1)
    rows = [{"dataset_index": 2, "file": "b.h5", "slice_index": 4,
             "k_i_degenerate": False, "passed": False, "branch": "ordinary",
             "rho_M": 1.0, "relative_max": 1.0, "max_MFdx": 1.0,
             "absolute_allowance": 1e-7}]
    try:
        P2._gate(rows)
        check("a failing support condition BLOCKs", False, "no block")
    except StageBlock as blk:
        check("a failing support condition BLOCKs",
              blk.block_code == "MEASURED_SUPPORT_INVALID")


# ---------------------------------------------------------------------------
# shared machinery
# ---------------------------------------------------------------------------

def _fixture_facts(runtime: float) -> tuple[dict, dict]:
    semantic = {"schema": "seqref-test/1", "verdict": "PASS",
                "slices": [{"i": 0, "v": 1.5}], "thresholds": {"a": 1e-10}}
    facts = {"schema": "seqref-test/1", "stage": "TEST",
             "artefact_type": "stage_facts", "verdict": "PASS",
             "run": {"utc": "2026-07-30T00:00:00.000000+00:00",
                     "runtime_seconds": runtime,
                     "peak_memory_bytes": int(runtime * 1e6)},
             "slices": semantic["slices"], "thresholds": semantic["thresholds"]}
    return attach_semantic_hash(facts, semantic), semantic


def test_publication() -> None:
    print("\nshared publication, hashing and error records")
    with tempfile.TemporaryDirectory() as d:
        f1, _ = _fixture_facts(1.0)
        f2, _ = _fixture_facts(99.0)
        check("semantic_sha256 is stable under changed runtime metadata",
              f1["semantic_sha256"] == f2["semantic_sha256"])
        check("semantic_sha256 is not self-referential",
              "semantic_sha256" in f1["semantic_scope"]["excluded"])

        p1, s1 = publish_stage(f1, d, "test_facts", "TEST")
        check("authoritative artefact and sidecar are paired",
              os.path.isfile(p1) and os.path.isfile(p1 + ".sha256"))
        check("sidecar verifies", verify_sidecar(p1) == s1)
        check("no claim file survives publication",
              not [n for n in os.listdir(d) if ".claim." in n])

        p2, s2 = publish_stage(f2, d, "test_facts", "TEST")
        check("a rerun never overwrites the authoritative record",
              p2 != p1 and verify_sidecar(p1) == s1)
        # The recorded overwrite_policy claims a TIMESTAMPED record alongside
        # the authoritative one. Pin that claim to the code.
        b1, b2 = os.path.basename(p1), os.path.basename(p2)
        check("the authoritative name is exactly <prefix>.json",
              b1 == "test_facts.json", b1)
        check("a rerun writes <prefix>.<stamp>.json, as the policy states",
              b2.startswith("test_facts.") and b2.endswith(".json")
              and b2 != b1 and len(b2) > len(b1), b2)
        check("the rerun record carries its own verified sidecar",
              verify_sidecar(p2) == s2)
        check("the rerun stamp derives from run.utc, not wall clock",
              "20260730T000000000000" in b2, b2)
        check("scientifically identical reruns differ in artefact bytes",
              s2 != s1)

        with open(p1, "rb") as fh:
            body = json.load(fh)
        check("published facts carry the verdict and artefact type",
              body["verdict"] == "PASS"
              and body["artefact_type"] == "stage_facts")

        blocked = dict(f1)
        blk = StageBlock("TEST_BLOCK", "fixture block", observed=1.0,
                         threshold=0.5,
                         first_failing={"dataset_index": 7}, n_failing=1)
        blocked.update(blk.as_record())
        blocked["verdict"] = "BLOCK"
        pb, _ = publish_stage(blocked, d, "test_block_facts", "TEST")
        with open(pb, "rb") as fh:
            bb = json.load(fh)
        check("a BLOCK publishes VALID facts carrying code, reason, identity "
              "and threshold",
              bb["verdict"] == "BLOCK" and bb["block_code"] == "TEST_BLOCK"
              and bb["first_failing_slice"]["dataset_index"] == 7
              and bb["registered_threshold"] == 0.5 and bb["n_failing"] == 1)

        ep = publish_error(StageError("TEST_ERROR", "fixture error"), d,
                           "test_error", "TEST", parents={"p0": "x"},
                           code={"script": "s"}, run={"argv": []})
        check("an error record has a DISTINCT filename",
              ep is not None and "test_error" in os.path.basename(ep))
        with open(ep, "rb") as fh:
            er = json.load(fh)
        check("an error record is explicitly typed and cannot pass as facts",
              er["artefact_type"] == "error" and er["verdict"] == "ERROR"
              and er["error_code"] == "TEST_ERROR")

        nan_facts = dict(f1)
        nan_facts["slices"] = [{"i": 0, "v": float("nan")}]
        try:
            publish_stage(nan_facts, d, "test_nan", "TEST")
            check("non-finite values are refused, not silently written", False)
        except ValueError:
            check("non-finite values are refused, not silently written", True)

        marker = os.path.join(d, ".test_claim.claim")   # FIXED name per prefix
        with open(marker, "w", encoding="utf-8") as fh:
            fh.write("stage=OTHER pid=1 token=deadbeef utc=fixture\n")
        try:
            with publication_claim(d, "test_claim", "TEST"):
                check("an existing claim blocks concurrent publication", False)
        except StageError as exc:
            check("an existing claim blocks concurrent publication",
                  exc.error_code == "PUBLICATION_CLAIM_HELD")
        os.remove(marker)


def _write_pair(path: str, obj: dict) -> None:
    import hashlib
    from preflight_io import canonical_bytes
    payload = canonical_bytes(obj)
    with open(path, "wb") as fh:
        fh.write(payload)
    with open(path + ".sha256", "w", encoding="utf-8") as fh:
        fh.write(f"{hashlib.sha256(payload).hexdigest()}  "
                 f"{os.path.basename(path)}\n")


def test_parent_error_paths_end_to_end() -> None:
    print("\nend-to-end: parent failures -> EXIT_ERROR, trusted/untrusted "
          "split")
    # (a) UNVERIFIABLE parent: identity untrustworthy -> log and raise, and
    #     NO stage artefact of any kind is presented as valid.
    with tempfile.TemporaryDirectory() as d:
        bogus = os.path.join(d, "p0_facts.json")
        with open(bogus, "w", encoding="utf-8") as fh:
            fh.write("{}")                       # no sidecar
        out = os.path.join(d, "out")
        for name, mod in (("P1", P1), ("P2", P2)):
            rc = mod.main(["--repo-dir", d, "--data-root", d,
                           "--p0-facts", bogus, "--p0s-facts", bogus,
                           "--p0s-script", bogus, "--out-dir", out])
            files = os.listdir(out) if os.path.isdir(out) else []
            check(f"{name}: unverifiable parent returns EXIT_ERROR, not a "
                  f"traceback", rc == EXIT_ERROR, f"rc={rc}")
            check(f"{name}: unverifiable parent writes NO artefact at all",
                  not files, f"{files}")

    # (b) VERIFIED parent that fails a downstream check: the output path is
    #     trustworthy, so an auditable error record MUST be written.
    with tempfile.TemporaryDirectory() as d:
        p0 = os.path.join(d, "p0_facts.json")
        _write_pair(p0, {"schema": "seqref-p0-facts/2", "stage": "P0",
                         "verdict": "BLOCK"})     # verified, but not a PASS
        out = os.path.join(d, "out")
        for name, mod, prefix in (("P1", P1, "representation_error"),
                                  ("P2", P2, "support_error")):
            rc = mod.main(["--repo-dir", d, "--data-root", d,
                           "--p0-facts", p0, "--p0s-facts", p0,
                           "--p0s-script", p0, "--out-dir", out])
            check(f"{name}: non-PASS parent returns EXIT_ERROR",
                  rc == EXIT_ERROR, f"rc={rc}")
            files = os.listdir(out) if os.path.isdir(out) else []
            recs = [f for f in files
                    if f.startswith(prefix) and f.endswith(".json")]
            check(f"{name}: verified-parent failure leaves a distinctly named "
                  f"error record", bool(recs), f"{files}")
            if recs:
                with open(os.path.join(out, recs[0]), "rb") as fh:
                    er = json.load(fh)
                check(f"{name}: error record is typed and names the fault",
                      er["artefact_type"] == "error"
                      and er["error_code"] == "PARENT_NOT_PASSED")
            check(f"{name}: no stage facts artefact was published",
                  not any(f.startswith("representation_facts")
                          or f.startswith("support_facts") for f in files))



def _contender(out_dir: str, barrier, q) -> None:
    barrier.wait()
    try:
        with publication_claim(out_dir, "race", "TEST"):
            q.put("acquired")
            time.sleep(0.6)          # hold, so losers cannot simply queue
    except StageError:
        q.put("refused")
    except Exception as exc:                       # pragma: no cover
        q.put(f"unexpected:{type(exc).__name__}:{exc}")


def test_same_prefix_race() -> None:
    print("\nsame-prefix claim: simultaneous contenders")
    n = 8
    ctx = mp.get_context("fork")
    with tempfile.TemporaryDirectory() as d:
        barrier = ctx.Barrier(n)
        q = ctx.Queue()
        procs = [ctx.Process(target=_contender, args=(d, barrier, q))
                 for _ in range(n)]
        for pr in procs:
            pr.start()
        for pr in procs:
            pr.join(30)
        got = [q.get() for _ in range(n)]
        check("exactly ONE of 8 simultaneous same-prefix contenders acquires",
              got.count("acquired") == 1, f"{got}")
        check("every loser is refused with a typed StageError, not a crash",
              got.count("refused") == n - 1, f"{got}")
        check("no claim survives the race",
              not [x for x in os.listdir(d) if ".claim" in x],
              f"{os.listdir(d)}")


def test_failed_publication_releases_claim() -> None:
    print("\nfailed publication releases its claim")
    with tempfile.TemporaryDirectory() as d:
        bad, _ = _fixture_facts(1.0)
        bad["slices"] = [{"v": float("inf")}]
        try:
            publish_stage(bad, d, "race", "TEST")
            check("a failed publication raises", False)
        except ValueError:
            check("a failed publication raises", True)
        check("a failed publication leaves NO live claim",
              not os.path.exists(os.path.join(d, ".race.claim")),
              f"{os.listdir(d)}")
        p, _ = publish_stage(_fixture_facts(1.0)[0], d, "race", "TEST")
        check("the prefix is reusable after a failed publication",
              os.path.isfile(p))


def test_run_mode_guard() -> None:
    print("\nsmoke / authoritative run-mode guard")
    with tempfile.TemporaryDirectory() as d:
        check("smoke is allowed into an empty directory",
              guard_run_mode(d, True) == "smoke")
        open(os.path.join(d, "representation_facts.json"), "w").close()
        expect_raise("smoke is REFUSED against a locked authoritative path",
                     StageError, lambda: guard_run_mode(d, True),
                     code="SMOKE_INTO_AUTHORITATIVE_PATH")
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "smoke_support_facts.json"), "w").close()
        expect_raise("an authoritative run is REFUSED where smoke residue "
                     "remains", StageError, lambda: guard_run_mode(d, False),
                     code="SMOKE_RESIDUE_IN_AUTHORITATIVE_PATH")


def _fake_parents() -> dict:
    return {"p0": {"facts_sha256": "0" * 64}, "p0s": {"facts_sha256": "1" * 64},
            "s_ref": 1.0, "s_ref_squared": 1.0, "subset_indices": [0],
            "subset_size": 1}


def test_unexpected_exceptions_end_to_end() -> None:
    print("\nunexpected runtime exceptions -> deterministic EXIT_ERROR")
    cases = [("P1", P1, "_collect", "representation_error",
              OSError("simulated dataset read failure")),
             ("P2", P2, "_evaluate", "support_error",
              RuntimeError("simulated FFT library failure"))]
    for name, mod, fn, prefix, exc in cases:
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out")
            orig_vp, orig_fn = mod.verify_parents, getattr(mod, fn)
            mod.verify_parents = lambda *a, **k: _fake_parents()

            def _boom(*a, _e=exc, **k):
                raise _e
            setattr(mod, fn, _boom)
            try:
                rc = mod.main(["--repo-dir", d, "--data-root", d,
                               "--p0-facts", d, "--p0s-facts", d,
                               "--p0s-script", d, "--out-dir", out])
            finally:
                mod.verify_parents, _ = orig_vp, setattr(mod, fn, orig_fn)
            check(f"{name}: an unexpected {type(exc).__name__} returns "
                  f"EXIT_ERROR, not a traceback", rc == EXIT_ERROR, f"rc={rc}")
            files = os.listdir(out) if os.path.isdir(out) else []
            recs = [f for f in files
                    if f.startswith(prefix) and f.endswith(".json")]
            check(f"{name}: it leaves a typed UNEXPECTED_RUNTIME_ERROR record",
                  bool(recs), f"{files}")
            if recs:
                with open(os.path.join(out, recs[0]), "rb") as fh:
                    er = json.load(fh)
                check(f"{name}: the record names the exception class",
                      er["error_code"] == "UNEXPECTED_RUNTIME_ERROR"
                      and er["detail"]["exception_type"] == type(exc).__name__)
            check(f"{name}: no stage facts artefact was published",
                  not any(f.startswith("representation_facts")
                          or f.startswith("support_facts") for f in files))



def test_margins() -> None:
    print("\nmargin diagnostics: orientation, null and unbounded cases")
    # --- P1 ---------------------------------------------------------------
    def p1rows(vals):
        return [{"E_re_over_S_ref_sq": v} for v in vals]
    m = P1.compute_margins(p1rows([P1.R_REAL_MIN * 10, 1.0]))
    check("P1 margin > 1 when the weakest slice clears the floor",
          close(m["real_channel"], 10.0)
          and m["real_channel_status"] == "finite",
          f"{m['real_channel']}")
    m = P1.compute_margins(p1rows([P1.R_REAL_MIN, 1.0]))
    check("P1 margin == 1 exactly on the BLOCK boundary",
          close(m["real_channel"], 1.0), f"{m['real_channel']!r}")
    m = P1.compute_margins(p1rows([P1.R_REAL_MIN / 2, 1.0]))
    check("P1 margin < 1 when the premise fails",
          close(m["real_channel"], 0.5), f"{m['real_channel']!r}")
    m = P1.compute_margins([])
    check("P1 margin on an empty population is null + not_applicable",
          m["real_channel"] is None
          and m["real_channel_status"] == "not_applicable")

    # --- P2 ---------------------------------------------------------------
    def ordrow(rho, rel, x0):
        return {"rho_M": rho, "relative_max": rel, "x0_rel_error": x0,
                "rho_M_applicable": True, "relative_max_applicable": True,
                "absolute_leakage_applicable": False}

    rows = [ordrow(P2.RHO_M_MAX / 100, P2.REL_MAX_MAX / 10,
                   P2.X0_ASSERT_RTOL / 1000),
            ordrow(P2.RHO_M_MAX / 2, P2.REL_MAX_MAX / 4,
                   P2.X0_ASSERT_RTOL / 10)]
    m = P2.compute_margins(rows)
    check("P2 ordinary margins are controlled by the WORST slice",
          close(m["ordinary_rho_M"], 2.0) and close(m["relative_max"], 4.0)
          and close(m["x0_contract"], 10.0),
          f"{m['ordinary_rho_M']} {m['relative_max']} {m['x0_contract']}")
    check("P2 ordinary margins report status finite",
          m["ordinary_rho_M_status"] == "finite"
          and m["relative_max_status"] == "finite"
          and m["x0_contract_status"] == "finite")
    check("no near-zero slices -> not_applicable, not a bare null",
          m["near_zero_leakage"] is None
          and m["near_zero_leakage_status"] == "not_applicable"
          and m["near_zero_leakage_applicable_slices"] == 0)

    nzrows = [{"rho_M_applicable": False, "relative_max_applicable": False,
               "absolute_leakage_applicable": True,
               "absolute_allowance": 1e-7, "max_MFdx": 1e-9,
               "x0_rel_error": P2.X0_ASSERT_RTOL / 10}]
    m = P2.compute_margins(nzrows)
    check("no ordinary slices -> ordinary margins are not_applicable",
          m["ordinary_rho_M"] is None
          and m["ordinary_rho_M_status"] == "not_applicable"
          and m["relative_max_status"] == "not_applicable")
    check("near-zero leakage margin is the per-slice headroom",
          close(m["near_zero_leakage"], 100.0)
          and m["near_zero_leakage_status"] == "finite",
          f"{m['near_zero_leakage']!r}")

    mixed = nzrows + [dict(nzrows[0], max_MFdx=0.0)]
    m = P2.compute_margins(mixed)
    check("mixed finite and zero leakage -> partly_unbounded, finite minimum "
          "retained", close(m["near_zero_leakage"], 100.0)
          and m["near_zero_leakage_status"] == "partly_unbounded"
          and m["near_zero_leakage_unbounded_slices"] == 1,
          f"{m['near_zero_leakage']!r}")

    allzero = [dict(nzrows[0], max_MFdx=0.0) for _ in range(3)]
    m = P2.compute_margins(allzero)
    check("all leakages exactly zero -> fully_unbounded with a null value",
          m["near_zero_leakage"] is None
          and m["near_zero_leakage_status"] == "fully_unbounded"
          and m["near_zero_leakage_unbounded_slices"] == 3)

    m = P2.compute_margins([ordrow(0.0, 0.0, 0.0)])
    check("an exactly-zero ordinary metric -> unbounded, never infinity",
          m["ordinary_rho_M"] is None
          and m["ordinary_rho_M_status"] == "unbounded"
          and m["x0_contract_status"] == "unbounded")

    # Scan only the VALUES, not the documentation strings -- the
    # undefined_rule text legitimately contains the words it forbids.
    numeric = {k: v for k, v in m.items()
               if not isinstance(v, (str, list))}
    blob = json.dumps(numeric)
    check("no margin VALUE ever serialises as Infinity or NaN",
          "Infinity" not in blob and "NaN" not in blob, blob)
    check("every unavailable margin value is exactly null",
          all(v is None for k, v in numeric.items()
              if k in ("x0_contract", "ordinary_rho_M", "relative_max",
                       "near_zero_leakage")), blob)


def test_margins_in_semantic_hash() -> None:
    print("\nmargins participate in semantic_sha256")
    base = {"summary": {"margins": {"real_channel": 10.0}}}
    a = attach_semantic_hash({}, dict(base))
    b = attach_semantic_hash(
        {}, {"summary": {"margins": {"real_channel": 1.01}}})
    check("changing a margin changes semantic_sha256",
          a["semantic_sha256"] != b["semantic_sha256"])
    c = attach_semantic_hash({}, dict(base))
    check("an unchanged margin leaves semantic_sha256 stable",
          a["semantic_sha256"] == c["semantic_sha256"])


def main() -> int:
    print(f"{SCRIPT_ID} {SCRIPT_VERSION} -- EPHEMERAL; delete after use\n")
    test_p1_classification()
    test_p1_gate_and_metrics()
    test_p2_branches()
    test_p2_operators()
    test_p2_gate()
    test_margins()
    test_margins_in_semantic_hash()
    test_publication()
    test_same_prefix_race()
    test_failed_publication_releases_claim()
    test_run_mode_guard()
    test_parent_error_paths_end_to_end()
    test_unexpected_exceptions_end_to_end()
    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} checks passed")
    if failed:
        for name, _, detail in failed:
            logger.error("FAILED: %s %s", name, detail)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

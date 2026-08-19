# SEQREF-TDIAGT v0.1 -- scripts.tdiag_selftest
# LIFETIME: KEEP
# =============================================================================
# Purpose: fixtures harness for SEQREF-TDIAG (scripts.tdiag), R0 slice.
#          Pure-synthetic fixtures: no dataset, no model, no parent
#          artefacts -- each fixture isolates one behaviour against the
#          REAL driver/package code (imported, never reimplemented).
# Coverage:
#   T1  taxonomy purity (structural): the driver source contains no
#       EXIT_PASS / EXIT_BLOCK tokens; the module constants are exactly
#       EXIT_REPORT=0 / EXIT_ERROR=2. Scientific BLOCK cannot leak in.
#   T2  deferred-probe guard: D4/D5/D6 (and any other probe request)
#       raise DEFERRED_PROBE_AMENDMENT_GATED -- no execution path exists.
#   T3  R0 comparison engine: exact equality per quantity (23 registered
#       quantities); a one-ulp float drift is a NAMED mismatch; NaN is a
#       mismatch, never a crash, never a pass.
#   T3b live parent-hash regression (2026-08-16 repair): a wrong FRESHLY
#       VERIFIED live IMPL semantic sha or TINY file sha fails exactly its
#       comparison row -- the tautological form could never fail.
#   T4  TINY parent dual-pin refusal: unpinned file -> PARENT_FILE_
#       MISMATCH; wrong semantic -> PARENT_SEMANTIC_MISMATCH; wrong
#       verdict -> PARENT_VERDICT_MISMATCH; non-authoritative ->
#       PARENT_NOT_AUTHORITATIVE.
#   T5  trace completeness: a missing replayed checkpoint raises
#       R0_TRACE_INCOMPLETE (never a silent skip of a grid point).
#   T6  state-hash determinism: insertion-order independent; identical
#       tensors hash equal; one flipped element changes the hash.
#   T7  facts schema purity: no 'verdict' key; completeness block marks
#       D1/D2/D3 pending; semantic hash present and stable across
#       identical assemblies.
#   T8  publication + ERROR taxonomy integration: happy path -> exit 0
#       and tdiag_facts published with a verifying sidecar; trusted-
#       context runtime failure -> exit 2, distinct tdiag_error record
#       with sidecar, NO facts; missing parent args -> exit 2 with NO
#       artefacts (untrusted-context rule).
#   T9  startup logging robustness: nested nonexistent --log-file/
#       --out-dir targets are CREATED; the subsequent parent-input
#       failure leaves as typed ERROR (2); exit 1 must never escape.
#   T10 preflight module identity (2026-08-17 repair): TDIAG and the
#       reused TINY code share ONE preflight_parents module object -- a
#       split (qualified vs legacy import paths) would create two
#       StageError classes and let TINY-raised errors escape the
#       driver's typed handler. Includes the cross-module catch probe.
#   T11 locked banks: Z_DIAG is exactly PCG64(0)/(128,13824)/f64->f32
#       and deterministic; JVP probes are exactly PCG64(2) Rademacher
#       float32 and deterministic; manifests pin the bank bytes.
#   T12 E0/R0 equivalence gate: exact records pass; a one-ulp drift or
#       an identity drift is D1_E0_R0_MISMATCH; an excluded (NMSE-None)
#       slice is D1_METRIC_INVALID -- never a silent N change.
#   T13 winner selection: E3 maximises, E4 minimises, ties resolve to
#       the lowest start index, non-finite starts are skipped, and
#       all-non-finite is D1_ALL_STARTS_NON_FINITE.
#   T14 aggregation: the arithmetic mean over the IDENTICAL slice set;
#       a permuted or truncated estimator slice set is
#       D1_SLICE_SET_MISMATCH.
#   T15 materiality bands: +2.0 dB and 0.5x NMSE boundaries are
#       inclusive and exact at one-ulp resolution; the E4-only
#       intermediate case yields usable=False/oracle=True with no
#       invented routing.
#   T17 D1 descriptive figures: four figures render non-empty; a broken
#       payload is D1_PLOT_FAILURE. T8 also pins the all-or-nothing
#       publication order (plots BEFORE publish; plot failure => exit 2
#       with a typed error record and NO facts artefact).
#   T18 gradient hygiene (2026-08-18 repair): after run_d1 the handoff
#       model's parameters are frozen with grads None, while E3/E4 still
#       update z (regression for the parameter-grad retention defect).
#   T16 estimator conventions (stub model, real decode machinery): E1
#       averages complex physical u BEFORE image formation; E2 shares
#       the bank and takes the coordinate-wise median; E3/E4 run exactly
#       the 8 locked starts with gridded trajectories and the exact
#       density decomposition; JVP is deterministic; run_d1 happy path
#       carries the E4 non-routing marker; D1 facts keep the recursive
#       no-verdict invariant.
#   T19 D2a state-swap identity invariant: the verified step-0 state is
#       swapped into the SAME model under hash checks at every boundary;
#       a tampered step-500/step-0 hash is D2A_STATE_MISMATCH, a missing
#       state0 is D2A_STATE0_MISSING; state0 is discarded after D2a.
#   T20 D2a Gaussian identity + percentile: the production log-prob
#       agrees with the analytic identity within the frozen tolerance
#       (and a wrong formula is caught); percentile below-all/above-all/
#       exact-tie/duplicate-tie conventions and the record fields.
#   T21 D2a z_true: exact stub closed form (bitwise), deterministic
#       vector sha, non-finite target -> D2A_Z_TRUE_NON_FINITE, missing
#       target -> D2A_TARGET_MISSING.
#   T22 D2a hygiene: no parameter mutation; no verdict/pattern keys;
#       slice-order drift -> D2A_SLICE_ORDER_MISMATCH; bank drift ->
#       D2A_BANK_MISMATCH.
#   T23 D2a happy path + facts: full block structure, top-K and global
#       top-K shapes, nested d2.completeness with top-level D2 partial,
#       run_mode validation-r0-d1-d2a, stable semantic hash.
#   T24 D2a figures: three figures render non-empty; a broken payload
#       is D2A_PLOT_FAILURE.
# Coverage registry: EXPECTED_COUNTS pins the check count of every
#   fixture plus the suite total; coverage_ok requires zero failures AND
#   exact count matches, so a green suite cannot silently shrink.
# Invocation: both `python seqref_mri/scripts/tdiag_selftest.py` and
#   `python -m seqref_mri.scripts.tdiag_selftest` are supported.
# Taxonomy: all fixtures PASS -> exit 0; any failure -> exit 2 (a failing
#   fixture is a construction/contract defect, ERROR class under LOCK 2;
#   never a scientific result). No fallback, no mock, no placeholder, no
#   silent pass: every failure path is logger.error + typed outcome.
#   D1 fixtures use a fixture-local elementwise STUB model (the unit
#   under test is the estimator logic) against the REAL decode/metric
#   machinery -- the same doctrine as T8's boundary patches.
# Changelog (NEW in v0.1):
#   * Introduced with the R0 slice after the 2026-08-15 EXEC SS10.6 lock.
#   * Review-repair round (2026-08-16, pre-execution; NO contract
#     change): all compare_registered call sites pass the freshly
#     verified live parent identities; new T3b fixture proves a wrong
#     live IMPL semantic sha or TINY file sha fails exactly its row, so
#     the tautological comparison form can never return.
# Update summary:
#   v0.1 pins the R0/D1/D2a-slice contracts as executable regressions:
#   the standalone 0/2 taxonomy, the amendment-gated deferred-probe
#   guard, the exact serialized-value comparison engine, the TINY
#   dual-pin refusal paths, trace-grid completeness, canonical state
#   hashing, the no-verdict facts schema, both ERROR-context boundaries,
#   the D1 estimator-slate conventions and the D2a state-swap/geometry
#   invariants, under a static expected-count coverage registry.
#   * D1 slice (2026-08-18, under the same SS10.6 lock; NO contract
#     change): T11-T16 pin the D1 contracts (locked banks, E0/R0 gate,
#     winner/tie-break, slice-set aggregation, materiality boundaries,
#     estimator conventions + run_d1 happy path + D1 facts invariants);
#     T10's bootstrap guard now also covers estimators.py and
#     d1_plots.py. Suite total 46 -> 87 checks.
#   * Review-repair round (2026-08-18, pre-execution; NO contract
#     change): T8 gains the all-or-nothing publication-order case (plot
#     failure pre-publication => exit 2, typed error record, NO facts);
#     T18 pins the gradient-hygiene freeze regression. +5 checks.
#   * D2a slice (2026-08-19, under the same SS10.6 lock; NO contract
#     change): T19-T24 pin the D2a contracts (state-swap identity,
#     Gaussian identity + percentile conventions, z_true encode/hash,
#     no-mutation/no-pattern hygiene, nested d2 completeness, figures);
#     T8 gains the D2a plot-failure case (+2), T10's bootstrap/identity
#     guards now cover d2a.py and d2a_plots.py (+2). 87 -> 118 checks.
# =============================================================================
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import sys
import tempfile

import numpy as np
import torch

if __package__:  # `python -m seqref_mri.scripts.tdiag_selftest`
    from seqref_mri.scripts import tdiag as td
    from seqref_mri.tdiag import d1_plots as tplots
    from seqref_mri.tdiag import d2a as td2a
    from seqref_mri.tdiag import d2a_plots as td2aplots
    from seqref_mri.tdiag import estimators as test
    from seqref_mri.tdiag import facts as tfacts
    from seqref_mri.tdiag import invariants as tinv
    from seqref_mri.tdiag import replay as treplay
else:  # direct script run: scripts/ is on sys.path; tdiag sets repo paths
    import tdiag as td
    from seqref_mri.tdiag import d1_plots as tplots
    from seqref_mri.tdiag import d2a as td2a
    from seqref_mri.tdiag import d2a_plots as td2aplots
    from seqref_mri.tdiag import estimators as test
    from seqref_mri.tdiag import facts as tfacts
    from seqref_mri.tdiag import invariants as tinv
    from seqref_mri.tdiag import replay as treplay
from preflight_io import file_sha256, verify_sidecar  # noqa: E402
from preflight_parents import StageError, publish_stage  # noqa: E402

SCRIPT_ID = "SEQREF-TDIAGT"
SCRIPT_VERSION = "v0.1"
logger = logging.getLogger(SCRIPT_ID)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    if not ok:
        logger.error("[%s] fixture FAILED: %s -- %s", SCRIPT_ID, name,
                     detail)


def expect_stage_error(name: str, fn, code: str) -> None:
    try:
        fn()
    except StageError as exc:
        check(name, exc.error_code == code,
              f"StageError code {exc.error_code!r} (expected {code!r})")
        return
    except Exception as exc:  # wrong failure class: still a failure
        check(name, False,
              f"raised {type(exc).__name__} instead of StageError "
              f"{code}: {exc}")
        return
    check(name, False, f"no error raised; expected StageError {code}")


# ---------------------------------------------------------------------------
# Synthetic registered-artefact shell shared by T3/T5/T7/T8.
# ---------------------------------------------------------------------------

def _tiny_facts_stub() -> dict:
    trace = {str(k): 100.0 - 0.1 * k for k in range(0, 501, 50)}
    ep0 = {"nll_batch_mean": 100.0, "mean_psnr_z0": 10.0,
           "mean_nmse_u_z0": 0.4}
    ep500 = {"nll_batch_mean": 50.0, "mean_psnr_z0": 12.0,
             "mean_nmse_u_z0": 0.2}
    return {"schema": "seqref-tiny-facts/1",
            "semantic_sha256": tinv.TINY_FACTS_SEMANTIC_SHA256,
            "verdict": "BLOCK", "authoritative": True,
            "endpoints": {"initial": ep0, "final": ep500},
            "nll_trace": trace,
            "selection": {"manifest_sha256": "d" * 64,
                          "draw_order_indices": [3, 1, 2]},
            "parents": {"impl_class_a": {"file_sha256": "e" * 64,
                                         "semantic_sha256": "f" * 64}}}


def _replay_match_stub(facts: dict) -> dict:
    ep0, ep500 = facts["endpoints"]["initial"], facts["endpoints"]["final"]
    return ({"manifest_sha256": facts["selection"]["manifest_sha256"],
             "draw_order_indices":
                 list(facts["selection"]["draw_order_indices"])},
            dict(facts["nll_trace"]),
            {"nll_batch_mean": ep0["nll_batch_mean"],
             "mean_psnr_z0": ep0["mean_psnr_z0"],
             "mean_nmse_u_z0": ep0["mean_nmse_u_z0"]},
            {"nll_batch_mean": ep500["nll_batch_mean"],
             "mean_psnr_z0": ep500["mean_psnr_z0"],
             "mean_nmse_u_z0": ep500["mean_nmse_u_z0"]})


# ---------------------------------------------------------------------------
# T1 -- taxonomy purity (structural guard)
# ---------------------------------------------------------------------------

def t1_taxonomy_purity() -> None:
    with open(td.__file__, "r", encoding="utf-8") as fh:
        src = fh.read()
    check("T1 no EXIT_PASS token in the TDIAG driver source",
          "EXIT_PASS" not in src)
    check("T1 no EXIT_BLOCK token in the TDIAG driver source",
          "EXIT_BLOCK" not in src)
    check("T1 exit constants are exactly 0 (report) and 2 (ERROR)",
          td.EXIT_REPORT == 0 and td.EXIT_ERROR == 2
          and not hasattr(td, "EXIT_PASS")
          and not hasattr(td, "EXIT_BLOCK"))


# ---------------------------------------------------------------------------
# T2 -- deferred-probe guard (D4/D5/D6 amendment-gated)
# ---------------------------------------------------------------------------

def t2_deferred_probe_guard() -> None:
    for probe in ("D4", "D5", "D6"):
        expect_stage_error(f"T2 {probe} refused (amendment-gated)",
                           lambda p=probe: tinv.refuse_deferred_probe(p),
                           "DEFERRED_PROBE_AMENDMENT_GATED")
    expect_stage_error("T2 unknown probe request refused",
                       lambda: tinv.refuse_deferred_probe("D9"),
                       "DEFERRED_PROBE_AMENDMENT_GATED")


# ---------------------------------------------------------------------------
# T3 -- R0 exact serialized-value comparison engine
# ---------------------------------------------------------------------------

def t3_comparison_engine() -> None:
    facts = _tiny_facts_stub()
    sel, trace, m0, m500 = _replay_match_stub(facts)
    res = treplay.compare_registered(facts, sel, trace, m0, m500,
                                     "e" * 64, "f" * 64,
                                     tinv.TINY_FACTS_FILE_SHA256)
    check("T3 exact replay is VALID", res["valid"] is True)
    check("T3 comparison count is the registered 23",
          len(res["comparisons"]) == 23,
          f"got {len(res['comparisons'])}")
    check("T3 per-quantity equality booleans all true",
          all(c["equal"] for c in res["comparisons"]))
    # one-ulp drift on a single registered value -> named mismatch
    drifted = dict(m500)
    drifted["mean_psnr_z0"] = math.nextafter(m500["mean_psnr_z0"],
                                             math.inf)
    res2 = treplay.compare_registered(facts, sel, trace, m0, drifted,
                                      "e" * 64, "f" * 64,
                                      tinv.TINY_FACTS_FILE_SHA256)
    bad = [c["quantity"] for c in res2["comparisons"] if not c["equal"]]
    check("T3 one-ulp drift invalidates the replay",
          res2["valid"] is False)
    check("T3 drifted quantity is NAMED",
          bad == ["step500_z0_mean_psnr"], f"got {bad}")
    # NaN replay value: mismatch, never a crash, never a pass
    nan_m = dict(m0)
    nan_m["nll_batch_mean"] = float("nan")
    res3 = treplay.compare_registered(facts, sel, trace, nan_m, m500,
                                      "e" * 64, "f" * 64,
                                      tinv.TINY_FACTS_FILE_SHA256)
    check("T3 NaN replay value is a mismatch (NaN != NaN)",
          res3["valid"] is False
          and any(c["quantity"] == "step0_nll" and not c["equal"]
                  for c in res3["comparisons"]))


# ---------------------------------------------------------------------------
# T3b -- live parent-hash comparison regression (2026-08-16 repair)
# ---------------------------------------------------------------------------

def t3b_live_parent_hash_comparisons() -> None:
    """Regression for the 2026-08-16 review repair: the parent-hash rows
    compare the registered records against the FRESHLY VERIFIED live
    identities -- a wrong live value must fail the matching row (the
    tautological form could never fail)."""
    facts = _tiny_facts_stub()
    sel, trace, m0, m500 = _replay_match_stub(facts)
    res = treplay.compare_registered(facts, sel, trace, m0, m500,
                                     "e" * 64, "0" * 64,
                                     tinv.TINY_FACTS_FILE_SHA256)
    bad = [c["quantity"] for c in res["comparisons"] if not c["equal"]]
    check("T3b wrong live IMPL semantic sha fails exactly its row",
          res["valid"] is False
          and bad == ["parent_impl_semantic_sha256"], f"got {bad}")
    res2 = treplay.compare_registered(facts, sel, trace, m0, m500,
                                      "e" * 64, "f" * 64, "0" * 64)
    bad2 = [c["quantity"] for c in res2["comparisons"] if not c["equal"]]
    check("T3b wrong live TINY file sha fails exactly its row",
          res2["valid"] is False
          and bad2 == ["parent_tiny_file_sha256"], f"got {bad2}")


# ---------------------------------------------------------------------------
# T4 -- TINY parent dual-pin refusal
# ---------------------------------------------------------------------------

def t4_tiny_parent_dual_pin() -> None:
    with tempfile.TemporaryDirectory() as td_:
        fake = os.path.join(td_, "tiny_facts.json")
        with open(fake, "w", encoding="utf-8") as fh:
            json.dump(_tiny_facts_stub(), fh)
        expect_stage_error(
            "T4 unpinned TINY file refused (PARENT_FILE_MISMATCH)",
            lambda: treplay.load_tiny_parent(fake),
            "PARENT_FILE_MISMATCH")
        # Reach the deeper checks: pin-satisfying file sha + no-op sidecar.
        saved_sha, saved_sidecar = treplay.file_sha256, treplay.verify_sidecar
        treplay.file_sha256 = lambda p: tinv.TINY_FACTS_FILE_SHA256  # noqa: E731
        treplay.verify_sidecar = lambda p: tinv.TINY_FACTS_FILE_SHA256  # noqa: E731
        try:
            good = _tiny_facts_stub()
            ok_path = os.path.join(td_, "ok.json")
            with open(ok_path, "w", encoding="utf-8") as fh:
                json.dump(good, fh)
            art, sha = treplay.load_tiny_parent(ok_path)
            check("T4 pinned+consistent artefact loads",
                  sha == tinv.TINY_FACTS_FILE_SHA256
                  and art["verdict"] == "BLOCK")
            bad_sem = dict(good, semantic_sha256="0" * 64)
            p = os.path.join(td_, "bad_sem.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(bad_sem, fh)
            expect_stage_error(
                "T4 semantic mismatch refused",
                lambda: treplay.load_tiny_parent(p),
                "PARENT_SEMANTIC_MISMATCH")
            bad_verdict = dict(good, verdict="PASS")
            p = os.path.join(td_, "bad_verdict.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(bad_verdict, fh)
            expect_stage_error(
                "T4 non-BLOCK verdict refused",
                lambda: treplay.load_tiny_parent(p),
                "PARENT_VERDICT_MISMATCH")
            bad_auth = dict(good, authoritative=False)
            p = os.path.join(td_, "bad_auth.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(bad_auth, fh)
            expect_stage_error(
                "T4 non-authoritative artefact refused",
                lambda: treplay.load_tiny_parent(p),
                "PARENT_NOT_AUTHORITATIVE")
        finally:
            treplay.file_sha256, treplay.verify_sidecar = (saved_sha,
                                                           saved_sidecar)


# ---------------------------------------------------------------------------
# T5 -- trace-grid completeness
# ---------------------------------------------------------------------------

def t5_trace_completeness() -> None:
    facts = _tiny_facts_stub()
    sel, trace, m0, m500 = _replay_match_stub(facts)
    broken = {k: v for k, v in trace.items() if k != "250"}
    expect_stage_error(
        "T5 missing replayed checkpoint raises R0_TRACE_INCOMPLETE",
        lambda: treplay.compare_registered(facts, sel, broken, m0, m500,
                                           "e" * 64, "f" * 64,
                                           tinv.TINY_FACTS_FILE_SHA256),
        "R0_TRACE_INCOMPLETE")


# ---------------------------------------------------------------------------
# T6 -- canonical state-hash determinism
# ---------------------------------------------------------------------------

def t6_state_hash_determinism() -> None:
    a = {"w": torch.arange(6, dtype=torch.float32).reshape(2, 3),
         "b": torch.tensor([1.5], dtype=torch.float32)}
    b = {"b": torch.tensor([1.5], dtype=torch.float32),
         "w": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    check("T6 insertion order does not affect the hash",
          treplay.state_hash(a) == treplay.state_hash(b))
    check("T6 identical state hashes identically (64-hex)",
          treplay.state_hash(a) == treplay.state_hash(a)
          and len(treplay.state_hash(a)) == 64)
    c = {"w": torch.arange(6, dtype=torch.float32).reshape(2, 3),
         "b": torch.tensor([1.5 + 1e-7], dtype=torch.float32)}
    check("T6 one flipped element changes the hash",
          treplay.state_hash(a) != treplay.state_hash(c))


# ---------------------------------------------------------------------------
# T7 -- facts schema purity (evidence-only)
# ---------------------------------------------------------------------------

def _r0_result_stub() -> dict:
    facts = _tiny_facts_stub()
    sel, trace, m0, m500 = _replay_match_stub(facts)
    sel = {**sel, "population": 34742, "ordered_identities": [],
           "canonical_sorted_identities": [], "p0s_overlap_rule": "x"}
    comp = treplay.compare_registered(facts, sel, trace, m0, m500,
                                      "e" * 64, "f" * 64,
                                      tinv.TINY_FACTS_FILE_SHA256)
    return {"selection": sel, "trace": trace,
            "endpoints": {"initial": m0, "final": m500},
            "step0_state_hash": "0" * 64, "step500_state_hash": "1" * 64,
            "replay_config_hash": "2" * 64, **comp}


def t7_facts_schema_purity() -> None:
    saved_code, saved_env = tfacts.code_record, tfacts.environment_record
    tfacts.code_record = lambda repo: {"fixture": "isolated"}  # noqa: E731
    tfacts.environment_record = lambda *a, **k: {"fixture": True}  # noqa: E731
    try:
        tiny = _tiny_facts_stub()
        impl = {"schema": "seqref-impl-facts/1",
                "semantic_sha256": "f" * 64, "verdict": "PASS"}
        parents = {"parents_id": "fixture", "p0": {}, "p0s": {}}
        f1 = tfacts.build_r0_facts(_r0_result_stub(), tiny, "9" * 64,
                                   impl, "e" * 64, parents, {}, {}, {},
                                   15.62704, "/nonexistent-repo", ["x"])
        f2 = tfacts.build_r0_facts(_r0_result_stub(), tiny, "9" * 64,
                                   impl, "e" * 64, parents, {}, {}, {},
                                   15.62704, "/nonexistent-repo", ["x"])
    finally:
        tfacts.code_record, tfacts.environment_record = (saved_code,
                                                         saved_env)
    check("T7 facts carry NO verdict key (evidence-only)",
          "verdict" not in f1)
    check("T7 schema + partial report status exact",
          f1["schema"] == "seqref-tdiag-facts/1"
          and f1["report_status"].startswith("partial")
          and f1["authoritative"] is False)
    check("T7 completeness block: R0 complete, D1/D2/D3 pending",
          f1["completeness"] == {"R0": "complete", "D1": "pending",
                                 "D2": "pending", "D3": "pending"})
    check("T7 semantic hash present and stable across assemblies",
          isinstance(f1.get("semantic_sha256"), str)
          and f1["semantic_sha256"] == f2["semantic_sha256"])


# ---------------------------------------------------------------------------
# T8 -- publication + ERROR taxonomy integration
# ---------------------------------------------------------------------------

def t8_publication_and_error_taxonomy() -> None:
    saved: dict = {}

    def _patch(name, value, owner):
        # save the ORIGINAL only once: a second patch of the same
        # attribute must not clobber the restore value with a stub
        if (owner, name) not in saved:
            saved[(owner, name)] = getattr(owner, name)
        setattr(owner, name, value)

    def _happy_patches():
        _patch("verify_parents",
               lambda *a, **k: {"parents_id": "fixture", "p0": {},
                                "p0s": {}}, td)
        for name in ("load_p3_parent", "load_p4s2_parent"):
            _patch(name, lambda p: {}, td.ffr)
        _patch("load_implb_parent", lambda p: {"spline_b": 1.0}, td.ffr)
        _patch("_load_impl_parent",
               lambda p: ({"schema": "seqref-impl-facts/1",
                           "semantic_sha256": "f" * 64,
                           "verdict": "PASS"}, "e" * 64), td.tg)
        _patch("_s_ref_from_p0s", lambda p: 15.62704, td.tg)
        _patch("load_tiny_parent",
               lambda p: (_tiny_facts_stub(), "9" * 64), td.replay)
        _patch("run_d1", lambda *a, **k: {"note": "fixture d1 block"},
               td.estimators)
        _patch("run_d2a", lambda *a, **k: {"note": "fixture d2a block"},
               td.d2a)
        _patch("render_d1_figures", lambda d1, out: ["f1.png", "f2.png",
                                                     "f3.png", "f4.png"],
               td.d1_plots)
        _patch("render_d2a_figures",
               lambda d2a, out: ["f5.png", "f6.png", "f7.png"],
               td.d2a_plots)
        _patch("code_record", lambda repo: {"fixture": True}, td.tfacts)
        _patch("environment_record", lambda *a, **k: {"fixture": True},
               td.tfacts)

    base_args = ["--repo-dir", os.path.realpath(td._REPO)]
    parents_args = ["--p0-facts", "x", "--p0s-facts", "x",
                    "--p0s-script", "x", "--p3-facts", "x",
                    "--p4-stats2", "x", "--implb-facts", "x",
                    "--impl-facts", "x", "--tiny-facts", "x"]
    try:
        _happy_patches()
        _patch("run_r0_with_context",
               lambda *a, **k: (_r0_result_stub(), None), td.replay)
        with tempfile.TemporaryDirectory() as td_:
            rc = td.main(base_args + ["--data-root", td_,
                                      "--out-dir", td_] + parents_args)
            check("T8 valid R0 + D1 -> exit 0 (report)", rc == 0)
            pubs = [p for p in os.listdir(td_)
                    if p.startswith("tdiag_facts")
                    and p.endswith(".json")]
            ok_pub = False
            if len(pubs) == 1:
                path = os.path.join(td_, pubs[0])
                ok_pub = verify_sidecar(path) == file_sha256(path)
            check("T8 evidence report published with verifying sidecar",
                  ok_pub, f"{pubs}")
        # trusted-context runtime failure AFTER parents verified
        _patch("run_r0_with_context",
               lambda *a, **k: (_ for _ in ()).throw(
                   RuntimeError("injected trusted-context failure")),
               td.replay)
        with tempfile.TemporaryDirectory() as td_:
            rc = td.main(base_args + ["--data-root", td_,
                                      "--out-dir", td_] + parents_args)
            check("T8 trusted-context runtime failure -> exit 2", rc == 2)
            errs = [p for p in os.listdir(td_)
                    if p.startswith("tdiag_error") and p.endswith(".json")]
            facts_left = [p for p in os.listdir(td_)
                          if p.startswith("tdiag_facts")]
            typed = False
            if len(errs) == 1:
                with open(os.path.join(td_, errs[0]),
                          encoding="utf-8") as fh:
                    rec = json.load(fh)
                typed = (rec.get("error_code") == "UNEXPECTED_RUNTIME_ERROR"
                         and rec.get("schema") == "seqref-tdiag-error/1"
                         and "injected trusted-context failure"
                         in rec.get("error_reason", ""))
            check("T8 distinct typed tdiag_error record written",
                  typed, f"{errs}")
            check("T8 NO evidence report after ERROR",
                  facts_left == [], f"{facts_left}")
        # plot failure BEFORE publication (2026-08-18 all-or-nothing
        # repair): typed D1_PLOT_FAILURE, exit 2, NO facts artefact
        _patch("run_r0_with_context",
               lambda *a, **k: (_r0_result_stub(), None), td.replay)

        def _plot_throw(*a, **k):
            raise td.d1_plots.StageError("D1_PLOT_FAILURE",
                                         "injected plot failure")
        _patch("render_d1_figures", _plot_throw, td.d1_plots)
        with tempfile.TemporaryDirectory() as td_:
            rc = td.main(base_args + ["--data-root", td_,
                                      "--out-dir", td_] + parents_args)
            check("T8 plot failure pre-publication -> exit 2", rc == 2)
            errs = [p for p in os.listdir(td_)
                    if p.startswith("tdiag_error") and p.endswith(".json")]
            facts_left = [p for p in os.listdir(td_)
                          if p.startswith("tdiag_facts")]
            typed = False
            if len(errs) == 1:
                with open(os.path.join(td_, errs[0]),
                          encoding="utf-8") as fh:
                    rec = json.load(fh)
                typed = rec.get("error_code") == "D1_PLOT_FAILURE"
            check("T8 typed D1_PLOT_FAILURE error record written",
                  typed, f"{errs}")
            check("T8 NO evidence report when plots fail "
                  "pre-publication", facts_left == [], f"{facts_left}")
        # symmetric D2a plot failure (2026-08-19): typed D2A_PLOT_FAILURE,
        # exit 2, NO facts artefact; render_d1_figures is first restored
        # to the happy stub (the previous case left it throwing)
        _patch("render_d1_figures", lambda d1, out: ["f1.png", "f2.png",
                                                     "f3.png", "f4.png"],
               td.d1_plots)
        def _d2a_plot_throw(*a, **k):
            raise td.d2a_plots.StageError("D2A_PLOT_FAILURE",
                                          "injected d2a plot failure")
        _patch("render_d2a_figures", _d2a_plot_throw, td.d2a_plots)
        with tempfile.TemporaryDirectory() as td_:
            rc = td.main(base_args + ["--data-root", td_,
                                      "--out-dir", td_] + parents_args)
            check("T8 D2a plot failure pre-publication -> exit 2",
                  rc == 2)
            errs = [p for p in os.listdir(td_)
                    if p.startswith("tdiag_error") and p.endswith(".json")]
            facts_left = [p for p in os.listdir(td_)
                          if p.startswith("tdiag_facts")]
            typed = False
            if len(errs) == 1:
                with open(os.path.join(td_, errs[0]),
                          encoding="utf-8") as fh:
                    rec = json.load(fh)
                typed = rec.get("error_code") == "D2A_PLOT_FAILURE"
            check("T8 typed D2A_PLOT_FAILURE record, NO facts artefact",
                  typed and facts_left == [], f"{errs} {facts_left}")
    finally:
        for (owner, name), value in saved.items():
            setattr(owner, name, value)
    with tempfile.TemporaryDirectory() as td_:
        rc = td.main(base_args + ["--data-root", td_, "--out-dir", td_])
        check("T8 missing parent arguments -> exit 2 (ERROR)", rc == 2)
        leftovers = [p for p in os.listdir(td_)
                     if p.startswith("tdiag_facts")
                     or p.startswith("tdiag_error")]
        check("T8 no artefact after PARENT_INPUT_MISSING",
              leftovers == [], f"found {leftovers}")


# ---------------------------------------------------------------------------
# T9 -- startup logging robustness (exit 1 must never escape)
# ---------------------------------------------------------------------------

def t9_startup_logging_robustness() -> None:
    with tempfile.TemporaryDirectory() as td_:
        nested_out = os.path.join(td_, "deep", "nested", "out")
        nested_log = os.path.join(td_, "deep", "nested", "logs",
                                  "tdiag_run.log")
        try:
            rc = td.main(["--repo-dir", os.path.realpath(td._REPO),
                          "--data-root", td_,
                          "--out-dir", nested_out,
                          "--log-file", nested_log])
        finally:
            # td.main reconfigures root logging (force=True) onto the
            # temporary file handler; restore stdout logging before the
            # tempdir disappears so later suite logging cannot hit a
            # stale descriptor.
            logging.basicConfig(level=logging.INFO,
                                format="%(asctime)s %(name)s %(message)s",
                                force=True)
        check("T9 startup failure leaves as typed ERROR (2), never raw 1",
              rc == 2, f"rc={rc}")
        check("T9 startup guard created the nested log file",
              os.path.isfile(nested_log))
        content = (open(nested_log, encoding="utf-8").read()
                   if os.path.isfile(nested_log) else "")
        check("T9 typed PARENT_INPUT_MISSING recorded in the log file",
              "PARENT_INPUT_MISSING" in content,
              content[-200:] if content else "log missing/empty")
        check("T9 startup guard created the nested out dir",
              os.path.isdir(nested_out))


# ---------------------------------------------------------------------------
# T10 -- preflight module identity shared with TINY (2026-08-17 repair)
# ---------------------------------------------------------------------------

def t10_preflight_module_identity() -> None:
    """Regression for the dual-identity defect: if TDIAG ever imports
    preflight through a DIFFERENT module path than tiny_gate (e.g. a
    package-qualified seqref_mri.src.preflight_parents alongside the
    legacy top-level preflight_parents), Python builds TWO module objects
    and their StageError classes diverge -- errors raised inside reused
    TINY primitives would escape the driver's `except StageError` and be
    mislabeled UNEXPECTED_RUNTIME_ERROR. One canonical identity only."""
    import sys as _sys
    import preflight_parents as pp
    check("T10 all TDIAG modules share the legacy StageError identity",
          td.StageError is pp.StageError
          and treplay.StageError is pp.StageError
          and tfacts.StageError is pp.StageError
          and tinv.StageError is pp.StageError
          and test.StageError is pp.StageError
          and tplots.StageError is pp.StageError
          and td2a.StageError is pp.StageError
          and td2aplots.StageError is pp.StageError)
    check("T10 reused tiny_gate shares the same StageError identity",
          td.tg.StageError is pp.StageError)
    check("T10 no duplicate qualified preflight_parents module object",
          "preflight_parents" in _sys.modules
          and "seqref_mri.src.preflight_parents" not in _sys.modules)
    try:
        raise td.tg._fail("IDENTITY_PROBE", "raised through tiny_gate")
    except td.StageError as exc:
        check("T10 StageError from TINY code caught by the driver class",
              exc.error_code == "IDENTITY_PROBE")
    except Exception as exc:  # noqa: BLE001 -- the regression witness
        check("T10 StageError from TINY code caught by the driver class",
              False, f"escaped as {type(exc).__name__}: identity split")
    # structural guard: the explicit bootstrap import must precede the
    # first preflight import in every TDIAG package module
    for mod in (treplay, tfacts, tinv, test, tplots, td2a,
                td2aplots):
        with open(mod.__file__, "r", encoding="utf-8") as fh:
            src = fh.read()
        boot_at = src.find("from seqref_mri.tdiag import _bootstrap")
        preflight_at = src.find("from preflight_")
        check(f"T10 {mod.__name__.rsplit('.', 1)[-1]} bootstraps before "
              f"preflight imports",
              0 <= boot_at < preflight_at)


# ---------------------------------------------------------------------------
# D1 fixtures: fixture-local elementwise STUB model (the estimator logic
# is the unit under test) against the REAL decode/metric machinery.
# ---------------------------------------------------------------------------

class _StubFlow:
    """Elementwise scaling flow: encode(u) = u * exp(log_scale) with a
    constant log-abs-det -- enough to exercise the E3 objective path
    (production flow.encode + gaussian log-prob) deterministically."""

    def __init__(self, dim):
        self.log_scale = torch.full((dim,), 0.02)

    def encode(self, u, h):
        z = u * torch.exp(self.log_scale)
        ldj = self.log_scale.sum().expand(u.shape[0])
        return z, ldj


class _StubModel(torch.nn.Module):
    """Elementwise decoder: decode_scalars(z) = s * z + b (s > 0). The
    physical-u map is then affine, so E1/E2 statistics and the JVP have
    exact closed forms the fixtures compare against."""

    def __init__(self):
        super().__init__()
        dim = td.tg.ffr.FLOW_DIM_REAL
        g = np.random.Generator(np.random.PCG64(11))
        self.s = torch.from_numpy(
            g.uniform(0.5, 1.5, size=dim).astype(np.float32))
        self.b = torch.from_numpy(
            g.normal(0.0, 0.1, size=dim).astype(np.float32))
        self.flow = _StubFlow(dim)

    def condition(self, cond_in, mask):
        return torch.zeros(cond_in.shape[0], 4)

    def decode_scalars(self, z, cond_in, mask):
        return z.to(torch.float32) * self.s + self.b


def _stub_states(n: int = 2) -> list:
    """Synthetic slice states through the REAL map/scaling machinery:
    24 acquired columns (indices 0..23), identity P4 affine (mean 0,
    scale 1), complex64 raw y, complex128 u_true."""
    ffr, dec = td.tg.ffr, td.tg.dec
    acq = list(range(24))
    cmap = dec.build_coordinate_map(acq, ffr.GRID_H, ffr.GRID_W)
    acq_set = set(acq)
    loc_index = {(r, c): {"applied_mean_re": 0.0, "applied_scale_re": 1.0,
                          "applied_mean_im": 0.0, "applied_scale_im": 1.0}
                 for r in range(ffr.GRID_H) for c in range(ffr.GRID_W)
                 if c not in acq_set}
    vecs = ffr.standardisation_vectors(cmap, loc_index)
    rng = np.random.Generator(np.random.PCG64(7))
    mask = torch.zeros(1, ffr.GRID_W)
    mask[0, :24] = 1.0
    states = []
    for i in range(n):
        y = torch.from_numpy(
            (rng.standard_normal((1, ffr.GRID_H, ffr.GRID_W))
             + 1j * rng.standard_normal((1, ffr.GRID_H, ffr.GRID_W))
             ).astype(np.complex64))
        u_true = (rng.standard_normal(cmap.n_free_complex)
                  + 1j * rng.standard_normal(cmap.n_free_complex)
                  ).astype(np.complex128)
        # registered standardised target (encode_target layout): with
        # the identity P4 affine this is exactly interleaved re/im f64
        target = np.empty((1, ffr.FLOW_DIM_REAL), dtype=np.float64)
        target[0, 0::2] = u_true.real
        target[0, 1::2] = u_true.imag
        states.append({
            "identity": {"split": "train", "file": f"fixture_{i}.h5",
                         "slice_index": i, "dataset_index": i},
            "cmap": cmap, "vecs": vecs,
            "cond": torch.zeros(1, 2, ffr.GRID_H, ffr.GRID_W),
            "mask": mask, "y": y, "amax": torch.ones(1),
            "x_true_mag": torch.from_numpy(
                np.abs(rng.standard_normal(
                    (ffr.GRID_H, ffr.GRID_W))).astype(np.float32)),
            "target": target,
            "u_true": u_true, "excluded": False})
    return states


def _expected_bank_u(model: _StubModel, bank: np.ndarray) -> np.ndarray:
    """Closed-form shared-bank decodes under the stub, replicating the
    implementation's dtype path EXACTLY (float32 elementwise arithmetic,
    float64 unpack, identity affine)."""
    us32 = (bank * model.s.numpy() + model.b.numpy()).astype(np.float32)
    us64 = us32.astype(np.float64)
    return (us64[:, 0::2] + 1j * us64[:, 1::2]).astype(np.complex128)


def _nmse_expected(u_hat: np.ndarray, u_true: np.ndarray) -> float:
    return float(np.sum(np.abs(u_hat - u_true) ** 2)
                 / np.sum(np.abs(u_true) ** 2))


def t11_locked_banks() -> None:
    b1, b2 = test.z_diag_bank(), test.z_diag_bank()
    dim = td.tg.ffr.FLOW_DIM_REAL
    check("T11 Z_DIAG bank is deterministic across builds",
          np.array_equal(b1["bank"], b2["bank"]))
    recipe = np.random.Generator(np.random.PCG64(0)).standard_normal(
        size=(128, dim), dtype=np.float64).astype(np.float32)
    check("T11 Z_DIAG matches the locked recipe exactly",
          np.array_equal(b1["bank"], recipe))
    check("T11 Z_DIAG shape/dtype + manifest pins the bank bytes",
          b1["bank"].shape == (128, dim)
          and b1["bank"].dtype == np.float32
          and b1["bank_sha256"] == hashlib.sha256(
              b1["bank"].tobytes(order="C")).hexdigest()
          and len(b1["manifest_sha256"]) == 64)
    p1, p2 = test.jvp_probes(), test.jvp_probes()
    bits = np.random.Generator(np.random.PCG64(2)).integers(
        0, 2, size=(16, dim))
    recipe_p = torch.from_numpy((2 * bits - 1).astype(np.float32))
    check("T11 JVP probes deterministic + locked recipe exact",
          torch.equal(p1["probes"], p2["probes"])
          and torch.equal(p1["probes"], recipe_p))
    check("T11 JVP probes are float32 Rademacher",
          p1["probes"].dtype == torch.float32
          and set(torch.unique(p1["probes"]).tolist()) <= {-1.0, 1.0})


def t12_e0_r0_equivalence_gate() -> None:
    recs = [{"identity": {"file": "a", "slice_index": 1},
             "psnr": 30.0, "nmse_u": 0.5},
            {"identity": {"file": "b", "slice_index": 2},
             "psnr": 31.0, "nmse_u": 0.6}]
    r0_ok = [{"identity": {"file": "a", "slice_index": 1},
              "psnr_z0": 30.0, "nmse_u_z0": 0.5},
             {"identity": {"file": "b", "slice_index": 2},
              "psnr_z0": 31.0, "nmse_u_z0": 0.6}]
    test.check_e0_r0_equivalence(recs, r0_ok)
    check("T12 exact E0/R0 records pass the gate", True)
    drift = [dict(r0_ok[0]),
             dict(r0_ok[1], psnr_z0=math.nextafter(31.0, math.inf))]
    expect_stage_error(
        "T12 one-ulp E0 drift -> D1_E0_R0_MISMATCH",
        lambda: test.check_e0_r0_equivalence(recs, drift),
        "D1_E0_R0_MISMATCH")
    excl = [dict(r0_ok[0]), dict(r0_ok[1], nmse_u_z0=None)]
    expect_stage_error(
        "T12 excluded slice (NMSE None) -> D1_METRIC_INVALID",
        lambda: test.check_e0_r0_equivalence(recs, excl),
        "D1_METRIC_INVALID")
    wrong_id = [dict(r0_ok[0]),
                dict(r0_ok[1],
                     identity={"file": "c", "slice_index": 9})]
    expect_stage_error(
        "T12 identity drift -> D1_E0_R0_MISMATCH",
        lambda: test.check_e0_r0_equivalence(recs, wrong_id),
        "D1_E0_R0_MISMATCH")


def _start_rec(final, finite=True):
    return {"record": {"final_total": final, "finite": finite}}


def t13_winner_selection() -> None:
    starts = [_start_rec(-10.0), _start_rec(-5.0), _start_rec(-7.0)]
    check("T13 E3 winner is the highest final density",
          test._select_winner(starts, maximize=True) == 1)
    tie = [_start_rec(-5.0), _start_rec(-5.0), _start_rec(-6.0)]
    check("T13 E3 tie resolves to the lowest start index",
          test._select_winner(tie, maximize=True) == 0)
    errs = [_start_rec(3.0), _start_rec(1.5), _start_rec(2.0)]
    check("T13 E4 winner is the lowest final error",
          test._select_winner(errs, maximize=False) == 1)
    check("T13 E4 tie resolves to the lowest start index",
          test._select_winner([_start_rec(1.5), _start_rec(1.5)],
                              maximize=False) == 0)
    mixed = [_start_rec(float("nan"), finite=False), _start_rec(2.0)]
    check("T13 non-finite start skipped, finite winner chosen",
          test._select_winner(mixed, maximize=False) == 1)
    expect_stage_error(
        "T13 all non-finite -> D1_ALL_STARTS_NON_FINITE",
        lambda: test._select_winner(
            [_start_rec(None, False),
             _start_rec(float("nan"), False)], maximize=True),
        "D1_ALL_STARTS_NON_FINITE")


def _per_est_records():
    return [{"identity": {"file": "a", "slice_index": 1},
             "psnr": 10.0, "nmse_u": 0.4},
            {"identity": {"file": "b", "slice_index": 2},
             "psnr": 20.0, "nmse_u": 0.6}]


def _per_est_full():
    return {k: [dict(r) for r in _per_est_records()]
            for k in ("E0", "E1", "E2", "E3", "E4")}


def t14_aggregation_slice_set() -> None:
    agg, thr = test.aggregate_estimators(_per_est_full())
    check("T14 identical slice sets aggregate",
          agg["E0"]["mean_psnr"] == 15.0)
    check("T14 aggregation is the arithmetic mean",
          agg["E0"]["mean_nmse_u"] == 0.5
          and thr["E0_mean_psnr"] == 15.0)
    broken = _per_est_full()
    broken["E2"] = [broken["E2"][1], broken["E2"][0]]
    expect_stage_error(
        "T14 permuted slice set -> D1_SLICE_SET_MISMATCH",
        lambda: test.aggregate_estimators(broken),
        "D1_SLICE_SET_MISMATCH")
    short = _per_est_full()
    short["E3"] = short["E3"][:1]
    expect_stage_error(
        "T14 truncated slice set -> D1_SLICE_SET_MISMATCH",
        lambda: test.aggregate_estimators(short),
        "D1_SLICE_SET_MISMATCH")


def _per_est_e1_gain(psnr_gain, nmse):
    # base 0.0 keeps the delta free of addition rounding so the one-ulp
    # boundary constructions are resolved exactly
    e0 = [{"identity": {"file": "a", "slice_index": 1},
           "psnr": 0.0, "nmse_u": 0.4}]
    e1 = [{"identity": {"file": "a", "slice_index": 1},
           "psnr": psnr_gain, "nmse_u": nmse}]
    return {"E0": e0, "E1": e1, "E2": [dict(e0[0])],
            "E3": [dict(e0[0])], "E4": [dict(e0[0])]}


def t15_materiality_bands() -> None:
    agg, _ = test.aggregate_estimators(_per_est_e1_gain(2.0, 0.4))
    check("T15 +2.0 dB exactly is material (boundary inclusive)",
          agg["E1"]["material_by_psnr"] is True)
    agg, _ = test.aggregate_estimators(
        _per_est_e1_gain(math.nextafter(2.0, 0.0), 0.4))
    check("T15 one ulp below +2.0 dB is NOT material",
          agg["E1"]["material_by_psnr"] is False)
    agg, _ = test.aggregate_estimators(_per_est_e1_gain(0.0, 0.2))
    check("T15 NMSE exactly 0.5 x E0 is material (boundary inclusive)",
          agg["E1"]["material_by_nmse"] is True)
    agg, _ = test.aggregate_estimators(
        _per_est_e1_gain(0.0, math.nextafter(0.2, 1.0)))
    check("T15 one ulp above 0.5 x E0 is NOT material",
          agg["E1"]["material_by_nmse"] is False)
    per = _per_est_e1_gain(0.0, 0.4)
    per["E4"] = [{"identity": {"file": "a", "slice_index": 1},
                  "psnr": 15.0, "nmse_u": 0.1}]
    agg, _ = test.aggregate_estimators(per)
    dec = test.decision_fields(agg)
    check("T15 E4-only improvement: usable False, oracle True, no "
          "invented routing",
          dec["usable_estimator_material_improvement"] is False
          and dec["oracle_material_improvement"] is True
          and dec["estimator_mismatch"] is False
          and dec["oracle_negative"] is False)


def t16_estimator_conventions() -> None:
    model = _StubModel()
    states = _stub_states(2)
    bank = test.z_diag_bank()
    counter: dict = {}
    decodes = test.decode_bank(model, states[0], bank["bank"], counter)
    e1, e2 = test.e1_e2_from_decodes(states[0], decodes)
    u_exp = _expected_bank_u(model, bank["bank"])
    check("T16 E1 averages complex physical u BEFORE image formation",
          e1["nmse_u"] == _nmse_expected(u_exp.mean(axis=0),
                                         states[0]["u_true"]))
    u_med = (np.median(u_exp.real, axis=0)
             + 1j * np.median(u_exp.imag, axis=0)).astype(np.complex128)
    check("T16 E2 shares the ONE bank pass (128 decodes) + median exact",
          counter["n"] == 128
          and e2["nmse_u"] == _nmse_expected(u_med,
                                             states[0]["u_true"]))
    e3 = test._map_slice(model, states[0], bank["bank"], oracle=False,
                         counter=counter)
    srcs = [s["start_source"] for s in e3["starts"]]
    check("T16 E3 runs exactly the 8 locked starts",
          srcs == ["z0"] + [f"Z_DIAG[{k}]" for k in range(7)])
    grid = {"0", "25", "50", "75", "100", "125", "150", "175", "200"}
    decomp = all(
        s["initial_total_log_density"]
        == s["initial_log_pz"] + s["initial_logabsdet"]
        for s in e3["starts"] if s["finite"])
    check("T16 E3 trajectories on the locked grid + decomposition exact",
          set(e3["starts"][0]["trajectory"].keys()) == grid and decomp)
    e4 = test._map_slice(model, states[0], bank["bank"], oracle=True,
                         counter=counter)
    check("T16 E4 runs the SAME 8 locked starts",
          [s["start_source"] for s in e4["starts"]] == srcs
          and set(e4["starts"][0]["trajectory"].keys()) == grid)
    probes = test.jvp_probes()
    j1 = test.jvp_slice(model, states[0], probes["probes"], counter)
    j2 = test.jvp_slice(model, states[0], probes["probes"], counter)
    check("T16 JVP deterministic, 16 probes per slice",
          j1["q"] == j2["q"] and len(j1["q"]) == 16)
    ctx = treplay.ReplayContext(model=model, states=states, selection={},
                                spline_b=1.0, s_ref=1.0)
    e0 = [test.e0_slice(model, st) for st in states]
    r0_stub = {"endpoints": {"final": {"per_slice": [
        {"identity": r["identity"], "psnr_z0": r["psnr"],
         "nmse_u_z0": r["nmse_u"]} for r in e0]}}}
    d1 = test.run_d1(ctx, r0_stub)
    check("T16 run_d1 happy path: blocks + E0 gate + E4 non-routing",
          all(k in d1 for k in ("estimators", "aggregate", "thresholds",
                                "decision", "jvp", "runtime",
                                "e0_r0_equivalence"))
          and d1["e0_r0_equivalence"]["equal"] is True
          and d1["estimators"]["E4"]["routing"].startswith(
              "diagnostic_only"))
    saved_code, saved_env = tfacts.code_record, tfacts.environment_record
    tfacts.code_record = lambda repo: {"fixture": "isolated"}  # noqa: E731
    tfacts.environment_record = lambda *a, **k: {"fixture": True}  # noqa: E731
    try:
        tiny = _tiny_facts_stub()
        impl = {"schema": "seqref-impl-facts/1",
                "semantic_sha256": "f" * 64, "verdict": "PASS"}
        parents = {"parents_id": "fixture", "p0": {}, "p0s": {}}
        f1 = tfacts.build_d1_facts(_r0_result_stub(), d1, tiny, "9" * 64,
                                   impl, "e" * 64, parents, {}, {}, {},
                                   15.62704, "/nonexistent-repo", ["x"])
        f2 = tfacts.build_d1_facts(_r0_result_stub(), d1, tiny, "9" * 64,
                                   impl, "e" * 64, parents, {}, {}, {},
                                   15.62704, "/nonexistent-repo", ["x"])
    finally:
        tfacts.code_record, tfacts.environment_record = (saved_code,
                                                         saved_env)

    def _has_verdict(node):
        if isinstance(node, dict):
            return any(k == "verdict" or _has_verdict(v)
                       for k, v in node.items())
        if isinstance(node, list):
            return any(_has_verdict(v) for v in node)
        return False

    check("T16 D1 facts: recursive no-verdict, D1 complete, semantic "
          "stable",
          not _has_verdict(f1["d1"]) and "verdict" not in f1
          and f1["completeness"] == {"R0": "complete", "D1": "complete",
                                     "D2": "pending", "D3": "pending"}
          and f1["run_mode"] == "validation-r0-d1"
          and f1["semantic_sha256"] == f2["semantic_sha256"])


# ---------------------------------------------------------------------------
# T17 -- D1 descriptive figures (non-evidence; typed failure, never skip)
# ---------------------------------------------------------------------------

def _d1_fig_stub() -> dict:
    ids = [{"file": "a", "slice_index": 1},
           {"file": "b", "slice_index": 2}]

    def rows(p, n):
        return [{"identity": i, "psnr": p, "nmse_u": n} for i in ids]

    traj = {str(k): float(k) for k in (0, 100, 200)}
    starts = [{"winner": True, "trajectory": traj},
              {"winner": False, "trajectory": traj}]
    map_rows = [{"identity": i, "psnr": 11.0, "nmse_u": 0.3,
                 "starts": starts, "winner_start_index": 0,
                 "nonfinite_count": 0} for i in ids]
    return {"aggregate": {k: {"mean_psnr": 10.0, "mean_nmse_u": 0.4}
                          for k in ("E0", "E1", "E2", "E3", "E4")},
            "thresholds": {"E0_plus_2db": 12.0, "E0_half_nmse_u": 0.2},
            "estimators": {"E0": {"per_slice": rows(10.0, 0.4)},
                           "E1": {"per_slice": rows(11.0, 0.3)},
                           "E2": {"per_slice": rows(10.5, 0.35)},
                           "E3": {"per_slice": map_rows},
                           "E4": {"per_slice": map_rows}},
            "jvp": {"per_slice": [{"identity": i, "q": [1.0, 2.0, 3.0],
                                   "sqrt_mean_q": 1.5} for i in ids]}}


def t17_d1_figures() -> None:
    with tempfile.TemporaryDirectory() as td_:
        paths = tplots.render_d1_figures(_d1_fig_stub(), td_)
        check("T17 four D1 figures rendered non-empty",
              len(paths) == 4
              and all(os.path.isfile(p) and os.path.getsize(p) > 0
                      for p in paths), f"{paths}")
        expect_stage_error(
            "T17 broken D1 payload -> D1_PLOT_FAILURE",
            lambda: tplots.render_d1_figures({"broken": True}, td_),
            "D1_PLOT_FAILURE")


# ---------------------------------------------------------------------------
# T18 -- gradient hygiene regression (2026-08-18 repair)
# ---------------------------------------------------------------------------

class _StubModelParam(_StubModel):
    """Stub with ONE trainable parameter inside the decode path, so
    E3/E4 backward WOULD populate its grad if the D1 handoff freeze were
    missing."""

    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(1))

    def decode_scalars(self, z, cond_in, mask):
        return (z.to(torch.float32) * self.s + self.b) * self.gain


def t18_gradient_hygiene() -> None:
    model = _StubModelParam()
    states = _stub_states(1)
    ctx = treplay.ReplayContext(model=model, states=states, selection={},
                                spline_b=1.0, s_ref=1.0)
    e0 = test.e0_slice(model, states[0])
    r0_stub = {"endpoints": {"final": {"per_slice": [
        {"identity": e0["identity"], "psnr_z0": e0["psnr"],
         "nmse_u_z0": e0["nmse_u"]}]}}}
    d1 = test.run_d1(ctx, r0_stub)
    check("T18 D1 freezes the handoff model (param grads stay None)",
          model.gain.grad is None
          and not model.gain.requires_grad)
    moved = any(
        s["finite"]
        and s["final_z_norm"] != s["initial_z_norm"]
        for s in d1["estimators"]["E4"]["per_slice"][0]["starts"])
    check("T18 E3/E4 still update z through the frozen model", moved)


# ---------------------------------------------------------------------------
# D2a fixtures (2026-08-19): state-swap identity, Gaussian identity +
# percentile conventions, z_true encode/hash, hygiene, facts, figures.
# ---------------------------------------------------------------------------

class _StubModelState(_StubModel):
    """Stub whose decode parameters are REGISTERED buffers, so
    state_dict/capture_state/state_hash exercise the real hash path
    (the plain _StubModel attributes never enter state_dict)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("s_r", self.s.clone())
        self.register_buffer("b_r", self.b.clone())

    def decode_scalars(self, z, cond_in, mask):
        return z.to(torch.float32) * self.s_r + self.b_r


def _d2a_setup(n: int = 2):
    """Consistent D2a fixture context: stub model with registered state,
    an r0 stub whose step0/step500 hashes match the captured states, and
    the REAL D1 block (run_d1 on the same states). state0 is a scaled
    copy of the step-500 state -- a different, hash-verifiable state."""
    model = _StubModelState()
    states = _stub_states(n)
    e0 = [test.e0_slice(model, st) for st in states]
    state500 = treplay.capture_state(model)
    state0 = {k: v * 1.01 for k, v in state500.items()}
    r0_stub = {"endpoints": {"final": {"per_slice": [
        {"identity": r["identity"], "psnr_z0": r["psnr"],
         "nmse_u_z0": r["nmse_u"]} for r in e0]}},
        "step0_state_hash": treplay.state_hash(state0),
        "step500_state_hash": treplay.state_hash(state500)}
    ctx = treplay.ReplayContext(model=model, states=states, selection={},
                                spline_b=1.0, s_ref=1.0, state0=state0)
    d1 = test.run_d1(ctx, r0_stub)
    return model, states, ctx, r0_stub, d1


def t19_state_swap_identity() -> None:
    model, states, ctx, r0_stub, d1 = _d2a_setup()
    block = td2a.run_d2a(ctx, r0_stub, d1)
    si = block["state_identity"]
    check("T19 happy path: all four swap boundaries verified",
          all(si[k]["equal"] for k in ("pre_swap_step500", "step0_loaded",
                                       "step500_restored",
                                       "post_measurement_step500")))
    check("T19 the step-0 state is discarded after D2a",
          ctx.state0 is None)
    _, _, ctx2, r0_stub2, d1_2 = _d2a_setup()
    r0_bad = {**r0_stub2, "step500_state_hash": "0" * 64}
    expect_stage_error(
        "T19 tampered step-500 hash -> D2A_STATE_MISMATCH",
        lambda: td2a.run_d2a(ctx2, r0_bad, d1_2), "D2A_STATE_MISMATCH")
    _, _, ctx3, r0_stub3, d1_3 = _d2a_setup()
    r0_bad0 = {**r0_stub3, "step0_state_hash": "0" * 64}
    expect_stage_error(
        "T19 tampered step-0 hash -> D2A_STATE_MISMATCH",
        lambda: td2a.run_d2a(ctx3, r0_bad0, d1_3), "D2A_STATE_MISMATCH")
    _, _, ctx4, r0_stub4, d1_4 = _d2a_setup()
    ctx4.state0 = None
    expect_stage_error(
        "T19 missing state0 -> D2A_STATE0_MISSING",
        lambda: td2a.run_d2a(ctx4, r0_stub4, d1_4), "D2A_STATE0_MISSING")


def t20_gaussian_identity_percentile() -> None:
    rng = np.random.Generator(np.random.PCG64(23))
    vecs = [rng.standard_normal(tinv.Z_DIAG_N).astype(np.float32)
            for _ in range(3)]
    worst = td2a._gaussian_identity_check(vecs)
    check("T20 production log-prob matches the analytic identity",
          worst <= tinv.GAUSS_LOGPROB_CHECK_TOL, f"worst={worst!r}")
    saved = td.tg.ffr._gaussian_logprob
    td.tg.ffr._gaussian_logprob = (
        lambda z: -0.5 * z.pow(2).sum(-1))  # dropped -d/2 log(2pi)
    try:
        expect_stage_error(
            "T20 a wrong production formula is caught",
            lambda: td2a._gaussian_identity_check(vecs),
            "D2A_GAUSSIAN_IDENTITY_MISMATCH")
    finally:
        td.tg.ffr._gaussian_logprob = saved
    bank = np.array([-10.0, -5.0, -5.0, -1.0])
    r_below = td2a.bank_percentile(-11.0, bank)
    check("T20 below-all -> rank 0, fraction 0.0",
          r_below["rank_le_count"] == 0
          and r_below["percentile_fraction"] == 0.0)
    r_above = td2a.bank_percentile(0.0, bank)
    check("T20 above-all -> rank n, fraction 1.0, percent 100",
          r_above["rank_le_count"] == 4
          and r_above["percentile_fraction"] == 1.0
          and r_above["percentile_percent"] == 100.0)
    r_tie = td2a.bank_percentile(-5.0, bank)
    check("T20 exact tie counts ALL tied members (<= rule)",
          r_tie["rank_le_count"] == 3)
    check("T20 percentile record fields + frozen tie rule",
          set(r_tie) == {"bank_n", "rank_le_count", "percentile_fraction",
                         "percentile_percent", "tie_rule"}
          and r_tie["bank_n"] == 4
          and r_tie["tie_rule"] == tinv.D2A_PERCENTILE_TIE_RULE)


def t21_ztrue_encode() -> None:
    model = _StubModel()
    st = _stub_states(1)[0]
    z1 = td2a.z_true_slice(model, st)
    t32 = np.ascontiguousarray(st["target"], dtype=np.float32)
    expected = (torch.from_numpy(t32)
                * torch.exp(model.flow.log_scale)).numpy()[0]
    check("T21 z_true matches the stub closed form BITWISE",
          np.array_equal(z1, expected))
    z2 = td2a.z_true_slice(model, st)
    check("T21 z_true deterministic + sha over f32 C-order bytes",
          np.array_equal(z1, z2)
          and td2a._z_sha(z1) == hashlib.sha256(
              np.ascontiguousarray(z1, dtype=np.float32)
              .tobytes(order="C")).hexdigest()
          and len(td2a._z_sha(z1)) == 64)
    st_nan = dict(st)
    bad = st["target"].copy()
    bad[0, 5] = np.nan
    st_nan["target"] = bad
    expect_stage_error(
        "T21 non-finite target -> D2A_Z_TRUE_NON_FINITE",
        lambda: td2a.z_true_slice(model, st_nan), "D2A_Z_TRUE_NON_FINITE")
    st_missing = {k: v for k, v in st.items() if k != "target"}
    expect_stage_error(
        "T21 missing target -> D2A_TARGET_MISSING",
        lambda: td2a.z_true_slice(model, st_missing),
        "D2A_TARGET_MISSING")


def _no_key(node, pred) -> bool:
    if isinstance(node, dict):
        return all(not pred(k) and _no_key(v, pred)
                   for k, v in node.items())
    if isinstance(node, list):
        return all(_no_key(v, pred) for v in node)
    return True


def t22_d2a_hygiene() -> None:
    model, states, ctx, r0_stub, d1 = _d2a_setup()
    block = td2a.run_d2a(ctx, r0_stub, d1)
    check("T22 D2a leaves the model at the registered step-500 state",
          treplay.state_hash(treplay.capture_state(model))
          == r0_stub["step500_state_hash"])
    check("T22 no verdict/pattern keys anywhere in the D2a block",
          _no_key(block, lambda k: k == "verdict"
                  or str(k).startswith("pattern"))
          and block["routing"].startswith("descriptive_mechanistic_only"))
    _, states3, ctx3, r0_stub3, d1_3 = _d2a_setup()
    ctx3.states = list(reversed(states3))
    expect_stage_error(
        "T22 slice-order drift -> D2A_SLICE_ORDER_MISMATCH",
        lambda: td2a.run_d2a(ctx3, r0_stub3, d1_3),
        "D2A_SLICE_ORDER_MISMATCH")
    _, _, ctx4, r0_stub4, d1_4 = _d2a_setup()
    d1_bad = dict(d1_4)
    d1_bad["z_diag"] = {**d1_4["z_diag"], "manifest_sha256": "0" * 64}
    expect_stage_error(
        "T22 bank-manifest drift -> D2A_BANK_MISMATCH",
        lambda: td2a.run_d2a(ctx4, r0_stub4, d1_bad), "D2A_BANK_MISMATCH")


def t23_d2a_facts() -> None:
    model, states, ctx, r0_stub, d1 = _d2a_setup()
    block = td2a.run_d2a(ctx, r0_stub, d1)
    check("T23 D2a block carries all top-level sections",
          all(k in block for k in
              ("spec", "routing", "z_true_rule", "state_identity",
               "bank_reference", "gaussian_identity", "slices",
               "global_topk_drift", "runtime")))
    stat_keys = {"mean", "std", "rms", "mean_abs", "median", "q05",
                 "q25", "q75", "q95", "min", "max", "max_abs"}
    delta_keys = {"delta_norm_z", "delta_norm_z_squared", "delta_log_pz",
                  "norm_ratio_500_over_0", "cosine_similarity_z0_z500",
                  "delta_z_l2", "delta_z_rms", "top_k_drift",
                  "top_k_rule"}
    s0 = block["slices"][0]
    check("T23 per-slice step records carry the frozen stat/delta sets",
          set(s0["step0"]["coordinate_stats"]) == stat_keys
          and set(s0["step500"]["coordinate_stats"]) == stat_keys
          and set(s0["delta"]) == delta_keys
          and set(s0["step0"]["percentile"])
          == {"bank_n", "rank_le_count", "percentile_fraction",
              "percentile_percent", "tie_rule"})
    check("T23 top-K drift: 20 entries, exact fields",
          len(s0["delta"]["top_k_drift"]) == tinv.D2A_TOP_K
          and set(s0["delta"]["top_k_drift"][0])
          == {"coordinate_index", "z_step0", "z_step500", "delta",
              "abs_delta"})
    g = block["global_topk_drift"]
    check("T23 global top-K: 20 indices x per-slice signed deltas",
          len(g["coordinate_indices"]) == tinv.D2A_TOP_K
          and len(g["delta_matrix"]) == len(states)
          and all(len(row) == tinv.D2A_TOP_K for row in g["delta_matrix"]))
    saved_code, saved_env = tfacts.code_record, tfacts.environment_record
    tfacts.code_record = lambda repo: {"fixture": "isolated"}  # noqa: E731
    tfacts.environment_record = lambda *a, **k: {"fixture": True}  # noqa: E731
    try:
        tiny = _tiny_facts_stub()
        impl = {"schema": "seqref-impl-facts/1",
                "semantic_sha256": "f" * 64, "verdict": "PASS"}
        parents = {"parents_id": "fixture", "p0": {}, "p0s": {}}
        f1 = tfacts.build_d2_facts(_r0_result_stub(), d1, block, tiny,
                                   "9" * 64, impl, "e" * 64, parents,
                                   {}, {}, {}, 15.62704,
                                   "/nonexistent-repo", ["x"])
        f2 = tfacts.build_d2_facts(_r0_result_stub(), d1, block, tiny,
                                   "9" * 64, impl, "e" * 64, parents,
                                   {}, {}, {}, 15.62704,
                                   "/nonexistent-repo", ["x"])
    finally:
        tfacts.code_record, tfacts.environment_record = (saved_code,
                                                         saved_env)
    check("T23 D2 facts: nested d2.completeness, top-level D2 partial",
          f1["completeness"] == {"R0": "complete", "D1": "complete",
                                 "D2": "partial", "D3": "pending"}
          and f1["d2"]["completeness"] == {"D2a": "complete",
                                           "D2b": "pending",
                                           "D2c": "pending"}
          and f1["run_mode"] == "validation-r0-d1-d2a")
    check("T23 D2 facts: recursive no-verdict + stable semantic hash",
          "verdict" not in f1
          and _no_key(f1["d1"], lambda k: k == "verdict")
          and _no_key(f1["d2"], lambda k: k == "verdict")
          and f1["semantic_sha256"] == f2["semantic_sha256"])


def t24_d2a_figures() -> None:
    _, _, ctx, r0_stub, d1 = _d2a_setup()
    block = td2a.run_d2a(ctx, r0_stub, d1)
    with tempfile.TemporaryDirectory() as td_:
        paths = td2aplots.render_d2a_figures(block, td_)
        check("T24 three D2a figures rendered non-empty",
              len(paths) == 3
              and all(os.path.isfile(p) and os.path.getsize(p) > 0
                      for p in paths), f"{paths}")
        expect_stage_error(
            "T24 broken D2a payload -> D2A_PLOT_FAILURE",
            lambda: td2aplots.render_d2a_figures({"broken": True}, td_),
            "D2A_PLOT_FAILURE")


# ---------------------------------------------------------------------------

EXPECTED_COUNTS = {  # static registry: a green suite cannot shrink
    "t1_taxonomy_purity": 3,
    "t2_deferred_probe_guard": 4,
    "t3_comparison_engine": 6,
    "t3b_live_parent_hash_comparisons": 2,
    "t4_tiny_parent_dual_pin": 5,
    "t5_trace_completeness": 1,
    "t6_state_hash_determinism": 3,
    "t7_facts_schema_purity": 4,
    "t8_publication_and_error_taxonomy": 12,
    "t9_startup_logging_robustness": 4,
    "t10_preflight_module_identity": 11,
    "t11_locked_banks": 5,
    "t12_e0_r0_equivalence_gate": 4,
    "t13_winner_selection": 6,
    "t14_aggregation_slice_set": 4,
    "t15_materiality_bands": 5,
    "t16_estimator_conventions": 8,
    "t17_d1_figures": 2,
    "t18_gradient_hygiene": 2,
    "t19_state_swap_identity": 5,
    "t20_gaussian_identity_percentile": 6,
    "t21_ztrue_encode": 4,
    "t22_d2a_hygiene": 4,
    "t23_d2a_facts": 6,
    "t24_d2a_figures": 2,
}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    fixtures = [t1_taxonomy_purity, t2_deferred_probe_guard,
                t3_comparison_engine, t3b_live_parent_hash_comparisons,
                t4_tiny_parent_dual_pin,
                t5_trace_completeness, t6_state_hash_determinism,
                t7_facts_schema_purity,
                t8_publication_and_error_taxonomy,
                t9_startup_logging_robustness,
                t10_preflight_module_identity, t11_locked_banks,
                t12_e0_r0_equivalence_gate, t13_winner_selection,
                t14_aggregation_slice_set, t15_materiality_bands,
                t16_estimator_conventions, t17_d1_figures,
                t18_gradient_hygiene, t19_state_swap_identity,
                t20_gaussian_identity_percentile, t21_ztrue_encode,
                t22_d2a_hygiene, t23_d2a_facts, t24_d2a_figures]
    counts_ok = True
    for fn in fixtures:
        before = len(RESULTS)
        fn()
        want = EXPECTED_COUNTS[fn.__name__]
        got = len(RESULTS) - before
        if got != want:
            counts_ok = False
            logger.error("[%s] coverage shrinkage: %s emitted %d checks, "
                         "registry pins %d", SCRIPT_ID, fn.__name__, got,
                         want)
    total = len(RESULTS)
    if total != EXPECTED_TOTAL:
        counts_ok = False
        logger.error("[%s] coverage shrinkage: suite emitted %d checks, "
                     "registry pins %d", SCRIPT_ID, total, EXPECTED_TOTAL)
    failed = [r for r in RESULTS if not r[1]]
    coverage_ok = not failed and counts_ok
    for name, ok, detail in RESULTS:
        logger.info("[%s] %s %s%s", SCRIPT_ID, "PASS" if ok else "FAIL",
                    name, f" -- {detail}" if (detail and not ok) else "")
    logger.info("[%s] fixtures: %d/%d PASS, coverage_ok=%s", SCRIPT_ID,
                total - len(failed), total, str(coverage_ok).lower())
    if failed or not counts_ok:
        if failed:
            logger.error("[%s] %d fixture(s) failed -- driver repair "
                         "required before any TDIAG execution", SCRIPT_ID,
                         len(failed))
        if not counts_ok:
            logger.error("[%s] coverage registry mismatch -- refusing "
                         "green exit", SCRIPT_ID)
        return 2
    logger.info("[%s] all fixtures green; tdiag v0.1 R0+D1+D2a-slice "
                "contracts hold", SCRIPT_ID)
    return 0


if __name__ == "__main__":
    sys.exit(main())

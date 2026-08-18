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
# Coverage registry: EXPECTED_COUNTS pins the check count of every
#   fixture plus the suite total; coverage_ok requires zero failures AND
#   exact count matches, so a green suite cannot silently shrink.
# Invocation: both `python seqref_mri/scripts/tdiag_selftest.py` and
#   `python -m seqref_mri.scripts.tdiag_selftest` are supported.
# Taxonomy: all fixtures PASS -> exit 0; any failure -> exit 2 (a failing
#   fixture is a construction/contract defect, ERROR class under LOCK 2;
#   never a scientific result). No fallback, no mock, no placeholder, no
#   silent pass: every failure path is logger.error + typed outcome.
# Changelog (NEW in v0.1):
#   * Introduced with the R0 slice after the 2026-08-15 EXEC SS10.6 lock.
#   * Review-repair round (2026-08-16, pre-execution; NO contract
#     change): all compare_registered call sites pass the freshly
#     verified live parent identities; new T3b fixture proves a wrong
#     live IMPL semantic sha or TINY file sha fails exactly its row, so
#     the tautological comparison form can never return.
# Update summary:
#   v0.1 pins the R0-slice contracts as executable regressions: the
#   standalone 0/2 taxonomy, the amendment-gated deferred-probe guard,
#   the exact serialized-value comparison engine, the TINY dual-pin
#   refusal paths, trace-grid completeness, canonical state hashing, the
#   no-verdict facts schema and both ERROR-context boundaries, under a
#   static expected-count coverage registry.
# =============================================================================
from __future__ import annotations

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
    from seqref_mri.tdiag import facts as tfacts
    from seqref_mri.tdiag import invariants as tinv
    from seqref_mri.tdiag import replay as treplay
else:  # direct script run: scripts/ is on sys.path; tdiag sets repo paths
    import tdiag as td
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
        _patch("run_r0", lambda *a, **k: _r0_result_stub(), td.replay)
        with tempfile.TemporaryDirectory() as td_:
            rc = td.main(base_args + ["--data-root", td_,
                                      "--out-dir", td_] + parents_args)
            check("T8 valid R0 replay -> exit 0 (report)", rc == 0)
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
        _patch("run_r0", lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("injected trusted-context failure")), td.replay)
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
          and tinv.StageError is pp.StageError)
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
    for mod in (treplay, tfacts, tinv):
        with open(mod.__file__, "r", encoding="utf-8") as fh:
            src = fh.read()
        boot_at = src.find("from seqref_mri.tdiag import _bootstrap")
        preflight_at = src.find("from preflight_")
        check(f"T10 {mod.__name__.rsplit('.', 1)[-1]} bootstraps before "
              f"preflight imports",
              0 <= boot_at < preflight_at)


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
    "t8_publication_and_error_taxonomy": 7,
    "t9_startup_logging_robustness": 4,
    "t10_preflight_module_identity": 7,
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
                t10_preflight_module_identity]
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
    logger.info("[%s] all fixtures green; tdiag v0.1 R0-slice contracts "
                "hold", SCRIPT_ID)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# SEQREF-TDIAG v0.1 -- tdiag.facts
# LIFETIME: KEEP
# =============================================================================
# Purpose: seqref-tdiag-facts/1 assembly for the TDIAG diagnostic stage
#          (EXEC SS10.6). TDIAG is EVIDENCE-ONLY: the facts carry NO
#          verdict field and no gate outcome -- only measurements,
#          comparisons against frozen bands (D1-D3, as they land), the R0
#          replay-validity record, provenance and hashes.
#          The R0-only builder assembles the replay-validity partial
#          report; the D1 builder extends it with the estimator-slate
#          block (E0-E4 + JVP, frozen-band materiality, decision
#          fields); the D2 builder nests the D2a latent-geometry block
#          under d2 with its own D2a/D2b/D2c sub-completeness (top-level
#          D2 stays "partial" until D2a+b+c all land); the D3 builder
#          adds the conditioner-sensitivity block and flips D3 to
#          complete (the full R0/D1/D2/D3 diagnostic suite).
# Publication: seqref_mri/results/_diag/diag/tdiag_facts.json under the
#   campaign claim/publish/sidecar machinery; reruns write a stamped
#   sibling, never overwrite.
# CONVENTION: logger.error + typed raise. No fallback, no mock, no
#   placeholder, no silent pass.
# Changelog (NEW in v0.1):
#   * Introduced with the R0 slice after the 2026-08-15 EXEC SS10.6 lock.
#   * D1 slice (2026-08-18, under the same SS10.6 lock; NO contract
#     change): build_d1_facts extends the R0 report with the D1 block,
#     flips completeness to D1 complete (D2/D3 pending), run_mode
#     validation-r0-d1; a RECURSIVE no-verdict scan now covers the D1
#     block; estimators.py and d1_plots.py joined the code record.
#   * D2a slice (2026-08-19, under the same SS10.6 lock; NO contract
#     change): build_d2_facts nests d2.completeness (D2a complete;
#     D2b/D2c pending) with top-level D2 "partial", run_mode
#     validation-r0-d1-d2a; d2a.py and d2a_plots.py joined the code
#     record.
#   * D2b slice (2026-08-19, under the same SS10.6 lock; NO contract
#     change): build_d2b_facts adds the D2b decomposition block, flips
#     d2.completeness.D2b to complete (top-level D2 stays "partial"
#     until D2c), run_mode validation-r0-d1-d2a-d2b; d2b.py and
#     d2b_plots.py joined the code record.
#   * D2c slice (2026-08-20, under the same SS10.6 lock; NO contract
#     change): build_d2c_facts adds the D2c holdout block, flips D2c
#     and the top-level D2 to complete (D3 pending), run_mode
#     validation-r0-d1-d2a-d2b-d2c; d2c.py and d2c_plots.py joined the
#     code record.
#   * D3 slice (2026-08-20, under the same SS10.6 lock; NO contract
#     change): build_d3_facts adds the D3 conditioner-sensitivity block,
#     flips D3 to complete (ALL of R0/D1/D2/D3 complete; the report
#     becomes the TDIAG D1-D3 suite closure -- D4/D5/D6 remain
#     amendment-gated), run_mode validation-r0-d1-d2a-d2b-d2c-d3; d3.py
#     and d3_plots.py joined the code record.
# Update summary:
#   v0.1 lands the R0 partial evidence assembly plus the D1, D2a, D2b,
#   D2c and D3 extensions: completeness tracking (with the nested D2
#   sub-block), recursive no-verdict schema invariant, TDIAG code
#   record (all measurement and figure modules through D3) and the
#   campaign semantic-hash attachment (run/ excluded as volatile).
# =============================================================================
from __future__ import annotations

import logging
import os
import sys

from seqref_mri.tdiag import _bootstrap  # noqa: F401

from preflight_io import file_sha256
from preflight_parents import (StageError, attach_semantic_hash,
                               environment_record, hash_project_code)

logger = logging.getLogger("SEQREF-TDIAG")

FACTS_SCHEMA = "seqref-tdiag-facts/1"
FACTS_PREFIX = "tdiag_facts"
ERROR_PREFIX = "tdiag_error"
ERROR_SCHEMA = "seqref-tdiag-error/1"
STAGE = "TDIAG"

# Every project-local file whose bytes pin the executing TDIAG path: the
# stage's own driver/selftest/package PLUS the reused TINY construction
# and every module the production path executes (same doctrine as
# tiny_gate.TINY_LOCAL_FILES).
TDIAG_LOCAL_FILES = [
    "seqref_mri/scripts/tdiag.py",
    "seqref_mri/scripts/tdiag_selftest.py",
    "seqref_mri/tdiag/__init__.py",
    "seqref_mri/tdiag/_bootstrap.py",
    "seqref_mri/tdiag/invariants.py",
    "seqref_mri/tdiag/replay.py",
    "seqref_mri/tdiag/estimators.py",
    "seqref_mri/tdiag/d1_plots.py",
    "seqref_mri/tdiag/d2a.py",
    "seqref_mri/tdiag/d2a_plots.py",
    "seqref_mri/tdiag/d2b.py",
    "seqref_mri/tdiag/d2b_plots.py",
    "seqref_mri/tdiag/d2c.py",
    "seqref_mri/tdiag/d2c_plots.py",
    "seqref_mri/tdiag/d3.py",
    "seqref_mri/tdiag/d3_plots.py",
    "seqref_mri/tdiag/facts.py",
    "seqref_mri/scripts/tiny_gate.py",
    "seqref_mri/scripts/tiny_selftest.py",
    "seqref_mri/src/free_flow_runtime.py",
    "seqref_mri/scripts/train_free_flow.py",
    "seqref_mri/src/conditioner.py",
    "seqref_mri/src/flows/nsf_layer.py",
    "seqref_mri/src/fastmri_data.py",
    "seqref_mri/src/residual_decoder.py",
    "seqref_mri/src/metrics.py",
    "seqref_mri/src/preflight_parents_p3.py",
    "seqref_mri/scripts/train_base.py",
]


def _fail(code: str, message: str, **kwargs) -> StageError:
    logger.error("[SEQREF-TDIAG] %s: %s", code, message)
    return StageError(code, message, **kwargs)


def _path_free(rec):
    if isinstance(rec, dict):
        return {k: _path_free(v) for k, v in rec.items() if k != "path"}
    if isinstance(rec, list):
        return [_path_free(v) for v in rec]
    return rec


def code_record(repo: str) -> dict:
    code = dict(hash_project_code(repo, os.path.abspath(
        os.path.join(repo, "seqref_mri", "scripts", "tdiag.py"))))
    hashed = []
    for rel in TDIAG_LOCAL_FILES:
        path = os.path.join(repo, rel)
        if not os.path.isfile(path):
            raise _fail("CODE_HASH_FILE_MISSING",
                        f"project-local file required for the TDIAG code "
                        f"hash is missing: {rel}")
        hashed.append({"relpath": rel, "sha256": file_sha256(path)})
    code["tdiag_local"] = hashed
    code["tdiag_local_note"] = (
        "the TDIAG stage's own driver/selftest/package plus the reused "
        "TINY construction and every module the production path "
        "executes; the frozen project hash block covers the preflight "
        "core")
    return code


def build_r0_facts(r0: dict, tiny_facts: dict, tiny_file_sha: str,
                   impl: dict, impl_file_sha: str, parents: dict,
                   p3: dict, p4: dict, implb: dict, s_ref: float,
                   repo: str, argv) -> dict:
    """Assemble the R0-only partial evidence report. INVARIANT: the facts
    carry no 'verdict' key -- TDIAG emits evidence, never a gate
    outcome."""
    parents_rec = {
        "parents_id": parents.get("parents_id"),
        "p0": _path_free(parents.get("p0")),
        "p0s": _path_free(parents.get("p0s")),
        "p3_coordinate_map": _path_free({k: v for k, v in p3.items()
                                         if k != "bindings"}),
        "p4_scaling_statistics": _path_free({k: v for k, v in p4.items()
                                             if k != "location_index"}),
        "implb_calibration": _path_free(implb),
        "impl_class_a": {"schema": impl["schema"],
                         "file_sha256": impl_file_sha,
                         "semantic_sha256": impl["semantic_sha256"],
                         "verdict": impl["verdict"]},
        "tiny_authoritative": {"schema": tiny_facts["schema"],
                               "file_sha256": tiny_file_sha,
                               "semantic_sha256":
                                   tiny_facts["semantic_sha256"],
                               "verdict": tiny_facts["verdict"]},
        "s_ref": {"value": s_ref, "source": "verified P0S artefact",
                  "used_for": "R_FREE_MIN exclusion ratios only"},
    }
    facts = {
        "schema": FACTS_SCHEMA,
        "script": {"id": "SEQREF-TDIAG", "version": "v0.1",
                   "lifetime": "KEEP"},
        "stage": STAGE,
        "artefact_type": "diagnostic_evidence",
        "report_status": ("partial -- R0 replay validity only; D1/D2/D3 "
                          "pending implementation"),
        "run_mode": "validation-r0-only",
        "authoritative": False,
        "spec": "EXEC SS10.6 (SEQREF-TDIAG v0.1, locked 2026-08-15 "
                "pre-implementation)",
        "verdict_note": ("TDIAG emits an evidence report only; no "
                         "PASS/BLOCK verdict exists in this schema, the "
                         "report cannot unblock PILOT/SCREEN/FORMAL and "
                         "never converts the TINY BLOCK into PASS"),
        "completeness": {"R0": "complete", "D1": "pending",
                         "D2": "pending", "D3": "pending"},
        "r0": {
            "registered_source": {
                "schema": tiny_facts["schema"],
                "file_sha256": tiny_file_sha,
                "semantic_sha256": tiny_facts["semantic_sha256"]},
            "rule": "exact equality of the registered serialized values; "
                    "no tolerance (EXEC SS10.6 R0)",
            "valid": r0["valid"],
            "comparisons": r0["comparisons"],
            "nll_trace_replayed": r0["trace"],
            "step0_state_hash": r0["step0_state_hash"],
            "step500_state_hash": r0["step500_state_hash"],
            "replay_config_hash": r0["replay_config_hash"],
            "selection": r0["selection"],
            "endpoints": r0["endpoints"],
        },
        "parents": parents_rec,
        "dataset_provenance": {
            "split": "train", "mode": "eval",
            "population": r0["selection"]["population"],
            "selection_rule": ("TINY-registered PCG64(0) draw, re-derived "
                               "and compared against the authoritative "
                               "artefact manifest")},
        "code": code_record(repo),
        "run": {**environment_record(repo, argv),
                "hash_note": "file sha256 + sidecar; semantic sha256 "
                             "over the path-free semantic payload (run/ "
                             "excluded as volatile)"},
    }
    if "verdict" in facts:
        raise _fail("FACTS_SCHEMA_VIOLATION",
                    "a verdict key entered the TDIAG facts; the stage is "
                    "evidence-only by preregistration")
    semantic = {k: v for k, v in facts.items() if k != "run"}
    attach_semantic_hash(facts, semantic)
    return facts


def _no_verdict_scan(node, path: str) -> None:
    """Recursive evidence-only invariant: NO 'verdict' key may appear
    anywhere inside the D1 block (the top-level check alone would miss
    nested leakage)."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "verdict":
                raise _fail("FACTS_SCHEMA_VIOLATION",
                            f"a verdict key entered the TDIAG facts at "
                            f"{path}.verdict; the stage is evidence-only "
                            f"by preregistration")
            _no_verdict_scan(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _no_verdict_scan(v, f"{path}[{i}]")


def build_d1_facts(r0: dict, d1: dict, tiny_facts: dict,
                   tiny_file_sha: str, impl: dict, impl_file_sha: str,
                   parents: dict, p3: dict, p4: dict, implb: dict,
                   s_ref: float, repo: str, argv) -> dict:
    """Assemble the R0+D1 partial evidence report: the R0 report plus
    the D1 estimator-slate block. Completeness flips D1 to complete;
    run_mode validation-r0-d1; still NOT the authoritative TDIAG closure
    (D2/D3 pending). INVARIANT: no 'verdict' key anywhere, enforced by
    the top-level check (R0 builder) plus a recursive scan over the D1
    block."""
    _no_verdict_scan(d1, "d1")
    facts = build_r0_facts(r0, tiny_facts, tiny_file_sha, impl,
                           impl_file_sha, parents, p3, p4, implb, s_ref,
                           repo, argv)
    facts["report_status"] = ("partial -- R0 replay validity + D1 "
                              "estimator slate; D2/D3 pending "
                              "implementation")
    facts["run_mode"] = "validation-r0-d1"
    facts["completeness"] = {"R0": "complete", "D1": "complete",
                             "D2": "pending", "D3": "pending"}
    facts["d1"] = d1
    semantic = {k: v for k, v in facts.items() if k != "run"}
    attach_semantic_hash(facts, semantic)
    return facts


def build_d2_facts(r0: dict, d1: dict, d2a: dict, tiny_facts: dict,
                   tiny_file_sha: str, impl: dict, impl_file_sha: str,
                   parents: dict, p3: dict, p4: dict, implb: dict,
                   s_ref: float, repo: str, argv) -> dict:
    """Assemble the R0+D1+D2a partial evidence report. The D2 block is
    NESTED: d2.completeness tracks D2a/D2b/D2c individually while the
    top-level D2 stays "partial" until D2a+b+c all land -- no new
    top-level field, no accidental contract change (review 2026-08-19).
    INVARIANT: no 'verdict' key anywhere, enforced by the top-level
    check (R0 builder) plus recursive scans over the D1 and D2a
    blocks."""
    _no_verdict_scan(d2a, "d2.d2a")
    facts = build_d1_facts(r0, d1, tiny_facts, tiny_file_sha, impl,
                           impl_file_sha, parents, p3, p4, implb, s_ref,
                           repo, argv)
    facts["report_status"] = ("partial -- R0 replay validity + D1 "
                              "estimator slate + D2a latent geometry; "
                              "D2b/D2c/D3 pending implementation")
    facts["run_mode"] = "validation-r0-d1-d2a"
    facts["completeness"] = {"R0": "complete", "D1": "complete",
                             "D2": "partial", "D3": "pending"}
    facts["d2"] = {"completeness": {"D2a": "complete", "D2b": "pending",
                                    "D2c": "pending"},
                   "d2a": d2a}
    semantic = {k: v for k, v in facts.items() if k != "run"}
    attach_semantic_hash(facts, semantic)
    return facts


def build_d2b_facts(r0: dict, d1: dict, d2a: dict, d2b: dict,
                    tiny_facts: dict, tiny_file_sha: str, impl: dict,
                    impl_file_sha: str, parents: dict, p3: dict,
                    p4: dict, implb: dict, s_ref: float, repo: str,
                    argv) -> dict:
    """Assemble the R0+D1+D2a+D2b partial evidence report: the D2a
    report plus the D2b decomposition block. d2.completeness flips D2b
    to complete; the top-level D2 stays "partial" until D2c lands;
    run_mode validation-r0-d1-d2a-d2b (the completed D2a stays visible
    in the mode string, review 2026-08-19). INVARIANT: no 'verdict'
    key anywhere, enforced by the top-level check (R0 builder) plus
    recursive scans over the D1/D2a/D2b blocks."""
    _no_verdict_scan(d2b, "d2.d2b")
    facts = build_d2_facts(r0, d1, d2a, tiny_facts, tiny_file_sha, impl,
                           impl_file_sha, parents, p3, p4, implb, s_ref,
                           repo, argv)
    facts["report_status"] = ("partial -- R0 replay validity + D1 "
                              "estimator slate + D2a latent geometry + "
                              "D2b likelihood decomposition; D2c/D3 "
                              "pending implementation")
    facts["run_mode"] = "validation-r0-d1-d2a-d2b"
    facts["d2"]["completeness"]["D2b"] = "complete"
    facts["d2"]["d2b"] = d2b
    semantic = {k: v for k, v in facts.items() if k != "run"}
    attach_semantic_hash(facts, semantic)
    return facts
def build_d2c_facts(r0: dict, d1: dict, d2a: dict, d2b: dict,
                    d2c: dict, tiny_facts: dict, tiny_file_sha: str,
                    impl: dict, impl_file_sha: str, parents: dict,
                    p3: dict, p4: dict, implb: dict, s_ref: float,
                    repo: str, argv) -> dict:
    """Assemble the R0+D1+D2a+D2b+D2c partial evidence report: the D2b
    report plus the D2c holdout-generalization block. d2.completeness
    flips D2c to complete and the TOP-LEVEL D2 flips to "complete"
    (D2a+b+c all landed); D3 stays pending; run_mode
    validation-r0-d1-d2a-d2b-d2c (review 2026-08-20). INVARIANT: no
    'verdict' key anywhere, enforced by the top-level check (R0
    builder) plus recursive scans over the D1/D2a/D2b/D2c blocks."""
    _no_verdict_scan(d2c, "d2.d2c")
    facts = build_d2b_facts(r0, d1, d2a, d2b, tiny_facts, tiny_file_sha,
                            impl, impl_file_sha, parents, p3, p4, implb,
                            s_ref, repo, argv)
    facts["report_status"] = ("partial -- R0 replay validity + D1 "
                              "estimator slate + D2a latent geometry + "
                              "D2b likelihood decomposition + D2c "
                              "holdout generalization; D3 pending "
                              "implementation")
    facts["run_mode"] = "validation-r0-d1-d2a-d2b-d2c"
    facts["completeness"]["D2"] = "complete"
    facts["d2"]["completeness"]["D2c"] = "complete"
    facts["d2"]["d2c"] = d2c
    semantic = {k: v for k, v in facts.items() if k != "run"}
    attach_semantic_hash(facts, semantic)
    return facts


def build_d3_facts(r0: dict, d1: dict, d2a: dict, d2b: dict,
                   d2c: dict, d3: dict, tiny_facts: dict,
                   tiny_file_sha: str, impl: dict, impl_file_sha: str,
                   parents: dict, p3: dict, p4: dict, implb: dict,
                   s_ref: float, repo: str, argv) -> dict:
    """Assemble the R0+D1+D2a+D2b+D2c+D3 evidence report: the D2c
    report plus the D3 conditioner-sensitivity block. ALL of R0/D1/D2/
    D3 flip to complete -- this is the TDIAG D1-D3 diagnostic-suite
    closure (D4/D5/D6 remain amendment-gated); run_mode
    validation-r0-d1-d2a-d2b-d2c-d3 (review 2026-08-20). INVARIANT: no
    'verdict' key anywhere, enforced by the top-level check (R0
    builder) plus recursive scans over the D1/D2/D3 blocks."""
    _no_verdict_scan(d3, "d3")
    facts = build_d2c_facts(r0, d1, d2a, d2b, d2c, tiny_facts,
                            tiny_file_sha, impl, impl_file_sha, parents,
                            p3, p4, implb, s_ref, repo, argv)
    facts["report_status"] = ("complete -- R0 replay validity + D1 "
                              "estimator slate + D2a latent geometry + "
                              "D2b likelihood decomposition + D2c "
                              "holdout generalization + D3 conditioner "
                              "sensitivity; the TDIAG D1-D3 diagnostic "
                              "suite is complete (deferred probes "
                              "D4/D5/D6 remain amendment-gated)")
    facts["run_mode"] = "validation-r0-d1-d2a-d2b-d2c-d3"
    facts["completeness"]["D3"] = "complete"
    facts["d3"] = d3
    semantic = {k: v for k, v in facts.items() if k != "run"}
    attach_semantic_hash(facts, semantic)
    return facts

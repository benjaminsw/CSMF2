# SEQREF-TDIAG v0.1 -- tdiag.facts
# LIFETIME: KEEP
# =============================================================================
# Purpose: seqref-tdiag-facts/1 assembly for the TDIAG diagnostic stage
#          (EXEC SS10.6). TDIAG is EVIDENCE-ONLY: the facts carry NO
#          verdict field and no gate outcome -- only measurements,
#          comparisons against frozen bands (D1-D3, as they land), the R0
#          replay-validity record, provenance and hashes.
#          This slice assembles the R0-only partial report; D1/D2/D3
#          blocks are added by later slices under the same schema, with
#          the completeness block tracking what is present.
# Publication: seqref_mri/results/_diag/diag/tdiag_facts.json under the
#   campaign claim/publish/sidecar machinery; reruns write a stamped
#   sibling, never overwrite.
# CONVENTION: logger.error + typed raise. No fallback, no mock, no
#   placeholder, no silent pass.
# Changelog (NEW in v0.1):
#   * Introduced with the R0 slice after the 2026-08-15 EXEC SS10.6 lock.
# Update summary:
#   v0.1 lands the R0-only partial evidence assembly: completeness block,
#   no-verdict schema invariant, TDIAG code record and the campaign
#   semantic-hash attachment (run/ excluded as volatile).
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

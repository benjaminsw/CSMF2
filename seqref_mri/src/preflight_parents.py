# SEQREF-PPAR v0.1 -- P1/P2 parent verification + shared stage machinery
# LIFETIME: KEEP
#
# Why this module exists, and why it is NEW rather than an addition to an
# existing shared module
#   Under EXEC v0.4 §9 any change to a stage's inputs invalidates that stage's
#   artefact. P0 and P0S have both PASSED. Adding parent-verification helpers
#   to preflight_io.py or parent_expectations.py -- which P0S imports -- would
#   have invalidated a passed gate for code it does not use. This module is
#   therefore imported by P1 onward and NEVER by P0 or P0S (EXEC v0.4 §8, A3).
#
# SCOPE NOTE (deviation to be recorded)
#   EXEC §8 A3 specifies this module as providing "parent verification for
#   both" P1 and P2. It additionally carries the SHARED STAGE MACHINERY those
#   two scripts must implement identically: the PASS/BLOCK/ERROR taxonomy, the
#   semantic-hash construction, project-local code hashing, and the
#   claim-guarded publication wrapper. The alternative was a sixth module (not
#   listed in §9) or duplicating ~150 lines across two scripts written in the
#   same amendment, which would drift. One module, one implementation.
#
# What is NOT re-implemented here
#   Canonical hashing, file hashing, sidecar verification, pairing checks and
#   publication all come from preflight_io. There is ONE canonicalisation rule
#   in this tree, and semantic_sha256 is computed through
#   preflight_io.canonical_hash so it cannot drift from the artefact rule.
#
# Publication exclusion (A3)
#   preflight_io.publish() derives its temporary names from os.getpid() alone.
#   That module is imported by the PASSED P0S, so editing it would invalidate
#   P0S under §9. Mutual exclusion is therefore provided HERE, and entirely
#   within this module: a FIXED-NAME claim path per prefix, acquired with
#   O_CREAT|O_EXCL so acquisition is ATOMIC and exactly one contender wins.
#   The PID, a UUID4 token and the timestamp are written INSIDE that file as
#   holder identity -- they identify the holder, they do not create the
#   exclusion. An earlier draft used a UUID-NAMED claim file: two contenders
#   would each create a DIFFERENT name, both pass the pre-scan and both
#   proceed, so it supplied collision resistance and no exclusion at all.
#
#   STALE CLAIM: a holder killed by SIGKILL leaves the fixed claim behind and
#   blocks every later same-prefix run. This is deliberate -- a silent
#   double-publication is worse than a halt -- and the in-file PID and
#   timestamp make the staleness diagnosable. Removal is an OPERATOR action;
#   nothing here reclaims a claim automatically, because "it looks old" is not
#   evidence the holder is dead.
#
# CONVENTION: every failure path -> logger.error + raise. No fallback, no mock,
#   no placeholder, no silent pass.
#
# Changelog
#   v0.1 (2026-07-30) Created under Amendment A3 for the parallel P1/P2
#     preflight stages. Verifies the P0 and P0S authoritative sidecars,
#     re-drives the live-code check from the LOCKED PROFILE rather than from
#     either parent record, and PROVES the S_ref median convention by
#     recomputing it from P0S's own per-entry energies instead of asserting a
#     descriptive string. Supplies the shared PASS/BLOCK/ERROR taxonomy so the
#     two stages cannot diverge on audit semantics.

from __future__ import annotations

import logging
import os
import sys
import uuid
from contextlib import contextmanager

import numpy as np

from preflight_io import (canonical_hash, check_pairing, file_sha256,
                          git_state, publish, utc_stamp, verify_sidecar)
from contract_hash import (ASSERT_PROCEDURE_ID, PROCEDURE_ID,
                           check_prepare_binding, contract_hash)
from normalisation_profile import (EXPECTED_ASSERTIONS, EXPECTED_CONTRACT,
                                   PROFILE_ID)
from parent_expectations import (EXPECTATIONS_ID, EXPECTED_SOURCE_FILE_COUNT,
                                 EXPECTED_SOURCE_ROOTS, validate_parent_scope,
                                 validate_parent_source_binding)

logger = logging.getLogger("seqref_mri.preflight_parents")

__version__ = "0.1"
__abbr__ = "SEQREF-PPAR"

PARENTS_ID = "seqref-preflight-parents/1"

P0_FACTS_SCHEMA = "seqref-p0-facts/2"
P0S_FACTS_SCHEMA = "seqref-p0s-facts/1"

# Project-local files whose content can affect a P1/P2 verdict. Third-party
# packages are recorded by NAME and VERSION only: content-hashing an installed
# package is not reproducible across environments and is already covered by the
# environment record.
CODE_HASH_FILES = [
    "seqref_mri/src/preflight_io.py",
    "seqref_mri/src/contract_hash.py",
    "seqref_mri/src/normalisation_profile.py",
    "seqref_mri/src/parent_expectations.py",
    "seqref_mri/src/preflight_parents.py",
    "seqref_mri/src/fastmri_data.py",
    "seqref_mri/src/forward_operator.py",
    "seqref_mri/scripts/train_base.py",
]

REQUIRED_PREPARE_KEYS = ("y", "x_norm", "cond_in", "tgt_norm", "amax", "ops")


# ---------------------------------------------------------------------------
# Verdict taxonomy (EXEC v0.4 §8, A3)
#   PASS  -> publish facts, exit 0
#   BLOCK -> publish VALID facts first, then exit 1. A BLOCK must never
#            disappear as a log line, so it must not raise before publication.
#   ERROR -> publish a DISTINCTLY IDENTIFIED error record where the output path
#            is still trustworthy, then exit non-zero. If the parent or the
#            output configuration is itself untrustworthy, log and raise with
#            no stage artefact presented as valid.
# ---------------------------------------------------------------------------

EXIT_PASS = 0
EXIT_BLOCK = 1
EXIT_ERROR = 2


class StageBlock(Exception):
    """A premise about the inspected DATA failed. Carries everything the facts
    artefact must record before the process exits non-zero."""

    def __init__(self, block_code: str, reason: str, *, observed=None,
                 threshold=None, first_failing=None, n_failing: int = 0):
        super().__init__(reason)
        self.block_code = block_code
        self.reason = reason
        self.observed = observed
        self.threshold = threshold
        self.first_failing = first_failing
        self.n_failing = n_failing

    def as_record(self) -> dict:
        return {"block_code": self.block_code, "block_reason": self.reason,
                "observed": self.observed, "registered_threshold":
                self.threshold, "first_failing_slice": self.first_failing,
                "n_failing": self.n_failing}


class StageError(Exception):
    """The code or the specification is wrong; no valid scientific verdict can
    be produced. An underdetermined threshold is an ERROR, not a BLOCK."""

    def __init__(self, error_code: str, reason: str, *, detail=None,
                 write_record: bool = True):
        super().__init__(reason)
        self.error_code = error_code
        self.reason = reason
        self.detail = detail
        # A3 / trusted-context split. write_record=False means the PARENT or
        # the OUTPUT CONTEXT is itself untrustworthy: log and raise, present no
        # stage artefact as valid. The ABSENCE of a record is then the signal.
        # write_record=True means the output path is still trustworthy and an
        # auditable error record must be written.
        self.write_record = write_record

    def as_record(self) -> dict:
        return {"error_code": self.error_code, "error_reason": self.reason,
                "detail": self.detail}


def _fail_error(code: str, msg: str, *args, detail=None,
                write_record: bool = True) -> None:
    logger.error(msg, *args)
    raise StageError(code, msg % args if args else msg, detail=detail,
                     write_record=write_record)


# ---------------------------------------------------------------------------
# Finiteness -- NaN > tol evaluates False, so finiteness is asserted BEFORE
# every numeric comparison, never inferred from one.
# ---------------------------------------------------------------------------

def require_finite(values: dict, context: str) -> None:
    bad = {k: v for k, v in values.items()
           if not (isinstance(v, (int, float)) and np.isfinite(v))}
    if bad:
        _fail_error("NON_FINITE_QUANTITY",
                    "non-finite quantity in %s: %r -- comparisons are refused "
                    "because a NaN silently passes every > test", context, bad)


# ---------------------------------------------------------------------------
# Parent verification
# ---------------------------------------------------------------------------

def _load_verified(path: str, schema: str, stage: str) -> tuple[dict, str]:
    import json
    # An unverifiable parent makes the parent IDENTITY untrustworthy, so no
    # stage artefact -- not even an error record naming it -- is presented as
    # valid. Everything AFTER this point has a verified parent and a
    # trustworthy output path, so it does write an auditable error record.
    try:
        sha = verify_sidecar(path)
    except Exception as exc:
        _fail_error("PARENT_ARTEFACT_UNVERIFIABLE",
                    "parent artefact %s could not be verified against its "
                    "authoritative sidecar (%s); the parent identity is "
                    "untrustworthy, so no stage artefact is written", path,
                    exc, write_record=False)
    with open(path, "rb") as fh:
        rec = json.load(fh)
    if rec.get("schema") != schema:
        _fail_error("PARENT_SCHEMA_MISMATCH",
                    "parent %s schema is %r, expected %r", path,
                    rec.get("schema"), schema)
    if rec.get("stage") != stage:
        _fail_error("PARENT_STAGE_MISMATCH",
                    "parent %s stage is %r, expected %r", path,
                    rec.get("stage"), stage)
    if rec.get("verdict") != "PASS":
        _fail_error("PARENT_NOT_PASSED",
                    "parent %s verdict is %r; downstream stages run only "
                    "after a PASS", path, rec.get("verdict"))
    return rec, sha


def _verify_live_code(repo_dir: str, p0_cv: dict, p0: dict) -> tuple[dict,
                                                                    list]:
    """Scope comes from the LOCKED PROFILE, never from a parent record. The
    parent supplies only the expected hash and the recorded binding names."""
    live_files = []
    for spec in EXPECTED_CONTRACT:
        path = os.path.join(repo_dir, spec["relpath"])
        if not os.path.isfile(path):
            _fail_error("CONTRACT_FILE_MISSING",
                        "contract file missing from the tree: %s", path)
        with open(path, "rb") as fh:
            live_files.append({"relpath": spec["relpath"],
                               "source_bytes": fh.read(),
                               "entities": list(spec["entities"])})
    live = contract_hash(live_files)
    if live["contract_hash"] != p0_cv.get("live_hash"):
        _fail_error("LIVE_CODE_DIVERGED",
                    "LIVE CODE DIVERGED FROM P0: contract hash is %s, P0 "
                    "recorded %s. The parent record is intact but no longer "
                    "describes the code about to execute. Re-run P0.",
                    live["contract_hash"], p0_cv.get("live_hash"))

    recorded = {r.get("id"): r.get("binding")
                for r in p0.get("assertion_verification", {})
                .get("results", [])}
    live_assertions = []
    for aid, relpath, function, callee in EXPECTED_ASSERTIONS:
        path = os.path.join(repo_dir, relpath)
        if not os.path.isfile(path):
            _fail_error("ASSERTION_FILE_MISSING",
                        "assertion target file missing: %s", path)
        with open(path, "rb") as fh:
            src = fh.read()
        try:
            a = check_prepare_binding(src, relpath, function, callee)
        except Exception as exc:                       # fail closed
            _fail_error("ASSERTION_CHECK_FAILED",
                        "prepare-binding assertion %s failed: %s", aid, exc)
        a["id"] = aid
        if a["binding"] != recorded.get(aid):
            _fail_error("ASSERTION_BINDING_CHANGED",
                        "assertion %s: binding changed (P0 recorded %r, live "
                        "%r)", aid, recorded.get(aid), a["binding"])
        live_assertions.append(a)
    return live, live_assertions


def _inherit_median_convention(p0s: dict, p0s_script: str) -> dict:
    """EXEC §8 P1 requires the median convention to be READ FROM the P0S
    implementation and RECORDED with the source hash it was read from.

    A descriptive string would only assert the inheritance. This PROVES it: the
    recorded per-entry energies are re-reduced with the convention P1 is about
    to use, and the result must equal the frozen S_ref BITWISE. If P0S had used
    a different median rule, the reproduction fails and the stage errors.
    """
    if not os.path.isfile(p0s_script):
        _fail_error("P0S_SCRIPT_MISSING",
                    "P0S implementation not found at %s; the median "
                    "convention cannot be inherited from a file that is not "
                    "there", p0s_script)
    script_sha = file_sha256(p0s_script)
    recorded_sha = p0s.get("code", {}).get("script_sha256")
    if script_sha != recorded_sha:
        _fail_error("P0S_SCRIPT_SHA_MISMATCH",
                    "P0S implementation at %s hashes to %s but the P0S facts "
                    "record its script as %s -- the convention would be "
                    "inherited from a file P0S did not run", p0s_script,
                    script_sha, recorded_sha)

    entries = p0s.get("entries")
    if not isinstance(entries, list) or not entries:
        _fail_error("P0S_ENTRIES_MISSING",
                    "P0S facts carry no per-entry energies; the median "
                    "convention cannot be reproduced")
    e = np.asarray([float(x["e_i"]) for x in entries], dtype=np.float64)
    if not np.all(np.isfinite(e)):
        _fail_error("P0S_ENTRIES_NON_FINITE",
                    "P0S per-entry energies contain non-finite values")
    reproduced = float(np.median(np.sqrt(e)))
    frozen = float(p0s["s_ref"]["value"])
    if reproduced != frozen:
        _fail_error("MEDIAN_CONVENTION_MISMATCH",
                    "re-reducing the P0S per-entry energies with numpy.median "
                    "on sqrt(e_i) gives %.17g, but P0S froze S_ref = %.17g. "
                    "The convention this stage would apply is NOT the "
                    "convention P0S applied.", reproduced, frozen)
    return {
        "convention": "numpy.median over sqrt(e_i), e_i the FULL two-channel "
                      "state energy; numpy's linear-interpolation rule on an "
                      "even count",
        "inherited_from": "seqref_mri/scripts/p0s_normalisation_scale.py",
        "source_sha256": script_sha,
        "numpy_version": np.__version__,
        "reproduction": {"method": "re-reduce the P0S per-entry e_i and "
                                   "require bitwise equality with the frozen "
                                   "S_ref",
                         "reproduced_value": reproduced,
                         "frozen_value": frozen,
                         "bitwise_equal": True},
    }


def verify_parents(repo_dir: str, p0_facts_path: str, p0s_facts_path: str,
                   p0s_script_path: str) -> dict:
    """Verify P0 and P0S and return everything P1/P2 may consume.

    Raises StageError on any failure. Neither P1 nor P2 consumes the other's
    verdict, and neither redraws the frozen subset.
    """
    p0, p0_sha = _load_verified(p0_facts_path, P0_FACTS_SCHEMA, "P0")
    p0_cv = p0.get("contract_verification", {})
    if p0_cv.get("reproduced") is not True:
        _fail_error("PARENT_CONTRACT_NOT_REPRODUCED",
                    "P0 contract_verification.reproduced is %r",
                    p0_cv.get("reproduced"))

    binding = p0.get("source_binding", {})
    try:
        validate_parent_source_binding(binding)
        validate_parent_scope(p0_cv, p0.get("assertion_verification", {}),
                              EXPECTED_CONTRACT, EXPECTED_ASSERTIONS)
    except Exception as exc:
        _fail_error("PARENT_SCOPE_INVALID",
                    "P0 parent record failed the locked expectations: %s", exc)

    live, live_assertions = _verify_live_code(repo_dir, p0_cv, p0)

    p0s, p0s_sha = _load_verified(p0s_facts_path, P0S_FACTS_SCHEMA, "P0S")
    s_ref_block = p0s.get("s_ref", {})
    if s_ref_block.get("valid_for_downstream") is not True:
        _fail_error("S_REF_NOT_VALID_FOR_DOWNSTREAM",
                    "P0S s_ref.valid_for_downstream is %r; P1/P2/P3 require "
                    "both verdict == PASS and this flag",
                    s_ref_block.get("valid_for_downstream"))

    linked = p0s.get("parent_artifact_shas", {}).get("p0_facts")
    if linked != p0_sha:
        _fail_error("PARENT_CHAIN_BROKEN",
                    "P0S records parent P0 facts SHA %r but the P0 artefact "
                    "supplied hashes to %r -- these are not the same chain",
                    linked, p0_sha)

    s_ref = float(s_ref_block.get("value"))
    require_finite({"S_ref": s_ref}, "P0S s_ref.value")
    if s_ref <= 0.0:
        _fail_error("S_REF_NON_POSITIVE",
                    "P0S S_ref is %r; it must be strictly positive", s_ref)

    sampling = p0s.get("sampling", {})
    canonical = sampling.get("canonical_sorted_indices")
    if not isinstance(canonical, list) or not canonical:
        _fail_error("SUBSET_MISSING",
                    "P0S facts carry no canonical_sorted_indices; the frozen "
                    "subset cannot be consumed")
    if sorted(set(canonical)) != list(canonical):
        _fail_error("SUBSET_NOT_CANONICAL",
                    "P0S canonical_sorted_indices are not strictly ascending "
                    "and unique")

    median = _inherit_median_convention(p0s, p0s_script_path)

    dataset = p0s.get("dataset", {})
    return {
        "parents_id": PARENTS_ID,
        "p0": {"path": os.path.abspath(p0_facts_path), "facts_sha256": p0_sha,
               "contract_hash": p0_cv.get("live_hash"),
               "source_manifest_sha256": binding.get("source_manifest_sha256"),
               "git": p0.get("code", {}).get("git", {})},
        "p0s": {"path": os.path.abspath(p0s_facts_path),
                "facts_sha256": p0s_sha,
                "subset_manifest_sha256":
                    sampling.get("subset_manifest_sha256"),
                "population_manifest_sha256":
                    dataset.get("population_manifest_sha256"),
                "population_size": dataset.get("population_size"),
                "git": p0s.get("code", {}).get("git", {})},
        "s_ref": s_ref,
        "s_ref_squared": s_ref * s_ref,
        "median_convention": median,
        "subset_indices": [int(i) for i in canonical],
        "subset_size": len(canonical),
        "dataset": {"data_root": dataset.get("data_root"),
                    "split": dataset.get("split"), "mode": dataset.get("mode"),
                    "epoch": dataset.get("epoch"),
                    "test0": dataset.get("test0")},
        "live_code_verification": {
            "blocking": True, "contract_procedure": PROCEDURE_ID,
            "assertion_procedure": ASSERT_PROCEDURE_ID, "profile": PROFILE_ID,
            "parent_expectations": EXPECTATIONS_ID,
            "expected_source_roots": list(EXPECTED_SOURCE_ROOTS),
            "expected_source_file_count": EXPECTED_SOURCE_FILE_COUNT,
            "scope_source": "normalisation_profile.EXPECTED_CONTRACT / "
                            "EXPECTED_ASSERTIONS -- NOT either parent record",
            "p0_recorded_hash": p0_cv.get("live_hash"),
            "live_hash": live["contract_hash"], "reproduced": True,
            "assertions": live_assertions},
    }


# ---------------------------------------------------------------------------
# Code hashing, environment, semantic hash
# ---------------------------------------------------------------------------

def hash_project_code(repo_dir: str, script_path: str) -> dict:
    files = []
    for relpath in CODE_HASH_FILES:
        path = os.path.join(repo_dir, relpath)
        if not os.path.isfile(path):
            _fail_error("CODE_HASH_FILE_MISSING",
                        "project-local file required for the code hash is "
                        "missing: %s", path)
        files.append({"relpath": relpath, "sha256": file_sha256(path)})
    script = os.path.abspath(script_path)
    files.append({"relpath": os.path.basename(script),
                  "sha256": file_sha256(script)})
    import torch
    return {
        "project_local": files,
        "project_local_sha256": canonical_hash(files),
        "resolved_dependency_list": list(CODE_HASH_FILES),
        "third_party": {"python": sys.version.split()[0],
                        "numpy": np.__version__, "torch": torch.__version__,
                        "note": "third-party packages are recorded by name and "
                                "version, never content-hashed: an installed "
                                "package hash is not reproducible across "
                                "environments"},
    }


def environment_record(repo_dir: str, argv) -> dict:
    import torch
    return {"utc": utc_stamp(), "python": sys.version.split()[0],
            "numpy": np.__version__, "torch": torch.__version__,
            "device": "cpu", "torch_threads": torch.get_num_threads(),
            "git": git_state(repo_dir), "argv": list(argv),
            "arithmetic_path": "reductions in NumPy float64 on CPU, "
                               "single-threaded; canonical so the record does "
                               "not vary with hardware"}


SEMANTIC_EXCLUDED = ["run", "code.third_party", "runtime_seconds",
                     "peak_memory_bytes", "paths", "timestamps",
                     "artifact_sha256", "semantic_sha256"]


def attach_semantic_hash(facts: dict, semantic_payload: dict) -> dict:
    """semantic_sha256 covers SCIENTIFIC CONTENT ONLY and is not
    self-referential: the payload is built separately and excludes the field
    itself, timestamps, runtime, memory, absolute paths and host details.
    Hashed through preflight_io.canonical_hash so there is exactly one
    canonicalisation rule in this tree."""
    facts["semantic_sha256"] = canonical_hash(semantic_payload)
    facts["semantic_scope"] = {
        "included_keys": sorted(semantic_payload.keys()),
        "excluded": list(SEMANTIC_EXCLUDED),
        "canonicalisation": "preflight_io.canonical_hash (UTF-8, sorted keys, "
                            "no insignificant whitespace, allow_nan=False)",
        "self_referential": False,
        "note": "two scientifically identical reruns MUST agree on "
                "semantic_sha256; they are NOT expected to agree on the "
                "authoritative artifact SHA, and no cross-reference may "
                "assume they do"}
    return facts


# ---------------------------------------------------------------------------
# Run-mode guard
#   A smoke run and an authoritative run must never share an output directory.
#   Without this, `--smoke N` pointed at the locked results directory writes
#   smoke_* artefacts and exits 0 -- an authoritative-SHAPED hole that no
#   downstream consumer would notice, because the artefacts it looks for are
#   simply absent rather than wrong.
# ---------------------------------------------------------------------------

AUTHORITATIVE_ARTEFACTS = ("representation_facts.json", "support_facts.json")
SMOKE_ARTEFACT_PREFIX = "smoke_"


def guard_run_mode(out_dir: str, smoke: bool) -> str:
    mode = "smoke" if smoke else "authoritative"
    if not os.path.isdir(out_dir):
        return mode
    names = os.listdir(out_dir)
    if smoke:
        clash = [n for n in names if n in AUTHORITATIVE_ARTEFACTS]
        if clash:
            _fail_error("SMOKE_INTO_AUTHORITATIVE_PATH",
                        "refusing to run in SMOKE mode against %s: it already "
                        "holds authoritative artefacts %s. A smoke run must "
                        "use an EPHEMERAL output directory.", out_dir, clash,
                        write_record=False)
    else:
        clash = [n for n in names if n.startswith(SMOKE_ARTEFACT_PREFIX)]
        if clash:
            _fail_error("SMOKE_RESIDUE_IN_AUTHORITATIVE_PATH",
                        "refusing to run AUTHORITATIVELY against %s: it holds "
                        "smoke residue %s. Delete the EPHEMERAL smoke "
                        "artefacts before the authoritative run.", out_dir,
                        clash, write_record=False)
    return mode


# ---------------------------------------------------------------------------
# Claim-guarded publication
# ---------------------------------------------------------------------------

@contextmanager
def publication_claim(out_dir: str, prefix: str, stage: str):
    """ATOMIC exclusive claim over one prefix, held across publication.

    The claim path is FIXED per prefix, so O_CREAT|O_EXCL decides the winner in
    the kernel. Holder identity (stage, PID, UUID4 token, timestamp) is written
    inside the file; it is provenance, not the exclusion mechanism.
    """
    os.makedirs(out_dir, exist_ok=True)
    claim = os.path.join(out_dir, f".{prefix}.claim")
    token = uuid.uuid4().hex
    info = {"claim_path": os.path.basename(claim), "claim_token": token,
            "pid": os.getpid(), "acquired": False, "released": None}
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        try:
            with open(claim, "r", encoding="utf-8") as fh:
                holder = fh.read().strip()
        except OSError:
            holder = "<unreadable>"
        _fail_error("PUBLICATION_CLAIM_HELD",
                    "prefix %r in %s is claimed by another %s publication "
                    "(holder: %s). Concurrent same-prefix publication is "
                    "refused. If that holder is dead, the claim is STALE and "
                    "must be removed by an operator -- nothing reclaims it "
                    "automatically.", prefix, out_dir, stage, holder)
    except OSError as exc:
        _fail_error("PUBLICATION_CLAIM_UNAVAILABLE",
                    "could not acquire the publication claim for prefix %r in "
                    "%s: %s", prefix, out_dir, exc)
    try:
        os.write(fd, f"stage={stage} pid={os.getpid()} token={token} "
                     f"utc={utc_stamp()}\n".encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    info["acquired"] = True

    stale = [n for n in os.listdir(out_dir)
             if n.startswith(prefix) and ".tmp" in n]
    if stale:
        try:
            os.remove(claim)
            info["released"] = True
        except OSError:
            logger.exception("could not release the publication claim %s",
                             claim)
            info["released"] = False
        _fail_error("STALE_TEMPORARY_FOUND",
                    "stale temporary files for prefix %r in %s: %s -- residue "
                    "of an interrupted write", prefix, out_dir, stale)
    try:
        yield info
    finally:
        try:
            os.remove(claim)
            info["released"] = True
        except OSError:
            logger.exception("could not release the publication claim %s -- "
                             "later same-prefix runs will halt until an "
                             "operator removes it", claim)
            info["released"] = False


def publish_stage(facts: dict, out_dir: str, prefix: str,
                  stage: str) -> tuple[str, str]:
    """Publish under an atomic claim and VERIFY the result before returning.

    The claim/publication provenance is injected into facts["run"] INSIDE the
    claim, before serialisation, so it travels with the artefact. It is
    volatile by nature and the semantic payload excludes run/, so it cannot
    perturb semantic_sha256.
    """
    with publication_claim(out_dir, prefix, stage) as info:
        run = facts.setdefault("run", {})
        run["publication"] = {"prefix": prefix,
                              "out_dir_basename": os.path.basename(
                                  os.path.abspath(out_dir)),
                              "claim_mechanism": "fixed-name O_CREAT|O_EXCL",
                              **{k: v for k, v in info.items()
                                 if k != "released"}}
        check_pairing(out_dir, prefix)
        path, sha = publish(facts, out_dir, prefix)
        verified = verify_sidecar(path)
        if verified != sha:
            _fail_error("PUBLISHED_ARTEFACT_UNVERIFIABLE",
                        "published %s but the sidecar verifies to %s, not %s",
                        path, verified, sha)
    logger.info("[%s] published %s sidecar_verified=True claim_token=%s "
                "claim_released=%s", stage, os.path.basename(path),
                info["claim_token"], info["released"])
    return path, sha


def publish_error(exc: StageError, out_dir: str, prefix: str, stage: str,
                  *, parents=None, code=None, run=None) -> str | None:
    """Write a DISTINCTLY IDENTIFIED error record. Distinct filename AND an
    explicit artefact_type, so no consumer can mistake it for stage facts.
    Returns the path, or None if the output path is not trustworthy."""
    record = {
        "schema": f"seqref-{stage.lower()}-error/1",
        "artefact_type": "error",
        "stage": stage,
        "verdict": "ERROR",
        "not_stage_facts": "this record is NOT a stage facts artefact and "
                           "must never be consumed as one",
        "run": run or {}, "code": code or {},
        "parent_identifiers": parents or {},
        **exc.as_record(),
    }
    if not exc.write_record:
        logger.error("[%s] %s -- the parent or output context is itself "
                     "untrustworthy, so NO stage artefact is presented as "
                     "valid", exc.error_code, exc.reason)
        return None
    try:
        with publication_claim(out_dir, prefix, stage):
            path, _ = publish(record, out_dir, prefix)
        return path
    except Exception:
        logger.exception("could not write an error record to %s; the output "
                         "path is not trustworthy, so no stage artefact is "
                         "presented as valid", out_dir)
        return None

# SEQREF-PEXP v0.1 -- locked parent expectations for downstream preflight
# LIFETIME: KEEP
#
# Why this is a SEPARATE module from normalisation_profile.py
#   The natural home for these constants would be normalisation_profile.py.
#   But P0 IMPORTS that module, and under EXEC v0.4 §9 any change to a stage's
#   inputs invalidates its artefact -- so adding constants there would
#   invalidate the PASSED P0 and force a ~100 GB re-run of the source binding,
#   for a change P0 does not even use.
#
#   This module is therefore imported by P0S onward and NEVER by P0. Scope
#   still lives outside the consumer that checks it, which is the point: a
#   parent record is data, and data must not define what is checked against it.
#
#   Changing anything below is a SPEC AMENDMENT to EXEC v0.4 §8 P0S.
#
# CONVENTION: every failure path -> logger.error + raise. No fallback, no mock,
#   no silent pass.
#
# Changelog
#   v0.1 (2026-07-29) Created after review found that P0S validated the LIVE
#     code against whatever scope the PARENT FACTS recorded, rather than
#     against the A2-required scope -- the self-defined-scope defect already
#     fixed for the declaration, relocated into the parent artefact.

from __future__ import annotations

import logging

logger = logging.getLogger("seqref_mri.parent_expectations")

__version__ = "0.1"
__abbr__ = "SEQREF-PEXP"

EXPECTATIONS_ID = "seqref-parent-expectations/1"

# Canonical, repo-relative POSIX roots the P0 source manifest must have bound,
# in the sorted order P0 canonicalises to.
EXPECTED_SOURCE_ROOTS = [
    "seqref_mri/data/fastmri/knee_singlecoil_train",
    "seqref_mri/data/fastmri/knee_singlecoil_val",
]

# Campaign sanity condition: 973 train + 199 val = 1,172 .h5 volumes.
# A genuine data change SHOULD invalidate P0 (its manifest SHA would move),
# so a mismatch here means P0 must be re-run, not that this number is wrong.
EXPECTED_SOURCE_FILE_COUNT = 1172

EXPECTED_SOURCE_MANIFEST_FIELD = "source_manifest_sha256"


def _fail(msg: str, *args) -> None:
    logger.error(msg, *args)
    raise ValueError(msg % args if args else msg)


def validate_parent_source_binding(binding: dict) -> None:
    """The parent's source binding must be full-content AND must cover the
    locked roots. A technically valid P0 run over different directories is not
    an admissible parent."""
    if not isinstance(binding, dict):
        _fail("parent source_binding is not an object")

    field = binding.get("field")
    if field != EXPECTED_SOURCE_MANIFEST_FIELD:
        _fail("parent source binding field is %r, required %r -- a "
              "structure-only manifest does not bind file contents", field,
              EXPECTED_SOURCE_MANIFEST_FIELD)
    if binding.get("hash_contents") is not True:
        _fail("parent source binding has hash_contents=%r; a P0 produced with "
              "--no-hash-contents is not an admissible parent",
              binding.get("hash_contents"))

    sha = binding.get(EXPECTED_SOURCE_MANIFEST_FIELD)
    if not isinstance(sha, str) or len(sha) != 64 or \
            any(c not in "0123456789abcdef" for c in sha.lower()):
        _fail("parent %s is not a valid 64-hex value: %r",
              EXPECTED_SOURCE_MANIFEST_FIELD, sha)

    roots = binding.get("roots_canonical")
    if roots != EXPECTED_SOURCE_ROOTS:
        logger.error("parent bound roots %r, required exactly %r in this "
                     "order. A P0 run over different directories is not an "
                     "admissible parent for P0S.", roots,
                     EXPECTED_SOURCE_ROOTS)
        raise ValueError("parent source roots mismatch")

    n = binding.get("n_files")
    if n != EXPECTED_SOURCE_FILE_COUNT:
        logger.error("parent bound %r files, expected %d for this campaign "
                     "(973 train + 199 val). A genuine data change moves the "
                     "manifest SHA and requires P0 to be RE-RUN; it does not "
                     "make this expectation wrong.", n,
                     EXPECTED_SOURCE_FILE_COUNT)
        raise ValueError("parent source file count mismatch")


def validate_parent_scope(p0_contract: dict, p0_assertions: dict,
                          expected_contract, expected_assertions) -> None:
    """The PARENT RECORD must itself cover the A2-required scope.

    Verifying live code against whatever scope the parent happens to record
    proves only that the code matches the parent's own list. A truncated parent
    would validate against its own truncation. `expected_*` are passed in from
    normalisation_profile so this module does not duplicate them.
    """
    got_files = p0_contract.get("files")
    if not isinstance(got_files, list):
        _fail("parent contract_verification.files is not a list")
    got = [(f.get("relpath"),
            tuple((e.get("kind"), e.get("name")) for e in f.get("entities", [])))
           for f in got_files]
    want = [(s["relpath"], tuple(s["entities"])) for s in expected_contract]
    if got != want:
        logger.error("PARENT SCOPE MISMATCH: P0 recorded contract scope %r, "
                     "required %r. The parent record does not cover the scope "
                     "Amendment A2 requires; a narrowed parent must never "
                     "become the basis of a live-code check.",
                     [(r, list(e)) for r, e in got],
                     [(r, list(e)) for r, e in want])
        raise ValueError("parent contract scope mismatch")

    results = p0_assertions.get("results")
    if not isinstance(results, list):
        _fail("parent assertion_verification.results is not a list")
    got_a = [(r.get("id"), r.get("relpath"), r.get("function"), r.get("callee"))
             for r in results]
    if got_a != list(expected_assertions):
        logger.error("PARENT SCOPE MISMATCH: P0 recorded assertions %r, "
                     "required %r. A partial or empty assertion list must "
                     "never pass.", got_a, list(expected_assertions))
        raise ValueError("parent assertion scope mismatch")

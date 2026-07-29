# SEQREF-NPROF v0.2 -- locked A2 normalisation profile (required scope)
# LIFETIME: KEEP
#
# Why this exists
#   P0 must not let the declaration define its own scope.  verify_contract()
#   proves a declaration is self-consistent with the code it NAMES; it cannot
#   prove the declaration still NAMES what Amendment A2 requires.  A
#   declaration regenerated without _prepare, or with an empty assertion list,
#   would reproduce its own reduced hash and PASS.  This module is the
#   independent statement of required scope.
#
#   Both SEQREF-NDECL (generator) and SEQREF-P0 (gate) import from here, so the
#   two can never disagree about what the contract covers.  P0 is thereby
#   independent of the DECLARATION -- the actual threat, since the declaration
#   is data that can be stale, narrowed or hand-edited.  P0 is NOT independent
#   of this module; that is accepted because the module is KEEP, version
#   controlled, carries a changelog, and its SHA is recorded in p0_facts.json.
#
#   Changing anything below is a SPEC AMENDMENT to concept v0.4 §3.1 and
#   EXEC v0.4 §8 P0, not a code tweak.
#
# THREE-STAGE VALIDATION -- the order is load-bearing for the locked
# PASS/BLOCK/RAISE semantics:
#   1. validate_scope()      ALWAYS, convention-independent. Does the
#                            declaration still COVER what A2 requires?
#                            Narrowed / reordered / empty -> RAISE.
#   2. classify()            Convention verdict. per_slice and none, and any
#                            readable label outside the vocabulary, are
#                            recorded BLOCKs -- they must NOT raise, or the
#                            gate can never return its most important finding.
#   3. validate_fields()     ONLY for a convention that has a locked profile.
#                            The A2 field profile is per_volume-specific;
#                            applying it to a per_slice declaration would
#                            raise on the divisor and destroy the BLOCK.
#
#   The PASS vocabulary stays as concept §3.1 and EXEC §8 state it. PASS is
#   conditional on a PROFILE EXISTING for the declared convention: global and
#   train_split are admissible per spec but have no locked profile, so they
#   RAISE rather than BLOCK. A missing profile is an implementation gap, not a
#   verdict about the representation, and must not be recorded as a finding.
#
# CONVENTION: every failure path -> logger.error + raise. No fallback, no mock,
#   no silent pass.
#
# Changelog
#   v0.2 (2026-07-29) Split validation into scope / classification / field
#     stages. v0.1 raised on any convention other than per_volume, which fired
#     BEFORE P0 could classify, so per_slice and none raised instead of
#     producing the recorded BLOCK the spec requires. Field validation is now
#     profile-keyed and runs only for a convention that has a locked profile.
#   v0.1 (2026-07-29) Created under Amendment A2 after review found that P0
#     validated declaration/code consistency but not declaration SCOPE.
#
# Update summary (v0.2): the scope check and the convention verdict were doing
#   each other's jobs. Requiring per_volume inside the always-run validator
#   turned the gate's most important finding -- an unsupported normalisation
#   convention -- into a crash, so a per_slice pipeline would have produced no
#   record at all. Scope is now checked for every declaration regardless of
#   convention, the convention is classified on its own terms, and the
#   per_volume field profile is applied only where it applies. Conventions the
#   spec admits but this implementation has no profile for are separated from
#   both outcomes: they raise, because not having an answer is not the same as
#   answering no.

from __future__ import annotations

import logging

logger = logging.getLogger("seqref_mri.normalisation_profile")

__version__ = "0.2"
__abbr__ = "SEQREF-NPROF"

PROFILE_ID = "seqref-normalisation-profile/1"

# Exact amendment identifier. A later amendment MUST update this deliberately;
# an unrecognised identifier is a stale-metadata error, not a pass.
AMENDMENT_ID = "A2 (2026-07-29)"

FASTMRI = "seqref_mri/src/fastmri_data.py"
TRAINBASE = "seqref_mri/scripts/train_base.py"

# File order and entity order are BOTH part of the contract hash.
EXPECTED_CONTRACT = [
    {"relpath": FASTMRI,
     "role": "provider — supplies meta.file_attr_max (the per-volume divisor)",
     "entities": [("assign", "__version__"),
                  ("assign", "__abbr__"),
                  ("method", "FastMRISliceDataset.__getitem__")]},
    {"relpath": TRAINBASE,
     "role": "division owner — applies the divisor and fixes the metric range",
     "entities": [("assign", "__version__"),
                  ("assign", "__abbr__"),
                  ("assign", "NORMALIZED_DATA_RANGE"),
                  ("function", "_collate"),
                  ("function", "_prepare"),
                  ("function", "_validate")]},
]

EXPECTED_ASSERTIONS = [
    ("validation_uses_prepare", TRAINBASE, "_validate", "_prepare"),
    ("training_uses_prepare", TRAINBASE, "run_training", "_prepare"),
]

# Vocabulary as stated in concept v0.4 §3.1 and EXEC v0.4 §8 P0.
CONVENTIONS_PASS_VOCAB = ("global", "train_split", "per_volume")
CONVENTIONS_BLOCK = ("per_slice", "none")

# Conventions for which a LOCKED FIELD PROFILE exists. A convention in the PASS
# vocabulary without a profile cannot be verified, so it raises.
PROFILED_CONVENTIONS = ("per_volume",)

EXPECTED_CONVENTION = "per_volume"

EXPECTED_NORMALISED = [
    "x_true / complex reconstruction state",
    "zero-filled image / conditioning state",
    "target magnitude",
]

EXPECTED_NOT_NORMALISED = [
    "y / measured k-space observation (RAW UNITS)",
]

EXPECTED_DIVISOR = {
    "symbol": "a_i",
    "source": "HDF5 file-level attribute 'max'",
    "scope": "one HDF5 file = one volume",
    "required": True,
    "finite": True,
    "strictly_positive": True,
    "fallback": "none",
}

EXPECTED_METRIC_DATA_RANGE = 1.0

EXPECTED_PREP_CONTRACT_VERIFIED_BY = [a[0] for a in EXPECTED_ASSERTIONS]

EXPECTED_PROVENANCE_FILES = [FASTMRI, TRAINBASE]


def _fail(msg: str, *args) -> None:
    logger.error(msg, *args)
    raise ValueError(msg % args if args else msg)


def validate_scope(decl: dict) -> None:
    """Independent scope check: does the declaration still COVER what A2
    requires? Missing, extra, reordered or duplicated entries all raise."""
    contract = decl.get("contract")
    if not isinstance(contract, dict):
        _fail("declaration has no contract block")
    if contract.get("blocking") is not True:
        _fail("declaration marks the contract non-blocking; A2 requires "
              "blocking")

    got_files = contract.get("files")
    if not isinstance(got_files, list):
        _fail("declaration contract.files is not a list")
    got = [(f.get("relpath"),
            tuple((e.get("kind"), e.get("name")) for e in f.get("entities", [])))
           for f in got_files]
    want = [(s["relpath"], tuple(s["entities"])) for s in EXPECTED_CONTRACT]
    if got != want:
        got_set = {f[0]: set(f[1]) for f in got}
        want_set = {f[0]: set(f[1]) for f in want}
        missing, extra = [], []
        for relpath, ents in want_set.items():
            have = got_set.get(relpath, set())
            missing += [f"{relpath}::{k} {n}" for k, n in sorted(ents - have)]
        for relpath, ents in got_set.items():
            keep = want_set.get(relpath, set())
            extra += [f"{relpath}::{k} {n}" for k, n in sorted(ents - keep)]
        logger.error("DECLARATION SCOPE MISMATCH against %s. missing: %s; "
                     "extra: %s; declared order: %s; required order: %s. The "
                     "declaration does not cover the scope Amendment A2 "
                     "requires -- a narrowed declaration must never pass.",
                     PROFILE_ID, missing or "none", extra or "none",
                     [f[0] for f in got], [f[0] for f in want])
        raise ValueError("declaration scope mismatch (contract entities)")

    block = decl.get("semantic_assertions")
    if not isinstance(block, dict):
        _fail("declaration has no semantic_assertions block")
    if block.get("blocking") is not True:
        _fail("declaration marks semantic assertions non-blocking; A2 "
              "requires blocking")
    results = block.get("results")
    if not isinstance(results, list):
        _fail("declaration semantic_assertions.results is not a list")
    got_a = [(r.get("id"), r.get("relpath"), r.get("function"), r.get("callee"))
             for r in results]
    if got_a != EXPECTED_ASSERTIONS:
        logger.error("DECLARATION SCOPE MISMATCH: assertions %s, required %s. "
                     "An empty or partial assertion list must never pass.",
                     got_a or "none", EXPECTED_ASSERTIONS)
        raise ValueError("declaration scope mismatch (semantic assertions)")


def normalise_label(raw) -> str:
    """Canonical label form. Non-string or empty -> raise (contradictory
    metadata, not a verdict about the representation)."""
    if not isinstance(raw, str) or not raw.strip():
        logger.error("declared convention is not a non-empty string: %r", raw)
        raise TypeError("convention must be a string label")
    return raw.strip().lower().replace("-", "_").replace(" ", "_")


def classify(label: str) -> tuple[str, str]:
    """Stage 2. Returns (verdict, reason). RAISES only for a convention the
    spec admits but this implementation has no locked profile for."""
    if label in CONVENTIONS_BLOCK:
        return "BLOCK", (
            "readable but unsupported convention; the A2 anomaly floors are "
            "not defined against this representation")
    if label in CONVENTIONS_PASS_VOCAB:
        if label in PROFILED_CONVENTIONS:
            return "PASS", (
                "recognised uniform convention with a locked field profile; "
                "the A2 dimensionless anomaly floors are defined relative to "
                "a recorded reference scale")
        logger.error("convention %r is in the PASS vocabulary but NO LOCKED "
                     "FIELD PROFILE exists for it (profiled: %s). This is an "
                     "implementation gap, not a verdict about the data: it is "
                     "neither a PASS (unverifiable) nor a BLOCK (the "
                     "representation is not unsupported). Write a locked "
                     "profile before running P0 against it.", label,
                     PROFILED_CONVENTIONS)
        raise NotImplementedError(f"no locked profile for convention: {label}")
    return "BLOCK", (
        "convention label outside the locked vocabulary; readable but "
        "unsupported, treated as BLOCK rather than assumed equivalent")


def validate_fields(decl: dict) -> None:
    """Stage 3. Field-level consistency for the per_volume profile ONLY. Must
    not be called for a BLOCK convention: the divisor and list checks below are
    A2/per_volume-specific and would turn a legitimate BLOCK into a crash."""
    if decl.get("amendment") != AMENDMENT_ID:
        _fail("declaration amendment is %r, this profile implements %r",
              decl.get("amendment"), AMENDMENT_ID)
    if normalise_label(decl.get("convention")) != EXPECTED_CONVENTION:
        _fail("validate_fields called for convention %r but this profile "
              "covers %r -- caller must classify first",
              decl.get("convention"), EXPECTED_CONVENTION)

    div = decl.get("divisor")
    if not isinstance(div, dict):
        _fail("declaration divisor is not an object")
    for key, want in EXPECTED_DIVISOR.items():
        if key not in div:
            _fail("declaration divisor is missing %r", key)
        if div[key] != want or type(div[key]) is not type(want):
            _fail("declaration divisor.%s is %r (%s), required %r (%s)",
                  key, div[key], type(div[key]).__name__, want,
                  type(want).__name__)

    if decl.get("normalised") != EXPECTED_NORMALISED:
        _fail("declaration normalised list is %r, required %r",
              decl.get("normalised"), EXPECTED_NORMALISED)
    if decl.get("not_normalised") != EXPECTED_NOT_NORMALISED:
        _fail("declaration not_normalised list is %r, required %r -- the raw-y "
              "declaration is load-bearing for the decoder scale (concept "
              "§2b)", decl.get("not_normalised"), EXPECTED_NOT_NORMALISED)

    dr = decl.get("metric_data_range")
    if isinstance(dr, bool) or not isinstance(dr, (int, float)):
        _fail("declaration metric_data_range is %r (%s); a non-boolean numeric "
              "is required", dr, type(dr).__name__)
    if float(dr) != EXPECTED_METRIC_DATA_RANGE:
        _fail("declaration metric_data_range is %r, required %r (locked G0 "
              "3.11 metric contract)", dr, EXPECTED_METRIC_DATA_RANGE)

    prep = decl.get("train_val_preparation_contract")
    if not isinstance(prep, dict):
        _fail("declaration train_val_preparation_contract is not an object")
    if prep.get("verified_by") != EXPECTED_PREP_CONTRACT_VERIFIED_BY:
        _fail("train_val_preparation_contract.verified_by is %r, required %r",
              prep.get("verified_by"), EXPECTED_PREP_CONTRACT_VERIFIED_BY)
    if not prep.get("not_verified"):
        _fail("train_val_preparation_contract must state what the static "
              "assertions do NOT verify; a bare claim overstates the evidence")

    prov = decl.get("provenance")
    if not isinstance(prov, list):
        _fail("declaration provenance is not a list")
    got_p = [p.get("relpath") for p in prov]
    if got_p != EXPECTED_PROVENANCE_FILES:
        _fail("declaration provenance covers %r, required %r", got_p,
              EXPECTED_PROVENANCE_FILES)
    for p in prov:
        if p.get("blocking") is not False:
            _fail("provenance entry %r must be marked non-blocking; whole-file "
                  "SHAs are provenance, not the blocking control",
                  p.get("relpath"))


def validate_declaration(decl: dict) -> tuple[str, str, str]:
    """Full three-stage validation. Returns (label, verdict, reason).

    Stage 1 (scope) runs for EVERY declaration, so a narrowed or reordered
    declaration raises whatever convention it claims. Stage 3 (fields) runs
    only for a profiled convention, so a per_slice declaration reaches its
    recorded BLOCK instead of crashing on an A2-specific field check.
    """
    validate_scope(decl)
    label = normalise_label(decl.get("convention"))
    verdict, reason = classify(label)
    if verdict == "PASS":
        validate_fields(decl)
    return label, verdict, reason

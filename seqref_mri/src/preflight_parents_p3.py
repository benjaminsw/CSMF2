# SEQREF-PP3 v0.3 -- P3-specific parent extension (P1/P2) over the frozen verifier
# LIFETIME: KEEP
#
# CHANGELOG
# - v0.3 (2026-07-30): rewritten against the ACTUAL frozen API after repository
#   inspection. v0.2 was written against the machinery's design and every call
#   site was contract-incompatible; it also re-implemented work the frozen
#   module already does.
# - DELETED as duplication of preflight_parents: S_ref bitwise reproduction
#   (done by _inherit_median_convention), the P0S script-SHA check (done by the
#   same function), the contract re-drive (done by _verify_live_code), canonical
#   hashing, publication, claims, run-mode guarding and error records.
# - CORRECTED: P0S carries `sampling.canonical_sorted_indices` -- plain integer
#   indices, NOT identity records. v0.2's frozen_subset() invented a per-entry
#   {dataset_index, file, slice_index} shape that does not exist. Slice
#   identities come from batch meta, exactly as P2 obtains them.
# - This module now contains ONLY what the frozen verifier does not do: the
#   byte-pinned P1/P2 loads, the COMPLEX ruling requirement, the P2 per-slice
#   index, and the P3-local code hash (CODE_HASH_FILES is frozen and cannot
#   list P3's new modules).
# - Parent field spellings are FROZEN to the actual names read from the
#   authoritative artefacts. The v0.2 candidate shim is retired.
#
# EXEC §9: this module IMPORTS preflight_parents and preflight_io unchanged.
# Importing invalidates nothing; only editing them would.

from __future__ import annotations

import logging
import os

from preflight_io import canonical_hash, file_sha256, verify_sidecar
from preflight_parents import StageError, _fail_error

logger = logging.getLogger("SEQREF-PP3")

__version__ = "0.3"
__abbr__ = "SEQREF-PP3"

P1_FACTS_SCHEMA = "seqref-p1-facts/1"
P2_FACTS_SCHEMA = "seqref-p2-facts/1"

# P3-local files whose content can affect a P3 verdict. The frozen
# CODE_HASH_FILES list cannot name them (editing it would invalidate P1/P2), so
# P3 hashes them separately and records both blocks.
P3_CODE_FILES = [
    "seqref_mri/src/preflight_parents_p3.py",
    "seqref_mri/src/residual_decoder.py",
]

# FROZEN parent field spellings, read from the authoritative artefacts.
P2_SLICE_KEYS = {
    "dataset_index": "dataset_index",
    "file": "file",
    "slice_index": "slice_index",
    "selected_columns": "mask_selected_columns",
    "mask_seed": "mask_seed",
    "mask_n_columns": "mask_n_columns",
    "mask_width": "mask_width",
    "max_MFdx": "max_MFdx",
    "k_i": "k_i",
    "relative_max": "relative_max",
    "residual_energy_ratio": "residual_energy_ratio",
    "x0_source_key": "x0_prepared_source_key",
    "x0_rel_error": "x0_rel_error",
}


def _load_pinned(path: str, schema: str, stage: str,
                 expected_sha: str | None,
                 expected_semantic: str | None = None) -> tuple[dict, str]:
    """Verify a P1/P2 artefact against its sidecar, then require schema, stage,
    artefact_type, verdict and -- where registered -- the exact bytes.

    A valid sidecar proves byte integrity, not semantic compatibility. Both are
    required before a branch ruling or a support verdict is consumed.
    """
    if not os.path.isfile(path):
        _fail_error("P3_PARENT_MISSING", "%s facts artefact missing at %s",
                    stage, path, write_record=False)
    try:
        sha = verify_sidecar(path)
    except Exception as exc:
        _fail_error("P3_PARENT_UNVERIFIABLE",
                    "%s artefact at %s failed sidecar verification: %s",
                    stage, path, exc, write_record=False)
    import json
    with open(path, "r", encoding="utf-8") as fh:
        facts = json.load(fh)
    if not isinstance(facts, dict):
        _fail_error("P3_PARENT_MALFORMED", "%s artefact is not a JSON object",
                    stage, write_record=False)
    for field, want in (("schema", schema), ("stage", stage),
                        ("artefact_type", "stage_facts"), ("verdict", "PASS")):
        got = facts.get(field)
        if got != want:
            _fail_error("P3_PARENT_IDENTITY_MISMATCH",
                        "%s artefact %s is %r, expected %r", stage, field, got,
                        want, write_record=False)
    semantic = facts.get("semantic_sha256")
    if semantic is None:
        _fail_error("P3_PARENT_SEMANTIC_HASH_MISSING",
                    "%s artefact carries no semantic_sha256; a stripped "
                    "artefact must not be consumed", stage, write_record=False)
    if expected_sha and sha != expected_sha:
        _fail_error("P3_PARENT_SHA_MISMATCH",
                    "%s facts hash to %s but %s was registered -- this is not "
                    "the artefact P3 was specified against", stage, sha,
                    expected_sha, write_record=False)
    # Presence is not identity. A semantic hash that merely exists proves only
    # that some P-stage wrote one; equality with the registered value proves it
    # is the closed artefact P3 was specified against.
    if expected_semantic and semantic != expected_semantic:
        _fail_error("P3_PARENT_SEMANTIC_SHA_MISMATCH",
                    "%s semantic_sha256 is %s but %s was registered", stage,
                    semantic, expected_semantic, write_record=False)
    logger.info("%s verified: sha256=%s semantic=%s", stage, sha,
                facts.get("semantic_sha256"))
    return facts, sha


def verify_p1_p2(p1_facts_path: str, p2_facts_path: str, *,
                 expected_p1_sha: str | None = None,
                 expected_p2_sha: str | None = None,
                 expected_p1_semantic_sha: str | None = None,
                 expected_p2_semantic_sha: str | None = None) -> dict:
    """P3-specific extension: the frozen verify_parents covers P0 and P0S only.

    The expectations are REQUIRED in a stage run -- the caller passes the EXEC
    §12 registered identities. They stay optional in the signature only so a
    fixture can exercise the schema/stage/verdict path independently.

    Requires the P1 ruling to be COMPLEX -- P3 v0.5 implements that branch, and
    a REAL ruling needs the real-target map and x_det construction that this
    stage deliberately does not contain.
    """
    p1, p1_sha = _load_pinned(p1_facts_path, P1_FACTS_SCHEMA, "P1",
                              expected_p1_sha, expected_p1_semantic_sha)
    p2, p2_sha = _load_pinned(p2_facts_path, P2_FACTS_SCHEMA, "P2",
                              expected_p2_sha, expected_p2_semantic_sha)

    ruling = p1.get("ruling")
    if ruling != "COMPLEX":
        _fail_error("P1_RULING_NOT_COMPLEX",
                    "P1 ruled %r. P3 v0.5 implements the COMPLEX branch only; "
                    "another ruling requires the real-target coordinate map "
                    "and x_det construction.", ruling, write_record=False)

    rows = p2.get("slices")
    if not isinstance(rows, list) or not rows:
        _fail_error("P2_SLICES_MISSING",
                    "P2 facts carry no per-slice records; the persisted mask "
                    "and leakage evidence cannot be consumed",
                    write_record=False)

    by_index: dict[int, dict] = {}
    for i, r in enumerate(rows):
        key = P2_SLICE_KEYS["dataset_index"]
        if key not in r:
            _fail_error("P2_SLICE_FIELD_MISSING",
                        "P2 slice record %d has no %r; the frozen field "
                        "spelling does not match this artefact", i, key,
                        write_record=False)
        by_index[int(r[key])] = r
    missing = [k for k in ("selected_columns", "mask_seed", "max_MFdx")
               if P2_SLICE_KEYS[k] not in rows[0]]
    if missing:
        _fail_error("P2_SLICE_FIELD_MISSING",
                    "P2 slice records lack the frozen fields %s (present: %s)",
                    [P2_SLICE_KEYS[k] for k in missing], sorted(rows[0]),
                    write_record=False)

    pinning = {
        "p1_facts_sha_expected": expected_p1_sha,
        "p1_facts_sha_live": p1_sha,
        "p1_facts_sha_pinned": expected_p1_sha is not None,
        "p1_semantic_sha_expected": expected_p1_semantic_sha,
        "p1_semantic_sha_live": p1.get("semantic_sha256"),
        "p1_semantic_sha_pinned": expected_p1_semantic_sha is not None,
        "p2_facts_sha_expected": expected_p2_sha,
        "p2_facts_sha_live": p2_sha,
        "p2_facts_sha_pinned": expected_p2_sha is not None,
        "p2_semantic_sha_expected": expected_p2_semantic_sha,
        "p2_semantic_sha_live": p2.get("semantic_sha256"),
        "p2_semantic_sha_pinned": expected_p2_semantic_sha is not None,
        "all_identities_pinned": all(x is not None for x in (
            expected_p1_sha, expected_p2_sha, expected_p1_semantic_sha,
            expected_p2_semantic_sha)),
    }
    logger.info("P1/P2 verified: ruling=COMPLEX, %d P2 slice records, "
                "all_identities_pinned=%s", len(by_index),
                pinning["all_identities_pinned"])
    return {
        "identity_pinning": pinning,
        "p1": {"path": os.path.abspath(p1_facts_path), "facts_sha256": p1_sha,
               "semantic_sha256": p1.get("semantic_sha256"),
               "schema": p1.get("schema"), "stage": "P1",
               "artefact_type": p1.get("artefact_type"),
               "verdict": p1.get("verdict"), "ruling": ruling},
        "p2": {"path": os.path.abspath(p2_facts_path), "facts_sha256": p2_sha,
               "semantic_sha256": p2.get("semantic_sha256"),
               "schema": p2.get("schema"), "stage": "P2",
               "artefact_type": p2.get("artefact_type"),
               "verdict": p2.get("verdict"), "n_slice_records": len(by_index),
               "verdict_dtype_path": p2.get("summary", {}).get(
                   "verdict_dtype_path")},
        "p2_by_index": by_index,
        "p2_slice_field_spellings": dict(P2_SLICE_KEYS),
        "branch": ruling,
    }


def hash_p3_local_code(repo_dir: str) -> dict:
    """The frozen CODE_HASH_FILES cannot name P3's new modules -- editing that
    list would invalidate the passed P1/P2. P3 hashes them here and both blocks
    are recorded, so no code that can affect the verdict is unhashed."""
    files = []
    for relpath in P3_CODE_FILES:
        path = os.path.join(repo_dir, relpath)
        if not os.path.isfile(path):
            _fail_error("P3_CODE_HASH_FILE_MISSING",
                        "P3-local file required for the code hash is missing: "
                        "%s", path)
        files.append({"relpath": relpath, "sha256": file_sha256(path)})
    return {"p3_local": files, "p3_local_sha256": canonical_hash(files),
            "note": "preflight_parents.CODE_HASH_FILES is frozen and cannot "
                    "name P3's modules; this block covers them separately"}


MASK_SEED_CANDIDATES = ("canonical_mask_seed",)
MASK_SEED_MODULES = ("seqref_mri.src.fastmri_data",)


def bind_mask_seed_provenance(repo_dir: str) -> dict:
    """M4: bind the eval-mask seed rule to the EXECUTING implementation.

    A textual seed-rule description asserts the rule; this reads it. The seed
    tuple is taken from the function's own signature, not declared, and the
    source file is hashed. Failure is a typed ERROR: the census can prove the
    live seeds equal P2's persisted seeds, but only this binding can show what
    produced them, and an unbindable provenance is a one-name fix, not a
    reason to publish an unevidenced rule.
    """
    import importlib
    import inspect

    for modname in MASK_SEED_MODULES:
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        for cand in MASK_SEED_CANDIDATES:
            fn = getattr(mod, cand, None)
            if fn is None or not callable(fn):
                continue
            try:
                sig = inspect.signature(fn)
                params = list(sig.parameters)
            except (TypeError, ValueError):
                params = None
            src = getattr(mod, "__file__", None)
            return {
                "mask_seed_module": modname,
                "mask_seed_qualname": getattr(fn, "__qualname__", cand),
                "mask_seed_source_sha256": (file_sha256(src) if src and
                                            os.path.isfile(src) else None),
                "mask_seed_signature": str(sig) if params is not None else None,
                "seed_tuple_fields_from_source": params,
                "derivation": "read from the executing implementation; the "
                              "seed tuple is the function's own parameter "
                              "list, not a declared description",
                "resolved": True,
            }
    _fail_error("MASK_SEED_PROVENANCE_UNBINDABLE",
                "could not bind the eval-mask seed implementation; tried %s in "
                "%s. M4 provenance is required and must be read from the "
                "executing code, never asserted.", list(MASK_SEED_CANDIDATES),
                list(MASK_SEED_MODULES))


def dataset_provenance(dataset_cls, dataset_obj=None) -> dict:
    """Record the dataset constructor actually used, and any sampling
    parameters the instance exposes. Absent attributes are recorded as absent,
    never defaulted."""
    import inspect
    try:
        sig = str(inspect.signature(dataset_cls.__init__))
    except (TypeError, ValueError):
        sig = None
    out = {"dataset_class": getattr(dataset_cls, "__qualname__", None),
           "dataset_init_signature": sig, "split": "train", "mode": "eval"}
    for attr in ("acceleration", "center_fraction", "center_fractions",
                 "shape", "cell_hw"):
        if dataset_obj is not None and hasattr(dataset_obj, attr):
            val = getattr(dataset_obj, attr)
            out[attr] = val if isinstance(val, (int, float, str, list, tuple)) \
                else repr(val)
        else:
            out[attr] = "not_exposed_by_dataset"
    return out


def p2_field(record: dict, logical: str, context: str):
    """Read a frozen P2 field. Absence is a typed failure, never a default."""
    key = P2_SLICE_KEYS.get(logical)
    if key is None:
        raise StageError("P2_FIELD_UNREGISTERED",
                         f"no frozen spelling registered for {logical!r}")
    if key not in record:
        _fail_error("P2_SLICE_FIELD_MISSING",
                    "%s: P2 record has no %r (present: %s)", context, key,
                    sorted(record))
    return record[key]

# SEQREF-NDECL v0.2 -- generate seqref_mri/configs/normalisation_declaration.json
# LIFETIME: EPHEMERAL
#
# Purpose
#   Turn the implicit runtime normalisation contract into an explicit, hashed
#   declaration that P0 can verify against the executing code (EXEC v0.4 A2).
#   The ENTITY LIST below is the declared contract scope; it is the thing P0
#   blocks on.  Whole-file SHAs are emitted as provenance only.
#
#   This generator and P0 share seqref_mri/src/contract_hash.py so the two can
#   never drift into computing the hash differently.
#
# CONVENTION: every failure path -> logger.error + raise. No fallback, no mock,
#   no silent pass.  The generator does NOT verify that the declaration is
#   scientifically correct -- it records what the code says.  P0 verifies.
#
# Changelog
#   v0.2 (2026-07-29) Publication is race-free: without --overwrite the file is
#     published by exclusive os.link, so exclusive creation IS the existence
#     check and no concurrent writer can slip between test and write.
#     Added blocking prepare-binding semantic assertions so the
#     training call site is covered without hashing all of run_training;
#     renamed train_val_identical -> train_val_preparation_contract, which
#     states what is verified AND what the static checks leave unverified;
#     required --overwrite before replacing an existing KEEP declaration.
#   v0.1 (2026-07-29) Created under Amendment A2.
#
# Update summary (v0.2): the previous declaration asserted that training and
#   validation share a preparation rule, but nothing blocking covered the
#   training call site, so the claim outran its evidence. Assertions now check
#   that each path calls the same preparation function exactly once, binds the
#   result to a single name, and never rebinds, aliases, mutates through or
#   passes that binding bare. The field is renamed and reshaped so it no longer
#   reads as full equivalence, and it names the one thing the static check
#   cannot see: tensors built directly from `batch` alongside the binding.

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))
from contract_hash import (contract_hash, check_prepare_binding,  # noqa: E402
                           PROCEDURE_ID, ASSERT_PROCEDURE_ID)
from normalisation_profile import (  # noqa: E402
    AMENDMENT_ID, EXPECTED_ASSERTIONS, EXPECTED_CONTRACT, EXPECTED_CONVENTION,
    EXPECTED_DIVISOR, EXPECTED_METRIC_DATA_RANGE, EXPECTED_NORMALISED,
    EXPECTED_NOT_NORMALISED, PROFILE_ID, validate_declaration)

logger = logging.getLogger("SEQREF-NDECL")

SCHEMA = "seqref-normalisation-declaration/1"

# ---- SCOPE COMES FROM THE LOCKED PROFILE, NOT FROM THIS FILE ---------------
# Duplicating the scope here would let the generator and the gate drift apart.
# normalisation_profile.py (KEEP) is the single source of truth; changing it is
# a spec amendment to concept v0.4 §3.1 and EXEC v0.4 §8 P0.
CONTRACT = EXPECTED_CONTRACT
ASSERTIONS = [{"id": i, "relpath": r, "function": f, "callee": c}
              for (i, r, f, c) in EXPECTED_ASSERTIONS]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError as exc:
        logger.error("unreadable file %s: %s", path, exc)
        raise
    return h.hexdigest()


def git_state(repo_dir: str) -> dict:
    try:
        head = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True)
        st = subprocess.run(["git", "-C", repo_dir, "status", "--porcelain"],
                            capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.error("could not resolve git state for %s: %s", repo_dir, exc)
        raise
    return {"commit": head.stdout.strip(), "dirty": bool(st.stdout.strip())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="SEQREF-NDECL v0.2 -- emit normalisation_declaration.json")
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--out", required=True,
                    help="output path, normally "
                         "seqref_mri/configs/normalisation_declaration.json")
    ap.add_argument("--overwrite", action="store_true",
                    help="required to replace an existing declaration. It is "
                         "a KEEP artefact; silent regeneration is exactly the "
                         "recording-without-verifying failure mode.")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    files, provenance = [], []
    for spec in CONTRACT:
        path = os.path.join(args.repo_dir, spec["relpath"])
        if not os.path.isfile(path):
            logger.error("declared contract file missing: %s", path)
            raise FileNotFoundError(path)
        with open(path, "rb") as fh:
            src = fh.read()
        files.append({"relpath": spec["relpath"], "source_bytes": src,
                      "entities": spec["entities"]})
        provenance.append({"relpath": spec["relpath"], "role": spec["role"],
                           "whole_file_sha256": sha256_file(path),
                           "blocking": False,
                           "note": "provenance only; the blocking check is the "
                                   "entity-level contract_hash"})


    ch = contract_hash(files)

    by_relpath = {f["relpath"]: f["source_bytes"] for f in files}
    assertions = []
    for spec in ASSERTIONS:
        src = by_relpath.get(spec["relpath"])
        if src is None:
            logger.error("assertion %s targets %s, which is not a declared "
                         "contract file", spec["id"], spec["relpath"])
            raise KeyError(spec["relpath"])
        rec = check_prepare_binding(src, spec["relpath"], spec["function"],
                                    spec["callee"])          # raises on failure
        rec.update({"id": spec["id"], "blocking": True})
        assertions.append(rec)

    decl = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "amendment": AMENDMENT_ID,
        "profile": PROFILE_ID,
        "convention": EXPECTED_CONVENTION,
        "divisor": dict(EXPECTED_DIVISOR),
        "normalised": list(EXPECTED_NORMALISED),
        "not_normalised": list(EXPECTED_NOT_NORMALISED),
        "scale_mixing_note": (
            "y is raw. Because y = M F x_true, the normalised acquired "
            "coefficients are y/a_i. Any k-space decode MUST state which "
            "scale it assembles in (concept v0.4 §2b)."
        ),
        "metric_data_range": EXPECTED_METRIC_DATA_RANGE,
        "train_val_preparation_contract": {
            "claim": ("training and validation both obtain prepared tensors "
                      "from train_base.py::_prepare"),
            "verified_by": [a["id"] for a in assertions],
            "verified_properties": [
                "each path calls _prepare exactly once",
                "the result is bound to a single bare name",
                "that binding is never rebound, aliased, mutated through, or "
                "passed bare to another call",
            ],
            "not_verified": (
                "construction of additional tensors directly from `batch` in "
                "either path is NOT detected by these static assertions; this "
                "is a narrower claim than full train/validation equivalence"
            ),
        },
        "semantic_assertions": {
            "procedure": ASSERT_PROCEDURE_ID,
            "blocking": True,
            "results": assertions,
        },
        "contract": {
            "procedure": PROCEDURE_ID,
            "blocking": True,
            "contract_hash": ch["contract_hash"],
            "files": ch["files"],
        },
        "provenance": provenance,
        "git": git_state(args.repo_dir),
        "generator": {"abbr": "SEQREF-NDECL", "version": "0.2",
                      "sha256": sha256_file(os.path.abspath(__file__))},
    }

    # The generator validates its own output against the locked profile, so a
    # declaration that would fail P0 is never written in the first place.
    validate_declaration(decl)

    payload = json.dumps(decl, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = f"{args.out}.tmp{os.getpid()}"
    try:
        with open(tmp, "xb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        if args.overwrite:
            os.replace(tmp, args.out)
        else:
            # No check-then-write race: exclusive creation IS the check, so a
            # concurrent writer cannot slip in between an existence test and
            # the publication.
            try:
                os.link(tmp, args.out)
            except FileExistsError:
                logger.error("declaration already exists at %s and "
                             "--overwrite was not given; refusing to "
                             "regenerate silently", args.out)
                raise
            os.unlink(tmp)
    except OSError as exc:
        if not isinstance(exc, FileExistsError):
            logger.error("could not write declaration to %s: %s",
                         args.out, exc)
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    logger.info("wrote %s  contract_hash=%s  git_dirty=%s",
                args.out, ch["contract_hash"], decl["git"]["dirty"])
    for a in assertions:
        logger.info("  assertion %s: %s::%s binds %r at L%d (call L%d) -> PASS",
                    a["id"], a["relpath"], a["function"], a["binding"],
                    a["binding_line"], a["call_line"])
    for f in ch["files"]:
        for e in f["entities"]:
            logger.info("  %s %s %s L%d-%d %s", f["relpath"], e["kind"],
                        e["name"], e["start_line"], e["end_line"],
                        e["sha256"][:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SEQREF-P0 v0.2 -- normalisation contract gate (EXEC v0.4 §8 P0, Amendment A2)
# LIFETIME: DIAGNOSTIC
#
# Purpose
#   Verify the declared normalisation contract against the EXECUTING code, then
#   emit PASS or BLOCK.  Nothing downstream is built until this passes: the R_*
#   anomaly floors are dimensionless ratios against S_ref, and S_ref is only
#   meaningful if the prepared representation is normalised as declared.
#
# Metadata-only guarantee (structural, not merely documented)
#   * torch is never imported and no tensor/HDF5 file is opened for its VALUES.
#   * Source files are parsed as TEXT for contract hashing; data files are
#     hashed as opaque bytes.  Byte-hashing is provenance, not a statistic.
#   * No dataset, loader or statistical-preflight path is instantiated.
#   * p0_facts.json carries "statistics_computed": false, machine-checkable.
#
# Verdict semantics (EXEC v0.4 §8 P0 -- BLOCK and RAISE are NEVER conflated)
#   PASS   {global, train_split, per_volume} is the ADMISSIBLE PASS
#          vocabulary (concept v0.4 §3.1, EXEC v0.4 §8 P0).  An actual PASS
#          additionally requires a LOCKED FIELD PROFILE for the declared
#          convention; only per_volume is profiled today, and it passes only
#          under Amendment A2, because every affected anomaly floor is defined
#          relative to a recorded reference scale.  Contract hash and semantic
#          assertions must also reproduce against live code.
#   BLOCK  declaration READABLE and internally consistent, but the convention
#          is unsupported (per_slice, none) or outside the locked vocabulary.
#          A recorded scientific verdict; exit 1.
#   RAISE  declaration missing, unreadable, schema-wrong, self-contradictory,
#          or STALE (contract hash / assertions do not reproduce); OR the
#          convention is admissible per spec but has NO LOCKED FIELD PROFILE
#          (global, train_split today).  The latter is an implementation gap,
#          not a verdict about the representation, so it must not be recorded
#          as a BLOCK.  Nothing is written; logger.error + raise.
#
#   INDEPENDENT SCOPE CHECK.  Reproducing the declaration's own hash proves
#   only that the declaration is self-consistent with the code it NAMES.  It
#   cannot prove the declaration still NAMES what A2 requires: a regenerated
#   declaration omitting _prepare, or carrying an empty assertion list, would
#   reproduce its own reduced hash and PASS.  normalisation_profile.py (KEEP)
#   therefore states the required scope independently of the declaration, and
#   P0 requires exact equality of file order, entity order, assertion ids,
#   paths, functions and callees before anything else runs.
#
#   Whole-file SHAs are PROVENANCE ONLY and never block: a comment or logging
#   edit in train_base.py must not halt the preflight.  The blocking controls
#   are the entity-level contract hash and the prepare-binding assertions.
#
# Artefact hashing / rerun safety (retained from v0.1, which was reviewed)
#   Authoritative SHA is the SHA-256 of the written file bytes in a
#   <facts>.sha256 sidecar; no self-referential embedded hash.  Facts and
#   sidecar are written as a temporary pair and published by exclusive os.link
#   (never an empty placeholder), sidecar FIRST so an interrupted write leaves
#   a loud residue.  A content-validating pairing pre-check runs before any
#   write.
#
# CONVENTION: every failure path -> logger.error + raise. No fallback, no mock,
#   no placeholder, no silent pass.
#
# Changelog
#   v0.2 (2026-07-29) REBUILT under Amendment A2, which invalidated v0.1.
#     Reads normalisation_declaration.json instead of an ad-hoc --norm-key;
#     recomputes the entity-level contract hash and the prepare-binding
#     assertions against live code; admits per_volume to the PASS vocabulary;
#     splits BLOCK from RAISE explicitly; binds the source split by a full
#     (relpath, size, sha256) manifest, downgradable only under a RENAMED
#     field.  Write path, sidecar authority and pairing pre-check retained.
#     Post-review additions: independent locked-scope validation via
#     normalisation_profile.py so the declaration cannot define what is
#     checked; field-level declaration validation (amendment id, divisor,
#     normalised/raw lists, metric range, preparation-contract profile,
#     provenance set); source roots canonicalised to repo-relative POSIX paths
#     with duplicate roots rejected, so equivalent CLI spellings hash alike.
#     Classification delegated to the profile's three-stage validator, so an
#     unsupported convention yields a RECORDED BLOCK rather than raising on an
#     A2-specific field check.
#   v0.1 (2026-07-28) Created. INVALIDATED by Amendment A2 (it assumed a fixed
#     global/train-split convention declared by a bare config key).
#
# Update summary (v0.2): the first P0 asked a config for a convention label and
#   trusted the answer.  It was invalidated when the executing code turned out
#   to normalise per volume and to leave the measured k-space raw, a
#   mixed-scale arrangement no config key expressed.  This version verifies
#   rather than trusts: it recomputes the hash of the specific code that
#   performs the division and re-runs the assertions that both training and
#   validation draw their tensors from it, so a declaration that has drifted
#   from the code is a stale-metadata error rather than a pass.
#   Unsupported-but-readable and missing-or-stale are kept apart, because only
#   the first is a finding.

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))
from contract_hash import (contract_hash, check_prepare_binding,  # noqa: E402
                           PROCEDURE_ID, ASSERT_PROCEDURE_ID)
from normalisation_profile import (CONVENTIONS_BLOCK,  # noqa: E402
                                   CONVENTIONS_PASS_VOCAB, PROFILE_ID,
                                   PROFILED_CONVENTIONS, validate_declaration)

SCRIPT_ID = "SEQREF-P0"
SCRIPT_VERSION = "v0.2"
FACTS_SCHEMA = "seqref-p0-facts/2"
DECL_SCHEMA = "seqref-normalisation-declaration/1"

# Vocabulary and classification live in normalisation_profile.py (KEEP) so the
# gate and the generator cannot disagree about them. P0 does not restate them.

REQUIRED_DECL_KEYS = (
    "schema", "convention", "divisor", "normalised", "not_normalised",
    "metric_data_range", "contract", "semantic_assertions", "provenance",
    "train_val_preparation_contract",
)

logger = logging.getLogger(SCRIPT_ID)


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError as exc:
        logger.error("unreadable file %s: %s", path, exc)
        raise
    return h.hexdigest()


def check_pairing(out_dir: str) -> None:
    """Every facts record must have its sidecar AND the pair must be internally
    consistent. Filename matching alone is insufficient: an interrupted
    publication could leave two names with no valid contents."""
    if not os.path.isdir(out_dir):
        return
    names = set(os.listdir(out_dir))
    facts = {n for n in names
             if n.startswith("p0_facts") and n.endswith(".json")}
    sidecars = {n for n in names
                if n.startswith("p0_facts") and n.endswith(".json.sha256")}
    orphan_facts = sorted(f for f in facts if f + ".sha256" not in sidecars)
    orphan_side = sorted(x for x in sidecars if x[:-7] not in facts)
    if orphan_facts or orphan_side:
        logger.error("unpaired P0 records in %s -- facts without sidecar: %s; "
                     "sidecar without facts: %s. Residue of an interrupted "
                     "write; resolve before rerunning.", out_dir,
                     orphan_facts or "none", orphan_side or "none")
        raise RuntimeError("unpaired P0 records")
    for name in sorted(facts):
        fpath = os.path.join(out_dir, name)
        spath = fpath + ".sha256"
        try:
            with open(spath, "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            logger.error("existing sidecar %s unreadable: %s", spath, exc)
            raise
        parts = text.split()
        if len(parts) != 2:
            logger.error("existing sidecar %s malformed: expected "
                         "'<sha256>  <filename>', got %r", spath, text)
            raise RuntimeError(f"malformed sidecar: {spath}")
        rec_sha, rec_name = parts
        if len(rec_sha) != 64 or any(c not in "0123456789abcdef"
                                     for c in rec_sha.lower()):
            logger.error("existing sidecar %s has no valid 64-hex SHA-256: %r",
                         spath, rec_sha)
            raise RuntimeError(f"invalid SHA in sidecar: {spath}")
        if rec_name != name:
            logger.error("existing sidecar %s names %r but sits beside %r",
                         spath, rec_name, name)
            raise RuntimeError(f"sidecar/facts name mismatch: {spath}")
        if file_sha256(fpath).lower() != rec_sha.lower():
            logger.error("existing record %s does not match its sidecar SHA",
                         fpath)
            raise RuntimeError(f"SHA mismatch for existing record: {fpath}")
        try:
            with open(fpath, "rb") as fh:
                json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("existing record %s is not valid JSON: %s", fpath, exc)
            raise RuntimeError(f"corrupt existing record: {fpath}") from exc


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


def load_declaration(path: str) -> dict:
    if not os.path.isfile(path):
        logger.error("normalisation declaration not found: %s -- generate it "
                     "with SEQREF-NDECL before running P0", path)
        raise FileNotFoundError(path)
    try:
        with open(path, "rb") as fh:
            decl = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("declaration %s is unreadable or not valid JSON: %s",
                     path, exc)
        raise
    if not isinstance(decl, dict):
        logger.error("declaration %s did not parse to a mapping", path)
        raise TypeError(f"declaration is not a mapping: {path}")
    if decl.get("schema") != DECL_SCHEMA:
        logger.error("declaration %s has schema %r, expected %r", path,
                     decl.get("schema"), DECL_SCHEMA)
        raise ValueError("declaration schema mismatch")
    missing = [k for k in REQUIRED_DECL_KEYS if k not in decl]
    if missing:
        logger.error("declaration %s is missing required keys: %s", path,
                     missing)
        raise KeyError(f"declaration missing keys: {missing}")
    return decl


def verify_contract(decl: dict, repo_dir: str) -> dict:
    """Recompute the entity-level contract hash from LIVE code. Stale -> raise."""
    contract = decl["contract"]
    if contract.get("procedure") != PROCEDURE_ID:
        logger.error("declaration contract procedure is %r, this P0 "
                     "implements %r", contract.get("procedure"), PROCEDURE_ID)
        raise ValueError("contract procedure mismatch")
    files = []
    for spec in contract["files"]:
        relpath = spec["relpath"]
        path = os.path.join(repo_dir, relpath)
        if not os.path.isfile(path):
            logger.error("declared contract file missing from the tree: %s",
                         path)
            raise FileNotFoundError(path)
        with open(path, "rb") as fh:
            src = fh.read()
        entities = [(e["kind"], e["name"]) for e in spec["entities"]]
        files.append({"relpath": relpath, "source_bytes": src,
                      "entities": entities})
    live = contract_hash(files)          # raises on missing/ambiguous entity
    declared = contract["contract_hash"]
    if live["contract_hash"] != declared:
        decl_map = {(f["relpath"], e["kind"], e["name"]): e["sha256"]
                    for f in contract["files"] for e in f["entities"]}
        drift = [f"{f['relpath']}::{e['kind']} {e['name']}"
                 for f in live["files"] for e in f["entities"]
                 if decl_map.get((f["relpath"], e["kind"], e["name"]))
                 != e["sha256"]]
        logger.error("STALE DECLARATION: contract hash does not reproduce "
                     "(declared %s, live %s). Entities that changed: %s. The "
                     "normalisation contract was edited without regenerating "
                     "the declaration.", declared, live["contract_hash"],
                     drift or "unknown (ordering or scope changed)")
        raise ValueError("stale declaration: contract hash mismatch")
    return live


def verify_assertions(decl: dict, repo_dir: str) -> list[dict]:
    """Re-run the prepare-binding assertions against LIVE code."""
    block = decl["semantic_assertions"]
    if block.get("procedure") != ASSERT_PROCEDURE_ID:
        logger.error("declaration assertion procedure is %r, this P0 "
                     "implements %r", block.get("procedure"),
                     ASSERT_PROCEDURE_ID)
        raise ValueError("assertion procedure mismatch")
    if not block.get("blocking", False):
        logger.error("declaration marks semantic assertions non-blocking; A2 "
                     "requires them to be blocking")
        raise ValueError("semantic assertions must be blocking")
    out = []
    for rec in block["results"]:
        path = os.path.join(repo_dir, rec["relpath"])
        if not os.path.isfile(path):
            logger.error("assertion target file missing: %s", path)
            raise FileNotFoundError(path)
        with open(path, "rb") as fh:
            src = fh.read()
        live = check_prepare_binding(src, rec["relpath"], rec["function"],
                                     rec["callee"])      # raises on failure
        live["id"] = rec["id"]
        if live["binding"] != rec["binding"]:
            logger.error("assertion %s: binding name changed (declared %r, "
                         "live %r) -- declaration is stale", rec["id"],
                         rec["binding"], live["binding"])
            raise ValueError(f"stale assertion binding: {rec['id']}")
        out.append(live)
    return out


def check_provenance(decl: dict, repo_dir: str) -> list[dict]:
    """Whole-file SHAs: RECORDED, NEVER BLOCKING."""
    out = []
    for prov in decl["provenance"]:
        path = os.path.join(repo_dir, prov["relpath"])
        if not os.path.isfile(path):
            logger.error("provenance file missing from the tree: %s", path)
            raise FileNotFoundError(path)
        live = file_sha256(path)
        matched = (live == prov["whole_file_sha256"])
        if not matched:
            logger.info("provenance note: %s whole-file SHA differs from the "
                        "declaration (declared %s, live %s). NON-BLOCKING -- "
                        "the contract hash is the blocking control.",
                        prov["relpath"], prov["whole_file_sha256"][:16],
                        live[:16])
        out.append({"relpath": prov["relpath"],
                    "declared_sha256": prov["whole_file_sha256"],
                    "live_sha256": live, "matched": matched,
                    "blocking": False})
    return out


def canonical_roots(dirs: list[str], repo_dir: str) -> list[tuple[str, str]]:
    """Canonicalise each source root to a repo-relative POSIX label, so that
    'a/b', './a/b' and '/abs/a/b' hash identically. Duplicates are rejected;
    roots outside the repo raise, because a stable logical label for them
    would be a spec decision, not a CLI convenience."""
    repo_real = os.path.realpath(repo_dir)
    out: list[tuple[str, str]] = []
    seen: dict[str, str] = {}
    for d in dirs:
        if not os.path.isdir(d):
            logger.error("source directory not found: %s", d)
            raise FileNotFoundError(d)
        real = os.path.realpath(d)
        if os.path.commonpath([real, repo_real]) != repo_real:
            logger.error("source root %s resolves outside the repository "
                         "(%s); a stable logical label is required and is a "
                         "spec decision, not a CLI option", d, repo_real)
            raise ValueError(f"source root outside repo: {d}")
        label = os.path.relpath(real, repo_real).replace(os.sep, "/")
        if label in seen:
            logger.error("duplicate source root: %r and %r both canonicalise "
                         "to %r", seen[label], d, label)
            raise ValueError(f"duplicate source root: {label}")
        seen[label] = d
        out.append((label, real))
    out.sort(key=lambda t: t[0])
    return out


def source_manifest(dirs: list[str], repo_dir: str,
                    hash_contents: bool) -> dict:
    """Deterministic manifest over the declared source split directories."""
    roots = canonical_roots(dirs, repo_dir)
    entries = []
    for label, real in roots:
        for root, _sub, files in os.walk(real):
            for name in sorted(files):
                full = os.path.join(root, name)
                rel = os.path.relpath(full, real).replace(os.sep, "/")
                rec = {"root": label, "relpath": rel,
                       "size_bytes": os.path.getsize(full)}
                if hash_contents:
                    rec["sha256"] = file_sha256(full)
                entries.append(rec)
    entries.sort(key=lambda r: (r["root"], r["relpath"]))
    field = ("source_manifest_sha256" if hash_contents
             else "structure_manifest_sha256")
    binding = ("full byte binding: (relpath, size, sha256) over every file"
               if hash_contents else
               "directory structure and file-size manifest hash; FILE "
               "CONTENTS ARE NOT BOUND")
    return {"field": field, field: canonical_hash(entries),
            "n_files": len(entries), "hash_contents": hash_contents,
            "binding": binding,
            "roots_canonical": [label for label, _ in roots],
            "roots_supplied": list(dirs),
            "root_canonicalisation": "repo-relative POSIX path via realpath; "
                                     "duplicates rejected"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="SEQREF-P0 v0.2 -- normalisation contract gate "
                    "(metadata only; computes no statistics)")
    ap.add_argument("--declaration", required=True, metavar="PATH",
                    help="seqref_mri/configs/normalisation_declaration.json")
    ap.add_argument("--repo-dir", required=True, metavar="PATH")
    ap.add_argument("--source-dir", action="append", required=True,
                    metavar="PATH", help="split directory to bind. Repeatable.")
    ap.add_argument("--no-hash-contents", action="store_true",
                    help="skip per-file content hashing. The manifest field is "
                         "then RENAMED to structure_manifest_sha256 and records "
                         "that file contents are not bound.")
    ap.add_argument("--out-dir", required=True, metavar="PATH",
                    help="EXEC v0.4 §9: results/_diag/residual_preflight/")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    decl = load_declaration(args.declaration)
    raw = decl["convention"]

    # THREE-STAGE VALIDATION (normalisation_profile.py):
    #   scope ALWAYS -> classify -> fields only for a profiled convention.
    # Scope runs first so a narrowed declaration raises whatever convention it
    # claims; field validation is withheld from BLOCK conventions so the gate
    # can still return its most important finding instead of crashing.
    label, verdict, reason = validate_declaration(decl)

    # Verification runs regardless of the convention verdict: a STALE
    # declaration is an error and must never be reported as a clean BLOCK.
    live_contract = verify_contract(decl, args.repo_dir)
    live_assertions = verify_assertions(decl, args.repo_dir)
    provenance = check_provenance(decl, args.repo_dir)

    if decl["metric_data_range"] != 1.0:
        logger.error("declared metric_data_range is %r; the locked G0 3.11 "
                     "metric contract requires 1.0 in the normalised "
                     "representation", decl["metric_data_range"])
        raise ValueError("metric_data_range mismatch")

    manifest = source_manifest(args.source_dir, args.repo_dir,
                               not args.no_hash_contents)

    check_pairing(args.out_dir)

    facts: dict[str, Any] = {
        "schema": FACTS_SCHEMA,
        "script": {"id": SCRIPT_ID, "version": SCRIPT_VERSION,
                   "lifetime": "DIAGNOSTIC"},
        "stage": "P0",
        "stage_description": "normalisation contract gate (Amendment A2)",
        "statistics_computed": False,
        "run": {
            "utc": datetime.now(timezone.utc).isoformat(
                timespec="microseconds"),
            "python": sys.version.split()[0],
            "argv": sys.argv[1:] if argv is None else list(argv),
        },
        "scope_profile": {
            "profile": PROFILE_ID,
            "module": "seqref_mri/src/normalisation_profile.py",
            "sha256": file_sha256(os.path.join(
                args.repo_dir, "seqref_mri/src/normalisation_profile.py")),
            "note": "required scope is stated independently of the "
                    "declaration; P0 is independent of the declaration, not "
                    "of this KEEP module",
        },
        "code": {
            "git": git_state(args.repo_dir),
            "repo_dir": os.path.abspath(args.repo_dir),
            "script_path": os.path.abspath(__file__),
            "script_sha256": file_sha256(os.path.abspath(__file__)),
        },
        "parent_artifact_shas": {
            "normalisation_declaration": file_sha256(args.declaration),
            "declaration_path": os.path.abspath(args.declaration),
        },
        "declaration": {
            "convention_declared_raw": raw,
            "convention_label": label,
            "vocabulary": {"pass": list(CONVENTIONS_PASS_VOCAB),
                           "block": list(CONVENTIONS_BLOCK),
                           "profiled": list(PROFILED_CONVENTIONS),
                           "note": "a convention in the PASS vocabulary "
                                   "without a locked profile RAISES; that is "
                                   "an implementation gap, not a verdict"},
            "divisor": decl["divisor"],
            "normalised": decl["normalised"],
            "not_normalised": decl["not_normalised"],
            "metric_data_range": decl["metric_data_range"],
            "train_val_preparation_contract":
                decl["train_val_preparation_contract"],
            "scale_mixing_note": decl.get("scale_mixing_note"),
        },
        "contract_verification": {
            "procedure": PROCEDURE_ID, "blocking": True,
            "declared_hash": decl["contract"]["contract_hash"],
            "live_hash": live_contract["contract_hash"],
            "reproduced": True,
            "files": live_contract["files"],
        },
        "assertion_verification": {
            "procedure": ASSERT_PROCEDURE_ID, "blocking": True,
            "results": live_assertions,
        },
        "provenance_whole_file": {
            "blocking": False,
            "note": "recorded for provenance; the blocking control is the "
                    "entity-level contract hash",
            "files": provenance,
        },
        "source_binding": manifest,
        "verdict": verdict,
        "verdict_reason": reason,
        "downstream": (
            "P0S may run; on P0S PASS, P1 and P2 run in parallel from the "
            "same P0-approved tensors, subset and frozen P0S artefact"
            if verdict == "PASS"
            else "HALT: amend concept v0.4 §3.1 and EXEC v0.4 §8 before any "
                 "statistical inspection"
        ),
        "verify_before_use": [
            "P0S must verify the written file bytes against the sidecar "
            "<facts>.sha256 before running",
            "P1 and P2 must verify both the P0 and P0S sidecars before running",
        ],
        "hash_note": (
            "the authoritative artefact SHA is the SHA-256 of THIS FILE'S "
            "bytes, recorded in the sidecar; no self-referential hash is "
            "embedded in the record"
        ),
        "overwrite_policy": (
            "authoritative file is never overwritten in place; reruns write a "
            "new timestamped record alongside it"
        ),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    authoritative = os.path.join(args.out_dir, "p0_facts.json")
    stamp = facts["run"]["utc"].replace(":", "").replace("-", "").replace(
        ".", "")
    target = authoritative
    if os.path.exists(authoritative):
        target = os.path.join(args.out_dir, f"p0_facts.{stamp}.json")
        logger.info("authoritative p0_facts.json exists; writing rerun record "
                    "to %s", target)

    payload = canonical_bytes(facts)
    sidecar = target + ".sha256"
    tmp_facts = f"{target}.tmp{os.getpid()}"
    tmp_sidecar = f"{sidecar}.tmp{os.getpid()}"

    def _cleanup() -> None:
        for path in (tmp_facts, tmp_sidecar):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                logger.exception("could not remove temporary file %s", path)

    try:
        with open(tmp_facts, "xb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        artifact_sha = file_sha256(tmp_facts)
        with open(tmp_sidecar, "x", encoding="utf-8") as fh:
            fh.write(f"{artifact_sha}  {os.path.basename(target)}\n")
            fh.flush()
            os.fsync(fh.fileno())
        # Publish by exclusive hard link from the COMPLETED temporaries; never
        # an empty placeholder. Sidecar first: an interrupted publication then
        # leaves an orphan sidecar, which the pairing pre-check catches.
        os.link(tmp_sidecar, sidecar)
        os.unlink(tmp_sidecar)
        os.link(tmp_facts, target)
        os.unlink(tmp_facts)
    except FileExistsError:
        logger.error("target already exists, refusing to overwrite: %s / %s",
                     target, sidecar)
        _cleanup()
        raise
    except OSError as exc:
        logger.error("could not write P0 record to %s: %s", target, exc)
        _cleanup()
        raise

    logger.info("P0 verdict=%s convention=%s contract_hash=%s facts=%s "
                "file_sha256=%s", verdict, label,
                live_contract["contract_hash"][:16], target, artifact_sha)
    for a in live_assertions:
        logger.info("  assertion %s: %s::%s binds %r L%d -> PASS", a["id"],
                    a["relpath"], a["function"], a["binding"],
                    a["binding_line"])
    logger.info("  source binding: %s over %d files (%s)", manifest["field"],
                manifest["n_files"], manifest["binding"])

    if verdict == "PASS":
        logger.info("P0 PASS -- %s", reason)
        return 0
    logger.error("P0 BLOCK -- %s (declared convention: %r)", reason, raw)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

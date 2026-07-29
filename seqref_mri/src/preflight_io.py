# SEQREF-PFIO v0.1 -- preflight artefact I/O (canonical hashing + publication)
# LIFETIME: KEEP
#
# Why this exists, and why it duplicates P0
#   P0 v0.2 carries its own copy of these helpers. That copy is FROZEN: P0 has
#   PASSED, and under EXEC v0.4 §9 any change to an artefact-producing stage's
#   code invalidates that artefact and everything downstream. Refactoring the
#   helpers OUT of P0 would therefore invalidate a passed gate and force a
#   ~100 GB re-run of the source binding for no scientific gain.
#
#   The duplication is deliberate and recorded here rather than left to be
#   discovered. P0S and every later preflight stage import from this module;
#   P0 keeps its frozen copy. If P0 is ever legitimately re-run for another
#   reason, it should be migrated to this module at that point.
#
# What is locked (identical semantics to the frozen P0 copy)
#   * canonical JSON: UTF-8, sorted keys, no insignificant whitespace, floats
#     via shortest round-trip repr();
#   * the AUTHORITATIVE artefact SHA is the SHA-256 of the WRITTEN FILE BYTES,
#     in a <artefact>.sha256 sidecar. No self-referential hash is embedded: an
#     embedded hash cannot cover itself, and a "strip this field then
#     recompute" protocol would have to be reimplemented identically by every
#     reader;
#   * publication writes a temporary pair, then links sidecar FIRST and facts
#     second by exclusive os.link. No empty placeholder is ever created -- a
#     placeholder pair would look valid to a name-only pairing check while
#     containing nothing. Two links cannot be jointly atomic, so the order is
#     chosen to make an interrupted write leave an ORPHAN SIDECAR, which the
#     pairing pre-check catches, rather than a facts file stranded without its
#     SHA while reruns quietly divert to timestamped records;
#   * the pairing pre-check validates pair CONTENTS, not filenames: sidecar
#     readable, exactly two fields, 64-hex SHA, name matches its neighbour,
#     recomputed SHA matches, facts parse as JSON.
#
# CONVENTION: every failure path -> logger.error + raise. No fallback, no mock,
#   no placeholder, no silent pass.
#
# Changelog
#   v0.1 (2026-07-29) Created for P0S under Amendment A2, carrying forward the
#     write semantics reviewed and hardened during P0 v0.1->v0.2.

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("seqref_mri.preflight_io")

__version__ = "0.1"
__abbr__ = "SEQREF-PFIO"


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


def utc_stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


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


def verify_sidecar(facts_path: str) -> str:
    """Verify a parent artefact against its authoritative sidecar and return
    the SHA. Any inconsistency raises: a parent that cannot be verified must
    never be silently consumed."""
    sidecar = facts_path + ".sha256"
    for p in (facts_path, sidecar):
        if not os.path.isfile(p):
            logger.error("parent artefact incomplete: %s is missing", p)
            raise FileNotFoundError(p)
    with open(sidecar, "r", encoding="utf-8") as fh:
        text = fh.read()
    parts = text.split()
    if len(parts) != 2:
        logger.error("parent sidecar %s malformed: expected "
                     "'<sha256>  <filename>', got %r", sidecar, text)
        raise RuntimeError(f"malformed parent sidecar: {sidecar}")
    rec_sha, rec_name = parts
    if len(rec_sha) != 64 or any(c not in "0123456789abcdef"
                                 for c in rec_sha.lower()):
        logger.error("parent sidecar %s has no valid 64-hex SHA-256: %r",
                     sidecar, rec_sha)
        raise RuntimeError(f"invalid SHA in parent sidecar: {sidecar}")
    if rec_name != os.path.basename(facts_path):
        logger.error("parent sidecar %s names %r but sits beside %r", sidecar,
                     rec_name, os.path.basename(facts_path))
        raise RuntimeError(f"parent sidecar name mismatch: {sidecar}")
    actual = file_sha256(facts_path)
    if actual.lower() != rec_sha.lower():
        logger.error("parent artefact %s does not match its sidecar SHA "
                     "(file %s, sidecar %s)", facts_path, actual, rec_sha)
        raise RuntimeError(f"parent SHA mismatch: {facts_path}")
    try:
        with open(facts_path, "rb") as fh:
            json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("parent artefact %s is not valid JSON: %s",
                     facts_path, exc)
        raise RuntimeError(f"corrupt parent artefact: {facts_path}") from exc
    return actual


def check_pairing(out_dir: str, prefix: str) -> None:
    """Every facts record must have its sidecar AND the pair must be
    internally consistent. Filename matching alone is insufficient: an
    interrupted publication could leave two names with no valid contents."""
    if not os.path.isdir(out_dir):
        return
    names = set(os.listdir(out_dir))
    facts = {n for n in names if n.startswith(prefix) and n.endswith(".json")}
    sides = {n for n in names
             if n.startswith(prefix) and n.endswith(".json.sha256")}
    orphan_facts = sorted(f for f in facts if f + ".sha256" not in sides)
    orphan_sides = sorted(s for s in sides if s[:-7] not in facts)
    if orphan_facts or orphan_sides:
        logger.error("unpaired %s records in %s -- facts without sidecar: %s; "
                     "sidecar without facts: %s. Residue of an interrupted "
                     "write; resolve before rerunning.", prefix, out_dir,
                     orphan_facts or "none", orphan_sides or "none")
        raise RuntimeError(f"unpaired {prefix} records")
    for name in sorted(facts):
        verify_sidecar(os.path.join(out_dir, name))


def publish(facts: dict, out_dir: str, prefix: str) -> tuple[str, str]:
    """Write facts + authoritative sidecar. Returns (path, artefact_sha).

    The authoritative name is <prefix>.json; if it exists, a microsecond
    stamped rerun record is written alongside and the authoritative file is
    left untouched.
    """
    os.makedirs(out_dir, exist_ok=True)
    check_pairing(out_dir, prefix)

    authoritative = os.path.join(out_dir, f"{prefix}.json")
    target = authoritative
    if os.path.exists(authoritative):
        stamp = (facts.get("run", {}).get("utc") or utc_stamp())
        stamp = stamp.replace(":", "").replace("-", "").replace(".", "")
        target = os.path.join(out_dir, f"{prefix}.{stamp}.json")
        logger.info("authoritative %s.json exists; writing rerun record to %s",
                    prefix, target)

    payload = canonical_bytes(facts)
    sidecar = target + ".sha256"
    tmp_facts = f"{target}.tmp{os.getpid()}"
    tmp_side = f"{sidecar}.tmp{os.getpid()}"

    def _cleanup() -> None:
        for p in (tmp_facts, tmp_side):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                logger.exception("could not remove temporary file %s", p)

    try:
        with open(tmp_facts, "xb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        artifact_sha = file_sha256(tmp_facts)
        with open(tmp_side, "x", encoding="utf-8") as fh:
            fh.write(f"{artifact_sha}  {os.path.basename(target)}\n")
            fh.flush()
            os.fsync(fh.fileno())
        # Publish by exclusive hard link from the COMPLETED temporaries; never
        # an empty placeholder. Sidecar first: an interrupted publication then
        # leaves an orphan sidecar, which check_pairing catches.
        os.link(tmp_side, sidecar)
        os.unlink(tmp_side)
        os.link(tmp_facts, target)
        os.unlink(tmp_facts)
    except FileExistsError:
        logger.error("target already exists, refusing to overwrite: %s / %s",
                     target, sidecar)
        _cleanup()
        raise
    except OSError as exc:
        logger.error("could not write %s record to %s: %s", prefix, target,
                     exc)
        _cleanup()
        raise
    return target, artifact_sha

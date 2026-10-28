# SEQREF-MRI-V05LTDH v0.2 -- scripts.v05lt_datahash
# LIFETIME: KEEP
# =============================================================================
# Purpose: data-content manifest for SEQREF-MRI-V05LT (195 PA5-A; I1.9 c2,
#   plan SEQREF-MRI-V05LTI19 v0.1; V05LTEXEC item 20). Resume data identity
#   covers orders, eligibility, P4 and S_ref but not MRI file contents; this
#   module closes that gap.
#   * generate (gate 2a, PA2-A / PA5-A.1): from the verified split manifest,
#     the sha256 and byte size of every TRAINING and VALIDATION file it
#     references (973 files), bound to the split-manifest sha256; holdout
#     identities refused (HOLDOUT_CONTACT); canonical JSON written once,
#     read-only, with a sha256 sidecar. Timing is NOT part of the manifest
#     (regeneration is byte-identical); it is logged and returned (PA5-A.3).
#   * verify (P1 and every session start, PA5-A.2): manifest sha256 = the
#     approved value; sidecar; schema and fields; split-manifest binding;
#     listed file set = the split manifest's training + validation files
#     (missing / extra entries refused); every listed file re-hashed and its
#     size checked; every directory holding a listed file contains exactly
#     the listed *.h5 files (the production dataset globs *.h5; an extra
#     .h5 on disk is refused, other file types are not data); holdout
#     identities refused. Returns counts and the measured hashing time.
# Registered output: results/v05lt/inputs/v05lt_data_content.json (+ sidecar),
#   written only with production=True (main `generate`, gate 2a only).
# CONVENTION: logger.error + V05LTError(code); exit 0 PASS / 2 failure.
#   No fallback, no mock, no placeholder, no silent pass.
# Changelog (v0.2, 2026-10-25; c2 revision-1 review fix):
#   * Every operational failure is typed: the manifest sidecar is read with
#     C.read_sidecar (regular file, strict ASCII) -> DATA_CONTENT_SIDECAR;
#     manifest / split-manifest reads -> DATA_CONTENT_UNREADABLE (missing
#     manifest -> DATA_CONTENT_MANIFEST_MISSING); generation writes ->
#     DATA_CONTENT_WRITE_FAILED.
# Changelog (NEW in v0.1, 2026-10-25; I1.9 c2, PA5-A):
#   * Introduced: generate / verify of the data-content manifest; holdout
#     refusal; directory-listing check for extra files; timing returned.
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import stat
import sys
import time
from pathlib import Path

from seqref_mri.scripts import v05lt_common as C

logger = logging.getLogger("seqref_mri.v05lt_datahash")

__version__ = "0.2"
__abbr__ = "SEQREF-MRI-V05LTDH"

SCHEMA = "seqref-v05lt-data-content/1"
OUT_NAME = "v05lt_data_content.json"
SPLITS = ("train", "validation")
_DOC_KEYS = {"schema", "abbr", "version", "split_manifest_sha256",
             "n_files", "total_bytes", "files"}
_ROW_KEYS = {"split", "volume_id", "file", "size", "sha256"}


def _io(code: str, what: str, fn, *a, **k):
    """Typed V05LT errors pass through; any other exception -> code."""
    try:
        return fn(*a, **k)
    except C.V05LTError:
        raise
    except Exception as exc:  # noqa: BLE001 -- converted to a typed failure
        C.fail(code, f"{what}: {type(exc).__name__}: {exc}")


def _sha_ok(x) -> bool:
    return isinstance(x, str) and len(x) == 64 and all(
        c in "0123456789abcdef" for c in x)


def _rel_ok(x) -> bool:
    if not isinstance(x, str) or not x or x.startswith("/") or "\\" in x:
        return False
    return all(p not in ("", ".", "..") for p in x.split("/"))


def hash_file(path: Path) -> tuple:
    """-> (sha256, size) in one read of a regular, non-symlink file."""
    path = Path(path)
    try:
        st = os.lstat(path)
    except OSError as exc:
        C.fail("DATA_CONTENT_MISSING", f"{path}: {exc}")
    C.require(stat.S_ISREG(st.st_mode), "DATA_CONTENT_NOT_REGULAR",
              f"{path}: not a regular file (symlinks refused)")
    h, n = hashlib.sha256(), 0
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8 << 20), b""):
                h.update(chunk)
                n += len(chunk)
    except OSError as exc:
        C.fail("DATA_CONTENT_UNREADABLE", f"{path}: {exc}")
    return h.hexdigest(), n


def expected_rows(split_doc: dict, holdout: frozenset) -> list:
    """Training + validation rows of a verified split manifest, sorted by
    relative path; holdout identities refused."""
    rows = []
    for sp in SPLITS:
        for r in split_doc[sp]:
            C.require(_rel_ok(r.get("file")), "DATA_CONTENT_PATH",
                      f"{sp}: {r.get('file')!r}")
            rows.append({"split": sp, "volume_id": r["volume_id"],
                         "file": r["file"]})
    C.assert_no_holdout([r["volume_id"] for r in rows], holdout,
                        "data-content manifest")
    files = [r["file"] for r in rows]
    C.require(len(set(files)) == len(files), "DATA_CONTENT_DUPLICATE",
              "a file is listed twice in the split manifest")
    return sorted(rows, key=lambda r: r["file"])


def _guard(out: Path, production: bool) -> Path:
    p = Path(os.path.abspath(out))
    reg = Path(os.path.abspath(C.Paths().v05lt))
    if p == reg or reg in p.parents:
        C.require(production, "DATAHASH_PRODUCTION_DISABLED",
                  f"{p} is under the registered results tree (gate 2a only)")
    return p


def generate(split_manifest: Path, split_sha256: str, holdout: frozenset,
             data_root: Path, out: Path, *, production: bool = False,
             expect: C.SplitExpect = C.REGISTERED_SPLIT) -> dict:
    """Write the data-content manifest once. Returns {sha256, n_files,
    total_bytes, seconds}."""
    out = _guard(out, production)
    side = Path(str(out) + ".sha256")
    C.require(not os.path.lexists(out) and not os.path.lexists(side),
              "ARTIFACT_EXISTS", f"{out}: data-content manifest is write-once")
    doc = _io("DATA_CONTENT_UNREADABLE", f"{split_manifest}: split manifest",
              C.load_split_manifest, Path(split_manifest), split_sha256,
              expect)
    rows = expected_rows(doc, holdout)
    t0, total = time.monotonic(), 0
    for r in rows:
        r["sha256"], r["size"] = hash_file(Path(data_root) / r["file"])
        total += r["size"]
    secs = time.monotonic() - t0
    body = {"schema": SCHEMA, "abbr": __abbr__, "version": __version__,
            "split_manifest_sha256": split_sha256, "n_files": len(rows),
            "total_bytes": total, "files": rows}
    def write() -> str:
        sha = C.write_durable_new(out, C.canonical_json(body))
        C.write_durable_new(side, f"{sha}  {out.name}\n".encode())
        os.chmod(out, 0o444)
        os.chmod(side, 0o444)
        return sha
    sha = _io("DATA_CONTENT_WRITE_FAILED", f"{out}: write", write)
    logger.info("[%s] data-content manifest %s: %d files, %d bytes, hashed "
                "in %.1f s (sha256 %s)", __abbr__, out, len(rows), total,
                secs, sha)
    return {"sha256": sha, "n_files": len(rows), "total_bytes": total,
            "seconds": secs}


def load_manifest(path: Path, approved_sha256: str) -> dict:
    """Approved hash first, sidecar, then exact structure."""
    path = Path(path)
    C.require(_sha_ok(approved_sha256), "DATA_CONTENT_UNAPPROVED",
              "approved sha256 required")
    C.require(path.is_file() and not path.is_symlink(),
              "DATA_CONTENT_MANIFEST_MISSING", str(path))
    got = C.sha256_file(path)
    C.require(got == approved_sha256, "DATA_CONTENT_UNAPPROVED",
              f"{path}: {got} != approved {approved_sha256}")
    side = Path(str(path) + ".sha256")
    C.require(C.read_sidecar(side, "DATA_CONTENT_SIDECAR") == got,
              "DATA_CONTENT_SIDECAR", f"{side}: sidecar != manifest")
    raw = _io("DATA_CONTENT_UNREADABLE", f"{path}: read", path.read_bytes)
    try:
        doc = json.loads(raw)
    except ValueError as exc:            # includes undecodable bytes
        C.fail("DATA_CONTENT_INVALID", f"{path}: {exc}")
    C.require(isinstance(doc, dict) and set(doc) == _DOC_KEYS and
              doc["schema"] == SCHEMA and isinstance(doc["files"], list),
              "DATA_CONTENT_INVALID", f"{path}: structure")
    for i, r in enumerate(doc["files"]):
        C.require(isinstance(r, dict) and set(r) == _ROW_KEYS and
                  r["split"] in SPLITS and _rel_ok(r["file"]) and
                  isinstance(r["volume_id"], str) and _sha_ok(r["sha256"]) and
                  isinstance(r["size"], int) and not isinstance(
                      r["size"], bool) and r["size"] >= 0,
                  "DATA_CONTENT_INVALID", f"{path}: row {i}")
    C.require(doc["n_files"] == len(doc["files"]) and doc["total_bytes"] ==
              sum(r["size"] for r in doc["files"]), "DATA_CONTENT_INVALID",
              f"{path}: counts")
    return doc


def verify(path: Path, approved_sha256: str, split_manifest: Path,
           split_sha256: str, holdout: frozenset, data_root: Path, *,
           expect: C.SplitExpect = C.REGISTERED_SPLIT) -> dict:
    """Full re-hash (P1 and every session start). Any difference ->
    V05LTError; returns {n_files, total_bytes, seconds, manifest_sha256}."""
    doc = load_manifest(path, approved_sha256)
    C.require(doc["split_manifest_sha256"] == split_sha256,
              "DATA_CONTENT_SPLIT_MISMATCH", "manifest bound to another "
              "split manifest")
    split = _io("DATA_CONTENT_UNREADABLE", f"{split_manifest}: split "
                "manifest", C.load_split_manifest, Path(split_manifest),
                split_sha256, expect)
    want = expected_rows(split, holdout)
    C.assert_no_holdout([r["volume_id"] for r in doc["files"]], holdout,
                        "data-content manifest")
    listed = [{k: r[k] for k in ("split", "volume_id", "file")}
              for r in doc["files"]]
    if listed != want:
        a = {r["file"] for r in listed}
        b = {r["file"] for r in want}
        C.fail("DATA_CONTENT_SET_MISMATCH", f"listed vs split manifest: "
               f"extra {sorted(a - b)[:3]} missing {sorted(b - a)[:3]} "
               f"(or split / order differs)")
    root = Path(data_root)
    dirs = {}
    for r in doc["files"]:
        rel = Path(r["file"])
        dirs.setdefault(rel.parent, set()).add(rel.name)
    for d, names in sorted(dirs.items()):
        try:
            on_disk = {e.name for e in os.scandir(root / d)
                       if e.name.endswith(".h5")}   # what the dataset reads
        except OSError as exc:
            C.fail("DATA_CONTENT_MISSING", f"{root / d}: {exc}")
        C.require(on_disk == names, "DATA_CONTENT_SET_MISMATCH",
                  f"{d}: extra on disk {sorted(on_disk - names)[:3]} "
                  f"missing {sorted(names - on_disk)[:3]}")
    t0, total, bad = time.monotonic(), 0, []
    for r in doc["files"]:
        sha, size = hash_file(root / r["file"])
        total += size
        if sha != r["sha256"] or size != r["size"]:
            bad.append(r["file"])
            logger.error("[%s] DATA_CONTENT_MISMATCH %s: sha256 %s size %d "
                         "(manifest %s / %d)", __abbr__, r["file"], sha, size,
                         r["sha256"], r["size"])
    secs = time.monotonic() - t0
    C.require(not bad, "DATA_CONTENT_MISMATCH",
              f"{len(bad)} altered file(s), first {bad[:3]}")
    logger.info("[%s] data content verified: %d files, %d bytes in %.1f s",
                __abbr__, len(doc["files"]), total, secs)
    return {"n_files": len(doc["files"]), "total_bytes": total,
            "seconds": secs, "manifest_sha256": approved_sha256}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=f"{__abbr__} v{__version__}")
    ap.add_argument("mode", choices=["generate", "verify"])
    ap.add_argument("--repo", default=str(C.Paths().repo))
    ap.add_argument("--approved-sha256", default=None)
    a = ap.parse_args(argv)
    paths = C.Paths(repo=Path(a.repo))
    split = paths.inputs / "v05lt_split_manifest.json"
    ssha = C.REGISTERED_INPUTS["v05lt_split_manifest.json"]
    out = paths.inputs / OUT_NAME
    try:
        hold = C.holdout_ids(paths.v02_manifests)
        if a.mode == "generate":      # gate 2a only (195 PA2-A, PA5-A.1)
            r = generate(split, ssha, hold, paths.data_root, out,
                         production=True)
            print(f"RESULT PASS data_content_sha256={r['sha256']} "
                  f"files={r['n_files']} bytes={r['total_bytes']} "
                  f"seconds={r['seconds']:.1f} path={out}")
        else:
            C.require(a.approved_sha256 is not None, "DATA_CONTENT_UNAPPROVED",
                      "--approved-sha256 is required")
            r = verify(out, a.approved_sha256, split, ssha, hold,
                       paths.data_root)
            print(f"RESULT PASS files={r['n_files']} bytes={r['total_bytes']} "
                  f"seconds={r['seconds']:.1f}")
    except C.V05LTError as exc:
        print(f"RESULT FAIL {exc.code}")
        return 2
    except Exception as exc:  # noqa: BLE001 -- logged, never swallowed
        logger.exception("UNEXPECTED: %s", exc)
        print(f"RESULT FAIL UNEXPECTED {type(exc).__name__}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

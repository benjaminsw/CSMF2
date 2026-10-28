# SEQREF-MRI-V04DLTDV v0.2 -- v04dlt_dataset_verify.py (pre-launch train-split verification)
# LIFETIME: DIAGNOSTIC
# =============================================================================
# Purpose: before V04DLT launch approval, verify that the chosen data root
#   holds exactly the training population the registered V02 epoch manifests
#   describe, read-only, WITHOUT opening holdout data.
# What it does (reads training HDF5 headers and registered manifests only;
#   writes only a fresh evidence folder; performs no training and no holdout
#   evaluation):
#   * code pin first: the repo must be at HEAD 9190735833573cbf2dc7f5fc375d
#     7b9acaa50f70 with a clean worktree, and the imported modules must
#     resolve inside that repo -- checked before and after import;
#   * registered anchor: the V04 pipeline reference table must hash to its
#     registered full SHA-256 (Article 179, a64f18fc...cfdb); each epoch
#     manifest's file hash must equal the table's full pin, and must match
#     the prefix/suffix recorded in 179 (file) and V02R/V03EXEC (content);
#   * builds the train index with the PINNED FastMRISliceDataset(split="train",
#     mode="train") from the pinned worktree -- the same traversal the trainer
#     uses (h5 headers only: key presence and kspace slice count);
#   * checks 973 volumes / 34,742 slices (locked V02 counts);
#   * for each registered epoch manifest v02_epoch{0,1,2}_manifest.json:
#     file hash == its sidecar, internal content hash == recomputed, and the
#     entries sorted by dataset_index == the traversal (file, slice_index,
#     dataset_index), exactly;
#   * records a population fingerprint: SHA-256 of the canonical list of
#     (relative path, file size, slice count). It covers paths, sizes and
#     slice counts only, NOT the dataset bytes.
# Holdout boundary: never constructs split="val", never lists or opens the
#   val directory, never reads the holdout manifest.
# Usage (pinned interpreter, from anywhere):
#   <python> v04dlt_dataset_verify.py --repo <pinned worktree> \
#     --data-root <fastMRI root> --manifest-dir <v02 manifests dir> \
#     --reference <v04_pipeline_reference.json> --out <fresh output dir>
# Output: <out>/dataset_verify.json (+ .sha256), dataset_verify.log; final
#   line "RESULT PASS" (exit 0) or "RESULT FAIL: <code>" (exit 2).
# CONVENTION: logger.error + typed failure; no fallback, no skip.
# Changelog (NEW in v0.1, 2026-10-17): introduced for the V04DLT launch-
#   approval dataset check (reviewer request after the data-root inference).
# Changelog (v0.2, 2026-10-17, review): code-pin check (HEAD, clean tree,
#   module origin) before import; manifests anchored to the registered
#   reference table (full hashes) plus registered prefixes; scope wording
#   (fingerprint excludes bytes; headers only; no training/holdout).
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("SEQREF-MRI-V04DLTDV")
N_TRAIN_VOLUMES, N_TRAIN_SLICES, N_EPOCHS = 973, 34742, 3
PIN = "9190735833573cbf2dc7f5fc375d7b9acaa50f70"
REF_SHA = "a64f18fc4eabc3fbf42e219cbdb3db06a405eea7c94ea151ed10835e2cb0cfdb"   # 179
REG = {  # epoch: ((file prefix, suffix) per 179, (content prefix, suffix) per V02R/V03EXEC)
    0: (("b0637f78", "fb56"), ("390220ab", "a8e9e")),
    1: (("5fb416f9", "21e6"), ("cbe54ac3", "e6b9b")),
    2: (("209e0454", "9d57"), ("048aba62", "e42fe")),
}


class DVError(RuntimeError):
    def __init__(self, code: str, msg: str):
        self.code = code
        super().__init__(f"{code}: {msg}")


def fail(code: str, msg: str) -> None:
    logger.error("%s: %s", code, msg)
    raise DVError(code, msg)


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(repo: Path, *args: str) -> str:
    import subprocess
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if r.returncode != 0:
        fail("GIT_FAILED", f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def check_pin(repo: Path) -> None:
    head = git(repo, "rev-parse", "HEAD").strip()
    if head != PIN:
        fail("COMMIT_PIN_MISMATCH", f"HEAD {head} != {PIN}")
    dirty = git(repo, "status", "--porcelain")
    if dirty.strip():
        fail("WORKTREE_DIRTY", dirty.strip()[:300])


def check_origin(repo: Path, *mods) -> dict:
    out = {}
    for m in mods:
        f = Path(m.__file__).resolve()
        if repo not in f.parents:
            fail("MODULE_ORIGIN_MISMATCH", f"{m.__name__} from {f}, not {repo}")
        out[m.__name__] = str(f)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    for a in ("--repo", "--data-root", "--manifest-dir", "--reference", "--out"):
        ap.add_argument(a, required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"RESULT FAIL: OUT_EXISTS {out}")
        return 2
    out.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(out / "dataset_verify.log"),
                                  logging.StreamHandler(sys.stdout)])
    try:
        repo = Path(args.repo).resolve()
        check_pin(repo)                                   # before import
        if any(k.startswith("seqref_mri") for k in sys.modules):
            fail("MODULE_PRELOADED", "seqref_mri already imported")
        sys.path.insert(0, str(repo))
        from seqref_mri.src import fastmri_data
        from seqref_mri.scripts import v02_manifests
        origins = check_origin(repo, fastmri_data, v02_manifests)
        FastMRISliceDataset = fastmri_data.FastMRISliceDataset
        manifest_sha256 = v02_manifests.manifest_sha256
        logger.info("code pin %s verified (clean); modules from %s", PIN, origins)
        refp = Path(args.reference)
        if not refp.is_file() or sha_file(refp) != REF_SHA:
            fail("REFERENCE_TABLE_MISMATCH", f"{refp} is missing or != registered {REF_SHA}")
        ref = json.loads(refp.read_text())["artifacts"]
        root = Path(args.data_root)
        if not root.is_dir():
            fail("DATA_ROOT_MISSING", str(root))
        ds = FastMRISliceDataset(str(root), split="train", mode="train")
        train_dir = ds.index[0][0].parent if ds.index else None
        if train_dir is None or "val" in train_dir.name:
            fail("TRAIN_DIR_INVALID", str(train_dir))
        trav = [{"file": p.relative_to(ds.data_root).as_posix(), "slice_index": int(s),
                 "dataset_index": int(k)} for k, (p, s) in enumerate(ds.index)]
        files = sorted({p for p, _ in ds.index})
        if len(files) != N_TRAIN_VOLUMES or len(trav) != N_TRAIN_SLICES:
            fail("POPULATION_MISMATCH",
                 f"{len(files)} volumes / {len(trav)} slices != {N_TRAIN_VOLUMES} / {N_TRAIN_SLICES}")
        counts = {}
        for p, _ in ds.index:
            counts[p] = counts.get(p, 0) + 1
        pop = [[p.relative_to(ds.data_root).as_posix(), p.stat().st_size, counts[p]] for p in files]
        fingerprint = hashlib.sha256(json.dumps(pop, separators=(",", ":")).encode()).hexdigest()
        logger.info("train split %s: %d volumes, %d slices; population fingerprint %s",
                    train_dir, len(files), len(trav), fingerprint)
        mdir = Path(args.manifest_dir)
        mres = {}
        for ep in range(N_EPOCHS):
            mp = mdir / f"v02_epoch{ep}_manifest.json"
            side = mdir / f"v02_epoch{ep}_manifest.json.sha256"
            if not mp.is_file() or not side.is_file():
                fail("MANIFEST_MISSING", f"{mp} or its sidecar")
            fsha = sha_file(mp)
            if side.read_text().split()[0] != fsha:
                fail("MANIFEST_SIDECAR_MISMATCH", mp.name)
            if ref.get(mp.name) != fsha:
                fail("MANIFEST_NOT_REGISTERED", f"{mp.name}: {fsha} != reference {ref.get(mp.name)}")
            (fp, fs), (cp, cs) = REG[ep]
            if not (fsha.startswith(fp) and fsha.endswith(fs)):
                fail("MANIFEST_NOT_REGISTERED", f"{mp.name}: file hash {fsha} vs 179 {fp}…{fs}")
            m = json.loads(mp.read_text())
            want = m.pop("manifest_sha256", None)
            if manifest_sha256(m) != want:
                fail("MANIFEST_CONTENT_HASH_MISMATCH", mp.name)
            if not (want.startswith(cp) and want.endswith(cs)):
                fail("MANIFEST_NOT_REGISTERED", f"{mp.name}: content hash {want} vs V02R {cp}…{cs}")
            if m.get("kind") != "train_epoch" or m.get("epoch") != ep or m.get("n_slices") != N_TRAIN_SLICES:
                fail("MANIFEST_HEADER_INVALID", mp.name)
            ents = sorted(m["entries"], key=lambda e: e["dataset_index"])
            if ents != trav:
                bad = next(i for i, (a, b) in enumerate(zip(ents, trav + [None] * len(ents))) if a != b)
                fail("MANIFEST_TRAVERSAL_MISMATCH", f"{mp.name}: first difference at dataset_index {bad}")
            mres[mp.name] = {"file_sha256": fsha, "content_sha256": want, "entries_match_traversal": True}
            logger.info("%s: registered (reference table + 179/V02R prefixes), sidecar OK, content hash OK, %d entries == traversal", mp.name, len(ents))
        check_pin(repo)                                   # unchanged after the run
        rec = {"script": "SEQREF-MRI-V04DLTDV v0.2", "result": "PASS",
               "code_pin": PIN, "module_origins": origins,
               "reference_table": str(refp), "reference_table_sha256": REF_SHA,
               "data_root": str(root), "data_root_realpath": os.path.realpath(root),
               "train_dir": str(train_dir), "n_volumes": len(files), "n_slices": len(trav),
               "population_fingerprint_sha256": fingerprint,
               "fingerprint_rule": "sha256 of compact JSON list [relpath, size_bytes, n_slices] sorted by relpath; covers paths, sizes and slice counts, NOT dataset bytes",
               "scope": "read training HDF5 headers and registered manifests; wrote only this evidence folder; no training, no holdout evaluation",
               "manifests": mres, "manifest_dir": str(mdir), "repo": str(repo),
               "holdout": "not opened (no split='val' construction; val directory and holdout manifest untouched)"}
        body = json.dumps(rec, indent=2, sort_keys=True).encode()
        (out / "dataset_verify.json").write_bytes(body)
        (out / "dataset_verify.json.sha256").write_text(
            f"{hashlib.sha256(body).hexdigest()}  dataset_verify.json\n")
        print("RESULT PASS")
        return 0
    except DVError as e:
        print(f"RESULT FAIL: {e.code}")
        return 2
    except Exception as e:  # noqa: BLE001 -- logged, never swallowed
        logger.exception("UNEXPECTED_ERROR")
        print(f"RESULT FAIL: UNEXPECTED_ERROR {type(e).__name__}")
        return 2


if __name__ == "__main__":
    sys.exit(main())

# SEQREF-MRI-S3 v0.7 -- s3_inspect.py
# LIFETIME: DIAGNOSTIC (delete after S3/G0 lock is recorded in EXEC)
# Implements SEQREF-MRI-EXEC v0.3 SS3a: data inspection & G0 closure inputs.
# Writes facts.json + the SS3a required plot set (plus diagnostic subplots).
# Test split: STRUCTURE ONLY (counts + keys; no statistics, no plots).
# v0.7 changelog (review-driven, closing-mode guards):
#   * tolerance validated: finite and >= 0 (NaN would silently pass the
#     max>tol comparison); fail immediately otherwise.
#   * exclude-record verified: must be an exploratory-mode facts.json AND
#     reference the same data_root.
#   * different pre-registered seed ENFORCED in closing mode (protocol
#     requirement, independent of file non-overlap).
#   * tolerance_policy text branches by mode (no "EXPLORATORY" banner
#     inside a closing record).
#   * v0.7.1: resolved-path data_root comparison; closing-mode fixed
#     visual sample joins the overlap guard (--sample-file must come from
#     outside the exploratory record).
#   * wording: zero-filled operator residual "mathematically zero;
#     expected near machine precision numerically".
# v0.6 changelog (review-driven):
#   * --mode exploratory|closing. CLOSING mode requires --exclude-record
#     (exploratory facts.json) and --target-rel-tolerance; exploratory
#     files are EXCLUDED from the candidate pool before sampling
#     (deterministic non-overlap) and an empty intersection is asserted;
#     the locked tolerance is applied UNCHANGED (fail if max rel-L2
#     exceeds it). Mode + exclusions recorded in facts.
#   * multi-sample A/B construction statistics: complex and magnitude
#     rel-L2 per seeded record, min/median/max aggregated. Fixed plots
#     remain the visual example.
#   * target key REQUIRED to be reconstruction_esc (single-coil ESC);
#     reconstruction_rss now fails loudly -- accepting it could hide an
#     accidental multi-coil dataset or wrong extraction.
#   * provenance: git_dirty + script_sha256 recorded (commit alone does
#     not identify a dirty tree's script).
# v0.5 changelog (review-driven):
#   * target/IFFT validation now MULTI-SAMPLE: seeded 5 train + 5 val
#     volumes x middle slice; min/median/max rel-L2 + per-file records.
#     Fixed inspection sample remains the visual example.
#   * TOLERANCE POLICY (explicit): this run is EXPLORATORY -- no auto
#     pass/fail; the observed rel-L2 distribution is recorded and an
#     engineering tolerance is locked in EXEC BEFORE the G0-closing rerun.
#     G0 is never closed on one unthresholded value.
#   * header check labelled SPOT CHECK + extended: signatures decoded for
#     the seeded validation-sample files as well; agreement recorded.
#   * n_cap removed from split_structural_summary (whole-split scan);
#     help text corrected (compare A/B BEFORE selection; rerun with the
#     SELECTED construction after G0 3.2); "atomic-ish" comment kept
#     accurate (old dir removed before rename).
# v0.4 changelog (review-driven):
#   * BLOCKING VALIDATION ADDED: supplied HDF5 target vs center-cropped
#     |IFFT(kspace)| -- rectangular crops supported; rel-L2 + plots
#     (supplied / from-kspace / absdiff) recorded. This is the sanity check
#     on FFT shifts, axes, and target alignment.
#   * A/B dependency explicit: --construction A|B (default A); reference
#     image named x_ref_<X>; all masked/zero-filled/oracle diagnostics
#     labelled "provisional construction <X>"; facts record the flag.
#     Rerun with the other letter after G0 3.2 selects.
#   * headers decoded for one train AND one val file; compared; mismatch
#     in matrix/limits recorded (not failed -- it is itself a finding).
#   * provenance block: script version, argv, git commit (if available).
#   * atomic output: write to <out>.tmp then replace -- a failed rerun cannot
#     destroy the previous successful record.
#   * removed misleading inspected_files_for_deep_stats_cap (the seeded
#     stat file list is the real record).
# v0.3 changelog (review-driven):
#   * BLOCKING FIX: ground-truth error e = x_true - x0 is ORACLE-ONLY
#     (diagnostics, never a model input). Deployable conditioner channels
#     [|x0|, Re/Im(A^H(y - A x0))] are identically 0 for zero-filled x0 --
#     recorded as a fact; their statistics DEFERRED until a flow-base x0
#     exists (I2+). Histograms now: |x0| + oracle-e (labelled).
#   * OUT_DIR cleared at startup (stale-PNG-safe reruns).
#   * ismrmrd_header XML decoded: encoded/recon matrix sizes, FOV,
#     phase-encode (kspace_encoding_step_1) limits -> sampling-axis input
#     for G0 3.7. Absent header recorded as a fact; unparseable -> fail.
#   * structural checks (target key, shapes, dtypes, slice counts) now run
#     over the WHOLE split, with per-file target/kspace slice-count
#     agreement enforced; deep stats remain capped+seeded.
#   * total_bytes_all_files + per-inspected-file byte sizes recorded.
# v0.2 changelog (review-driven):
#   * ONE residual convention documented: r = y - A(x); for zero-filled
#     x0 = A^H y the operator residual is IDENTICALLY 0 (recorded as a
#     sanity fact); inspection channel = reconstruction error
#     e = x_true - x0 = ifft2c(kspace - y); plots/channels share this sign.
#   * construction comparison records BOTH complex and magnitude rel-L2.
#   * sampling axis explicitly PROVISIONAL (columns/last axis); native dims
#     + ismrmrd header presence recorded; axis is a G0 3.7 lock item.
#   * honest stats labels (train_sampled_middle_slice_*), SEEDED volume
#     sample (not alphabetical head), filenames + byte sizes recorded.
#   * validation split gets the same structural summary as train; target
#     key agreement enforced per split and across splits (fail on mismatch).
#   * --max-stat-volumes >= 1 validated; sample-file key checks fail()
#     cleanly; paths recorded relative to data_root; total slice counts.
# v0.1: initial SS3a implementation.
# Usage (from repo root, .venv_mri active):
#   python -m seqref_mri.diagnostics.s3_inspect \
#       --data-root seqref_mri/data/fastmri [--max-stat-volumes 20] \
#       [--stat-seed 20260721] [--sample-file <name.h5>] [--sample-slice -1]
import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("s3_inspect")

SPLITS_FULL = ("knee_singlecoil_train", "knee_singlecoil_val")   # stats allowed
SPLIT_TEST = "knee_singlecoil_test"                              # structure ONLY
OUT_DIR = Path("seqref_mri/results/_diag/s3_inspection")
CROP = 96
ACCEL = 4
CENTER_FRAC = 0.08          # provisional, inspection-only (G0 3.7 pending)
MASK_SEED = 20260720        # fixed so plots are reproducible


def fail(msg: str) -> None:
    logger.error("[s3_inspect] %s", msg)
    raise RuntimeError(msg)


def list_h5(split_dir: Path) -> list[Path]:
    if not split_dir.is_dir():
        fail(f"split directory missing: {split_dir}")
    files = sorted(split_dir.rglob("*.h5"))
    if not files:
        fail(f"no .h5 files under {split_dir}")
    return files


def file_structure(fp: Path) -> dict:
    with h5py.File(fp, "r") as f:
        keys = sorted(f.keys())
        info = {"keys": keys, "attrs": {k: str(v) for k, v in f.attrs.items()},
                "file_bytes": fp.stat().st_size}
        for k in keys:
            d = f[k]
            if isinstance(d, h5py.Dataset):
                info[k] = {"shape": list(d.shape), "dtype": str(d.dtype)}
    return info


def target_key_of(f: h5py.File, fp_name: str) -> str:
    # Campaign is locked to knee SINGLE-COIL: the ESC target is REQUIRED.
    # RSS present without ESC suggests multi-coil data / wrong extraction.
    if "reconstruction_esc" in f:
        return "reconstruction_esc"
    if "reconstruction_rss" in f:
        fail(f"{fp_name} has reconstruction_rss but NOT reconstruction_esc "
             "-- possible multi-coil dataset or wrong extraction; the "
             "single-coil campaign requires ESC")
    fail(f"no reconstruction target in {fp_name}: {sorted(f.keys())}")
    return ""  # unreachable


def split_structural_summary(files: list[Path], root: Path) -> dict:
    # WHOLE-SPLIT structural scan (metadata only, cheap): every file checked
    # for kspace presence, target key, shapes, dtypes, and per-file
    # target/kspace slice-count agreement. Deep stats remain capped elsewhere.
    n_slices, kshapes, tshapes, kdtypes, tkeys = [], set(), set(), set(), set()
    total_slices, total_bytes = 0, 0
    for fp in files:
        total_bytes += fp.stat().st_size
        with h5py.File(fp, "r") as f:
            if "kspace" not in f:
                fail(f"'kspace' key missing in {fp.name}: {sorted(f.keys())}")
            ks = f["kspace"]
            tk = target_key_of(f, fp.name)
            if f[tk].shape[0] != ks.shape[0]:
                fail(f"target/kspace slice-count mismatch in {fp.name}: "
                     f"{f[tk].shape[0]} vs {ks.shape[0]}")
            n_slices.append(ks.shape[0])
            total_slices += int(ks.shape[0])
            kshapes.add(tuple(ks.shape[1:]))
            kdtypes.add(str(ks.dtype))
            tkeys.add(tk)
            tshapes.add(tuple(f[tk].shape[1:]))
    if len(tkeys) != 1:
        fail(f"inconsistent target keys within split: {sorted(tkeys)}")
    return {"n_files": len(files),
            "total_slices_all_files": total_slices,
            "total_bytes_all_files": total_bytes,
            "structural_scan": "ALL files in split",
            "slices_per_volume": {
                "min": int(np.min(n_slices)), "median": float(np.median(n_slices)),
                "max": int(np.max(n_slices))},
            "kspace_slice_shapes": sorted(str(s) for s in kshapes),
            "kspace_dtypes": sorted(kdtypes),
            "target_key": sorted(tkeys)[0],
            "target_slice_shapes": sorted(str(s) for s in tshapes)}


def decode_ismrmrd_header(fp: Path) -> dict:
    # Decode the ismrmrd XML header if present: matrix sizes, FOV,
    # phase-encode limits -> the factual input for the G0 3.7 axis lock.
    with h5py.File(fp, "r") as f:
        if "ismrmrd_header" not in f:
            return {"present": False,
                    "note": "no ismrmrd_header dataset in this file"}
        raw = f["ismrmrd_header"][()]
    try:
        xml_text = raw.decode() if isinstance(raw, bytes) else str(raw)
        tree = ET.fromstring(xml_text)
    except Exception as exc:  # present but unparseable = loud failure
        fail(f"ismrmrd_header present but unparseable in {fp.name}: {exc}")

    def local(tag):  # namespace-agnostic findall
        return [el for el in tree.iter() if el.tag.split('}')[-1] == tag]

    def matrix_of(space_tag):
        for sp in local(space_tag):
            for ms in sp.iter():
                if ms.tag.split('}')[-1] == "matrixSize":
                    vals = {c.tag.split('}')[-1]: c.text for c in ms}
                    return vals
        return None

    def fov_of(space_tag):
        for sp in local(space_tag):
            for ms in sp.iter():
                if ms.tag.split('}')[-1] == "fieldOfView_mm":
                    return {c.tag.split('}')[-1]: c.text for c in ms}
        return None

    enc1 = None
    for el in local("kspace_encoding_step_1"):
        enc1 = {c.tag.split('}')[-1]: c.text for c in el}
        break
    return {"present": True,
            "encodedSpace_matrix": matrix_of("encodedSpace"),
            "reconSpace_matrix": matrix_of("reconSpace"),
            "encodedSpace_fov_mm": fov_of("encodedSpace"),
            "reconSpace_fov_mm": fov_of("reconSpace"),
            "phase_encode_step1_limits": enc1}


def center_crop2d(x: np.ndarray, size) -> np.ndarray:
    sh, sw = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
    h, w = x.shape[-2], x.shape[-1]
    if h < sh or w < sw:
        fail(f"cannot center-crop {x.shape} to {sh}x{sw}")
    top, left = (h - sh) // 2, (w - sw) // 2
    return x[..., top:top + sh, left:left + sw]


def fft2c(img: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img), norm="ortho"))


def ifft2c(ksp: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ksp), norm="ortho"))


def cartesian_mask(shape_hw, accel: int, center_frac: float, seed: int) -> np.ndarray:
    # PROVISIONAL inspection-only mask. AXIS PROVISIONAL TOO: samples COLUMNS
    # (last axis) -- whether that is the true phase-encode axis is a G0 3.7
    # lock item, decided from native dims + header metadata.
    h, w = shape_hw
    rng = np.random.default_rng(seed)
    n_center = max(1, int(round(w * center_frac)))
    n_total = max(n_center, int(round(w / accel)))
    cols = np.zeros(w, dtype=bool)
    c0 = (w - n_center) // 2
    cols[c0:c0 + n_center] = True
    remaining = np.setdiff1d(np.arange(w), np.nonzero(cols)[0])
    n_rand = n_total - int(cols.sum())
    if n_rand > 0:
        cols[rng.choice(remaining, size=n_rand, replace=False)] = True
    return np.broadcast_to(cols[None, :], (h, w)).copy()


def pcts(a: np.ndarray) -> dict:
    return {"p1": float(np.percentile(a, 1)), "p50": float(np.percentile(a, 50)),
            "p99": float(np.percentile(a, 99)),
            "min": float(a.min()), "max": float(a.max())}


def imsave(path: Path, img: np.ndarray, title: str, log: bool = False) -> None:
    plt.figure(figsize=(5, 5))
    disp = np.log1p(np.abs(img)) if log else img
    plt.imshow(disp, cmap="gray")
    plt.title(title, fontsize=9)
    plt.axis("off")
    plt.colorbar(fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--max-stat-volumes", type=int, default=20)
    ap.add_argument("--stat-seed", type=int, default=20260721,
                    help="seed for choosing which train volumes feed stats")
    ap.add_argument("--sample-file", default=None,
                    help="val .h5 name for the fixed inspection sample "
                         "(default: first val file)")
    ap.add_argument("--sample-slice", type=int, default=-1,
                    help="-1 = middle slice")
    ap.add_argument("--mode", choices=["exploratory", "closing"],
                    default="exploratory",
                    help="closing mode enforces non-overlap with the "
                         "exploratory record and applies the locked "
                         "tolerance unchanged")
    ap.add_argument("--exclude-record", default=None,
                    help="path to the EXPLORATORY facts.json (required in "
                         "closing mode)")
    ap.add_argument("--target-rel-tolerance", type=float, default=None,
                    help="locked target/IFFT rel-L2 tolerance (required in "
                         "closing mode; applied unchanged)")
    ap.add_argument("--construction", choices=["A", "B"], default="A",
                    help="provisional 96x96 construction for the masked/"
                         "zero-filled diagnostics; A and B are COMPARED "
                         "before G0 3.2 selects -- after selection, rerun "
                         "with the SELECTED letter for the closing record")
    args = ap.parse_args()
    if args.max_stat_volumes < 1:
        fail(f"--max-stat-volumes must be >= 1, got {args.max_stat_volumes}")
    excluded_files: set[str] = set()
    if args.mode == "closing":
        if not args.exclude_record:
            fail("closing mode requires --exclude-record (exploratory facts)")
        if (args.target_rel_tolerance is None
                or not np.isfinite(args.target_rel_tolerance)
                or args.target_rel_tolerance < 0):
            fail("--target-rel-tolerance must be finite and >= 0 "
                 f"(got {args.target_rel_tolerance})")
        with open(args.exclude_record) as fh:
            prior = json.load(fh)
        if prior.get("mode") != "exploratory":
            fail("--exclude-record must be an exploratory-mode facts.json "
                 f"(its mode = {prior.get('mode')!r})")
        if (Path(prior.get("data_root", "")).resolve()
                != Path(args.data_root).resolve()):
            fail("--exclude-record data_root mismatch: "
                 f"{prior.get('data_root')!r} vs {args.data_root!r}")
        prior_seed = prior["target_vs_ifft_multisample"].get("sample_seed")
        if args.stat_seed + 1 == prior_seed:
            fail("closing mode requires a DIFFERENT pre-registered sample "
                 f"seed (exploratory used stat_seed+1 = {prior_seed})")
        excluded_files = {r["file"] for r in
                          prior["target_vs_ifft_multisample"]["records"]}
        if not excluded_files:
            fail("exclude-record contains no prior sample files")

    root = Path(args.data_root)
    tmp_out = OUT_DIR.with_name(OUT_DIR.name + ".tmp")
    if tmp_out.exists():
        shutil.rmtree(tmp_out)
    tmp_out.mkdir(parents=True)
    plots = tmp_out / "plots"
    plots.mkdir()
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, timeout=5).stdout.strip() or "unknown"
        git_dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5).stdout.strip())
    except Exception:
        git_commit, git_dirty = "unknown", "unknown"
    script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    facts: dict = {
        "script_version": "SEQREF-MRI-S3 v0.7",
        "argv": sys.argv[1:],
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "script_sha256": script_sha256,
        "mode": args.mode,
        "provisional_construction": args.construction,
        "data_root": str(root),
        "residual_convention": {
            "definition": "r = y - A(x); A = M o F (provisional), "
                          "A^H = F^H o M",
            "zero_filled_note": "for x0 = A^H y the operator residual is "
                                "mathematically zero (A A^H y = M y = y); "
                                "expected near machine precision numerically",
            "deployable_conditioner_channels": [
                "|x0|", "Re(A^H(y - A x0))", "Im(A^H(y - A x0))"],
            "deployable_note": "residual channels are mathematically zero "
                               "for zero-filled x0; their statistics are "
                               "DEFERRED until a flow-base x0 exists (I2+)",
            "oracle_note": "e = x_true - x0 = ifft2c(kspace_96 - y) is an "
                           "ORACLE diagnostic (uses ground truth) -- "
                           "NEVER a model input"},
        "provisional_mask": {
            "family": "random-cartesian-columns+center (INSPECTION ONLY)",
            "axis": "columns (last axis) -- PROVISIONAL, G0 3.7 lock item",
            "accel": ACCEL, "center_frac": CENTER_FRAC, "seed": MASK_SEED}}

    # ---- 1. counts + structure per split (test: structure ONLY) ----
    for split in (*SPLITS_FULL, SPLIT_TEST):
        files = list_h5(root / split)
        facts[split] = {"n_files": len(files),
                        "first_file_structure": file_structure(files[0])}
        logger.info("%s: %d files; keys=%s", split, len(files),
                    facts[split]["first_file_structure"]["keys"])

    # ---- 2. structural summaries: train AND val (same checks) ----
    tr_files = list_h5(root / SPLITS_FULL[0])
    va_files = list_h5(root / SPLITS_FULL[1])
    facts["train_structure"] = split_structural_summary(tr_files, root)
    facts["val_structure"] = split_structural_summary(va_files, root)
    if facts["train_structure"]["target_key"] != facts["val_structure"]["target_key"]:
        fail("train/val target keys differ: "
             f"{facts['train_structure']['target_key']} vs "
             f"{facts['val_structure']['target_key']}")
    tkey = facts["train_structure"]["target_key"]

    # ---- 3. value ranges: SEEDED train volume sample, middle slices ----
    rng = np.random.default_rng(args.stat_seed)
    n_pick = min(len(tr_files), args.max_stat_volumes)
    picked = [tr_files[i] for i in
              sorted(rng.choice(len(tr_files), size=n_pick, replace=False))]
    kmag, tmag = [], []
    for fp in picked:
        with h5py.File(fp, "r") as f:
            mid = f["kspace"].shape[0] // 2
            kmag.append(np.abs(f["kspace"][mid]).ravel())
            tmag.append(np.asarray(f[tkey][mid]).ravel())
    facts["train_sampled_middle_slice_stats"] = {
        "stat_seed": args.stat_seed, "n_volumes": n_pick,
        "files": [str(fp.relative_to(root)) for fp in picked],
        "kspace_magnitude": pcts(np.concatenate(kmag)),
        "target_intensity": pcts(np.concatenate(tmag)),
        "note": "middle slice of each seeded-sampled train volume; NOT "
                "full-dataset statistics"}

    # MULTI-SAMPLE target/IFFT validation: seeded 5+5 volumes, middle slice.
    def target_rel_for(fp: Path, split: str) -> dict:
        with h5py.File(fp, "r") as f:
            if "kspace" not in f or tkey not in f:
                fail(f"validation sample missing keys: {fp.name}")
            m = f["kspace"].shape[0] // 2
            k = np.asarray(f["kspace"][m])
            t = np.asarray(f[tkey][m])
        tf = center_crop2d(np.abs(ifft2c(k)), (t.shape[-2], t.shape[-1]))
        # A/B construction stats on the same slice
        imn = ifft2c(k)
        iA = center_crop2d(imn, CROP)
        iB = ifft2c(center_crop2d(k, CROP))
        return {"split": split, "file": str(fp.relative_to(root)),
                "slice": int(m),
                "rel_l2": float(np.linalg.norm(tf - t) /
                                max(np.linalg.norm(t), 1e-12)),
                "ab_complex_rel_l2": float(np.linalg.norm(iA - iB) /
                                           max(np.linalg.norm(iA), 1e-12)),
                "ab_magnitude_rel_l2": float(
                    np.linalg.norm(np.abs(iA) - np.abs(iB)) /
                    max(np.linalg.norm(np.abs(iA)), 1e-12))}
    vrng = np.random.default_rng(args.stat_seed + 1)
    tv_records = []
    for flist, sname in ((tr_files, "train"), (va_files, "val")):
        pool = [fp for fp in flist
                if str(fp.relative_to(root)) not in excluded_files]
        nsel = min(5, len(pool))
        if args.mode == "closing" and nsel < 5:
            fail(f"closing mode: fewer than 5 non-excluded files in {sname} "
                 f"({len(pool)} available after excluding "
                 f"{len(flist) - len(pool)})")
        idx = sorted(vrng.choice(len(pool), size=nsel, replace=False))
        tv_records += [target_rel_for(pool[i], sname) for i in idx]
    overlap = {r["file"] for r in tv_records} & excluded_files
    if overlap:
        fail(f"closing-sample overlap with exploratory record: {sorted(overlap)}")
    if args.mode == "closing":
        # the fixed VISUAL sample must also avoid the exploratory files
        default_visual = str(va_files[0].relative_to(root))
        visual = (args.sample_file if args.sample_file else default_visual)
        vmatch = [v for v in excluded_files
                  if v == visual or v.endswith("/" + str(visual))]
        if vmatch:
            fail(f"closing-mode fixed visual sample {visual!r} overlaps the "
                 "exploratory record; pass --sample-file from the closing "
                 "sample instead")
    rels = [r["rel_l2"] for r in tv_records]
    ab_c = [r["ab_complex_rel_l2"] for r in tv_records]
    ab_m = [r["ab_magnitude_rel_l2"] for r in tv_records]
    facts["construction_comparison_multisample"] = {
        "n_checked": len(tv_records),
        "complex_rel_l2": {"min": float(np.min(ab_c)),
                           "median": float(np.median(ab_c)),
                           "max": float(np.max(ab_c))},
        "magnitude_rel_l2": {"min": float(np.min(ab_m)),
                             "median": float(np.median(ab_m)),
                             "max": float(np.max(ab_m))}}
    facts["target_vs_ifft_multisample"] = {
        "mode": args.mode,
        "n_excluded_prior_files": len(excluded_files),
        "n_checked": len(tv_records),
        "rel_l2_min": float(np.min(rels)),
        "rel_l2_median": float(np.median(rels)),
        "rel_l2_max": float(np.max(rels)),
        "records": tv_records,
        "sample_seed": args.stat_seed + 1,
        "tolerance_policy": (
            "Exploratory; no automatic pass/fail -- an engineering "
            "tolerance is locked in EXEC from this observed distribution "
            "BEFORE the G0-closing rerun (different pre-registered seed, "
            "non-overlapping sample, tolerance applied unchanged)"
            if args.mode == "exploratory" else
            "Closing; locked tolerance applied unchanged to a "
            "non-overlapping, differently-seeded sample")}
    if args.mode == "closing":
        facts["target_vs_ifft_multisample"]["locked_tolerance"] = \
            args.target_rel_tolerance
        if float(np.max(rels)) > args.target_rel_tolerance:
            fail(f"CLOSING CHECK FAILED: max rel_l2 {float(np.max(rels)):.3e} "
                 f"> locked tolerance {args.target_rel_tolerance:.3e}")
        facts["target_vs_ifft_multisample"]["closing_check"] = "PASSED"


    # ---- 4. fixed inspection sample (VAL) + SS3a plots ----
    if args.sample_file:
        matches = [p for p in va_files if p.name == args.sample_file]
        if not matches:
            fail(f"--sample-file {args.sample_file} not found under val split")
        if len(matches) > 1:
            fail(f"--sample-file {args.sample_file} ambiguous: "
                 f"{[str(m.relative_to(root)) for m in matches]}")
        sample_fp = matches[0]
    else:
        sample_fp = va_files[0]
    with h5py.File(sample_fp, "r") as f:
        if "kspace" not in f:
            fail(f"'kspace' missing in sample {sample_fp.name}")
        if tkey not in f:
            fail(f"target key '{tkey}' missing in sample {sample_fp.name}")
        n = f["kspace"].shape[0]
        sl = n // 2 if args.sample_slice < 0 else args.sample_slice
        if not (0 <= sl < n):
            fail(f"slice {sl} out of range 0..{n-1} for {sample_fp.name}")
        ksp = np.asarray(f["kspace"][sl])     # complex, native
        tgt = np.asarray(f[tkey][sl])         # real magnitude, native crop
    facts["inspection_sample"] = {
        "file": str(sample_fp.relative_to(root)), "slice": int(sl),
        "file_bytes": sample_fp.stat().st_size,
        "kspace_shape": list(ksp.shape), "kspace_dtype": str(ksp.dtype),
        "target_shape": list(tgt.shape)}
    hdr_tr = decode_ismrmrd_header(tr_files[0])
    hdr_va = decode_ismrmrd_header(sample_fp)
    facts["ismrmrd_header_train_file0"] = hdr_tr
    facts["ismrmrd_header_val_sample"] = hdr_va
    sigs = []
    for rec in tv_records:                       # SPOT CHECK, seeded files
        h = decode_ismrmrd_header(root / rec["file"])
        sigs.append({"file": rec["file"],
                     "matrix": h.get("encodedSpace_matrix"),
                     "pe_limits": h.get("phase_encode_step1_limits"),
                     "present": h.get("present")})
    n_present = sum(1 for s in sigs if s["present"])
    uniq = {json.dumps({"m": s["matrix"], "p": s["pe_limits"]}, sort_keys=True)
            for s in sigs if s["present"]}
    facts["ismrmrd_header_spot_check"] = {
        "scope": "SPOT CHECK (seeded sample files), not a whole-dataset proof",
        "n_checked": len(sigs), "n_present": n_present,
        "n_absent": len(sigs) - n_present,
        "n_unique_present_signatures": len(uniq),
        "axis_note": "matrix sizes + PE limits alone do not prove which "
                     "NumPy axis is sampled; confirm from observed array "
                     "dims + mask visualization before locking 3.7",
        "signatures": sigs}
    facts["sampling_axis_note"] = ("axis roles resolved at G0 3.7 from the "
                                   "decoded headers above; mask axis remains "
                                   "PROVISIONAL until then")

    img_native = ifft2c(ksp)

    # Visual example (fixed sample): supplied HDF5 target vs |IFFT(kspace)| center-
    # cropped to the target's own (possibly rectangular) shape -- the sanity
    # check on FFT shifts, axes, and alignment.
    target_from_kspace = center_crop2d(np.abs(img_native),
                                       (tgt.shape[-2], tgt.shape[-1]))
    t_rel = float(np.linalg.norm(target_from_kspace - tgt) /
                  max(np.linalg.norm(tgt), 1e-12))
    facts["target_vs_ifft_validation"] = {
        "target_shape": list(tgt.shape),
        "rel_l2": t_rel,
        "mean_abs_diff": float(np.abs(target_from_kspace - tgt).mean()),
        "note": "supplied reconstruction target vs center-cropped "
                "|IFFT(kspace)|; large rel_l2 indicates an FFT-shift/axis/"
                "normalization convention mismatch to resolve before G0 lock"}
    imsave(plots / "0a_target_supplied.png", tgt, "supplied HDF5 target")
    imsave(plots / "0b_target_from_kspace.png", target_from_kspace,
           "center-cropped |IFFT(kspace)|")
    imsave(plots / "0c_target_absdiff.png", np.abs(target_from_kspace - tgt),
           f"target |diff| (rel_l2={t_rel:.3e})")

    # Construction A: image-center-crop -> FFT · B: k-space-center-crop -> IFFT
    imgA = center_crop2d(img_native, CROP)
    kspA = fft2c(imgA)
    kspB = center_crop2d(ksp, CROP)
    imgB = ifft2c(kspB)
    mag_diff = np.abs(np.abs(imgA) - np.abs(imgB))
    facts["construction_comparison"] = {
        "complex_rel_l2": float(np.linalg.norm(imgA - imgB) /
                                max(np.linalg.norm(imgA), 1e-12)),
        "magnitude_rel_l2": float(np.linalg.norm(mag_diff) /
                                  max(np.linalg.norm(np.abs(imgA)), 1e-12)),
        "magnitude_mean_abs_diff": float(mag_diff.mean()),
        "magnitude_max_abs_diff": float(mag_diff.max()),
        "note": "A=image-crop->FFT, B=kspace-crop->IFFT; NOT equivalent -- "
                "G0 3.2 decides which defines the 96x96 problem; complex "
                "rel-L2 included because state and operator are complex"}

    # Reference under the PROVISIONAL construction chosen by --construction:
    x_ref = imgA if args.construction == "A" else imgB
    ksp_ref = kspA if args.construction == "A" else kspB
    cname = f"provisional construction {args.construction}"

    mask = cartesian_mask((CROP, CROP), ACCEL, CENTER_FRAC, MASK_SEED)
    facts["provisional_mask"]["sampled_fraction"] = float(mask.mean())
    y = ksp_ref * mask                    # measurement under x_ref
    zf = ifft2c(y)                        # x0 = A^H y (zero-filled)
    # Sanity fact: operator residual of zero-filled recon is mathematically
    # zero; numerically expected near machine precision.
    op_resid = y - mask * fft2c(zf)
    facts["zero_filled_operator_residual_norm"] = float(np.linalg.norm(op_resid))
    # ORACLE diagnostic (uses the x_ref of the provisional construction;
    # NEVER a model input): e = x_ref - x0 = ifft2c(ksp_ref - y).
    e = x_ref - zf
    # Deployable residual channels A^H(y - A x0) are mathematically zero for
    # zero-filled x0 -- recorded above; stats deferred to a flow-base x0.

    # SS3a required plot set (+ diagnostic subplots)
    imsave(plots / "1_native_target_magnitude.png", tgt, "native target |x|")
    imsave(plots / "2a_96_target_imgcrop_fft.png", np.abs(imgA),
           "96x96 target -- A: image-crop->FFT")
    imsave(plots / "2b_96_target_kspcrop_ifft.png", np.abs(imgB),
           "96x96 target -- B: kspace-crop->IFFT")
    imsave(plots / "2c_construction_absdiff.png", mag_diff,
           "| |A| - |B| | (decides G0 3.2)")
    imsave(plots / "3_full_kspace_logmag.png", ksp, "native k-space log|k|",
           log=True)
    imsave(plots / "4_sampling_mask.png", mask.astype(float),
           f"provisional mask {ACCEL}x, axis PROVISIONAL (INSPECTION ONLY)")
    imsave(plots / "5_masked_kspace_logmag.png", y,
           f"masked k-space log|y| ({cname})", log=True)
    imsave(plots / "6_zero_filled_magnitude.png", np.abs(zf),
           f"zero-filled |x0| = |A^H y| ({cname})")
    imsave(plots / "7a_oracle_error_real.png", np.real(e),
           f"ORACLE Re(e) = Re(x_ref - x0), {cname} -- diagnostics only")
    imsave(plots / "7b_oracle_error_imag.png", np.imag(e),
           f"ORACLE Im(e) = Im(x_ref - x0), {cname} -- diagnostics only")

    # channel histograms: |x0| (deployable) + oracle-e (diagnostics only).
    # Deployable residual-channel stats are deferred (see residual_convention).
    chans = {"abs_x0_zf__deployable": np.abs(zf).ravel(),
             "re_e__ORACLE_diag_only": np.real(e).ravel(),
             "im_e__ORACLE_diag_only": np.imag(e).ravel()}
    plt.figure(figsize=(9, 3))
    for i, (name, v) in enumerate(chans.items(), 1):
        plt.subplot(1, 3, i)
        plt.hist(v, bins=100)
        plt.title(name, fontsize=8)
        plt.yscale("log")
        facts[f"channel_{name}"] = pcts(v)
    plt.tight_layout()
    plt.savefig(plots / "8_channel_histograms.png", dpi=140)
    plt.close()

    with open(tmp_out / "facts.json", "w") as fh:
        json.dump(facts, fh, indent=2)
    if OUT_DIR.exists():                  # atomic-ish replace on success only
        shutil.rmtree(OUT_DIR)
    tmp_out.rename(OUT_DIR)
    logger.info("facts.json + %d plots written to %s",
                len(list((OUT_DIR / 'plots').glob('*.png'))), OUT_DIR)
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("[s3_inspect] FAILED")
        sys.exit(1)
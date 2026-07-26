# =============================================================================
# SEQREF-I2 v0.1 -- scripts._diag.i2_baseline
# LIFETIME: DIAGNOSTIC
# Purpose: I2 evidence run. Sections:
#   self_tests        contract smoke tests (run FIRST, cheap, CPU):
#                     conditioner accepts (B,3,96,96); rejects 1-channel;
#                     CplRegRefiner without dim/in_channels -> TypeError;
#                     normalize_channels without ChannelScales -> TypeError;
#                     model_channels REJECTS provisional provenance.
#   data_range        provisional metric range per slice: HDF5 file-attr
#                     `max` -- REQUIRED present, finite, > 0; NO fallback to
#                     per-slice max. Label: provisional-I2-file-attr-max.
#                     RECORDED CAVEAT: the attr is computed over the full
#                     320x320 ESC volume; the 96x96 crop may exclude the
#                     volume max, so this range slightly OVERESTIMATES the
#                     crop's own range (small uniform upward PSNR bias,
#                     consistent + reproducible). Verified + locked at the
#                     section-6 hand-computed metric-sanity check.
#   baseline          zero-filled x0 = A^H y over the FULL official val
#                     split (eval-mode deterministic masks): per-slice PSNR/
#                     SSIM vs target_mag -> per-slice distribution summary +
#                     per-volume table + global mean (3.13 baseline).
#   identity_checks   EXPLICITLY LABELLED identity checks, NOT informative
#                     residual evidence: consistency(x0,y) and residual-
#                     channel magnitudes for exact zero-filled x0 are
#                     expected ~0 (idempotent-mask identity); the first
#                     informative residuals come from non-data-consistent
#                     x0 (I4+ flow bases).
#   provisional_stats |x0| channel p50/p99/max over a seeded subset --
#                     provenance provisional-zero-filled, DIAGNOSTIC ONLY
#                     (final scales from the frozen I7 winner, EXEC 3.8).
#   provenance        git commit + dirty (dirty -> RAISE), script hashes,
#                     argv, versions.
# CONVENTION: logger.error + raise; any failed check aborts.
# Invocation (repo root, venv active):
#   python -m seqref_mri.scripts._diag.i2_baseline \
#       --data-root seqref_mri/data/fastmri [--smoke N]
# Changelog (NEW in v0.1): Introduced.
# Update summary: produces the 3.13 zero-filled baseline and the I2
#   evidence set under the adopted framing (identity checks labelled;
#   provisional vs locked scales structurally separated).
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from seqref_mri.src.fastmri_data import FastMRISliceDataset
from seqref_mri.src.forward_operator import (MaskedFourierOperator,
                                             two_channel_to_complex)
from seqref_mri.src.refiners.channel_assembly import (ChannelScales,
                                                      assemble_raw_channels,
                                                      normalize_channels,
                                                      model_channels)
from seqref_mri.src.metrics import psnr_per_sample, ssim_per_sample
from seqref_mri.src.conditioner import Conditioner
from seqref_mri.src.refiners.coupling_regressor import CplRegRefiner

logger = logging.getLogger("seqref_mri.i2_baseline")

__version__ = "0.1"
__abbr__ = "SEQREF-I2"

DATA_RANGE_LABEL = "provisional-I2-file-attr-max"
STAT_SUBSET = 256          # seeded subset for provisional |x0| stats
STAT_SEED = 20260902


def _fail(msg: str) -> None:
    logger.error("[i2] %s", msg)
    raise RuntimeError(msg)


def provenance(argv: list[str]) -> dict:
    # NO FALLBACK; formal evidence requires clean git provenance.
    try:
        commit = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"],
                                capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    capture_output=True, text=True,
                                    check=True).stdout.strip())
    except Exception as e:
        logger.error("[i2] git provenance unobtainable: %r", e)
        raise RuntimeError(f"git provenance unobtainable: {e!r}") from e
    if dirty:
        _fail("working tree DIRTY -- commit before the formal I2 run")
    import seqref_mri.src.refiners.channel_assembly as ca
    import seqref_mri.src.metrics as me
    import seqref_mri.src.conditioner as co
    import seqref_mri.src.refiners.coupling_regressor as cr
    hashes = {}
    for mod in (ca, me, co, cr, sys.modules[__name__]):
        p = Path(mod.__file__)
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return {"git_commit": commit, "git_dirty": dirty,
            "script_sha256": hashes, "argv": argv,
            "python": sys.version.split()[0], "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available()}


# ---- contract self-tests -----------------------------------------------------
def self_tests() -> dict:
    out = {}
    cond = Conditioner(in_channels=3, width=64, h_dim=128)
    h = cond(torch.zeros(2, 3, 96, 96))
    if h.shape != (2, 128):
        _fail(f"conditioner output shape {tuple(h.shape)} != (2,128)")
    out["conditioner_accepts_3ch_96"] = True

    try:
        cond(torch.zeros(2, 1, 96, 96))
        _fail("conditioner ACCEPTED 1-channel input -- contract broken")
    except ValueError:
        out["conditioner_rejects_1ch"] = True

    try:
        Conditioner(width=64, h_dim=128)          # type: ignore[call-arg]
        _fail("Conditioner constructed WITHOUT in_channels")
    except TypeError:
        out["conditioner_requires_in_channels"] = True

    try:
        CplRegRefiner(flavor="nice", in_channels=3)   # type: ignore[call-arg]
        _fail("CplRegRefiner constructed WITHOUT dim")
    except TypeError:
        out["cplreg_requires_dim"] = True

    raw = torch.rand(3, 96, 96)
    try:
        normalize_channels(raw, {"s_mag": 1.0})       # type: ignore[arg-type]
        _fail("normalize_channels ACCEPTED non-ChannelScales scales")
    except TypeError:
        out["normalize_requires_declared_scales"] = True

    try:
        ChannelScales(1.0, 1e-12, 1.0, "provisional-zero-filled")
        _fail("ChannelScales ACCEPTED a scale <= 1e-8")
    except ValueError:
        out["scales_reject_below_min"] = True

    prov = ChannelScales(1.0, 1.0, 1.0, "provisional-zero-filled")
    mask = torch.ones(96, dtype=torch.bool)
    op = MaskedFourierOperator(mask)
    x0 = torch.complex(torch.rand(96, 96), torch.rand(96, 96))
    y = op.A(x0)
    try:
        model_channels(x0, y, op, prov)
        _fail("model_channels ACCEPTED provisional provenance")
    except ValueError:
        out["model_rejects_provisional_scales"] = True

    locked = ChannelScales(1.0, 1.0, 1.0, "locked-I7")
    ch = model_channels(x0, y, op, locked)
    if ch.shape != (3, 96, 96):
        _fail(f"model_channels shape {tuple(ch.shape)} != (3,96,96)")
    out["model_accepts_locked_scales"] = True
    return out


# ---- data range (provisional convention) -------------------------------------
def file_attr_max(ds: FastMRISliceDataset, i: int) -> float:
    path, _ = ds.index[i]
    with h5py.File(path, "r") as h:
        if "max" not in h.attrs:
            _fail(f"{path.name}: file attr 'max' MISSING -- no fallback "
                  "(provisional data-range convention requires it)")
        v = float(h.attrs["max"])
    if not np.isfinite(v) or v <= 0.0:
        _fail(f"{path.name}: file attr 'max' = {v!r} not finite/>0")
    return v


# ---- baseline ----------------------------------------------------------------
def run_baseline(ds: FastMRISliceDataset, device: str,
                 limit: int | None) -> dict:
    n = len(ds) if limit is None else min(limit, len(ds))
    per_slice_psnr = np.zeros(n)
    per_slice_ssim = np.zeros(n)
    per_slice_cons = np.zeros(n)
    max_res_abs = 0.0
    vols: dict[str, list[int]] = {}
    with torch.no_grad():
        for i in range(n):
            item = ds[i]
            dr = file_attr_max(ds, i)
            y = item["y"].to(device)
            mask = item["mask"].to(device)
            op = MaskedFourierOperator(mask)
            x0_c = op.A_adjoint(y)                       # zero-filled state
            raw = assemble_raw_channels(x0_c, y, op)
            max_res_abs = max(max_res_abs,
                              float(raw[1:].abs().max().item()))
            x0_mag = raw[0].unsqueeze(0).unsqueeze(0).cpu()
            tgt = item["target_mag"].unsqueeze(0).unsqueeze(0)
            per_slice_psnr[i] = psnr_per_sample(
                x0_mag, tgt, data_range=dr).item()
            per_slice_ssim[i] = ssim_per_sample(
                x0_mag, tgt, data_range=dr).item()
            per_slice_cons[i] = float(op.consistency(x0_c, y).item())
            vols.setdefault(item["meta"]["file"], []).append(i)
            if (i + 1) % 500 == 0:
                logger.info("[i2] baseline %d/%d", i + 1, n)

    def _summary(a: np.ndarray) -> dict:
        return {"mean": float(a.mean()), "median": float(np.median(a)),
                "std": float(a.std()), "p5": float(np.percentile(a, 5)),
                "p95": float(np.percentile(a, 95)),
                "min": float(a.min()), "max": float(a.max())}

    per_volume = [
        {"file": f, "n_slices": len(idx),
         "psnr_mean": float(per_slice_psnr[idx].mean()),
         "ssim_mean": float(per_slice_ssim[idx].mean())}
        for f, idx in sorted(vols.items())]
    return {
        "n_slices_evaluated": n,
        "data_range_convention": {
            "label": DATA_RANGE_LABEL,
            "definition": "HDF5 file-level attr 'max'; required present, "
                          "finite, > 0; no per-slice-max fallback",
            "caveat": "attr computed over the full 320x320 ESC volume; the "
                      "96x96 crop may exclude the volume max, so the range "
                      "slightly overestimates the crop's own range (small "
                      "uniform upward PSNR bias). Verify + lock at the "
                      "section-6 hand-computed metric-sanity check."},
        "per_slice": {"psnr": _summary(per_slice_psnr),
                      "ssim": _summary(per_slice_ssim)},
        "per_volume": per_volume,
        "global": {"psnr_mean": float(per_slice_psnr.mean()),
                   "ssim_mean": float(per_slice_ssim.mean())},
        "identity_checks": {
            "LABEL": "IDENTITY CHECKS -- expected ~0 for exact zero-filled "
                     "x0 = A^H y (idempotent-mask identity). NOT informative "
                     "residual evidence; first informative residuals come "
                     "from non-data-consistent x0 (I4+ flow bases).",
            "consistency_max": float(per_slice_cons.max()),
            "consistency_mean": float(per_slice_cons.mean()),
            "residual_channels_max_abs": max_res_abs},
        "_arrays": {"psnr": per_slice_psnr, "ssim": per_slice_ssim},
    }


def provisional_stats(ds: FastMRISliceDataset, device: str) -> dict:
    rng = np.random.Generator(np.random.PCG64(STAT_SEED))
    idx = rng.choice(len(ds), size=min(STAT_SUBSET, len(ds)), replace=False)
    mags = []
    with torch.no_grad():
        for i in idx:
            item = ds[int(i)]
            op = MaskedFourierOperator(item["mask"].to(device))
            x0_c = op.A_adjoint(item["y"].to(device))
            mags.append(x0_c.abs().flatten().cpu())
    v = torch.cat(mags).numpy()
    return {"PROVENANCE": "provisional-zero-filled -- DIAGNOSTIC ONLY; "
                          "rejected by model_channels by construction. Final "
                          "scales from the frozen I7 winner (EXEC 3.8).",
            "channel": "|x0| (zero-filled)",
            "subset": {"n_slices": int(len(idx)), "seed": STAT_SEED},
            "p50": float(np.percentile(v, 50)),
            "p99": float(np.percentile(v, 99)),
            "max": float(v.max()),
            "residual_channels_note": "Re/Im(A^H r) statistics NOT estimated "
                                      "-- mathematically ~0 for zero-filled "
                                      "x0 (see identity_checks)."}


def make_plots(bl: dict, ds: FastMRISliceDataset, plots_dir: Path,
               device: str) -> list[str]:
    written = []
    ps = bl["_arrays"]["psnr"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(ps, bins=60, color="gray")
    ax.set_xlabel("per-slice PSNR (dB)"); ax.set_ylabel("count")
    ax.set_title("zero-filled baseline PSNR, val split", fontsize=9)
    f = plots_dir / "01_psnr_hist.png"
    fig.savefig(f, dpi=120, bbox_inches="tight"); plt.close(fig)
    written.append(f.name)

    pv = [r["psnr_mean"] for r in bl["per_volume"]]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(sorted(pv), marker=".", linestyle="none", color="black")
    ax.set_xlabel("volume (sorted)"); ax.set_ylabel("mean PSNR (dB)")
    ax.set_title("per-volume mean PSNR", fontsize=9)
    f = plots_dir / "02_per_volume_psnr.png"
    fig.savefig(f, dpi=120, bbox_inches="tight"); plt.close(fig)
    written.append(f.name)

    item = ds[len(ds) // 2]
    op = MaskedFourierOperator(item["mask"].to(device))
    x0_mag = op.A_adjoint(item["y"].to(device)).abs().cpu()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, img, title in ((axes[0], x0_mag, "|x0| zero-filled"),
                           (axes[1], item["target_mag"], "target |x_true|")):
        ax.imshow(img.numpy(), cmap="gray"); ax.set_title(title, fontsize=9)
        ax.axis("off")
    f = plots_dir / "03_zero_filled_vs_target.png"
    fig.savefig(f, dpi=120, bbox_inches="tight"); plt.close(fig)
    written.append(f.name)
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--smoke", type=int, default=None,
                   help="limit to N slices (smoke only; formal run omits)")
    a = p.parse_args()
    out_dir = Path(a.out) if a.out else (
        Path(a.data_root).parent.parent / "results" / "_diag" / "i2")
    tmp = out_dir.parent / (out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "plots").mkdir(parents=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    facts: dict = {"script": f"{__abbr__} v{__version__} i2_baseline",
                   "smoke_limit": a.smoke}
    facts["provenance"] = provenance(sys.argv)
    facts["self_tests"] = self_tests()
    ds = FastMRISliceDataset(a.data_root, split="val", mode="eval")
    bl = run_baseline(ds, device, a.smoke)
    facts["plots"] = make_plots(bl, ds, tmp / "plots", device)
    bl.pop("_arrays")
    facts["baseline"] = bl
    facts["provisional_stats"] = provisional_stats(ds, device)
    facts["verdict"] = ("PASS (smoke)" if a.smoke else "PASS")

    with open(tmp / "facts.json", "w") as f:
        json.dump(facts, f, indent=2)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    logger.info("[i2] %s -- report at %s", facts["verdict"], out_dir)


if __name__ == "__main__":
    main()

# =============================================================================
# SEQREF-I1 v0.2 -- scripts._diag.i1_operator_report
# LIFETIME: DIAGNOSTIC
# Purpose: I1 competence report -- BOTH operator identities/edge cases AND
#   loader/data-contract competence (amended I1 scope). Emits facts.json +
#   plots to results/_diag/i1/ (atomic .tmp rename).
# Sections in facts.json:
#   loader_contract     one train + one eval sample: shapes/dtypes of every
#                       item field, sample provenance (file/slice/seed),
#                       split enforcement note, ESC-crop vs |x_true|
#                       agreement (construction-A check, rel-L2)
#   mask_invariants     for widths {96, 32, 64, 100}: exact n_center, exact
#                       n_total, centred low-frequency block, no duplicate
#                       columns (bool mask => structural), sampled fraction
#                       == n_total/W exactly; determinism: eval mask
#                       reproducible (same seed twice identical), train mask
#                       epoch-varying (epoch 0 vs 1 differ)
#   operator            full-sample round-trip |x - AᴴAx| (full mask);
#                       zero-mask and full-mask edge cases; imaginary
#                       leakage where a real quantity is expected; sampled-
#                       entry fraction under the campaign mask; FFT
#                       normalization mode recorded ('ortho', centred)
#   adjoint_preflight   embedded run_preflight() result (CPU always; GPU if
#                       CUDA available, else explicit 'cuda_not_available')
# Plots (6 + full-mask identity panel): x · |AᴴAx| · |x - AᴴAx| · |Aᴴr| ·
#   sampled-vs-unsampled k-space · residual k-space magnitude.
# CONVENTION: logger.error + raise; any failed check aborts the report.
# Invocation (from repo root, venv active):
#   python -m seqref_mri.scripts._diag.i1_operator_report \
#       --data-root seqref_mri/data/fastmri
# Changelog (v0.1 -> v0.2, pre-deployment review fixes):
#   * Epoch-varying mask check is now a HARD FAILURE on the fixed
#     competence-test tuple (was warn-and-pass).
#   * Zero-mask adjoint test uses a NONZERO complex residual (the trivial
#     x*0 input would pass even with broken masking).
#   * Report provenance recorded: git commit + dirty status, script
#     sha256s, argv, python/torch versions, CUDA availability. Provenance
#     failure RAISES (no "unknown" fallback -- a formal report must not
#     PASS without valid git provenance).
#   * Sampled-vs-unsampled plot is a true side-by-side comparison.
#   * Split enforcement actively tested: constructing split='test' must
#     raise; recorded as a pass/fail check, not prose.
# Changelog (NEW in v0.1):
#   * Introduced.
# Update summary (v0.2): the report's competence checks can no longer pass
#   vacuously -- every promised invariant is either verified or fails hard.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from seqref_mri.src.fastmri_data import (FastMRISliceDataset,
                                         make_cartesian_mask, mask_counts,
                                         canonical_mask_seed, fft2c, ifft2c,
                                         CELL_HW, TRAIN_BASE_SEED,
                                         EVAL_BASE_SEED)
from seqref_mri.src.forward_operator import (MaskedFourierOperator,
                                             two_channel_to_complex)
from seqref_mri.src.adjoint_check import run_preflight

logger = logging.getLogger("seqref_mri.i1_report")

__version__ = "0.2"
__abbr__ = "SEQREF-I1"

_MASK_WIDTHS = (96, 32, 64, 100)


def provenance(argv: list[str]) -> dict:
    # NO FALLBACK: a formal competence report must not PASS without valid
    # git provenance -- failure to obtain commit/dirty status raises.
    try:
        commit = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"],
                                capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    capture_output=True, text=True,
                                    check=True).stdout.strip())
    except Exception as e:
        logger.error("[i1] git provenance unobtainable: %r", e)
        raise RuntimeError(f"git provenance unobtainable: {e!r}") from e
    hashes = {}
    import seqref_mri.src.fastmri_data as fd
    import seqref_mri.src.forward_operator as fo
    import seqref_mri.src.adjoint_check as ac
    for mod in (fd, fo, ac, sys.modules[__name__]):
        p = Path(mod.__file__)
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return {"git_commit": commit, "git_dirty": dirty,
            "script_sha256": hashes, "argv": argv,
            "python": sys.version.split()[0], "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available()}


def _fail(msg: str) -> None:
    logger.error("[i1] %s", msg)
    raise RuntimeError(msg)


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    den = torch.linalg.vector_norm(b).item()
    if den == 0.0:
        _fail("rel_l2 reference has zero norm")
    return (torch.linalg.vector_norm(a - b) / den).item()


# ---- loader / data contract --------------------------------------------------
def loader_contract(data_root: str) -> dict:
    out: dict = {}
    for mode, split in (("train", "train"), ("eval", "val")):
        ds = FastMRISliceDataset(data_root, split=split, mode=mode)
        if mode == "train":
            ds.set_epoch(0)
        item = ds[len(ds) // 2]
        rec = {"n_slices": len(ds), "meta": item["meta"],
               "x_true": [str(item["x_true"].dtype),
                          list(item["x_true"].shape)],
               "y": [str(item["y"].dtype), list(item["y"].shape)],
               "mask": [str(item["mask"].dtype), list(item["mask"].shape)],
               "target_mag": [str(item["target_mag"].dtype),
                              list(item["target_mag"].shape)]}
        if list(item["x_true"].shape) != [2, CELL_HW, CELL_HW]:
            _fail(f"x_true shape {list(item['x_true'].shape)} != contract")
        if item["y"].dtype != torch.complex64:
            _fail(f"y dtype {item['y'].dtype} != complex64")
        # Construction-A agreement: centre-crop of the dataset's own ESC
        # target vs |x_true| (S3-proven convention).
        esc = ds.esc_crop(len(ds) // 2)
        rec["esc_vs_xtrue_rel_l2"] = _rel_l2(item["target_mag"], esc)
        if rec["esc_vs_xtrue_rel_l2"] > 1e-6:   # S3-locked tolerance class
            _fail(f"ESC/|x_true| disagreement {rec['esc_vs_xtrue_rel_l2']:.3e}"
                  " > 1e-6 -- construction A broken")
        out[mode] = rec
    out["split_enforcement"] = ("ESC presence enforced at index build; "
                                "official singlecoil_train / singlecoil_val "
                                "dirs; test split unused (no targets)")
    return out


# ---- mask invariants ---------------------------------------------------------
def mask_invariants() -> dict:
    out: dict = {"widths": {}}
    for w in _MASK_WIDTHS:
        n_center, n_total = mask_counts(w)
        m = make_cartesian_mask(w, seed=12345)
        start = (w - n_center) // 2
        if int(m.sum()) != n_total:
            _fail(f"W={w}: total {int(m.sum())} != {n_total}")
        if not m[start:start + n_center].all():
            _fail(f"W={w}: centred block not fully sampled")
        # bool mask => no duplicates structurally; verify count consistency
        frac = float(m.mean())
        if abs(frac - n_total / w) > 1e-12:
            _fail(f"W={w}: fraction {frac} != {n_total / w}")
        out["widths"][w] = {"n_center": n_center, "n_total": n_total,
                            "fraction_exact": frac}
    # determinism: eval reproducible, train epoch-varying
    rel, sl = "singlecoil_val/file_x.h5", 17
    e_seed = canonical_mask_seed(EVAL_BASE_SEED, rel, sl)
    m1 = make_cartesian_mask(CELL_HW, e_seed)
    m2 = make_cartesian_mask(CELL_HW, e_seed)
    if not np.array_equal(m1, m2):
        _fail("eval mask not deterministic")
    t0 = canonical_mask_seed(TRAIN_BASE_SEED, rel, sl, epoch=0)
    t1 = canonical_mask_seed(TRAIN_BASE_SEED, rel, sl, epoch=1)
    if t0 == t1:
        _fail("train seeds identical across epochs")
    epoch_masks_differ = not np.array_equal(
        make_cartesian_mask(CELL_HW, t0), make_cartesian_mask(CELL_HW, t1))
    if not epoch_masks_differ:
        _fail("train masks identical across epochs 0/1 on the fixed "
              "competence tuple -- fresh-mask-per-epoch policy violated")
    out["determinism"] = {"eval_reproducible": True,
                          "train_seed_epoch0": t0, "train_seed_epoch1": t1,
                          "train_epoch_masks_differ": True}
    return out


def split_enforcement_test(data_root: str) -> dict:
    # Constructing an invalid split MUST raise (only train/val sanctioned;
    # test split has no targets -- EXEC 3.3).
    try:
        FastMRISliceDataset(data_root, split="test", mode="eval")
    except ValueError:
        return {"invalid_split_rejected": True}
    _fail("split='test' was accepted -- split enforcement broken")


# ---- operator identities & edge cases ---------------------------------------
def operator_checks(x_c: torch.Tensor) -> dict:
    out: dict = {"fft_normalization": "ortho, centred (fftshift/ifftshift)"}
    full = MaskedFourierOperator(torch.ones(CELL_HW, dtype=torch.bool))
    zero = MaskedFourierOperator(torch.zeros(CELL_HW, dtype=torch.bool))
    camp_mask = torch.from_numpy(make_cartesian_mask(CELL_HW, seed=777))
    camp = MaskedFourierOperator(camp_mask)
    # full-mask round trip: AᴴA = Fᴴ M F with M = I  =>  identity
    xr = full.A_adjoint(full.A(x_c))
    out["full_mask_roundtrip_rel_l2"] = _rel_l2(xr, x_c)
    if out["full_mask_roundtrip_rel_l2"] > 1e-5:
        _fail("full-mask round trip exceeds 1e-5")
    # zero-mask edge: A x == 0, and Aᴴ of a NONZERO complex residual == 0
    # exactly (a zero input would pass even with broken masking).
    if zero.A(x_c).abs().max().item() != 0.0:
        _fail("zero-mask A(x) not identically zero")
    nonzero_r = fft2c(x_c)          # nonzero complex k-space residual
    if nonzero_r.abs().max().item() == 0.0:
        _fail("test residual unexpectedly zero -- invalid test input")
    if zero.A_adjoint(nonzero_r).abs().max().item() != 0.0:
        _fail("zero-mask adjoint of NONZERO residual not identically zero "
              "-- masking inside A_adjoint broken")
    out["zero_mask_ok"] = True
    # imaginary leakage where a real quantity is expected: symmetric real
    # test image -> |Im(AᴴA x)| under full mask should be numerical noise
    real_x = torch.complex(x_c.real.abs(), torch.zeros_like(x_c.real))
    leak = full.A_adjoint(full.A(real_x)).imag.abs().max().item()
    out["max_imag_leakage_real_input_full_mask"] = leak
    if leak > 1e-4 * real_x.real.abs().max().item():
        _fail(f"imaginary leakage {leak:.3e} too large")
    # campaign-mask numbers
    n_center, n_total = mask_counts(CELL_HW)
    out["campaign_mask"] = {"n_center": n_center, "n_total": n_total,
                            "sampled_fraction": float(camp_mask.float().mean())}
    y = camp.A(x_c)
    out["consistency_on_truth"] = float(camp.consistency(x_c, y).item())
    if out["consistency_on_truth"] > 1e-6:
        _fail("consistency(x_true, A x_true) not ~0")
    return out


# ---- plots -------------------------------------------------------------------
def make_plots(x_c: torch.Tensor, plots_dir: Path) -> list[str]:
    camp_mask = torch.from_numpy(make_cartesian_mask(CELL_HW, seed=777))
    camp = MaskedFourierOperator(camp_mask)
    full = MaskedFourierOperator(torch.ones(CELL_HW, dtype=torch.bool))
    y = camp.A(x_c)
    r = y - camp.A(camp.A_adjoint(y))            # residual in k-space
    panels = [
        ("01_x", x_c.abs(), "|x|"),
        ("02_AhAx_full", full.A_adjoint(full.A(x_c)).abs(),
         "|AᴴAx| (full mask ≈ x)"),
        ("03_x_minus_AhAx_full", (x_c - full.A_adjoint(full.A(x_c))).abs(),
         "|x − AᴴAx| (full mask)"),
        ("04_Ah_r", camp.A_adjoint(r).abs(), "|Aᴴr| (campaign mask)"),
        ("06_residual_kspace", r.abs().log1p(), "log |residual k-space|"),
    ]
    written = []
    for name, img, title in panels:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img.cpu().numpy(), cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        f = plots_dir / f"{name}.png"
        fig.savefig(f, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(f.name)
    # true side-by-side: sampled columns vs unsampled columns of log k-space
    k_log = fft2c(x_c).abs().log1p()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, sel, title in (
            (axes[0], camp_mask.float(), "sampled columns"),
            (axes[1], (~camp_mask).float(), "unsampled columns")):
        ax.imshow((k_log * sel).cpu().numpy(), cmap="gray")
        ax.set_title(f"log k-space, {title}", fontsize=9)
        ax.axis("off")
    f = plots_dir / "05_kspace_sampled_vs_unsampled.png"
    fig.savefig(f, dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append(f.name)
    return sorted(written)


# ---- main --------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", default=None,
                   help="output dir (default <data-root>/../../results/_diag/i1)")
    a = p.parse_args()
    out_dir = Path(a.out) if a.out else (
        Path(a.data_root).parent.parent / "results" / "_diag" / "i1")
    tmp = out_dir.parent / (out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "plots").mkdir(parents=True)

    facts: dict = {"script": f"{__abbr__} v{__version__} i1_operator_report"}
    facts["provenance"] = provenance(sys.argv)
    facts["loader_contract"] = loader_contract(a.data_root)
    facts["loader_contract"]["split_enforcement_test"] = \
        split_enforcement_test(a.data_root)
    facts["mask_invariants"] = mask_invariants()

    ds = FastMRISliceDataset(a.data_root, split="val", mode="eval")
    item = ds[len(ds) // 2]
    x_c = two_channel_to_complex(item["x_true"])
    facts["operator"] = operator_checks(x_c)
    facts["operator"]["sample_provenance"] = item["meta"]
    facts["adjoint_preflight"] = run_preflight()
    facts["plots"] = make_plots(x_c, tmp / "plots")
    facts["verdict"] = "PASS"

    with open(tmp / "facts.json", "w") as f:
        json.dump(facts, f, indent=2)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    logger.info("[i1] PASS -- report at %s", out_dir)


if __name__ == "__main__":
    main()

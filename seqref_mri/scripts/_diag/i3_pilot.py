# =============================================================================
# SEQREF-I3 v0.3 -- scripts._diag.i3_pilot
# LIFETIME: DIAGNOSTIC
# Purpose: the section-6 competence pilot. Sections (run in this order):
#   provenance        clean tree required for formal runs; --smoke permits a
#                     dirty tree with prominent PROVISIONAL labelling.
#   loader_checks     HARD requirements: (1) eval masks identical across two
#                     passes; (2) train masks identical for the same seed AND
#                     epoch; (3) train masks different across epochs.
#   metric_sanity     hand-computed NumPy PSNR + SSIM replica (SAME 11x11
#                     Gaussian window sigma=1.5, reflect padding, K1/K2
#                     constants, per-sample map mean) vs the torch metrics
#                     on a random magnitude pair; agreement <= 1e-5 required.
#                     On PASS this VERIFIES AND LOCKS the data-range
#                     convention: D2-normalized values with data_range = 1.0
#                     (raw file attr recorded alongside -- applied ONCE).
#   spline_stats      D3, FULLY PREDECLARED (nothing chosen after seeing the
#                     data): TRAINING split only; values = normalized Re AND
#                     Im of x_true; seeded 512-slice subset (seed 20260903);
#                     q = percentile(|values|, 99.9); B = 1.10 * q (margin
#                     FIXED in advance); K = 8; linear tails outside [-B, B].
#   memory_smoke      per expert (nice/realnvp/nsf): build at dim 18432,
#                     1 batch (batch=8) forward+backward on GPU; peak VRAM
#                     recorded; ACCEPT <= 10.0 GB (>= 2 GB headroom on 12).
#   short_runs        per expert: 2 epochs, 2000 train / 500 val slices,
#                     seeded, via train_base.run_training (imported, no
#                     subprocess). NLL GATE (numeric): epoch-2 mean train
#                     NLL < epoch-1 mean train NLL. Timing recorded SPLIT
#                     (t_data / t_fb / t_val / t_sample); ACCEPT epoch total
#                     <= 1800 s per expert on the pilot subset.
# Any failed acceptance criterion = pilot FAIL; no budgets locked.
# CONVENTION: logger.error + raise. No fallback.
# Invocation: python -m seqref_mri.scripts._diag.i3_pilot \
#     --data-root seqref_mri/data/fastmri [--smoke]
#   (--smoke: dirty tree permitted + short_runs shrunk to 200/50 slices,
#    1 epoch, memory smoke still full-size -- catches code bugs cheaply.)
# Changelog (v0.2 -> v0.3, pre-deployment review fixes):
#   * metric_sanity is now END-TO-END: in addition to the synthetic pair,
#     one REAL validation item is normalized by its recorded file_attr_max
#     (applied exactly once), evaluated as zero-filled-vs-target through
#     BOTH torch and the independent NumPy replica at data_range=1.0;
#     agreement <= 1e-5 required; raw attr + normalized range recorded.
#   * torch.cuda.synchronize() before reading peak VRAM in memory_smoke.
# Changelog (v0.1 -> v0.2, pre-deployment review fixes):
#   * loader_checks now exercises the REAL pipeline: actual train/eval
#     datasets, make_train_loader with num_workers=2 (worker propagation),
#     set_epoch, and repeated full iteration -- comparing the masks
#     actually returned per (file, slice): eval pass1==pass2; train
#     epoch-0 pass1==pass2; train epoch-0 != epoch-1. (v0.1 only re-tested
#     the helper functions, missing the guarded failure mode.)
#   * Run paths recorded RELATIVE to the report root (runs/<name>) so the
#     staging-rename cannot leave stale absolute paths in facts.json.
#   * Provenance hash set extended: + train_utils.py, conditioner.py,
#     forward_operator.py (all affect the executed pilot).
# Changelog (NEW in v0.1): Introduced.
# Update summary: single diagnostic producing every number EXEC 3.15 needs
#   (s/epoch, VRAM, D3 spline values) plus the metric-convention lock.
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from seqref_mri.src.fastmri_data import FastMRISliceDataset
from seqref_mri.src.metrics import psnr_per_sample, ssim_per_sample
from seqref_mri.scripts.train_base import run_training

logger = logging.getLogger("seqref_mri.i3_pilot")

__version__ = "0.3"
__abbr__ = "SEQREF-I3"

EXPERTS = ("nice", "realnvp", "nsf")
SPLINE_SUBSET, SPLINE_SEED = 512, 20260903
SPLINE_PERCENTILE, SPLINE_MARGIN, SPLINE_K = 99.9, 1.10, 8
VRAM_LIMIT_GB = 10.0
EPOCH_LIMIT_S = 1800.0
PILOT = {"epochs": 2, "train_slices": 2000, "val_slices": 500, "batch": 8,
         "lr": 1e-4, "h_dim": 128, "hidden": 256, "cond_width": 64,
         "n_post": 4, "seed_index": 0, "subset_seed": 20260904,
         "n_layers": {"nice": 4, "realnvp": 6, "nsf": 6}}


def _fail(msg: str) -> None:
    logger.error("[i3] %s", msg)
    raise RuntimeError(msg)


def provenance(argv, *, allow_dirty: bool) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"],
                                capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    capture_output=True, text=True,
                                    check=True).stdout.strip())
    except Exception as e:
        logger.error("[i3] git provenance unobtainable: %r", e)
        raise RuntimeError(f"git provenance unobtainable: {e!r}") from e
    if dirty and not allow_dirty:
        _fail("working tree DIRTY -- commit before the formal I3 run")
    if dirty:
        logger.warning("[i3] DIRTY TREE PERMITTED (smoke): output is "
                       "PROVISIONAL -- NOT FORMAL EVIDENCE")
    import seqref_mri.src.fastmri_data as fd
    import seqref_mri.src.base_experts as be
    import seqref_mri.src.metrics as me
    import seqref_mri.src.train_utils as tu
    import seqref_mri.src.conditioner as co
    import seqref_mri.src.forward_operator as fo
    import seqref_mri.scripts.train_base as tb
    hashes = {Path(m.__file__).name:
              hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()[:16]
              for m in (fd, be, me, tu, co, fo, tb, sys.modules[__name__])}
    return {"git_commit": commit, "git_dirty": dirty, "argv": argv,
            "script_sha256": hashes, "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available()}


def _collect_masks(loader) -> dict:
    # {(file, slice): mask bytes} from a REAL iteration.
    out = {}
    for batch in loader:
        for i, meta in enumerate(batch["meta"]):
            out[(meta["file"], meta["slice_index"])] = \
                batch["mask"][i].numpy().tobytes()
    return out


def loader_checks(data_root: str) -> dict:
    # v0.2: exercises the REAL pipeline -- datasets, make_train_loader,
    # num_workers=2 (worker propagation), set_epoch, repeated iteration.
    from torch.utils.data import DataLoader, SubsetRandomSampler
    from seqref_mri.scripts.train_base import _collate
    from seqref_mri.src.fastmri_data import make_train_loader

    n_check = 16
    ev = FastMRISliceDataset(data_root, split="val", mode="eval")
    ev_loader = lambda: DataLoader(
        torch.utils.data.Subset(ev, list(range(n_check))),
        batch_size=4, shuffle=False, num_workers=2, collate_fn=_collate)
    if _collect_masks(ev_loader()) != _collect_masks(ev_loader()):
        _fail("REAL eval loader: masks differ across repeated passes")

    tr = FastMRISliceDataset(data_root, split="train", mode="train")
    def tr_masks(epoch: int) -> dict:
        tr.set_epoch(epoch)
        sampler = SubsetRandomSampler(
            list(range(n_check)),
            generator=torch.Generator().manual_seed(999))
        loader = make_train_loader(tr, batch_size=4, sampler=sampler,
                                   num_workers=2, collate_fn=_collate)
        return _collect_masks(loader)
    e0a, e0b, e1 = tr_masks(0), tr_masks(0), tr_masks(1)
    if e0a != e0b:
        _fail("REAL train loader: masks differ for the SAME seed and epoch")
    if e0a == e1:
        _fail("REAL train loader: masks identical across epochs 0/1")
    return {"pipeline": "real datasets + make_train_loader, num_workers=2, "
                        f"{n_check} slices",
            "eval_repeat_identical": True,
            "train_same_seed_epoch_identical": True,
            "train_epochs_differ": True}


# ---- faithful NumPy SSIM/PSNR replica ---------------------------------------
def _np_gaussian_window(size=11, sigma=1.5):
    ax = np.arange(size) - size // 2
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    w = np.outer(g, g)
    return w / w.sum()


def _np_reflect_conv(img: np.ndarray, w: np.ndarray) -> np.ndarray:
    pad = w.shape[0] // 2
    padded = np.pad(img, pad, mode="reflect")
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = (padded[i:i + w.shape[0], j:j + w.shape[1]] * w).sum()
    return out


def np_psnr(a, b, data_range):
    m = max(float(np.mean((a - b) ** 2)), 1e-12)
    return 10.0 * np.log10(data_range ** 2 / m)


def np_ssim(a, b, data_range, K1=0.01, K2=0.03):
    w = _np_gaussian_window()
    mu1, mu2 = _np_reflect_conv(a, w), _np_reflect_conv(b, w)
    mu1s, mu2s, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = _np_reflect_conv(a * a, w) - mu1s
    s2 = _np_reflect_conv(b * b, w) - mu2s
    s12 = _np_reflect_conv(a * b, w) - mu12
    C1, C2 = (K1 * data_range) ** 2, (K2 * data_range) ** 2
    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / \
               ((mu1s + mu2s + C1) * (s1 + s2 + C2))
    return float(ssim_map.mean())


def _metric_pair_check(a: np.ndarray, b: np.ndarray, tag: str) -> dict:
    ta = torch.from_numpy(a)[None, None]
    tb = torch.from_numpy(b)[None, None]
    d_psnr = abs(float(psnr_per_sample(ta, tb, data_range=1.0)) -
                 np_psnr(a, b, 1.0))
    d_ssim = abs(float(ssim_per_sample(ta, tb, data_range=1.0)) -
                 np_ssim(a, b, 1.0))
    if d_psnr > 1e-5 or d_ssim > 1e-5:
        _fail(f"metric sanity FAILED ({tag}): |dPSNR|={d_psnr:.2e} "
              f"|dSSIM|={d_ssim:.2e} > 1e-5")
    return {"abs_diff_psnr": d_psnr, "abs_diff_ssim": d_ssim}


def metric_sanity(data_root: str) -> dict:
    # (a) synthetic mechanics check
    rng = np.random.Generator(np.random.PCG64(20260905))
    a = rng.random((64, 64), dtype=np.float64).astype(np.float32)
    b = np.clip(a + 0.05 * rng.standard_normal((64, 64)).astype(np.float32),
                0.0, 1.0)
    synth = _metric_pair_check(a, b, "synthetic")
    # (b) END-TO-END on one REAL validation item: normalize by the recorded
    # file attr EXACTLY ONCE, evaluate zero-filled vs target at range 1.0.
    from seqref_mri.src.forward_operator import MaskedFourierOperator
    ds = FastMRISliceDataset(data_root, split="val", mode="eval")
    item = ds[len(ds) // 2]
    amax = item["meta"]["file_attr_max"]
    op = MaskedFourierOperator(item["mask"])
    x0_mag = op.A_adjoint(item["y"]).abs()
    a_real = (x0_mag / amax).numpy().astype(np.float32)
    b_real = (item["target_mag"] / amax).numpy().astype(np.float32)
    real = _metric_pair_check(a_real, b_real, "real-sample")
    return {"synthetic": synth,
            "real_sample": real | {"file": item["meta"]["file"],
                                   "slice_index":
                                       item["meta"]["slice_index"],
                                   "raw_file_attr_max": amax,
                                   "normalized_data_range": 1.0,
                                   "note": "attr applied exactly once"},
            "verdict": "PASS",
            "data_range_convention_LOCKED":
                "D2-normalized values (state+target / file_attr_max), "
                "metrics with data_range = 1.0; raw attr recorded per "
                "sample and applied exactly once"}


def spline_stats(data_root: str, smoke: bool) -> dict:
    ds = FastMRISliceDataset(data_root, split="train", mode="eval")
    # eval mode: deterministic access; masks irrelevant (x_true used).
    n = min(64 if smoke else SPLINE_SUBSET, len(ds))
    rng = np.random.Generator(np.random.PCG64(SPLINE_SEED))
    idx = rng.choice(len(ds), size=n, replace=False)
    vals = []
    for i in idx:
        item = ds[int(i)]
        v = (item["x_true"] / item["meta"]["file_attr_max"]).flatten()
        vals.append(v.numpy())
    v = np.concatenate(vals)
    q = float(np.percentile(np.abs(v), SPLINE_PERCENTILE))
    B = SPLINE_MARGIN * q
    return {"rule": f"q = p{SPLINE_PERCENTILE}(|Re,Im normalized x_true|), "
                    f"B = {SPLINE_MARGIN} * q (margin FIXED pre-run), "
                    f"K = {SPLINE_K}, linear tails outside [-B, B]; "
                    "TRAINING split only",
            "subset": {"n_slices": int(n), "seed": SPLINE_SEED},
            "q_p999": q, "B": float(B), "K": SPLINE_K,
            "abs_max_observed": float(np.abs(v).max())}


def memory_smoke(data_root: str, nsf_B: float) -> dict:
    if not torch.cuda.is_available():
        _fail("memory smoke requires CUDA (the 12 GB constraint is the "
              "question being answered)")
    from seqref_mri.src.conditioner import Conditioner
    from seqref_mri.src.base_experts import build_expert, CondNSF
    ds = FastMRISliceDataset(data_root, split="val", mode="eval")
    from seqref_mri.scripts.train_base import _collate, _prepare, DIM
    batch = _collate([ds[i] for i in range(PILOT["batch"])])
    out = {}
    for name in EXPERTS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cond = Conditioner(in_channels=2, width=PILOT["cond_width"],
                           h_dim=PILOT["h_dim"])
        if name == "nsf":
            model = CondNSF(dim=DIM, h_dim=PILOT["h_dim"], conditioner=cond,
                            hidden=PILOT["hidden"],
                            n_layers=PILOT["n_layers"]["nsf"], K=SPLINE_K,
                            B=nsf_B, use_film=True).cuda()
        else:
            model = build_expert(name, dim=DIM, h_dim=PILOT["h_dim"],
                                 conditioner=cond, hidden=PILOT["hidden"],
                                 use_film=True,
                                 n_layers=PILOT["n_layers"][name]).cuda()
        p = _prepare(batch, "cuda", test0=False)
        nll = -model.log_prob(p["x_norm"].flatten(1), p["cond_in"]).mean()
        nll.backward()
        torch.cuda.synchronize()
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        n_params = sum(q.numel() for q in model.parameters())
        out[name] = {"peak_vram_gb": peak_gb, "n_params": n_params,
                     "pass": peak_gb <= VRAM_LIMIT_GB}
        logger.info("[i3] memory smoke %s: %.2f GB (%.1fM params)",
                    name, peak_gb, n_params / 1e6)
        if peak_gb > VRAM_LIMIT_GB:
            _fail(f"{name}: peak VRAM {peak_gb:.2f} GB > {VRAM_LIMIT_GB} GB")
        del model, cond, nll, p
        torch.cuda.empty_cache()
    return out


def short_runs(data_root: str, nsf_B: float, smoke: bool,
               out_root: str) -> dict:
    out = {}
    for name in EXPERTS:
        cfg = dict(PILOT)
        cfg.pop("n_layers")
        cfg.update({"data_root": data_root, "expert": name,
                    "n_layers": PILOT["n_layers"][name],
                    "nsf_B": nsf_B if name == "nsf" else None,
                    "out_root": out_root, "test0": False})
        if smoke:
            cfg.update({"epochs": 1, "train_slices": 200, "val_slices": 50})
        facts = run_training(cfg)
        h = facts["history"]
        rec = {"epochs": [dict(e, val={k: v for k, v in e["val"].items()})
                          for e in h],
               "best_val_psnr": facts["best_val_psnr"],
               "n_params": facts["n_params"],
               "run_dir_rel": f"runs/{facts['run_dir_rel']}"}
        if not smoke:
            if not (h[1]["train_nll_mean"] < h[0]["train_nll_mean"]):
                _fail(f"{name}: NLL gate FAILED -- epoch-2 mean "
                      f"{h[1]['train_nll_mean']:.4f} >= epoch-1 "
                      f"{h[0]['train_nll_mean']:.4f}")
            worst_epoch = max(e["t_epoch_total_s"] for e in h)
            if worst_epoch > EPOCH_LIMIT_S:
                _fail(f"{name}: epoch total {worst_epoch:.0f}s > "
                      f"{EPOCH_LIMIT_S:.0f}s on the pilot subset")
            rec["nll_gate"] = "PASS (epoch2 < epoch1)"
        out[name] = rec
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    out_dir = Path(a.out) if a.out else (
        Path(a.data_root).parent.parent / "results" / "_diag" / "i3")
    tmp = out_dir.parent / (out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    facts: dict = {"script": f"{__abbr__} v{__version__} i3_pilot",
                   "smoke": a.smoke,
                   "acceptance_criteria": {
                       "peak_vram_gb": VRAM_LIMIT_GB,
                       "epoch_total_s": EPOCH_LIMIT_S,
                       "nll_gate": "epoch2 mean < epoch1 mean",
                       "metric_agreement": 1e-5,
                       "loader": "eval repeat identical; train same-seed "
                                 "identical; train epochs differ"}}
    facts["provenance"] = provenance(sys.argv, allow_dirty=a.smoke)
    if facts["provenance"]["git_dirty"]:
        facts["EVIDENCE_STATUS"] = ("PROVISIONAL -- dirty tree permitted "
                                    "for smoke; NOT formal evidence")
    facts["loader_checks"] = loader_checks(a.data_root)
    facts["metric_sanity"] = metric_sanity(a.data_root)
    facts["spline_stats"] = spline_stats(a.data_root, a.smoke)
    nsf_B = facts["spline_stats"]["B"]
    facts["memory_smoke"] = memory_smoke(a.data_root, nsf_B)
    facts["short_runs"] = short_runs(a.data_root, nsf_B, a.smoke,
                                     str(tmp / "runs"))
    facts["verdict"] = "PASS (smoke, provisional)" if a.smoke else "PASS"

    with open(tmp / "facts.json", "w") as f:
        json.dump(facts, f, indent=2)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    logger.info("[i3] %s -- report at %s", facts["verdict"], out_dir)


if __name__ == "__main__":
    main()

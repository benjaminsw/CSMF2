# SEQREF-MRI-V05LTM v0.3 -- scripts.v05lt_metrics
# LIFETIME: KEEP
# =============================================================================
# Purpose: the ONE metric implementation shared by the trainer's per-epoch
#   validation (I1.6) and the evaluator (I1.7) of SEQREF-MRI-V05LT
#   (194 §8.2 + AM2-C + AM4-A; 195 §6 + PA1):
#   * slice states: tiny_gate._build_slice_states (production u_true,
#     target, |x_true|, map, standardisation) -- reused, not reimplemented;
#   * R0 decode: tiny_gate._decode_z at z = 0;
#   * PSNR: tiny_gate._psnr (metrics.psnr_per_sample, DATA_RANGE 1.0) plus
#     raw MSE (metrics.mse_per_sample) and the AM4-A flags; the cap test is
#     evaluated in the dtype of the computed MSE, exactly mirroring the
#     production clamp (mse.clamp_min(_EPS));
#   * SSIM: metrics.ssim_per_sample on float64 inputs (the decoded image is
#     float64; the float32 target is promoted exactly) -- metrics.py needs
#     equal dtypes and no earlier evaluator fixed this; NMSE_u: tiny_gate._nmse on
#     AM2-C-eligible slices only (None otherwise);
#   * NLL per free real dimension: -model.log_prob_free(target) / d_free in
#     fixed batches of the view order;
#   * aggregation: slice -> volume mean -> equal-weight mean over volumes;
#     NMSE_u volume mean over eligible slices, volume without any -> None
#     and outside the NMSE population; cap/zero-error counts + identities.
#   Validation never changes model state: eval mode + no_grad, previous
#   mode restored; a non-finite metric is a validity failure (194 §8.2).
#   * NLL components (I1.8 b0, G3; 194 §10 plot 5 / tracked values): per
#     slice the prior and log-determinant contributions, from the same
#     production internals log_prob_free uses. Sign convention:
#       log p(u) = log N(z; 0, I) + ldj,  (z, ldj) = flow.encode(u, h),
#       nll_per_dim        = -log p(u) / d_free   (PRODUCTION value from
#                            model.log_prob_free; used for stopping and
#                            selection; never replaced),
#       nll_prior_per_dim  = -log N(z; 0, I) / d_free,
#       nll_logdet_per_dim = -ldj / d_free.
#     Consistency check (defined): the recomposed float32 log-density
#     log N(z) + ldj (same operation order as production) must satisfy
#     |recomposed - production| <= NLL_DECOMP_TOL_REL * max(1, |production|)
#     per slice (compared in float64), else VAL_NLL_DECOMPOSITION_MISMATCH
#     (failed_validity); the absolute deviation is recorded per slice.
# CONVENTION: logger.error + V05LTError(code); no fallback, no placeholder.
# Changelog (v0.3, 2026-10-24; I1.8 b0, G3):
#   * validate(): per-slice nll_prior_per_dim, nll_logdet_per_dim and the
#     recorded consistency deviation; production nll_per_dim unchanged.
# Changelog (v0.2, 2026-10-22; I1.7(c2), item 18):
#   * validate(): typed seams for data loading (VAL_DATA_LOAD), NLL
#     (VAL_NLL_FAILED) and R0 decode (VAL_DECODE_FAILED); non-finite ->
#     VAL_NON_FINITE; all failed_validity (194 §8.2). No metric changed.
# Changelog (NEW in v0.1, 2026-10-21):
#   * Introduced under I1.6(a) (SEQREF-MRI-V05LTI16A v0.1): shared metric,
#     validation and aggregation code with AM4-A cap/zero-error flags.
# =============================================================================
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from seqref_mri.scripts import v05lt_common as C
from seqref_mri.scripts import v05lt_eligibility as E

logger = logging.getLogger("seqref_mri.v05lt_metrics")

__version__ = "0.3"
__abbr__ = "SEQREF-MRI-V05LTM"

BATCH = 32
NLL_DECOMP_TOL_REL = 1e-6        # G3 consistency tolerance (relative)
METRICS = ("psnr", "ssim", "nll_per_dim")       # all slices
NMSE = "nmse_u"                                  # eligible slices only


def _prod():
    import torch
    from seqref_mri.scripts import tiny_gate as tg
    from seqref_mri.src import metrics as mt
    from seqref_mri.src import free_flow_runtime as ffr
    from seqref_mri.scripts.train_base import _collate
    return torch, tg, mt, ffr, _collate


def cap_flags(mse_tensor) -> dict:
    """AM4-A flags for a 0-dim/1-element MSE tensor, compared in the
    tensor's own dtype (the production clamp threshold)."""
    torch, _, mt, _, _ = _prod()
    eps = torch.tensor(mt._EPS, dtype=mse_tensor.dtype)
    m = mse_tensor.reshape(())
    return {"mse": float(m), "psnr_capped": bool(m <= eps),
            "zero_error": bool(m == 0)}


def psnr_record(mag_hat, mag_true) -> dict:
    torch, tg, mt, _, _ = _prod()
    shp = (1, 1, *mag_hat.shape)
    try:
        psnr = tg._psnr(mag_hat, mag_true)
    except Exception as exc:  # TINY StageError on non-finite; re-raised typed
        C.fail("METRIC_PSNR_INVALID", f"{type(exc).__name__}: {exc}")
    rec = cap_flags(mt.mse_per_sample(mag_hat.reshape(shp),
                                      mag_true.reshape(shp))[0])
    rec["psnr"] = float(psnr)
    return rec


def ssim_value(mag_hat, mag_true) -> float:
    torch, tg, mt, _, _ = _prod()
    shp = (1, 1, *mag_hat.shape)
    a = mag_hat.to(torch.float64).reshape(shp)   # decode output is float64;
    b = mag_true.to(torch.float64).reshape(shp)  # float32 target promoted
    try:                                          # exactly (no rounding)
        v = float(mt.ssim_per_sample(a, b, data_range=tg.DATA_RANGE)[0])
    except ValueError as exc:
        C.fail("METRIC_SSIM_INVALID", str(exc))
    C.require(math.isfinite(v), "METRIC_SSIM_INVALID", repr(v))
    return v


def nmse_value(u_hat, u_true) -> float:
    _, tg, _, _, _ = _prod()
    try:
        v = float(tg._nmse(u_hat, u_true))
    except Exception as exc:  # TINY StageError on non-finite; re-raised typed
        C.fail("METRIC_NMSE_INVALID", f"{type(exc).__name__}: {exc}")
    return v


def nll_components(model, u, cond, mask):
    """(log N(z; 0, I), ldj) per sample, float32, from the production
    internals of log_prob_free (condition -> flow.encode -> prior)."""
    torch, _, _, ffr, _ = _prod()
    h = model.condition(cond, mask)
    z, ldj = model.flow.encode(u.to(torch.float32), h)
    prior = ffr._gaussian_logprob(z)
    C.require(bool(torch.isfinite(prior).all() and torch.isfinite(ldj).all()),
              "VAL_NON_FINITE", "NLL components non-finite")
    return prior, ldj


def _val_seam(code: str, where: str, fn, *a, **k):
    """Item 18 (shared validation): V05LT errors pass through; any other
    exception becomes a typed failed_validity (194 §8.2: a non-finite
    metric/decode in validation is a validity failure). Non-finite
    failures get the code VAL_NON_FINITE."""
    try:
        return fn(*a, **k)
    except C.V05LTError:
        raise
    except Exception as exc:  # noqa: BLE001 -- typed, logged, re-raised
        t = f"{type(exc).__name__}: {exc}"
        nonfinite = "non-finite" in t.lower() or "NON_FINITE" in t.upper()
        C.fail("VAL_NON_FINITE" if nonfinite else code, f"{where}: {t}")


def validate(model, view, eligibility: dict, p4: dict, s_ref: float,
             indices=None) -> list:
    """Per-slice R0/NLL rows for view (or the given view indices).
    eligibility: {(file, slice): {"eligible", "midpoint", "volume_id"}}."""
    torch, tg, _, ffr, collate = _prod()
    idx = list(range(len(view))) if indices is None else list(indices)
    was_training = model.training
    model.eval()
    rows = []
    z0 = torch.zeros(1, ffr.FLOW_DIM_REAL)
    try:
        with torch.no_grad():
            for start in range(0, len(idx), BATCH):
                part = idx[start:start + BATCH]
                batch = _val_seam("VAL_DATA_LOAD", f"positions {part[:4]}",
                                  lambda: collate([view[i] for i in part]))
                ents = [view.entry(i) for i in part]
                sel = {"ordered_identities": [{"file": f, "slice_index": s}
                                              for f, s in ents]}
                try:
                    states = tg._build_slice_states(batch, sel, p4, s_ref)
                except Exception as exc:  # TINY StageError; re-raised typed
                    C.fail("VAL_STATE_FAILED", f"{type(exc).__name__}: {exc}")
                tgt = torch.from_numpy(np.concatenate(
                    [st["target"] for st in states], 0).astype(np.float32))
                cond = torch.cat([st["cond"] for st in states], 0)
                msk = torch.cat([st["mask"] for st in states], 0)
                d_free = int(tgt.shape[1])
                C.require(d_free == ffr.FLOW_DIM_REAL, "VAL_DFREE", str(d_free))
                lp = _val_seam("VAL_NLL_FAILED", "log_prob_free",
                               model.log_prob_free, tgt, cond, msk)
                nll = (-lp.to(torch.float64) / d_free).tolist()
                prior, ldj = _val_seam("VAL_NLL_FAILED", "nll components",
                                       nll_components, model, tgt, cond, msk)
                lp64 = lp.to(torch.float64)
                dev = ((prior + ldj).to(torch.float64) - lp64).abs()
                lim = NLL_DECOMP_TOL_REL * lp64.abs().clamp_min(1.0)
                C.require(bool((dev <= lim).all()),
                          "VAL_NLL_DECOMPOSITION_MISMATCH",
                          f"max |recomposed - production| "
                          f"{float(dev.max())} at positions {part[:4]}")
                n_prior = (-prior.to(torch.float64) / d_free).tolist()
                n_ldj = (-ldj.to(torch.float64) / d_free).tolist()
                devs = dev.tolist()
                for j, (st, (f, s)) in enumerate(zip(states, ents)):
                    key = (f, s)
                    C.require(key in eligibility, "VAL_ELIGIBILITY_MISSING",
                              f"{f}|{s}")
                    el = eligibility[key]
                    C.require(bool(el["eligible"]) == (not st["excluded"]),
                              "VAL_ELIGIBILITY_MISMATCH", f"{f}|{s}")
                    x0, u0 = _val_seam("VAL_DECODE_FAILED", f"R0 {f}|{s}",
                                       tg._decode_z, model, z0, st)
                    mag = x0.abs()
                    rec = psnr_record(mag, st["x_true_mag"])
                    row = {"volume_id": el["volume_id"], "file": f,
                           "slice_index": s, "midpoint": bool(el["midpoint"]),
                           "eligible": bool(el["eligible"]),
                           **rec, "ssim": ssim_value(mag, st["x_true_mag"]),
                           NMSE: (nmse_value(u0, st["u_true"])
                                  if el["eligible"] else None),
                           "nll_per_dim": float(nll[j]),
                           "nll_prior_per_dim": float(n_prior[j]),
                           "nll_logdet_per_dim": float(n_ldj[j]),
                           "nll_decomp_abs_dev": float(devs[j])}
                    C.require(math.isfinite(row["nll_per_dim"]),
                              "VAL_NLL_NON_FINITE", f"{f}|{s}")
                    rows.append(row)
    finally:
        model.train(was_training)
    return rows


def _agg(rows: list) -> dict:
    ent = [(r["volume_id"], r["file"], r["slice_index"]) for r in rows]
    key = lambda r: (r["file"], r["slice_index"])  # noqa: E731
    out = {"slices": len(rows), "volumes": len({r["volume_id"] for r in rows})}
    for m in METRICS:
        vm = E.volume_means({key(r): r[m] for r in rows}, ent)
        out[m] = float(np.mean(list(vm.values())))
    el = {key(r): r["eligible"] for r in rows}
    nm = {key(r): r[NMSE] for r in rows if r["eligible"]}
    vm = E.volume_means(nm, ent, el)
    pop = E.nmse_population(vm)
    out[NMSE] = float(np.mean([vm[v] for v in pop])) if pop else None
    out["nmse_population_volumes"] = len(pop)
    out["nmse_null_volumes"] = sorted(v for v, x in vm.items() if x is None)
    for flag in ("psnr_capped", "zero_error"):
        ids = [f"{r['file']}|{r['slice_index']}" for r in rows if r[flag]]
        out[flag] = {"count": len(ids), "identities": ids}
    return out


def summarise(rows: list) -> dict:
    """194 §8.2 aggregation for all slices and the midpoint subset."""
    C.require(len({(r["file"], r["slice_index"]) for r in rows}) == len(rows),
              "AGG_DUPLICATE_SLICE", "")
    mid = [r for r in rows if r["midpoint"]]
    return {"all_slices": _agg(rows), "midpoint": _agg(mid) if mid else None}


def eligibility_index(manifest: dict) -> dict:
    """{(file, slice): {...}} from a loaded eligibility manifest."""
    return {(r["file"], r["slice_index"]): {"eligible": r["eligible"],
                                            "midpoint": r["midpoint"],
                                            "volume_id": r["volume_id"]}
            for r in manifest["slices"]}

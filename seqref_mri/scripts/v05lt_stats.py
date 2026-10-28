# SEQREF-MRI-V05LTSTATS v0.5 -- scripts.v05lt_stats
# LIFETIME: KEEP
# =============================================================================
# Purpose: the preregistered analysis of SEQREF-MRI-V05LT results, including
#   confidence intervals for reconstruction and NLL changes (194 §9 + AM2-C,
#   AM3, AM4-A; 195 §6, §9.2; I1.8 plan SEQREF-MRI-V05LTI18 v0.1, delivery
#   (a): core engine + A + B + C). The intervals describe uncertainty in the
#   MEASURED EXPERIMENTAL RESULT (variation across seeds and volumes), not
#   the model's predictive uncertainty (D-UNC etc. are out of scope).
#   * aggregation: per-slice rows -> volume mean -> equal-weight volumes;
#     NMSE_u = mean of per-slice ratios over eligible slices (registered
#     eligibility manifest); a volume with none -> None, outside the fixed
#     NMSE population (AM2-C); capped finite PSNR (120 dB) retained;
#     cap / zero-error counts with identities per source (AM4-A);
#   * paired crossed percentile bootstrap (194 §9.1): B = 10,000; per
#     resample 3 seeds with replacement and ONE volume list applied to every
#     sampled seed and to both sides; statistics dPSNR (mean difference) and
#     r = (N - N_ref) / |N_ref| (ratio of means); per-seed volume-only
#     bootstrap; SSIM / NMSE_u differences likewise;
#   * labels (194 §9.1-9.3): support, practical (+0.5 dB), density (+1 %),
#     outcomes 1-6; AM3 epoch-0 / ZF block descriptive (no labels; raw NLL
#     differences only trained vs epoch 0; ZF on PSNR, SSIM, NMSE_u only);
#     AM3-E4 "unavailable" for missing epoch-0 / ZF values only (AM3-G2:
#     NMSE ineligibility and PSNR caps are never "unavailable");
#   * decisions D1-D4 (196, 2026-10-23): D1 index arrays shared only by
#     comparisons with the same ordered seed and volume population; D2 NMSE
#     population fixed from the eligibility manifest, empty -> NMSE
#     unavailable (never zero); D3 campaign-level matched milestone needs
#     both arms at that milestone in all three seeds, else per-seed
#     descriptive results only; D4 linear-interpolation percentiles at
#     2.5 % / 97.5 % (formula below, recorded in the configuration);
#   * statistics stream: ONLY numpy.random.default_rng seeded 20261022,
#     consumed in the registered draw order; sha256 of every index array
#     recorded; global Python / NumPy RNG states must be unchanged (F04);
#   * analysis configuration (addition C): canonical JSON, sha256; analyse()
#     refuses any configuration whose hash differs from the approved one.
# Arithmetic (declared): every mean in this module is math.fsum(values) / n
#   (correctly rounded, independent of summation order); percentiles by the
#   formula in percentile_linear (exact rational position), not by
#   numpy.percentile. Bitwise reproducibility is established only in the
#   tested environments (fixture BSB); it is not guaranteed across NumPy
#   versions.
# Inputs: per-slice result rows (keys volume_id, file, slice_index and the
#   metric values), keyed by source (arm, seed, checkpoint, readout, scope).
#   Reading those rows from run / evaluator outputs is delivery (b). This
#   module never reads MRI data or the holdout.
# Production: main() refuses (STATS_PRODUCTION_DISABLED); the registered
#   analysis directory is refused without production=True, which only the
#   gated analysis stage may pass, after the approved configuration hash is
#   recorded in 196.
# CONVENTION: logger.error + V05LTError(code); no substituted values.
# Changelog (v0.5, 2026-10-24; b3 revision-1 review fixes):
#   * The holdout addendum freezes the expected mask-seed identity
#     (sha256 over the registered slices' canonical evaluation mask seeds,
#     computed from the manifest only) before any holdout contact.
# Changelog (v0.4, 2026-10-24; I1.8 b3, SEQREF-MRI-V05LTI18B3 v0.1):
#   * Holdout addendum (registered in 196 before coding): 21 comparisons
#     (between arms at selected; within each arm selected vs epoch 3; R0
#     and primary-bank R1 on PSNR, SSIM, NMSE_u; NLL ratio), population
#     "hold" appended AFTER the validation populations (validation arrays,
#     estimates, CIs and labels unchanged); NMSE_u per AM4-B (all slices,
#     no exclusion; a missing value is a validity failure).
#   * build_holdout_config / analyse(holdout_rows=...) ; label scope as
#     registered (PSNR support + practical, NLL density, SSIM/NMSE_u none).
# Changelog (v0.2, 2026-10-23; I1.8(a) review):
#   * Slice completeness: every source is bound to the expected slice
#     identities of its scope (from the eligibility-manifest rows; config
#     records their sha256). Missing trained slice -> STATS_SLICE_MISSING;
#     missing epoch-0 / ZF slice -> affected comparisons unavailable with
#     identities; unexpected slice / wrong volume -> refused.
#   * Readout comparisons (194 §8, §8.1; plot 3): R1-R0, R0+DC-R0,
#     R1+DC-R1 on identical midpoint slices per run and checkpoint.
#   * DC identity (194 §8): a comparison whose participating DC sources are
#     identical within tolerance on all their evaluated slices is reported
#     as an identity result (no labels, no outcome); decided per comparison
#     (v0.3 fix), campaign-wide summary reported separately. PA3 reporting: D3 results list seeds reached / absent and
#     why. Cross-NumPy bitwise guarantee withdrawn.
# Changelog (NEW in v0.1, 2026-10-23):
#   * Introduced under I1.8(a) (SEQREF-MRI-V05LTI18 v0.1): aggregation,
#     crossed and per-seed bootstrap, labels, outcomes, AM3 descriptive
#     block, D1-D4, statistics-stream record, analysis configuration hash.
# =============================================================================
from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
import random
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

from seqref_mri.scripts import v05lt_common as C

logger = logging.getLogger("seqref_mri.v05lt_stats")

__version__ = "0.5"
__abbr__ = "SEQREF-MRI-V05LTSTATS"

CONFIG_SCHEMA = "seqref-v05lt-analysis-config/1"
RESULT_SCHEMA = "seqref-v05lt-analysis/1"
STATS_SEED = 20261022                       # 194 §3.1 bootstrap stream
B_REGISTERED = 10_000                       # 194 §9.1
SEEDS = (2026, 2027, 2028)                  # 194 §4, order = matrix rows
ARMS = ("lam0", "lam2")                     # lam2 = reconstruction-aware
MILESTONES = (3, 10, 20, 30, 50, 75, 100)   # 194 §6.1 (= v05lt_rules)
LEVELS = (Fraction(25, 1000), Fraction(975, 1000))   # D4
PRACTICAL_DB = 0.5                          # 194 §9.2
DENSITY_BOUND = 0.01                        # 194 §9.2 (+1 %)
READOUT_SCOPES = (("r0", "all"), ("r1", "mid"), ("r0_dc", "mid"),
                  ("r1_dc", "mid"))
READOUT_PAIRS = (("r1", "r0"), ("r0_dc", "r0"), ("r1_dc", "r1"))   # 194 §8
DC_READOUTS = ("r0_dc", "r1_dc")
IDENTITY_TEXT = ("DC identical within tolerance on all evaluated slices of "
                 "the sources in this comparison: identity result, not an "
                 "independent treatment (194 §8)")
METRICS = ("psnr", "ssim", "nmse_u", "nll_per_dim")
HOLDOUT_POP = "hold"                        # population / scope id
HOLDOUT_READOUTS = ("r0", "r1")             # R1 = primary bank, 128 decodes
HOLDOUT_ROLES = ("selected", "epoch3")      # evaluation scope (196 b3)
HOLDOUT_NMSE_RULE = (
    "194 AM4-B: holdout NMSE_u over all registered slices, no threshold "
    "exclusion; exactly zero free-space energy is a validity error.")
HOLDOUT_SCOPE_TEXT = (
    "The 21 registered holdout comparisons use the selected and epoch-3 "
    "checkpoints only; final and epoch-0 checkpoints listed in the "
    "selection manifest are not evaluated on the holdout (196 b3).")
HOLDOUT_STATEMENT = (
    "The 199-volume holdout has already been used (V02, V02G, V03): these "
    "holdout results are NOT an independent confirmation (194 §10.1).")
RECON = ("psnr", "ssim", "nmse_u")
OUTCOME_TEXT = {
    1: "Reconstruction improves with density preserved (NLL deterioration "
       "bounded below 1%), under the tested conditions",
    2: "Reconstruction-density trade-off",
    3: "Reconstruction improves; density effect unclear",
    4: "No practically important change (precisely estimated)",
    5: "Uncertain",
    6: "Reconstruction deteriorates"}
CAP_STATEMENT = ("PSNR cannot distinguish errors below its floor (MSE <= "
                 "1e-12, 120 dB); comparisons including capped slices "
                 "estimate differences in the capped metric (194 AM4-A).")
DECISIONS = {
    "D1": "Resample index arrays are reused only across comparisons with "
          "the same ordered seed and volume populations; pairing preserved "
          "on both sides of every comparison.",
    "D2": "NMSE eligible-volume population fixed from the registered "
          "eligibility manifest (per scope); if no volume is eligible, NMSE "
          "is reported unavailable, never zero.",
    "D3": "A campaign-level matched-milestone comparison requires both arms "
          "to reach that milestone in all three seeds; otherwise the "
          "available paired results are reported descriptively (no labels).",
    "D4": "Percentiles: linear interpolation at 2.5% and 97.5%."}
DRAW_ORDER = (
    "One generator, numpy.random.default_rng seeded 20261022, for the whole "
    "analysis. Populations in configuration order; for each non-empty "
    "population of V volumes: crossed seed indices integers(0, 3, size=(B, "
    "3), dtype=int64), then crossed volume indices integers(0, V, size=(B, "
    "V), dtype=int64), then volume-only indices integers(0, V, size=(B, V), "
    "dtype=int64) for seeds 2026, 2027, 2028 in that order. Seed index k "
    "denotes SEEDS[k]; volume index j denotes the population's j-th volume. "
    "Array identity = sha256 of the C-order int64 bytes.")
PERCENTILE_METHOD = (
    "Sort the B finite resample statistics ascending (x_0..x_{B-1}); for "
    "level q (exact fraction 1/40 or 39/40) h = (B - 1) q exactly, lo = "
    "floor(h), t = h - lo; value = x_lo if t = 0, else x_lo + float(t) * "
    "(x_{lo+1} - x_lo) in float64 (linear interpolation, D4).")
ARITHMETIC = ("Every mean = math.fsum(values) / n; dPSNR (and SSIM, NMSE_u, "
              "raw NLL) statistic = mean over all (sampled seed, sampled "
              "volume) cells of (side - reference); r = (mean(side) - "
              "mean(ref)) / |mean(ref)| over the same cells.")


# ---------------------------------------------------------------------------
# Aggregation (194 §8.2, AM2-C, AM4-A)
# ---------------------------------------------------------------------------
def fmean(values) -> float:
    vals = list(values)
    C.require(len(vals) > 0, "STATS_EMPTY_MEAN", "mean of no values")
    return math.fsum(vals) / len(vals)


def _check_finite(v, code: str, where: str) -> float:
    C.require(isinstance(v, float) and math.isfinite(v), code,
              f"{where}: {v!r}")
    return v


def expected_slices(validation_volumes: list, slice_rows: list) -> dict:
    """Expected slice identities per scope from the registered eligibility-
    manifest rows (volume_id, file, slice_index, eligible, midpoint):
    {"all": [(vid, file, slice)], "mid": [...], "eligible": {(f, s): bool}},
    ordered by the validation order, then file, then slice."""
    pos = {v: i for i, v in enumerate(validation_volumes)}
    seen, mids, per = set(), {}, {v: 0 for v in validation_volumes}
    for r in slice_rows:
        ident = (r["file"], r["slice_index"])
        C.require(ident not in seen, "STATS_CONFIG_INVALID",
                  f"duplicate slice row {ident}")
        seen.add(ident)
        C.require(r["volume_id"] in pos, "STATS_CONFIG_INVALID",
                  f"slice row {ident} of unregistered volume")
        C.require(isinstance(r["eligible"], bool) and
                  isinstance(r["midpoint"], bool), "STATS_CONFIG_INVALID",
                  f"{ident}: eligible / midpoint must be bool")
        per[r["volume_id"]] += 1
        if r["midpoint"]:
            mids[r["volume_id"]] = mids.get(r["volume_id"], 0) + 1
    bad = [v for v in validation_volumes if per[v] == 0 or mids.get(v) != 1]
    C.require(not bad, "STATS_CONFIG_INVALID",
              f"volumes without rows or without exactly one midpoint: "
              f"{bad[:5]}")
    key = lambda t: (pos[t[0]], t[1], t[2])  # noqa: E731
    all_ = sorted(((r["volume_id"], r["file"], r["slice_index"])
                   for r in slice_rows), key=key)
    mid = sorted(((r["volume_id"], r["file"], r["slice_index"])
                  for r in slice_rows if r["midpoint"]), key=key)
    return {"all": all_, "mid": mid,
            "eligible": {(r["file"], r["slice_index"]): r["eligible"]
                         for r in slice_rows}}


def slices_sha256(expected: list, eligible: dict) -> str:
    """Identity of a scope's expected slices incl. their eligibility."""
    return C.sha256_bytes(C.canonical_json(
        [[v, f, s, eligible[(f, s)]] for v, f, s in expected]))


def volume_values(rows: list, metric: str, expected: list, *,
                  eligible: dict = None, reference: bool = False) -> tuple:
    """Per-slice rows -> {volume_id: mean | None} over the volumes of the
    ordered `expected` slice identities [(volume_id, file, slice)] of the
    scope. Every expected slice must have exactly one row and no other row
    may exist (STATS_SLICE_UNEXPECTED, STATS_SLICE_VOLUME_MISMATCH,
    STATS_DUPLICATE_SLICE). PSNR/SSIM/NLL: every slice; NMSE_u: eligible
    slices only (eligible from the registered manifest), a volume without
    one -> None. A missing row or value on a required slice is a validity
    failure for trained checkpoints (STATS_SLICE_MISSING /
    STATS_VALUE_MISSING) and, for epoch-0 / ZF references (reference=True),
    makes that volume None with the identity recorded (AM3-E4). Returns
    (values, missing, nmse_volumes)."""
    C.require(metric in METRICS, "STATS_METRIC_UNKNOWN", metric)
    nmse = metric == "nmse_u"
    C.require(not nmse or eligible is not None, "STATS_ELIGIBILITY_MISSING",
              "NMSE_u needs the registered eligibility index")
    exp = {(f, s): v for v, f, s in expected}
    volumes = list(dict.fromkeys(v for v, _, _ in expected))
    got = {}
    for r in rows:
        ident = (r["file"], r["slice_index"])
        C.require(ident not in got, "STATS_DUPLICATE_SLICE", f"{ident}")
        C.require(ident in exp, "STATS_SLICE_UNEXPECTED",
                  f"{ident} is not an expected slice of this scope")
        C.require(r["volume_id"] == exp[ident], "STATS_SLICE_VOLUME_MISMATCH",
                  f"{ident}: volume {r['volume_id']} != {exp[ident]}")
        got[ident] = r
    acc = {v: [] for v in volumes}
    bad_vol, missing, elig_vol = set(), [], set()
    for vid, f, s in expected:
        ident = (f, s)
        if nmse:
            C.require(ident in eligible, "STATS_ELIGIBILITY_MISSING",
                      f"{ident} not in the eligibility index")
            if not eligible[ident]:
                r = got.get(ident)
                C.require(r is None or r.get(metric) is None,
                          "STATS_NMSE_ELIGIBILITY_MISMATCH",
                          f"{ident}: ineligible slice carries NMSE_u")
                continue
            elig_vol.add(vid)
        r = got.get(ident)
        if r is None:
            C.require(reference, "STATS_SLICE_MISSING",
                      f"trained checkpoint row missing for slice {ident}")
            missing.append(f"{f}|{s} (row absent)")
            bad_vol.add(vid)
            continue
        v = r.get(metric)
        if v is None:
            C.require(reference, "STATS_VALUE_MISSING",
                      f"{metric} missing for trained checkpoint slice {ident}")
            missing.append(f"{f}|{s}")
            bad_vol.add(vid)
            continue
        acc[vid].append(_check_finite(v, "STATS_NON_FINITE",
                                      f"{metric} {ident}"))
    out = {v: (None if (v in bad_vol or not acc[v]) else fmean(acc[v]))
           for v in volumes}
    return out, missing, [v for v in volumes if v in elig_vol]


def cap_counts(rows: list) -> dict:
    """AM4-A4 counts and identities of psnr_capped / zero_error slices."""
    out = {}
    for flag in ("psnr_capped", "zero_error"):
        ids = sorted(f"{r['file']}|{r['slice_index']}" for r in rows
                     if r.get(flag) is True)
        out[flag] = {"count": len(ids), "identities": ids}
    return out


# ---------------------------------------------------------------------------
# Registered comparison list (194 §9.0, §9.1, §7, AM3-F, AM3-G3)
# ---------------------------------------------------------------------------
def _side(arm, ckpt, readout):
    return {"arm": arm, "ckpt": ckpt, "readout": readout}


def registered_comparisons(milestones=MILESTONES) -> tuple:
    """-> (comparisons, outcome_groups); deterministic, recorded in the
    configuration verbatim."""
    comps, groups = [], []

    def add(cid, family, metric, stat, scope, side, ref, labels=None,
            milestone=None):
        pop = (f"nmse_{scope}" if metric == "nmse_u" else "val")
        comps.append({"id": cid, "family": family, "metric": metric,
                      "stat": stat, "scope": scope, "side": side, "ref": ref,
                      "labels": labels, "milestone": milestone,
                      "population": pop})

    def recon_block(prefix, family, side_arm, side_ckpt, ref_arm, ref_ckpt,
                    readouts, labelled, milestone=None, ref_readout=None):
        ids = {}
        for ro, sc in readouts:
            for m in RECON:
                cid = f"{prefix}|{ro}|{m}"
                lab = "psnr" if (labelled and m == "psnr") else None
                add(cid, family, m, "delta", sc, _side(side_arm, side_ckpt, ro),
                    _side(ref_arm, ref_ckpt, ref_readout or ro), lab,
                    milestone)
                ids[(ro, m)] = cid
        return ids

    # Between arms (Q2) at the primary selected checkpoint: labelled.
    p = "between|selected"
    ids = recon_block(p, "between", "lam2", "selected", "lam0", "selected",
                      READOUT_SCOPES, True)
    add(f"{p}|nll|ratio", "between", "nll_per_dim", "ratio", "all",
        _side("lam2", "selected", "r0"), _side("lam0", "selected", "r0"),
        "density")
    for ro, _ in READOUT_SCOPES:
        groups.append({"id": f"{p}|{ro}", "family": "between",
                       "psnr": ids[(ro, "psnr")], "nll": f"{p}|nll|ratio",
                       "milestone": None})
    # Between arms at each matched-budget milestone (D3): R0 all-slice.
    for m in milestones:
        p = f"between|epoch{m}"
        ids = recon_block(p, "between", "lam2", f"epoch{m}", "lam0",
                          f"epoch{m}", (("r0", "all"),), True, milestone=m)
        add(f"{p}|nll|ratio", "between", "nll_per_dim", "ratio", "all",
            _side("lam2", f"epoch{m}", "r0"), _side("lam0", f"epoch{m}", "r0"),
            "density", milestone=m)
        groups.append({"id": f"{p}|r0", "family": "between",
                       "psnr": ids[("r0", "psnr")], "nll": f"{p}|nll|ratio",
                       "milestone": m})
    # Secondary lowest-NLL checkpoint (§7): reported alongside, no labels.
    p = "secondary|best_nll"
    recon_block(p, "secondary", "lam2", "best_nll", "lam0", "best_nll",
                (("r0", "all"),), False)
    add(f"{p}|nll|ratio", "secondary", "nll_per_dim", "ratio", "all",
        _side("lam2", "best_nll", "r0"), _side("lam0", "best_nll", "r0"))
    # Within arm (Q1): selected vs epoch 3 of the same run: labelled.
    for arm in ARMS:
        p = f"within|{arm}"
        ids = recon_block(p, "within", arm, "selected", arm, "epoch3",
                          READOUT_SCOPES, True)
        add(f"{p}|nll|ratio", "within", "nll_per_dim", "ratio", "all",
            _side(arm, "selected", "r0"), _side(arm, "epoch3", "r0"),
            "density")
        for ro, _ in READOUT_SCOPES:
            groups.append({"id": f"{p}|{ro}", "family": "within",
                           "psnr": ids[(ro, "psnr")],
                           "nll": f"{p}|nll|ratio", "milestone": None})
    # AM3 descriptive: trained vs epoch 0 (raw NLL differences).
    for arm in ARMS:
        for ck in ("epoch3", "selected", "final"):
            p = f"am3_epoch0|{arm}|{ck}"
            recon_block(p, "am3_epoch0", arm, ck, "shared", "epoch0",
                        READOUT_SCOPES, False)
            add(f"{p}|nll|delta", "am3_epoch0", "nll_per_dim", "delta",
                "all", _side(arm, ck, "r0"), _side("shared", "epoch0", "r0"))
    # AM3 descriptive: each readout vs ZF (PSNR, SSIM, NMSE_u only).
    points = [("shared", "epoch0")] + [(a, c) for a in ARMS
                                       for c in ("epoch3", "selected",
                                                 "final")]
    for arm, ck in points:
        recon_block(f"am3_zf|{arm}|{ck}", "am3_zf", arm, ck, "zf", "zf",
                    READOUT_SCOPES, False, ref_readout="zf")
    # Readout comparisons (194 §8, §8.1; plot 3): same run and checkpoint,
    # identical midpoint slices. Support label (194 §9.1 "R1") on dPSNR of
    # R1 - R0 at trained checkpoints only; no practical label or outcome;
    # epoch 0 descriptive (AM3); DC pairs descriptive / identity results.
    for arm, ck in points:
        for ro, base in READOUT_PAIRS:
            for m in RECON:
                lab = ("support" if (m == "psnr" and arm != "shared"
                                     and (ro, base) == ("r1", "r0")) else None)
                add(f"readout|{arm}|{ck}|{ro}-{base}|{m}", "readout", m,
                    "delta", "mid", _side(arm, ck, ro), _side(arm, ck, base),
                    lab)
    return comps, groups


def holdout_comparisons() -> tuple:
    """196 b3 registration: 21 comparisons and 6 outcome groups on the
    holdout population, in this fixed order. Labels: PSNR -> support +
    practical; NLL ratio -> density; SSIM / NMSE_u -> none."""
    comps, groups = [], []

    def add(cid, family, metric, stat, side, ref, labels):
        comps.append({"id": cid, "family": family, "metric": metric,
                      "stat": stat, "scope": HOLDOUT_POP, "side": side,
                      "ref": ref, "labels": labels, "milestone": None,
                      "population": HOLDOUT_POP})
    contrasts = [("holdout|between|selected", "holdout_between", "lam2",
                  "selected", "lam0", "selected")]
    contrasts += [(f"holdout|within|{a}", "holdout_within", a, "selected",
                   a, "epoch3") for a in ARMS]
    for p, fam, sa, sc, ra, rc in contrasts:
        for ro in HOLDOUT_READOUTS:
            for m in RECON:
                add(f"{p}|{ro}|{m}", fam, m, "delta", _side(sa, sc, ro),
                    _side(ra, rc, ro), "psnr" if m == "psnr" else None)
        add(f"{p}|nll|ratio", fam, "nll_per_dim", "ratio",
            _side(sa, sc, "r0"), _side(ra, rc, "r0"), "density")
        for ro in HOLDOUT_READOUTS:
            groups.append({"id": f"{p}|{ro}", "family": fam,
                           "psnr": f"{p}|{ro}|psnr", "nll": f"{p}|nll|ratio",
                           "milestone": None})
    return comps, groups


def holdout_expected(holdout_rows: list) -> list:
    """Registered holdout slices [(volume_id, file, slice_index)] in the
    holdout-manifest order: one slice per volume, no duplicates."""
    out, vols, ids = [], set(), set()
    for r in holdout_rows:
        C.require(isinstance(r, dict) and {"volume_id", "file",
                                           "slice_index"} <= set(r),
                  "STATS_CONFIG_INVALID", f"holdout row {r!r}"[:120])
        t = (r["volume_id"], r["file"], r["slice_index"])
        C.require(t[0] not in vols and (t[1], t[2]) not in ids,
                  "STATS_CONFIG_INVALID", f"holdout duplicate {t}")
        vols.add(t[0])
        ids.add((t[1], t[2]))
        out.append(t)
    C.require(bool(out), "STATS_CONFIG_INVALID", "empty holdout population")
    return out


def holdout_slices_sha256(expected: list) -> str:
    return C.sha256_bytes(C.canonical_json([list(t) for t in expected]))


def build_holdout_config(val_cfg: dict, holdout_rows: list, *,
                         manifest_content_sha256: str,
                         mask_seeds_sha256: str) -> dict:
    """Holdout addendum: the validation configuration unchanged plus the
    holdout population (appended after every validation population), its
    21 registered comparisons and 6 outcome groups. Frozen and approved
    (sha256 recorded in 196) before any holdout contact."""
    import copy
    _validate_config(val_cfg)
    C.require("holdout" not in val_cfg, "STATS_CONFIG_INVALID",
              "configuration already carries a holdout addendum")
    ex = holdout_expected(holdout_rows)
    cfg = copy.deepcopy(val_cfg)
    comps, groups = holdout_comparisons()
    cfg["populations"].append({"id": HOLDOUT_POP,
                               "volumes": [v for v, _, _ in ex]})
    cfg["comparisons"] = cfg["comparisons"] + comps
    cfg["outcome_groups"] = cfg["outcome_groups"] + groups
    C.require(isinstance(mask_seeds_sha256, str) and
              re.fullmatch(r"[0-9a-f]{64}", mask_seeds_sha256) is not None,
              "STATS_CONFIG_INVALID", "mask-seed identity must be a sha256")
    cfg["holdout"] = {"manifest_content_sha256": manifest_content_sha256,
                      "mask_seeds_sha256": mask_seeds_sha256,
                      "slices": {"n": len(ex),
                                 "sha256": holdout_slices_sha256(ex)},
                      "readouts": list(HOLDOUT_READOUTS),
                      "roles": list(HOLDOUT_ROLES),
                      "nmse_rule": HOLDOUT_NMSE_RULE,
                      "evaluation_scope": HOLDOUT_SCOPE_TEXT,
                      "statement": HOLDOUT_STATEMENT}
    return cfg


def validation_part(cfg: dict) -> dict:
    """The validation configuration inside a holdout-addendum
    configuration (addendum population, comparisons, groups and section
    removed); equals the approved validation configuration byte for
    byte."""
    import copy
    C.require("holdout" in cfg, "STATS_CONFIG_INVALID", "no holdout addendum")
    hc, hg = holdout_comparisons()
    v = copy.deepcopy(cfg)
    del v["holdout"]
    C.require(v["populations"][-1]["id"] == HOLDOUT_POP and
              v["comparisons"][-len(hc):] == hc and
              v["outcome_groups"][-len(hg):] == hg,
              "STATS_CONFIG_INVALID", "holdout addendum not at the end")
    v["populations"] = v["populations"][:-1]
    v["comparisons"] = v["comparisons"][:-len(hc)]
    v["outcome_groups"] = v["outcome_groups"][:-len(hg)]
    return v


# ---------------------------------------------------------------------------
# Analysis configuration (addition C)
# ---------------------------------------------------------------------------
def build_config(validation_volumes: list, slice_rows: list, *,
                 eligibility_sha256: str, split_sha256: str,
                 B: int = B_REGISTERED, fixture_override: bool = False) -> dict:
    """Validation volumes in the registered validation order; slice_rows =
    the registered eligibility-manifest rows (volume_id, file, slice_index,
    eligible, midpoint). Expected slice identities per scope and the NMSE
    populations (D2) are derived from them and bound by sha256."""
    C.require(isinstance(B, int) and B >= 2, "STATS_CONFIG_INVALID", f"B={B}")
    C.require(B == B_REGISTERED or fixture_override, "STATS_CONFIG_INVALID",
              f"B={B} differs from the registered {B_REGISTERED}")
    vols = list(validation_volumes)
    C.require(len(set(vols)) == len(vols) and vols, "STATS_CONFIG_INVALID",
              "validation volumes empty or duplicated")
    ex = expected_slices(vols, slice_rows)
    nmse = {sc: [v for v in vols if any(ex["eligible"][(f, s)]
                                        for vv, f, s in ex[sc] if vv == v)]
            for sc in ("all", "mid")}
    comps, groups = registered_comparisons()
    return {
        "schema": CONFIG_SCHEMA, "abbr": __abbr__, "stats_version": __version__,
        "fixture_override": bool(fixture_override),
        "bootstrap": {"B": B, "generator": "numpy.random.default_rng "
                      "(PCG64)", "seed": STATS_SEED,
                      "draw_order": DRAW_ORDER, "index_dtype": "int64"},
        "percentile": {"levels": [str(q) for q in LEVELS],
                       "method": PERCENTILE_METHOD},
        "arithmetic": ARITHMETIC,
        "seeds": list(SEEDS), "arms": list(ARMS),
        "milestones": list(MILESTONES),
        "thresholds": {"practical_db": PRACTICAL_DB,
                       "density_bound": DENSITY_BOUND},
        "decisions": DECISIONS,
        "inputs": {"eligibility_manifest_sha256": eligibility_sha256,
                   "split_manifest_sha256": split_sha256},
        "slices": {sc: {"n": len(ex[sc]), "sha256": slices_sha256(ex[sc], ex["eligible"])}
                   for sc in ("all", "mid")},
        "populations": [{"id": "val", "volumes": vols},
                        {"id": "nmse_all", "volumes": nmse["all"]},
                        {"id": "nmse_mid", "volumes": nmse["mid"]}],
        "comparisons": comps, "outcome_groups": groups,
        "dc_identity_rule": IDENTITY_TEXT,
        "out_of_scope": ["D-UNC", "D-SPEC", "D-LAT", "D-JAC",
                         "new subgroup tests (D-HET)"],
    }


def config_bytes(cfg: dict) -> bytes:
    return C.canonical_json(cfg)


def config_sha256(cfg: dict) -> str:
    return C.sha256_bytes(config_bytes(cfg))


def _registered_dir_guard(path: Path, production: bool) -> Path:
    p = Path(os.path.abspath(path))            # final name not followed
    reg = Path(os.path.abspath(C.Paths().v05lt))
    if p == reg or reg in p.parents:
        C.require(production, "STATS_PRODUCTION_DISABLED",
                  f"{p} is under the registered results tree")
    return p


def write_json_once(path: Path, doc: dict, production: bool = False) -> str:
    """Canonical JSON + .sha256 sidecar, write-once (no overwrite)."""
    p = _registered_dir_guard(path, production)
    data = C.canonical_json(doc)
    C.require(not os.path.lexists(p) and not os.path.lexists(
        str(p) + ".sha256"), "ARTIFACT_EXISTS", str(p))
    sha = C.write_durable_new(p, data)
    C.write_durable_new(Path(str(p) + ".sha256"),
                        f"{sha}  {p.name}\n".encode())
    return sha


def _validate_config(cfg: dict) -> dict:
    C.require(cfg.get("schema") == CONFIG_SCHEMA, "STATS_CONFIG_INVALID",
              "schema")
    pops = {p["id"]: p["volumes"] for p in cfg["populations"]}
    hold = "holdout" in cfg
    C.require(list(pops) == ["val", "nmse_all", "nmse_mid"] +
              ([HOLDOUT_POP] if hold else []),
              "STATS_CONFIG_INVALID", "population list")
    comps, groups = registered_comparisons(tuple(cfg["milestones"]))
    if hold:                       # v0.4: registered holdout addendum only
        hc, hg = holdout_comparisons()
        comps, groups = comps + hc, groups + hg
    C.require(cfg["comparisons"] == comps and cfg["outcome_groups"] == groups,
              "STATS_CONFIG_INVALID", "comparison list is not the registered "
              "list")
    C.require(cfg["seeds"] == list(SEEDS) and cfg["bootstrap"]["seed"] ==
              STATS_SEED, "STATS_CONFIG_INVALID", "seeds / stream")
    return pops


# ---------------------------------------------------------------------------
# Statistics stream (194 §3.1; F04) and percentiles (D4)
# ---------------------------------------------------------------------------
def _arr_sha(a: np.ndarray) -> str:
    return C.sha256_bytes(np.ascontiguousarray(a, dtype=np.int64).tobytes())


def draw_indices(cfg: dict) -> tuple:
    """All resample index arrays in the registered draw order (DRAW_ORDER).
    Identical ordered populations share one set of arrays (D1)."""
    B = cfg["bootstrap"]["B"]
    rng = np.random.default_rng(STATS_SEED)
    draws, record, seen = {}, [], {}
    for pop in cfg["populations"]:
        key = tuple(pop["volumes"])
        if key in seen:                         # D1: same ordered population
            draws[pop["id"]] = draws[seen[key]]
            record.append({"population": pop["id"], "shares": seen[key]})
            continue
        seen[key] = pop["id"]
        V = len(key)
        if V == 0:
            draws[pop["id"]] = None
            record.append({"population": pop["id"], "volumes": 0,
                           "draws": "none (empty population)"})
            continue
        s = rng.integers(0, len(SEEDS), size=(B, len(SEEDS)), dtype=np.int64)
        v = rng.integers(0, V, size=(B, V), dtype=np.int64)
        vo = {}
        for sd in SEEDS:
            vo[sd] = rng.integers(0, V, size=(B, V), dtype=np.int64)
        draws[pop["id"]] = {"seed": s, "vol": v, "vo": vo}
        record.append({"population": pop["id"], "volumes": V, "B": B,
                       "crossed_seed_sha256": _arr_sha(s),
                       "crossed_volume_sha256": _arr_sha(v),
                       "volume_only_sha256": {str(sd): _arr_sha(vo[sd])
                                              for sd in SEEDS}})
    return draws, record


def percentile_linear(sorted_vals: list, level: Fraction) -> float:
    n = len(sorted_vals)
    C.require(n >= 2, "STATS_PERCENTILE_INVALID", f"n={n}")
    h = Fraction(n - 1) * level
    lo = h.numerator // h.denominator
    t = h - lo
    if t == 0:
        return float(sorted_vals[lo])
    a, b = sorted_vals[lo], sorted_vals[lo + 1]
    return float(a + float(t) * (b - a))


def ci(stats: list) -> list:
    for x in stats:
        _check_finite(x, "STATS_NON_FINITE", "resample statistic")
    srt = sorted(stats)
    return [percentile_linear(srt, q) for q in LEVELS]


# ---------------------------------------------------------------------------
# Bootstrap statistics (194 §9.1)
# ---------------------------------------------------------------------------
def _stat(A: np.ndarray, R: np.ndarray, stat: str) -> float:
    if stat == "delta":
        return fmean((A - R).ravel().tolist())
    a, r = fmean(A.ravel().tolist()), fmean(R.ravel().tolist())
    C.require(r != 0.0, "STATS_R_ZERO_REFERENCE",
              "r undefined: reference mean NLL is exactly 0")
    return (a - r) / abs(r)


def bootstrap_comparison(A: np.ndarray, R: np.ndarray, stat: str,
                         draws: dict, seeds_available=None,
                         keep: bool = False) -> dict:
    """A, R: (3, V) float64, rows = SEEDS, columns = population order.
    Crossed statistic + 95 % CI and per-seed volume-only results. With
    seeds_available (D3 descriptive), only per-seed results for those.
    keep=True (fixtures only) also returns the crossed resample values."""
    C.require(stat in ("delta", "ratio"), "STATS_STAT_UNKNOWN", stat)
    C.require(A.shape == R.shape and A.shape[0] == len(SEEDS),
              "STATS_SHAPE", f"{A.shape} vs {R.shape}")
    per_seed = []
    for k, sd in enumerate(SEEDS):
        if seeds_available is not None and sd not in seeds_available:
            continue
        vo = draws["vo"][sd]
        res = [_stat(A[k, vo[b]], R[k, vo[b]], stat)
               for b in range(vo.shape[0])]
        per_seed.append({"seed": sd, "estimate": _stat(A[k], R[k], stat),
                         "ci": ci(res)})
    out = {"per_seed": per_seed}
    if seeds_available is not None:
        return out
    s, v = draws["seed"], draws["vol"]
    res = []
    for b in range(s.shape[0]):
        ix = np.ix_(s[b], v[b])
        res.append(_stat(A[ix], R[ix], stat))
    out.update({"estimate": _stat(A, R, stat), "ci": ci(res)})
    if keep:
        out["resamples"] = res
    return out


# ---------------------------------------------------------------------------
# Labels (194 §9.1-9.3)
# ---------------------------------------------------------------------------
def support_label(lo: float, hi: float) -> str:
    if lo > 0:
        return "supported improvement"
    if hi < 0:
        return "supported deterioration"
    return "not established"


def practical_label(lo: float, hi: float) -> str:
    if lo > PRACTICAL_DB:
        return "above reference"
    if hi < PRACTICAL_DB:
        return "below reference"
    return "straddles reference"


def density_label(lo: float, hi: float) -> str:
    if hi < DENSITY_BOUND:
        return "preserved"
    if lo > DENSITY_BOUND:
        return "worse"
    return "unclear"


def outcome(support: str, lo: float, hi: float, density: str) -> int:
    if support == "supported improvement":
        return {"preserved": 1, "worse": 2, "unclear": 3}[density]
    if support == "supported deterioration":
        return 6
    if -PRACTICAL_DB < lo and hi < PRACTICAL_DB:
        return 4
    return 5


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def source_key(arm, seed, ckpt, readout, scope) -> str:
    return f"{arm}|{'-' if seed is None else seed}|{ckpt}|{readout}|{scope}"


def _global_rng_state() -> bytes:
    return pickle.dumps((random.getstate(), np.random.get_state()))


def _matrix(sources, side, seeds, comp, ex, pop, cache):
    """(len(SEEDS), V) values over the comparison population (rows of seeds
    outside `seeds` are None) + missing identities. Each source is bound to
    the expected slices of the comparison's scope; for NMSE_u its eligible
    volumes must equal the configured population (D2)."""
    ref = side["arm"] in ("shared", "zf")
    metric, scope = comp["metric"], comp["scope"]
    rows_out, missing = [], []
    for sd in SEEDS:
        if sd not in seeds:
            rows_out.append(None)
            continue
        k = source_key(side["arm"], None if side["arm"] == "zf" else sd,
                       side["ckpt"], side["readout"], scope)
        C.require(k in sources, "STATS_SOURCE_MISSING", k)
        ck = (k, metric)
        if ck not in cache:
            cache[ck] = volume_values(
                sources[k], metric, ex[scope],
                eligible=ex["eligible"] if metric == "nmse_u" else None,
                reference=ref)
        vals, miss, elig_vols = cache[ck]
        if metric == "nmse_u":
            C.require(elig_vols == list(pop),
                      "STATS_NMSE_POPULATION_MISMATCH",
                      f"{k}: eligible volumes differ from the configured "
                      f"population {comp['population']}")
        rows_out.append([vals[v] for v in pop])
        missing += [f"{k}: {m}" for m in miss]
    return rows_out, missing


def dc_identity(sources: dict) -> dict:
    """Per DC source key: identical within tolerance on all of that source's
    evaluated slices (rows carry the evaluator's per-slice boolean
    `dc_identical`, 194 §8). Returns {"per_source": {key: {...}},
    "summary": {readout: {...}}}; the campaign-wide summary is reported
    only -- comparison decisions use their own participating sources."""
    per = {}
    for k in sorted(sources):
        if k.split("|")[3] not in DC_READOUTS:
            continue
        non = []
        for r in sources[k]:
            f = r.get("dc_identical")
            C.require(isinstance(f, bool), "STATS_DC_IDENTITY_MISSING",
                      f"{k}: {r['file']}|{r['slice_index']}")
            if not f:
                non.append(f"{r['file']}|{r['slice_index']}")
        per[k] = {"identical_all": not non, "non_identical": non}
    summary = {}
    for ro in DC_READOUTS:
        keys = [k for k in per if k.split("|")[3] == ro]
        C.require(bool(keys), "STATS_SOURCE_MISSING", f"no {ro} source")
        non = [f"{k}: {x}" for k in keys for x in per[k]["non_identical"]]
        summary[ro] = {"identical_all": not non,
                       "non_identical_count": len(non),
                       "non_identical": non}
    return {"per_source": per, "summary": summary}


def _comp_dc_identity(comp, seeds, dcid) -> bool:
    """Identity decision for ONE comparison: every DC source participating
    in it (its DC sides, the seeds it uses) identical on all its slices."""
    keys = []
    for side in (comp["side"], comp["ref"]):
        if side["readout"] in DC_READOUTS:
            for sd in sorted(seeds):
                keys.append(source_key(side["arm"], sd, side["ckpt"],
                                       side["readout"], comp["scope"]))
    for k in keys:
        C.require(k in dcid["per_source"], "STATS_SOURCE_MISSING", k)
    return bool(keys) and all(dcid["per_source"][k]["identical_all"]
                              for k in keys)


def _d3(m, run_info) -> tuple:
    """PA3: seeds where both arms reached milestone m, and for the others
    which arms lacked it and how their runs ended."""
    reached, absent = [], []
    for sd in SEEDS:
        lack = [a for a in ARMS if m not in run_info[(a, sd)]["milestones"]]
        if not lack:
            reached.append(sd)
            continue
        absent.append({"seed": sd, "arms_without_milestone": lack,
                       "runs": {a: {"last_epoch": run_info[(a, sd)][
                           "last_epoch"], "status": run_info[(a, sd)][
                           "status"]} for a in lack}})
    return reached, absent


def analyse(cfg: dict, approved_sha256: str, sources: dict,
            slice_rows: list, run_info: dict, production: bool = False,
            holdout_rows: list = None) -> dict:
    """sources: {source_key: per-slice rows}; slice_rows: the registered
    eligibility-manifest rows (bound to the configuration by sha256);
    run_info: {(arm, seed): {"milestones": [...], "last_epoch": int,
    "status": str}} from the run ledgers. Returns the analysis document."""
    got = config_sha256(cfg)
    C.require(got == approved_sha256, "STATS_CONFIG_UNAPPROVED",
              f"configuration sha256 {got} != approved {approved_sha256}")
    C.require(not (production and cfg["fixture_override"]),
              "STATS_FIXTURE_CONFIG", "fixture configuration in production")
    pops = _validate_config(cfg)
    ex = expected_slices(pops["val"], slice_rows)
    C.require(all(slices_sha256(ex[sc], ex["eligible"]) ==
                  cfg["slices"][sc]["sha256"]
                  for sc in ("all", "mid")), "STATS_SLICES_MISMATCH",
              "slice rows differ from those bound in the configuration")
    C.require(("holdout" in cfg) == (holdout_rows is not None),
              "STATS_HOLDOUT_MISMATCH", "holdout rows given iff the "
              "configuration carries the holdout addendum")
    if holdout_rows is not None:                # v0.4 (AM4-B: all slices)
        hx = holdout_expected(holdout_rows)
        C.require(holdout_slices_sha256(hx) == cfg["holdout"]["slices"][
            "sha256"] and [v for v, _, _ in hx] == pops[HOLDOUT_POP],
            "STATS_SLICES_MISMATCH", "holdout slices differ from those "
            "bound in the configuration")
        C.require(not ({(f, s) for _, f, s in hx} & set(ex["eligible"])),
                  "STATS_CONFIG_INVALID", "holdout slice inside the "
                  "validation population")
        elig = dict(ex["eligible"])
        elig.update({(f, s): True for _, f, s in hx})   # AM4-B: all slices
        ex = dict(ex, eligible=elig)
        ex[HOLDOUT_POP] = hx
    for a in ARMS:
        for sd in SEEDS:
            ri = run_info.get((a, sd))
            C.require(isinstance(ri, dict) and {"milestones", "last_epoch",
                                                 "status"} <= set(ri),
                      "STATS_RUN_INFO_MISSING", f"{a} seed {sd}")
    g0 = _global_rng_state()
    draws, stream = draw_indices(cfg)
    dcid = dc_identity(sources)
    results, cache = {}, {}
    for comp in cfg["comparisons"]:
        pop = pops[comp["population"]]
        res = {"id": comp["id"], "family": comp["family"],
               "metric": comp["metric"], "stat": comp["stat"],
               "scope": comp["scope"], "population": comp["population"],
               "n_volumes": len(pop), "milestone": comp["milestone"]}
        m = comp["milestone"]
        seeds = set(SEEDS)
        if m is not None:
            reached, absent = _d3(m, run_info)
            seeds = set(reached)
            if absent:
                res.update({"seeds_reached": reached, "seeds_absent": absent})
        if {comp["side"]["readout"], comp["ref"]["readout"]} & set(
                DC_READOUTS):
            res["dc_identity"] = _comp_dc_identity(comp, seeds, dcid)
        if comp["metric"] == "nmse_u" and not pop:
            res.update({"status": "unavailable_no_eligible_volume",
                        "reason": "D2: no NMSE-eligible volume in scope"})
            results[comp["id"]] = res
            continue
        if not seeds:
            res.update({"status": "descriptive_d3", "per_seed": [],
                        "reason": f"D3: milestone {m} reached by both arms "
                                  "in no seed"})
            results[comp["id"]] = res
            continue
        A, ma = _matrix(sources, comp["side"], seeds, comp, ex, pop, cache)
        R, mr = _matrix(sources, comp["ref"], seeds, comp, ex, pop, cache)
        if ma or mr:                            # AM3-E4 (references only)
            res.update({"status": "unavailable",
                        "reason": "AM3-E4: epoch-0 / ZF value or slice "
                                  "missing; the comparison is not "
                                  "re-bootstrapped on the remaining volumes",
                        "identities": sorted(set(ma + mr))})
            results[comp["id"]] = res
            continue
        nan = float("nan")
        Af = np.array([[nan] * len(pop) if row is None else row for row in A],
                      dtype=np.float64)
        Rf = np.array([[nan] * len(pop) if row is None else row for row in R],
                      dtype=np.float64)
        used = [k for k, sd in enumerate(SEEDS) if sd in seeds]
        C.require(bool(np.isfinite(Af[used]).all() and
                       np.isfinite(Rf[used]).all()), "STATS_VALUE_MISSING",
                  f"{comp['id']}: population volume without a value")
        if seeds != set(SEEDS):
            bs = bootstrap_comparison(Af, Rf, comp["stat"],
                                      draws[comp["population"]], seeds)
            res.update({"status": "descriptive_d3", **bs,
                        "reason": f"D3 / PA3: milestone {m} not reached by "
                                  "both arms in all three seeds; available "
                                  "paired seeds only, no crossed CI, no "
                                  "labels, no outcome"})
            results[comp["id"]] = res
            continue
        bs = bootstrap_comparison(Af, Rf, comp["stat"],
                                  draws[comp["population"]])
        res.update({"status": "ok", **bs})
        lo, hi = bs["ci"]
        if res.get("dc_identity"):
            if comp["labels"]:
                res["identity_result"] = IDENTITY_TEXT
        elif comp["labels"] in ("psnr", "support"):
            res["support"] = support_label(lo, hi)
            if comp["labels"] == "psnr":
                res["practical"] = practical_label(lo, hi)
        elif comp["labels"] == "density":
            res["density"] = density_label(lo, hi)
        results[comp["id"]] = res
    outcomes = []
    for g in cfg["outcome_groups"]:
        p, n = results[g["psnr"]], results[g["nll"]]
        base = {"id": g["id"], "family": g["family"],
                "milestone": g["milestone"]}
        if p["status"] != "ok" or n["status"] != "ok":
            outcomes.append({**base, "outcome": None,
                             "reason": f"not labelled (psnr {p['status']}, "
                                       f"nll {n['status']})"})
            continue
        if p.get("dc_identity"):
            outcomes.append({**base, "outcome": None,
                             "reason": IDENTITY_TEXT})
            continue
        lo, hi = p["ci"]
        o = outcome(p["support"], lo, hi, n["density"])
        outcomes.append({**base, "outcome": o, "text": OUTCOME_TEXT[o],
                         "support": p["support"],
                         "practical": p["practical"],
                         "density": n["density"]})
    caps = {k: cap_counts(sources[k]) for k in sorted(sources)}
    C.require(_global_rng_state() == g0, "STATS_GLOBAL_RNG_CONSUMED",
              "global Python / NumPy RNG state changed during the analysis")
    return {"schema": RESULT_SCHEMA, "abbr": __abbr__,
            "stats_version": __version__, "config_sha256": got,
            "stream": stream, "comparisons": list(results.values()),
            "outcomes": outcomes, "caps": caps,
            "dc_identity": dcid["summary"],
            "dc_identity_per_source": dcid["per_source"],
            "cap_statement": CAP_STATEMENT,
            "notes": ["Intervals describe uncertainty in the measured "
                      "experimental result (seeds and volumes), not model "
                      "predictive uncertainty.",
                      "With three seeds the seed component is coarse.",
                      "AM3 comparisons are descriptive: no labels."] +
                     ([HOLDOUT_STATEMENT, HOLDOUT_NMSE_RULE,
                       HOLDOUT_SCOPE_TEXT] if holdout_rows is not None
                      else [])}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=f"{__abbr__} v{__version__}")
    ap.parse_args(argv)
    logger.error("[failed_validity] STATS_PRODUCTION_DISABLED: the V05LT "
                 "analysis runs only at the gated analysis stage with the "
                 "approved configuration hash (195 §8.3, 196 I1.8)")
    return 2


if __name__ == "__main__":
    sys.exit(main())

# =============================================================================
# SEQREF-DCP v0.3 -- scripts._diag.dcprobe
# LIFETIME: DIAGNOSTIC
# Purpose: DESCRIPTIVE (no gate) DC-projection probe over the EXISTING
#   NICE ep29 / NICEXT ep57 checkpoints, separating two failure
#   components of the whole-image outputs:
#     (1) damage caused by altering MEASURED k-space;
#     (2) error carried by the model's estimates on UNMEASURED k-space.
#   It TESTS (not assumes) how much image-quality recovery exact DC
#   restoration of acquired coefficients provides.
# Projection (defined explicitly; computed ONLY through the verified
#   operator pair so no FFT convention can silently mismatch):
#     x_DC = x_hat + A^H(y - A x_hat)
#          = F^H[ M y + (1-M) F x_hat ]          (unitary F, exact DC)
#   After projection only the model's unmeasured-frequency contribution
#   remains effective; acquired coefficients are restored from y.
# Energy split (unitary Parseval): for k_err = F(x_hat_un) - F(x_true_un),
#     measured   E_m = ||A x_hat - A x_true||^2      (= ||M k_err||^2)
#     unmeasured E_u = ||x_hat - x_true||^2 - E_m    (>= 0 guarded)
# Recorded per variant (ep29/ep57 x PM/z0), on the SAME formal 64-slice
#   subset, SAME verified checkpoints, SAME shared z bank as VISINSPECT:
#   pre-DC and post-DC PSNR/SSIM; per-slice (post - pre) deltas;
#   per-slice (post - x0) deltas; measured/unmeasured Fourier error
#   energies + fractions; consistency before and after projection
#   (post expected ~ numerical precision).
# Panels: the SAME six registered slice positions as formal VISINSPECT
#   (loaded from its facts; no new selection), 2x5 per slice:
#   target | x0 | DC(ep29 PM) | DC(ep57 PM) | DC(ep57 z=0)
#   pre-DC ep57 PM | x0 |err| | DC(ep29 PM) |err| | DC(ep57 PM) |err|
#   | DC(ep57 z0) |err| -- shared scales as in VISINSPECT.
# INTERPRETATION RULES (recorded up front; descriptive only):
#   post-DC much better, near x0  -> much of the failure came from
#     corrupting acquired data; the current unmeasured prediction adds
#     little (near-x0 does NOT mean little work remains -- it may mean
#     the model contributes ~nothing useful beyond the baseline);
#   post-DC above x0              -> some useful unmeasured-line
#     information exists that DC can preserve;
#   post-DC below x0              -> the unmeasured prediction itself is
#     harmful; residual learning must improve that component;
#   little improvement after DC   -> most loss lies in the unmeasured
#     component or broader representation failure.
# Verification chain: REUSED from visinspect (same functions imported --
#   one verification implementation): basediag subset + ep29 anchor,
#   ep57 facts binding + recomputed competence, SHA args, provenance
#   pinned to the repo. VISINSPECT formal facts REQUIRED (positions).
# CONVENTION: logger.error + raise; smoke = dirty tree permitted,
#   PROVISIONAL, separate output dir; no fallback.
# Changelog (v0.2 -> v0.3, final pre-smoke fixes):
#   * NON-FINITE HARD FAILURES on every diagnostic scalar (pre/post
#     consistency, measured/total energy, per slice) and every summary
#     array (cons, Em, Eu, Et, fractions) BEFORE any threshold
#     comparison -- NaN > tol is False, so the v0.2 DC validity check
#     could be bypassed (the same trap fixed in BASEDIAG; now applied
#     here too).
#   * DATA IDENTITY cross-bound, not just positions: current
#     (file, slice_index) identifiers and the ordered subset manifest
#     are rebuilt exactly as VISINSPECT recorded them and must EQUAL
#     the formal VISINSPECT records (smoke: first-8 identifier prefix
#     compared; manifest equality formal-only). Same-positions !=
#     same-identity -- VISINSPECT's own forward-binding lesson applied.
#   * PRE-DC EQUIVALENCE (formal): x0 PSNR must reproduce VISINSPECT's
#     criteria_inputs, and ep29-PM pre-DC PSNR must equal x0 + the
#     recorded signed diff (atol 1e-6) -- the whole binding chain
#     becomes one falsifiable assertion: "same evidence, with DC
#     applied".
#   * total_energy recorded from authoritative Et; Parseval closure
#     |Em+Eu-Et| max recorded; fraction denominator (post-guard Em+Eu)
#     stated explicitly.
#   * VISINSPECT source verified (script string, active_subset_size 64,
#     formal_n_post 32, verdict string); compute device recorded (latent
#     hashes depend on generator/device implementation).
# Changelog (v0.1 -> v0.2, pre-smoke review fixes):
#   * VISINSPECT reuse is now VERIFIED, not claimed: the loaded formal
#     facts must match this run's z_seed, formal n_post, the full
#     BASEDIAG subset indices, the BASEDIAG facts sha, and BOTH declared
#     checkpoint SHAs; positions validated (ints, unique, in-range,
#     no coercion); on a FORMAL run the freshly generated latent-bank
#     hashes must EQUAL the formal VISINSPECT z_bank hashes (smoke uses
#     a reduced provisional bank and records that fact instead).
#   * Parseval guard fixed: a negative unmeasured complement beyond
#     1e-6 * max(E_total, 1) HARD-FAILS (an operator/mask/scaling
#     mismatch must not be silently clamped); tiny roundoff clamped to
#     0 as before. Fractions computed from the RECORDED Em+Eu (the v0.1
#     1-frac_m path could go negative despite the clamp). Absolute
#     energy statistics, per-slice energies, and the aggregate fraction
#     (a different question from the mean per-slice fraction) recorded.
#   * DC exactness is a VALIDITY CONDITION, not an outcome: per-variant
#     max post-DC consistency must be <= DC_CONSISTENCY_TOL = 1e-5
#     (chosen to cover the verified c64 adjoint-preflight scale
#     1.46e-06 with margin); tol + max stored in facts.
#   * SHA args validated via the shared 64-hex helper; measured-error
#     definition note recorded (vs F(x_true), distinct from consistency
#     vs acquired y); win counts over x0 at >0 / >0.01 / >0.10 dB.
# Changelog (NEW in v0.1): Introduced.
# Update summary: measures the decomposition the residual-base v0.4
#   specification will cite; runs in minutes; changes no rulings.
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from seqref_mri.src.fastmri_data import FastMRISliceDataset
from seqref_mri.src.forward_operator import two_channel_to_complex
from seqref_mri.src.metrics import psnr_per_sample, ssim_per_sample
from seqref_mri.scripts.train_base import (_collate, _prepare,
                                           NORMALIZED_DATA_RANGE, CELL_HW,
                                           DIM)
from seqref_mri.scripts._diag.visinspect import (
    provenance, load_basediag, load_ckpt_nice29, load_ckpt_nicext57,
    verify_nicext_facts, _require_finite, _fail, _tensor_sha,
    _valid_sha_arg, Z_SEED, N_POST)

logger = logging.getLogger("seqref_mri.dcprobe")

__version__ = "0.3"

DC_CONSISTENCY_TOL = 1e-5   # validity tolerance for exact DC (covers the
                            # verified c64 adjoint-preflight 1.46e-06)
__abbr__ = "SEQREF-DCP"


def _load_visinspect(path: str) -> tuple[list[int], dict, str]:
    p = Path(path)
    if not p.is_file():
        _fail(f"--visinspect-facts missing at {path}")
    raw = p.read_bytes()          # read ONCE: hash and parse same bytes
    sha = hashlib.sha256(raw).hexdigest()
    vf = json.loads(raw)
    if vf.get("smoke") is not False:
        _fail("visinspect facts: smoke must be false (formal only)")
    if vf.get("provenance", {}).get("git_dirty") is not False:
        _fail("visinspect facts: git_dirty must be false")
    sel = vf.get("selection", {}).get("positions")
    if not isinstance(sel, list) or len(sel) != 6:
        _fail("visinspect facts: selection.positions missing/wrong length")
    if not all(type(x) is int for x in sel):
        _fail("visinspect positions must be integers")
    if len(set(sel)) != 6:
        _fail("visinspect positions must be unique")
    if min(sel) < 0 or max(sel) >= 64:
        _fail("visinspect position outside the formal 64-slice subset")
    return sel, vf, sha


def _crossbind_visinspect(vf: dict, full_subset_idx: list[int],
                          bd_sha: str, sha_nice: str,
                          sha_nicext: str) -> None:
    # The header claim "SAME subset / checkpoints / z bank" must be
    # VERIFIED, not asserted.
    if vf.get("script") != "SEQREF-VINS v0.4 visinspect":
        _fail(f"unexpected VISINSPECT facts version: {vf.get('script')!r}")
    if vf.get("verdict") != "EVIDENCE RECORDED (no gate)":
        _fail("visinspect verdict string unexpected")
    c = vf.get("constants", {})
    if c.get("z_seed") != Z_SEED:
        _fail("visinspect z_seed mismatch")
    if c.get("n_post") != N_POST:
        _fail("visinspect formal n_post mismatch")
    if c.get("formal_n_post") != N_POST:
        _fail("visinspect formal_n_post mismatch")
    if c.get("active_subset_size") != 64:
        _fail("visinspect active_subset_size != 64")
    if vf.get("subset_indices") != full_subset_idx:
        _fail("visinspect subset indices differ from the BASEDIAG subset")
    if vf.get("basediag_facts_sha256") != bd_sha:
        _fail("visinspect BASEDIAG facts anchor mismatch")
    cks = vf.get("checkpoints", {})
    if cks.get("nice_ep29", {}).get("sha256") != sha_nice:
        _fail("visinspect ep29 checkpoint mismatch")
    if cks.get("nicext_ep57", {}).get("sha256") != sha_nicext:
        _fail("visinspect ep57 checkpoint mismatch")


@torch.no_grad()
def _outputs(model, p, z_bank):
    h = model.cond(p["cond_in"])
    _require_finite("h", h)
    acc = None
    for z in z_bank:
        x = model.decode(z, h)
        _require_finite("decode", x)
        acc = x if acc is None else acc + x
    pm = acc / len(z_bank)
    z0 = model.decode(torch.zeros(p["cond_in"].shape[0], model.dim,
                                  device=p["cond_in"].device), h)
    _require_finite("z0", z0)
    return {"pm": pm, "z0": z0}


def _to_complex_un(flat, p):
    # normalized flat state -> UN-normalized complex image
    c = two_channel_to_complex(flat.view(-1, 2, CELL_HW, CELL_HW))
    return c * p["amax"].view(-1, 1, 1)


def _metrics_norm(c_un, p):
    mag = (c_un / p["amax"].view(-1, 1, 1)).abs().unsqueeze(1).cpu()
    psnr = psnr_per_sample(mag, p["tgt_norm"].cpu(),
                           data_range=NORMALIZED_DATA_RANGE)
    ssim = ssim_per_sample(mag, p["tgt_norm"].cpu(),
                           data_range=NORMALIZED_DATA_RANGE)
    _require_finite("psnr", psnr)
    _require_finite("ssim", ssim)
    return psnr, ssim


def _stats(v: np.ndarray) -> dict:
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "p5": float(np.percentile(v, 5)),
            "p95": float(np.percentile(v, 95)),
            "min": float(v.min()), "max": float(v.max())}


def probe_variant(tag, flat, p, x_true_un, x0_psnr):
    B = flat.shape[0]
    x_un = _to_complex_un(flat, p)
    pre_psnr, pre_ssim = _metrics_norm(x_un, p)
    pre_cons, post_cons = [], []
    Em, Eu, tot = [], [], []
    x_dc = torch.empty_like(x_un)
    for i in range(B):
        op, y = p["ops"][i], p["y"][i]
        # exact DC through the verified operator pair
        x_dc[i] = x_un[i] + op.A_adjoint(y - op.A(x_un[i]))
        pre_c = float(op.consistency(x_un[i], y))
        post_c = float(op.consistency(x_dc[i], y))
        # unitary Parseval energy split
        e_m = float(torch.sum(torch.abs(op.A(x_un[i])
                                        - op.A(x_true_un[i])) ** 2))
        e_t = float(torch.sum(torch.abs(x_un[i] - x_true_un[i]) ** 2))
        for nm, val in (("pre consistency", pre_c),
                        ("post consistency", post_c),
                        ("measured energy", e_m),
                        ("total energy", e_t)):
            if not np.isfinite(val):
                _fail(f"{tag} slice {i}: {nm} is non-finite")
        pre_cons.append(pre_c)
        post_cons.append(post_c)
        raw_eu = e_t - e_m
        tol = 1e-6 * max(e_t, 1.0)
        if raw_eu < -tol:
            _fail(f"{tag} slice {i}: measured energy exceeds total "
                  f"beyond roundoff (Em={e_m:.6e}, Et={e_t:.6e}) -- "
                  "operator/mask/scaling mismatch, NOT clampable")
        e_u = max(raw_eu, 0.0)             # tiny roundoff only
        Em.append(e_m); Eu.append(e_u); tot.append(e_t)
    _require_finite("x_dc", torch.view_as_real(x_dc))
    Em, Eu, Et = map(np.asarray, (Em, Eu, tot))
    den = np.maximum(Em + Eu, 1e-30)   # declared post-guard denominator
    frac_m = Em / den
    frac_u = Eu / den
    post_arr = np.asarray(post_cons)
    for nm, arr in (("pre_cons", np.asarray(pre_cons)),
                    ("post_cons", post_arr), ("Em", Em), ("Eu", Eu),
                    ("Et", Et), ("frac_m", frac_m), ("frac_u", frac_u)):
        if not np.isfinite(arr).all():
            _fail(f"{tag}: {nm} contains non-finite values")
    # VALIDITY: exact DC is the mechanism under test -- a single failed
    # slice invalidates the probe (mean alone could hide it; finiteness
    # asserted ABOVE so NaN cannot bypass this comparison).
    if post_arr.max() > DC_CONSISTENCY_TOL:
        _fail(f"{tag}: post-DC consistency max {post_arr.max():.3e} "
              f"exceeds validity tol {DC_CONSISTENCY_TOL:.1e}")
    post_psnr, post_ssim = _metrics_norm(x_dc, p)
    pre_np, post_np = pre_psnr.numpy(), post_psnr.numpy()
    rec = {
        "pre_dc": {"psnr": _stats(pre_np),
                   "ssim": _stats(pre_ssim.numpy()),
                   "consistency_mean": float(np.mean(pre_cons))},
        "post_dc": {"psnr": _stats(post_np),
                    "ssim": _stats(post_ssim.numpy()),
                    "consistency_mean": float(np.mean(post_cons))},
        "delta_post_minus_pre_psnr": _stats(post_np - pre_np),
        "delta_post_minus_x0_psnr": _stats(post_np - x0_psnr),
        "n_slices_post_above_x0": {
            "gt_0dB": int((post_np > x0_psnr).sum()),
            "gt_0.01dB": int((post_np > x0_psnr + 0.01).sum()),
            "gt_0.10dB": int((post_np > x0_psnr + 0.10).sum())},
        "dc_validity": {"tolerance": DC_CONSISTENCY_TOL,
                        "post_consistency_max": float(post_arr.max()),
                        "post_consistency_mean": float(post_arr.mean())},
        "fourier_error_energy": {
            "measured_energy": _stats(Em),
            "unmeasured_energy": _stats(Eu),
            "total_energy": _stats(Et),
            "parseval_closure_abs_max": float(np.abs(Em + Eu - Et).max()),
            "fraction_denominator": "post-guard Em + Eu (declared)",
            "measured_frac_per_slice": _stats(frac_m),
            "unmeasured_frac_per_slice": _stats(frac_u),
            "aggregate_measured_frac": float(
                Em.sum() / max((Em + Eu).sum(), 1e-30)),
            "definition_note": "measured-subspace RECONSTRUCTION error "
                               "is defined vs the target's forward "
                               "projection A(x_true); data CONSISTENCY "
                               "is separately measured vs acquired y; "
                               "negative complement beyond roundoff "
                               "hard-fails"},
        "per_slice": {"pre_psnr": pre_np.tolist(),
                      "post_psnr": post_np.tolist(),
                      "measured_energy": Em.tolist(),
                      "unmeasured_energy": Eu.tolist(),
                      "total_energy": Et.tolist()},
    }
    logger.info("[dcp] %s: pre %.2f -> post %.2f dB (x0 %.2f) | "
                "post cons %.2e | measured-frac %.3f", tag,
                pre_np.mean(), post_np.mean(), x0_psnr.mean(),
                np.mean(post_cons), frac_m.mean())
    return rec, x_dc


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ckpt-nice", required=True)
    ap.add_argument("--sha-nice", required=True)
    ap.add_argument("--ckpt-nicext", required=True)
    ap.add_argument("--sha-nicext", required=True)
    ap.add_argument("--nicext-facts", required=True)
    ap.add_argument("--basediag-facts", required=True)
    ap.add_argument("--visinspect-facts", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    script_path = Path(__file__).resolve()
    seqref_root = script_path.parents[2]
    repo_root = script_path.parents[3]
    if seqref_root.name != "seqref_mri":
        _fail(f"script not at seqref_mri/scripts/_diag (got {seqref_root})")
    base = seqref_root / "results" / "_diag"
    default_formal = (base / "dcprobe").resolve()
    default_smoke = (base / "dcprobe_smoke").resolve()
    out_dir = (Path(a.out).resolve() if a.out
               else (default_smoke if a.smoke else default_formal))
    if a.smoke and out_dir == default_formal:
        _fail("smoke output may not target the formal DCPROBE directory")
    if a.out and out_dir.exists():
        _fail(f"explicit --out target exists: {out_dir}")
    if not a.smoke and out_dir.exists():
        _fail(f"formal output already exists: {out_dir} -- delete "
              "manually to rerun")
    tmp = out_dir.parent / (out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "panels").mkdir(parents=True)

    actual_n_post = 4 if a.smoke else N_POST
    facts: dict = {"script": f"{__abbr__} v{__version__} dcprobe",
                   "smoke": a.smoke,
                   "constants": {"z_seed": Z_SEED,
                                 "n_post": actual_n_post,
                                 "formal_n_post": N_POST},
                   "projection": "x_DC = x_hat + A^H(y - A x_hat) "
                                 "(exact DC via the verified operator "
                                 "pair; unitary F)"}
    facts["provenance"] = provenance(sys.argv, allow_dirty=a.smoke,
                                     repo_root=repo_root)
    facts["provenance"]["device"] = device
    if facts["provenance"]["git_dirty"]:
        facts["EVIDENCE_STATUS"] = "PROVISIONAL (smoke, dirty tree)"

    sha_nice = _valid_sha_arg("--sha-nice", a.sha_nice)
    sha_nicext = _valid_sha_arg("--sha-nicext", a.sha_nicext)
    subset_idx, bd_facts, bd_sha = load_basediag(a.basediag_facts,
                                                 sha_nice)
    vpos, vf, v_sha = _load_visinspect(a.visinspect_facts)
    _crossbind_visinspect(vf, subset_idx, bd_sha, sha_nice, sha_nicext)
    facts["basediag_facts_sha256"] = bd_sha
    facts["visinspect_facts_sha256"] = v_sha
    facts["visinspect_crossbind"] = ("VERIFIED: z_seed, formal n_post, "
                                     "full subset indices, basediag "
                                     "anchor, both checkpoint SHAs")
    if a.smoke:
        subset_idx = subset_idx[:8]
        vpos = [p_ for p_ in vpos if p_ < 8] or [0]
    facts["subset_indices"] = subset_idx
    facts["panel_positions_from_visinspect"] = vpos

    ds = FastMRISliceDataset(a.data_root, split="val", mode="eval")
    if min(subset_idx) < 0 or max(subset_idx) >= len(ds):
        _fail("subset index outside current dataset")
    items = [ds[i] for i in subset_idx]
    # DATA IDENTITY: rebuild identifiers + manifest exactly as VISINSPECT
    # recorded them; positions matching is NOT identity matching.
    current_ids = [
        {"subset_position": pos, "dataset_index": int(ds_idx),
         "file": it["meta"]["file"],
         "slice_index": it["meta"]["slice_index"]}
        for pos, (ds_idx, it) in enumerate(zip(subset_idx, items))]
    manifest_text = "\n".join(
        f'{it["meta"]["file"]}|{it["meta"]["slice_index"]}'
        for it in items)
    current_manifest = {"n_slices": len(items), "ordered": True,
                        "sha256": hashlib.sha256(
                            manifest_text.encode()).hexdigest()}
    facts["slice_identifiers"] = current_ids
    facts["subset_manifest"] = current_manifest
    if not a.smoke:
        if current_ids != vf.get("slice_identifiers"):
            _fail("current slice identities differ from formal VISINSPECT")
        if current_manifest != vf.get("subset_manifest"):
            _fail("current ordered subset manifest differs from VISINSPECT")
        facts["data_identity"] = "VERIFIED equal to formal VISINSPECT"
    else:
        vids = (vf.get("slice_identifiers") or [])[:len(current_ids)]
        if current_ids != vids:
            _fail("smoke slice identities differ from the VISINSPECT "
                  "prefix")
        facts["data_identity"] = ("smoke: first-%d identifier prefix "
                                  "verified; manifest equality formal-"
                                  "only" % len(current_ids))
    batch = _collate(items)
    p = _prepare(batch, device, test0=False)
    for key in ("cond_in", "x_norm", "tgt_norm", "amax"):
        _require_finite(f"prepared {key}", p[key])

    with torch.no_grad():
        nice29, ck29 = load_ckpt_nice29(a.ckpt_nice, sha_nice, device)
        next57, ck57, next_cfg, next_init = load_ckpt_nicext57(
            a.ckpt_nicext, sha_nicext, sha_nice, device)
        ck57["best_verified"] = verify_nicext_facts(
            a.nicext_facts, sha_nicext, sha_nice, next_cfg,
            a.ckpt_nicext, next_init)
        facts["checkpoints"] = {"nice_ep29": ck29, "nicext_ep57": ck57}

        torch.manual_seed(Z_SEED)
        z_bank = [torch.randn(len(subset_idx), DIM, device=device)
                  for _ in range(actual_n_post)]
        z_hashes = [_tensor_sha(z) for z in z_bank]
        facts["z_bank_sha256"] = z_hashes
        if not a.smoke:
            if z_hashes != vf.get("z_bank_sha256"):
                _fail("DCPROBE latent bank does not match the formal "
                      "VISINSPECT z bank -- identity claim unverifiable")
            facts["z_bank_identity"] = "VERIFIED equal to VISINSPECT"
        else:
            facts["z_bank_identity"] = ("REDUCED PROVISIONAL smoke bank "
                                        "(shape/count differ; identity "
                                        "with VISINSPECT not claimed)")
        o29 = _outputs(nice29, p, z_bank)
        o57 = _outputs(next57, p, z_bank)

    x_true_un = _to_complex_un(p["x_norm"].flatten(1), p)
    x0_un = _to_complex_un(p["cond_in"].flatten(1), p)
    x0_psnr, _ = _metrics_norm(x0_un, p)
    x0_psnr = x0_psnr.numpy()
    facts["x0_psnr_per_slice"] = x0_psnr.tolist()

    dc_imgs = {}
    facts["variants"] = {}
    for tag, flat in (("ep29_pm", o29["pm"]), ("ep29_z0", o29["z0"]),
                      ("ep57_pm", o57["pm"]), ("ep57_z0", o57["z0"])):
        rec, x_dc = probe_variant(tag, flat, p, x_true_un, x0_psnr)
        facts["variants"][tag] = rec
        dc_imgs[tag] = x_dc

    # PRE-DC EQUIVALENCE (formal): the binding chain as one falsifiable
    # assertion -- DCPROBE must REPRODUCE VISINSPECT's pre-DC evidence.
    if not a.smoke:
        exp_x0 = np.asarray(vf["criteria_inputs"]["x0_psnr_per_slice"])
        exp_ep29 = exp_x0 + np.asarray(
            vf["criteria_inputs"]["ep29pm_minus_x0_signed"])
        got_ep29 = np.asarray(
            facts["variants"]["ep29_pm"]["per_slice"]["pre_psnr"])
        if not np.allclose(x0_psnr, exp_x0, rtol=0, atol=1e-6):
            _fail("x0 metrics do not reproduce formal VISINSPECT")
        if not np.allclose(got_ep29, exp_ep29, rtol=0, atol=1e-6):
            _fail("ep29 PM pre-DC metrics do not reproduce formal "
                  "VISINSPECT")
        facts["pre_dc_equivalence"] = ("VERIFIED: x0 and ep29-PM pre-DC "
                                       "PSNR reproduce VISINSPECT "
                                       "(atol 1e-6)")

    # panels: the SAME six registered positions, shared scales
    tgt = p["tgt_norm"].cpu()
    amax = p["amax"].view(-1, 1, 1)
    pre57 = _to_complex_un(o57["pm"], p)
    for pos in vpos:
        t_img = tgt[pos, 0]
        vmax = float(torch.quantile(t_img.flatten(), 0.995))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        def m(c):
            return (c[pos] / amax[pos]).abs().cpu()
        imgs_top = [(t_img, "target"), (m(x0_un), "x0"),
                    (m(dc_imgs["ep29_pm"]), "DC(ep29 PM)"),
                    (m(dc_imgs["ep57_pm"]), "DC(ep57 PM)"),
                    (m(dc_imgs["ep57_z0"]), "DC(ep57 z=0)")]
        errs = {"x0": (m(x0_un) - t_img).abs(),
                "dc29pm": (m(dc_imgs["ep29_pm"]) - t_img).abs(),
                "dc57pm": (m(dc_imgs["ep57_pm"]) - t_img).abs(),
                "dc57z0": (m(dc_imgs["ep57_z0"]) - t_img).abs()}
        evmax = float(torch.quantile(
            torch.cat([e.flatten() for e in errs.values()]), 0.995))
        if not np.isfinite(evmax) or evmax <= 0:
            evmax = 1.0
        imgs_bot = [(m(pre57), "pre-DC ep57 PM", "img"),
                    (errs["x0"], "x0 |err|", "err"),
                    (errs["dc29pm"], "DC(ep29 PM) |err|", "err"),
                    (errs["dc57pm"], "DC(ep57 PM) |err|", "err"),
                    (errs["dc57z0"], "DC(ep57 z0) |err|", "err")]
        fig, axes = plt.subplots(2, 5, figsize=(16, 6.6))
        for ax, (img, title) in zip(axes[0], imgs_top):
            ax.imshow(img.numpy(), cmap="gray", vmin=0.0, vmax=vmax)
            ax.set_title(title, fontsize=8); ax.axis("off")
        for ax, (img, title, kind) in zip(axes[1], imgs_bot):
            ax.imshow(img.numpy(), cmap="gray", vmin=0.0,
                      vmax=(vmax if kind == "img" else evmax))
            ax.set_title(title, fontsize=8); ax.axis("off")
        fig.suptitle(f"DCPROBE — subset position {pos} (ds index "
                     f"{subset_idx[pos]})", fontsize=9)
        fig.savefig(tmp / "panels" / f"slice_pos{pos:02d}.png",
                    dpi=110, bbox_inches="tight")
        plt.close(fig)

    facts["interpretation_rules"] = (
        "post-DC near x0: much of the failure came from corrupting "
        "acquired data; current unmeasured prediction adds little (does "
        "NOT mean little work remains). post-DC above x0: some useful "
        "unmeasured-line information exists. post-DC below x0: the "
        "unmeasured prediction itself is harmful. little improvement: "
        "loss lies mainly in the unmeasured component or broader "
        "representation failure.")
    facts["verdict"] = ("EVIDENCE RECORDED (smoke, provisional)"
                        if a.smoke else "EVIDENCE RECORDED (no gate)")
    with open(tmp / "facts.json", "w") as f:
        json.dump(facts, f, indent=2)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    logger.info("[dcp] %s -- report at %s", facts["verdict"], out_dir)


if __name__ == "__main__":
    main()

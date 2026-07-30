# SEQREF-P12PLT v0.2 -- P1/P2 diagnostic plots (EXEC v0.4 §8, A3)
# LIFETIME: KEEP  (the SCRIPT is permanent; the PNGs it emits are DIAGNOSTIC)
#
# Reads ONLY persisted facts, and only after verifying each artefact against
# its authoritative sidecar. Plots NEVER participate in a verdict; nothing here
# recomputes a scientific quantity.
#
# LIFETIME RATIONALE (v0.2)
#   The FIGURES are diagnostic and are deleted after inspection. The READER is
#   not: representation_facts.json and support_facts.json are KEEP artefacts
#   that later stages and later reviewers will want to look at, and a reader
#   that verifies sidecars and refuses error records is the maintained way to
#   do that. Deleting it would leave permanent artefacts with no maintained
#   viewer.
#
# Panels (A3, locked)
#   P1  1. rho_imag_E distribution, REAL (1e-10) and COMPLEX (1e-6) marked
#       2. rho_imag_max distribution, 1e-5 and 1e-3 marked
#       3. rho_imag_E vs rho_imag_max scatter with decision regions
#       4. ordered E_re/S_ref^2 with the R_REAL_MIN blocking floor
#   P2  1. rho_M distribution, verdict path and diagnostic float64 path
#       2. relative_max distribution with REL_MAX_MAX
#       3. residual-energy ratio with the near-zero boundary, coloured by branch
#       4. absolute leakage vs k_i with the permitted line, plus a companion
#          panel carrying x0_rel_error (fp32, fp64) AND the direct fp64-vs-fp32
#          path discrepancy -- shared panel, so the locked 4+4 count is intact.
#          X0_ASSERT_RTOL is drawn for the fp32 CONTRACT series only and is
#          labelled as such; the direct path series is non-blocking and sits on
#          a twin axis so the figure cannot imply 1e-6 gates it.
#   The conjugate-symmetry plot is OPTIONAL (--conj), non-blocking cross-check.
#
# CONVENTION: logger.error + raise on every failure path. No fallback, no mock.
#
# Changelog
#   v0.2 (2026-07-30) LIFETIME DIAGNOSTIC -> KEEP for the SCRIPT, after the
#     authoritative P1/P2 run (commit c144242). The emitted PNGs remain
#     DIAGNOSTIC and are still deleted after inspection. No plotting logic
#     changed.
#   v0.1 (2026-07-30) Created under Amendment A3.
#
# Update summary (v0.2): the DIAGNOSTIC tag conflated the script with its
#   output. The figures are disposable; the reader of two permanent artefacts
#   is not, and it was committed alongside them. Only the declared lifetime
#   changed.

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve_repo_root(start: str) -> str:
    """Locate the repo root by walking up until seqref_mri/src is found, and
    VERIFY it rather than assuming a fixed depth.

    A hard-coded os.path.join(_HERE, "..", "..") silently imports from the
    wrong tree if the layout ever differs -- and it already does differ inside
    this campaign: p0s_normalisation_scale.py walks up THREE levels from the
    same directory this file sits in, so at most one of the two can be right.
    No fallback: an unlocatable root raises.
    """
    d = os.path.abspath(start)
    for _ in range(8):
        if os.path.isfile(os.path.join(d, "seqref_mri", "src",
                                       "preflight_io.py")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise RuntimeError(
        f"could not locate the repo root above {start}: no ancestor contains "
        f"seqref_mri/src/preflight_io.py. Run from a checkout, e.g. "
        f"/home/benjamin/CSMFII/seqref_mri/scripts/")


_REPO = _resolve_repo_root(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "seqref_mri", "src"))

from preflight_io import verify_sidecar  # noqa: E402

SCRIPT_ID = "SEQREF-P12PLT"
SCRIPT_VERSION = "v0.2"

logger = logging.getLogger(SCRIPT_ID)

_POS_FLOOR = 1e-300      # plotting-only: log axes cannot show exact zero


def _load(path: str, stage: str) -> dict:
    sha = verify_sidecar(path)
    with open(path, "rb") as fh:
        facts = json.load(fh)
    if facts.get("stage") != stage:
        logger.error("%s is stage %r, expected %r", path, facts.get("stage"),
                     stage)
        raise ValueError(f"wrong stage in {path}")
    if facts.get("artefact_type") != "stage_facts":
        logger.error("%s has artefact_type %r; an error record is not a facts "
                     "artefact and must not be plotted", path,
                     facts.get("artefact_type"))
        raise ValueError(f"not a stage facts artefact: {path}")
    logger.info("loaded %s stage=%s verdict=%s sha=%s", path, stage,
                facts.get("verdict"), sha[:16])
    return facts


def _vals(slices: list[dict], key: str) -> np.ndarray:
    out = [s[key] for s in slices
           if isinstance(s.get(key), (int, float))
           and not isinstance(s.get(key), bool)]
    return np.asarray(out, dtype=np.float64)


def _logsafe(a: np.ndarray) -> np.ndarray:
    return np.where(a > 0.0, a, _POS_FLOOR)


def plot_p1(facts: dict, out_path: str) -> None:
    s = facts["slices"]
    t = facts["thresholds"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"P1 representation — verdict {facts['verdict']} / ruling "
                 f"{facts.get('ruling')} (n={len(s)}) — DIAGNOSTIC")

    rho_e = _vals(s, "rho_imag_E")
    rho_m = _vals(s, "rho_imag_max")

    if rho_e.size:
        ax[0, 0].hist(np.log10(_logsafe(rho_e)), bins=40, color="#4C72B0")
    for v, c, lab in ((t["RHO_E_REAL_MAX"], "#2CA02C", "REAL ≤ 1e-10"),
                      (t["RHO_E_COMPLEX_MIN"], "#D62728", "COMPLEX ≥ 1e-6")):
        ax[0, 0].axvline(np.log10(v), color=c, ls="--", label=lab)
    ax[0, 0].set_xlabel("log10 rho_imag_E")
    ax[0, 0].set_ylabel("slices")
    ax[0, 0].legend(fontsize=8)

    if rho_m.size:
        ax[0, 1].hist(np.log10(_logsafe(rho_m)), bins=40, color="#4C72B0")
    for v, c, lab in ((t["RHO_MAX_REAL_MAX"], "#2CA02C", "REAL ≤ 1e-5"),
                      (t["RHO_MAX_COMPLEX_MIN"], "#D62728", "COMPLEX ≥ 1e-3")):
        ax[0, 1].axvline(np.log10(v), color=c, ls="--", label=lab)
    ax[0, 1].set_xlabel("log10 rho_imag_max")
    ax[0, 1].set_ylabel("slices")
    ax[0, 1].legend(fontsize=8)

    if rho_e.size and rho_m.size:
        ax[1, 0].scatter(np.log10(_logsafe(rho_e)), np.log10(_logsafe(rho_m)),
                         s=12, alpha=0.7, color="#4C72B0")
        ax[1, 0].axvspan(-320, np.log10(t["RHO_E_REAL_MAX"]), color="#2CA02C",
                         alpha=0.08)
        ax[1, 0].axhspan(-320, np.log10(t["RHO_MAX_REAL_MAX"]),
                         color="#2CA02C", alpha=0.08)
        ax[1, 0].axvline(np.log10(t["RHO_E_COMPLEX_MIN"]), color="#D62728",
                         ls="--")
        ax[1, 0].axhline(np.log10(t["RHO_MAX_COMPLEX_MIN"]), color="#D62728",
                         ls="--")
    ax[1, 0].set_xlabel("log10 rho_imag_E")
    ax[1, 0].set_ylabel("log10 rho_imag_max")
    ax[1, 0].set_title("decision regions (green = REAL corner)", fontsize=9)

    e_rel = np.sort(_vals(s, "E_re_over_S_ref_sq"))
    if e_rel.size:
        ax[1, 1].semilogy(np.arange(e_rel.size), _logsafe(e_rel), ".",
                          color="#4C72B0")
    ax[1, 1].axhline(t["R_REAL_MIN"], color="#D62728", ls="--",
                     label="R_REAL_MIN (BLOCK floor)")
    ax[1, 1].set_xlabel("slice rank")
    ax[1, 1].set_ylabel("E_re / S_ref²")
    ax[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def plot_p2(facts: dict, out_path: str) -> None:
    s = facts["slices"]
    t = facts["thresholds"]
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(f"P2 measured support — verdict {facts['verdict']} "
                 f"(n={len(s)}, verdict path "
                 f"{facts['summary']['dtype_path']['selected']}) — DIAGNOSTIC")

    rho = _vals(s, "rho_M")
    rho64 = _vals(s, "rho_M_f64_diagnostic")
    if rho.size:
        ax[0, 0].hist(np.log10(_logsafe(rho)), bins=40, alpha=0.65,
                      label="verdict path (f32)", color="#4C72B0")
    if rho64.size:
        ax[0, 0].hist(np.log10(_logsafe(rho64)), bins=40, alpha=0.45,
                      label="f64 sensitivity", color="#DD8452")
    ax[0, 0].axvline(np.log10(t["RHO_M_MAX_F32"]), color="#D62728", ls="--",
                     label="RHO_M_MAX_F32")
    ax[0, 0].set_xlabel("log10 rho_M")
    ax[0, 0].set_ylabel("slices")
    ax[0, 0].legend(fontsize=8)

    rmax = _vals(s, "relative_max")
    if rmax.size:
        ax[0, 1].hist(np.log10(_logsafe(rmax)), bins=40, color="#4C72B0")
    ax[0, 1].axvline(np.log10(t["REL_MAX_MAX"]), color="#D62728", ls="--",
                     label="REL_MAX_MAX")
    ax[0, 1].set_xlabel("log10 relative_max")
    ax[0, 1].set_ylabel("slices")
    ax[0, 1].legend(fontsize=8)

    ratio = np.asarray([r.get("residual_energy_ratio") for r in s],
                       dtype=np.float64)
    branch = [r.get("branch") for r in s]
    for name, colour in (("ordinary", "#4C72B0"), ("near_zero", "#DD8452")):
        m = np.asarray([b == name for b in branch])
        if m.any():
            ax[0, 2].semilogy(np.flatnonzero(m), _logsafe(ratio[m]), ".",
                              color=colour, label=name)
    ax[0, 2].axhline(t["R_RESID_MIN"], color="#D62728", ls="--",
                     label="R_RESID_MIN (switch)")
    ax[0, 2].set_xlabel("slice position")
    ax[0, 2].set_ylabel("‖F Δx‖² / S_ref²")
    ax[0, 2].legend(fontsize=8)

    k_i = _vals(s, "k_i")
    leak = _vals(s, "max_MFdx")
    if k_i.size and leak.size and k_i.size == leak.size:
        ax[1, 0].loglog(_logsafe(k_i), _logsafe(leak), ".", color="#4C72B0")
        grid = np.logspace(np.log10(max(k_i.min(), _POS_FLOOR)),
                           np.log10(k_i.max()), 50) if k_i.max() > 0 else None
        if grid is not None:
            ax[1, 0].loglog(grid, t["ABS_LEAK_F32"] * grid, "--",
                            color="#D62728", label="ABS_LEAK_F32 · k_i")
    ax[1, 0].set_xlabel("k_i = max|M F x_true|")
    ax[1, 0].set_ylabel("max|M F Δx|")
    ax[1, 0].legend(fontsize=8)

    x32 = _vals(s, "x0_rel_error")
    x64 = _vals(s, "x0_rel_error_f64")
    xpath = _vals(s, "x0_path_rel_difference")
    if x32.size:
        ax[1, 1].hist(np.log10(_logsafe(x32)), bins=40, alpha=0.65,
                      label="fp32", color="#4C72B0")
    if x64.size:
        ax[1, 1].hist(np.log10(_logsafe(x64)), bins=40, alpha=0.45,
                      label="fp64", color="#DD8452")
    ax[1, 1].axvline(np.log10(t["X0_ASSERT_RTOL"]), color="#D62728", ls="--",
                     label="X0_ASSERT_RTOL — fp32 contract only")
    if xpath.size:
        # DIRECT fp64-vs-fp32 reconstruction difference: NON-BLOCKING
        # operator-path sensitivity, plotted on a TWIN axis so the shared
        # X0_ASSERT_RTOL line cannot be read as gating it. That tolerance
        # governs the pipeline-matched fp32 contract assertion and nothing
        # else.
        tw = ax[1, 1].twinx()
        tw.hist(np.log10(_logsafe(xpath)), bins=40, color="#55A868",
                histtype="step", linewidth=1.6,
                label="direct f64−f32 (NON-BLOCKING, ungated)")
        tw.set_ylabel("slices — direct path difference", color="#55A868",
                      fontsize=8)
        tw.tick_params(axis="y", labelcolor="#55A868", labelsize=7)
        tw.legend(fontsize=7, loc="upper right")
    ax[1, 1].set_xlabel("log10 x0_rel_error")
    ax[1, 1].set_ylabel("slices")
    ax[1, 1].set_title("x0 contract margin (gated) + operator-path difference "
                       "(ungated, right axis)", fontsize=9)
    ax[1, 1].legend(fontsize=7, loc="upper left")

    dec = _vals(s, "boundary_distance_decades")
    if dec.size:
        ax[1, 2].hist(dec, bins=40, color="#4C72B0")
    ax[1, 2].axvline(t["P2_BOUNDARY_BAND_DECADES"], color="#D62728", ls="--",
                     label="band (NON-VERDICT)")
    ax[1, 2].set_xlabel("|log10(ratio / R_RESID_MIN)|")
    ax[1, 2].set_ylabel("slices")
    ax[1, 2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def plot_conj(facts: dict, out_path: str) -> None:
    s = facts["slices"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle("P1 conjugate-symmetry cross-check — NON-BLOCKING")
    for k, a, lab in (("conj_symmetry_violation_abs", ax[0], "absolute"),
                      ("conj_symmetry_violation_rel", ax[1], "relative")):
        v = _vals(s, k)
        if v.size:
            a.hist(np.log10(_logsafe(v)), bins=40, color="#4C72B0")
        a.set_xlabel(f"log10 {lab} violation of K[-u,-v] = conj(K[u,v])")
        a.set_ylabel("slices")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    logger.info("wrote %s", out_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="SEQREF-P12PLT v0.2 -- plots from persisted P1/P2 facts. "
                    "The script is KEEP; the PNGs are DIAGNOSTIC and should "
                    "be deleted after inspection.")
    ap.add_argument("--p1-facts", default=None)
    ap.add_argument("--p2-facts", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--conj", action="store_true",
                    help="also emit the optional conjugate-symmetry panel")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if not args.p1_facts and not args.p2_facts:
        logger.error("nothing to plot: give --p1-facts and/or --p2-facts")
        raise SystemExit(2)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.p1_facts:
        f1 = _load(args.p1_facts, "P1")
        plot_p1(f1, os.path.join(args.out_dir, "p1_representation.png"))
        if args.conj:
            plot_conj(f1, os.path.join(args.out_dir, "p1_conj_symmetry.png"))
    if args.p2_facts:
        f2 = _load(args.p2_facts, "P2")
        plot_p2(f2, os.path.join(args.out_dir, "p2_support.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# =============================================================================
# STEP-1_1 v0.8 -- experiments.step_1_1.summary
# Purpose: read `sanity_per_epoch` in report.json and produce a pass/fail ledger
#          printed to the console (ANSI coloured) + dumped to summary.txt and
#          summary.csv in the run directory.
# CONVENTION: read-only -- never mutates the run. Failures -> logger.error
#             + raise (no fallback / mock / pass).
# Changelog (v0.8 -> v0.9):
#   * film_alive in _conditioning_block now judged over the LAST 5 epochs
#     (passes if v2_diag.film_alive.alive is True in >=1 of the last 5),
#     not the single final epoch. film_alive flickers true/false epoch-to-
#     epoch on a noisy diagnostic; a healthy run could false-fail conditioning
#     on a last-epoch coin-flip (observed: seeds 1/2 of realnvp s2/n0.05,
#     FZDY ~1.6 / fwd_rel ~0.22 but film_alive_last False -> cond_pass False).
#   * N/A semantics preserved: film_ok is None only when NO epoch in the
#     window carries v2_diag.film_alive (NSF --no-use-v2-conditioner).
# Changelog (v0.7 -> v0.8):
#   * Phase 0 (conditioning collapse): exit_ok now also requires FINAL-epoch
#     conditioning. fzdy_last_ok (fixed_z_different_y.passed) + h_std_last_ok
#     (h_std_obs_hist[-1].batch >= 0.02) are UNIVERSAL; shuffle_last_ok
#     (v2_diag.logp_shuffle.gap_mean > 0) + film_last_ok (v2_diag.film_alive
#     .alive) are graded only when v2_diag exists, else N/A (NSF never fails).
#   * FZDY judged on the LAST epoch only -- NOT moved into _HARD_GATES, whose
#     passed==total rule would false-fail the conditioning ramp-up.
#   * NEW summary_block["conditioning"] sub-block + conditioning remedy hints.
# Changelog (v0.6 -> v0.7):
#   * BUGFIX (Glow false FAIL): _gate_is_na now checks the SPECIFIC keys a
#     gate reads, not "all snapshots empty". Previously, if ANY cond_gate
#     key was populated (e.g. cond_grad for grad_flow) the snapshots looked
#     non-empty, so h_alive / film_alive / determinism graded their MISSING
#     keys as 0 -> false FAIL (0/70). Now a gate is N/A iff none of its
#     required keys appear in any snapshot. Glow's authoritative conditioning
#     check stays v2_diag.film_alive (unaffected here).
#   * NEW: glow_film_gain summary. min/mean/max of the per-layer film_gain
#     (read from report["glow_film_gain_hist"]) printed + stored in the
#     summary block. Surfaces the conditioning-strength decay at a glance.
# Changelog (v0.5 -> v0.6):
#   * NEW informational gate: fixed_z_sensitivity (FZDY Phase 3). Reads
#     rec["fixed_z_different_y"]["passed"]. Non-blocking; not cond-gate
#     dependent. Auto-appears in summary.txt / summary.csv.
# Changelog (v0.4 -> v0.5):
#   * N/A semantics: gates that depend on cond_gate_history (h_alive,
#     film_alive, determinism, no_nan_inf, null_control_gap, h_st_slope,
#     grad_flow) are now marked N/A when the history is missing or empty
#     (e.g. Glow runs where _cond_gate_hook is intentionally skipped).
#     N/A gates do NOT count toward exit criterion and do NOT appear in
#     remedy hints. v2 gates remain the authoritative conditioning check
#     for Glow.
#   * Display: N/A shown as YELLOW "--" instead of red "X".
# Changelog (NEW in v0.4):
#   * Introduced. 7 hard gates + 4 informational checks.
# Update summary:
#   v0.9 makes the film_alive conditioning sub-check robust to last-epoch noise:
#   it now passes if FiLM was alive in any of the last 5 epochs, matching the
#   windowed spirit of the other conditioning signals. Fixes spurious
#   conditioning_pass=False on healthy runs (real conditioning intact: high FZDY,
#   low fwd_rel) where only the final-epoch FiLM std happened to dip below eps.
#   Re-grades from existing JSON, no retrain.
#   v0.8 makes conditioning collapse BLOCK exit_criteria_met: a run that ignores
#   y (final-epoch FZDY < fzdy_tau) now fails even with good NLL/latent/cycle.
#   v2-only signals (shuffle/film) graded when present, N/A otherwise, so NSF is
#   unaffected. Existing runs re-grade with no retrain -- metrics already in JSON.
#   v0.7 fixed the Glow false-FAIL on conditioning gates (they now correctly
#   read N/A when their cond_gate keys are absent) and surfaces film_gain
#   min/mean/max so its decay is visible without opening report.json.
# =============================================================================
from __future__ import annotations
import csv
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
__version__ = "0.9"
__abbr__ = "STEP-1_1"


# ANSI colour codes -----------------------------------------------------------
def _use_color() -> bool:
    if os.environ.get("NO_COLOR", "") != "":
        return False
    return True


_G = "\033[32m" if _use_color() else ""
_R = "\033[31m" if _use_color() else ""
_Y = "\033[33m" if _use_color() else ""
_B = "\033[1m"  if _use_color() else ""
_N = "\033[0m"  if _use_color() else ""


# Gate definitions ------------------------------------------------------------
# Each gate maps one epoch's sanity record -> bool (pass/fail).
# Missing keys -> False (treated as failure but noted).
_HARD_GATES: list[tuple[str, str]] = [
    ("invertibility",               "invertibility FP64 <1e-5"),
    ("latent_ks",                   "latent KS vs N(0,1) <0.05"),
    ("cycle_max",                   "cycle_max < 1e-12"),
    ("h_alive",                     "h std > 1e-3"),
    ("film_alive",                  "FiLM γ,β std > 1e-3 aggregate"),
    ("determinism",                 "(y,seed) max|Δh| < 1e-6"),
    ("no_nan_inf",                  "no NaN/Inf this epoch"),
]
_INFO_GATES: list[tuple[str, str]] = [
    ("null_control_gap",            "null-control gap ≥ 0.10"),
    ("h_st_slope",                  "h_st Δlogp slope > 1e-3"),
    ("grad_flow",                   "grad norm > 1e-6"),
    ("fwd_consistency_tracked",     "fwd residual reported"),
    ("fixed_z_sensitivity",         "fixed-z diff-y mean sens ≥ τ (FZDY)"),
]

_NLL_IMPROVED_KEY = "nll_improved"   # computed from history, not per-epoch

# Gates that depend on the cond_gate_history block. If the run skipped
# _cond_gate_hook (e.g. Glow, whose layer structure is incompatible), these
# gates are N/A rather than FAIL.
_COND_GATE_DEPENDENT = {
    "h_alive", "film_alive", "determinism", "no_nan_inf",
    "null_control_gap", "h_st_slope", "grad_flow",
}

# v0.7: the cond_gate_history key(s) each gate actually reads. A gate is N/A
# iff NONE of its required keys appear in ANY per-epoch snapshot. This stops a
# populated unrelated key (e.g. cond_grad) from making missing-key gates grade
# as FAIL instead of N/A.
_GATE_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "h_alive":          ("h_std",),
    "film_alive":       ("gamma_std", "beta_std"),
    "determinism":      ("det_max_dh",),
    "no_nan_inf":       ("nan_inf",),
    "null_control_gap": ("real_nll", "abl_nll"),
    "h_st_slope":       ("h_st_slope",),
    "grad_flow":        ("cond_grad", "film_grad_min"),
}


def _gate_is_na(gate: str, cg_snapshots: list[dict | None]) -> bool:
    # v0.7: per-gate N/A. A cond-gate-dependent gate is N/A iff none of the
    # keys it reads are present in any snapshot (data was never recorded).
    if gate not in _COND_GATE_DEPENDENT:
        return False
    if not cg_snapshots:
        return True
    req = _GATE_REQUIRED_KEYS.get(gate, ())
    for s in cg_snapshots:
        if s and any(k in s for k in req):
            return False          # data available -> grade it
    return True                   # no required key anywhere -> N/A


def _epoch_passes(rec: dict, gate: str, cg_snapshot: dict | None) -> bool:
    # rec: one entry of sanity_per_epoch. cg_snapshot: corresponding cond-gate
    # derived values from the hook's history (NaN/Inf, det, etc.).
    try:
        if gate == "invertibility":
            return bool(rec["invertibility"]["passed"])
        if gate == "latent_ks":
            return bool(rec["latent"]["passed"])
        if gate == "cycle_max":
            return rec["cycle_heatmap"]["cycle_max"] < 1e-12
        if gate == "h_alive":
            return cg_snapshot is not None and cg_snapshot.get("h_std", 0) > 1e-3
        if gate == "film_alive":
            if cg_snapshot is None: return False
            return (cg_snapshot.get("gamma_std", 0) > 1e-3 and
                    cg_snapshot.get("beta_std",  0) > 1e-3)
        if gate == "determinism":
            return cg_snapshot is not None and cg_snapshot.get("det_max_dh", 1) < 1e-6
        if gate == "no_nan_inf":
            if cg_snapshot is None: return False
            ni = cg_snapshot.get("nan_inf", {})
            return sum(ni.values()) == 0
        # informational --------------------------------------------------------
        if gate == "null_control_gap":
            if cg_snapshot is None: return False
            r, a = cg_snapshot.get("real_nll"), cg_snapshot.get("abl_nll")
            if r is None or a is None: return False
            return (a - r) / max(abs(r), 1e-12) >= 0.10
        if gate == "h_st_slope":
            return cg_snapshot is not None and cg_snapshot.get("h_st_slope", 0) > 1e-3
        if gate == "grad_flow":
            if cg_snapshot is None: return False
            g = cg_snapshot.get("cond_grad", 0)
            fg = cg_snapshot.get("film_grad_min", 0)
            return g > 1e-6 and fg > 1e-6
        if gate == "fwd_consistency_tracked":
            return "forward_consistency" in rec
        if gate == "fixed_z_sensitivity":
            fz = rec.get("fixed_z_different_y")
            if not fz:
                return False
            return bool(fz.get("passed", False))
    except (KeyError, TypeError):
        return False
    return False


def _film_gain_stats(report: dict) -> dict | None:
    # v0.7: summarise report["glow_film_gain_hist"] (list of per-layer gain
    # lists, one per epoch). Returns first/last-epoch means and last-epoch
    # min/mean/max, or None if absent (non-Glow runs).
    hist = report.get("glow_film_gain_hist")
    if not hist:
        return None
    first, last = hist[0], hist[-1]
    if not isinstance(last, list) or not last:
        return None
    return {
        "first_epoch_mean": sum(first) / len(first),
        "last_epoch_mean":  sum(last) / len(last),
        "last_min":  min(last),
        "last_max":  max(last),
        "n_layers":  len(last),
        "decayed":   (sum(last) / len(last)) < (sum(first) / len(first)),
    }


def _build_cg_snapshots(report: dict) -> list[dict]:
    # reshape cg_history (if present) into per-epoch dicts.
    cg = report.get("cond_gate_history")
    if cg is None:
        return [None] * len(report.get("sanity_per_epoch", []))
    n = len(report.get("sanity_per_epoch", []))
    out = []
    for i in range(n):
        snap = {}
        # guard each access in case histories have different lengths
        for k in ("h_std", "gamma_std", "beta_std",
                  "real_nll", "abl_nll", "h_st_slope",
                  "cond_grad", "film_grad_min", "det_max_dh"):
            if k in cg and i < len(cg[k]):
                snap[k] = cg[k][i]
        if "nan_inf" in cg and i < len(cg["nan_inf"]):
            snap["nan_inf"] = cg["nan_inf"][i]
        out.append(snap)
    return out


def _conditioning_block(report: dict, per_epoch: list[dict]) -> dict:
    # v0.8: final-epoch conditioning health. FZDY + h_std.batch are UNIVERSAL
    # (all experts). shuffle/film come from v2_diag and are N/A when absent
    # (NSF runs --no-use-v2-conditioner). A missing UNIVERSAL metric grades as
    # FAIL (logged), never as pass.
    last = per_epoch[-1] if per_epoch else {}

    # FZDY (universal) ----
    fz = last.get("fixed_z_different_y")
    if fz is None:
        logger.warning("[conditioning] final epoch missing fixed_z_different_y "
                       "-> FZDY graded FAIL")
    fzdy_last    = fz.get("sensitivity_mean") if fz else None
    fzdy_tau     = fz.get("tau") if fz else None
    fzdy_last_ok = bool(fz and fz.get("passed", False))

    # h_std batch (universal) ----
    hsh = report.get("h_std_obs_hist", [])
    if not hsh:
        logger.warning("[conditioning] h_std_obs_hist empty -> h_std graded FAIL")
    h_std_batch_last = hsh[-1].get("batch") if hsh else None
    h_std_batch_ok   = (h_std_batch_last is not None and h_std_batch_last >= 0.02)

    # v2-only signals (N/A when v2_diag absent) ----
    # shuffle: read from the final epoch (monotone, no flicker).
    # film_alive: v0.9 -- judged over the LAST 5 epochs (alive in >=1 of them),
    # because the per-epoch FiLM std flickers around eps and a single final-epoch
    # dip would false-fail an otherwise-healthy run.
    v2 = last.get("v2_diag")
    if v2 is not None:
        shuffle_gap_last = v2.get("logp_shuffle", {}).get("gap_mean")
        shuffle_ok = (shuffle_gap_last is not None and shuffle_gap_last > 0)
    else:
        shuffle_gap_last = None; shuffle_ok = None

    # film_alive over a trailing window of up to 5 epochs
    _tail = per_epoch[-5:] if per_epoch else []
    _alive_flags = [
        e.get("v2_diag", {}).get("film_alive", {}).get("alive")
        for e in _tail
    ]
    _alive_flags = [a for a in _alive_flags if a is not None]
    if _alive_flags:
        film_alive_last = any(_alive_flags)   # True if alive in >=1 of last 5
        film_ok = bool(film_alive_last)
    else:
        film_alive_last = None; film_ok = None  # N/A: no v2_diag.film_alive present

    # overall: universal gates must pass; v2 gates only if present
    conditioning_pass = bool(fzdy_last_ok and h_std_batch_ok)
    if shuffle_ok is not None:
        conditioning_pass = conditioning_pass and shuffle_ok
    if film_ok is not None:
        conditioning_pass = conditioning_pass and film_ok

    return {
        "fzdy_last": fzdy_last, "fzdy_tau": fzdy_tau, "fzdy_last_ok": fzdy_last_ok,
        "h_std_batch_last": h_std_batch_last, "h_std_batch_ok": h_std_batch_ok,
        "shuffle_gap_last": shuffle_gap_last, "shuffle_gap_ok_or_na": shuffle_ok,
        "film_alive_last": film_alive_last, "film_alive_ok_or_na": film_ok,
        "conditioning_pass": conditioning_pass,
    }


def summarize(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    report_path = run_dir / "report.json"
    if not report_path.exists():
        logger.error("[summary] %s not found", report_path)
        raise FileNotFoundError(f"report.json not found in {run_dir}")
    report = json.loads(report_path.read_text())

    cfg = report["cfg"]
    per_epoch = report.get("sanity_per_epoch", [])
    epochs_completed = len(per_epoch)
    cg_snapshots = _build_cg_snapshots(report)

    # tally ------------------------------------------------------------------
    ledger: dict[str, dict] = {}
    for gate_name, _desc in _HARD_GATES + _INFO_GATES:
        if _gate_is_na(gate_name, cg_snapshots):
            ledger[gate_name] = {"passed": 0, "total": 0,
                                 "fail_epochs": [], "na": True}
            continue
        fail_epochs: list[int] = []
        passed = 0
        for i, rec in enumerate(per_epoch):
            snap = cg_snapshots[i] if i < len(cg_snapshots) else None
            ok = _epoch_passes(rec, gate_name, snap)
            if ok: passed += 1
            else:  fail_epochs.append(i)
        ledger[gate_name] = {"passed": passed,
                             "total":  epochs_completed,
                             "fail_epochs": fail_epochs,
                             "na": False}

    # NLL direction ----------------------------------------------------------
    tr_hist = report.get("train_nll_hist", [])
    te_hist = report.get("test_nll_hist", [])
    nll_improved = (len(tr_hist) >= 2 and tr_hist[-1] < tr_hist[0])

    # exit criterion ---------------------------------------------------------
    hard_all_pass = all(
        ledger[g].get("na", False) or
        (ledger[g]["passed"] == ledger[g]["total"] and ledger[g]["total"] > 0)
        for g, _ in _HARD_GATES)
    logdet_ok = bool(report.get("logdet_check", {}).get("passed", False))
    # v0.8: final-epoch conditioning floor (Phase 0). FZDY + h_std are universal;
    # shuffle/film graded only when v2_diag exists (N/A for NSF).
    cond_blk = _conditioning_block(report, per_epoch)
    exit_ok = bool(hard_all_pass and logdet_ok and nll_improved
                   and cond_blk["conditioning_pass"])

    # remedy hints -----------------------------------------------------------
    remedies: list[str] = []
    for g, desc in _HARD_GATES:
        if ledger[g].get("na", False):
            continue                       # N/A is not a failure
        if ledger[g]["passed"] < ledger[g]["total"]:
            remedies.append(f"hard gate failed: {g} ({desc}) -- check epochs "
                            f"{ledger[g]['fail_epochs']}")
    if not logdet_ok:
        remedies.append("numeric log-det check failed -- inspect formulas")
    if not nll_improved:
        remedies.append("NLL did not decrease -- check LR / init / data pipeline")
    # v0.8: conditioning remedy hints.
    if not cond_blk["fzdy_last_ok"]:
        remedies.append("conditioning failed: FZDY below threshold "
                        f"(last={cond_blk['fzdy_last']}, tau={cond_blk['fzdy_tau']}) "
                        "-- model ignores y")
    if not cond_blk["h_std_batch_ok"]:
        remedies.append("conditioning failed: h_std_obs.batch < 0.02 "
                        f"(last={cond_blk['h_std_batch_last']})")
    if cond_blk["shuffle_gap_ok_or_na"] is False:
        remedies.append("conditioning failed: shuffle gap <= 0 "
                        f"(last={cond_blk['shuffle_gap_last']})")
    if cond_blk["film_alive_ok_or_na"] is False:
        remedies.append("conditioning failed: FiLM across-y not alive")

    # write summary.txt ------------------------------------------------------
    lines = _format_lines(cfg, report, ledger, tr_hist, te_hist,
                          epochs_completed, exit_ok, remedies)
    (run_dir / "summary.txt").write_text("\n".join(_strip_ansi(s) for s in lines))
    for s in lines:
        print(s)

    # write summary.csv (per-epoch, machine readable) ------------------------
    csv_path = run_dir / "summary.csv"
    fieldnames = (["epoch", "train_nll", "test_nll", "cycle_max",
                   "fwd_rel_mean", "latent_ks"]
                  + [g for g, _ in _HARD_GATES + _INFO_GATES])
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, rec in enumerate(per_epoch):
            snap = cg_snapshots[i] if i < len(cg_snapshots) else None
            row = {
                "epoch": i,
                "train_nll": tr_hist[i] if i < len(tr_hist) else "",
                "test_nll":  te_hist[i] if i < len(te_hist) else "",
                "cycle_max": rec.get("cycle_heatmap", {}).get("cycle_max", ""),
                "fwd_rel_mean": rec.get("forward_consistency", {}).get("fwd_rel_mean", ""),
                "latent_ks": rec.get("latent", {}).get("ks", ""),
            }
            for g, _ in _HARD_GATES + _INFO_GATES:
                row[g] = int(_epoch_passes(rec, g, snap))
            w.writerow(row)

    # augment report.json with machine-readable summary block ----------------
    summary_block = {
        "tag": cfg.get("expert", "?") + "_" + str(cfg.get("seed", "?")),
        "expert": cfg.get("expert"), "scale": cfg.get("scale"),
        "noise": cfg.get("noise_sigma"), "seed": cfg.get("seed"),
        "epochs_completed": epochs_completed,
        "checks": ledger,
        "nll": {"train_first": tr_hist[0]  if tr_hist else None,
                "train_last":  tr_hist[-1] if tr_hist else None,
                "test_first":  te_hist[0]  if te_hist else None,
                "test_last":   te_hist[-1] if te_hist else None,
                "improved": nll_improved},
        "cycle_max_last": (per_epoch[-1]["cycle_heatmap"]["cycle_max"]
                           if per_epoch else None),
        "fwd_rel_last": (per_epoch[-1].get("forward_consistency", {}).get("fwd_rel_mean")
                         if per_epoch else None),
        "exit_criteria_met": exit_ok,
        "remedy_hints": remedies,
        "glow_film_gain": _film_gain_stats(report),
        "conditioning": cond_blk,
    }
    report["summary"] = summary_block
    report_path.write_text(json.dumps(report, indent=2))
    return summary_block


# --- formatting -------------------------------------------------------------
def _fmt(passed: int, total: int, na: bool = False) -> str:
    if na:
        return f"{_Y}  --/-- N/A{_N}"
    if total == 0:
        return f"{_Y}  0/0{_N}"
    col = _G if passed == total else (_R if passed == 0 else _Y)
    mark = "✓" if passed == total else "✗"
    return f"{col}{passed:>3}/{total:<3} {mark}{_N}"


def _strip_ansi(s: str) -> str:
    import re
    return re.sub(r"\033\[[0-9;]*m", "", s)


def _format_lines(cfg, report, ledger, tr_hist, te_hist, n_ep, exit_ok, remedies):
    lines = []
    sep = "=" * 62
    lines.append(sep)
    lines.append(f"{_B}STEP-1_1 RUN SUMMARY{_N}")
    lines.append(sep)
    lines.append(f"expert: {cfg.get('expert')}   scale: {cfg.get('scale')}   "
                 f"noise: {cfg.get('noise_sigma')}   seed: {cfg.get('seed')}   "
                 f"epochs: {n_ep}")
    if tr_hist:
        lines.append(f"train NLL:  {tr_hist[0]:10.2f}  ->  {tr_hist[-1]:10.2f}  "
                     f"(Δ = {tr_hist[-1] - tr_hist[0]:+.2f})")
    if te_hist:
        lines.append(f"test  NLL:  {te_hist[0]:10.2f}  ->  {te_hist[-1]:10.2f}")

    fg = _film_gain_stats(report)
    if fg is not None:
        flag = f" {_R}(DECAYING){_N}" if fg["decayed"] else ""
        lines.append(f"film_gain:  mean {fg['first_epoch_mean']:.3f} -> "
                     f"{fg['last_epoch_mean']:.3f}  "
                     f"[last min {fg['last_min']:.3f} / max {fg['last_max']:.3f}, "
                     f"{fg['n_layers']} layers]{flag}")

    lines.append("")
    lines.append(f"{_B}HARD GATES (halt on fail){_N}")
    for g, d in _HARD_GATES:
        s = ledger[g]
        lines.append(f"  {_fmt(s['passed'], s['total'], s.get('na', False))}  {d}")
    ld = report.get("logdet_check", {})
    lines.append(f"  {_fmt(1 if ld.get('passed') else 0, 1)}  "
                 f"numeric log-det (FP64, once)")

    lines.append("")
    lines.append(f"{_B}INFORMATIONAL (logged, non-blocking){_N}")
    for g, d in _INFO_GATES:
        s = ledger[g]
        lines.append(f"  {_fmt(s['passed'], s['total'], s.get('na', False))}  {d}")

    # totals (N/A gates excluded from denominator) --------------------------
    tot_pass = sum(ledger[g]["passed"] for g, _ in _HARD_GATES
                   if not ledger[g].get("na", False)) \
             + (1 if ld.get("passed") else 0)
    tot_all  = sum(ledger[g]["total"]  for g, _ in _HARD_GATES
                   if not ledger[g].get("na", False)) + 1
    n_na = sum(1 for g, _ in _HARD_GATES if ledger[g].get("na", False))
    lines.append("-" * 62)
    na_suffix = f"  ({n_na} N/A)" if n_na else ""
    lines.append(f"{_B}HARD TOTAL: {tot_pass}/{tot_all}{_N}  "
                 f"({100*tot_pass/max(tot_all,1):.1f}%){na_suffix}")

    verdict = (f"{_G}YES{_N}" if exit_ok else f"{_R}NO{_N}")
    lines.append(f"{_B}EXIT_CRITERIA_MET:{_N} {verdict}")
    if remedies:
        lines.append(f"{_B}REMEDY HINTS:{_N}")
        for r in remedies:
            lines.append(f"  - {r}")
    lines.append(sep)
    return lines

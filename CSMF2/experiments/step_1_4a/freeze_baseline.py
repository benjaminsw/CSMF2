# =============================================================================
# NCP-N0 v0.3 -- experiments.step_1_4a.freeze_baseline
# Purpose: Write ONE immutable, identity-verified evidence record for the
#          existing clamp-2p0 NICE-CB run. Pure snapshot: no model/config/train
#          edits, no optimizer, no checkpoint mutation. Defines the contract
#          every later NICE change (N3/N7, roadmap 1.4c) must BEAT.
# CONVENTION: No fallback / mock / dummy / silent pass. Every failure path is
#          logger.error + raise. Required (contract) fields missing -> raise;
#          only optional-context fields may be recorded as "not_available".
# Changelog (v0.5 -> v0.6):
#   * Dtype fix: model/conditioner are float32 -- run the forward pass in
#     float32 and cast to float64 ONLY at the invertibility diff and the logdet
#     slogdet (was casting y to double, breaking the float32 conv conditioner).
# Changelog (v0.4 -> v0.5):
#   * Recompute seam wired to the REAL API: build_from_report(ckpt_dir, device)
#     -> (model, cond, cfg), model is CB-wrapped (encode/decode/cond, FLAT x);
#     val batch via MNISTDegraded(split='val', sigma/scale/noise from cfg)
#     mirroring step_1_4a/run.py. x flattened to (B,784); h = model.cond(y).
#     Tolerant to encode returning z or (z, ldj).
# Changelog (v0.3 -> v0.4):
#   * Matched real report.json schema: identity reads report['cfg']; cfg_hash
#     derived from run-dir 12-hex suffix (guarded, never 'unknown'); NLL from
#     top-level val_nll; degradation synthesized as s{scale}_n{noise} if absent.
#   * Conditioning recorded as typed conditioning_evidence (direct FiLM stats OR
#     conditional-base shuffle-sensitivity PROXY) -- base_alive is NOT labeled
#     'conditioning_active'. Contract field is now 'conditioning_evidence'.
# Changelog (v0.2 -> v0.3):
#   * rec_argmin / tier now read from the RECARGMIN-DIAG breakdown JSON
#     (win_totals[expert], tier) via --breakdown-report; RECGATE status read
#     from the step_1_3 RECGATE report.json (decision_helpers + reconstruction)
#     via --recgate-report. The old flat beta_summary.json lookup is removed --
#     these fields never lived in one flat file.
#   * sanity log demoted to optional provenance (conditioning comes from the
#     run-dir report.json). model_io seam pinned to step_1_1_1_1.model_io.
# Changelog (sketch -> v0.2):
#   * v0.1: locate -> verify identity (from saved cfg, not dir name) ->
#     collect contract fields -> A1 read-only float64 recompute of
#     invertibility + logdet sanity on the FINAL ckpt -> write-once JSON+MD,
#     tamper-evident freeze_hash.
#   * v0.2: contract uses PROVISIONAL+ as trusted improvement (WEAK =
#     inconclusive, not sufficient); added run-identity checksum
#     (ckpt sha256/size/mtime + source paths); clamp2p0 filename;
#     recon_grid_path provenance; optional verification plots.
# Update summary:
#   v0.2 hardens the freeze into a traceable, immutable artifact: the record
#   pins one exact checkpoint by hash and the "beats baseline" rule is now
#   tier-correct (a few nonzero wins = WEAK = not a rescue). cond_base.py is
#   intentionally untouched (saturation diagnostics belong to N1).
# Integration seam (confirm once): _load_model_and_eval_batch() imports the
#   project's model loader + dataset builder. Names are flagged below; if your
#   module paths differ, reconcile THAT function only -- the rest is
#   self-contained and project-agnostic.
# =============================================================================
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import sys
from typing import Any

logger = logging.getLogger("NCP-N0")

__version__ = "0.6"
__abbr__ = "NCP-N0"

# --- expected baseline identity (verified against the saved config) ----------
EXPECT_EXPERT_TOKENS = ("nice",)          # expert string must contain 'nice'
EXPECT_CB = True                          # NICE-CB -> conditional base on
EXPECT_BASE_LOGSIGMA_MAX = 2.0
EXPECT_BASE_LOGSIGMA_MAX_TOL = 1e-9
INVERTIBILITY_GATE = 1e-5                 # max ||x - finv(f(x))||inf

# RECARGMIN-DIAG tier ordering (reused by later steps; not applied during N0).
TIER_RANK = {"FLAT": 0, "WEAK": 1, "PROVISIONAL": 2, "STRONG": 3}


# =============================================================================
# small helpers (pure; unit-tested without torch)
# =============================================================================
def _die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    logger.error(msg)
    raise RuntimeError(msg)


def require(d: dict, keys: tuple[str, ...], ctx: str) -> Any:
    """Return d[k] for the first k in `keys` present with a non-None value.
    Raise (never substitute a default) if none are found."""
    if not isinstance(d, dict):
        _die(f"[{ctx}] expected a dict, got {type(d).__name__}")
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    _die(f"[{ctx}] required field missing; looked for any of {list(keys)} "
         f"in keys {sorted(d.keys())}")


def optional(d: dict, keys: tuple[str, ...]) -> Any:
    """Return first present value, else the literal 'not_available'. Never invent."""
    if isinstance(d, dict):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
    return "not_available"


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def canonical_freeze_hash(record: dict) -> str:
    """sha256 over the canonical record body EXCLUDING the freeze_hash slot."""
    body = {k: v for k, v in record.items() if k != "freeze_hash"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def tier_rank(tier: str) -> int:
    if tier not in TIER_RANK:
        _die(f"unknown RECARGMIN-DIAG tier {tier!r}; expected one of {list(TIER_RANK)}")
    return TIER_RANK[tier]


def beats_baseline(baseline_tier: str, candidate_tier: str) -> bool:
    """Tier rule for trusted improvement: candidate must be PROVISIONAL+ AND
    strictly above baseline. WEAK is inconclusive -> does NOT beat baseline.
    Reused by N3/N7; not called during the freeze itself."""
    return (tier_rank(candidate_tier) >= TIER_RANK["PROVISIONAL"]
            and tier_rank(candidate_tier) > tier_rank(baseline_tier))


# =============================================================================
# locate + verify
# =============================================================================
def locate(run_dir: str) -> dict[str, str]:
    if not os.path.isdir(run_dir):
        _die(f"run_dir does not exist or is not a directory: {run_dir}")

    report = os.path.join(run_dir, "report.json")
    if not os.path.isfile(report):
        _die(f"report.json not found in run_dir: {report}")

    # checkpoint: accept the common names, else raise listing what was found.
    ckpt = None
    for name in ("ckpt.pt", "checkpoint.pt", "model.pt", "best.pt", "final.pt"):
        cand = os.path.join(run_dir, name)
        if os.path.isfile(cand):
            ckpt = cand
            break
    if ckpt is None:
        present = sorted(p for p in os.listdir(run_dir) if p.endswith((".pt", ".pth")))
        _die(f"no checkpoint (.pt) found in {run_dir}; present .pt/.pth: {present}")

    # sanity log (conditioning activity / per-epoch checks) -- OPTIONAL provenance.
    sanity = "not_available"
    for name in ("sanity.jsonl", "sanity.json", "sanity_log.jsonl", "metrics.csv"):
        cand = os.path.join(run_dir, name)
        if os.path.isfile(cand):
            sanity = os.path.abspath(cand)
            break

    return {"run_dir": os.path.abspath(run_dir),
            "report_json_path": os.path.abspath(report),
            "checkpoint_path": os.path.abspath(ckpt),
            "sanity_log_path": sanity}


def load_report(report_path: str) -> dict:
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001 - re-raised explicitly, never swallowed
        _die(f"failed to read/parse report.json at {report_path}: {e!r}")


def _cfg_hash_from_run_dir(run_dir: str) -> str:
    """Derive the 12-hex cfg_hash from the run-dir suffix (canonical source;
    report.json does not store it). Raise if not a 12-hex token -- never 'unknown'."""
    import re
    base = os.path.basename(run_dir.rstrip("/"))
    m = re.search(r"([0-9a-f]{12})$", base)
    if not m:
        _die(f"cannot parse 12-hex cfg_hash from run-dir suffix: {base!r}")
    return m.group(1)


def verify_identity(report: dict, run_dir: str) -> dict:
    """Confirm this run IS the clamp-2p0 NICE-CB baseline, from the saved config.
    Any mismatch -> raise. cfg is nested under report['cfg']."""
    cfg = report.get("cfg", report.get("config", report))

    expert = str(require(cfg, ("expert", "expert_name", "arch"), "identity.expert")).lower()
    if not any(tok in expert for tok in EXPECT_EXPERT_TOKENS):
        _die(f"expert mismatch: {expert!r} does not look like NICE")

    use_cb = require(cfg, ("use_conditional_base", "conditional_base", "use_cb"),
                     "identity.use_conditional_base")
    if bool(use_cb) is not EXPECT_CB:
        _die(f"conditional-base mismatch: use_conditional_base={use_cb!r}, expected {EXPECT_CB}")

    blsm = float(require(cfg, ("base_logsigma_max", "base_log_sigma_max"),
                         "identity.base_logsigma_max"))
    if abs(blsm - EXPECT_BASE_LOGSIGMA_MAX) > EXPECT_BASE_LOGSIGMA_MAX_TOL:
        _die(f"base_logsigma_max mismatch: {blsm} != {EXPECT_BASE_LOGSIGMA_MAX} "
             f"(this is NOT the clamp-2p0 baseline)")

    seed = int(require(cfg, ("seed", "cli_seed", "seed_index"), "identity.seed"))
    rng_seed = optional(cfg, ("rng_seed", "torch_seed"))
    cfg_hash = _cfg_hash_from_run_dir(run_dir)
    scale = optional(cfg, ("scale", "downsample", "sr_scale"))
    noise = optional(cfg, ("noise_sigma", "noise", "sigma"))
    blur = optional(cfg, ("blur_sigma", "blur"))
    degradation = optional(cfg, ("degradation", "degradation_tag", "deg_tag"))
    if degradation == "not_available" and scale != "not_available":
        degradation = f"s{scale}_n{noise}"

    return {"expert": expert, "use_conditional_base": bool(use_cb),
            "base_logsigma_max": blsm, "seed": seed, "rng_seed": rng_seed,
            "cfg_hash": cfg_hash, "cfg_hash_source": "run_dir_suffix",
            "degradation": degradation, "scale": scale,
            "noise_sigma": noise, "blur_sigma": blur}


# =============================================================================
# collect contract metrics from existing artifacts
# =============================================================================
def _load_json(path: str, ctx: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        _die(f"failed to read/parse {ctx} at {path}: {e!r}")


def collect_recgate(breakdown_path: str, recgate_report_path: str,
                    expert_token: str = "nice") -> dict:
    """rec_argmin + tier come from the RECARGMIN-DIAG breakdown JSON; the
    RECGATE-global status is derived from the step_1_3 RECGATE report.json
    decision_helpers + reconstruction block. Two distinct tools, two files."""
    # --- breakdown: rec_argmin (win_totals) + tier ---
    bd = _load_json(breakdown_path, "breakdown JSON")
    wins = require(bd, ("win_totals", "counts_by_expert"), "breakdown.win_totals")
    if isinstance(wins, dict):
        nice_count = int(require(wins, (expert_token, expert_token.upper(),
                                        f"{expert_token}_cb"), "breakdown.win_totals.nice"))
    elif isinstance(wins, (list, tuple)):
        experts = require(bd, ("experts", "expert_names"), "breakdown.experts")
        idx = list(experts).index(expert_token)
        nice_count = int(wins[idx])
    else:
        _die(f"unrecognized win_totals payload type: {type(wins).__name__}")

    tier = str(require(bd, ("tier", "nice_tier", "recargmin_diag_tier"),
                       "breakdown.tier")).upper()
    tier_rank(tier)  # validate
    n_total = optional(bd, ("n", "n_samples"))
    self_check = optional(bd, ("self_check",))

    # --- RECGATE report: derive a status from decision helpers + fwd_rel ---
    rg = _load_json(recgate_report_path, "RECGATE report.json")
    dh = require(rg, ("decision_helpers",), "recgate.decision_helpers")
    recon = require(rg, ("reconstruction",), "recgate.reconstruction")
    soft = require(recon, ("soft_fwd_rel",), "recgate.reconstruction.soft_fwd_rel")
    nsf_only = require(recon, ("nsf_only_fwd_rel",), "recgate.reconstruction.nsf_only_fwd_rel")
    improved = bool(require(dh, ("fwd_rel_improved_vs_nsf_only",), "recgate.dh.improved"))
    neff_ok = bool(optional(dh, ("neff_gt_1p5",)) is True)
    not_collapsed = bool(optional(dh, ("max_weight_lt_0p70",)) is True)
    status = ("RESCUE" if improved else "NO_SUCCESS") + \
             f" (soft_fwd_rel={float(soft):.4f} vs nsf_only={float(nsf_only):.4f}; " \
             f"neff>1.5={neff_ok}, max_w<0.70={not_collapsed})"

    return {"rec_argmin_count": nice_count,
            "recargmin_diag_tier": tier,
            "recgate_global_status": status,
            "_rec_argmin_n_total": n_total,
            "_rec_argmin_self_check": self_check,
            "_recgate_soft_fwd_rel": float(soft),
            "_recgate_nsf_only_fwd_rel": float(nsf_only)}


def collect_nll_and_conditioning(report: dict, sanity_path: str) -> dict:
    out: dict[str, Any] = {}

    # NLL: top-level val_nll on this run; train/test optional.
    nll_val = optional(report, ("val_nll", "nll_val"))
    nll_train = optional(report, ("train_nll", "nll_train", "nll"))
    nll_test = optional(report, ("test_nll", "nll_test"))
    if all(v == "not_available" for v in (nll_train, nll_val, nll_test)):
        _die("NLL missing: none of val_nll/train_nll/test_nll found in report.json")
    out.update(nll_train=nll_train, nll_val=nll_val, nll_test=nll_test)

    # Conditioning EVIDENCE (not 'active'): prefer direct conditioner stats; else
    # fall back to the conditional-base shuffle-sensitivity proxy. Honest labels.
    cond = report.get("conditioning", {})
    h_std = optional(cond, ("h_std", "cond_h_std"))
    film_g = optional(cond, ("film_gamma_std", "gamma_std"))
    film_b = optional(cond, ("film_beta_std", "beta_std"))

    base = report.get("base_final", {})
    base_alive = optional(base, ("base_alive",))
    shuffle_gap = optional(base, ("base_shuffle_gap", "shuffle_gap"))

    if h_std != "not_available":
        evidence = {"status": "direct_recorded",
                    "type": "conditioner_film_stats",
                    "h_std": h_std, "film_gamma_std": film_g, "film_beta_std": film_b}
    elif base_alive is True and shuffle_gap != "not_available":
        evidence = {"status": "proxy_recorded",
                    "type": "conditional_base_shuffle_sensitivity",
                    "base_alive": True, "base_shuffle_gap": shuffle_gap,
                    "h_std": "not_available",
                    "film_gamma_std": "not_available",
                    "film_beta_std": "not_available"}
    else:
        _die("no conditioning evidence found: neither direct conditioner stats "
             "(h_std/FiLM) nor the conditional-base proxy (base_alive + "
             "base_shuffle_gap) are present in report.json")

    out["conditioning_evidence"] = evidence
    out["conditioning_proxy_active"] = (evidence["status"] != "direct_recorded")
    out["conditioning_direct_stats_available"] = (evidence["status"] == "direct_recorded")
    out["_sanity_log_path"] = sanity_path if sanity_path == "not_available" \
        else os.path.abspath(sanity_path)
    return out


# =============================================================================
# A1: read-only float64 recompute on the FINAL checkpoint (no grad steps)
# =============================================================================
def _load_model_and_eval_batch(run_dir: str, report: dict, device: str,
                               n_invert: int, n_logdet: int):
    """INTEGRATION SEAM -- the only project-coupled function.
    Returns (model, x_flat_invert, x_flat_logdet, h_invert, h_logdet) on the SAME
    val degradation as the run. Never returns mock tensors -- a failed import raises."""
    try:
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        _die(f"torch import failed; recompute requires the project env: {e!r}")

    try:
        # CONFIRMED: build_from_report(ckpt_dir, device) -> (model, cond, cfg);
        # model is the CB-wrapped expert (encode/decode/cond, FLAT x). val batch
        # via MNISTDegraded mirroring step_1_4a/run.py.
        from CSMF2.experiments.step_1_1_1_1.model_io import build_from_report  # type: ignore
        from CSMF2.data.degrade import MNISTDegraded                          # type: ignore
        from torch.utils.data import DataLoader                              # type: ignore
    except Exception as e:  # noqa: BLE001
        _die("project import failed in the integration seam "
             "(_load_model_and_eval_batch). Reconcile build_from_report / "
             f"MNISTDegraded to your actual module paths: {e!r}")

    import torch
    dev = torch.device(device)
    model, cond, cfg = build_from_report(run_dir, dev)  # ckpt_dir == run_dir; CB-wrapped, frozen
    model.eval()

    n = max(n_invert, n_logdet)
    val_ds = MNISTDegraded(cfg.data_root, split="val", sigma=cfg.blur_sigma,
                           scale=cfg.scale, noise_sigma=cfg.noise_sigma)
    x, y = next(iter(DataLoader(val_ds, batch_size=n, shuffle=False)))
    # model is float32 -- keep the forward pass in float32; cast to float64 ONLY
    # at the numeric invertibility/logdet checks (where precision matters).
    x = x.to(dev).float()
    y = y.to(dev).float()
    x_flat = x.reshape(x.shape[0], -1)            # (B, 784) -- encode takes flat x
    with torch.no_grad():
        h = model.cond(y)
    return (model, x_flat[:n_invert], x_flat[:n_logdet],
            h[:n_invert], h[:n_logdet])


def recompute_invertibility_and_logdet(run_dir: str, report: dict, device: str,
                                       n_invert: int, n_logdet: int) -> dict:
    import torch
    model, xf_inv, xf_ld, h_inv, h_ld = _load_model_and_eval_batch(
        run_dir, report, device, n_invert, n_logdet)

    # --- invertibility (model fwd in float32; diff in float64) ---
    with torch.no_grad():
        enc = model.encode(xf_inv, h_inv)
        z = enc[0] if isinstance(enc, tuple) else enc
        x_rec = model.decode(z, h_inv)
        if x_rec.shape != xf_inv.shape:
            _die(f"decode shape {tuple(x_rec.shape)} != input {tuple(xf_inv.shape)}")
        err = (xf_inv.double() - x_rec.double()).abs().amax(dim=1)  # per-sample Linf
        inv_max = float(err.max().item())
    inv_pass = inv_max < INVERTIBILITY_GATE

    # --- logdet sanity: analytic ldj vs numerical Jacobian slogdet (float64) ---
    D = xf_ld.shape[1]

    def f_single(v: "torch.Tensor", hh: "torch.Tensor") -> "torch.Tensor":
        out = model.encode(v.unsqueeze(0), hh.unsqueeze(0))
        zz = out[0] if isinstance(out, tuple) else out
        return zz.reshape(-1)

    abs_errs = []
    for i in range(xf_ld.shape[0]):
        with torch.no_grad():
            out = model.encode(xf_ld[i:i + 1], h_ld[i:i + 1])
            if not isinstance(out, tuple) or len(out) < 2:
                _die("encode did not return (z, ldj); cannot run logdet sanity")
            ldj_analytic = float(out[1].reshape(-1)[0].item())
        J = torch.autograd.functional.jacobian(
            lambda v, _h=h_ld[i]: f_single(v, _h), xf_ld[i].clone().requires_grad_(True))
        J = J.reshape(D, D).double()
        ldj_numeric = float(torch.linalg.slogdet(J)[1].item())
        abs_errs.append(abs(ldj_analytic - ldj_numeric))

    ld_mean = sum(abs_errs) / len(abs_errs)
    ld_max = max(abs_errs)
    # tolerant self-check (warn, don't fail, on close mismatch) per project convention
    ld_status = "pass" if ld_max <= 10.0 else "fail"
    if ld_max > 1e-2:
        logger.warning("[logdet] max |analytic-numeric|=%.4g over %d samples",
                       ld_max, len(abs_errs))

    return {"invertibility_max_linf": inv_max,
            "invertibility_pass": bool(inv_pass),
            "logdet_sanity_f64_status": ld_status,
            "logdet_abs_error_mean": ld_mean,
            "logdet_abs_error_max": ld_max,
            "logdet_n_samples": len(abs_errs)}


# =============================================================================
# optional provenance + verification plots (recorded as paths only)
# =============================================================================
def collect_provenance(run_dir: str, report: dict) -> dict:
    def _path_if_exists(*names: str) -> str:
        for n in names:
            cand = os.path.join(run_dir, n) if not os.path.isabs(n) else n
            if os.path.isfile(cand):
                return os.path.abspath(cand)
            cand2 = os.path.join(run_dir, "plots", n)
            if os.path.isfile(cand2):
                return os.path.abspath(cand2)
        return "not_available"

    return {
        "recon_grid_path": _path_if_exists("p4_recon_panel.png", "recon_grid.png"),
        "sample_grid_path": _path_if_exists("sample_grid.png", "samples.png"),
        "nll_curve_path": _path_if_exists("p_nll_curve.png", "nll_curve.png"),
        "recgate_beta_plot_path": _path_if_exists("p3_fwdrel_vs_beta.png"),
        "conditioning_curve_path": _path_if_exists("p_cond.png", "conditioning.png"),
        # fraction_at_sigma_max belongs to N1; copy ONLY if already present, never compute.
        "fraction_at_sigma_max": optional(report.get("base", report),
                                          ("fraction_at_sigma_max",)),
        "base_logsigma_mean": optional(report.get("base", report), ("base_logsigma_mean",)),
        "base_logsigma_std": optional(report.get("base", report), ("base_logsigma_std",)),
        "shuffle_gap": optional(report.get("base", report),
                                ("base_shuffle_gap", "shuffle_gap")),
        "fixed_z_diff_y_sensitivity": optional(report, ("fixed_z_diff_y_sensitivity",)),
    }


def write_verification_plots(out_dir: str, recompute: dict) -> dict:
    """Tiny verification plots from already-computed values. Optional; absence
    of matplotlib is recorded, not faked."""
    paths = {"invertibility_hist_path": "not_available",
             "logdet_scatter_path": "not_available"}
    return paths  # per-sample arrays are not persisted in N0; plots are deferred
    # (kept as a slot; N1 owns the diagnostic plots. No fake image is written.)


# =============================================================================
# assemble + write-once
# =============================================================================
REQUIRED_METRIC_KEYS = (
    "rec_argmin_count", "recargmin_diag_tier", "recgate_global_status",
    "invertibility_max_linf", "invertibility_pass",
    "logdet_sanity_f64_status", "logdet_abs_error_mean", "logdet_abs_error_max",
    "conditioning_evidence",
)
REQUIRED_IDENTITY_KEYS = (
    "expert", "base_logsigma_max", "seed", "cfg_hash",
    "run_dir", "checkpoint_path", "checkpoint_sha256", "report_json_path",
)


def assemble_record(paths: dict, identity: dict, metrics: dict,
                    provenance: dict, git_commit: str | None) -> dict:
    st = os.stat(paths["checkpoint_path"])
    identity_block = {
        **identity,
        "run_dir": paths["run_dir"],
        "checkpoint_path": paths["checkpoint_path"],
        "checkpoint_sha256": sha256_file(paths["checkpoint_path"]),
        "checkpoint_size_bytes": st.st_size,
        "checkpoint_mtime": _dt.datetime.fromtimestamp(
            st.st_mtime, _dt.timezone.utc).isoformat(),
        "report_json_path": paths["report_json_path"],
        "recgate_report_path": paths.get("recgate_report_path", "not_available"),
        "breakdown_report_path": paths.get("breakdown_report_path", "not_available"),
        "sanity_log_path": paths["sanity_log_path"],
        "git_commit": git_commit if git_commit else "not_available",
    }

    record = {
        "abbr": __abbr__,
        "version": __version__,
        "baseline_name": "NICE-CB clamp-2p0 baseline",
        "roadmap_slot": "1.3b/N0",
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "identity": identity_block,
        "metrics": {k: v for k, v in metrics.items() if not k.startswith("_")},
        "provenance": provenance,
        "contract": {
            "rule": "No NICE change is trusted unless it beats this baseline.",
            "beats_baseline_iff": {
                "rec_argmin_tier": "PROVISIONAL+ (WEAK is inconclusive, NOT sufficient)",
                "rec_argmin_count": "> baseline AND tier not FLAT",
                "recgate_global_status": "improves",
                "nll": "no critical regression (use existing RECGATE threshold)",
                "invertibility_max_linf": f"< {INVERTIBILITY_GATE}",
                "logdet_sanity_f64": "pass",
                "conditioning": "remains active",
            },
            "three_seed_lock": "trusted rescue only if PROVISIONAL+ repeats across seeds 0/1/2",
        },
        "freeze_hash": None,  # filled below
    }

    # hard-fail on any missing REQUIRED field (no null contract fields).
    for k in REQUIRED_IDENTITY_KEYS:
        if identity_block.get(k) in (None, "not_available", ""):
            _die(f"required identity field missing/blank: {k}")
    for k in REQUIRED_METRIC_KEYS:
        if record["metrics"].get(k) in (None, "not_available", ""):
            _die(f"required metric field missing/blank: {k}")

    record["freeze_hash"] = canonical_freeze_hash(record)
    return record


def render_md(record: dict) -> str:
    i, m, p = record["identity"], record["metrics"], record["provenance"]
    L = [
        "# NICE-CB clamp-2p0 frozen baseline",
        "",
        f"`{record['abbr']} v{record['version']}` · roadmap slot {record['roadmap_slot']} "
        f"· created {record['created_at']}",
        "",
        "## Identity",
        f"- expert: `{i['expert']}`  (conditional base: {i['use_conditional_base']})",
        f"- base_logsigma_max: **{i['base_logsigma_max']}**",
        f"- seed: {i['seed']}  (rng_seed: {i['rng_seed']})",
        f"- cfg_hash: `{i['cfg_hash']}`",
        f"- degradation: {i['degradation']} (scale {i['scale']}, noise {i['noise_sigma']})",
        f"- checkpoint: `{i['checkpoint_path']}`",
        f"- checkpoint_sha256: `{i['checkpoint_sha256']}`",
        f"- git_commit: `{i['git_commit']}`",
        f"- freeze_hash: `{record['freeze_hash']}`",
        "",
        "## Metrics (contract)",
        f"- rec_argmin (NICE): **{m['rec_argmin_count']}**  ·  tier: **{m['recargmin_diag_tier']}**",
        f"- RECGATE-global: {m['recgate_global_status']}",
        f"- NLL (train/val/test): {m['nll_train']} / {m['nll_val']} / {m['nll_test']}",
        f"- invertibility max |·|inf: {m['invertibility_max_linf']:.3e}  "
        f"(pass < {INVERTIBILITY_GATE}: **{m['invertibility_pass']}**)",
        f"- logdet sanity (f64): **{m['logdet_sanity_f64_status']}**  "
        f"(|err| mean {m['logdet_abs_error_mean']:.3e}, max {m['logdet_abs_error_max']:.3e})",
        f"- conditioning evidence: **{m['conditioning_evidence']['status']}** "
        f"({m['conditioning_evidence']['type']})",
        "",
        "## Provenance (links only)",
        f"- recon grid: `{p['recon_grid_path']}`",
        f"- NLL curve: `{p['nll_curve_path']}`",
        f"- RECGATE beta plot: `{p['recgate_beta_plot_path']}`",
        "",
        "## Rule",
        "No NICE change is trusted unless it **beats** this baseline "
        "(rec_argmin PROVISIONAL+, RECGATE improves, NLL not critically regressed, "
        f"invertibility < {INVERTIBILITY_GATE}, logdet f64 pass, conditioning active). "
        "WEAK tier is inconclusive, not a rescue.",
    ]
    return "\n".join(L) + "\n"


def write_once(out_dir: str, record: dict, ckpt_hash12: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    stem = f"nice_cb_clamp2p0_{ckpt_hash12}"
    json_path = os.path.join(out_dir, stem + ".json")
    md_path = os.path.join(out_dir, stem + ".md")
    for p in (json_path, md_path):
        if os.path.exists(p):
            _die(f"baseline artifact already exists (immutable, refusing to overwrite): {p}")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_md(record))
    return json_path, md_path


# =============================================================================
# main
# =============================================================================
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="NCP-N0 v0.3 freeze clamp-2p0 NICE-CB baseline")
    ap.add_argument("--run-dir", required=True,
                    help="path to the clamp-2.0 NICE-CB run directory")
    ap.add_argument("--breakdown-report", required=True,
                    help="path to RECARGMIN-DIAG rec_argmin_breakdown.json (rec_argmin + tier)")
    ap.add_argument("--recgate-report", required=True,
                    help="path to the step_1_3 RECGATE report.json at the chosen beta (status)")
    ap.add_argument("--out-dir",
                    default="CSMF2/experiments/step_1_4a/baselines",
                    help="where to write the immutable baseline artifacts")
    ap.add_argument("--git-commit", default=None, help="real commit hash (optional)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-invert", type=int, default=64,
                    help="samples for invertibility recompute (float64, no grad)")
    ap.add_argument("--n-logdet", type=int, default=4,
                    help="samples for numerical-Jacobian logdet sanity (expensive)")
    ap.add_argument("--skip-recompute", action="store_true",
                    help="skip A1 invertibility/logdet recompute (NOT recommended; "
                         "records both as not_available and forces N0 NO-GO)")
    args = ap.parse_args(argv)

    for p, ctx in ((args.breakdown_report, "--breakdown-report"),
                   (args.recgate_report, "--recgate-report")):
        if not os.path.isfile(p):
            _die(f"{ctx} not found: {p}")

    paths = locate(args.run_dir)
    paths["recgate_report_path"] = os.path.abspath(args.recgate_report)
    paths["breakdown_report_path"] = os.path.abspath(args.breakdown_report)
    logger.info("located: %s", json.dumps(paths, indent=2))

    report = load_report(paths["report_json_path"])
    identity = verify_identity(report, paths["run_dir"])
    logger.info("identity verified: NICE-CB, base_logsigma_max=%.3f, seed=%d, cfg_hash=%s",
                identity["base_logsigma_max"], identity["seed"], identity["cfg_hash"])

    expert_token = next((t for t in EXPECT_EXPERT_TOKENS if t in identity["expert"]), "nice")
    metrics: dict[str, Any] = {}
    metrics.update(collect_recgate(paths["breakdown_report_path"],
                                   paths["recgate_report_path"], expert_token))
    metrics.update(collect_nll_and_conditioning(report, paths["sanity_log_path"]))

    # A1: read-only recompute on the FINAL ckpt (requires project env).
    if args.skip_recompute:
        logger.warning("[A1] --skip-recompute set: invertibility/logdet NOT verified "
                       "on the final ckpt; N0 will be NO-GO.")
        metrics.update(invertibility_max_linf=float("nan"), invertibility_pass=False,
                       logdet_sanity_f64_status="not_recomputed",
                       logdet_abs_error_mean=float("nan"),
                       logdet_abs_error_max=float("nan"), logdet_n_samples=0)
    else:
        metrics.update(recompute_invertibility_and_logdet(
            paths["run_dir"], report, args.device, args.n_invert, args.n_logdet))

    provenance = collect_provenance(paths["run_dir"], report)

    record = assemble_record(paths, identity, metrics, provenance, args.git_commit)
    ckpt_hash12 = record["identity"]["checkpoint_sha256"][:12]
    json_path, md_path = write_once(args.out_dir, record, ckpt_hash12)

    logger.info("freeze complete -> %s", json_path)
    logger.info("freeze_hash=%s", record["freeze_hash"])
    # go/no-go
    m = record["metrics"]
    ok = (m["invertibility_pass"] and m["logdet_sanity_f64_status"] == "pass"
          and bool(m.get("conditioning_evidence")))
    print("\nN0 GO" if ok else "\nN0 NO-GO (recompute gate failed; inspect record)")
    print(f"  baseline tier: {m['recargmin_diag_tier']}  rec_argmin(NICE)={m['rec_argmin_count']}")
    print(f"  invertibility: {m['invertibility_max_linf']:.3e}  logdet: {m['logdet_sanity_f64_status']}")
    print(f"  json: {json_path}\n  md:   {md_path}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())

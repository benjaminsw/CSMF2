# =============================================================================
# NCP-N2 v0.3 -- experiments.step_1_4a.clamp_probe
# Purpose: FREE clamp-binding pre-check. On the EXISTING (frozen) clamp-2.0
#          NICE-CB checkpoint, compute fraction_at_sigma_max via the N1
#          base_diagnostics and decide whether the logsigma clamp is BINDING.
#          One forward pass, read-only: no training, no optimizer, no
#          checkpoint mutation. This is the GPU-saving gate for N3-N7.
# Decision (the verdict lives HERE, not in cond_base):
#     fraction_at_sigma_max <= gate_frac -> NOT_BINDING -> COUPLING-limited
#         -> STOP clamp branch (skip N3-N7); one-shot N8 or Stage 2.3.
#     fraction_at_sigma_max  > gate_frac -> BINDING (may bind)
#         -> continue to N3 (retrain NICE-CB clamp 3.0, seed0).
# IMPORTANT: decide on fraction_at_sigma_max, NOT on sigma_max alone -- a few
#          elements touching the ceiling is not a globally clamp-limited base.
# CONVENTION: No fallback / mock / dummy / silent pass. Every failure path is
#          logger.error + raise. The JSON record is re-runnable (NOT write-once
#          / immutable -- unlike the N0 baseline).
# Changelog (v0.2 -> v0.3):
#   * Clamp expectation is now a CLI parameter (--expect-logsigma-max, default
#     2.0) instead of a hardcoded 2.0. v0.2 hard-asserted clamp==2.0, so re-using
#     the probe at N5 on the clamp-3.0 ckpt raised. Now the same probe works at
#     any rung (2.0 baseline / 3.0 N3 / 4.0 N4). The BINDING next_step text is
#     also clamp-aware (points to the next rung). Verdict logic unchanged.
# Changelog (v0.1 -> v0.2):
#   * Read CB identity from the loaded model.base, not cfg. The typed StepCfg
#     from build_from_report does not carry use_conditional_base /
#     base_logsigma_max / base_tau, so the v0.1 cfg lookups raised on a valid
#     NICE-CB run. Now: verify .base/.params present (the real CB evidence),
#     read the clamp from base.logsigma_max, and fall back to base_tau=1e-3 if
#     cfg lacks it. Data fields (data_root/blur_sigma/scale/noise_sigma) still
#     come from cfg (StepCfg carries those; same seam freeze_baseline uses).
# Changelog (NEW in v0.1):
#   * Introduced. Loads the ckpt via build_from_report (same seam as N0 /
#     freeze_baseline), asserts NICE-CB identity + base_logsigma_max == 2.0 +
#     presence of .base/.cond, runs one MNISTDegraded val batch through the
#     v0.2 base_diagnostics, prints the saturation metrics + verdict, and
#     writes a small re-runnable JSON record.
# Update summary:
#   v0.1 answers exactly one question -- "is NICE pinned at logsigma_max=2.0 on
#   a meaningful fraction of latent dims?" -- cheaply, before any GPU is spent
#   on N3. N1 measures; N2 decides; the clamp retrain (N3) only runs if BINDING.
# Integration seam (confirm once): build_from_report / MNISTDegraded module
#   paths mirror freeze_baseline.py; reconcile ONLY the import block if your
#   paths differ. base_diagnostics is imported from this package's cond_base.
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

logger = logging.getLogger("NCP-N2")

__version__ = "0.3"
__abbr__ = "NCP-N2"

# --- expected baseline identity (verified against the saved config) ----------
EXPECT_EXPERT_TOKENS = ("nice",)          # expert string must contain 'nice'
EXPECT_BASE_LOGSIGMA_MAX = 2.0
EXPECT_BASE_LOGSIGMA_MAX_TOL = 1e-9


def _die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    logger.error(msg)
    raise RuntimeError(msg)


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _expert_token(cfg: Any) -> str:
    """Pull the expert string from the loaded cfg (typed StepCfg or dict)."""
    val = getattr(cfg, "expert", None)
    if val is None and isinstance(cfg, dict):
        val = cfg.get("expert")
    if not isinstance(val, str) or not val:
        _die(f"could not read 'expert' from cfg ({type(cfg).__name__})")
    return val.lower()


def _cfg_get(cfg: Any, name: str) -> Any:
    val = getattr(cfg, name, None)
    if val is None and isinstance(cfg, dict):
        val = cfg.get(name)
    if val is None:
        _die(f"cfg missing required field {name!r} ({type(cfg).__name__})")
    return val


def _load_model_and_val_y(run_dir: str, device: str, n_y: int):
    """Load the CB-wrapped expert and ONE val batch of y (size n_y). Read-only.
    Mirrors the freeze_baseline loader seam; reconcile only this block if your
    module paths differ."""
    try:
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        _die(f"torch import failed; N2 requires the project env: {e!r}")

    try:
        from CSMF2.experiments.step_1_1_1_1.model_io import build_from_report  # type: ignore
        from CSMF2.data.degrade import MNISTDegraded                          # type: ignore
        from torch.utils.data import DataLoader                              # type: ignore
    except Exception as e:  # noqa: BLE001
        _die("project import failed in the integration seam "
             "(_load_model_and_val_y). Reconcile build_from_report / "
             f"MNISTDegraded to your actual module paths: {e!r}")

    import torch
    dev = torch.device(device)
    model, cond, cfg = build_from_report(run_dir, dev)  # CB-wrapped, frozen
    model.eval()

    val_ds = MNISTDegraded(_cfg_get(cfg, "data_root"), split="val",
                           sigma=_cfg_get(cfg, "blur_sigma"),
                           scale=_cfg_get(cfg, "scale"),
                           noise_sigma=_cfg_get(cfg, "noise_sigma"))
    _x, y = next(iter(DataLoader(val_ds, batch_size=n_y, shuffle=False)))
    y = y.to(dev).float()
    return model, cfg, y


def run_probe(run_dir: str, n_y: int, eps: float, gate_frac: float,
              device: str, expect_logsigma_max: float = 2.0) -> dict:
    import torch  # noqa: F401
    from CSMF2.experiments.step_1_4a.cond_base import base_diagnostics  # type: ignore

    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    if not os.path.isfile(ckpt_path):
        _die(f"checkpoint not found: {ckpt_path}")

    model, cfg, y = _load_model_and_val_y(run_dir, device, n_y)

    # --- identity guards (fail loud on the wrong run) ------------------------
    # NOTE (v0.2): the typed StepCfg from build_from_report does NOT carry
    # use_conditional_base / base_logsigma_max / base_tau as cfg fields. The
    # authoritative evidence that this is a CB run is the loaded model.base
    # object itself (it loaded as "CBExpert, base=conditional"). So verify CB
    # identity and read the clamp from model.base, not from cfg.
    tok = _expert_token(cfg)
    if not any(t in tok for t in EXPECT_EXPERT_TOKENS):
        _die(f"expert {tok!r} is not NICE; refusing to probe the wrong run")
    base = getattr(model, "base", None)
    if base is None or not hasattr(base, "params"):
        _die("loaded model has no conditional base (.base/.params); not a CB expert")
    if not hasattr(model, "cond"):
        _die("loaded model has no .cond; cannot compute h for diagnostics")
    if not hasattr(base, "logsigma_max"):
        _die("model.base has no logsigma_max; cannot verify the clamp")
    blsm = float(base.logsigma_max)
    if abs(blsm - expect_logsigma_max) > EXPECT_BASE_LOGSIGMA_MAX_TOL:
        _die(f"base.logsigma_max={blsm} != expected {expect_logsigma_max} "
             "(pass --expect-logsigma-max to match the clamp you trained)")

    # base_tau (for the base_alive flag inside base_diagnostics): prefer cfg,
    # fall back to the documented default if this cfg type doesn't carry it.
    tau_b = float(getattr(cfg, "base_tau", None)
                  or (cfg.get("base_tau") if isinstance(cfg, dict) else None)
                  or 1e-3)

    # --- the N1 metric, reused (no recompute logic duplicated here) ----------
    diag = base_diagnostics(model.base, model.cond, y, tau_b=tau_b,
                            n_y=n_y, eps=eps)

    frac_max = float(diag["fraction_at_sigma_max"])
    binding = frac_max > gate_frac
    verdict = "BINDING" if binding else "NOT_BINDING"
    if binding:
        next_step = (f"BINDING at clamp {blsm:g} -> retrain at the next clamp rung "
                     "(N3 clamp 3.0 if probing 2.0; N4 clamp 4.0 if probing 3.0), seed0")
    else:
        next_step = ("NOT_BINDING -> coupling-limited -> STOP clamp branch "
                     "(skip remaining rungs); one-shot N8 or Stage 2.3")

    record = {
        "abbr": __abbr__,
        "version": __version__,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "run_dir": os.path.abspath(run_dir),
        "checkpoint_path": os.path.abspath(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "expert": tok,
        "base_logsigma_max": blsm,
        "n_y": int(n_y),
        "saturation_eps": float(eps),
        "gate_frac": float(gate_frac),
        # --- metrics (from the v0.2 N1 base_diagnostics) ---
        "fraction_at_sigma_max": frac_max,
        "fraction_at_sigma_min": float(diag["fraction_at_sigma_min"]),
        "base_logsigma_q05": float(diag["base_logsigma_q05"]),
        "base_logsigma_q50": float(diag["base_logsigma_q50"]),
        "base_logsigma_q95": float(diag["base_logsigma_q95"]),
        "base_logsigma_std_overall": float(diag["base_logsigma_std_overall"]),
        "sigma_max": float(diag["sigma_max"]),
        "sigma_min": float(diag["sigma_min"]),
        # --- verdict (decided HERE, not in cond_base) ---
        "clamp_binding": bool(binding),
        "verdict": verdict,
        "next_step": next_step,
        "note": ("decide on fraction_at_sigma_max, NOT sigma_max alone -- a few "
                 "elements at the ceiling is not a globally clamp-limited base"),
    }
    return record


def write_record(out_dir: str, record: dict, ckpt_sha12: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"n2_clamp_probe_{ckpt_sha12}.json")
    # re-runnable (NOT write-once): overwrite is allowed, unlike the N0 baseline.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return path


def _print_summary(record: dict) -> None:
    print()
    print(f"N2 clamp-binding pre-check ({record['abbr']} v{record['version']})")
    print(f"  expert={record['expert']}  base_logsigma_max={record['base_logsigma_max']}"
          f"  n_y={record['n_y']}  eps={record['saturation_eps']}"
          f"  gate_frac={record['gate_frac']}")
    print(f"  fraction_at_sigma_max = {record['fraction_at_sigma_max']:.6f}")
    print(f"  fraction_at_sigma_min = {record['fraction_at_sigma_min']:.6f}")
    print(f"  base_logsigma q05/q50/q95 = "
          f"{record['base_logsigma_q05']:.4f} / {record['base_logsigma_q50']:.4f} "
          f"/ {record['base_logsigma_q95']:.4f}")
    print(f"  base_logsigma_std_overall = {record['base_logsigma_std_overall']:.4f}")
    print(f"  sigma_max = {record['sigma_max']:.4f}   sigma_min = {record['sigma_min']:.4f}")
    print(f"  note: {record['note']}")
    print(f"  clamp_binding = {'YES' if record['clamp_binding'] else 'NO'}")
    print(f"  verdict = {record['verdict']}")
    print(f"  next_step = {record['next_step']}")
    print()


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
    ap = argparse.ArgumentParser(
        description="NCP-N2: free clamp-binding pre-check on an existing NICE-CB ckpt.")
    ap.add_argument("--run-dir", required=True,
                    help="the clamp-2.0 NICE-CB checkpoint dir")
    ap.add_argument("--n-y", type=int, default=512,
                    help="val samples for the diagnostic (default 512)")
    ap.add_argument("--eps", type=float, default=1e-3,
                    help="logsigma-space distance to count as 'at the clamp' (default 1e-3)")
    ap.add_argument("--gate-frac", type=float, default=1e-4,
                    help="fraction_at_sigma_max cutoff for BINDING (default 1e-4)")
    ap.add_argument("--expect-logsigma-max", type=float, default=2.0,
                    help="assert the ckpt's base.logsigma_max equals this "
                         "(clamp sweep: 2.0 baseline, 3.0 for N3, 4.0 for N4)")
    ap.add_argument("--device", default="cpu", help="torch device (default cpu)")
    ap.add_argument("--out-dir",
                    default="CSMF2/experiments/step_1_4a/baselines",
                    help="where to write the re-runnable JSON record")
    ap.add_argument("--no-write-json", action="store_true",
                    help="print to stdout only; do not write the JSON record")
    args = ap.parse_args()

    if args.n_y <= 0:
        _die(f"--n-y must be positive, got {args.n_y}")
    if not (args.eps > 0.0):
        _die(f"--eps must be positive, got {args.eps}")
    if not (args.gate_frac >= 0.0):
        _die(f"--gate-frac must be >= 0, got {args.gate_frac}")

    record = run_probe(args.run_dir, args.n_y, args.eps, args.gate_frac,
                       args.device, args.expect_logsigma_max)
    _print_summary(record)

    if not args.no_write_json:
        ckpt_sha12 = record["checkpoint_sha256"][:12]
        path = write_record(args.out_dir, record, ckpt_sha12)
        logger.info("N2 record written -> %s", path)

    # exit 0 regardless of verdict: BINDING / NOT_BINDING are both valid
    # outcomes, not failures. Read the printed verdict to choose N3 vs STOP.
    return 0


if __name__ == "__main__":
    sys.exit(main())

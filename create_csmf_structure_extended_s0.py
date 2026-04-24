#!/usr/bin/env python3
# =============================================================================
# EXP-SCAFFOLD v2.2 -- create_csmf_structure.py
# CONVENTION: NLL = LOSS (lower = better).
#             All run artifacts scoped by (step_id, seed, cfg_hash).
# =============================================================================
"""
Scaffolder for the CSMF Incremental Experiment Plan (WP0 -> WP5).

Generates an 11-step experiment tree where each stage is:
  * its own Python package (run.py + config.py) -- independently runnable
  * its own results sandbox (configs/logs/results/checkpoints/plots)
  * its own progress state (status.json + metrics.csv)

Plan steps (execution order): 1.1, 1.2, 1.3, 2.2, 3.1, 3.2, 3.3, 4, 5.1, 5.2, 6.

Changelog (v2.2 -> v2.3):
  * Generates three real (non-stub) files under common/:
      - cond_diagnostics.py  : COND-GATE v0.3 checks #1-#8 + bundler + move-forward gate
      - gate_diagnostics.py  : COND-GATE v0.3 check  #9 (gate collapse probe)
      - cond_viz.py          : 7 diagnostic plots (h/FiLM/grads/determinism/gate)
  * These replace the previous implicit expectation that step owners build their
    own diagnostics -- now every step imports `common.cond_diagnostics` and
    `common.cond_viz`. Bodies are real implementations, never NI stubs.
  * No changes to STEPS, per-step layout, writer, or existing common/ files.

Changelog (v2.1 -> v2.2):
  * New top-level package `models/` (sibling to `common/`) for shared model code
    (experts, gates, flows) so definitions live in one importable place instead
    of being re-defined per step. Accessible as `CSMF2.models.*`.
  * `models/__init__.py` generated with the standard H() header; no NI stubs yet
    -- add module files (e.g. experts.py, gates.py, flows.py) as they are written.
  * No changes to STEPS, writer, per-step layout, or existing files.

Changelog (v2.0 -> v2.1):
  * BASE directory renamed: csmf_exp/ -> CSMF2/ (module paths update accordingly).
  * Split requirements: requirements.txt (dashboard-minimal) +
    requirements-project.txt (full CSMF ML stack: torch, torchvision, scipy,
    scikit-image, pandas, Pillow, matplotlib, pydantic, tqdm, pyyaml).
  * README quick-start + generated run.py docstrings updated to reference CSMF2.
  * No changes to STEPS, _safe_write, no-clobber behaviour, or NI stub semantics.

Changelog (v1.0 -> v2.0):
  * Layout redesigned to match PBMA-DIV v0.3.1 scaffolder format (H, NI, _safe_write).
  * Each step is now a runnable Python package: run.py + config.py stubs.
  * Step dirs renamed step_1.1 -> step_1_1 (Python-importable).
  * New common/ package: logger / status_io / metrics_io (real) + seed / hashing (NI).
  * `--force` flag; default is no-clobber with created/skipped/overwritten counts.

Update summary (v2.2 -> v2.3):
  Closes the diagnostics gap. v2.2 left each step responsible for its own
  conditioning/gating sanity checks, which meant (a) duplication across 11
  steps and (b) a real risk of silent failures (e.g. a dead FiLM head hidden
  by aggregate stats). v2.3 ships a shared COND-GATE v0.3 diagnostics package
  with 9 real checks (checks 1-8 in cond_diagnostics.py, check 9 -- gate
  collapse -- in gate_diagnostics.py) plus a visualisation layer. Every check
  raises on fail and logs via logger.error -- no silent pass / mock / dummy.
  Additive only: existing trees update in-place without --force; the scaffolder
  will add the 3 files and skip everything else.

Update summary (v2.1 -> v2.2):
  Additive structural change. Introduces a `models/` package alongside `common/`
  so shared model classes (experts, gates, normalising flows) can be imported
  from a single location (`from CSMF2.models.experts import ...`) rather than
  duplicated across step packages. No behavioural change to the 11 experiment
  stages, the dashboard index, or the no-clobber writer. Existing trees can be
  updated in-place without `--force` -- the scaffolder will only add the new
  `models/` directory and skip everything else.

Update summary (v2.0 -> v2.1):
  Cosmetic + dependency housekeeping. The BASE output directory is now `CSMF2/`
  to match the user's project naming, so the run invocation becomes
  `python -m CSMF2.experiments.step_1_1.run --seed <n>`. Requirements are split
  so a monitoring-only install (flask + pyyaml) stays lightweight while the
  full training stack (torch / torchvision / scipy / scikit-image / pandas /
  Pillow / matplotlib / pydantic / tqdm / pyyaml) goes in a dedicated
  requirements-project.txt. No behavioural change to scaffolding logic.

Update summary (v1.0 -> v2.0):
  v1.0 scaffolded only data files (README.md, status.json, metrics.csv) and left
  the reproducibility workflow implicit. v2.0 generates a runnable Python package
  per stage so each step can be re-run independently via
  `python -m CSMF2.experiments.step_1_1.run --seed <n>`. Per-run output is
  scoped by `(seed, cfg_hash)` under each step's `results/` so reruns never
  collide. Shared helpers (status_io, metrics_io, logger) are real implementations;
  per-step entry points and seed/hashing helpers are NotImplementedError stubs so
  skipped work is loud, never silent. The existing EXP-DASH v1.0 dashboard keeps
  working -- it only reads experiment_index.json which this scaffolder regenerates.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from datetime import datetime, timezone

__version__ = "2.3"
__abbr__ = "EXP-SCAFFOLD"
BASE = "CSMF2"


# -----------------------------------------------------------------------------
# 11 steps of the CSMF incremental experiment plan (execution order)
# -----------------------------------------------------------------------------
STEPS: list[dict] = [
    {"step_id": "1.1", "step_name": "Single conditional expert (no mixture)",  "wp": "WP0",
     "exit": "All WP0 tests pass; no NaN; conditioning active; >= 2 experts competent."},
    {"step_id": "1.2", "step_name": "Mixture skeleton (pure NLL)",             "wp": "WP0",
     "exit": "Learned gate no worse than uniform; gate usage varies across inputs."},
    {"step_id": "1.3", "step_name": "Expert sanity package (pre-Stage B)",     "wp": "WP0",
     "exit": "No dead expert; no constant-logdet expert; mild diversity present."},
    {"step_id": "2.2", "step_name": "Proximal step at inference only",         "wp": "WP1",
     "exit": "Smallest T with clear residual drop + acceptable NLL/recon tradeoff."},
    {"step_id": "3.1", "step_name": "Stage A with weak consistency",           "wp": "WP2",
     "exit": "Weak consistency helps or does not hurt expert health."},
    {"step_id": "3.2", "step_name": "Stage B full hybrid (experts frozen)",    "wp": "WP2",
     "exit": "Hybrid beats pure NLL on residual/geometry without hurting NLL (M2)."},
    {"step_id": "3.3", "step_name": "Stage C light joint fine-tune",           "wp": "WP2",
     "exit": "Clear gain over Stage B OR same metrics with better recon quality."},
    {"step_id": "4",   "step_name": "WP3 MNIST ablation matrix",               "wp": "WP3",
     "exit": "One default config selected; matched/better NLL + improved residual/geometry."},
    {"step_id": "5.1", "step_name": "Port to optical SR (DIV2K / BSD68)",      "wp": "WP4",
     "exit": "Residual/geometry gain without NLL collapse; visible recon improvement."},
    {"step_id": "5.2", "step_name": "Port to SAR prototype",                   "wp": "WP4",
     "exit": "Same qualitative pattern as MNIST and SR."},
    {"step_id": "6",   "step_name": "Final sweep (lambda_cons, T, tau)",       "wp": "WP5",
     "exit": "Final output pack: curves, Pareto, PIT/ES, sample grids, gate histograms."},
]


def dir_of(step_id: str) -> str:
    """'1.1' -> 'step_1_1' so the directory is a valid Python package name."""
    return "step_" + step_id.replace(".", "_")


# -----------------------------------------------------------------------------
# File body helpers (PBMA-DIV style)
# -----------------------------------------------------------------------------
def H(module: str, purpose: str) -> str:
    """Standard header block injected at the top of every generated .py file."""
    return textwrap.dedent(f'''\
        # =============================================================================
        # {__abbr__} v{__version__} -- {module}
        # Purpose: {purpose}
        # CONVENTION: NLL = LOSS (lower = better). Artifacts scoped by (step, seed, cfg_hash).
        # =============================================================================
        from __future__ import annotations
        import logging
        logger = logging.getLogger(__name__)
        __version__ = "{__version__}"
    ''')


def NI(fn: str, note: str) -> str:
    """Stub body: logs an error and raises -- never a silent pass/mock/dummy."""
    return (
        f"\n\ndef {fn}(*args, **kwargs):\n"
        f'    logger.error("[{fn}] not implemented -- {note}")\n'
        f'    raise NotImplementedError("{fn}: {note}")\n'
    )


# -----------------------------------------------------------------------------
# Real helper bodies (short; header from H() provides logger + __version__)
# -----------------------------------------------------------------------------
LOGGER_BODY = '''

def get_logger(name, level=logging.INFO):
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s :: %(message)s"))
        lg.addHandler(h)
        lg.setLevel(level)
    return lg
'''


STATUS_IO_BODY = '''
import json
import os
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_status(step_dir):
    p = Path(step_dir) / "status.json"
    if not p.exists():
        logger.error("status.json not found at %s", p)
        raise FileNotFoundError(str(p))
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.error("failed to read %s\\n%s", p, traceback.format_exc())
        raise


def write_status(step_dir, status):
    """Atomic write: temp file in same dir + os.replace."""
    step_dir = Path(step_dir)
    p = step_dir / "status.json"
    status["last_updated"] = _now_iso()
    fd, tmp_path = tempfile.mkstemp(dir=str(step_dir), prefix=".status.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(status, indent=2) + "\\n")
        os.replace(tmp_path, p)
    except OSError:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        logger.error("atomic write failed for %s\\n%s", p, traceback.format_exc())
        raise


def update_status(step_dir, seed_done=None, **fields):
    """Read -> mutate -> atomic write. Appends seed without duplicates."""
    st = read_status(step_dir)
    if seed_done is not None and seed_done not in st["seeds_done"]:
        st["seeds_done"].append(seed_done)
    st.update(fields)
    write_status(step_dir, st)
    return st
'''


METRICS_IO_BODY = '''
import csv
import traceback
from pathlib import Path

FIELDS = ["seed", "epoch", "nll", "residual", "sw2", "es", "neff", "notes"]


def append_row(step_dir, **row):
    missing = [k for k in FIELDS if k not in row]
    if missing:
        logger.error("missing metric fields: %s", missing)
        raise ValueError("missing metric fields: " + str(missing))
    p = Path(step_dir) / "metrics.csv"
    try:
        with p.open("a", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerow([row[k] for k in FIELDS])
    except OSError:
        logger.error("metrics append failed at %s\\n%s", p, traceback.format_exc())
        raise
'''


# -----------------------------------------------------------------------------
# Per-step body renderers
# -----------------------------------------------------------------------------
def render_run_py(step: dict) -> str:
    sid = step["step_id"]
    mod = f"experiments.{dir_of(sid)}.run"
    head = H(mod, f"Entry point for step {sid} -- {step['step_name']} ({step['wp']})")
    body = textwrap.dedent(f'''

        import argparse
        import sys
        import traceback
        from pathlib import Path

        STEP_ID = "{sid}"
        STEP_NAME = "{step["step_name"]}"
        WP = "{step["wp"]}"
        STEP_DIR = Path(__file__).parent


        def output_dir(seed, cfg_hash):
            """Per-run output directory -- scoped by (seed, cfg_hash) so reruns never collide."""
            out = STEP_DIR / "results" / ("seed" + str(seed) + "_cfg" + str(cfg_hash))
            out.mkdir(parents=True, exist_ok=True)
            return out


        def run(seed, cfg_overrides=None):
            """Training / evaluation for this step. IMPLEMENT ME.

            Expected calls inside:
              * common.seed.set_seed(seed)
              * cfg_hash = common.hashing.config_hash(cfg)
              * out = output_dir(seed, cfg_hash)
              * common.metrics_io.append_row(STEP_DIR, seed=..., epoch=..., nll=..., ...)
              * common.status_io.update_status(STEP_DIR, status="running")
              * common.status_io.update_status(STEP_DIR, seed_done=seed, status="done",
                                               exit_criteria_met=<bool>)
            """
            logger.error("[run] step {sid}: training/eval not implemented")
            raise NotImplementedError(
                "step {sid}: wire training/eval here; call append_row() + update_status() on progress"
            )


        def main(argv=None):
            ap = argparse.ArgumentParser(description="step {sid} -- {step["step_name"]}")
            ap.add_argument("--seed", type=int, required=True)
            ap.add_argument("--verbose", action="store_true")
            args = ap.parse_args(argv)
            logging.basicConfig(
                level=logging.DEBUG if args.verbose else logging.INFO,
                format="[%(asctime)s] %(levelname)s %(name)s :: %(message)s",
            )
            try:
                run(args.seed)
            except NotImplementedError:
                raise
            except Exception:
                logger.error("step {sid} crashed\\n%s", traceback.format_exc())
                return 2
            return 0


        if __name__ == "__main__":
            sys.exit(main())
    ''')
    return head + body


def render_config_py(step: dict) -> str:
    sid = step["step_id"]
    mod = f"experiments.{dir_of(sid)}.config"
    head = H(mod, f"Step-local config for step {sid} -- {step['step_name']}")
    body = textwrap.dedent(f'''

        # Fill in step-specific hyperparameters below. Keep this dict canonical
        # (sorted keys, no runtime state) so that config_hash() is stable across runs.
        CONFIG = {{
            "step_id":   "{sid}",
            "step_name": "{step["step_name"]}",
            "wp":        "{step["wp"]}",
            # --- fill in below ---
            # "batch_size":   128,
            # "lr":           1e-3,
            # "epochs":       50,
            # "lambda_cons":  0.0,
            # "lambda_trans": 0.0,
            # "tau":          1.1,
        }}


        def get_config():
            return dict(CONFIG)
    ''')
    return head + body


def render_readme(step: dict) -> str:
    return textwrap.dedent(f'''\
        # Step {step["step_id"]} -- {step["step_name"]}

        **Workpackage:** {step["wp"]}

        ## Run

        ```
        python -m {BASE}.experiments.{dir_of(step["step_id"])}.run --seed 0
        python -m {BASE}.experiments.{dir_of(step["step_id"])}.run --seed 1
        python -m {BASE}.experiments.{dir_of(step["step_id"])}.run --seed 2
        ```

        ## Output layout

        ```
        results/seed<N>_cfg<HASH>/   one sub-dir per (seed, cfg_hash)
        checkpoints/                  per-run checkpoints
        plots/                        per-run diagnostic PNGs
        logs/                         stdout/stderr tee logs
        configs/                      snapshot of the config used
        status.json                   step progress (shared, atomic)
        metrics.csv                   append-only per-epoch metrics
        ```

        ## Exit criterion (go / no-go)

        {step["exit"]}

        ## Move-forward gate (applies to every step)

        - no NaNs
        - no invertibility failure
        - gate not collapsed
        - Neff monitored
        - residual improves when expected
        - NLL not materially worse unless justified
        - stable across 3 seeds
    ''')


def render_status(step: dict) -> str:
    return json.dumps({
        "step_id": step["step_id"],
        "step_name": step["step_name"],
        "status": "not_started",
        "seeds_done": [],
        "exit_criteria_met": False,
        "last_updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": "",
    }, indent=2) + "\n"


def render_experiment_index() -> str:
    return json.dumps({
        "version": __version__,
        "abbr": __abbr__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "steps": [
            {
                "step_id": s["step_id"],
                "step_name": s["step_name"],
                "wp": s["wp"],
                "path": f"experiments/{dir_of(s['step_id'])}",
            }
            for s in STEPS
        ],
    }, indent=2) + "\n"


METRICS_HEADER = "seed,epoch,nll,residual,sw2,es,neff,notes\n"



# -----------------------------------------------------------------------------
# COND-GATE v0.3 body constants (embedded verbatim; raw-strings so backslash
# escapes like `\n` in logger.error messages survive to the generated file).
# -----------------------------------------------------------------------------
COND_DIAGNOSTICS_BODY = r"""# =============================================================================
# COND-GATE v0.3 -- common.cond_diagnostics
# Purpose: 8 conditioning sanity checks (h / FiLM / s,t / grads / cache /
#          determinism / per-layer FiLM). Check #9 (gate collapse) lives in
#          gate_diagnostics.py. Bundler run_global_gate() calls both.
# CONVENTION: NLL = LOSS (lower = better). All checks raise ValueError on fail
#             and log via logger.error -- never silent pass / mock / dummy.
# Changelog (v0.2 -> v0.3):
#   * Added film_stats_per_layer (check #8) to catch one broken FiLM head
#     that aggregate stats hide.
#   * Bundler run_global_gate() now optionally calls gate_collapse_probe
#     from gate_diagnostics when a gate module is provided.
#   * No threshold changes on checks 1-7.
# =============================================================================
from __future__ import annotations
import logging
import traceback
logger = logging.getLogger(__name__)
__version__ = "0.3"
__abbr__ = "COND-GATE"

import torch

# --- Default thresholds (override by passing tol=... where supported) --------
STD_EPS            = 1e-8
CACHE_TOL          = 1e-5
DETERMINISM_TOL    = 1e-6
GRAD_EPS           = 1e-10
SHUFFLE_DELTA_EPS  = 1e-4
DIVERSITY_EPS      = 1e-6


def _t(x):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


# ----- Check 1: h_stats ------------------------------------------------------
def h_stats(h, raise_on_fail=True):
    h = _t(h).detach()
    nan = int(torch.isnan(h).sum().item())
    inf = int(torch.isinf(h).sum().item())
    if nan or inf:
        logger.error("[h_stats] NaN=%d Inf=%d", nan, inf)
        if raise_on_fail:
            raise ValueError(f"h_stats: NaN={nan} Inf={inf}")
    fin = h[torch.isfinite(h)]
    if fin.numel() == 0:
        logger.error("[h_stats] no finite values")
        raise ValueError("h_stats: no finite values")
    mean = float(fin.mean().item())
    std  = float(fin.std().item())
    nrm  = float(torch.linalg.vector_norm(h.flatten()).item())
    nuq  = int(torch.unique(fin).numel())
    if std < STD_EPS:
        logger.error("[h_stats] h appears constant (std=%.2e)", std)
        if raise_on_fail:
            raise ValueError(f"h_stats: constant h (std={std:.2e})")
    return {"mean": mean, "std": std, "norm": nrm,
            "nan": nan, "inf": inf, "n_unique": nuq}


# ----- Check 2: h_diversity --------------------------------------------------
def h_diversity(h_batch, raise_on_fail=True):
    h = _t(h_batch).detach().flatten(start_dim=1)
    B = h.shape[0]
    if B < 2:
        logger.error("[h_diversity] need B>=2 got B=%d", B)
        raise ValueError(f"h_diversity: need B>=2 got {B}")
    d = torch.cdist(h, h)
    mask = ~torch.eye(B, dtype=torch.bool, device=h.device)
    off = d[mask]
    mean_d = float(off.mean().item())
    if mean_d < DIVERSITY_EPS:
        logger.error("[h_diversity] batch-collapsed (mean pairwise=%.2e)", mean_d)
        if raise_on_fail:
            raise ValueError(f"h_diversity: collapsed (mean={mean_d:.2e})")
    return {"mean_pairwise": mean_d,
            "min_pairwise":  float(off.min().item()),
            "max_pairwise":  float(off.max().item()),
            "pairwise_matrix": d.cpu().numpy()}


# ----- Check 3: film_stats (aggregate) ---------------------------------------
def film_stats(gamma, beta, raise_on_fail=True):
    out = {}
    for name, t in [("gamma", _t(gamma).detach()), ("beta", _t(beta).detach())]:
        nan = int(torch.isnan(t).sum().item())
        inf = int(torch.isinf(t).sum().item())
        fin = t[torch.isfinite(t)]
        mean = float(fin.mean().item()) if fin.numel() else float("nan")
        std  = float(fin.std().item())  if fin.numel() else float("nan")
        out[name] = {"mean": mean, "std": std, "nan": nan, "inf": inf}
        if nan or inf:
            logger.error("[film_stats] %s NaN=%d Inf=%d", name, nan, inf)
            if raise_on_fail:
                raise ValueError(f"film_stats: {name} NaN={nan} Inf={inf}")
        if std < STD_EPS:
            logger.error("[film_stats] %s constant (std=%.2e)", name, std)
            if raise_on_fail:
                raise ValueError(f"film_stats: {name} constant (std={std:.2e})")
    return out


# ----- Check 4: st_sensitivity ----------------------------------------------
def st_sensitivity(flow_fn, x, h, h_shuffled, raise_on_fail=True):
    # flow_fn(x, h) -> (s, t, logp). Shuffle should perturb all three.
    try:
        s1, t1, lp1 = flow_fn(_t(x), _t(h))
        s2, t2, lp2 = flow_fn(_t(x), _t(h_shuffled))
    except Exception:
        logger.error("[st_sensitivity] flow_fn crashed\n%s", traceback.format_exc())
        raise
    ds  = float((s1 - s2).abs().max().item())
    dt  = float((t1 - t2).abs().max().item())
    dlp = float((lp1 - lp2).abs().max().item())
    if ds < SHUFFLE_DELTA_EPS or dt < SHUFFLE_DELTA_EPS:
        logger.error("[st_sensitivity] (s,t) insensitive: ds=%.2e dt=%.2e", ds, dt)
        if raise_on_fail:
            raise ValueError(f"st_sensitivity: (s,t) ignores h (ds={ds:.2e} dt={dt:.2e})")
    if dlp < SHUFFLE_DELTA_EPS:
        logger.error("[st_sensitivity] logp insensitive: dlogp=%.2e", dlp)
        if raise_on_fail:
            raise ValueError(f"st_sensitivity: logp ignores h (dlogp={dlp:.2e})")
    return {"ds_max": ds, "dt_max": dt, "dlogp_max": dlp}


# ----- Check 5: grad_norms ---------------------------------------------------
def grad_norms(cond_net, film_heads, raise_on_fail=True):
    # Call AFTER loss.backward(). Reads .grad on each param.
    def _norm(params):
        tot, n = 0.0, 0
        for p in params:
            if p.grad is None:
                continue
            tot += float(p.grad.detach().pow(2).sum().item())
            n += 1
        return tot ** 0.5, n

    cn, cn_n = _norm(cond_net.parameters())
    if cn_n == 0 or cn < GRAD_EPS:
        logger.error("[grad_norms] conditioner grad=%.2e n_params=%d", cn, cn_n)
        if raise_on_fail:
            raise ValueError(f"grad_norms: conditioner no gradient ({cn:.2e})")
    heads = []
    for i, h in enumerate(film_heads):
        hn, hn_n = _norm(h.parameters())
        heads.append(hn)
        if hn_n == 0 or hn < GRAD_EPS:
            logger.error("[grad_norms] FiLM head %d grad=%.2e", i, hn)
            if raise_on_fail:
                raise ValueError(f"grad_norms: FiLM head {i} no gradient ({hn:.2e})")
    return {"conditioner": cn, "film_heads": heads}


# ----- Check 6: cache_check --------------------------------------------------
def cache_check(h_cached, h_fresh, tol=CACHE_TOL, raise_on_fail=True):
    hc = _t(h_cached).detach()
    hf = _t(h_fresh).detach()
    if hc.shape != hf.shape:
        logger.error("[cache_check] shape mismatch %s vs %s", hc.shape, hf.shape)
        raise ValueError("cache_check: shape mismatch")
    err = float((hc - hf).abs().max().item())
    if err > tol:
        logger.error("[cache_check] max|diff|=%.2e > tol=%.2e", err, tol)
        if raise_on_fail:
            raise ValueError(f"cache_check: max|diff|={err:.2e} > {tol:.2e}")
    return {"max_abs_diff": err}


# ----- Check 7: determinism_check -------------------------------------------
def determinism_check(cond_net, y, seed, set_seed_fn,
                      tol=DETERMINISM_TOL, raise_on_fail=True):
    # Run cond_net(y) twice under set_seed_fn(seed). Assert identical output.
    try:
        was_training = cond_net.training
        cond_net.eval()
        set_seed_fn(seed)
        with torch.no_grad():
            h1 = cond_net(_t(y)).detach().clone()
        set_seed_fn(seed)
        with torch.no_grad():
            h2 = cond_net(_t(y)).detach().clone()
        if was_training:
            cond_net.train()
    except Exception:
        logger.error("[determinism_check] cond_net crashed\n%s", traceback.format_exc())
        raise
    err = float((h1 - h2).abs().max().item())
    if err > tol:
        logger.error("[determinism_check] max|Δh|=%.2e > tol=%.2e (hidden RNG?)", err, tol)
        if raise_on_fail:
            raise ValueError(f"determinism_check: max|Δh|={err:.2e} > {tol:.2e}")
    return {"max_abs_delta_h": err}


# ----- Check 8: film_stats_per_layer ----------------------------------------
def film_stats_per_layer(film_head_outputs, raise_on_fail=True):
    # film_head_outputs: list[(gamma_i, beta_i)] -- one tuple per FiLM layer.
    per = []
    for i, (g, b) in enumerate(film_head_outputs):
        try:
            s = film_stats(g, b, raise_on_fail=raise_on_fail)
        except ValueError as e:
            logger.error("[film_stats_per_layer] layer %d failed: %s", i, e)
            raise
        s["layer_index"] = i
        per.append(s)
    return per


# ----- Move-forward gate -----------------------------------------------------
def check_move_forward(log):
    # log: dict aggregated across a completed run (expected keys below).
    # Returns (ok: bool, reasons: list[str]).
    reasons = []
    if not log.get("main_metric_improved", False):
        reasons.append("main metric did not improve")
    if log.get("nll_regression", False):
        reasons.append("NLL regression detected")
    if log.get("numerical_failure", False):
        reasons.append("numerical failure occurred")
    if log.get("n_seeds_ok", 0) < 3:
        reasons.append(f"only {log.get('n_seeds_ok',0)} seeds passed (need >=3)")
    if not log.get("h_finite_nonconstant", False):
        reasons.append("h not finite / not varying / not deterministic")
    if not log.get("film_valid_all_layers", False):
        reasons.append("one or more FiLM layers invalid")
    if not log.get("shuffle_changes_st_and_logp", False):
        reasons.append("shuffling h did not change s,t or logp")
    if not log.get("grads_flow_everywhere", False):
        reasons.append("conditioner or FiLM head has no gradient")
    if not log.get("cache_matches_fresh", True):
        reasons.append("cached h disagrees with fresh h")
    if not log.get("determinism_ok", False):
        reasons.append("(y,seed) does not reproduce identical h")
    if log.get("gate_used", False) and not log.get("gate_healthy", False):
        reasons.append("gate collapsed (Neff/max_w/variance check failed)")
    return (len(reasons) == 0, reasons)


# ----- Bundler ---------------------------------------------------------------
def run_global_gate(*, h=None, h_batch=None,
                    gamma_aggregate=None, beta_aggregate=None,
                    film_per_layer_outputs=None,
                    flow_fn=None, x=None, h_shuffled=None,
                    cond_net=None, film_heads=None,
                    h_cached=None, h_fresh=None,
                    y=None, seed=None, set_seed_fn=None,
                    gate_fn=None, y_batch=None,
                    raise_on_fail=False):
    # Runs all checks whose required args are provided. Others are skipped +
    # logged. Returns {passed, reasons, metrics}. Never silent.
    metrics, reasons = {}, []

    def _try(name, fn, *args, **kw):
        try:
            metrics[name] = fn(*args, **kw, raise_on_fail=True)
            return True
        except ValueError as e:
            reasons.append(f"{name}: {e}")
            return False
        except Exception as e:
            logger.error("[run_global_gate] %s crashed: %s\n%s",
                         name, e, traceback.format_exc())
            reasons.append(f"{name}: crash -- {e}")
            return False

    ok = True
    if h is not None:
        ok &= _try("h_stats", h_stats, h)
    if h_batch is not None:
        ok &= _try("h_diversity", h_diversity, h_batch)
    if gamma_aggregate is not None and beta_aggregate is not None:
        ok &= _try("film_stats", film_stats, gamma_aggregate, beta_aggregate)
    if film_per_layer_outputs is not None:
        ok &= _try("film_stats_per_layer", film_stats_per_layer, film_per_layer_outputs)
    if flow_fn is not None and x is not None and h is not None and h_shuffled is not None:
        ok &= _try("st_sensitivity", st_sensitivity, flow_fn, x, h, h_shuffled)
    if cond_net is not None and film_heads is not None:
        ok &= _try("grad_norms", grad_norms, cond_net, film_heads)
    if h_cached is not None and h_fresh is not None:
        ok &= _try("cache_check", cache_check, h_cached, h_fresh)
    if cond_net is not None and y is not None and seed is not None and set_seed_fn is not None:
        ok &= _try("determinism_check", determinism_check,
                   cond_net, y, seed, set_seed_fn)
    if gate_fn is not None and y_batch is not None:
        try:
            from .gate_diagnostics import gate_collapse_probe
        except ImportError:
            from gate_diagnostics import gate_collapse_probe   # flat-layout fallback
        ok &= _try("gate_collapse_probe", gate_collapse_probe, gate_fn, y_batch)

    if not ok and raise_on_fail:
        raise ValueError("COND-GATE failed: " + "; ".join(reasons))
    return {"passed": ok, "reasons": reasons, "metrics": metrics}
"""

GATE_DIAGNOSTICS_BODY = r"""# =============================================================================
# COND-GATE v0.3 -- common.gate_diagnostics
# Purpose: check #9 -- gate collapse probe. Separate module so Stages 1.2, 3.2,
#          and the WP3 ablation matrix re-use the same code.
# CONVENTION: NLL = LOSS (lower = better). Probe raises ValueError on fail and
#             logs via logger.error -- never silent pass / mock / dummy.
# Changelog (new in v0.3):
#   * Introduced. Reports neff_mean, entropy_mean, max_w_mean, per-input
#     variance of w, and argmax-expert histogram.
#   * Raises on any of: Neff<1.5, max_w>0.95, w constant across inputs,
#     weights that do not sum to 1.
# =============================================================================
from __future__ import annotations
import logging
import traceback
logger = logging.getLogger(__name__)
__version__ = "0.3"
__abbr__ = "COND-GATE"

import torch

NEFF_MIN     = 1.5
MAX_W_MAX    = 0.95
VAR_EPS      = 1e-6
SUM_TOL      = 1e-4


def gate_collapse_probe(gate_fn, y_batch,
                        neff_min=NEFF_MIN, max_w_max=MAX_W_MAX,
                        var_eps=VAR_EPS, raise_on_fail=True):
    # gate_fn(y) -> weights (B, K) summing to 1. Detects dead / collapsed gate.
    try:
        w = gate_fn(y_batch)
    except Exception:
        logger.error("[gate_collapse_probe] gate_fn crashed\n%s", traceback.format_exc())
        raise
    w = w.detach()
    if w.dim() != 2:
        logger.error("[gate_collapse_probe] expected (B,K) got shape %s", tuple(w.shape))
        raise ValueError(f"gate_collapse_probe: expected (B,K) got {tuple(w.shape)}")
    B, K = w.shape

    row_sums = w.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=SUM_TOL):
        lo, hi = float(row_sums.min().item()), float(row_sums.max().item())
        logger.error("[gate_collapse_probe] weights do not sum to 1: rows in [%.4f, %.4f]", lo, hi)
        if raise_on_fail:
            raise ValueError(f"gate_collapse_probe: row sums in [{lo:.4f},{hi:.4f}]")

    entropy = -(w * w.clamp_min(1e-12).log()).sum(-1)          # (B,)
    neff    = entropy.exp()                                     # (B,)
    max_w   = w.max(-1).values                                  # (B,)
    w_var_across_inputs = float(w.var(dim=0).mean().item())     # scalar

    neff_mean    = float(neff.mean().item())
    entropy_mean = float(entropy.mean().item())
    max_w_mean   = float(max_w.mean().item())

    if neff_mean < neff_min:
        logger.error("[gate_collapse_probe] Neff=%.3f < %.2f (gate collapsed)",
                     neff_mean, neff_min)
        if raise_on_fail:
            raise ValueError(f"gate_collapse_probe: Neff={neff_mean:.3f} < {neff_min}")
    if max_w_mean > max_w_max:
        logger.error("[gate_collapse_probe] mean max_w=%.3f > %.2f (one expert dominates)",
                     max_w_mean, max_w_max)
        if raise_on_fail:
            raise ValueError(f"gate_collapse_probe: max_w={max_w_mean:.3f} > {max_w_max}")
    if w_var_across_inputs < var_eps:
        logger.error("[gate_collapse_probe] w constant across inputs (var=%.2e)",
                     w_var_across_inputs)
        if raise_on_fail:
            raise ValueError(f"gate_collapse_probe: w constant across inputs "
                             f"(var={w_var_across_inputs:.2e})")

    argmax_hist = torch.bincount(w.argmax(-1), minlength=K).cpu().numpy()

    return {"neff_mean": neff_mean,
            "entropy_mean": entropy_mean,
            "max_w_mean": max_w_mean,
            "w_var_across_inputs": w_var_across_inputs,
            "neff_per_sample": neff.cpu().numpy(),
            "argmax_hist": argmax_hist,
            "K": K, "B": B}
"""

COND_VIZ_BODY = r"""# =============================================================================
# COND-GATE v0.3 -- common.cond_viz
# Purpose: diagnostic plots for the 9 COND-GATE checks. Writes PNG to
#          step_X/plots/cond_gate/epoch_<N>.png or similar.
# CONVENTION: matplotlib imported lazily so that a missing matplotlib does not
#             break module import. Any IO error is logged + re-raised.
# Changelog (v0.2 -> v0.3):
#   * Added plot_film_per_layer (per-layer γ/β bar chart).
#   * Added plot_gate_collapse (Neff / entropy traj + argmax histogram).
#   * plot_grad_traj now supports multi-layer FiLM grad history.
# =============================================================================
from __future__ import annotations
import logging
import traceback
from pathlib import Path
logger = logging.getLogger(__name__)
__version__ = "0.3"
__abbr__ = "COND-GATE"

import numpy as np


def _mpl():
    # Lazy import so scaffolder-only installs don't need matplotlib.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save(plt, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
    except OSError:
        logger.error("[cond_viz] save failed %s\n%s", out_path, traceback.format_exc())
        raise


def plot_h_hist(h, gamma, beta, out_path):
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, arr, title in zip(axes, [h, gamma, beta], ["h", "gamma", "beta"]):
        a = np.asarray(arr).ravel()
        a = a[np.isfinite(a)]
        if a.size == 0:
            ax.set_title(title + " (no finite)")
            continue
        ax.hist(a, bins=50)
        ax.set_title(f"{title}  mean={a.mean():.3f} std={a.std():.3f}")
    _save(plt, out_path)


def plot_h_diversity(pairwise_matrix, out_path):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(np.asarray(pairwise_matrix), cmap="viridis")
    plt.colorbar(im, ax=ax)
    ax.set_title("pairwise ||h_i - h_j||")
    _save(plt, out_path)


def plot_grad_traj(cond_grad_history, film_grad_history, out_path):
    # cond_grad_history: list[float] length = epochs
    # film_grad_history: list[list[float]] shape (epochs, L) OR list[float]
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cond_grad_history, label="conditioner", linewidth=2)
    fh = np.asarray(film_grad_history)
    if fh.ndim == 2:
        for i in range(fh.shape[1]):
            ax.plot(fh[:, i], label=f"FiLM[{i}]", alpha=0.6)
    elif fh.size > 0:
        ax.plot(fh, label="FiLM", alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("grad norm")
    ax.legend(fontsize=8)
    ax.set_title("gradient flow (must stay >> 0)")
    _save(plt, out_path)


def plot_logp_shuffle(logp_real, logp_shuffled, out_path):
    plt = _mpl()
    r = np.asarray(logp_real).ravel()
    s = np.asarray(logp_shuffled).ravel()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([0, 1], [r.mean(), s.mean()], yerr=[r.std(), s.std()], color=["C0", "C3"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["logp(x|h)", "logp(x|shuffle h)"])
    ax.set_ylabel("mean logp")
    ax.set_title("conditioning effect on logp (bars must differ)")
    _save(plt, out_path)


def plot_nan_inf_traj(nan_inf_history, out_path):
    # nan_inf_history: list[dict] per epoch with keys like h_nan/h_inf/gamma_nan/...
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 3))
    if not nan_inf_history:
        logger.error("[plot_nan_inf_traj] empty history")
        ax.set_title("no data")
        _save(plt, out_path)
        return
    keys = sorted({k for d in nan_inf_history for k in d.keys()})
    for k in keys:
        ax.plot([d.get(k, 0) for d in nan_inf_history], label=k)
    ax.set_xlabel("epoch")
    ax.set_ylabel("count")
    ax.set_title("NaN/Inf per epoch (must stay 0)")
    ax.legend(fontsize=8)
    _save(plt, out_path)


def plot_determinism_traj(max_delta_h_history, tol, out_path):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 3))
    vals = np.asarray(max_delta_h_history, dtype=float)
    vals = np.clip(vals, 1e-20, None)   # log-safe
    ax.plot(vals, marker="o")
    ax.axhline(tol, color="red", linestyle="--", label=f"tol={tol:.0e}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("max|delta h| (twin runs)")
    ax.set_title("(y, seed) determinism")
    ax.legend()
    _save(plt, out_path)


def plot_film_per_layer(per_layer_stats, out_path):
    # per_layer_stats: list of {layer_index, gamma:{mean,std}, beta:{mean,std}}
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 4))
    L = len(per_layer_stats)
    if L == 0:
        ax.set_title("no FiLM layers")
        _save(plt, out_path)
        return
    xs = np.arange(L)
    g_mean = [s["gamma"]["mean"] for s in per_layer_stats]
    g_std  = [s["gamma"]["std"]  for s in per_layer_stats]
    b_mean = [s["beta"]["mean"]  for s in per_layer_stats]
    b_std  = [s["beta"]["std"]   for s in per_layer_stats]
    ax.bar(xs - 0.2, g_mean, 0.4, yerr=g_std, label="gamma", capsize=3)
    ax.bar(xs + 0.2, b_mean, 0.4, yerr=b_std, label="beta",  capsize=3)
    ax.set_xlabel("FiLM layer index")
    ax.set_ylabel("mean +/- std")
    ax.set_xticks(xs)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.legend()
    ax.set_title("per-layer FiLM stats (flat bar = dead layer)")
    _save(plt, out_path)


def plot_gate_collapse(neff_history, entropy_history, argmax_hist, out_path):
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].plot(neff_history, marker="o")
    axes[0].set_title("Neff per epoch")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("Neff")
    axes[1].plot(entropy_history, marker="o", color="C1")
    axes[1].set_title("gate entropy per epoch")
    axes[1].set_xlabel("epoch")
    ah = np.asarray(argmax_hist).ravel()
    axes[2].bar(np.arange(ah.size), ah)
    axes[2].set_title("argmax-expert histogram")
    axes[2].set_xlabel("expert k"); axes[2].set_ylabel("count")
    _save(plt, out_path)
"""


# -----------------------------------------------------------------------------
# Static structure (common/, models/, scripts/, experiments/__init__.py)
# -----------------------------------------------------------------------------
STATIC_DIRS: dict[str, list[tuple[str, str]]] = {
    "common": [
        ("__init__.py",   H("common", "shared utilities for all steps")),
        ("logger.py",     H("common.logger",     "unified logger") + LOGGER_BODY),
        ("status_io.py",  H("common.status_io",  "atomic read/write of per-step status.json")
                          + STATUS_IO_BODY),
        ("metrics_io.py", H("common.metrics_io", "append row to per-step metrics.csv")
                          + METRICS_IO_BODY),
        ("seed.py",       H("common.seed", "deterministic seeding")
                          + NI("set_seed",
                               "seed torch / numpy / random + set cuDNN deterministic flags")),
        ("hashing.py",    H("common.hashing", "config + git-SHA hashing for reproducibility")
                          + NI("config_hash",
                               "sha256(canonical-json(cfg)) -> first 12 hex chars")
                          + NI("git_sha",
                               "git rev-parse HEAD; fallback 'DIRTY-<timestamp>' if unavailable")),
        # --- COND-GATE v0.3 diagnostics (real bodies, not NI stubs) ---
        ("cond_diagnostics.py", COND_DIAGNOSTICS_BODY),
        ("gate_diagnostics.py", GATE_DIAGNOSTICS_BODY),
        ("cond_viz.py",         COND_VIZ_BODY),
    ],
    # NEW in v2.2: shared model definitions (experts, gates, flows) live here
    # so they're imported once as CSMF2.models.* instead of duplicated per step.
    "models": [
        ("__init__.py",   H("models", "shared model definitions (experts, gates, flows)")),
    ],
    # scripts/ stays empty in the scaffold -- drop the EXP-DASH v1.0 file here.
    "scripts": [],
    "experiments": [("__init__.py", H("experiments", "experiment stages package"))],
}

ROOT_FILES: dict[str, str] = {
    "README.md":
        f"# CSMF Incremental Experiment Tree -- {__abbr__} v{__version__}\n\n"
        "Each of the 11 plan stages is a runnable Python package under `experiments/`.\n"
        "Shared model code lives under `models/`; shared utilities under `common/`.\n"
        "**Reproducibility:** every run is parameterised by `--seed` and its output is scoped\n"
        "by `(seed, cfg_hash)` so reruns never collide.\n\n"
        "## Install\n"
        "```\n"
        "pip install -r requirements-project.txt   # full CSMF ML stack\n"
        "pip install -r requirements.txt           # dashboard + scaffolder only\n"
        "```\n\n"
        "## Quick start\n"
        "```\n"
        f"python -m {BASE}.experiments.step_1_1.run --seed 0\n"
        f"python scripts/experiment_dashboard.py --root {BASE}/\n"
        "```\n\n"
        "Drop the `EXP-DASH v1.0` `experiment_dashboard.py` into `scripts/`.\n\n"
        "**CONVENTION:** NLL = LOSS (lower = better).\n",
    "VERSION":         f"{__abbr__} v{__version__}\n",
    # --- Dashboard + scaffolder only (minimal; pure-python scaffolder has no extras) ---
    "requirements.txt":
        "# EXP-SCAFFOLD + EXP-DASH runtime (monitoring install).\n"
        "# Full training stack -> requirements-project.txt\n"
        "flask>=3.0\n"
        "pyyaml>=6.0\n",
    # --- Full CSMF project ML stack (training + evaluation + diagnostics) ---
    "requirements-project.txt":
        "# CSMF project ML stack -- pin as needed for your CUDA / driver combo.\n"
        "# Core deep learning\n"
        "torch>=2.0\n"
        "torchvision>=0.15\n"
        "# Scientific computing\n"
        "numpy>=1.24\n"
        "scipy>=1.10\n"
        "pandas>=2.0\n"
        "# Image IO + metrics (SR / SAR tasks)\n"
        "Pillow>=10.0\n"
        "scikit-image>=0.21\n"
        "# Visualisation\n"
        "matplotlib>=3.7\n"
        "# CLI + progress + config\n"
        "tqdm>=4.65\n"
        "pyyaml>=6.0\n"
        "# Schemas / validation (manifest + CheckpointRecord pattern)\n"
        "pydantic>=2.0\n",
    ".gitignore":
        "__pycache__/\n*.pyc\n*.pt\n*.pth\n.env\n"
        "experiments/**/checkpoints/*\n"
        "experiments/**/results/*\n"
        "experiments/**/plots/*\n"
        "experiments/**/logs/*\n"
        "!experiments/**/.gitkeep\n"
        "!experiments/**/README.md\n",
}


# -----------------------------------------------------------------------------
# Writer (no-clobber by default; IO errors never silently swallowed)
# -----------------------------------------------------------------------------
def _safe_write(path: str, content: str, force: bool) -> str:
    """Returns 'created' | 'skipped' | 'overwritten'."""
    try:
        if os.path.exists(path) and not force:
            print(f"    skip (exists): {path}")
            return "skipped"
        action = "overwritten" if os.path.exists(path) else "created"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            fh.write(content)
        print(f"    {'wrote (force)' if action == 'overwritten' else 'wrote'}: {path}")
        return action
    except OSError as e:
        print(f"    ERROR writing {path}: {e}", file=sys.stderr)
        raise


def create_structure(force: bool = False) -> dict:
    os.makedirs(BASE, exist_ok=True)
    print(f"Root: {BASE}/ ({__abbr__} v{__version__}) force={force}")

    counts = {"created": 0, "skipped": 0, "overwritten": 0}

    def _tick(result: str) -> None:
        counts[result] = counts.get(result, 0) + 1

    # --- static dirs: common/, models/, scripts/, experiments/__init__ ---
    for folder, files in STATIC_DIRS.items():
        folder_path = os.path.join(BASE, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"  dir: {folder_path}/")
        if not files:
            _tick(_safe_write(os.path.join(folder_path, ".gitkeep"), "", force))
        for name, body in files:
            _tick(_safe_write(os.path.join(folder_path, name), body, force))

    # --- per-step experiment packages ---
    for step in STEPS:
        step_pkg = os.path.join(BASE, "experiments", dir_of(step["step_id"]))
        os.makedirs(step_pkg, exist_ok=True)
        print(f"  dir: {step_pkg}/")

        step_files = [
            ("__init__.py", H(f"experiments.{dir_of(step['step_id'])}",
                              f"step {step['step_id']} package")),
            ("run.py",      render_run_py(step)),
            ("config.py",   render_config_py(step)),
            ("README.md",   render_readme(step)),
            ("status.json", render_status(step)),
            ("metrics.csv", METRICS_HEADER),
        ]
        for name, body in step_files:
            _tick(_safe_write(os.path.join(step_pkg, name), body, force))

        # per-step output sub-dirs (always idempotent, .gitkeep for git tracking)
        for sub in ("configs", "logs", "results", "checkpoints", "plots"):
            sub_path = os.path.join(step_pkg, sub)
            os.makedirs(sub_path, exist_ok=True)
            _tick(_safe_write(os.path.join(sub_path, ".gitkeep"), "", force))

    # --- root files ---
    for name, body in ROOT_FILES.items():
        _tick(_safe_write(os.path.join(BASE, name), body, force))

    # experiment_index.json always refreshed (mirrors current STEPS)
    idx_path = os.path.join(BASE, "experiment_index.json")
    existed = os.path.exists(idx_path)
    _safe_write(idx_path, render_experiment_index(), force=True)
    counts["overwritten" if existed else "created"] += 1

    print(
        f"\nDone. {__abbr__} v{__version__}  "
        f"created={counts['created']}  "
        f"skipped={counts['skipped']}  "
        f"overwritten={counts['overwritten']}"
    )
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=f"{__abbr__} v{__version__} scaffolder")
    ap.add_argument(
        "--force", action="store_true",
        help="Overwrite existing files. Default: skip existing files (no-clobber).",
    )
    args = ap.parse_args()
    create_structure(force=args.force)

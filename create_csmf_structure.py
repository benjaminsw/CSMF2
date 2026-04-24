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

__version__ = "2.2"
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

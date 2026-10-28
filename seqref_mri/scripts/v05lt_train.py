# SEQREF-MRI-V05LTT v0.5 -- scripts.v05lt_train
# LIFETIME: KEEP
# =============================================================================
# Purpose: one V05LT training run (arm x seed) -- 194 §4-§7 + AM2-A/B/C +
#   AM3; 195 §4, §5. Adapted from V03DLT/V03 (objective, lambda use, Adam,
#   constant LR, failure taxonomy) with the V05LT additions:
#   * data: split-manifest views (v05lt_common); training epoch e uses the
#     production set_epoch(e - 1); validation views use eval masks;
#   * epoch-0 identity: the freshly built model must equal the seed's
#     epoch-0 model-state hash before the first update;
#   * data order (AM2-A): base order L; a dedicated numpy.default_rng(seed)
#     draws ONE permutation per epoch at the start of that epoch; the
#     permutation's identity hash is ledgered; the generator state after the
#     draw is part of the resume state, so an interrupted-and-resumed run
#     draws the identical sequence;
#   * validation every epoch through v05lt_metrics (shared with I1.7);
#   * stopping (v05lt_rules.Plateau) and selection (v05lt_rules.Selection)
#     are independent;
#   * retention (194 §6.1): weights_e<NNN>.pt is staged once per epoch when
#     any role needs it (milestone / best PSNR / best NLL; overlapping roles
#     share one file); resume_e<NNN>.pt every epoch; a single store commit
#     per epoch lists every retained role; superseded files are deleted
#     only after that commit is durable (v05lt_common.ArtifactStore);
#   * validation rows (v0.4; I1.8 b0, G1): every epoch the per-slice rows
#     returned by the shared validation (the values that drive stopping
#     and selection) are staged as val_e<NNN>.json (schema below) BEFORE
#     that epoch's commit; the commit lists it as role val_rows_e<NNN>
#     (sha256) and records its epoch, row count and slice-identity hash.
#     Row files are retained for the life of the run (never superseded);
#     recovery verifies committed files and quarantines uncommitted ones
#     (ArtifactStore.recover, unchanged).
#   * full resume state (AM2-B): model, Adam, Python / NumPy-global /
#     torch-CPU RNG states, shuffle-generator state, next epoch and mask
#     epoch, step, plateau and selection state, ledger position; max 3
#     sessions per run;
#   * failure taxonomy (195 §1): non-finite loss/gradient ->
#     numerical_failure; structural breaches -> failed_validity; both
#     terminal and recorded in the ledger.
# Recovery rules (v0.2; declared):
#   * Refusals (no state change except a `resume_refused` record):
#     REFUSAL_CODES below. They leave the run resumable after the owner
#     acts (e.g. restores the environment).
#   * Terminal failures: any other V05LTError raised at initialisation,
#     resume validation or training -> `run_end` {status = taxonomy, code,
#     epoch, stage}; never resumed (RESUME_RUN_ENDED).
#   * Known production seams (v0.3, item 18) are typed at the point of
#     failure (_typed): data loading, prepare, mask/standardisation/
#     targets, objective, backward/optimiser, mask epoch, validation
#     summary, checkpoint stage/commit -> failed_validity (non-finite in
#     the objective/step -> numerical_failure), hence terminal.
#   * Unexpected (non-V05LT) exceptions outside those seams -> `session_error`
#     record, logged, re-raised; not terminal (owner decision). Process kills leave no
#     record by nature and are handled by resume.
#   * Interruption before the first epoch commit: resume quarantines
#     uncommitted files, records `restart_uncommitted` and restarts from
#     the seed-defined initial state (identical to a fresh start).
#   * A durable final commit (status != continue) without `run_end`:
#     resume records `run_end` (recovered: true) and refuses.
# Production entry is deliberately DISABLED in this version: main() refuses
#   (TRAIN_PRODUCTION_DISABLED) until the launcher/gate delivery (I1.9).
#   Fixtures call Run with a registered fixture config in temporary
#   directories; a fixture config can never point at results/v05lt/
#   (FIXTURE_TARGETS_PRODUCTION).
# `model_factory` / `hook` / `env_override` arguments exist only for
#   fixtures (stub or instrumented model, fault injection, incompatibility
#   tests); production passes none. Declared, not a fallback.
# CONVENTION: logger.error + V05LTError(code); no fallback, no placeholder.
# Changelog (v0.5, 2026-10-24; b0 review fix):
#   * load_val_rows validates the document's fields and types (and each
#     row's identity fields) before use: malformed documents ->
#     VAL_ROWS_INVALID, never a raw KeyError/TypeError.
# Changelog (v0.4, 2026-10-24; I1.8 b0, G1):
#   * Per-epoch validation-row files (val_e<NNN>.json) staged before the
#     epoch commit and retained permanently; commit field validation_rows.
#   * No change to the objective, data order, RNG use, stopping, selection
#     or the commit / recovery protocol.
# Changelog (v0.3, 2026-10-22; I1.7(c2), item 18):
#   * _typed seams: TRAIN_DATA_LOAD, TRAIN_PREPARE, TRAIN_TARGETS,
#     OBJECTIVE_FAILED (finite backend errors; non-finite -> NUMERICAL_FAILURE),
#     TRAIN_STEP_FAILED, VAL_SUMMARY_FAILED, CHECKPOINT_WRITE_FAILED.
#   * No change to the objective, data order, RNG use or commit protocol.
# Changelog (v0.2, 2026-10-21; I1.6(c), SEQREF-MRI-V05LTI16C v0.2, reviewer
#   findings 13-17):
#   * 13: complete resume-state validation -- commit records the resume-
#     state and Adam-state hashes; resume checks model, Adam, state hash,
#     epoch/mask epoch/step/rules vs the commit, RNG structures and the
#     ledger position (RESUME_STATE_MISMATCH, terminal failed_validity).
#   * 14: interruption before the first commit -> explicit restart; fresh
#     sessions seed the Python and NumPy global RNGs from the run seed so
#     the restart is identical to a fresh start.
#   * 15: commit carries `status`; final commit without run_end recognised.
#   * 16: run_start records code, environment and data identity; resume
#     compares all three (RESUME_INCOMPATIBLE, refusal).
#   * 17: one terminal-recording rule for initialisation, resume validation
#     and training; refusal and unexpected-error records.
#   * Memory: resume frees the loaded model/Adam tensors once copied;
#     checkpoint writers save without defensive copies (synchronous).
# Changelog (NEW in v0.1, 2026-10-21):
#   * Introduced under I1.6(b) (SEQREF-MRI-V05LTI16B v0.1): trainer,
#     retention, full-state resume, fixed per-epoch order, taxonomy.
# =============================================================================
from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from seqref_mri.scripts import v05lt_common as C
from seqref_mri.scripts import v05lt_metrics as M
from seqref_mri.scripts import v05lt_rules as R

logger = logging.getLogger("seqref_mri.v05lt_train")

__version__ = "0.5"
__abbr__ = "SEQREF-MRI-V05LTT"

ARMS = {"lam0": 0.0, "lam2": 0.6130341651761398}       # 194 §4
LEARNING_RATE = 1e-4                                    # 194 §6 (= V03)
BATCH = 32
MAX_SESSIONS = 3                                        # AM2-B
RUN_SCHEMA = "seqref-v05lt-run/2"
RESUME_SCHEMA = "seqref-v05lt-resume/1"
WEIGHTS_SCHEMA = "seqref-v05lt-weights/1"
VAL_ROWS_SCHEMA = "seqref-v05lt-val-rows/1"
VAL_ROLE_PREFIX = "val_rows_e"
INTERRUPTED_STATUS = "interrupted — awaiting owner decision"
TERMINAL = ("numerical_failure", "failed_validity")
REFUSAL_CODES = frozenset({
    "ARTIFACT_EXISTS", "RESUME_NO_RUN", "RESUME_CONFIG_MISMATCH",
    "RESUME_INCOMPATIBLE", "RESUME_RUN_ENDED", "SESSION_LIMIT"})
# Files whose identity defines "the trainer code" for resume compatibility.
CODE_REPO = Path(__file__).resolve().parents[2]   # repo holding this code
CODE_FILES = ("v05lt_train.py", "v05lt_common.py", "v05lt_metrics.py",
              "v05lt_rules.py", "v05lt_epoch0.py")


@dataclass
class TrainConfig:
    seed: int
    arm: str
    run_dir: str
    epoch0_model_state: str
    max_epochs: int = R.MAX_EPOCHS
    fixture: bool = False
    train_indices: list | None = None     # fixture subsets only
    val_indices: list | None = None
    note: str = ""
    extra: dict = field(default_factory=dict)

    def identity(self) -> dict:
        d = asdict(self)
        d.pop("run_dir")
        return d


def _fail_num(code, msg):
    C.fail(code, msg, taxonomy="numerical_failure")


def _nonfinite_text(exc) -> bool:
    t = f"{type(exc).__name__}: {exc}"
    return "non-finite" in t.lower() or "NON_FINITE" in t.upper()


def _typed(code: str, where: str, fn, *a, numeric: bool = False, **k):
    """Item 18: a known production seam. V05LT errors pass through; any
    other exception becomes a typed failure (registered taxonomy, 195 §1)
    instead of an untyped session_error. numeric=True: a non-finite
    failure here is a numerical breakdown of training (numerical_failure);
    otherwise every failure is failed_validity. BaseException (process
    kill / fixture interruption) is never caught."""
    try:
        return fn(*a, **k)
    except C.V05LTError:
        raise
    except Exception as exc:  # noqa: BLE001 -- typed, logged, re-raised
        tax = ("numerical_failure" if numeric and _nonfinite_text(exc)
               else "failed_validity")
        C.fail(code, f"{where}: {type(exc).__name__}: {exc}", taxonomy=tax)


def check_config(cfg: TrainConfig, paths: C.Paths) -> None:
    C.require(cfg.arm in ARMS, "TRAIN_ARM_INVALID", cfg.arm)
    C.require(isinstance(cfg.seed, int), "TRAIN_SEED_INVALID", repr(cfg.seed))
    run = Path(cfg.run_dir).resolve()
    prod_root = paths.v05lt.resolve()
    if cfg.fixture:
        C.require(prod_root not in run.parents and run != prod_root,
                  "FIXTURE_TARGETS_PRODUCTION", str(run))
        C.require(1 <= cfg.max_epochs <= R.MAX_EPOCHS, "FIXTURE_EPOCHS",
                  str(cfg.max_epochs))
    else:
        C.require(cfg.max_epochs == R.MAX_EPOCHS and cfg.train_indices is None
                  and cfg.val_indices is None, "PRODUCTION_CONFIG_ALTERED",
                  "production runs use the full registered budget and data")
        C.require(cfg.seed in (2026, 2027, 2028), "TRAIN_SEED_INVALID",
                  str(cfg.seed))


# ---------------------------------------------------------------------------
# RNG state (JSON-safe, exact)
# ---------------------------------------------------------------------------
def rng_state() -> dict:
    import torch
    py = random.getstate()
    npst = np.random.get_state()
    return {"python": [py[0], list(py[1]), py[2]],
            "numpy": [npst[0], npst[1].tolist(), int(npst[2]), int(npst[3]),
                      float(npst[4])],
            "torch_cpu": torch.get_rng_state().tolist()}


def set_rng_state(s: dict) -> None:
    """Restore exactly; any structural problem -> RESUME_STATE_MISMATCH."""
    import torch
    try:
        random.setstate((s["python"][0], tuple(s["python"][1]),
                         s["python"][2]))
        n = s["numpy"]
        np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2],
                             n[3], n[4]))
        torch.set_rng_state(torch.tensor(s["torch_cpu"], dtype=torch.uint8))
    except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as exc:
        C.fail("RESUME_STATE_MISMATCH", f"RNG state not restorable: {exc}")
    C.require(rng_state() == s, "RESUME_STATE_MISMATCH",
              "RNG state differs after restore")


def seed_global_rngs(seed: int) -> None:
    """Fresh sessions only (v0.2): Python and NumPy global RNGs from the run
    seed (torch CPU is seeded by build_model). Declared determinism measure;
    the training path itself draws from neither (F04/C)."""
    random.seed(int(seed))
    np.random.seed(int(seed))


def state_sha256(state: dict) -> str:
    return C.sha256_bytes(C.canonical_json(state))


def scientific_state_sha256(state: dict) -> str:
    """Resume state without its ledger position (session-dependent by
    design); used to compare interrupted and uninterrupted runs."""
    return state_sha256({k: v for k, v in state.items()
                         if k != "ledger_seq_before_commit"})


def optimizer_state_sha256(od: dict) -> str:
    """Serialisation-independent Adam state identity: param_groups as
    canonical JSON, then every per-parameter entry (sorted) with dtype,
    shape and raw bytes for tensors, repr for scalars."""
    import torch
    h = hashlib.sha256(b"seqref-v05lt-optimizer-state/1\n")
    h.update(C.canonical_json(od["param_groups"]))
    for pid in sorted(od["state"]):
        for name in sorted(od["state"][pid]):
            v = od["state"][pid][name]
            if isinstance(v, torch.Tensor):
                t = v.detach().to("cpu").contiguous()
                h.update(f"{pid}|{name}|{t.dtype}|{tuple(t.shape)}\n".encode())
                h.update(t.numpy().tobytes() if t.dtype != torch.bfloat16
                         else t.view(torch.int16).numpy().tobytes())
            else:
                h.update(f"{pid}|{name}|{v!r}\n".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Data order (AM2-A)
# ---------------------------------------------------------------------------
def order_sha256(view, order) -> str:
    h = hashlib.sha256()
    for i in order:
        rel, s = view.entry(int(i))
        h.update(f"{rel}|{s}\n".encode("utf-8"))
    return h.hexdigest()


def draw_epoch_order(order_rng, n: int) -> list:
    return [int(i) for i in order_rng.permutation(n)]


# ---------------------------------------------------------------------------
# One epoch of updates (V03 objective unchanged)
# ---------------------------------------------------------------------------
def _train_epoch(model, opt, tv, positions, p4, lam, step0, hook=None):
    import torch
    from seqref_mri.scripts import train_free_flow as tff
    from seqref_mri.scripts.train_base import _collate, _prepare
    from seqref_mri.scripts.train_free_flow import targets_from_prepared
    from seqref_mri.scripts.v02_train import derive_cmap_from_mask
    from seqref_mri.scripts.v03_probe import global_grad_norm
    from seqref_mri.src import free_flow_runtime as ffr
    from seqref_mri.src.flows import v03_objective as v03l
    model.train()
    step, sums = step0, {"L_density": 0.0, "L_rec": 0.0, "L_total": 0.0,
                         "grad_norm_max": 0.0}
    n_batches = 0
    for start in range(0, len(positions), BATCH):
        chunk = positions[start:start + BATCH]
        batch = _typed("TRAIN_DATA_LOAD", f"batch at {start}",
                       lambda: _collate([tv[i] for i in chunk]))
        prep = _typed("TRAIN_PREPARE", f"batch at {start}", _prepare, batch,
                      "cpu", test0=False)

        def build_targets():
            ts, ms = [], []
            for j in range(len(batch["meta"])):
                cmap = derive_cmap_from_mask(batch["mask"][j])
                vecs = ffr.standardisation_vectors(cmap,
                                                   p4["location_index"])
                one = {"x_norm": prep["x_norm"][j:j + 1],
                       "cond_in": prep["cond_in"][j:j + 1]}
                ts.append(targets_from_prepared(one, cmap, vecs))
                ms.append((cmap, vecs))
            return torch.cat(ts, 0), ms
        tgt, maps = _typed("TRAIN_TARGETS", f"mask/standardisation, batch at "
                           f"{start}", build_targets)
        cond, msk = prep["cond_in"], batch["mask"]
        step += 1
        if hook is not None:
            hook(f"train:before_step:{step}")
        opt.zero_grad(set_to_none=True)
        try:
            nll = tff.nll_objective(model, tgt, cond, msk)
            if lam == 0.0:
                loss, rec = nll, None
            else:
                rec = v03l.l_rec(model, cond, msk, prep["y"], prep["amax"],
                                 prep["tgt_norm"], maps)
                loss = nll + lam * rec
        except v03l.V03LError as exc:
            C.fail(exc.code, f"objective failure at step {step}: {exc}",
                   taxonomy=("numerical_failure" if exc.code == "U_NON_FINITE"
                             else "failed_validity"))
        except ffr.FreeFlowError as exc:          # production runtime gates
            C.fail(exc.code if hasattr(exc, "code") else "FREEFLOW_ERROR",
                   f"step {step}: {exc}",
                   taxonomy=("numerical_failure" if "NON_FINITE"
                             in str(getattr(exc, "code", "")) else
                             "failed_validity"))
        except C.V05LTError:
            raise
        except Exception as exc:  # noqa: BLE001 -- NSF backend etc.; typed
            if _nonfinite_text(exc):
                _fail_num("NUMERICAL_FAILURE", f"step {step}: {exc}")
            C.fail("OBJECTIVE_FAILED", f"step {step}: {type(exc).__name__}: "
                   f"{exc}")
        if not torch.isfinite(loss):
            _fail_num("NUMERICAL_FAILURE", f"step {step}: objective non-finite")
        _typed("TRAIN_STEP_FAILED", f"backward, step {step}", loss.backward,
               numeric=True)
        gnorm = global_grad_norm(model)
        if not np.isfinite(gnorm):
            _fail_num("NUMERICAL_FAILURE", f"step {step}: gradient non-finite")
        _typed("TRAIN_STEP_FAILED", f"optimiser, step {step}", opt.step,
               numeric=True)
        if hook is not None:
            hook(f"train:step:{step}")
        sums["L_density"] += float(nll)
        sums["L_rec"] += 0.0 if rec is None else float(rec)
        sums["L_total"] += float(loss)
        sums["grad_norm_max"] = max(sums["grad_norm_max"], float(gnorm))
        n_batches += 1
    for k in ("L_density", "L_rec", "L_total"):
        sums[k] /= max(n_batches, 1)
    return step, n_batches, sums


# ---------------------------------------------------------------------------
# Checkpoint payloads
# ---------------------------------------------------------------------------
def _weights_writer(model, meta):
    """The writer runs synchronously inside ArtifactStore.stage, before any
    further update, so tensors are saved without a defensive copy."""
    import torch
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    return lambda fh: torch.save({"schema": WEIGHTS_SCHEMA, "model": sd,
                                  "meta": meta}, fh)


def slice_ids_sha256(rows: list) -> str:
    """Identity of the validated slice population, in validation order."""
    return C.sha256_bytes(C.canonical_json(
        [[r["volume_id"], r["file"], r["slice_index"]] for r in rows]))


def val_rows_doc(cfg, epoch: int, msh: str, rows: list) -> dict:
    return {"schema": VAL_ROWS_SCHEMA, "seed": cfg.seed, "arm": cfg.arm,
            "epoch": epoch, "model_state_sha256": msh,
            "metrics_version": M.__version__,
            "nll_decomp_tol_rel": M.NLL_DECOMP_TOL_REL,
            "n_rows": len(rows), "slice_ids_sha256": slice_ids_sha256(rows),
            "rows": rows}


_VAL_ROWS_FIELDS = {"schema": str, "seed": int, "arm": str, "epoch": int,
                    "model_state_sha256": str, "metrics_version": str,
                    "nll_decomp_tol_rel": float, "n_rows": int,
                    "slice_ids_sha256": str, "rows": list}
_ROW_ID_FIELDS = {"volume_id": str, "file": str, "slice_index": int}


def _typed_ok(d: dict, spec: dict) -> list:
    """Names of fields in spec missing from d or of the wrong type (bool is
    never accepted as int)."""
    return [k for k, t in spec.items() if k not in d or
            isinstance(d[k], bool) or not isinstance(d[k], t)]


def load_val_rows(path, expected_sha256: str, epoch: int) -> dict:
    """Committed val_e<NNN>.json -> verified document. Order: file hash
    (VAL_ROWS_HASH_MISMATCH); JSON; exact field set and types; row identity
    fields; schema, epoch, n_rows and slice_ids_sha256 recomputed
    (VAL_ROWS_INVALID). Never a raw exception for a malformed document."""
    import json
    from pathlib import Path
    path = Path(path)
    got = C.sha256_file(path)
    C.require(got == expected_sha256, "VAL_ROWS_HASH_MISMATCH",
              f"{path.name}: {got} != committed {expected_sha256}")
    try:
        doc = json.loads(path.read_bytes())
    except ValueError as exc:
        C.fail("VAL_ROWS_INVALID", f"{path.name}: {exc}")
    C.require(isinstance(doc, dict) and set(doc) == set(_VAL_ROWS_FIELDS),
              "VAL_ROWS_INVALID", f"{path.name}: field set "
              f"{sorted(doc) if isinstance(doc, dict) else type(doc)}")
    bad = _typed_ok(doc, _VAL_ROWS_FIELDS)
    C.require(not bad, "VAL_ROWS_INVALID", f"{path.name}: field types {bad}")
    badrows = [i for i, r in enumerate(doc["rows"])
               if not isinstance(r, dict) or _typed_ok(r, _ROW_ID_FIELDS)]
    C.require(not badrows, "VAL_ROWS_INVALID",
              f"{path.name}: row identity fields, rows {badrows[:5]}")
    C.require(doc["schema"] == VAL_ROWS_SCHEMA and doc["epoch"] == epoch and
              doc["n_rows"] == len(doc["rows"]) and
              doc["slice_ids_sha256"] == slice_ids_sha256(doc["rows"]),
              "VAL_ROWS_INVALID", f"{path.name}: schema/epoch/rows")
    return doc


def _resume_writer(model, od, state):
    import torch
    sd = {k: v.detach() for k, v in model.state_dict().items()}
    return lambda fh: torch.save({"schema": RESUME_SCHEMA, "model": sd,
                                  "adam": od, "state": state}, fh)


def load_resume(path: Path):
    import torch
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 -- converted to a typed failure
        C.fail("RESUME_FILE_INVALID", f"{path}: {type(exc).__name__}: {exc}")
    C.require(isinstance(obj, dict) and obj.get("schema") == RESUME_SCHEMA
              and set(obj) == {"schema", "model", "adam", "state"},
              "RESUME_FILE_INVALID", str(path))
    return obj


# ---------------------------------------------------------------------------
# Identity records for resume compatibility (finding 16)
# ---------------------------------------------------------------------------
def _git(repo: Path, *args) -> str:
    import subprocess
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                       text=True)
    C.require(r.returncode == 0, "GIT_FAILED", r.stderr.strip())
    return r.stdout.strip()


def code_identity(repo: Path, model_factory) -> dict:
    here = Path(__file__).resolve().parent
    out = {"abbr": __abbr__, "version": __version__,
           "files": {f: C.sha256_file(here / f) for f in CODE_FILES},
           "repo_head": _git(repo, "rev-parse", "HEAD"),
           "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
           "model_factory": "production:free_flow_runtime.build_model"}
    if model_factory is not None:
        mod = getattr(model_factory, "__module__", "?")
        out["model_factory"] = (f"fixture:{mod}."
                                f"{getattr(model_factory, '__qualname__', '?')}")
        mfile = getattr(sys.modules.get(mod), "__file__", None)
        if mfile:
            out["model_factory_sha256"] = C.sha256_file(Path(mfile))
    return out


def _p4_identity(p4: dict) -> dict:
    if "file_sha256" in p4:
        return {"file_sha256": p4["file_sha256"],
                "semantic_sha256": p4.get("semantic_sha256")}
    rows = [p4["location_index"][k] for k in sorted(p4["location_index"])]
    return {"fixture_location_index_sha256": state_sha256({"rows": rows})}


def _eligibility_sha256(elig: dict) -> str:
    rows = [[f, s, v["eligible"], v["midpoint"], v["volume_id"]]
            for (f, s), v in sorted(elig.items())]
    return state_sha256({"rows": rows})


def data_identity(cfg: TrainConfig, data: dict, positions: list) -> dict:
    vv = data["vv"]
    return {"train_base_order_sha256": order_sha256(data["tv"], positions),
            "n_train": len(positions),
            "val_order_sha256": vv.order_lines_sha256(),
            "n_val": len(vv), "val_indices": cfg.val_indices,
            "eligibility_sha256": _eligibility_sha256(data["eligibility"]),
            "p4": _p4_identity(data["p4"]),
            "s_ref": repr(float(data["s_ref"]))}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
class Run:
    def __init__(self, cfg: TrainConfig, paths: C.Paths, data: dict,
                 model_factory=None, hook=None, env_override=None):
        """data: {"tv", "vv", "p4", "s_ref", "eligibility"} prepared by the
        caller (production: registered inputs; fixtures: temp data)."""
        check_config(cfg, paths)
        self.cfg, self.paths, self.data = cfg, paths, data
        self.hook = hook
        self.model_factory = model_factory
        self.env_override = env_override
        self.dir = Path(cfg.run_dir)
        self.ledger = C.Ledger(self.dir / "ledger.jsonl")
        self.store = None

    # -- helpers ----------------------------------------------------------
    def _build_model(self):
        if self.model_factory is not None:
            return self.model_factory(self.cfg.seed)
        from seqref_mri.src import free_flow_runtime as ffr
        return ffr.build_model(init_seed=self.cfg.seed)

    def _train_positions(self) -> list:
        tv = self.data["tv"]
        return list(range(len(tv))) if self.cfg.train_indices is None \
            else list(self.cfg.train_indices)

    def _environment(self) -> dict:
        from seqref_mri.scripts import v05lt_epoch0 as E0
        return E0.environment() if self.env_override is None \
            else self.env_override

    def _identities(self) -> dict:
        return {"implementation": code_identity(CODE_REPO,
                                                self.model_factory),
                "env": self._environment(),
                "data": data_identity(self.cfg, self.data,
                                      self._train_positions())}

    def _sessions(self) -> int:
        return sum(1 for r in self.ledger.records()
                   if r["kind"] == "session_start")

    def _record_end(self, status: str, code: str | None, epoch: int,
                    **extra) -> None:
        self.ledger.append("run_end", {"status": status, "code": code,
                                       "epoch": epoch, **extra})

    def _record_terminal(self, exc: C.V05LTError, epoch: int,
                         stage: str) -> None:
        """Finding 17: one rule for every stage. Never masks `exc`."""
        try:
            self._record_end(exc.taxonomy, exc.code, epoch, stage=stage)
        except Exception as rec_exc:  # noqa: BLE001 -- logged, original re-raised
            logger.error("[%s] TERMINAL_RECORD_FAILED: %s while recording %s",
                         __abbr__, rec_exc, exc.code)

    def _record_refusal(self, exc: C.V05LTError) -> None:
        try:
            self.ledger.append("resume_refused", {"code": exc.code,
                                                  "message": str(exc)})
        except Exception as rec_exc:  # noqa: BLE001 -- logged, original re-raised
            logger.error("[%s] REFUSAL_RECORD_FAILED: %s", __abbr__, rec_exc)

    def _classify(self, exc: C.V05LTError) -> None:
        if exc.code in REFUSAL_CODES:
            if self.dir.is_dir() and self.ledger.path.exists():
                self._record_refusal(exc)
        elif exc.taxonomy in TERMINAL:
            self._record_terminal(exc, self._epoch, self._stage)

    def _guard(self, fn):
        """Run fn(); classify failures (refusal / terminal / unexpected)
        at the current stage (self._stage) and epoch (self._epoch)."""
        try:
            return fn()
        except C.V05LTError as exc:
            self._classify(exc)
            raise
        except Exception as exc:  # noqa: BLE001 -- recorded, logged, re-raised
            logger.exception("[%s] SESSION_ERROR at %s", __abbr__,
                             self._stage)
            try:
                self.ledger.append("session_error", {
                    "stage": self._stage, "epoch": self._epoch, "type": type(exc).__name__,
                    "message": str(exc)[:500]})
            except Exception as rec_exc:  # noqa: BLE001 -- logged
                logger.error("[%s] SESSION_ERROR_RECORD_FAILED: %s",
                             __abbr__, rec_exc)
            raise

    # -- entry points -----------------------------------------------------
    def start(self) -> dict:
        C.require(not self.dir.exists(), "ARTIFACT_EXISTS",
                  f"{self.dir} exists; a fresh start is refused (resume mode "
                  "is explicit)")
        self.dir.mkdir(parents=True)
        self.store = C.ArtifactStore(self.dir / "ckpt", self.ledger,
                                     self.dir / "_incomplete")
        ids = self._identities()
        self.ledger.append("run_start", {
            "schema": RUN_SCHEMA, "config": self.cfg.identity(),
            "lambda": ARMS[self.cfg.arm], "lr": LEARNING_RATE, "batch": BATCH,
            "base_order_sha256": ids["data"]["train_base_order_sha256"],
            "n_train": ids["data"]["n_train"], **ids})
        self._epoch, self._stage = 0, "initialisation"
        return self._guard(lambda: self._session(resume=None))

    def resume(self) -> dict:
        C.require(self.dir.is_dir() and self.ledger.path.exists(),
                  "RESUME_NO_RUN", str(self.dir))
        self.store = C.ArtifactStore(self.dir / "ckpt", self.ledger,
                                     self.dir / "_incomplete")
        self._epoch, self._stage = 0, "resume_validation"
        body = self._guard(self.store.recover)
        self._epoch = body.get("epoch", 0) if body else 0
        return self._guard(lambda: self._resume_checked(body))

    def _resume_checked(self, body: dict) -> dict:
        recs = self.ledger.records()
        C.require(recs and recs[0]["kind"] == "run_start", "RESUME_NO_RUN",
                  "ledger has no run_start")
        start = recs[0]["body"]
        C.require(start["config"] == self.cfg.identity(),
                  "RESUME_CONFIG_MISMATCH", "run configuration differs")
        ends = [r for r in recs if r["kind"] == "run_end"]
        if ends:
            C.fail("RESUME_RUN_ENDED", f"run already ended: {ends[-1]['body']}")
        if body and body.get("status") != R.CONTINUE:          # finding 15
            self._record_end(body["status"], None, body["epoch"],
                             recovered=True, stage="resume_validation")
            C.fail("RESUME_RUN_ENDED", f"final commit (status "
                   f"{body['status']!r}, epoch {body['epoch']}) found without "
                   "run_end; run_end recorded")
        ids = self._identities()                                # finding 16
        diffs = [k for k in ("implementation", "env", "data")
                 if start.get(k) != ids[k]]
        if diffs:
            for k in diffs:
                logger.error("[%s] resume %s differs:\n start  %s\n now    %s",
                             __abbr__, k, start.get(k), ids[k])
            C.fail("RESUME_INCOMPATIBLE", f"{diffs} differ from run_start; "
                   "owner decision required")
        if self._sessions() >= MAX_SESSIONS:
            self._record_end(INTERRUPTED_STATUS, "SESSION_LIMIT",
                             body.get("epoch", 0) if body else 0)
            C.fail("SESSION_LIMIT", f"{MAX_SESSIONS} sessions used; "
                   f"status '{INTERRUPTED_STATUS}'")
        if not body:                                            # finding 14
            self._stage = "initialisation"
            logger.warning("[%s] RESUME_RESTART_UNCOMMITTED: no epoch was "
                           "committed; restarting from the seed-defined "
                           "initial state", __abbr__)
            self.ledger.append("restart_uncommitted", {})
            return self._session(resume=None)
        return self._session(resume=body)

    # -- resume validation (finding 13) -----------------------------------
    def _restore(self, model, commit: dict):
        import torch
        rname = commit["retained"]["resume"]["name"]
        obj = load_resume(self.store.dir / rname)
        st, epoch = obj["state"], commit["epoch"]

        def need(cond, what):
            C.require(cond, "RESUME_STATE_MISMATCH", f"epoch {epoch}: {what}")
        need(state_sha256(st) == commit.get("resume_state_sha256"),
             "resume-state hash != committed hash")
        need(set(st) == {"next_epoch", "mask_epoch_next", "step", "rng",
                         "order_rng", "plateau", "selection",
                         "ledger_seq_before_commit"}, "resume-state keys")
        need(st["next_epoch"] == epoch + 1 and st["mask_epoch_next"] == epoch,
             "next epoch / mask epoch")
        need(st["step"] == commit["step"], "step")
        need(st["plateau"] == commit["plateau"]
             and st["selection"] == commit["selection"], "rule state")
        recs = self.ledger.records()
        seq = st["ledger_seq_before_commit"]
        need(isinstance(seq, int) and 0 <= seq < len(recs)
             and recs[seq]["kind"] == "commit"
             and recs[seq]["body"].get("epoch") == epoch
             and recs[seq]["body"].get("resume_state_sha256")
             == commit["resume_state_sha256"], "ledger position")
        need(optimizer_state_sha256(obj["adam"])
             == commit.get("adam_state_sha256"), "Adam-state hash")
        try:
            model.load_state_dict(obj["model"], strict=True)
        except (RuntimeError, KeyError) as exc:
            C.fail("RESUME_STATE_MISMATCH", f"model state: {exc}")
        need(C.model_state_sha256(model) == commit["model_state_sha256"],
             "model-state hash")
        del obj["model"]              # copied into the model; free memory
        opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        try:
            opt.load_state_dict(obj.pop("adam"))
        except (ValueError, KeyError, RuntimeError) as exc:
            C.fail("RESUME_STATE_MISMATCH", f"Adam state: {exc}")
        gc.collect()
        groups = opt.state_dict()["param_groups"]
        need(len(groups) == 1 and groups[0]["lr"] == LEARNING_RATE,
             "Adam param groups")
        steps = {float(s["step"]) for s in opt.state_dict()["state"].values()}
        need(steps == {float(st["step"])}, f"Adam steps {sorted(steps)[:3]}")
        cfg = self.cfg
        order_rng = np.random.default_rng(cfg.seed)
        try:
            need(st["order_rng"].get("bit_generator") == "PCG64",
                 "shuffle generator type")
            order_rng.bit_generator.state = st["order_rng"]
        except (KeyError, TypeError, ValueError) as exc:
            C.fail("RESUME_STATE_MISMATCH", f"shuffle generator: {exc}")
        need(order_rng.bit_generator.state == st["order_rng"],
             "shuffle generator state")
        plateau = R.Plateau.from_dict(st["plateau"])
        selection = R.Selection.from_dict(st["selection"], plateau.last_epoch)
        need(plateau.last_epoch == epoch, "plateau epoch")
        set_rng_state(st["rng"])
        return opt, order_rng, plateau, selection, st["next_epoch"], st["step"]

    # -- one session --------------------------------------------------------
    def _session(self, resume) -> dict:
        import torch
        from seqref_mri.scripts import v05lt_epoch0 as E0
        n_sess = self._sessions() + 1
        cfg, tv, vv = self.cfg, self.data["tv"], self.data["vv"]
        lam = ARMS[cfg.arm]
        if resume is None:
            seed_global_rngs(cfg.seed)
        model = self._build_model()
        if resume is None:
            init_rng = state_sha256(rng_state())
            self.ledger.append("session_start", {"session": n_sess,
                                                 "mode": "fresh",
                                                 "initial_rng_sha256": init_rng})
            E0.assert_matches_epoch0(model, cfg.epoch0_model_state)
            opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            order_rng = np.random.default_rng(cfg.seed)
            plateau, selection = R.Plateau(), R.Selection()
            next_epoch, step = 1, 0
        else:
            self.ledger.append("session_start", {"session": n_sess,
                                                 "mode": "resume",
                                                 "from_epoch": resume["epoch"]})
            (opt, order_rng, plateau, selection, next_epoch,
             step) = self._restore(model, resume)
        pos = self._train_positions()
        status = R.CONTINUE
        self._epoch, self._stage = next_epoch - 1, "training"
        for epoch in range(next_epoch, cfg.max_epochs + 1):
            self._epoch = epoch
            order = draw_epoch_order(order_rng, len(pos))
            positions = [pos[i] for i in order]
            o_sha = order_sha256(tv, positions)
            _typed("TRAIN_DATA_LOAD", f"mask epoch {epoch}",
                   tv.set_training_epoch, epoch)
            t0 = time.time()
            step, nb, losses = _train_epoch(model, opt, tv, positions,
                                            self.data["p4"], lam, step,
                                            self.hook)
            rows = M.validate(model, vv, self.data["eligibility"],
                              self.data["p4"], self.data["s_ref"],
                              cfg.val_indices)
            summ = _typed("VAL_SUMMARY_FAILED", f"epoch {epoch}",
                          M.summarise, rows)
            psnr = summ["all_slices"]["psnr"]
            nll = summ["all_slices"]["nll_per_dim"]
            status = plateau.update(epoch, psnr, nll)
            changed = selection.update(epoch, psnr, nll)
            if status == R.CONTINUE and epoch == cfg.max_epochs:
                status = "fixture_budget_end"
            self._commit_epoch(model, opt, order_rng, epoch, step, plateau,
                               selection, changed, status, {
                                   "order_sha256": o_sha, "batches": nb,
                                   "losses": losses, "validation": summ,
                                   "session": n_sess,
                                   "seconds": time.time() - t0}, rows)
            if status != R.CONTINUE:
                if self.hook is not None:
                    self.hook("run:before_run_end")
                self._record_end(status, None, epoch, stage=self._stage)
                break
        return {"status": status, "epoch": self._epoch, "step": step,
                "selection": selection.to_dict(),
                "plateau": plateau.to_dict()}

    def _commit_epoch(self, model, opt, order_rng, epoch, step, plateau,
                      selection, changed, status, record, val_rows) -> None:
        msh = C.model_state_sha256(model)
        prev = self.store.committed()
        retained = {k: v["name"] for k, v in prev.items()
                    if k.startswith("milestone_")
                    or k.startswith(VAL_ROLE_PREFIX)}
        vdoc = val_rows_doc(self.cfg, epoch, msh, val_rows)
        vname = f"val_e{epoch:03d}.json"
        vdata = C.canonical_json(vdoc)
        vsha = _typed("CHECKPOINT_WRITE_FAILED", f"validation rows {vname}",
                      self.store.stage, vname,
                      lambda fh: fh.write(vdata), self.hook)
        retained[f"{VAL_ROLE_PREFIX}{epoch:03d}"] = vname
        record = dict(record, validation_rows={
            "name": vname, "sha256": vsha, "epoch": epoch,
            "n_rows": vdoc["n_rows"],
            "slice_ids_sha256": vdoc["slice_ids_sha256"]})
        roles = []
        if R.is_milestone(epoch):
            roles.append(f"milestone_{epoch:03d}")
        if changed["psnr"]:
            roles.append("best_psnr")
        if changed["nll"]:
            roles.append("best_nll")
        for k in ("best_psnr", "best_nll"):
            if k not in roles and k in prev:
                retained[k] = prev[k]["name"]
        if roles:
            wname = f"weights_e{epoch:03d}.pt"
            meta = {"seed": self.cfg.seed, "arm": self.cfg.arm,
                    "lambda": ARMS[self.cfg.arm], "epoch": epoch, "step": step,
                    "model_state_sha256": msh, "roles": roles}
            _typed("CHECKPOINT_WRITE_FAILED", f"weights {wname}",
                   self.store.stage, wname, _weights_writer(model, meta),
                   self.hook)
            for r in roles:
                retained[r] = wname
        state = {"next_epoch": epoch + 1, "mask_epoch_next": epoch,
                 "step": step, "rng": rng_state(),
                 "order_rng": order_rng.bit_generator.state,
                 "plateau": plateau.to_dict(), "selection": selection.to_dict(),
                 "ledger_seq_before_commit": len(self.ledger.records())}
        od = opt.state_dict()
        rname = f"resume_e{epoch:03d}.pt"
        _typed("CHECKPOINT_WRITE_FAILED", f"resume {rname}", self.store.stage,
               rname, _resume_writer(model, od, state), self.hook)
        retained["resume"] = rname
        _typed("CHECKPOINT_WRITE_FAILED", f"commit epoch {epoch}",
               self.store.commit, retained, {
            "epoch": epoch, "step": step, "status": status,
            "model_state_sha256": msh,
            "adam_state_sha256": optimizer_state_sha256(od),
            "resume_state_sha256": state_sha256(state),
            "resume_scientific_sha256": scientific_state_sha256(state),
            "plateau": plateau.to_dict(), "selection": selection.to_dict(),
            "roles_this_epoch": roles, **record}, self.hook)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    argparse.ArgumentParser(description="V05LT trainer").parse_args(argv)
    try:
        C.fail("TRAIN_PRODUCTION_DISABLED", "production training is entered "
               "only through the gated launcher (I1.9, 195 §8.3); this "
               "version exposes run objects to fixtures only")
    except C.V05LTError as exc:
        print(f"TRAINER REFUSED {exc.code}")
        return 2


if __name__ == "__main__":
    sys.exit(main())

# SEQREF-MRI-V05LTHO v0.2 -- scripts.v05lt_holdout
# LIFETIME: KEEP
# =============================================================================
# Purpose: the gated V05LT holdout runner (I1.9 c3; plan
#   SEQREF-MRI-V05LTI19C3 v0.1, core + A + B + C + D, reviewer refinements
#   1-5; 195 §9.3; 196 b3 registration). It evaluates the frozen selected
#   and epoch-3 checkpoints of the six runs on the 199-volume V02 holdout
#   and writes the b3 records (schema seqref-v05lt-holdout-eval/1).
#   Order (refinement 2, no contact before permission):
#     1. approval record: its sha256 must equal the EXTERNALLY approved value
#        (recorded in 196); exact schema; mode = stage mode;
#     2. launch chain (approved sha256 = the approval's binding) and the
#        launch-inputs record through ProductionStages.bind_chain (pin,
#        loaded modules, backend, inputs);
#     3. analysis-chain manifest (F18, v05lt_evidence_check.check_chain;
#        approved sha256 = the approval's binding; must carry the
#        holdout_config role): selection manifest, validation and frozen
#        holdout configurations;
#     4. holdout manifest (metadata): file + canonical content sha256 and the
#        expected mask-seed identity = approval = frozen configuration;
#     5. `gate_passed` appended to the holdout ledger -- only then is any
#        holdout MRI file opened.
#   Scope (refinement 3): selected and epoch 3 of every run (final, epoch 0
#   refused); a run whose selected and epoch-3 checkpoints are the same model
#   is evaluated ONCE and one record per role is written from that
#   evaluation (both role bindings preserved).
#   Measurements: R0 (z = 0) and primary-bank R1 (exactly 128 decodes,
#   registered u-space mean), PSNR (+ cap flags), SSIM, NMSE_u under AM4-B
#   (rule v02_holdout_all: exact zero free-space energy = validity error),
#   NLL per free real dimension on R0 (production log_prob_free / d_free).
#   Width check (item 27): per slice, width of the loaded k-space = recorded
#   width; loaded mask and seed = make_cartesian_mask(width,
#   canonical_mask_seed(EVAL_BASE_SEED, file, slice)); recorded mask
#   identity = recomputation.
#   Outputs (refinement 4): one directory per evaluation, staged and
#   published with renameat2(RENAME_NOREPLACE); classified absent /
#   complete / incomplete / mismatched; complete outputs verified and
#   reused; incomplete or mismatched outputs and failed tasks are recorded
#   halts; a retry needs a decision bound to that halt AND the same
#   approval; the failed attempt is moved to _failed/ (never deleted).
#   Gated invocation (refinement 5): `run` with --approval PATH:SHA256,
#   --launch-chain, --analysis-chain [--decision PATH:SHA256]; production
#   paths registered (holdout manifest, data root, results/v05lt/holdout).
# CONVENTION: logger.error + V05LTError(code); no fallback, no placeholder,
#   no mock. Fixture-only parameters: `launch_root` (separate fixture launch
#   layout root) and `hooks` (fault injection); refused in production.
# Changelog (v0.2, 2026-10-27; revision-1 review):
#   * One post-gate stage boundary (Runner._stage): plan, data loading and
#     width checks, classification, existing-output verification,
#     parent / state preparation, evaluation and the index all record a
#     halt on a typed failure or an unexpected ordinary exception
#     (HOLDOUT_STAGE_ERROR with traceback), so clearing a fault alone
#     never resumes the stage.
#   * check_out_dir before any directory, lock or ledger is created:
#     symlinked output root refused; containment on the resolved
#     destination (fixture alias into results/v05lt refused); production:
#     registered root with resolved destination = repo/<registered>
#     (redirection by a symlinked root or parent refused); ledger / lock /
#     index / _failed entries
#     must not be symlinks.
# Changelog (NEW in v0.1, 2026-10-27; SEQREF-MRI-V05LTI19C3 v0.1):
#   * Introduced: approval record (make / verify), gate in no-contact order,
#     holdout view (val split, manifest order only after the gate), width /
#     mask check, task plan with coincident-checkpoint reuse, evaluation and
#     write-once publication, output classification and verification,
#     holdout ledger, recorded halts, bound retry / halt decisions with
#     failed-attempt retention, index file, CLI (approve, decide, run,
#     status).
# =============================================================================
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from seqref_mri.scripts import v05lt_common as C

logger = logging.getLogger("seqref_mri.v05lt_holdout")

__version__ = "0.2"
__abbr__ = "SEQREF-MRI-V05LTHO"

APPROVAL_SCHEMA = "seqref-v05lt-holdout-approval/1"
DECISION_SCHEMA = "seqref-v05lt-holdout-decision/1"
INDEX_SCHEMA = "seqref-v05lt-holdout-index/1"
DECISIONS = ("retry", "halt")
LEDGER_NAME = "holdout_ledger.jsonl"
LOCK_NAME = ".holdout.lock"
INDEX_NAME = "holdout_index.json"
FAILED_DIR = "_failed"
SUMS = "SHA256SUMS"
NMSE_RULE = "v02_holdout_all"                    # AM4-B (holdout path)
REGISTERED_OUT = ("results", "v05lt", "holdout")
TRACEBACK_MAX = 8000
STATEMENT_MIN = 20
_APPROVAL_KEYS = {"schema", "abbr", "version", "statement",
                  "launch_chain_sha256", "analysis_chain_sha256",
                  "selection_sha256", "validation_config_sha256",
                  "holdout_config_sha256", "holdout_manifest",
                  "mask_seeds_sha256", "evaluation_scope", "fixture",
                  "time_utc"}
_GATE_TOKEN = object()


def _mods():
    from seqref_mri.scripts import v05lt_analysis as A
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts import v05lt_evidence_check as EVC
    from seqref_mri.scripts import v05lt_launch as L
    from seqref_mri.scripts import v05lt_prodstages as PS
    from seqref_mri.scripts import v05lt_results as RS
    from seqref_mri.scripts import v05lt_stats as S
    return A, EV, EVC, L, PS, RS, S


def _is_sha(x) -> bool:
    return isinstance(x, str) and len(x) == 64 and all(
        c in "0123456789abcdef" for c in x)


def _strict_json(path: Path, code: str):
    try:
        return json.loads(Path(path).read_bytes(),
                          parse_constant=lambda c: C.fail(
                              code, f"{path}: non-finite constant {c}"))
    except (OSError, ValueError) as exc:
        C.fail(code, f"{path}: {type(exc).__name__}: {exc}")


def _regular(path: Path, code: str) -> Path:
    p = Path(path)
    C.require(p.is_file() and not p.is_symlink(), code,
              f"{p}: missing, not a regular file or a symlink")
    return p


def _rel(p: str, code: str) -> str:
    C.require(isinstance(p, str) and p and not p.startswith("/") and
              all(x not in ("", ".", "..") for x in p.split("/")), code,
              f"unsafe relative path {p!r}")
    return p


def registered_out(paths: C.Paths) -> Path:
    return paths.repo.joinpath("seqref_mri", *REGISTERED_OUT)


def registered_manifest(paths: C.Paths) -> Path:
    return paths.v02_manifests / C.V02_SOURCES["holdout"][0]


# ---------------------------------------------------------------------------
# Analysis chain (F18) -> roles and verified files
# ---------------------------------------------------------------------------
def _analysis_chain(path: Path, approved: str, root: Path) -> dict:
    """check_chain (approved sha256 first; every entry re-hashed; bindings)
    then the role -> (path, sha256) map; the holdout_config role is
    required."""
    A, EV, EVC, *_ = _mods()
    res = EVC.check_chain(Path(path), approved, Path(root))
    doc = _strict_json(path, "HOLDOUT_CHAIN_INVALID")
    roles = doc.get("roles") if isinstance(doc, dict) else None
    C.require(isinstance(roles, dict) and "holdout_config" in roles,
              "HOLDOUT_CHAIN_NO_HOLDOUT_CONFIG", "the approved analysis chain "
              "has no holdout_config role (configuration not frozen)")
    sha = {e["path"]: e["sha256"] for e in doc["entries"]}
    out = {}
    for role in ("config", "selection", "holdout_config"):
        rel = _rel(roles[role], "HOLDOUT_CHAIN_INVALID")
        C.require(rel in sha, "HOLDOUT_CHAIN_INVALID", f"{role}: no entry")
        out[role] = (Path(root) / rel, sha[rel])
    out["analysis_sha256"] = res["analysis_sha256"]
    return out


def _holdout_config(path: Path, sha: str, val_sha: str) -> dict:
    A, EV, EVC, L, PS, RS, S = _mods()
    cfg = A.load_config(path, sha)
    C.require(isinstance(cfg.get("holdout"), dict),
              "HOLDOUT_CONFIG_INVALID", "configuration has no holdout addendum")
    C.require(S.config_sha256(S.validation_part(cfg)) == val_sha,
              "HOLDOUT_CONFIG_INVALID", "holdout configuration does not "
              "extend the approved validation configuration")
    h = cfg["holdout"]
    C.require(h.get("roles") == list(RS.HOLDOUT_ROLES) and
              h.get("readouts") == list(RS.HOLDOUT_READOUTS) and
              h.get("evaluation_scope") == S.HOLDOUT_SCOPE_TEXT and
              _is_sha(h.get("manifest_content_sha256")) and
              _is_sha(h.get("mask_seeds_sha256")),
              "HOLDOUT_CONFIG_INVALID", "holdout addendum fields")
    return cfg


def _manifest(path: Path, file_sha: str, content_sha: str, fixture: bool):
    """Holdout manifest -- METADATA ONLY (identities); no MRI file opened."""
    A, EV, EVC, L, PS, RS, S = _mods()
    man = RS.load_holdout_manifest(Path(path), RS.Identities(),
                                   file_sha256=file_sha,
                                   content_sha256=content_sha,
                                   fixture=fixture)
    man["seeds"] = RS.holdout_mask_seeds(man["rows"])
    man["seeds_sha256"] = RS.holdout_mask_seeds_sha256(man["rows"])
    man["path"] = Path(path)
    return man


# ---------------------------------------------------------------------------
# Approval record (refinement 1)
# ---------------------------------------------------------------------------
def make_approval(out_path: Path, *, root: Path, launch_chain: tuple,
                  analysis_chain: tuple, manifest_path: Path,
                  statement: str, fixture: bool = False,
                  launch_root: Path | None = None) -> str:
    """Write-once approval record built from VERIFIED evidence: launch
    chain (path, approved sha256), analysis chain (path, approved sha256),
    holdout manifest. Returns the record sha256, which must be recorded in
    196 before `run` relies on it. Reads no MRI file."""
    A, EV, EVC, L, PS, RS, S = _mods()
    C.require(isinstance(statement, str) and
              len(statement.strip()) >= STATEMENT_MIN,
              "HOLDOUT_APPROVAL_INVALID", "a written approval statement "
              f"(>= {STATEMENT_MIN} characters) is required")
    C.require(fixture or launch_root is None, "HOLDOUT_FIXTURE_IN_PRODUCTION",
              "launch_root is fixture-only")
    L.verify_launch_chain(Path(launch_chain[0]), launch_chain[1],
                          Path(launch_root or root))
    ch = _analysis_chain(analysis_chain[0], analysis_chain[1], root)
    cfg = _holdout_config(*ch["holdout_config"], ch["config"][1])
    A.load_selection(*ch["selection"])
    fsha = C.sha256_file(_regular(manifest_path, "HOLDOUT_MANIFEST_MISSING"))
    man = _manifest(manifest_path, fsha,
                    cfg["holdout"]["manifest_content_sha256"], fixture)
    C.require(man["seeds_sha256"] == cfg["holdout"]["mask_seeds_sha256"],
              "HOLDOUT_MASK_MISMATCH", "manifest seed identity != frozen "
              "holdout configuration")
    doc = {"schema": APPROVAL_SCHEMA, "abbr": __abbr__,
           "version": __version__, "statement": statement.strip(),
           "launch_chain_sha256": launch_chain[1],
           "analysis_chain_sha256": analysis_chain[1],
           "selection_sha256": ch["selection"][1],
           "validation_config_sha256": ch["config"][1],
           "holdout_config_sha256": ch["holdout_config"][1],
           "holdout_manifest": {"file_sha256": man["file_sha256"],
                                "content_sha256": man["content_sha256"]},
           "mask_seeds_sha256": man["seeds_sha256"],
           "evaluation_scope": S.HOLDOUT_SCOPE_TEXT,
           "fixture": bool(fixture), "time_utc": C.utc_now()}
    p = Path(out_path)
    C.require(not os.path.lexists(p), "ARTIFACT_EXISTS", str(p))
    return C.write_durable_new(p, C.canonical_json(doc))


def load_approval(path: Path, approved_sha256: str, production: bool) -> dict:
    C.require(_is_sha(approved_sha256), "HOLDOUT_APPROVAL_UNAPPROVED",
              "an externally approved sha256 is required")
    p = _regular(path, "HOLDOUT_APPROVAL_MISSING")
    got = C.sha256_file(p)
    C.require(got == approved_sha256, "HOLDOUT_APPROVAL_UNAPPROVED",
              f"{p}: {got} != approved {approved_sha256}")
    doc = _strict_json(p, "HOLDOUT_APPROVAL_INVALID")
    C.require(isinstance(doc, dict) and set(doc) == _APPROVAL_KEYS and
              doc["schema"] == APPROVAL_SCHEMA, "HOLDOUT_APPROVAL_INVALID",
              f"{p}: schema / keys")
    hm = doc["holdout_manifest"]
    C.require(isinstance(hm, dict) and set(hm) == {"file_sha256",
                                                   "content_sha256"} and
              all(_is_sha(v) for v in hm.values()) and all(
                  _is_sha(doc[k]) for k in (
                      "launch_chain_sha256", "analysis_chain_sha256",
                      "selection_sha256", "validation_config_sha256",
                      "holdout_config_sha256", "mask_seeds_sha256")) and
              isinstance(doc["statement"], str) and
              len(doc["statement"]) >= STATEMENT_MIN and
              isinstance(doc["fixture"], bool),
              "HOLDOUT_APPROVAL_INVALID", f"{p}: field values")
    C.require(doc["fixture"] is (not production), "HOLDOUT_APPROVAL_MODE",
              f"approval fixture={doc['fixture']}, run production="
              f"{production}")
    return doc


# ---------------------------------------------------------------------------
# Gate (refinements 1, 2) -- verifies everything, opens no MRI file
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Gate:
    token: object
    production: bool
    approval: dict
    approval_sha256: str
    launch_chain_sha256: str
    launch_inputs_sha256: str
    analysis_chain_sha256: str
    analysis_sha256: str
    selection: dict
    selection_sha256: str
    validation_config_sha256: str
    holdout_config_sha256: str
    manifest: dict
    root: Path
    stages: object

    def summary(self) -> dict:
        return {"approval_sha256": self.approval_sha256,
                "launch_chain_sha256": self.launch_chain_sha256,
                "launch_inputs_sha256": self.launch_inputs_sha256,
                "analysis_chain_sha256": self.analysis_chain_sha256,
                "analysis_sha256": self.analysis_sha256,
                "selection_sha256": self.selection_sha256,
                "validation_config_sha256": self.validation_config_sha256,
                "holdout_config_sha256": self.holdout_config_sha256,
                "manifest_file_sha256": self.manifest["file_sha256"],
                "manifest_content_sha256": self.manifest["content_sha256"],
                "mask_seeds_sha256": self.manifest["seeds_sha256"],
                "n_slices": len(self.manifest["rows"]),
                "production": self.production}


def _check_stages(stages) -> bool:
    A, EV, EVC, L, PS, RS, S = _mods()
    C.require(isinstance(stages, PS.ProductionStages),
              "HOLDOUT_STAGES_INVALID", "the holdout runner needs "
              "v05lt_prodstages.ProductionStages (pin / launch-inputs binding)")
    if stages.production:
        C.require(type(stages) is PS.ProductionStages,
                  "HOLDOUT_STAGES_INVALID", "production requires the exact "
                  f"class, got {type(stages).__name__}")
    return bool(stages.production)


def verify_gate(approval: tuple, *, root: Path, launch_chain: Path,
                analysis_chain: Path, manifest_path: Path, stages,
                launch_root: Path | None = None) -> Gate:
    """approval = (path, externally approved sha256). Every check before any
    holdout MRI file is opened (refinement 2). Returns the only object that
    allows the holdout view to be built."""
    A, EV, EVC, L, PS, RS, S = _mods()
    production = _check_stages(stages)
    C.require(not production or launch_root is None,
              "HOLDOUT_FIXTURE_IN_PRODUCTION", "launch_root is fixture-only")
    if production:
        C.require(Path(manifest_path) == registered_manifest(stages.paths),
                  "HOLDOUT_MANIFEST_PATH", f"{manifest_path} != registered "
                  f"{registered_manifest(stages.paths)}")
    ap = load_approval(Path(approval[0]), approval[1], production)
    # launch chain + launch-inputs record (pin, modules, backend, inputs)
    lroot = Path(launch_root or root)
    ldoc = L.verify_launch_chain(Path(launch_chain),
                                 ap["launch_chain_sha256"], lroot)
    bound = stages.bind_chain(ldoc, lroot)
    want = [e["sha256"] for e in ldoc["entries"]
            if e["role"] == PS.LAUNCH_INPUTS_ROLE]
    C.require(isinstance(bound, dict) and len(want) == 1 and
              bound.get("launch_inputs_sha256") == want[0] ==
              (stages.inputs or {}).get("record_sha256"),
              "HOLDOUT_STAGES_UNBOUND", "stages not bound to the approved "
              "launch chain's launch-inputs record")
    # analysis chain -> selection, configurations
    ch = _analysis_chain(Path(analysis_chain), ap["analysis_chain_sha256"],
                         Path(root))
    bind = {"selection_sha256": ch["selection"][1],
            "validation_config_sha256": ch["config"][1],
            "holdout_config_sha256": ch["holdout_config"][1]}
    diff = {k: (ap[k], v) for k, v in bind.items() if ap[k] != v}
    C.require(not diff, "HOLDOUT_APPROVAL_BINDING",
              f"approval vs approved analysis chain (approval, chain): {diff}")
    sel = A.load_selection(*ch["selection"])
    cfg = _holdout_config(*ch["holdout_config"], ch["config"][1])
    # holdout manifest -- metadata only
    hm = ap["holdout_manifest"]
    if production:
        C.require((hm["file_sha256"], hm["content_sha256"]) ==
                  tuple(C.V02_SOURCES["holdout"][2:]),
                  "HOLDOUT_MANIFEST_MISMATCH", "approval names an "
                  "unregistered holdout manifest")
    man = _manifest(manifest_path, hm["file_sha256"], hm["content_sha256"],
                    not production)
    C.require(man["content_sha256"] ==
              cfg["holdout"]["manifest_content_sha256"],
              "HOLDOUT_MANIFEST_MISMATCH", "manifest content != frozen "
              "holdout configuration")
    C.require(man["seeds_sha256"] == ap["mask_seeds_sha256"] ==
              cfg["holdout"]["mask_seeds_sha256"], "HOLDOUT_MASK_MISMATCH",
              "expected mask-seed identity: manifest / approval / frozen "
              "configuration differ")
    logger.info("[%s] holdout gate verified (approval %s, production=%s); "
                "no holdout MRI file opened", __abbr__, approval[1][:12],
                production)
    return Gate(token=_GATE_TOKEN, production=production, approval=ap,
                approval_sha256=approval[1],
                launch_chain_sha256=ap["launch_chain_sha256"],
                launch_inputs_sha256=want[0],
                analysis_chain_sha256=ap["analysis_chain_sha256"],
                analysis_sha256=ch["analysis_sha256"], selection=sel,
                selection_sha256=ch["selection"][1],
                validation_config_sha256=ch["config"][1],
                holdout_config_sha256=ch["holdout_config"][1],
                manifest=man, root=Path(root), stages=stages)


def _gate_ok(gate) -> Gate:
    C.require(isinstance(gate, Gate) and gate.token is _GATE_TOKEN,
              "HOLDOUT_GATE_REQUIRED", "holdout data are accessed only after "
              "verify_gate()")
    return gate


# ---------------------------------------------------------------------------
# Task plan (refinement 3)
# ---------------------------------------------------------------------------
def plan_tasks(selection: dict) -> list:
    """One task per distinct model of a run among {selected, epoch3};
    coincident checkpoints -> one task with both roles (evaluated once).
    Final and epoch-0 checkpoints are never planned."""
    A, EV, EVC, L, PS, RS, S = _mods()
    tasks = []
    for key in sorted(selection["runs"]):
        arm, seed = key.split("|")
        run = selection["runs"][key]
        by_state = {}
        for role in RS.HOLDOUT_ROLES:
            ck = run[role]
            t = by_state.get(ck["model_state_sha256"])
            if t is None:
                t = by_state[ck["model_state_sha256"]] = {
                    "arm": arm, "seed": int(seed), "epoch": ck["epoch"],
                    "model_state_sha256": ck["model_state_sha256"],
                    "file": ck["file"], "sha256": ck["sha256"], "roles": []}
            C.require(t["epoch"] == ck["epoch"] and t["file"] == ck["file"]
                      and t["sha256"] == ck["sha256"],
                      "HOLDOUT_PLAN_CONFLICT", f"{key}: roles share model "
                      f"{ck['model_state_sha256'][:12]} with different "
                      "epoch / checkpoint file")
            t["roles"].append(role)
        tasks += list(by_state.values())
    for t in tasks:
        t["name"] = f"{t['seed']}_{t['arm']}_{t['model_state_sha256']}"
    return tasks


def task_scope_check(task: dict) -> None:
    A, EV, EVC, L, PS, RS, S = _mods()
    C.require(isinstance(task.get("roles"), list) and task["roles"] and
              set(task["roles"]) <= set(RS.HOLDOUT_ROLES),
              "HOLDOUT_SCOPE", f"roles {task.get('roles')} outside the "
              f"registered scope {list(RS.HOLDOUT_ROLES)}")


# ---------------------------------------------------------------------------
# Holdout data (only after the gate; item 27)
# ---------------------------------------------------------------------------
def holdout_view(gate: Gate, data_root: Path):
    """Production dataset class on split 'val' (the V02 holdout), restricted
    to the manifest slices in manifest order. Only with a verified gate."""
    gate = _gate_ok(gate)
    from seqref_mri.src import fastmri_data as FD
    rows = gate.manifest["rows"]
    vols = frozenset(r["volume_id"] for r in rows)
    try:
        base = FD.FastMRISliceDataset(str(data_root), split="val",
                                      mode="eval")
    except (OSError, KeyError, ValueError) as exc:
        C.fail("HOLDOUT_DATA_LOAD", f"{type(exc).__name__}: {exc}")
    idx = [(Path(p).relative_to(base.data_root).as_posix(), int(s))
           for p, s in base.index]
    on_disk = {Path(r).name for r, _ in idx}
    C.require(on_disk == set(vols), "HOLDOUT_DATA_POPULATION",
              f"on disk {len(on_disk)} volumes vs manifest {len(vols)}; "
              f"missing {sorted(set(vols) - on_disk)[:3]} extra "
              f"{sorted(on_disk - set(vols))[:3]}")
    pos = {e: i for i, e in enumerate(idx)}
    miss = [(r["file"], r["slice_index"]) for r in rows
            if (r["file"], r["slice_index"]) not in pos]
    C.require(not miss, "HOLDOUT_DATA_POPULATION",
              f"manifest slices not in the dataset: {miss[:3]}")
    return C.SplitView(base, [pos[(r["file"], r["slice_index"])]
                              for r in rows], vols, "holdout")


def check_widths(gate: Gate, view) -> dict:
    """Item 27: per slice the loaded k-space width, the loaded mask and seed
    against the registered rule; returns widths and both mask identities
    (recomputation and loaded bytes; must agree)."""
    gate = _gate_ok(gate)
    A, EV, EVC, L, PS, RS, S = _mods()
    from seqref_mri.src import fastmri_data as FD
    rows, seeds = gate.manifest["rows"], gate.manifest["seeds"]
    C.require(len(view) == len(rows), "HOLDOUT_DATA_POPULATION",
              f"view {len(view)} vs manifest {len(rows)}")
    widths, h = [], hashlib.sha256()
    for i, (r, sd) in enumerate(zip(rows, seeds)):
        C.require(view.entry(i) == (r["file"], r["slice_index"]),
                  "HOLDOUT_DATA_ORDER", f"position {i}: {view.entry(i)} vs "
                  f"{(r['file'], r['slice_index'])}")
        try:
            item = view[i]
        except C.V05LTError:
            raise
        except Exception as exc:  # noqa: BLE001 -- typed, logged, re-raised
            C.fail("HOLDOUT_DATA_LOAD", f"{r['file']}|{r['slice_index']}: "
                   f"{type(exc).__name__}: {exc}")
        w = int(item["y"].shape[-1])
        m = FD.make_cartesian_mask(w, int(sd)).astype(bool)
        got = np.asarray(item["mask"]).astype(bool).reshape(-1)
        C.require(int(item["meta"]["mask_seed"]) == int(sd),
                  "HOLDOUT_MASK_MISMATCH", f"{r['file']}|{r['slice_index']}: "
                  "loaded seed != canonical_mask_seed(EVAL_BASE_SEED, file, "
                  "slice)")
        C.require(got.shape == (w,) and np.array_equal(got, m),
                  "HOLDOUT_WIDTH_MISMATCH", f"{r['file']}|{r['slice_index']}:"
                  f" loaded mask (shape {got.shape}) != make_cartesian_mask("
                  f"width={w}, seed)")
        widths.append(w)
        h.update(np.ascontiguousarray(got).tobytes(order="C"))
    rec = RS.holdout_masks_sha256(seeds, widths)
    C.require(rec == h.hexdigest(), "HOLDOUT_MASK_MISMATCH",
              "recomputed mask identity != loaded mask bytes")
    return {"widths": widths, "masks_sha256": rec}


def _p4_sref(stages) -> tuple:
    A, EV, EVC, L, PS, RS, S = _mods()
    if stages.production:
        from seqref_mri.src import free_flow_runtime as ffr
        s_ref = C.verify_p0s(stages.paths.p0s)
        try:
            p4 = ffr.load_p4s2_parent(str(stages.repo / PS.P4_REL))
        except Exception as exc:  # noqa: BLE001 -- typed, logged, re-raised
            C.fail("HOLDOUT_PARENT_LOAD", f"{type(exc).__name__}: {exc}")
        return p4, s_ref
    return stages.fx["p4"], stages.fx["s_ref"]


# ---------------------------------------------------------------------------
# Evaluation of one task (core 4)
# ---------------------------------------------------------------------------
def code_identity() -> dict:
    A, EV, *_ = _mods()
    return {"eval": EV.code_identity(),
            "holdout": {"sha256": C.sha256_file(Path(__file__).resolve()),
                        "version": __version__}}


def _nll(model, states: list) -> list:
    """NLL per free real dimension on R0 inputs: -log_prob_free / d_free in
    float64, batched as the shared validation (v05lt_metrics.validate)."""
    import torch
    from seqref_mri.scripts import v05lt_metrics as M
    from seqref_mri.src import free_flow_runtime as ffr
    out = []
    for i in range(0, len(states), M.BATCH):
        part = states[i:i + M.BATCH]
        tgt = torch.from_numpy(np.concatenate(
            [st["target"] for st in part], 0).astype(np.float32))
        cond = torch.cat([st["cond"] for st in part], 0)
        msk = torch.cat([st["mask"] for st in part], 0)
        d_free = int(tgt.shape[1])
        C.require(d_free == ffr.FLOW_DIM_REAL, "HOLDOUT_DFREE", str(d_free))
        try:
            lp = model.log_prob_free(tgt, cond, msk)
        except C.V05LTError:
            raise
        except Exception as exc:  # noqa: BLE001 -- typed, logged, re-raised
            C.fail("HOLDOUT_NLL_FAILED", f"{type(exc).__name__}: {exc}")
        v = (-lp.to(torch.float64) / d_free).tolist()
        C.require(all(math.isfinite(x) for x in v), "HOLDOUT_NON_FINITE",
                  f"NLL non-finite at positions {i}..{i + len(part) - 1}")
        out += [float(x) for x in v]
    return out


def evaluate_task(model, checkpoint, states: list, gate: Gate,
                  bank: np.ndarray) -> tuple:
    """R0, R1 (primary bank, 128 decodes), NLL on every holdout state for
    one verified checkpoint. Parameters, mode and RNG states unchanged.
    Returns (rows, info)."""
    gate = _gate_ok(gate)
    import torch
    A, EV, EVC, L, PS, RS, S = _mods()
    C.require(isinstance(checkpoint, EV.VerifiedCheckpoint),
              "HOLDOUT_CHECKPOINT_UNVERIFIED", "checkpoint must come from "
              "load_trained_checkpoint")
    C.require(EV._array_sha(bank) == EV.REGISTERED_BANK_ARRAY["primary"] and
              tuple(bank.shape) == C.BANK_SHAPE, "HOLDOUT_BANK",
              "R1 bank is not the registered primary bank")
    rows_m, seeds = gate.manifest["rows"], gate.manifest["seeds"]
    C.require(len(states) == len(rows_m), "HOLDOUT_DATA_POPULATION",
              f"{len(states)} states vs {len(rows_m)} manifest slices")
    rng0 = PS._rng_save()
    params0 = {k: v.detach().clone() for k, v in model.state_dict().items()}
    was_training = model.training
    model.eval()
    t0 = time.perf_counter()
    decodes = 0
    rows = []
    try:
        with torch.no_grad():
            nll = _nll(model, states)
            for st, r, sd, n in zip(states, rows_m, seeds, nll):
                ident = (str(st["identity"]["file"]),
                         int(st["identity"]["slice_index"]))
                C.require(ident == (r["file"], r["slice_index"]),
                          "HOLDOUT_DATA_ORDER", f"state {ident} vs manifest "
                          f"{(r['file'], r['slice_index'])}")
                r0 = EV.r0_readout(model, st, True, NMSE_RULE)
                r0["nll_per_dim"] = n
                dec = EV.decode_bank(model, st, bank)
                decodes += int(dec.shape[0])
                r1 = EV.r1_from_decodes(st, dec, EV.BANK_ROWS, True,
                                        NMSE_RULE)
                rows.append({"volume_id": r["volume_id"], "file": r["file"],
                             "slice_index": r["slice_index"],
                             "mask_seed": int(sd), "r0": r0, "r1": r1})
    finally:
        model.train(was_training)
        rng1 = PS._rng_save()
        consumed = not PS._rng_equal(rng0, rng1)
        PS._rng_restore(rng0)
    after = model.state_dict()
    C.require(all(torch.equal(params0[k], after[k]) for k in params0),
              "HOLDOUT_STATE_CHANGED", "evaluation changed parameters")
    C.require(decodes == EV.BANK_ROWS * len(rows), "HOLDOUT_BANK",
              f"{decodes} decodes != 128 x {len(rows)}")
    return rows, {"decodes": decodes,
                  "evaluate_s": time.perf_counter() - t0,
                  "rng": {"consumed_during_eval": consumed,
                          "restored": True}}


def build_records(task: dict, gate: Gate, checkpoint, rows: list,
                  info: dict, masks: dict) -> dict:
    """One b3 record per role of the task, all from the same evaluation."""
    A, EV, EVC, L, PS, RS, S = _mods()
    gate = _gate_ok(gate)
    base = expected_binding(task, gate, masks)
    out = {}
    for role in task["roles"]:
        out[role] = dict(base, role=role, rows=rows,
                         checkpoint=checkpoint.as_record(),
                         timing={"evaluate_s": info["evaluate_s"]},
                         rng=info["rng"], decodes=info["decodes"])
    return out


def expected_binding(task: dict, gate: Gate, masks: dict) -> dict:
    """Every record field that is fixed by the task, the gate, the approved
    execution and the loaded masks (used to write and to verify)."""
    A, EV, EVC, L, PS, RS, S = _mods()
    m = gate.manifest
    return {"schema": RS.HOLDOUT_SCHEMA, "abbr": __abbr__,
            "version": __version__, "seed": task["seed"], "arm": task["arm"],
            "epoch": task["epoch"],
            "model_state_sha256": task["model_state_sha256"],
            "checkpoint_sha256": task["sha256"],
            "checkpoint_file": task["file"],
            "evaluation": {"name": task["name"], "roles": list(task["roles"])},
            "manifest_content_sha256": m["content_sha256"],
            "manifest_file_sha256": m["file_sha256"],
            "masks": {"convention": RS.HOLDOUT_MASK_CONVENTION,
                      "sha256": masks["masks_sha256"],
                      "seeds_sha256": m["seeds_sha256"],
                      "widths": list(masks["widths"])},
            "bank": {"name": "primary",
                     "array_sha256": EV.REGISTERED_BANK_ARRAY["primary"],
                     "n_decodes": EV.BANK_ROWS},
            "nmse_rule": NMSE_RULE,
            "approval_sha256": gate.approval_sha256,
            "bindings": {k: v for k, v in gate.summary().items()
                         if k.endswith("_sha256")},
            "code": code_identity(), "backend": EV.backend_identity(),
            "fixture": not gate.production}


# ---------------------------------------------------------------------------
# Publication and verification (core 6)
# ---------------------------------------------------------------------------
def _staging(d: Path) -> Path:
    return d.with_name(f".{d.name}.staging")


def write_task(out_dir: Path, task: dict, records: dict, hook=None) -> dict:
    """Stage <role>.json + sidecars + SHA256SUMS, publish the evaluation
    directory with RENAME_NOREPLACE. Returns {role: (path, sha256)}."""
    A, EV, *_ = _mods()
    d = EV.dest_path(Path(out_dir) / task["name"])
    stg = _staging(d)
    C.require(not EV.entry_exists(d), "ARTIFACT_EXISTS", str(d))
    C.require(not EV.entry_exists(stg), "HOLDOUT_OUTPUT_INCOMPLETE",
              f"{stg}: staging present (interrupted write)")
    files = {}
    for role, doc in sorted(records.items()):
        try:
            b = (json.dumps(doc, sort_keys=True, allow_nan=False) +
                 "\n").encode()
        except (TypeError, ValueError) as exc:
            C.fail("HOLDOUT_JSON_INVALID", f"{role}: {exc}")
        files[f"{role}.json"] = b
        files[f"{role}.json.sha256"] = \
            f"{C.sha256_bytes(b)}  {role}.json\n".encode()
    try:
        stg.mkdir()
        shas = {n: C.write_durable_new(stg / n, b)
                for n, b in sorted(files.items())}
        C.write_durable_new(stg / SUMS, "".join(
            f"{h}  {n}\n" for n, h in sorted(shas.items())).encode())
    except OSError as exc:
        C.fail("HOLDOUT_WRITE_FAILED", f"{stg}: {exc}")
    if hook is not None:
        hook("holdout:staged")
    EV.publish_noreplace(stg, d)
    return {role: (d / f"{role}.json", shas[f"{role}.json"])
            for role in records}


def classify(out_dir: Path, task: dict) -> str:
    A, EV, *_ = _mods()
    d = EV.dest_path(Path(out_dir) / task["name"])
    if EV.entry_exists(_staging(d)):
        return "incomplete"
    if not EV.entry_exists(d):
        return "absent"
    return "present"


def verify_task(out_dir: Path, task: dict, gate: Gate, masks: dict) -> dict:
    """A published evaluation directory: file set and hashes (else
    HOLDOUT_OUTPUT_INCOMPLETE), every record = the expected binding for
    this task, gate, approved execution and loaded masks, rows = manifest,
    values valid for the b3 loader, roles share one evaluation (else
    HOLDOUT_OUTPUT_MISMATCH). Returns {role: (path, sha256)}."""
    A, EV, EVC, L, PS, RS, S = _mods()
    gate = _gate_ok(gate)
    d = Path(out_dir) / task["name"]
    C.require(d.is_dir() and not d.is_symlink(), "HOLDOUT_OUTPUT_INCOMPLETE",
              f"{d}: not a directory")
    sums = d / SUMS
    C.require(sums.is_file() and not sums.is_symlink(),
              "HOLDOUT_OUTPUT_INCOMPLETE", f"{d}: {SUMS}")
    listed = {}
    try:
        lines = sums.read_bytes().decode("ascii").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        C.fail("HOLDOUT_OUTPUT_INCOMPLETE", f"{sums}: {exc}")
    for line in lines:
        h, sep, n = line.partition("  ")
        C.require(bool(sep) and _is_sha(h) and n and "/" not in n and
                  n not in listed, "HOLDOUT_OUTPUT_INCOMPLETE",
                  f"{SUMS} line {line!r}")
        listed[n] = h
    present = {p.name for p in d.iterdir()} - {SUMS}
    C.require(present == set(listed), "HOLDOUT_OUTPUT_INCOMPLETE",
              f"{d}: listed {sorted(listed)} present {sorted(present)}")
    for n, h in listed.items():
        f = d / n
        C.require(f.is_file() and not f.is_symlink() and
                  C.sha256_file(f) == h, "HOLDOUT_OUTPUT_INCOMPLETE",
                  f"{d.name}/{n}: hash")
    want = {f"{r}.json" for r in task["roles"]} | \
        {f"{r}.json.sha256" for r in task["roles"]}
    C.require(set(listed) == want, "HOLDOUT_OUTPUT_MISMATCH",
              f"{d.name}: files {sorted(listed)} != roles {task['roles']}")
    exp = expected_binding(task, gate, masks)
    out, rows0 = {}, None
    for role in task["roles"]:
        p = d / f"{role}.json"
        sha = listed[p.name]
        C.require(C.read_sidecar(Path(str(p) + ".sha256"),
                                 "HOLDOUT_OUTPUT_INCOMPLETE") == sha,
                  "HOLDOUT_OUTPUT_INCOMPLETE", f"{d.name}/{p.name}: sidecar")
        doc = _strict_json(p, "HOLDOUT_OUTPUT_MISMATCH")
        C.require(isinstance(doc, dict), "HOLDOUT_OUTPUT_MISMATCH",
                  f"{p.name}: not an object")
        bad = sorted(k for k, v in exp.items() if doc.get(k) != v)
        C.require(not bad and doc.get("role") == role,
                  "HOLDOUT_OUTPUT_MISMATCH", f"{d.name}/{p.name}: fields "
                  f"{bad or ['role']} differ from this task / approval / "
                  "approved execution / loaded masks")
        ck = doc.get("checkpoint")
        C.require(isinstance(ck, dict) and
                  ck.get("model_state_sha256") == task["model_state_sha256"]
                  and ck.get("file_sha256") == task["sha256"] and
                  ck.get("seed") == task["seed"] and
                  ck.get("arm") == task["arm"] and
                  ck.get("epoch") == task["epoch"],
                  "HOLDOUT_OUTPUT_MISMATCH", f"{p.name}: checkpoint record")
        rows = doc.get("rows")
        man = gate.manifest
        C.require(isinstance(rows, list) and len(rows) == len(man["rows"]),
                  "HOLDOUT_OUTPUT_MISMATCH", f"{p.name}: row count")
        for r, m, sd in zip(rows, man["rows"], man["seeds"]):
            C.require(isinstance(r, dict) and
                      (r.get("volume_id"), r.get("file"),
                       r.get("slice_index"), r.get("mask_seed")) ==
                      (m["volume_id"], m["file"], m["slice_index"], sd),
                      "HOLDOUT_OUTPUT_MISMATCH", f"{p.name}: row "
                      f"{str(r)[:100]} vs manifest")
            for ro in RS.HOLDOUT_READOUTS:
                RS._holdout_readout(r.get(ro), f"{p.name} {m['file']} {ro}",
                                    ro == "r0")
        C.require(doc.get("decodes") == EV.BANK_ROWS * len(rows),
                  "HOLDOUT_OUTPUT_MISMATCH", f"{p.name}: decode count")
        if rows0 is None:
            rows0 = rows
        C.require(rows == rows0, "HOLDOUT_OUTPUT_MISMATCH",
                  f"{d.name}: roles do not share one evaluation")
        out[role] = (p, sha)
    return out


# ---------------------------------------------------------------------------
# Ledger, halts and decisions (addition D; refinement 4)
# ---------------------------------------------------------------------------
class Halted(C.V05LTError):
    pass


def _halt_state(ledger: "C.Ledger") -> dict:
    unresolved, halted, gates = None, False, []
    for r in ledger.records() if ledger.path.exists() else []:
        k = r["kind"]
        if k == "gate_passed":
            gates.append(r["body"]["approval_sha256"])
        elif k == "halt" and unresolved is None:
            unresolved = r
        elif k == "halt_resolved" and unresolved is not None and \
                r["body"].get("halt_seq") == unresolved["seq"]:
            halted = halted or r["body"].get("decision") == "halt"
            unresolved = None
    return {"unresolved": unresolved, "halted": halted, "approvals": gates}


def _halt_binding(rec: dict) -> dict:
    b = rec["body"]
    return {"halt_seq": rec["seq"],
            "halt_line_sha256": C.sha256_bytes(C.canonical_json(rec)),
            "halt_code": b.get("code"), "task": b.get("task"),
            "approval_sha256": b.get("approval_sha256")}


def make_decision(out_path: Path, out_dir: Path, decision: str,
                  note: str) -> str:
    """Write-once decision bound to the CURRENT unresolved holdout halt
    (and through it to the approval it was recorded under)."""
    C.require(decision in DECISIONS, "HOLDOUT_DECISION_INVALID", decision)
    C.require(isinstance(note, str) and note.strip() != "",
              "HOLDOUT_DECISION_INVALID", "a reason (note) is required")
    hs = _halt_state(C.Ledger(Path(out_dir) / LEDGER_NAME))
    C.require(hs["unresolved"] is not None, "HOLDOUT_DECISION_NOT_APPLICABLE",
              f"{out_dir}: no unresolved holdout halt")
    doc = {"schema": DECISION_SCHEMA,
           "binding": _halt_binding(hs["unresolved"]),
           "decision": decision, "note": note.strip(),
           "time_utc": C.utc_now()}
    p = Path(out_path)
    C.require(not os.path.lexists(p), "ARTIFACT_EXISTS", str(p))
    return C.write_durable_new(p, C.canonical_json(doc))


def check_decision(decision, halt: dict, approval_sha256: str) -> str:
    C.require(decision is not None, "HOLDOUT_HALT_UNRESOLVED",
              f"unresolved holdout halt {halt['body'].get('code')} at seq "
              f"{halt['seq']}: a decision bound to it is required")
    path, sha = decision
    p = _regular(path, "HOLDOUT_DECISION_UNAPPROVED")
    C.require(C.sha256_file(p) == sha, "HOLDOUT_DECISION_UNAPPROVED",
              f"{p}: hash != approved {sha}")
    doc = _strict_json(p, "HOLDOUT_DECISION_INVALID")
    C.require(isinstance(doc, dict) and doc.get("schema") == DECISION_SCHEMA
              and doc.get("decision") in DECISIONS,
              "HOLDOUT_DECISION_INVALID", str(p))
    C.require(doc.get("binding") == _halt_binding(halt),
              "HOLDOUT_DECISION_STALE", f"decision bound to "
              f"{doc.get('binding')}, unresolved halt is {_halt_binding(halt)}")
    C.require(doc["binding"]["approval_sha256"] == approval_sha256,
              "HOLDOUT_DECISION_STALE", "decision was recorded under another "
              "approval")
    return doc["decision"]


class Runner:
    def __init__(self, gate: Gate, out_dir: Path, data_root: Path, *,
                 hooks: dict | None = None):
        self.gate = _gate_ok(gate)
        C.require(gate.production is False or hooks is None,
                  "HOLDOUT_FIXTURE_IN_PRODUCTION", "hooks are fixture-only")
        self.out = Path(out_dir)
        self.data_root = Path(data_root)
        self.hooks = hooks or {}
        self.ledger = C.Ledger(self.out / LEDGER_NAME)

    def _hook(self, name: str):
        return self.hooks.get(name)

    def _log(self, kind: str, body: dict) -> None:
        self.ledger.append(kind, body)

    def _halt(self, code: str, msg: str, **body) -> Halted:
        try:
            self._log("halt", {"code": code, "message": msg[:1000],
                               "approval_sha256": self.gate.approval_sha256,
                               **body})
        except Exception as rec_exc:  # noqa: BLE001 -- logged, halt raised
            logger.error("[%s] HALT_RECORD_FAILED: %s", __abbr__, rec_exc)
        logger.error("[failed_validity] %s: %s", code, msg)
        return Halted(code, msg)

    def _retain_failed(self, halt: dict) -> list:
        """Move the failed attempt's outputs, and the index that lists
        them, to _failed/ (never deleted)."""
        A, EV, *_ = _mods()
        name = halt["body"].get("task")
        dst_root = self.out / FAILED_DIR
        dst_root.mkdir(exist_ok=True)
        idx = self.out / INDEX_NAME          # lists the failed attempt
        srcs = [idx, Path(str(idx) + ".sha256")]
        if name:
            d = self.out / name
            srcs += [_staging(d), d]
        moved = []
        for src in srcs:
            if EV.entry_exists(src):
                dst = dst_root / f"seq{halt['seq']}_{src.name}"
                EV.publish_noreplace(src, dst)
                moved.append({"from": src.name,
                              "to": f"{FAILED_DIR}/{dst.name}"})
        return moved

    def _resolve(self, decision) -> None:
        hs = _halt_state(self.ledger)
        C.require(not hs["halted"], "HOLDOUT_HALTED",
                  f"{self.out}: ended by an owner `halt` decision")
        other = sorted(set(hs["approvals"]) - {self.gate.approval_sha256})
        C.require(not other, "HOLDOUT_APPROVAL_CHANGED", f"{self.out}: "
                  f"earlier runs used approval(s) {[o[:12] for o in other]}; "
                  "one output directory per approval")
        halt = hs["unresolved"]
        if halt is None:
            return
        dec = check_decision(decision, halt, self.gate.approval_sha256)
        moved = self._retain_failed(halt) if dec == "retry" else []
        self._log("halt_resolved", {"halt_seq": halt["seq"], "decision": dec,
                                    "sha256": decision[1],
                                    "binding": _halt_binding(halt),
                                    "retained": moved})
        if dec == "halt":
            C.fail("HOLDOUT_HALTED", f"owner `halt` for "
                   f"{halt['body'].get('code')} at seq {halt['seq']}")
        logger.warning("[%s] holdout halt %s (seq %d) resolved by `retry`; "
                       "failed attempt retained: %s", __abbr__,
                       halt["body"].get("code"), halt["seq"], moved)

    def _index(self, recs: dict) -> dict:
        g = self.gate
        return {"schema": INDEX_SCHEMA, "abbr": __abbr__,
                "version": __version__, "approval_sha256": g.approval_sha256,
                "manifest_file_sha256": g.manifest["file_sha256"],
                "manifest_content_sha256": g.manifest["content_sha256"],
                "records": {k: {"path": f"{v[0].parent.name}/{v[0].name}",
                                "sha256": v[1]}
                            for k, v in sorted(recs.items())}}

    def _write_index(self, recs: dict) -> str:
        A, EV, EVC, L, PS, RS, S = _mods()
        h = self._hook("holdout:index")
        if h is not None:
            h()
        doc = self._index(recs)
        p = self.out / INDEX_NAME
        data = C.canonical_json(doc)
        if os.path.lexists(p):
            C.require(p.is_file() and not p.is_symlink() and
                      p.read_bytes() == data, "HOLDOUT_OUTPUT_MISMATCH",
                      f"{p}: existing index differs from the verified records")
            C.require(C.read_sidecar(Path(str(p) + ".sha256"),
                                     "HOLDOUT_OUTPUT_INCOMPLETE") ==
                      C.sha256_bytes(data), "HOLDOUT_OUTPUT_INCOMPLETE",
                      f"{p}: sidecar")
            return C.sha256_bytes(data)
        return S.write_json_once(p, doc, production=self.gate.production)

    def _stage(self, stage: str, fn, *a, code: str | None = None,
               task: str | None = None):
        """Post-gate stage boundary (revision-1 review, finding 1): every
        stage after `gate_passed` runs here. Recorded halts pass through;
        a typed failure becomes a recorded halt (`code`, or the failure's
        own code when None) with its cause; any other ordinary exception is
        logged with its traceback and recorded as HOLDOUT_STAGE_ERROR. So
        clearing a fault alone never resumes the stage: the next run needs
        a decision bound to the halt. BaseException (process interruption)
        passes through unchanged."""
        try:
            return fn(*a)
        except Halted:
            raise
        except C.V05LTError as exc:
            raise self._halt(code or exc.code, f"{stage}: {exc.code}: {exc}",
                             stage=stage, task=task, cause=exc.code)
        except Exception as exc:  # noqa: BLE001 -- recorded halt, raised
            tb = traceback.format_exc()
            logger.error("[%s] unexpected %s in stage %s:\n%s", __abbr__,
                         type(exc).__name__, stage, tb)
            raise self._halt("HOLDOUT_STAGE_ERROR", f"{stage}: "
                             f"{type(exc).__name__}: {exc}", stage=stage,
                             task=task, type=type(exc).__name__,
                             traceback=tb[-TRACEBACK_MAX:])

    def run(self, decision=None) -> dict:
        g = self.gate
        self._resolve(decision)
        self._log("gate_passed", {**g.summary(), "abbr": __abbr__,
                                  "version": __version__})
        return self._stage("run", self._body)

    def _data(self) -> tuple:
        h = self._hook("holdout:data")
        if h is not None:
            h()
        view = holdout_view(self.gate, self.data_root)
        return view, check_widths(self.gate, view)

    def _body(self) -> dict:
        A, EV, EVC, L, PS, RS, S = _mods()
        g = self.gate
        tasks = self._stage("plan", plan_tasks, g.selection)
        for t in tasks:
            self._stage("plan", task_scope_check, t, task=t["name"])
        # ---- holdout contact starts here (after the recorded gate) ----
        view, masks = self._stage("data", self._data,
                                  code="HOLDOUT_DATA_FAILED")
        self._log("data_verified", {"n_slices": len(masks["widths"]),
                                    "masks_sha256": masks["masks_sha256"]})
        states, recs = None, {}
        for t in tasks:
            st = self._stage("classify", classify, self.out, t,
                             task=t["name"])
            if st == "incomplete":
                raise self._halt("HOLDOUT_OUTPUT_INCOMPLETE", f"{t['name']}: "
                                 "staging present (interrupted write)",
                                 task=t["name"])
            if st == "present":
                got = self._stage("verify", self._verify, t, masks,
                                  task=t["name"])
                self._log("task_reused", {"task": t["name"],
                                          "roles": t["roles"],
                                          "sha256": {r: v[1] for r, v in
                                                     got.items()}})
            else:
                if states is None:
                    states = self._stage("states", self._states, view,
                                         code="HOLDOUT_DATA_FAILED")
                got = self._evaluate(t, states, masks)
            for role, v in got.items():
                recs[f"{t['arm']}|{t['seed']}|{role}"] = v
        want = {f"{a}|{s}|{r}" for a in A.SELECTION_ARMS
                for s in A.SELECTION_SEEDS for r in RS.HOLDOUT_ROLES}
        C.require(set(recs) == want, "HOLDOUT_SCOPE",
                  f"records {sorted(recs)[:4]} != registered scope")
        isha = self._stage("index", self._write_index, recs)
        self._log("complete", {"index_sha256": isha, "records": len(recs)})
        logger.info("[%s] holdout stage complete: %d records, index %s",
                    __abbr__, len(recs), isha[:12])
        return {"status": "complete", "index_sha256": isha,
                "records": {k: {"path": v[0], "sha256": v[1]}
                            for k, v in recs.items()}}

    def _verify(self, t: dict, masks: dict) -> dict:
        h = self._hook("holdout:verify")
        if h is not None:
            h(t)
        return verify_task(self.out, t, self.gate, masks)

    def _states(self, view) -> list:
        A, EV, *_ = _mods()
        h = self._hook("holdout:states")
        if h is not None:
            h()
        p4, s_ref = _p4_sref(self.gate.stages)
        return EV.build_states(view, list(range(len(view))), p4, s_ref)

    def _evaluate(self, t: dict, states: list, masks: dict) -> dict:
        A, EV, EVC, L, PS, RS, S = _mods()
        g = self.gate
        stg = g.stages
        try:
            h = self._hook("holdout:before_evaluate")
            if h is not None:
                h(t)
            model, vck = EV.load_trained_checkpoint(
                g.root / t["file"], t["sha256"], t["model_state_sha256"],
                seed=t["seed"], arm=t["arm"], epoch=t["epoch"],
                model_factory=stg._factory())
            rows, info = evaluate_task(model, vck, states, g,
                                       stg.banks()["primary"])
            del model
            recs = build_records(t, g, vck, rows, info, masks)
            write_task(self.out, t, recs, hook=self._hook("holdout:staging"))
            got = verify_task(self.out, t, g, masks)
            stg._check_loaded(stg.inputs["pin_code"])
        except Halted:
            raise
        except C.V05LTError as exc:
            raise self._halt("HOLDOUT_TASK_FAILED", f"{t['name']}: "
                             f"{exc.code}: {exc}", task=t["name"],
                             cause=exc.code)
        except Exception as exc:  # noqa: BLE001 -- recorded halt, raised
            tb = traceback.format_exc()
            logger.error("[%s] unexpected %s in task %s:\n%s", __abbr__,
                         type(exc).__name__, t["name"], tb)
            raise self._halt("HOLDOUT_STAGE_ERROR", f"{t['name']}: "
                             f"{type(exc).__name__}: {exc}", task=t["name"],
                             type=type(exc).__name__,
                             traceback=tb[-TRACEBACK_MAX:])
        self._log("task_done", {"task": t["name"], "roles": t["roles"],
                                "sha256": {r: v[1] for r, v in got.items()},
                                "decodes": info["decodes"],
                                "evaluate_s": round(info["evaluate_s"], 3)})
        return got


def check_out_dir(out_dir: Path, data_root: Path, paths: C.Paths,
                  production: bool) -> Path:
    """Output-root guard (revision-1 review, finding 2), applied BEFORE any
    directory, lock or ledger is created. The output root is never a
    symlink; containment is decided on the RESOLVED destination.
    Production: the registered root exactly, whose resolved destination
    must equal repo/<registered> (a symlinked root or parent redirects it
    and is refused); data root registered. Fixture: neither the spelling nor the resolved
    destination may lie at or under the registered results/v05lt. Existing
    ledger, lock, index and _failed entries must not be symlinks."""
    out = Path(os.path.abspath(out_dir))
    C.require(not out.is_symlink(), "HOLDOUT_OUT_PATH",
              f"{out}: output root is a symlink")
    res = out.resolve()
    if production:
        reg = Path(os.path.abspath(registered_out(paths)))
        repo = Path(os.path.abspath(paths.repo))
        C.require(out == reg and Path(os.path.abspath(data_root)) ==
                  Path(os.path.abspath(paths.data_root)),
                  "HOLDOUT_OUT_PATH", f"production writes only {reg} from the "
                  "registered data root")
        C.require(res == repo.resolve() / out.relative_to(repo),
                  "HOLDOUT_OUT_PATH", f"{out} resolves to {res}")
    else:
        for prod in {Path(os.path.abspath(paths.v05lt)),
                     Path(os.path.abspath(paths.v05lt)).resolve()}:
            for o in (out, res):
                C.require(o != prod and prod not in o.parents,
                          "FIXTURE_TARGETS_PRODUCTION",
                          f"{out} (resolved {res}) is under {prod}")
    for name in (LEDGER_NAME, LOCK_NAME, INDEX_NAME, INDEX_NAME + ".sha256",
                 FAILED_DIR):
        C.require(not (out / name).is_symlink(), "HOLDOUT_OUT_PATH",
                  f"{out / name}: symlink")
    return out


def run_holdout(approval: tuple, *, root: Path, launch_chain: Path,
                analysis_chain: Path, manifest_path: Path, stages,
                data_root: Path, out_dir: Path, decision: tuple | None = None,
                launch_root: Path | None = None,
                hooks: dict | None = None) -> dict:
    """The gated holdout stage (refinement 5). Production: registered
    manifest, data root and output directory; fixture: temporary layouts
    outside results/v05lt."""
    A, EV, EVC, L, PS, RS, S = _mods()
    gate = verify_gate(approval, root=root, launch_chain=launch_chain,
                       analysis_chain=analysis_chain,
                       manifest_path=manifest_path, stages=stages,
                       launch_root=launch_root)
    out = check_out_dir(out_dir, data_root, stages.paths, gate.production)
    out.mkdir(parents=True, exist_ok=True)
    with L.Lock(out / LOCK_NAME, "holdout"):
        return Runner(gate, out, data_root, hooks=hooks).run(decision)


def status(out_dir: Path) -> dict:
    """Read-only: ledger halt state and the published evaluations."""
    led = C.Ledger(Path(out_dir) / LEDGER_NAME)
    hs = _halt_state(led)
    recs = led.records() if led.path.exists() else []
    return {"records": len(recs),
            "unresolved_halt": None if hs["unresolved"] is None else
            _halt_binding(hs["unresolved"]),
            "halted": hs["halted"],
            "complete": any(r["kind"] == "complete" for r in recs),
            "approvals": sorted(set(hs["approvals"]))}


def _pathsha(s: str, what: str) -> tuple:
    path, sep, sha = s.rpartition(":")
    C.require(bool(sep) and bool(path) and _is_sha(sha),
              "HOLDOUT_ARGS_INVALID", f"{what}: expected PATH:SHA256, got "
              f"{s!r}")
    return (Path(path), sha)


def main(argv=None) -> int:
    """Gated invocation (refinement 5). `run` evaluates the real holdout
    only with an approval record whose sha256 is the externally approved
    value recorded in 196 (195 §9.3), bound to the approved launch chain,
    analysis chain, selection manifest and frozen holdout configuration."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    A, EV, EVC, L, PS, RS, S = _mods()
    ap = argparse.ArgumentParser(description=f"{__abbr__} v{__version__}")
    sub = ap.add_subparsers(dest="mode", required=True)
    a1 = sub.add_parser("approve", help="write the approval record")
    a1.add_argument("--out", required=True)
    a1.add_argument("--launch-chain", required=True, help="PATH:SHA256")
    a1.add_argument("--analysis-chain", required=True, help="PATH:SHA256")
    a1.add_argument("--statement", required=True)
    a2 = sub.add_parser("decide", help="decision on the unresolved halt")
    a2.add_argument("--out", required=True)
    a2.add_argument("--decision", required=True, choices=DECISIONS)
    a2.add_argument("--note", required=True)
    a3 = sub.add_parser("run", help="gated holdout evaluation")
    a3.add_argument("--approval", required=True, help="PATH:SHA256")
    a3.add_argument("--launch-chain", required=True)
    a3.add_argument("--analysis-chain", required=True)
    a3.add_argument("--decision", help="PATH:SHA256")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    paths = C.Paths()
    out = registered_out(paths)
    try:
        if a.mode == "status":
            print(json.dumps(status(out), sort_keys=True))
            return 0
        if a.mode == "decide":
            sha = make_decision(Path(a.out), out, a.decision, a.note)
            print(f"HOLDOUT DECISION {sha} {a.out}")
            return 0
        if a.mode == "approve":
            sha = make_approval(
                Path(a.out), root=paths.repo,
                launch_chain=_pathsha(a.launch_chain, "--launch-chain"),
                analysis_chain=_pathsha(a.analysis_chain, "--analysis-chain"),
                manifest_path=registered_manifest(paths),
                statement=a.statement)
            print(f"HOLDOUT APPROVAL {sha} {a.out} (record this sha256 in "
                  "196 before `run`)")
            return 0
        stages = PS.ProductionStages(paths, paths.data_root)
        res = run_holdout(
            _pathsha(a.approval, "--approval"), root=paths.repo,
            launch_chain=Path(a.launch_chain),
            analysis_chain=Path(a.analysis_chain),
            manifest_path=registered_manifest(paths), stages=stages,
            data_root=paths.data_root, out_dir=out,
            decision=(_pathsha(a.decision, "--decision")
                      if a.decision else None))
    except C.V05LTError as exc:
        print(f"HOLDOUT REFUSED {exc.code}")
        return 2
    except Exception as exc:  # noqa: BLE001 -- logged, never swallowed
        logger.error("[%s] UNEXPECTED %s:\n%s", __abbr__, type(exc).__name__,
                     traceback.format_exc())
        print(f"HOLDOUT FAILED UNEXPECTED {type(exc).__name__}")
        return 2
    print(f"HOLDOUT RESULT {res['status']} index={res['index_sha256']} "
          f"records={len(res['records'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

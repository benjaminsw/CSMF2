# SEQREF-MRI-V05LTSMK v0.2 -- scripts.v05lt_smoke
# LIFETIME: KEEP
# =============================================================================
# Purpose: F17 registered smoke run (195 §7 + PA6; plan SEQREF-MRI-V05LTI110
#   v0.1, core + A + B + C, D1(a)). One small real campaign on the frozen
#   subset (k = 4 training + 4 validation volumes of the registered split
#   manifest), seed 2026, both arms, epoch-0 stage + 3 epochs, through the
#   ACCEPTED components only (ProductionStages in fixture mode, real
#   architecture, no model factory in production use of this runner).
#   Outputs are DIAGNOSTIC and live only under seqref_mri/results/v05lt_smoke/;
#   any path in results/v05lt is refused. No holdout contact.
# Commands:
#   freeze  data only (no model): verifies the registered split / case
#           manifests, P0S and P4; copies the frozen subset (copy sha256 =
#           source); writes the subset split manifest, the panel-case manifest
#           (the 4 validation midpoints), the eligibility manifest (registered
#           v05lt_eligibility code, pinned S_ref), the data-content manifest,
#           the ZF record, smoke DC tolerances (registered calibration on the
#           subset midpoints; NOT P5), bank B/C copies and smoke_config.json
#           (+ sidecar), published write-once (RENAME_NOREPLACE).
#   run     --config PATH:SHA256 (the sha256 recorded in V05LTEXEC): every
#           input re-verified; smoke launch chain + launch-inputs record; each
#           stage in a fresh spawned process: epoch-0 stage; train + post-run
#           for lam0 and lam2 (item 24: trained epoch-3 / selected / final
#           through load_trained_checkpoint); interrupted lam2 run (SIGKILL at
#           commit:after_ledger, 2nd occurrence) + resume, compared with the
#           uninterrupted lam2 run (PA6-D); stability (B): lam2 selected
#           checkpoint in two fresh processes; descriptive report (PA6-C; no
#           CI, labels or outcomes; one seed, never duplicated); timing and
#           disk (C). Record smoke_record.json (+ sidecar).
# Declared: the smoke launch chain's pin / gate2a / P1-P6 entries are smoke
#   evidence records (code identity, smoke config, smoke tolerances), NOT the
#   registered gate evidence; the chain sha256 is computed by this runner
#   (not an owner launch approval). These exist only to drive the accepted
#   ProductionStages binding in fixture mode; nothing here is a production
#   gate. Disk use is sampled after each stage (not a continuous peak).
# Fixture-only: `fixture` = {"ref": "module:function", "arg": str} resolving
#   to synthetic sources, P4 / S_ref, holdout ids, banks and a model factory
#   (local pre-check SMK). The CLI never passes it.
# CONVENTION: logger.error + V05LTError(code); no fallback, no placeholder,
#   no mock in this module.
# Changelog (v0.2, 2026-10-28; revision-1 review; run path only -- freeze()
#   is unchanged, so the existing frozen configuration stays as written):
#   * Interruption comparison covers the complete optimiser state (per-
#     parameter step counters and moments, every param-group setting) via
#     adam_diff(); any difference is named.
#   * Stability: both outputs must hold exactly the expected midpoint slices
#     and keys; every required PSNR / SSIM / NLL value present and finite;
#     null NMSE_u only on a slice the frozen eligibility marks ineligible
#     (SMOKE_STABILITY_INVALID); backends of both processes must match.
#   * Backend enforced in every child: torch threads / interop threads /
#     deterministic = the registered CPU setup (12 / 12 / off), checked never
#     set (SMOKE_BACKEND_MISMATCH); recorded per child and re-checked.
#   * Code provenance: the frozen configuration's freeze-code sha256 is kept
#     as written and must be the current file or a registered accepted freeze
#     code (rev-1 98c33b18...); the run code is verified separately against
#     the approved --run-code-sha256 in the parent and every child; both are
#     recorded.
# Changelog (NEW in v0.1, 2026-10-28; SEQREF-MRI-V05LTI110 v0.1):
#   * Introduced freeze / run, frozen subset and panel cases (PA6-B), real
#     child-process kill and comparison with permitted differences (PA6-D),
#     trained-checkpoint stability in two fresh processes (B), descriptive
#     one-seed report (PA6-C), timing and disk record (C).
# =============================================================================
from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import logging
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from seqref_mri.scripts import v05lt_common as C

logger = logging.getLogger("seqref_mri.v05lt_smoke")

__version__ = "0.2"
__abbr__ = "SEQREF-MRI-V05LTSMK"

PLAN = "SEQREF-MRI-V05LTI110 v0.1"
K = 4
SEED = 2026
ARMS = ("lam0", "lam2")
EPOCHS = 3                                         # 195 PA6-A
INTERRUPT = {"arm": "lam2", "stage": "commit:after_ledger", "occurrence": 2,
             "mode": "sigkill"}                    # 195 PA6-D
STAB_ARM, STAB_ROLE = "lam2", "selected"           # addition B
STAB_TOL = {"psnr_db": 0.001, "nll_per_dim": 0.0001}   # P2 limits (195 §8.2)
SMOKE_REL = "seqref_mri/results/v05lt_smoke"
P4_REL = "seqref_mri/results/_diag/p4/scaling_statistics.json"
CONFIG_SCHEMA = "seqref-v05lt-smoke-config/1"
RECORD_SCHEMA = "seqref-v05lt-smoke-record/1"
REPORT_SCHEMA = "seqref-v05lt-smoke-report/1"
CONFIG_NAME = "smoke_config.json"
RECORD_NAME = "smoke_record.json"
MIN_FREE = {"freeze": 10e9, "run": 60e9}
CHILD_TIMEOUT_S = 6 * 3600
# Registered CPU setup (195 §8.1; T3 / STB), enforced in every child (v0.2).
BACKEND = {"threads": 12, "interop_threads": 12, "deterministic": False}
# Freeze code accepted as provenance of an existing frozen configuration
# (v0.2): revision 1, which wrote results/v05lt_smoke/freeze on the host
# (config sha256 9bffdcd6...be80, T23). freeze() is unchanged since.
ACCEPTED_FREEZE_CODE = frozenset({
    "98c33b1809ae1a8ef7654e34870ffd0ce423cb997f123fe223b4429079fb8bbf"})
STAB_KEYS = ("file", "slice_index", "r0_psnr", "r0_ssim", "r0_nmse_u",
             "r0_nll_per_dim", "r1_psnr", "r1_ssim", "r1_nmse_u",
             "r0_dc_psnr", "r1_dc_psnr", "r0_eval_psnr")
STAB_REQUIRED = ("r0_psnr", "r0_ssim", "r0_nll_per_dim", "r1_psnr",
                 "r1_ssim", "r0_dc_psnr", "r1_dc_psnr", "r0_eval_psnr")
STAB_NMSE = ("r0_nmse_u", "r1_nmse_u")
BANK_FILES = {"bank_B": "v05lt_bank_20261023.npy",
              "bank_C": "v05lt_bank_20261024.npy"}
# Registered permitted differences between the interrupted and the
# uninterrupted run (195 PA6-D; V05LTI16C v0.2 item 2).
PERMITTED_COMMIT_KEYS = ("seconds", "session", "store", "resume_state_sha256")
PERMITTED_DIFFS = (
    "commit.seconds (wall-clock timing)",
    "commit.session (session number)",
    "commit.store (store path)",
    "commit.resume_state_sha256 and the resume file's sha256 (both embed the "
    "ledger position)",
    "resume state ledger_seq_before_commit (ledger position)",
    "session_start / restart_uncommitted / resume_refused / run_end-recovered "
    "ledger records and therefore ledger positions")
STATEMENTS = (
    "F17 smoke run (195 §7, PA6): one seed (2026); not evidence about the "
    "lambda effect; no bootstrap, CI, labels or outcomes (PA6-C).",
    "The smoke seed is never duplicated; three-seed statistics are covered "
    "by the synthetic fixtures F06, F15, E2E.",
    "Smoke DC tolerances are calibrated on the 4 subset midpoints by the "
    "registered procedure; they are NOT the P5 tolerances.",
    "Training and validation data of the frozen subset only; no holdout "
    "contact.")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _fx(fixture):
    """Fixture context (fixture-only) or None."""
    if fixture is None:
        return None
    C.require(isinstance(fixture, dict) and set(fixture) == {"ref", "arg"},
              "SMOKE_FIXTURE_INVALID", repr(fixture)[:120])
    mod, _, fn = fixture["ref"].partition(":")
    ctx = getattr(importlib.import_module(mod), fn)(fixture["arg"])
    C.require(isinstance(ctx, dict), "SMOKE_FIXTURE_INVALID", "context")
    return ctx


def _check_out(out: Path, paths: C.Paths, production: bool) -> Path:
    """Smoke root: production = repo/SMOKE_REL exactly; never in or under
    results/v05lt (judged on the resolved destination); not a symlink."""
    out = Path(out)
    C.require(not out.is_symlink(), "SMOKE_OUT_INVALID", f"{out}: symlink")
    res = out.resolve()
    prod = paths.v05lt.resolve()
    C.require(res != prod and prod not in res.parents,
              "SMOKE_TARGETS_PRODUCTION", f"{out} -> {res}")
    if production:
        want = (paths.repo / SMOKE_REL).resolve()
        C.require(res == want, "SMOKE_OUT_INVALID",
                  f"{out}: smoke outputs belong in {want}")
    return out


def _free(path: Path, need: float, what: str) -> int:
    free = shutil.disk_usage(path).free
    C.require(free >= need, "SMOKE_DISK_SPACE", f"{what}: {free / 1e9:.1f} GB "
              f"free at {path}, need >= {need / 1e9:.0f} GB")
    return free


def _du(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in Path(p).rglob("*")
               if f.is_file() and not f.is_symlink())


def _git_head(repo: Path) -> str:
    r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    C.require(r.returncode == 0, "SMOKE_GIT_FAILED", r.stderr.strip())
    return r.stdout.strip()


def _write_with_sidecar(p: Path, doc) -> str:
    sha = C.write_durable_new(p, C.canonical_json(doc))
    C.write_durable_new(Path(str(p) + ".sha256"),
                        f"{sha}  {p.name}\n".encode())
    return sha


def _strict(p: Path, code: str):
    try:
        return json.loads(Path(p).read_bytes(), parse_constant=lambda c:
                          C.fail(code, f"{p}: non-finite constant {c}"))
    except (OSError, ValueError) as exc:
        C.fail(code, f"{p}: {type(exc).__name__}: {exc}")


def _expect(sub: dict, n_holdout: int) -> C.SplitExpect:
    tr = [r["volume_id"] for r in sub["train"]]
    va = [r["volume_id"] for r in sub["validation"]]
    return C.SplitExpect(len(tr), len(va), n_holdout,
                         sum(r["n_slices"] for r in sub["train"]),
                         sum(r["n_slices"] for r in sub["validation"]),
                         C._ids_sha(tr), C._ids_sha(va))


def _registered_sources(paths: C.Paths, fx):
    """-> (split doc, split path, split sha, expect, case doc, case sha,
    data_root, holdout ids, s_ref, p4, p4 sha, bank dir)."""
    if fx is None:
        from seqref_mri.src import free_flow_runtime as ffr
        sp = paths.inputs / "v05lt_split_manifest.json"
        cp = paths.inputs / "v05lt_case_manifest.json"
        ssha = C.REGISTERED_INPUTS[sp.name]
        sdoc = C.load_split_manifest(sp, ssha)
        csha = C.verify_file(cp, C.REGISTERED_INPUTS[cp.name], "CASE")
        cdoc = _strict(cp, "SMOKE_INPUT_INVALID")
        hold = C.holdout_ids(paths.v02_manifests)
        s_ref = C.verify_p0s(paths.p0s)
        p4p = paths.repo / P4_REL
        p4 = ffr.load_p4s2_parent(str(p4p))
        return (sdoc, sp, ssha, C.REGISTERED_SPLIT, cdoc, csha,
                paths.data_root, hold, s_ref, p4, C.sha256_file(p4p),
                paths.inputs)
    sdoc = C.load_split_manifest(Path(fx["source_split"]),
                                 fx["source_split_sha256"], fx["source_expect"])
    cp = Path(fx["source_case"])
    csha = C.sha256_file(cp)
    return (sdoc, Path(fx["source_split"]), fx["source_split_sha256"],
            fx["source_expect"], _strict(cp, "SMOKE_INPUT_INVALID"), csha,
            Path(fx["data_root"]), frozenset(fx["holdout"]), fx["s_ref"],
            fx["p4"], "fixture", Path(fx["bank_dir"]))


# ---------------------------------------------------------------------------
# freeze (data only)
# ---------------------------------------------------------------------------
def freeze(out: Path, paths: C.Paths, fixture=None) -> dict:
    from seqref_mri.scripts import v05lt_datahash as DH
    from seqref_mri.scripts import v05lt_eligibility as E
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts import v05lt_prodstages as PS
    from seqref_mri.scripts import v05lt_train as T
    from seqref_mri.scripts.v02_manifests import midpoint_slice
    fx = _fx(fixture)
    out = _check_out(out, paths, fx is None)
    out.mkdir(parents=True, exist_ok=True)
    fdir, stg = out / "freeze", out / ".freeze.staging"
    for p in (fdir, stg):
        C.require(not os.path.lexists(p), "SMOKE_EXISTS",
                  f"{p} exists (write-once; inspect, keep or remove)")
    free = _free(out, fx.get("min_free", MIN_FREE["freeze"]) if fx else
                 MIN_FREE["freeze"], "freeze")
    t0 = time.monotonic()
    (sdoc, sp, ssha, sexp, cdoc, csha, data_root, hold, s_ref, p4, p4sha,
     bank_dir) = _registered_sources(paths, fx)
    train, val = sdoc["train"][:K], sdoc["validation"][:K]
    C.require(len(train) == K and len(val) == K, "SMOKE_SUBSET",
              f"{len(train)}/{len(val)} volumes")
    stg.mkdir()
    # 1. frozen subset copied; copy sha256 = source
    files = {}
    for r in train + val:
        src, dst = Path(data_root) / r["file"], stg / "data" / r["file"]
        C.require(src.is_file() and not src.is_symlink(), "SMOKE_INPUT_MISSING",
                  str(src))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        a, b = C.sha256_file(src), C.sha256_file(dst)
        C.require(a == b, "SMOKE_COPY_MISMATCH", f"{r['file']}: {a} != {b}")
        files[r["volume_id"]] = {"file": r["file"], "n_slices": r["n_slices"],
                                 "sha256": a, "size": dst.stat().st_size}
    # 2. subset split manifest
    sub = {"schema": C.SPLIT_SCHEMA, "train": train, "validation": val,
           "train_ids_sha256": C._ids_sha([r["volume_id"] for r in train]),
           "validation_ids_sha256": C._ids_sha([r["volume_id"] for r in val]),
           "holdout_overlap": 0}
    ind = stg / "inputs"
    ind.mkdir()
    msha = C.write_durable_new(ind / "split.json", C.canonical_json(sub))
    ex = _expect(sub, len(hold))
    sd = C.load_split_manifest(ind / "split.json", msha, ex)
    _, vv = C.build_split_views(stg / "data", sd, hold, ex)
    # 3. panel cases = the 4 validation midpoints (production cases noted)
    prod_cases = {(c["file"], c["slice_index"]): c["case"]
                  for c in cdoc["cases"]}
    cases, panel = [], []
    for j, r in enumerate(val):
        s = midpoint_slice(r["n_slices"])
        cases.append({"case": j, "file": r["file"], "slice_index": s})
        panel.append({"case": j, "volume_id": r["volume_id"],
                      "file": r["file"], "slice_index": s,
                      "production_case": prod_cases.get((r["file"], s))})
    cdoc2 = {"schema": "seqref-v05lt-smoke-case-manifest/1", "cases": cases}
    cpsha = C.write_durable_new(ind / "case_manifest.json",
                                C.canonical_json(cdoc2))
    # 4. eligibility manifest (registered code)
    recs = E.compute_records(vv, s_ref)
    prov = {"smoke": True, "plan": PLAN, "source_split_sha256": ssha,
            "source_case_sha256": csha, "s_ref": s_ref,
            "s_ref_squared": s_ref ** 2,
            "p0s_sha256": C.P0S_SHA256 if fx is None else "fixture",
            "validation_order_sha256": vv.order_lines_sha256()}
    edoc = E.build_manifest(recs, {"validation": val}, cdoc2, prov)
    esha = _write_with_sidecar(ind / "eligibility.json", edoc)
    from seqref_mri.scripts import v05lt_metrics as M
    el = M.eligibility_index(edoc)
    # 5. data-content manifest (registered code)
    dc = DH.generate(ind / "split.json", msha, hold, stg / "data",
                     ind / DH.OUT_NAME, expect=ex)
    # 6. ZF on the subset (registered writer)
    sts = EV.build_states(vv, list(range(len(vv))), p4, s_ref)
    zrows = [EV.zf_row(st, el[(st["identity"]["file"],
                               int(st["identity"]["slice_index"]))])
             for st in sts]
    zsum = EV.zf_summary(zrows)
    del sts
    EV.write_zf(stg / "zf", zrows, zsum, {"smoke": True, "plan": PLAN})
    zsha = EV.verify_zf(stg / "zf")
    # 7. smoke DC tolerances (registered calibration on the subset midpoints)
    mids = [i for i in range(len(vv)) if el[vv.entry(i)]["midpoint"]]
    C.require(len(mids) == K, "SMOKE_SUBSET", f"{len(mids)} midpoints")
    cal = EV.calibrate_dc(EV.build_states(vv, mids, p4, s_ref))
    (stg / "p5").mkdir()
    tprov = {"smoke": True, "not_P5": True, "plan": PLAN,
             "floors": cal["floors"],
             "argmax_identity": cal["argmax_identity"],
             **cal["provenance"]}
    tsha = PS.make_tolerances(stg / "p5" / "tolerances.json",
                              cal["tolerances"], tprov)
    # 8. bank copies
    banks = {}
    for k, n in BANK_FILES.items():
        dst = ind / n
        shutil.copyfile(Path(bank_dir) / n, dst)
        exp = C.REGISTERED_INPUTS[n] if fx is None else C.sha256_file(dst)
        C.verify_bank_npy(dst, exp, C.BANK_ARRAY_SHA256[n])
        banks[k] = exp
    rel = {"split_manifest": "freeze/inputs/split.json",
           "case_manifest": "freeze/inputs/case_manifest.json",
           "bank_B": f"freeze/inputs/{BANK_FILES['bank_B']}",
           "bank_C": f"freeze/inputs/{BANK_FILES['bank_C']}",
           "data_content": f"freeze/inputs/{DH.OUT_NAME}",
           "eligibility": "freeze/inputs/eligibility.json",
           "zf": f"freeze/zf/{EV.ZF_FILE}",
           "dc_tolerances": "freeze/p5/tolerances.json"}
    shas = {"split_manifest": msha, "case_manifest": cpsha,
            "bank_B": banks["bank_B"], "bank_C": banks["bank_C"],
            "data_content": dc["sha256"], "eligibility": esha, "zf": zsha,
            "dc_tolerances": tsha}
    smry = edoc["summary"]
    cfg = {
        "schema": CONFIG_SCHEMA, "abbr": __abbr__, "version": __version__,
        "plan": PLAN, "amendment": "195 PA6", "fixture": fx is not None,
        "k": K, "seed": SEED,
        "arms": {a: T.ARMS[a] for a in ARMS}, "epochs": EPOCHS,
        "batch": T.BATCH, "lr": T.LEARNING_RATE,
        "interrupt": INTERRUPT, "permitted_differences": list(PERMITTED_DIFFS),
        "stability": {"arm": STAB_ARM, "role": STAB_ROLE, "processes": 2,
                      "population": "subset midpoints", "bank": "primary",
                      "tolerances": STAB_TOL,
                      "checked": ["r0 psnr", "r1 psnr (128)", "r0_dc psnr",
                                  "r1_dc psnr", "r0 nll_per_dim"],
                      "reported": ["ssim", "nmse_u", "bitwise identity"]},
        "subset": {"train": [dict(files[r["volume_id"]],
                                  volume_id=r["volume_id"]) for r in train],
                   "validation": [dict(files[r["volume_id"]],
                                       volume_id=r["volume_id"],
                                       midpoint=midpoint_slice(r["n_slices"]))
                                  for r in val],
                   "train_slices": sum(r["n_slices"] for r in train),
                   "validation_slices": sum(r["n_slices"] for r in val)},
        "panel_cases": panel,
        "eligibility": {"all_slices": {k: smry["all_slices"][k] for k in (
            "slices", "eligible", "excluded", "excluded_identities",
            "zero_energy", "volumes_without_eligible_slice")},
            "midpoint": {k: smry["midpoint"][k] for k in (
                "slices", "eligible", "excluded", "excluded_identities")}},
        "zf_summary": zsum, "dc_tolerances": cal["tolerances"],
        "dc_floors": cal["floors"],
        "data_content": {"n_files": dc["n_files"],
                         "total_bytes": dc["total_bytes"]},
        "sources": {"split_manifest_sha256": ssha, "case_manifest_sha256":
                    csha, "p0s_sha256": prov["p0s_sha256"], "s_ref": s_ref,
                    "p4_sha256": p4sha},
        "inputs": {k: {"path": rel[k], "sha256": shas[k]} for k in rel},
        "code": {"smoke_sha256": C.sha256_file(Path(__file__)),
                 "repo_head": _git_head(Path(paths.repo))},
        "free_bytes_at_freeze": free}
    csha2 = _write_with_sidecar(stg / CONFIG_NAME, cfg)
    for f in sorted(stg.rglob("*")):
        if f.is_file():
            os.chmod(f, 0o444)
    from seqref_mri.scripts.v05lt_eval import publish_noreplace
    publish_noreplace(stg, fdir)
    secs = round(time.monotonic() - t0, 1)
    logger.info("[%s] freeze published %s config sha256 %s (%.1f s)",
                __abbr__, fdir, csha2, secs)
    return {"config": str(fdir / CONFIG_NAME), "config_sha256": csha2,
            "seconds": secs, "bytes": _du(fdir),
            "eligible": smry["all_slices"]["eligible"],
            "slices": smry["all_slices"]["slices"]}


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def parse_config(p: Path, sha: str, production: bool) -> dict:
    """The frozen configuration file itself (no input re-verification):
    sha256 = the supplied recorded value, sidecar, strict JSON, schema, mode
    and frozen values, freeze-code provenance. Never modifies the file."""
    p = Path(p)
    C.require(isinstance(sha, str) and len(sha) == 64,
              "SMOKE_CONFIG_UNAPPROVED", "recorded sha256 required")
    C.require(p.is_file() and not p.is_symlink(), "SMOKE_CONFIG_INVALID",
              f"{p}: not a regular file")
    got = C.sha256_file(p)
    C.require(got == sha, "SMOKE_CONFIG_UNAPPROVED",
              f"{p}: {got} != recorded {sha}")
    C.require(C.read_sidecar(Path(str(p) + ".sha256"), "SMOKE_CONFIG_INVALID")
              == got, "SMOKE_CONFIG_INVALID", "sidecar")
    cfg = _strict(p, "SMOKE_CONFIG_INVALID")
    C.require(isinstance(cfg, dict) and cfg.get("schema") == CONFIG_SCHEMA
              and cfg.get("fixture") is (not production) and
              cfg.get("k") == K and cfg.get("seed") == SEED and
              cfg.get("epochs") == EPOCHS and cfg.get("interrupt") == INTERRUPT,
              "SMOKE_CONFIG_INVALID", "schema / mode / frozen values")
    check_freeze_code(cfg)
    return {**cfg, "_sha256": got, "_path": str(p)}


def load_config(out: Path, ref: str, production: bool) -> dict:
    """ref = PATH:SHA256; PATH must be <out>/freeze/smoke_config.json; the
    file is checked against the supplied sha256 (recording it in V05LTEXEC
    and supplying that value is the owner's approval step); every listed
    input re-verified."""
    path, _, sha = ref.rpartition(":")
    C.require(len(sha) == 64 and path, "SMOKE_CONFIG_UNAPPROVED",
              "--config PATH:SHA256 required")
    p = Path(path)
    C.require(p.resolve() == (Path(out) / "freeze" / CONFIG_NAME).resolve(),
              "SMOKE_CONFIG_INVALID", f"{p}: not the frozen configuration "
              f"of {out}")
    cfg = parse_config(p, sha, production)
    for k, e in cfg["inputs"].items():
        f = Path(out) / e["path"]
        C.require(f.is_file() and not f.is_symlink() and
                  C.sha256_file(f) == e["sha256"], "SMOKE_INPUT_HASH",
                  f"{k}: {f}")
    return cfg


def check_freeze_code(cfg: dict) -> str:
    """The frozen configuration's freeze-code sha256 is kept as written; it
    must be the current smoke file or a registered accepted freeze code."""
    code = cfg.get("code")
    fc = code.get("smoke_sha256") if isinstance(code, dict) else None
    cur = C.sha256_file(Path(__file__))
    C.require(isinstance(fc, str) and (fc == cur or fc in
                                       ACCEPTED_FREEZE_CODE),
              "SMOKE_FREEZE_CODE_UNKNOWN", f"freeze code {fc!r} is neither "
              f"this file ({cur}) nor a registered accepted freeze code")
    return fc


def check_run_code(approved: str) -> str:
    """The running smoke file must be the approved run code."""
    cur = C.sha256_file(Path(__file__))
    C.require(isinstance(approved, str) and cur == approved,
              "SMOKE_RUN_CODE_UNAPPROVED", f"running {cur} != approved "
              f"{approved!r}")
    return cur


def check_backend(expected: dict) -> dict:
    """Registered CPU setup, checked (never set)."""
    from seqref_mri.scripts import v05lt_prodstages as PS
    got = PS.backend_now()
    C.require(got == expected, "SMOKE_BACKEND_MISMATCH",
              f"backend {got} != expected {expected}")
    return got


def _stages(out: Path, cfg: dict, paths: C.Paths, fx, hooks=None):
    """Fresh ProductionStages (fixture mode) over the frozen subset."""
    from seqref_mri.scripts import v05lt_prodstages as PS
    from seqref_mri.src import free_flow_runtime as ffr
    sub = {"train": cfg["subset"]["train"],
           "validation": cfg["subset"]["validation"]}
    if fx is None:
        hold = C.holdout_ids(paths.v02_manifests)
        s_ref = C.verify_p0s(paths.p0s)
        p4p = paths.repo / P4_REL
        C.require(C.sha256_file(p4p) == cfg["sources"]["p4_sha256"],
                  "SMOKE_INPUT_HASH", "P4 parent changed since freeze")
        p4 = ffr.load_p4s2_parent(str(p4p))
    else:
        hold, s_ref, p4 = frozenset(fx["holdout"]), fx["s_ref"], fx["p4"]
    C.require(s_ref == cfg["sources"]["s_ref"], "SMOKE_INPUT_HASH", "S_ref")
    fxd = {"root": Path(out) / "run" / "campaign", "max_epochs": EPOCHS,
           "p4": p4, "s_ref": s_ref, "split_expect": _expect(sub, len(hold)),
           "holdout": hold, "repo": Path(paths.repo),
           "hooks": hooks if hooks is not None else {}}
    if fx is not None and fx.get("model_factory") is not None:
        fxd["model_factory"] = fx["model_factory"]
    return PS.ProductionStages(paths, Path(out) / "freeze" / "data",
                               fixture=fxd)


def _bind(st, out: Path, chain: dict) -> None:
    from seqref_mri.scripts import v05lt_launch as L
    doc = L.verify_launch_chain(Path(chain["path"]), chain["sha256"], out)
    st.bind_chain(doc, out)


def _make_chain(out: Path, cfg: dict, paths: C.Paths, fx) -> dict:
    """Smoke launch chain (declared: smoke evidence records, not gate
    evidence) + launch-inputs record over the frozen inputs."""
    from seqref_mri.scripts import v05lt_launch as L
    from seqref_mri.scripts import v05lt_prodstages as PS
    ev = Path(out) / "run" / "ev"
    ev.mkdir(parents=True)
    st = _stages(out, cfg, paths, fx)
    roles = {}
    for r in L.CHAIN_ROLES:
        if r == PS.LAUNCH_INPUTS_ROLE:
            continue
        body = {"role": r, "smoke": True, "plan": PLAN,
                "note": "F17 smoke chain entry; not registered gate evidence"}
        if r == "pin":
            body["code"] = st.code_identity()
        if r == "gate2a":
            body["smoke_config_sha256"] = cfg["_sha256"]
        if r == "P5":
            body["smoke_tolerances_sha256"] = \
                cfg["inputs"]["dc_tolerances"]["sha256"]
        C.write_durable_new(ev / f"{r}.json", C.canonical_json(body))
        roles[r] = f"run/ev/{r}.json"
    li = PS.make_launch_inputs(
        ev / "launch_inputs.json", Path(out), st,
        chain_files={r: roles[r] for r in PS.BOUND_ROLES},
        inputs={k: e["path"] for k, e in cfg["inputs"].items()},
        commit=_git_head(Path(paths.repo)))
    roles[PS.LAUNCH_INPUTS_ROLE] = "run/ev/launch_inputs.json"
    cp = Path(out) / "run" / "chain.json"
    sha = L.make_launch_chain(cp, Path(out), roles)
    return {"path": str(cp), "sha256": sha, "launch_inputs_sha256": li}


# -- children (fresh spawned processes) ---------------------------------------
def _interrupt_config(st):
    """lam2 TrainConfig with the same identity, separate run directory."""
    cfg = st.train_config(SEED, INTERRUPT["arm"])
    return dataclasses.replace(cfg, run_dir=str(
        Path(st.root).parent / "interrupt" / f"{SEED}_{INTERRUPT['arm']}"))


def _stability_eval(st, out: Path) -> dict:
    """Addition B: the lam2 selected checkpoint (verified against its commit)
    on the subset midpoints: R0 (PSNR, SSIM, NMSE_u, NLL/dim) via the shared
    validation, R1 (128) and R0+DC / R1+DC via evaluate_states."""
    import torch
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts import v05lt_metrics as M
    from seqref_mri.scripts import v05lt_results as RS
    man = st.manifests()
    run = RS.load_run(st.run_dir(SEED, STAB_ARM), STAB_ARM, SEED, man,
                      RS.Identities(), fixture=True)
    ck = run["checkpoints"][STAB_ROLE]
    name = f"weights_e{ck['epoch']:03d}.pt"
    hit = [v for v in run["retained"].values() if v["name"] == name]
    C.require(len(hit) >= 1, "SMOKE_STABILITY_SOURCE", f"{name} not retained")
    model, vck = EV.load_trained_checkpoint(
        Path(run["ckpt_dir"]) / name, hit[0]["sha256"],
        ck["model_state_sha256"], seed=SEED, arm=STAB_ARM, epoch=ck["epoch"],
        model_factory=st._factory())
    data = st.data()
    vv, el = data["vv"], data["eligibility"]
    mids = [i for i in range(len(vv)) if el[vv.entry(i)]["midpoint"]]
    rows = M.validate(model, vv, el, data["p4"], data["s_ref"], mids)
    with torch.no_grad():
        states = EV.build_states(vv, mids, data["p4"], data["s_ref"])
        ev_rows = EV.evaluate_states(
            model, states, [bool(el[vv.entry(i)]["eligible"]) for i in mids],
            {"primary": st.banks()["primary"]}, prefixes=(EV.BANK_ROWS,),
            dc_tol=st._bound()["tolerances"])
    out_rows = []
    for r, e in zip(rows, ev_rows):
        out_rows.append({
            "file": r["file"], "slice_index": r["slice_index"],
            "r0_psnr": r["psnr"], "r0_ssim": r["ssim"],
            "r0_nmse_u": r.get("nmse_u"), "r0_nll_per_dim": r["nll_per_dim"],
            "r1_psnr": e["r1"]["primary"][str(EV.BANK_ROWS)]["psnr"],
            "r1_ssim": e["r1"]["primary"][str(EV.BANK_ROWS)]["ssim"],
            "r1_nmse_u": e["r1"]["primary"][str(EV.BANK_ROWS)]["nmse_u"],
            "r0_dc_psnr": e["r0_dc"]["psnr"], "r1_dc_psnr": e["r1_dc"]["psnr"],
            "r0_eval_psnr": e["r0"]["psnr"]})
    return {"checkpoint": vck.as_record(), "rows": out_rows,
            "backend": EV.backend_identity(),
            "threads": torch.get_num_threads()}


def _child(spec_path: str) -> None:
    """Fresh spawned process: one stage. Writes a JSON result; a SIGKILL on
    purpose (interruption test) leaves no result."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s "
                        "%(name)s[child]: %(message)s")
    spec = json.loads(Path(spec_path).read_text())
    res = {"action": spec["action"], "pid": os.getpid()}
    t0 = time.monotonic()
    try:
        from seqref_mri.scripts import v05lt_train as T
        paths = C.Paths(repo=Path(spec["repo"]))
        fx = _fx(spec.get("fixture"))
        out = Path(spec["out"])
        res["run_code_sha256"] = check_run_code(spec["run_code_sha256"])
        res["backend"] = check_backend(spec["backend"])
        cfg = load_config(out, spec["config"], fx is None)
        counts = {"n": 0}

        def kill_hook(stage):
            if stage == INTERRUPT["stage"]:
                counts["n"] += 1
                if counts["n"] == INTERRUPT["occurrence"]:
                    logger.warning("smoke interruption: SIGKILL at %s "
                                   "(occurrence %d)", stage, counts["n"])
                    os.kill(os.getpid(), signal.SIGKILL)
        st = _stages(out, cfg, paths, fx)
        _bind(st, out, spec["chain"])
        a = spec["action"]
        if a == "epoch0":
            st.verify_data()
            res["result"] = st.epoch0(SEED)
            res["result"]["done"] = st.epoch0_done(SEED)
        elif a.startswith("train:"):
            st.verify_data()
            r = st.train(SEED, a.split(":")[1], "start")
            res["result"] = {"status": r["status"], "epoch": r["epoch"]}
        elif a.startswith("post:"):
            res["result"] = st.post_run(SEED, a.split(":")[1])
        elif a in ("interrupt:start", "interrupt:resume"):
            st.verify_data()
            run = T.Run(_interrupt_config(st), paths, st.data(),
                        model_factory=st._factory(),
                        hook=kill_hook if a == "interrupt:start" else None)
            r = run.start() if a == "interrupt:start" else run.resume()
            res["result"] = {"status": r["status"], "epoch": r["epoch"]}
        elif a.startswith("stability:"):
            r = _stability_eval(st, out)
            p = Path(spec["out_file"])
            C.write_durable_new(p, C.canonical_json(r))
            res["result"] = {"file": str(p), "sha256": C.sha256_file(p),
                             "threads": r["threads"]}
        else:
            C.fail("SMOKE_ACTION_INVALID", a)
        res["ok"] = True
    except C.V05LTError as exc:
        logger.error("child %s refused: %s: %s", spec["action"], exc.code, exc)
        res["error"] = {"code": exc.code, "message": str(exc)[:600]}
    except Exception as exc:  # noqa: BLE001 -- reported to the parent, logged
        logger.exception("child %s failed", spec["action"])
        res["error"] = {"code": "SMOKE_CHILD_UNEXPECTED",
                        "message": f"{type(exc).__name__}: {exc}"[:600],
                        "traceback": traceback.format_exc()[-4000:]}
    res["seconds"] = round(time.monotonic() - t0, 1)
    Path(spec["result"]).write_text(json.dumps(res, default=str))


def _spawn(out: Path, spec: dict, n: int, expect_kill: bool = False) -> dict:
    d = Path(out) / "run" / "children"
    d.mkdir(parents=True, exist_ok=True)
    sp, rp = d / f"spec_{n:02d}.json", d / f"result_{n:02d}.json"
    sp.write_text(json.dumps({**spec, "result": str(rp)}, default=str))
    logger.info("[%s] child %d: %s", __abbr__, n, spec["action"])
    t0 = time.monotonic()
    p = multiprocessing.get_context("spawn").Process(target=_child,
                                                     args=(str(sp),))
    p.start()
    p.join(CHILD_TIMEOUT_S)
    if p.is_alive():
        p.kill()
        p.join(30)
        C.fail("SMOKE_CHILD_TIMEOUT", f"{spec['action']}: > {CHILD_TIMEOUT_S} s")
    secs = round(time.monotonic() - t0, 1)
    if expect_kill:
        C.require(p.exitcode == -signal.SIGKILL and not rp.exists(),
                  "SMOKE_NOT_KILLED", f"{spec['action']}: exit {p.exitcode}")
        return {"exit": p.exitcode, "seconds": secs}
    C.require(rp.is_file(), "SMOKE_CHILD_FAILED",
              f"{spec['action']}: exit {p.exitcode}, no result")
    res = json.loads(rp.read_text())
    if "error" in res:
        logger.error("child %s error %s: %s", spec["action"],
                     res["error"]["code"], res["error"]["message"])
        C.fail(res["error"]["code"], f"child {spec['action']}: "
               f"{res['error']['message']}")
    C.require(p.exitcode == 0 and res.get("ok") is True, "SMOKE_CHILD_FAILED",
              f"{spec['action']}: exit {p.exitcode}")
    res["wall_seconds"] = secs
    return res


# -- comparison (PA6-D) ---------------------------------------------------------
def check_kill_boundary(run_dir: Path) -> dict:
    """After the SIGKILL, before the resume: exactly the declared boundary
    (occurrence-th epoch commit durable, no run_end)."""
    kinds = [r["kind"] for r in C.Ledger(Path(run_dir) / "ledger.jsonl")
             .records()]
    got = {"commits": kinds.count("commit"), "run_end": kinds.count("run_end")}
    C.require(got == {"commits": INTERRUPT["occurrence"], "run_end": 0},
              "SMOKE_INTERRUPT", f"kill boundary not as declared: {got}")
    return got


def _commits(run_dir: Path) -> dict:
    out = {}
    for r in C.Ledger(Path(run_dir) / "ledger.jsonl").records():
        if r["kind"] == "commit":
            b = json.loads(json.dumps({k: v for k, v in r["body"].items()
                                       if k not in PERMITTED_COMMIT_KEYS}))
            b["retained"]["resume"].pop("sha256", None)
            out[b["epoch"]] = b
    return out


def _final_state(run_dir: Path):
    from seqref_mri.scripts import v05lt_train as T
    run_dir = Path(run_dir)
    st = C.ArtifactStore(run_dir / "ckpt", C.Ledger(run_dir / "ledger.jsonl"),
                         run_dir / "_incomplete").committed()
    obj = T.load_resume(run_dir / "ckpt" / st["resume"]["name"])
    state = {k: v for k, v in obj["state"].items()
             if k != "ledger_seq_before_commit"}
    return C.model_state_sha256(obj["model"]), state, obj["adam"]


def adam_diff(a, b, where: str = "adam"):
    """First difference between two complete optimiser state dicts (state:
    per-parameter step counters and moments; param_groups: every setting),
    or None. Tensors compared by dtype, shape and value (torch.equal);
    everything else by type and value."""
    import torch
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return f"{where}: tensor vs {type(b).__name__}"
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            return f"{where}: tensor differs"
        return None
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return f"{where}: dict vs {type(b).__name__}"
        if set(a) != set(b):
            return f"{where}: keys {sorted(map(str, set(a) ^ set(b)))[:4]}"
        for k in sorted(a, key=str):
            d = adam_diff(a[k], b[k], f"{where}.{k}")
            if d:
                return d
        return None
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            return f"{where}: sequence type / length"
        for i, (x, y) in enumerate(zip(a, b)):
            d = adam_diff(x, y, f"{where}[{i}]")
            if d:
                return d
        return None
    if type(a) is not type(b) or a != b:
        return f"{where}: {a!r} != {b!r}"
    return None


def compare_runs(ref: Path, other: Path) -> dict:
    ca, cb = _commits(ref), _commits(other)
    diff_epochs = sorted(e for e in set(ca) | set(cb) if ca.get(e) != cb.get(e))
    a, b = _final_state(ref), _final_state(other)
    ad = adam_diff(a[2], b[2])
    kinds = [r["kind"] for r in C.Ledger(Path(other) / "ledger.jsonl")
             .records()]
    return {"epoch_records_identical": not diff_epochs,
            "differing_epochs": diff_epochs,
            "model_state_identical": a[0] == b[0],
            "resume_state_identical": a[1] == b[1],
            "adam_identical": ad is None, "adam_difference": ad,
            "sessions": kinds.count("session_start"),
            "run_end": kinds.count("run_end"),
            "epochs": sorted(ca)}


# -- descriptive report (PA6-C) -------------------------------------------------
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def descriptive(st, out: Path, cfg: dict) -> dict:
    from seqref_mri.scripts import v05lt_eval as EV
    from seqref_mri.scripts import v05lt_results as RS
    man = st.manifests()
    ids, shared = RS.Identities(), {}
    e0ck = st.epoch0_checkpoint(SEED)
    e0 = RS.load_epoch0_eval(st._e0_dir(SEED) / "eval", SEED,
                             e0ck["model_state_sha256"], man, ids, {})
    zdoc = _strict(Path(out) / cfg["inputs"]["zf"]["path"],
                   "SMOKE_INPUT_INVALID")
    C.require(EV.verify_zf(Path(out) / "freeze" / "zf") ==
              cfg["inputs"]["zf"]["sha256"], "SMOKE_INPUT_HASH", "zf")

    def ro(src):
        return {k: {"psnr": _mean(r.get("psnr") for r in v),
                    "ssim": _mean(r.get("ssim") for r in v),
                    "nmse_u": _mean(r.get("nmse_u") for r in v), "n": len(v)}
                for k, v in src["readouts"].items()}

    arms = {}
    for a in ARMS:
        run = RS.load_run(st.run_dir(SEED, a), a, SEED, man, ids, fixture=True)
        roles = {}
        for role in RS.POSTEVAL_ROLES:
            pe = RS.load_posteval_for(st._posteval_root(), run, role, man, ids,
                                      shared)
            dcid = {k: sum(1 for r in pe["dc"][k] if r["identical"])
                    for k in pe["dc"]}
            roles[role] = {"epoch": pe["epoch"],
                           "model_state_sha256": pe["model_state_sha256"],
                           "record_sha256": pe["record_sha256"],
                           "midpoint_means": ro(pe),
                           "dc_identical_slices": dcid}
        arms[a] = {"status": run["status"], "last_epoch": run["last_epoch"],
                   "selected_epoch": run["checkpoints"]["selected"]["epoch"],
                   "best_nll_epoch": run["checkpoints"]["best_nll"]["epoch"],
                   "learning_curve": {e: v["validation"]["all_slices"]
                                      for e, v in run["epochs"].items()},
                   "post_run": roles}
    return {"schema": REPORT_SCHEMA, "abbr": __abbr__, "version": __version__,
            "statements": list(STATEMENTS), "seed": SEED,
            "zf": {"summary": zdoc.get("summary")},
            "epoch0": {"all_slice_summary": e0["summary"],
                       "midpoint_means": ro(e0)},
            "arms": arms}


def _report_md(rep: dict) -> str:
    lines = [f"# F17 smoke report ({PLAN})", ""]
    lines += [f"- {s}" for s in rep["statements"]] + [""]
    lines.append("| Arm | Status | Last | Selected | Best NLL |")
    lines.append("|---|---|---|---|---|")
    for a, v in rep["arms"].items():
        lines.append(f"| {a} | {v['status']} | {v['last_epoch']} | "
                     f"{v['selected_epoch']} | {v['best_nll_epoch']} |")
    lines += ["", "| Arm | Role | Epoch | R0 PSNR | R1 PSNR | R0+DC | R1+DC |",
              "|---|---|---|---|---|---|---|"]

    def f(x):
        return "—" if x is None else f"{x:.4f}"
    for a, v in rep["arms"].items():
        for role, r in v["post_run"].items():
            m = r["midpoint_means"]
            lines.append(f"| {a} | {role} | {r['epoch']} | "
                         f"{f(m.get('r0', {}).get('psnr'))} | "
                         f"{f(m.get('r1', {}).get('psnr'))} | "
                         f"{f(m.get('r0_dc', {}).get('psnr'))} | "
                         f"{f(m.get('r1_dc', {}).get('psnr'))} |")
    lines += ["", "Learning curves (all-slice validation):", ""]
    for a, v in rep["arms"].items():
        for e, s in sorted(v["learning_curve"].items(), key=lambda x:
                           int(x[0])):
            lines.append(f"- {a} epoch {e}: PSNR {f(s.get('psnr'))} dB, "
                         f"NLL/dim {f(s.get('nll_per_dim'))}")
    return "\n".join(lines) + "\n"


def expected_midpoints(cfg: dict) -> dict:
    """{(file, midpoint slice): eligible} from the frozen configuration (all-
    slice eligibility exclusions), in validation order."""
    excl = set(cfg["eligibility"]["all_slices"]["excluded_identities"])
    return {(r["file"], r["midpoint"]): f"{r['file']}|{r['midpoint']}"
            not in excl for r in cfg["subset"]["validation"]}


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and \
        bool(np.isfinite(x))


def check_stability_doc(doc: dict, expected: dict, where: str) -> None:
    """Exactly the expected slices (in order) and keys; required values
    present and finite; null NMSE_u only on an ineligible slice."""
    rows = doc.get("rows") if isinstance(doc, dict) else None
    C.require(isinstance(rows, list) and [(r.get("file"), r.get("slice_index"))
                                         for r in rows] == list(expected),
              "SMOKE_STABILITY_INVALID", f"{where}: slices != expected "
              f"{list(expected)}")
    for r in rows:
        key = (r["file"], r["slice_index"])
        C.require(set(r) == set(STAB_KEYS), "SMOKE_STABILITY_INVALID",
                  f"{where} {key}: keys {sorted(set(r) ^ set(STAB_KEYS))}")
        bad = [k for k in STAB_REQUIRED if not _finite(r[k])]
        C.require(not bad, "SMOKE_STABILITY_INVALID",
                  f"{where} {key}: missing or non-finite {bad}")
        for k in STAB_NMSE:
            ok = _finite(r[k]) or (r[k] is None and not expected[key])
            C.require(ok, "SMOKE_STABILITY_INVALID", f"{where} {key}: {k}="
                      f"{r[k]!r} (eligible={expected[key]})")
    C.require(isinstance(doc.get("backend"), dict) and
              isinstance(doc.get("checkpoint"), dict),
              "SMOKE_STABILITY_INVALID", f"{where}: backend / checkpoint")


def _stability_compare(a: dict, b: dict, expected: dict) -> dict:
    check_stability_doc(a, expected, "stability 1")
    check_stability_doc(b, expected, "stability 2")
    C.require(a["checkpoint"] == b["checkpoint"], "SMOKE_STABILITY_MISMATCH",
              "different checkpoint")
    C.require(a["backend"] == b["backend"] and a["threads"] == b["threads"],
              "SMOKE_STABILITY_MISMATCH", f"backends differ: {a['backend']} "
              f"vs {b['backend']}")
    d = {}
    for k in STAB_KEYS[2:]:
        vals = [(x[k], y[k]) for x, y in zip(a["rows"], b["rows"])]
        C.require(all((x is None) == (y is None) for x, y in vals),
                  "SMOKE_STABILITY_MISMATCH", f"{k}: null in one process only")
        d[k] = max((abs(x - y) for x, y in vals if x is not None), default=0.0)
    psnr_ok = all(d[k] <= STAB_TOL["psnr_db"] for k in
                  ("r0_psnr", "r1_psnr", "r0_dc_psnr", "r1_dc_psnr",
                   "r0_eval_psnr"))
    nll_ok = d["r0_nll_per_dim"] <= STAB_TOL["nll_per_dim"]
    return {"max_abs_diff": d, "psnr_within": psnr_ok, "nll_within": nll_ok,
            "bitwise_identical": a["rows"] == b["rows"],
            "backend_equal": True, "backend": a["backend"],
            "n_slices": len(a["rows"])}


def run(out: Path, paths: C.Paths, config_ref: str, run_code_sha256: str,
        fixture=None) -> dict:
    fx = _fx(fixture)
    out = _check_out(out, paths, fx is None)
    run_code = check_run_code(run_code_sha256)
    backend = BACKEND if fx is None else fx["backend"]
    check_backend(backend)
    cfg = load_config(out, config_ref, fx is None)
    freeze_code = check_freeze_code(cfg)
    rdir = out / "run"
    C.require(not os.path.lexists(rdir), "SMOKE_EXISTS",
              f"{rdir} exists (DIAGNOSTIC output; inspect and remove before a "
              "new run)")
    free0 = _free(out, fx.get("min_free", MIN_FREE["run"]) if fx else
                  MIN_FREE["run"], "run")
    rdir.mkdir()
    t_all = time.monotonic()
    chain = _make_chain(out, cfg, paths, fx)
    base = {"repo": str(paths.repo), "out": str(out), "config": config_ref,
            "chain": chain, "fixture": fixture, "run_code_sha256": run_code,
            "backend": backend}
    timing, disk, n = {}, {}, 0

    def step(action, **kw):
        nonlocal n
        n += 1
        r = _spawn(out, {**base, "action": action, **kw.pop("spec", {})}, n,
                   **kw)
        timing[action] = r.get("wall_seconds", r.get("seconds"))
        disk[action] = {"run_bytes": _du(rdir),
                        "free_bytes": shutil.disk_usage(out).free}
        return r
    results = {"epoch0": step("epoch0")}
    for a in ARMS:
        results[f"train:{a}"] = step(f"train:{a}")
        results[f"post:{a}"] = step(f"post:{a}")
    results["interrupt:start"] = step("interrupt:start", expect_kill=True)
    idir = rdir / "interrupt" / f"{SEED}_{INTERRUPT['arm']}"
    killed_at = check_kill_boundary(idir)
    results["interrupt:resume"] = step("interrupt:resume")
    st = _stages(out, cfg, paths, fx)
    _bind(st, out, chain)
    cmp = compare_runs(st.run_dir(SEED, INTERRUPT["arm"]), idir)
    cmp["killed_at"] = killed_at
    sdir = rdir / "stability"
    sdir.mkdir()
    stab = []
    for i in (1, 2):
        r = step(f"stability:{i}", spec={"out_file": str(sdir /
                                                         f"stability_{i}.json")})
        results[f"stability:{i}"] = r
        stab.append(json.loads(Path(r["result"]["file"]).read_bytes()))
    scmp = _stability_compare(*stab, expected_midpoints(cfg))
    rep = descriptive(st, out, cfg)
    repd = rdir / "report"
    repd.mkdir()
    rsha = _write_with_sidecar(repd / "smoke_report.json", rep)
    msha = C.write_durable_new(repd / "smoke_report.md",
                               _report_md(rep).encode())
    # item 24: trained (not epoch-0) checkpoints evaluated from their files
    item24 = {}
    e0ms = st.epoch0_checkpoint(SEED)["model_state_sha256"]
    for a in ARMS:
        tasks = results[f"post:{a}"]["result"]["tasks"]
        roles = sorted(x for t in tasks for x in t["roles"])
        item24[a] = {"roles": roles, "tasks": len(tasks),
                     "trained": all(t["model_state"] != e0ms and t["epoch"] >= 1
                                    for t in tasks)}
    sizes = {}
    for a in ARMS:
        ck = st.run_dir(SEED, a) / "ckpt"
        sizes[a] = {"run_dir_bytes": _du(st.run_dir(SEED, a)),
                    "files": {p.name: p.stat().st_size for p in ck.iterdir()
                              if p.is_file() and not p.name.endswith(".sha256")}}
    checks = {
        # RNG restoration around the epoch-0 stage is enforced inside
        # ProductionStages.epoch0 (PROD_RNG_NOT_RESTORED), so a result exists
        # only if it held.
        "epoch0_done": results["epoch0"]["result"].get("done") is True,
        "runs_completed": all(results[f"train:{a}"]["result"]["epoch"] ==
                              EPOCHS for a in ARMS),
        "item24_roles": all(set(v["roles"]) >= {"epoch3", "selected", "final"}
                            and v["trained"] for v in item24.values()),
        "post_run_verified": all(results[f"post:{a}"]["result"]["verified"]
                                 for a in ARMS),
        "interrupt_identical": cmp["epoch_records_identical"] and
        cmp["model_state_identical"] and cmp["resume_state_identical"] and
        cmp["adam_identical"] and cmp["sessions"] == 2 and cmp["run_end"] == 1,
        "stability_within_P2": scmp["psnr_within"] and scmp["nll_within"],
        "children_backend": all(r.get("backend") == backend for k, r in
                                results.items() if k != "interrupt:start"),
        "children_run_code": all(r.get("run_code_sha256") == run_code
                                 for k, r in results.items()
                                 if k != "interrupt:start")}
    record = {
        "schema": RECORD_SCHEMA, "abbr": __abbr__, "version": __version__,
        "plan": PLAN, "fixture": fx is not None,
        "config_sha256": cfg["_sha256"], "chain": chain,
        "code": {"freeze_code_sha256": freeze_code,
                 "run_code_sha256": run_code,
                 "freeze_code_is_run_code": freeze_code == run_code},
        "backend_expected": backend,
        "results": results, "interruption": cmp,
        "permitted_differences": list(PERMITTED_DIFFS),
        "stability": {**scmp, "files": [str(sdir / f"stability_{i}.json")
                                        for i in (1, 2)]},
        "item24": item24, "report": {"json_sha256": rsha, "md_sha256": msha},
        "timing_seconds": {**timing,
                           "total": round(time.monotonic() - t_all, 1)},
        "disk": {"after_stage": disk, "free_at_start": free0,
                 "run_sizes": sizes, "peak_sampled_run_bytes":
                 max(v["run_bytes"] for v in disk.values())},
        "checks": checks, "statements": list(STATEMENTS)}
    rec_sha = _write_with_sidecar(rdir / RECORD_NAME, record)
    bad = sorted(k for k, v in checks.items() if not v)
    C.require(not bad, "SMOKE_CHECK_FAILED", f"failed checks {bad}; record "
              f"{rec_sha}")
    logger.info("[%s] smoke run PASS record %s", __abbr__, rec_sha)
    return {"record": str(rdir / RECORD_NAME), "record_sha256": rec_sha,
            "checks": checks}


# ---------------------------------------------------------------------------
# CLI (production paths only; fixtures call freeze / run directly)
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="V05LT F17 smoke run (195 PA6)")
    ap.add_argument("command", choices=["freeze", "run"])
    ap.add_argument("--repo", default=str(C.Paths().repo))
    ap.add_argument("--config", default=None,
                    help="run: PATH:SHA256 recorded in V05LTEXEC")
    ap.add_argument("--run-code-sha256", default=None,
                    help="run: approved sha256 of this smoke file")
    a = ap.parse_args(argv)
    paths = C.Paths(repo=Path(a.repo))
    out = paths.repo / SMOKE_REL
    try:
        if a.command == "freeze":
            r = freeze(out, paths)
            print(f"RESULT FREEZE config={r['config']} "
                  f"sha256={r['config_sha256']} eligible={r['eligible']}/"
                  f"{r['slices']} seconds={r['seconds']}")
        else:
            C.require(a.config is not None, "SMOKE_CONFIG_UNAPPROVED",
                      "--config PATH:SHA256 required")
            C.require(a.run_code_sha256 is not None,
                      "SMOKE_RUN_CODE_UNAPPROVED", "--run-code-sha256 required")
            r = run(out, paths, a.config, a.run_code_sha256)
            print(f"RESULT PASS record={r['record']} "
                  f"sha256={r['record_sha256']}")
    except C.V05LTError as exc:
        logger.error("RESULT FAIL %s: %s", exc.code, exc)
        print(f"RESULT FAIL {exc.code}")
        return 2
    except Exception as exc:  # noqa: BLE001 -- logged, never swallowed
        logger.exception("UNEXPECTED: %s", exc)
        print(f"RESULT FAIL UNEXPECTED {type(exc).__name__}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

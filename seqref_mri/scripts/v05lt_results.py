# SEQREF-MRI-V05LTRES v0.6 -- scripts.v05lt_results
# LIFETIME: KEEP
# =============================================================================
# Purpose: verified, read-only loading of V05LT result records into the
#   inputs of the registered analysis (I1.8(b) stage b1, plan
#   SEQREF-MRI-V05LTI18B v0.2: B1; decisions D5-D8; 194 §7-§9, AM3, AM4;
#   195 §9.2, PA3, PA4). It reads ONLY result records and the registered
#   manifests -- never MRI data, never the holdout -- and writes nothing.
#   * Runs (one per arm x seed): hash-chained ledger; exactly one run_start
#     (seed/arm/config) and one run_end; consecutive epoch commits 1..N;
#     final retained set = directory listing (+ sidecars), every file and
#     sidecar hash verified; every epoch's validation-row file
#     (val_e<NNN>.json, b0 G1) verified against its commit and loaded with
#     v05lt_train.load_val_rows; the committed validation summary is
#     recomputed from the rows (v05lt_metrics.summarise) and must be equal;
#     the per-slice NLL-decomposition field must be present and finite
#     (the b0 G3 bound itself is enforced terminally at training time).
#   * D7: selection from the final committed selection state, re-derived
#     from the epoch records (highest R0 PSNR, earliest tie; lowest NLL,
#     earliest tie); a milestone M is reached ONLY when its exact epoch
#     commit, its row file and its retained milestone weights file verify;
#     never inferred from a later epoch. Status from run_end.
#   * D5: trained R0 all-slice / NLL rows from the row file of each
#     checkpoint's epoch; midpoint R0/R1/R0+DC/R1+DC rows from the post-run
#     evaluation record (v05lt_eval.load_posteval) of that checkpoint's
#     model state (epoch 3, selected, final).
#   * D6: duplicated midpoint R0 values (row file vs post-run record; epoch-0
#     all-slice vs epoch-0 midpoint) must agree: identical slice identity,
#     model state and mask convention / masks hash; PSNR |d| <= 0.001 dB
#     (registered P2 tolerance); MSE relative <= 10**(0.001/10) - 1 (the
#     same tolerance expressed in MSE); SSIM |d| <= 1e-6; NMSE_u both None
#     or relative <= 1e-6; psnr_capped, zero_error, eligible exactly equal.
#     Any disagreement -> LOAD_DUPLICATE_MISMATCH (validity failure); the
#     largest deviations are recorded.
#   * D8: a missing required trained result is a validity failure; an
#     absent milestone is reported through run_info (PA3); a missing
#     epoch-0 / ZF value (field absent or null, or a readout record or
#     slice absent) is PRESERVED as missing and listed with its identity,
#     so the engine reports the affected comparisons unavailable (AM3-E4,
#     v0.2). Wrong types (LOAD_RECORD_INVALID) and non-finite numbers
#     (LOAD_NONFINITE) always fail, for references and trained records.
#   * R1 measurements (v0.2): every post-run row must carry exactly the
#     declared banks; the primary bank exactly the declared prefixes
#     (selected: 16, 32, 64, 128), every other bank exactly 128 --
#     declarations never stand in for measurements.
#   * D6 (v0.3): checked field by field; with a missing reference value
#     the affected check is listed as not performed (AM3-E4); trained
#     values must all be present.
#   * Panel cases (v0.2): declared cases and every panel's case identity
#     (number, file, slice) must equal the registered case manifest, for
#     epoch-0 and post-run records.
#   * Epoch-0 evaluation records (per seed) and the ZF record: SHA256SUMS /
#     sidecar verified, schema, recorded sha256 (ZF: gate-2a value), panel
#     files = declared cases = registered case manifest (v0.2); a summary
#     is recomputed and compared when the record is complete (with missing
#     reference values it is recorded as not verifiable); epoch-0 context
#     must bind seed and epoch-0
#     model state (contract for the gated epoch-0 stage, I1.9).
#   * Output: the engine inputs (sources, slice_rows, run_info, validation
#     volumes) plus every file read with its sha256 (source identities) and
#     the D6 check record.
# CONVENTION: logger.error + V05LTError(code); no substituted values.
# Fixture-only: `fixture=True` accepts fixture run configurations and the
#   fixture end status "fixture_budget_end"; production passes False.
# Changelog (v0.6, 2026-10-24; b3 revision-1 review fixes):
#   * Holdout masks bound to an independently fixed identity: every row's
#     mask seed = canonical_mask_seed(EVAL_BASE_SEED, file, slice) from the
#     registered manifest; seed identity = the one frozen in the holdout
#     configuration; masks sha256 must be a valid sha256 (LOAD_MASK_INVALID)
#     and equal the
#     recomputation with the production make_cartesian_mask(width, seed);
#     widths identical across records. Agreement alone is not accepted.
# Changelog (v0.5, 2026-10-24; I1.8 b3, SEQREF-MRI-V05LTI18B3 v0.1):
#   * load_run also returns the committed retained-file records (name,
#     sha256) and the checkpoint directory (selection manifest, 195 §9.3).
#   * Holdout: schema seqref-v05lt-holdout-eval/1, verified manifest and
#     record loader (manifest file + canonical content, exact slice
#     identities and order, mask convention and one masks sha256, primary
#     bank with 128 decodes, checkpoint bindings to the verified commit and
#     the approved selection manifest, roles selected / epoch 3 only,
#     AM4-B NMSE_u). Synthetic records only until the 195 §9.3 approval.
# Changelog (v0.4, 2026-10-24; b1 revision-3 review fix):
#   * load_epoch0_eval: the midpoint-readout "missing" list no longer
#     overwrites the combined list in the returned record; absent slice rows
#     (midpoint and all-slice) now reach "missing" and reference_missing.
# Changelog (v0.3, 2026-10-24; b1 revision-2 review fixes):
#   * R1 bank / prefix structures type-checked before use (a malformed
#     bank or prefix entry -> LOAD_RECORD_INVALID; genuine absence kept
#     under AM3-E4).
#   * DC measures validated and normalised per readout ("dc" lists, with
#     missing measures listed) for the descriptive stage.
#   * D6 field by field: each available duplicated value is checked; a
#     check that cannot be performed because a reference value is missing
#     is listed (checks_not_performed), never treated as a mismatch.
# Changelog (v0.2, 2026-10-24; b1 review fixes):
#   * Post-run R1 rows must contain every declared bank and prefix (no
#     reliance on declarations).
#   * Missing epoch-0 / ZF values preserved for the engine's AM3-E4 rule;
#     strict type and finiteness checks on every record value.
#   * Declared panel cases bound to the registered case manifest (epoch-0
#     and post-run); layout key case_manifest / case_sha256.
# Changelog (NEW in v0.1, 2026-10-24; I1.8(b) b1, SEQREF-MRI-V05LTI18B v0.2):
#   * Introduced: verified run, post-run, epoch-0, ZF and manifest loading;
#     D5-D8; engine source mapping; source identities; production layout.
# =============================================================================
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from seqref_mri.scripts import v05lt_common as C

logger = logging.getLogger("seqref_mri.v05lt_results")

__version__ = "0.6"
__abbr__ = "SEQREF-MRI-V05LTRES"

SEEDS = (2026, 2027, 2028)
ARMS = ("lam0", "lam2")
POSTEVAL_ROLES = ("epoch3", "selected", "final")     # 194 §8.1 checkpoints
COMPLETE_STATUSES = ("stopped_plateau",
                     "budget exhausted; plateau not established")
FIXTURE_STATUS = "fixture_budget_end"
PSNR_TOL_DB = 0.001                                  # registered P2 (195 §8.2)
MSE_TOL_REL = 10 ** (PSNR_TOL_DB / 10) - 1           # same tolerance in MSE
SSIM_TOL_ABS = 1e-6
NMSE_TOL_REL = 1e-6
D6_RULE = {
    "identity": "same file, slice_index and volume_id; same model state "
                "(commit = row file = post-run record); same mask "
                "convention, one masks sha256 across all post-run records",
    "psnr_abs_db": PSNR_TOL_DB, "mse_rel": MSE_TOL_REL,
    "ssim_abs": SSIM_TOL_ABS, "nmse_u_rel": NMSE_TOL_REL,
    "exact": ["psnr_capped", "zero_error", "eligible"]}


def _T():
    from seqref_mri.scripts import v05lt_train as T
    return T


def _EV():
    from seqref_mri.scripts import v05lt_eval as EV
    return EV


def _M():
    from seqref_mri.scripts import v05lt_metrics as M
    return M


def _R():
    from seqref_mri.scripts import v05lt_rules as R
    return R


class Identities:
    """Every file read, with its sha256 (source identities)."""

    def __init__(self):
        self.files = {}

    def add(self, path: Path, sha: str) -> str:
        p = str(Path(path))
        C.require(self.files.get(p, sha) == sha, "LOAD_IDENTITY_CONFLICT", p)
        self.files[p] = sha
        return sha

    def hash(self, path: Path) -> str:
        return self.add(path, C.sha256_file(Path(path)))


def _json(path: Path, code: str):
    try:
        return json.loads(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        C.fail(code, f"{path}: {type(exc).__name__}: {exc}")


def _sums(d: Path, code_missing: str, code_invalid: str) -> dict:
    sums = d / "SHA256SUMS"
    C.require(sums.is_file(), code_missing, f"{d}: SHA256SUMS")
    listed = {}
    try:
        lines = sums.read_text().splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        C.fail(code_invalid, f"{sums}: {exc}")
    for line in lines:
        h, sep, n = line.partition("  ")
        C.require(sep and len(h) == 64 and n and "/" not in n and
                  n not in listed, code_invalid, f"{sums}: {line!r}")
        listed[n] = h
    present = {p.name for p in d.iterdir()} - {"SHA256SUMS"}
    C.require(present == set(listed), code_missing,
              f"{d}: listed {sorted(listed)[:4]} present {sorted(present)[:4]}")
    for n, h in listed.items():
        C.require((d / n).is_file() and C.sha256_file(d / n) == h,
                  code_missing, f"{d / n}")
    return listed


# ---------------------------------------------------------------------------
# Registered manifests
# ---------------------------------------------------------------------------
def load_manifests(layout: dict, ids: Identities) -> dict:
    """Split manifest (validation volume order) and eligibility manifest
    rows; both against their recorded sha256."""
    E = __import__("seqref_mri.scripts.v05lt_eligibility",
                   fromlist=["SCHEMA"])
    sp = Path(layout["split_manifest"])
    sdoc = C.load_split_manifest(sp, layout["split_sha256"],
                                 layout.get("split_expect",
                                            C.REGISTERED_SPLIT))
    ids.add(sp, layout["split_sha256"])
    ep = Path(layout["eligibility"])
    got = ids.hash(ep)
    C.require(got == layout["eligibility_sha256"], "LOAD_ELIGIBILITY_HASH",
              f"{ep}: {got} != recorded {layout['eligibility_sha256']}")
    side = Path(str(ep) + ".sha256")
    C.require(side.is_file() and side.read_text().split()[:1] == [got],
              "LOAD_ELIGIBILITY_HASH", f"{side}: sidecar")
    edoc = _json(ep, "LOAD_ELIGIBILITY_INVALID")
    C.require(isinstance(edoc, dict) and edoc.get("schema") == E.SCHEMA and
              isinstance(edoc.get("slices"), list),
              "LOAD_ELIGIBILITY_INVALID", f"{ep}: schema")
    vols = [r["volume_id"] for r in sdoc["validation"]]
    rows = []
    for r in edoc["slices"]:
        C.require(isinstance(r, dict) and {"volume_id", "file", "slice_index",
                                           "eligible", "midpoint"} <= set(r),
                  "LOAD_ELIGIBILITY_INVALID", f"{ep}: row {r!r}"[:200])
        rows.append({k: r[k] for k in ("volume_id", "file", "slice_index",
                                       "eligible", "midpoint")})
    C.require({r["volume_id"] for r in rows} == set(vols),
              "LOAD_ELIGIBILITY_INVALID", "eligibility volumes != split "
              "validation volumes")
    cp = Path(layout["case_manifest"])
    C.require(ids.hash(cp) == layout["case_sha256"], "LOAD_CASE_MANIFEST",
              f"{cp}: sha256 != recorded {layout['case_sha256']}")
    cdoc = _json(cp, "LOAD_CASE_MANIFEST")
    C.require(isinstance(cdoc, dict) and isinstance(cdoc.get("cases"), list)
              and cdoc["cases"], "LOAD_CASE_MANIFEST", f"{cp}: structure")
    index = {(r["file"], r["slice_index"]): r for r in rows}
    cases = []
    for c in cdoc["cases"]:
        ok = (isinstance(c, dict) and isinstance(c.get("case"), int) and
              (c.get("file"), c.get("slice_index")) in index and
              index[(c["file"], c["slice_index"])]["midpoint"])
        C.require(ok, "LOAD_CASE_MANIFEST", f"{cp}: case {c!r}"[:200])
        cases.append({"case": c["case"], "file": c["file"],
                      "slice_index": c["slice_index"]})
    C.require(len({c["case"] for c in cases}) == len(cases),
              "LOAD_CASE_MANIFEST", f"{cp}: duplicate case numbers")
    return {"validation_volumes": vols, "slice_rows": rows, "cases": cases,
            "case_sha256": layout["case_sha256"],
            "index": index,
            "mid_ids": [[r["file"], r["slice_index"]] for r in rows
                        if r["midpoint"]],
            "eligibility_sha256": got, "split_sha256": layout["split_sha256"]}


# ---------------------------------------------------------------------------
# Runs (D7, D5 trained R0 / NLL rows)
# ---------------------------------------------------------------------------
FLOAT_FIELDS = ("psnr", "ssim", "mse")
FLAG_FIELDS = ("psnr_capped", "zero_error")
VALUE_FIELDS = FLOAT_FIELDS + ("nmse_u",) + FLAG_FIELDS


def _values(rec, where: str, reference: bool, missing: list) -> dict:
    """Strict record values. Absent / null value: reference (epoch 0, ZF)
    -> None, listed in `missing` (AM3-E4, decided by the engine); trained
    checkpoint -> LOAD_VALUE_MISSING (D8). nmse_u null is a registered
    value (ineligible slice; the engine checks it against eligibility).
    Wrong type -> LOAD_RECORD_INVALID; non-finite -> LOAD_NONFINITE."""
    C.require(isinstance(rec, dict), "LOAD_RECORD_INVALID",
              f"{where}: record is {type(rec).__name__}")
    out = {}
    for k in VALUE_FIELDS:
        absent = k not in rec or (rec[k] is None and k != "nmse_u")
        if absent:
            C.require(reference, "LOAD_VALUE_MISSING",
                      f"{where}: {k} missing on a trained checkpoint (D8)")
            missing.append(f"{where}: {k}")
            out[k] = None
            continue
        v = rec[k]
        if k in FLAG_FIELDS:
            C.require(isinstance(v, bool), "LOAD_RECORD_INVALID",
                      f"{where}: {k}={v!r}")
        elif v is not None:
            C.require(isinstance(v, (int, float)) and not isinstance(v, bool),
                      "LOAD_RECORD_INVALID", f"{where}: {k}={v!r}")
            C.require(math.isfinite(v), "LOAD_NONFINITE", f"{where}: {k}={v!r}")
            v = float(v)
        out[k] = v
    return out


def _engine_row_from_val(r: dict, where: str = "row file",
                         reference: bool = False, missing=None,
                         with_nll: bool = True) -> dict:
    C.require(isinstance(r, dict) and isinstance(r.get("file"), str) and
              isinstance(r.get("slice_index"), int) and
              isinstance(r.get("volume_id"), str), "LOAD_RECORD_INVALID",
              f"{where}: row identity")
    out = {"volume_id": r["volume_id"], "file": r["file"],
           "slice_index": r["slice_index"],
           **_values(r, f"{where} {r['file']}|{r['slice_index']}",
                     reference, missing if missing is not None else []),
           "eligible": r.get("eligible"), "midpoint": r.get("midpoint")}
    if with_nll:
        out["nll_per_dim"] = _nll(r, where, reference, missing)
    return out


def _nll(r, where, reference, missing):
    v = r.get("nll_per_dim")
    if v is None:
        C.require(reference, "LOAD_VALUE_MISSING",
                  f"{where} {r['file']}|{r['slice_index']}: nll_per_dim (D8)")
        if missing is not None:
            missing.append(f"{where} {r['file']}|{r['slice_index']}: "
                           "nll_per_dim")
        return None
    C.require(isinstance(v, (int, float)) and not isinstance(v, bool),
              "LOAD_RECORD_INVALID", f"{where}: nll_per_dim={v!r}")
    C.require(math.isfinite(v), "LOAD_NONFINITE", f"{where}: nll_per_dim")
    return float(v)


def load_run(run_dir: Path, arm: str, seed: int, man: dict,
             ids: Identities, fixture: bool = False) -> dict:
    """One finished run -> verified epochs, checkpoints and run_info."""
    T, M, R = _T(), _M(), _R()
    run_dir = Path(run_dir)
    tag = f"{arm}/{seed}"
    lp = run_dir / "ledger.jsonl"
    C.require(lp.is_file(), "LOAD_RUN_MISSING", f"{tag}: {lp}")
    recs = C.Ledger(lp).records()             # chain + self-hash verified
    ids.hash(lp)
    C.require(bool(recs) and recs[0]["kind"] == "run_start",
              "LOAD_RUN_INVALID", f"{tag}: first record is not run_start")
    cfg = recs[0]["body"].get("config", {})
    C.require(cfg.get("seed") == seed and cfg.get("arm") == arm,
              "LOAD_RUN_IDENTITY", f"{tag}: run_start config "
              f"{cfg.get('arm')}/{cfg.get('seed')}")
    C.require(bool(cfg.get("fixture")) == fixture, "LOAD_RUN_IDENTITY",
              f"{tag}: fixture flag {cfg.get('fixture')!r}")
    ends = [r for r in recs if r["kind"] == "run_end"]
    C.require(len(ends) == 1, "LOAD_RUN_NOT_ENDED",
              f"{tag}: {len(ends)} run_end records")
    end = ends[0]["body"]
    allowed = COMPLETE_STATUSES + ((FIXTURE_STATUS,) if fixture else ())
    C.require(end.get("status") in allowed, "LOAD_RUN_NOT_COMPLETE",
              f"{tag}: run ended with status {end.get('status')!r} (code "
              f"{end.get('code')!r}); analysis requires a completed run -- "
              "owner decision outside the analysis")
    commits = [r["body"] for r in recs if r["kind"] == "commit"]
    epochs = [b["epoch"] for b in commits]
    C.require(epochs == list(range(1, len(epochs) + 1)) and epochs,
              "LOAD_RUN_COMMITS_INVALID", f"{tag}: commit epochs {epochs}")
    n = epochs[-1]
    C.require(end.get("epoch") == n and commits[-1].get("status") ==
              end["status"], "LOAD_RUN_COMMITS_INVALID",
              f"{tag}: run_end epoch {end.get('epoch')} / status vs last "
              f"commit {n} / {commits[-1].get('status')}")
    # final retained set == directory (+ sidecars); every hash verified
    ck = run_dir / "ckpt"
    ret = commits[-1]["retained"]
    names = {v["name"]: v["sha256"] for v in ret.values()}
    listing = {p.name for p in ck.iterdir()} if ck.is_dir() else set()
    C.require(listing == set(names) | {f"{x}.sha256" for x in names},
              "LOAD_RUN_DIR_UNEXPECTED", f"{tag}: extra "
              f"{sorted(listing - set(names) - {f'{x}.sha256' for x in names})[:4]}"
              f" missing {sorted(set(names) - listing)[:4]}")
    for name, sha in sorted(names.items()):
        side = (ck / f"{name}.sha256").read_text().split()[:1]
        C.require(side == [sha], "LOAD_HASH_MISMATCH", f"{tag}: sidecar {name}")
        got = ids.hash(ck / name)
        C.require(got == sha, "LOAD_HASH_MISMATCH",
                  f"{tag}: {name} {got} != committed {sha}")
    # every epoch: row file bound to its commit and retained to the end
    by_epoch, rows_by_epoch = {}, {}
    exp_ids = {(r["file"], r["slice_index"]) for r in man["slice_rows"]}
    for b in commits:
        e = b["epoch"]
        vr = b.get("validation_rows")
        C.require(isinstance(vr, dict), "LOAD_ROWS_MISSING",
                  f"{tag}: epoch {e} commit has no validation_rows")
        role = f"{T.VAL_ROLE_PREFIX}{e:03d}"
        C.require(ret.get(role) == {"name": vr["name"], "sha256": vr["sha256"]},
                  "LOAD_ROWS_MISSING", f"{tag}: epoch {e} row file not "
                  "retained as committed")
        doc = T.load_val_rows(ck / vr["name"], vr["sha256"], e)
        C.require(doc["seed"] == seed and doc["arm"] == arm and
                  doc["model_state_sha256"] == b["model_state_sha256"] and
                  doc["n_rows"] == vr["n_rows"] and
                  doc["slice_ids_sha256"] == vr["slice_ids_sha256"],
                  "LOAD_ROWS_MISMATCH", f"{tag}: epoch {e} row file vs commit")
        got_ids = {(r["file"], r["slice_index"]) for r in doc["rows"]}
        C.require(got_ids == exp_ids and len(doc["rows"]) == len(exp_ids),
                  "LOAD_ROWS_SLICES", f"{tag}: epoch {e} row file slices != "
                  "eligibility manifest slices")
        for r in doc["rows"]:
            el = man["index"][(r["file"], r["slice_index"])]
            C.require(r["volume_id"] == el["volume_id"] and r["eligible"] ==
                      el["eligible"] and r["midpoint"] == el["midpoint"],
                      "LOAD_ROWS_SLICES", f"{tag}: epoch {e} "
                      f"{r['file']}|{r['slice_index']} vs eligibility")
            C.require(isinstance(r.get("nll_decomp_abs_dev"), float) and
                      math.isfinite(r["nll_decomp_abs_dev"]) and
                      r["nll_decomp_abs_dev"] >= 0.0, "LOAD_ROWS_INVALID",
                      f"{tag}: epoch {e} {r['file']}|{r['slice_index']} "
                      "decomposition deviation field")
        summ = json.loads(json.dumps(M.summarise(doc["rows"])))
        C.require(summ == b["validation"], "LOAD_SUMMARY_MISMATCH",
                  f"{tag}: epoch {e} summary recomputed from the row file != "
                  "committed validation summary")
        by_epoch[e], rows_by_epoch[e] = b, doc
    # D7: selection re-derived from the epoch records
    sel = commits[-1]["selection"]

    def best(key, better):
        bestv = None
        for e in epochs:
            v = by_epoch[e]["validation"]["all_slices"][key]
            if bestv is None or better(v, bestv[1]):
                bestv = (e, v)
        return {"epoch": bestv[0], "value": bestv[1]}
    want = {"best_psnr": best("psnr", lambda a, b: a > b),
            "best_nll": best("nll_per_dim", lambda a, b: a < b)}
    C.require(sel == want, "LOAD_SELECTION_MISMATCH",
              f"{tag}: committed selection {sel} != re-derived {want}")
    for role in ("best_psnr", "best_nll"):
        e = sel[role]["epoch"]
        C.require(ret.get(role, {}).get("name") == f"weights_e{e:03d}.pt",
                  "LOAD_SELECTION_MISMATCH", f"{tag}: {role} weights file "
                  f"{ret.get(role)} != epoch {e}")
    # D7: milestones only from their exact verified commit and artefacts
    reached = []
    for m in R.MILESTONES:
        if m > n:
            continue
        mrole = f"milestone_{m:03d}"
        ok = (mrole in by_epoch[m].get("roles_this_epoch", []) and
              ret.get(mrole, {}).get("name") == f"weights_e{m:03d}.pt")
        C.require(ok, "LOAD_MILESTONE_INCOMPLETE", f"{tag}: milestone {m} "
                  "committed epoch lacks its role or retained weights file")
        reached.append(m)
    C.require(n >= 3, "LOAD_REQUIRED_MISSING", f"{tag}: epoch 3 (required "
              "within-arm reference) not committed")
    ckpts = {"epoch3": 3, "selected": sel["best_psnr"]["epoch"],
             "best_nll": sel["best_nll"]["epoch"], "final": n}
    for m in reached:
        ckpts[f"epoch{m}"] = m
    out = {"arm": arm, "seed": seed, "dir": str(run_dir), "last_epoch": n,
           "status": end["status"], "code": end.get("code"),
           "milestones": reached,
           "epoch0_model_state": cfg.get("epoch0_model_state"),
           "checkpoints": {k: {"epoch": e, "model_state_sha256":
                               by_epoch[e]["model_state_sha256"]}
                           for k, e in ckpts.items()},
           "epochs": {e: {"step": by_epoch[e]["step"],
                          "validation": by_epoch[e]["validation"],
                          "plateau": by_epoch[e]["plateau"],
                          "status": by_epoch[e]["status"],
                          "model_state_sha256":
                              by_epoch[e]["model_state_sha256"]}
                      for e in epochs},
           "rows": rows_by_epoch,
           "retained": {k: {"name": v["name"], "sha256": v["sha256"]}
                        for k, v in sorted(ret.items())},
           "ckpt_dir": str(ck)}
    return out


# ---------------------------------------------------------------------------
# Evaluator records (D5, D6)
# ---------------------------------------------------------------------------
def _ident(r: dict, man: dict, where: str) -> tuple:
    ident = r.get("identity") if isinstance(r, dict) else None
    C.require(isinstance(ident, dict) and isinstance(ident.get("file"), str)
              and isinstance(ident.get("slice_index"), int),
              "LOAD_RECORD_INVALID", f"{where}: row identity")
    key = (ident["file"], ident["slice_index"])
    C.require(key in man["index"], "LOAD_SLICE_UNKNOWN", f"{where}: {key}")
    return key


def _engine_row_from_rec(rec, key: tuple, man: dict, where: str,
                         reference: bool, missing: list,
                         dc: bool = False) -> dict:
    out = {"volume_id": man["index"][key]["volume_id"], "file": key[0],
           "slice_index": key[1],
           **_values(rec, f"{where} {key[0]}|{key[1]}", reference, missing)}
    if dc:
        C.require(isinstance(rec.get("dc_identity"), bool),
                  "LOAD_RECORD_INVALID", f"{where} {key}: dc_identity")
        out["dc_identical"] = rec["dc_identity"]
    return out


DC_MEASURES = ("a", "b", "c_rms", "c_max")


def _dc_meas(rec: dict, where: str, reference: bool, missing: list):
    """Validated DC measures of one R+DC record, or None when absent
    (reference only; listed)."""
    m = rec.get("dc")
    if m is None:
        C.require(reference, "LOAD_VALUE_MISSING",
                  f"{where}: DC measures absent (trained checkpoint, D8)")
        missing.append(f"{where}: DC measures absent")
        return None
    C.require(isinstance(m, dict) and set(m) >= set(DC_MEASURES),
              "LOAD_RECORD_INVALID", f"{where}: DC measures {m!r}"[:200])
    out = {}
    for k in DC_MEASURES:
        v = m[k]
        C.require(isinstance(v, (int, float)) and not isinstance(v, bool),
                  "LOAD_RECORD_INVALID", f"{where}: DC {k}={v!r}")
        C.require(math.isfinite(v), "LOAD_NONFINITE", f"{where}: DC {k}")
        out[k] = float(v)
    return out


def _mid_sources(mrows: list, man: dict, where: str, *, banks: dict,
                 reference: bool) -> dict:
    """Evaluator midpoint rows -> {readout: engine rows}, per-bank and
    per-prefix R1 rows and validated DC measures. banks = {bank: [prefix
    keys]} declared by the record. Structure is type-checked first
    (LOAD_RECORD_INVALID). Trained records: every row carries exactly the
    declared banks and prefixes, nothing absent (D8). Reference records
    (epoch 0): an absent readout / bank / prefix entry is listed
    (AM3-E4); entries outside the declaration are refused."""
    C.require(isinstance(mrows, list), "LOAD_RECORD_INVALID", where)
    out = {"r0": [], "r1": [], "r0_dc": [], "r1_dc": []}
    dc = {"r0_dc": [], "r1_dc": []}
    per_bank = {b: {k: [] for k in ks} for b, ks in banks.items()}
    missing, seen = [], set()
    for r in mrows:
        key = _ident(r, man, where)
        C.require(key not in seen, "LOAD_DUPLICATE_SLICE", f"{where}: {key}")
        seen.add(key)
        tag = f"{where} {key[0]}|{key[1]}"

        def absent(what):
            C.require(reference, "LOAD_VALUE_MISSING",
                      f"{tag}: {what} absent (trained checkpoint, D8)")
            missing.append(f"{tag}: {what} absent")
        for ro in ("r0", "r0_dc", "r1_dc"):
            if r.get(ro) is None:
                absent(ro)
                continue
            row = _engine_row_from_rec(r[ro], key, man, f"{where} {ro}",
                                       reference, missing,
                                       dc=ro.endswith("_dc"))
            out[ro].append(row)
            if ro.endswith("_dc"):
                dc[ro].append({"file": key[0], "slice_index": key[1],
                               "identical": row["dc_identical"],
                               "measures": _dc_meas(r[ro], f"{tag} {ro}",
                                                    reference, missing)})
        r1 = r.get("r1")
        if r1 is None:
            absent("r1")
            continue
        C.require(isinstance(r1, dict), "LOAD_RECORD_INVALID", f"{tag}: r1 "
                  f"is {type(r1).__name__}")
        for b_, v in r1.items():
            C.require(b_ in banks, "LOAD_R1_MISSING" if not reference else
                      "LOAD_RECORD_INVALID", f"{tag}: undeclared bank {b_!r}")
            C.require(isinstance(v, dict), "LOAD_RECORD_INVALID",
                      f"{tag}: bank {b_} is {type(v).__name__}")
            for k, rec in v.items():
                C.require(k in banks[b_], "LOAD_POSTEVAL_PREFIXES" if
                          not reference else "LOAD_RECORD_INVALID",
                          f"{tag}: bank {b_} undeclared prefix {k!r}")
                C.require(rec is None or isinstance(rec, dict),
                          "LOAD_RECORD_INVALID", f"{tag}: bank {b_} prefix "
                          f"{k} is {type(rec).__name__}")
        if not reference:
            C.require(set(r1) == set(banks), "LOAD_R1_MISSING",
                      f"{tag}: banks {sorted(r1)} != declared "
                      f"{sorted(banks)}")
            for b_, ks in banks.items():
                C.require(set(k for k, x in r1[b_].items() if x is not None)
                          == set(ks), "LOAD_POSTEVAL_PREFIXES" if
                          b_ == "primary" else "LOAD_R1_MISSING",
                          f"{tag}: bank {b_} carries {sorted(r1[b_])} != "
                          f"declared {sorted(ks)}")
        for b_, ks in banks.items():
            for k in ks:
                rec = r1.get(b_, {}).get(k)
                if rec is None:
                    absent(f"r1 {b_} {k}")
                    continue
                row = _engine_row_from_rec(rec, key, man, f"{where} r1 {b_} "
                                           f"{k}", reference, missing)
                per_bank[b_][k].append(row)
                if b_ == "primary" and k == "128":
                    out["r1"].append(row)
    return {"readouts": out, "per_bank": per_bank, "dc": dc,
            "missing": missing, "slices": seen}


D6_FIELDS = ("psnr", "mse", "ssim", "nmse_u", "psnr_capped", "zero_error")


def _d6_block(rows_all: list, mid_r0: list, where: str, man: dict,
              reference: bool = False) -> dict:
    """D6 field by field on the duplicated midpoint R0 values. Every
    available pair is checked; a check impossible because a reference value
    (or the all-slice row) is missing is listed, never a mismatch. NMSE_u
    on an ineligible slice must be null on both sides."""
    by = {(r["file"], r["slice_index"]): r for r in rows_all}
    dev, not_done, n = {}, [], 0
    lim = {"psnr": PSNR_TOL_DB, "mse": MSE_TOL_REL, "ssim": SSIM_TOL_ABS,
           "nmse_u": NMSE_TOL_REL}
    dname = {"psnr": "psnr_abs_db", "mse": "mse_rel", "ssim": "ssim_abs",
             "nmse_u": "nmse_u_rel"}
    for r in mid_r0:
        k = (r["file"], r["slice_index"])
        tag = f"{where} {k[0]}|{k[1]}"
        if k not in by:
            C.require(reference, "LOAD_DUPLICATE_MISMATCH",
                      f"{tag}: absent from the all-slice rows")
            not_done.append(f"{k[0]}|{k[1]}: all fields (all-slice row "
                            "absent)")
            continue
        a = by[k]
        n += 1
        C.require((a["volume_id"], a["file"], a["slice_index"]) ==
                  (r["volume_id"], r["file"], r["slice_index"]),
                  "LOAD_DUPLICATE_MISMATCH", f"{tag}: identity")
        elig = man["index"][k]["eligible"]
        for f in D6_FIELDS:
            x, y = r[f], a[f]
            if f == "nmse_u" and not elig:
                C.require(x is None and y is None, "LOAD_DUPLICATE_MISMATCH",
                          f"{tag}: NMSE_u on an ineligible slice")
                continue
            if x is None or y is None:
                C.require(reference, "LOAD_DUPLICATE_MISMATCH",
                          f"{tag}: {f} missing on a trained record")
                not_done.append(f"{k[0]}|{k[1]}: {f} (reference value "
                                "missing)")
                continue
            if f in ("psnr_capped", "zero_error"):
                C.require(x == y, "LOAD_DUPLICATE_MISMATCH", f"{tag}: {f}")
                continue
            d = abs(x - y)
            if f in ("mse", "nmse_u"):
                d = d / max(abs(y), 1e-300)
            dev[dname[f]] = max(dev.get(dname[f], 0.0), d)
            C.require(d <= lim[f], "LOAD_DUPLICATE_MISMATCH",
                      f"{tag}: {f} deviation {d:.3g} > {lim[f]:.3g} (D6 "
                      f"{D6_RULE})")
    return {"slices": n, "max_deviation": dev,
            "checks_not_performed": not_done}


def _check_mid_population(keys: set, n_rows: int, man: dict, where: str,
                          reference: bool = False) -> list:
    """Trained: rows = exactly the eligibility midpoints. Reference: a
    subset (absent slices listed for AM3-E4); never an extra slice."""
    want = {(f, s) for f, s in man["mid_ids"]}
    C.require(keys <= want and len(keys) == n_rows,
              "LOAD_POPULATION_MISMATCH", f"{where}: rows outside the "
              "eligibility midpoints or duplicated")
    C.require(reference or keys == want, "LOAD_POPULATION_MISMATCH",
              f"{where}: midpoint slices differ from the eligibility "
              "manifest midpoints")
    return [f"{where} {f}|{s}: slice absent" for f, s in man["mid_ids"]
            if (f, s) not in keys]


def _check_cases(doc: dict, man: dict, where: str) -> None:
    """Declared cases and each panel's case identity = the registered case
    manifest (number, file, slice)."""
    want = {c["case"]: (c["file"], c["slice_index"]) for c in man["cases"]}
    C.require(isinstance(doc.get("cases"), list) and
              sorted(doc["cases"]) == sorted(want) and
              len(doc["cases"]) == len(want), "LOAD_CASES_MISMATCH",
              f"{where}: declared cases {doc.get('cases')} != registered "
              f"{sorted(want)}")
    pans = doc.get("panels")
    C.require(isinstance(pans, list) and len(pans) == len(want),
              "LOAD_CASES_MISMATCH", f"{where}: panel list")
    got = {}
    for m in pans:
        c = m.get("case") if isinstance(m, dict) else None
        C.require(isinstance(c, dict) and c.get("case") in want and
                  c.get("case") not in got, "LOAD_CASES_MISMATCH",
                  f"{where}: panel case {c!r}"[:200])
        got[c["case"]] = (c.get("file"), c.get("slice_index"))
    C.require(got == want, "LOAD_CASES_MISMATCH", f"{where}: panel case "
              f"identities {got} != registered {want}")


def load_posteval_for(root: Path, run: dict, role: str, man: dict,
                      ids: Identities, shared: dict) -> dict:
    """Post-run record of one (run, role) checkpoint (D5); provenance bound
    to the run's commit; bank and prefix requirements per role."""
    EV = _EV()
    ck = run["checkpoints"][role]
    ms = ck["model_state_sha256"]
    d = Path(root) / ms
    doc = EV.load_posteval(d)
    for n in sorted(p.name for p in d.iterdir()):
        ids.hash(d / n)
    tag = f"{run['arm']}/{run['seed']} {role}"
    cp = doc["checkpoint"]
    C.require(cp.get("model_state_sha256") == ms and cp.get("epoch") ==
              ck["epoch"] and cp.get("seed") == run["seed"] and
              cp.get("arm") == run["arm"], "LOAD_POSTEVAL_MISMATCH",
              f"{tag}: record checkpoint {cp} vs commit epoch {ck['epoch']}")
    uses = {(u.get("seed"), u.get("arm"), u.get("role"))
            for u in doc.get("uses", [])}
    C.require((run["seed"], run["arm"], role) in uses,
              "LOAD_POSTEVAL_MISMATCH", f"{tag}: role not among the record's "
              f"uses {sorted(uses)}")
    banks = set(doc["banks"])
    need = {"primary", "B", "C"} if role == "selected" else {"primary"}
    C.require(need <= banks and all(doc["banks"][b] ==
                                    EV.REGISTERED_BANK_ARRAY[b]
                                    for b in banks),
              "LOAD_POSTEVAL_BANKS", f"{tag}: banks {sorted(banks)}")
    C.require(doc.get("prefix_bank", "primary") == "primary",
              "LOAD_POSTEVAL_PREFIXES", f"{tag}: prefix bank")
    want_pfx = sorted(EV.PREFIXES) if role == "selected" else None
    C.require(want_pfx is None or sorted(int(k) for k in doc["prefixes"])
              == want_pfx, "LOAD_POSTEVAL_PREFIXES",
              f"{tag}: prefixes {doc['prefixes']}")
    C.require(doc["masks"]["convention"] == EV.MASK_CONVENTION,
              "LOAD_MASK_MISMATCH", f"{tag}: mask convention")
    C.require(shared.setdefault("masks_sha256", doc["masks"]["sha256"]) ==
              doc["masks"]["sha256"], "LOAD_MASK_MISMATCH",
              f"{tag}: masks sha256 differs between post-run records")
    C.require(shared.setdefault("dc_tolerances", doc["dc_tolerances"]) ==
              doc["dc_tolerances"], "LOAD_DC_TOLERANCE_MISMATCH",
              f"{tag}: DC tolerances differ between records")
    pop = doc["population"]
    C.require(pop["eligibility_manifest_sha256"] == man["eligibility_sha256"]
              and pop["n"] == len(man["mid_ids"]) and
              pop["identities_sha256"] == C.sha256_bytes(
                  C.canonical_json(man["mid_ids"])),
              "LOAD_POPULATION_MISMATCH", f"{tag}: population {pop}")
    _check_cases(doc, man, tag)
    rows = doc["midpoint"]["rows"]
    declared = {b: ([str(int(k)) for k in doc["prefixes"]]
                    if b == doc.get("prefix_bank", "primary") else ["128"])
                for b in banks}
    srcs = _mid_sources(rows, man, tag, banks=declared, reference=False)
    _check_mid_population(srcs["slices"], len(rows), man, tag)
    return {"model_state_sha256": ms, "epoch": ck["epoch"], "dir": str(d),
            "record_sha256": C.sha256_file(d / EV.POSTEVAL_FILE),
            "banks": doc["banks"], "prefixes": doc["prefixes"],
            "code": doc["code"], "backend": doc["backend"],
            "dc_tolerances": doc["dc_tolerances"], "rows": rows, **srcs}


def load_epoch0_eval(d: Path, seed: int, e0_state: str, man: dict,
                     ids: Identities, shared: dict) -> dict:
    """Epoch-0 evaluation record of one seed (AM3-C)."""
    EV, M = _EV(), _M()
    d = Path(d)
    listed = _sums(d, "LOAD_EPOCH0_INCOMPLETE", "LOAD_EPOCH0_INVALID")
    for n in sorted(listed):
        ids.add(d / n, listed[n])
    C.require(EV.E0_EVAL_FILE in listed, "LOAD_EPOCH0_INCOMPLETE",
              f"{d}: {EV.E0_EVAL_FILE}")
    doc = _json(d / EV.E0_EVAL_FILE, "LOAD_EPOCH0_INVALID")
    ok = (isinstance(doc, dict) and doc.get("schema") == EV.E0_EVAL_SCHEMA
          and isinstance(doc.get("context"), dict) and
          isinstance(doc.get("all_slice"), dict) and
          isinstance(doc.get("midpoint"), dict) and
          isinstance(doc.get("cases"), list) and doc["cases"])
    C.require(ok, "LOAD_EPOCH0_INVALID", f"{d}: record structure")
    ctx = doc["context"]
    C.require(ctx.get("seed") == seed and ctx.get(
        "epoch0_model_state_sha256") == e0_state, "LOAD_EPOCH0_IDENTITY",
        f"{d}: context seed/model state {ctx.get('seed')}/"
        f"{str(ctx.get('epoch0_model_state_sha256'))[:12]} vs seed {seed} "
        f"runs' epoch-0 state {str(e0_state)[:12]}")
    _check_cases(doc, man, f"epoch0 seed {seed}")
    want = set().union(*({f"panel_case{c}.png", f"panel_case{c}.npz"}
                         for c in doc["cases"]))
    C.require(set(listed) - {EV.E0_EVAL_FILE} == want,
              "LOAD_EPOCH0_INCOMPLETE", f"{d}: panel files vs declared cases "
              f"{doc['cases']}")
    C.require(shared.setdefault("dc_tolerances", doc["dc_tolerances"]) ==
              doc["dc_tolerances"], "LOAD_DC_TOLERANCE_MISMATCH",
              f"{d}: DC tolerances differ between records")
    where = f"epoch0 seed {seed}"
    missing = []
    rows_all = doc["all_slice"].get("rows")
    C.require(isinstance(rows_all, list), "LOAD_EPOCH0_INVALID",
              f"{d}: all-slice rows")
    eng_all, seen = [], set()
    for r in rows_all:
        row = _engine_row_from_val(r, f"{where} all-slice", True, missing)
        k = (row["file"], row["slice_index"])
        el = man["index"].get(k)
        C.require(el is not None and el["volume_id"] == row["volume_id"] and
                  k not in seen, "LOAD_EPOCH0_INVALID", f"{d}: {k[0]}|{k[1]} "
                  "vs eligibility (unknown, wrong volume or duplicate)")
        seen.add(k)
        eng_all.append(row)
    missing += [f"{where} all-slice {f}|{s_}: slice absent"
                for f, s_ in man["index"] if (f, s_) not in seen]
    srcs = _mid_sources(doc["midpoint"].get("rows"), man, f"{where} midpoint",
                        banks={"primary": ["128"]}, reference=True)
    missing += srcs["missing"]
    missing += _check_mid_population(srcs["slices"], len(
        doc["midpoint"]["rows"]), man, f"{where} midpoint", reference=True)
    if not missing:
        summ = json.loads(json.dumps(M.summarise(rows_all)))
        C.require(summ == doc["all_slice"]["summary"], "LOAD_SUMMARY_MISMATCH",
                  f"{d}: epoch-0 summary recomputed != stored")
    d6 = _d6_block(eng_all, srcs["readouts"]["r0"], where, man,
                   reference=True)
    # v0.4: srcs first, so its readout-only "missing" can never replace the
    # combined list (slice-absent and all-slice entries included).
    return {**srcs, "dir": str(d), "record_sha256": listed[EV.E0_EVAL_FILE],
            "summary": doc["all_slice"]["summary"],
            "summary_verified": not missing, "missing": missing,
            "rows_all": eng_all, "rows_all_raw": rows_all,
            "rows": doc["midpoint"]["rows"],
            "dc_tolerances": doc["dc_tolerances"], "d6": d6}


def load_zf(d: Path, recorded_sha: str, man: dict, ids: Identities) -> dict:
    """Gate-2a ZF record (AM3-D): sidecar pair, recorded sha256, rows."""
    EV = _EV()
    d = Path(d)
    sha = EV.verify_zf(d)
    C.require(sha == recorded_sha, "LOAD_ZF_HASH",
              f"{d}: {sha} != recorded gate-2a sha256 {recorded_sha}")
    ids.add(d / EV.ZF_FILE, sha)
    doc = _json(d / EV.ZF_FILE, "LOAD_ZF_INVALID")
    C.require(isinstance(doc, dict) and doc.get("schema") == EV.ZF_SCHEMA and
              isinstance(doc.get("rows"), list), "LOAD_ZF_INVALID",
              f"{d}: structure")
    rows, missing, seen = [], [], set()
    for r in doc["rows"]:
        row = _engine_row_from_val(r, "ZF", True, missing,
                                   with_nll=False)     # ZF has no NLL (AM3-D)
        k = (row["file"], row["slice_index"])
        el = man["index"].get(k)
        C.require(el is not None and el["volume_id"] == row["volume_id"] and
                  el["eligible"] == r.get("eligible") and el["midpoint"] ==
                  r.get("midpoint") and k not in seen, "LOAD_ZF_INVALID",
                  f"{d}: {k[0]}|{k[1]} vs eligibility (unknown, flags or "
                  "duplicate)")
        seen.add(k)
        rows.append(row)
    missing += [f"ZF {f}|{s_}: slice absent" for f, s_ in man["index"]
                if (f, s_) not in seen]
    if not missing:
        summ = json.loads(json.dumps(EV.zf_summary(doc["rows"])))
        C.require(summ == doc["summary"], "LOAD_SUMMARY_MISMATCH",
                  f"{d}: ZF summary recomputed != stored")
    return {"dir": str(d), "record_sha256": sha, "rows": rows,
            "summary": doc["summary"], "summary_verified": not missing,
            "missing": missing}


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------
def production_layout(paths: C.Paths, recorded: dict) -> dict:
    """Registered layout (195 §3-§6, PA2): recorded = {'eligibility_sha256',
    'zf_sha256', 'difficult_sha256'} as recorded in V05LTEXEC."""
    v = paths.v05lt
    return {"split_manifest": paths.inputs / "v05lt_split_manifest.json",
            "split_sha256": C.REGISTERED_INPUTS["v05lt_split_manifest.json"],
            "case_manifest": paths.inputs / "v05lt_case_manifest.json",
            "case_sha256": C.REGISTERED_INPUTS["v05lt_case_manifest.json"],
            "eligibility": paths.inputs / "v05lt_nmse_eligibility.json",
            "eligibility_sha256": recorded["eligibility_sha256"],
            "zf_dir": v / "zf", "zf_sha256": recorded["zf_sha256"],
            "difficult": v / "analysis" / "v05lt_difficult_cases.json",
            "difficult_sha256": recorded["difficult_sha256"],
            "epoch0": {s: v / "epoch0" / f"seed_{s}" / "eval" for s in SEEDS},
            "runs": {f"{a}|{s}": v / f"{s}_{a}" for s in SEEDS for a in ARMS},
            "posteval_root": v / "posteval", "seeds": list(SEEDS),
            "arms": list(ARMS), "fixture": False}


def load_campaign(layout: dict) -> dict:
    """All result records of the campaign -> engine inputs + identities +
    loaded records for the descriptive block. Read-only."""
    ids = Identities()
    fixture = bool(layout.get("fixture"))
    seeds, arms = list(layout["seeds"]), list(layout["arms"])
    if not fixture:
        C.require(seeds == list(SEEDS) and arms == list(ARMS),
                  "LOAD_LAYOUT_INVALID", "production layout must list the "
                  "registered seeds and arms")
    man = load_manifests(layout, ids)
    shared = {}
    runs, sources, run_info, d6 = {}, {}, {}, {}
    posteval = {}
    for s in seeds:
        for a in arms:
            key = f"{a}|{s}"
            C.require(key in layout["runs"], "LOAD_RUN_MISSING", key)
            run = load_run(layout["runs"][key], a, s, man, ids, fixture)
            runs[key] = run
            run_info[(a, s)] = {"milestones": run["milestones"],
                                "last_epoch": run["last_epoch"],
                                "status": run["status"]}
            for ck, info in run["checkpoints"].items():
                rows = [_engine_row_from_val(r, f"{key} {ck} row file")
                        for r in run["rows"][info["epoch"]]["rows"]]
                sources[f"{a}|{s}|{ck}|r0|all"] = rows
            for role in POSTEVAL_ROLES:
                pe = load_posteval_for(layout["posteval_root"], run, role,
                                       man, ids, shared)
                posteval[f"{key}|{role}"] = pe
                for ro, rows in pe["readouts"].items():
                    sources[f"{a}|{s}|{role}|{ro}|mid"] = rows
                rows_all = sources[f"{a}|{s}|{role}|r0|all"]
                d6[f"{key}|{role}"] = _d6_block(
                    [dict(r) for r in rows_all], pe["readouts"]["r0"],
                    f"{key} {role}", man)
    epoch0 = {}
    for s in seeds:
        states = {runs[f"{a}|{s}"]["epoch0_model_state"] for a in arms}
        C.require(len(states) == 1, "LOAD_EPOCH0_IDENTITY",
                  f"seed {s}: arms disagree on the epoch-0 model state")
        C.require(s in layout["epoch0"] or str(s) in layout["epoch0"],
                  "LOAD_EPOCH0_MISSING", f"seed {s}")
        d = layout["epoch0"].get(s, layout["epoch0"].get(str(s)))
        e0 = load_epoch0_eval(d, s, states.pop(), man, ids, shared)
        epoch0[s] = e0
        d6[f"epoch0|{s}"] = e0["d6"]
        sources[f"shared|{s}|epoch0|r0|all"] = e0["rows_all"]
        for ro, rows in e0["readouts"].items():
            sources[f"shared|{s}|epoch0|{ro}|mid"] = rows
    zf = load_zf(layout["zf_dir"], layout["zf_sha256"], man, ids)
    ref_missing = ([m for s in seeds for m in epoch0[s]["missing"]] +
                   zf["missing"])
    sources["zf|-|zf|zf|all"] = zf["rows"]
    sources["zf|-|zf|zf|mid"] = [r for r in zf["rows"] if r["midpoint"]]
    for k in sources:                      # engine rows carry metric keys only
        sources[k] = [{kk: vv for kk, vv in r.items()
                       if kk not in ("eligible", "midpoint")}
                      for r in sources[k]]
    return {"abbr": __abbr__, "version": __version__,
            "validation_volumes": man["validation_volumes"],
            "slice_rows": man["slice_rows"], "mid_ids": man["mid_ids"],
            "eligibility_sha256": man["eligibility_sha256"],
            "split_sha256": man["split_sha256"],
            "sources": sources, "run_info": run_info, "runs": runs,
            "posteval": posteval, "epoch0": epoch0, "zf": zf,
            "dc_tolerances": shared.get("dc_tolerances"),
            "masks_sha256": shared.get("masks_sha256"),
            "d6": {"rule": D6_RULE, "checks": d6},
            "reference_missing": ref_missing,
            "cases": man["cases"], "case_sha256": man["case_sha256"],
            "identities": dict(sorted(ids.files.items()))}


# ---------------------------------------------------------------------------
# Holdout records (I1.8 b3; 196 SEQREF-MRI-V05LTI18B3 v0.1)
# ---------------------------------------------------------------------------
HOLDOUT_SCHEMA = "seqref-v05lt-holdout-eval/1"
HOLDOUT_ROLES = ("selected", "epoch3")                 # evaluation scope
HOLDOUT_READOUTS = ("r0", "r1")
HOLDOUT_MASK_CONVENTION = (
    "holdout evaluation masks: canonical_mask_seed(EVAL_BASE_SEED, file, "
    "slice), no epoch (194 §3; V02 convention unchanged)")
HOLDOUT_N_VOLUMES = 199


def _is_sha(x) -> bool:
    return isinstance(x, str) and len(x) == 64 and all(
        c in "0123456789abcdef" for c in x)


def holdout_mask_seeds(rows: list) -> list:
    """Registered evaluation-mask seed per holdout slice, from the manifest
    identities only (production canonical_mask_seed; no data read)."""
    from seqref_mri.src import fastmri_data as FD
    return [FD.canonical_mask_seed(FD.EVAL_BASE_SEED, r["file"],
                                   r["slice_index"]) for r in rows]


def holdout_mask_seeds_sha256(rows: list) -> str:
    """Expected mask identity frozen in the holdout configuration."""
    return C.sha256_bytes(C.canonical_json(
        [[r["file"], r["slice_index"], sd]
         for r, sd in zip(rows, holdout_mask_seeds(rows))]))


def holdout_masks_sha256(seeds: list, widths: list) -> str:
    """sha256 over the C-order bool bytes of make_cartesian_mask(width,
    seed) in manifest order (the identity the I1.9 runner records)."""
    import hashlib
    from seqref_mri.src import fastmri_data as FD
    h = hashlib.sha256()
    for sd, w in zip(seeds, widths):
        h.update(FD.make_cartesian_mask(int(w), int(sd)).astype(
            bool).tobytes(order="C"))
    return h.hexdigest()


def load_holdout_manifest(path: Path, ids: Identities, *,
                          file_sha256: str = C.V02_SOURCES["holdout"][2],
                          content_sha256: str = C.V02_SOURCES["holdout"][3],
                          fixture: bool = False) -> dict:
    """V02 holdout manifest: file sha256 + sidecar, schema / kind and the
    canonical content sha256 (C.load_v02_source); one registered slice per
    volume, unique volume IDs (basename rule). Production: the registered
    hashes and 199 volumes; other hashes only with fixture=True."""
    path = Path(path)
    C.require(fixture or (file_sha256, content_sha256) ==
              C.V02_SOURCES["holdout"][2:], "LOAD_HOLDOUT_MANIFEST",
              "unregistered holdout manifest hashes outside fixtures")
    doc = C.load_v02_source(path.parent, "holdout", {"holdout": (
        path.name, "holdout", file_sha256, content_sha256)})
    ids.add(path, file_sha256)
    rows, vols = [], set()
    for e in doc.get("entries", []):
        C.require(isinstance(e, dict) and isinstance(e.get("file"), str) and
                  isinstance(e.get("slice_index"), int),
                  "LOAD_HOLDOUT_MANIFEST", f"entry {e!r}"[:160])
        v = Path(e["file"]).name
        C.require(v not in vols, "LOAD_HOLDOUT_MANIFEST", f"duplicate {v}")
        vols.add(v)
        rows.append({"volume_id": v, "file": e["file"],
                     "slice_index": e["slice_index"]})
    C.require(bool(rows) and (fixture or len(rows) == HOLDOUT_N_VOLUMES),
              "LOAD_HOLDOUT_MANIFEST", f"{len(rows)} holdout volumes")
    return {"rows": rows, "file_sha256": file_sha256,
            "content_sha256": content_sha256}


def _holdout_readout(rec, where: str, with_nll: bool) -> dict:
    C.require(isinstance(rec, dict), "LOAD_RECORD_INVALID",
              f"{where}: readout is {type(rec).__name__}")
    C.require(rec.get("nmse_u") is not None, "HOLDOUT_NMSE_UNDEFINED",
              f"{where}: NMSE_u missing or undefined -- AM4-B: exactly zero "
              "free-space energy is a validity error, never a dropped slice")
    out = _values(rec, where, False, [])
    if with_nll:
        v = rec.get("nll_per_dim")
        C.require(v is not None, "LOAD_VALUE_MISSING",
                  f"{where}: nll_per_dim missing on a trained checkpoint")
        C.require(isinstance(v, (int, float)) and not isinstance(v, bool),
                  "LOAD_RECORD_INVALID", f"{where}: nll_per_dim={v!r}")
        C.require(math.isfinite(v), "LOAD_NONFINITE",
                  f"{where}: nll_per_dim={v!r}")
        out["nll_per_dim"] = float(v)
    else:
        C.require("nll_per_dim" not in rec, "LOAD_RECORD_INVALID",
                  f"{where}: nll_per_dim on an R1 readout")
    return out


def load_holdout_records(holdout: dict, camp: dict, selection: dict,
                         ids: Identities, *, fixture: bool = False,
                         mask_seeds_sha256: str = None) -> dict:
    """holdout = {"manifest": path, "manifest_file_sha256",
    "manifest_content_sha256", "records": {"<arm>|<seed>|<role>": {"path",
    "sha256"}}}; camp = load_campaign output; selection = the verified,
    approved selection manifest. Returns engine sources keyed
    <arm>|<seed>|<role>|<r0|r1>|hold plus identities."""
    EV = _EV()
    man = load_holdout_manifest(
        holdout["manifest"], ids, file_sha256=holdout["manifest_file_sha256"],
        content_sha256=holdout["manifest_content_sha256"], fixture=fixture)
    want = {f"{r['arm']}|{r['seed']}|{role}" for r in camp["runs"].values()
            for role in HOLDOUT_ROLES}
    got = set(holdout.get("records", {}))
    C.require(got <= want, "LOAD_HOLDOUT_SCOPE", "records outside the "
              f"registered evaluation scope (selected, epoch 3): "
              f"{sorted(got - want)[:4]}")
    C.require(got == want, "LOAD_HOLDOUT_MISSING",
              f"missing holdout records {sorted(want - got)[:4]}")
    exp = [(r["volume_id"], r["file"], r["slice_index"]) for r in man["rows"]]
    seeds = holdout_mask_seeds(man["rows"])
    C.require(_is_sha(mask_seeds_sha256) and mask_seeds_sha256 ==
              holdout_mask_seeds_sha256(man["rows"]), "LOAD_MASK_MISMATCH",
              "expected mask-seed identity (frozen in the holdout "
              "configuration) does not match the registered manifest")
    sources, recs, masks, widths = {}, {}, None, None
    for key in sorted(want):
        arm, seed, role = key.split("|")
        run = camp["runs"][f"{arm}|{seed}"]
        ck = run["checkpoints"][role]
        ent = holdout["records"][key]
        path, sha = Path(ent["path"]), ent["sha256"]
        C.require(ids.hash(path) == sha, "LOAD_HOLDOUT_HASH",
                  f"{key}: {path} != recorded {sha}")
        side = Path(str(path) + ".sha256")
        C.require(side.is_file() and side.read_text().split()[:1] == [sha],
                  "LOAD_HOLDOUT_HASH", f"{key}: sidecar")
        doc = _json(path, "LOAD_HOLDOUT_INVALID")
        C.require(isinstance(doc, dict) and doc.get("schema") ==
                  HOLDOUT_SCHEMA, "LOAD_HOLDOUT_INVALID", f"{key}: schema")
        try:
            selrec = selection["runs"][f"{arm}|{seed}"][role]
        except (KeyError, TypeError):
            C.fail("LOAD_HOLDOUT_BINDING", f"{key}: not in the selection "
                   "manifest")
        bind = {"seed": int(seed), "arm": arm, "role": role,
                "epoch": ck["epoch"],
                "model_state_sha256": ck["model_state_sha256"],
                "checkpoint_sha256": selrec["sha256"]}
        C.require(selrec["epoch"] == ck["epoch"] and
                  selrec["model_state_sha256"] == ck["model_state_sha256"],
                  "LOAD_HOLDOUT_BINDING", f"{key}: selection manifest vs "
                  "verified commit")
        bad = {k: doc.get(k) for k, v in bind.items() if doc.get(k) != v}
        C.require(not bad, "LOAD_HOLDOUT_BINDING", f"{key}: {bad}")
        C.require(doc.get("manifest_content_sha256") ==
                  man["content_sha256"], "LOAD_HOLDOUT_BINDING",
                  f"{key}: holdout manifest content sha256")
        mk = doc.get("masks")
        C.require(isinstance(mk, dict) and _is_sha(mk.get("sha256")),
                  "LOAD_MASK_INVALID", f"{key}: masks sha256 must be 64 "
                  "lower-case hex")
        C.require(mk.get("convention") == HOLDOUT_MASK_CONVENTION and
                  mk.get("seeds_sha256") == mask_seeds_sha256,
                  "LOAD_MASK_MISMATCH", f"{key}: mask convention or seed "
                  "identity")
        w = mk.get("widths")
        C.require(isinstance(w, list) and len(w) == len(exp) and all(
            isinstance(x, int) and not isinstance(x, bool) and x >= 2
            for x in w), "LOAD_MASK_MISMATCH", f"{key}: mask widths")
        widths = w if widths is None else widths
        C.require(w == widths, "LOAD_MASK_MISMATCH",
                  f"{key}: mask widths differ between holdout records")
        if masks is None:
            masks = holdout_masks_sha256(seeds, widths)
        C.require(mk["sha256"] == masks, "LOAD_MASK_MISMATCH",
                  f"{key}: masks sha256 != recomputation from the "
                  "registered seeds (make_cartesian_mask)")
        C.require(doc.get("bank") == {"name": "primary", "array_sha256":
                                      EV.REGISTERED_BANK_ARRAY["primary"],
                                      "n_decodes": EV.BANK_ROWS},
                  "LOAD_HOLDOUT_BANK", f"{key}: R1 bank / decode count")
        rows = doc.get("rows")
        C.require(isinstance(rows, list) and len(rows) == len(exp),
                  "LOAD_POPULATION_MISMATCH", f"{key}: holdout row count")
        out = {ro: [] for ro in HOLDOUT_READOUTS}
        for (vid, f, sl), r in zip(exp, rows):
            C.require(isinstance(r, dict) and (r.get("volume_id"),
                                               r.get("file"),
                                               r.get("slice_index")) ==
                      (vid, f, sl), "LOAD_POPULATION_MISMATCH",
                      f"{key}: row {str(r)[:120]} vs manifest {(vid, f, sl)}")
            C.require(r.get("mask_seed") == seeds[len(out["r0"])],
                      "LOAD_MASK_MISMATCH", f"{key} {f}|{sl}: mask seed "
                      "!= canonical_mask_seed(EVAL_BASE_SEED, file, slice)")
            for ro in HOLDOUT_READOUTS:
                v = _holdout_readout(r.get(ro), f"{key} {f}|{sl} {ro}",
                                     ro == "r0")
                out[ro].append({"volume_id": vid, "file": f,
                                "slice_index": sl, **v})
        for ro in HOLDOUT_READOUTS:
            sources[f"{arm}|{seed}|{role}|{ro}|hold"] = out[ro]
        recs[key] = sha
    return {"sources": sources, "rows": man["rows"],
            "manifest_content_sha256": man["content_sha256"],
            "manifest_file_sha256": man["file_sha256"],
            "records": recs, "masks_sha256": masks}

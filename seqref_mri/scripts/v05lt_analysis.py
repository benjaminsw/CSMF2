# SEQREF-MRI-V05LTANA v0.7 -- scripts.v05lt_analysis
# LIFETIME: KEEP
# =============================================================================
# Purpose: the gated V05LT analysis stage (I1.8(b) stage b1, plan
#   SEQREF-MRI-V05LTI18B v0.2: B2 + the descriptive calculations; 194 §9,
#   §10; 195 §9.2, PA3, PA4). Three write-once steps, each refused under the
#   registered results tree unless production=True (gated stages only):
#   1. select_difficult (after gate 2a, before training; 195 PA4): the 10
#      validation volumes with the lowest midpoint ZF PSNR, ties broken by
#      volume ID, from the approved gate-2a ZF record -> difficult-case file
#      (+ sidecar); "difficult under the ZF-PSNR definition".
#   2. write_config (after gate 2a, before training; B2 timing): the
#      manifest-dependent analysis configuration of v05lt_stats (structure
#      and comparison rules frozen in v05lt_stats v0.3) -> config file
#      (+ sidecar); its sha256 is approved and recorded in V05LTEXEC.
#   3. run_analysis (after all runs): v05lt_results.load_campaign (verified
#      inputs) -> v05lt_stats.analyse against the APPROVED configuration
#      hash -> descriptive block computed once here -> analysis document
#      (+ sidecar) with source identities, D6 record and code identity.
#   Descriptive block (no CI, no labels; definitions and source identities
#   saved with the values; the report and plots read only these values):
#   per-case effects (PA4), R1 bank sensitivity and sample-count stability
#   (194 §8.1, §10), learning curves (plots 1, 4), NLL decomposition
#   (plot 5), DC measures vs tolerances (plot 6), epoch-0 / ZF references
#   recomputed from the loaded rows (v0.2: a metric with a missing
#   reference value is unavailable with identities; stored summaries are
#   not used).
#   Arithmetic: means are math.fsum / n (v05lt_stats.fmean); volume values
#   via v05lt_stats.volume_values (slice -> volume -> equal weight).
# Production: main() refuses (ANALYSIS_PRODUCTION_DISABLED).
# CONVENTION: logger.error + V05LTError(code); no substituted values.
# Fixture-only: n_difficult other than 10 and fixture configurations
#   (v05lt_stats fixture_override) are refused when production=True.
# Changelog (v0.7, 2026-10-25; I1.9 c2, PA5-B.3, item 26):
#   * Selection manifest: `final` = the published final-weights file
#     (<run>/final/weights_final.pt, kind `final_weights`), verified by
#     v05lt_final.verify_final and bound to the run's final epoch and model
#     state. A resume file is no longer accepted anywhere in the manifest
#     (FINAL_FROM_RESUME removed); epoch3 / selected stay committed weights.
# Changelog (v0.6, 2026-10-24; b3 revision-1 review fixes):
#   * write_holdout_config freezes the expected mask-seed identity from the
#     manifest; run_analysis passes it to the holdout loader.
#   * load_selection requires all 6 runs x {epoch3, selected, final} and
#     all 3 epoch-0 seeds with validated fields (SELECTION_INCOMPLETE /
#     SELECTION_INVALID); resume only for final, with FINAL_FROM_RESUME.
#   * Statements conditional: holdout result records vs MRI data.
# Changelog (v0.5, 2026-10-24; I1.8 b3, SEQREF-MRI-V05LTI18B3 v0.1):
#   * write_holdout_config: validation configuration + registered holdout
#     addendum (stats v0.4), write-once; frozen before holdout contact.
#   * Selection manifest (195 §9.3 step 1): per run epoch-3, selected and
#     final checkpoint files + sha256, epoch-0 checkpoint per seed, the
#     validation configuration / analysis / report identities and the
#     explicit holdout evaluation scope; write-once; load_selection.
#   * run_analysis(holdout=..., selection=...): holdout records loaded and
#     bound to the approved selection manifest; holdout block recorded.
# Changelog (v0.4, 2026-10-24; I1.8 b2, SEQREF-MRI-V05LTI18B v0.2):
#   * descriptive.nmse_exclusion: AM2-C §5 exclusion report (excluded
#     slices with identities for all slices, midpoints and visual cases;
#     null-NMSE volumes and population size per scope) so the b2 report
#     reads it from the analysis document instead of recomputing.
# Changelog (v0.3, 2026-10-24; b1 revision-3 review fix):
#   * dc_measures: coverage checked against the expected midpoint set; an
#     absent midpoint slice row is listed for both DC readouts and the
#     block is partial; n_expected / n_observed / n_with_measures reported;
#     maxima renamed max_ratio_to_tolerance_available_slices.
# Changelog (v0.2, 2026-10-24; b1 revision-2 review fixes):
#   * Missing reference values carried through the descriptive stage: DC
#     measures per readout with the missing slices listed; epoch-0 and ZF
#     reference values recomputed from the loaded rows (complete metric
#     populations only, else unavailable with identities); stored summaries
#     are never presented as verified.
# Changelog (NEW in v0.1, 2026-10-24; I1.8(b) b1, SEQREF-MRI-V05LTI18B v0.2):
#   * Introduced: PA4 difficult-case selection, configuration write, gated
#     analysis run, descriptive block.
# =============================================================================
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from seqref_mri.scripts import v05lt_common as C
from seqref_mri.scripts import v05lt_final as FW
from seqref_mri.scripts import v05lt_results as RS
from seqref_mri.scripts import v05lt_stats as S

logger = logging.getLogger("seqref_mri.v05lt_analysis")

__version__ = "0.7"
__abbr__ = "SEQREF-MRI-V05LTANA"

DIFFICULT_SCHEMA = "seqref-v05lt-difficult-cases/1"
RESULT_SCHEMA = "seqref-v05lt-analysis-stage/1"
SELECTION_SCHEMA = "seqref-v05lt-selection/1"
SELECTION_KINDS = {"epoch3": "weights", "selected": "weights",
                   "final": "final_weights"}          # PA5-B.3
N_DIFFICULT = 10                                         # 195 PA4
DIFFICULT_RULE = ("195 PA4: the 10 validation volumes with the lowest "
                  "midpoint ZF PSNR (AM3-D zero-filled reference), ties "
                  "broken by volume ID (ascending); 'difficult under the "
                  "ZF-PSNR definition'")
CODE_FILES = ("v05lt_analysis.py", "v05lt_results.py", "v05lt_stats.py",
              "v05lt_common.py", "v05lt_eval.py", "v05lt_metrics.py",
              "v05lt_train.py", "v05lt_rules.py")
SIGN_CONVENTION = ("nll_prior_per_dim = -log N(z; 0, I) / d_free; "
                   "nll_logdet_per_dim = -ldj / d_free; nll_per_dim = "
                   "-log p / d_free with log p = log N(z) + ldj (production "
                   "value). Larger = worse for all three.")


def _out_guard(path: Path, production: bool) -> Path:
    p = S._registered_dir_guard(path, production)     # refuses under v05lt
    return p


def _load_once(path: Path, sha: str, code: str) -> dict:
    path = Path(path)
    got = C.sha256_file(path)
    C.require(got == sha, code, f"{path}: {got} != recorded {sha}")
    side = Path(str(path) + ".sha256")
    C.require(side.is_file() and side.read_text().split()[:1] == [got],
              code, f"{side}: sidecar")
    try:
        return json.loads(path.read_bytes())
    except ValueError as exc:
        C.fail(code, f"{path}: {exc}")


# ---------------------------------------------------------------------------
# 1. PA4 difficult cases
# ---------------------------------------------------------------------------
def difficult_doc(zf: dict, validation_volumes: list,
                  eligibility_sha256: str, n: int = N_DIFFICULT) -> dict:
    mids = {}
    for r in zf["rows"]:
        if r["midpoint"]:
            C.require(r["volume_id"] not in mids, "DIFFICULT_INPUT",
                      f"{r['volume_id']}: two midpoint ZF rows")
            mids[r["volume_id"]] = r
    C.require(set(mids) == set(validation_volumes), "DIFFICULT_INPUT",
              "ZF midpoint rows do not cover exactly the validation volumes")
    C.require(isinstance(n, int) and 1 <= n <= len(mids), "DIFFICULT_INPUT",
              f"n={n}")
    order = sorted(mids.values(), key=lambda r: (r["psnr"], r["volume_id"]))
    sel = order[:n]
    return {"schema": DIFFICULT_SCHEMA, "abbr": __abbr__,
            "version": __version__, "rule": DIFFICULT_RULE, "n": n,
            "zf_record_sha256": zf["record_sha256"],
            "eligibility_sha256": eligibility_sha256,
            "volumes": [r["volume_id"] for r in sel],
            "values": [{"volume_id": r["volume_id"], "file": r["file"],
                        "slice_index": r["slice_index"],
                        "zf_psnr": r["psnr"]} for r in sel]}


def select_difficult(layout: dict, out_path: Path, *, production: bool =
                     False, n: int = N_DIFFICULT) -> str:
    """Gate-2a ZF record (recorded sha256) -> write-once difficult-case
    file. Returns its sha256 (recorded in V05LTEXEC before training)."""
    C.require(not production or n == N_DIFFICULT, "DIFFICULT_INPUT",
              f"production n must be {N_DIFFICULT}")
    ids = RS.Identities()
    man = RS.load_manifests(layout, ids)
    zf = RS.load_zf(layout["zf_dir"], layout["zf_sha256"], man, ids)
    doc = difficult_doc(zf, man["validation_volumes"],
                        man["eligibility_sha256"], n)
    p = _out_guard(out_path, production)
    return S.write_json_once(p, doc, production=production)


def load_difficult(path: Path, sha: str, zf_sha: str, validation_volumes:
                   list, production: bool = False) -> dict:
    doc = _load_once(path, sha, "DIFFICULT_INVALID")
    C.require(isinstance(doc, dict) and doc.get("schema") == DIFFICULT_SCHEMA
              and doc.get("zf_record_sha256") == zf_sha and
              isinstance(doc.get("volumes"), list) and
              len(doc["volumes"]) == doc.get("n") and
              set(doc["volumes"]) <= set(validation_volumes),
              "DIFFICULT_INVALID", f"{path}: structure / ZF binding")
    C.require(not production or doc["n"] == N_DIFFICULT, "DIFFICULT_INVALID",
              f"{path}: n = {doc['n']}")
    return doc


# ---------------------------------------------------------------------------
# 2. Analysis configuration (B2 timing)
# ---------------------------------------------------------------------------
def write_config(layout: dict, out_path: Path, *, production: bool = False,
                 B: int = S.B_REGISTERED, fixture_override: bool = False) -> str:
    """Manifest-dependent configuration (v05lt_stats.build_config) ->
    write-once file; its sha256 is approved and recorded before training."""
    C.require(not (production and fixture_override), "STATS_FIXTURE_CONFIG",
              "fixture configuration in production")
    ids = RS.Identities()
    man = RS.load_manifests(layout, ids)
    cfg = S.build_config(man["validation_volumes"], man["slice_rows"],
                         eligibility_sha256=man["eligibility_sha256"],
                         split_sha256=man["split_sha256"], B=B,
                         fixture_override=fixture_override)
    sha = S.write_json_once(_out_guard(out_path, production), cfg,
                            production=production)
    C.require(sha == S.config_sha256(cfg), "ANALYSIS_CONFIG_WRITE",
              "written bytes differ from the canonical configuration")
    return sha


def load_config(path: Path, approved_sha: str) -> dict:
    cfg = _load_once(path, approved_sha, "STATS_CONFIG_UNAPPROVED")
    C.require(S.config_sha256(cfg) == approved_sha, "STATS_CONFIG_UNAPPROVED",
              f"{path}: canonical sha256 != approved")
    return cfg


# ---------------------------------------------------------------------------
# 2b. Holdout addendum configuration (196 b3; frozen before holdout contact)
# ---------------------------------------------------------------------------
def write_holdout_config(val_config_path: Path, val_sha: str, holdout: dict,
                         out_path: Path, *, production: bool = False,
                         fixture: bool = False) -> str:
    """Approved validation configuration + the registered holdout addendum
    (21 comparisons; population appended after the validation ones) ->
    write-once file. Reads only the holdout MANIFEST (identities), never
    holdout data."""
    C.require(not (production and fixture), "ANALYSIS_FIXTURE_LAYOUT",
              "fixture holdout manifest in production")
    val = load_config(val_config_path, val_sha)
    ids = RS.Identities()
    man = RS.load_holdout_manifest(
        holdout["manifest"], ids,
        file_sha256=holdout["manifest_file_sha256"],
        content_sha256=holdout["manifest_content_sha256"], fixture=fixture)
    cfg = S.build_holdout_config(
        val, man["rows"], manifest_content_sha256=man["content_sha256"],
        mask_seeds_sha256=RS.holdout_mask_seeds_sha256(man["rows"]))
    sha = S.write_json_once(_out_guard(out_path, production), cfg,
                            production=production)
    C.require(sha == S.config_sha256(cfg), "ANALYSIS_CONFIG_WRITE",
              "written bytes differ from the canonical configuration")
    return sha


# ---------------------------------------------------------------------------
# 2c. Selection manifest (195 §9.3 step 1)
# ---------------------------------------------------------------------------
def _rel(path: Path, root: Path) -> str:
    p, r = Path(os.path.abspath(path)), Path(os.path.abspath(root))
    C.require(r in p.parents, "SELECTION_PATH", f"{p} not under {r}")
    return str(p.relative_to(r))


def _epoch0_ckpt(store_dir: Path, seed: int, model_state: str,
                 ids) -> dict:
    """Committed epoch-0 checkpoint of one seed: last commit of its store
    ledger, file hash and sidecar re-verified, model state = the runs'."""
    d = Path(store_dir)
    recs = [r for r in C.Ledger(d / "ledger.jsonl").records()
            if r["kind"] == "commit"]
    C.require(bool(recs), "SELECTION_EPOCH0", f"seed {seed}: no commit")
    body = recs[-1]["body"]
    rec = body.get("retained", {}).get("epoch0")
    C.require(isinstance(rec, dict) and body.get("seed") == seed and
              body.get("model_state_sha256") == model_state,
              "SELECTION_EPOCH0", f"seed {seed}: commit / model state")
    f = d / "ckpt" / rec["name"]
    C.require(ids.hash(f) == rec["sha256"], "SELECTION_EPOCH0",
              f"seed {seed}: {f.name} hash")
    side = Path(str(f) + ".sha256")
    C.require(side.is_file() and side.read_text().split()[:1] ==
              [rec["sha256"]], "SELECTION_EPOCH0", f"seed {seed}: sidecar")
    return {"path": f, "sha256": rec["sha256"],
            "model_state_sha256": model_state}


def selection_doc(layout: dict, root: Path, *, config: tuple,
                  analysis: tuple, report_dir: Path) -> dict:
    """config / analysis = (path, approved sha256); report_dir verified
    against the analysis. Checkpoints from the verified campaign."""
    from seqref_mri.scripts import v05lt_report as R
    load_config(config[0], config[1])
    doc = _load_once(analysis[0], analysis[1], "SELECTION_ANALYSIS")
    C.require(isinstance(doc, dict) and doc.get("config_sha256") ==
              config[1], "SELECTION_ANALYSIS", "analysis not bound to the "
              "configuration")
    rj = R.verify_report(report_dir, analysis[1])
    camp = RS.load_campaign(layout)
    ids = RS.Identities()
    runs = {}
    for key, run in sorted(camp["runs"].items()):
        ret, ckd = run["retained"], Path(run["ckpt_dir"])
        n = run["last_epoch"]
        out = {}
        for role in ("epoch3", "selected"):
            ck = run["checkpoints"][role]
            name = f"weights_e{ck['epoch']:03d}.pt"
            hit = [v for v in ret.values() if v["name"] == name]
            C.require(bool(hit), "SELECTION_CHECKPOINT",
                      f"{key} {role}: no committed weights file")
            out[role] = {"epoch": ck["epoch"],
                         "model_state_sha256": ck["model_state_sha256"],
                         "file": _rel(ckd / hit[0]["name"], root),
                         "sha256": hit[0]["sha256"], "kind": "weights"}
        ck = run["checkpoints"]["final"]           # PA5-B.3: final weights
        try:
            fw = FW.verify_final(run["dir"], fixture=bool(layout.get(
                "fixture")))
        except C.V05LTError as exc:
            C.fail("SELECTION_CHECKPOINT", f"{key} final: final weights "
                   f"not verified ({exc.code})")
        C.require(fw["epoch"] == ck["epoch"] == n and
                  fw["model_state_sha256"] == ck["model_state_sha256"],
                  "SELECTION_CHECKPOINT", f"{key} final: final weights "
                  f"epoch {fw['epoch']} / state not the run's final checkpoint")
        out["final"] = {"epoch": ck["epoch"],
                        "model_state_sha256": ck["model_state_sha256"],
                        "file": _rel(fw["path"], root),
                        "sha256": fw["sha256"], "kind": "final_weights"}
        runs[key] = out
    e0 = {}
    for s in layout["seeds"]:
        state = camp["runs"][f"{layout['arms'][0]}|{s}"]["epoch0_model_state"]
        st = layout["epoch0_store"].get(s, layout["epoch0_store"].get(str(s)))
        c = _epoch0_ckpt(st, s, state, ids)
        e0[str(s)] = {"file": _rel(c["path"], root), "sha256": c["sha256"],
                      "model_state_sha256": state}
    sums = C.sha256_file(Path(report_dir) / R.SUMS)
    return {"schema": SELECTION_SCHEMA, "abbr": __abbr__,
            "version": __version__, "runs": runs, "epoch0": e0,
            "validation": {"config_sha256": config[1],
                           "analysis_sha256": analysis[1],
                           "report_json_sha256": C.sha256_file(
                               Path(report_dir) / R.REPORT_JSON),
                           "report_sums_sha256": sums,
                           "report_version": rj["version"]},
            "evaluation_scope": S.HOLDOUT_SCOPE_TEXT,
            "paths_relative_to": "repository root (evidence-checker root)"}


def write_selection_manifest(layout: dict, root: Path, out_path: Path, *,
                             config: tuple, analysis: tuple,
                             report_dir: Path,
                             production: bool = False) -> str:
    """Write-once selection manifest; its sha256 is recorded in 196 and
    checked by the evidence checker before the holdout stage."""
    C.require(not (production and layout.get("fixture")),
              "ANALYSIS_FIXTURE_LAYOUT", "fixture layout in production")
    doc = selection_doc(layout, root, config=config, analysis=analysis,
                        report_dir=report_dir)
    return S.write_json_once(_out_guard(out_path, production), doc,
                             production=production)


SELECTION_ARMS = ("lam0", "lam2")
SELECTION_SEEDS = (2026, 2027, 2028)
SELECTION_ROLES = ("epoch3", "selected", "final")


def _sha_ok(x) -> bool:
    return isinstance(x, str) and len(x) == 64 and all(
        c in "0123456789abcdef" for c in x)


def _relfile_ok(x) -> bool:
    if not isinstance(x, str) or not x or x.startswith("/") or "\\" in x:
        return False
    return all(part not in ("", ".", "..") for part in x.split("/"))


def _int_ok(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def validate_selection(doc, *, seeds=SELECTION_SEEDS,
                       arms=SELECTION_ARMS) -> dict:
    """Completeness + field validation of a selection manifest. The
    registered 6 runs (arms x seeds), each with exactly epoch3, selected and
    final, and exactly the 3 epoch-0 seeds are required; an empty or partial
    manifest is refused even when its hashes are valid."""
    C.require(isinstance(doc, dict) and doc.get("schema") == SELECTION_SCHEMA
              and isinstance(doc.get("runs"), dict) and
              isinstance(doc.get("epoch0"), dict) and
              doc.get("evaluation_scope") == S.HOLDOUT_SCOPE_TEXT,
              "SELECTION_INVALID", "structure")
    want = {f"{a}|{s}" for a in arms for s in seeds}
    C.require(set(doc["runs"]) == want, "SELECTION_INCOMPLETE",
              f"runs {sorted(doc['runs'])} != registered {sorted(want)}")
    C.require(set(doc["epoch0"]) == {str(s) for s in seeds},
              "SELECTION_INCOMPLETE", f"epoch-0 seeds {sorted(doc['epoch0'])}")
    for key, run in sorted(doc["runs"].items()):
        C.require(isinstance(run, dict) and set(run) == set(SELECTION_ROLES),
                  "SELECTION_INCOMPLETE", f"{key}: roles "
                  f"{sorted(run) if isinstance(run, dict) else run!r}")
        for role in SELECTION_ROLES:
            ck = run[role]
            C.require(isinstance(ck, dict), "SELECTION_INVALID",
                      f"{key} {role}: entry")
            kind = ck.get("kind")
            base = {"epoch", "model_state_sha256", "file", "sha256", "kind"}
            C.require(set(ck) == base, "SELECTION_INVALID",
                      f"{key} {role}: fields {sorted(ck)}")
            C.require(_int_ok(ck["epoch"]) and (role != "epoch3" or
                                                ck["epoch"] == 3),
                      "SELECTION_INVALID", f"{key} {role}: epoch")
            C.require(_sha_ok(ck["model_state_sha256"]) and
                      _sha_ok(ck["sha256"]), "SELECTION_INVALID",
                      f"{key} {role}: sha256 form")
            C.require(_relfile_ok(ck["file"]), "SELECTION_INVALID",
                      f"{key} {role}: file path")
            C.require(kind == SELECTION_KINDS[role], "SELECTION_INVALID",
                      f"{key} {role}: kind {kind!r} != "
                      f"{SELECTION_KINDS[role]!r} (resume files refused)")
        C.require(run["selected"]["epoch"] <= run["final"]["epoch"] and
                  run["epoch3"]["epoch"] <= run["final"]["epoch"],
                  "SELECTION_INVALID", f"{key}: epoch order")
    for s, ck in sorted(doc["epoch0"].items()):
        C.require(isinstance(ck, dict) and set(ck) == {
            "file", "sha256", "model_state_sha256"} and _relfile_ok(
            ck["file"]) and _sha_ok(ck["sha256"]) and _sha_ok(
            ck["model_state_sha256"]), "SELECTION_INVALID",
            f"epoch-0 seed {s}: fields")
    v = doc.get("validation")
    C.require(isinstance(v, dict) and all(_sha_ok(v.get(k)) for k in (
        "config_sha256", "analysis_sha256", "report_json_sha256",
        "report_sums_sha256")), "SELECTION_INVALID", "validation identities")
    return doc


def load_selection(path: Path, sha: str, *, seeds=SELECTION_SEEDS,
                   arms=SELECTION_ARMS) -> dict:
    doc = _load_once(path, sha, "SELECTION_INVALID")
    try:
        return validate_selection(doc, seeds=seeds, arms=arms)
    except C.V05LTError as exc:
        logger.error("%s: selection manifest refused (%s)", path, exc.code)
        raise


# ---------------------------------------------------------------------------
# Descriptive block (computed once; definitions + source identities)
# ---------------------------------------------------------------------------
def _vol_means(rows, metric, ex, scope):
    vals, miss, _ = S.volume_values(rows, metric, ex[scope],
                                    eligible=ex["eligible"]
                                    if metric == "nmse_u" else None)
    C.require(not miss, "DESC_VALUE_MISSING", f"{metric}: {miss[:3]}")
    return vals


def _mean_present(d: dict):
    v = [x for x in d.values() if x is not None]
    return S.fmean(v) if v else None


def per_case(camp: dict, ex: dict, difficult: dict, seeds, arms) -> dict:
    """195 PA4 §3: Δ_v = mean over seeds of [PSNR(lam2, selected) -
    PSNR(lam0, selected)] on R0 all-slice volume means."""
    vols = camp["validation_volumes"]
    per_seed = {}
    for s in seeds:
        a = _vol_means(camp["sources"][f"lam2|{s}|selected|r0|all"], "psnr",
                       ex, "all")
        b = _vol_means(camp["sources"][f"lam0|{s}|selected|r0|all"], "psnr",
                       ex, "all")
        per_seed[s] = {v: a[v] - b[v] for v in vols}
    delta = {v: S.fmean([per_seed[s][v] for s in seeds]) for v in vols}

    def counts(sel):
        d = [delta[v] for v in sel]
        pos = sum(1 for x in d if x > 0)
        return {"n": len(d), "positive": pos,
                "zero": sum(1 for x in d if x == 0),
                "negative": sum(1 for x in d if x < 0),
                "fraction_improved": pos / len(d)}
    return {"definition": "195 PA4 §3-§4: case = validation volume (all); "
                          "delta_v = seed-mean of lam2 - lam0 R0 all-slice "
                          "volume PSNR at the selected checkpoint; improved "
                          "= delta_v > 0; descriptive only (no CI, no "
                          "subgroup test, no labels)",
            "difficult_rule": difficult["rule"],
            "difficult_volumes": difficult["volumes"],
            "delta_db": [{"volume_id": v, "delta": delta[v],
                          "difficult": v in difficult["volumes"],
                          "per_seed": {str(s): per_seed[s][v]
                                       for s in seeds}} for v in vols],
            "all": counts(vols),
            "difficult": counts(difficult["volumes"]),
            "sources": [f"{a}|{s}|selected|r0|all" for s in seeds
                        for a in ("lam2", "lam0")]}


def bank_sensitivity(camp: dict, ex: dict) -> dict:
    """194 §8.1, §10: R1 per-bank results at the selected checkpoint
    (k = 128) and their spread; descriptive."""
    out = {}
    for key, pe in sorted(camp["posteval"].items()):
        if not key.endswith("|selected"):
            continue
        per = {}
        vm = {}
        for b, by_k in pe["per_bank"].items():
            rows = by_k["128"]
            vm[b] = _vol_means(rows, "psnr", ex, "mid")
            per[b] = {m: _mean_present(_vol_means(rows, m, ex, "mid"))
                      for m in ("psnr", "ssim", "nmse_u")}
        vols = list(vm["primary"])
        spread = {v: max(vm[b][v] for b in vm) - min(vm[b][v] for b in vm)
                  for v in vols}
        out[key[:-len("|selected")]] = {
            "banks": per, "psnr_spread_db": {
                "mean": S.fmean(list(spread.values())),
                "max": max(spread.values())},
            "record_sha256": pe["record_sha256"]}
    return {"definition": "R1 (k = 128) on the midpoint slices per bank at "
                          "the selected checkpoint: mean over volumes of the "
                          "volume values; spread = max - min over banks of "
                          "the volume PSNR, summarised by mean and max; "
                          "descriptive only",
            "runs": out}


def prefix_stability(camp: dict, ex: dict) -> dict:
    out = {}
    for key, pe in sorted(camp["posteval"].items()):
        if not key.endswith("|selected"):
            continue
        by_k = pe["per_bank"]["primary"]
        means = {k: _mean_present(_vol_means(by_k[k], "psnr", ex, "mid"))
                 for k in sorted(by_k, key=int)}
        out[key[:-len("|selected")]] = {
            "psnr_db": means,
            "minus_k128_db": {k: means[k] - means["128"] for k in means},
            "record_sha256": pe["record_sha256"]}
    return {"definition": "R1 on the primary bank from the first k decodes "
                          "(k = 16, 32, 64, 128; no extra decodes) at the "
                          "selected checkpoint: mean PSNR over midpoint "
                          "volumes and difference to k = 128; descriptive",
            "runs": out}


def _ref_metric(rows: list, metric: str, ex: dict) -> dict:
    """Reference value on the all-slice scope (slice -> volume -> equal
    weight) from the loaded rows; unavailable with identities when any
    required value is missing (AM3-E4)."""
    vals, miss, _ = S.volume_values(rows, metric, ex["all"],
                                    eligible=ex["eligible"]
                                    if metric == "nmse_u" else None,
                                    reference=True)
    if miss:
        return {"value": None, "status": "unavailable",
                "reason": "AM3-E4: reference value missing",
                "identities": sorted(miss)}
    v = [x for x in vals.values() if x is not None]
    if not v:
        return {"value": None, "status": "unavailable_no_eligible_volume"}
    return {"value": S.fmean(v), "status": "ok"}


def reference_values(camp: dict, ex: dict) -> dict:
    e0 = {}
    for s_, rec in sorted(camp["epoch0"].items()):
        rows = camp["sources"][f"shared|{s_}|epoch0|r0|all"]
        e0[str(s_)] = {m: _ref_metric(rows, m, ex)
                       for m in ("psnr", "ssim", "nmse_u", "nll_per_dim")}
        e0[str(s_)].update(record_sha256=rec["record_sha256"],
                           stored_summary_verified=rec["summary_verified"],
                           missing=rec["missing"])
    zrows = camp["sources"]["zf|-|zf|zf|all"]
    zf = {m: _ref_metric(zrows, m, ex) for m in ("psnr", "ssim", "nmse_u")}
    zf.update(record_sha256=camp["zf"]["record_sha256"],
              stored_summary_verified=camp["zf"]["summary_verified"],
              missing=camp["zf"]["missing"], note="ZF has no NLL (AM3-D)")
    return {"definition": "Epoch-0 and ZF all-slice reference values "
                          "recomputed from the loaded rows (slice -> volume "
                          "-> equal-weight; NMSE_u over eligible volumes); a "
                          "metric with any missing value is unavailable "
                          "with identities (AM3-E4); stored record "
                          "summaries are not used",
            "epoch0": e0, "zf": zf}


def learning_curves(camp: dict, ex: dict) -> dict:
    out = {}
    for key, run in sorted(camp["runs"].items()):
        eps = []
        for e in sorted(run["epochs"]):
            ep = run["epochs"][e]
            a, m = ep["validation"]["all_slices"], ep["validation"]["midpoint"]
            eps.append({"epoch": e, "step": ep["step"],
                        "psnr": a["psnr"], "nll_per_dim": a["nll_per_dim"],
                        "ssim": a["ssim"], "nmse_u": a["nmse_u"],
                        "psnr_mid": m["psnr"] if m else None,
                        "nll_mid": m["nll_per_dim"] if m else None,
                        "plateau_ref": ep["plateau"]["ref"],
                        "plateau_count": ep["plateau"]["count"]})
        last = run["epochs"][run["last_epoch"]]["plateau"]
        out[key] = {"epochs": eps, "resets": last["resets"],
                    "selected_epoch": run["checkpoints"]["selected"]["epoch"],
                    "best_nll_epoch": run["checkpoints"]["best_nll"]["epoch"],
                    "stop_epoch": run["last_epoch"], "status": run["status"],
                    "epoch3": 3, "milestones": run["milestones"]}
    refs = reference_values(camp, ex)
    return {"definition": "Per run and epoch: committed all-slice and "
                          "midpoint validation summaries (verified equal to "
                          "the row files), update count, plateau references "
                          "and counters; resets, selected, best-NLL and stop "
                          "epochs; epoch-0 and ZF references (plots 1, 4)",
            "runs": out, "references": refs}


def nll_decomposition(camp: dict, ex: dict) -> dict:
    out = {}
    for key, run in sorted(camp["runs"].items()):
        eps = []
        for e in sorted(run["rows"]):
            rows = run["rows"][e]["rows"]
            vals = {}
            for f in ("nll_prior_per_dim", "nll_logdet_per_dim",
                      "nll_per_dim"):
                per_v = {}
                for r in rows:
                    per_v.setdefault(r["volume_id"], []).append(r[f])
                vals[f] = S.fmean([S.fmean(per_v[v])
                                   for v in camp["validation_volumes"]])
            eps.append({"epoch": e, **vals})
        out[key] = eps
    return {"definition": "Per run and epoch, all validation slices: slice "
                          "-> volume mean -> equal-weight volume mean of the "
                          "prior and log-determinant contributions and the "
                          "production NLL per free real dimension (plot 5)",
            "sign_convention": SIGN_CONVENTION, "runs": out}


def dc_measures(camp: dict) -> dict:
    """Per record and DC readout: validated measures per slice (loader).
    Coverage is checked against the expected midpoint set (camp["mid_ids"]):
    every expected slice without measures is listed (absent slice row,
    absent readout record or absent measures; AM3-E4) and the block is then
    partial. Maxima cover the AVAILABLE slices only (v0.3)."""
    tol = camp["dc_tolerances"]
    want = [(f, s) for f, s in camp["mid_ids"]]
    recs = [(k, pe["dc"], pe.get("missing", []), pe["record_sha256"])
            for k, pe in sorted(camp["posteval"].items())]
    recs += [(f"shared|{s_}|epoch0", e0["dc"], e0["missing"],
              e0["record_sha256"]) for s_, e0 in sorted(camp["epoch0"].items())]
    out = {}
    for key, dc, miss, sha in recs:
        blk = {}
        for ro in ("r0_dc", "r1_dc"):
            sl = dc[ro]
            seen = {(x["file"], x["slice_index"]): x for x in sl}
            C.require(len(seen) == len(sl) and set(seen) <= set(want),
                      "ANALYSIS_DC_COVERAGE", f"{key} {ro}: duplicate or "
                      "unexpected slice in DC measures")
            have = [x for x in sl if x["measures"] is not None]
            gone = []
            for f, s in want:
                x = seen.get((f, s))
                if x is not None and x["measures"] is not None:
                    continue
                tag = f"{f}|{s}"
                why = sorted(m for m in miss if m.endswith((
                    f" {tag}: slice absent", f" {tag}: {ro} absent",
                    f" {tag} {ro}: DC measures absent")))
                gone += why or [f"{key} {tag}: {ro} measures not observed"]
            blk[ro] = {
                "slices": sl, "n_expected": len(want),
                "n_observed": len(sl), "n_with_measures": len(have),
                "n_identical": sum(1 for x in sl if x["identical"]),
                "max_ratio_to_tolerance_available_slices": (
                    {m: max(x["measures"][m] / tol[m] for x in have)
                     for m in RS.DC_MEASURES} if have else None),
                "missing": gone,
                "status": "ok" if len(have) == len(want) else
                          "partial: maxima cover available slices only; "
                          "missing slices listed"}
        out[key] = {**blk, "record_sha256": sha}
    return {"definition": "DC measures (a), (b), (c_rms), (c_max) per "
                          "midpoint slice for R0+DC and R1+DC against the "
                          "calibrated tolerances (194 §8; plot 6). "
                          "n_expected = registered midpoint slices; "
                          "n_observed = slices with a readout record; "
                          "maxima cover the available slices (with "
                          "measures) only; every expected slice without "
                          "measures is listed and the block is partial "
                          "(AM3-E4)",
            "tolerances": tol, "sources": out}


def nmse_exclusion(camp: dict) -> dict:
    """194 AM2-C §5 (v0.4): NMSE_u exclusion report from the verified
    eligibility-manifest rows -- counts and identities of excluded slices
    (all slices, the midpoint slices, the visual cases) and the volumes
    with no eligible slice per scope (NMSE null, outside the population)."""
    rows = camp["slice_rows"]
    cases = {(c["file"], c["slice_index"]): c["case"] for c in camp["cases"]}

    def ident(r):
        return {"volume_id": r["volume_id"], "file": r["file"],
                "slice_index": r["slice_index"]}

    def scope(sel):
        ex = [ident(r) for r in sel if not r["eligible"]]
        vols = sorted({r["volume_id"] for r in sel})
        elig = {r["volume_id"] for r in sel if r["eligible"]}
        return {"n_slices": len(sel), "n_excluded": len(ex), "excluded": ex,
                "null_volumes": [v for v in vols if v not in elig],
                "population_volumes": len(elig)}
    mid = [r for r in rows if r["midpoint"]]
    return {"definition": "194 AM2-C: a slice is NMSE-eligible iff E / "
                          "S_ref^2 > 1e-10 (registered eligibility "
                          "manifest); excluded slices are counted and "
                          "identified (all slices, midpoint slices, visual "
                          "cases); a volume with no eligible slice in a "
                          "scope has NMSE null and is outside that scope's "
                          "NMSE population. PSNR, SSIM and NLL keep all "
                          "slices",
            "eligibility_sha256": camp["eligibility_sha256"],
            "all": scope(rows), "mid": scope(mid),
            "cases_excluded": sorted(cases[(r["file"], r["slice_index"])]
                                     for r in mid if not r["eligible"] and
                                     (r["file"], r["slice_index"]) in cases)}


def descriptive(camp: dict, difficult: dict, seeds, arms) -> dict:
    ex = S.expected_slices(camp["validation_volumes"], camp["slice_rows"])
    return {"per_case": per_case(camp, ex, difficult, seeds, arms),
            "bank_sensitivity": bank_sensitivity(camp, ex),
            "prefix_stability": prefix_stability(camp, ex),
            "learning_curves": learning_curves(camp, ex),
            "nll_decomposition": nll_decomposition(camp, ex),
            "dc_measures": dc_measures(camp),
            "nmse_exclusion": nmse_exclusion(camp)}


# ---------------------------------------------------------------------------
# 3. Analysis run
# ---------------------------------------------------------------------------
def code_identity() -> dict:
    here = Path(__file__).resolve().parent
    return {n: C.sha256_file(here / n) for n in CODE_FILES}


def run_analysis(layout: dict, config_path: Path, approved_sha: str,
                 out_path: Path, *, production: bool = False,
                 holdout: dict = None, selection: tuple = None) -> str:
    """Verified inputs -> registered analysis (approved configuration) +
    descriptive block -> write-once analysis document. Returns its sha256."""
    p = _out_guard(out_path, production)
    C.require(not (production and layout.get("fixture")),
              "ANALYSIS_FIXTURE_LAYOUT", "fixture layout in production")
    cfg = load_config(config_path, approved_sha)
    camp = RS.load_campaign(layout)
    C.require(cfg["inputs"] == {"eligibility_manifest_sha256":
                                camp["eligibility_sha256"],
                                "split_manifest_sha256": camp["split_sha256"]},
              "STATS_CONFIG_INPUTS", "configuration inputs differ from the "
              "loaded manifests")
    difficult = load_difficult(layout["difficult"], layout["difficult_sha256"],
                               camp["zf"]["record_sha256"],
                               camp["validation_volumes"], production)
    hres, sel_sha, hid = None, None, RS.Identities()
    C.require(("holdout" in cfg) == (holdout is not None) ==
              (selection is not None), "STATS_HOLDOUT_MISMATCH",
              "holdout records and selection manifest are given iff the "
              "configuration carries the holdout addendum")
    sources = camp["sources"]
    if holdout is not None:                     # 196 b3 holdout addendum
        sel = load_selection(selection[0], selection[1])
        C.require(sel["validation"]["config_sha256"] ==
                  S.config_sha256(S.validation_part(cfg)),
                  "SELECTION_INVALID", "selection manifest not bound to the "
                  "validation part of this configuration")
        hres = RS.load_holdout_records(
            holdout, camp, sel, hid, fixture=bool(layout.get("fixture")),
            mask_seeds_sha256=cfg["holdout"]["mask_seeds_sha256"])
        C.require(hres["manifest_content_sha256"] ==
                  cfg["holdout"]["manifest_content_sha256"],
                  "STATS_HOLDOUT_MISMATCH", "holdout manifest differs from "
                  "the configuration")
        sources = dict(sources, **hres["sources"])
        sel_sha = selection[1]
    res = S.analyse(cfg, approved_sha, sources, camp["slice_rows"],
                    camp["run_info"], production=production,
                    holdout_rows=None if hres is None else hres["rows"])
    desc = descriptive(camp, difficult, layout["seeds"], layout["arms"])
    doc = {"schema": RESULT_SCHEMA, "abbr": __abbr__, "version": __version__,
           "config_sha256": approved_sha, "analysis": res,
           "descriptive": desc,
           "inputs": {"identities": dict(sorted({**camp["identities"],
                                                **hid.files}.items())),
                      "d6": camp["d6"],
                      "run_info": {f"{a}|{s}": v for (a, s), v in
                                   sorted(camp["run_info"].items())},
                      "difficult_sha256": layout["difficult_sha256"],
                      "masks_sha256": camp["masks_sha256"]},
           **({} if hres is None else {"holdout": {
               "statement": S.HOLDOUT_STATEMENT,
               "nmse_rule": S.HOLDOUT_NMSE_RULE,
               "evaluation_scope": S.HOLDOUT_SCOPE_TEXT,
               "manifest_content_sha256": hres["manifest_content_sha256"],
               "manifest_file_sha256": hres["manifest_file_sha256"],
               "records": hres["records"],
               "masks_sha256": hres["masks_sha256"],
               "mask_seeds_sha256": cfg["holdout"]["mask_seeds_sha256"],
               "selection_sha256": sel_sha}}),
           "code": code_identity(),
           "statements": [
               ("Validation only; the holdout is not read by this stage."
                if hres is None else
                "Validation and holdout: this stage reads the holdout "
                "RESULT RECORDS (per-slice metrics written by the holdout "
                "runner) and the holdout manifest; it never reads MRI data "
                "(validation or holdout)."),
               "The 199-volume holdout has already been used (V02, V02G, "
               "V03); any later holdout result is not an independent "
               "confirmation (194 §10.1)."]}
    return S.write_json_once(p, json.loads(json.dumps(doc)),
                             production=production)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    argparse.ArgumentParser(description=f"{__abbr__} v{__version__}"
                            ).parse_args(argv)
    try:
        C.fail("ANALYSIS_PRODUCTION_DISABLED", "the V05LT analysis stage runs "
               "only from the gated stage with recorded approvals (195 §8.3, "
               "§9.2; 196 I1.8)")
    except C.V05LTError as exc:
        print(f"ANALYSIS REFUSED {exc.code}")
        return 2


if __name__ == "__main__":
    sys.exit(main())

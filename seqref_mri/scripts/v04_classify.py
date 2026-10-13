# SEQREF-MRI-V04C v0.1 -- scripts.v04_classify
# LIFETIME: KEEP
# =============================================================================
# Purpose: mechanical V04 classification + within-architecture routing
#          (V04SPEC v1.0 SS14/SS15, article 176 FROZEN; V04PLAN v1.0,
#          article 178 FROZEN + Amendment A1, article 180 FROZEN). The
#          routing matrix is inherited verbatim from V03SPEC SS15
#          (article 163 FROZEN) with classes renamed V04-A..V04-E; the
#          threshold values are inherited verbatim from V03SPEC SS14
#          (176 SS14; 173 SS7 cross-flow comparability). Consumes ONLY
#          the frozen seqref-v04-facts/1 artifact plus the pinned
#          199-volume holdout manifest: no model, no data, no metric
#          computation, no training, no holdout contact.
# Rules (locked):
#   - The only classified arm is the parity arm lambda_2' (arm 1); the
#     overall V04 class IS the parity arm's class (176 SS7/SS15). The
#     lambda=0 anchor arm 0 is never classified.
#   - D_rec readout: delta_psnr_r0 (R0 z=0) per Amendment A4 (167);
#     delta_psnr_r1 and the NMSE_u deltas are sensitivity evidence,
#     never classified (176 SS14).
#   - SS14 thresholds: material improvement mean >= +0.5 dB with the
#     95% CI excluding 0; material degradation mean <= -0.5 dB with CI
#     excluding 0; density degraded when relative > +1% with CI
#     excluding 0; S_inst = numerical failure in >=2 of 3 seeds. Any
#     CI straddling a classification boundary, an incomplete arm, or an
#     uncovered matrix cell routes to V04-E (163 SS15). Density-CI
#     precedence (Amendment A5, 168): a CI touching or containing the
#     +1% boundary is a straddle, overriding the preservation clauses.
#   - Interpretation (176 SS4, locked matrix): the SS4 verdict is
#     assigned mechanically per SS15 -- never by hand. The verdict is
#     strictly within-architecture; the SS13 DeltaDelta cross-
#     architecture comparison is V04X's responsibility. V04C makes no
#     cross-flow claim.
# Reuse (single source, no competing implementations):
#   - classify_reconstruction / classify_density are IMPORTED from the
#     registered v03_classify v0.3: pure predicates over (mean, ci95)
#     carrying the inherited threshold values.
#   - scan_for_verdict / validate_facts / validate_parent_references
#     are imported from v04_eval v0.1 (the f07/f08 integrity layer);
#     the holdout manifest is loaded through v02_eval._load_manifest.
#     (Importing v04_eval pulls the evaluation dependency chain incl.
#     torch; V04C itself performs no tensor work.)
# Output: seqref-v04-routing-record/1 (v04_routing_record.json) +
#   SHA-256 sidecar + semantic hash. Deterministic by construction:
#   re-runs on the same frozen inputs are bit-identical; the record
#   carries no wall-clock fields and no timestamps.
# CONVENTION: logger.error + typed raise (V04CError). PASS(0)/ERROR(2).
#   No fallback, no mock, no placeholder, no silent pass.
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from pathlib import Path

from seqref_mri.scripts import v02_eval as v02e
from seqref_mri.scripts import v02_manifests as v02m
from seqref_mri.scripts import v03_classify as v03c
from seqref_mri.scripts import v04_eval as v04e
from seqref_mri.scripts.v02_manifests import canonical_json

logger = logging.getLogger("seqref_mri.v04_classify")

__version__ = "0.1"
__abbr__ = "SEQREF-MRI-V04C"

FACTS_SCHEMA = "seqref-v04-facts/1"
ROUTING_SCHEMA = "seqref-v04-routing-record/1"
HOLDOUT_MANIFEST = v04e.HOLDOUT_MANIFEST
PRIMARY_ARM = 1                      # lambda_2' parity arm (176 SS7/SS15)
CLASS_ARMS = (1,)                    # anchor arm 0 is never classified
ARMS_COVERAGE = [0, 1]               # 2-arm design (176 SS7)
N_SEEDS = 3                          # locked seed set {101, 202, 303}
SEED_FAILURE_D = 2                   # SS14 S_inst: >=2 of 3 seeds
# Threshold values inherited verbatim (176 SS14); the single source is
# the registered V03 implementation -- referenced, not redeclared.
REC_IMPROVE_DB = v03c.REC_IMPROVE_DB
REC_DEGRADE_DB = v03c.REC_DEGRADE_DB
NLL_REL_TOLERANCE = v03c.NLL_REL_TOLERANCE
classify_reconstruction = v03c.classify_reconstruction
classify_density = v03c.classify_density
_SEMANTIC_EXCLUDE = ("run", "semantic_sha256", "semantic_scope")
# The closed top-level key set of the production facts artifact
# (v04_eval._build_facts + the semantic block). Any other top-level
# key -- including a pre-made verdict smuggled under an innocent
# name -- is refused.
_FACTS_TOP_LEVEL_KEYS = frozenset({
    "schema", "script", "spec", "plan", "code", "seeds", "arms",
    "evaluation", "v03_facts_reference", "parent_references",
    "per_arm_seed", "per_arm", "environment",
    "semantic_sha256", "semantic_scope"})
# 176 SS4 interpretation matrix (locked); registered wording verbatim
# (markdown emphasis stripped). V04-B and V04-D share one matrix row.
INTERPRETATION_MAP = {
    "V04-A": ("RealNVP meets the material-improvement criterion; "
              "suggests architecture-dependent objective alignment \u2014 "
              "confirmed only if the \u00a713 \u0394\u0394 CI excludes "
              "0; H1 re-opens for the RNVP track only"),
    "V04-B": ("Degradation or instability evidence; report; no "
              "automatic experiment expansion"),
    "V04-C": ("Both architectures fail the material-improvement test; "
              "the null is supported across the two tested "
              "architectures (NSF and RealNVP) \u2014 decisive_null "
              "stands, strengthened. Not a claim about all normalising "
              "flows: NICE remains untested and parked"),
    "V04-D": ("Degradation or instability evidence; report; no "
              "automatic experiment expansion"),
    "V04-E": ("Inconclusive; further decisions require the registered "
              "amendment process (\u00a723)")}
INTERPRETATION_SCOPE = ("within-architecture only (176 \u00a74): the "
                        "\u00a713 \u0394\u0394 cross-architecture "
                        "comparison is V04X's responsibility; V04C "
                        "makes no cross-flow claim")


class V04CError(Exception):
    """Typed classification failure: .code is the machine-readable
    reason."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise V04CError(code, message)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _register_src_path() -> Path:
    """The preflight modules import their siblings top-level; register
    src/ so the module is import-safe in any environment. Derived from
    this file's own location -- deterministic, not a fallback."""
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def _canonical_hash():
    _register_src_path()
    try:
        from preflight_io import canonical_hash
    except ImportError as exc:
        _fail("ENV_IMPORT_FAILED",
              f"preflight_io is not importable in this environment: "
              f"{exc}")
    return canonical_hash


def _verify_sidecar_file(path: Path) -> str:
    side = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not side.is_file():
        _fail("CLASSIFY_INPUT_MISSING",
              f"frozen input or sidecar missing at {path}")
    recorded = side.read_text().split()[0]
    digest = _file_sha256(path)
    if digest != recorded:
        _fail("CLASSIFY_INPUT_DRIFT",
              f"{path.name}: sha256 != the recorded sidecar digest")
    return digest


def _semantic_sha(payload: dict) -> str:
    return _canonical_hash()(
        {k: v for k, v in payload.items() if k not in _SEMANTIC_EXCLUDE})


# ---------------------------------------------------------------------------
# Facts artifact loading and integrity gates
# ---------------------------------------------------------------------------

def _finite_pair(node: dict, key: str, what: str) -> tuple:
    m = node.get(key)
    ci = node.get("ci95")
    if not isinstance(m, (int, float)) or isinstance(m, bool) \
            or not math.isfinite(float(m)):
        _fail("FACTS_RECORD_INVALID",
              f"{what}: {key} {m!r} is not a finite number")
    if not isinstance(ci, list) or len(ci) != 2 \
            or not all(isinstance(x, (int, float))
                       and not isinstance(x, bool)
                       and math.isfinite(float(x)) for x in ci):
        _fail("FACTS_RECORD_INVALID",
              f"{what}: ci95 {ci!r} is not a finite pair")
    return float(m), (float(ci[0]), float(ci[1]))


def _load_holdout_identities(manifest_dir: Path) -> tuple:
    """The pinned 199-volume holdout manifest (178 SS10/SS11): schema,
    kind, sidecar and internal hash are verified by the registered V02
    loader (its typed V02Error propagates unchanged); the locked volume
    count is gated here, identical to V04E."""
    holdman = v02e._load_manifest(manifest_dir, HOLDOUT_MANIFEST,
                                  "holdout")
    if int(holdman["n_volumes"]) != v02m.N_HOLDOUT_VOLUMES \
            or len(holdman["entries"]) != v02m.N_HOLDOUT_VOLUMES:
        _fail("MANIFEST_CONTENT_MISMATCH",
              "holdout manifest diverges from the locked "
              f"{v02m.N_HOLDOUT_VOLUMES} volumes")
    identities = [{"file": e["file"],
                   "slice_index": int(e["slice_index"])}
                  for e in holdman["entries"]]
    return identities, _file_sha256(manifest_dir / HOLDOUT_MANIFEST)


def validate_classification_inputs(rows: dict) -> None:
    """Decision-input gates on the classified arm(s): a complete
    classified arm must carry finite delta_psnr_r0 (mean/ci95) and
    delta_nll_relative (relative/ci95) vs the lambda=0 anchor.
    Incomplete arms carry no aggregates by the V04E contract -- that is
    accepted and routes mechanically, never coerced."""
    for arm in CLASS_ARMS:
        r = rows[arm]
        if r["ensemble_status"] != "complete":
            continue
        agg = r.get("aggregates")
        if not isinstance(agg, dict) or "means_ci" not in agg:
            _fail("FACTS_RECORD_INVALID",
                  f"complete arm {arm}: aggregates.means_ci missing")
        deltas = agg.get("delta_vs_lambda0")
        if not isinstance(deltas, dict):
            _fail("FACTS_RECORD_INVALID",
                  f"complete arm {arm}: delta_vs_lambda0 missing; the "
                  f"anchor ensemble is required for classification")
        _finite_pair(deltas.get("delta_psnr_r0") or {}, "mean",
                     f"arm {arm} delta_psnr_r0")
        _finite_pair(deltas.get("delta_nll_relative") or {},
                     "relative", f"arm {arm} delta_nll_relative")


def load_verified_facts(path: Path, manifest_dir: Path) -> tuple:
    """The frozen facts artifact, pinned four ways: SHA-256 sidecar,
    schema, the registered V04E integrity layer (injected-verdict
    scan, per-volume holdout identities vs the pinned manifest, parent
    references), and semantic self-consistency; then the structural
    arm gates (coverage exactly {0, 1}, explicit ensemble_status,
    stability-count range/sum/status agreement). The content screens
    run BEFORE the content hash so a non-finite tamper lands on the
    typed finite-scan refusal (canonical_hash refuses NaN serialisation
    by design, preflight_io allow_nan=False)."""
    digest = _verify_sidecar_file(path)
    facts = json.loads(path.read_text())
    if facts.get("schema") != FACTS_SCHEMA:
        _fail("FACTS_RECORD_INVALID",
              f"facts schema {facts.get('schema')!r} != "
              f"{FACTS_SCHEMA!r}")
    try:
        v04e.scan_for_verdict(facts)
    except v04e.V04EError as exc:
        _fail(exc.code, f"facts integrity: {exc}")
    extra = sorted(set(facts) - _FACTS_TOP_LEVEL_KEYS)
    if extra:
        _fail("FACTS_RECORD_INVALID",
              f"unexpected top-level facts keys {extra}; the schema "
              f"is closed -- a pre-made verdict under any name is "
              f"refused")
    identities, manifest_sha = _load_holdout_identities(manifest_dir)
    if "parent_references" not in facts:
        _fail("FACTS_RECORD_INVALID", "parent_references missing")
    try:
        v04e.validate_facts(facts, identities)
        v04e.validate_parent_references(facts["parent_references"])
    except v04e.V04EError as exc:
        _fail(exc.code, f"facts integrity: {exc}")
    if _semantic_sha(facts) != facts.get("semantic_sha256"):
        _fail("FACTS_RECORD_INVALID",
              "facts semantic_sha256 != the recomputed content hash; "
              "the frozen artifact drifted")
    per_arm = facts.get("per_arm")
    if not isinstance(per_arm, list) \
            or sorted(int(r.get("arm", -1)) for r in per_arm) \
            != ARMS_COVERAGE:
        _fail("FACTS_RECORD_INVALID",
              f"per_arm must cover arms {ARMS_COVERAGE} exactly")
    rows = {}
    for r in per_arm:
        arm = int(r["arm"])
        status = r.get("ensemble_status")
        if status not in ("complete", "incomplete"):
            _fail("FACTS_RECORD_INVALID",
                  f"arm {arm}: ensemble_status {status!r} is not "
                  f"complete|incomplete")
        stab = r.get("stability")
        if not isinstance(stab, dict) or not all(
                isinstance(stab.get(k), int)
                and not isinstance(stab.get(k), bool)
                for k in ("seeds_completed", "seeds_numerical_failure",
                          "seeds_failed_validity")):
            _fail("FACTS_RECORD_INVALID",
                  f"arm {arm}: stability counts missing or malformed")
        nc = stab["seeds_completed"]
        nf = stab["seeds_numerical_failure"]
        nfv = stab["seeds_failed_validity"]
        if not all(0 <= v <= N_SEEDS for v in (nc, nf, nfv)):
            _fail("FACTS_RECORD_INVALID",
                  f"arm {arm}: stability count outside "
                  f"[0, {N_SEEDS}]")
        if nc + nf + nfv != N_SEEDS:
            _fail("FACTS_RECORD_INVALID",
                  f"arm {arm}: stability counts sum to "
                  f"{nc + nf + nfv}, not {N_SEEDS}")
        if status == "complete" and nc != N_SEEDS:
            _fail("FACTS_RECORD_INVALID",
                  f"arm {arm}: ensemble_status complete but "
                  f"seeds_completed {nc} != {N_SEEDS}")
        if status == "incomplete" and nc >= N_SEEDS:
            _fail("FACTS_RECORD_INVALID",
                  f"arm {arm}: ensemble_status incomplete but "
                  f"seeds_completed {nc} == {N_SEEDS}")
        rows[arm] = r
    validate_classification_inputs(rows)
    return rows, digest, manifest_sha


# ---------------------------------------------------------------------------
# The inherited SS14/SS15 matrix (pure, total, mechanical)
# ---------------------------------------------------------------------------

def classify_stability(row: dict) -> bool:
    """176 SS14 S_inst: numerical failure in >=2 of the 3 locked
    seeds, from the recorded stability counts alone."""
    return row["stability"]["seeds_numerical_failure"] \
        >= SEED_FAILURE_D


def route_arm(row: dict) -> dict:
    """One arm through the locked 163 SS15 matrix (classes renamed per
    176 SS15). D is checked first (S_inst from stability counts alone,
    or material degradation on a complete ensemble); an incomplete arm
    without S_inst routes to E (never silently aggregated); straddles
    and uncovered cells route to E by construction."""
    arm = int(row["arm"])
    stab = row["stability"]
    s_inst = classify_stability(row)
    detail = {"arm": arm, "lambda": row["lambda"],
              "stability": dict(stab),
              "ensemble_status": row["ensemble_status"],
              "s_inst": s_inst}
    if s_inst:
        detail.update(**{"class": "V04-D"}, rec=None, density=None,
                      reason=">=2 of 3 seeds numerical_failure "
                             "(176 SS14 S_inst)")
        return detail
    if row["ensemble_status"] != "complete":
        detail.update(**{"class": "V04-E"}, rec=None, density=None,
                      reason="incomplete arm (163 SS15: matched "
                             "3-seed ensemble required)")
        return detail
    deltas = row["aggregates"]["delta_vs_lambda0"]
    pm, pci = _finite_pair(deltas["delta_psnr_r0"], "mean",
                           f"arm {arm} delta_psnr_r0")
    rl, rci = _finite_pair(deltas["delta_nll_relative"], "relative",
                           f"arm {arm} delta_nll_relative")
    rec = classify_reconstruction(pm, pci)
    dens = classify_density(rl, rci)
    detail["decision_inputs"] = {
        "delta_psnr_r0": {"mean": pm, "ci95": [pci[0], pci[1]]},
        "delta_nll_relative": {"relative": rl,
                               "ci95": [rci[0], rci[1]]}}
    if rec == "degradation":
        cls, reason = "V04-D", "material reconstruction degradation " \
                               "(176 SS14 D_rec)"
    elif rec == "straddle" or dens == "straddle":
        cls, reason = "V04-E", (f"CI straddle (rec={rec}, "
                                f"density={dens})")
    elif rec == "improvement" and dens == "preserved":
        cls, reason = "V04-A", "aligned improvement"
    elif rec == "improvement" and dens == "degraded":
        cls, reason = "V04-B", "reconstruction/density tradeoff"
    elif rec == "no_change" and dens == "preserved":
        cls, reason = "V04-C", "no reconstruction support"
    else:
        cls, reason = "V04-E", (f"uncovered matrix cell (rec={rec}, "
                                f"density={dens})")
    detail.update(**{"class": cls}, rec=rec, density=dens,
                  reason=reason)
    return detail


def route_overall(per_arm: list) -> dict:
    """The overall V04 class is the locked parity arm's class (176
    SS7/SS15); the anchor arm 0 is never classified and never
    consulted."""
    return next(d for d in per_arm if d["arm"] == PRIMARY_ARM)


# ---------------------------------------------------------------------------
# Routing record writer, run, CLI
# ---------------------------------------------------------------------------

def _write_with_sidecar(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(payload))
    digest = _file_sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n")
    logger.info("[%s] written: %s (sha256 %s...)", __abbr__, path,
                digest[:12])
    return digest


def run(cfg: dict) -> int:
    """Mechanical classification on the frozen facts artifact. cfg
    keys: facts, manifest_dir, out_dir, code_commit, code_branch.
    Deterministic: re-runs on the same frozen inputs are
    bit-identical."""
    facts_path = Path(cfg["facts"])
    rows, facts_sha, manifest_sha = load_verified_facts(
        facts_path, Path(cfg["manifest_dir"]))
    per_arm = [route_arm(rows[a]) for a in CLASS_ARMS]
    overall = route_overall(per_arm)
    record = {
        "schema": ROUTING_SCHEMA,
        "script": f"{__abbr__} v{__version__}",
        "spec": {"article": 176, "status": "FROZEN"},
        "plan": {"article": 178, "status": "FROZEN",
                 "amendments": ["A1"]},
        "inherited_rules": {
            "routing_matrix": "V03SPEC SS15 (article 163, FROZEN), "
                              "classes renamed V04-A..V04-E per "
                              "V04SPEC SS15",
            "thresholds": "V04SPEC SS14: values identical to V03SPEC "
                          "SS14, inherited verbatim for cross-flow "
                          "comparability (173 SS7)",
            "readout_designation": {
                "delta_rec": "delta_psnr_r0 (R0 z=0)",
                "amendment": "A4 (article 167)",
                "sensitivity": "delta_psnr_r1 and the NMSE_u deltas "
                               "are sensitivity evidence, never "
                               "classified"},
            "density_ci_precedence": "a CI touching or containing the "
                                     "+1% boundary is a straddle, "
                                     "overriding the preservation "
                                     "clauses (Amendment A5, article "
                                     "168)"},
        "code": {"commit": cfg["code_commit"],
                 "branch": cfg["code_branch"]},
        "inputs": {"facts": str(facts_path),
                   "facts_sha256": facts_sha,
                   "holdout_manifest": HOLDOUT_MANIFEST,
                   "holdout_manifest_sha256": manifest_sha},
        "thresholds": {
            "delta_rec_improve_db": REC_IMPROVE_DB,
            "delta_rec_degrade_db": REC_DEGRADE_DB,
            "delta_nll_relative_tolerance": NLL_REL_TOLERANCE,
            "ci": "95% bootstrap over the 199 holdout volumes on "
                  "ensemble-mean per-volume values (V02G conventions)",
            "s_inst": "numerical failure in >=2 of 3 seeds"},
        "per_arm": per_arm,
        "overall": {"class": overall["class"],
                    "primary_arm": PRIMARY_ARM,
                    "designation": "locked parity arm lambda_2' "
                                   "(176 SS7/SS15); the lambda=0 "
                                   "anchor arm 0 is never classified"},
        "interpretation": {
            "conclusion": INTERPRETATION_MAP[overall["class"]],
            "matrix": "V04SPEC SS4 interpretation matrix (locked), "
                      "assigned mechanically per SS15",
            "scope": INTERPRETATION_SCOPE},
    }
    semantic = {k: v for k, v in record.items()
                if k not in _SEMANTIC_EXCLUDE}
    record["semantic_sha256"] = _canonical_hash()(semantic)
    record["semantic_scope"] = {
        "included_keys": sorted(semantic.keys()),
        "excluded": list(_SEMANTIC_EXCLUDE),
        "canonicalisation": "preflight_io.canonical_hash"}
    out = Path(cfg["out_dir"])
    _write_with_sidecar(out / "v04_routing_record.json", record)
    print(f"PASS {__abbr__} v{__version__}: overall "
          f"{overall['class']} (locked parity arm lambda_2' = arm "
          f"{PRIMARY_ARM}); v04_routing_record.json + sidecar")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=f"{__abbr__} v{__version__} -- mechanical V04 "
                    f"classification + within-architecture routing "
                    f"(V04SPEC SS14/SS15; inherited V03SPEC SS15 "
                    f"matrix, Amendments A4/A5)")
    ap.add_argument("--mode", required=True, choices=("run",))
    ap.add_argument("--facts")
    ap.add_argument("--manifest-dir")
    ap.add_argument("--out")
    ap.add_argument("--commit")
    ap.add_argument("--branch")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    required = ("facts", "manifest_dir", "out", "commit", "branch")
    missing = [k for k in required if getattr(args, k) is None]
    if missing:
        logger.error("[%s] missing required arguments: %s", __abbr__,
                     ", ".join("--" + m.replace("_", "-")
                               for m in missing))
        return 2
    cfg = {"facts": args.facts, "manifest_dir": args.manifest_dir,
           "out_dir": args.out, "code_commit": args.commit,
           "code_branch": args.branch}
    try:
        return run(cfg)
    except V04CError as exc:
        logger.error("[%s] ERROR: %s", __abbr__, exc)
        return 2
    except Exception:
        logger.exception("[%s] unexpected exception", __abbr__)
        return 2


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# Changelog
# ---------------------------------------------------------------------------
# v0.1 (2026-10-12): initial implementation. Mechanical V04-A..V04-E
#   classification on the frozen seqref-v04-facts/1 artifact: V03SPEC
#   SS15 matrix inherited verbatim (163) with V04 class names, A4 R0
#   readout designation, A5 density-CI precedence, S_inst >=2/3 checked
#   first, the parity arm (lambda_2', arm 1) alone determines the
#   overall class, and the locked 176 SS4 interpretation is assigned
#   mechanically -- strictly within-architecture (no cross-flow claim;
#   that is V04X). Reuses the registered v03_classify predicates and
#   the v04_eval integrity validators; consumes only sidecar-pinned
#   frozen inputs.

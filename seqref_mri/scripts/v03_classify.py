# SEQREF-V03C v0.3 -- scripts.v03_classify
# LIFETIME: KEEP
# =============================================================================
# Purpose: mechanical V03 classification + cross-flow routing (V03SPEC
#          v0.3 SS14/SS15, article 163 FROZEN; V03PLAN v0.3 SS12,
#          article 164 FROZEN; Amendment A4 via the SPEC SS23 route:
#          SS14 D_rec readout designated R0 z=0). Consumes ONLY the
#          frozen seqref-v03-facts/1 artifact: no model, no data, no
#          metric computation, no holdout contact.
# Rules (locked):
#   - Per-arm class for lambda_1/lambda_2/lambda_3 by the SS15 matrix;
#     the overall V03 class is the class of the locked primary arm
#     lambda_2 (SS7, designated pre-training); sensitivity-arm classes
#     are reported alongside and never alter the overall.
#   - D_rec readout: delta_psnr_r0 (R0 z=0) per Amendment A4;
#     delta_psnr_r1 and the NMSE_u deltas are sensitivity evidence,
#     never classified.
#   - SS14 thresholds: material improvement mean >= +0.5 dB with the
#     95% CI excluding 0; material degradation mean <= -0.5 dB with CI
#     excluding 0; density preserved when relative <= +1% or the CI
#     includes 0, degraded when relative > +1% with CI excluding 0;
#     S_inst = numerical failure in >=2 of 3 seeds. Any CI straddling
#     a classification boundary, an incomplete arm, or an uncovered
#     matrix cell routes to V03-E (SS15). Density-CI precedence
#     (Amendment A5): a CI touching or containing the +1% boundary
#     is a straddle, overriding the preservation clauses.
#   - Cross-flow gate (CH2-EXEC SS7; class->gate mapping registered
#     pre-results 2026-10-06): V03-A -> decisive_positive; V03-C/V03-D
#     -> decisive_null; V03-B/V03-E -> inconclusive.
# Output: seqref-v03-routing-record/1 (v03_routing_record.json) +
#   SHA-256 sidecar + semantic hash. Deterministic by construction:
#   re-runs on the same frozen facts are bit-identical (PLAN SS12);
#   the record carries no wall-clock fields and no timestamps.
# CONVENTION: logger.error + typed raise (V03CError). PASS(0)/ERROR(2).
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

from seqref_mri.scripts.v02_manifests import canonical_json

logger = logging.getLogger("seqref_mri.v03_classify")

__version__ = "0.3"
__abbr__ = "SEQREF-V03C"

FACTS_SCHEMA = "seqref-v03-facts/1"
ROUTING_SCHEMA = "seqref-v03-routing-record/1"
PRIMARY_ARM = 2                      # lambda_2, locked pre-training (SS7)
CLASS_ARMS = (1, 2, 3)               # anchor arm 0 is never classified
REC_IMPROVE_DB = 0.5                 # SS14 lock 5
REC_DEGRADE_DB = -0.5                # SS14 lock 5
NLL_REL_TOLERANCE = 0.01             # SS14 lock 5 (+1%)
SEED_FAILURE_D = 2                   # SS14 S_inst: >=2 of 3 seeds
N_SEEDS = 3                        # locked seed set {101, 202, 303} (SS7)
GATE_MAP = {"V03-A": "decisive_positive",
            "V03-B": "inconclusive",
            "V03-C": "decisive_null",
            "V03-D": "decisive_null",
            "V03-E": "inconclusive"}
GATE_EFFECT = {
    "decisive_positive": "Phase D opens: generalisation port to "
                         "RealNVP and NICE (CH2-EXEC SS7)",
    "decisive_null": "Phase D limited to one RealNVP robustness "
                     "port; NICE stays parked (CH2-EXEC SS7)",
    "inconclusive": "no cross-flow work (CH2-EXEC SS7)"}
GATE_MAPPING_TEXT = ("V03-A -> decisive_positive; V03-C/V03-D -> "
                     "decisive_null; V03-B/V03-E -> inconclusive "
                     "(registered pre-results 2026-10-06)")
_SEMANTIC_EXCLUDE = ("run", "semantic_sha256", "semantic_scope")


class V03CError(Exception):
    """Typed classification failure: .code is the machine reason."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise V03CError(code, message)


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
# Facts artifact loading and structural gates
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


def load_facts(path: Path) -> tuple:
    """The frozen facts artifact, pinned three ways: SHA-256 sidecar,
    schema, semantic self-consistency. Structural gates: per_arm must
    cover arms 0..3 with explicit ensemble_status and stability counts;
    every complete classified arm must carry the decision inputs
    (delta_psnr_r0 mean/ci95, delta_nll_relative relative/ci95, all
    finite). Incomplete arms carry no aggregates by V03E v0.2 contract
    -- that is accepted and routes mechanically, never coerced."""
    digest = _verify_sidecar_file(path)
    facts = json.loads(path.read_text())
    if facts.get("schema") != FACTS_SCHEMA:
        _fail("FACTS_RECORD_INVALID",
              f"facts schema {facts.get('schema')!r} != {FACTS_SCHEMA!r}")
    if _semantic_sha(facts) != facts.get("semantic_sha256"):
        _fail("FACTS_RECORD_INVALID",
              "facts semantic_sha256 != the recomputed content hash; "
              "the frozen artifact drifted")
    per_arm = facts.get("per_arm")
    if not isinstance(per_arm, list) \
            or sorted(int(r.get("arm", -1)) for r in per_arm) \
            != [0, 1, 2, 3]:
        _fail("FACTS_RECORD_INVALID",
              "per_arm must cover arms 0, 1, 2, 3 exactly")
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
        if status == "complete":
            agg = r.get("aggregates")
            if not isinstance(agg, dict) or "means_ci" not in agg:
                _fail("FACTS_RECORD_INVALID",
                      f"complete arm {arm}: aggregates.means_ci missing")
            if arm != 0:
                deltas = agg.get("delta_vs_lambda0")
                if not isinstance(deltas, dict):
                    _fail("FACTS_RECORD_INVALID",
                          f"complete arm {arm}: delta_vs_lambda0 "
                          f"missing; the anchor ensemble is required "
                          f"for classification")
                _finite_pair(deltas.get("delta_psnr_r0") or {},
                             "mean", f"arm {arm} delta_psnr_r0")
                _finite_pair(deltas.get("delta_nll_relative") or {},
                             "relative",
                             f"arm {arm} delta_nll_relative")
        rows[arm] = r
    return rows, digest


# ---------------------------------------------------------------------------
# The SS14/SS15 matrix (pure, total, mechanical)
# ---------------------------------------------------------------------------

def classify_reconstruction(mean: float, ci: tuple) -> str:
    """SS14 D_rec on the A4-designated R0 z=0 readout. A CI containing
    a +/-0.5 dB cell boundary is a straddle; improvement/degradation
    additionally require the CI to exclude 0."""
    lo, hi = ci
    if lo <= REC_IMPROVE_DB <= hi or lo <= REC_DEGRADE_DB <= hi:
        return "straddle"
    if mean >= REC_IMPROVE_DB:
        return "improvement" if lo > 0.0 else "straddle"
    if mean <= REC_DEGRADE_DB:
        return "degradation" if hi < 0.0 else "straddle"
    return "no_change"


def classify_density(relative: float, ci: tuple) -> str:
    """SS14 D_nll. CI containing the +1% boundary -> straddle (the
    generic CI rule dominates); CI entirely above +1% -> degraded
    (relative > +1% with CI excluding 0 follows); otherwise preserved
    (relative <= +1%, or the CI includes 0)."""
    lo, hi = ci
    if lo <= NLL_REL_TOLERANCE <= hi:
        return "straddle"
    if lo > NLL_REL_TOLERANCE:
        return "degraded"
    return "preserved"


def classify_arm(row: dict) -> dict:
    """One arm through the locked SS15 matrix. D is checked first
    (>=2/3 numerical failures from stability counts alone, or material
    degradation on a complete ensemble); an incomplete arm without
    S_inst routes to E (never silently aggregated); straddles and
    uncovered cells route to E by construction."""
    arm = int(row["arm"])
    stab = row["stability"]
    s_inst = stab["seeds_numerical_failure"] >= SEED_FAILURE_D
    detail = {"arm": arm, "lambda": row["lambda"],
              "stability": dict(stab),
              "ensemble_status": row["ensemble_status"],
              "s_inst": s_inst}
    if s_inst:
        detail.update(**{"class": "V03-D"}, rec=None, density=None,
                      reason=">=2 of 3 seeds numerical_failure "
                             "(SS14 S_inst)")
        return detail
    if row["ensemble_status"] != "complete":
        detail.update(**{"class": "V03-E"}, rec=None, density=None,
                      reason="incomplete arm (SS15: matched 3-seed "
                             "ensemble required)")
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
        cls, reason = "V03-D", "material reconstruction degradation " \
                               "(SS14 D_rec)"
    elif rec == "straddle" or dens == "straddle":
        cls, reason = "V03-E", (f"CI straddle (rec={rec}, "
                                f"density={dens})")
    elif rec == "improvement" and dens == "preserved":
        cls, reason = "V03-A", "aligned improvement"
    elif rec == "improvement" and dens == "degraded":
        cls, reason = "V03-B", "reconstruction/density tradeoff"
    elif rec == "no_change" and dens == "preserved":
        cls, reason = "V03-C", "no reconstruction support"
    else:
        cls, reason = "V03-E", (f"uncovered matrix cell (rec={rec}, "
                                f"density={dens})")
    detail.update(**{"class": cls}, rec=rec, density=dens,
                  reason=reason)
    return detail


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
    keys: facts, out_dir, code_commit, code_branch. Deterministic:
    re-runs on the same facts are bit-identical (PLAN SS12)."""
    facts_path = Path(cfg["facts"])
    rows, facts_sha = load_facts(facts_path)
    per_arm = [classify_arm(rows[a]) for a in CLASS_ARMS]
    overall = next(d for d in per_arm if d["arm"] == PRIMARY_ARM)
    gate = GATE_MAP[overall["class"]]
    record = {
        "schema": ROUTING_SCHEMA,
        "script": f"{__abbr__} v{__version__}",
        "spec": {"article": 163, "status": "FROZEN",
                 "amendments": ["A3", "A4", "A5"]},
        "plan": {"article": 164, "status": "FROZEN",
                 "amendments": ["A1", "A2", "A3"]},
        "code": {"commit": cfg["code_commit"],
                 "branch": cfg["code_branch"]},
        "inputs": {"facts": str(facts_path),
                   "facts_sha256": facts_sha},
        "readout_designation": {
            "delta_rec": "delta_psnr_r0 (R0 z=0)",
            "amendment": "A4",
            "sensitivity": "delta_psnr_r1 and the NMSE_u deltas are "
                           "evidence only, never classified"},
        "thresholds": {
            "delta_rec_improve_db": REC_IMPROVE_DB,
            "delta_rec_degrade_db": REC_DEGRADE_DB,
            "delta_nll_relative_tolerance": NLL_REL_TOLERANCE,
            "ci": "95% bootstrap over the 199 holdout volumes on "
                  "ensemble-mean per-volume values (V02G conventions)",
            "s_inst": "numerical failure in >=2 of 3 seeds",
            "density_ci_precedence": "a CI touching or containing the "
                                     "+1% boundary is a straddle, "
                                     "overriding the preservation "
                                     "clauses (Amendment A5)"},
        "per_arm": per_arm,
        "overall": {"class": overall["class"],
                    "primary_arm": PRIMARY_ARM,
                    "designation": "locked pre-training primary arm "
                                   "lambda_2 (SPEC SS7/SS15)"},
        "cross_flow_gate": {"gate": gate,
                            "effect": GATE_EFFECT[gate],
                            "mapping": GATE_MAPPING_TEXT,
                            "source": "CH2-EXEC SS7 (article 162)"},
    }
    semantic = {k: v for k, v in record.items()
                if k not in _SEMANTIC_EXCLUDE}
    record["semantic_sha256"] = _canonical_hash()(semantic)
    record["semantic_scope"] = {
        "included_keys": sorted(semantic.keys()),
        "excluded": list(_SEMANTIC_EXCLUDE),
        "canonicalisation": "preflight_io.canonical_hash"}
    out = Path(cfg["out_dir"])
    _write_with_sidecar(out / "v03_routing_record.json", record)
    print(f"PASS {__abbr__} v{__version__}: overall "
          f"{overall['class']} (locked primary arm lambda_2 = arm "
          f"{PRIMARY_ARM}); cross-flow gate {gate}; "
          f"v03_routing_record.json + sidecar")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=f"{__abbr__} v{__version__} -- mechanical V03 "
                    f"classification + routing (V03SPEC SS14/SS15, "
                    f"Amendment A4)")
    ap.add_argument("--mode", required=True,
                    choices=("selftest", "run"))
    ap.add_argument("--facts")
    ap.add_argument("--out")
    ap.add_argument("--commit")
    ap.add_argument("--branch")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.mode == "selftest":
        return _selftest()
    required = ("facts", "out", "commit", "branch")
    missing = [k for k in required if getattr(args, k) is None]
    if missing:
        logger.error("[%s] missing required arguments: %s", __abbr__,
                     ", ".join("--" + m.replace("_", "-")
                               for m in missing))
        return 2
    cfg = {"facts": args.facts, "out_dir": args.out,
           "code_commit": args.commit, "code_branch": args.branch}
    try:
        return run(cfg)
    except V03CError as exc:
        logger.error("[%s] ERROR: %s", __abbr__, exc)
        return 2
    except Exception:
        logger.exception("[%s] unexpected exception", __abbr__)
        return 2


# ---------------------------------------------------------------------------
# Selftest (pure fixtures -- no model, no data, no metrics)
# ---------------------------------------------------------------------------

def _expect_fail(fn, code: str, what: str) -> None:
    try:
        fn()
    except V03CError as exc:
        if exc.code != code:
            _fail("CLASSIFY_SELFTEST_WRONG_CODE",
                  f"{what}: expected {code}, got {exc.code}")
        return
    _fail("CLASSIFY_SELFTEST_NO_FAILURE",
          f"{what}: expected {code}, no failure raised")


def _arm_row(arm: int, status: str = "complete", nf: int = 0,
             nfv: int = 0, nc: int | None = None,
             psnr=(0.8, (0.55, 1.05)),
             nll=(0.005, (-0.01, 0.008))) -> dict:
    """Synthetic per-arm facts row. nf/nfv count failed seeds; the
    completed count defaults to the remainder of the 3 locked seeds."""
    if nc is None:
        nc = 3 - nf - nfv
    row = {"arm": arm, "lambda": {0: 0.0, 1: 0.1, 2: 0.5,
                                  3: 0.9}[arm],
           "ensemble_status": status,
           "stability": {"seeds_completed": nc,
                         "seeds_numerical_failure": nf,
                         "seeds_failed_validity": nfv}}
    if status == "complete":
        row["aggregates"] = {
            "means_ci": {"nll": {"mean": 1.0, "ci95": [0.9, 1.1]}},
            "delta_vs_lambda0": None if arm == 0 else {
                "delta_psnr_r0": {"mean": psnr[0],
                                  "ci95": [psnr[1][0], psnr[1][1]]},
                "delta_nll_relative": {
                    "relative": nll[0],
                    "ci95": [nll[1][0], nll[1][1]]}}}
    return row


def _selftest() -> int:
    import tempfile as _tf

    tmp = Path(_tf.mkdtemp(prefix="v03c_selftest_"))

    # f11: SS14 threshold functions -- every branch.
    if classify_reconstruction(0.8, (0.55, 1.05)) != "improvement" \
            or classify_reconstruction(0.6, (0.51, 0.7)) \
            != "improvement":
        _fail("CLASSIFY_SELFTEST_WRONG_VALUE", "f11 rec improvement")
    if classify_reconstruction(-0.8, (-1.05, -0.55)) != "degradation":
        _fail("CLASSIFY_SELFTEST_WRONG_VALUE", "f11 rec degradation")
    if classify_reconstruction(0.2, (-0.1, 0.45)) != "no_change":
        _fail("CLASSIFY_SELFTEST_WRONG_VALUE", "f11 rec no_change")
    for m, ci in ((0.8, (0.4, 1.2)), (0.2, (-0.6, 0.9)),
                  (0.8, (-0.1, 1.7)), (-0.6, (-0.7, 0.1)),
                  (0.5, (0.0, 0.9)), (-0.5, (-0.9, 0.2))):
        if classify_reconstruction(m, ci) != "straddle":
            _fail("CLASSIFY_SELFTEST_WRONG_VALUE",
                  f"f11 rec straddle {(m, ci)}")
    if classify_density(0.005, (-0.01, 0.008)) != "preserved" \
            or classify_density(-0.02, (-0.03, -0.01)) != "preserved":
        _fail("CLASSIFY_SELFTEST_WRONG_VALUE", "f11 density preserved")
    if classify_density(0.02, (0.015, 0.025)) != "degraded":
        _fail("CLASSIFY_SELFTEST_WRONG_VALUE", "f11 density degraded")
    for rel, ci in ((0.02, (0.005, 0.03)), (0.005, (-0.02, 0.03)),
                    (-0.005, (-0.02, 0.01))):
        if classify_density(rel, ci) != "straddle":
            _fail("CLASSIFY_SELFTEST_WRONG_VALUE",
                  f"f11 density straddle {(rel, ci)}")

    # f11: SS15 matrix rows + precedence.
    cases = [
        ("V03-A", _arm_row(1)),
        ("V03-B", _arm_row(1, nll=(0.02, (0.015, 0.025)))),
        ("V03-C", _arm_row(1, psnr=(0.2, (-0.1, 0.45)))),
        ("V03-D", _arm_row(1, psnr=(-0.8, (-1.05, -0.55)))),
        ("V03-D", _arm_row(1, status="incomplete", nf=2)),
        ("V03-D", _arm_row(1, status="incomplete", nf=2, nfv=1,
                           nc=0)),
        ("V03-E", _arm_row(1, status="incomplete", nf=1)),
        ("V03-E", _arm_row(1, status="incomplete", nfv=2)),
        ("V03-E", _arm_row(1, psnr=(0.8, (0.4, 1.2)))),
        ("V03-E", _arm_row(1, nll=(0.02, (0.005, 0.03)))),
        ("V03-E", _arm_row(1, psnr=(0.2, (-0.1, 0.45)),
                           nll=(0.02, (0.015, 0.025)))),
    ]
    for want, row in cases:
        got = classify_arm(row)
        if got["class"] != want:
            _fail("CLASSIFY_SELFTEST_WRONG_VALUE",
                  f"f11 matrix: expected {want}, got {got['class']} "
                  f"({got['reason']})")
    logger.info("[%s] f11 PASS: SS14 thresholds + SS15 matrix rows",
                __abbr__)

    # Facts loading: happy path + every refusal branch.
    def _facts_file(name: str, mutate=None, semantic_ok: bool = True,
                    sidecar_ok: bool = True) -> Path:
        facts = {"schema": FACTS_SCHEMA,
                 "spec": {"article": 163, "status": "FROZEN"},
                 "per_arm": [_arm_row(0),
                             _arm_row(1),
                             _arm_row(2, psnr=(0.2, (-0.1, 0.45))),
                             _arm_row(3, status="incomplete", nf=1)],
                 "environment": {"python": "x"}}
        if mutate is not None:
            mutate(facts)
        payload = {k: v for k, v in facts.items()
                   if k not in _SEMANTIC_EXCLUDE}
        facts["semantic_sha256"] = (
            _canonical_hash()(payload) if semantic_ok else "0" * 64)
        p = tmp / name
        p.write_bytes(canonical_json(facts))
        digest = _file_sha256(p) if sidecar_ok else "0" * 64
        p.with_suffix(".json.sha256").write_text(
            f"{digest}  {p.name}\n")
        return p

    good = _facts_file("facts_ok.json")
    rows, digest = load_facts(good)
    if set(rows) != {0, 1, 2, 3} or digest != _file_sha256(good):
        _fail("CLASSIFY_SELFTEST_WRONG_VALUE", "facts happy path")
    _expect_fail(lambda: load_facts(
        _facts_file("facts_schema.json",
                    lambda f: f.update(schema="wrong/9"))),
        "FACTS_RECORD_INVALID", "facts wrong schema")
    _expect_fail(lambda: load_facts(
        _facts_file("facts_semantic.json", semantic_ok=False)),
        "FACTS_RECORD_INVALID", "facts semantic drift")
    _expect_fail(lambda: load_facts(
        _facts_file("facts_sidecar.json", sidecar_ok=False)),
        "CLASSIFY_INPUT_DRIFT", "facts sidecar drift")
    _expect_fail(lambda: load_facts(tmp / "absent.json"),
                 "CLASSIFY_INPUT_MISSING", "facts missing")

    def _drop_deltas(f: dict) -> None:
        f["per_arm"][1]["aggregates"]["delta_vs_lambda0"] = None

    _expect_fail(lambda: load_facts(
        _facts_file("facts_nodeltas.json", _drop_deltas)),
        "FACTS_RECORD_INVALID", "complete arm missing deltas")

    def _bad_status(f: dict) -> None:
        f["per_arm"][2]["ensemble_status"] = "partial"

    _expect_fail(lambda: load_facts(
        _facts_file("facts_status.json", _bad_status)),
        "FACTS_RECORD_INVALID", "unknown ensemble_status")

    def _drop_arm(f: dict) -> None:
        f["per_arm"] = f["per_arm"][:3]

    _expect_fail(lambda: load_facts(
        _facts_file("facts_arms.json", _drop_arm)),
        "FACTS_RECORD_INVALID", "arm coverage")

    # Stability-count consistency gates (input hardening): range,
    # sum, agreement with ensemble_status.
    def _stab_range(f: dict) -> None:
        f["per_arm"][1]["stability"]["seeds_completed"] = 4

    _expect_fail(lambda: load_facts(
        _facts_file("facts_stab_range.json", _stab_range)),
        "FACTS_RECORD_INVALID", "stability count > N_SEEDS")

    def _stab_sum(f: dict) -> None:
        f["per_arm"][3]["stability"] = {
            "seeds_completed": 1, "seeds_numerical_failure": 1,
            "seeds_failed_validity": 0}

    _expect_fail(lambda: load_facts(
        _facts_file("facts_stab_sum.json", _stab_sum)),
        "FACTS_RECORD_INVALID", "stability counts sum != N_SEEDS")

    def _stab_complete(f: dict) -> None:
        f["per_arm"][1]["stability"].update(
            seeds_completed=2, seeds_numerical_failure=1)

    _expect_fail(lambda: load_facts(
        _facts_file("facts_stab_comp.json", _stab_complete)),
        "FACTS_RECORD_INVALID", "complete status with <3 completed")

    def _stab_incomplete(f: dict) -> None:
        f["per_arm"][3]["stability"].update(
            seeds_completed=3, seeds_numerical_failure=0)

    _expect_fail(lambda: load_facts(
        _facts_file("facts_stab_inc.json", _stab_incomplete)),
        "FACTS_RECORD_INVALID", "incomplete status with 3 completed")

    # End-to-end: run twice on the same frozen facts -> bit-identical
    # routing records (PLAN SS12); overall is the locked primary arm
    # lambda_2 (arm 2 -> V03-C here), never the sensitivity arms.
    cfg = {"facts": str(good), "code_commit": "c", "code_branch": "b"}
    cfg1 = dict(cfg, out_dir=str(tmp / "o1"))
    cfg2 = dict(cfg, out_dir=str(tmp / "o2"))
    if run(cfg1) != 0 or run(cfg2) != 0:
        _fail("CLASSIFY_SELFTEST_WRONG_VALUE", "run exit status")
    b1 = (tmp / "o1" / "v03_routing_record.json").read_bytes()
    b2 = (tmp / "o2" / "v03_routing_record.json").read_bytes()
    if b1 != b2:
        _fail("CLASSIFY_SELFTEST_WRONG_VALUE",
              "re-run is not bit-identical")
    rec = json.loads(b1.decode())
    classes = {d["arm"]: d["class"] for d in rec["per_arm"]}
    if classes != {1: "V03-A", 2: "V03-C", 3: "V03-E"} \
            or rec["overall"]["class"] != "V03-C" \
            or rec["overall"]["primary_arm"] != PRIMARY_ARM \
            or rec["cross_flow_gate"]["gate"] != "decisive_null" \
            or rec["schema"] != ROUTING_SCHEMA:
        _fail("CLASSIFY_SELFTEST_WRONG_VALUE",
              f"routing record content {classes}")
    if _semantic_sha(rec) != rec["semantic_sha256"]:
        _fail("CLASSIFY_SELFTEST_WRONG_VALUE",
              "routing record semantic mismatch")
    logger.info("[%s] end-to-end PASS: overall=primary arm, bit-"
                "identical re-run", __abbr__)

    # f10: static audit over this module.
    from seqref_mri.scripts import v03_train as v03t
    v03t.static_audit([Path(__file__)])
    logger.info("[%s] f10 PASS: static audit clean", __abbr__)

    print(f"PASS {__abbr__} v{__version__} selftest (f11, f10)")
    return 0


# ---------------------------------------------------------------------------
# Changelog
# v0.2 (2026-10-06): review hardening. Stability-count consistency
#   gates in load_facts (counts in [0, 3], sum to 3, agreement with
#   ensemble_status) -- input validation only, no scientific rule.
#   Density-CI precedence now cited as Amendment A5 (article 168);
#   classification logic unchanged. Selftest: four new refusal
#   branches.
# v0.1 (2026-10-06): initial implementation under V03SPEC v0.3 (163,
#   FROZEN) SS14/SS15 + Amendment A4 (D_rec readout designated R0 z=0)
#   and V03PLAN v0.3 (164, FROZEN) SS12. Pure mechanical classification
#   on the frozen seqref-v03-facts/1 artifact: sidecar + schema +
#   semantic verification, structural gates (arm coverage, explicit
#   ensemble_status, finite decision inputs on complete arms), the
#   locked matrix per arm lambda_1/lambda_2/lambda_3, overall = locked
#   primary arm lambda_2, cross-flow gate per CH2-EXEC SS7 with the
#   pre-results mapping A->positive, C/D->null, B/E->inconclusive.
#   seqref-v03-routing-record/1 + SHA-256 sidecar + semantic hash;
#   bit-identical re-runs; no model, no data, no metric computation.
#   Selftest: f11 (threshold branches, all matrix rows, precedence,
#   loading refusals, end-to-end determinism) + f10 static audit.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())

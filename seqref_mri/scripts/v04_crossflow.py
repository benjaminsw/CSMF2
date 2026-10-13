# SEQREF-MRI-V04X v0.1 -- Cross-flow comparison (NSF vs RealNVP)
# LIFETIME: KEEP
# STATUS: fixture-verified (f17 registered + supplementary X01-X13
#         synthetic smoke); authoritative preflight/audit PENDING (the
#         frozen V03/V04 facts artifacts and a registered V04R do not
#         exist in this environment)
"""SEQREF-MRI-V04X -- cross-flow readiness + difference-of-improvements.

Frozen contract (verified verbatim before implementation):

- 173 (CROSSFLOW, FROZEN) SS5 equivalence matrix; SS6 notation and
  pairing (Delta_a = M_{a,rec} - M_{a,0}; Delta_{a,b} = Delta_a -
  Delta_b; lower-is-better metrics sign-reversed so positive = helped;
  pairing at MRI-volume level; matching seed numbers is NOT pairing;
  an inconclusive outcome is a legitimate terminal state); SS7 NLL
  comparability contract; SS8 deviations register; SS8.1 six-item
  readiness gate (failure of any item -> the comparison is not
  registered, exploratory at most); SS10 activation rules (a V03
  decisive null activates exactly one RealNVP robustness port; NICE
  stays parked).
- 176 (V04SPEC, FROZEN) SS13 locked DeltaDelta: per holdout volume v,
  Delta_V03(v) = V03 parity-arm ensemble-mean PSNR R0(v) - V03 anchor
  ensemble-mean PSNR R0(v) from the frozen V03 facts artifact;
  Delta_V04(v) likewise; DeltaDelta(v) = Delta_V04(v) - Delta_V03(v);
  the mean over the 199 volumes with a 95% bootstrap CI (V02G: ONE
  PCG64(3), B=10000, percentile, paired volume-index draws); for NLL
  the sign is reversed (higher-is-better); seed averaging happens
  WITHIN each architecture first -- seeds never pair across
  architectures; the uncertainty-scope limitation is restated in the
  record and must be restated in V04R.
- 178 (V04PLAN, FROZEN) SS12a two-phase lifecycle: (1) preflight
  (non-authoritative, after V04E): V03 facts sha256 pin-verify,
  evaluation-manifest identity match, case-set alignment -- explicitly
  NOT SS8.1 admissibility; (2) authoritative readiness audit after
  V04R is registered: the six SS8.1 items; any failure ->
  CROSSFLOW_GATE_FAILURE, exploratory at most, the within-architecture
  V04 result unaffected; a comparability failure discovered at this
  post-evaluation stage follows the amendment/reporting path, not the
  pre-evaluation abort terminology; (3) DeltaDelta only if the audit
  passes, appended to V04R via the registered update process. 178 SS5
  f17 freezes the readiness-refusal fixture.

Two phase-distinct records (implementation recommendation of the
registered plan review): seqref-v04-crossflow-preflight/1 with an
explicit non_authoritative status, and seqref-v04-crossflow-record/1.

Governance attestation (implementation-defined input contract): the
SS8.1 items that artifacts cannot prove (activation under SS10,
registered RESULT articles, pre-results registration timing, zero
holdout tuning) are attested by an operator-supplied chain-state
document (seqref-v04-crossflow-chain-state/1, sidecar-pinned). Every
artifact-checkable item (seeds, bootstrap conventions, data_range,
readouts, metric keys, case alignment, ensemble completeness) is
verified from the pin-verified facts themselves -- the attestation
never overrides artifact evidence.

No fallback, no mock, no placeholder, no silent pass: every failure
path logs an error and exits 2 with a typed code; a failed readiness
item is recorded FAIL and yields CROSSFLOW_GATE_FAILURE with no
confirmatory DeltaDelta in the record -- never a registered claim.
Determinism: re-runs on the same frozen inputs are bit-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

from seqref_mri.scripts import v02_eval as v02e
from seqref_mri.scripts import v02_manifests as v02m
from seqref_mri.scripts import v04_classify as v04c
from seqref_mri.scripts import v04_eval as v04e
from seqref_mri.scripts import v04_train as v04t
from seqref_mri.scripts.v02_manifests import canonical_json

logger = logging.getLogger("seqref_mri.v04_crossflow")

__version__ = "0.1"
__abbr__ = "SEQREF-MRI-V04X"

PREFLIGHT_SCHEMA = "seqref-v04-crossflow-preflight/1"
RECORD_SCHEMA = "seqref-v04-crossflow-record/1"
CHAIN_STATE_SCHEMA = "seqref-v04-crossflow-chain-state/1"
PREFLIGHT_RECORD = "v04_crossflow_preflight.json"
AUDIT_RECORD = "v04_crossflow_record.json"
HOLDOUT_MANIFEST = v04c.HOLDOUT_MANIFEST
N_VOLUMES = v02m.N_HOLDOUT_VOLUMES

V03_ANCHOR_ARM = 0
V03_PARITY_ARM = 2                 # V03SPEC SS7: lambda_2 locked primary
V04_ANCHOR_ARM = 0
V04_PARITY_ARM = v04c.PRIMARY_ARM  # 1: lambda_2' parity arm (176 SS7)
SEEDS = v04t.SEEDS                 # locked set {101, 202, 303} (173 SS5)
METRIC_KEYS = v04e.METRIC_KEYS

GATE_FAILURE = "CROSSFLOW_GATE_FAILURE"
PREFLIGHT_FAILURE = "CROSSFLOW_PREFLIGHT_FAILED"

LIKELIHOOD_CONTRACT = ("per-volume mean of per-slice NLL in nats on "
                       "identically scaled data (173 \u00a77)")
NLL_CONVENTION = ("higher-is-better (173 \u00a76 / 176 \u00a713): "
                  "delta_NLL(v) = anchor ensemble NLL(v) - arm "
                  "ensemble NLL(v); a positive value means the arm "
                  "helped")
PAIRING_RULE = ("MRI-volume level in manifest order; seed averaging "
                "within each architecture first -- seeds never pair "
                "across architectures (176 \u00a713)")
UNCERTAINTY_SCOPE = ("Uncertainty scope (registered limitation, 176 "
                     "\u00a713, restated): the paired bootstrap over "
                     "the 199 volumes measures evaluation-volume "
                     "variability conditional on the fitted three-seed "
                     "ensembles; it does not capture uncertainty over "
                     "possible training initialisations beyond the "
                     "three seeds")
INTERPRETATION_NOTE = ("interpretation per 176 \u00a715 / 178 "
                       "\u00a712a: V04-C -> the null is robust across "
                       "the two tested architectures; V04-A with the "
                       "\u0394\u0394 CI excluding 0 -> an "
                       "architecture-dependent effect; the admitted "
                       "result is appended to V04R via the registered "
                       "update process, never by silent edit")

READINESS_GATE = "173 \u00a78.1"
READINESS_ITEMS = (
    ("activation_rules",
     "The architectures were activated under the registered Chapter 2 "
     "rules (\u00a710)."),
    ("evidence_chains",
     "Each architecture has its own completed, valid experimental "
     "evidence chain (SPEC \u2192 PLAN \u2192 EXEC \u2192 RESULT)."),
    ("shared_settings",
     "The mandatory shared settings (\u00a75) match across the "
     "compared experiments."),
    ("deviations_registered",
     "All architecture-specific deviations were registered (\u00a78) "
     "before results were seen."),
    ("statistics_frozen",
     "The statistical comparison procedures and primary metrics were "
     "frozen pre-results."),
    ("no_holdout_tuning",
     "No held-out data were used for tuning at any point."),
)
# Registered article identities of the two evidence chains (governance
# constants; RESULT article ids are intentionally not pinned -- V04R
# does not exist at implementation time).
EXPECTED_CHAINS = {"v03": {"spec": 163, "plan": 164, "exec": 169},
                   "v04": {"spec": 176, "plan": 178, "exec": 179}}
ACTIVATION_EXPECTED = {"v03_overall": "decisive_null",
                       "activated_port": "realnvp",
                       "nice_status": "parked"}      # 173 SS10
_CHAIN_STATE_KEYS = frozenset({
    "schema", "activation", "chains", "deviations", "statistics",
    "holdout_tuning", "shared"})
_SHARED_KEYS = frozenset({
    "data_range", "likelihood", "holdout_manifest_sha256"})
_SEMANTIC_EXCLUDE = ("run", "semantic_sha256", "semantic_scope")


class V04XError(Exception):
    """Typed cross-flow failure: .code is the machine-readable reason."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise V04XError(code, message)


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


def _semantic_sha(payload: dict) -> str:
    return _canonical_hash()(
        {k: v for k, v in payload.items() if k not in _SEMANTIC_EXCLUDE})


def _verify_sidecar_file(path: Path) -> str:
    side = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not side.is_file():
        _fail("CROSSFLOW_INPUT_MISSING",
              f"frozen input or sidecar missing at {path}")
    recorded = side.read_text().split()[0]
    digest = _file_sha256(path)
    if digest != recorded:
        _fail("CROSSFLOW_INPUT_DRIFT",
              f"{path.name}: sha256 != the recorded sidecar digest")
    return digest


def _write_with_sidecar(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(payload))
    digest = _file_sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n")
    logger.info("[%s] written: %s (sha256 %s...)", __abbr__, path,
                digest[:12])
    return digest


def _load_holdout_identities(manifest_dir: Path) -> tuple:
    """The pinned 199-volume holdout manifest (178 SS12a): schema,
    kind, sidecar and internal hash are verified by the registered V02
    loader (its typed V02Error propagates unchanged); the locked volume
    count is gated here, identical to V04E/V04C."""
    holdman = v02e._load_manifest(manifest_dir, HOLDOUT_MANIFEST,
                                  "holdout")
    if int(holdman["n_volumes"]) != N_VOLUMES \
            or len(holdman["entries"]) != N_VOLUMES:
        _fail("MANIFEST_CONTENT_MISMATCH",
              "holdout manifest diverges from the locked "
              f"{N_VOLUMES} volumes")
    identities = [{"file": e["file"],
                   "slice_index": int(e["slice_index"])}
                  for e in holdman["entries"]]
    return identities, _file_sha256(manifest_dir / HOLDOUT_MANIFEST)


# ---------------------------------------------------------------------------
# Frozen-input loading (integrity failures are hard typed aborts)
# ---------------------------------------------------------------------------

def load_verified_v03_facts(path, *,
                            expected_sha: str = v04e.V03_FACTS_SHA256
                            ) -> tuple:
    """178 SS12a preflight step 1: the frozen V03 facts artifact,
    pin-verified at load by the registered V04E loader (sidecar, sha
    pin, schema, recorded V02G CI settings). Typed V04E codes
    propagate with the same code as V04XError."""
    p = Path(path)
    try:
        facts = v04e.load_v03_facts_reference(str(p),
                                              expected_sha=expected_sha)
    except v04e.V04EError as exc:
        _fail(exc.code, f"V03 facts: {exc}")
    return facts, _file_sha256(p)


def load_verified_v04_facts(path, manifest_dir) -> tuple:
    """The frozen V04 facts artifact through the registered V04C
    battery (sidecar, schema, verdict scan, closed key allowlist,
    manifest identity, parent references, finite scan, semantic hash,
    arm coverage, stability gates, classification inputs). Typed V04C
    codes propagate with the same code as V04XError; the V02 loader's
    typed V02Error propagates unchanged."""
    try:
        rows, digest, _msha = v04c.load_verified_facts(
            Path(path), Path(manifest_dir))
    except v04c.V04CError as exc:
        _fail(exc.code, f"V04 facts: {exc}")
    full = json.loads(Path(path).read_text())
    return rows, full, digest


def load_chain_state(path) -> tuple:
    """The operator-supplied governance attestation
    (seqref-v04-crossflow-chain-state/1): sidecar-pinned, schema-gated,
    closed key set -- an attestation smuggled under an innocent key is
    refused exactly like a pre-made verdict in the facts artifact."""
    p = Path(path)
    _verify_sidecar_file(p)
    digest = _file_sha256(p)
    try:
        state = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        _fail("CHAIN_STATE_INVALID", f"{p.name}: not JSON: {exc}")
    if not isinstance(state, dict) \
            or state.get("schema") != CHAIN_STATE_SCHEMA:
        _fail("CHAIN_STATE_INVALID",
              f"{p.name}: schema "
              f"{(state or {}).get('schema')!r} != "
              f"{CHAIN_STATE_SCHEMA!r}")
    extra = set(state) - _CHAIN_STATE_KEYS
    if extra:
        _fail("CHAIN_STATE_INVALID",
              f"{p.name}: unexpected keys {sorted(extra)}")
    return state, digest


# ---------------------------------------------------------------------------
# Case alignment and ensemble extraction
# ---------------------------------------------------------------------------

def _per_arm_rows(facts: dict) -> dict:
    return {int(r.get("arm", -1)): r for r in facts.get("per_arm", [])}


def _anchor_identities(facts: dict, anchor_arm: int, what: str) -> list:
    """Per-volume identities of a completed anchor run (every complete
    arm was verified against the manifest at its own evaluation stage,
    so any completed run carries the canonical order)."""
    for row in facts.get("per_arm_seed", []):
        if int(row.get("arm", -1)) == anchor_arm \
                and row.get("status") == "completed":
            pv = row.get("per_volume") or {}
            ids = pv.get("identities")
            if not isinstance(ids, list) or not ids:
                _fail("CROSSFLOW_FACTS_INVALID",
                      f"{what}: a completed anchor run carries no "
                      f"identities")
            return [{"file": e.get("file"),
                     "slice_index": int(e.get("slice_index", -1))}
                    for e in ids]
    _fail("CROSSFLOW_ARMS_INCOMPLETE",
          f"{what}: no completed anchor (arm {anchor_arm}) run -- the "
          f"evidence chain is not complete")


def check_case_alignment(v03_facts: dict, v04_facts: dict,
                         manifest_ids: list) -> list:
    """173 SS5 (matched evaluation cases, volume-level pairing):
    order-sensitive identity equality against the pinned holdout
    manifest on BOTH sides. No canonicalisation is registered -- a
    missing, duplicated, or reordered identity refuses the
    comparison."""
    ids_v03 = _anchor_identities(v03_facts, V03_ANCHOR_ARM,
                                 "V03 facts")
    ids_v04 = _anchor_identities(v04_facts, V04_ANCHOR_ARM,
                                 "V04 facts")
    for what, ids in (("V03 facts", ids_v03), ("V04 facts", ids_v04)):
        if ids != manifest_ids:
            _fail("CROSSFLOW_CASE_MISMATCH",
                  f"{what}: per-volume identities != the pinned "
                  f"{N_VOLUMES}-volume holdout manifest "
                  f"(order-sensitive)")
    if ids_v03 != ids_v04:
        _fail("CROSSFLOW_CASE_MISMATCH",
              "V03 and V04 per-volume identities diverge; volume-level "
              "pairing is impossible")
    return ids_v03


def _ens_vec(rows: dict, arm: int, key: str, what: str) -> np.ndarray:
    row = rows.get(arm)
    if row is None:
        _fail("CROSSFLOW_FACTS_INVALID", f"{what}: arm {arm} row "
                                         f"missing")
    if row.get("ensemble_status") != "complete":
        _fail("CROSSFLOW_ARMS_INCOMPLETE",
              f"{what}: arm {arm} ensemble_status "
              f"{row.get('ensemble_status')!r} != 'complete'")
    ens = row.get("per_volume_ensemble")
    vec = (ens or {}).get(key)
    if not isinstance(vec, list) or len(vec) != N_VOLUMES:
        _fail("CROSSFLOW_FACTS_INVALID",
              f"{what}: arm {arm} per_volume_ensemble.{key} missing "
              f"or != {N_VOLUMES} volumes")
    a = np.asarray(vec, dtype=np.float64)
    if not np.isfinite(a).all():
        _fail("CROSSFLOW_FACTS_INVALID",
              f"{what}: arm {arm} per_volume_ensemble.{key} contains "
              f"non-finite values")
    return a


# ---------------------------------------------------------------------------
# Readiness audit (173 SS8.1): six items, each confirmed or failed with
# evidence; an unconfirmable item is NOT confirmed and therefore fails.
# ---------------------------------------------------------------------------

def _audit_activation(ctx: dict) -> tuple:
    act = ctx["chain"].get("activation")
    ev = []
    ok = isinstance(act, dict)
    if ok:
        for k, want in ACTIVATION_EXPECTED.items():
            got = act.get(k)
            ev.append(f"{k}={got!r} (required {want!r})")
            ok = ok and got == want
    else:
        ev.append("activation block missing or malformed")
    ev.append("173 SS10: a V03 decisive null activates exactly one "
              "RealNVP robustness port; NICE stays parked")
    return ("PASS" if ok else "FAIL"), ev


def _audit_chains(ctx: dict) -> tuple:
    chains = ctx["chain"].get("chains")
    ev = []
    ok = isinstance(chains, dict)
    if ok:
        for exp in ("v03", "v04"):
            blk = chains.get(exp)
            want = EXPECTED_CHAINS[exp]
            if not isinstance(blk, dict):
                ev.append(f"{exp}: chain block missing")
                ok = False
                continue
            for stage in ("spec", "plan", "exec"):
                ent = blk.get(stage) or {}
                want_status = "FROZEN" if stage in ("spec", "plan") \
                    else "registered"
                good = ent.get("article") == want[stage] \
                    and ent.get("status") == want_status
                ev.append(f"{exp}.{stage}: article "
                          f"{ent.get('article')!r} status "
                          f"{ent.get('status')!r} (required article "
                          f"{want[stage]} {want_status})")
                ok = ok and good
            res = blk.get("result") or {}
            good = isinstance(res.get("article"), int) \
                and not isinstance(res.get("article"), bool) \
                and res.get("article", 0) > 0 \
                and res.get("status") == "registered"
            ev.append(f"{exp}.result: article {res.get('article')!r} "
                      f"status {res.get('status')!r} (a registered "
                      f"RESULT is required; V04R registration is the "
                      f"178 SS12a lifecycle gate)")
            ok = ok and good
    else:
        ev.append("chains block missing or malformed")
    for what, rows, arms in (
            ("V03", ctx["v03_rows"], (V03_ANCHOR_ARM, V03_PARITY_ARM)),
            ("V04", ctx["v04_rows"], (V04_ANCHOR_ARM, V04_PARITY_ARM))):
        for a in arms:
            row = rows.get(a) or {}
            st = row.get("ensemble_status")
            ev.append(f"{what} arm {a} (anchor/parity) "
                      f"ensemble_status={st!r}")
            ok = ok and st == "complete"
    return ("PASS" if ok else "FAIL"), ev


def _audit_shared(ctx: dict) -> tuple:
    ev = []
    ok = True
    s3 = [int(s) for s in ctx["v03_facts"].get("seeds", [])]
    s4 = [int(s) for s in ctx["v04_facts"].get("seeds", [])]
    want = [int(s) for s in SEEDS]
    ev.append(f"seeds V03={s3} V04={s4} (locked {want}, seed-major "
              f"ordering per the registered plans)")
    ok = ok and s3 == want and s4 == want
    eva = ctx["v04_facts"].get("evaluation") or {}
    boot = eva.get("bootstrap") or {}
    readouts = eva.get("readouts") or []
    good = eva.get("data_range") == 1.0 \
        and int(boot.get("seed", -1)) == v04e.BOOTSTRAP_SEED \
        and int(boot.get("B", -1)) == v04e.BOOTSTRAP_B \
        and boot.get("method") == "percentile" \
        and len(readouts) == 2 \
        and any(str(r).startswith("R0") for r in readouts)
    ev.append(f"V04 evaluation block: data_range="
              f"{eva.get('data_range')!r} bootstrap={boot!r} "
              f"readouts={readouts!r} (V02G locked conventions, R0 "
              f"primary + R1 sensitivity)")
    ok = ok and good
    ev.append(f"V03 bootstrap conventions pin-verified at load "
              f"(B={v04e.BOOTSTRAP_B}, seed={v04e.BOOTSTRAP_SEED}, "
              f"percentile) by the registered V04E loader")
    k3 = set((ctx["v03_rows"].get(V03_PARITY_ARM) or {})
             .get("per_volume_ensemble") or {})
    k4 = set((ctx["v04_rows"].get(V04_PARITY_ARM) or {})
             .get("per_volume_ensemble") or {})
    ev.append(f"metric keys V03={sorted(k3)} V04={sorted(k4)} "
              f"(locked {sorted(METRIC_KEYS)})")
    ok = ok and k3 == set(METRIC_KEYS) and k4 == set(METRIC_KEYS)
    ev.append(f"NLL comparability (173 SS7): {LIKELIHOOD_CONTRACT}; "
              f"exact analytic log-determinants on both sides (NSF "
              f"spline / RealNVP affine) -- no numerical log-det "
              f"approximation")
    try:
        ids = check_case_alignment(ctx["v03_facts"], ctx["v04_facts"],
                                   ctx["manifest_ids"])
        ev.append(f"case alignment: V03/V04 per-volume identities == "
                  f"the pinned manifest, {len(ids)} volumes, "
                  f"order-sensitive")
    except V04XError as exc:
        ev.append(f"case alignment refused: {exc}")
        ok = False
    shared = ctx["chain"].get("shared")
    if not isinstance(shared, dict) or set(shared) != _SHARED_KEYS:
        ev.append("shared attestation block missing or malformed "
                  f"(required keys {sorted(_SHARED_KEYS)})")
        ok = False
    else:
        g1 = shared.get("data_range") == 1.0
        g2 = shared.get("likelihood") == LIKELIHOOD_CONTRACT
        g3 = shared.get("holdout_manifest_sha256") \
            == ctx["manifest_sha"]
        ev.append(f"attested data_range={shared.get('data_range')!r} "
                  f"(required 1.0); attested likelihood matches the "
                  f"SS7 contract: {g2}; attested holdout-manifest sha "
                  f"== current manifest sha: {g3}")
        ok = ok and g1 and g2 and g3
    return ("PASS" if ok else "FAIL"), ev


def _audit_deviations(ctx: dict) -> tuple:
    devs = ctx["chain"].get("deviations")
    ev = []
    ok = isinstance(devs, list)
    if ok:
        hit = [d for d in devs
               if isinstance(d, dict)
               and (d.get("registered_in") or {}).get("article") == 173
               and str((d.get("registered_in") or {})
                       .get("section")) == "8"
               and d.get("timing") == "pre-results"]
        ok = len(hit) >= 1
        ev.append(f"{len(hit)} deviation(s) attested as registered in "
                  f"173 SS8 pre-results (the affine-coupling-vs-spline "
                  f"register row is the declared architecture-intrinsic "
                  f"difference, conditionally admissible)")
        if not ok:
            ev.append("the registered affine-vs-spline deviation "
                      "(173 SS8) is not attested")
    else:
        ev.append("deviations block missing or malformed")
    spec_ref = (ctx["v04_facts"].get("spec") or {}).get("article")
    ev.append(f"artifact cross-check: V04 facts spec article "
              f"{spec_ref!r} (required 176, the architecture lock)")
    ok = ok and spec_ref == 176
    return ("PASS" if ok else "FAIL"), ev


def _audit_statistics(ctx: dict) -> tuple:
    st = ctx["chain"].get("statistics")
    ev = []
    ok = isinstance(st, dict)
    if ok:
        lock = st.get("delta_delta_lock") or {}
        good = lock.get("article") == 176 \
            and str(lock.get("section")) == "13" \
            and st.get("frozen_timing") == "pre-results"
        ev.append(f"delta_delta_lock={lock!r} "
                  f"frozen_timing={st.get('frozen_timing')!r} "
                  f"(required article 176 section 13, pre-results)")
        ok = good
    else:
        ev.append("statistics block missing or malformed")
    ev.append("artifact cross-check: V02G bootstrap conventions on "
              "both sides (V03 CI blocks pin-verified at load; the V04 "
              "evaluation block is checked under shared_settings)")
    return ("PASS" if ok else "FAIL"), ev


def _audit_no_tuning(ctx: dict) -> tuple:
    ht = ctx["chain"].get("holdout_tuning")
    return ("PASS" if ht == "none" else "FAIL"), \
        [f"holdout_tuning={ht!r} (required 'none')"]


_AUDIT_FNS = {"activation_rules": _audit_activation,
              "evidence_chains": _audit_chains,
              "shared_settings": _audit_shared,
              "deviations_registered": _audit_deviations,
              "statistics_frozen": _audit_statistics,
              "no_holdout_tuning": _audit_no_tuning}


def audit_readiness(ctx: dict) -> tuple:
    """173 SS8.1: all six items must be confirmed; any failed (or
    unconfirmable) item -> CROSSFLOW_GATE_FAILURE."""
    items = []
    for name, text in READINESS_ITEMS:
        try:
            status, evidence = _AUDIT_FNS[name](ctx)
        except V04XError as exc:
            status, evidence = "FAIL", [f"unconfirmable: {exc}"]
        items.append({"item": name, "text": text, "status": status,
                      "evidence": evidence})
    verdict = "PASS" if all(i["status"] == "PASS" for i in items) \
        else GATE_FAILURE
    return items, verdict


# ---------------------------------------------------------------------------
# Locked DeltaDelta (176 SS13) and the V02G paired bootstrap
# ---------------------------------------------------------------------------

def compute_delta_delta(v03_rows: dict, v04_rows: dict) -> dict:
    """176 SS13 (locked): per-volume difference of improvements on the
    frozen per-volume ensemble values. Seed averaging happened WITHIN
    each architecture at its own evaluation stage (per_volume_ensemble
    is the 3-seed ensemble mean) -- seeds never pair across
    architectures. NLL is sign-reversed to higher-is-better."""
    d_v03_psnr = _ens_vec(v03_rows, V03_PARITY_ARM, "psnr_r0",
                          "V03 facts") \
        - _ens_vec(v03_rows, V03_ANCHOR_ARM, "psnr_r0", "V03 facts")
    d_v04_psnr = _ens_vec(v04_rows, V04_PARITY_ARM, "psnr_r0",
                          "V04 facts") \
        - _ens_vec(v04_rows, V04_ANCHOR_ARM, "psnr_r0", "V04 facts")
    d_v03_nll = _ens_vec(v03_rows, V03_ANCHOR_ARM, "nll", "V03 facts") \
        - _ens_vec(v03_rows, V03_PARITY_ARM, "nll", "V03 facts")
    d_v04_nll = _ens_vec(v04_rows, V04_ANCHOR_ARM, "nll", "V04 facts") \
        - _ens_vec(v04_rows, V04_PARITY_ARM, "nll", "V04 facts")
    return {"delta_v03_psnr_r0": d_v03_psnr,
            "delta_v04_psnr_r0": d_v04_psnr,
            "dd_psnr_r0": d_v04_psnr - d_v03_psnr,
            "delta_v03_nll": d_v03_nll,
            "delta_v04_nll": d_v04_nll,
            "dd_nll": d_v04_nll - d_v03_nll}


def paired_bootstrap(dd: np.ndarray) -> dict:
    """V02G conventions via the registered single source
    v04_eval.bootstrap_ci: ONE PCG64(3) stream, B=10000 resampled
    volume-index draws, arithmetic means, percentile 2.5/97.5. The
    cross-experiment pairing lives in the elementwise DeltaDelta
    construction, so one shared resample matrix resamples paired
    volumes jointly. Deterministic by construction."""
    return v04e.bootstrap_ci(dd)


def _ci_excludes_zero(ci: dict) -> bool:
    lo, hi = ci["ci95"]
    return lo > 0.0 or hi < 0.0


# ---------------------------------------------------------------------------
# Phase runners and CLI
# ---------------------------------------------------------------------------

def _seal_record(record: dict) -> dict:
    semantic = {k: v for k, v in record.items()
                if k not in _SEMANTIC_EXCLUDE}
    record["semantic_sha256"] = _canonical_hash()(semantic)
    record["semantic_scope"] = {
        "included_keys": sorted(semantic.keys()),
        "excluded": list(_SEMANTIC_EXCLUDE),
        "canonicalisation": "preflight_io.canonical_hash"}
    return record


def run_preflight(cfg: dict, *,
                  v03_facts_sha: str = v04e.V03_FACTS_SHA256) -> int:
    """178 SS12a phase 1 (non-authoritative): V03 facts sha pin-verify,
    V04 facts integrity, evaluation-manifest identity, case-set
    alignment. The record is ALWAYS written (diagnostic evidence); a
    FAIL result exits 2 with CROSSFLOW_PREFLIGHT_FAILED. cfg keys:
    v03_facts, v04_facts, manifest_dir, out, commit, branch. The
    v03_facts_sha override exists for synthetic fixtures only; the CLI
    always uses the locked pin."""
    out = Path(cfg["out"])
    checks = []
    loaded = {}

    def _check(name, fn) -> bool:
        try:
            detail = fn()
            checks.append({"check": name, "status": "PASS",
                           "detail": detail})
            return True
        except (V04XError, v04c.V04CError, v04e.V04EError,
                v02m.V02Error) as exc:
            code = getattr(exc, "code", None) \
                or str(exc).split(":")[0]
            checks.append({"check": name, "status": "FAIL",
                           "code": code, "detail": str(exc)})
            return False

    def c_v03() -> str:
        facts, sha = load_verified_v03_facts(cfg["v03_facts"],
                                             expected_sha=v03_facts_sha)
        loaded["v03_facts"] = facts
        loaded["v03_sha"] = sha
        return (f"sha256 {sha[:12]}... == the pin "
                f"{v03_facts_sha[:12]}...; schema and recorded V02G "
                f"CI settings verified by the registered V04E loader")

    def c_v04() -> str:
        rows, facts, sha = load_verified_v04_facts(
            cfg["v04_facts"], cfg["manifest_dir"])
        loaded["v04_rows"] = rows
        loaded["v04_facts"] = facts
        loaded["v04_sha"] = sha
        return f"registered V04C battery PASS; facts sha {sha[:12]}..."

    def c_man() -> str:
        ids, msha = _load_holdout_identities(Path(cfg["manifest_dir"]))
        loaded["manifest_ids"] = ids
        loaded["manifest_sha"] = msha
        return (f"{HOLDOUT_MANIFEST}: {len(ids)} volumes, sha256 "
                f"{msha[:12]}...")

    def c_case() -> str:
        ids = check_case_alignment(loaded["v03_facts"],
                                   loaded["v04_facts"],
                                   loaded["manifest_ids"])
        return (f"V03/V04 per-volume identities == the pinned "
                f"manifest, {len(ids)} volumes, order-sensitive")

    ok = _check("v03_facts_pin", c_v03)
    ok = _check("v04_facts_integrity", c_v04) and ok
    ok = _check("manifest_identity", c_man) and ok
    if all(c["status"] == "PASS" for c in checks):
        ok = _check("case_alignment", c_case) and ok
    else:
        checks.append({"check": "case_alignment", "status": "not_run",
                       "detail": "prerequisite check(s) failed"})
        ok = False
    record = {
        "schema": PREFLIGHT_SCHEMA,
        "script": f"{__abbr__} v{__version__}",
        "phase": "preflight",
        "status": "non_authoritative",
        "notice": "178 SS12a: artifact/data compatibility only -- a "
                  "preflight PASS is NOT 173 SS8.1 admissibility and "
                  "grants no permission for an authoritative "
                  "cross-flow claim",
        "spec": {"article": 173, "status": "FROZEN"},
        "plan": {"article": 178, "status": "FROZEN"},
        "code": {"commit": cfg["commit"], "branch": cfg["branch"]},
        "inputs": {
            "v03_facts": {"path": str(cfg["v03_facts"]),
                          "sha256": loaded.get("v03_sha"),
                          "pin": v03_facts_sha},
            "v04_facts": {"path": str(cfg["v04_facts"]),
                          "sha256": loaded.get("v04_sha")},
            "holdout_manifest": {"name": HOLDOUT_MANIFEST,
                                 "sha256": loaded.get("manifest_sha")}},
        "checks": checks,
        "result": "PASS" if ok else "FAIL",
    }
    _write_with_sidecar(out / PREFLIGHT_RECORD, _seal_record(record))
    if not ok:
        failed = [c["check"] for c in checks if c["status"] == "FAIL"]
        logger.error("[%s] %s: preflight check(s) failed (%s); %s "
                     "written (non-authoritative diagnostic)",
                     __abbr__, PREFLIGHT_FAILURE, ", ".join(failed),
                     PREFLIGHT_RECORD)
        return 2
    print(f"PASS {__abbr__} v{__version__}: preflight (NON-AUTHORITATIVE"
          f" compatibility only, 178 SS12a) 4/4 checks PASS; "
          f"{PREFLIGHT_RECORD} + sidecar")
    return 0


def run_audit(cfg: dict, *,
              v03_facts_sha: str = v04e.V03_FACTS_SHA256) -> int:
    """178 SS12a phase 2 (authoritative, requires a registered V04R):
    the 173 SS8.1 six-item readiness audit; DeltaDelta (176 SS13,
    locked) only if every item passes. Any failed item -> the record
    is written with CROSSFLOW_GATE_FAILURE, no confirmatory
    DeltaDelta, claim 'exploratory at most'; exit 2. cfg keys:
    v03_facts, v04_facts, manifest_dir, chain_state, out, commit,
    branch."""
    out = Path(cfg["out"])
    v03_facts, v03_sha = load_verified_v03_facts(
        cfg["v03_facts"], expected_sha=v03_facts_sha)
    v04_rows, v04_facts, v04_sha = load_verified_v04_facts(
        cfg["v04_facts"], cfg["manifest_dir"])
    manifest_ids, manifest_sha = _load_holdout_identities(
        Path(cfg["manifest_dir"]))
    chain, chain_sha = load_chain_state(cfg["chain_state"])
    v03_rows = _per_arm_rows(v03_facts)
    ctx = {"v03_facts": v03_facts, "v04_facts": v04_facts,
           "v03_rows": v03_rows, "v04_rows": v04_rows,
           "manifest_ids": manifest_ids, "manifest_sha": manifest_sha,
           "chain": chain}
    items, verdict = audit_readiness(ctx)
    record = {
        "schema": RECORD_SCHEMA,
        "script": f"{__abbr__} v{__version__}",
        "phase": "audit",
        "spec": {"article": 173, "status": "FROZEN"},
        "plan": {"article": 178, "status": "FROZEN"},
        "delta_delta_lock": {"article": 176, "section": "13"},
        "code": {"commit": cfg["commit"], "branch": cfg["branch"]},
        "inputs": {
            "v03_facts": {"path": str(cfg["v03_facts"]),
                          "sha256": v03_sha, "pin": v03_facts_sha},
            "v04_facts": {"path": str(cfg["v04_facts"]),
                          "sha256": v04_sha},
            "holdout_manifest": {"name": HOLDOUT_MANIFEST,
                                 "sha256": manifest_sha},
            "chain_state": {"path": str(cfg["chain_state"]),
                            "sha256": chain_sha}},
        "readiness": {"gate": READINESS_GATE, "items": items,
                      "verdict": verdict},
        "claim": ("registered cross-flow comparison"
                  if verdict == "PASS" else
                  "exploratory at most (173 SS8.1); the "
                  "within-architecture V04 result is unaffected "
                  "(178 SS12a)"),
        "uncertainty_scope": UNCERTAINTY_SCOPE,
    }
    if verdict == "PASS":
        dd = compute_delta_delta(v03_rows, v04_rows)
        ci_psnr = paired_bootstrap(dd["dd_psnr_r0"])
        ci_nll = paired_bootstrap(dd["dd_nll"])
        record["delta_delta"] = {
            "definition": "DeltaDelta(v) = Delta_V04(v) - "
                          "Delta_V03(v); Delta_a(v) = parity-arm "
                          "ensemble value(v) - anchor ensemble "
                          "value(v) (176 SS13, locked)",
            "pairing": PAIRING_RULE,
            "n_volumes": N_VOLUMES,
            "arms": {"v03_anchor": V03_ANCHOR_ARM,
                     "v03_parity": V03_PARITY_ARM,
                     "v04_anchor": V04_ANCHOR_ARM,
                     "v04_parity": V04_PARITY_ARM},
            "psnr_r0": {
                "delta_v03_mean": float(dd["delta_v03_psnr_r0"]
                                        .mean()),
                "delta_v04_mean": float(dd["delta_v04_psnr_r0"]
                                        .mean()),
                "dd": ci_psnr,
                "dd_ci95_excludes_zero": _ci_excludes_zero(ci_psnr)},
            "nll": {
                "convention": NLL_CONVENTION,
                "delta_v03_mean": float(dd["delta_v03_nll"].mean()),
                "delta_v04_mean": float(dd["delta_v04_nll"].mean()),
                "dd": ci_nll,
                "dd_ci95_excludes_zero": _ci_excludes_zero(ci_nll)},
            "bootstrap": {"seed": v04e.BOOTSTRAP_SEED,
                          "B": v04e.BOOTSTRAP_B,
                          "method": "percentile",
                          "source": "v04_eval.bootstrap_ci (V02G "
                                    "conventions, single source)"},
            "interpretation_note": INTERPRETATION_NOTE,
        }
    _write_with_sidecar(out / AUDIT_RECORD, _seal_record(record))
    if verdict != "PASS":
        failed = [i["item"] for i in items if i["status"] != "PASS"]
        logger.error("[%s] %s: readiness item(s) failed (%s); %s "
                     "written with no confirmatory DeltaDelta; the "
                     "comparison is exploratory at most and the "
                     "within-architecture V04 result is unaffected",
                     __abbr__, GATE_FAILURE, ", ".join(failed),
                     AUDIT_RECORD)
        return 2
    print(f"PASS {__abbr__} v{__version__}: readiness gate PASS "
          f"(6/6); DeltaDelta PSNR R0 mean {ci_psnr['mean']:+.4f} dB "
          f"CI95 [{ci_psnr['ci95'][0]:+.4f}, "
          f"{ci_psnr['ci95'][1]:+.4f}]; {AUDIT_RECORD} + sidecar")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=f"{__abbr__} v{__version__} -- cross-flow "
                    f"readiness + DeltaDelta (173 SS8.1, 176 SS13, "
                    f"178 SS12a)")
    ap.add_argument("--mode", required=True,
                    choices=("preflight", "audit"))
    ap.add_argument("--v03-facts")
    ap.add_argument("--v04-facts")
    ap.add_argument("--manifest-dir")
    ap.add_argument("--chain-state")
    ap.add_argument("--out")
    ap.add_argument("--commit")
    ap.add_argument("--branch")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    need = ["v03_facts", "v04_facts", "manifest_dir", "out", "commit",
            "branch"]
    if args.mode == "audit":
        need.append("chain_state")
    missing = ["--" + k.replace("_", "-") for k in need
               if getattr(args, k) is None]
    if missing:
        logger.error("[%s] missing required arguments: %s",
                     __abbr__, ", ".join(missing))
        return 2
    cfg = {"v03_facts": args.v03_facts, "v04_facts": args.v04_facts,
           "manifest_dir": args.manifest_dir,
           "chain_state": args.chain_state, "out": args.out,
           "commit": args.commit, "branch": args.branch}
    try:
        if args.mode == "preflight":
            return run_preflight(cfg)
        return run_audit(cfg)
    except V04XError:
        return 2
    except Exception as exc:  # untyped unexpected failure: still exit 2
        logger.error("[%s] UNEXPECTED: %s", __abbr__, exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())


# Changelog
# - v0.1 (2026-10-12): initial module (advisor V04X plan assessed
#   against the frozen texts 173 SS5-SS8.1/SS10, 176 SS13, 178
#   SS12a/SS5-f17 -- GO, no frozen-text changes). Two-phase lifecycle
#   per 178 SS12a: --mode preflight (non-authoritative compatibility
#   record, explicit non_authoritative status) and --mode audit (173
#   SS8.1 six-item readiness + locked 176 SS13 DeltaDelta, V02G
#   bootstrap via v04_eval.bootstrap_ci as the single source).
#   Governance items are attested by a sidecar-pinned chain-state
#   document (seqref-v04-crossflow-chain-state/1, closed key set);
#   every artifact-checkable item is verified from the pin-verified
#   facts. CROSSFLOW_GATE_FAILURE on any failed readiness item, with
#   no confirmatory DeltaDelta in the record. Fixture-verified (f17 +
#   supplementary X01-X13); no authoritative execution.

# SEQREF-DLAT-STAT v0.3 -- diag_lat_stats.py: D-LAT bootstrap, contrasts, classes, tags, L6, outcome
# LIFETIME: KEEP
"""
STATUS: PRE-FREEZE ENGINEERING; NOT ACTIVATED (236 LE19; 238 LE21). Written against the
mini-spec candidate v0.5.2 (not frozen) and the interpretation clarifications I1-I4 of
238 LE21; outcome rule, seed tag and fixed-epoch tag follow candidate v0.6.2 (not frozen;
amendments A14, A16 with its clarification, A19; plan SEQREF-DLAT-P23REV v0.1, 242 LE51).
Provisional until freeze. Mathematics only: no I/O, no model, no random draw
except the single registered bootstrap call.

PORTED from diag_unc_stats.py v0.5 lineage (sha256 b8706513...8f55) and diag_unc_core.py
v0.2 (sha256 a0d3b4a4...8b06), logic unchanged, D-LAT codes; equivalence checked by
fixture T19: bootstrap_matrix, matrix_identity, verify_matrix, _percentile_ci, sign_label,
multiplicities, _boot_midranks, boot_spearman, spearman, stability_label, _pair,
_interaction, TIER1_CONTRASTS, TIER2_CONTRASTS, and the paired additive pooling rule
(effect = mean over seeds of per-seed volume means; replicates diff[:, idx]).
NOT ported: the D-UNC interpretation matrix, cross rows, D-GAP labels, any limit projection.

Interaction sign: X-TIME = (lam2 change) - (lam0 change). For a measure predicted to rise
(s = +1) "gap-selective" needs the X-TIME interval entirely BELOW 0; for L5 (s = -1) above 0.

Stability: per contrast (D-UNC rule, four labels on THREE distinct subsets: pre64 == halfA);
measure-level summary "stable" only if S-GAP0 and C-TIME0 are both stable (I4).

No fallback, placeholder, mock or dummy return. "not computed" is an explicit status.

Changelog
- v0.3 (2026-11-22): revision SEQREF-DLAT-P23REV v0.1 (242 LE51), candidate v0.6.2.
  outcome(): provisional only if a result not computed could change the letter (A14); new
  argument (class, predicted sign and labels per measure); new field open_contrasts.
  seed_tag(): four values in first-match order, zero and missing effects defined (A16).
  fixed_epoch_tag(): new argument (all labels); "not computed" while gap association is
  still possible, "not applicable" when excluded (A19). Stability summary of L1 and L5 is
  the exact string "not applicable" (A17). New result field rules_version. Constant
  SELECTIVITY_OPEN removed (used by outcome() only). Nothing else changed.
- v0.2 (2026-11-21): BUG FIX (found in the P2 review). l6_contrast took its per-volume changes
  from per_seed_differences, which covers all seeds, so a state not computed in one seed
  suppressed the L6 correlations of the other seeds. L6 now uses seed_differences (the
  requested seed only). The pooled rule for classified contrasts is unchanged.
- v0.1 (2026-11-21): first version. Ported bootstrap and contrasts; Population with explicit
  not-computed states; class rules with I1-I4 (missing-contrast branches, provisional
  outcome); seed, fixed-epoch, coordinate and stability tags; L6 statistics.
"""
import hashlib
import logging
import math

import numpy as np

from diag_lat_core import DiagLatError, NOT_COMPUTED, NONFINITE, INPUT

__version__ = "0.3"
ABBR = "SEQREF-DLAT-STAT"
log = logging.getLogger("diag_lat.stats")

DIAG_LAT_SEED = 20261120
N_BOOT = 10000
N_VOLUMES = 97
UNDEFINED_DRAW_LIMIT = 0.01
STABILITY_SUBSETS = ("pre64", "full", "halfA", "halfB")
BOOTSTRAP_IDENTITY = "DIAG_LAT_BOOTSTRAP_IDENTITY"
POPULATION_MISMATCH = "DIAG_LAT_POPULATION_MISMATCH"
NC = "not computed"
RULES_VERSION = "SEQREF-MRI-DIAG-LAT v0.6.2 CANDIDATE (A14, A16-CL, A19; not frozen)"

# classified measures: id -> (record key, predicted sign on gap contrasts, bank-dependent)
MEASURES = {
    "L1": ("KS", +1, False),
    "L2": ("log_gamma_ratio", +1, True),
    "L3a": ("d01", +1, True),
    "L3b": ("a01", +1, True),
    "L4": ("TC_G", +1, True),
    "L5": ("P_swap", -1, False),
}
COORDINATE_TAGGED = ("L2", "L3a", "L3b")
READ_GAP = ("S-GAP0", "C-TIME0")
FIXED_EPOCH = "F-TRAJ0(10→30)"

# class labels (candidate section 5 + 238 LE21 I1, I2)
C_SELECTIVE = "gap-selective"
C_SHARED = "shared"
C_GA_PLAIN = "gap-associated (not selective, not shared)"
C_GA_SEL_UNRES = "gap-associated; selectivity and shared status unresolved — measure not computed"
C_GA_SHARED_UNRES = "gap-associated; shared status unresolved — measure not computed"
C_CONTRADICTED = "contradicted"
C_UNRESOLVED = "unresolved"
C_UNRES_NC = "unresolved — measure not computed"
GAP_ASSOCIATED = (C_SELECTIVE, C_SHARED, C_GA_PLAIN, C_GA_SEL_UNRES, C_GA_SHARED_UNRES)
ALL_CLASSES = GAP_ASSOCIATED + (C_CONTRADICTED, C_UNRESOLVED, C_UNRES_NC)


# ---- ported: bootstrap --------------------------------------------------------------
def _default_generator(seed):
    return np.random.Generator(np.random.PCG64(seed))


def bootstrap_matrix(seed=DIAG_LAT_SEED, n_volumes=N_VOLUMES, n_boot=N_BOOT, generator_factory=_default_generator):
    """The single registered call. No other draw is taken from this generator."""
    return generator_factory(seed).integers(0, n_volumes, size=(n_boot, n_volumes), dtype=np.int64)


def matrix_identity(idx):
    """SHA-256 of the raw bytes as little-endian int64, C order, no header."""
    if not (isinstance(idx, np.ndarray) and idx.ndim == 2 and idx.dtype.kind == "i" and idx.dtype.itemsize == 8):
        raise DiagLatError(BOOTSTRAP_IDENTITY, "index matrix is not a 2-D signed 64-bit integer array")
    return hashlib.sha256(np.ascontiguousarray(idx.astype("<i8", copy=False)).tobytes(order="C")).hexdigest()


def verify_matrix(idx, registered_sha256, n_volumes=N_VOLUMES, n_boot=N_BOOT):
    """Validity gate: a mismatch is a failure, never a reason to regenerate."""
    digest = matrix_identity(idx)
    if idx.shape != (n_boot, n_volumes) or idx.min() < 0 or idx.max() >= n_volumes:
        raise DiagLatError(BOOTSTRAP_IDENTITY, "index matrix has the wrong shape or range")
    if digest != registered_sha256:
        raise DiagLatError(BOOTSTRAP_IDENTITY, "index matrix hash %s differs from the registered %s"
                           % (digest, registered_sha256))
    return digest


def _percentile_ci(boot):
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return (float(lo), float(hi))


def sign_label(ci):
    if ci is None:
        return NC
    if ci[1] < 0:
        return "below0"
    if ci[0] > 0:
        return "above0"
    return "unresolved"


def multiplicities(idx):
    """(B, V) number of times each original volume appears in each replicate."""
    counts = np.zeros(idx.shape, dtype=float)
    np.add.at(counts, (np.repeat(np.arange(idx.shape[0]), idx.shape[1]), idx.ravel()), 1.0)
    return counts


def _boot_midranks(v, counts):
    uniq, inv = np.unique(v, return_inverse=True)
    onehot = np.zeros((len(v), len(uniq)))
    onehot[np.arange(len(v)), inv] = 1.0
    n_g = counts @ onehot
    rank_g = np.cumsum(n_g, axis=1) - n_g + (n_g + 1.0) / 2.0
    return rank_g[:, inv], (n_g > 0).sum(axis=1)


def boot_spearman(a, b, idx, counts=None):
    """Spearman rho (average ranks) on the resampled volumes of every replicate; nan where undefined."""
    counts = multiplicities(idx) if counts is None else counts
    n = idx.shape[1]
    ra, da = _boot_midranks(a, counts)
    rb, db = _boot_midranks(b, counts)
    ca = ra - (counts * ra).sum(axis=1, keepdims=True) / n
    cb = rb - (counts * rb).sum(axis=1, keepdims=True) / n
    va, vb = (counts * ca * ca).sum(axis=1), (counts * cb * cb).sum(axis=1)
    rho = np.full(idx.shape[0], np.nan)
    ok = (da > 1) & (db > 1)
    rho[ok] = (counts * ca * cb).sum(axis=1)[ok] / np.sqrt(va[ok] * vb[ok])
    return rho


def average_ranks(v):
    """1-based average ranks (ties share the mean rank)."""
    v = np.asarray(v, dtype=float)
    uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    start = np.cumsum(cnt) - cnt
    return (start + (cnt + 1.0) / 2.0)[inv]


def spearman(a, b):
    """Spearman rho with average ranks; nan if either input is constant (undefined)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra, rb = average_ranks(a), average_ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra * rb).sum() / math.sqrt((ra * ra).sum() * (rb * rb).sum()))


def stability_label(labels):
    """'stable' iff the direction label is identical on the four subset labels (three distinct subsets)."""
    vals = [labels.get(s) for s in STABILITY_SUBSETS]
    return "stable" if None not in vals and len(set(vals)) == 1 else "Monte Carlo adequacy unresolved"


# ---- ported: contrasts (names, weights and signs unchanged) ---------------------------
def _pair(plus, minus):
    return {plus: 1.0, minus: -1.0}


def _interaction(a_plus, a_minus, b_plus, b_minus):
    """(a_plus - a_minus) - (b_plus - b_minus), with coinciding roles combined."""
    w = {}
    for role, v in ((a_plus, 1.0), (a_minus, -1.0), (b_plus, -1.0), (b_minus, 1.0)):
        w[role] = w.get(role, 0.0) + v
    return w


TIER1_CONTRASTS = {
    "S-GAP0": _pair("lam0/fin", "lam0/sel"),
    "S-CTL0": _pair("lam0/sel", "lam0/e3"),
    "C-TIME0": _pair("lam0/fin", "lam0/e3"),
    "S-CTL2": _pair("lam2/sel", "lam2/e3"),
    "C-TIME2": _pair("lam2/fin", "lam2/e3"),
    "S-GAP2": _pair("lam2/fin", "lam2/sel"),
    "C-OBJ": _pair("lam2/sel", "lam0/sel"),
    "C-OBJ@3": _pair("lam2/e3", "lam0/e3"),
    "X-TIME": _interaction("lam2/fin", "lam2/e3", "lam0/fin", "lam0/e3"),
    "X-CTL": _interaction("lam2/sel", "lam2/e3", "lam0/sel", "lam0/e3"),
}
TIER2_CONTRASTS = {"C-OBJ@10": _pair("lam2/e10", "lam0/e10"), "C-OBJ@30": _pair("lam2/e30", "lam0/e30")}
for _a, _b in (("e3", "e10"), ("e10", "e30"), ("e3", "e30")):
    _tag = "(%s→%s)" % (_a[1:], _b[1:])
    TIER2_CONTRASTS["F-TRAJ0" + _tag] = _pair("lam0/" + _b, "lam0/" + _a)
    TIER2_CONTRASTS["F-TRAJ2" + _tag] = _pair("lam2/" + _b, "lam2/" + _a)
    TIER2_CONTRASTS["X-FIX" + _tag] = _interaction("lam2/" + _b, "lam2/" + _a, "lam0/" + _b, "lam0/" + _a)
ALL_CONTRASTS = dict(TIER1_CONTRASTS, **TIER2_CONTRASTS)


# ---- population ----------------------------------------------------------------------
class Population:
    """Per-volume values for every (checkpoint role, seed, key), in one fixed volume order.
    A key is (name, subset) with subset None for measures that do not depend on the bank.
    A state with any NOT_COMPUTED volume is stored as not computed for that key (S6)."""

    def __init__(self, volume_ids):
        self.volume_ids = list(volume_ids)
        if len(set(self.volume_ids)) != len(self.volume_ids) or self.volume_ids != sorted(self.volume_ids):
            raise DiagLatError(POPULATION_MISMATCH, "volume list must be sorted and unique")
        self._v = {}
        self.not_computed = {}        # (role, seed, key) -> number of volumes not computed
        self.sealed = False

    def set(self, role, seed, name, values, subset=None):
        if self.sealed:
            raise DiagLatError(POPULATION_MISMATCH, "population is sealed")
        k = (role, seed, (name, subset))
        if k in self._v:
            raise DiagLatError(POPULATION_MISMATCH, "duplicate entry %r" % (k,))
        values = list(values)
        if len(values) != len(self.volume_ids):
            raise DiagLatError(POPULATION_MISMATCH, "%r: %d values for %d volumes" % (k, len(values), len(self.volume_ids)))
        if any(isinstance(v, str) and v != NOT_COMPUTED for v in values):
            raise DiagLatError(INPUT, "unknown status string for %r" % (k,))
        missing = sum(1 for v in values if isinstance(v, str))
        if missing:
            self._v[k] = NOT_COMPUTED
            self.not_computed[k] = missing
            log.warning("state %r: %d volume(s) not computed -> state not computed", k, missing)
            return
        a = np.array(values, dtype=float)
        if not np.all(np.isfinite(a)):
            raise DiagLatError(NONFINITE, "non-finite value for %r" % (k,))
        self._v[k] = a

    def seal(self):
        self.roles = sorted({r for r, _, _ in self._v})
        self.seeds = sorted({s for _, s, _ in self._v})
        keys = sorted({k for _, _, k in self._v}, key=repr)
        for role in self.roles:
            for seed in self.seeds:
                for key in keys:
                    if (role, seed, key) not in self._v:
                        raise DiagLatError(POPULATION_MISMATCH, "incomplete population: %s seed %s key %r" % (role, seed, key))
        self.keys, self.sealed = keys, True
        return self

    def col(self, role, seed, name, subset=None):
        if not self.sealed:
            raise DiagLatError(POPULATION_MISMATCH, "population used before it was sealed")
        k = (role, seed, (name, subset))
        if k not in self._v:
            raise DiagLatError(POPULATION_MISMATCH, "no entry %r" % (k,))
        return self._v[k]


# ---- effects -------------------------------------------------------------------------
_NOT_COMPUTED_EFFECT = {"point": None, "per_seed": None, "ci": None, "label": NC, "undefined_draws": 0}


def per_seed_differences(pop, weights, name, subset=None):
    """(S, V) per-volume paired differences sum_role w * value, or NOT_COMPUTED."""
    diff = []
    for seed in pop.seeds:
        d = np.zeros(len(pop.volume_ids))
        for role, w in weights.items():
            v = pop.col(role, seed, name, subset)
            if v is NOT_COMPUTED:
                return NOT_COMPUTED
            d = d + w * v
        diff.append(d)
    return np.array(diff)


def seed_differences(pop, weights, name, seed, subset=None):
    """(V,) per-volume paired differences for ONE seed, or NOT_COMPUTED if a state of THAT seed
    is not computed. Used by L6, which is a separate analysis for each seed; a missing state in
    another seed has no effect here. (The pooled contrasts use per_seed_differences.)"""
    if seed not in pop.seeds:
        raise DiagLatError(INPUT, "unknown seed %r" % (seed,))
    d = np.zeros(len(pop.volume_ids))
    for role, w in weights.items():
        v = pop.col(role, seed, name, subset)
        if v is NOT_COMPUTED:
            return NOT_COMPUTED
        d = d + w * v
    return d


def additive_effect(pop, weights, name, idx, subset=None):
    """Paired effect: per-volume differences, then means; replicates diff[:, idx] (223 section 3.4)."""
    diff = per_seed_differences(pop, weights, name, subset)
    if diff is NOT_COMPUTED:
        return dict(_NOT_COMPUTED_EFFECT)
    per_seed = diff.mean(axis=1)
    boot = diff[:, idx].mean(axis=2).mean(axis=0)
    ci = _percentile_ci(boot)
    return {"point": float(per_seed.mean()), "per_seed": [float(v) for v in per_seed], "ci": ci,
            "label": sign_label(ci), "undefined_draws": 0}


# ---- reading rules -------------------------------------------------------------------
def support(label, s):
    """Sign label -> 'supported' / 'opposite' / 'unresolved' / 'not computed' for predicted sign s."""
    if s not in (1, -1):
        raise DiagLatError(INPUT, "predicted sign must be +1 or -1")
    if label == NC:
        return NC
    if label == "unresolved":
        return "unresolved"
    if label not in ("above0", "below0"):
        raise DiagLatError(INPUT, "unknown sign label %r" % (label,))
    return "supported" if (label == "above0") == (s > 0) else "opposite"


def classify(s, labels):
    """Class of one measure from its sign labels on S-GAP0, C-TIME0, X-TIME, C-TIME2
    (candidate section 5 with clarifications I1 and I2 of 238 LE21)."""
    a, b = support(labels["S-GAP0"], s), support(labels["C-TIME0"], s)
    if a == "supported" and b == "supported":
        x = labels["X-TIME"]
        if x == NC:
            return C_GA_SEL_UNRES
        if x not in ("above0", "below0", "unresolved"):
            raise DiagLatError(INPUT, "unknown sign label %r" % (x,))
        if x == ("below0" if s > 0 else "above0"):       # larger predicted-direction change under lam0
            return C_SELECTIVE
        c2 = support(labels["C-TIME2"], s)
        if c2 == NC:
            return C_GA_SHARED_UNRES
        return C_SHARED if c2 == "supported" else C_GA_PLAIN
    if a == "opposite" and b == "opposite":
        return C_CONTRADICTED
    if NC in (a, b):
        other = b if a == NC else a
        return C_UNRESOLVED if other == "unresolved" else C_UNRES_NC
    return C_UNRESOLVED


def gap_possibility(s, labels):
    """Candidate v0.6.2 section 5 (A14). A measure CAN STILL BE GAP-ASSOCIATED if neither S-GAP0
    nor C-TIME0 is computed and unsupported (each is supported or not computed); it CAN STILL BE
    GAP-SELECTIVE if that holds and X-TIME is on the selective side or not computed.
    Returns the two flags and the not-computed contrasts each of them depends on."""
    sup = {c: support(labels[c], s) for c in READ_GAP}
    x = labels["X-TIME"]
    if x not in ("above0", "below0", "unresolved", NC):
        raise DiagLatError(INPUT, "unknown sign label %r" % (x,))
    can_ga = all(v in ("supported", NC) for v in sup.values())
    can_sel = can_ga and x in (NC, "below0" if s > 0 else "above0")
    gap_open = [c for c in READ_GAP if sup[c] == NC]
    return {"can_ga": can_ga, "can_sel": can_sel, "gap_open": gap_open,
            "sel_open": gap_open + (["X-TIME"] if x == NC else [])}


def outcome(measures):
    """Report outcome from {measure id: {"class", "predicted_sign", "labels"}} (candidate v0.6.2
    section 5, A14). PROVISIONAL only if a result that is not computed could change the letter:
    A confirmed if a measure is gap-selective; else provisional if one can still be gap-selective
    (letter B if a measure is gap-associated, else C); else B confirmed if a measure is
    gap-associated; else provisional (letter C) if one can still be gap-associated; else C confirmed.
    open_measures / open_contrasts name exactly the measures, with their not-computed contrasts,
    whose completion alone could change the letter: at letter B those that can still be
    gap-selective; at letter C those that can still be gap-associated."""
    classes = {m: v["class"] for m, v in measures.items()}
    for m, v in measures.items():
        if classes[m] != classify(v["predicted_sign"], v["labels"]):
            raise DiagLatError(INPUT, "class %r of %s disagrees with its labels" % (classes[m], m))
    if any(c == C_SELECTIVE for c in classes.values()):
        return {"outcome": "A", "provisional": False, "open_measures": [], "open_contrasts": {}}
    poss = {m: gap_possibility(v["predicted_sign"], v["labels"]) for m, v in measures.items()}
    any_ga = any(c in GAP_ASSOCIATED for c in classes.values())
    flag = "can_sel" if any_ga else "can_ga"
    open_c = {m: poss[m]["sel_open"] for m in sorted(poss) if poss[m][flag]}
    return {"outcome": "B" if any_ga else "C", "provisional": bool(open_c),
            "open_measures": list(open_c), "open_contrasts": open_c}


def seed_tag(effects):
    """Candidate v0.6.2 section 5 (A16 with its clarification). An effect has a sign only if it is
    computed and not exactly 0. On S-GAP0 and C-TIME0 together, the first that applies:
    "mixed-seed" if on either contrast the pooled effect has a sign and a computed per-seed effect
    does not share it (a per-seed effect of exactly 0 does not share it); "no pooled direction" if
    on either contrast the pooled effect is exactly 0; "not computed" if on either contrast the
    pooled or a per-seed effect is not computed; otherwise "seed-consistent"."""
    pooled, seeds = {}, {}
    for c in READ_GAP:
        p, ps = effects[c]["point"], effects[c]["per_seed"]
        ps = [None] if ps is None else list(ps)
        for v in [p] + ps:
            if v is not None and not math.isfinite(v):
                raise DiagLatError(NONFINITE, "seed_tag: non-finite effect on %s" % c)
        pooled[c], seeds[c] = p, ps
    for c in READ_GAP:
        p = pooled[c]
        if p is not None and p != 0.0 and any(v is not None and (v == 0.0 or (v > 0) != (p > 0)) for v in seeds[c]):
            return "mixed-seed"
    if any(pooled[c] is not None and pooled[c] == 0.0 for c in READ_GAP):
        return "no pooled direction"
    if any(pooled[c] is None or any(v is None for v in seeds[c]) for c in READ_GAP):
        return NC
    return "seed-consistent"


def fixed_epoch_tag(cls, s, labels):
    """Candidate v0.6.2 section 5 (I3, A15, A19), from the class and the labels on S-GAP0, C-TIME0
    and F-TRAJ0(10->30). Gap-associated: corroborated / not corroborated / "not computed" if
    F-TRAJ0(10->30) is not computed. Not gap-associated: "not computed" while gap association is
    still possible (a gap contrast not computed, none computed and unsupported); "not applicable"
    if a computed, unsupported gap contrast excludes it."""
    if cls not in ALL_CLASSES:
        raise DiagLatError(INPUT, "unknown class %r" % (cls,))
    if cls in GAP_ASSOCIATED:
        sup = support(labels[FIXED_EPOCH], s)
        if sup == NC:
            return NC
        return "corroborated at fixed epochs" if sup == "supported" else "not corroborated at fixed epochs"
    gap = [support(labels[c], s) for c in READ_GAP]
    if any(v in ("opposite", "unresolved") for v in gap):
        return "not applicable"
    if NC not in gap:
        raise DiagLatError(INPUT, "class %r disagrees with two supported gap contrasts" % (cls,))
    return NC


def coordinate_tag(s, physical_labels):
    """L2, L3a, L3b: the primary statistic in physical norms on S-GAP0 and C-TIME0."""
    sup = [support(physical_labels[c], s) for c in READ_GAP]
    if all(v == "supported" for v in sup):
        return "same direction in physical coordinates"
    if all(v == "opposite" for v in sup):
        return "opposite in physical coordinates"
    return "not resolved in physical coordinates"


def analyse_measure(pop, mid, idx, physical_name=None):
    """All 21 contrasts for one classified measure, with class and tags."""
    name, s, bank = MEASURES[mid]
    subset = "full" if bank else None
    effects = {c: additive_effect(pop, w, name, idx, subset) for c, w in ALL_CONTRASTS.items()}
    labels = {c: e["label"] for c, e in effects.items()}
    cls = classify(s, labels)
    out = {"measure": mid, "key": name, "predicted_sign": s, "effects": effects, "class": cls,
           "seed_tag": seed_tag(effects), "fixed_epoch_tag": fixed_epoch_tag(cls, s, labels),
           "x_fix": effects["X-FIX(10→30)"]}
    if bank:
        per_contrast = {}
        for c, w in ALL_CONTRASTS.items():
            sub = {sb: (labels[c] if sb == "full" else additive_effect(pop, w, name, idx, sb)["label"])
                   for sb in STABILITY_SUBSETS}
            per_contrast[c] = {"labels": sub, "stability": stability_label(sub)}
        out["stability_per_contrast"] = per_contrast
        out["stability_summary"] = ("stable" if all(per_contrast[c]["stability"] == "stable" for c in READ_GAP)
                                    else "Monte Carlo adequacy unresolved")
    else:
        out["stability_per_contrast"] = None
        out["stability_summary"] = "not applicable"
    if mid in COORDINATE_TAGGED:
        if physical_name is None:
            raise DiagLatError(INPUT, "%s needs its physical-coordinate key" % mid)
        phys = {c: additive_effect(pop, ALL_CONTRASTS[c], physical_name, idx, "full") for c in READ_GAP}
        out["physical_effects"] = phys
        out["coordinate_tag"] = coordinate_tag(s, {c: e["label"] for c, e in phys.items()})
    else:
        out["coordinate_tag"] = "not applicable"
    return out


def run_registered(pop, idx, physical_names):
    """Six classified measures, their classes and tags, and the report outcome."""
    res = {m: analyse_measure(pop, m, idx, physical_names.get(m)) for m in MEASURES}
    res_out = outcome({m: {"class": r["class"], "predicted_sign": r["predicted_sign"],
                           "labels": {c: e["label"] for c, e in r["effects"].items()}} for m, r in res.items()})
    return {"rules_version": RULES_VERSION, "measures": res, "outcome": res_out,
            "note": "exploratory classification; unadjusted 95 % intervals; no outcome asserts a cause"}


# ---- L6 (descriptive; per seed) --------------------------------------------------------
def spearman_result(a, b, idx, counts=None):
    """Point rho and bootstrap interval with the 223 section 5 U7 (ii) rule for undefined draws."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size != idx.shape[1]:
        raise DiagLatError(INPUT, "spearman_result: shapes %s, %s, idx %s" % (a.shape, b.shape, idx.shape))
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise DiagLatError(NONFINITE, "spearman_result: non-finite input")
    point = spearman(a, b)
    boot = boot_spearman(a, b, idx, counts)
    undefined = int(np.isnan(boot).sum())
    if math.isnan(point):
        return {"rho": NC, "ci": None, "undefined_draws": undefined, "reason": "a variable is constant"}
    if undefined > UNDEFINED_DRAW_LIMIT * len(boot):
        return {"rho": point, "ci": None, "undefined_draws": undefined,
                "reason": "more than 1 % undefined draws: interval not computed"}
    return {"rho": point, "ci": _percentile_ci(boot[~np.isnan(boot)]), "undefined_draws": undefined, "reason": None}


def share_both_lower(d_nll, d_psnr, idx):
    """Share of volumes with NLL lower and R0 PSNR lower, with its bootstrap interval."""
    ind = ((np.asarray(d_nll, dtype=float) < 0) & (np.asarray(d_psnr, dtype=float) < 0)).astype(float)
    if ind.size != idx.shape[1]:
        raise DiagLatError(INPUT, "share_both_lower: %d volumes for idx %s" % (ind.size, idx.shape))
    return {"share": float(ind.mean()), "ci": _percentile_ci(ind[idx].mean(axis=1))}


def l6_contrast(pop, contrast, seed, idx, tau, counts=None, nll="NLL", psnr="R0_PSNR"):
    """L6 for one read contrast and one seed."""
    w = ALL_CONTRASTS[contrast]
    counts = multiplicities(idx) if counts is None else counts

    def change(name, subset=None):
        return seed_differences(pop, w, name, seed, subset)
    d_nll, d_psnr = change(nll), change(psnr)
    if d_nll is NOT_COMPUTED or d_psnr is NOT_COMPUTED:
        raise DiagLatError(INPUT, "L6 needs NLL and R0 PSNR for every volume of the seed (%s, seed %s)" % (contrast, seed))
    out = {"contrast": contrast, "seed": seed, "share_nll_lower_psnr_lower": share_both_lower(d_nll, d_psnr, idx),
           "measures": {}, "tail": {}}
    for mid, (name, _s, bank) in MEASURES.items():
        d = change(name, "full" if bank else None)
        if d is NOT_COMPUTED:
            out["measures"][mid] = {"vs_psnr": {"rho": NC, "ci": None, "undefined_draws": 0, "reason": "measure not computed"},
                                    "vs_nll": {"rho": NC, "ci": None, "undefined_draws": 0, "reason": "measure not computed"}}
        else:
            out["measures"][mid] = {"vs_psnr": spearman_result(d, d_psnr, idx, counts),
                                    "vs_nll": spearman_result(d, d_nll, idx, counts)}
    d_ks = change("KS")
    for lab, d in (("vs_dKS", d_ks), ("vs_dNLL", d_nll), ("vs_dPSNR", d_psnr)):
        out["tail"][lab] = ({"rho": NC, "ci": None, "undefined_draws": 0, "reason": "measure not computed"}
                            if d is NOT_COMPUTED else spearman_result(tau, d, idx, counts))
    return out

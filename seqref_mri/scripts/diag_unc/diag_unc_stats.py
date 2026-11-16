# SEQREF-DUNC-STATS v0.2 -- D-UNC stage 2: population measures, bootstrap, labels, sensitivity (223 v1.0 + AM1 §3.4-§8)
# LIFETIME: KEEP
"""Stage 2 of the D-UNC extractor (SEQREF-MRI-DIAG-UNC v1.0 + AM1, LudWiki 223).

Works only on the per-slice records retained by stage 1 (diag_unc_core.slice_stats):
population-level measures, paired contrasts, bootstrap intervals, Monte Carlo
sensitivity and labels. It reads no file and imports no V05LT code.

Changelog
- v0.2 (2026-11-16): registered contrast table (Tier 1, Tier 2) and one entry point, run_registered;
  interpretation rows per 223 AM1-3 with three-valued logic (assigned / ruled out / unresolved) and
  deciding effects for row 10; bootstrap count cache removed (counts computed once per call and
  passed explicitly); injectable generator factory so the single-call rule of §3.4 can be verified.
- v0.1 (2026-11-16): first version.
"""
import hashlib
import logging
import math

import numpy as np

from diag_unc_core import LEVELS, N_BANDS, DiagUncError, attainable, spearman

__version__ = "0.2"
LOG = logging.getLogger("diag_unc")

DIAG_UNC_SEED = 20261114
N_BOOT = 10000
N_VOLUMES = 97
PRACTICAL_DB = 0.5          # V05LT practical reference (194 §9.2); context only
CALIBRATION_REFERENCE = 0.05
TIE_FLAG_FRACTION = 0.001
UNDEFINED_DRAW_LIMIT = 0.01
STABILITY_SUBSETS = ("pre64", "full", "halfA", "halfB")
FAIR_SCORES = ("ucrps", "mcrps", "energy")
SPACES = ("U", "M")


# ---------------------------------------------------------------- bootstrap matrix (§3.4)
def _default_generator(seed):
    return np.random.Generator(np.random.PCG64(seed))


def bootstrap_matrix(seed=DIAG_UNC_SEED, n_volumes=N_VOLUMES, n_boot=N_BOOT, generator_factory=_default_generator):
    """The single registered call. No other draw is taken from this generator.

    generator_factory exists only so a fixture can pass an instrumented generator and count the calls.
    """
    return generator_factory(seed).integers(0, n_volumes, size=(n_boot, n_volumes), dtype=np.int64)


def matrix_identity(idx):
    """SHA-256 of the raw bytes as little-endian int64, C order, no header."""
    if not (isinstance(idx, np.ndarray) and idx.ndim == 2 and idx.dtype.kind == "i" and idx.dtype.itemsize == 8):
        raise DiagUncError("DIAG_UNC_BOOTSTRAP_IDENTITY", "index matrix is not a 2-D signed 64-bit integer array")
    return hashlib.sha256(np.ascontiguousarray(idx.astype("<i8", copy=False)).tobytes(order="C")).hexdigest()


def verify_matrix(idx, registered_sha256, n_volumes=N_VOLUMES, n_boot=N_BOOT):
    """Validity gate: a mismatch is a failure, never a reason to regenerate."""
    digest = matrix_identity(idx)
    if idx.shape != (n_boot, n_volumes) or idx.min() < 0 or idx.max() >= n_volumes:
        raise DiagUncError("DIAG_UNC_BOOTSTRAP_IDENTITY", "index matrix has the wrong shape or range")
    if digest != registered_sha256:
        raise DiagUncError("DIAG_UNC_BOOTSTRAP_IDENTITY", "index matrix hash %s differs from the registered %s"
                           % (digest, registered_sha256))
    return digest


def verify_hash(actual, registered, what):
    """Exact identity check for bound inputs (bank, checkpoint, mask)."""
    if actual != registered:
        raise DiagUncError("DIAG_UNC_REPRODUCTION_MISMATCH", "%s hash %s differs from the registered %s"
                           % (what, actual, registered))


# ---------------------------------------------------------------- population
class Population:
    """Stage-1 records for every (checkpoint role, seed), in one fixed volume order."""

    def __init__(self, volume_ids):
        self.volume_ids = list(volume_ids)
        if len(set(self.volume_ids)) != len(self.volume_ids) or self.volume_ids != sorted(self.volume_ids):
            raise DiagUncError("DIAG_UNC_POPULATION_MISMATCH", "volume list must be sorted and unique")
        self.records = {}
        self.sealed = False

    def add(self, role, seed, volume_id, record):
        recs = self.records.setdefault((role, seed), [])
        if len(recs) >= len(self.volume_ids) or self.volume_ids[len(recs)] != volume_id:
            raise DiagUncError("DIAG_UNC_POPULATION_MISMATCH", "unexpected volume %r for %s seed %s"
                               % (volume_id, role, seed))
        recs.append(record)

    def seal(self):
        roles = sorted({r for r, _ in self.records})
        seeds = sorted({s for _, s in self.records})
        for role in roles:
            for seed in seeds:
                if len(self.records.get((role, seed), ())) != len(self.volume_ids):
                    raise DiagUncError("DIAG_UNC_POPULATION_MISMATCH", "incomplete population for %s seed %s"
                                       % (role, seed))
        self.roles, self.seeds, self.sealed = roles, seeds, True
        return self

    def col(self, role, seed, fn, allow_nan=False):
        if not self.sealed:
            raise DiagUncError("DIAG_UNC_POPULATION_MISMATCH", "population used before it was sealed")
        a = np.array([fn(r) for r in self.records[(role, seed)]], dtype=float)
        if not allow_nan and not np.all(np.isfinite(a)):
            raise DiagUncError("DIAG_UNC_NONFINITE", "non-finite stage-1 value for %s seed %s" % (role, seed))
        return a


# ---------------------------------------------------------------- measures
def additive_measures(k_samples):
    """Per-volume measures whose contrasts are means of paired differences."""
    m = {"ucrps": lambda r: r["ucrps"], "mcrps": lambda r: r["mcrps"], "energy": lambda r: r["energy"],
         "sd_U": lambda r: r["sd_mean"]["U"], "sd_M": lambda r: r["sd_mean"]["M"],
         "rho_r0": lambda r: r["rho_r0"], "rho_r1": lambda r: r["rho_r1"]}
    for b, name in enumerate(("low", "mid", "high")[:N_BANDS]):
        m["band_crps_" + name] = (lambda r, b=b: r["band_crps"][b])
    for space in SPACES:
        for name, level in LEVELS:
            if attainable(level, k_samples)[1] is not None:
                m["width_%s_%s" % (space, name)] = (lambda r, s=space, n=name: r[s]["cov"][n][2])
    return m


def nonlinear_measures(k_samples):
    names = ["ssr", "rho_vol"]
    for space in SPACES:
        for name, level in LEVELS:
            if attainable(level, k_samples)[1] is not None:
                names.append("abscov_%s_%s" % (space, name))
    return names


def _percentile_ci(boot):
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return (float(lo), float(hi))


def sign_label(ci):
    if ci is None:
        return "not computed"
    if ci[1] < 0:
        return "below0"
    if ci[0] > 0:
        return "above0"
    return "unresolved"


def _subset_k(pop, subset):
    return pop.records[(pop.roles[0], pop.seeds[0])][0]["subsets"][subset]["K"]


def additive_effect(pop, weights, measure, subset, idx):
    """Paired effect sum_role w * value: per-volume differences, then means (223 §3.4)."""
    fn = additive_measures(_subset_k(pop, subset))[measure]
    is_rho = measure in ("rho_r0", "rho_r1")
    diff = []
    for seed in pop.seeds:
        d = np.zeros(len(pop.volume_ids))
        for role, w in weights.items():
            d = d + w * pop.col(role, seed, lambda r: fn(r["subsets"][subset]), allow_nan=is_rho)
        diff.append(d)
    diff = np.array(diff)
    if np.isnan(diff).any():  # case-level undefined within-slice rho: no substitution, no dropped volume
        return {"point": None, "per_seed": None, "ci": None, "label": "not computed", "undefined_draws": 0}
    per_seed = diff.mean(axis=1)
    boot = diff[:, idx].mean(axis=2).mean(axis=0)
    ci = _percentile_ci(boot)
    return {"point": float(per_seed.mean()), "per_seed": [float(v) for v in per_seed], "ci": ci,
            "label": sign_label(ci), "undefined_draws": 0}


def multiplicities(idx):
    """(B, V) number of times each original volume appears in each replicate. No caching."""
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


def _role_seed_values(pop, role, seed, measure, subset, idx, counts):
    """Point value and (B,) replicate values of a nonlinear aggregate for one checkpoint and seed."""
    sub = lambda r: r["subsets"][subset]
    k = _subset_k(pop, subset)
    if measure == "ssr":
        sig2, e2 = pop.col(role, seed, lambda r: sub(r)["sig2"]), pop.col(role, seed, lambda r: sub(r)["e2"])
        den, den_b = e2.mean(), e2[idx].mean(axis=1)
        if den <= 0 or np.any(den_b <= 0):
            raise DiagUncError("DIAG_UNC_ZERO_DENOMINATOR", "SSR denominator is zero for %s seed %s" % (role, seed))
        f = (k + 1.0) / k
        return math.sqrt(f * sig2.mean()) / math.sqrt(den), np.sqrt(f * sig2[idx].mean(axis=1)) / np.sqrt(den_b)
    if measure == "rho_vol":
        sd, mse = pop.col(role, seed, lambda r: sub(r)["sd_slice"]), pop.col(role, seed, lambda r: sub(r)["mse_r0"])
        return spearman(sd, mse), boot_spearman(sd, mse, idx, counts)
    _, space, name = measure.split("_")
    pi = attainable(dict(LEVELS)[name], k)[1]
    cov = pop.col(role, seed, lambda r: sub(r)[space]["cov"][name][0])
    n = pop.col(role, seed, lambda r: sub(r)[space]["cov"][name][1])
    return abs(cov.sum() / n.sum() - pi), np.abs(cov[idx].sum(axis=1) / n[idx].sum(axis=1) - pi)


def nonlinear_effect(pop, weights, measure, subset, idx, counts=None):
    """Aggregate recomputed per checkpoint and seed inside every replicate, then combined (223 §3.4)."""
    if measure == "rho_vol" and counts is None:
        counts = multiplicities(idx)
    per_seed = np.zeros(len(pop.seeds))
    boot = np.zeros(idx.shape[0])
    for role, w in weights.items():
        for j, seed in enumerate(pop.seeds):
            p, b = _role_seed_values(pop, role, seed, measure, subset, idx, counts)
            per_seed[j] += w * p
            boot += w * b / len(pop.seeds)
    undefined = int(np.isnan(boot).sum())
    if np.isnan(per_seed).any():
        return {"point": None, "per_seed": None, "ci": None, "label": "not computed", "undefined_draws": undefined}
    if undefined > UNDEFINED_DRAW_LIMIT * len(boot):
        ci = None
    else:
        ci = _percentile_ci(boot[~np.isnan(boot)])
    return {"point": float(per_seed.mean()), "per_seed": [float(v) for v in per_seed], "ci": ci,
            "label": sign_label(ci), "undefined_draws": undefined}


# ---------------------------------------------------------------- Monte Carlo sensitivity (§6)
def stability_label(labels):
    """'stable' iff the direction label is identical at K = 64, K = 128 and both halves."""
    vals = [labels.get(s) for s in STABILITY_SUBSETS]
    return "stable" if None not in vals and len(set(vals)) == 1 else "Monte Carlo adequacy unresolved"


def batch_means_se(effects):
    """Mean and SE = sd/sqrt(G) of G group effects (7 degrees of freedom for G = 8)."""
    e = np.asarray(effects, dtype=float)
    return float(e.mean()), float(e.std(ddof=1) / math.sqrt(len(e)))


def group_effects(pop, weights, measure):
    """Population-level paired effect of a fair score from each of the eight disjoint groups of 16."""
    names = sorted(pop.records[(pop.roles[0], pop.seeds[0])][0]["groups"])
    out = []
    for g in names:
        per_seed = [sum(w * pop.col(role, seed, lambda r: r["groups"][g][measure]).mean()
                        for role, w in weights.items()) for seed in pop.seeds]
        out.append(float(np.mean(per_seed)))
    return out


def analyse_contrast(pop, weights, idx):
    """Every measure, subset, interval, label and sensitivity result for one contrast.

    weights: {role: w}; (+1, -1) for a paired contrast, (+1, -1, -1, +1) for an interaction.
    """
    for role in weights:
        if role not in pop.roles:
            raise DiagUncError("DIAG_UNC_POPULATION_MISMATCH", "checkpoint role %r is not in the population" % (role,))
    subsets = sorted(pop.records[(pop.roles[0], pop.seeds[0])][0]["subsets"])
    counts = multiplicities(idx)
    out = {"effects": {}, "stability": {}, "batch_means": {}}
    for subset in subsets:
        k = _subset_k(pop, subset)
        for measure in additive_measures(k):
            out["effects"][(measure, subset)] = additive_effect(pop, weights, measure, subset, idx)
        for measure in nonlinear_measures(k):
            out["effects"][(measure, subset)] = nonlinear_effect(pop, weights, measure, subset, idx, counts)
    for measure in sorted({m for m, _ in out["effects"]}):
        labels = {s: out["effects"][(measure, s)]["label"] for s in subsets if (measure, s) in out["effects"]}
        out["stability"][measure] = stability_label(labels)
    if pop.records[(pop.roles[0], pop.seeds[0])][0]["groups"]:
        for measure in FAIR_SCORES:
            eff = group_effects(pop, weights, measure)
            mean, se = batch_means_se(eff)
            out["batch_means"][measure] = {"effects": eff, "mean": mean, "se": se}
    return out


# ---------------------------------------------------------------- per-checkpoint summaries
def checkpoint_calibration(pop, role, space, level_name, idx, subset="full"):
    """Coverage, signed error with interval, absolute-calibration label, reference label, tie flag."""
    k = _subset_k(pop, subset)
    pi = attainable(dict(LEVELS)[level_name], k)[1]
    if pi is None:
        return {"status": "level not resolvable at K = %d" % k}
    signed, boot, ties, total = [], np.zeros(idx.shape[0]), 0.0, 0.0
    for seed in pop.seeds:
        sub = lambda r: r["subsets"][subset][space]
        cov, n = pop.col(role, seed, lambda r: sub(r)["cov"][level_name][0]), pop.col(role, seed, lambda r: sub(r)["cov"][level_name][1])
        signed.append(cov.sum() / n.sum() - pi)
        boot += (cov[idx].sum(axis=1) / n[idx].sum(axis=1) - pi) / len(pop.seeds)
        ties += pop.col(role, seed, lambda r: sub(r)["ties"]).sum()
        total += pop.col(role, seed, lambda r: sub(r)["n"]).sum()
    ci = _percentile_ci(boot)
    label = {"below0": "over-confident supported", "above0": "under-confident supported"}.get(
        sign_label(ci), "miscalibration not established")
    if -CALIBRATION_REFERENCE <= ci[0] and ci[1] <= CALIBRATION_REFERENCE:
        ref = "within"
    elif ci[0] > CALIBRATION_REFERENCE or ci[1] < -CALIBRATION_REFERENCE:
        ref = "outside"
    else:
        ref = "straddles"
    frac = ties / total
    return {"pi": pi, "signed": float(np.mean(signed)), "coverage": float(np.mean(signed) + pi),
            "per_seed_signed": [float(v) for v in signed], "ci": ci, "label": label, "reference": ref,
            "tie_fraction": float(frac), "ties_material": bool(frac > TIE_FLAG_FRACTION)}


def pooled_deciles(pop, role, subset="full"):
    """Pooled error-vs-SD bins (descriptive figure only): mean SD, mean R0 error, mean R1 error per bin."""
    tot = sum(pop.col(role, seed, lambda r: r["subsets"][subset]["deciles"]).sum(axis=0) for seed in pop.seeds)
    return tot[:, :3] / tot[:, 3:4]


def pooled_histogram(pop, role, space, subset="full"):
    return sum(pop.col(role, seed, lambda r: r["subsets"][subset][space]["hist"]).sum(axis=0) for seed in pop.seeds)


# ---------------------------------------------------------------- D-GAP labels (§4)
def nll_label(ci):
    return {"below0": "improvement supported", "above0": "deterioration supported"}.get(
        sign_label(ci), "direction unresolved")


def psnr_label(ci):
    return {"above0": "improvement supported", "below0": "deterioration supported"}.get(
        sign_label(ci), "direction unresolved")


def practical_label(ci, reference=PRACTICAL_DB):
    """PSNR practical size against the 0.5 dB context reference (not a decision rule)."""
    if ci[0] > reference:
        return "supported"
    if ci[1] < reference:
        return "ruled out at this interval level"
    return "unresolved"


def gap_pattern(nll, psnr_r0):
    return nll == "improvement supported" and psnr_r0 != "improvement supported"


# ---------------------------------------------------------------- interpretation matrix (§8, AM1-3)
LEVEL_NAMES = tuple(name for name, _ in LEVELS)
CELLS = tuple((space, name) for space in SPACES for name in LEVEL_NAMES)
UNRESOLVED_REPORT = "interpretation unresolved — measure not computed"


def _and(*vals):
    """Three-valued AND: False if any is False, else None if any is None, else True."""
    if any(v is False for v in vals):
        return False
    return None if any(v is None for v in vals) else True


def _or(*vals):
    if any(v is True for v in vals):
        return True
    return None if any(v is None for v in vals) else False


def _not(v):
    return None if v is None else (not v)


def _state(label, improving):
    """True = improvement supported; False = improvement not supported (valid interval); None = not computed."""
    if label == "not computed":
        return None
    return label == improving


def matrix_rows(nll, psnr_r0, labels, stability=None, validity_ok=True):
    """Rows of 223 §8 for one contrast, assigned as registered in AM1-3 (K = 128 labels).

    labels: {measure: sign label} for ucrps, mcrps, energy, rho_r0 and, for each of the eight cells,
            abscov_<space>_<level> and width_<space>_<level>.
    stability: {measure: 'stable' | 'Monte Carlo adequacy unresolved'} (§6.4); None = not assessed here.
    Each of rows 1-7 is True (assigned), False (ruled out by the available measures) or None
    (unresolved: a measure that is not computed prevents the decision).
    """
    out = {"rows": [], "unresolved": [], "deciding": [], "report": None, "truth": {}}
    if not validity_ok:
        out["rows"].append(10)
    if nll == "direction unresolved":
        out["rows"].append(9)
        out["rows"].sort()
        return out
    if nll == "improvement supported" and psnr_r0 == "improvement supported":
        out["rows"].append(8)
        out["rows"].sort()
        return out
    if not gap_pattern(nll, psnr_r0):
        return out
    u, m = _state(labels["ucrps"], "below0"), _state(labels["mcrps"], "below0")
    energy, assoc = _state(labels["energy"], "below0"), _state(labels["rho_r0"], "above0")
    crps = {"U": u, "M": m}
    falls = {c: _state(labels["abscov_%s_%s" % c], "below0") for c in CELLS}
    rises = {c: _state(labels["width_%s_%s" % c], "above0") for c in CELLS}
    bought = {c: _and(falls[c], rises[c]) for c in CELLS}
    any_cov = _or(*falls.values())
    cov_names = ["abscov_%s_%s" % c for c in CELLS]
    width_names = ["width_%s_%s" % c for c in CELLS]
    space_cov = lambda s: ["abscov_%s_%s" % c for c in CELLS if c[0] == s]
    row3 = {s: _and(crps[s], _or(*[_not(falls[c]) for c in CELLS if c[0] == s])) for s in SPACES}
    row4 = {c: _and(bought[c], _not(crps[c[0]])) for c in CELLS}
    truth = {
        1: _and(u, m, _not(_or(*bought.values()))),
        2: _and(u, _not(m)),
        3: _or(*row3.values()),
        4: _or(*row4.values()),
        5: _and(energy, _not(u), _not(m)),
        6: _and(assoc, _not(u), _not(m), _not(energy), _not(any_cov)),
        7: _and(_not(u), _not(m), _not(energy), _not(any_cov), _not(assoc)),
    }
    crps_name = {"U": "ucrps", "M": "mcrps"}
    refers = {
        1: ["ucrps", "mcrps"] + cov_names + width_names,
        2: ["ucrps", "mcrps"],
        3: sorted({n for s in SPACES if row3[s] is not False for n in [crps_name[s]] + space_cov(s)}),
        4: sorted({n for c in CELLS if row4[c] is not False
                   for n in ("abscov_%s_%s" % c, "width_%s_%s" % c, crps_name[c[0]])}),
        5: ["energy", "ucrps", "mcrps"],
        6: ["rho_r0", "ucrps", "mcrps", "energy"] + cov_names,
        7: ["ucrps", "mcrps", "energy", "rho_r0"] + cov_names,
    }
    deciding = {"ucrps", "mcrps"}
    for row in range(1, 8):
        if truth[row] is True:
            out["rows"].append(row)
            deciding.update(refers[row])
        elif truth[row] is None:
            missing = sorted(n for n in refers[row] if labels[n] == "not computed")
            out["unresolved"].append((row, missing))
    out["truth"] = truth
    out["deciding"] = sorted(deciding)
    if out["unresolved"]:
        out["report"] = UNRESOLVED_REPORT
    mc_unresolved = stability is not None and any(stability.get(n) != "stable" for n in deciding)
    if (out["unresolved"] or mc_unresolved) and 10 not in out["rows"]:
        out["rows"].append(10)
    out["rows"].sort()
    return out


def cross_rows(dgap, xtime_labels, tier2_labels=()):
    """Rows of 223 §8.1. dgap: {contrast: (nll label, R0 psnr label)} with S-GAP0, C-TIME0, C-TIME2."""
    rows = []
    both = lambda c: dgap[c] == ("improvement supported", "improvement supported")
    if gap_pattern(*dgap["S-GAP0"]) and gap_pattern(*dgap["C-TIME0"]) and both("C-TIME2"):
        rows.append("X1")
    if gap_pattern(*dgap["C-TIME2"]):
        rows.append("X2")
    x3 = any(lab in ("below0", "above0") for lab in xtime_labels)
    x5 = any(lab in ("below0", "above0") for lab in tier2_labels)
    if x3:
        rows.append("X3")
    if x5:
        rows.append("X5")
    if not (x3 or x5):
        rows.append("X4")
    return rows


# ---------------------------------------------------------------- registered contrasts (§1)
ARMS = ("lam0", "lam2")
TIER1_STAGES = ("e3", "sel", "fin")
TIER2_STAGES = ("e10", "e30")


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
PROBABILISTIC = ("ucrps", "mcrps", "ssr") + tuple("abscov_%s_%s" % c for c in CELLS) + tuple(
    "width_%s_%s" % c for c in CELLS)


def registered_roles(tier2):
    stages = TIER1_STAGES + (TIER2_STAGES if tier2 else ())
    return [arm + "/" + stage for arm in ARMS for stage in stages]


def run_registered(pop, idx, tier2=False, dgap=None, validity_ok=True):
    """Every registered D-UNC result for a sealed population (the single entry point of stage 2).

    pop   roles named '<arm>/<stage>' with arm in lam0, lam2 and stage in e3, sel, fin (+ e10, e30 for Tier 2)
    idx   the verified bootstrap index matrix
    dgap  {contrast: {"nll": label, "psnr_r0": label}} from the D-GAP step (not part of this module);
          without it no interpretation row is assigned and the result says so.
    Cross rows X3/X5 use the probabilistic measures named in 223 §8.1 (CRPS, coverage error, width, SSR);
    the NLL-PSNR pair of X5 belongs to the D-GAP step and is not evaluated here.
    """
    needed = registered_roles(tier2)
    missing = [r for r in needed if r not in pop.roles]
    if missing:
        raise DiagUncError("DIAG_UNC_POPULATION_MISMATCH", "registered checkpoint roles missing: %s" % missing)
    if idx.shape[1] != len(pop.volume_ids):
        raise DiagUncError("DIAG_UNC_POPULATION_MISMATCH", "index matrix width does not match the volume count")
    contrasts = dict(TIER1_CONTRASTS)
    if tier2:
        contrasts.update(TIER2_CONTRASTS)
    out = {"tier2": bool(tier2), "contrasts": {}, "checkpoints": {}, "cross": None}
    for name, weights in contrasts.items():
        res = analyse_contrast(pop, weights, idx)
        if dgap is not None and name in dgap:
            labels = {m: res["effects"][(m, "full")]["label"] for (m, sub) in res["effects"] if sub == "full"}
            res["interpretation"] = matrix_rows(dgap[name]["nll"], dgap[name]["psnr_r0"], labels,
                                                stability=res["stability"], validity_ok=validity_ok)
        else:
            res["interpretation"] = "not assigned: D-GAP labels not supplied"
        out["contrasts"][name] = res
    for role in needed:
        out["checkpoints"][role] = {
            "calibration": {c: checkpoint_calibration(pop, role, c[0], c[1], idx) for c in CELLS},
            "deciles": pooled_deciles(pop, role),
            "histogram": {s: pooled_histogram(pop, role, s) for s in SPACES}}
    if dgap is not None and all(c in dgap for c in ("S-GAP0", "C-TIME0", "C-TIME2")):
        lab = lambda c: [out["contrasts"][c]["effects"][(m, "full")]["label"] for m in PROBABILISTIC]
        t2 = [l for c in TIER2_CONTRASTS if c.startswith(("C-OBJ@", "X-FIX")) for l in lab(c)] if tier2 else []
        out["cross"] = cross_rows({c: (dgap[c]["nll"], dgap[c]["psnr_r0"]) for c in ("S-GAP0", "C-TIME0", "C-TIME2")},
                                  lab("X-TIME"), t2)
    return out

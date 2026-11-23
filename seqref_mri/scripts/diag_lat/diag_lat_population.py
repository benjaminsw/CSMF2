# SEQREF-DLAT-POP v0.1 -- diag_lat_population.py: D-LAT population stage (plan SEQREF-DLAT-POP v0.1, steps P1-P9)
# LIFETIME: KEEP
"""Population stage of D-LAT: from the per-slice records of the 28 model states to every
population-level result of the frozen specification SEQREF-MRI-DIAG-LAT v1.0 (LudWiki 235).

Plan SEQREF-DLAT-POP v0.1 with decisions Q-P1 to Q-P5 and clarifications C1, C2 (LudWiki 248
LE76, LE77). Pure NumPy. No model, no file access, no random draw: the registered bootstrap
index matrix is an argument and is verified against its registered sha256 before use.

What is classified: the six measures of diag_lat_stats.MEASURES, through run_registered,
unchanged. Everything else here is REPORTED, NOT CLASSIFIED:
  reported()     every reported key on the 21 contrasts: effect, 95 % interval, per-seed effects
                 (no direction label: Q-P2)
  state_means()  mean over the volumes per role and seed, and pooled over seeds; descriptive,
                 no interval (Q-P3)
  e_cov_table()  E_cov per model state from the stored z_proj, beside the analytic reference;
                 no contrast, no interval (Q-P4)
  l6_tables()    section 4 quantities on the seven read contrasts, per seed (Q-P1)
  tail_table()   tau_i and q_tail_i over the volumes
  events_table() every "not computed" state and every adapter event
The per-figure data files of the figure plan are NOT written here (Q-P5): they belong to the
G runner.

Deduplicated states (C1): the measurements of a model state are held once, under its state id.
role_map gives the state id of every (role, seed). Two roles of one seed may share a state;
both roles then read the same values, and every contrast keeps all its roles.

"not computed" follows the Population rule of diag_lat_stats (a state with any volume not
computed is not computed for that key) and is never replaced by a number. No fallback,
placeholder, mock or dummy return. Every failure is a typed DiagLatError, logged at construction.

Changelog:
  v0.1 (2026-11-23) first version: P1 build_population, P2 classified, P3 reported,
        P4 state_means, P5 e_cov_table, P6 l6_tables, P7 tail_table, P8 events_table,
        P9 run_population with the main table.
"""
import logging

import numpy as np

import diag_lat_adapter as A
import diag_lat_core as C
import diag_lat_stats as ST
from diag_lat_core import DiagLatError

__version__ = "0.1"
ABBR = "SEQREF-DLAT-POP"
LOG = logging.getLogger("diag_lat.population")

POPULATION_MISMATCH = ST.POPULATION_MISMATCH
NC = ST.NC
# 235 section 1: gap contrasts (read), controls (read), interactions (read) -- seven in all
READ_CONTRASTS = ("S-GAP0", "C-TIME0", "F-TRAJ0(10→30)", "S-CTL0", "C-TIME2", "X-TIME", "X-FIX(10→30)")
MAIN_TABLE_CONTRASTS = ("S-GAP0", "C-TIME0", "X-TIME", "X-FIX(10→30)")          # figure plan v0.2 section 3
ROLES = tuple(sorted({r for w in ST.ALL_CONTRASTS.values() for r in w}))
TAIL_KEYS = ("tau", "q_tail")                   # fixed per slice; not read on contrasts (235 section 3)
INPUT_KEYS = ("NLL", "R0_PSNR")                 # stored rows, inputs of L6; not D-LAT measures
CLASSIFIED_KEYS = tuple(m[0] for m in ST.MEASURES.values())
SCALAR_KEYS = tuple(k for k in A.RECORD_KEYS if k["type"] != "float[8]")
REPORTED_KEYS = tuple(k for k in SCALAR_KEYS
                      if k["name"] not in CLASSIFIED_KEYS + TAIL_KEYS + INPUT_KEYS)
MEAN_KEYS = tuple(k for k in SCALAR_KEYS if k["name"] not in TAIL_KEYS)


def _stop(code, message):
    raise DiagLatError(code, message)            # DiagLatError logs at construction


def _subset(k):
    return "full" if k["bank"] else None


# ---- P1 ---------------------------------------------------------------------------------------
def build_population(records, role_map, volume_ids, seeds, labels=ST.STABILITY_SUBSETS):
    """records: {state_id: [one validated record per volume, in the order of volume_ids]};
    role_map: {(role, seed): state_id} for every registered role and every seed.
    Returns {"pop": sealed Population, "z_proj": {state_id: (V, 8) array}, "tau", "q_tail",
    "states": sorted state ids, "role_map"}."""
    volume_ids, seeds, labels = list(volume_ids), list(seeds), list(labels)
    if len(set(seeds)) != len(seeds) or not seeds:
        _stop(C.INPUT, "seeds must be a non-empty list of distinct values, got %r" % (seeds,))
    want = {(r, s) for r in ROLES for s in seeds}
    if not isinstance(role_map, dict) or set(role_map) != want:
        got = set(role_map) if isinstance(role_map, dict) else set()
        _stop(POPULATION_MISMATCH, "role map: missing %s, unexpected %s"
              % (sorted(want - got, key=repr), sorted(got - want, key=repr)))
    if not isinstance(records, dict):
        _stop(C.INPUT, "records must be a dict {state id: list of records}")
    used = set(role_map.values())
    if used != set(records):
        _stop(POPULATION_MISMATCH, "states: in the role map but without records %s; with records but in no role %s"
              % (sorted(used - set(records), key=repr), sorted(set(records) - used, key=repr)))
    seed_of = {}
    for (role, seed), sid in sorted(role_map.items(), key=repr):
        if seed_of.setdefault(sid, seed) != seed:
            _stop(POPULATION_MISMATCH, "state %r is mapped to two seeds (%r and %r)" % (sid, seed_of[sid], seed))
    for sid, recs in records.items():
        if not isinstance(recs, (list, tuple)) or len(recs) != len(volume_ids):
            _stop(POPULATION_MISMATCH, "state %r: %s records for %d volumes"
                  % (sid, len(recs) if isinstance(recs, (list, tuple)) else type(recs).__name__, len(volume_ids)))
        for rec in recs:
            A.validate_record(rec, labels)
    pop = ST.Population(volume_ids)
    for (role, seed), sid in sorted(role_map.items(), key=repr):
        recs = records[sid]
        for k in SCALAR_KEYS:
            if k["bank"]:
                for lab in labels:
                    pop.set(role, seed, k["name"], [r[A.BANK_FIELD][lab][k["name"]] for r in recs], lab)
            else:
                pop.set(role, seed, k["name"], [r[k["name"]] for r in recs])
    pop.seal()
    states = sorted(records, key=repr)
    z_proj = {sid: np.array([r["z_proj"] for r in records[sid]], dtype=np.float64) for sid in states}   # validated above
    tail = {}
    for name in TAIL_KEYS:
        ref = np.array([r[name] for r in records[states[0]]], dtype=float)
        for sid in states[1:]:
            if not np.array_equal(ref, np.array([r[name] for r in records[sid]], dtype=float)):
                _stop(POPULATION_MISMATCH, "%s differs between states %r and %r; it is fixed per slice by construction"
                      % (name, states[0], sid))
        tail[name] = ref
    return {"pop": pop, "z_proj": z_proj, "tau": tail["tau"], "q_tail": tail["q_tail"], "states": states,
            "role_map": dict(role_map), "seeds": seeds, "labels": labels}


# ---- P2 ---------------------------------------------------------------------------------------
def classified(pop, idx):
    """The six classified measures on the 21 contrasts, with classes, tags and outcome. Unchanged rules."""
    return ST.run_registered(pop, idx, A.PHYSICAL_NAMES)


# ---- P3 ---------------------------------------------------------------------------------------
def _plain(effect):
    """Effect, interval and per-seed effects; the direction label is not carried (Q-P2)."""
    if effect["label"] == NC:
        return {"point": NC, "ci": None, "per_seed": None}
    return {"point": effect["point"], "ci": [float(effect["ci"][0]), float(effect["ci"][1])],
            "per_seed": list(effect["per_seed"])}


def reported(pop, idx):
    """{key: {"bank", "subset", "contrasts": {contrast: {"point", "ci", "per_seed"}}}} for every reported key."""
    out = {}
    for k in REPORTED_KEYS:
        sub = _subset(k)
        out[k["name"]] = {"bank": k["bank"], "subset": sub,
                          "contrasts": {c: _plain(ST.additive_effect(pop, w, k["name"], idx, sub))
                                        for c, w in ST.ALL_CONTRASTS.items()}}
    return out


# ---- P4 ---------------------------------------------------------------------------------------
def state_means(pop):
    """Mean over the volumes of every key (bank keys on "full"), per role and seed and pooled over
    seeds. Descriptive: no interval (Q-P3). A state not computed gives "not computed", also pooled."""
    out = {}
    for k in MEAN_KEYS:
        sub = _subset(k)
        rows = {}
        for role in pop.roles:
            per_seed = {}
            for seed in pop.seeds:
                v = pop.col(role, seed, k["name"], sub)
                per_seed[str(seed)] = NC if v is C.NOT_COMPUTED else float(v.mean())
            vals = list(per_seed.values())
            rows[role] = {"per_seed": per_seed,
                          "pooled": NC if any(v == NC for v in vals) else float(np.mean(vals))}
        out[k["name"]] = {"bank": k["bank"], "subset": sub, "roles": rows}
    return out


# ---- P5 ---------------------------------------------------------------------------------------
def e_cov_table(built):
    """E_cov per model state from the stored projections (V x M_PROJ), beside the analytic
    reference. Descriptive: per state only (Q-P4). The full truth codes are not needed."""
    eye = np.eye(C.M_PROJ)
    n = len(built["pop"].volume_ids)
    per_state = {}
    for sid in built["states"]:
        y = built["z_proj"][sid]
        if y.shape != (n, C.M_PROJ):
            _stop(POPULATION_MISMATCH, "z_proj of state %r has shape %s, expected (%d, %d)" % (sid, y.shape, n, C.M_PROJ))
        per_state[str(sid)] = C.projected_cov_mismatch(y, eye)
    by_role = {role: {str(seed): per_state[str(built["role_map"][(role, seed)])] for seed in built["seeds"]}
               for role in ROLES}
    return {"reference": C.e_cov_reference(n, C.M_PROJ), "n_volumes": n, "m": C.M_PROJ,
            "per_state": per_state, "by_role_and_seed": by_role}


# ---- P6 ---------------------------------------------------------------------------------------
def l6_tables(built, idx):
    """235 section 4 on each of the seven read contrasts, per seed (Q-P1)."""
    pop = built["pop"]
    counts = ST.multiplicities(idx)
    return {c: {str(seed): ST.l6_contrast(pop, c, seed, idx, built["tau"], counts) for seed in pop.seeds}
            for c in READ_CONTRASTS}


# ---- P7 ---------------------------------------------------------------------------------------
def tail_table(built):
    tau, q = built["tau"], built["q_tail"]
    return {"volume_ids": list(built["pop"].volume_ids), "tau": [float(v) for v in tau], "q_tail": [float(v) for v in q],
            "tau_summary": {"min": float(tau.min()), "median": float(np.median(tau)), "max": float(tau.max()),
                            "volumes_with_empty_tail": int(np.count_nonzero(tau == 0.0))},
            "tau_constant": bool(np.all(tau == tau[0]))}


# ---- P8 ---------------------------------------------------------------------------------------
def events_table(built, events):
    """Every state that is "not computed" for a key, and the adapter events handed in by the driver."""
    if not isinstance(events, list) or any(not isinstance(e, dict) for e in events):
        _stop(C.INPUT, "events must be a list of dicts")
    nc = [{"role": role, "seed": seed, "key": key[0], "subset": key[1], "volumes_not_computed": n,
           "state": built["role_map"][(role, seed)]}
          for (role, seed, key), n in sorted(built["pop"].not_computed.items(), key=repr)]
    return {"not_computed_states": nc, "adapter_events": list(events), "n_adapter_events": len(events)}


# ---- P9 ---------------------------------------------------------------------------------------
def main_table(cls):
    """Figure plan v0.2 section 3: one row per classified measure, in the fixed order."""
    rows = []
    for mid in ST.MEASURES:
        m = cls["measures"][mid]
        row = {"measure": mid, "key": m["key"], "predicted_sign": m["predicted_sign"], "class": m["class"],
               "fixed_epoch_tag": m["fixed_epoch_tag"], "seed_tag": m["seed_tag"],
               "stability_summary": m["stability_summary"], "coordinate_tag": m["coordinate_tag"], "effects": {}}
        for c in MAIN_TABLE_CONTRASTS:
            e = m["effects"][c]
            row["effects"][c] = dict(_plain(e), label=e["label"])
        rows.append(row)
    return {"rows": rows, "outcome": cls["outcome"], "rules_version": cls["rules_version"]}


def run_population(records, role_map, volume_ids, seeds, idx, registered_matrix_sha256, expected_states, events,
                   n_boot=ST.N_BOOT):
    """P1-P9. idx is verified against registered_matrix_sha256 before any use; the number of
    distinct model states must equal expected_states (28 in production, given by the driver)."""
    volume_ids = list(volume_ids)
    digest = ST.verify_matrix(idx, registered_matrix_sha256, n_volumes=len(volume_ids), n_boot=n_boot)
    built = build_population(records, role_map, volume_ids, seeds)
    if len(built["states"]) != expected_states:
        _stop(POPULATION_MISMATCH, "%d distinct model states, expected %d" % (len(built["states"]), expected_states))
    for c in READ_CONTRASTS + MAIN_TABLE_CONTRASTS:
        if c not in ST.ALL_CONTRASTS:
            _stop(C.INPUT, "contrast %r is not registered" % c)
    cls = classified(built["pop"], idx)
    LOG.info("population stage: %d states, %d role-seed columns, %d volumes, %d draws",
             len(built["states"]), len(role_map), len(volume_ids), idx.shape[0])
    return {"population_version": __version__, "stats_version": ST.__version__, "adapter_version": A.__version__,
            "core_version": C.__version__, "rules_version": cls["rules_version"],
            "identity": {"bootstrap_matrix_sha256": digest, "n_boot": int(idx.shape[0]), "n_volumes": len(volume_ids),
                         "seeds": list(seeds), "roles": list(ROLES), "n_states": len(built["states"]),
                         "states": [str(s) for s in built["states"]],
                         "role_map": {"%s|%s" % (r, s): str(sid) for (r, s), sid in sorted(role_map.items(), key=repr)},
                         "contrasts": list(ST.ALL_CONTRASTS), "read_contrasts": list(READ_CONTRASTS)},
            "classified": cls, "main_table": main_table(cls), "reported": reported(built["pop"], idx),
            "state_means": state_means(built["pop"]), "e_cov": e_cov_table(built),
            "l6": l6_tables(built, idx), "tail": tail_table(built), "events": events_table(built, events)}

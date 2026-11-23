# SEQREF-DLAT-BIND v0.1 -- diag_lat_bind.py: D-LAT production binding to the registered D-UNC F2 manifest (plan SEQREF-DLAT-F1D v0.1, stages D2-D3, hash level)
# LIFETIME: KEEP
"""Binding of D-LAT to the 28 V05LT model states, the 97 validation slices and the stored rows,
through the REGISTERED D-UNC execution manifest (manifest_f2.json, schema seqref-dunc-f2-manifest/1,
sha256 d1956bf3...6215; LudWiki 230, 231, 236). Field names and file locations are those read in
the manifest and in diag_unc_g.py g1_bind (sha256 9ad5f6d6...0ff2); none is guessed (249 LE81).

Two functions, both WITHOUT loading a checkpoint tensor, importing PyTorch or project code, or
running a model (condition C6): they read JSON and hash files. "Read-only" here means read-only
with respect to the existing evidence and the checkpoints; this module writes nothing.

  bind_manifest()  the manifest alone: identity, schema, pin; the 30 role assignments resolved to
                   exactly 28 model states (condition C3); the 97 slice identities
  verify_files()   against the V05LT results folder: every role file by sha256 and sidecar; the
                   eligibility manifest and the midpoint positions; the stored validation rows of
                   the 28 states by sha256, schema, state and order

Role names: D-UNC "epoch3, epoch10, epoch30, selected, final" are the D-LAT roles
"e3, e10, e30, sel, fin" (diag_unc_g.ROLE_SHORT); a D-LAT role is "<arm>/<short>".
A model state is identified by its model_state_sha256. Where two roles of one seed share a
state, the state is held once and both roles point to it. Where a state has two files (a
selected epoch that is also the final one), every file is hashed and the file of the first role
in manifest order is the one named for loading; the pinned loader verifies the state hash.

The stored rows are EXISTING V05LT results (NLL, B, J, R0 PSNR per slice). verify_files returns
them to the driver for the fixtures; they are not D-LAT measures and are not written by this module.

No fallback, placeholder, mock or dummy return. Every failure is a typed DiagLatError, logged.

Changelog:
  v0.1 (2026-11-23) first version: bind_manifest, verify_files.
"""
import hashlib
import json
import logging
import math
import os
import re

import numpy as np

import diag_lat_core as C
import diag_lat_stats as ST
from diag_lat_core import DiagLatError

__version__ = "0.1"
ABBR = "SEQREF-DLAT-BIND"
LOG = logging.getLogger("diag_lat.bind")

ORIGIN = "DIAG_LAT_ORIGIN"
POPULATION_MISMATCH = ST.POPULATION_MISMATCH
REGISTERED_F2_SHA256 = "d1956bf370dc6c1cc82caf9804dc9731cbec970c0f8205265c14bdf72e3a6215"
F2_SCHEMA = "seqref-dunc-f2-manifest/1"
ROWS_SCHEMA = "seqref-v05lt-val-rows/1"
PIN = "7455da8027f33a6cca4dbf72f04e72420e938f45"
RESULTS_SUBDIR = ("seqref_mri", "results", "v05lt")          # diag_unc_g.py line 1142
ROLE_SHORT = {"epoch3": "e3", "selected": "sel", "final": "fin", "epoch10": "e10", "epoch30": "e30"}
ARMS = ("lam0", "lam2")
ROLES = tuple(sorted({r for w in ST.ALL_CONTRASTS.values() for r in w}))
N_STATES, N_ROLES, N_VOLUMES = 28, 30, ST.N_VOLUMES
ROW_FIELDS = ("psnr", "nll_per_dim", "nll_prior_per_dim", "nll_logdet_per_dim")      # R0 PSNR, NLL, B, J
ROW_NAMES = ("R0_PSNR", "NLL", "B", "J")
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")


def _stop(code, message):
    raise DiagLatError(code, message)            # DiagLatError logs at construction


def sha_file(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    except OSError as exc:
        _stop(ORIGIN, "cannot read %s: %s" % (path, exc))
    return h.hexdigest()


def sidecar(path):
    """First 64-hex token of <path>.sha256, or None (the V05LT sidecar convention)."""
    sc = path + ".sha256"
    if not os.path.isfile(sc):
        return None
    with open(sc, "r", encoding="utf-8", errors="replace") as fh:
        m = HEX64.search(fh.read())
    return m.group(0) if m else None


def _load_json(path):
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError) as exc:
        _stop(ORIGIN, "cannot read JSON %s: %s" % (path, exc))


def _is_hex(v):
    return isinstance(v, str) and HEX64.fullmatch(v) is not None


# ---- the manifest alone ---------------------------------------------------------------------
def bind_manifest(manifest_path, f2_sha256=REGISTERED_F2_SHA256, n_states=N_STATES, n_volumes=N_VOLUMES, pin=PIN):
    """Identity and structure of the D-UNC F2 manifest; the role table resolved to model states."""
    mf, sha = _load_json(manifest_path)
    if sha != f2_sha256:
        _stop(ORIGIN, "manifest %s has sha256 %s, registered %s" % (manifest_path, sha, f2_sha256))
    if not isinstance(mf, dict) or mf.get("schema") != F2_SCHEMA or mf.get("selftest") is not False or mf.get("pin") != pin:
        _stop(ORIGIN, "manifest schema, selftest flag or pin differ from the registered values (schema %r, selftest %r, pin %r)"
              % (mf.get("schema") if isinstance(mf, dict) else None, mf.get("selftest") if isinstance(mf, dict) else None,
                 mf.get("pin") if isinstance(mf, dict) else None))
    roles_in = mf.get("roles")
    if not isinstance(roles_in, list) or len(roles_in) != N_ROLES:
        _stop(POPULATION_MISMATCH, "manifest holds %s role assignments, expected %d"
              % (len(roles_in) if isinstance(roles_in, list) else "no", N_ROLES))
    role_map, states, files = {}, {}, {}
    for r in roles_in:
        ok = (isinstance(r, dict) and r.get("arm") in ARMS and r.get("role") in ROLE_SHORT and isinstance(r.get("seed"), int)
              and isinstance(r.get("epoch"), int) and r["epoch"] >= 1 and _is_hex(r.get("file_sha256"))
              and _is_hex(r.get("model_state_sha256")) and isinstance(r.get("path"), str) and r["path"]
              and not os.path.isabs(r["path"]) and ".." not in r["path"].split("/"))
        if not ok:
            _stop(POPULATION_MISMATCH, "malformed role assignment: %r" % (r,))
        key = ("%s/%s" % (r["arm"], ROLE_SHORT[r["role"]]), r["seed"])
        if key in role_map:
            _stop(POPULATION_MISMATCH, "role assignment %r occurs twice" % (key,))
        sid = r["model_state_sha256"]
        role_map[key] = sid
        st = states.setdefault(sid, {"state": sid, "arm": r["arm"], "seed": r["seed"], "epoch": r["epoch"], "path": r["path"],
                                     "file_sha256": r["file_sha256"], "roles": [], "files": {}})
        if (st["arm"], st["seed"], st["epoch"]) != (r["arm"], r["seed"], r["epoch"]):
            _stop(POPULATION_MISMATCH, "state %s is assigned to (%s, %d, epoch %d) and to (%s, %d, epoch %d)"
                  % (sid[:12], st["arm"], st["seed"], st["epoch"], r["arm"], r["seed"], r["epoch"]))
        st["roles"].append("%s|%d|%s" % (r["arm"], r["seed"], r["role"]))
        st["files"][r["path"]] = r["file_sha256"]
        if files.setdefault(r["path"], (r["file_sha256"], sid)) != (r["file_sha256"], sid):
            _stop(POPULATION_MISMATCH, "file %s is listed with two hashes or two model states" % r["path"])
    seeds = sorted({s for _r, s in role_map})
    if set(role_map) != {(r, s) for r in ROLES for s in seeds} or len(seeds) * len(ROLES) != N_ROLES:
        _stop(POPULATION_MISMATCH, "role assignments are not the %d registered roles for each of %d seeds (seeds %s)"
              % (len(ROLES), N_ROLES // len(ROLES), seeds))
    if len(states) != n_states:
        _stop(POPULATION_MISMATCH, "%d role assignments resolve to %d model states, expected %d" % (N_ROLES, len(states), n_states))
    table = mf.get("states")
    ok = (isinstance(table, list) and len(table) == n_states and [t.get("index") for t in table] == list(range(n_states))
          and {t.get("model_state_sha256") for t in table} == set(states)
          and all(sorted(t.get("roles") or []) == sorted(states[t["model_state_sha256"]]["roles"]) for t in table))
    if not ok:
        _stop(POPULATION_MISMATCH, "the state table of the manifest does not agree with its role table")
    order = [t["model_state_sha256"] for t in table]                 # registered processing order
    pop = mf.get("population") or {}
    ids = pop.get("identities")
    ok = (isinstance(ids, list) and len(ids) == n_volumes == pop.get("n_midpoint")
          and all(isinstance(x, list) and len(x) == 2 and isinstance(x[0], str) and isinstance(x[1], int) for x in ids)
          and len({tuple(x) for x in ids}) == n_volumes and len({x[0] for x in ids}) == n_volumes
          and _is_hex(pop.get("regenerated_masks_sha256")) and _is_hex(pop.get("validation_order_sha256")))
    if not ok:
        _stop(POPULATION_MISMATCH, "manifest population: not %d distinct midpoint slices, one per file, with mask and order hashes" % n_volumes)
    inputs = mf.get("inputs") or {}
    rows = inputs.get("stored_rows") or {}
    want = {"%s|%d|e%03d" % (st["arm"], st["seed"], st["epoch"]) for st in states.values()}
    if set(rows) != want or not all(_is_hex(v) for v in rows.values()):
        _stop(POPULATION_MISMATCH, "stored-row entries of the manifest are not exactly those of the %d states" % n_states)
    for k in ("eligibility_sha256", "launch_inputs_sha256", "split_sha256", "p4_sha256", "p0s_sha256", "z_diag_sha256", "z_manifest_sha256"):
        if not _is_hex(inputs.get(k)):
            _stop(ORIGIN, "manifest inputs.%s is missing or not a sha256" % k)
    LOG.info("manifest bound: %d roles -> %d states, %d seeds, %d slices", N_ROLES, len(states), len(seeds), n_volumes)
    return {"manifest_sha256": sha, "manifest": mf, "seeds": seeds, "role_map": role_map, "states": states, "state_order": order,
            "identities": [tuple(x) for x in ids], "inputs": inputs,
            "regenerated_masks_sha256": pop["regenerated_masks_sha256"], "validation_order_sha256": pop["validation_order_sha256"],
            "n_files": len(files), "shared_states": sorted(s for s, st in states.items() if len(st["roles"]) > 1)}


# ---- against the results folder ---------------------------------------------------------------
def results_root(repo):
    return os.path.join(repo, *RESULTS_SUBDIR)


def _hashed(path, want, what):
    if not os.path.isfile(path):
        _stop(ORIGIN, "%s is missing: %s" % (what, path))
    got, sc = sha_file(path), sidecar(path)
    if got != want or sc != want:
        _stop(ORIGIN, "%s: sha256 %s, sidecar %s, registered %s (%s)" % (what, got, sc, want, path))
    return got


def verify_files(bound, root):
    """Every role file, the eligibility manifest, the launch inputs and the stored rows of the 28
    states, against the bound manifest. Returns {"stored": {state: {name: (V,) float64}}, "volume_ids",
    "volume_order", "eligible", "n_rows_per_file", "files_hashed"}. Values in manifest slice order."""
    if not os.path.isdir(root):
        _stop(ORIGIN, "results folder not found: %s" % root)
    inputs, n_vol = bound["inputs"], len(bound["identities"])
    hashed = 0
    for st in bound["states"].values():
        for rel, want in sorted(st["files"].items()):
            _hashed(os.path.join(root, rel), want, "checkpoint file of state %s" % st["state"][:12])
            hashed += 1
    li = os.path.join(root, "launch", "v05lt_launch_inputs.json")
    if not os.path.isfile(li) or sha_file(li) != inputs["launch_inputs_sha256"]:
        _stop(ORIGIN, "launch inputs differ from the manifest or are missing: %s" % li)
    el_path = os.path.join(root, "inputs", "v05lt_nmse_eligibility.json")
    _hashed(el_path, inputs["eligibility_sha256"], "eligibility manifest")
    doc, _sha = _load_json(el_path)
    slices = doc.get("slices") if isinstance(doc, dict) else None
    if not isinstance(slices, list) or not slices:
        _stop(POPULATION_MISMATCH, "eligibility manifest holds no slice list")
    entries = [(x.get("file"), x.get("slice_index")) for x in slices]
    mid_pos = [q for q, x in enumerate(slices) if x.get("midpoint") is True]
    vols = [slices[q].get("volume_id") for q in mid_pos]
    ok = ([entries[q] for q in mid_pos] == bound["identities"] and len(set(vols)) == n_vol
          and all(isinstance(v, str) and v for v in vols) and all(isinstance(slices[q].get("eligible"), bool) for q in mid_pos)
          and (doc.get("provenance") or {}).get("validation_order_sha256") == bound["validation_order_sha256"])
    if not ok:
        _stop(POPULATION_MISMATCH, "eligibility manifest: the midpoint slices, their order, the volume ids or the order hash differ from the F2 manifest")
    stored = {}
    for sid in bound["state_order"]:
        st = bound["states"][sid]
        key = "%s|%d|e%03d" % (st["arm"], st["seed"], st["epoch"])
        p = os.path.join(root, "%d_%s" % (st["seed"], st["arm"]), "ckpt", "val_e%03d.json" % st["epoch"])
        _hashed(p, inputs["stored_rows"][key], "stored rows %s" % key)
        j, _sha = _load_json(p)
        rows = j.get("rows") if isinstance(j, dict) else None
        ok = (isinstance(rows, list) and j.get("schema") == ROWS_SCHEMA and j.get("arm") == st["arm"] and j.get("seed") == st["seed"]
              and j.get("epoch") == st["epoch"] and j.get("model_state_sha256") == sid and j.get("n_rows") == len(entries) == len(rows)
              and [(r.get("file"), r.get("slice_index")) for r in rows] == entries)
        if not ok:
            _stop(POPULATION_MISMATCH, "stored rows %s: schema, arm, seed, epoch, model state, row count or row order differ" % key)
        vals = {}
        for name, field in zip(ROW_NAMES, ROW_FIELDS):
            a = [rows[q].get(field) for q in mid_pos]
            if not all(isinstance(v, float) and math.isfinite(v) for v in a):
                _stop(C.NONFINITE, "stored rows %s: field %s is missing or not finite on a midpoint slice" % (key, field))
            vals[name] = np.array(a, dtype=np.float64)
        stored[sid] = vals
        hashed += 1
    LOG.info("files verified: %d hashed; stored rows of %d states on %d midpoint slices", hashed + 2, len(stored), n_vol)
    return {"stored": stored, "volume_ids": vols, "volume_order": sorted(range(n_vol), key=lambda q: vols[q]),
            "eligible": [bool(slices[q]["eligible"]) for q in mid_pos], "n_rows_per_file": len(entries), "files_hashed": hashed + 2}

# SEQREF-DLAT-F1D v0.3 -- diag_lat_f1.py: D-LAT F1 preparation driver, stages D1-D9, with the bind-only preflight (plan SEQREF-DLAT-F1D v0.1; F1 plan SEQREF-DLAT-F1 v0.2)
# LIFETIME: KEEP
"""F1 of the frozen specification SEQREF-MRI-DIAG-LAT v1.0 (LudWiki 235, sha256 6afa3eda...7327):
binding, device, registered arrays, convention fixtures, timing probe, resource projection.

F1 WRITES NO MEASURE, CONTRAST, CLASS OR FIGURE (condition C5). The probe's records and the
synthetic population result exist in memory only and are discarded; the report keeps timings,
byte counts, identities and fixture residuals, and its schema refuses any other key (A11).

The driver holds no science: every model operation is an adapter call (diag_lat_adapter), every
statistic a population-stage call (diag_lat_population), every hash-level binding a binder call
(diag_lat_bind). Arithmetic of this file (complete list): maxima of absolute and relative
residuals; the comparison of each with its run-card tolerance; the fixture 3 identity
log RMSE_R0 - (log gamma_truth + 0.5 log q_z); W, R, L, D of 235 section 7 (C1 v0.3).

Modes:
  full run     run_f1(binding, card, out_dir); production main() builds ProductionBinding
  --bind-only  bind_manifest + verify_files on the real results folder (conditions C6, C8):
               no PyTorch, no checkpoint tensor, no inference; one small report in a new folder

Conditions: C3 (30 role-seed assignments -> exactly n_states), C5, C6, C7 (D8 times the whole
population stage with the missing-value branch and serialisation; F1 wall, memory and disk
guards; no device fallback), C8 (bind-only reports the complete file-level outcome), C9 (the
model-state hash of every loaded model is compared with the manifest; verify_files hashes every
file of every state, both files of a two-file state).

The run card is the only source of tolerances, ceilings and the device. Device: the card's
device must be "cpu" (owner's decision, LudWiki 249 LE82); every model parameter and every
tensor of the slice states must be on it; anything else stops. There is no fallback.

DISCLOSED for review (not in the wording of the approved plans):
  (1) card tolerance fx3_l2_identity_abs (the plans name none for fixture 3; 1e-12 is the value
      of the approved self-test T3);
  (2) a wrong device stops with DIAG_LAT_REPRODUCTION_MISMATCH (the code 235 section 7 names for G);
  (3) a complete probe record costs 264 flow passes, not 262: slice_state_record repeats the
      target encode and the R0 decode that the driver needs beforehand for L5. t is therefore
      slightly conservative;
  (4) R_fix: the G runner does not exist yet, so its run-level files are not known. Counted:
      twice the serialised population result (result, and tables and figure data derived from
      it), one manifest line per record file and run-level file, the log bytes of this F1 run
      as the allowance for the remaining logs, 193 bytes per run-level file for RUN_LEVEL_FILES
      files, and the ledger creation;
  (5) the synthetic population of D8 (synthetic_population) is the labelled synthetic input
      235 section 7 prescribes for t_stat. It feeds timings and byte counts only.
No fallback, placeholder, mock or dummy return. Every failure is a typed DiagLatError, logged;
the report then records the stop and the verdict is F1: FAIL.

Changelog:
  v0.3 (2026-11-23) review of v0.2 (LudWiki 250 LE86): a FINAL elapsed-time check directly before the
        verdict. The wall ceiling was checked after every model state and at the end of D8; it is
        now checked once more after the last stage, so a run that crosses the ceiling anywhere
        cannot be reported as F1: PASS. The stop (DIAG_LAT_RESOURCE_LIMIT, criterion A9) records
        the elapsed time; the final elapsed time is written to the report. Nothing else changed.
  v0.2 (2026-11-23) review of v0.1 (LudWiki 250 LE84), conditions C10-C12. C10: a production run card
        must name the REGISTERED D-UNC manifest hash and the pin (constants of diag_lat_bind); another
        internally consistent card is refused. C11: if the report fails its schema, a separate
        failure report (schema seqref-dlat-f1-failure/1, file f1_failure_report.json, exact keys,
        verdict F1: FAIL only, itself validated) is written INSTEAD of f1_report.json; it cannot be
        taken for F1 evidence. C12: new required card text not_verified_before_run (what no test
        has exercised before the run); copied into the bind-only report. No stage, fixture,
        tolerance or formula changed.
  v0.1 (2026-11-23) first version.
"""
import sys

FORBIDDEN = ("--checkpoint", "--data-root")
EXIT_REFUSED = 3


def refuse_forbidden(argv):
    """A4: standard library only; runs before NumPy, PyTorch or any project module is imported."""
    hit = sorted({f for a in argv for f in FORBIDDEN if a == f or a.startswith(f + "=")})
    if hit:
        sys.stderr.write("ERROR DIAG_LAT_REFUSED: option(s) %s are not accepted: F1 takes checkpoints and data "
                         "only through the registered manifest\n" % ", ".join(hit))
        raise SystemExit(EXIT_REFUSED)


if __name__ == "__main__":
    refuse_forbidden(sys.argv[1:])

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import diag_lat_adapter as A  # noqa: E402
import diag_lat_bind as BD  # noqa: E402
import diag_lat_core as C  # noqa: E402
import diag_lat_guard as GD  # noqa: E402
import diag_lat_population as PO  # noqa: E402
import diag_lat_stats as ST  # noqa: E402
from diag_lat_core import DiagLatError  # noqa: E402

__version__ = "0.3"
ABBR = "SEQREF-DLAT-F1D"
LOG = logging.getLogger("diag_lat.f1")

CARD_SCHEMA = "seqref-dlat-f1-runcard/1"
REPORT_SCHEMA = "seqref-dlat-f1-report/1"
FAILURE_SCHEMA = "seqref-dlat-f1-failure/1"       # C11: written instead of the report when the report fails its schema
FAILURE_KEYS = {"schema", "banner", "driver", "run_card_sha256", "criteria", "stop", "verdict"}
BIND_SCHEMA = "seqref-dlat-bind-report/1"
RUN_CARD = "DIAG_LAT_RUN_CARD"
REPRODUCTION_MISMATCH = "DIAG_LAT_REPRODUCTION_MISMATCH"
NOISE_SEED, PROJ_SEED = 20261121, 20261122
SYNTHETIC_SEED = 778899                         # D8 synthetic population only; not a registered seed
PROBE_ROLES = ("lam0/e3", "lam0/fin", "lam2/e3", "lam2/fin")       # 235 section 7, on the first seed
RECORD_ALLOWANCE = GD.SIDECAR_BYTES + GD.LEDGER_REWRITE_BYTES      # 193 bytes (235 section 7, b)
RUN_LEVEL_FILES = 64                            # disclosed item (4)
CRITERIA = tuple("A%d" % k for k in range(1, 13))
NOT_REACHED = "NOT_REACHED"
REGISTERED = {"n_states": 28, "n_volumes": 97, "n_boot": 10000, "bank_rows": C.K_BANK}
CARD_KEYS = {"schema", "selftest", "rules_version", "device", "manifest_f2_sha256", "pin", "registered",
             "tolerances", "fx7_validity", "not_verified_before_run", "ceilings"}
TOL_KEYS = {"fx1_roundtrip_max_abs", "fx2ii_roundtrip_max_abs", "stored_rows_nats_per_dim", "s1_s2_p1_relative",
            "fx7_relative", "fx3_l2_identity_abs"}
CEIL_KEYS = {"f1_wall_s", "g_wall_s", "disk_bytes", "reserve_bytes", "ram_bytes", "vram_bytes"}
REPORT_KEYS = {"schema", "banner", "versions", "run_card", "environment", "criteria", "binding", "device",
               "arrays", "fixtures", "probe", "projection", "memory", "stop", "verdict"}
REFUSED_KEYS = ({k["name"] for k in A.RECORD_KEYS} | set(ST.MEASURES) | set(ST.ALL_CONTRASTS)
                | {"record", "records", "classified", "class", "classes", "contrast", "contrasts", "figure", "figures",
                   "main_table", "reported", "state_means", "e_cov", "l6", "tail", "events", A.BANK_FIELD})


def _stop(code, message):
    raise DiagLatError(code, message)            # DiagLatError logs at construction


def pinned(where, fn, *args):
    """A pinned call outside the adapter (fixture 7 only: the pinned R0 image); typed stop on failure."""
    try:
        return fn(*args)
    except DiagLatError:
        raise
    except Exception as exc:
        _stop(A.ADAPTER, "%s: %s: %s" % (where, type(exc).__name__, exc))


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def dumps(obj):
    return json.dumps(obj, indent=1, sort_keys=True).encode()


class MemoryLog(logging.Handler):
    """Log lines kept in memory: the log file is written once, through the write guard."""

    def __init__(self):
        super().__init__()
        self.lines, self.nbytes = [], 0

    def emit(self, record):
        line = self.format(record)
        self.lines.append(line)
        self.nbytes += len(line.encode()) + 1


# ---- D1: run card ---------------------------------------------------------------------------
def load_card(path, sha256):
    """The reviewed run card: sha256 given on the command line; exact key sets; no default."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        _stop(RUN_CARD, "cannot read the run card %s: %s" % (path, exc))
    got = sha_bytes(raw)
    if got != sha256:
        _stop(RUN_CARD, "run card %s has sha256 %s, expected %s" % (path, got, sha256))
    try:
        card = json.loads(raw)
    except ValueError as exc:
        _stop(RUN_CARD, "run card is not JSON: %s" % exc)
    check_card(card)
    return card, got


def check_card(card):
    if not isinstance(card, dict) or set(card) != CARD_KEYS:
        got = set(card) if isinstance(card, dict) else set()
        _stop(RUN_CARD, "run card keys: missing %s, unexpected %s" % (sorted(CARD_KEYS - got), sorted(got - CARD_KEYS)))
    tol, ceil, reg = card["tolerances"], card["ceilings"], card["registered"]
    ok = (card["schema"] == CARD_SCHEMA and isinstance(card["selftest"], bool) and card["rules_version"] == ST.RULES_VERSION
          and isinstance(card["device"], str) and isinstance(card["fx7_validity"], str)
          and isinstance(card["not_verified_before_run"], str) and card["not_verified_before_run"].strip()
          and isinstance(card["manifest_f2_sha256"], str) and isinstance(card["pin"], str)
          and isinstance(tol, dict) and set(tol) == TOL_KEYS and isinstance(ceil, dict) and set(ceil) == CEIL_KEYS
          and isinstance(reg, dict) and set(reg) == set(REGISTERED)
          and all(isinstance(v, float) and math.isfinite(v) and v > 0.0 for v in tol.values())
          and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in list(ceil.values()) + list(reg.values())))
    if not ok:
        _stop(RUN_CARD, "run card: schema, rules version, a tolerance, a ceiling or a registered size is missing or malformed")
    if card["device"] != "cpu":
        _stop(REPRODUCTION_MISMATCH, "run card device %r: this code runs the flow passes on cpu only; there is no "
              "transfer code and no fallback" % card["device"])
    if tol["s1_s2_p1_relative"] != A.SWAP_REL_TOL:
        _stop(RUN_CARD, "run card S1/S2/P1 tolerance %r differs from the adapter's %r" % (tol["s1_s2_p1_relative"], A.SWAP_REL_TOL))
    if not card["selftest"] and reg != REGISTERED:
        _stop(RUN_CARD, "run card sizes %r differ from the registered %r" % (reg, REGISTERED))
    if not card["selftest"] and (card["manifest_f2_sha256"] != BD.REGISTERED_F2_SHA256 or card["pin"] != BD.PIN):       # C10
        _stop(RUN_CARD, "run card names manifest %s and pin %s; the registered ones are %s and %s"
              % (card["manifest_f2_sha256"], card["pin"], BD.REGISTERED_F2_SHA256, BD.PIN))


# ---- production binding (D2, D3) ------------------------------------------------------------
class ProductionBinding:
    """The 28 V05LT states, the 97 validation slices, the bank and the stored rows, through the
    registered D-UNC manifest (diag_lat_bind) and the hash-verified D-UNC adapter."""
    selftest = False

    def __init__(self, repo, manifest_path, card):
        self.repo = repo
        self.root = BD.results_root(repo)
        self.bound = BD.bind_manifest(manifest_path, f2_sha256=card["manifest_f2_sha256"],
                                      n_states=card["registered"]["n_states"], n_volumes=card["registered"]["n_volumes"],
                                      pin=card["pin"])
        self.files = BD.verify_files(self.bound, self.root)
        b = self.bound
        self.manifest_sha256, self.pin = b["manifest_sha256"], card["pin"]
        self.seeds, self.role_map = list(b["seeds"]), dict(b["role_map"])
        self.state_order, self.states = list(b["state_order"]), b["states"]
        self.identities, self.volume_ids = [tuple(x) for x in b["identities"]], list(self.files["volume_ids"])
        self.stored = self.files["stored"]
        self.masks_sha256, self.bank_sha256 = b["regenerated_masks_sha256"], b["inputs"]["z_diag_sha256"]
        self.bank_manifest_sha256 = b["inputs"]["z_manifest_sha256"]
        self.files_hashed = self.files["files_hashed"]

    def _dua(self):
        A.verify_origin()
        import diag_unc_adapter as dua
        return dua

    def slice_states(self):
        """The 97 production slice states in manifest order (validation view only; no holdout item)."""
        dua = self._dua()
        try:
            inputs = dua.load_inputs(self.repo)
            view = dua.validation_view(inputs)
            return dua.build_states(view, dua.positions_for(view, self.identities), inputs)
        except dua.AdapterError as exc:
            _stop(A.ADAPTER, "D-UNC adapter (slice states): %s" % exc)
        except DiagLatError:
            raise
        except Exception as exc:
            _stop(A.ADAPTER, "slice states: %s: %s" % (type(exc).__name__, exc))

    def bank(self):
        dua = self._dua()
        try:
            rec = dua.z_bank()
        except Exception as exc:
            _stop(A.ADAPTER, "Z_DIAG bank: %s: %s" % (type(exc).__name__, exc))
        if rec["bank_sha256"] != self.bank_sha256 or rec["manifest_sha256"] != self.bank_manifest_sha256:
            _stop(A.ORIGIN, "Z_DIAG bank or its manifest differ from the registered hashes")
        return rec["bank"]

    def load_state(self, sid):
        st = self.states[sid]
        dua = self._dua()
        try:
            model, ver = dua.load_checkpoint(os.path.join(self.root, st["path"]), st["file_sha256"], sid,
                                             seed=st["seed"], arm=st["arm"], epoch=st["epoch"])
        except Exception as exc:
            _stop(A.ORIGIN, "state %s (%s): the pinned loader refused it: %s: %s" % (sid[:12], st["path"], type(exc).__name__, exc))
        return model, {"model_state_sha256": ver.model_state_sha256, "file_sha256": ver.file_sha256}


# ---- D8: synthetic population (labelled; 235 section 7, t_stat) --------------------------------
def synthetic_population(role_map, state_order, n_volumes, labels, seed=SYNTHETIC_SEED):
    """SYNTHETIC records with the full schema for every state: a role level, a volume term and
    noise; one not-computed value with its event (the missing-value branch). No real measurement
    enters; the result is used for timing and byte counts only and is discarded."""
    g = np.random.Generator(np.random.PCG64(seed))
    roles_of = {}
    for (role, _seed), sid in sorted(role_map.items(), key=repr):
        roles_of.setdefault(sid, role)
    level = {r: 0.1 * i for i, r in enumerate(PO.ROLES)}
    tau = np.abs(g.standard_normal(n_volumes)) * 0.01
    vol = np.sin(np.arange(n_volumes) * 1.3)
    records = {}
    for sid in state_order:
        lv = level[roles_of[sid]]
        noise = g.standard_normal((n_volumes, len(PO.SCALAR_KEYS), len(labels))) * 0.05
        zp = g.standard_normal((n_volumes, C.M_PROJ))
        rows = []
        for v in range(n_volumes):
            rec, bank = {}, {lab: {} for lab in labels}
            for ki, k in enumerate(PO.SCALAR_KEYS):
                base = lv * (1.0 + 0.1 * ki) + 0.3 * vol[v] + 0.01 * ki
                if k["name"] in PO.TAIL_KEYS:
                    rec[k["name"]] = float(tau[v] * (1.0 if k["name"] == "tau" else 30.0))
                elif k["bank"]:
                    for li, lab in enumerate(labels):
                        bank[lab][k["name"]] = float(base + noise[v, ki, li])
                else:
                    rec[k["name"]] = float(base + noise[v, ki, 0])
            rec["z_proj"] = [float(x) for x in zp[v]]
            rec[A.BANK_FIELD] = bank
            rows.append(rec)
        records[sid] = rows
    nc_key = next(k["name"] for k in A.RECORD_KEYS if k["bank"] and k["type"] == "float|NOT_COMPUTED")
    first = state_order[0]
    records[first][0][A.BANK_FIELD][labels[0]][nc_key] = C.NOT_COMPUTED
    events = [{"code": C.UNDEFINED_STATISTIC, "measure": nc_key, "state": str(first), "volume": "v000"}]
    return records, events


# ---- the run ----------------------------------------------------------------------------------
class F1Run:
    def __init__(self, card, out_dir, card_sha256):
        self.b, self.card, self.out, self.card_sha = None, card, out_dir, card_sha256
        self.tol, self.ceil, self.reg = card["tolerances"], card["ceilings"], card["registered"]
        self.t0 = time.monotonic()
        self.crit = {c: {"status": NOT_REACHED, "observed": None, "limit": None, "stop": None} for c in CRITERIA}
        self.current = "A1"
        self.rep = {k: None for k in REPORT_KEYS}
        self.fx = {}

    # -- helpers
    def at(self, criterion):
        self.current = criterion

    def passed(self, criterion, observed, limit):
        self.crit[criterion].update(status="PASS", observed=observed, limit=limit)

    def worst(self, name, value, tol):
        """Record the largest residual of a fixture; the first value above the tolerance stops."""
        e = self.fx.setdefault(name, {"max": 0.0, "tolerance": tol, "n": 0})
        if not math.isfinite(value):
            _stop(C.NONFINITE, "fixture %s: non-finite residual" % name)
        e["max"], e["n"] = max(e["max"], float(value)), e["n"] + 1
        return value <= tol

    def wall(self):
        el = time.monotonic() - self.t0
        if el > self.ceil["f1_wall_s"]:
            _stop(GD.RESOURCE_LIMIT, "F1 wall time %.0f s exceeds the ceiling %d s" % (el, self.ceil["f1_wall_s"]))
        return el

    # -- D1
    def d1(self):
        self.at("A1")
        if self.card["selftest"] is not bool(self.b.selftest):
            _stop(RUN_CARD, "run card selftest = %r but the binding's is %r" % (self.card["selftest"], self.b.selftest))
        if not (sys.flags.isolated and sys.dont_write_bytecode):
            _stop(A.ORIGIN, "the interpreter must run with -I -B (isolated %d, no bytecode %s)"
                  % (sys.flags.isolated, sys.dont_write_bytecode))
        here = os.path.dirname(os.path.abspath(__file__))
        code = {n: BD.sha_file(os.path.join(here, n)) for n in sorted(os.listdir(here)) if n.endswith(".py")}
        self.rep["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
                                   "isolated": bool(sys.flags.isolated), "no_bytecode": bool(sys.dont_write_bytecode),
                                   "code_sha256": code, "pin": self.b.pin}
        self.rep["versions"] = {"driver": __version__, "adapter": A.__version__, "binder": BD.__version__, "core": C.__version__,
                                "guard": GD.__version__, "population": PO.__version__, "stats": ST.__version__,
                                "rules": ST.RULES_VERSION}
        self.rep["run_card"] = {"sha256": self.card_sha, "content": self.card}
        self.passed("A1", "driver part: -I -B, code hashes recorded, pin %s; launcher part: exit 90, 91, 92" % self.b.pin[:12], "see plan A1")
        self.passed("A4", "driver part: no checkpoint or data option accepted; every write through one write guard", "see plan A4")

    # -- D2, D3
    def d2_d3(self):
        b, reg = self.b, self.reg
        self.at("A2")
        want = {(r, s) for r in PO.ROLES for s in b.seeds}
        ok = (set(b.role_map) == want and set(b.role_map.values()) == set(b.state_order)
              and len(b.state_order) == len(set(b.state_order)) == reg["n_states"] and set(b.states) == set(b.state_order))
        if not ok:
            _stop(ST.POPULATION_MISMATCH, "%d role-seed assignments resolve to %d states, expected the %d registered roles "
                  "for each seed and exactly %d states" % (len(b.role_map), len(set(b.role_map.values())), len(PO.ROLES), reg["n_states"]))
        self.at("A3")
        n = reg["n_volumes"]
        if not (len(b.identities) == len(set(b.identities)) == n and len(b.volume_ids) == len(set(b.volume_ids)) == n):
            _stop(ST.POPULATION_MISMATCH, "%d slices and %d volumes, expected %d distinct of each"
                  % (len(b.identities), len(set(b.volume_ids)), n))
        for sid in b.state_order:
            row = b.stored.get(sid)
            if not (isinstance(row, dict) and set(row) == set(BD.ROW_NAMES)
                    and all(np.asarray(row[k]).shape == (n,) and np.all(np.isfinite(row[k])) for k in BD.ROW_NAMES)):
                _stop(ST.POPULATION_MISMATCH, "stored rows of state %s are missing, incomplete or non-finite" % str(sid)[:12])
        self.states = b.slice_states()
        ids = [(str(st["identity"]["file"]), int(st["identity"]["slice_index"])) for st in self.states]
        if ids != [(str(f), int(i)) for f, i in b.identities]:
            _stop(ST.POPULATION_MISMATCH, "the slice states are not the registered identities in the registered order")
        h = hashlib.sha256()
        for st in self.states:
            h.update(st["mask"][0].numpy().tobytes())
        if h.hexdigest() != b.masks_sha256:
            _stop(A.ORIGIN, "mask bytes %s differ from the registered %s" % (h.hexdigest(), b.masks_sha256))
        self.bank = np.asarray(b.bank())
        d = int(A.seams().ffr.FLOW_DIM_REAL)
        bank_sha = sha_bytes(np.ascontiguousarray(self.bank).tobytes(order="C"))
        if self.bank.shape != (reg["bank_rows"], d) or self.bank.dtype != np.float32 or bank_sha != b.bank_sha256:
            _stop(A.ORIGIN, "bank %s %s sha256 %s, registered (%d, %d) float32 %s"
                  % (self.bank.dtype, self.bank.shape, bank_sha, reg["bank_rows"], d, b.bank_sha256))
        self.rep["binding"] = {"manifest_sha256": b.manifest_sha256, "seeds": [int(s) for s in b.seeds],
                               "n_role_assignments": len(b.role_map), "n_states": len(b.state_order),
                               "states": [{"index": k, "model_state_sha256": str(sid), "arm": b.states[sid]["arm"],
                                           "seed": b.states[sid]["seed"], "epoch": b.states[sid]["epoch"],
                                           "file_sha256": b.states[sid]["file_sha256"],
                                           "roles": sorted(r for (r, _s), q in b.role_map.items() if q == sid)}
                                          for k, sid in enumerate(b.state_order)],
                               "slices": [[f, i] for f, i in ids], "volume_ids": list(b.volume_ids),
                               "masks_sha256": h.hexdigest(), "bank_sha256": bank_sha, "bank_shape": list(self.bank.shape),
                               "bank_beyond_spline_bound": int(np.count_nonzero(np.abs(self.bank) > np.float32(C.SPLINE_B))),
                               "files_hashed": int(b.files_hashed), "holdout": "not requested: validation view only"}
        self.passed("A3", "%d slices in the registered order; masks, bank and stored rows at their registered hashes" % n, "equality")

    # -- D4
    def d4_tensors(self):
        self.at("A5")
        torch = A.seams().torch
        devs = sorted({str(v.device) for st in self.states for v in st.values() if isinstance(v, torch.Tensor)})
        if devs != [self.card["device"]]:
            _stop(REPRODUCTION_MISMATCH, "slice-state tensors are on %s, the run card says %s" % (devs, self.card["device"]))
        self.rep["device"] = {"run_card": self.card["device"], "slice_state_tensors": devs, "models": None,
                              "precision": {"flow_passes": "float32", "statistics": "float64"},
                              "vram": "not applicable (no CUDA use)"}

    def check_model(self, model, sid):
        rec = A.device_record(model)
        if rec["devices"] != [self.card["device"]] or rec["dtypes"] != ["torch.float32"]:
            _stop(REPRODUCTION_MISMATCH, "state %s: parameters on %s with %s; the run card says %s float32"
                  % (str(sid)[:12], rec["devices"], rec["dtypes"], self.card["device"]))
        if self.rep["device"]["models"] is None:
            self.rep["device"]["models"] = rec
        elif {k: rec[k] for k in ("devices", "dtypes", "n_parameters", "torch_threads")} != \
                {k: self.rep["device"]["models"][k] for k in ("devices", "dtypes", "n_parameters", "torch_threads")}:
            _stop(REPRODUCTION_MISMATCH, "state %s: device record %r differs from the first state's" % (str(sid)[:12], rec))

    # -- D5
    def d5(self):
        self.at("A8")
        n, reg = self.reg["n_volumes"], self.reg
        t = time.monotonic()
        self.idx = ST.bootstrap_matrix(n_volumes=n, n_boot=reg["n_boot"])
        self.t_matrix = time.monotonic() - t
        self.idx_sha = ST.matrix_identity(self.idx)
        self.noise = np.random.Generator(np.random.PCG64(NOISE_SEED)).standard_normal(size=(n, C.N_ACQUIRED, 2))
        raw = np.random.Generator(np.random.PCG64(PROJ_SEED)).standard_normal(size=(C.D_FREE, C.M_PROJ))
        self.proj = C.projection_matrix(raw)
        calls = {"bootstrap": "Generator(PCG64(%d)).integers(0, %d, size=(%d, %d), dtype=int64)" % (ST.DIAG_LAT_SEED, n, reg["n_boot"], n),
                 "noise": "Generator(PCG64(%d)).standard_normal(size=(%d, %d, 2))" % (NOISE_SEED, n, C.N_ACQUIRED),
                 "projection": "Generator(PCG64(%d)).standard_normal(size=(%d, %d)), reduced QR, diag R > 0" % (PROJ_SEED, C.D_FREE, C.M_PROJ)}
        self.rep["arrays"] = {"bootstrap": dict(C.array_identity(self.idx), call=calls["bootstrap"], matrix_identity=self.idx_sha),
                              "noise": dict(C.array_identity(self.noise), call=calls["noise"]),
                              "projection": dict(C.array_identity(self.proj), call=calls["projection"]),
                              "registered_sizes": not self.card["selftest"]}
        LOG.info("registered arrays: bootstrap %s, noise %s, projection %s", self.idx_sha[:16],
                 self.rep["arrays"]["noise"]["sha256"][:16], self.rep["arrays"]["projection"]["sha256"][:16])
        self.scaled = []
        for i, st in enumerate(self.states):                         # layout and degenerate cases, every slice
            r, c = A.acquired_cells(st)
            cols = np.unique(c)
            y = st["y"][0].detach().cpu().numpy()
            off = np.ones(y.shape, dtype=bool)
            off[r, c] = False
            ok = (r.size == C.N_ACQUIRED and cols.size * y.shape[0] == C.N_ACQUIRED
                  and np.array_equal(r, np.repeat(np.arange(y.shape[0]), cols.size)) and np.array_equal(c, np.tile(cols, y.shape[0]))
                  and not np.any(y[off]))
            if not ok:
                _stop(ST.BOOTSTRAP_IDENTITY, "slice %d: the acquired cells are not %d cells in row-major order with the "
                      "measurement zero elsewhere (%d cells, %d columns)" % (i, C.N_ACQUIRED, r.size, cols.size))
            self.scaled.append(C.scale_noise(self.noise[i], A.acquired_values(st)))       # stops on zero RMS or non-finite
        self.rep["arrays"]["layout"] = {"cells_per_slice": C.N_ACQUIRED, "order": "row-major; rows ascending, acquired columns ascending",
                                        "slices_checked": len(self.states)}
        self.passed("A8", "three one-call draws, identities recorded; %d cells per slice in the assumed order on %d slices"
                    % (C.N_ACQUIRED, len(self.states)), "equality")

    # -- slice-only fixtures (S2, S3, S4, P1 input)
    def slice_fixtures(self):
        self.at("A7")
        s, n = A.seams(), len(self.states)
        donors = list(s.d3.derangement(n))
        if donors != [(i + 1) % n for i in range(n)] or any(self.b.volume_ids[i] == self.b.volume_ids[donors[i]] for i in range(n)):
            _stop(C.SWAP_FIXTURE, "S3: the donor map is not (i + 1) mod %d or pairs a volume with itself" % n)
        self.donors, self.swaps, self.zero_cond = donors, [], []
        s4 = []
        for i, st in enumerate(self.states):
            sw = A.swap_input(st, self.states[donors[i]])            # S2 and S4 inside; typed stops
            if sw["mask"] is not st["mask"]:
                _stop(C.SWAP_FIXTURE, "S2: slice %d: the mask passed on is not the recipient's stored mask" % i)
            self.worst("S2_excluded_over_tolerance", sw["s2_excluded_max"] / sw["s2_tolerance"], 1.0)
            self.worst("S2_acquired_over_tolerance", sw["s2_acquired_max_diff"] / sw["s2_tolerance"], 1.0)
            s4.append(sw["s4_relative_l2"])
            self.swaps.append(sw)
            zero = A.perturbed_input(st, np.zeros(C.N_ACQUIRED, dtype=np.complex128))
            self.worst("P1_cond_over_tolerance", A.perturb_check_p1(st, zero)                 # P1 stops inside at the S1 tolerance
                       / (A.SWAP_REL_TOL * float(np.max(np.abs(st["cond"].detach().cpu().numpy())))), 1.0)
            self.zero_cond.append(zero)
        self.fx["S3"] = {"pairs": [[list(self.b.identities[i]), list(self.b.identities[donors[i]])] for i in range(n)]}
        self.fx["S4_relative_l2"] = {"min": float(min(s4)), "max": float(max(s4)), "n": n}

    # -- D6 (fixtures) and D7 (probe), one state loaded at a time
    def d6_d7(self):
        b, tol, s = self.b, self.tol, A.seams()
        row_tol, n = tol["stored_rows_nats_per_dim"], len(self.states)
        probe_states = [b.role_map[(r, b.seeds[0])] for r in PROBE_ROLES]
        if len(set(probe_states)) != len(PROBE_ROLES):
            _stop(ST.POPULATION_MISMATCH, "the four probe roles of the first seed do not name four distinct states")
        probe_slices = sorted({0, n // 2, n - 1})
        loads, t_rec, b_rec, rt2, passes = {}, [], [], [], 0
        for k, sid in enumerate(b.state_order):
            self.at("A2")
            t = time.monotonic()
            model, ver = b.load_state(sid)
            loads[sid] = time.monotonic() - t
            if ver["model_state_sha256"] != sid or ver["file_sha256"] != b.states[sid]["file_sha256"]:       # C9
                _stop(A.ORIGIN, "state %s: the loader returned model state %s, file %s; the manifest names %s, %s"
                      % (str(sid)[:12], ver["model_state_sha256"], ver["file_sha256"], sid, b.states[sid]["file_sha256"]))
            self.at("A5")
            self.check_model(model, sid)
            stored = b.stored[sid]
            for i, st in enumerate(self.states):
                self.at("A6")
                enc = A.encode_truth(model, st)                                        # fixtures 2 (i), 4, 6
                for name in ("NLL", "B", "J"):
                    if not self.worst("stored_%s_abs_diff" % name, abs(enc[name] - float(stored[name][i])), row_tol):
                        _stop(A.ADAPTER, "state %s slice %d: recomputed %s %.6f, stored %.6f (tolerance %g)"
                              % (str(sid)[:12], i, name, enc[name], float(stored[name][i]), row_tol))
                t64, t32 = A.target_std(st)
                z = enc["z_star"]
                mask, _tau, _q = C.tail_set(t32)                                       # fixture 8
                C.check_tail_invariance(np.asarray(z), t32, mask)
                self.fx.setdefault("fx8_beyond_bound_outside_tail", {"total": 0})["total"] += int(
                    np.count_nonzero((np.abs(np.asarray(z)) > np.float32(C.SPLINE_B)) & ~mask))
                r0 = A.decode_r0_std(model, st).astype(np.float64)
                z64 = np.asarray(z).astype(np.float64)
                num, zn = float(np.linalg.norm(t64 - r0)), float(np.linalg.norm(z64))
                if num == 0.0 or zn == 0.0:
                    _stop(C.ZERO_DENOMINATOR, "state %s slice %d: ||u_true - R0|| = %r, ||z*|| = %r" % (str(sid)[:12], i, num, zn))
                l2 = abs(math.log(num / math.sqrt(t64.size)) - (math.log(num / zn) + 0.5 * math.log(C.q_of(z64))))     # fixture 3
                if not self.worst("fx3_l2_identity_abs", l2, tol["fx3_l2_identity_abs"]):
                    _stop(A.ADAPTER, "state %s slice %d: L2 identity residual %.3e" % (str(sid)[:12], i, l2))
                scale, mean = A.scale_mean_interleaved(st)                             # fixture 7
                d_phys = C.shift_norm(C.to_physical(t64, scale, mean), C.to_physical(r0, scale, mean))
                x0, _u0 = pinned("r0_image", s.EV.r0_image, model, st)
                d_img = float(np.linalg.norm(st["x_true_c"].to(s.torch.complex128).numpy() - x0.to(s.torch.complex128).numpy()))
                if d_img == 0.0:
                    _stop(C.ZERO_DENOMINATOR, "state %s slice %d: image distance 0" % (str(sid)[:12], i))
                if not self.worst("fx7_relative", abs(d_phys - d_img) / d_img, tol["fx7_relative"]):
                    _stop(A.ADAPTER, "state %s slice %d: physical distance %.9g, image distance %.9g" % (str(sid)[:12], i, d_phys, d_img))
                back = A.decode_bank_std(model, st, np.ascontiguousarray(np.asarray(z, dtype=np.float32)[None, :]))[0]       # fixture 1
                if not self.worst("fx1_roundtrip_max_abs", float(np.max(np.abs(back.astype(np.float64) - t32.astype(np.float64)))),
                                  tol["fx1_roundtrip_max_abs"]):
                    _stop(A.ADAPTER, "state %s slice %d: round trip g(f(u)) differs from u by more than %g"
                          % (str(sid)[:12], i, tol["fx1_roundtrip_max_abs"]))
                self.at("A7")
                own = A.swap_input(st, st, check_identity=False)                       # S1
                if not self.worst("S1_cond_over_tolerance", own["max_abs_diff_to_own"] / (A.SWAP_REL_TOL * own["max_abs_own"]), 1.0):
                    _stop(C.SWAP_FIXTURE, "S1: state %s slice %d: self-swap input differs by %.3e" % (str(sid)[:12], i, own["max_abs_diff_to_own"]))
                if not self.worst("S1_nll_abs_diff", abs(A.encode_truth(model, st, own["cond_swap"], own["mask"])["NLL"]
                                                         - float(stored["NLL"][i])), row_tol):
                    _stop(C.SWAP_FIXTURE, "S1: state %s slice %d: self-swap NLL differs from the stored row" % (str(sid)[:12], i))
                if not self.worst("P1_nll_abs_diff", abs(A.encode_truth(model, st, self.zero_cond[i], st["mask"])["NLL"]
                                                         - float(stored["NLL"][i])), row_tol):
                    _stop(A.PERTURB_FIXTURE, "P1: state %s slice %d: zero-noise NLL differs from the stored row" % (str(sid)[:12], i))
                passes += 6
                if sid in probe_states and i in probe_slices:                           # D7
                    self.at("A9")
                    t = time.monotonic()
                    nbytes, rt = self.probe_record(model, st, i, sid)
                    t_rec.append(time.monotonic() - t)
                    b_rec.append(nbytes)
                    rt2.append(rt)
                    self.at("A6")
                    if not self.worst("fx2ii_roundtrip_max_abs", rt, tol["fx2ii_roundtrip_max_abs"]):
                        _stop(A.ADAPTER, "state %s slice %d: re-encode consistency f(g(z_k)) = z_k: residual %.3e above %g"
                              % (str(sid)[:12], i, rt, tol["fx2ii_roundtrip_max_abs"]))
            del model
            self.at("A10")
            peak = GD.check_memory(self.ceil["ram_bytes"])
            el = self.wall()
            LOG.info("state %d of %d done: load %.2f s, elapsed %.0f s, peak memory %d MiB", k + 1, len(b.state_order),
                     loads[sid], el, peak // 2 ** 20)
        if len(rt2) != len(probe_states) * len(probe_slices):
            _stop(ST.POPULATION_MISMATCH, "%d probe records, expected %d" % (len(rt2), len(probe_states) * len(probe_slices)))
        self.passed("A2", "%d states loaded through the pinned loader; file and model-state hash of each equal the manifest"
                    % len(b.state_order), "equality")
        self.passed("A5", "device %s, float32 flow passes, float64 statistics; %d threads"
                    % (self.card["device"], self.rep["device"]["models"]["torch_threads"]), self.card["device"])
        self.passed("A6", {k: v["max"] for k, v in self.fx.items() if k.startswith(("stored_", "fx")) and "max" in v},
                    {k: v["tolerance"] for k, v in self.fx.items() if k.startswith(("stored_", "fx")) and "max" in v})
        self.passed("A7", {k: v["max"] for k, v in self.fx.items() if k.startswith(("S1", "S2", "P1"))},
                    "S1, S2, P1: %g relative; NLL: %g nats/dim; S4 > 0" % (A.SWAP_REL_TOL, row_tol))
        self.probe = {"states": [str(x) for x in probe_states], "slices": probe_slices, "load_s": [loads[x] for x in probe_states],
                      "record_s": t_rec, "record_bytes": b_rec, "fixture_passes": passes,
                      "load_s_all_states_max": max(loads.values())}

    def probe_record(self, model, st, i, sid):
        """One complete slice-state record at the registered K, in memory; returns its charged bytes
        and the bank round-trip residual. The record is discarded here."""
        sw = self.swaps[i]
        own = A.encode_truth(model, st)
        p_swap = C.swap_penalty(A.encode_truth(model, st, sw["cond_swap"], sw["mask"])["NLL"], own["NLL"])
        cond_p = A.perturbed_input(st, self.scaled[i])
        enc_p = A.encode_truth(model, st, cond_p, st["mask"])
        r0 = A.decode_r0_std(model, st).astype(np.float64)
        sens = C.l5b_sensitivities(r0, A.decode_r0_std(model, st, cond_p, st["mask"]).astype(np.float64),
                                   own["z_star"].astype(np.float64), enc_p["z_star"].astype(np.float64), own["NLL"], enc_p["NLL"])
        r0_sw = A.decode_r0_std(model, st, sw["cond_swap"], sw["mask"]).astype(np.float64)
        scale, mean = A.scale_mean_interleaved(st)
        rec, diag = A.slice_state_record(
            model, st, self.bank, subsets=C.STABILITY_SUBSETS, p_swap=p_swap, r0_psnr=float(self.b.stored[sid]["R0_PSNR"][i]),
            swap_shift_r0=C.shift_norm(r0_sw, r0),
            phys_swap_shift_r0=C.shift_norm(C.to_physical(r0_sw, scale, mean), C.to_physical(r0, scale, mean)),
            sensitivities=sens, proj=self.proj)
        payload = dumps({"identity": st["identity"], "state": str(sid), "record": rec,
                         "diagnostics": {k: v for k, v in diag.items() if k != "events"}, "events": diag["events"]})
        line = "probe slice-state done: state %s slice %d (timing and byte count only)" % (str(sid)[:12], i)
        LOG.info(line)
        return len(payload) + RECORD_ALLOWANCE + len(line.encode()) + 64, diag["roundtrip_max_abs"]      # 64: log line prefix allowance

    # -- D8
    def d8(self, log_bytes):
        self.at("A9")
        b, reg, p = self.b, self.reg, self.probe
        vols = ["v%03d" % v for v in range(reg["n_volumes"])]
        records, events = synthetic_population(b.role_map, b.state_order, reg["n_volumes"], list(ST.STABILITY_SUBSETS))
        t = time.monotonic()
        result = PO.run_population(records, b.role_map, vols, b.seeds, self.idx, self.idx_sha, reg["n_states"], events,
                                   n_boot=reg["n_boot"])
        blob = dumps(result)
        t_stat = time.monotonic() - t + self.t_matrix
        nc = len(events)
        n_ss = reg["n_states"] * reg["n_volumes"]
        r_fix = (2 * len(blob) + 110 * (n_ss + RUN_LEVEL_FILES) + log_bytes + RECORD_ALLOWANCE * RUN_LEVEL_FILES + GD.LEDGER_BYTES)
        del result, blob, records
        s, tt, bb = max(p["load_s"]), max(p["record_s"]), max(p["record_bytes"])
        w = reg["n_states"] * s + n_ss * tt + t_stat
        r = n_ss * bb + r_fix
        lim_l = 1.5 * (w + max(600.0, 0.10 * w))
        lim_d = 1.5 * (r + max(0.5 * GD.GIB, r))
        self.rep["probe"] = dict(p, s=s, t=tt, t_max_over_min=tt / min(p["record_s"]), t_stat=t_stat, b=bb, R_fix=r_fix,
                                 passes_per_record=2 * (reg["bank_rows"] + 4), missing_value_events=nc,
                                 population="SYNTHETIC: %d states, %d role-seed columns, %d volumes, %d contrasts, %d draws"
                                            % (reg["n_states"], len(b.role_map), reg["n_volumes"], len(ST.ALL_CONTRASTS), reg["n_boot"]))
        self.rep["projection"] = {"W_s": w, "R_bytes": r, "L_s": lim_l, "D_bytes": lim_d, "slice_states": n_ss,
                                  "g_wall_ceiling_s": self.ceil["g_wall_s"], "disk_ceiling_bytes": self.ceil["disk_bytes"]}
        if lim_l > self.ceil["g_wall_s"] or lim_d > self.ceil["disk_bytes"]:
            _stop(GD.RESOURCE_LIMIT, "L = %.0f s (ceiling %d s), D = %d bytes (ceiling %d): the ceiling returns to review"
                  % (lim_l, self.ceil["g_wall_s"], lim_d, self.ceil["disk_bytes"]))
        self.passed("A9", {"L_s": lim_l, "D_bytes": lim_d, "f1_elapsed_s": self.wall()},
                    {"L_s": self.ceil["g_wall_s"], "D_bytes": self.ceil["disk_bytes"], "f1_wall_s": self.ceil["f1_wall_s"]})


def check_report(rep):
    """A11: exact top-level keys; no key of a record, measure, contrast, class or figure anywhere."""
    if set(rep) != REPORT_KEYS:
        _stop(A.RECORD_SCHEMA, "report keys: missing %s, unexpected %s" % (sorted(REPORT_KEYS - set(rep)), sorted(set(rep) - REPORT_KEYS)))

    def walk(o, where):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in REFUSED_KEYS:
                    _stop(A.RECORD_SCHEMA, "report holds the key %r at %s: F1 writes no measure" % (k, where))
                walk(v, "%s.%s" % (where, k))
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v, where)
    walk(rep, "report")
    return True


def failure_report(crit, code, message, banner, card_sha256):
    """C11: the report written INSTEAD of f1_report.json when the report fails its schema.
    Criterion statuses and stop codes only; verdict F1: FAIL; validated by check_failure_report."""
    rep = {"schema": FAILURE_SCHEMA, "banner": banner, "driver": __version__, "run_card_sha256": card_sha256,
           "criteria": {c: {"status": crit[c]["status"], "stop": crit[c]["stop"]} for c in CRITERIA},
           "stop": {"code": code, "criterion": "A11", "message": message}, "verdict": "F1: FAIL"}
    check_failure_report(rep)
    return rep


def check_failure_report(rep):
    ok = (isinstance(rep, dict) and set(rep) == FAILURE_KEYS and rep["schema"] == FAILURE_SCHEMA and rep["verdict"] == "F1: FAIL"
          and all(isinstance(rep[k], str) for k in ("banner", "driver", "run_card_sha256"))
          and isinstance(rep["criteria"], dict) and set(rep["criteria"]) == set(CRITERIA)
          and all(isinstance(v, dict) and set(v) == {"status", "stop"} and v["status"] in ("PASS", "FAIL", NOT_REACHED)
                  and (v["stop"] is None or isinstance(v["stop"], str)) for v in rep["criteria"].values())
          and rep["criteria"]["A11"]["status"] == "FAIL"
          and isinstance(rep["stop"], dict) and set(rep["stop"]) == {"code", "criterion", "message"}
          and all(isinstance(v, str) for v in rep["stop"].values()))
    if not ok:
        _stop(A.RECORD_SCHEMA, "the failure report does not have its registered form; nothing is written")
    return True


def run_f1(binding, card, out_dir, card_sha256="(not from a file)", banner="D-LAT F1 PREPARATION EVIDENCE -- NO MEASURE"):
    """Stages D1-D9. binding: a binding object or a function that builds one (production).
    Returns (report, exit code). out_dir must not exist; it is created here."""
    check_card(card)
    try:
        os.makedirs(out_dir, exist_ok=False)
    except OSError as exc:
        _stop(GD.GUARD_STATE, "cannot create the new output folder %s: %s" % (out_dir, exc))
    mem = MemoryLog()
    mem.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(mem)
    run = F1Run(card, out_dir, card_sha256)
    rep = run.rep
    rep.update(schema=REPORT_SCHEMA, banner=banner, stop=None)
    try:
        guard = GD.WriteGuard(out_dir, card["ceilings"]["disk_bytes"], card["ceilings"]["reserve_bytes"])
        try:
            with GD.memory_guarded(card["ceilings"]["ram_bytes"]):
                run.at("A2")
                run.b = binding() if callable(binding) else binding      # production: manifest and files bound here
                run.d1()
                run.d2_d3()
                run.d4_tensors()
                run.d5()
                run.slice_fixtures()
                run.d6_d7()
                run.d8(mem.nbytes)
            run.at("A9")
            elapsed = run.wall()                 # final check: the F1 wall ceiling is a condition of F1: PASS
            rep["projection"]["f1_elapsed_final_s"] = elapsed
            run.at("A10")
            peak = GD.check_memory(card["ceilings"]["ram_bytes"])
            rep["memory"] = {"peak_rss_bytes": peak, "limit_bytes": card["ceilings"]["ram_bytes"], "vram": "not applicable (no CUDA use)",
                             "disk_written_bytes": guard.written, "reserve_bytes": card["ceilings"]["reserve_bytes"]}
            run.passed("A10", {"peak_rss_bytes": peak}, {"ram_bytes": card["ceilings"]["ram_bytes"]})
            run.passed("A12", "report, verdict, log and sha256 manifest written through the write guard; launcher part: logs", "see plan A12")
            run.at("A11")
            run.passed("A11", "report schema checked; probe records and the synthetic result discarded", "no key outside the schema")
        except DiagLatError as exc:
            where = "A10" if exc.code == GD.RESOURCE_LIMIT and run.current != "A9" else run.current
            rep["stop"] = {"code": exc.code, "message": exc.message, "criterion": where}
            run.crit[where].update(status="FAIL", stop=exc.code, observed=exc.message)
            LOG.error("F1 stopped at criterion %s: %s", where, exc)
        rep["fixtures"] = run.fx
        rep["criteria"] = run.crit
        ok = rep["stop"] is None and all(run.crit[c]["status"] == "PASS" for c in CRITERIA)
        rep["verdict"] = "F1: PASS" if ok else "F1: FAIL"
        name = "f1_report.json"
        try:
            check_report(rep)
        except DiagLatError as exc:                  # C11: no f1_report.json; a separate, validated failure report
            run.crit["A11"].update(status="FAIL", stop=exc.code, observed=exc.message)
            ok, name = False, "f1_failure_report.json"
            rep = failure_report(run.crit, exc.code, exc.message, banner, card_sha256)
            LOG.error("the F1 report failed its schema; %s is written instead and is not F1 evidence", name)
        verdict = "".join("%s: %s\n" % (c, run.crit[c]["status"]) for c in CRITERIA) + ("F1: PASS\n" if ok else "F1: FAIL\n")
        files = {name: dumps(rep), "f1_verdict.txt": verdict.encode(), "f1.log": ("\n".join(mem.lines) + "\n").encode()}
        for name, data in files.items():
            guard.write(name, data)
        guard.write("outputs_sha256.txt", "".join("%s  %s\n" % (sha_bytes(d), n) for n, d in sorted(files.items())).encode())
        return rep, (0 if ok else 1)
    finally:
        root.removeHandler(mem)


# ---- bind-only preflight (C6, C8) -------------------------------------------------------------
def bind_only(repo, manifest_path, card, out_dir, card_sha256):
    """Manifest structure, file identities, role-to-state mapping and stored-row order on the real
    results folder. No PyTorch, no checkpoint tensor, no inference. Read-only with respect to the
    existing evidence and the checkpoints; writes one small report into a new folder."""
    try:
        os.makedirs(out_dir, exist_ok=False)
    except OSError as exc:
        _stop(GD.GUARD_STATE, "cannot create the new output folder %s: %s" % (out_dir, exc))
    rep = {"schema": BIND_SCHEMA, "run_card_sha256": card_sha256, "driver": __version__, "binder": BD.__version__,
           "results_root": BD.results_root(repo), "stop": None, "outcome": None,
           "not_verified_by_this_preflight": card["not_verified_before_run"]}
    try:
        reg = card["registered"]
        bound = BD.bind_manifest(manifest_path, f2_sha256=card["manifest_f2_sha256"], n_states=reg["n_states"],
                                 n_volumes=reg["n_volumes"], pin=card["pin"])
        files = BD.verify_files(bound, rep["results_root"])
        order = bound["state_order"]
        rep["outcome"] = {
            "manifest_sha256": bound["manifest_sha256"], "n_role_assignments": len(bound["role_map"]), "n_states": len(order),
            "n_checkpoint_files": bound["n_files"], "shared_states": bound["shared_states"], "seeds": list(bound["seeds"]),
            "role_map": {"%s|%s" % k: v for k, v in sorted(bound["role_map"].items(), key=repr)},
            "states": [{"index": k, "model_state_sha256": sid, "arm": bound["states"][sid]["arm"], "seed": bound["states"][sid]["seed"],
                        "epoch": bound["states"][sid]["epoch"], "files": bound["states"][sid]["files"]} for k, sid in enumerate(order)],
            "files_hashed": files["files_hashed"], "n_rows_per_stored_file": files["n_rows_per_file"],
            "stored_row_order": "every stored-row file lists the validation slices in the order of the eligibility manifest",
            "n_midpoint_slices": len(bound["identities"]), "n_volumes": len(set(files["volume_ids"])),
            "volume_order": files["volume_order"], "n_eligible": int(sum(files["eligible"])),
            "stored_values_finite": True, "torch_imported": "torch" in sys.modules}
        if rep["outcome"]["torch_imported"]:
            _stop(A.ADAPTER, "PyTorch was imported in the bind-only mode")
    except DiagLatError as exc:
        rep["stop"] = {"code": exc.code, "message": exc.message}
    ok = rep["stop"] is None
    guard = GD.WriteGuard(out_dir, card["ceilings"]["disk_bytes"], card["ceilings"]["reserve_bytes"])
    files = {"bind_report.json": dumps(rep), "bind_verdict.txt": ("BIND: PASS\n" if ok else "BIND: FAIL\n").encode()}
    for name, data in files.items():
        guard.write(name, data)
    guard.write("outputs_sha256.txt", "".join("%s  %s\n" % (sha_bytes(d), n) for n, d in sorted(files.items())).encode())
    return rep, (0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description="D-LAT F1 preparation driver (writes no measure)", allow_abbrev=False)
    ap.add_argument("--run-card", required=True)
    ap.add_argument("--run-card-sha256", required=True)
    ap.add_argument("--repo", required=True, help="the pinned V05LT checkout (read-only)")
    ap.add_argument("--manifest", required=True, help="the registered D-UNC manifest_f2.json")
    ap.add_argument("--out", required=True, help="a new output folder; it must not exist")
    ap.add_argument("--dunc-dir", help="folder of the registered diag_unc_adapter.py (full run only; verified by sha256)")
    ap.add_argument("--bind-only", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        card, card_sha = load_card(args.run_card, args.run_card_sha256)
        if card["selftest"]:
            _stop(RUN_CARD, "a self-test run card is not accepted by the production entry point")
        if args.bind_only:
            rep, code = bind_only(args.repo, args.manifest, card, args.out, card_sha)
            print("BIND: PASS" if code == 0 else "BIND: FAIL (%s)" % rep["stop"]["code"])
            return code
        if not args.dunc_dir or not os.path.isdir(args.dunc_dir):
            _stop(A.ORIGIN, "--dunc-dir is required for the full run and must be an existing folder")
        sys.path.append(os.path.realpath(args.dunc_dir))             # after the D-LAT code folder: nothing is shadowed
        sys.path.append(os.path.realpath(args.repo))                 # the pinned checkout (as the D-UNC drivers do)
        rep, code = run_f1(lambda: ProductionBinding(args.repo, args.manifest, card), card, args.out, card_sha)
        print(rep["verdict"])
        return code
    except DiagLatError as exc:
        LOG.error("%s did not start or could not write its evidence: %s", "the bind-only preflight" if args.bind_only else "F1", exc)
        print("%s: FAIL (%s)" % ("BIND" if args.bind_only else "F1", exc.code))
        return 1
    except Exception as exc:                     # logged and re-raised: never swallowed
        LOG.error("unexpected error: %s: %s", type(exc).__name__, exc)
        raise


if __name__ == "__main__":
    sys.exit(main())

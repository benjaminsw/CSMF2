# SEQREF-DLAT-G v0.3 -- diag_lat_g.py: D-LAT step G driver: the 2,716 slice-state records, the population stage, rule E-G1, figure and table data (plan SEQREF-DLAT-F2-PLAN v0.3; figure plan SEQREF-DLAT-FIGURE-PLAN v0.2)
# LIFETIME: KEEP
"""Step G of the frozen specification SEQREF-MRI-DIAG-LAT v1.0 (LudWiki 235, sha256 6afa3eda...7327).

DRAFT FOR REVIEW. NOT REGISTERED. NOT TO BE RUN ON THE HOST before the implementation review, the
Git registration, the F2 verdict GO and the owner's recorded authorisation of G.

The driver holds no science. Every model operation is a call of the registered adapter
(diag_lat_adapter), every statistic, class, tag and the outcome a call of the registered
population stage (diag_lat_population.run_population), every binding check a stage of the
registered F1 driver (diag_lat_f1.F1Run: d1, d2_d3, d4_tensors, check_model, d5, slice_fixtures),
inherited unchanged. The inputs of L5 are assembled exactly as the F1 probe assembled them.
Arithmetic of this file (complete list): absolute differences from the stored rows and their
comparison with the manifest tolerance; the runtime guard W_k of 235 section 7; the comparison
of each bank round-trip residual with the E-G1 threshold; set membership for "affected".

Stages:
  G1-G6  binding, device, registered arrays (verified against the manifest), slice fixtures
  G7     for each of the n_states model states, for each slice: one record through
         slice_state_record, written at once as one file through the write guard
  G8     after every state: memory check and runtime guard (235 section 7); disk limit D through
         the write guard on every write
  G9     population stage on the records held in memory (the same objects that were written)
  G10    Table 1, E-G1 annotation, one data file per figure and supplementary output of the figure
         plan v0.2 (figdata/), report with the sha256 of every result file, log, verdict, sha256 lists

Rule E-G1 (F2 plan v0.3 section 4; introduced for G at F2, NOT part of 235): the bank round-trip
residual of every slice-state is recorded; a residual above the manifest threshold is flagged;
nothing is replaced or discarded; a non-finite residual is fatal. E-G1 is written to its own file
as a numerical-validity annotation of L4. It changes no value, interval, class, tag or outcome:
the population result is written exactly as run_population returns it.

Verdict: "G: COMPLETE" if every stage finished, else "G: FAIL". G has no scientific pass or fail.
One run, no resume: a stop ends the run; the record files written so far stay as evidence.

DISCLOSED for review (not in the wording of the plan):
  (1) the stored-row check (recomputed NLL, B, J against the stored rows, 1e-4 nats/dim) is
      repeated for every slice-state from the encode the record needs anyway; a failure stops;
  (2) python, torch and numpy versions and the torch thread count must equal those of the F1
      run named in the manifest (235 section 7: same device and environment for F1 and G);
  (3) whether an outcome letter "rests on an affected L4 result alone" is NOT decided by this
      code: if L4 is affected the annotation says that the outcome statement must address it;
  (4) figure plan v0.2 (sha256 83ad1502...ded1), rule 9: one data file per figure. Every value in
      figdata/ is SELECTED from the population result, the stored rows or the E-G1 annotation, or
      returned by a registered function (seed_differences for Figure 5 (c), pr_isotropic_reference
      for Figure 4 (c)); this file computes nothing for a figure. Figure 1 holds the stored
      per-slice values of every state; the means by checkpoint role are formed at rendering;
  (8) SOURCE BINDINGS (v0.3; bind_sources), read-only, verified at the start of G before the first
      record and by --check-sources in the launcher's preflight:
      (a) Figure 1 (c): R1 PSNR is COPIED from the D-UNC G record chain (manifest_g.json at its
          registered sha256 -> records/index.json -> 28 state indexes -> 2,716 record files, each
          against its sidecar and its index entry); states are matched by model-state hash, slices
          by identity, position and volume. No PSNR is computed;
      (b) S6: the fixture tables of three evidence files are COPIED at their registered sha256:
          the F1 report of V15, and the self-test and P3 reports made on the host with the
          registered code. No fixture is re-run. Paths and hashes come from the manifest only;
  (7) a record costs 264 flow passes (132 encode, 132 decode), as the F1 probe record did; 235
      section 7 counts 262. The two extra are the target encode and the R0 decode that
      slice_state_record repeats after this driver has made them for the L5 inputs. They use no
      random draw and return the same values (tested);
  (6) the write guard is created with the registered disk CEILING; the operative limit D is
      enforced by this driver before every record file and every result file (put). The report,
      verdict, log and the two sha256 lists are written after that check, so that a run stopped
      at D can still write its failure evidence; they count against the ceiling only;
  (5) the population stage reads the records from memory, not from the written files: the
      registered code tests "not computed" by object identity, which a file round trip loses.
No fallback, placeholder, mock or dummy return. Every failure is a typed DiagLatError, logged.

Changelog:
  v0.3 (2026-11-24) focused completion after the review of v0.2 (LudWiki 256): source bindings for
        Figure 1 (c) and S6 (bind_sources, manifest part "sources"), verified at the start of G;
        mode --check-sources for the read-only preflight (no model, no inference, writes nothing).
        Unchanged: the record pathway, the population call, the guards and limits, E-G1.
  v0.2 (2026-11-24) narrow completion release after the review of v0.1 (LudWiki 255 LE100, LE101):
        figure and table data of the figure plan v0.2 (figure_files; folder figdata/; output schema
        v0.2); the E-G1 reading text approved at F2.4 replaces the v0.1 text; the report holds the
        sha256 of every result file; Table 1 carries the exploratory-status sentence. Unchanged:
        the record pathway, the population call, the guards and limits, the E-G1 flag logic.
  v0.1 (2026-11-24) first version (draft for review).
"""
import sys

FORBIDDEN = ("--checkpoint", "--data-root")
EXIT_REFUSED = 3


def refuse_forbidden(argv):
    """Standard library only; runs before NumPy, PyTorch or any project module is imported."""
    hit = sorted({f for a in argv for f in FORBIDDEN if a == f or a.startswith(f + "=")})
    if hit:
        sys.stderr.write("ERROR DIAG_LAT_REFUSED: option(s) %s are not accepted: G takes checkpoints and data "
                         "only through the registered manifest\n" % ", ".join(hit))
        raise SystemExit(EXIT_REFUSED)


if __name__ == "__main__":
    refuse_forbidden(sys.argv[1:])

import argparse  # noqa: E402
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
import diag_lat_f1 as F  # noqa: E402
import diag_lat_guard as GD  # noqa: E402
import diag_lat_population as PO  # noqa: E402
import diag_lat_stats as ST  # noqa: E402
from diag_lat_core import DiagLatError  # noqa: E402

__version__ = "0.3"
ABBR = "SEQREF-DLAT-G"
LOG = logging.getLogger("diag_lat.g")

MANIFEST_SCHEMA = "seqref-dlat-g-manifest/1"
REPORT_SCHEMA = "seqref-dlat-g-report/1"
RECORD_SCHEMA = "seqref-dlat-g-record/1"
EG1_SCHEMA = "seqref-dlat-g-eg1/1"
TABLE_SCHEMA = "seqref-dlat-g-main-table/1"
FIG_SCHEMA = "seqref-dlat-g-figdata/1"
G_MANIFEST = "DIAG_LAT_G_MANIFEST"
CRITERIA = tuple("G%d" % k for k in range(1, 11))
FROM_F1 = {"A1": "G1", "A4": "G1", "A2": "G2", "A3": "G3", "A5": "G4", "A8": "G5", "A7": "G6"}   # inherited stages
MAN_KEYS = {"schema", "selftest", "rules_version", "specification", "device", "manifest_f2_sha256", "pin", "registered",
            "arrays", "tolerances", "e_g1", "limits", "environment", "f1_evidence", "figure_plan", "sources", "review_status",
            "not_verified_before_run"}
SUB_KEYS = {"specification": {"article", "version", "sha256"},
            "arrays": {"bootstrap_sha256", "noise_sha256", "projection_sha256"},
            "tolerances": {"stored_rows_nats_per_dim", "s1_s2_p1_relative"},
            "e_g1": {"rule", "bank_roundtrip_flag_above", "statement", "provenance"},
            "limits": {"L_s", "D_bytes", "t_stat_s", "g_wall_ceiling_s", "disk_ceiling_bytes", "ram_bytes", "reserve_bytes"},
            "environment": {"python", "torch", "numpy", "torch_threads"},
            "f1_evidence": {"run", "report_sha256", "run_card_sha256", "commit", "target_roundtrip_max_abs",
                            "target_roundtrip_n", "target_roundtrip_tolerance", "peak_rss_bytes"},
            "figure_plan": {"name", "version", "sha256", "status"},
            "sources": {"dunc_g_out", "dunc_g_manifest_sha256", "f1_report", "f1_report_sha256", "selftest_report",
                        "selftest_report_sha256", "p3_report", "p3_report_sha256", "note"}}
EVIDENCE = ("f1_report", "selftest_report", "p3_report")
DUNC_SCHEMAS = ("seqref-dunc-g-manifest/1", "seqref-dunc-g-index/1", "seqref-dunc-g-state-index/1", "seqref-dunc-g-record/1")
REPORT_KEYS = {"schema", "banner", "review_status", "versions", "manifest", "environment", "criteria", "binding", "device",
               "arrays", "fixtures", "records", "resources", "numerical", "outputs", "stop", "verdict"}
EG1_KEYS = {"schema", "rule", "threshold", "statement", "provenance", "reading", "n_slice_states", "n_flagged", "max",
            "flagged", "affected_states", "contrasts", "l4_summary", "l4_outcome_note", "residuals"}
UNRESOLVED, NO_FLAG = "numerical adequacy unresolved", "no numerical flag"
READING = ("An affected L4 result is reported with the annotation 'numerical adequacy unresolved' beside its class, tags and "
           "stability label. The annotation withdraws and changes none of them and does not make an outcome provisional. A flag is "
           "not a finding of numerical invalidity: nothing is deleted or replaced. Whether a flagged L4 result is numerically "
           "adequate is decided by review, outside the frozen rules.")       # approved at F2.4 (LudWiki 255 LE101)
OUTCOME_NOTE = ("L4 is affected on a read gap contrast: the outcome statement must say whether the outcome letter rests on "
                "L4 alone. This code does not decide that.")


def _stop(code, message):
    raise DiagLatError(code, message)            # DiagLatError logs at construction


# ---- manifest -------------------------------------------------------------------------------
def load_manifest(path, sha256):
    """The reviewed G manifest: sha256 given on the command line; exact key sets; no default."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        _stop(G_MANIFEST, "cannot read the G manifest %s: %s" % (path, exc))
    got = F.sha_bytes(raw)
    if got != sha256:
        _stop(G_MANIFEST, "G manifest %s has sha256 %s, expected %s" % (path, got, sha256))
    try:
        man = json.loads(raw)
    except ValueError as exc:
        _stop(G_MANIFEST, "G manifest is not JSON: %s" % exc)
    check_manifest(man)
    return man, got


def _hex64(v):
    return isinstance(v, str) and len(v) == 64 and all(ch in "0123456789abcdef" for ch in v)


def _posint(v):
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def _posfloat(v):
    return isinstance(v, float) and math.isfinite(v) and v > 0.0


def check_manifest(man):
    if not isinstance(man, dict) or set(man) != MAN_KEYS:
        got = set(man) if isinstance(man, dict) else set()
        _stop(G_MANIFEST, "G manifest keys: missing %s, unexpected %s" % (sorted(MAN_KEYS - got), sorted(got - MAN_KEYS)))
    for name, keys in SUB_KEYS.items():
        if not isinstance(man[name], dict) or set(man[name]) != keys:
            _stop(G_MANIFEST, "G manifest part %r must have exactly the keys %s" % (name, sorted(keys)))
    lim, env, tol, reg, eg, ev, fig = (man[k] for k in ("limits", "environment", "tolerances", "registered", "e_g1",
                                                         "f1_evidence", "figure_plan"))
    src = man["sources"]
    if not (all(isinstance(src[k], str) and os.path.isabs(src[k]) for k in ("dunc_g_out",) + EVIDENCE)
            and all(_hex64(src[k]) for k in ("dunc_g_manifest_sha256",) + tuple(e + "_sha256" for e in EVIDENCE))
            and isinstance(src["note"], str) and src["note"].strip()):
        _stop(G_MANIFEST, "G manifest part 'sources': every path must be absolute and every sha256 complete")
    ok = (man["schema"] == MANIFEST_SCHEMA and isinstance(man["selftest"], bool) and man["rules_version"] == ST.RULES_VERSION
          and all(isinstance(man[k], str) and man[k].strip() for k in ("device", "manifest_f2_sha256", "pin", "review_status",
                                                                       "not_verified_before_run"))
          and isinstance(reg, dict) and set(reg) == set(F.REGISTERED) and all(_posint(v) for v in reg.values())
          and all(_hex64(v) for v in man["arrays"].values()) and _hex64(man["specification"]["sha256"])
          and all(_posfloat(v) for v in tol.values()) and _posfloat(eg["bank_roundtrip_flag_above"])
          and all(isinstance(eg[k], str) and eg[k].strip() for k in ("rule", "statement", "provenance"))
          and _posfloat(lim["L_s"]) and _posfloat(lim["t_stat_s"])
          and all(_posint(lim[k]) for k in ("D_bytes", "g_wall_ceiling_s", "disk_ceiling_bytes", "ram_bytes", "reserve_bytes"))
          and all(isinstance(env[k], str) and env[k] for k in ("python", "torch", "numpy")) and _posint(env["torch_threads"])
          and all(isinstance(ev[k], str) and ev[k] for k in ("run", "commit")) and _hex64(ev["report_sha256"])
          and _hex64(ev["run_card_sha256"]) and _posfloat(ev["target_roundtrip_max_abs"])
          and _posfloat(ev["target_roundtrip_tolerance"]) and _posint(ev["target_roundtrip_n"]) and _posint(ev["peak_rss_bytes"])
          and all(isinstance(fig[k], str) and fig[k] for k in ("name", "version", "status")) and _hex64(fig["sha256"]))
    if not ok:
        _stop(G_MANIFEST, "G manifest: schema, rules version, a hash, a tolerance, a limit or a text is missing or malformed")
    if man["device"] != "cpu":
        _stop(F.REPRODUCTION_MISMATCH, "G manifest device %r: this code runs the flow passes on cpu only; there is no "
              "transfer code and no fallback" % man["device"])
    if tol["s1_s2_p1_relative"] != A.SWAP_REL_TOL:
        _stop(G_MANIFEST, "G manifest S1/S2/P1 tolerance %r differs from the adapter's %r" % (tol["s1_s2_p1_relative"], A.SWAP_REL_TOL))
    if lim["L_s"] > lim["g_wall_ceiling_s"] or lim["D_bytes"] > lim["disk_ceiling_bytes"]:
        _stop(G_MANIFEST, "G manifest: L or D exceeds its ceiling")
    if ev["target_roundtrip_max_abs"] > ev["target_roundtrip_tolerance"]:
        _stop(G_MANIFEST, "G manifest: the cited F1 target round trip exceeds its tolerance")
    if not man["selftest"]:
        if reg != F.REGISTERED:
            _stop(G_MANIFEST, "G manifest sizes %r differ from the registered %r" % (reg, F.REGISTERED))
        if src["f1_report_sha256"] != ev["report_sha256"]:
            _stop(G_MANIFEST, "G manifest: the bound F1 report is not the cited F1 evidence")
        if man["manifest_f2_sha256"] != BD.REGISTERED_F2_SHA256 or man["pin"] != BD.PIN:
            _stop(G_MANIFEST, "G manifest names manifest %s and pin %s; the registered ones are %s and %s"
                  % (man["manifest_f2_sha256"], man["pin"], BD.REGISTERED_F2_SHA256, BD.PIN))


def card_view(man):
    """The fields the inherited F1 stages read (F1Run reads card[...]); no F1 tolerance is used by G."""
    lim = man["limits"]
    return {"selftest": man["selftest"], "device": man["device"], "manifest_f2_sha256": man["manifest_f2_sha256"],
            "pin": man["pin"], "registered": man["registered"], "tolerances": man["tolerances"],
            "ceilings": {"g_wall_s": lim["g_wall_ceiling_s"], "disk_bytes": lim["disk_ceiling_bytes"],
                         "reserve_bytes": lim["reserve_bytes"], "ram_bytes": lim["ram_bytes"]}}


# ---- source bindings (v0.3): existing results, read-only ------------------------------------
def _read(path, want, what, sidecar=False):
    """Bytes of an existing evidence file at its registered sha256 (and, for D-UNC files, equal to its sidecar)."""
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        side = None
        if sidecar:
            with open(path + ".sha256", "r") as fh:
                side = (fh.read().split() or [""])[0]
    except OSError as exc:
        _stop(A.ORIGIN, "%s: %s or its sidecar cannot be read (%s)" % (what, path, type(exc).__name__))
    got = F.sha_bytes(data)
    if got != want or (sidecar and got != side):
        _stop(A.ORIGIN, "%s: %s has sha256 %s, expected %s%s" % (what, path, got, want, " (sidecar %s)" % side if sidecar else ""))
    return data


def _json(data, what):
    try:
        return json.loads(data)
    except ValueError as exc:
        _stop(A.ORIGIN, "%s is not JSON: %s" % (what, exc))


def bind_r1(src, manifest_f2_sha256, state_order, identities, volume_ids):
    """R1 PSNR of every slice-state from the D-UNC G record chain: {state id: [value per slice, D-LAT slice order]}.
    Every file against its sidecar and the hash its parent names; states by model-state hash; slices by identity,
    position and volume. Values are copied; nothing is computed."""
    root, n = src["dunc_g_out"], len(identities)
    man_g = _json(_read(os.path.join(root, "manifest_g.json"), src["dunc_g_manifest_sha256"], "D-UNC G manifest", True), "D-UNC G manifest")
    if not (isinstance(man_g, dict) and man_g.get("schema") == DUNC_SCHEMAS[0] and man_g.get("selftest") is False
            and man_g.get("manifest_f2_sha256") == manifest_f2_sha256 and _hex64(man_g.get("stage1_index_sha256"))):
        _stop(ST.POPULATION_MISMATCH, "D-UNC G manifest: schema, self-test flag, F2 manifest or stage-1 index hash differ from the registered binding")
    final = _json(_read(os.path.join(root, "records", "index.json"), man_g["stage1_index_sha256"], "D-UNC final index", True), "D-UNC final index")
    states = final.get("states") if isinstance(final, dict) else None
    if not (isinstance(states, list) and final.get("schema") == DUNC_SCHEMAS[1] and final.get("n_states") == len(states) == len(state_order)
            and final.get("n_records") == len(state_order) * n
            and sorted(str(s.get("model_state_sha256")) for s in states) == sorted(str(s) for s in state_order)):
        _stop(ST.POPULATION_MISMATCH, "D-UNC final index: schema, counts or the set of model states differ from the D-LAT binding")
    out, n_files = {}, 2
    for s in states:
        sid = s["model_state_sha256"]
        sidx = _json(_read(os.path.join(root, s["file"]), s["sha256"], "D-UNC state index %s" % sid[:12], True), "D-UNC state index")
        ents = sidx.get("records") if isinstance(sidx, dict) else None
        if not (isinstance(ents, list) and sidx.get("schema") == DUNC_SCHEMAS[2] and sidx.get("model_state_sha256") == sid
                and sidx.get("state_index") == s.get("state_index") and len(ents) == n
                and [e.get("position") for e in ents] == list(range(n))
                and [tuple(e.get("identity") or ()) for e in ents] == [tuple(x) for x in identities]
                and [e.get("volume_id") for e in ents] == list(volume_ids) and all(e.get("gate") == "pass" for e in ents)):
            _stop(ST.POPULATION_MISMATCH, "D-UNC state index %s: schema, state, positions, slice identities, volumes or gates differ from the "
                  "D-LAT binding" % sid[:12])
        vals = []
        for i, e in enumerate(ents):
            body = _json(_read(os.path.join(root, e["file"]), e["sha256"], "D-UNC record %s/%d" % (sid[:12], i), True), "D-UNC record")
            if not (isinstance(body, dict) and body.get("schema") == DUNC_SCHEMAS[3] and body.get("model_state_sha256") == sid
                    and body.get("position") == i and body.get("volume_id") == volume_ids[i]
                    and (body.get("file"), body.get("slice_index")) == tuple(identities[i])):
                _stop(ST.POPULATION_MISMATCH, "D-UNC record %s/%d: schema, state, position, volume or slice identity differ" % (sid[:12], i))
            v = body.get("r1_psnr_db")
            if not (isinstance(v, float) and math.isfinite(v)):
                _stop(C.NONFINITE, "D-UNC record %s/%d: r1_psnr_db is missing or not finite" % (sid[:12], i))
            vals.append(v)
        out[sid], n_files = vals, n_files + 1 + n
    return out, {"files_verified": n_files, "stage1_index_sha256": man_g["stage1_index_sha256"], "n_states": len(out), "n_records": len(out) * n}


def bind_sources(man, state_order, identities, volume_ids):
    """All source bindings of v0.3. Returns {"r1", "r1_info", "evidence": {name: parsed report}}."""
    src = man["sources"]
    r1, info = bind_r1(src, man["manifest_f2_sha256"], state_order, identities, volume_ids)
    evidence = {}
    for name in EVIDENCE:
        doc = _json(_read(src[name], src[name + "_sha256"], name), name)
        if not (isinstance(doc, dict) and isinstance(doc.get("fixtures"), (dict, list)) and doc["fixtures"]):
            _stop(A.ORIGIN, "%s: %s holds no fixture table" % (name, src[name]))
        evidence[name] = doc
    LOG.info("source bindings verified: D-UNC chain of %d files (R1 PSNR of %d slice-states); F1, self-test and P3 reports at their sha256",
             info["files_verified"], info["n_records"])
    return {"r1": r1, "r1_info": info, "evidence": evidence}


# ---- rule E-G1 ------------------------------------------------------------------------------
def eg1(residuals, state_order, identities, role_map, seeds, spec):
    """residuals: {state id: [bank round-trip residual of each slice, registered order]}.
    Flag: residual > threshold. Affected state: at least one flagged slice-state. Affected
    contrast: a role with non-zero weight is mapped, for some seed, to an affected state.
    Labels for L4 only. Nothing here touches a measure, class, tag or the outcome."""
    thr = spec["bank_roundtrip_flag_above"]
    flagged, worst = [], None
    for k, sid in enumerate(state_order):
        for i, r in enumerate(residuals[sid]):
            if not math.isfinite(r):
                _stop(C.NONFINITE, "E-G1: state %s slice %d: non-finite bank round-trip residual" % (str(sid)[:12], i))
            item = {"state": str(sid), "state_index": k, "slice_index": i, "identity": list(identities[i]), "residual": float(r)}
            if worst is None or r > worst["residual"]:
                worst = item
            if r > thr:
                flagged.append(item)
    affected = sorted({f["state"] for f in flagged})
    contrasts = {}
    for name, weights in ST.ALL_CONTRASTS.items():
        hits = sorted("%s|%s" % (role, seed) for role, w in weights.items() if w != 0.0 for seed in seeds
                      if str(role_map[(role, seed)]) in affected)
        contrasts[name] = {"label": UNRESOLVED if hits else NO_FLAG, "affected_role_seed": hits}
    gap = any(contrasts[c]["label"] == UNRESOLVED for c in ST.READ_GAP)
    return {"schema": EG1_SCHEMA, "rule": spec["rule"], "threshold": thr, "statement": spec["statement"],
            "provenance": spec["provenance"], "reading": READING,
            "n_slice_states": sum(len(residuals[s]) for s in state_order), "n_flagged": len(flagged), "max": worst,
            "flagged": flagged, "affected_states": affected, "contrasts": contrasts,
            "l4_summary": UNRESOLVED if gap else NO_FLAG, "l4_outcome_note": OUTCOME_NOTE if gap else None,
            "residuals": {str(s): [float(r) for r in residuals[s]] for s in state_order}}


def runtime_guard(elapsed, k, n_states, t_max, t_stat, limit):
    """235 section 7: W_k = t_elapsed + (n - k) T_max + t_stat; stop if W_k > L. Timing figures only."""
    w = elapsed + (n_states - k) * t_max + t_stat
    if w > limit:
        _stop(GD.RESOURCE_LIMIT, "runtime guard after state %d of %d: W_k = %.0f s exceeds L = %.3f s (elapsed %.0f s, "
              "longest state %.1f s, t_stat %.1f s)" % (k, n_states, w, limit, elapsed, t_max, t_stat))
    return w


# ---- the run --------------------------------------------------------------------------------
class GRun(F.F1Run):
    """The binding stages are those of the registered F1 driver; their criterion names are mapped."""

    def __init__(self, man, out_dir, man_sha256):
        super().__init__(card_view(man), out_dir, man_sha256)
        self.man, self.lim = man, man["limits"]
        self.crit = {c: {"status": F.NOT_REACHED, "observed": None, "limit": None, "stop": None} for c in CRITERIA}
        self.current = "G1"
        self.rep = {k: None for k in REPORT_KEYS}

    def at(self, criterion):
        self.current = FROM_F1.get(criterion, criterion)

    def passed(self, criterion, observed, limit):
        self.crit[FROM_F1.get(criterion, criterion)].update(status="PASS", observed=observed, limit=limit)

    def wall(self):
        el = time.monotonic() - self.t0
        if el > self.lim["L_s"]:
            _stop(GD.RESOURCE_LIMIT, "G wall time %.0f s exceeds the operative limit L = %.3f s" % (el, self.lim["L_s"]))
        return el

    def put(self, guard, name, data):
        """A record or result file: the operative disk limit D first (235 section 7), then the write guard."""
        self.at("G8")
        total = guard.written + guard.charge(len(data))
        if total > self.lim["D_bytes"]:
            _stop(GD.RESOURCE_LIMIT, "writing %s would bring the cumulative disk count to %d bytes, above the operative limit "
                  "D = %d" % (name, total, self.lim["D_bytes"]))
        return guard.write(name, data)

    def g1(self):
        self.d1()                                # -I -B, code hashes, versions (inherited)
        self.rep.pop("run_card", None)           # the F1 stage's key; G reports "manifest"
        self.rep["manifest"] = {"sha256": self.card_sha, "content": self.man}
        self.rep["versions"]["g_driver"] = __version__
        self.rep["versions"]["f1_driver_stages"] = self.rep["versions"].pop("driver")
        env, want = self.rep["environment"], self.man["environment"]
        if (env["python"], env["numpy"]) != (want["python"], want["numpy"]):
            _stop(F.REPRODUCTION_MISMATCH, "python %s and numpy %s; the manifest (F1 run) names %s and %s"
                  % (env["python"], env["numpy"], want["python"], want["numpy"]))

    def g5(self):
        self.d5()                                # the three one-call draws, layout, scaled noise (inherited)
        self.at("G5")
        arr, want = self.rep["arrays"], self.man["arrays"]
        if self.idx_sha != want["bootstrap_sha256"]:
            _stop(ST.BOOTSTRAP_IDENTITY, "bootstrap matrix %s differs from the registered %s" % (self.idx_sha, want["bootstrap_sha256"]))
        for name in ("noise", "projection"):
            if arr[name]["sha256"] != want[name + "_sha256"]:
                _stop(F.REPRODUCTION_MISMATCH, "%s array %s differs from the registered %s: a validity failure, not a reason "
                      "to draw again" % (name, arr[name]["sha256"], want[name + "_sha256"]))
        arr["verified_against_manifest"] = True
        self.passed("G5", "bootstrap, noise and projection equal the identities registered in F1; layout checked on %d slices"
                    % len(self.states), "equality")

    def g6(self):
        self.slice_fixtures()                    # S2, S3, S4 and the P1 input on every slice (inherited)
        self.passed("G6", {q: v["max"] for q, v in self.fx.items() if q.startswith(("S2", "P1"))},
                    "S2, P1: %g relative; S3 donor map; S4 > 0" % A.SWAP_REL_TOL)

    def check_environment(self):
        m, want = self.rep["device"]["models"], self.man["environment"]
        if (m["torch"], m["torch_threads"]) != (want["torch"], want["torch_threads"]):
            _stop(F.REPRODUCTION_MISMATCH, "torch %s with %d threads; the manifest (F1 run) names %s with %d"
                  % (m["torch"], m["torch_threads"], want["torch"], want["torch_threads"]))

    def record(self, model, st, i, sid):
        """One slice-state record. The inputs of L5 are assembled as in the F1 probe (diag_lat_f1.probe_record)."""
        sw = self.swaps[i]
        own = A.encode_truth(model, st)
        stored = self.b.stored[sid]
        for name in ("NLL", "B", "J"):           # disclosed item (1)
            if not self.worst("stored_%s_abs_diff" % name, abs(own[name] - float(stored[name][i])), self.tol["stored_rows_nats_per_dim"]):
                _stop(A.ADAPTER, "state %s slice %d: recomputed %s %.6f, stored %.6f (tolerance %g)"
                      % (str(sid)[:12], i, name, own[name], float(stored[name][i]), self.tol["stored_rows_nats_per_dim"]))
        p_swap = C.swap_penalty(A.encode_truth(model, st, sw["cond_swap"], sw["mask"])["NLL"], own["NLL"])
        cond_p = A.perturbed_input(st, self.scaled[i])
        enc_p = A.encode_truth(model, st, cond_p, st["mask"])
        r0 = A.decode_r0_std(model, st).astype(np.float64)
        sens = C.l5b_sensitivities(r0, A.decode_r0_std(model, st, cond_p, st["mask"]).astype(np.float64),
                                   own["z_star"].astype(np.float64), enc_p["z_star"].astype(np.float64), own["NLL"], enc_p["NLL"])
        r0_sw = A.decode_r0_std(model, st, sw["cond_swap"], sw["mask"]).astype(np.float64)
        scale, mean = A.scale_mean_interleaved(st)
        return A.slice_state_record(
            model, st, self.bank, subsets=C.STABILITY_SUBSETS, p_swap=p_swap, r0_psnr=float(stored["R0_PSNR"][i]),
            swap_shift_r0=C.shift_norm(r0_sw, r0),
            phys_swap_shift_r0=C.shift_norm(C.to_physical(r0_sw, scale, mean), C.to_physical(r0, scale, mean)),
            sensitivities=sens, proj=self.proj)

    def g7_g8(self, guard):
        b, n, n_states = self.b, len(self.states), len(self.b.state_order)
        thr = self.man["e_g1"]["bank_roundtrip_flag_above"]
        self.records, self.events, self.residuals, self.record_files = {}, [], {}, {}
        per_state, t_max, w_max = [], 0.0, 0.0
        for k, sid in enumerate(b.state_order):
            self.at("G2")
            t_state = time.monotonic()
            model, ver = b.load_state(sid)
            t_load = time.monotonic() - t_state
            if ver["model_state_sha256"] != sid or ver["file_sha256"] != b.states[sid]["file_sha256"]:
                _stop(A.ORIGIN, "state %s: the loader returned model state %s, file %s; the manifest names %s, %s"
                      % (str(sid)[:12], ver["model_state_sha256"], ver["file_sha256"], sid, b.states[sid]["file_sha256"]))
            self.at("G4")
            self.check_model(model, sid)
            self.check_environment()
            rows, res = [], []
            for i, st in enumerate(self.states):
                self.at("G7")
                rec, diag = self.record(model, st, i, sid)
                rt = diag["roundtrip_max_abs"]
                if not math.isfinite(rt):
                    _stop(C.NONFINITE, "state %s slice %d: non-finite bank round-trip residual" % (str(sid)[:12], i))
                events = [dict(e, state=str(sid), state_index=k, slice_index=i, volume=str(b.volume_ids[i])) for e in diag["events"]]
                name = "records/s%02d_%s/v%03d.json" % (k, str(sid)[:12], i)
                self.record_files[name] = self.put(guard, name, F.dumps(
                    {"schema": RECORD_SCHEMA, "state": str(sid), "state_index": k, "slice_index": i, "identity": st["identity"],
                     "volume": str(b.volume_ids[i]), "record": rec,
                     "diagnostics": {q: v for q, v in diag.items() if q != "events"}, "events": events}))
                rows.append(rec)
                res.append(rt)
                self.events.extend(events)
                if rt > thr:
                    LOG.warning("E-G1 flag: state %s (index %d) slice %d: bank round-trip residual %.6e above %g; recorded, "
                                "nothing replaced", str(sid)[:12], k, i, rt, thr)
            self.records[sid], self.residuals[sid] = rows, res
            del model
            self.at("G8")
            peak = GD.check_memory(self.lim["ram_bytes"])
            t_this = time.monotonic() - t_state
            t_max = max(t_max, t_this)
            el = self.wall()
            w = runtime_guard(el, k + 1, n_states, t_max, self.lim["t_stat_s"], self.lim["L_s"])
            w_max = max(w_max, w)
            per_state.append({"state_index": k, "state": str(sid), "load_s": t_load, "state_s": t_this, "elapsed_s": el,
                              "W_k_s": w, "peak_rss_bytes": peak, "events": sum(1 for e in self.events if e["state_index"] == k),
                              "bank_roundtrip_max_abs": max(res)})
            LOG.info("state %d of %d done: %d records, %.1f s (load %.2f s), elapsed %.0f s, W_k %.0f s of L %.0f s, peak memory "
                     "%d MiB, bank round-trip max %.3e", k + 1, n_states, n, t_this, t_load, el, w, self.lim["L_s"],
                     peak // 2 ** 20, max(res))
        n_rec = sum(len(v) for v in self.records.values())
        if n_rec != n_states * n or len(self.record_files) != n_rec:
            _stop(ST.POPULATION_MISMATCH, "%d records and %d record files, expected %d" % (n_rec, len(self.record_files), n_states * n))
        self.passed("G2", "%d states loaded through the pinned loader; file and model-state hash of each equal the manifest" % n_states, "equality")
        self.passed("G4", "device %s, float32 flow passes, float64 statistics; torch %s, %d threads, as in the F1 run"
                    % (self.card["device"], self.rep["device"]["models"]["torch"], self.rep["device"]["models"]["torch_threads"]),
                    self.man["environment"])
        self.passed("G7", {"records": n_rec, "record_files": len(self.record_files), "adapter_events": len(self.events),
                           "stored_rows_max_abs_diff": {q: v["max"] for q, v in self.fx.items() if q.startswith("stored_")}},
                    {"records": n_states * n, "stored_rows_nats_per_dim": self.tol["stored_rows_nats_per_dim"]})
        self.rep["records"] = {"n": n_rec, "layout": "records/s<state index>_<state id, 12 characters>/v<slice index>.json",
                               "schema": RECORD_SCHEMA, "adapter_events": len(self.events)}
        self.rep["resources"] = {"per_state": per_state, "T_max_s": t_max, "W_k_max_s": w_max, "L_s": self.lim["L_s"],
                                 "t_stat_manifest_s": self.lim["t_stat_s"]}

    def g9(self):
        self.at("G9")
        b, reg = self.b, self.reg
        t = time.monotonic()
        self.result = PO.run_population(self.records, b.role_map, list(b.volume_ids), b.seeds, self.idx,
                                        self.man["arrays"]["bootstrap_sha256"], reg["n_states"], self.events, n_boot=reg["n_boot"])
        self.numerical = eg1(self.residuals, b.state_order, [tuple(x) for x in b.identities], b.role_map, b.seeds, self.man["e_g1"])
        self.at("G8")
        el = self.wall()                         # during the population stage G stops if elapsed exceeds L
        self.rep["resources"]["population_stage_s"] = time.monotonic() - t
        self.rep["resources"]["elapsed_after_population_s"] = el
        self.passed("G9", "population stage complete: %d states, %d contrasts, %d draws; rules %s"
                    % (reg["n_states"], len(ST.ALL_CONTRASTS), reg["n_boot"], self.result["rules_version"]), "registered code only")


def main_table_file(man, result, numerical):
    """235 section 5.1 main results table as the registered population stage builds it, with the
    review status, the figure plan reference and the E-G1 annotation of L4 beside it."""
    return {"schema": TABLE_SCHEMA, "id": "table1", "review_status": man["review_status"], "figure_plan": man["figure_plan"],
            "main_table": result["main_table"], "exploratory_status": result["classified"]["note"],
            "numerical_annotation_L4": {"rule": numerical["rule"], "summary": numerical["l4_summary"],
                                        "contrasts": {c: numerical["contrasts"][c]["label"] for c in PO.MAIN_TABLE_CONTRASTS},
                                        "n_flagged": numerical["n_flagged"], "max_residual": numerical["max"]["residual"],
                                        "threshold": numerical["threshold"], "reading": numerical["reading"],
                                        "outcome_note": numerical["l4_outcome_note"], "provenance": numerical["provenance"]}}


# ---- figure plan v0.2: one data file per figure, table and supplementary output (selection only) ----
EXISTING, MEASURE = "existing result", "D-LAT measure"       # figure plan rule 10
UNITS = {"NLL": "nats per dimension", "B": "nats per dimension", "J": "nats per dimension", "TC_G": "nats per dimension",
         "P_swap": "nats per dimension", "R0_PSNR": "dB"}    # figure plan rule 7; every other quantity: as defined in 235 section 3
FIG_NAMES = tuple("figdata/%s.json" % n for n in ("fig1", "fig2", "fig3", "fig4", "fig5", "s1", "s2", "s3", "s4", "s5", "s6", "s7"))
GAP_ROW = ("S-GAP0", "C-TIME0", ST.FIXED_EPOCH)              # Figure 5 (a)
INTERACTIONS = ("X-TIME", "X-FIX(10→30)")                   # Figure 5 (b); rule 2: never one without the other
SENSITIVITIES = A.SENSITIVITY_KEYS


def _unit(key):
    return UNITS.get(key, "as defined in 235 section 3")


def figure_files(man, result, numerical, run, population_sha256):
    """{relative name: bytes} for the five main figures and S1-S7 of the figure plan v0.2.
    Table 1 is g_main_table.json. Nothing is computed here: values are selected from `result`
    (g_population.json), from the stored rows and from the E-G1 annotation, or returned by the
    registered seed_differences and pr_isotropic_reference."""
    b, cls, means, reported = run.b, result["classified"]["measures"], result["state_means"], result["reported"]
    src, ev = man["sources"], run.sources["evidence"]
    pop_src = {"file": "g_population.json", "sha256": population_sha256}
    rows_src = {"file": "registered D-UNC manifest_f2.json, which binds the stored-row file of every state by sha256",
                "sha256": man["manifest_f2_sha256"]}
    roles = {role: {str(seed): {"state": str(b.role_map[(role, seed)]), "epoch": b.states[b.role_map[(role, seed)]]["epoch"],
                                "arm": b.states[b.role_map[(role, seed)]]["arm"]} for seed in b.seeds} for role in PO.ROLES}

    def contrast_panel(pid, mid, contrasts=PO.READ_CONTRASTS):
        m = cls[mid]
        return {"panel": pid, "kind": "contrast panel", "measure": mid, "key": m["key"], "unit": _unit(m["key"]), "provenance": MEASURE,
                "source": dict(pop_src, path="classified.measures.%s.effects" % mid),
                "data": {c: m["effects"][c] for c in contrasts}}

    def trajectory(pid, keys, note=None):
        return {"panel": pid, "kind": "trajectory by checkpoint role (descriptive; interpretation rests on the registered contrasts only)",
                "keys": list(keys), "units": {k: _unit(k) for k in keys}, "provenance": MEASURE,
                "source": dict(pop_src, path="state_means"), "note": note, "data": {k: means[k] for k in keys}}

    def doc(fid, title, panels, **extra):
        d = {"schema": FIG_SCHEMA, "id": fid, "title": title, "figure_plan": man["figure_plan"], "review_status": man["review_status"],
             "checkpoint_roles": roles, "panels": panels}
        d.update(extra)
        return d

    pop = PO.build_population(run.records, b.role_map, list(b.volume_ids), b.seeds)["pop"]      # registered; for seed_differences only
    per_volume = {}
    for c in PO.READ_CONTRASTS:
        per_volume[c] = {}
        for seed in b.seeds:
            d_nll = ST.seed_differences(pop, ST.ALL_CONTRASTS[c], "NLL", seed)
            d_psnr = ST.seed_differences(pop, ST.ALL_CONTRASTS[c], "R0_PSNR", seed)
            per_volume[c][str(seed)] = {"d_NLL": ST.NC if d_nll is C.NOT_COMPUTED else [float(v) for v in d_nll],
                                        "d_R0_PSNR": ST.NC if d_psnr is C.NOT_COMPUTED else [float(v) for v in d_psnr]}
    stored = {str(sid): {k: [float(v) for v in b.stored[sid][k]] for k in ("NLL", "B", "J", "R0_PSNR")} for sid in b.state_order}
    l4_note = {"rule": numerical["rule"], "summary": numerical["l4_summary"], "reading": numerical["reading"],
               "contrasts": {c: numerical["contrasts"][c]["label"] for c in PO.READ_CONTRASTS}}
    files = {
        "fig1": doc("fig1", "The gap D-LAT examines: context, existing results only, not a D-LAT result", [
            {"panel": "a", "kind": "stored per-slice values of every state; means by checkpoint role are formed at rendering",
             "keys": ["NLL", "B", "J"], "units": {k: _unit(k) for k in ("NLL", "B", "J")}, "provenance": EXISTING, "source": rows_src,
             "data": {sid: {k: v[k] for k in ("NLL", "B", "J")} for sid, v in stored.items()}},
            {"panel": "b", "kind": "stored per-slice values of every state", "keys": ["R0_PSNR"], "units": {"R0_PSNR": _unit("R0_PSNR")},
             "provenance": EXISTING, "source": rows_src, "data": {sid: {"R0_PSNR": v["R0_PSNR"]} for sid, v in stored.items()}},
            {"panel": "c", "kind": "R1 PSNR per slice of every state, copied from the D-UNC G records (not computed in step G)",
             "keys": ["R1_PSNR"], "units": {"R1_PSNR": "dB"}, "provenance": EXISTING,
             "source": {"file": os.path.join(src["dunc_g_out"], "manifest_g.json"), "sha256": src["dunc_g_manifest_sha256"],
                        "chain": "manifest_g.json -> records/index.json (%s) -> 28 state indexes -> record files, each at the sha256 its "
                                 "parent names" % run.sources["r1_info"]["stage1_index_sha256"]},
             "data": {str(sid): {"R1_PSNR": list(run.sources["r1"][sid])} for sid in b.state_order}}],
            volume_ids=list(b.volume_ids)),
        "fig2": doc("fig2", "Latent fit and decoder geometry", [
            contrast_panel("a", "L1"), contrast_panel("b", "L2"), trajectory("c", ("log_gamma_truth", "log_gamma_bank")),
            trajectory("d", ("q_z",))]),
        "fig3": doc("fig3", "z = 0 readout against the posterior centre and spread", [
            contrast_panel("a", "L3a"), contrast_panel("b", "L3b"), trajectory("c", ("e0_sq", "e1_sq", "tr_sigma_per_dim"))]),
        "fig4": doc("fig4", "Distributional structure; observation-content discrimination", [
            dict(contrast_panel("a", "L4"), label="plug-in Gaussian-marginal proxy; negative values are plotted as they are",
                 numerical_annotation=l4_note),
            contrast_panel("b", "L5"),
            dict(trajectory("c", ("PR",)), isotropic_reference=C.pr_isotropic_reference(run.reg["bank_rows"], C.D_FREE),
                 isotropic_reference_source="diag_lat_core.pr_isotropic_reference(K, d): analytic, not a draw"),
            dict(trajectory("d", SENSITIVITIES), contrasts={k: {c: reported[k]["contrasts"][c] for c in PO.READ_CONTRASTS}
                                                            for k in SENSITIVITIES})]),
        "fig5": doc("fig5", "Summary of which changes accompany the gap", [
            {"panel": "a", "kind": "forest layout, one row per measure, each in its own units on its own axis", "provenance": MEASURE,
             "source": dict(pop_src, path="classified.measures"), "row_order": list(ST.MEASURES), "contrasts": list(GAP_ROW),
             "units": {mid: _unit(cls[mid]["key"]) for mid in ST.MEASURES},
             "data": {mid: {c: cls[mid]["effects"][c] for c in GAP_ROW} for mid in ST.MEASURES}},
            {"panel": "b", "kind": "interactions, shown together (235 section 5.1)", "provenance": MEASURE,
             "source": dict(pop_src, path="classified.measures"), "row_order": list(ST.MEASURES), "contrasts": list(INTERACTIONS),
             "data": {mid: {c: cls[mid]["effects"][c] for c in INTERACTIONS} for mid in ST.MEASURES}},
            {"panel": "c", "kind": "per-volume change of NLL against change of R0 PSNR, per read contrast and seed (the inputs of L6)",
             "provenance": MEASURE, "source": {"file": "records/ (records_sha256.txt)", "function": "diag_lat_stats.seed_differences"},
             "units": {"d_NLL": _unit("NLL"), "d_R0_PSNR": _unit("R0_PSNR")}, "volume_ids": list(b.volume_ids), "data": per_volume},
            {"panel": "d", "kind": "L6 Spearman rho with intervals (descriptive; no class)", "provenance": MEASURE,
             "source": dict(pop_src, path="l6"), "data": result["l6"]}],
            rule_11="no common scale and no ranking: raw effects of the six measures are never put on one numerical axis, never "
                    "rescaled and never ordered by size; rows follow the fixed order L1, L2, L3a, L3b, L4, L5"),
        "s1": doc("s1", "Every classified and reported measure on all 21 contrasts", [
            {"panel": "classified", "provenance": MEASURE, "source": dict(pop_src, path="classified.measures.*.effects"),
             "data": {mid: cls[mid]["effects"] for mid in ST.MEASURES}},
            {"panel": "reported", "provenance": MEASURE, "source": dict(pop_src, path="reported"), "data": reported}]),
        "s2": doc("s2", "Monte Carlo stability: direction labels at K = 64, K = 128 and both halves", [
            {"panel": mid, "provenance": MEASURE, "source": dict(pop_src, path="classified.measures.%s.stability_per_contrast" % mid),
             "summary": cls[mid]["stability_summary"], "data": cls[mid]["stability_per_contrast"]} for mid in ("L2", "L3a", "L3b", "L4")]),
        "s3": doc("s3", "Physical complements beside their standardised versions (reported, not classified)", [
            {"panel": mid, "provenance": MEASURE, "source": dict(pop_src, path="classified.measures.%s" % mid),
             "coordinate_tag": cls[mid]["coordinate_tag"],
             "data": {"standardised": {c: cls[mid]["effects"][c] for c in ST.READ_GAP}, "physical": cls[mid]["physical_effects"]}}
            for mid in ST.COORDINATE_TAGGED] + [
            {"panel": "%s and %s" % pair, "provenance": MEASURE, "source": dict(pop_src, path="reported; state_means"),
             "data": {k: {"contrasts": reported[k]["contrasts"], "state_means": means[k]} for k in pair}}
            for pair in (("PR", "phys_PR"), ("swap_shift_r0", "phys_swap_shift_r0"))]),
        "s4": doc("s4", "Tail covariate over the slices and its L6 correlations", [
            {"panel": "tail", "provenance": MEASURE, "source": dict(pop_src, path="tail"), "data": result["tail"]},
            {"panel": "L6 tail correlations", "provenance": MEASURE, "source": dict(pop_src, path="l6.*.*.tail"),
             "data": {c: {seed: v["tail"] for seed, v in per_seed.items()} for c, per_seed in result["l6"].items()}}]),
        "s5": doc("s5", "L1: skewness, excess kurtosis and E_cov beside the N(0, I) reference", [
            {"panel": k, "provenance": MEASURE, "source": dict(pop_src, path="reported.%s; state_means.%s" % (k, k)),
             "data": {"contrasts": reported[k]["contrasts"], "state_means": means[k]}} for k in ("skewness", "excess_kurtosis")] + [
            {"panel": "E_cov", "provenance": MEASURE, "source": dict(pop_src, path="e_cov"), "data": result["e_cov"]}]),
        "s6": doc("s6", "Fixtures", [
            {"panel": "step G", "provenance": MEASURE, "source": {"file": "g_report.json", "path": "fixtures"}, "data": run.fx},
            {"panel": "bank round trip on all slice-states (rule E-G1)", "provenance": MEASURE, "source": {"file": "g_eg1.json"},
             "data": {k: numerical[k] for k in ("threshold", "n_slice_states", "n_flagged", "max", "l4_summary")}},
            {"panel": "convention fixtures 1-11 (F1 run V15)", "provenance": EXISTING,
             "source": {"file": src["f1_report"], "sha256": src["f1_report_sha256"]},
             "data": {"fixtures": ev["f1_report"]["fixtures"], "cited": man["f1_evidence"]}},
            {"panel": "synthetic fixtures, incl. finite-K behaviour of TC_G (self-test; synthetic, not D-LAT evidence)", "provenance": EXISTING,
             "source": {"file": src["selftest_report"], "sha256": src["selftest_report_sha256"]},
             "data": {"banner": ev["selftest_report"].get("banner"), "fixtures": ev["selftest_report"]["fixtures"]}},
            {"panel": "numerical Jacobian fixture, S5 and integration fixtures (P3 test; synthetic, not D-LAT evidence)", "provenance": EXISTING,
             "source": {"file": src["p3_report"], "sha256": src["p3_report_sha256"]},
             "data": {"banner": ev["p3_report"].get("banner"), "fixtures": ev["p3_report"]["fixtures"]}}]),
        "s7": doc("s7", "Events: every 'not computed' case and typed non-fatal event", [
            {"panel": "events", "provenance": MEASURE, "source": dict(pop_src, path="events"), "data": result["events"]}]),
    }
    out = {"figdata/%s.json" % k: F.dumps(v) for k, v in files.items()}
    if tuple(out) != FIG_NAMES:
        _stop(A.RECORD_SCHEMA, "figure data files %s differ from the registered list" % sorted(out))
    return out


def check_outputs(rep, numerical):
    if rep is not None and set(rep) != REPORT_KEYS:
        _stop(A.RECORD_SCHEMA, "report keys: missing %s, unexpected %s" % (sorted(REPORT_KEYS - set(rep)), sorted(set(rep) - REPORT_KEYS)))
    if numerical is not None and set(numerical) != EG1_KEYS:
        _stop(A.RECORD_SCHEMA, "E-G1 annotation keys: missing %s, unexpected %s"
              % (sorted(EG1_KEYS - set(numerical)), sorted(set(numerical) - EG1_KEYS)))
    return True


def run_g(binding, man, out_dir, man_sha256="(not from a file)", banner="D-LAT STEP G -- RESULT FILES OF THE SINGLE RUN"):
    """Stages G1-G10. binding: a binding object or a function that builds one (production).
    Returns (report, exit code). out_dir must not exist; it is created here."""
    check_manifest(man)
    try:
        os.makedirs(out_dir, exist_ok=False)
    except OSError as exc:
        _stop(GD.GUARD_STATE, "cannot create the new output folder %s: %s" % (out_dir, exc))
    mem = F.MemoryLog()
    mem.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(mem)
    run = GRun(man, out_dir, man_sha256)
    lim = man["limits"]
    try:
        guard = GD.WriteGuard(out_dir, lim["disk_ceiling_bytes"], lim["reserve_bytes"])       # disclosed item (6)
        results, done = {}, False
        try:
            with GD.memory_guarded(lim["ram_bytes"]):
                run.at("G2")
                run.b = binding() if callable(binding) else binding
                run.at("G1")
                run.g1()
                run.d2_d3()
                run.at("G3")
                run.sources = bind_sources(man, run.b.state_order, [tuple(x) for x in run.b.identities], list(run.b.volume_ids))
                run.rep["binding"]["sources"] = dict(run.sources["r1_info"], evidence={e: man["sources"][e + "_sha256"] for e in EVIDENCE})
                run.d4_tensors()
                run.g5()
                run.g6()
                run.g7_g8(guard)
                run.g9()
            check_outputs(None, run.numerical)
            pop_bytes = F.dumps(run.result)
            out_files = {"g_population.json": pop_bytes, "g_eg1.json": F.dumps(run.numerical),
                         "g_main_table.json": F.dumps(main_table_file(man, run.result, run.numerical))}
            run.at("G10")
            out_files.update(figure_files(man, run.result, run.numerical, run, F.sha_bytes(pop_bytes)))
            for name, data in out_files.items():
                results[name] = run.put(guard, name, data)       # the sha256 the write guard returns
            run.at("G8")
            run.rep["resources"]["elapsed_final_s"] = run.wall()
            peak = GD.check_memory(lim["ram_bytes"])
            done = True
        except DiagLatError as exc:
            where = "G8" if exc.code == GD.RESOURCE_LIMIT else run.current
            run.rep["stop"] = {"code": exc.code, "message": exc.message, "criterion": where}
            run.crit[where].update(status="FAIL", stop=exc.code, observed=exc.message)
            LOG.error("G stopped at criterion %s: %s", where, exc)
        rep = run.rep
        rep.update(schema=REPORT_SCHEMA, banner=banner, review_status=man["review_status"], fixtures=run.fx)
        if rep["resources"] is None:
            rep["resources"] = {}
        rep["resources"].update(disk_written_records_and_results_bytes=guard.written, D_bytes=lim["D_bytes"],
                                reserve_bytes=lim["reserve_bytes"], ram_limit_bytes=lim["ram_bytes"],
                                peak_rss_bytes=GD.peak_rss_bytes())
        if done:
            num = run.numerical
            rep["numerical"] = {q: num[q] for q in ("rule", "threshold", "n_slice_states", "n_flagged", "max", "affected_states",
                                                    "l4_summary", "l4_outcome_note")}
            rep["numerical"]["target_round_trip"] = "not repeated in G (F2 decision Q2); F1 evidence: see manifest.f1_evidence"
            run.passed("G8", {"elapsed_s": rep["resources"]["elapsed_final_s"], "W_k_max_s": rep["resources"]["W_k_max_s"],
                              "peak_rss_bytes": peak, "disk_written_bytes": guard.written},
                       {"L_s": lim["L_s"], "ram_bytes": lim["ram_bytes"], "D_bytes": lim["D_bytes"]})
            run.passed("G10", "population result, Table 1, E-G1 annotation, %d figure and supplementary data files, report, log, verdict "
                              "and sha256 lists written through the write guard; Figure 1 (c) and S6 from the bound sources"
                       % len(FIG_NAMES), "output schema v0.3")
        rep["outputs"] = {"schema": "v0.3", "result_files": dict(sorted(results.items())),
                          "evidence_files": ["g.log", "g_report.json", "g_verdict.txt"],
                          "record_files_manifest": "records_sha256.txt", "n_record_files": len(getattr(run, "record_files", {})),
                          "figure_data": "figdata/: one file per figure and supplementary output of the figure plan v0.2; Table 1 is "
                                         "g_main_table.json; Figure 1 (c) and S6 hold existing results copied from the bound sources"}
        rep["criteria"] = run.crit
        ok = done and rep["stop"] is None and all(run.crit[c]["status"] == "PASS" for c in CRITERIA)
        try:
            check_outputs(rep, None)
        except DiagLatError as exc:
            run.crit["G10"].update(status="FAIL", stop=exc.code, observed=exc.message)
            rep["stop"] = {"code": exc.code, "message": exc.message, "criterion": "G10"}
            ok = False
            LOG.error("the G outputs failed their schema: %s", exc)
        rep["verdict"] = "G: COMPLETE" if ok else "G: FAIL"
        verdict = "".join("%s: %s\n" % (c, run.crit[c]["status"]) for c in CRITERIA) + rep["verdict"] + "\n"
        rec_files = getattr(run, "record_files", {})
        files = {"g_report.json": F.dumps(rep), "g_verdict.txt": verdict.encode(), "g.log": ("\n".join(mem.lines) + "\n").encode(),
                 "records_sha256.txt": "".join("%s  %s\n" % (d, n) for n, d in sorted(rec_files.items())).encode()}
        for name, data in files.items():
            results[name] = guard.write(name, data)
        guard.write("outputs_sha256.txt", "".join("%s  %s\n" % (d, n) for n, d in sorted(results.items())).encode())
        return rep, (0 if ok else 1)
    finally:
        root.removeHandler(mem)


def check_sources(repo, manifest_path, man):
    """Read-only preflight: the registered F2 manifest bound as in the run (no PyTorch), then every source binding."""
    reg = man["registered"]
    bound = BD.bind_manifest(manifest_path, f2_sha256=man["manifest_f2_sha256"], n_states=reg["n_states"], n_volumes=reg["n_volumes"], pin=man["pin"])
    files = BD.verify_files(bound, BD.results_root(repo))
    info = bind_sources(man, list(bound["state_order"]), [tuple(x) for x in bound["identities"]], list(files["volume_ids"]))["r1_info"]
    if "torch" in sys.modules:
        _stop(A.ADAPTER, "PyTorch was imported in the source check")
    return info


def main():
    ap = argparse.ArgumentParser(description="D-LAT step G driver", allow_abbrev=False)
    ap.add_argument("--g-manifest", required=True)
    ap.add_argument("--g-manifest-sha256", required=True)
    ap.add_argument("--repo", required=True, help="the pinned V05LT checkout (read-only)")
    ap.add_argument("--manifest", required=True, help="the registered D-UNC manifest_f2.json")
    ap.add_argument("--out", help="a new output folder; it must not exist (required unless --check-sources)")
    ap.add_argument("--check-sources", action="store_true", help="read-only: bind the F2 manifest and verify every source binding; "
                                                                 "no PyTorch, no model, no inference; writes nothing")
    ap.add_argument("--dunc-dir", required=True, help="folder of the registered diag_unc_adapter.py (verified by sha256)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        man, man_sha = load_manifest(args.g_manifest, args.g_manifest_sha256)
        if man["selftest"]:
            _stop(G_MANIFEST, "a self-test manifest is not accepted by the production entry point")
        if args.check_sources:
            info = check_sources(args.repo, args.manifest, man)
            print("SOURCES: PASS (%d D-UNC files verified, R1 PSNR of %d slice-states; three evidence reports at their sha256)"
                  % (info["files_verified"], info["n_records"]))
            return 0
        if not args.out:
            _stop(G_MANIFEST, "--out is required for the run")
        if not os.path.isdir(args.dunc_dir):
            _stop(A.ORIGIN, "--dunc-dir must be an existing folder")
        sys.path.append(os.path.realpath(args.dunc_dir))             # after the D-LAT code folder: nothing is shadowed
        sys.path.append(os.path.realpath(args.repo))                 # the pinned checkout (as the F1 driver does)
        LOG.info("G driver %s on %s, python %s", __version__, platform.node(), platform.python_version())
        rep, code = run_g(lambda: F.ProductionBinding(args.repo, args.manifest, card_view(man)), man, args.out, man_sha)
        print(rep["verdict"])
        return code
    except DiagLatError as exc:
        LOG.error("%s: %s", "the source check failed" if args.check_sources else "G did not start or could not write its evidence", exc)
        print("%s: FAIL (%s)" % ("SOURCES" if args.check_sources else "G", exc.code))
        return 1
    except Exception as exc:                     # logged and re-raised: never swallowed
        LOG.error("unexpected error: %s: %s", type(exc).__name__, exc)
        raise


if __name__ == "__main__":
    sys.exit(main())

# SEQREF-MRI-V05LTRPT v0.3 -- scripts.v05lt_report
# LIFETIME: KEEP
# =============================================================================
# Purpose: the V05LT validation report and plots 1-7 (I1.8(b) stage b2,
#   plan SEQREF-MRI-V05LTI18B v0.2: B3 report, B4 plots; 194 §9.2-§10;
#   195 §9.2, §9.4, PA1, PA3, PA4). Input: ONE verified analysis document
#   (v05lt_analysis, recorded sha256 + sidecar). Output: a write-once
#   directory, staged and published with renameat2(RENAME_NOREPLACE):
#     report.md            human-readable validation report
#     report.json          the same content as data + bindings (analysis
#                          and configuration sha256, code identity, plot
#                          files and their sha256)
#     plot<k>_<name>.png   plots 1-7 (matplotlib Agg; PNG metadata removed)
#     plot<k>_<name>.json  the exact values drawn, with their source keys
#     SHA256SUMS
#   Nothing is re-estimated: every statistic, label, count and descriptive
#   value is read from the analysis document. Display-only arithmetic:
#   plot 4 draws descriptive per-seed references at N_ref + 0.01 |N_ref|
#   (registered r definition; SCHEMA b2 §3).
#   The registered results tree is refused unless production=True (gated
#   reporting stage only); main() refuses production.
# CONVENTION: logger.error + V05LTError(code); no substituted values; a
#   malformed analysis document -> REPORT_ANALYSIS_INVALID (never a raw
#   exception); a missing plotting backend -> REPORT_PLOT_BACKEND (no
#   fallback renderer).
# Changelog (v0.3, 2026-10-24; I1.8 b3, SEQREF-MRI-V05LTI18B3 v0.1):
#   * Holdout section when the analysis carries the registered addendum:
#     prominent "not independent confirmation" statement, AM4-B NMSE_u
#     rule, explicit evaluation scope, manifest / record / selection
#     identities and the 21 holdout comparisons with per-seed results.
# Changelog (v0.2, 2026-10-24; b2 revision-1 review fixes):
#   * plot 4 references from r = (N - N_ref)/|N_ref| (correct for negative
#     NLL), labelled descriptive; CIs drawn independently of the estimate.
#   * verify_report: exact required file set, valid hashes, complete plot
#     entries before bindings. Report: per-seed CIs; complete identity
#     lists in appendices (no truncation).
# Changelog (NEW in v0.1, 2026-10-24; I1.8(b) b2, SEQREF-MRI-V05LTI18B v0.2):
#   * Introduced: verified analysis loading, report (Markdown + JSON),
#     plots 1-7 with data files, write-once publication, report verifier.
# =============================================================================
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from pathlib import Path

from seqref_mri.scripts import v05lt_analysis as A
from seqref_mri.scripts import v05lt_common as C
from seqref_mri.scripts import v05lt_stats as S

logger = logging.getLogger("seqref_mri.v05lt_report")

__version__ = "0.3"
__abbr__ = "SEQREF-MRI-V05LTRPT"

REPORT_SCHEMA = "seqref-v05lt-report/1"
PLOT_SCHEMA = "seqref-v05lt-plot-data/1"
REPORT_MD = "report.md"
REPORT_JSON = "report.json"
SUMS = "SHA256SUMS"
DPI = 100
AM4B_STATEMENT = (
    "Validation NMSE_u excludes slices with E / S_ref^2 <= 1e-10 (194 "
    "AM2-C, AM4-B); the holdout protocol keeps all slices (V02E rule; "
    "exact zero energy is a validity error). The two NMSE_u protocols "
    "therefore differ (195 PA1).")
PRACTICAL_TEXT = (
    "+0.5 dB is the provisional, descriptive practical reference (194 "
    "§9.2): about 10.9 % lower MSE; not a pass/fail rule. Density "
    "safeguard: preserved if the CI upper bound of r < +1 %, worse if the "
    "lower bound > +1 %, otherwise unclear.")


class _Bad(Exception):
    """Internal: malformed analysis document (converted to a typed V05LT
    error at the public boundary)."""


# ---------------------------------------------------------------------------
# 1. Verified input
# ---------------------------------------------------------------------------
def load_analysis(path: Path, sha: str) -> dict:
    """Analysis document against its recorded sha256 and sidecar; schema
    and configuration binding checked before use."""
    doc = A._load_once(Path(path), sha, "REPORT_ANALYSIS_HASH")
    C.require(isinstance(doc, dict) and doc.get("schema") == A.RESULT_SCHEMA,
              "REPORT_ANALYSIS_INVALID", f"{path}: schema")
    an = doc.get("analysis")
    C.require(isinstance(an, dict) and an.get("schema") == S.RESULT_SCHEMA
              and an.get("config_sha256") == doc.get("config_sha256"),
              "REPORT_ANALYSIS_INVALID", f"{path}: analysis block / "
              "configuration binding")
    for k in ("descriptive", "inputs", "code", "statements"):
        C.require(k in doc, "REPORT_ANALYSIS_INVALID", f"{path}: {k}")
    return doc


def _cmp_index(doc: dict) -> dict:
    out = {}
    for c in doc["analysis"]["comparisons"]:
        if not isinstance(c, dict) or "id" not in c or c["id"] in out:
            raise _Bad(f"comparison entry {str(c)[:80]}")
        out[c["id"]] = c
    return out


# ---------------------------------------------------------------------------
# 2. Report content (read from the analysis document only)
# ---------------------------------------------------------------------------
def _fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.6g}"
    return str(x)


def _ci(c: dict) -> str:
    ci = c.get("ci")
    return "—" if not ci else f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"


def _per_seed(c: dict) -> str:
    ps = c.get("per_seed") or []
    return "; ".join(f"{p['seed']}: {_fmt(p.get('estimate'))} {_ci(p)}"
                     for p in ps) or "—"


def _row(c: dict) -> dict:
    return {"id": c["id"], "family": c["family"], "status": c["status"],
            "metric": c["metric"], "stat": c["stat"], "scope": c["scope"],
            "milestone": c.get("milestone"), "estimate": c.get("estimate"),
            "ci": c.get("ci"), "per_seed": c.get("per_seed", []),
            "support": c.get("support"), "practical": c.get("practical"),
            "density": c.get("density"), "n_volumes": c.get("n_volumes"),
            "reason": c.get("reason"), "identities": c.get("identities"),
            "seeds_absent": c.get("seeds_absent")}


def build_content(doc: dict) -> dict:
    """Structured report content; every value is copied from the analysis
    document (selection and formatting only)."""
    an, de = doc["analysis"], doc["descriptive"]
    cmp_ = _cmp_index(doc)
    outcomes = []
    for o in an["outcomes"]:
        prim = cmp_.get(f"{o['id']}|psnr")
        if prim is None:
            raise _Bad(f"outcome {o['id']}: primary PSNR comparison absent")
        outcomes.append({**o, "primary": _row(prim)})
    rows = [_row(c) for c in an["comparisons"]]
    not_ok = [r for r in rows if r["status"] != "ok"]
    lc = de["learning_curves"]
    runs = {k: {"status": v["status"], "stop_epoch": v["stop_epoch"],
                "selected_epoch": v["selected_epoch"],
                "best_nll_epoch": v["best_nll_epoch"],
                "milestones": v["milestones"], "resets": v["resets"]}
            for k, v in sorted(lc["runs"].items())}
    for k, v in doc["inputs"]["run_info"].items():
        if k not in runs or runs[k]["status"] != v["status"]:
            raise _Bad(f"run_info {k} disagrees with learning curves")
    hold = doc.get("holdout")
    hrows = [r for r in rows if r["family"].startswith("holdout_")]
    if (hold is None) != (not hrows):
        raise _Bad("holdout block and holdout comparisons disagree")
    return {
        "holdout": None if hold is None else {**hold, "comparisons": hrows},
        "statements": doc["statements"], "notes": an["notes"],
        "practical_text": PRACTICAL_TEXT,
        "outcomes": outcomes, "comparisons": rows,
        "not_ok": not_ok,
        "caps": {"statement": an["cap_statement"], "per_source": an["caps"]},
        "nmse_exclusion": de["nmse_exclusion"],
        "am4b_statement": AM4B_STATEMENT,
        "dc": {"identity": an["dc_identity"],
               "identity_per_source": an["dc_identity_per_source"],
               "measures": de["dc_measures"]},
        "bank_sensitivity": de["bank_sensitivity"],
        "prefix_stability": de["prefix_stability"],
        "per_case": de["per_case"], "runs": runs,
        "references": lc["references"],
        "stream": an["stream"], "d6_rule": doc["inputs"]["d6"]["rule"]}


def _md_table(head, rows) -> list:
    out = ["| " + " | ".join(head) + " |",
           "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(x).replace("|", "\\|") for x in r) + " |"
            for r in rows]
    return out + [""]


def render_markdown(meta: dict, ct: dict, plots: list) -> str:
    L = [f"# V05LT validation report ({meta['abbr']} v{meta['version']})", "",
         f"- Analysis document sha256: `{meta['analysis_sha256']}`",
         f"- Analysis configuration sha256: `{meta['config_sha256']}`",
         f"- Report code sha256: `{meta['code']['v05lt_report.py']}`", ""]
    ho = ct["holdout"]
    if ho is not None:
        L += ["## HOLDOUT — NOT AN INDEPENDENT CONFIRMATION", "",
              f"**{ho['statement']}**", "", f"- {ho['nmse_rule']}",
              f"- {ho['evaluation_scope']}",
              f"- Holdout manifest content sha256: "
              f"`{ho['manifest_content_sha256']}`; selection manifest "
              f"sha256: `{ho['selection_sha256']}`; masks sha256: "
              f"`{ho['masks_sha256']}`",
              f"- Holdout records: {len(ho['records'])} (Appendix D)", ""]
        L += _md_table(["Holdout comparison", "Status", "Estimate", "95% CI",
                        "Support", "Practical", "Density", "Per seed"],
                       [[r["id"], r["status"], _fmt(r["estimate"]), _ci(r),
                         _fmt(r["support"]), _fmt(r["practical"]),
                         _fmt(r["density"]), _per_seed(r)]
                        for r in ho["comparisons"]])
    L += ["## Statements", ""] + [f"- {s}" for s in ct["statements"]]
    L += [f"- {s}" for s in ct["notes"]] + [f"- {ct['practical_text']}", ""]
    L += ["## Outcome table (194 §9.3)", ""]
    L += _md_table(
        ["Outcome id", "#", "Outcome", "Support", "Practical", "Density",
         "ΔPSNR (dB)", "95% CI", "Per seed"],
        [[o["id"], _fmt(o["outcome"]),
          o["text"] if o["outcome"] is not None else o["reason"],
          _fmt(o.get("support")),
          _fmt(o.get("practical")), _fmt(o.get("density")),
          _fmt(o["primary"]["estimate"]), _ci(o["primary"]),
          _per_seed(o["primary"])] for o in ct["outcomes"]])
    L += ["## Runs", ""]
    L += _md_table(["Run", "Status", "Stop epoch", "Selected", "Best NLL",
                    "Milestones", "Resets (epoch, metric)"],
                   [[k, v["status"], v["stop_epoch"], v["selected_epoch"],
                     v["best_nll_epoch"], v["milestones"], v["resets"]]
                    for k, v in ct["runs"].items()])
    L += ["## All comparisons", ""]
    L += _md_table(["Id", "Status", "Estimate", "95% CI", "Support",
                    "Practical", "Density", "Per seed", "Volumes"],
                   [[r["id"], r["status"], _fmt(r["estimate"]), _ci(r),
                     _fmt(r["support"]), _fmt(r["practical"]),
                     _fmt(r["density"]), _per_seed(r), _fmt(r["n_volumes"])]
                    for r in ct["comparisons"]])
    L += ["## Comparisons not estimated as registered (PA3 / AM3-E4)", ""]
    if ct["not_ok"]:
        L += _md_table(["Id", "Status", "Reason", "Identities / absent "
                        "seeds"],
                       [[r["id"], r["status"], _fmt(r["reason"]),
                         f"Appendix A.{k + 1}"]
                        for k, r in enumerate(ct["not_ok"])])
    else:
        L += ["None.", ""]
    L += ["## PSNR caps and zero-error slices (AM4-A)", "",
          ct["caps"]["statement"], ""]
    nz = [(k, v) for k, v in sorted(ct["caps"]["per_source"].items())
          if v["psnr_capped"]["count"] or v["zero_error"]["count"]]
    L += _md_table(["Source", "Capped", "Zero error", "Identities"],
                   [[k, v["psnr_capped"]["count"], v["zero_error"]["count"],
                     f"Appendix B.{j + 1}"] for j, (k, v) in enumerate(nz)]) \
        if nz else [f"No capped or zero-error slices in "
                    f"{len(ct['caps']['per_source'])} populations.", ""]
    ne = ct["nmse_exclusion"]
    L += ["## NMSE_u exclusion report (AM2-C)", "", ne["definition"], "",
          f"- Eligibility manifest sha256: `{ne['eligibility_sha256']}`"]
    for sc in ("all", "mid"):
        L.append(f"- {sc}: {ne[sc]['n_excluded']} of {ne[sc]['n_slices']} "
                 f"slices excluded; null volumes {ne[sc]['null_volumes']}; "
                 f"NMSE population {ne[sc]['population_volumes']} volumes")
    L += [f"- Visual cases with an excluded midpoint slice: "
          f"{ne['cases_excluded'] or 'none'}",
          f"- Excluded slices ({ne['all']['n_excluded']}): Appendix C "
          "(complete list)",
          f"- {ct['am4b_statement']}", ""]
    dc = ct["dc"]
    L += ["## DC identity (194 §8)", ""]
    L += _md_table(["Readout", "Identical on all sources", "Non-identical "
                    "sources"],
                   [[ro, dc["identity"][ro]["identical_all"],
                     dc["identity"][ro]["non_identical_count"]]
                    for ro in sorted(dc["identity"])])
    L += [dc["measures"]["definition"], ""]
    L += _md_table(["Source", "Readout", "Expected", "Observed",
                    "With measures", "Status", "Max ratio to tolerance "
                    "(available slices)", "Missing"],
                   [[k, ro, v[ro]["n_expected"], v[ro]["n_observed"],
                     v[ro]["n_with_measures"], v[ro]["status"],
                     json.dumps(v[ro]["max_ratio_to_tolerance_available_"
                                     "slices"], sort_keys=True),
                     len(v[ro]["missing"])]
                    for k, v in sorted(dc["measures"]["sources"].items())
                    for ro in S.DC_READOUTS])
    bs, ps = ct["bank_sensitivity"], ct["prefix_stability"]
    L += ["## R1 bank sensitivity and sample-count stability "
          "(descriptive)", "", bs["definition"], ""]
    L += _md_table(["Run", "Bank", "PSNR", "SSIM", "NMSE_u",
                    "PSNR spread max / mean (dB)"],
                   [[k, b, _fmt(v["banks"][b]["psnr"]),
                     _fmt(v["banks"][b]["ssim"]),
                     _fmt(v["banks"][b]["nmse_u"]),
                     f"{_fmt(v['psnr_spread_db']['max'])} / "
                     f"{_fmt(v['psnr_spread_db']['mean'])}"]
                    for k, v in sorted(bs["runs"].items())
                    for b in sorted(v["banks"])])
    L += [ps["definition"], ""]
    ks = sorted(next(iter(ps["runs"].values()))["psnr_db"], key=int)
    L += _md_table(["Run"] + [f"k={k} PSNR (Δ vs 128)" for k in ks],
                   [[k] + [f"{_fmt(v['psnr_db'][x])} "
                           f"({_fmt(v['minus_k128_db'][x])})" for x in ks]
                    for k, v in sorted(ps["runs"].items())])
    pc = ct["per_case"]
    L += ["## Per-case effects (195 PA4; plot 7; descriptive)", "",
          pc["definition"], "", f"- Difficult cases: {pc['difficult_rule']}"
          f"; volumes {pc['difficult_volumes']}"]
    for g in ("all", "difficult"):
        v = pc[g]
        L.append(f"- {g}: n {v['n']}, improved {v['positive']}, zero "
                 f"{v['zero']}, worse {v['negative']}, fraction improved "
                 f"{_fmt(v['fraction_improved'])}")
    L += ["", "## References (AM3, descriptive)", ""]
    rf = ct["references"]
    L += _md_table(["Reference", "PSNR", "SSIM", "NMSE_u", "NLL/dim",
                    "Missing"],
                   [[f"epoch0 {s}"] + [
                       f"{_fmt(v[m]['value'])} ({v[m]['status']})"
                       for m in ("psnr", "ssim", "nmse_u", "nll_per_dim")] +
                    [len(v["missing"])]
                    for s, v in sorted(rf["epoch0"].items())] +
                   [["zero-filled"] + [
                       f"{_fmt(rf['zf'][m]['value'])} "
                       f"({rf['zf'][m]['status']})"
                       for m in ("psnr", "ssim", "nmse_u")] +
                    ["no NLL (AM3-D)", len(rf["zf"]["missing"])]])
    L += ["## Plots", ""]
    L += _md_table(["Plot", "PNG sha256", "Data sha256"],
                   [[p["name"], p["png_sha256"], p["data_sha256"]]
                    for p in plots])
    # Appendices: complete identity lists (never truncated); the same lists
    # are in report.json under "content".
    L += ["## Appendix A — comparisons not estimated as registered: "
          "complete identities / absent seeds", ""]
    for k, r in enumerate(ct["not_ok"]):
        L += [f"A.{k + 1} `{r['id']}`", "", "```json",
              json.dumps({"identities": r["identities"],
                          "seeds_absent": r["seeds_absent"]},
                         sort_keys=True), "```", ""]
    L += ["## Appendix B — capped / zero-error slices: complete "
          "identities", ""]
    for j, (k, v) in enumerate(nz):
        L += [f"B.{j + 1} `{k}`", "", "```json",
              json.dumps(v, sort_keys=True), "```", ""]
    L += ["## Appendix C — NMSE_u excluded slices: complete list", "",
          "```json", json.dumps({"all": ne["all"]["excluded"],
                                 "mid": ne["mid"]["excluded"]},
                                sort_keys=True), "```"]
    if ho is not None:
        L += ["", "## Appendix D — holdout record identities", "", "```json",
              json.dumps(ho["records"], sort_keys=True), "```"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# 3. Plots 1-7 (194 §10); values drawn = values in the data file
# ---------------------------------------------------------------------------
def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:
        C.fail("REPORT_PLOT_BACKEND", f"matplotlib unavailable: {exc}")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans",
                                "font.size": 8, "svg.hashsalt": "v05lt",
                                "figure.dpi": DPI})
    return matplotlib, plt


def _png(fig, plt) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, metadata={"Software": None})
    plt.close(fig)
    return buf.getvalue()


def _runs(doc):
    return sorted(doc["descriptive"]["learning_curves"]["runs"].items())


def plot1(doc, plt):
    """Learning curves: R0 PSNR and NLL/dim vs updates; epoch 3, plateau
    references, resets, selected and stop marked; epoch-0 value at
    update 0; ZF line on the PSNR row only (AM3)."""
    lc = doc["descriptive"]["learning_curves"]
    refs, runs = lc["references"], _runs(doc)
    fig, ax = plt.subplots(2, len(runs), figsize=(3 * len(runs), 5),
                           squeeze=False)
    data = {"runs": {}, "zf_psnr": refs["zf"]["psnr"]}
    for j, (key, r) in enumerate(runs):
        seed = key.split("|")[1]
        e0 = refs["epoch0"][seed]
        st = {e["epoch"]: e["step"] for e in r["epochs"]}
        d = {"step": [e["step"] for e in r["epochs"]],
             "psnr": [e["psnr"] for e in r["epochs"]],
             "nll_per_dim": [e["nll_per_dim"] for e in r["epochs"]],
             "ref_psnr": [e["plateau_ref"]["psnr"] for e in r["epochs"]],
             "ref_nll": [e["plateau_ref"]["nll"] for e in r["epochs"]],
             "epoch3_step": st.get(3), "selected_step":
             st[r["selected_epoch"]], "stop_step": st[r["stop_epoch"]],
             "resets": [[st[e], m] for e, m in r["resets"]],
             "epoch0": {"psnr": e0["psnr"], "nll_per_dim":
                        e0["nll_per_dim"]}}
        data["runs"][key] = d
        for i, (m, rk, lab) in enumerate((("psnr", "ref_psnr", "PSNR (dB)"),
                                          ("nll_per_dim", "ref_nll",
                                           "NLL/dim"))):
            a = ax[i][j]
            a.plot(d["step"], d[m], "-o", ms=2, lw=1, label="validation R0")
            a.step(d["step"], d[rk], where="post", ls="--", lw=0.8,
                   label="plateau ref")
            if d["epoch0"][m]["status"] == "ok":
                a.plot([0], [d["epoch0"][m]["value"]], "s", ms=4,
                       label="epoch 0")
            if m == "psnr" and refs["zf"]["psnr"]["status"] == "ok":
                a.axhline(refs["zf"]["psnr"]["value"], ls=":", lw=0.8,
                          label="ZF")
            if d["epoch3_step"] is not None:
                a.axvline(d["epoch3_step"], ls=":", lw=0.6, c="grey")
            a.axvline(d["selected_step"], ls="-", lw=0.8, c="green",
                      label="selected")
            a.axvline(d["stop_step"], ls="-", lw=0.8, c="red", label="stop")
            for s_, rm in d["resets"]:
                if rm == ("psnr" if m == "psnr" else "nll"):
                    a.axvline(s_, ls="-.", lw=0.6, c="orange")
            a.set_xlabel("updates")
            a.set_ylabel(lab)
            if i == 0:
                a.set_title(key)
    ax[0][0].legend(fontsize=6)
    fig.suptitle("Plot 1 - learning curves (validation, R0 all-slice)")
    fig.tight_layout()
    return fig, data


def _cmp_plot(a, items, ref_lines):
    for x, (lab, c) in enumerate(items):
        if c["status"] == "ok":
            lo, hi = c["ci"]              # percentile CI; may exclude the
            a.plot([x, x], [lo, hi], "-", lw=1.2, c="black")   # estimate
            a.plot([x - 0.08, x + 0.08], [lo, lo], "-", lw=1, c="black")
            a.plot([x - 0.08, x + 0.08], [hi, hi], "-", lw=1, c="black")
            a.plot([x], [c["estimate"]], "o", ms=4, c="black")
        else:
            a.annotate(c["status"], (x, 0), rotation=90, fontsize=6,
                       ha="center", va="bottom")
        for p in c.get("per_seed") or []:
            a.plot([x + 0.15], [p["estimate"]], ".", ms=4, c="tab:blue",
                   mfc="none" if c["status"] != "ok" else "tab:blue")
    for y, ls in ref_lines:
        a.axhline(y, ls=ls, lw=0.8, c="grey")
    a.set_xticks(range(len(items)))
    a.set_xticklabels([lab for lab, _ in items], rotation=45, ha="right",
                      fontsize=6)


def plot2(doc, plt):
    """Objective comparison lam2 - lam0: R0 ΔPSNR at the selected
    checkpoint and each matched milestone; primary CI, individual seeds
    (open = descriptive under PA3); 0 and +0.5 dB marked."""
    cm = _cmp_index(doc)
    ids = ["between|selected|r0|psnr"] + [f"between|epoch{m}|r0|psnr"
                                          for m in S.MILESTONES]
    items = [(i.split("|")[1], cm[i]) for i in ids]
    fig, a = plt.subplots(figsize=(7, 4))
    _cmp_plot(a, items, [(0.0, "-"), (S.PRACTICAL_DB, "--")])
    a.set_ylabel("ΔPSNR lam2 - lam0 (dB)")
    a.set_title("Plot 2 - objective comparison (R0, all slices)")
    fig.tight_layout()
    return fig, {"comparisons": {i: _row(cm[i]) for i in ids},
                 "reference_lines_db": [0.0, S.PRACTICAL_DB]}


def plot3(doc, plt):
    """Readout comparison: paired R1 - R0 ΔPSNR and ΔNMSE_u on the
    midpoint slices per run and checkpoint; R1 per-bank PSNR at the
    selected checkpoint shown separately."""
    cm = _cmp_index(doc)
    srcs = [c["id"].rsplit("|", 2)[0] for c in doc["analysis"]["comparisons"]
            if c["family"] == "readout" and c["id"].endswith("|r1-r0|psnr")]
    bs = doc["descriptive"]["bank_sensitivity"]["runs"]
    fig, ax = plt.subplots(3, 1, figsize=(9, 9))
    data = {"psnr": {}, "nmse_u": {}, "banks": {}}
    for a, m in zip(ax[:2], ("psnr", "nmse_u")):
        items = [(s.split("|", 1)[1], cm[f"{s}|r1-r0|{m}"]) for s in srcs]
        data[m] = {f"{s}|r1-r0|{m}": _row(cm[f"{s}|r1-r0|{m}"])
                   for s in srcs}
        _cmp_plot(a, items, [(0.0, "-")])
        a.set_ylabel(f"R1 - R0 Δ{m}")
    banks = sorted({b for v in bs.values() for b in v["banks"]})
    for k, b in enumerate(banks):
        xs = list(range(len(bs)))
        ys = [v["banks"][b]["psnr"] for _, v in sorted(bs.items())]
        ax[2].plot([x + 0.1 * k for x in xs], ys, "o", ms=3, label=b)
        data["banks"][b] = dict(zip(sorted(bs), ys))
    ax[2].set_xticks(range(len(bs)))
    ax[2].set_xticklabels(sorted(bs), fontsize=6)
    ax[2].set_ylabel("R1 PSNR at selected (dB)")
    ax[2].legend(fontsize=6)
    fig.suptitle("Plot 3 - readout comparison (midpoint slices)")
    fig.tight_layout()
    return fig, data


def plot4(doc, plt):
    """Reconstruction-density trade-off: R0 PSNR vs NLL/dim across
    training per run. Descriptive per-seed reference lines at the NLL/dim
    where r = (N - N_ref) / |N_ref| = +1 % (N_ref = that seed's lam0
    selected-checkpoint NLL/dim), i.e. N_ref + 0.01 |N_ref|. Display only;
    the campaign density safeguard is the analysis label from r and its
    CI."""
    runs = dict(_runs(doc))
    fig, a = plt.subplots(figsize=(7, 5))
    data = {"runs": {}, "bound": {}}
    for key, r in sorted(runs.items()):
        xs = [e["nll_per_dim"] for e in r["epochs"]]
        ys = [e["psnr"] for e in r["epochs"]]
        a.plot(xs, ys, "-o", ms=2, lw=0.8, label=key)
        data["runs"][key] = {"nll_per_dim": xs, "psnr": ys}
    for s in S.SEEDS:
        r0 = runs[f"lam0|{s}"]
        sel = {e["epoch"]: e for e in r0["epochs"]}[r0["selected_epoch"]]
        n_ref = sel["nll_per_dim"]
        b = n_ref + S.DENSITY_BOUND * abs(n_ref)
        data["bound"][str(s)] = {"lam0_selected_nll_per_dim":
                                 sel["nll_per_dim"], "bound": b}
        a.axvline(b, ls="--", lw=0.6, c="grey")
    a.set_xlabel("NLL/dim (validation, all slices)")
    a.set_ylabel("PSNR (dB, R0)")
    a.legend(fontsize=6)
    a.set_title("Plot 4 - reconstruction-density trade-off; dashed: "
                "descriptive per-seed r = +1 % references (not the "
                "safeguard decision)", fontsize=7)
    fig.tight_layout()
    return fig, data


def plot5(doc, plt):
    """NLL decomposition: prior and log-determinant contributions (and
    the production NLL/dim) per epoch; sign convention in the data."""
    nd = doc["descriptive"]["nll_decomposition"]
    fig, ax = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    data = {"sign_convention": nd["sign_convention"], "runs": {}}
    for key, eps in sorted(nd["runs"].items()):
        d = {f: [e[f] for e in eps] for f in
             ("epoch", "nll_prior_per_dim", "nll_logdet_per_dim",
              "nll_per_dim")}
        data["runs"][key] = d
        for a, f in zip(ax, ("nll_prior_per_dim", "nll_logdet_per_dim",
                             "nll_per_dim")):
            a.plot(d["epoch"], d[f], "-o", ms=2, lw=0.8, label=key)
            a.set_ylabel(f)
    ax[-1].set_xlabel("epoch")
    ax[0].legend(fontsize=6)
    fig.suptitle("Plot 5 - NLL decomposition")
    fig.tight_layout()
    return fig, data


def plot6(doc, plt):
    """DC measures (a), (b), (c_rms), (c_max) against their calibrated
    tolerances: maximum ratio over the AVAILABLE slices per source and
    readout (ratio 1 = tolerance); partial sources marked."""
    dm = doc["descriptive"]["dc_measures"]
    srcs = sorted(dm["sources"])
    fig, ax = plt.subplots(len(S.DC_READOUTS), 1, figsize=(9, 6),
                           squeeze=False)
    data = {"tolerances": dm["tolerances"], "sources": {}}
    from seqref_mri.scripts import v05lt_results as RS
    for a, ro in zip(ax[:, 0], S.DC_READOUTS):
        for k, m in enumerate(RS.DC_MEASURES):
            ys, xs = [], []
            for x, s in enumerate(srcs):
                mx = dm["sources"][s][ro][
                    "max_ratio_to_tolerance_available_slices"]
                if mx is not None:
                    xs.append(x + 0.1 * k)
                    ys.append(mx[m])
            a.plot(xs, ys, "o", ms=3, label=m)
        for x, s in enumerate(srcs):
            b = dm["sources"][s][ro]
            data["sources"].setdefault(s, {})[ro] = {
                k: b[k] for k in ("max_ratio_to_tolerance_available_slices",
                                  "n_expected", "n_observed",
                                  "n_with_measures", "status")}
            if b["status"] != "ok":
                a.annotate("partial", (x, 1.0), fontsize=6, rotation=90)
        a.axhline(1.0, ls="--", lw=0.8, c="red")
        a.set_yscale("symlog", linthresh=1e-3)
        a.set_ylim(bottom=0.0)                 # ratios are >= 0
        a.set_ylabel(f"{ro}: max ratio")
        a.set_xticks(range(len(srcs)))
        a.set_xticklabels(srcs, rotation=60, ha="right", fontsize=5)
        a.legend(fontsize=6)
    fig.suptitle("Plot 6 - DC measures vs tolerance (available slices)")
    fig.tight_layout()
    return fig, data


def plot7(doc, plt):
    """Per-case effects (PA4): Δ_v per validation volume, sorted, with the
    difficult set marked; counts and fractions from the analysis."""
    pc = doc["descriptive"]["per_case"]
    pts = sorted(pc["delta_db"], key=lambda d: (d["delta"], d["volume_id"]))
    fig, a = plt.subplots(figsize=(7, 4))
    for diff, mk in ((False, "o"), (True, "^")):
        xs = [i for i, d in enumerate(pts) if d["difficult"] is diff]
        a.plot(xs, [pts[i]["delta"] for i in xs], mk, ms=3,
               label="difficult (ZF-PSNR definition)" if diff else "other")
    a.axhline(0.0, ls="-", lw=0.6, c="grey")
    al, df = pc["all"], pc["difficult"]
    a.set_title(f"Plot 7 - per-case effects: improved {al['positive']}/"
                f"{al['n']} (fraction {al['fraction_improved']:.3g}); "
                f"difficult {df['positive']}/{df['n']}", fontsize=8)
    a.set_xlabel("validation volumes (sorted by Δ)")
    a.set_ylabel("Δ_v PSNR lam2 - lam0 (dB, R0, selected)")
    a.legend(fontsize=6)
    fig.tight_layout()
    return fig, {"points": [{k: d[k] for k in ("volume_id", "delta",
                                               "difficult")} for d in pts],
                 "all": al, "difficult": df}


PLOTS = (("plot1_learning_curves", plot1), ("plot2_objective", plot2),
         ("plot3_readouts", plot3), ("plot4_tradeoff", plot4),
         ("plot5_nll_decomposition", plot5), ("plot6_dc_measures", plot6),
         ("plot7_per_case", plot7))


def render_plots(doc: dict) -> tuple:
    """-> ({file name: bytes}, [plot entries]). Data files are canonical
    JSON of exactly the values drawn."""
    mpl, plt = _mpl()
    files, entries = {}, []
    for name, fn in PLOTS:
        fig, data = fn(doc, plt)
        png = _png(fig, plt)
        dj = C.canonical_json({"schema": PLOT_SCHEMA, "plot": name,
                               "definition": fn.__doc__.strip(),
                               "data": data})
        files[f"{name}.png"], files[f"{name}.json"] = png, dj
        entries.append({"name": name, "png": f"{name}.png",
                        "data": f"{name}.json",
                        "png_sha256": C.sha256_bytes(png),
                        "data_sha256": C.sha256_bytes(dj)})
    return files, entries, mpl.__version__


# ---------------------------------------------------------------------------
# 4. Write-once report directory and verifier
# ---------------------------------------------------------------------------
def build_report(analysis_path: Path, analysis_sha: str, out_dir: Path, *,
                 production: bool = False, hook=None) -> dict:
    """Verified analysis -> report.md, report.json, plots 1-7, SHA256SUMS
    in a staged directory published with RENAME_NOREPLACE. Returns
    {file: sha256}."""
    from seqref_mri.scripts import v05lt_eval as EV
    out_dir = EV.dest_path(out_dir)
    S._registered_dir_guard(out_dir, production)
    C.require(not EV.entry_exists(out_dir), "ARTIFACT_EXISTS",
              f"{out_dir}: an entry already exists (write-once)")
    stg = out_dir.with_name(f".{out_dir.name}.staging")
    C.require(not EV.entry_exists(stg), "STAGING_PRESENT",
              f"{stg}: left by an interrupted write; owner inspection "
              "required (not deleted)")
    doc = load_analysis(analysis_path, analysis_sha)
    here = Path(__file__).resolve().parent
    try:
        content = build_content(doc)
        files, plots, mpl_version = render_plots(doc)
        meta = {"schema": REPORT_SCHEMA, "abbr": __abbr__,
                "version": __version__, "analysis_sha256": analysis_sha,
                "config_sha256": doc["config_sha256"],
                "analysis_code": doc["code"],
                "code": {"v05lt_report.py": C.sha256_file(
                    here / "v05lt_report.py")},
                "environment": {"matplotlib": mpl_version}}
        md = render_markdown(meta, content, plots).encode()
    except (_Bad, KeyError, TypeError, IndexError, ValueError,
            AttributeError) as exc:
        C.fail("REPORT_ANALYSIS_INVALID", f"{type(exc).__name__}: "
               f"{str(exc)[:300]}")
    rj = C.canonical_json({**meta, "content": content, "plots": plots,
                           "report_md_sha256": C.sha256_bytes(md)})
    files[REPORT_MD], files[REPORT_JSON] = md, rj
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stg.mkdir()
    shas = {n: C.write_durable_new(stg / n, b)
            for n, b in sorted(files.items())}
    C.write_durable_new(stg / SUMS, "".join(
        f"{h}  {n}\n" for n, h in sorted(shas.items())).encode())
    if hook is not None:
        hook("report:staged")
    EV.publish_noreplace(stg, out_dir)
    for n, h in shas.items():
        C.require(C.sha256_file(out_dir / n) == h, "REPORT_INCOMPLETE", n)
    return shas


def verify_report(out_dir: Path, analysis_sha: str) -> dict:
    """SHA256SUMS = directory listing and file hashes; report.json bound to
    the analysis sha256, the Markdown and every plot file."""
    d = Path(out_dir)
    C.require((d / SUMS).is_file(), "REPORT_INCOMPLETE", f"{d}: {SUMS}")
    listed = {}
    for ln in (d / SUMS).read_text().splitlines():
        h, _, n = ln.partition("  ")
        listed[n] = h
    expected = {REPORT_MD, REPORT_JSON} | {f"{n}.{x}" for n, _ in PLOTS
                                           for x in ("png", "json")}
    names = {p.name for p in d.iterdir()} - {SUMS}
    C.require(set(listed) == expected and names == expected,
              "REPORT_INCOMPLETE", f"{d}: required files missing or extra "
              f"(listed {sorted(set(listed) ^ expected)}, present "
              f"{sorted(names ^ expected)})")
    for n, h in listed.items():
        C.require(isinstance(h, str) and len(h) == 64 and
                  all(ch in "0123456789abcdef" for ch in h),
                  "REPORT_INCOMPLETE", f"{SUMS}: invalid hash for {n}")
        C.require(C.sha256_file(d / n) == h, "REPORT_HASH_MISMATCH", n)
    try:
        rj = json.loads((d / REPORT_JSON).read_bytes())
    except ValueError as exc:
        C.fail("REPORT_INVALID", f"{REPORT_JSON}: {exc}")
    C.require(isinstance(rj, dict) and rj.get("schema") == REPORT_SCHEMA and
              rj.get("analysis_sha256") == analysis_sha,
              "REPORT_INVALID", f"{d}: schema / analysis binding")
    C.require(rj.get("report_md_sha256") == listed[REPORT_MD],
              "REPORT_INVALID", f"{d}: report.md binding")
    pl = rj.get("plots")
    C.require(isinstance(pl, list) and len(pl) == len(PLOTS),
              "REPORT_INVALID", f"{d}: plot entries")
    for p, (n, _) in zip(pl, PLOTS):
        C.require(isinstance(p, dict) and p.get("name") == n and
                  p.get("png") == f"{n}.png" and
                  p.get("data") == f"{n}.json" and
                  p.get("png_sha256") == listed[f"{n}.png"] and
                  p.get("data_sha256") == listed[f"{n}.json"],
                  "REPORT_INVALID", f"{d}: plot {n} entry / binding")
    return rj


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    argparse.ArgumentParser(description=f"{__abbr__} v{__version__}"
                            ).parse_args(argv)
    try:
        C.fail("REPORT_PRODUCTION_DISABLED", "the V05LT report runs only "
               "from the gated reporting stage after an approved analysis "
               "(195 §9.2, §9.4; 196 I1.8)")
    except C.V05LTError as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())

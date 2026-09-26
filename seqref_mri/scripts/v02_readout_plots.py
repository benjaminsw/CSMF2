# SEQREF-V02I v0.2 -- scripts.v02_readout_plots
# LIFETIME: DIAGNOSTIC
"""Registered optional facts-only figures for the V02 readout
diagnostic (V02RD report, LudWiki id 155).

This module is a PURE facts reader: it loads ONLY the authoritative
V02G facts artifact (hard-pinned by artifact sha256 AND semantic
sha256), renders five descriptive figures, and publishes them
atomically with a provenance manifest. It NEVER touches the model,
the checkpoint, the data root, or any manifest; it NEVER recomputes
bootstrap CIs, D1/D2 decisions, or the route -- every statistic
annotated on a figure is read verbatim from the authoritative
payload. The only arithmetic allowed is straightforward plotting
transforms (per-volume differences and sort orders).

Exit taxonomy: 0 = mode completed; 2 = any typed refusal or
unexpected failure (logged; nothing partially published).
"""

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless deterministic rendering; no display
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

__abbr__ = "SEQREF-V02I"
__version__ = "0.2"

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s "
                                      "%(name)s %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# --- locked pins (V02G authoritative run 2026-09-26, commit 6ffce4e) -
FACTS_ARTIFACT_SHA256 = ("0d9ed10c480641cdd1ffe2e260e51fbba9d694dcde79"
                         "afbb0322740cf257fc98")
FACTS_SEMANTIC_SHA256 = ("92b2ee1ff449d7a4e561945d6de98667f343bd52674c"
                         "ab017a98b1ed6a06df59")
FACTS_SCHEMA = "seqref-v02rd-facts/1"
N_HOLDOUT = 199
MANIFEST_SCHEMA = "seqref-v02i-plots-manifest/1"

EXIT_PASS = 0
EXIT_ERROR = 2


class V02IError(Exception):
    """Typed refusal: carries a stable error code and a reason."""

    def __init__(self, error_code: str, reason: str):
        super().__init__(f"{error_code}: {reason}")
        self.error_code = error_code
        self.reason = reason


def _fail(error_code: str, reason: str) -> None:
    """Every failure path: logger.error + typed raise. Never coerced,
    never skipped, never silently passed."""
    logger.error("[%s] %s: %s", __abbr__, error_code, reason)
    raise V02IError(error_code, reason)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(cond: bool, reason: str) -> None:
    if not cond:
        _fail("FACTS_CONTENT_INVALID", reason)


def _identity_key(entry: dict):
    return (str(entry["file"]), int(entry["slice_index"]))


def _load_verified_facts(path: str, expected_sha: str,
                         expected_semantic: str) -> dict:
    """Load the facts artifact ONLY after hard pin verification:
    artifact sha256 -> schema -> semantic sha256 -> structural
    validation (required keys, n=199, identity alignment across
    endpoints and against population.identities, finiteness of every
    plotted quantity). Any drift is a typed ERROR; nothing is read
    for plotting before every check passes."""
    p = Path(path)
    if not p.is_file():
        _fail("FACTS_MISSING", f"facts artifact not found: {path}")
    sha = _file_sha256(p)
    if sha != expected_sha:
        _fail("FACTS_ARTIFACT_DRIFT",
              f"facts artifact sha256 {sha[:12]}... != pinned "
              f"{expected_sha[:12]}...")
    try:
        facts = json.loads(p.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        _fail("FACTS_CONTENT_INVALID", f"facts artifact unparsable: {exc}")
    if str(facts.get("schema")) != FACTS_SCHEMA:
        _fail("FACTS_SCHEMA_MISMATCH",
              f"schema {facts.get('schema')!r} != pinned {FACTS_SCHEMA!r}")
    if str(facts.get("semantic_sha256")) != expected_semantic:
        _fail("FACTS_SEMANTIC_DRIFT",
              f"semantic sha256 {str(facts.get('semantic_sha256'))[:12]}"
              f"... != pinned {expected_semantic[:12]}...")
    # structural validation (everything the figures consume)
    _require(int(facts.get("population", {}).get("n_volumes", -1))
             == N_HOLDOUT, "population.n_volumes != 199")
    pop = facts["population"].get("identities", [])
    _require(len(pop) == N_HOLDOUT,
             "population.identities is not 199 entries")
    for ep in ("step0", "final"):
        block = facts.get("metrics", {}).get(ep, {})
        pv = block.get("per_volume", [])
        _require(len(pv) == N_HOLDOUT,
                 f"metrics.{ep}.per_volume is not 199 entries")
        for j, rec in enumerate(pv):
            _require(_identity_key(rec["identity"])
                     == _identity_key(pop[j]),
                     f"metrics.{ep}.per_volume[{j}] identity misaligned "
                     f"with population.identities[{j}]")
            for readout in ("r0", "r1"):
                for key in ("psnr", "nmse_u"):
                    v = rec[readout][key]
                    _require(isinstance(v, (int, float))
                             and math.isfinite(v),
                             f"metrics.{ep}.per_volume[{j}].{readout}."
                             f"{key} is not finite")
        agg = block.get("aggregates", {})
        for readout in ("r0", "r1"):
            for key in ("psnr", "nmse_u"):
                m = agg.get(readout, {}).get(key, {}).get("mean")
                _require(isinstance(m, (int, float)) and math.isfinite(m),
                         f"metrics.{ep}.aggregates.{readout}.{key}.mean "
                         f"missing or non-finite")
    pc = facts.get("primary_comparison", {})
    for key in ("observed", "ci95_lower"):
        v = pc.get("d1", {}).get(key)
        _require(isinstance(v, (int, float)) and math.isfinite(v),
                 f"primary_comparison.d1.{key} missing or non-finite")
    for key in ("pass", "practical_effect_ok", "uncertainty_ok"):
        _require(isinstance(pc.get("d1", {}).get(key), bool),
                 f"primary_comparison.d1.{key} missing or not boolean")
    for key in ("observed", "ci95_upper"):
        v = pc.get("d2", {}).get(key)
        _require(isinstance(v, (int, float)) and math.isfinite(v),
                 f"primary_comparison.d2.{key} missing or non-finite")
    for key in ("pass", "practical_effect_ok", "uncertainty_ok"):
        _require(isinstance(pc.get("d2", {}).get(key), bool),
                 f"primary_comparison.d2.{key} missing or not boolean")
    ci1 = pc.get("paired_psnr", {}).get("bootstrap", {}).get(
        "d1_mean_paired_difference", {}).get("ci95")
    ci2 = pc.get("nmse_ratio", {}).get("bootstrap", {}).get(
        "d2_ratio_of_resampled_means", {}).get("ci95")
    _require(isinstance(ci1, list) and len(ci1) == 2
             and all(math.isfinite(x) for x in ci1),
             "d1 bootstrap ci95 missing or non-finite")
    _require(isinstance(ci2, list) and len(ci2) == 2
             and all(math.isfinite(x) for x in ci2),
             "d2 bootstrap ci95 missing or non-finite")
    _require(isinstance(facts.get("routing", {}).get("label"), str),
             "routing.label missing")
    logger.info("[%s] facts verified: artifact %s... semantic %s...",
                __abbr__, sha[:12], expected_semantic[:12])
    return facts


# ---------------------------------------------------------------------------
# Figures. Every annotated statistic is read verbatim from the verified
# payload; the only arithmetic is per-volume plotting transforms.
# ---------------------------------------------------------------------------

FIGSIZE = (7.0, 4.5)
DPI = 150


def _final_pairs(facts, key):
    pv = facts["metrics"]["final"]["per_volume"]
    r0 = np.array([rec["r0"][key] for rec in pv], dtype=np.float64)
    r1 = np.array([rec["r1"][key] for rec in pv], dtype=np.float64)
    return r0, r1


def _fig_f1(facts, workdir: Path):
    """Headline: sorted per-identity paired dPSNR (final R1 - R0)."""
    r0, r1 = _final_pairs(facts, "psnr")
    d = np.sort(r1 - r0)  # plotting transform: difference + sort
    pc = facts["primary_comparison"]
    d1 = pc["d1"]
    obs = d1["observed"]
    ci = pc["paired_psnr"]["bootstrap"]["d1_mean_paired_difference"][
        "ci95"]
    verdict = (f"D1 {'PASS' if d1['pass'] else 'FAIL'} "
               f"(practical_effect_ok={d1['practical_effect_ok']}, "
               f"uncertainty_ok={d1['uncertainty_ok']})")
    n_pos = int(np.sum(d > 0))
    n_ge = int(np.sum(d >= 0.5))
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(np.arange(1, N_HOLDOUT + 1), d, ".", ms=3, color="C0")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(0.5, color="C3", lw=0.8, ls="--")
    ax.text(N_HOLDOUT, 0.5, " +0.5 dB (D1 threshold)", va="bottom",
            ha="right", fontsize=8, color="C3")
    ax.set_xlabel("holdout identity (sorted by paired difference)")
    ax.set_ylabel("dPSNR (dB): final R1 - R0")
    ax.set_title("Primary comparison: paired dPSNR, final endpoint\n"
                 "(199 frozen holdout identities)", fontsize=10)
    ax.text(0.02, 0.86,
            f"observed mean {obs:+.4f} dB; CI95 [{ci[0]:+.4f}, "
            f"{ci[1]:+.4f}]\n{n_pos}/{N_HOLDOUT} > 0; "
            f"{n_ge}/{N_HOLDOUT} >= +0.5 dB\n"
            f"{verdict}; read from authoritative facts",
            transform=ax.transAxes, va="top", fontsize=8)
    fig.tight_layout()
    out = workdir / "dpsnr_final_r1_minus_r0.png"
    fig.savefig(out)
    plt.close(fig)
    return {"file": out.name,
            "source_fields": ["metrics.final.per_volume[*].r0.psnr",
                              "metrics.final.per_volume[*].r1.psnr",
                              "primary_comparison.d1.observed",
                              "primary_comparison.d1.pass",
                              "primary_comparison.d1.practical_effect_ok",
                              "primary_comparison.d1.uncertainty_ok",
                              "primary_comparison.paired_psnr.bootstrap."
                              "d1_mean_paired_difference.ci95"]}


def _fig_scatter(facts, workdir: Path, key: str, fname: str,
                 ylabel: str, title: str):
    """Paired scatter final R0 vs R1 with the identity line (F2/F3,
    descriptive only)."""
    r0, r1 = _final_pairs(facts, key)
    lo = float(min(r0.min(), r1.min()))
    hi = float(max(r0.max(), r1.max()))
    pad = 0.02 * (hi - lo)
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.plot(r0, r1, ".", ms=3, color="C0", alpha=0.6)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k-", lw=0.8,
            label="y = x")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"final R0 {ylabel}")
    ax.set_ylabel(f"final R1 {ylabel}")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.98, 0.02, "descriptive only; not a routing input",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            style="italic")
    fig.tight_layout()
    out = workdir / fname
    fig.savefig(out)
    plt.close(fig)
    base = f"metrics.final.per_volume[*].{{rd}}.{key}"
    return {"file": out.name,
            "source_fields": [base.format(rd="r0"), base.format(rd="r1")]}


def _fig_endpoint_means(facts, workdir: Path, key: str, fname: str,
                        ylabel: str, title: str):
    """Endpoint aggregate means (step0/final x R0/R1), read verbatim
    from metrics.*.aggregates (F4/F5, descriptive only). Rendered as a
    paired-marker dumbbell plot: pure position encoding, no bar area
    from a truncated non-zero baseline."""
    m = facts["metrics"]
    labels = ["step0", "final"]
    r0 = [m[ep]["aggregates"]["r0"][key]["mean"] for ep in labels]
    r1 = [m[ep]["aggregates"]["r1"][key]["mean"] for ep in labels]
    x = [0, 1]
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    for i in x:  # dumbbell connector per endpoint
        ax.plot([i, i], [r0[i], r1[i]], color="0.6", lw=1.0, zorder=1)
    ax.plot(x, r0, "o", color="C1", ms=9, zorder=2, label="R0 (z=0)")
    ax.plot(x, r1, "s", color="C2", ms=9, zorder=2,
            label="R1 (posterior mean)")
    for i in x:  # value labels split left/right of the marker pair
        ax.annotate(f"{r0[i]:.4f}", (i, r0[i]), xytext=(-10, -3),
                    textcoords="offset points", ha="right", fontsize=8,
                    color="C1")
        ax.annotate(f"{r1[i]:.4f}", (i, r1[i]), xytext=(10, -3),
                    textcoords="offset points", ha="left", fontsize=8,
                    color="C2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.6, 1.6)
    lo = min(r0 + r1)
    hi = max(r0 + r1)
    span = (hi - lo) if hi > lo else 0.01
    ax.set_ylim(lo - 0.4 * span, hi + 0.4 * span)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.text(0.98, 0.02, "descriptive only; not a routing input",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            style="italic")
    fig.tight_layout()
    out = workdir / fname
    fig.savefig(out)
    plt.close(fig)
    return {"file": out.name,
            "source_fields": [
                f"metrics.{{ep}}.aggregates.{{rd}}.{key}.mean "
                "(ep in step0/final, rd in r0/r1)"]}


FIGURE_BUILDERS = (
    _fig_f1,
    lambda f, w: _fig_scatter(
        f, w, "psnr", "psnr_final_r0_vs_r1.png", "PSNR (dB)",
        "Final endpoint: R0 vs R1 PSNR per holdout identity"),
    lambda f, w: _fig_scatter(
        f, w, "nmse_u", "nmse_final_r0_vs_r1.png", "NMSE_u",
        "Final endpoint: R0 vs R1 NMSE_u per holdout identity"),
    lambda f, w: _fig_endpoint_means(
        f, w, "psnr", "psnr_endpoint_means.png",
        "aggregate mean PSNR (dB)",
        "Endpoint aggregate means: PSNR (step0 vs final)"),
    lambda f, w: _fig_endpoint_means(
        f, w, "nmse_u", "nmse_endpoint_means.png",
        "aggregate mean NMSE_u",
        "Endpoint aggregate means: NMSE_u (step0 vs final)"),
)


def _render_all(facts: dict, workdir: Path) -> dict:
    """Render every figure into workdir, hash each, and assemble the
    provenance manifest. No timestamps anywhere -> byte-reproducible
    within one environment."""
    figures = []
    for build in FIGURE_BUILDERS:
        rec = build(facts, workdir)
        rec["sha256"] = _file_sha256(workdir / rec["file"])
        figures.append(rec)
        logger.info("[%s] rendered %s (sha256 %s...)", __abbr__,
                    rec["file"], rec["sha256"][:12])
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "stage": "V02I",
        "stage_version": f"{__abbr__} v{__version__}",
        "facts": {"artifact_sha256": FACTS_ARTIFACT_SHA256,
                  "semantic_sha256": FACTS_SEMANTIC_SHA256,
                  "schema": FACTS_SCHEMA,
                  "route": facts["routing"]["label"]},
        "figures": figures}
    (workdir / "v02_readout_plots_manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n")
    return manifest


def _publish(workdir: Path, out_dir: str) -> None:
    """All-or-nothing, write-once publication: the whole staged
    directory is the publication unit. The final output directory must
    NOT already exist (earlier V02I products are never silently
    overwritten); the staged sibling directory is renamed into place in
    ONE operation, so a failure can never leave a partial set behind."""
    out = Path(out_dir)
    if out.exists():
        _fail("OUT_DIR_EXISTS",
              f"output directory already exists: {out_dir} (write-once "
              f"publication; remove it explicitly to republish)")
    n = sum(1 for _ in workdir.iterdir())
    os.replace(workdir, out)  # single same-filesystem rename
    logger.info("[%s] published %d files to %s", __abbr__, n, out)


def run_render(facts_path: str, out_dir: str) -> dict:
    facts = _load_verified_facts(facts_path, FACTS_ARTIFACT_SHA256,
                                 FACTS_SEMANTIC_SHA256)
    # the render workspace must sit on the SAME filesystem as the
    # publication directory or os.replace cannot rename atomically
    # (cross-device link); a hidden sibling of out_dir guarantees that.
    # The named output directory itself is still created only after
    # every figure rendered and hashed successfully.
    parent = Path(out_dir).parent
    parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=".v02i_render_",
                                    dir=str(parent)))
    try:
        manifest = _render_all(facts, workdir)
        _publish(workdir, out_dir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return manifest


# ---------------------------------------------------------------------------
# Selftest: synthetic facts payload (labeled test double; never on the
# render path). Refusal branches must raise the typed code; the render
# branch must produce all five figures + manifest.
# ---------------------------------------------------------------------------


def _synthetic_facts() -> dict:
    """A small synthetic facts payload mirroring the pinned schema.
    SELFTEST ONLY -- it is never substituted for the authoritative
    artifact (the render mode hard-pins the authoritative shas)."""
    g = np.random.Generator(np.random.PCG64(0))
    base = np.linspace(24.0, 38.0, N_HOLDOUT)
    identities = [{"file": f"synthetic/vol{j:03d}.h5", "slice_index": j}
                  for j in range(N_HOLDOUT)]

    def endpoint(level_r0, level_r1, nm0, nm1):
        pv = []
        for j in range(N_HOLDOUT):
            pv.append({"identity": dict(identities[j]),
                       "r0": {"psnr": float(base[j] + level_r0
                                            + g.normal(0, 0.05)),
                              "nmse_u": float(nm0 + g.normal(0, 0.005))},
                       "r1": {"psnr": float(base[j] + level_r1
                                            + g.normal(0, 0.05)),
                              "nmse_u": float(nm1 + g.normal(0, 0.005))}})
        agg = {}
        for rd in ("r0", "r1"):
            agg[rd] = {k: {"mean": float(np.mean(
                [rec[rd][k] for rec in pv]))} for k in ("psnr", "nmse_u")}
        return {"per_volume": pv, "aggregates": agg}

    return {"schema": FACTS_SCHEMA,
            "semantic_sha256": "synthetic-selftest-semantic-pin",
            "population": {"n_volumes": N_HOLDOUT,
                           "identities": identities},
            "metrics": {"step0": endpoint(0.06, 0.01, 0.996, 1.011),
                        "final": endpoint(0.00, -0.01, 1.010, 1.013)},
            "primary_comparison": {
                "d1": {"observed": -0.0096, "ci95_lower": -0.0130,
                       "pass": False, "practical_effect_ok": False,
                       "uncertainty_ok": False},
                "d2": {"observed": 1.00283, "ci95_upper": 1.00355,
                       "pass": False, "practical_effect_ok": False,
                       "uncertainty_ok": False},
                "paired_psnr": {"bootstrap": {
                    "d1_mean_paired_difference": {
                        "ci95": [-0.0130, -0.0063]}}},
                "nmse_ratio": {"bootstrap": {
                    "d2_ratio_of_resampled_means": {
                        "ci95": [1.00214, 1.00355]}}}},
            "routing": {"label": "SYNTHETIC_SELFTEST_ROUTE"}}


_SELFTEST_RESULTS = []


def _st_record(name: str, ok: bool, detail: str) -> None:
    _SELFTEST_RESULTS.append({"check": name, "ok": bool(ok),
                              "detail": detail})
    (logger.info if ok else logger.error)(
        "[%s] selftest %s: %s -- %s", __abbr__, name,
        "PASS" if ok else "FAIL", detail)


def _st_expect_error(fn, code_fragment: str, label: str) -> None:
    try:
        fn()
    except V02IError as exc:
        if code_fragment not in str(exc):
            _fail("SELFTEST_ERROR_CODE_MISMATCH",
                  f"{label}: expected {code_fragment!r}, got {exc}")
        return
    _fail("SELFTEST_EXPECTED_ERROR_MISSING",
          f"{label}: expected {code_fragment!r}, none raised")


def run_selftest() -> dict:
    _SELFTEST_RESULTS.clear()
    tmp = Path(tempfile.mkdtemp(prefix="v02i_selftest_"))
    try:
        facts = _synthetic_facts()
        fpath = tmp / "synthetic_facts.json"
        fpath.write_text(json.dumps(facts))
        sha = _file_sha256(fpath)
        sem = facts["semantic_sha256"]
        ok = _load_verified_facts(str(fpath), sha, sem)
        _st_record("verified_load_accepts_valid",
                   int(ok["population"]["n_volumes"]) == N_HOLDOUT,
                   "synthetic payload passes every pin + structural check")
        _st_expect_error(
            lambda: _load_verified_facts(str(fpath), "0" * 64, sem),
            "FACTS_ARTIFACT_DRIFT", "wrong artifact sha")
        _st_record("refusal_bad_artifact_sha", True,
                   "FACTS_ARTIFACT_DRIFT raised")
        _st_expect_error(
            lambda: _load_verified_facts(str(fpath), sha, "1" * 64),
            "FACTS_SEMANTIC_DRIFT", "wrong semantic sha")
        _st_record("refusal_bad_semantic_sha", True,
                   "FACTS_SEMANTIC_DRIFT raised")
        bad_schema = dict(facts, schema="wrong-schema/0")
        p = tmp / "bad_schema.json"
        p.write_text(json.dumps(bad_schema))
        _st_expect_error(
            lambda: _load_verified_facts(str(p), _file_sha256(p), sem),
            "FACTS_SCHEMA_MISMATCH", "schema drift")
        _st_record("refusal_schema_drift", True,
                   "FACTS_SCHEMA_MISMATCH raised")
        no_metrics = dict(facts)
        no_metrics.pop("metrics")
        p = tmp / "no_metrics.json"
        p.write_text(json.dumps(no_metrics))
        _st_expect_error(
            lambda: _load_verified_facts(str(p), _file_sha256(p), sem),
            "FACTS_CONTENT_INVALID", "missing metrics key")
        _st_record("refusal_missing_key", True,
                   "FACTS_CONTENT_INVALID raised (missing key)")
        nan = json.loads(json.dumps(facts))
        nan["metrics"]["final"]["per_volume"][3]["r1"]["psnr"] = (
            float("nan"))
        p = tmp / "nan.json"
        p.write_text(json.dumps(nan))
        _st_expect_error(
            lambda: _load_verified_facts(str(p), _file_sha256(p), sem),
            "FACTS_CONTENT_INVALID", "non-finite psnr")
        nan_d1 = json.loads(json.dumps(facts))
        nan_d1["primary_comparison"]["d1"]["observed"] = float("nan")
        p = tmp / "nan_d1.json"
        p.write_text(json.dumps(nan_d1))
        _st_expect_error(
            lambda: _load_verified_facts(str(p), _file_sha256(p), sem),
            "FACTS_CONTENT_INVALID", "non-finite d1.observed")
        _st_record("refusal_non_finite", True,
                   "FACTS_CONTENT_INVALID raised (non-finite per-volume "
                   "psnr and non-finite primary_comparison.d1.observed)")
        swapped = json.loads(json.dumps(facts))
        (swapped["metrics"]["final"]["per_volume"][0]["identity"],
         swapped["metrics"]["final"]["per_volume"][1]["identity"]) = (
            swapped["metrics"]["final"]["per_volume"][1]["identity"],
            swapped["metrics"]["final"]["per_volume"][0]["identity"])
        p = tmp / "swapped.json"
        p.write_text(json.dumps(swapped))
        _st_expect_error(
            lambda: _load_verified_facts(str(p), _file_sha256(p), sem),
            "FACTS_CONTENT_INVALID", "identity misalignment")
        _st_record("refusal_identity_misalignment", True,
                   "FACTS_CONTENT_INVALID raised (misalignment)")
        # render branch: all five figures + manifest, manifest audited
        out = tmp / "plots_out"
        workdir = tmp / "plots_work"
        workdir.mkdir()
        manifest = _render_all(ok, workdir)
        _publish(workdir, str(out))
        expected = {"dpsnr_final_r1_minus_r0.png",
                    "psnr_final_r0_vs_r1.png",
                    "nmse_final_r0_vs_r1.png",
                    "psnr_endpoint_means.png",
                    "nmse_endpoint_means.png",
                    "v02_readout_plots_manifest.json"}
        produced = {f.name for f in out.iterdir()}
        man = json.loads(
            (out / "v02_readout_plots_manifest.json").read_text())
        audit_ok = (
            produced == expected
            and man["schema"] == MANIFEST_SCHEMA
            and man["facts"]["route"] == "SYNTHETIC_SELFTEST_ROUTE"
            and len(man["figures"]) == 5
            and all(len(fg["sha256"]) == 64 and fg["source_fields"]
                    for fg in man["figures"])
            and all(_file_sha256(out / fg["file"]) == fg["sha256"]
                    for fg in man["figures"]))
        _st_record("render_branch_complete", audit_ok,
                   "5 figures + manifest produced; manifest shas verify")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_fail = sum(1 for r in _SELFTEST_RESULTS if not r["ok"])
    report = {"stage": "V02I",
              "stage_version": f"{__abbr__} v{__version__}",
              "mode": "selftest",
              "checks": list(_SELFTEST_RESULTS),
              "n_pass": sum(1 for r in _SELFTEST_RESULTS if r["ok"]),
              "n_fail": n_fail}
    print(json.dumps(report, indent=1))
    if n_fail:
        _fail("SELFTEST_FAILED",
              f"{n_fail} selftest check(s) failed; do not render")
    logger.info("[%s] selftest %d/%d green; NOTHING published",
                __abbr__, report["n_pass"], len(_SELFTEST_RESULTS))
    return report


# ---------------------------------------------------------------------------
# CLI. 0 = mode completed; 2 = any typed refusal or unexpected failure
# (logged; nothing partially published).
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=f"{__abbr__} v{__version__} -- facts-only V02 "
                    f"readout-diagnostic figures (V02RD report, "
                    f"LudWiki id 155)")
    ap.add_argument("--mode", choices=["render", "selftest"],
                    default="selftest")
    ap.add_argument("--facts", default=None,
                    help="path to the authoritative "
                         "v02_readout_facts.json (render mode)")
    ap.add_argument("--out-dir", default=None,
                    help="publication directory for figures + manifest "
                         "(render mode)")
    args = ap.parse_args()
    try:
        if args.mode == "selftest":
            run_selftest()
            return EXIT_PASS
        if not args.facts or not args.out_dir:
            logger.error("[%s] render mode requires --facts and "
                         "--out-dir", __abbr__)
            return EXIT_ERROR
        run_render(args.facts, args.out_dir)
        return EXIT_PASS
    except V02IError as exc:
        logger.error("[%s] %s: %s", __abbr__, exc.error_code, exc.reason)
        return EXIT_ERROR
    except Exception:  # noqa: BLE001 -- the registered boundary
        logger.exception("[%s] unexpected runtime failure", __abbr__)
        return EXIT_ERROR  # every unexpected failure is logged -> exit 2


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Changelog
#
# v0.1 (2026-09-26): initial facts-only figures module (registered
#   optional V02I scope, DIAGNOSTIC) closing the V02RD (id 155) Figures
#   section. Pure facts reader: verified load hard-pins the
#   authoritative artifact sha256 (0d9ed10c...), schema
#   (seqref-v02rd-facts/1), semantic sha256 (92b2ee1f...), and
#   structurally validates n=199, identity alignment, and finiteness
#   before any output directory exists. Five figures: F1 sorted paired
#   dPSNR (headline, decision statistics annotated verbatim), F2/F3
#   paired R0-vs-R1 scatters (descriptive only), F4/F5 endpoint
#   aggregate-mean bars (descriptive only). No bootstrap/CI/route
#   recomputation -- only per-volume plotting transforms; the D1
#   verdict annotated on F1 is read from the stored
#   primary_comparison.d1 fields, never hardcoded. Write-once atomic
#   publication: render+hash in a hidden sibling staging directory,
#   then ONE directory rename (an existing output directory is the
#   typed refusal OUT_DIR_EXISTS; a partial publication set is
#   impossible by construction). sha256-provenance manifest
#   (seqref-v02i-plots-manifest/1) with per-figure source_fields;
#   selftest mode with six typed refusal branches (the non-finite
#   branch covers both a per-volume value and
#   primary_comparison.d1.observed) plus a full render audit. No
#   fallback, no mock, no placeholder, no silent pass on any path.
#
# v0.2 (2026-09-26): presentation-only patch after the v0.1 render
#   review. No verification logic, pin, schema, statistic, source
#   field, or figure filename changed. F1: title wrapped to two lines
#   (the one-line title was clipped at the right edge) and the
#   annotation block lowered clear of the +0.5 dB threshold line.
#   F4/F5: bar charts replaced by paired-marker dumbbell plots -- pure
#   position encoding instead of bar area from a truncated non-zero
#   baseline, which visually exaggerated the tiny endpoint
#   differences. Selftest unchanged (8/8); render remains write-once.

#!/usr/bin/env python3
# SEQREF-P3PLT v0.4 -- P3 diagnostic plots (persisted facts ONLY)
# LIFETIME: KEEP
#
# CHANGELOG
# - v0.4 (2026-08-07): REWRITTEN FOR seqref-p3-facts/2 (A4). The /1 reader
#   assumed a single serialised global map (map_serialized / mask_columns /
#   free_coordinates) and carried a BLOCK-summary path. Under A4 neither
#   exists: the artefact is the enumeration RULE plus per-realisation
#   bindings, and the stage publishes facts only on PASS.
#     * load_verified REJECTS /1 by name: a /1 artefact is the superseded
#       global-map format, readable only as historical evidence about the
#       v0.3.x implementation, never interpretable as per-realisation facts.
#     * class maps are drawn PER REALISATION from
#       coordinate_map.realisations[].acquired_columns (up to 4 panels).
#     * the BLOCK panel is gone; a census/A4-regime summary panel is emitted
#       first instead.
#     * empirical-energy bars use the grid-keyed /2 field names
#       (*_location_mean_energy); packed-index keying is gone with /1.
#     * the map-payload panel becomes the enumeration-rule + realisation-hash
#       panel.
# - v0.3 (2026-07-30): field names realigned to the flat per-slice row shape
#   the v0.5 stage publishes (P1/P2 row convention), replacing v0.2's nested
#   block assumptions. Reader now verifies via preflight_io.verify_sidecar and
#   checks schema, stage and artefact_type, as the P1/P2 readers do.
# - Plot 7 is a GENUINE fp32-vs-fp64 comparison across all three metrics; the
#   operation-order diagnostic it once substituted has its own figure.
# - The SCRIPT is KEEP; the emitted PNGs are DIAGNOSTIC and deleted after
#   inspection.

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "seqref_mri", "src"))

from preflight_io import verify_sidecar  # noqa: E402

logger = logging.getLogger("SEQREF-P3PLT")

EXPECTED_SCHEMA = "seqref-p3-facts/2"
LEGACY_SCHEMA = "seqref-p3-facts/1"   # v0.3.x global-map format, SUPERSEDED
EXPECTED_STAGE = "P3"
EXPECTED_TYPE = "stage_facts"


def load_verified(path: str) -> dict:
    """Sidecar first, then schema / stage / artefact_type. A valid sidecar
    proves byte integrity, not that this is a P3 facts artefact.

    /1 is rejected BY NAME, not by generic mismatch: the A4 contract
    (seqref-p3-facts/2 schema_compatibility) requires a consumer to refuse
    the superseded global-map format outright rather than reinterpret it as
    per-realisation facts.
    """
    verify_sidecar(path)
    with open(path, "r", encoding="utf-8") as fh:
        facts = json.load(fh)
    if facts.get("artefact_type") == "error":
        logger.error("refusing to plot an ERROR record: %s", path)
        raise ValueError("artefact is an ERROR record, not stage facts")
    if facts.get("schema") == LEGACY_SCHEMA:
        logger.error("rejecting %s artefact: %s", LEGACY_SCHEMA, path)
        raise ValueError(
            f"schema is {LEGACY_SCHEMA}: the SUPERSEDED v0.3.x global-map "
            f"format. It is readable only as historical evidence about that "
            f"implementation and must never be plotted as per-realisation "
            f"facts. Re-run P3 (>= v0.4) to produce {EXPECTED_SCHEMA}.")
    for field, want in (("schema", EXPECTED_SCHEMA), ("stage", EXPECTED_STAGE),
                        ("artefact_type", EXPECTED_TYPE)):
        if facts.get(field) != want:
            logger.error("reader refuses artefact: %s=%r, expected %r", field,
                         facts.get(field), want)
            raise ValueError(f"{field} is {facts.get(field)!r}, expected {want!r}")
    return facts


def _save(fig, out_dir: Path, name: str) -> Path:
    p = out_dir / name
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    logger.info("wrote %s", p.name)
    return p


def _col(rows, key, probe=None):
    out = []
    for r in rows:
        v = r["probes"].get(probe, {}).get(key) if probe else r.get(key)
        out.append(np.nan if v is None else float(v))
    return np.asarray(out, dtype=float)


def _lg(v):
    return np.log10(np.maximum(np.nan_to_num(v, nan=1e-20), 1e-20))


def _annotate(ax, facts, gate):
    for w in facts.get("summary", {}).get("worst_slices") or []:
        if w and w.get("gate") == gate:
            m = w.get("margin")
            ax.set_xlabel(f"{ax.get_xlabel()}   |   worst: idx "
                          f"{w['dataset_index']} slice {w['slice_index']}, "
                          f"obs {w['observed']:.3e}, margin "
                          f"{'n/a' if m is None else format(m, '.3g')}",
                          fontsize=8)
            return


def plot_all(facts: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    rows = facts.get("slices") or []
    thr = facts.get("thresholds", {})
    census = facts.get("mask_census") or {}
    cm = facts.get("coordinate_map") or {}
    realisations = cm.get("realisations") or []
    H, W = cm.get("grid_shape", [96, 96])
    ma = facts.get("map_audit") or {}
    dims = (ma.get("c6_dimensions") or {}) if ma else {}

    # 0. census / A4-regime summary FIRST. There is no BLOCK panel: under A4
    #    the stage publishes facts only on PASS, and set variation is the
    #    expected regime, so this panel records the regime, not a failure.
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.axis("off")
    ax.set_title(f"P3 {facts.get('verdict')} -- mask census (A4: per-realisation)")
    head = [f"verdict                    : {facts.get('verdict')}",
            f"schema                     : {facts.get('schema')}",
            f"run_mode                   : {facts.get('run_mode')}",
            f"artefact format            : {cm.get('format')}",
            ""]
    head += [f"{k:38s} {v}" for k, v in census.items()
             if not isinstance(v, (list, dict))]
    ax.text(0.0, 0.97, "\n".join(head), fontsize=8, va="top", family="monospace")
    written.append(_save(fig, out_dir, "p3_00_census_regime.png"))

    # 1. class maps, one panel PER REALISATION (up to 4). Under A4 there is no
    #    single mask: each realisation has its own acquired column set.
    shown = realisations[:4]
    if shown:
        fig, axes = plt.subplots(1, len(shown),
                                 figsize=(3.2 * len(shown) + 1.5, 4))
        if len(shown) == 1:
            axes = [axes]
        for ax, rel in zip(axes, shown):
            grid = np.zeros((H, W))
            for c in (int(c) for c in rel.get("acquired_columns", [])):
                grid[:, c] = 1.0
            ax.imshow(grid, cmap="viridis", aspect="auto",
                      interpolation="nearest")
            ax.set_title(f"sha {str(rel.get('map_sha256'))[:8]}\n"
                         f"{rel.get('n_slices')} slice(s)", fontsize=8)
            ax.set_xlabel("k-space column", fontsize=8)
        axes[0].set_ylabel("k-space row")
        fig.suptitle(f"Class maps: {cm.get('n_realisations')} realisation(s) "
                     f"(first {len(shown)} shown) -- acquired light, free dark",
                     fontsize=10)
        written.append(_save(fig, out_dir, "p3_01_realisation_class_maps.png"))
    else:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.axis("off")
        ax.set_title("Class maps")
        ax.text(0.0, 0.8, "no realisations recorded in this artefact",
                fontsize=9, va="top", family="monospace")
        written.append(_save(fig, out_dir, "p3_01_realisation_class_maps.png"))

    # 2. two-link map audit across realisations (both gate)
    l1 = ma.get("link1_ordered_list_identity") or {}
    l2 = ma.get("link2_unique_valued_oracle") or {}
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.axis("off"); ax.set_title("Two-link map audit, ALL realisations (both gate)")
    ax.text(0.0, 0.95, "\n".join([
        f"realisations audited          : {l1.get('n_realisations_audited')}",
        f"link 1 all published == indep : {l1.get('all_published_vs_independent_equal')}",
        f"       first failing           : {l1.get('first_failing_realisation')}",
        f"link 2 all oracle bitwise     : {l2.get('all_unique_oracle_bitwise_equal')}",
        f"       first failing           : {l2.get('first_failing_realisation')}",
        "",
        "the true-coefficient probe is BLIND to a shared permutation;",
        "these two links are the per-realisation map validators",
    ]), fontsize=8.5, va="top", family="monospace")
    written.append(_save(fig, out_dir, "p3_02_map_audit.png"))

    if rows:
        # 3. C3a
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.hist(_lg(_col(rows, "c3a_decode_zero_rel")), bins=40, color="#4573a7")
        ax.axvline(np.log10(thr["P3_DECODE_TOL"]), color="crimson",
                   label="P3_DECODE_TOL")
        ax.axvline(np.log10(facts["summary"]["c3a_expected_rel"]), color="green",
                   linestyle="--", label="expected ~1e-7 (operation order)")
        ax.set_xlabel("log10 c3a_decode_zero_rel"); ax.set_ylabel("slices")
        ax.set_title("C3a zero-state decode vs the live anchor cond_in")
        _annotate(ax, facts, "c3a_decode_zero"); ax.legend(fontsize=8)
        written.append(_save(fig, out_dir, "p3_03_c3a.png"))

        # 4. fixity by probe, not pooled
        fig, ax = plt.subplots(figsize=(7.5, 4))
        for lbl, col in (("scale_A", "#4573a7"), ("scale_B", "#a75745"),
                         ("true", "#45a76a")):
            ax.hist(_lg(_col(rows, "fixity_rel", lbl)), bins=30, alpha=0.55,
                    label=lbl, color=col)
        ax.axvline(np.log10(thr["P3_FIXITY_TOL"]), color="crimson",
                   label="P3_FIXITY_TOL")
        ax.set_xlabel("log10 fixity_rel"); ax.set_ylabel("slices")
        ax.set_title("C3c measured fixity, three probes")
        _annotate(ax, facts, "c3c_fixity"); ax.legend(fontsize=8)
        written.append(_save(fig, out_dir, "p3_04_fixity.png"))

        # 5. free-coordinate round trip
        fig, ax = plt.subplots(figsize=(7.5, 4))
        for lbl in ("scale_A", "scale_B", "true"):
            ax.hist(_lg(_col(rows, "unmeasured_roundtrip_rel", lbl)), bins=30,
                    alpha=0.55, label=lbl)
        ax.axvline(np.log10(thr["P3_ROUNDTRIP_TOL"]), color="crimson",
                   label="P3_ROUNDTRIP_TOL")
        ax.set_xlabel("log10 unmeasured_roundtrip_rel"); ax.set_ylabel("slices")
        ax.set_title("Free-coordinate encode/decode round trip")
        _annotate(ax, facts, "unmeasured_roundtrip"); ax.legend(fontsize=8)
        written.append(_save(fig, out_dir, "p3_05_roundtrip.png"))

        # 6. LIKE-FOR-LIKE: P3 max|M F dx| vs P2 persisted max_MFdx.
        a = _col(rows, "max_MFdx_p2")
        b = _col(rows, "max_MFdx_p3")
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(a, b, s=10, alpha=0.6)
        lim = [np.nanmin([a.min(), b.min()]), np.nanmax([a.max(), b.max()])]
        ax.plot(lim, lim, "k--", linewidth=1, label="identity")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(fontsize=8)
        ax.set_xlabel("P2 persisted max|M F dx|")
        ax.set_ylabel("P3 live max|M F dx|")
        ax.set_title("Measured-support leakage continuity\n"
                     "same quantity, same domain -- NON-VERDICT", fontsize=10)
        written.append(_save(fig, out_dir, "p3_06_max_MFdx_continuity.png"))

        # 6b. residual-energy-ratio continuity, also like-for-like
        a = _col(rows, "residual_energy_ratio_p2")
        b = _col(rows, "residual_energy_ratio_p3")
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(a, b, s=10, alpha=0.6)
        lim = [np.nanmin([a.min(), b.min()]), np.nanmax([a.max(), b.max()])]
        ax.plot(lim, lim, "k--", linewidth=1, label="identity")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(fontsize=8)
        ax.set_xlabel("P2 residual energy ratio")
        ax.set_ylabel("P3 residual energy ratio")
        ax.set_title("Residual-ratio continuity -- NON-VERDICT", fontsize=10)
        written.append(_save(fig, out_dir, "p3_06b_residual_ratio_continuity.png"))

        # 6c. image-domain reconstruction discrepancy, on its OWN axes.
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.hist(_lg(_col(rows, "observed_true_reconstruction_abs")), bins=40,
                color="#45a76a")
        ax.set_xlabel("log10 max_image |decode(u_true) - x_norm|")
        ax.set_ylabel("slices")
        ax.set_title("Image-domain reconstruction discrepancy\n"
                     "= max |F^H(M F dx)| -- NOT comparable to max_MFdx",
                     fontsize=10)
        written.append(_save(fig, out_dir, "p3_06c_image_domain_discrepancy.png"))

        # 7. GENUINE fp32 vs fp64, all three metrics
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, (a, b, name) in zip(axes, (
                ("c3a_rel_f32", "c3a_rel_f64", "C3a"),
                ("fixity_rel_f32", "fixity_rel_f64", "fixity"),
                ("roundtrip_rel_f32", "roundtrip_rel_f64", "round trip"))):
            x, y = _col(rows, a), _col(rows, b)
            ax.scatter(np.maximum(x, 1e-20), np.maximum(y, 1e-20), s=10,
                       alpha=0.6)
            hi = max(np.nanmax(x), np.nanmax(y), 1e-19)
            ax.plot([1e-20, hi], [1e-20, hi], "k--", linewidth=1)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(f"{name} fp32"); ax.set_ylabel(f"{name} fp64")
            ax.set_title(f"{name}: complex128 operator arithmetic over\nfloat32-prepared inputs -- NON-VERDICT",
                         fontsize=10)
        written.append(_save(fig, out_dir, "p3_07_fp32_vs_fp64.png"))

        # 8. operation-order diagnostic, separate from precision
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.hist(_lg(_col(rows, "normalized_vs_raw_path_rel")), bins=40,
                color="#8a6bbf")
        ax.set_xlabel("log10 normalized_vs_raw_path_rel"); ax.set_ylabel("slices")
        ax.set_title("Operation-order difference: normalised vs raw assembly\n"
                     "NON-VERDICT, distinct from precision", fontsize=10)
        ax.text(0.02, 0.92, f"raw path bitwise == cond_in on "
                            f"{facts['summary'].get('raw_path_bitwise_equal_count')}"
                            f"/{len(rows)} slices",
                transform=ax.transAxes, fontsize=8, family="monospace")
        written.append(_save(fig, out_dir, "p3_08_operation_order.png"))

        # 10. P3-vs-P2 continuity
        fig, (b1, b2) = plt.subplots(1, 2, figsize=(11, 4))
        b1.hist(_lg(_col(rows, "max_MFdx_rel_diff")), bins=40,
                color="#c08a3e")
        b1.set_xlabel("log10 rel diff, max|M F dx|"); b1.set_ylabel("slices")
        b1.set_title("Leakage continuity", fontsize=10)
        b2.hist(_lg(_col(rows, "residual_energy_ratio_rel_diff")),
                bins=40, color="#3e8ac0")
        b2.set_xlabel("log10 rel diff, residual energy ratio")
        b2.set_title("Residual-ratio continuity", fontsize=10)
        fig.suptitle("P3 vs persisted P2 -- NON-VERDICT", fontsize=11)
        written.append(_save(fig, out_dir, "p3_10_p2_continuity.png"))

    # 9. constraint audit -- measured values
    ca = facts.get("constraint_audit") or {}
    emp = ca.get("empirical_diagnostics") or {}
    fig, (c1, c2) = plt.subplots(1, 2, figsize=(12, 5))
    c1.axis("off")
    c1.set_title("STRUCTURAL, per realisation (failure => ERROR)", fontsize=10)
    c1.text(0.0, 0.95, "\n".join(f"{k:46s} {v}" for k, v in
                                 (ca.get("structural_checks") or {}).items()),
            fontsize=7.5, va="top", family="monospace")
    c2.axis("off")
    c2.set_title("EMPIRICAL -- NON-VERDICT, never blocks", fontsize=10)
    c2.text(0.0, 0.95, "\n".join(f"{k:46s} {v}" for k, v in emp.items()),
            fontsize=7.5, va="top", family="monospace")
    written.append(_save(fig, out_dir, "p3_09_constraint_audit.png"))

    # 11. per-LOCATION energy and conjugate-pair violation summary. /2 fields
    #     are keyed by PHYSICAL GRID LOCATION, never by packed index.
    fig, (d1, d2) = plt.subplots(1, 2, figsize=(11, 4))
    e = [emp.get("min_location_mean_energy"),
         emp.get("median_location_mean_energy"),
         emp.get("max_location_mean_energy")]
    if all(v is not None for v in e):
        d1.bar(["min", "median", "max"], e, color="#4573a7")
        d1.set_yscale("log")
    else:
        d1.axis("off")
    d1.set_title("Per-location mean energy, grid-keyed (NON-VERDICT)",
                 fontsize=10)
    v = [emp.get("conjugate_pair_violation_min"),
         emp.get("conjugate_pair_violation_median"),
         emp.get("conjugate_pair_violation_p95"),
         emp.get("conjugate_pair_violation_max")]
    if all(x is not None for x in v):
        d2.bar(["min", "median", "p95", "max"], v, color="#a75745")
        d2.set_yscale("log")
    else:
        d2.axis("off")
    d2.set_title(f"Unmeasured conjugate-pair violation\n"
                 f"{emp.get('n_unmeasured_pairs_tested')} pairs -- NON-VERDICT",
                 fontsize=10)
    written.append(_save(fig, out_dir, "p3_11_empirical_distributions.png"))

    # 12. dimensions and the enumeration-rule / realisation-hash panel
    fig, (e1, e2) = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["n_acquired", "n_free_complex", "flow_dim_real"]
    values = [dims.get("n_acquired", 0), dims.get("n_free_enumerated", 0),
              dims.get("flow_dim_real", 0)]
    e1.bar(labels, values, color=["#a7a7a7", "#4573a7", "#45a76a"])
    for i, val in enumerate(values):
        e1.text(i, val, f"{val:,}", ha="center", va="bottom", fontsize=9)
    e1.set_ylabel("count")
    e1.set_title("Dimensions (state size only -- NOT model feasibility)",
                 fontsize=10)
    e2.axis("off"); e2.set_title("Enumeration rule + realisations", fontsize=10)
    first_rel = realisations[0] if realisations else {}
    e2.text(0.0, 0.95, "\n".join([
        f"format             : {cm.get('format')}",
        f"enumeration_rule   : {cm.get('enumeration_rule')}",
        f"packing_order      : {cm.get('complex_packing_order')}",
        f"n_realisations     : {cm.get('n_realisations')}",
        f"first map_sha256   : {str(first_rel.get('map_sha256'))[:40]}",
        f"flow_dim_invariant : {dims.get('flow_dim_invariant')}",
        f"distinct flow_dims : {dims.get('distinct_flow_dim_real')}",
        f"n_free_formula     : {dims.get('n_free_formula')}",
        f"n_free_enumerated  : {dims.get('n_free_enumerated')}",
        f"counts agree       : {dims.get('all_n_free_counts_agree')}",
        f"state fp32 bytes   : {dims.get('bytes_per_state_fp32', 0):,}",
        f"batch b8 fp32      : {dims.get('bytes_per_batch_fp32_b8', 0):,}",
    ]), fontsize=8.5, va="top", family="monospace")
    written.append(_save(fig, out_dir, "p3_12_dimensions_and_rule.png"))
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="SEQREF-P3PLT v0.4 -- P3 diagnostic plots (schema /2)")
    ap.add_argument("--facts", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        facts = load_verified(args.facts)
        written = plot_all(facts, Path(args.out_dir))
        print(json.dumps({"n_figures": len(written),
                          "figures": [p.name for p in written]}, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.exception("plotting failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

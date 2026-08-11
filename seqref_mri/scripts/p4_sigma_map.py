# =============================================================================
# SEQREF-P4S2D v0.1 -- scripts.p4_sigma_map
# LIFETIME: DIAGNOSTIC
# Purpose: read-only diagnostic rendering of a P4 /2 statistics artefact
#   (schema seqref-p4-stats/2): per-location raw_std maps, count map,
#   floor-margin ratio map, mean maps, and raw_std distribution histograms
#   with the FROZEN floor lines. Provenance-bound: every output filename
#   and figure title carries the artefact's recomputed file sha256 prefix.
# INSPECTION CONTRACT (frozen):
#   Visual inspection may identify apparent implementation/indexing/
#   rendering defects requiring investigation (centre band populated,
#   row/column transpose, NaN/Inf, missing/duplicate locations, gross
#   striping inconsistent with count geometry, rendered values disagreeing
#   with the JSON). It MUST NOT change FLOOR_FACTOR, the branch threshold,
#   the eligible set, any statistic definition, or choose a branch. This
#   script derives NO new threshold and publishes NO facts artefact; it is
#   not a stage, not in CODE_HASH_FILES, and has no EXEC 9.1 registry
#   entry.
# FAILURE DISCIPLINE: every failure path is logger.error + typed raise;
#   no fallback, no mock, no placeholder, no silent pass. Exit codes:
#   0 = rendered, 2 = ERROR (construction/contract).
# Changelog
#   v0.1 (2026-08-10) Created after the P4 /2 256-row smoke PASS; scope
#     per the tightened plan: 2x3 map figure (raw_std re/im, count,
#     floor-margin ratio, mean re/im) + distribution figure (re/im
#     histograms with frozen floor lines); quantile console summary;
#     structural assertions on the location records.
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402

logger = logging.getLogger("seqref_mri.p4_sigma_map")

SCRIPT_ID = "SEQREF-P4S2D"
SCRIPT_VERSION = "v0.1"
FACTS_SCHEMA = "seqref-p4-stats/2"
GRID_HW = 96
CENTRE_COLS = tuple(range(44, 52))  # always acquired at 4x; never observed
LOCATION_KEYS = {
    "row", "column", "count", "floor_hit",
    "mean_re", "mean_im", "raw_std_re", "raw_std_im",
    "per_location_scale_re", "per_location_scale_im",
    "applied_mean_re", "applied_mean_im",
    "applied_scale_re", "applied_scale_im",
}
EXIT_OK, EXIT_ERROR = 0, 2


class DiagnosticError(Exception):
    """Construction/contract failure of the diagnostic (exit 2)."""


def _fail(code: str, msg: str) -> None:
    logger.error("[%s] %s -> %s", SCRIPT_ID, code, msg)
    raise DiagnosticError(f"{code}: {msg}")


def req(node, *path):
    """Required-key walker: missing key is a contract error, never a
    default."""
    cur = node
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            _fail("ARTEFACT_KEY_MISSING",
                  f"required key path {path!r} missing at {key!r}")
        cur = cur[key]
    return cur


def load_artefact(path: str) -> tuple[dict, str]:
    """Read-only load + contract checks. Returns (payload, file_sha256)."""
    if not os.path.isfile(path):
        _fail("ARTEFACT_NOT_FOUND", f"no artefact at {path!r}")
    with open(path, "rb") as fh:
        raw = fh.read()
    sha = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        _fail("ARTEFACT_NOT_JSON", f"{path!r} does not parse: {exc}")
    if payload.get("schema") != FACTS_SCHEMA:
        _fail("SCHEMA_MISMATCH",
              f"schema {payload.get('schema')!r} != {FACTS_SCHEMA!r}; "
              f"this diagnostic renders /2 statistics artefacts only")
    for key in ("run_mode", "authoritative", "branch", "c7", "channels",
                "inheritance", "locations", "population", "parents",
                "semantic_sha256", "summary"):
        req(payload, key)
    logger.info("%s %s loaded %s file_sha256=%s run_mode=%s",
                SCRIPT_ID, SCRIPT_VERSION, os.path.basename(path),
                sha, payload["run_mode"])
    return payload, sha


def assemble_grids(payload: dict) -> dict:
    """Reshape the location records into (H, W) float64 grids. NaN marks
    never-observed positions (the always-acquired centre columns 44..51);
    NaN is RENDERED as a distinct band, never zero-filled. Structural
    assertions are construction gates -> ERROR on any violation."""
    locs = payload["locations"]
    n_eligible_cols = req(payload, "inheritance", "n_eligible_columns")
    n_eligible_locs = req(payload, "inheritance", "n_eligible_locations")
    if n_eligible_locs != GRID_HW * n_eligible_cols:
        _fail("INHERITANCE_INCONSISTENT",
              f"n_eligible_locations {n_eligible_locs} != "
              f"{GRID_HW} * n_eligible_columns {n_eligible_cols}")
    if len(locs) != n_eligible_locs:
        _fail("LOCATION_COUNT_MISMATCH",
              f"{len(locs)} records != n_eligible_locations "
              f"{n_eligible_locs}")
    seen = set()
    grids = {name: np.full((GRID_HW, GRID_HW), np.nan, dtype=np.float64)
             for name in ("raw_std_re", "raw_std_im", "mean_re", "mean_im",
                          "per_location_scale_re", "per_location_scale_im",
                          "count", "floor_hit")}
    for i, rec in enumerate(locs):
        if set(rec.keys()) != LOCATION_KEYS:
            _fail("LOCATION_RECORD_KEYS",
                  f"record {i} keys {sorted(rec.keys())} != the frozen "
                  f"14-key location record")
        r, c = rec["row"], rec["column"]
        if not (0 <= r < GRID_HW and 0 <= c < GRID_HW):
            _fail("LOCATION_OUT_OF_RANGE",
                  f"record {i} at (r={r}, c={c}) outside 0..{GRID_HW - 1}")
        if c in CENTRE_COLS:
            _fail("CENTRE_POPULATED",
                  f"record {i} at centre column {c}; centre columns are "
                  f"always acquired and must never be observed")
        if (r, c) in seen:
            _fail("DUPLICATE_LOCATION",
                  f"duplicate record at (r={r}, c={c})")
        seen.add((r, c))
        for name in grids:
            grids[name][r, c] = float(rec[name])
    if len(seen) != n_eligible_locs:
        _fail("LOCATION_UNIQUENCY",
              f"{len(seen)} unique (r,c) != {n_eligible_locs}")
    # Observed entries are those a record wrote (NaN = never observed);
    # Inf written by a record must NOT slip through the NaN filter.
    observed = {name: ~np.isnan(grids[name]) for name in grids}
    # raw_std: finite and non-negative. raw_std == 0 is LEGITIMATE under
    # the frozen /2 semantics: a STRICT-< floor hit, which is exactly
    # what the positive floor is for. The registered pre-vote validity
    # gate applies to per_location_scale, not to raw_std.
    for name in ("raw_std_re", "raw_std_im"):
        obs = grids[name][observed[name]]
        if not np.isfinite(obs).all():
            _fail("NON_FINITE_RAW_STD", f"{name} has NaN/Inf observations")
        if (obs < 0).any():
            _fail("NEGATIVE_RAW_STD",
                  f"{name} has a negative observed entry; a raw "
                  f"standard deviation is non-negative by definition")
    # The actual frozen positivity condition: per_location_scale > 0 at
    # every eligible location (pre-vote validity, EXEC 13 P4 /2 block).
    for name in ("per_location_scale_re", "per_location_scale_im"):
        obs = grids[name][observed[name]]
        if not np.isfinite(obs).all() or not (obs > 0).all():
            _fail("INVALID_PER_LOCATION_SCALE",
                  f"{name} has non-finite or non-positive entries; the "
                  f"frozen pre-vote validity requires per_location_scale "
                  f"> 0 at every eligible location")
    # Remaining rendered quantities: finite wherever observed (no silent
    # drop of Inf/NaN written by a record).
    for name in ("mean_re", "mean_im", "count", "floor_hit"):
        obs = grids[name][observed[name]]
        if not np.isfinite(obs).all():
            _fail("NON_FINITE_FIELD", f"{name} has NaN/Inf observations")
    logger.info("%s grids assembled: %d observed locations, centre "
                "columns %d-%d masked", SCRIPT_ID, len(seen),
                CENTRE_COLS[0], CENTRE_COLS[-1])
    return grids


def floor_margins(grids: dict, floor_re: float,
                  floor_im: float) -> tuple[np.ndarray, dict]:
    """Dimensionless margin to the FROZEN floor:
    margin(r,c) = min(raw_std_re/floor_re, raw_std_im/floor_im).
    <1 floor hit; ==1 exactly at floor (NOT a hit, STRICT-<); >1 above.
    This renders the frozen rule; it derives no new threshold."""
    if floor_re <= 0 or floor_im <= 0:
        _fail("NON_POSITIVE_FLOOR",
              f"floor_re={floor_re}, floor_im={floor_im}; the frozen "
              f"floor must be positive")
    ratio_re = grids["raw_std_re"] / floor_re
    ratio_im = grids["raw_std_im"] / floor_im
    margin = np.fmin(ratio_re, ratio_im)
    obs = np.isfinite(margin)
    flat = margin[obs]
    idx = int(np.argmin(flat))
    rr, cc = np.argwhere(obs)[idx]
    ch = "re" if ratio_re[rr, cc] <= ratio_im[rr, cc] else "im"
    info = {"min_ratio": float(flat[idx]),
            "min_location": {"row": int(rr), "column": int(cc),
                             "channel": ch},
            "n_below_1": int((flat < 1.0).sum())}
    return margin, info


def _observed(grid: np.ndarray) -> np.ndarray:
    return grid[np.isfinite(grid)]


def _log_panel(ax, grid: np.ndarray, title: str,
               cmap: str) -> tuple[float, float, int]:
    """Log-scaled heatmap. Same scaling METHOD per quantity; the norm is
    per-panel from positive observed data (a shared cross-channel norm is
    NOT forced: it can flatten one channel when ranges differ). Observed
    ZEROS are legitimate (raw_std == 0 -> STRICT-< floor hit): they are
    excluded from the log layer and overlaid as magenta squares, clearly
    distinct from the grey never-observed centre band."""
    obs_mask = ~np.isnan(grid)
    obs = grid[obs_mask]
    pos = obs[obs > 0]
    n_zero = int((obs == 0).sum())
    if pos.size == 0:
        _fail("LOG_NORM_DOMAIN",
              f"{title}: no positive observed values to log-render "
              f"({n_zero} zeros); nothing to normalize")
    vmin, vmax = float(pos.min()), float(pos.max())
    masked = np.ma.masked_invalid(grid)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="0.75")
    im = ax.imshow(masked, norm=LogNorm(vmin=vmin, vmax=vmax),
                   cmap=cmap_obj, origin="upper", interpolation="nearest")
    if n_zero > 0:
        zr, zc = np.where(obs_mask & (grid == 0))
        ax.scatter(zc, zr, marker="s", s=18, facecolors="magenta",
                   edgecolors="black", linewidths=0.4,
                   label=f"observed zero (floor hit): {n_zero}",
                   zorder=3)
        ax.legend(fontsize=8, loc="upper right")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("k-space column")
    ax.set_ylabel("k-space row")
    plt.colorbar(im, ax=ax, fraction=0.046)
    return vmin, vmax, n_zero


def render_map_figure(payload: dict, sha: str, grids: dict,
                      margin: np.ndarray, margin_info: dict,
                      out_dir: str) -> str:
    """Core 2x3 figure: raw_std re / raw_std im (log), count, floor-margin
    ratio (log, contour at the frozen boundary 1.0, markers on any <1),
    mean re / mean im (symmetric about zero, shared vmax across channels
    for direct comparison)."""
    branch = req(payload, "branch")
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    ranges = {}
    ranges["raw_std_re"] = _log_panel(
        axes[0, 0], grids["raw_std_re"], "raw_std_re (log)", "viridis")
    ranges["raw_std_im"] = _log_panel(
        axes[0, 1], grids["raw_std_im"], "raw_std_im (log)", "viridis")

    counts = grids["count"]
    c_masked = np.ma.masked_invalid(counts)
    cmap_c = plt.get_cmap("cividis").copy()
    cmap_c.set_bad(color="0.75")
    im_c = axes[0, 2].imshow(c_masked, cmap=cmap_c, origin="upper",
                             interpolation="nearest")
    axes[0, 2].set_title(
        f"count (min {int(np.nanmin(counts))}, "
        f"max {int(np.nanmax(counts))})", fontsize=10)
    axes[0, 2].set_xlabel("k-space column")
    axes[0, 2].set_ylabel("k-space row")
    plt.colorbar(im_c, ax=axes[0, 2], fraction=0.046)

    axm = axes[1, 0]
    m_obs = _observed(margin)
    # margin == 0 is a legitimate raw_std==0 floor hit; the symmetric
    # log-decade range must be computed from POSITIVE margins only, and
    # the zeros are shown by the <1 red-marker overlay below.
    m_pos = m_obs[m_obs > 0]
    if m_pos.size == 0:
        _fail("MARGIN_ALL_ZERO",
              "every observed floor-margin ratio is zero; the log panel "
              "has no positive support (all locations are zero-std "
              "floor hits) -- investigate the artefact before any "
              "visual inspection")
    lo, hi = float(m_pos.min()), float(m_pos.max())
    decades = max(-np.log10(min(lo, 1.0)), np.log10(max(hi, 1.0)), 0.05)
    m_masked = np.ma.masked_invalid(margin)
    cmap_m = plt.get_cmap("RdYlGn").copy()
    cmap_m.set_bad(color="0.75")
    im_m = axm.imshow(m_masked,
                      norm=LogNorm(vmin=10 ** (-decades),
                                   vmax=10 ** decades),
                      cmap=cmap_m, origin="upper",
                      interpolation="nearest")
    axm.contour(np.ma.filled(m_masked, np.nan), levels=[1.0],
                colors="black", linewidths=0.8)
    if margin_info["n_below_1"] > 0:
        hits = np.argwhere(np.ma.filled(m_masked, np.inf) < 1.0)
        axm.scatter(hits[:, 1], hits[:, 0], s=6, facecolors="none",
                    edgecolors="red", linewidths=0.7)
    axm.set_title(
        "floor-margin ratio min(raw_std/floor) over channels (log)\n"
        f"min {margin_info['min_ratio']:.3f} at "
        f"(r={margin_info['min_location']['row']}, "
        f"c={margin_info['min_location']['column']}, "
        f"{margin_info['min_location']['channel']}); "
        f"<1: {margin_info['n_below_1']}", fontsize=9)
    axm.set_xlabel("k-space column")
    axm.set_ylabel("k-space row")
    plt.colorbar(im_m, ax=axm, fraction=0.046)

    vmax_mean = float(max(np.abs(_observed(grids["mean_re"])).max(),
                          np.abs(_observed(grids["mean_im"])).max()))
    if vmax_mean == 0.0:
        logger.warning("%s both mean maps are exactly 0.0; rendering "
                       "guard vmax=1e-300 (display only, no data change)",
                       SCRIPT_ID)
        vmax_mean = 1e-300
    mean_norm = Normalize(vmin=-vmax_mean, vmax=vmax_mean)
    for ax, name in ((axes[1, 1], "mean_re"), (axes[1, 2], "mean_im")):
        masked = np.ma.masked_invalid(grids[name])
        cmap_b = plt.get_cmap("RdBu_r").copy()
        cmap_b.set_bad(color="0.75")
        im_b = ax.imshow(masked, norm=mean_norm, cmap=cmap_b,
                         origin="upper", interpolation="nearest")
        ax.set_title(f"{name} (symmetric, vmax={vmax_mean:.3e})",
                     fontsize=10)
        ax.set_xlabel("k-space column")
        ax.set_ylabel("k-space row")
        plt.colorbar(im_b, ax=ax, fraction=0.046)

    fig.suptitle(
        f"{SCRIPT_ID} {SCRIPT_VERSION} -- raw_std / count / floor-margin "
        f"/ mean | {payload['schema']} | run_mode={payload['run_mode']} | "
        f"branch={branch['selected']} (smoke_scale="
        f"{branch['smoke_scale']}) | artefact {sha[:12]} | grey band = "
        f"centre columns 44-51 (always acquired, never observed)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(out_dir, f"sigma_map_{sha[:12]}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("%s map figure written: %s", SCRIPT_ID, path)
    logger.info("%s log-norm ranges (positive support; per-channel, NOT "
                "shared): raw_std_re [%.6e, %.6e] zeros=%d, raw_std_im "
                "[%.6e, %.6e] zeros=%d", SCRIPT_ID,
                ranges["raw_std_re"][0], ranges["raw_std_re"][1],
                ranges["raw_std_re"][2], ranges["raw_std_im"][0],
                ranges["raw_std_im"][1], ranges["raw_std_im"][2])
    return path


def render_distribution_figure(payload: dict, sha: str, grids: dict,
                               floor_re: float, floor_im: float,
                               out_dir: str) -> str:
    """raw_std histograms with the FROZEN floor lines. Log-x only when the
    observed range warrants it (max/min > 10)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, name, floor in ((axes[0], "raw_std_re", floor_re),
                            (axes[1], "raw_std_im", floor_im)):
        obs = _observed(grids[name])
        n_zero = int((obs == 0).sum())
        pos = obs[obs > 0]
        if pos.size == 0:
            _fail("ALL_ZERO_RAW_STD",
                  f"{name}: all {n_zero} observed values are zero; no "
                  f"distribution to render -- investigate the artefact "
                  f"before any visual inspection")
        # Zeros (legitimate floor hits) are counted/reported separately,
        # never passed into log-x bins.
        span = float(pos.max() / pos.min())
        if span > 10.0:
            bins = np.logspace(np.log10(pos.min()), np.log10(pos.max()),
                               101)
            ax.set_xscale("log")
            mode = f"log-x (range ratio {span:.1f} > 10)"
        else:
            bins = np.linspace(pos.min(), pos.max(), 101)
            mode = f"linear-x (range ratio {span:.1f} <= 10)"
        ax.hist(pos, bins=bins, color="0.35", edgecolor="none",
                label=f"positive: {pos.size}")
        ax.axvline(floor, color="red", linestyle="--", linewidth=1.4,
                   label=f"frozen floor = {floor:.3e}")
        if n_zero > 0:
            ax.text(0.02, 0.95,
                    f"observed zeros (floor hits): {n_zero}",
                    transform=ax.transAxes, fontsize=9,
                    color="darkmagenta", va="top")
        ax.set_title(f"{name} distribution ({mode})", fontsize=10)
        ax.set_xlabel(name)
        ax.set_ylabel("locations")
        ax.legend(fontsize=9)
    fig.suptitle(f"{SCRIPT_ID} {SCRIPT_VERSION} -- raw_std distributions "
                 f"vs FROZEN floors | artefact {sha[:12]}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out_dir, f"sigma_distribution_{sha[:12]}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("%s distribution figure written: %s", SCRIPT_ID, path)
    return path


def console_summary(payload: dict, sha: str, grids: dict,
                    margin_info: dict) -> None:
    """Quantitative summary; quantiles so a single unusual pixel cannot
    masquerade as the distribution."""
    pop, br = payload["population"], payload["branch"]
    inh, chans = payload["inheritance"], payload["channels"]
    parent = req(payload, "parents", "p4_s1")
    lines = [
        f"artefact file sha256     {sha}",
        f"semantic_sha256          {payload['semantic_sha256']}",
        f"schema                   {payload['schema']}",
        f"run_mode                 {payload['run_mode']}",
        f"authoritative            {payload['authoritative']}",
        f"parent file sha256       {parent['file_sha256']}",
        f"parent semantic sha256   {parent['semantic_sha256']}",
        f"rows                     {pop['n_slices']}",
        f"files                    {pop['n_files']}",
        f"eligible columns         {inh['n_eligible_columns']}",
        f"eligible locations       {inh['n_eligible_locations']}",
        f"branch selected          {br['selected']}",
        f"smoke_scale              {br['smoke_scale']}",
        f"n_floor_hits             {br['n_floor_hits']}",
        f"20*n_floor_hits          {br['lhs_20x_hits']}",
        f"n_eligible               {br['rhs_n_eligible']}",
        f"sigma_global_re          {chans['re']['sigma_global']:.10e}",
        f"sigma_global_im          {chans['im']['sigma_global']:.10e}",
        f"floor_re                 {chans['re']['floor']:.10e}",
        f"floor_im                 {chans['im']['floor']:.10e}",
    ]
    for name in ("raw_std_re", "raw_std_im"):
        obs = _observed(grids[name])
        q = np.percentile(obs, (1, 5, 50, 95, 99))
        lines.append(
            f"{name:24s} min {obs.min():.6e}  p01 {q[0]:.6e}  "
            f"p05 {q[1]:.6e}  median {q[2]:.6e}  p95 {q[3]:.6e}  "
            f"p99 {q[4]:.6e}  max {obs.max():.6e}")
        lines.append(
            f"{name:24s} observed zeros (legitimate STRICT-< floor "
            f"hits): {int((obs == 0).sum())}")
    lines.append(
        f"floor_hit==True locs     {int(_observed(grids['floor_hit']).sum())}"
        f" (artefact record; EITHER channel, STRICT-<)")
    ml = margin_info["min_location"]
    lines += [
        f"min raw_std/floor ratio  {margin_info['min_ratio']:.4f} at "
        f"(r={ml['row']}, c={ml['column']}, ch={ml['channel']})",
        f"locations below floor    {margin_info['n_below_1']}",
    ]
    counts = _observed(grids["count"])
    lines.append(
        f"count                    min {int(counts.min())}  median "
        f"{int(np.median(counts))}  max {int(counts.max())}")
    c7 = payload["c7"]
    lines += [
        f"C7 max_rel_err           {c7['max_rel_err']:.6e}",
        f"C7 tolerance             {c7['tolerance']:.1e}",
    ]
    text = "\n".join(lines)
    print(f"\n{'=' * 78}\n{SCRIPT_ID} {SCRIPT_VERSION} diagnostic summary\n"
          f"{'=' * 78}\n{text}\n{'=' * 78}")
    logger.info("%s summary emitted (%d lines)", SCRIPT_ID, len(lines))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=f"{SCRIPT_ID} {SCRIPT_VERSION} -- read-only "
                    f"diagnostic rendering of a /2 statistics artefact")
    ap.add_argument("--p4-stats2", required=True,
                    help="path to a schema seqref-p4-stats/2 artefact")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args(argv)

    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s "
                               "%(message)s", handlers=handlers,
                        force=True)
    try:
        payload, sha = load_artefact(args.p4_stats2)
        os.makedirs(args.out_dir, exist_ok=True)
        grids = assemble_grids(payload)
        floor_re = float(req(payload, "channels", "re", "floor"))
        floor_im = float(req(payload, "channels", "im", "floor"))
        margin, margin_info = floor_margins(grids, floor_re, floor_im)
        map_path = render_map_figure(payload, sha, grids, margin,
                                     margin_info, args.out_dir)
        dist_path = render_distribution_figure(payload, sha, grids,
                                               floor_re, floor_im,
                                               args.out_dir)
        console_summary(payload, sha, grids, margin_info)
        logger.info("%s done: %s , %s", SCRIPT_ID, map_path, dist_path)
        return EXIT_OK
    except DiagnosticError as exc:
        logger.error("%s ERROR: %s", SCRIPT_ID, exc)
        return EXIT_ERROR
    except Exception as exc:  # no silent pass: unexpected is still logged
        logger.exception("%s unexpected ERROR: %s", SCRIPT_ID, exc)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())

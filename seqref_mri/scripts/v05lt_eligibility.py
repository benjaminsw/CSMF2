# SEQREF-MRI-V05LTELIG v0.2 -- scripts.v05lt_eligibility
# LIFETIME: KEEP
# =============================================================================
# Purpose: NMSE_u eligibility (194 AM2-C; 195 §6, PA1/PA2) for the 3,461
#   V05LT validation slices. Data only -- no model.
#   * u_true: production definition, computed exactly as TINY does it
#     (tiny_gate._build_slice_states): train_base._prepare (D2 / file attr
#     max) -> dx = complex(x_norm) - complex(cond_in) -> fastmri_data.fft2c
#     -> residual_decoder.gather_unmeasured(cmap from the evaluation mask)
#     -> complex128; E = sum(abs(u_true)**2); ratio = E / (S_ref ** 2).
#   * Rule: eligible iff ratio > 1e-10 (equality ineligible). Applies to
#     NMSE_u ONLY; PSNR, SSIM and NLL keep every slice (TINY's broader
#     exclusion of all metrics is NOT carried over).
#   * Aggregation helpers: NMSE per volume = mean over eligible slices;
#     volume with none -> None (never 0) and outside the NMSE population.
#   * Manifest: one explicit row per slice (volume ID, relpath, slice index,
#     energy, ratio, eligible, midpoint, case) + provenance (split/case
#     manifest, P0S hashes, S_ref, mask convention, implementation
#     identity) + summaries; canonical JSON written once (no overwrite),
#     sha256 sidecar, read-only.
# Generation of the real manifest is a registered pre-training step
#   (195 PA2 gate 2a), never part of a fixture run.
# CONVENTION: logger.error + V05LTError(code); exit 0 PASS / 2 failure.
#   No fallback, no mock, no placeholder, no silent pass.
# Changelog (v0.2, 2026-10-21; I1.6(a), open item 2):
#   * Explicit zero-energy handling: E == 0 exactly is a valid, ineligible
#     validation slice (ratio 0 <= threshold), flagged per row
#     (`zero_energy`) and counted per scope; never an error and never
#     dropped from PSNR/SSIM/NLL. (Holdout: exact zero = validity error,
#     194 AM4-B -- a different path, I1.7/F23.)
# Changelog (NEW in v0.1, 2026-10-21):
#   * Introduced under I1.4 (SEQREF-MRI-V05LTI14 v0.2, core + A): per-slice
#     records, provenance pins, NMSE-only scope, TINY cross-check in F19.
# =============================================================================
from __future__ import annotations

import argparse
import logging
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from seqref_mri.scripts import v05lt_common as C
from seqref_mri.scripts.v02_manifests import midpoint_slice  # reused unchanged

logger = logging.getLogger("seqref_mri.v05lt_eligibility")

__version__ = "0.2"
__abbr__ = "SEQREF-MRI-V05LTELIG"

SCHEMA = "seqref-v05lt-nmse-eligibility/1"
R_FREE_MIN = 1e-10                    # 194 AM2-C (= TINY R_FREE_MIN)
OUT_NAME = "v05lt_nmse_eligibility.json"
BATCH = 32
U_TRUE_DEFINITION = (
    "train_base._prepare(batch, 'cpu', test0=False); dx = complex(x_norm) - "
    "complex(cond_in); u_true = residual_decoder.gather_unmeasured("
    "fastmri_data.fft2c(dx), build_coordinate_map(acquired columns)) as "
    "complex128; E = sum(abs(u_true)**2); ratio = E / (S_ref ** 2)")
MASK_CONVENTION = ("evaluation: canonical_mask_seed(EVAL_BASE_SEED=20261001, "
                   "relpath, slice_index), no epoch (production eval-mode "
                   "FastMRISliceDataset over split='train')")


def _prod():
    """Production modules (legacy residual_decoder name via the TDIAG
    bootstrap, i.e. the same module object TINY uses)."""
    from seqref_mri.tdiag import _bootstrap  # noqa: F401 (sys.path setup)
    import residual_decoder as dec
    from seqref_mri.src import fastmri_data as fdm
    from seqref_mri.src import free_flow_runtime as ffr
    from seqref_mri.scripts.train_base import _collate, _prepare
    return dec, fdm, ffr, _collate, _prepare


def is_eligible(ratio: float) -> bool:
    if not (isinstance(ratio, float) and math.isfinite(ratio) and ratio >= 0):
        C.fail("ELIG_RATIO_INVALID", repr(ratio))
    return ratio > R_FREE_MIN


def slice_energy(x_norm_j, cond_in_j, mask_j) -> tuple:
    """-> (E, n_free_complex) for one slice; arithmetic as TINY."""
    import torch
    dec, fdm, ffr, _, _ = _prod()
    acq = np.flatnonzero(mask_j.to(torch.bool).cpu().numpy())
    cmap = dec.build_coordinate_map([int(c) for c in acq], ffr.GRID_H,
                                    ffr.GRID_W)
    n_free = int(cmap.free_rows.shape[0])
    C.require(2 * n_free == ffr.FLOW_DIM_REAL, "ELIG_MAP_DIM",
              f"2 * {n_free} != {ffr.FLOW_DIM_REAL}")
    dx = torch.complex(x_norm_j[0], x_norm_j[1]) - torch.complex(
        cond_in_j[0], cond_in_j[1])
    u_true = np.asarray(dec.gather_unmeasured(fdm.fft2c(dx).unsqueeze(0), cmap)
                        .detach().to(torch.complex128).cpu().numpy())[0]
    energy = float(np.sum(np.abs(u_true) ** 2))
    C.require(math.isfinite(energy), "ELIG_ENERGY_NON_FINITE", repr(energy))
    return energy, n_free


def compute_records(view, s_ref: float, indices=None) -> list:
    """Per-slice rows in view order (or for the given view indices)."""
    from torch.utils.data import DataLoader
    _, _, _, collate, prepare = _prod()
    idx = list(range(len(view))) if indices is None else list(indices)

    class _Sub:
        def __len__(self):
            return len(idx)

        def __getitem__(self, i):
            return view[idx[i]]
    loader = DataLoader(_Sub(), batch_size=BATCH, shuffle=False,
                        num_workers=0, collate_fn=collate)
    s2 = s_ref ** 2
    rows, k = [], 0
    for batch in loader:
        prep = prepare(batch, "cpu", test0=False)
        for j, meta in enumerate(batch["meta"]):
            rel, s = view.entry(idx[k])
            C.require(meta["file"] == rel and int(meta["slice_index"]) == s,
                      "ELIG_ORDER_MISMATCH", f"{meta['file']}|"
                      f"{meta['slice_index']} != {rel}|{s}")
            e, _ = slice_energy(prep["x_norm"][j], prep["cond_in"][j],
                                batch["mask"][j])
            ratio = e / s2
            if e == 0.0:
                logger.info("[%s] ELIG_ZERO_ENERGY %s|%d: valid slice, "
                            "ineligible for NMSE_u", __abbr__, rel, s)
            rows.append({"volume_id": Path(rel).name, "file": rel,
                         "slice_index": s, "energy": e, "ratio": ratio,
                         "eligible": is_eligible(ratio),
                         "zero_energy": e == 0.0})
            k += 1
    C.require(k == len(idx), "ELIG_COUNT", f"{k} != {len(idx)}")
    return rows


def volume_means(values: dict, entries: list, eligible=None) -> dict:
    """values: {(file, slice): float}; entries: [(volume_id, file, slice)].
    Without `eligible` (PSNR, SSIM, NLL) every slice counts; with it
    (NMSE_u) only eligible slices count and a volume with none -> None."""
    acc = {}
    for vid, f, s in entries:
        acc.setdefault(vid, [])
        if eligible is None or eligible[(f, s)]:
            v = values[(f, s)]
            C.require(isinstance(v, float) and math.isfinite(v),
                      "METRIC_NON_FINITE", f"{f}|{s}: {v!r}")
            acc[vid].append(v)
    return {vid: (float(np.mean(v)) if v else None) for vid, v in acc.items()}


def nmse_population(volume_values: dict) -> list:
    """Fixed NMSE population: volumes with a value (None excluded)."""
    return sorted(v for v, x in volume_values.items() if x is not None)


def build_manifest(rows: list, split_doc: dict, case_doc: dict,
                   provenance: dict) -> dict:
    nslices = {r["volume_id"]: r["n_slices"] for r in split_doc["validation"]}
    cases = {(c["file"], c["slice_index"]): c["case"]
             for c in case_doc["cases"]}
    out = []
    for r in rows:
        mid = r["slice_index"] == midpoint_slice(nslices[r["volume_id"]])
        out.append({**r, "midpoint": mid,
                    "case": cases.get((r["file"], r["slice_index"]))})
    C.require(len({(r["file"], r["slice_index"]) for r in out}) == len(out),
              "ELIG_DUPLICATE_SLICE", "")

    def summary(sel):
        ex = [f"{r['file']}|{r['slice_index']}" for r in sel
              if not r["eligible"]]
        vols = {}
        for r in sel:
            vols.setdefault(r["volume_id"], []).append(r["eligible"])
        null = sorted(v for v, el in vols.items() if not any(el))
        zero = [f"{r['file']}|{r['slice_index']}" for r in sel
                if r.get("zero_energy")]
        return {"slices": len(sel), "eligible": len(sel) - len(ex),
                "excluded": len(ex), "excluded_identities": ex,
                "zero_energy": len(zero), "zero_energy_identities": zero,
                "volumes": len(vols), "volumes_without_eligible_slice": null,
                "nmse_population_volumes": len(vols) - len(null)}
    return {"schema": SCHEMA, "provenance": provenance, "rule": {
        "threshold": R_FREE_MIN, "eligible_iff": "ratio > threshold "
        "(equality ineligible)", "scope": "NMSE_u on validation only; PSNR, "
        "SSIM and NLL keep every slice; holdout keeps V02E rule (194 AM4-B)",
        "u_true": U_TRUE_DEFINITION, "mask_convention": MASK_CONVENTION},
        "summary": {"all_slices": summary(out),
                    "midpoint": summary([r for r in out if r["midpoint"]]),
                    "cases": summary([r for r in out if r["case"] is not None])},
        "slices": out}


def _git_head(repo: Path) -> str:
    r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    C.require(r.returncode == 0, "GIT_FAILED", r.stderr.strip())
    return r.stdout.strip()


def generate(paths: C.Paths, data_root: Path, out: Path) -> str:
    """Registered generation (195 PA2 gate 2a). Returns manifest sha256."""
    C.require(not out.exists() and not Path(str(out) + ".sha256").exists(),
              "ARTIFACT_EXISTS", str(out))
    s_ref = C.verify_p0s(paths.p0s)
    sp = paths.inputs / "v05lt_split_manifest.json"
    cp = paths.inputs / "v05lt_case_manifest.json"
    split = C.load_split_manifest(sp, C.REGISTERED_INPUTS[sp.name])
    import json
    C.verify_file(cp, C.REGISTERED_INPUTS[cp.name], "CASE")
    case = json.loads(cp.read_bytes())
    _, vv = C.build_split_views(data_root, split,
                                C.holdout_ids(paths.v02_manifests))
    rows = compute_records(vv, s_ref)
    prov = {"split_manifest_sha256": C.REGISTERED_INPUTS[sp.name],
            "case_manifest_sha256": C.REGISTERED_INPUTS[cp.name],
            "p0s_path": str(paths.p0s.relative_to(paths.repo)),
            "p0s_sha256": C.P0S_SHA256, "s_ref": s_ref,
            "s_ref_squared": s_ref ** 2,
            "validation_order_sha256": vv.order_lines_sha256(),
            "implementation": {"abbr": __abbr__, "version": __version__,
                               "script_sha256": C.sha256_file(Path(__file__)),
                               "common_sha256": C.sha256_file(Path(C.__file__)),
                               "repo_head": _git_head(paths.repo)}}
    doc = build_manifest(rows, split, case, prov)
    data = C.canonical_json(doc)
    sha = C.write_durable_new(out, data)
    C.write_durable_new(Path(str(out) + ".sha256"),
                        f"{sha}  {out.name}\n".encode())
    for f in (out, Path(str(out) + ".sha256")):
        os.chmod(f, 0o444)
    s = doc["summary"]
    logger.info("[%s] written %s sha256 %s: all %s / midpoint %s / cases %s",
                __abbr__, out, sha, s["all_slices"]["eligible"],
                s["midpoint"]["eligible"], s["cases"]["eligible"])
    return sha


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="V05LT NMSE eligibility (AM2-C)")
    ap.add_argument("mode", choices=["generate"])
    ap.add_argument("--repo", default=str(C.Paths().repo))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    paths = C.Paths(repo=Path(a.repo))
    out = Path(a.out) if a.out else paths.inputs / OUT_NAME
    try:
        sha = generate(paths, Path(a.data_root) if a.data_root
                       else paths.data_root, out)
    except C.V05LTError as exc:
        print(f"RESULT FAIL {exc.code}")
        return 2
    except Exception as exc:  # noqa: BLE001 -- logged, never swallowed
        logger.exception("UNEXPECTED: %s", exc)
        print(f"RESULT FAIL UNEXPECTED {type(exc).__name__}")
        return 2
    print(f"RESULT PASS eligibility_manifest_sha256={sha} path={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

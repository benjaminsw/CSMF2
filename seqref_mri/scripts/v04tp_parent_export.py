# SEQREF-V04TP v0.5 -- v04tp_parent_export
# LIFETIME: KEEP
# =============================================================================
# Purpose: V04TP stage 3 -- export the parent (V03-pin) tensor reference ONCE.
#          Reproduces the data path of v03_probe.py v0.5 (registered source
#          sha256 ac4997e7...412026) exactly: p4 load -> ffr.build_model() ->
#          FastMRISliceDataset(split="train", mode="eval") -> DataLoader
#          (batch 32, sequential, workers 0, _collate) -> first 100 batches
#          -> _prepare(batch, "cpu", test0=False) -> per-slice
#          derive_cmap_from_mask -> standardisation_vectors ->
#          targets_from_prepared. The model is built ONLY to mirror V03P
#          v0.5's call order (RNG-state fidelity); it is never run: no
#          forward, no backward, no optimiser (card SS4).
# Fields exported per case (six tensor fields plus maps; x_norm
#   informational, parent-side only): u_scaled (the V03P `tgt`, NOT
#   x_norm), cond_in, mask, y_raw, amax, tgt_norm, maps=(cmap, vecs) with
#   explicit dataclass attribute trees for CoordinateMap/vectors.
# Batch-list digest uses the PINNED repo canonical_json in V04P v0.3's exact
#   canonical form, comparable to the registered c516ff49...916a.
# Data-file integrity (amendment 3 + reviewer findings 4/5): the dataset's
#   own ordered case index identifies the touched training files BEFORE any
#   tensor loading; all are hashed pre-load and re-hashed after the loop;
#   loader meta is bound to the index case-by-case (TP_SOURCE_MISMATCH);
#   the registry must cover exactly the locked cases. CURRENT integrity
#   checks, not proof of historical data identity.
# Stability: module registry re-walked after the loop (TP_SOURCE_DRIFT);
#   pin/tree re-verified at the end of the attempt.
# Outputs: v04tp_parent_manifest.json (seqref-v04-tensor-parity-parent/1)
#   + sidecar on success; handled failure -> v04tp_parent_status.json +
#   sidecar, no manifest; unhandled failure -> log only (amendment 5).
#   The attempt directory must be EMPTY before anything is created;
#   existing evidence is never touched (reviewer finding 1).
# CONVENTION: logger.error + typed raise (TPError). PASS(0)/ERROR(2).
#   No fallback, no mock, no placeholder, no silent pass.
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys

import v04tp_common as tc

logger = logging.getLogger("seqref_mri.v04tp.parent_export")

__version__ = "0.5"
__abbr__ = "SEQREF-V04TP"

# Frozen map allowlists (card SS5), bound from the pinned-source audit
# (V04 pin 70f35623): residual_decoder.CoordinateMap dataclass declaration
# (10 fields; n_free_complex/n_acquired/flow_dim_real are @property, NOT
# fields) and free_flow_runtime.standardisation_vectors return literal.
MAP_CMAP_FIELDS = frozenset({
    "height", "width", "mask_columns", "free_rows", "free_cols",
    "flatten_order", "packing_order", "conjugate_filled",
    "determined_from_partner", "self_conjugate_real"})
MAP_VECS_KEYS = frozenset({
    "mean_re", "scale_re", "mean_im", "scale_im", "n_free_complex"})

V03_PIN = "3393fbee8f3c2d3e430c81fcd48b909c67d366f8"   # locked (article 169)
N_PROBE_BATCHES = 100                                   # locked (176 SS9)
MANIFEST_SCHEMA = "seqref-v04-tensor-parity-parent/1"
STATUS_SCHEMA = "seqref-v04-tensor-parity-parent-status/1"
CONTRACT_FIELDS = ("u_scaled", "cond_in", "mask", "y_raw", "amax",
                   "tgt_norm", "maps")
INFORMATIONAL_FIELDS = ("x_norm",)
CARD_ID = "SEQREF-MRI-V04TP v0.5"


def _run(v03_root: str, data_root: str, p4_stats2: str, out_dir: str,
         command: str) -> int:
    tc.verify_pin_and_clean_tree(v03_root, V03_PIN)
    root = tc.activate_import_root(v03_root)

    from seqref_mri.src import free_flow_runtime as ffr
    from seqref_mri.src import residual_decoder as dec  # noqa: F401
    from seqref_mri.src.fastmri_data import FastMRISliceDataset
    from seqref_mri.scripts.train_base import _collate, _prepare
    from seqref_mri.scripts.train_free_flow import targets_from_prepared
    from seqref_mri.scripts.v02_manifests import BATCH_SIZE, canonical_json
    from seqref_mri.scripts.v02_train import derive_cmap_from_mask
    import torch
    from torch.utils.data import DataLoader
    modules = tc.collect_modules(root)

    # preflight: the frozen map allowlists must be bound BEFORE model
    # construction or any MRI read (reviewer rev 3)
    if MAP_CMAP_FIELDS is None or MAP_VECS_KEYS is None:
        tc.fail("TP_MAP_CONTRACT_UNBOUND",
                "map allowlists are not bound; freeze them from the "
                "pinned-source audit (card SS5) before any execution")

    p4 = ffr.load_p4s2_parent(p4_stats2)
    logger.info("[%s] P4 /2 parent verified (file %s...)", __abbr__,
                p4["file_sha256"][:12])
    model = ffr.build_model()   # call-order fidelity to V03P v0.5; never run
    model.train(False)

    ds = FastMRISliceDataset(data_root, split="train", mode="eval")
    n_cases_locked = N_PROBE_BATCHES * BATCH_SIZE
    # reviewer finding 4/5: identify + hash touched files BEFORE tensor
    # loading, from the dataset's own ordered index
    case_index = tc.extract_case_index(ds, n_cases_locked, data_root)
    registry = tc.build_file_registry(case_index, data_root)
    file_sha_first = {fp: tc.sha256_file(fp) for fp in registry}
    logger.info("[%s] pre-load hashing complete: %d files", __abbr__,
                len(file_sha_first))

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, collate_fn=_collate)
    if len(loader) < N_PROBE_BATCHES:
        tc.fail("TP_DATA_SHORT",
                f"train split yields {len(loader)} batches; the locked "
                f"procedure requires {N_PROBE_BATCHES}")

    batch_list_hash = hashlib.sha256()
    cases = []
    seen_files = []
    n_cases = 0
    for bi, batch in enumerate(loader):
        if bi >= N_PROBE_BATCHES:
            break
        meta = [{"file": m["file"], "slice_index": m["slice_index"]}
                for m in batch["meta"]]
        batch_list_hash.update(canonical_json({"batch": bi, "meta": meta}))
        # bind the loader's enumeration to the pre-registered index (both
        # sides live in the root-relative POSIX identity space)
        for j, m in enumerate(meta):
            exp_file, exp_slice = case_index[bi * BATCH_SIZE + j]
            if tc.root_relative_identity(m["file"], data_root) != \
                    exp_file or int(m["slice_index"]) != exp_slice:
                tc.fail("TP_SOURCE_MISMATCH",
                        f"batch {bi} case {j}: loader meta "
                        f"({m['file']}, {m['slice_index']}) != dataset "
                        f"index ({exp_file}, {exp_slice})")
            fp = tc.resolve_touched_file(data_root, m["file"])
            if fp not in file_sha_first:
                tc.fail("TP_FILE_REGISTRY_INCOMPLETE",
                        f"batch {bi} case {j} touches {fp}, absent from the "
                        f"pre-load registry")
            seen_files.append(fp)
        prep = _prepare(batch, "cpu", test0=False)
        targets, cond_ins, masks, maps = [], [], [], []
        for j in range(len(meta)):
            cmap = derive_cmap_from_mask(batch["mask"][j])
            vecs = ffr.standardisation_vectors(cmap, p4["location_index"])
            one = {"x_norm": prep["x_norm"][j:j + 1],
                   "cond_in": prep["cond_in"][j:j + 1]}
            targets.append(targets_from_prepared(one, cmap, vecs))
            cond_ins.append(prep["cond_in"][j:j + 1])
            masks.append(batch["mask"][j:j + 1])
            maps.append((cmap, vecs))
        u_scaled = torch.cat(targets, 0)
        cond_in = torch.cat(cond_ins, 0)
        msk = torch.cat(masks, 0)
        n_j = len(meta)
        for name, val in (("u_scaled", u_scaled), ("cond_in", cond_in),
                          ("mask", msk), ("y_raw", prep["y"]),
                          ("amax", prep["amax"]),
                          ("tgt_norm", prep["tgt_norm"])):
            if int(val.shape[0]) != n_j:
                tc.fail("TP_BATCH_LAYOUT",
                        f"batch {bi} field {name} leading dim "
                        f"{int(val.shape[0])} != batch cases {n_j}")
        if len(maps) != n_j:
            tc.fail("TP_BATCH_LAYOUT",
                    f"batch {bi} maps length {len(maps)} != {n_j}")
        for j in range(n_j):
            ctx = f"batch {bi} case {j} ({meta[j]['file']}#" \
                  f"{meta[j]['slice_index']})"
            # frozen map-structure contract (allowlists bound, card SS5)
            tc.assert_map_contract(maps[j], MAP_CMAP_FIELDS, MAP_VECS_KEYS,
                                   ctx)
            fields = {}
            for name, val in (("u_scaled", u_scaled[j]),
                              ("cond_in", cond_in[j]), ("mask", msk[j]),
                              ("y_raw", prep["y"][j]),
                              ("amax", prep["amax"][j]),
                              ("tgt_norm", prep["tgt_norm"][j]),
                              ("maps", maps[j])):
                node = tc.encode_value(val, f"{ctx} {name}")
                fields[name] = tc.field_descriptor(
                    bi, j, meta[j]["file"], meta[j]["slice_index"], name,
                    node)
                if name == "maps":
                    fields[name]["subtree"] = node
            xnode = tc.encode_value(prep["x_norm"][j], f"{ctx} x_norm")
            informational = {"x_norm": tc.field_descriptor(
                bi, j, meta[j]["file"], meta[j]["slice_index"], "x_norm",
                xnode)}
            cases.append({"batch": bi, "index": j,
                          "file": meta[j]["file"],
                          "slice_index": meta[j]["slice_index"],
                          "fields": fields, "informational": informational})
            n_cases += 1
        if (bi + 1) % 10 == 0:
            logger.info("[%s] exported batch %d/%d", __abbr__, bi + 1,
                        N_PROBE_BATCHES)
    del model  # never used; built only for call-order fidelity

    # exact registry coverage (reviewer finding 5)
    tc.validate_registry_coverage(registry, [c["file"] for c in cases],
                                  data_root)
    if set(seen_files) != set(registry):
        tc.fail("TP_FILE_REGISTRY_INCOMPLETE",
                "files seen during loading do not equal the registry")

    # post-loop re-hash (within-run stability, amendment 3)
    file_sha_end = {}
    for fp in registry:
        file_sha_end[fp] = tc.sha256_file(fp)
        if file_sha_end[fp] != file_sha_first[fp]:
            tc.fail("TP_INPUT_DRIFT",
                    f"touched file changed during export: {fp} "
                    f"({file_sha_first[fp][:12]}... -> "
                    f"{file_sha_end[fp][:12]}...)")

    # reviewer finding 8: late-import capture + source/tree stability
    modules = tc.refresh_modules(modules, root, "after export loop")
    tc.verify_pin_and_clean_tree(root, V03_PIN)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "script": f"{__abbr__} v{__version__} v04tp_parent_export",
        "card": CARD_ID,
        "pins": {"v03_pin": V03_PIN, "v03_head_verified": tc.git_head(root)},
        "modules": modules,
        "inputs": {"data_root": os.path.realpath(data_root),
                   "p4_stats2": {"path": p4["path"],
                                 "file_sha256": p4["file_sha256"],
                                 "semantic_sha256": p4["semantic_sha256"]}},
        "n_batches": N_PROBE_BATCHES, "batch_size": BATCH_SIZE,
        "n_cases": n_cases,
        "batch_list_sha256": batch_list_hash.hexdigest(),
        "contract_fields": list(CONTRACT_FIELDS),
        "informational_fields": list(INFORMATIONAL_FIELDS),
        "cases": cases,
        "data_files": {fp: {"sha_first": file_sha_first[fp],
                            "sha_end": file_sha_end[fp]}
                       for fp in registry},
        "data_integrity_note": "current-run integrity checks; not proof of "
                               "historical data identity (amendment 3)",
        "environment": tc.environment(),
        "command": command,
        "harness": {"v04tp_common_sha256": tc.sha256_file(
            os.path.abspath(tc.__file__)),
            "script_sha256": tc.sha256_file(os.path.abspath(__file__))},
    }
    manifest_path = os.path.join(out_dir, "v04tp_parent_manifest.json")
    digest = tc.write_json_with_sidecar(manifest_path, manifest)
    print(f"PASS {__abbr__} v{__version__} parent export: batches="
          f"{N_PROBE_BATCHES} cases={n_cases} files={len(registry)} "
          f"batch_list={manifest['batch_list_sha256'][:12]}... "
          f"manifest={manifest_path} sha256={digest[:12]}...")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="v04tp_parent_export")
    ap.add_argument("--v03-root", required=True,
                    help="isolated V03-pin checkout (git worktree)")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--p4-stats2", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    # reviewer finding 1: refuse a non-empty attempt directory BEFORE any
    # log or record is created; previous artifacts remain untouched
    if not tc.attempt_dir_is_fresh(args.out):
        print(f"ERROR {__abbr__} TP_OUTPUT_EXISTS: attempt directory "
              f"{args.out} is not empty; refusing to touch existing "
              f"evidence", file=sys.stderr)
        return 2
    command = f"{sys.executable} " + " ".join(sys.argv)
    os.makedirs(args.out, exist_ok=True)
    tc.setup_logging(os.path.join(args.out, "v04tp_parent_export.log"))
    try:
        return _run(args.v03_root, args.data_root, args.p4_stats2, args.out,
                    command)
    except tc.TPError as exc:
        logger.error("[%s] ERROR: %s", __abbr__, exc)
        # amendment 5: handled outcome -> status record written; no manifest
        status = {"schema": STATUS_SCHEMA,
                  "script": f"{__abbr__} v{__version__} v04tp_parent_export",
                  "failure": {"code": exc.code, "message": str(exc)},
                  "environment": tc.environment(), "command": command}
        try:
            tc.write_json_with_sidecar(
                os.path.join(args.out, "v04tp_parent_status.json"), status)
        except Exception:
            logger.exception("[%s] could not write status record; log is "
                             "the preserved evidence", __abbr__)
        return 2
    except Exception:
        logger.exception("[%s] unexpected exception; log preserved, no "
                         "record written (amendment 5)", __abbr__)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Changelog
# v0.5 (2026-10-13): map allowlists BOUND from the pinned-source audit --
#   CoordinateMap 10 declared dataclass fields (properties excluded) and
#   the 5 standardisation-vector keys; per-case contract call passes both.
# v0.4 (2026-10-13): reviewer rev-3 items -- unbound-allowlist check moved
#   to preflight (before model construction or any MRI read; per-case
#   contract checks remain); CARD_ID set to the final approved card
#   revision (v0.5). Behaviour otherwise unchanged.
# v0.3 (2026-10-13): reviewer NO-GO rev-2 corrections -- case-index
#   extraction and loader binding moved to the root-relative POSIX identity
#   space (Path entries, absolute or relative); frozen map allowlist
#   contract wired per case (MAP_CMAP_FIELDS unbound = fail-closed
#   TP_MAP_CONTRACT_UNBOUND pending the pinned-source audit).
# v0.2 (2026-10-13): reviewer NO-GO corrections -- attempt freshness;
#   pre-load registry hashing with index binding and exact coverage;
#   module refresh + end-of-attempt re-verification; explicit log levels.
# v0.1 (2026-10-13): initial implementation (superseded before execution).
# ---------------------------------------------------------------------------

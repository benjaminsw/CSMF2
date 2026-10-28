# SEQREF-V04TP v0.5 -- v04tp_compare
# LIFETIME: KEEP
# =============================================================================
# Purpose: V04TP stage 4 -- independently prepare the candidate (V04-pin)
# cases through the VERIFIED production entry point v04_probe
# .prepare_probe_batches (V04P v0.3, registered source sha256
# 722d9c06...20cd42) and compare every required case/field byte-exactly
# against the stage-3 parent manifest. The parent manifest supplies
# expected DESCRIPTORS only (dtype/shape/C-order SHA-256), never candidate
# tensors and never retained raw parent bytes (card SS6/SS8).
# Reference provenance is validated against the approved bindings BEFORE
# the bundle is accepted (reviewer finding 6): parent pin and verified
# HEAD, producer/common harness hashes, schema, card id, counts, contract
# field list, exact file-registry coverage.
# Checks (any failure -> typed code, record written for handled outcomes,
#   exit 2; unhandled failure -> log preserved, no record -- amendment 5):
#   pin/tree fail-closed at start AND end; import-root isolation with
#   dependency hashing refreshed after preparation (TP_SOURCE_DRIFT);
#   shared-seven cross-side equality; probe-record identity (registered
#   e32b8fcd...602bd) and schema; batch-list three-way equality (parent /
#   candidate live / registered c516ff49...916a) with the manifest's
#   ordered metadata re-hashed to bind it to the digest; candidate file
#   registry derived from the candidate's OWN dataset index before any
#   candidate tensor loading, hashed before/after, set-equal to the parent
#   registry and byte-equal across runs (amendment 3 / findings 4/5);
#   six tensor fields plus maps exact match (x_norm parent-side
#   informational only). PASS requires zero mismatches AND every integrity
#   check green. The attempt directory must be EMPTY before anything is
#   created; existing evidence is never touched (finding 1).
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

logger = logging.getLogger("seqref_mri.v04tp.compare")

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
V04_PIN = "70f35623ee4ca08b942b9204775415fe972b82df"   # locked (178 SS6.2)
N_PROBE_BATCHES = 100                                   # locked (176 SS9)
RECORD_SCHEMA = "seqref-v04-tensor-parity-record/1"
CARD_ID = "SEQREF-MRI-V04TP v0.5"
EXPECTED_BATCH_LIST_SHA256 = ("c516ff49dc1940b659e4d8cfa12faeece676b591fbdf"
                              "40c13b15053bb579916a")   # registered (179)
PROBE_RECORD_SHA256 = ("e32b8fcd80e690929f236bee4c2f77b7d95fd131544575f4294e"
                       "16ef180602bd")                  # registered (179)
EXPECTED_COMMON_SHA256 = ("25c457226dae52d3b7ab167f83f06cc81bcb2907c2a44c"
                          "62eb3b40c6255fb798")          # v04tp_common v0.4
EXPECTED_PARENT_EXPORT_SHA256 = ("f32eb9241434f13c54d15e3a1f42e75d1abfc881"
                                 "a9bc534ed312fef87a13af69")  # parent v0.5
CONTRACT_FIELDS = ("u_scaled", "cond_in", "mask", "y_raw", "amax",
                   "tgt_norm", "maps")
SHARED_SEVEN = ("seqref_mri/src/fastmri_data.py",
                "seqref_mri/scripts/train_base.py",
                "seqref_mri/scripts/train_free_flow.py",
                "seqref_mri/scripts/v02_train.py",
                "seqref_mri/src/free_flow_runtime.py",
                "seqref_mri/scripts/v02_manifests.py",
                "seqref_mri/src/residual_decoder.py")


def _validate_reference(manifest: dict) -> None:
    """Reviewer finding 6: every declared provenance value is checked
    against the approved bindings before the bundle is accepted."""
    if manifest.get("schema") != "seqref-v04-tensor-parity-parent/1":
        tc.fail("TP_REFERENCE_DRIFT",
                f"parent manifest schema {manifest.get('schema')!r}")
    if manifest.get("card") != CARD_ID:
        tc.fail("TP_REFERENCE_DRIFT",
                f"parent manifest card {manifest.get('card')!r} != {CARD_ID!r}")
    pins = manifest.get("pins", {})
    if pins.get("v03_pin") != V03_PIN or \
            pins.get("v03_head_verified") != V03_PIN:
        tc.fail("TP_REFERENCE_DRIFT",
                f"parent manifest pins {pins} do not match the approved "
                f"V03 pin {V03_PIN}")
    hw = manifest.get("harness", {})
    if hw.get("v04tp_common_sha256") != EXPECTED_COMMON_SHA256:
        tc.fail("TP_REFERENCE_DRIFT",
                "parent export ran with an unapproved v04tp_common "
                f"({hw.get('v04tp_common_sha256')})")
    if hw.get("script_sha256") != EXPECTED_PARENT_EXPORT_SHA256:
        tc.fail("TP_REFERENCE_DRIFT",
                "parent export script hash "
                f"{hw.get('script_sha256')} != approved "
                f"{EXPECTED_PARENT_EXPORT_SHA256}")
    if tc.sha256_file(os.path.abspath(tc.__file__)) != \
            EXPECTED_COMMON_SHA256:
        tc.fail("TP_REFERENCE_DRIFT",
                "the RUNNING v04tp_common does not match the approved hash")
    if manifest.get("n_batches") != N_PROBE_BATCHES or \
            manifest.get("batch_size") != 32 or \
            manifest.get("n_cases") != N_PROBE_BATCHES * 32:
        counts = (manifest.get("n_batches"), manifest.get("batch_size"),
                  manifest.get("n_cases"))
        tc.fail("TP_REFERENCE_DRIFT",
                f"parent manifest counts {counts} != (100, 32, 3200)")
    if manifest.get("contract_fields") != list(CONTRACT_FIELDS):
        tc.fail("TP_REFERENCE_DRIFT",
                f"parent manifest contract fields "
                f"{manifest.get('contract_fields')}")


def _first_mismatch(mismatches: list, first: dict, batch: int, index: int,
                    file: str, slice_index: int, field: str, path: str,
                    code: str, detail: str) -> None:
    mismatches.append(field)
    if not first:
        first.update({"batch": batch, "index": index, "file": file,
                      "slice_index": slice_index, "field": field,
                      "path": path, "code": code, "detail": detail})


def _run(v04_repo: str, data_root: str, p4_stats2: str, parent_manifest: str,
         probe_record: str, out_dir: str, command: str) -> int:
    tc.verify_pin_and_clean_tree(v04_repo, V04_PIN)
    root = tc.activate_import_root(v04_repo)

    from seqref_mri.src import residual_decoder as dec  # noqa: F401
    from seqref_mri.src.fastmri_data import FastMRISliceDataset
    from seqref_mri.scripts import v04_probe as v04p
    from seqref_mri.scripts.v02_manifests import canonical_json
    modules_c = tc.collect_modules(root)

    # preflight: the frozen map allowlists must be bound BEFORE any MRI
    # read (reviewer rev 3)
    if MAP_CMAP_FIELDS is None or MAP_VECS_KEYS is None:
        tc.fail("TP_MAP_CONTRACT_UNBOUND",
                "map allowlists are not bound; freeze them from the "
                "pinned-source audit (card SS5) before any execution")

    manifest, manifest_sha = tc.load_json_verified(parent_manifest)
    _validate_reference(manifest)
    modules_p = manifest["modules"]
    shared_checked, shared_bad = [], []
    for rel in SHARED_SEVEN:
        if rel in modules_p and rel in modules_c:
            shared_checked.append(rel)
            if modules_p[rel] != modules_c[rel]:
                shared_bad.append(rel)
    if len(shared_checked) != len(SHARED_SEVEN):
        tc.fail("TP_DEPENDENCY_MISMATCH",
                f"shared-seven runtime coverage incomplete: resolved "
                f"{len(shared_checked)}/{len(SHARED_SEVEN)} on both sides")
    if shared_bad:
        tc.fail("TP_DEPENDENCY_MISMATCH",
                f"shared preprocessing modules differ across pins at "
                f"runtime: {shared_bad}")
    probe, probe_sha = tc.load_json_verified(probe_record)
    if probe.get("schema") != "seqref-v04-probe-record/1":
        tc.fail("TP_REFERENCE_DRIFT",
                f"probe record schema {probe.get('schema')!r}")
    if probe_sha != PROBE_RECORD_SHA256:
        tc.fail("TP_REFERENCE_DRIFT",
                f"probe record sha256 {probe_sha} != registered "
                f"{PROBE_RECORD_SHA256}")
    registered_batch_list = probe.get("batch_list_sha256")
    if registered_batch_list != EXPECTED_BATCH_LIST_SHA256:
        tc.fail("TP_REFERENCE_DRIFT",
                f"probe record batch_list_sha256 {registered_batch_list} != "
                f"registered {EXPECTED_BATCH_LIST_SHA256}")
    if manifest["batch_list_sha256"] != EXPECTED_BATCH_LIST_SHA256:
        tc.fail("TP_REFERENCE_DRIFT",
                f"parent manifest batch-list {manifest['batch_list_sha256']} "
                f"!= registered {EXPECTED_BATCH_LIST_SHA256}")
    # bind the manifest's ordered metadata list to the digest: validate
    # flat case order, regroup per batch, re-hash in V04P's canonical form
    cases_p = manifest["cases"]
    h = hashlib.sha256()
    for bi in range(N_PROBE_BATCHES):
        meta = []
        for j, c in enumerate(cases_p[bi * 32:(bi + 1) * 32]):
            if c["batch"] != bi or c["index"] != j:
                tc.fail("TP_CASE_MISMATCH",
                        f"parent manifest case order broken at batch {bi} "
                        f"position {j}")
            meta.append({"file": c["file"],
                         "slice_index": c["slice_index"]})
        h.update(canonical_json({"batch": bi, "meta": meta}))
    if h.hexdigest() != EXPECTED_BATCH_LIST_SHA256:
        tc.fail("TP_CASE_MISMATCH",
                "parent manifest ordered metadata does not re-hash to the "
                "registered batch-list digest")
    # reviewer finding 5: the reference registry must cover exactly the
    # locked cases -- no missing, no extra, none empty
    data_files_p = manifest["data_files"]
    tc.validate_registry_coverage(
        list(data_files_p), [c["file"] for c in cases_p], data_root)

    # reviewer findings 4/5 (candidate side): derive the file registry from
    # the candidate's OWN dataset index and hash BEFORE any candidate
    # tensor loading (prepare_probe_batches is called only afterwards)
    ds_probe = FastMRISliceDataset(data_root, split="train", mode="eval")
    case_index_c = tc.extract_case_index(ds_probe, N_PROBE_BATCHES * 32,
                                         data_root)
    del ds_probe  # index-scan instance only; no tensor loading
    manifest_meta = [(tc.root_relative_identity(c["file"], data_root),
                      int(c["slice_index"])) for c in cases_p]
    if case_index_c != manifest_meta:
        tc.fail("TP_SOURCE_MISMATCH",
                "candidate dataset case index differs from the parent "
                "manifest's ordered metadata")
    registry_c = tc.build_file_registry(case_index_c, data_root)
    if set(registry_c) != set(data_files_p):
        tc.fail("TP_SOURCE_MISMATCH",
                "candidate touched-file set differs from the parent "
                "registry")
    file_sha_first = {fp: tc.sha256_file(fp) for fp in registry_c}
    logger.info("[%s] candidate pre-load hashing complete: %d files",
                __abbr__, len(file_sha_first))

    batches, batch_list_c, p4_c = v04p.prepare_probe_batches(data_root,
                                                             p4_stats2)
    if len(batches) != N_PROBE_BATCHES:
        tc.fail("TP_CASE_MISMATCH",
                f"candidate produced {len(batches)} batches, expected "
                f"{N_PROBE_BATCHES}")
    if batch_list_c != EXPECTED_BATCH_LIST_SHA256:
        tc.fail("TP_CASE_MISMATCH",
                f"candidate live batch-list {batch_list_c} != registered "
                f"{EXPECTED_BATCH_LIST_SHA256}")
    p4_p = manifest["inputs"]["p4_stats2"]
    if (p4_c["file_sha256"] != p4_p["file_sha256"]
            or p4_c["semantic_sha256"] != p4_p["semantic_sha256"]):
        tc.fail("TP_INPUT_DRIFT",
                "P4 statistics identity differs between parent export and "
                "candidate comparison")
    # reviewer finding 8: capture late imports from preparation
    modules_c = tc.refresh_modules(modules_c, root, "after candidate "
                                   "preparation")

    mismatches, first = [], {}
    per_field = {f: 0 for f in CONTRACT_FIELDS}
    observed = 0
    ctx = "pre-comparison"
    try:
        for bi in range(N_PROBE_BATCHES):
            batch = batches[bi]
            n_j = int(batch["u_scaled"].shape[0])
            if n_j != 32:
                tc.fail("TP_CASE_MISMATCH",
                        f"candidate batch {bi} holds {n_j} cases, expected "
                        f"32 (locked batch size)")
            for name in CONTRACT_FIELDS:
                if name not in batch:
                    tc.fail("TP_FIELD_MISSING",
                            f"candidate batch {bi} missing field {name}")
            extra = [k for k in batch if k not in CONTRACT_FIELDS]
            if extra:
                tc.fail("TP_FIELD_SET_MISMATCH",
                        f"candidate batch {bi} carries unexpected fields "
                        f"{extra}")
            for j in range(n_j):
                c = cases_p[bi * 32 + j]
                ctx = f"batch {bi} case {j} ({c['file']}#" \
                      f"{c['slice_index']})"
                if set(c["fields"]) != set(CONTRACT_FIELDS):
                    tc.fail("TP_FIELD_SET_MISMATCH",
                            f"parent manifest {ctx} field set "
                            f"{sorted(c['fields'])}")
                # frozen map-structure contract (allowlists bound, card SS5)
                tc.assert_map_contract(batch["maps"][j], MAP_CMAP_FIELDS,
                                       MAP_VECS_KEYS, ctx)
                for name in CONTRACT_FIELDS:
                    node_c = tc.encode_value(
                        batch[name][j] if name != "maps"
                        else batch["maps"][j], f"{ctx} {name}")
                    desc_p = c["fields"][name]
                    if name == "maps":
                        diffs = []
                        tc.compare_nodes(desc_p["subtree"], node_c,
                                         "maps", diffs)
                        if diffs:
                            path, code, detail = diffs[0]
                            _first_mismatch(mismatches, first, bi, j,
                                            c["file"], c["slice_index"],
                                            name, path, code, detail)
                            per_field[name] += 1
                        continue
                    for key, code in (("dtype", "TP_DTYPE_MISMATCH"),
                                      ("shape", "TP_SHAPE_MISMATCH"),
                                      ("content_sha256",
                                       "TP_TENSOR_MISMATCH")):
                        if desc_p[key] != node_c[key]:
                            _first_mismatch(mismatches, first, bi, j,
                                            c["file"], c["slice_index"],
                                            name, name, code,
                                            f"{key}: {desc_p[key]!r} != "
                                            f"{node_c[key]!r}")
                            per_field[name] += 1
                            break
                # informational x_norm: parent-side provenance only (the
                # pinned candidate entry point does not expose the
                # intermediate); recorded, never compared, never verdict
                observed += 1
            if (bi + 1) % 10 == 0:
                logger.info("[%s] compared batch %d/%d", __abbr__, bi + 1,
                            N_PROBE_BATCHES)
    except tc.TPError as exc:
        if exc.code in ("TP_FIELD_MISSING", "TP_FIELD_SET_MISMATCH",
                        "TP_CASE_MISMATCH"):
            raise
        # comparison-domain finding (e.g. TP_NONFINITE in candidate data):
        # handled outcome -> FAIL record with partial counts
        mismatches.append("aborted")
        first = first or {"code": exc.code, "detail": str(exc)}
        logger.error("[%s] comparison aborted at %s: %s", __abbr__, ctx,
                     exc)

    file_sha_end = {}
    within_stable_c = True
    for fp in registry_c:
        file_sha_end[fp] = tc.sha256_file(fp)
        if file_sha_end[fp] != file_sha_first[fp]:
            within_stable_c = False
            logger.error("[%s] TP_INPUT_DRIFT: touched file changed during "
                         "candidate pass: %s", __abbr__, fp)
    cross_run_equal = within_stable_c
    files_join = {}
    for fp in registry_c:
        p = data_files_p[fp]
        eq = (p["sha_first"] == p["sha_end"] == file_sha_first[fp]
              == file_sha_end[fp])
        cross_run_equal = cross_run_equal and eq
        files_join[fp] = {"parent_first": p["sha_first"],
                          "parent_end": p["sha_end"],
                          "candidate_first": file_sha_first[fp],
                          "candidate_end": file_sha_end[fp],
                          "all_equal": eq}
    if not cross_run_equal:
        logger.error("[%s] TP_INPUT_DRIFT: data-file integrity mismatch "
                     "(within-run or cross-run)", __abbr__)

    # reviewer finding 8: source/tree stability across the whole attempt
    modules_c = tc.refresh_modules(modules_c, root, "before record write")
    tc.verify_pin_and_clean_tree(root, V04_PIN)

    total = len(mismatches)
    passed = total == 0 and cross_run_equal
    if passed:
        code = "TP_MATCH"
    elif total:
        code = first.get("code", "TP_TENSOR_MISMATCH")
    else:
        code = "TP_INPUT_DRIFT"
    record = {
        "schema": RECORD_SCHEMA,
        "script": f"{__abbr__} v{__version__} v04tp_compare",
        "card": CARD_ID,
        "pins": {"v03_pin": V03_PIN, "v04_pin": V04_PIN,
                 "v04_head_verified": tc.git_head(root)},
        "modules": {"parent": modules_p, "candidate": modules_c,
                    "shared_checked": shared_checked, "shared_equal": True},
        "inputs": {"data_root": os.path.realpath(data_root),
                   "p4_stats2": {"parent": p4_p, "candidate": {
                       "path": p4_c["path"],
                       "file_sha256": p4_c["file_sha256"],
                       "semantic_sha256": p4_c["semantic_sha256"]}},
                   "probe_record": {"path": os.path.realpath(probe_record),
                                    "sha256": probe_sha}},
        "batch_list": {"parent": manifest["batch_list_sha256"],
                       "candidate": batch_list_c,
                       "registered": EXPECTED_BATCH_LIST_SHA256,
                       "all_equal": True,
                       "ordered_metadata_bound": True},
        "data_files": {"n_files": len(files_join), "files": files_join,
                       "within_run_stable": {
                           "parent": all(p["sha_first"] == p["sha_end"]
                                         for p in data_files_p.values()),
                           "candidate": within_stable_c},
                       "cross_run_equal": cross_run_equal,
                       "note": "current-run integrity checks; not proof of "
                               "historical data identity (amendment 3)"},
        "cases": {"n_batches": N_PROBE_BATCHES,
                  "batch_size": manifest["batch_size"],
                  "expected_cases": N_PROBE_BATCHES * 32,
                  "observed_cases": observed},
        "fields": {"contract": list(CONTRACT_FIELDS),
                   "informational_parent_only": ["x_norm"]},
        "mismatches": {"total": total, "per_field": per_field,
                       "first": first},
        "verdict": {"pass": passed, "code": code},
        "environment": {"parent": manifest["environment"],
                        "candidate": tc.environment()},
        "artifacts": {"parent_manifest": {
            "path": os.path.realpath(parent_manifest),
            "sha256": manifest_sha}},
        "command": command,
        "harness": {"v04tp_common_sha256": EXPECTED_COMMON_SHA256,
                    "compare_script_sha256": tc.sha256_file(
                        os.path.abspath(__file__)),
                    "parent_script_sha256": EXPECTED_PARENT_EXPORT_SHA256},
        "outcome_policy": "record written for handled outcomes; the log is "
                          "preserved if an early failure prevents writing "
                          "(amendment 5)",
    }
    record_path = os.path.join(out_dir, "v04tp_parity_record.json")
    digest = tc.write_json_with_sidecar(record_path, record)
    if passed:
        print(f"PASS {__abbr__} v{__version__} tensor parity: cases="
              f"{observed} mismatches=0 files={len(files_join)} "
              f"record={record_path} sha256={digest[:12]}...")
        return 0
    print(f"FAIL {__abbr__} v{__version__} tensor parity: code={code} "
          f"mismatches={total} cross_run_equal={cross_run_equal} "
          f"record={record_path} sha256={digest[:12]}...")
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(prog="v04tp_compare")
    ap.add_argument("--v04-repo", required=True,
                    help="V04 repository at the execution pin")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--p4-stats2", required=True)
    ap.add_argument("--parent-manifest", required=True)
    ap.add_argument("--probe-record", required=True)
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
    tc.setup_logging(os.path.join(args.out, "v04tp_compare.log"))
    try:
        return _run(args.v04_repo, args.data_root, args.p4_stats2,
                    args.parent_manifest, args.probe_record, args.out,
                    command)
    except tc.TPError as exc:
        logger.error("[%s] ERROR: %s", __abbr__, exc)
        # amendment 5: handled outcome -> FAIL record written (sections not
        # reached are absent, never null placeholders)
        record = {"schema": RECORD_SCHEMA,
                  "script": f"{__abbr__} v{__version__} v04tp_compare",
                  "card": CARD_ID,
                  "failure": {"code": exc.code, "message": str(exc)},
                  "verdict": {"pass": False, "code": exc.code},
                  "environment": {"candidate": tc.environment()},
                  "command": command}
        try:
            tc.write_json_with_sidecar(
                os.path.join(args.out, "v04tp_parity_record.json"), record)
        except Exception:
            logger.exception("[%s] could not write record; log is the "
                             "preserved evidence", __abbr__)
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
#   the 5 standardisation-vector keys; approved harness hashes updated to
#   common v0.4 / parent v0.5.
# v0.4 (2026-10-13): reviewer rev-3 items -- unbound-allowlist check moved
#   to preflight (before any MRI read); CARD_ID set to the final approved
#   card revision (v0.5); approved parent-export hash updated to v0.4.
# v0.3 (2026-10-13): reviewer NO-GO rev-2 corrections -- candidate case
#   index and manifest metadata compared in the root-relative POSIX
#   identity space; frozen map allowlist contract wired per case (unbound =
#   fail-closed TP_MAP_CONTRACT_UNBOUND); approved harness hashes updated
#   to common/parent v0.3.
# v0.2 (2026-10-13): reviewer NO-GO corrections -- attempt freshness; full
#   reference-provenance validation; candidate pre-load registry from its
#   own dataset index; module refresh + end-of-attempt re-verification;
#   explicit log levels.
# v0.1 (2026-10-13): initial implementation (superseded before execution).
# ---------------------------------------------------------------------------

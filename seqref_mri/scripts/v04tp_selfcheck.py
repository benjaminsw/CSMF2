# SEQREF-V04TP v0.3 -- v04tp_selfcheck
# LIFETIME: DIAGNOSTIC
# =============================================================================
# Purpose: targeted checks for harness v0.3 (run card SEQREF-MRI-V04TP
#          v0.4). Synthetic data only; no MRI pipeline, no gate, no training.
#          v0.2 adds the reviewer-required integration cases: (a) encode ->
#          write -> reload -> compare with a NON-alphabetical dataclass
#          field layout (the rev-1 fixture was alphabetical and blind to
#          the JSON round-trip ordering bug); (b) production-style
#          (Path, slice) index entries, absolute and relative. Also covers
#          the frozen map allowlist contract (bound / mismatch / unbound).
# CONVENTION: each check prints PASS/FAIL; any FAIL -> exit 2.
# =============================================================================
import dataclasses
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v04tp_common as tc

RESULTS = []


def check(name, fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - diagnostic harness
        RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        print(f"FAIL {name}: {exc}")
        return
    RESULTS.append((name, True, ""))
    print(f"PASS {name}")


def expect_tp(code, fn):
    try:
        fn()
    except tc.TPError as exc:
        assert exc.code == code, f"got {exc.code}, expected {code}"
        return
    raise AssertionError(f"no TPError raised, expected {code}")


# --- INTEGRATION A: encode -> write -> reload -> compare, non-alpha layout ---
@dataclasses.dataclass
class FakeCoordinateMap:
    z_center: np.ndarray     # declaration order deliberately NOT
    alpha_extent: np.ndarray # alphabetical: rev-1's fixture was and it
    mid_ndim: int            # hid the JSON round-trip ordering bug


def _fake_maps(seed=0.0):
    cmap = FakeCoordinateMap(
        np.array([1.5 + seed, 2.5], dtype=np.float32),
        np.array([10, 20], dtype=np.int64), 2)
    vecs = {"location": np.array([0.1 + seed], dtype=np.float32),
            "scale": np.array([2.0], dtype=np.float32)}
    return (cmap, vecs)


def t_map_json_roundtrip():
    d = tempfile.mkdtemp()
    node_parent = tc.encode_value(_fake_maps(), "t")
    desc = tc.field_descriptor(0, 0, "f.h5", 3, "maps", node_parent)
    desc["subtree"] = node_parent
    p = os.path.join(d, "manifest.json")
    tc.write_json_with_sidecar(p, {"cases": [{"fields": {"maps": desc}}]})
    loaded, _ = tc.load_json_verified(p)
    reloaded_subtree = loaded["cases"][0]["fields"]["maps"]["subtree"]
    node_candidate = tc.encode_value(_fake_maps(), "t")  # declaration order
    diffs = []
    tc.compare_nodes(reloaded_subtree, node_candidate, "maps", diffs)
    assert not diffs, f"identical maps must survive JSON round-trip: {diffs}"
    # and a real mismatch must still be found, with a precise path
    node_bad = tc.encode_value(_fake_maps(seed=0.25), "t")
    diffs2 = []
    tc.compare_nodes(reloaded_subtree, node_bad, "maps", diffs2)
    assert diffs2 and diffs2[0][1] == "TP_MAP_MISMATCH", diffs2
    assert "z_center" in diffs2[0][0] or "location" in diffs2[0][0], diffs2


# --- INTEGRATION B: production-style (Path, slice) index entries -------------
def t_path_index_identity():
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "train"))
    vol = os.path.join(root, "train", "vol.h5")
    Path(vol).write_bytes(b"data")
    rel_identity = "train/vol.h5"

    class DSAbs:   # absolute Path entries (production form)
        index = [(Path(vol), 3), (Path(vol), 4)]

    class DSRel:   # root-relative Path entries
        index = [(Path(rel_identity), 3), Path(rel_identity) and
                 (Path(rel_identity), 4)]

    class DSDict:  # dict entries with Path values
        index = [{"file": Path(vol), "slice_index": 3},
                 {"file": Path(rel_identity), "slice_index": 4}]

    for ds in (DSAbs(), DSRel(), DSDict()):
        ci = tc.extract_case_index(ds, 2, root)
        assert ci == [(rel_identity, 3), (rel_identity, 4)], ci
    # meta emitted relative by the loader binds against the index identity
    meta = [{"file": rel_identity, "slice_index": 3}]
    exp_file, exp_slice = tc.extract_case_index(DSAbs(), 2, root)[0]
    assert tc.root_relative_identity(meta[0]["file"], root) == exp_file
    # hashing still lands on the resolved absolute path under the root
    assert tc.resolve_touched_file(root, exp_file) == os.path.realpath(vol)
    # escape rejected
    expect_tp("TP_INPUT_MISSING",
              lambda: tc.root_relative_identity("/etc/hostname", root))


# --- map allowlist contract (bound / mismatch / unbound) ----------------------
VEC_KEYS = frozenset({"mean_re", "scale_re", "mean_im", "scale_im",
                      "n_free_complex"})


def t_map_contract():
    allow = frozenset({"z_center", "alpha_extent", "mid_ndim"})
    vecs5 = {"mean_re": np.ones(1), "scale_re": np.ones(1),
             "mean_im": np.ones(1), "scale_im": np.ones(1),
             "n_free_complex": 1}
    cmap, vecs = tc.assert_map_contract((_fake_maps()[0], vecs5), allow,
                                        VEC_KEYS, "t")
    assert cmap.mid_ndim == 2 and "scale_re" in vecs

    @dataclasses.dataclass
    class WrongMap:
        z_center: np.ndarray
        alpha_extent: np.ndarray

    expect_tp("TP_FIELD_SET_MISMATCH",
              lambda: tc.assert_map_contract(
                  (WrongMap(np.zeros(1, dtype=np.float32),
                            np.zeros(1, dtype=np.int64)), vecs5), allow,
                  VEC_KEYS, "t"))
    expect_tp("TP_FIELD_SET_MISMATCH",
              lambda: tc.assert_map_contract(
                  (_fake_maps()[0], {"wrong_key": 1}), allow, VEC_KEYS, "t"))
    expect_tp("TP_MAP_MISMATCH",
              lambda: tc.assert_map_contract(("not-a-dataclass", vecs5),
                                             allow, VEC_KEYS, "t"))
    expect_tp("TP_MAP_MISMATCH",
              lambda: tc.assert_map_contract(("only-one",), allow,
                                             VEC_KEYS, "t"))
    expect_tp("TP_MAP_CONTRACT_UNBOUND",
              lambda: tc.assert_map_contract(_fake_maps(), None, VEC_KEYS,
                                             "t"))
    expect_tp("TP_MAP_CONTRACT_UNBOUND",
              lambda: tc.assert_map_contract(_fake_maps(), allow, None,
                                             "t"))


# --- PRODUCTION layout: CoordinateMap declaration from the audit --------------
@dataclasses.dataclass(frozen=True)
class CoordinateMap:
    """Mirrors the pinned residual_decoder declaration exactly (10 fields;
    n_free_complex/n_acquired/flow_dim_real are properties, NOT fields)."""
    height: int
    width: int
    mask_columns: tuple
    free_rows: np.ndarray
    free_cols: np.ndarray
    flatten_order: str = "rows"
    packing_order: str = "re_im"
    conjugate_filled: tuple = ()
    determined_from_partner: tuple = ()
    self_conjugate_real: tuple = ()

    @property
    def n_free_complex(self):
        return int(self.free_rows.shape[0])


PROD_CMAP_FIELDS = frozenset({
    "height", "width", "mask_columns", "free_rows", "free_cols",
    "flatten_order", "packing_order", "conjugate_filled",
    "determined_from_partner", "self_conjugate_real"})


def _prod_maps(delta=0.0):
    cmap = CoordinateMap(4, 5, (1, 3),
                         np.array([0, 2], dtype=np.int64),
                         np.array([0, 2], dtype=np.int64) + int(delta))
    vecs = {"mean_re": np.zeros(2), "scale_re": np.ones(2),
            "mean_im": np.zeros(2) + delta, "scale_im": np.ones(2),
            "n_free_complex": 2}
    return (cmap, vecs)


def t_production_layout_roundtrip():
    d = tempfile.mkdtemp()
    node = tc.encode_value(_prod_maps(), "t")
    desc = tc.field_descriptor(0, 0, "f.h5", 1, "maps", node)
    desc["subtree"] = node
    p = os.path.join(d, "m.json")
    tc.write_json_with_sidecar(p, {"cases": [{"fields": {"maps": desc}}]})
    loaded, _ = tc.load_json_verified(p)
    reloaded = loaded["cases"][0]["fields"]["maps"]["subtree"]
    # contract accepts the production layout with the frozen allowlists
    tc.assert_map_contract(_prod_maps(), PROD_CMAP_FIELDS, VEC_KEYS, "t")
    # encode -> write -> reload -> compare is clean for identical maps
    diffs = []
    tc.compare_nodes(reloaded, tc.encode_value(_prod_maps(), "t"),
                     "maps", diffs)
    assert not diffs, diffs
    # one-value drift inside the production layout is found with its path
    diffs2 = []
    tc.compare_nodes(reloaded, tc.encode_value(_prod_maps(delta=1.0), "t"),
                     "maps", diffs2)
    assert diffs2 and diffs2[0][1] == "TP_MAP_MISMATCH", diffs2
    # a property (n_free_complex) must NOT appear as an encoded field
    cmap_node = reloaded["items"][0]
    assert cmap_node["kind"] == "dataclass"
    assert "n_free_complex" not in cmap_node["fields"]
    assert len(cmap_node["fields"]) == 10


# --- carried-forward rev-1 checks ---------------------------------------------
def t_object_dtype_rejected():
    expect_tp("TP_UNSUPPORTED_DTYPE",
              lambda: tc.encode_value(np.array([object()], dtype=object),
                                      "t"))
    expect_tp("TP_UNSUPPORTED_DTYPE",
              lambda: tc.encode_value(np.zeros(2, dtype="V8"), "t"))
    expect_tp("TP_UNSUPPORTED_DTYPE",
              lambda: tc.encode_value(np.array(["ab"]), "t"))


def t_torch_tensor_encoding():
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    n = tc.encode_value(x, "t")
    assert n["dtype"] == "<f4" and n["shape"] == [2, 3]
    z = torch.complex(torch.ones(2), torch.zeros(2))
    assert tc.encode_value(z, "t")["dtype"] == "<c8"
    expect_tp("TP_NONFINITE",
              lambda: tc.encode_value(np.array([np.nan]), "t"))


def t_one_byte_drift():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "f.bin")
    with open(p, "wb") as fh:
        fh.write(b"A" * 4096)
    h1 = tc.sha256_file(p)
    with open(p, "r+b") as fh:
        fh.seek(2048)
        fh.write(b"B")
    assert tc.sha256_file(p) != h1


def t_invalid_reference():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "ref.json")
    tc.write_json_with_sidecar(p, {"schema": "x", "v": 1})
    obj, sha = tc.load_json_verified(p)
    assert obj["v"] == 1 and len(sha) == 64
    with open(p, "ab") as fh:
        fh.write(b" ")
    expect_tp("TP_REFERENCE_DRIFT", lambda: tc.load_json_verified(p))
    os.remove(p + ".sha256")
    expect_tp("TP_INPUT_MISSING", lambda: tc.load_json_verified(p))


def t_registry_and_resolution():
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "train"))
    real = os.path.join(root, "train", "vol.h5")
    with open(real, "wb") as fh:
        fh.write(b"real-data")
    trap = os.path.join(tempfile.mkdtemp(), "vol.h5")
    with open(trap, "wb") as fh:
        fh.write(b"trap-data")
    cwd = os.getcwd()
    os.chdir(os.path.dirname(trap))
    try:
        resolved = tc.resolve_touched_file(root, "vol.h5")
        assert resolved == os.path.realpath(real), resolved
    finally:
        os.chdir(cwd)
    expect_tp("TP_INPUT_MISSING",
              lambda: tc.resolve_touched_file(root, "../../etc/hostname"))
    tc.validate_registry_coverage([real], ["vol.h5"], root)
    expect_tp("TP_FILE_REGISTRY_EMPTY",
              lambda: tc.validate_registry_coverage([], ["vol.h5"], root))
    expect_tp("TP_FILE_REGISTRY_INCOMPLETE",
              lambda: tc.validate_registry_coverage([real], [], root))
    other = os.path.join(root, "train", "other.h5")
    with open(other, "wb") as fh:
        fh.write(b"x")
    expect_tp("TP_FILE_REGISTRY_INCOMPLETE",
              lambda: tc.validate_registry_coverage([real, other],
                                                    ["vol.h5"], root))


def t_case_index_extraction():
    root = tempfile.mkdtemp()

    class DS1:
        index = [{"file": "a.h5", "slice_index": 3},
                 {"file": "a.h5", "slice_index": 4}]

    class DS2:
        samples = [("b.h5", 0), ("b.h5", 1)]

    class DS3:
        pass

    assert tc.extract_case_index(DS1(), 2, root) == [("a.h5", 3),
                                                     ("a.h5", 4)]
    assert tc.extract_case_index(DS2(), 2, root) == [("b.h5", 0),
                                                     ("b.h5", 1)]
    expect_tp("TP_INDEX_UNAVAILABLE",
              lambda: tc.extract_case_index(DS3(), 2, root))


def t_output_preservation():
    d = tempfile.mkdtemp()
    assert tc.attempt_dir_is_fresh(d)
    with open(os.path.join(d, "v04tp_parity_record.json"), "w") as fh:
        fh.write("previous PASS evidence")
    assert not tc.attempt_dir_is_fresh(d)
    assert open(os.path.join(d, "v04tp_parity_record.json")).read() == \
        "previous PASS evidence"


def t_log_capture():
    d = tempfile.mkdtemp()
    lp = os.path.join(d, "run.log")
    tc.setup_logging(lp)
    logging.getLogger("seqref_mri.v04tp.test").info("progress-message-42")
    logging.getLogger().handlers.clear()
    content = open(lp).read()
    assert "progress-message-42" in content and "INFO" in content


def t_source_drift():
    root = tempfile.mkdtemp()
    pkg = os.path.join(root, "seqref_mri")
    os.makedirs(pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    mod = os.path.join(pkg, "latemod.py")
    with open(mod, "w") as fh:
        fh.write("X = 1\n")
    sys.path.insert(0, root)
    try:
        import seqref_mri.latemod  # noqa: F401
        reg = tc.collect_modules(os.path.realpath(root))
        with open(mod, "a") as fh:
            fh.write("Y = 2\n")
        expect_tp("TP_SOURCE_DRIFT",
                  lambda: tc.refresh_modules(reg, os.path.realpath(root),
                                             "test"))
    finally:
        sys.path.remove(root)
        for m in [m for m in sys.modules if m.startswith("seqref_mri")]:
            del sys.modules[m]


def t_module_registry_late_import():
    root = tempfile.mkdtemp()
    pkg = os.path.join(root, "seqref_mri")
    os.makedirs(pkg)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    with open(os.path.join(pkg, "early.py"), "w") as fh:
        fh.write("X = 1\n")
    with open(os.path.join(pkg, "late.py"), "w") as fh:
        fh.write("Y = 2\n")
    sys.path.insert(0, root)
    try:
        import seqref_mri.early  # noqa: F401
        reg = tc.collect_modules(os.path.realpath(root))
        import seqref_mri.late  # noqa: F401
        reg = tc.refresh_modules(reg, os.path.realpath(root), "test")
        assert any(k.endswith("late.py") for k in reg)
    finally:
        sys.path.remove(root)
        for m in [m for m in sys.modules if m.startswith("seqref_mri")]:
            del sys.modules[m]


check("INTEG_map_json_roundtrip_nonalpha", t_map_json_roundtrip)
check("INTEG_path_index_identity", t_path_index_identity)
check("INTEG_production_layout_roundtrip", t_production_layout_roundtrip)
check("map_allowlist_contract", t_map_contract)
check("object_dtype_rejected", t_object_dtype_rejected)
check("torch_tensor_encoding", t_torch_tensor_encoding)
check("one_byte_drift", t_one_byte_drift)
check("invalid_reference", t_invalid_reference)
check("registry_and_resolution", t_registry_and_resolution)
check("case_index_extraction", t_case_index_extraction)
check("output_preservation", t_output_preservation)
check("log_capture", t_log_capture)
check("source_drift", t_source_drift)
check("module_registry_late_import", t_module_registry_late_import)

n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks PASS")
sys.exit(2 if n_fail else 0)


# ---------------------------------------------------------------------------
# Changelog
# v0.3 (2026-10-13): production-layout integration case -- frozen dataclass
#   mirroring the audited CoordinateMap declaration (10 fields, defaults
#   included) + 5-key vecs dict; contract accepts, JSON round-trip clean,
#   one-value drift localised, properties not encoded as fields; contract
#   tests updated to the two-allowlist signature.
# v0.2 (2026-10-13): reviewer rev-2 integration cases -- encode/write/
#   reload/compare with NON-alphabetical dataclass layout (the rev-1
#   fixture was alphabetical and blind to the JSON ordering bug);
#   production-style (Path, slice) index entries (absolute + relative);
#   map allowlist contract (bound/mismatch/unbound); extract_case_index
#   signature updated (data_root).
# v0.1 (2026-10-13): initial targeted checks for harness v0.2 (11/11 PASS).
# ---------------------------------------------------------------------------

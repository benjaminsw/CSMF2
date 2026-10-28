# SEQREF-V04TP v0.4 -- v04tp_common
# LIFETIME: KEEP
# =============================================================================
# Purpose: shared helpers for the V04TP tensor-parity harness (run card
#          SEQREF-MRI-V04TP v0.3). Recursive exact-encoding of runtime values
#          (dtype/shape/C-order bytes/SHA-256 descriptor), harness-canonical
#          JSON, import-root isolation with dependency hashing AND mid-run
#          stability re-validation, git pin/tree verification (start and
#          end of attempt), dataset-bound touched-file resolution and exact
#          registry coverage, JSON+sidecar IO, explicit dual-stream logging.
# Encoding contract (card SS6): no precision change, rounding, clipping,
#   tolerance, dtype coercion or signed-zero/NaN normalisation; non-finite
#   floating/complex values rejected BEFORE hashing; little-endian host
#   required; ONLY bool/int/uint/float/complex ndarray kinds are hashable
#   (object/void/string dtypes carry pointers or non-numeric bytes and are
#   rejected); dataclasses encode as explicit named-attribute trees (the
#   CoordinateMap/standardisation-vector export); tuple/list both map to
#   "sequence" so container-syntax differences cannot force false mismatch.
# CONVENTION: logger.error + typed raise (TPError). No fallback, no mock,
#   no placeholder, no silent pass. All errors are logged.
# =============================================================================
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import subprocess
import sys

import numpy as np
import torch  # noqa: F401  (isinstance checks only; harness runs in the locked venv)

logger = logging.getLogger("seqref_mri.v04tp")

__version__ = "0.4"
__abbr__ = "SEQREF-V04TP"

DESCRIPTOR_SCHEMA = "seqref-v04-tp-descriptor/1"
HASHABLE_KINDS = "biufc"  # bool, signed/unsigned int, float, complex only


class TPError(RuntimeError):
    """Single typed identity for the V04TP harness."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def fail(code: str, message: str) -> None:
    logger.error("[%s] %s: %s", __abbr__, code, message)
    raise TPError(code, message)


# ---------------------------------------------------------------------------
# Harness-canonical JSON (independent of either repo pin; the batch-list
# digest is the ONLY digest that must use the repo's canonical_json)
# ---------------------------------------------------------------------------

def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Recursive exact encoder -> descriptor tree
# ---------------------------------------------------------------------------

def _array_node(arr: np.ndarray, context: str) -> dict:
    if arr.dtype.kind not in HASHABLE_KINDS:
        fail("TP_UNSUPPORTED_DTYPE",
             f"{context}: dtype {arr.dtype.str} (kind {arr.dtype.kind!r}) is "
             f"not hashable content -- object/void/string arrays carry "
             f"pointers or non-numeric bytes, not stable tensor content")
    if arr.dtype.byteorder == ">":
        fail("TP_BYTEORDER_UNSUPPORTED",
             f"{context}: big-endian dtype {arr.dtype.str}; the reviewed "
             f"convention is little-endian and silent conversion is forbidden")
    if arr.dtype.byteorder == "=" and sys.byteorder != "little":
        fail("TP_BYTEORDER_UNSUPPORTED",
             f"{context}: native-endian dtype on a non-little-endian host")
    if arr.dtype.kind in "fc" and not bool(np.isfinite(arr).all()):
        fail("TP_NONFINITE",
             f"{context}: non-finite floating/complex value rejected before "
             f"hashing (card SS6)")
    raw = np.ascontiguousarray(arr).tobytes(order="C")
    node = {"kind": "array", "dtype": arr.dtype.str,
            "shape": [int(s) for s in arr.shape], "order": "C",
            "nbytes": len(raw), "content_sha256": sha256_bytes(raw)}
    node["node_sha256"] = sha256_bytes(canonical(
        {k: v for k, v in node.items() if k != "node_sha256"}))
    return node


def encode_value(obj, context: str) -> dict:
    """Encode one runtime value as a descriptor tree. Dataclasses are
    exported as explicit named-attribute trees (reviewer finding 2:
    CoordinateMap / standardisation vectors). Unsupported object or dtype
    representations are errors (card SS5), never skipped."""
    if isinstance(obj, torch.Tensor):
        return _array_node(obj.detach().cpu().numpy(), context)
    if isinstance(obj, np.ndarray):
        return _array_node(obj, context)
    if isinstance(obj, np.generic):
        return _array_node(np.asarray(obj), context)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = {}
        for f in dataclasses.fields(obj):
            fields[f.name] = encode_value(getattr(obj, f.name),
                                          f"{context}.{f.name}")
        node = {"kind": "dataclass", "type": type(obj).__name__,
                "fields": fields}
    elif isinstance(obj, bool):
        node = {"kind": "scalar", "pytype": "bool", "value": obj}
    elif isinstance(obj, int):
        node = {"kind": "scalar", "pytype": "int", "value": obj}
    elif isinstance(obj, float):
        return _array_node(np.asarray(obj), context)
    elif isinstance(obj, str):
        node = {"kind": "scalar", "pytype": "str", "value": obj}
    elif obj is None:
        node = {"kind": "none"}
    elif isinstance(obj, dict):
        items = {}
        for k in sorted(obj.keys(), key=str):
            if not isinstance(k, str):
                fail("TP_FIELD_ENCODING",
                     f"{context}: non-string dict key {k!r}")
            items[k] = encode_value(obj[k], f"{context}.{k}")
        node = {"kind": "dict", "keys": sorted(items), "items": items}
    elif isinstance(obj, (list, tuple)):
        node = {"kind": "sequence",
                "items": [encode_value(v, f"{context}[{i}]")
                          for i, v in enumerate(obj)]}
    else:
        fail("TP_FIELD_ENCODING",
             f"{context}: unsupported type {type(obj).__name__}; card SS5 "
             f"forbids repr/pickle/address encodings")
    node["node_sha256"] = sha256_bytes(canonical(
        {k: v for k, v in node.items() if k != "node_sha256"}))
    return node


def field_descriptor(batch_i: int, case_j: int, file: str,
                     slice_index: int, field: str, node: dict) -> dict:
    """Flat per-field descriptor exactly as frozen in card SS6; provenance
    stays outside so source-origin differences cannot force false mismatch."""
    is_array = node["kind"] == "array"
    desc = {"schema": DESCRIPTOR_SCHEMA, "batch": batch_i, "index": case_j,
            "file": file, "slice_index": slice_index, "field": field,
            "dtype": node["dtype"] if is_array else None,
            "shape": node["shape"] if is_array else None,
            "order": "C" if is_array else None,
            "nbytes": node["nbytes"] if is_array else None,
            "content_sha256": (node["content_sha256"] if is_array
                               else node["node_sha256"])}
    desc["field_sha256"] = sha256_bytes(canonical(desc))
    return desc


def _named_items(node: dict):
    """(key, child) pairs for named container kinds (dict / dataclass),
    ALWAYS in sorted key order on both sides. Canonical JSON serialisation
    alphabetises object keys, so a descriptor reloaded from disk orders
    dataclass fields alphabetically while a freshly encoded descriptor
    keeps declaration order; comparison must be order-insensitive for
    named attributes (reviewer finding 1, rev 2)."""
    if node["kind"] == "dict":
        return [(k, node["items"][k]) for k in sorted(node["items"])]
    return [(k, node["fields"][k]) for k in sorted(node["fields"])]


def compare_nodes(pnode: dict, cnode: dict, path: str, out: list) -> None:
    """First-mismatch path + counts for container (maps) subtrees."""
    if pnode.get("kind") != cnode.get("kind"):
        out.append((path, "TP_FIELD_SET_MISMATCH",
                    f"kind {pnode.get('kind')} != {cnode.get('kind')}"))
        return
    kind = pnode["kind"]
    if kind == "array":
        for key, code in (("dtype", "TP_DTYPE_MISMATCH"),
                          ("shape", "TP_SHAPE_MISMATCH"),
                          ("content_sha256", "TP_MAP_MISMATCH")):
            if pnode[key] != cnode[key]:
                out.append((path, code,
                            f"{key}: {pnode[key]!r} != {cnode[key]!r}"))
                return
        return
    if kind in ("scalar", "none"):
        if pnode.get("node_sha256") != cnode.get("node_sha256"):
            out.append((path, "TP_MAP_MISMATCH", "scalar subtree differs"))
        return
    if kind == "dataclass" and pnode.get("type") != cnode.get("type"):
        out.append((path, "TP_MAP_MISMATCH",
                    f"dataclass type {pnode.get('type')} != "
                    f"{cnode.get('type')}"))
        return
    pitems = _named_items(pnode) if kind in ("dict", "dataclass") else \
        [(str(i), v) for i, v in enumerate(pnode["items"])]
    citems = _named_items(cnode) if kind in ("dict", "dataclass") else \
        [(str(i), v) for i, v in enumerate(cnode["items"])]
    if [k for k, _ in pitems] != [k for k, _ in citems]:
        out.append((path, "TP_FIELD_SET_MISMATCH",
                    f"container keys {[k for k, _ in pitems]} != "
                    f"{[k for k, _ in citems]}"))
        return
    for (k, pv), (_, cv) in zip(pitems, citems):
        compare_nodes(pv, cv, f"{path}.{k}", out)


# ---------------------------------------------------------------------------
# Import-root isolation + dependency hashing with mid-run stability
# ---------------------------------------------------------------------------

def activate_import_root(root: str) -> str:
    root = os.path.realpath(root)
    if not os.path.isdir(root):
        fail("TP_INPUT_MISSING", f"import root {root} does not exist")
    if any(m == "seqref_mri" or m.startswith("seqref_mri.")
           for m in sys.modules):
        fail("TP_IMPORT_ROOT_MISMATCH",
             "seqref_mri was imported before isolation was activated")
    sys.path.insert(0, root)
    logger.info("[%s] import root activated: %s", __abbr__, root)
    return root


def collect_modules(root: str) -> dict:
    """SHA-256 every imported seqref_mri.* module file; abort on any module
    resolved OUTSIDE the activated root (import-root contamination)."""
    out = {}
    for name, mod in sorted(sys.modules.items()):
        if not (name == "seqref_mri" or name.startswith("seqref_mri.")):
            continue
        f = getattr(mod, "__file__", None)
        if not f:
            continue  # namespace package node carries no file
        real = os.path.realpath(f)
        if not real.startswith(root + os.sep):
            fail("TP_IMPORT_ROOT_MISMATCH",
                 f"module {name} resolved outside the activated root: {real}")
        out[os.path.relpath(real, root)] = sha256_file(real)
    logger.info("[%s] dependency registry: %d module files hashed under %s",
                __abbr__, len(out), root)
    return out


def refresh_modules(registry: dict, root: str, when: str) -> dict:
    """Re-walk imports AFTER execution stages (reviewer finding 8): modules
    imported late are added and hashed; a CHANGED hash for an already
    registered module is source drift and aborts the attempt."""
    current = collect_modules(root)
    for rel, sha in current.items():
        if rel in registry and registry[rel] != sha:
            fail("TP_SOURCE_DRIFT",
                 f"module file changed during the attempt ({when}): {rel}")
    late = sorted(k for k in current if k not in registry)
    if late:
        logger.info("[%s] late-imported modules registered at %s: %s",
                    __abbr__, when, late)
    registry.update(current)
    return registry


# ---------------------------------------------------------------------------
# Git pin / tree verification (fail-closed; called at start AND end)
# ---------------------------------------------------------------------------

def git_head(repo: str) -> str:
    r = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        fail("TP_PIN_MISMATCH",
             f"git rev-parse HEAD failed in {repo}: {r.stderr.strip()}")
    return r.stdout.strip()


def git_porcelain(repo: str) -> str:
    r = subprocess.run(["git", "-C", repo, "status",
                        "--porcelain=v1", "--untracked-files=all"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        fail("TP_PIN_MISMATCH",
             f"git status failed in {repo}: {r.stderr.strip()}")
    return r.stdout


def verify_pin_and_clean_tree(repo: str, pin: str) -> None:
    head = git_head(repo)
    if head != pin:
        fail("TP_PIN_MISMATCH", f"{repo} HEAD {head} != pinned {pin}")
    dirty = git_porcelain(repo)
    if dirty.strip():
        fail("TP_TREE_DIRTY",
             f"{repo} working tree is not clean:\n{dirty}")


# ---------------------------------------------------------------------------
# Dataset-bound touched-file resolution + exact registry coverage
# ---------------------------------------------------------------------------

def root_relative_identity(raw_file, data_root: str) -> str:
    """The dataset's exact root-relative POSIX identity for metadata
    comparisons (reviewer finding 2, rev 2). The production dataset index
    stores (Path, slice) entries whose paths may be ABSOLUTE, while loader
    meta["file"] is emitted relative to data_root; comparison identities
    must live in the relative space, hashing in the resolved-absolute
    space. Absolute inputs must resolve UNDER the data root."""
    root = os.path.realpath(data_root)
    p = os.fspath(raw_file)
    if os.path.isabs(p):
        real = os.path.realpath(p)
        if not real.startswith(root + os.sep):
            fail("TP_INPUT_MISSING",
                 f"index path {real} escapes the data root {root}; cannot "
                 f"derive a root-relative identity")
        return os.path.relpath(real, root).replace(os.sep, "/")
    return p.replace(os.sep, "/")


def resolve_touched_file(data_root: str, rel) -> str:
    """Resolve a case file identity to an existing absolute path UNDER the
    data root (reviewer finding 5: never the working directory)."""
    root = os.path.realpath(data_root)
    rel = os.fspath(rel)
    candidates = [rel] if os.path.isabs(rel) else [
        os.path.join(root, rel), os.path.join(root, "train", rel)]
    for cand in candidates:
        real = os.path.realpath(cand)
        if os.path.isfile(real):
            if not real.startswith(root + os.sep):
                fail("TP_INPUT_MISSING",
                     f"resolved path {real} escapes the data root {root}; "
                     f"refusing to bind a hash to it")
            return real
    fail("TP_INPUT_MISSING",
         f"cannot resolve touched training file {rel!r} under {root} "
         f"(tried {[os.path.realpath(c) for c in candidates]})")


def extract_case_index(ds, n: int, data_root: str) -> list:
    """The dataset's own ordered case index as (root-relative POSIX file
    identity, slice_index) pairs -- the SAME identity space the loader
    emits in meta["file"] (reviewer finding 2, rev 2: production index
    entries are (Path, slice) and may be absolute). Used to identify and
    hash touched files BEFORE any tensor loading (finding 4).
    Deterministic candidate-attribute scan with a hard typed error; the
    loader meta is cross-checked against it case-by-case, so a wrong
    attribute cannot survive batch 0."""
    for attr in ("index", "slice_index", "samples", "entries", "index_map",
                 "meta"):
        idx = getattr(ds, attr, None)
        if isinstance(idx, (list, tuple)) and len(idx) >= n:
            first = idx[0]
            if isinstance(first, dict) and "file" in first and \
                    "slice_index" in first:
                return [(root_relative_identity(e["file"], data_root),
                         int(e["slice_index"])) for e in idx[:n]]
            if isinstance(first, (list, tuple)) and len(first) >= 2 and \
                    isinstance(first[0], (str, os.PathLike)) and \
                    isinstance(first[1], (int, np.integer)):
                return [(root_relative_identity(e[0], data_root),
                         int(e[1])) for e in idx[:n]]
    fail("TP_INDEX_UNAVAILABLE",
         f"no ordered case-index attribute found on {type(ds).__name__} "
         f"(tried index/slice_index/samples/entries/index_map/meta); the "
         f"pre-load file registry cannot be built")


def assert_map_contract(maps_obj, cmap_fields, vecs_keys,
                        context: str) -> tuple:
    """Frozen map-structure contract (card SS5): maps must be a 2-element
    (cmap, vecs) container; cmap must be a dataclass instance whose field
    set EQUALS the frozen allowlist; vecs must be a dict whose key set
    EQUALS the frozen standardisation-vector keys. Both allowlists bind
    from the pinned-source audit (residual_decoder.CoordinateMap
    declaration; free_flow_runtime.standardisation_vectors return
    literal) BEFORE execution; while either is unbound the harness fails
    closed -- it cannot run, and cannot silently pass.
    Returns (cmap, vecs)."""
    if cmap_fields is None or vecs_keys is None:
        fail("TP_MAP_CONTRACT_UNBOUND",
             f"{context}: the map allowlists are not frozen; bind "
             f"MAP_CMAP_FIELDS / MAP_VECS_KEYS from the pinned-source "
             f"audit (card SS5) before any execution")
    if not isinstance(maps_obj, (list, tuple)) or len(maps_obj) != 2:
        mlen = (len(maps_obj) if isinstance(maps_obj, (list, tuple))
                else "n/a")
        fail("TP_MAP_MISMATCH",
             f"{context}: maps is {type(maps_obj).__name__} of length "
             f"{mlen}; expected a 2-element (cmap, vecs) container")
    cmap, vecs = maps_obj
    if not (dataclasses.is_dataclass(cmap) and not isinstance(cmap, type)):
        fail("TP_MAP_MISMATCH",
             f"{context}: cmap is {type(cmap).__name__}; expected the "
             f"CoordinateMap dataclass")
    observed = frozenset(f.name for f in dataclasses.fields(cmap))
    if observed != frozenset(cmap_fields):
        fail("TP_FIELD_SET_MISMATCH",
             f"{context}: CoordinateMap fields {sorted(observed)} != the "
             f"frozen allowlist {sorted(cmap_fields)}")
    if not isinstance(vecs, dict):
        fail("TP_MAP_MISMATCH",
             f"{context}: vecs is {type(vecs).__name__}; expected the "
             f"standardisation-vector dict")
    observed_v = frozenset(vecs.keys())
    if observed_v != frozenset(vecs_keys):
        fail("TP_FIELD_SET_MISMATCH",
             f"{context}: standardisation-vector keys "
             f"{sorted(observed_v)} != the frozen allowlist "
             f"{sorted(vecs_keys)}")
    return cmap, vecs


def build_file_registry(case_index: list, data_root: str) -> list:
    """Ordered unique resolved paths for the locked case list. Empty or
    unresolvable registries are errors (reviewer finding 5)."""
    if not case_index:
        fail("TP_FILE_REGISTRY_EMPTY", "case index is empty")
    seen, ordered = set(), []
    for raw, _slice in case_index:
        fp = resolve_touched_file(data_root, raw)
        if fp not in seen:
            seen.add(fp)
            ordered.append(fp)
    if not ordered:
        fail("TP_FILE_REGISTRY_EMPTY",
             "file registry resolved to zero files")
    logger.info("[%s] file registry: %d touched training files", __abbr__,
                len(ordered))
    return ordered


def validate_registry_coverage(registry_files, case_files_raw,
                               data_root: str) -> None:
    """Registry must cover EXACTLY the required cases -- no missing, no
    extra (reviewer finding 5)."""
    if not registry_files:
        fail("TP_FILE_REGISTRY_EMPTY", "file registry is empty")
    registry = set(registry_files)
    required = {resolve_touched_file(data_root, raw)
                for raw in case_files_raw}
    missing = sorted(required - registry)
    extra = sorted(registry - required)
    if missing:
        fail("TP_FILE_REGISTRY_INCOMPLETE",
             f"registry misses {len(missing)} required file(s), first: "
             f"{missing[0]}")
    if extra:
        fail("TP_FILE_REGISTRY_INCOMPLETE",
             f"registry carries {len(extra)} file(s) beyond the locked case "
             f"list, first: {extra[0]}")


# ---------------------------------------------------------------------------
# Attempt-directory freshness (reviewer finding 1): checked BEFORE any log
# or record is created; previous artifacts remain untouched.
# ---------------------------------------------------------------------------

def attempt_dir_is_fresh(out_dir: str) -> bool:
    return not (os.path.isdir(out_dir) and os.listdir(out_dir))


# ---------------------------------------------------------------------------
# JSON + sidecar IO
# ---------------------------------------------------------------------------

def write_json_with_sidecar(path: str, obj) -> str:
    raw = canonical(obj)
    with open(path, "wb") as fh:
        fh.write(raw)
    digest = sha256_bytes(raw)
    with open(path + ".sha256", "w") as fh:
        fh.write(f"{digest}  {os.path.basename(path)}\n")
    logger.info("[%s] wrote %s (sha256 %s...)", __abbr__, path, digest[:12])
    return digest


def load_json_verified(path: str) -> tuple:
    """Load a JSON artifact and verify its .sha256 sidecar; drift is an
    error, never a silent pass."""
    if not os.path.isfile(path):
        fail("TP_INPUT_MISSING", f"required artifact {path} is missing")
    side = path + ".sha256"
    if not os.path.isfile(side):
        fail("TP_INPUT_MISSING", f"required sidecar {side} is missing")
    with open(path, "rb") as fh:
        raw = fh.read()
    actual = sha256_bytes(raw)
    declared = open(side).read().split()[0]
    if actual != declared:
        fail("TP_REFERENCE_DRIFT",
             f"{path} sha256 {actual} != sidecar {declared}")
    return json.loads(raw.decode("utf-8")), actual


# ---------------------------------------------------------------------------
# Explicit dual-stream logging (reviewer finding 7: levels set explicitly on
# the root logger AND both handlers; no reliance on basicConfig no-ops)
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"))
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)
    root.addHandler(sh)


def environment() -> dict:
    import platform
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "platform": platform.platform()}


# ---------------------------------------------------------------------------
# Changelog
# v0.4 (2026-10-13): map contract extended to the standardisation-vector
#   key set (assert_map_contract now takes vecs_keys; vecs must be a dict
#   with exactly the frozen keys). Audit-complete bindings: CoordinateMap
#   declares 10 dataclass fields (n_free_complex/n_acquired/flow_dim_real
#   are @property, NOT fields); standardisation_vectors returns
#   {mean_re, scale_re, mean_im, scale_im, n_free_complex}.
# v0.3 (2026-10-13): reviewer NO-GO rev-2 corrections -- named-container
#   comparison by sorted keys (JSON round-trip alphabetises dataclass
#   fields; declaration order must not fire TP_FIELD_SET_MISMATCH);
#   root_relative_identity normalises (Path, slice) index entries -- incl.
#   absolute Paths -- into the loader's relative POSIX identity space;
#   extract_case_index takes data_root and accepts PathLike entries;
#   assert_map_contract with fail-closed unbound allowlist
#   (TP_MAP_CONTRACT_UNBOUND) pending the pinned-source audit freeze.
# v0.2 (2026-10-13): reviewer NO-GO corrections -- dataclass attribute-tree
#   encoding; object/void/string dtype rejection; dataset-bound resolution;
#   pre-load index extraction + registry coverage; module registry re-walk
#   with TP_SOURCE_DRIFT; explicit log levels; attempt freshness helper.
# v0.1 (2026-10-13): initial implementation (superseded before execution).
# ---------------------------------------------------------------------------

# SEQREF-MRI-V05LTMAN v0.1 -- V05LT split + case manifest generator (194 §3, §3.1, §10.1)
# LIFETIME: KEEP
"""Build the V05LT train/validation split manifest and the visual-case
manifest from the frozen V02 manifests (no MRI data is read).

Modes
  build  : verify sources, generate payloads, write files (refuses to overwrite).
  verify : verify sources, regenerate payloads in memory and compare them
           byte-for-byte with the files already written (independent process).

Rules (194):
  * Volume ID = basename of entries[].file (uniqueness and train/holdout
    disjointness are checked); original relative path kept.
  * Split: sort the 973 training IDs; perm = default_rng(20261020).permutation(973);
    validation = first 97 of sorted[perm], train = remaining 876; both stored sorted.
  * Cases: from the SORTED validation list, idx = default_rng(20261021)
    .choice(97, size=6, replace=False); case order = draw order.
  * Middle slice: V02 midpoint_slice (v02_manifests.py:103 @6ffce4e5):
    n even -> n/2 - 1, n odd -> (n-1)/2, i.e. (n-1)//2.
  * Payload files are canonical JSON (sort_keys, compact, UTF-8) with no
    timestamps; file sha256 == payload sha256. Run metadata goes to a
    separate run-record file.
Exit 0 = all checks passed; exit 2 = any failure (logged). No fallbacks.
"""
import argparse
import datetime
import hashlib
import json
import logging
import os
import platform
import sys
from pathlib import Path

import numpy as np

__version__ = "SEQREF-MRI-V05LTMAN v0.1"

SRC_REL = "seqref_mri/results/v02/manifests"
SRC = {
    "train": {"name": "v02_epoch0_manifest.json", "kind": "train_epoch",
              "file_sha256": "b0637f787ce198e4424f6b506fed264300ac585921ffaf17995004bb317efb56",
              "content_sha256": "390220ab031455584bbf5df4d9485550dea5c9109894e6b335c9444d973a8e9e"},
    "holdout": {"name": "v02_holdout_manifest.json", "kind": "holdout",
                "file_sha256": "0f2f96073fad6bdaad1314ae1319b643a1b67c06424e757f1fea3cfc1e002019",
                "content_sha256": "ce27550cdcdfa627372a53d51df6c7a97e4c6cff0f5a80622e0fa110b0eea165"},
}
SRC_SCHEMA = "seqref-v02-manifest/1"
N_TRAIN_VOLS, N_TRAIN_SLICES, N_HOLDOUT = 973, 34742, 199
N_VAL, N_CASES = 97, 6
SPLIT_SEED, CASE_SEED = 20261020, 20261021

SPLIT_FILE = "v05lt_split_manifest.json"
CASE_FILE = "v05lt_case_manifest.json"
RUN_FILE = "v05lt_manifests_run.json"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s [V05LTMAN] %(message)s")
log = logging.getLogger("v05ltman")


class ManError(RuntimeError):
    pass


def require(cond, code, msg):
    if not cond:
        raise ManError(f"{code}: {msg}")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def midpoint_slice(n: int) -> int:
    """Copy of V02 midpoint_slice (v02_manifests.py:103 @6ffce4e5)."""
    require(isinstance(n, (int, np.integer)) and n > 0, "MIDPOINT_INVALID",
            f"slice count {n!r}")
    m = n // 2 - 1 if n % 2 == 0 else (n - 1) // 2
    require(m == (n - 1) // 2, "MIDPOINT_RULE", f"n={n} m={m}")
    return int(m)


def load_source(root: Path, key: str) -> dict:
    s = SRC[key]
    path = root / SRC_REL / s["name"]
    raw = path.read_bytes()
    fh = sha(raw)
    side = Path(str(path) + ".sha256").read_text().strip().split()[0]
    require(fh == side, "SRC_SIDECAR", f"{s['name']}: {fh} != sidecar {side}")
    require(fh == s["file_sha256"], "SRC_PIN", f"{s['name']}: {fh} != pinned")
    doc = json.loads(raw)
    require(doc.get("schema") == SRC_SCHEMA, "SRC_SCHEMA", s["name"])
    require(doc.get("kind") == s["kind"], "SRC_KIND", s["name"])
    body = {k: v for k, v in doc.items() if k != "manifest_sha256"}
    ch = sha(canon(body))
    require(ch == doc.get("manifest_sha256") == s["content_sha256"],
            "SRC_CONTENT_HASH", f"{s['name']}: {ch}")
    log.info("source %s verified (file %s, content %s)", s["name"], fh, ch)
    return doc


def build_payloads(root: Path):
    tr = load_source(root, "train")
    ho = load_source(root, "holdout")

    # training volumes: slices per basename
    vols = {}
    for e in tr["entries"]:
        f, si, di = e["file"], int(e["slice_index"]), int(e["dataset_index"])
        vid = Path(f).name
        v = vols.setdefault(vid, {"file": f, "slices": {}})
        require(v["file"] == f, "ID_AMBIGUOUS", f"{vid}: {v['file']} vs {f}")
        require(si not in v["slices"], "DUP_SLICE", f"{vid} slice {si}")
        v["slices"][si] = di
    require(len(tr["entries"]) == N_TRAIN_SLICES, "TRAIN_SLICES", len(tr["entries"]))
    require(len(vols) == N_TRAIN_VOLS, "TRAIN_VOLS", len(vols))
    for vid, v in vols.items():
        n = len(v["slices"])
        require(sorted(v["slices"]) == list(range(n)), "SLICES_NOT_CONTIGUOUS", vid)
        v["n"] = n

    hfiles = [e["file"] for e in ho["entries"]]
    hids = {Path(f).name for f in hfiles}
    require(len(hfiles) == len(set(hfiles)) == len(hids) == N_HOLDOUT,
            "HOLDOUT_POP", f"{len(hfiles)}/{len(hids)}")
    require(not hids & set(vols), "TRAIN_HOLDOUT_OVERLAP",
            sorted(hids & set(vols))[:5])

    # split
    ids = sorted(vols)
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(ids))
    shuffled = [ids[int(i)] for i in perm]
    val_ids, train_ids = sorted(shuffled[:N_VAL]), sorted(shuffled[N_VAL:])
    require(len(val_ids) == N_VAL and len(train_ids) == N_TRAIN_VOLS - N_VAL,
            "SPLIT_COUNTS", f"{len(val_ids)}/{len(train_ids)}")
    require(not set(val_ids) & set(train_ids), "SPLIT_OVERLAP", "val/train")
    require(set(val_ids) | set(train_ids) == set(vols), "SPLIT_COVER", "union")

    def rows(lst):
        return [{"volume_id": i, "file": vols[i]["file"], "n_slices": vols[i]["n"]}
                for i in lst]

    def ids_sha(lst):
        return sha(("\n".join(lst) + "\n").encode("utf-8"))

    src_block = {k: {"path": f"{SRC_REL}/{SRC[k]['name']}",
                     "file_sha256": SRC[k]["file_sha256"],
                     "content_sha256": SRC[k]["content_sha256"]} for k in SRC}
    split = {
        "schema": "seqref-v05lt-split-manifest/1",
        "study": "SEQREF-MRI-V05LT",
        "spec": "194 §3, §3.1",
        "sources": src_block,
        "id_rule": "basename of V02 entries[].file",
        "recipe": ("ids = sorted(973 training basenames); "
                   f"perm = numpy.random.default_rng({SPLIT_SEED}).permutation(973); "
                   f"validation = sorted(ids[perm][:{N_VAL}]); "
                   f"train = sorted(ids[perm][{N_VAL}:])"),
        "counts": {"train_volumes": len(train_ids),
                   "validation_volumes": len(val_ids),
                   "holdout_volumes": N_HOLDOUT,
                   "train_slices": sum(vols[i]["n"] for i in train_ids),
                   "validation_slices": sum(vols[i]["n"] for i in val_ids)},
        "train_ids_sha256": ids_sha(train_ids),
        "validation_ids_sha256": ids_sha(val_ids),
        "holdout_overlap": 0,
        "validation": rows(val_ids),
        "train": rows(train_ids),
    }
    require(split["counts"]["train_slices"] + split["counts"]["validation_slices"]
            == N_TRAIN_SLICES, "SLICE_SUM", split["counts"])
    split_bytes = canon(split)

    # cases
    pick = np.random.default_rng(CASE_SEED).choice(N_VAL, size=N_CASES, replace=False)
    require(len({int(i) for i in pick}) == N_CASES, "CASE_DUP", pick.tolist())
    cases = []
    for rank, i in enumerate(pick.tolist()):
        vid = val_ids[int(i)]
        n = vols[vid]["n"]
        m = midpoint_slice(n)
        cases.append({"case": rank, "validation_sorted_index": int(i),
                      "volume_id": vid, "file": vols[vid]["file"],
                      "n_slices": n, "slice_index": m,
                      "v02_dataset_index": vols[vid]["slices"][m]})
    case = {
        "schema": "seqref-v05lt-case-manifest/1",
        "study": "SEQREF-MRI-V05LT",
        "spec": "194 §10.1",
        "split_manifest_sha256": sha(split_bytes),
        "recipe": (f"idx = numpy.random.default_rng({CASE_SEED}).choice({N_VAL}, "
                   f"size={N_CASES}, replace=False) over the sorted validation list; "
                   "slice = midpoint_slice(n) = (n-1)//2 (V02 v02_manifests.py:103 "
                   "@6ffce4e5); case order = draw order"),
        "cases": cases,
    }
    return split_bytes, canon(case)


def write_new(path: Path, data: bytes):
    with open(path, "xb") as f:      # 'x' refuses to overwrite
        f.write(data)
    require(path.read_bytes() == data, "READBACK", str(path))
    with open(str(path) + ".sha256", "x", encoding="utf-8") as f:
        f.write(f"{sha(data)}  {path.name}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["build", "verify"])
    ap.add_argument("--root", default="/home/benjamin/CSMFII")
    ap.add_argument("--out-dir", default=os.path.expanduser("~/v05lt_manifests"))
    a = ap.parse_args()
    out = Path(a.out_dir)
    try:
        log.info("%s; mode %s; numpy %s; root %s; out %s",
                 __version__, a.mode, np.__version__, a.root, out)
        split_b, case_b = build_payloads(Path(a.root))
        targets = {SPLIT_FILE: split_b, CASE_FILE: case_b}
        if a.mode == "build":
            out.mkdir(parents=True, exist_ok=True)
            for n in list(targets) + [RUN_FILE]:
                for p in (out / n, out / (n + ".sha256")):
                    require(not p.exists(), "ARTIFACT_EXISTS", f"{p} (refusing to overwrite)")
            for n, b in targets.items():
                write_new(out / n, b)
            run = {"generator": __version__, "mode": "build",
                   "numpy_version": np.__version__, "python": platform.python_version(),
                   "host": platform.node(), "root": a.root,
                   "created_utc": datetime.datetime.now(datetime.timezone.utc)
                   .isoformat(timespec="seconds"),
                   "outputs": {n: sha(b) for n, b in targets.items()}}
            write_new(out / RUN_FILE, json.dumps(run, indent=2, sort_keys=True).encode() + b"\n")
        else:
            for n, b in targets.items():
                p = out / n
                require(p.exists(), "MISSING", str(p))
                disk = p.read_bytes()
                require(disk == b, "VERIFY_MISMATCH",
                        f"{n}: disk {sha(disk)} != regenerated {sha(b)}")
                side = Path(str(p) + ".sha256").read_text().split()[0]
                require(side == sha(b), "VERIFY_SIDECAR", n)
                log.info("verify %s: byte-identical (%s)", n, sha(b))
    except ManError as exc:
        log.error("%s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001 -- log every unexpected failure
        log.exception("UNEXPECTED: %s", exc)
        return 2
    split = json.loads(split_b)
    case = json.loads(case_b)
    print(f"\nRESULT ({a.mode}, paste back):")
    print(f"  split_manifest_sha256={sha(split_b)}")
    print(f"  case_manifest_sha256={sha(case_b)}")
    print(f"  train_ids_sha256={split['train_ids_sha256']}")
    print(f"  validation_ids_sha256={split['validation_ids_sha256']}")
    print(f"  counts={json.dumps(split['counts'], sort_keys=True)}")
    for c in case["cases"]:
        print(f"  case {c['case']}: {c['volume_id']} n={c['n_slices']} slice={c['slice_index']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

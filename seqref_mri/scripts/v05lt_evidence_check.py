# SEQREF-MRI-V05LTEV v0.3 -- scripts.v05lt_evidence_check
# LIFETIME: KEEP
# =============================================================================
# Purpose: evidence checker for SEQREF-MRI-V05LT (195 F18, P1, §9.3).
#   Mode `inputs`  : every registered input (194 §3.1 / 195 §3) is checked
#                    against the REGISTERED constants in v05lt_common --
#                    inputs/ copies, bank arrays, V02 source manifests,
#                    P0S S_ref; optional Z_DIAG regeneration (--zdiag).
#   Mode `manifest`: an evidence manifest is accepted only if its own
#                    sha256 equals an APPROVED value supplied by the
#                    operator (recorded in V05LTEXEC beforehand); each
#                    entry is anchored ("registered:<name>" must equal the
#                    registered constant, or "approved_manifest") and then
#                    re-hashed on disk. A file and a freshly generated hash
#                    agreeing is never sufficient on its own.
# Limits: hash checks establish identity with registered/approved bytes,
#   not scientific correctness of an artefact.
# CONVENTION: exit 0 = PASS, 2 = any failure (logged with a typed code).
#   No fallback, no silent pass.
# Changelog (v0.3, 2026-10-24; b3 revision-1 review fixes):
#   * check_chain re-hashes every source identity recorded in the analysis
#     (inputs.identities) and requires equality (EVIDENCE_SOURCE_MISSING /
#     EVIDENCE_SOURCE_MISMATCH); an empty identity set is refused.
#   * Selection manifest bound to the analysis learning curves: same runs,
#     selected epoch = selected_epoch, final epoch = stop_epoch.
# Changelog (v0.2, 2026-10-24; I1.8 b3, SEQREF-MRI-V05LTI18B3 v0.1):
#   * Analysis-chain manifest (schema seqref-v05lt-chain-manifest/1):
#     configuration, difficult-case file, analysis document, every report
#     file, selection manifest and (optionally) the holdout configuration,
#     with named roles. Accepted only if its own sha256 equals the
#     externally APPROVED value (jointly altered files + manifest refused);
#     every entry re-hashed on disk; then the bindings are checked:
#     analysis -> configuration and difficult cases, report (verify_report
#     incl. the exact file set) -> analysis, selection -> configuration,
#     analysis and report. Mode `chain`.
# Changelog (NEW in v0.1, 2026-10-21):
#   * Introduced under I1.3 (SEQREF-MRI-V05LTI13 v0.2): anchored manifest
#     checks per review (registered or approved anchors only).
# =============================================================================
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path, PurePosixPath

from seqref_mri.scripts import v05lt_common as C

logger = logging.getLogger("seqref_mri.v05lt_evidence_check")

__version__ = "0.3"
__abbr__ = "SEQREF-MRI-V05LTEV"

MANIFEST_SCHEMA = "seqref-v05lt-evidence-manifest/1"

# name -> registered sha256 (every constant registered in 194 / 195 §3)
REGISTERED_ANCHORS = {
    **C.REGISTERED_INPUTS,
    **{v[0]: v[2] for v in C.V02_SOURCES.values()},
    "normalisation_scale_facts.json": C.P0S_SHA256,
}


def check_inputs(paths: C.Paths, zdiag: bool = False) -> list:
    """-> list of (name, sha256, anchor) for every check executed."""
    out = []
    for n, s in C.verify_registered_inputs(paths.inputs).items():
        out.append((f"inputs/{n}", s, f"registered:{n}"))
    for n, shape in C.verify_extra_banks(paths.inputs).items():
        out.append((f"inputs/{n}#array", C.BANK_ARRAY_SHA256[n],
                    f"registered:array {shape}"))
    for key in ("train", "holdout"):
        C.load_v02_source(paths.v02_manifests, key)
        name, _, fsha, csha = C.V02_SOURCES[key]
        out.append((f"v02/manifests/{name}", fsha,
                    f"registered:{name} (content {csha[:12]})"))
    v = C.verify_p0s(paths.p0s)
    out.append(("_diag/residual_preflight/normalisation_scale_facts.json",
                C.P0S_SHA256, f"registered:P0S S_ref={v!r}"))
    if zdiag:
        z = C.verify_zdiag()
        out.append(("Z_DIAG (regenerated)", z["bank_sha256"],
                    f"registered:manifest {z['manifest_sha256'][:12]}"))
    return out


def _safe_rel(p: str) -> str:
    pp = PurePosixPath(p)
    C.require(not pp.is_absolute() and ".." not in pp.parts and p == str(pp)
              and p != ".", "EVIDENCE_PATH_INVALID", repr(p))
    return p


def make_manifest(out_path: Path, root: Path, entries: list,
                  anchors: dict = REGISTERED_ANCHORS) -> str:
    """entries: [(relpath, anchor)]; anchor 'registered:<name>' or
    'approved_manifest'. Writes a new manifest; returns its sha256 (to be
    recorded/approved in V05LTEXEC before any check relies on it)."""
    rows = []
    for rel, anchor in entries:
        _safe_rel(rel)
        sha = C.sha256_file(root / rel)
        if anchor.startswith("registered:"):
            key = anchor.split(":", 1)[1]
            C.require(anchors.get(key) == sha, "EVIDENCE_ANCHOR_MISMATCH",
                      f"{rel}: {sha} vs registered {anchors.get(key)}")
        else:
            C.require(anchor == "approved_manifest", "EVIDENCE_ANCHOR_INVALID",
                      anchor)
        rows.append({"path": rel, "sha256": sha, "anchor": anchor})
    doc = {"schema": MANIFEST_SCHEMA, "entries": rows}
    return C.write_durable_new(out_path, C.canonical_json(doc))


def check_manifest(manifest: Path, approved_sha256: str, root: Path,
                   anchors: dict = REGISTERED_ANCHORS) -> list:
    C.require(isinstance(approved_sha256, str) and len(approved_sha256) == 64,
              "EVIDENCE_APPROVAL_MISSING", "approved sha256 required")
    got = C.sha256_file(manifest)
    C.require(got == approved_sha256, "EVIDENCE_MANIFEST_NOT_APPROVED",
              f"{manifest}: {got} != approved {approved_sha256}")
    doc = json.loads(manifest.read_bytes())
    C.require(doc.get("schema") == MANIFEST_SCHEMA, "EVIDENCE_SCHEMA",
              repr(doc.get("schema")))
    rows = doc.get("entries")
    C.require(isinstance(rows, list) and rows, "EVIDENCE_EMPTY", str(manifest))
    seen, out = set(), []
    for r in rows:
        rel = _safe_rel(r.get("path", ""))
        C.require(rel not in seen, "EVIDENCE_DUPLICATE", rel)
        seen.add(rel)
        sha, anchor = r.get("sha256"), r.get("anchor", "")
        if anchor.startswith("registered:"):
            key = anchor.split(":", 1)[1]
            C.require(key in anchors, "EVIDENCE_ANCHOR_UNKNOWN", anchor)
            C.require(anchors[key] == sha, "EVIDENCE_ANCHOR_MISMATCH",
                      f"{rel}: manifest {sha} vs registered {anchors[key]}")
        else:
            C.require(anchor == "approved_manifest", "EVIDENCE_ANCHOR_INVALID",
                      f"{rel}: {anchor!r}")
        disk = C.sha256_file(root / rel) if (root / rel).is_file() else None
        C.require(disk is not None, "EVIDENCE_MISSING", rel)
        C.require(disk == sha, "EVIDENCE_HASH_MISMATCH",
                  f"{rel}: disk {disk} != manifest {sha}")
        out.append((rel, sha, anchor))
    return out


CHAIN_SCHEMA = "seqref-v05lt-chain-manifest/1"
CHAIN_ROLES = ("config", "difficult", "analysis", "report_dir", "selection",
               "holdout_config")


def make_chain_manifest(out_path: Path, root: Path, roles: dict) -> str:
    """roles: {role: relpath} for config, difficult, analysis, report_dir
    (a directory: every file in it is listed), selection and optionally
    holdout_config. Returns the manifest sha256, which must be recorded /
    approved in V05LTEXEC before check_chain relies on it."""
    C.require(set(roles) - {"holdout_config"} == set(CHAIN_ROLES) -
              {"holdout_config"} and set(roles) <= set(CHAIN_ROLES),
              "EVIDENCE_CHAIN_ROLES", f"{sorted(roles)}")
    rows = []
    for role in CHAIN_ROLES:
        if role not in roles:
            continue
        rel = _safe_rel(roles[role])
        files = ([f"{rel}/{p.name}" for p in sorted((root / rel).iterdir())]
                 if role == "report_dir" else [rel])
        C.require(bool(files), "EVIDENCE_EMPTY", rel)
        for f in files:
            C.require((root / f).is_file() and not (root / f).is_symlink(),
                      "EVIDENCE_MISSING", f)
            rows.append({"path": f, "sha256": C.sha256_file(root / f),
                         "anchor": "approved_manifest", "role": role})
    doc = {"schema": CHAIN_SCHEMA, "roles": dict(roles), "entries": rows}
    return C.write_durable_new(out_path, C.canonical_json(doc))


def check_chain(manifest: Path, approved_sha256: str, root: Path) -> dict:
    """Approved chain manifest -> every entry on disk -> bindings. Internal
    agreement alone is never sufficient: the manifest's own sha256 must
    equal the externally approved value."""
    from seqref_mri.scripts import v05lt_analysis as A
    from seqref_mri.scripts import v05lt_report as R
    C.require(isinstance(approved_sha256, str) and len(approved_sha256) == 64,
              "EVIDENCE_APPROVAL_MISSING", "approved sha256 required")
    got = C.sha256_file(manifest)
    C.require(got == approved_sha256, "EVIDENCE_MANIFEST_NOT_APPROVED",
              f"{manifest}: {got} != approved {approved_sha256}")
    try:
        doc = json.loads(Path(manifest).read_bytes())
    except ValueError as exc:
        C.fail("EVIDENCE_SCHEMA", f"{manifest}: {exc}")
    C.require(isinstance(doc, dict) and doc.get("schema") == CHAIN_SCHEMA and
              isinstance(doc.get("roles"), dict) and
              isinstance(doc.get("entries"), list) and doc["entries"],
              "EVIDENCE_SCHEMA", str(manifest))
    roles, sha, seen = doc["roles"], {}, set()
    for r in doc["entries"]:
        rel = _safe_rel(r.get("path", ""))
        C.require(rel not in seen, "EVIDENCE_DUPLICATE", rel)
        seen.add(rel)
        C.require(r.get("anchor") == "approved_manifest" and
                  r.get("role") in CHAIN_ROLES, "EVIDENCE_ANCHOR_INVALID", rel)
        f = root / rel
        C.require(f.is_file() and not f.is_symlink(), "EVIDENCE_MISSING", rel)
        disk = C.sha256_file(f)
        C.require(disk == r.get("sha256"), "EVIDENCE_HASH_MISMATCH",
                  f"{rel}: disk {disk} != manifest {r.get('sha256')}")
        sha[rel] = disk
    one = {k: sha.get(_safe_rel(v)) for k, v in roles.items()
           if k != "report_dir"}
    C.require(all(one.values()), "EVIDENCE_CHAIN_ROLES", "role without entry")
    rep_rel = _safe_rel(roles["report_dir"])
    listed = {p for p in sha if p.startswith(rep_rel + "/")}
    present = {f"{rep_rel}/{p.name}" for p in (root / rep_rel).iterdir()}
    C.require(listed == present, "EVIDENCE_CHAIN_BINDING",
              f"report directory files differ from the manifest: "
              f"{sorted(listed ^ present)[:4]}")
    an = A._load_once(root / roles["analysis"], one["analysis"],
                      "EVIDENCE_CHAIN_BINDING")
    C.require(an.get("config_sha256") == one["config"] and
              an.get("inputs", {}).get("difficult_sha256") ==
              one["difficult"], "EVIDENCE_CHAIN_BINDING",
              "analysis not bound to the configuration / difficult cases")
    A.load_config(root / roles["config"], one["config"])
    srcs = an.get("inputs", {}).get("identities")
    C.require(isinstance(srcs, dict) and bool(srcs), "EVIDENCE_SOURCE_MISSING",
              "analysis records no source identities")
    for k, want in sorted(srcs.items()):
        f = Path(k) if Path(k).is_absolute() else root / _safe_rel(k)
        C.require(f.is_file() and not f.is_symlink(),
                  "EVIDENCE_SOURCE_MISSING", f"analysis source {k}")
        disk = C.sha256_file(f)
        C.require(disk == want, "EVIDENCE_SOURCE_MISMATCH",
                  f"analysis source {k}: disk {disk} != recorded {want}")
    rj = R.verify_report(root / rep_rel, one["analysis"])
    sel = A.load_selection(root / roles["selection"], one["selection"])
    v = sel["validation"]
    C.require(v["config_sha256"] == one["config"] and
              v["analysis_sha256"] == one["analysis"] and
              v["report_json_sha256"] == sha[f"{rep_rel}/{R.REPORT_JSON}"] and
              v["report_sums_sha256"] == sha[f"{rep_rel}/{R.SUMS}"],
              "EVIDENCE_CHAIN_BINDING", "selection manifest not bound to the "
              "validation configuration / analysis / report")
    lc = an.get("descriptive", {}).get("learning_curves", {}).get("runs")
    C.require(isinstance(lc, dict) and set(sel["runs"]) == set(lc),
              "EVIDENCE_CHAIN_BINDING",
              "selection runs differ from the analysis runs")
    for key, run in sel["runs"].items():
        C.require(run["selected"]["epoch"] == lc[key]["selected_epoch"] and
                  run["final"]["epoch"] == lc[key]["stop_epoch"],
                  "EVIDENCE_CHAIN_BINDING", f"{key}: selected / final epoch "
                  "differ from the analysis learning curves")
    for run in sel["runs"].values():
        for ck in run.values():
            f = root / _safe_rel(ck["file"])
            C.require(f.is_file() and C.sha256_file(f) == ck["sha256"],
                      "EVIDENCE_CHAIN_BINDING", f"checkpoint {ck['file']}")
    for ck in sel["epoch0"].values():
        f = root / _safe_rel(ck["file"])
        C.require(f.is_file() and C.sha256_file(f) == ck["sha256"],
                  "EVIDENCE_CHAIN_BINDING", f"epoch-0 checkpoint {ck['file']}")
    if "holdout_config" in roles:
        hc = A.load_config(root / roles["holdout_config"],
                           one["holdout_config"])
        from seqref_mri.scripts import v05lt_stats as S
        C.require(S.config_sha256(S.validation_part(hc)) == one["config"],
                  "EVIDENCE_CHAIN_BINDING", "holdout configuration does "
                  "not extend the approved validation configuration")
    return {"entries": len(sha), "analysis_sha256": one["analysis"],
            "report_version": rj["version"]}


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="V05LT evidence checker")
    sub = ap.add_subparsers(dest="mode", required=True)
    a1 = sub.add_parser("inputs")
    a1.add_argument("--repo", default=str(C.Paths().repo))
    a1.add_argument("--zdiag", action="store_true")
    a2 = sub.add_parser("manifest")
    a2.add_argument("--manifest", required=True)
    a2.add_argument("--approved-sha256", required=True)
    a2.add_argument("--root", default=str(C.Paths().repo))
    a3 = sub.add_parser("chain")
    a3.add_argument("--manifest", required=True)
    a3.add_argument("--approved-sha256", required=True)
    a3.add_argument("--root", default=str(C.Paths().repo))
    a = ap.parse_args(argv)
    logger.info("%s v%s mode=%s", __abbr__, __version__, a.mode)
    try:
        if a.mode == "inputs":
            rows = check_inputs(C.Paths(repo=Path(a.repo)), a.zdiag)
        elif a.mode == "chain":
            res = check_chain(Path(a.manifest), a.approved_sha256,
                              Path(a.root))
            print(f"RESULT PASS chain entries={res['entries']} analysis="
                  f"{res['analysis_sha256']}")
            return 0
        else:
            rows = check_manifest(Path(a.manifest), a.approved_sha256,
                                  Path(a.root))
    except C.V05LTError as exc:
        print(f"RESULT FAIL {exc.code}")
        return 2
    except Exception as exc:  # noqa: BLE001 -- logged, never swallowed
        logger.exception("UNEXPECTED: %s", exc)
        print(f"RESULT FAIL UNEXPECTED {type(exc).__name__}")
        return 2
    for rel, sha, anchor in rows:
        print(f"EVIDENCE_OK {sha} {rel} [{anchor}]")
    print(f"RESULT PASS n={len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

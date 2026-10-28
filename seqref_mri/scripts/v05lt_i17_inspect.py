# SEQREF-MRI-V05LTI17A v0.1 -- scripts.v05lt_i17_inspect
# LIFETIME: DIAGNOSTIC
# =============================================================================
# Purpose: I1.7 addition A (SEQREF-MRI-V05LTI17 v0.2) -- READ-ONLY source and
#   input inspection before writing v05lt_eval.py:
#   S1 re-verify the existing exports (i0_src, i0c_src, i12_source) against
#      their SHA256SUMS; reuse them where a required source is identical;
#   S2 required sources for R0/R1/DC/ZF/holdout NMSE: working-tree sha256,
#      HEAD blob, last commit, identity to the pins 3393fbee / 6ffce4e5;
#      export ONLY sources not already exported (git show HEAD:path) to
#      results/v05lt/diagnostics/i17_source/ with SHA256SUMS;
#   S3 search tracked code for any existing data-consistency (DC)
#      implementation and list the k-space / operator helpers;
#   S4 pin the F10 inputs from the V02G facts record: argv paths, V02
#      checkpoints (steps 0 and 3258; file sha256 vs train record), holdout
#      and D3-subset manifests (file + internal sha256 vs V02G constants),
#      PM anchors (v02_facts.json vs V02G constants), V02H evidence, parents;
#   S5 record V02G's deterministic-backend contract and the current torch
#      settings (F10 backend decision at I1.7(a)).
# Writes ONLY the report and the i17_source export (both refuse to
#   overwrite). Reads manifests and facts files; reads NO MRI data, builds
#   NO model, loads NO checkpoint into torch (checkpoint files are hashed).
# CONVENTION: every failure logged with a code; exit 0 = report complete,
#   2 = any check failed (report still written, failures listed).
# Changelog (NEW in v0.1, 2026-10-21):
#   * Introduced under I1.7 addition A.
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("seqref_mri.v05lt_i17_inspect")
__version__ = "0.1"
__abbr__ = "SEQREF-MRI-V05LTI17A"

PINS = {"chain": "3393fbee", "metric": "6ffce4e5"}
EXPORTS = ("i0_src", "i0c_src", "i12_source")
REQUIRED = (
    "seqref_mri/scripts/v02_readout_diag.py",
    "seqref_mri/scripts/v02_readout_common.py",
    "seqref_mri/scripts/v02_eval.py",
    "seqref_mri/scripts/v02_manifests.py",
    "seqref_mri/scripts/tiny_gate.py",
    "seqref_mri/scripts/train_base.py",
    "seqref_mri/scripts/v02_train.py",
    "seqref_mri/tdiag/_bootstrap.py",
    "seqref_mri/tdiag/d2b.py",
    "seqref_mri/tdiag/d3.py",
    "seqref_mri/tdiag/estimators.py",
    "seqref_mri/src/metrics.py",
    "seqref_mri/src/free_flow_runtime.py",
    "seqref_mri/src/fastmri_data.py",
    "seqref_mri/src/forward_operator.py",
    "seqref_mri/src/residual_decoder.py",
    "seqref_mri/src/preflight_parents.py",
    "seqref_mri/scripts/_diag/dcprobe.py",
)
# Exact bodies printed in S3 (DC-relevant existing code; reuse candidates).
DC_FUNCS = (
    ("seqref_mri/src/forward_operator.py", ("A", "A_adjoint", "consistency")),
    ("seqref_mri/src/residual_decoder.py", ("decode_normalised",
                                            "measured_fixity",
                                            "scatter_unmeasured",
                                            "column_mask_tensor")),
    ("seqref_mri/src/fastmri_data.py", ("fft2c", "ifft2c")),
    ("seqref_mri/scripts/_diag/dcprobe.py", ("_outputs", "probe_variant")),
    ("seqref_mri/scripts/tiny_gate.py", ("_decode_z", "_psnr", "_nmse")),
    ("seqref_mri/tdiag/estimators.py", ("image_from_u", "z_diag_bank")),
    ("seqref_mri/tdiag/d3.py", ("_decode_bank_conditioned", "_pm_metrics")),
)
DC_CONSTS = ("_CONSISTENCY_EPS", "DATA_RANGE", "_EPS")
DC_PATTERN = (r"data[_ -]?consisten|hard[_ -]?dc|apply_dc|\bdc_|_dc\b|"
              r"\bDC\b|replace[_ ]measured|measured[_ ]columns|"
              r"keep[_ ]measured|project[_ ]onto")
V02G = "seqref_mri/scripts/v02_readout_diag.py"
CONST_NAMES = ("PM_STEP0_PSNR", "PM_STEP0_NMSE", "PM_FINAL_PSNR",
               "PM_FINAL_NMSE", "PM_BANK_SHA256", "PM_BANK_MANIFEST_SHA256",
               "D3_SUBSET_MANIFEST_SHA256", "HOLDOUT_MANIFEST_SHA256",
               "V02H_EVIDENCE_SHA256", "PSNR_REPRO_TOL_DB",
               "NMSE_REPRO_TOL_REL", "STEP0", "FINAL_STEP")


class Report:
    def __init__(self):
        self.lines, self.fails = [], []

    def add(self, s: str = "") -> None:
        self.lines.append(s)
        print(s)

    def fail(self, code: str, msg: str) -> None:
        logger.error("%s: %s", code, msg)
        self.fails.append(f"{code}: {msg}")
        self.add(f"  FAIL {code}: {msg}")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(repo: Path, *args, ok_fail: bool = False) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if r.returncode != 0:
        if ok_fail:
            return ""
        raise RuntimeError(f"git {' '.join(args)}: "
                           f"{r.stderr.decode(errors='replace').strip()}")
    return r.stdout.decode(errors="replace")


def git_bytes(repo: Path, spec: str) -> bytes | None:
    r = subprocess.run(["git", "-C", str(repo), "show", spec],
                       capture_output=True)
    return r.stdout if r.returncode == 0 else None


# ---------------------------------------------------------------------------
def s1_exports(rep: Report, diag: Path) -> dict:
    """-> {sha256: 'export/relpath'} for verified export files."""
    rep.add("== S1 existing exports ==")
    known = {}
    for name in EXPORTS:
        d = diag / name
        sums = d / "SHA256SUMS"
        if not d.is_dir():
            rep.add(f"  {name}: absent")
            continue
        if not sums.is_file():
            rep.add(f"  {name}: no SHA256SUMS (files not used for reuse)")
            continue
        bad = 0
        for line in sums.read_text().splitlines():
            if not line.strip():
                continue
            h, rel = line.split(maxsplit=1)
            rel = rel.lstrip("*")
            f = d / rel
            got = sha256_file(f) if f.is_file() else "MISSING"
            if got != h:
                bad += 1
                rep.fail("EXPORT_HASH_MISMATCH", f"{name}/{rel}")
            else:
                known.setdefault(h, f"{name}/{rel}")
        rep.add(f"  {name}: {len(known)} verified so far, {bad} mismatches")
    return known


def s2_sources(rep: Report, repo: Path, diag: Path, known: dict) -> None:
    rep.add("== S2 required sources ==")
    head = git(repo, "rev-parse", "HEAD").strip()
    rep.add(f"  repo HEAD {head}  branch "
            f"{git(repo, 'rev-parse', '--abbrev-ref', 'HEAD').strip()}")
    out = diag / "i17_source"
    to_export = []
    for rel in REQUIRED:
        p = repo / rel
        if not p.is_file():
            rep.fail("SOURCE_MISSING", rel)
            continue
        wt = sha256_file(p)
        hb = git_bytes(repo, f"HEAD:{rel}")
        head_sha = hashlib.sha256(hb).hexdigest() if hb is not None else None
        last = git(repo, "log", "-1", "--format=%H %cs", "--", rel,
                   ok_fail=True).strip()
        pin_eq = {}
        for k, c in PINS.items():
            b = git_bytes(repo, f"{c}:{rel}")
            pin_eq[k] = None if b is None else (
                hashlib.sha256(b).hexdigest() == wt)
        reuse = known.get(wt)
        rep.add(f"  {rel}\n    wt {wt}  head={'same' if head_sha == wt else head_sha}"
                f"  last {last or 'untracked'}\n    = pin chain:{pin_eq['chain']}"
                f" metric:{pin_eq['metric']}  reuse-export: {reuse or '-'}")
        if head_sha != wt:
            rep.fail("SOURCE_NOT_COMMITTED",
                     f"{rel}: working tree differs from HEAD")
        if reuse is None and hb is not None:
            to_export.append((rel, hb))
    if not to_export:
        rep.add("  nothing new to export")
        return
    if out.exists():
        rep.fail("EXPORT_EXISTS", f"{out} exists; not overwritten")
        return
    lines = []
    for rel, data in to_export:
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "xb") as fh:
            fh.write(data)
        dst.chmod(0o444)
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {rel}")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    rep.add(f"  exported {len(to_export)} sources (HEAD) to {out}")
    for ln in lines:
        rep.add(f"    {ln}")


def s3_dc(rep: Report, repo: Path) -> None:
    rep.add("== S3 data-consistency search (tracked .py, excl. V04DLT) ==")
    r = subprocess.run(["git", "-C", str(repo), "grep", "-n", "-I", "-i",
                        "-E", DC_PATTERN, "--", "seqref_mri/*.py",
                        ":(exclude)V04DLT_HOSTDEV_10cedfde"],
                       capture_output=True)
    hits = r.stdout.decode(errors="replace").splitlines()
    per_file = {}
    for h in hits:
        per_file[h.split(":", 1)[0]] = per_file.get(h.split(":", 1)[0], 0) + 1
    rep.add(f"  {len(hits)} matching lines in {len(per_file)} files:")
    for f, n in sorted(per_file.items()):
        rep.add(f"    {n:4d}  {f}")
    import ast
    for rel, names in DC_FUNCS:
        p = repo / rel
        if not p.is_file():
            rep.fail("DC_SOURCE_MISSING", rel)
            continue
        text = p.read_text()
        lines = text.splitlines()
        for ln in lines:
            if re.match(rf"^({'|'.join(DC_CONSTS)})\s*=", ln):
                rep.add(f"  {rel}: {ln.strip()}")
        found = {}
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name in names and node.name not in found:
                found[node.name] = (node.lineno, node.end_lineno)
        for name in names:
            if name not in found:
                rep.fail("DC_FUNCTION_MISSING", f"{rel}::{name}")
                continue
            a, b = found[name]
            rep.add(f"  --- {rel}::{name} (lines {a}-{b}) ---")
            for ln in lines[a - 1:b]:
                rep.add(f"  | {ln}")


def _constants(repo: Path) -> dict:
    text = (repo / V02G).read_text()
    out = {}
    for name in CONST_NAMES:
        m = re.search(rf"^{name}\s*=\s*(.+?)$", text, re.M)
        if m is None:
            m = re.search(rf"^{name}, \w+\s*=\s*(.+?)$", text, re.M)
        out[name] = m.group(1).strip() if m else None
    # multi-line parenthesised string constants
    for name in ("PM_BANK_SHA256", "PM_BANK_MANIFEST_SHA256",
                 "D3_SUBSET_MANIFEST_SHA256", "HOLDOUT_MANIFEST_SHA256",
                 "V02H_EVIDENCE_SHA256"):
        m = re.search(rf"^{name}\s*=\s*\((.*?)\)", text, re.M | re.S)
        if m:
            out[name] = "".join(re.findall(r'"([0-9a-f]+)"', m.group(1)))
    m = re.search(r"^STEP0, FINAL_STEP\s*=\s*(\d+),\s*(\d+)", text, re.M)
    if m:
        out["STEP0"], out["FINAL_STEP"] = m.group(1), m.group(2)
    return out


def _argv_value(argv: list, flag: str):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _find(obj, key):
    """Yield every value stored under `key` anywhere in a JSON object."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            yield from _find(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from _find(v, key)


def s4_f10(rep: Report, repo: Path) -> None:
    rep.add("== S4 F10 inputs (V02 checkpoints only) ==")
    c = _constants(repo)
    rep.add(f"  V02G constants: {json.dumps(c)}")
    res = repo / "seqref_mri" / "results"
    facts = sorted(p for p in res.rglob("v02_readout_facts*.json")
                   if not p.name.endswith(".sha256"))
    rep.add(f"  V02G facts files: {[str(p.relative_to(repo)) for p in facts]}")
    if not facts:
        rep.fail("V02G_FACTS_MISSING", "no v02_readout_facts*.json under results/")
        return
    for fp in facts:
        doc = json.loads(fp.read_text())
        rep.add(f"  {fp.relative_to(repo)} sha256 {sha256_file(fp)}")
        argv = (doc.get("run") or {}).get("argv") or []
        rep.add(f"    argv: {argv}")
        for k in ("repro_pm", "gate11", "pm_reproduction"):
            for v in _find(doc, k):
                rep.add(f"    {k}: {json.dumps(v)[:600]}")
        for k in ("code_provenance", "provenance"):
            for v in _find(doc, k):
                rep.add(f"    {k}: {json.dumps(v)[:400]}")
    argv = (json.loads(facts[-1].read_text()).get("run") or {}).get("argv") or []
    paths = {f: _argv_value(argv, f) for f in (
        "--data-root", "--manifest-dir", "--train-record", "--run-root",
        "--p0s-facts", "--p4-stats2", "--implb-facts", "--v02h-evidence")}
    rep.add(f"  paths from the latest facts argv: {json.dumps(paths)}")

    def resolve(v):
        if v is None:
            return None
        p = Path(v)
        return p if p.is_absolute() else repo / p
    for flag in ("--p0s-facts", "--p4-stats2", "--implb-facts",
                 "--v02h-evidence", "--train-record"):
        p = resolve(paths[flag])
        if p is None or not p.is_file():
            rep.fail("F10_INPUT_MISSING", f"{flag} {paths[flag]}")
            continue
        rep.add(f"  {flag} {p} sha256 {sha256_file(p)}")
    vh = resolve(paths["--v02h-evidence"])
    if vh is not None and vh.is_file():
        ok = sha256_file(vh) == c.get("V02H_EVIDENCE_SHA256")
        rep.add(f"  V02H evidence = V02G constant: {ok}")
        if not ok:
            rep.fail("V02H_EVIDENCE_MISMATCH", str(vh))
    md = resolve(paths["--manifest-dir"])
    for name, const in (("v02_holdout_manifest.json", "HOLDOUT_MANIFEST_SHA256"),
                        ("v02_d3_monitor_manifest.json",
                         "D3_SUBSET_MANIFEST_SHA256")):
        p = None if md is None else md / name
        if p is None or not p.is_file():
            rep.fail("F10_MANIFEST_MISSING", name)
            continue
        man = json.loads(p.read_text())
        internal = man.get("manifest_sha256")
        rep.add(f"  {name}: file {sha256_file(p)} internal {internal} "
                f"n={man.get('n')} entries={len(man.get('entries', []))} "
                f"= V02G constant: {internal == c.get(const)}")
        if internal != c.get(const):
            rep.fail("F10_MANIFEST_MISMATCH", name)
    tr = resolve(paths["--train-record"])
    rr = resolve(paths["--run-root"])
    if tr is not None and tr.is_file() and rr is not None:
        rec = json.loads(tr.read_text())
        steps = {int(x["step"]): x for x in rec.get("checkpoints", [])}
        for step in (int(c["STEP0"] or 0), int(c["FINAL_STEP"] or 3258)):
            x = steps.get(step)
            if x is None:
                rep.fail("F10_CHECKPOINT_RECORD_MISSING", str(step))
                continue
            f = rr / x["file"]
            got = sha256_file(f) if f.is_file() else "MISSING"
            rep.add(f"  checkpoint step {step}: {f} record {x['file_sha256']} "
                    f"file {got} match {got == x['file_sha256']}")
            if got != x["file_sha256"]:
                rep.fail("F10_CHECKPOINT_MISMATCH", str(f))
    facts_v02 = sorted(res.rglob("v02_facts.json"))
    for fv in facts_v02:
        doc = json.loads(fv.read_text())
        pm = (((doc.get("secondary_monitoring") or {}).get("d3_monitor")
               or {}).get("pm"))
        rep.add(f"  {fv.relative_to(repo)} sha256 {sha256_file(fv)}; "
                f"d3_monitor.pm = {json.dumps(pm)[:700]}")


def s5_backend(rep: Report, repo: Path) -> None:
    rep.add("== S5 backend contracts ==")
    text = (repo / "seqref_mri/scripts/v02_readout_common.py").read_text()
    m = re.search(r"def pin_deterministic_backend\(torch\).*?return state",
                  text, re.S)
    rep.add("  V02G pin_deterministic_backend (source):")
    for ln in (m.group(0).splitlines() if m else ["NOT FOUND"]):
        if re.search(r"torch\.|os\.environ|cublas", ln):
            rep.add(f"    {ln.strip()[:120]}")
    if m is None:
        rep.fail("BACKEND_PIN_NOT_FOUND", "v02_readout_common.py")
    import torch
    rep.add(f"  current (registered V05LT setup): threads "
            f"{torch.get_num_threads()} interop "
            f"{torch.get_num_interop_threads()} deterministic "
            f"{torch.are_deterministic_algorithms_enabled()} torch "
            f"{torch.__version__}")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="/home/benjamin/CSMFII")
    a = ap.parse_args(argv)
    repo = Path(a.repo)
    diag = repo / "seqref_mri/results/v05lt/diagnostics"
    rpt = diag / "v05lt_i17_inspect.txt"
    if rpt.exists():
        logger.error("REPORT_EXISTS: %s", rpt)
        print(f"RESULT FAIL REPORT_EXISTS {rpt}")
        return 2
    rep = Report()
    rep.add(f"{__abbr__} v{__version__} -- read-only I1.7 inspection")
    for step in (lambda: s2_sources(rep, repo, diag, s1_exports(rep, diag)),
                 lambda: s3_dc(rep, repo), lambda: s4_f10(rep, repo),
                 lambda: s5_backend(rep, repo)):
        try:
            step()
        except Exception as exc:  # noqa: BLE001 -- recorded as a failure
            logger.exception("inspection step failed")
            rep.fail("STEP_ERROR", f"{type(exc).__name__}: {exc}")
    rep.add(f"RESULT {'FAIL' if rep.fails else 'PASS'} "
            f"failures={len(rep.fails)}")
    with open(rpt, "x") as fh:
        fh.write("\n".join(rep.lines) + "\n")
    print(f"report: {rpt}  sha256 {sha256_file(rpt)}")
    return 2 if rep.fails else 0


if __name__ == "__main__":
    sys.exit(main())

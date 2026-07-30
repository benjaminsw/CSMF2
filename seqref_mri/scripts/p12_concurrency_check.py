# SEQREF-P12CC v0.1 -- P1 ∥ P2 concurrency verification (EXEC v0.4 §8, A3)
# LIFETIME: EPHEMERAL
#
# A3 requires concurrency to be "verified ONCE against isolated runs, permitting
# divergence only in volatile metadata". This script runs the pair concurrently
# and then in isolation, into EPHEMERAL output directories, and checks:
#   1. concurrent and isolated exit codes MATCH per stage, and every run
#      produced a scientific outcome (0 PASS or 1 BLOCK). A BLOCK is a verdict
#      about the data, not a concurrency defect; only ERROR (2) fails outright,
#      because an ERROR publishes no comparable facts;
#   2. semantic_sha256 is IDENTICAL concurrent vs isolated (artefact bytes are
#      NOT compared: timestamps and runtime legitimately differ);
#   3. no temporary-name collision and no residue survives either mode;
#   4. no shared mutable cache -- the two stages' facts differ in stage
#      identity and neither reads the other's output;
#   5. ARTEFACT IDENTITY ISOLATION -- a P1 facts file never carries P2's stage
#      fields and vice versa;
#   6. every published sidecar verifies in both modes.
#
# Run it in SMOKE mode (default) so it never touches the locked authoritative
# path. Delete this script and its output directories after inspection.
#
# CONVENTION: logger.error + raise on every failure path. No fallback, no mock.
#
# Changelog
#   v0.1 (2026-07-30) Created under Amendment A3 as build addition 3.

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "seqref_mri", "src"))

from preflight_io import verify_sidecar  # noqa: E402
from preflight_parents import EXIT_BLOCK, EXIT_PASS  # noqa: E402

SCRIPT_ID = "SEQREF-P12CC"
SCRIPT_VERSION = "v0.1"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(SCRIPT_ID)

STAGES = {
    "P1": ("p1_representation.py", "smoke_representation_facts", "P1"),
    "P2": ("p2_support.py", "smoke_support_facts", "P2"),
}

_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}"
          + (f"   [{detail}]" if detail and not ok else ""))


def _run(stage: str, args, out_dir: str) -> int:
    script, _, _ = STAGES[stage]
    cmd = [sys.executable, os.path.join(_HERE, script),
           "--repo-dir", args.repo_dir, "--data-root", args.data_root,
           "--p0-facts", args.p0_facts, "--p0s-facts", args.p0s_facts,
           "--p0s-script", args.p0s_script, "--out-dir", out_dir,
           "--batch", str(args.batch), "--smoke", str(args.smoke)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("%s exited %d\n%s", stage, proc.returncode,
                     proc.stderr[-2000:])
    return proc.returncode


def _read(out_dir: str, stage: str) -> dict:
    _, prefix, _ = STAGES[stage]
    path = os.path.join(out_dir, f"{prefix}.json")
    if not os.path.isfile(path):
        logger.error("%s published no facts at %s", stage, path)
        raise FileNotFoundError(path)
    verify_sidecar(path)
    with open(path, "rb") as fh:
        return json.load(fh)


def _residue(out_dir: str) -> list[str]:
    return [n for n in os.listdir(out_dir)
            if ".tmp" in n or ".claim" in n]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="SEQREF-P12CC v0.1 -- EPHEMERAL concurrency verification")
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--p0-facts", required=True)
    ap.add_argument("--p0s-facts", required=True)
    ap.add_argument("--p0s-script", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--smoke", type=int, default=8,
                    help="frozen indices per run; keeps this check off the "
                         "locked authoritative path")
    args = ap.parse_args(argv)
    print(f"{SCRIPT_ID} {SCRIPT_VERSION} -- EPHEMERAL; delete after use\n")

    with tempfile.TemporaryDirectory(prefix="p12cc_") as root:
        conc = os.path.join(root, "concurrent")
        iso1 = os.path.join(root, "isolated_p1")
        iso2 = os.path.join(root, "isolated_p2")
        for d in (conc, iso1, iso2):
            os.makedirs(d, exist_ok=True)

        print("concurrent run (P1 ∥ P2, shared output directory)")
        with cf.ThreadPoolExecutor(max_workers=2) as ex:
            rc = dict(zip(STAGES, ex.map(lambda s: _run(s, args, conc),
                                         list(STAGES))))
        check("no temporary or claim residue after the concurrent run",
              not _residue(conc), f"{_residue(conc)}")

        print("\nisolated runs")
        rc_iso = {"P1": _run("P1", args, iso1), "P2": _run("P2", args, iso2)}
        check("no residue after the isolated runs",
              not _residue(iso1) and not _residue(iso2))

        # The requirement is that concurrency does not CHANGE the outcome, not
        # that the outcome is PASS. A legitimate scientific BLOCK on a smoke
        # subset exits 1 and still publishes valid facts; reporting that as a
        # concurrency defect would be wrong. Only exit 2 (ERROR) fails this
        # check outright, because an ERROR produces no comparable facts.
        check("every run produced a scientific outcome, not an ERROR",
              all(r in (EXIT_PASS, EXIT_BLOCK)
                  for r in (*rc.values(), *rc_iso.values())),
              f"concurrent={rc} isolated={rc_iso}")
        check("concurrent and isolated exit codes MATCH per stage",
              rc["P1"] == rc_iso["P1"] and rc["P2"] == rc_iso["P2"],
              f"concurrent={rc} isolated={rc_iso}")
        if any(r == EXIT_BLOCK for r in (*rc.values(), *rc_iso.values())):
            logger.warning("at least one stage BLOCKed. That is a SCIENTIFIC "
                           "outcome, not a concurrency defect; the comparison "
                           "below still applies to the published facts.")

        c1, c2 = _read(conc, "P1"), _read(conc, "P2")
        i1, i2 = _read(iso1, "P1"), _read(iso2, "P2")

        print("\ncomparison")
        check("P1 semantic_sha256 is identical concurrent vs isolated",
              c1["semantic_sha256"] == i1["semantic_sha256"],
              f"{c1['semantic_sha256'][:16]} vs {i1['semantic_sha256'][:16]}")
        check("P2 semantic_sha256 is identical concurrent vs isolated",
              c2["semantic_sha256"] == i2["semantic_sha256"],
              f"{c2['semantic_sha256'][:16]} vs {i2['semantic_sha256'][:16]}")
        check("P1 scientific observations and verdict match exactly",
              c1["slices"] == i1["slices"] and c1["verdict"] == i1["verdict"],
              f"{c1['verdict']} vs {i1['verdict']}")
        check("P2 scientific observations and verdict match exactly",
              c2["slices"] == i2["slices"] and c2["verdict"] == i2["verdict"],
              f"{c2['verdict']} vs {i2['verdict']}")

        check("P1 and P2 carry DISTINCT stage identities",
              c1["stage"] == "P1" and c2["stage"] == "P2"
              and c1["schema"] != c2["schema"])
        check("neither facts file carries the other's stage fields",
              "rho_imag_E" not in json.dumps(c2["thresholds"])
              and "RHO_M_MAX_F32" not in json.dumps(c1["thresholds"]))
        check("the two stages published to distinct filenames",
              len({n for n in os.listdir(conc) if n.endswith(".json")}) == 2,
              f"{sorted(os.listdir(conc))}")
        check("both stages saw the SAME frozen subset and S_ref",
              c1["thresholds"]["S_ref"] == c2["thresholds"]["S_ref"]
              and c1["parents"]["p0s"]["subset_manifest_sha256"]
              == c2["parents"]["p0s"]["subset_manifest_sha256"])
        check("both runs are recorded as SMOKE, not authoritative",
              c1["run_mode"] == "smoke" and c2["run_mode"] == "smoke"
              and c1["authoritative"] is False)
        check("neither stage consumed the other's verdict",
              "P2" not in json.dumps(c1["parents"])
              and "P1" not in json.dumps(c2["parents"]))

    failed = [r for r in _RESULTS if not r[1]]
    print(f"\n{len(_RESULTS) - len(failed)}/{len(_RESULTS)} checks passed")
    if failed:
        for name, _, detail in failed:
            logger.error("FAILED: %s %s", name, detail)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

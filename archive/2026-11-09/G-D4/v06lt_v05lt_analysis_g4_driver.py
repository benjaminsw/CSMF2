# SEQREF-MRI-V06LTG4DRV v0.2 -- G-D4 driver: V05LT registered validation analysis through the pinned run_analysis
# LIFETIME: EPHEMERAL
# =============================================================================
# Run only by v06lt_v05lt_analysis_g4_v0.2.sh (which checks this file's
# sha256), with the V05LT interpreter -B -E -s from the V05LT tree.
# Orchestrates ONLY pinned V05LT functions (pin 7455da80; nothing changed):
#   bind()     Campaign._verify_chain() + verify_completion() (read-only, as
#              G-D1); recorded eligibility / ZF hashes == launch-inputs record;
#              production_layout from the recorded hashes
#   analyse()  pinned v05lt_analysis.run_analysis(layout, config, APPROVED
#              config sha256, out) -- validation only (holdout=None,
#              selection=None); output and sidecar re-hashed; log records of
#              the run collected and checked to contain no numeric values
#   finish()   pinned loaded-module rule + source identity re-checked
# The same analyse() is exercised by test_v06lt_g4.py on a completed pinned
# synthetic campaign (production=False there: the pinned code refuses a
# fixture layout or fixture configuration in production; that refusal is
# itself tested).
# Prints only RESULT PASS/REFUSED/FAILED, the refusal code, analysis sha256
# and size, completion evidence sha256. No metric values.
# Exit: 0 PASS; 1 REFUSED (typed V05LT / G4 code); 2 unexpected error.
# Changelog (v0.2, 2026-11-09; review of v0.1): driver moved out of the
#   heredoc into this hashed file with testable functions; log content check
#   over all records (root at DEBUG during the analysis; console at INFO);
#   records with a decimal value withheld from the console/log, then refused.
# =============================================================================
from __future__ import annotations

import logging
import re
import sys
import traceback
from pathlib import Path

__abbr__ = "SEQREF-MRI-V06LTG4DRV"
__version__ = "0.2"
V05LT_REPO = Path("/home/benjamin/CSMFII")
NUMERIC = re.compile(r"(?<![\w.])-?\d+\.\d+")      # any decimal number


class _Withhold(logging.Filter):
    """Console/log-file handlers drop any record containing a decimal value
    during the analysis (prevention); _Collect still counts it (refusal)."""
    def filter(self, record):
        return not NUMERIC.search(record.getMessage())


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append((record.levelname, record.name,
                             record.getMessage()))


def bind(chain: str, csha: str, elig: str, zf: str, diff: str):
    from seqref_mri.scripts import v05lt_common as C
    from seqref_mri.scripts import v05lt_launch as L
    from seqref_mri.scripts import v05lt_prodstages as PS
    from seqref_mri.scripts import v05lt_results as RS
    paths = C.Paths()
    stages = PS.ProductionStages(paths, paths.data_root)
    camp = L.Campaign(paths.v05lt, stages, chain=(Path(chain), csha,
                                                  paths.repo), fixture=False)
    camp._verify_chain()
    _ev, ev_sha = camp.verify_completion()
    ins = stages._bound()
    C.require(ins["eligibility"]["sha256"] == elig and
              ins["zf"]["sha256"] == zf, "G4_INPUT_MISMATCH",
              "recorded eligibility / ZF hashes != launch-inputs record")
    layout = RS.production_layout(paths, {"eligibility_sha256": elig,
                                          "zf_sha256": zf,
                                          "difficult_sha256": diff})
    return {"paths": paths, "stages": stages, "ins": ins, "layout": layout,
            "ev_sha": ev_sha,
            "config": paths.v05lt / "analysis" / "v05lt_analysis_config.json",
            "out_dir": paths.v05lt / "analysis"}


def analyse(layout: dict, cfg_path: Path, cfg_sha: str, out: Path, *,
            production: bool) -> dict:
    """Pinned run_analysis (validation only) + output/sidecar check + log
    content check. Returns {"sha256", "bytes", "log_records"}."""
    from seqref_mri.scripts import v05lt_analysis as A
    from seqref_mri.scripts import v05lt_common as C
    col = _Collect()
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.DEBUG)          # collect every record (console
    wh = _Withhold()                      # handler stays at INFO, see main)
    others = list(root.handlers)
    for h in others:
        h.addFilter(wh)
    root.addHandler(col)
    try:
        sha = A.run_analysis(layout, Path(cfg_path), cfg_sha, Path(out),
                             production=production)
    finally:
        root.removeHandler(col)
        for h in others:
            h.removeFilter(wh)
        root.setLevel(level)
    out = Path(out)
    side = Path(str(out) + ".sha256")
    C.require(C.sha256_file(out) == sha and side.is_file() and
              side.read_text().split() == [sha, out.name],
              "G4_OUTPUT_MISMATCH", f"{out}: output or sidecar")
    leaked = [r for r in col.records if NUMERIC.search(r[2])]
    C.require(not leaked, "G4_LOG_VALUES", f"{len(leaked)} log records "
              "with numeric values during the analysis")
    return {"sha256": sha, "bytes": out.stat().st_size,
            "log_records": col.records}


def finish(ctx: dict) -> None:
    from seqref_mri.scripts import v05lt_common as C
    from seqref_mri.scripts import v05lt_prodstages as PS
    ctx["stages"]._check_loaded(ctx["ins"]["pin_code"])
    C.require(PS.source_identity(ctx["paths"].repo) == ctx["ins"]["pin_code"],
              "PROD_PIN_MISMATCH", "execution sources changed during run")


def main(argv) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for h in logging.getLogger().handlers:
        h.setLevel(logging.INFO)
    chain, csha, cfg_sha, elig, zf, diff, rel_out = argv
    sys.path.insert(0, str(V05LT_REPO))
    from seqref_mri.scripts import v05lt_common as C
    try:
        ctx = bind(chain, csha, elig, zf, diff)
        out = ctx["paths"].repo / rel_out
        C.require(out.parent == ctx["out_dir"], "G4_OUTPUT_PATH", str(out))
        res = analyse(ctx["layout"], ctx["config"], cfg_sha, out,
                      production=True)
        finish(ctx)
        print(f"RESULT V05LT_ANALYSIS PASS analysis_sha256={res['sha256']} "
              f"bytes={res['bytes']} completion_evidence_sha256="
              f"{ctx['ev_sha']} scope=validation_only "
              f"log_records_during_analysis={len(res['log_records'])}")
        return 0
    except C.V05LTError as exc:
        print(f"RESULT V05LT_ANALYSIS REFUSED {exc.code}")
        return 1
    except Exception as exc:                   # logged, never swallowed
        logging.error("UNEXPECTED %s:\n%s", type(exc).__name__,
                      traceback.format_exc())
        print(f"RESULT V05LT_ANALYSIS FAILED UNEXPECTED {type(exc).__name__}")
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

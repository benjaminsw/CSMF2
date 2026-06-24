# =============================================================================
# STEP-2_3 v0.1 -- experiments.step_2_3.export_experts  (S2.3 V1 re-score shim)
# Purpose: after 2.3-A training, write each TRAINED CB expert back out as its OWN
#          per-expert run-dir (report.json + ckpt.pt) in the EXACT layout
#          step_1_1_1_1.build_from_report loads, so the canonical
#          step_1_3a.breakdown (which calls step_1_2.load_experts(ckpt_dirs))
#          can re-score the trained experts and produce the real RECARGMIN tier
#          (V1). We do NOT recompute the tier here -- the whole point is to feed
#          the trained experts into the SAME breakdown N3/N8 used, so the tier
#          definition stays canonical.
# CONVENTION: no fallback/mock. The trained 2.3 ckpt stores per-expert
#          state_dicts; each must reload cleanly via build_from_report or we
#          logger.error+raise. report.json['cfg'] is COPIED from the source 1.4a
#          run-dir (architecture + CB fields), with seed/out unchanged -- so the
#          rebuilt architecture matches the trained weights exactly.
# Layout written per expert (mirrors step_1_4a/run.py save):
#   <out>/<expert_tag>/report.json   # {"cfg": <source 1.4a cfg>, ...}
#   <out>/<expert_tag>/ckpt.pt        # {"expert":..., "cond":..., "base":...}
# Changelog (NEW in v0.1):
#   * export_trained_experts(s23_ckpt, s23_report, source_ckpt_dirs, out_root):
#     decompose each trained CBExpert -> (expert, cond, base) state_dicts +
#     copied source report.json -> per-expert dir. Verifies each reloads.
# Update summary:
#   v0.1 is the bridge from a {gate, experts} 2.3 ckpt to the per-expert run-dirs
#   step_1_3a.breakdown expects. No new model math; pure (de)serialization +
#   a reload self-check so a bad export fails loudly, not at breakdown time.
# =============================================================================
from __future__ import annotations
import json
import logging
import shutil
from pathlib import Path

import torch

from ..step_1_1_1_1.model_io import build_from_report

logger = logging.getLogger(__name__)
__version__ = "0.1"
__abbr__ = "STEP-2_3"


def _decompose_cbexpert_state(expert_state: dict) -> dict:
    """A trained 2.3 CBExpert.state_dict() has keys prefixed:
        expert.*   -> the inner flow (INCLUDES expert.cond.* as a submodule)
        base.*     -> the conditional base
    step_1_1_1_1.build_from_report loads THREE separate keys: 'expert', 'cond',
    'base', where 'expert' is the INNER expert state (with its own cond.* keys)
    and 'cond' is the conditioner alone. So we re-split:
        inner_expert = { strip 'expert.' prefix }   # has cond.* inside (canonical)
        cond         = { keys of inner_expert under 'cond.' , prefix stripped }
        base         = { strip 'base.' prefix }
    This mirrors exactly how step_1_4a/run.py saved (expert.state_dict() then
    cond.state_dict() -- cond appears in both, consistent on reload)."""
    inner_expert, base, cond = {}, {}, {}
    for k, v in expert_state.items():
        if k.startswith("expert."):
            sub = k[len("expert."):]
            inner_expert[sub] = v
            if sub.startswith("cond."):
                cond[sub[len("cond."):]] = v
        elif k.startswith("base."):
            base[k[len("base."):]] = v
        else:
            logger.error("[export] unexpected CBExpert key %r (not expert./base.)",
                         k)
            raise KeyError(f"unexpected CBExpert state key {k!r}")
    if not inner_expert:
        logger.error("[export] no 'expert.*' keys in trained state")
        raise KeyError("no expert.* keys -- not a CBExpert state_dict?")
    if not cond:
        logger.error("[export] no 'expert.cond.*' keys -- conditioner missing")
        raise KeyError("no expert.cond.* keys")
    if not base:
        logger.error("[export] no 'base.*' keys -- 2.3 experts are CB-wrapped")
        raise KeyError("no base.* keys (expected CB expert)")
    return {"expert": inner_expert, "cond": cond, "base": base}


def export_trained_experts(s23_ckpt_path, s23_report_path, source_ckpt_dirs,
                           out_root, device=None):
    """Write each trained 2.3 expert as a per-expert run-dir for breakdown.

    s23_ckpt_path   : the 2.3-A ckpt.pt = {"gate":..., "experts":[state,...]}.
    s23_report_path : the 2.3-A report.json (for expert_set ORDER + provenance).
    source_ckpt_dirs: the ORIGINAL 1.4a CB run-dirs the experts warm-started from,
                      SAME ORDER as expert_set -- their report.json['cfg'] is the
                      architecture spec build_from_report needs to rebuild.
    out_root        : where to write <out_root>/<source_dir_name>/{report,ckpt}.
    Returns the list of exported per-expert dirs (feed to breakdown --ckpt-dirs).
    """
    device = device or torch.device("cpu")
    s23_ckpt = torch.load(Path(s23_ckpt_path), map_location=device)
    report = json.loads(Path(s23_report_path).read_text())
    expert_set = report["expert_set"]

    if "experts" not in s23_ckpt:
        logger.error("[export] 2.3 ckpt missing 'experts' list")
        raise KeyError("2.3 ckpt has no 'experts'")
    trained = s23_ckpt["experts"]
    if not (len(trained) == len(source_ckpt_dirs) == len(expert_set)):
        logger.error("[export] length mismatch: trained=%d source_dirs=%d "
                     "expert_set=%d", len(trained), len(source_ckpt_dirs),
                     len(expert_set))
        raise ValueError("trained / source_dirs / expert_set lengths differ")

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    exported = []
    for i, (state, src_dir, name) in enumerate(
            zip(trained, source_ckpt_dirs, expert_set)):
        src = Path(src_dir)
        src_report = src / "report.json"
        if not src_report.exists():
            logger.error("[export] source report.json missing: %s", src_report)
            raise FileNotFoundError(f"{src_report} not found")
        parts = _decompose_cbexpert_state(state)
        dst = out_root / f"{src.name}__s23A"
        dst.mkdir(parents=True, exist_ok=True)
        # copy the source cfg report (architecture spec); record provenance
        rep = json.loads(src_report.read_text())
        rep["exported_from_stage_2_3"] = {
            "expert_index": i, "expert_name": name,
            "s23_ckpt": str(s23_ckpt_path), "s23_report": str(s23_report_path)}
        (dst / "report.json").write_text(json.dumps(rep, indent=2))
        torch.save(parts, dst / "ckpt.pt")
        # ---- reload self-check: a bad export must fail HERE, not in breakdown ----
        try:
            _m, _c, _cfg = build_from_report(str(dst), device)
        except Exception:
            logger.error("[export] exported %s does NOT reload via "
                         "build_from_report -- aborting", dst)
            raise
        exported.append(str(dst))
        logger.info("[export] %s (%s) -> %s [reload OK]", name, src.name, dst)

    logger.info("[export] wrote %d per-expert dirs under %s; feed to "
                "step_1_3a.breakdown --ckpt-dirs %s", len(exported), out_root,
                " ".join(exported))
    return exported


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Export trained 2.3 experts as per-expert run-dirs for breakdown")
    p.add_argument("--s23-ckpt", required=True, help="2.3-A ckpt.pt")
    p.add_argument("--s23-report", required=True, help="2.3-A report.json")
    p.add_argument("--source-ckpt-dirs", nargs="+", required=True,
                   help="ORIGINAL 1.4a CB run-dirs, ORDER matching expert_set")
    p.add_argument("--out-root", required=True)
    a = p.parse_args()
    dirs = export_trained_experts(a.s23_ckpt, a.s23_report, a.source_ckpt_dirs,
                                  a.out_root)
    print("EXPORTED_DIRS " + " ".join(dirs))


if __name__ == "__main__":
    main()

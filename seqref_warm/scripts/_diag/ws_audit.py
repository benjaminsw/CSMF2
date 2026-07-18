# SEQREF-WSAUDIT v0.1 -- ws_audit
# LIFETIME: DIAGNOSTIC
# Step 2 prep (EXEC SS3 precondition, live path item 1): DRY-RUN warm-start
# compatibility audit. Builds the refiner skeleton exactly as train_refiner
# v0.7 does (same CplRegRefiner kwargs from the same config schema), then
# calls load_warm_start with min_loaded_fraction=0.0 (audit-only floor) to
# report: loaded fraction, transferred vs fresh-init parameter counts, and
# per-tensor excluded/mismatched lists. NO training, NO x0 cache, NO run dir.
# min_loaded_fraction for the real W1 config is set FROM this audit's
# numbers, not guessed. Forward-pass check included (dummy batch).
# No fallback/mock/silent-pass. Failures: logger.error + raise.
from __future__ import annotations
import argparse
import json
import logging

import torch
import yaml

from seqref_warm.src.refiners.coupling_regressor import (CplRegRefiner,
                                                          load_warm_start,
                                                          DEFAULT_EXCLUDE)

logger = logging.getLogger("seqref_warm.ws_audit")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s :: %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="a refiner config carrying the refiner: block "
                         "(refine_w0_rnvp.yaml works -- only the refiner "
                         "block and warm-start source are read)")
    ap.add_argument("--source", required=True,
                    help="standalone checkpoint.pt to audit as warm-start "
                         "source (RealNVP: a2ed576b4171 run)")
    ap.add_argument("--out", default=None, help="optional JSON output")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    r = cfg["refiner"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # EXACT constructor mirror of train_refiner v0.7 -- do not simplify.
    model = CplRegRefiner(flavor=r["flavor"],
                          dim=int(r.get("dim", 784)),
                          h_dim=int(r.get("h_dim", 256)),
                          hidden=int(r.get("hidden", 256)),
                          n_layers=r.get("n_layers"),
                          cond_width=int(r.get("cond_width", 128)),
                          film_hidden=int(r.get("film_hidden", 128)),
                          film_depth=int(r.get("film_depth", 2)),
                          film_use_gelu=bool(r.get("film_use_gelu", True)),
                          s_max=float(r.get("s_max", 4.0)),
                          post_init_std=float(r.get("post_init_std", 1e-3)),
                          g_max=float(r.get("g_max", 0.5)),
                          g_init=float(r.get("g_init", 0.05))).to(device)

    skel = model.state_dict()
    n_skel = sum(v.numel() for v in skel.values())
    logger.info("[ws_audit] refiner skeleton: %d tensors, %d params",
                len(skel), n_skel)

    src = torch.load(args.source, map_location=device)
    src_sd = src["model"] if "model" in src else src
    n_src = sum(v.numel() for v in src_sd.values())
    logger.info("[ws_audit] source: %d tensors, %d params (%s)",
                len(src_sd), n_src, args.source)

    # Pre-audit tensor-level mapping (name+shape), independent of loader
    # policy, so the loader's report can be cross-checked.
    matched, shape_mismatch, src_only, skel_only = [], [], [], []
    for k, v in src_sd.items():
        if k in skel:
            (matched if skel[k].shape == v.shape else shape_mismatch).append(k)
        else:
            src_only.append(k)
    for k in skel:
        if k not in src_sd:
            skel_only.append(k)
    n_matched = sum(src_sd[k].numel() for k in matched)
    logger.info("[ws_audit] name+shape matched: %d tensors (%d params, "
                "%.1f%% of skeleton)", len(matched), n_matched,
                100.0 * n_matched / n_skel)
    logger.info("[ws_audit] shape-mismatched: %d | source-only: %d | "
                "skeleton-only (fresh-init): %d",
                len(shape_mismatch), len(src_only), len(skel_only))
    for k in shape_mismatch:
        logger.info("[ws_audit]   MISMATCH %s: src %s vs skel %s", k,
                    tuple(src_sd[k].shape), tuple(skel[k].shape))

    # Official loader audit with audit-only floor 0.0 (dry run must not
    # raise on low fraction -- we are here to MEASURE the fraction).
    audit = load_warm_start(model, args.source, min_loaded_fraction=0.0,
                            exclude_patterns=tuple(DEFAULT_EXCLUDE))
    logger.info("[ws_audit] load_warm_start audit: %s",
                json.dumps(audit, indent=2, default=str))

    # Forward-pass verification on a dummy batch (B=4).
    model.eval()
    with torch.no_grad():
        inp = torch.zeros(4, 3, 28, 28, device=device)
        x0 = torch.zeros(4, 1, 28, 28, device=device)
        x1, dx, g = model(inp, x0)
    logger.info("[ws_audit] forward pass OK: x1 %s dx %s g %s",
                tuple(x1.shape), tuple(dx.shape), tuple(g.shape))

    result = {"source": args.source,
              "skeleton_tensors": len(skel), "skeleton_params": n_skel,
              "source_tensors": len(src_sd), "source_params": n_src,
              "matched_tensors": len(matched), "matched_params": n_matched,
              "matched_param_fraction_of_skeleton": n_matched / n_skel,
              "shape_mismatch": shape_mismatch, "source_only": src_only,
              "skeleton_only_fresh_init": skel_only,
              "loader_audit": audit,
              "default_exclude": list(DEFAULT_EXCLUDE)}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info("[ws_audit] written: %s", args.out)


if __name__ == "__main__":
    main()

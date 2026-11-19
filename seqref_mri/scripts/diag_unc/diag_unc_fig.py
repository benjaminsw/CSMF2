# SEQREF-DUNC-FIG v0.2 -- G-FIG: mechanical figures F1-F9 of DIAG-UNC from the verified outputs of one G run
# LIFETIME: KEEP
"""G-FIG for DIAG-UNC (mini-spec 223 section 9), implementing figure specification SEQREF-DUNC-FIGS v0.2 with its
addendum FIGS AM1 only.

Reads, after hash verification: <g-run>/out/manifest_g.json, the nine figure-data files, the registered
F2 manifest (contrast order), the stage-1 final index and state indexes (integrity metadata only) and the
map files of the fixed slices.  Computes no scientific statistic: the only derived quantities are
visualisation-only (axis limits, reference percentile, map limits, clipped fraction, saturation ratio).
Writes, once each, with sha256 sidecars: figures (PNG + PDF), manifest_fig.json, fig.log, RESULT.txt.
The log and the terminal carry names, counts and hashes only.  No fallback, placeholder or mock path:
every missing or mismatched input is an explicit FAIL (exit 1).

Run:  python -B -E -s <run>/code/diag_unc_fig.py --g-run <g run dir> --manifest-g-sha <sha256>
      --figs-sha <sha256 of the figure specification> [--selftest]

Changelog
- v0.2 (2026-11-19): --figs-sha must equal the registered sha256 of FIGS v0.2, checked before any input is read;
  every reference image is checked for non-finite values before its percentile is taken; F3 prints the tie
  fractions stored per coverage cell in fd_coverage (FIGS AM1-2).  No figure rule, scale or order changed.
- v0.1 (2026-11-19): first version.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import textwrap
import time
import traceback

VERSION = "SEQREF-DUNC-FIG v0.2"
SPEC = "SEQREF-DUNC-FIGS v0.2 + AM1"
FIGS_SHA = "bbe3774dd80e4b3b04993b246ef6f406fb313fc15d59d845f67f76bb6c6aa1e0"   # FIGS v0.2, frozen in 232 U67
FIGDATA = ("fd_histories", "fd_dgap", "fd_rankhist", "fd_coverage", "fd_crps", "fd_band", "fd_maps", "fd_convergence", "fd_deciles")
ARM_COLOUR = {"lam0": "#0072B2", "lam2": "#D55E00"}           # Okabe-Ito (FIGS section 2)
NEUTRAL, REFERENCE = "#000000", "#999999"
SEEDS = (2026, 2027, 2028)
SEED_MARKER = {2026: "o", 2027: "^", 2028: "s"}
ROLE_STYLE = {"e3": ":", "e10": "-.", "e30": "--", "sel": "-", "fin": (0, (5, 1))}
ROLE_EPOCH_KEYS = ("epoch3", "epoch10", "epoch30", "selected", "final")
LEVELS = ("0.50", "0.80", "0.90", "0.95")
SPACES = ("U", "M")
SUBSETS = ("pre16", "pre32", "pre64", "full", "halfA", "halfB")   # the stored subsets, fixed order
BANDS = ("low", "mid", "high")
UNSTABLE = "Monte Carlo adequacy unresolved"
MAP_FRACTION = {"reference": 1.0, "error": 0.20, "sd": 0.10}        # FIGS section 4 (approved, frozen)
MAP_PERCENTILE = 99.5
DPI = 200


class Fail(Exception):
    """An explicit, typed failure of G-FIG."""


class Log:
    def __init__(self, path):
        self.path, self.lines = path, []

    def __call__(self, level, msg):
        line = "%s %s" % (level, msg)
        self.lines.append(line)
        print(line, flush=True)

    def close(self):
        data = ("\n".join(self.lines) + "\n").encode("utf-8")
        write_once(self.path, data)
        write_once(self.path + ".sha256", (sha_bytes(data) + "\n").encode("ascii"))
        return sha_bytes(data)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_once(path, data):
    with open(path, "xb") as fh:                              # 'x': never overwrite
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def need(cond, msg):
    if not cond:
        raise Fail(msg)


def read_verified(path, expected, what):
    """Return the bytes of `path` after checking them against `expected` and against the sidecar."""
    need(os.path.isfile(path), "%s: file missing (%s)" % (what, os.path.basename(path)))
    with open(path, "rb") as fh:
        data = fh.read()
    got = sha_bytes(data)
    need(got == expected, "%s: sha256 differs from the registered value (%s)" % (what, os.path.basename(path)))
    side = path + ".sha256"
    need(os.path.isfile(side), "%s: sidecar missing (%s)" % (what, os.path.basename(side)))
    with open(side, "r", encoding="ascii") as fh:
        tok = fh.read().split()
    need(bool(tok) and tok[0] == got, "%s: sidecar differs from the file (%s)" % (what, os.path.basename(side)))
    return data


def strict_json(data, what):
    def bad(tok):
        raise Fail("%s: non-finite JSON token %s" % (what, tok))
    return json.loads(data.decode("utf-8"), parse_constant=bad)


# ----------------------------------------------------------------------------------------------- verification
def verify_inputs(args, log):
    g = os.path.abspath(args.g_run)
    out = os.path.join(g, "out")
    need(os.path.isdir(out), "G run directory has no out/")
    for stop in ("stop_record_watchdog.json", "RESULT_INVALID.txt", "stop_record.json"):
        need(not os.path.exists(os.path.join(g, stop)), "G run directory holds %s" % stop)
    res = os.path.join(g, "RESULT.txt")
    need(os.path.isfile(res), "G run has no RESULT.txt")
    with open(res, "r", encoding="utf-8") as fh:
        rl = fh.read().splitlines()
    first = "G_SELFTEST_RESULT: PASS" if args.selftest else "G_RESULT: PASS"
    need(bool(rl) and rl[0] == first, "RESULT.txt of the G run does not start with '%s'" % first)
    need(("manifest_sha256 " + args.manifest_g_sha) in rl, "RESULT.txt of the G run does not name the given manifest sha256")
    M = strict_json(read_verified(os.path.join(out, "manifest_g.json"), args.manifest_g_sha, "manifest_g.json"), "manifest_g.json")
    need(M.get("mode") == "g" and M.get("result") == "PASS", "manifest_g.json is not a PASS manifest of mode g")
    need(M.get("selftest") is bool(args.selftest), "self-test flag of the G manifest differs from this invocation")
    log("INFO", "[IN] PASS  G run verified: RESULT.txt, no stop evidence, manifest_g.json %s (mode g, PASS, selftest %s)" % (args.manifest_g_sha, M["selftest"]))
    FD = {}
    for name in FIGDATA:
        rel = "figdata/%s.json" % name
        need(rel in M["outputs"], "manifest_g.json does not list %s" % rel)
        doc = strict_json(read_verified(os.path.join(out, rel), M["outputs"][rel], rel), rel)
        need(set(doc) == {"data", "meta"}, "%s: unexpected top-level keys" % rel)
        need(doc["meta"].get("manifest_f2_sha256") == M["manifest_f2_sha256"] and doc["meta"].get("selftest") is bool(args.selftest),
             "%s: meta does not match the G manifest" % rel)
        FD[name] = doc
    log("INFO", "[IN] PASS  %d figure-data files == manifest_g.json == sidecars; meta consistent" % len(FD))
    f2p = os.path.join(g, "code", "manifest_f2.json")
    need(os.path.isfile(f2p) and sha_file(f2p) == M["manifest_f2_sha256"], "registered F2 manifest missing or different from manifest_g.json")
    with open(f2p, "rb") as fh:
        f2 = strict_json(fh.read(), "manifest_f2.json")
    contrasts = list(f2["contrasts"]["tier1"]) + list(f2["contrasts"]["tier2"])
    for name in ("fd_dgap", "fd_crps", "fd_band", "fd_convergence"):
        need(set(FD[name]["data"]) == set(contrasts), "%s: contrast set differs from the F2 manifest" % name)
    checkpoints = sorted(FD["fd_coverage"]["data"])
    for name in ("fd_rankhist", "fd_deciles"):
        need(sorted(FD[name]["data"]) == checkpoints, "%s: checkpoint set differs from fd_coverage" % name)
    log("INFO", "[IN] PASS  %d contrasts in the registered order of the F2 manifest; %d checkpoint keys" % (len(contrasts), len(checkpoints)))
    # maps: final index -> state indexes -> map entries -> files (integrity metadata only; no record is opened)
    idx = strict_json(read_verified(os.path.join(out, "records", "index.json"), M["stage1_index_sha256"], "records/index.json"), "records/index.json")
    reg, roles = {}, {}
    for st in idx["states"]:
        sdoc = strict_json(read_verified(os.path.join(out, st["file"]), st["sha256"], st["file"]), st["file"])
        si = sdoc["state_index"]
        roles[si] = list(sdoc["roles"])
        for rec in sdoc["records"]:
            if "maps" in rec:
                reg[(si, rec["position"])] = (rec["maps"], rec.get("reference"))
    fixed = [f["position"] for f in FD["fd_maps"]["data"]["fixed_slices"]]
    need(fixed == [f["position"] for f in idx["fixed_slices"]], "fd_maps: fixed slices differ from the final index")
    files = FD["fd_maps"]["data"]["files"]
    need(len(files) == len(idx["states"]) * len(fixed) == len(reg), "fd_maps: number of map entries differs from states x fixed slices")
    maps, refs, npix = {}, {}, None
    for ent in files:
        key = (ent["state_index"], ent["position"])
        need(key in reg and reg[key][0]["file"] == ent["maps"]["file"] and reg[key][0]["sha256"] == ent["maps"]["sha256"],
             "fd_maps: entry of state %d, position %d differs from its state index" % key)
        data = read_verified(os.path.join(out, ent["maps"]["file"]), ent["maps"]["sha256"], ent["maps"]["file"])
        need(len(data) % 12 == 0 and len(data) > 0, "%s: size is not 3 x N float32" % ent["maps"]["file"])
        n = len(data) // 12
        need(npix in (None, n), "map files differ in size")
        npix = n
        maps[key] = data
        ref = ent.get("reference")                              # stored once per fixed slice (with the first state)
        need((ref is None) == (reg[key][1] is None), "fd_maps: reference entry of state %d, position %d differs from its state index" % key)
        if ref is not None:
            need(reg[key][1]["sha256"] == ref["sha256"] and reg[key][1]["file"] == ref["file"], "fd_maps: reference of position %d differs from its state index" % key[1])
            need(key[1] not in refs, "fd_maps: more than one reference entry for position %d" % key[1])
            rdata = read_verified(os.path.join(out, ref["file"]), ref["sha256"], ref["file"])
            need(len(rdata) == 4 * n, "%s: size differs from the maps" % ref["file"])
            refs[key[1]] = (rdata, ref["sha256"])
    side = math.isqrt(npix)
    need(side * side == npix, "map size %d is not a square image; no geometry is assumed" % npix)
    need(set(refs) == set(fixed), "reference files do not cover the fixed slices")
    log("INFO", "[MAPS] PASS  %d map files and %d reference files == fd_maps == state indexes == sidecars; %d x %d pixels" % (len(maps), len(refs), side, side))
    return {"M": M, "FD": FD, "contrasts": contrasts, "checkpoints": checkpoints, "maps": maps, "refs": refs, "fixed": fixed,
            "roles": roles, "side": side, "n_states": len(idx["states"]), "g": g}


# ----------------------------------------------------------------------------------------------- drawing helpers
def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def round_up_2sig(x):
    if not x > 0:
        return 1.0
    e = math.floor(math.log10(x)) - 1
    return math.ceil(x / 10 ** e - 1e-9) * 10 ** e


def sym_limit(effects):
    """Half-range = 1.1 x the largest absolute interval endpoint or per-seed value drawn, rounded up to 2 significant figures."""
    vals = []
    for ef in effects:
        if ef is None:
            continue
        for v in [ef.get("point")] + list(ef.get("ci") or []) + list(ef.get("per_seed") or []):
            if finite(v):
                vals.append(abs(v))
    return round_up_2sig(1.1 * max(vals)) if vals and max(vals) > 0 else 1.0


def pad_limits(values):
    vals = [v for v in values if finite(v)]
    if not vals:
        return (0.0, 1.0)
    lo, hi = min(vals), max(vals)
    d = (hi - lo) * 0.05 if hi > lo else (abs(hi) * 0.05 or 0.5)
    return (lo - d, hi + d)


def computed(ef):
    return ef is not None and finite(ef.get("point")) and ef.get("ci") is not None and all(finite(v) for v in ef["ci"])


def forest(ax, names, effects, lim, labels=None, unstable=None, show_names=True, title=None):
    """Paired effects: filled diamond + 95 % interval; per-seed open markers; stored label at the right margin."""
    n = len(names)
    ax.axvline(0.0, color=REFERENCE, lw=0.8, zorder=1)
    for i, (name, ef) in enumerate(zip(names, effects)):
        y = n - 1 - i
        un = bool(unstable and unstable[i])
        if not computed(ef):
            ax.axhspan(y - 0.4, y + 0.4, facecolor="none", edgecolor=REFERENCE, hatch="////", lw=0.0, zorder=0)
            ax.text(0.0, y, "not computed", ha="center", va="center", fontsize=6, color=NEUTRAL, zorder=3)
        else:
            ax.plot(ef["ci"], [y, y], color=NEUTRAL, lw=1.2, zorder=2)
            ax.plot([ef["point"]], [y], marker="D", ms=5, mfc=("none" if un else NEUTRAL), mec=NEUTRAL, zorder=3)
            for seed, v in zip(SEEDS, ef.get("per_seed") or []):
                if finite(v):
                    ax.plot([v], [y + 0.26], marker=SEED_MARKER[seed], ms=3, mfc="none", mec=NEUTRAL, mew=0.6, zorder=3)
        txt = (labels[i] if labels else ((ef or {}).get("label") or "not computed")) + (" †" if un else "")
        ax.text(1.01, y, txt, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=5.5, color=NEUTRAL)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(-lim, lim)
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(reversed(names)) if show_names else [""] * n, fontsize=6.5)
    ax.tick_params(axis="x", labelsize=6)
    if title:
        ax.set_title(title, fontsize=8)


def caption_text(S, ident, what):
    st = S["FD"]["fd_dgap"]["meta"]["statement"]
    parts = ["%s." % what,
             "Population: %s." % st["population"], "Seeds: %s." % st["seeds"], "Interval: %s." % st["interval"],
             "Multiplicity: %s." % st["multiplicity"], "Validation reuse: %s." % st["validation_reuse"],
             "Selection limitation: %s." % st["selection_limitation"],
             "Markers: filled diamond = pooled estimate with its 95 % interval; open diamond and † = Monte Carlo adequacy unresolved; "
             "small open markers = seeds 2026 (circle), 2027 (triangle), 2028 (square); hatched row = not computed. "
             "Colours: λ₀ blue, λ₂ vermilion; contrasts black. No colour encodes a direction or a label.",
             "Run: %s; manifest_g.json %s; %s; %s%s." % (ident["run"], ident["manifest"], VERSION, SPEC, " (SELF-TEST, NOT G EVIDENCE)" if ident["selftest"] else "")]
    return "\n".join(textwrap.fill(p, 190) for p in parts)


class Figures:
    def __init__(self, S, outdir, log, plt):
        self.S, self.outdir, self.log, self.plt = S, outdir, log, plt
        self.written, self.visual = {}, {}
        self.ident = {"run": os.path.basename(S["g"]), "manifest": S["M"] and sha_file(os.path.join(S["g"], "out", "manifest_g.json")), "selftest": S["M"]["selftest"]}

    def save(self, fig, name, what, visual=None):
        cap = caption_text(self.S, self.ident, what)
        nlines = cap.count("\n") + 1
        h = fig.get_size_inches()[1]
        bottom = min(0.45, (0.115 * nlines + 0.35) / h)
        fig.tight_layout(rect=(0.0, bottom, 1.0, 0.975))
        fig.text(0.01, 0.004, cap, fontsize=5.5, va="bottom", ha="left", color=NEUTRAL, linespacing=1.15)
        for ext, kw in (("png", {"dpi": DPI, "metadata": {"Software": VERSION}}), ("pdf", {"metadata": {"CreationDate": None, "ModDate": None, "Producer": VERSION, "Creator": VERSION}})):
            path = os.path.join(self.outdir, "%s.%s" % (name, ext))
            need(not os.path.exists(path), "output exists already: %s" % os.path.basename(path))
            tmp = path + ".part"
            fig.savefig(tmp, format=ext, **kw)
            with open(tmp, "rb") as fh:
                data = fh.read()
            os.remove(tmp)
            write_once(path, data)
            write_once(path + ".sha256", (sha_bytes(data) + "\n").encode("ascii"))
            self.written["%s.%s" % (name, ext)] = sha_bytes(data)
        self.plt.close(fig)
        self.visual[name] = visual or {}
        self.log("INFO", "[FIG] PASS  %s written (png, pdf) with sidecars" % name)

    # ---- F1
    def f1(self):
        runs = self.S["FD"]["fd_histories"]["data"]["runs"]
        cols = None
        for r in runs.values():
            need(cols in (None, r["columns"]), "fd_histories: runs differ in columns")
            cols = r["columns"]
        maxep = max(e["epoch"] for r in runs.values() for e in r["epochs"])
        fig, axes = self.plt.subplots(len(cols), len(SEEDS), figsize=(11.5, 2.5 * len(cols) + 2.4), squeeze=False, sharex=True)
        lims = {}
        for qi, q in enumerate(cols):
            vals = [v[qi] for r in runs.values() for e in r["epochs"] for v in (e["midpoint"], e["all_slices"])]
            lims[q] = pad_limits(vals)
            for si, seed in enumerate(SEEDS):
                ax = axes[qi][si]
                for arm in ("lam0", "lam2"):
                    key = "%s|%d" % (arm, seed)
                    need(key in runs, "fd_histories: run %s missing" % key)
                    ep = [e["epoch"] for e in runs[key]["epochs"]]
                    ax.plot(ep, [e["midpoint"][qi] for e in runs[key]["epochs"]], color=ARM_COLOUR[arm], lw=1.1, ls="-")
                    ax.plot(ep, [e["all_slices"][qi] for e in runs[key]["epochs"]], color=ARM_COLOUR[arm], lw=0.9, ls="--")
                    for rk in ROLE_EPOCH_KEYS:
                        if rk in runs[key]["roles"]:
                            x = runs[key]["roles"][rk]
                            ax.axvline(x, color=ARM_COLOUR[arm], lw=0.5, ls=":", zorder=0)
                            ax.text(x, 1.0 if arm == "lam0" else 0.93, rk, transform=ax.get_xaxis_transform(), rotation=90, fontsize=4.5,
                                    ha="right", va="top", color=ARM_COLOUR[arm])
                ax.set_xlim(1, maxep)
                ax.set_ylim(*lims[q])
                ax.tick_params(labelsize=6)
                if qi == 0:
                    ax.set_title("seed %d" % seed, fontsize=8)
                if si == 0:
                    ax.set_ylabel(q, fontsize=7)
                if qi == len(cols) - 1:
                    ax.set_xlabel("epoch", fontsize=7)
        self.save(fig, "F1_histories", "F1 - training histories with checkpoint roles, both arms on common axes; solid = the midpoint slices, dashed = all validation slices; "
                  "dotted verticals = role epochs of each arm", {"y_limits": lims, "x_limits": [1, maxep]})

    # ---- F2
    def f2(self):
        D, C = self.S["FD"]["fd_dgap"]["data"], self.S["contrasts"]
        for name, qs, unit, title in (("F2_dgap_nll", ("nll", "B", "J"), "nats/dim", "F2 - paired change of NLL and of its two additive parts per contrast"),
                                      ("F2b_dgap_psnr", ("psnr_r0", "psnr_r1"), "dB", "F2b - paired change of R0 and R1 PSNR per contrast")):
            lim = sym_limit([D[c]["effects"][q] for c in C for q in qs])
            fig, axes = self.plt.subplots(1, len(qs), figsize=(4.6 * len(qs) + 1.2, 0.33 * len(C) + 3.6), squeeze=False)
            for j, q in enumerate(qs):
                if q in ("nll", "psnr_r0", "psnr_r1"):
                    labs = [D[c]["labels"][q] + ("; gap pattern" if (q == "nll" and D[c]["labels"]["gap_pattern"]) else "")
                            + ("; 0.5 dB: " + D[c]["labels"]["practical_" + q[5:]] if q.startswith("psnr") else "") for c in C]
                else:
                    labs = None
                forest(axes[0][j], C, [D[c]["effects"][q] for c in C], lim, labels=labs, show_names=(j == 0), title="%s (%s)" % (q, unit))
                if q.startswith("psnr"):
                    if lim >= 0.5:
                        axes[0][j].axvline(0.5, color=REFERENCE, lw=0.8, ls="--")
                    else:
                        axes[0][j].text(0.99, 0.01, "+0.5 dB reference lies outside the axis", transform=axes[0][j].transAxes, ha="right", va="bottom", fontsize=5.5, color=REFERENCE)
            extra = ""
            if name == "F2_dgap_nll":
                extra = ". Additive share as stored: " + "; ".join("%s: %s" % (c, D[c]["share"] if isinstance(D[c]["share"], str) else json.dumps(D[c]["share"])) for c in C)
            self.save(fig, name, title + "; panels share one axis" + extra, {"half_range": lim})

    # ---- F3
    def f3(self):
        R, K = self.S["FD"]["fd_rankhist"]["data"], self.S["checkpoints"]
        Cv = self.S["FD"]["fd_coverage"]["data"]
        top = max(v for k in K for sp in SPACES for v in R[k][sp])
        fig, axes = self.plt.subplots(len(K), len(SPACES), figsize=(9.5, 1.25 * len(K) + 2.6), squeeze=False, sharex=True)
        for i, k in enumerate(K):
            for j, sp in enumerate(SPACES):
                ax, h = axes[i][j], R[k][sp]
                ax.bar(range(len(h)), h, width=1.0, color=ARM_COLOUR[k.split("/")[0]], lw=0)
                ax.axhline(sum(h) / len(h), color=REFERENCE, lw=0.8)
                ax.set_ylim(0, top * 1.05)
                ax.tick_params(labelsize=5.5)
                ax.set_ylabel(k, fontsize=6)
                ties = [Cv[k]["calibration"]["%s|%s" % (sp, lv)]["tie_fraction"] for lv in LEVELS]
                need(all(finite(t) for t in ties), "fd_coverage: tie fraction missing for %s, space %s" % (k, sp))
                ax.text(0.5, 0.97, "tie fraction at levels %s: %s" % ("/".join(LEVELS), " / ".join("%.3g" % t for t in ties)), transform=ax.transAxes,
                        ha="center", va="top", fontsize=5, color=NEUTRAL)
                if i == 0:
                    ax.set_title("space %s" % sp, fontsize=8)
                if i == len(K) - 1:
                    ax.set_xlabel("rank of the reference among the decodes", fontsize=7)
        self.save(fig, "F3_rank_histograms", "F3 - rank histograms per checkpoint key (sorted order) and space; grey line = the uniform expectation (total count / number of bins); "
                  "common y-axis from 0. Tie fractions are printed as stored: one per coverage level of that checkpoint and space (from the coverage figure data)", {"y_max": top * 1.05})

    # ---- F4
    def f4(self):
        Cv, K = self.S["FD"]["fd_coverage"]["data"], self.S["checkpoints"]
        fig, axes = self.plt.subplots(1, 2, figsize=(10.5, 5.6), squeeze=False)
        for j, sp in enumerate(SPACES):
            ax = axes[0][j]
            ax.plot([0, 1], [0, 1], color=REFERENCE, lw=0.8)
            for k in K:
                arm, role = k.split("/")
                need(role in ROLE_STYLE, "fd_coverage: unknown role %s" % role)
                cells = [Cv[k]["calibration"]["%s|%s" % (sp, lv)] for lv in LEVELS]
                ax.plot([c["pi"] for c in cells], [c["coverage"] for c in cells], color=ARM_COLOUR[arm], ls=ROLE_STYLE[role], lw=1.0, marker="o", ms=2.5, label=k)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_xlabel("attainable level", fontsize=7)
            ax.set_ylabel("empirical coverage", fontsize=7)
            ax.set_title("space %s" % sp, fontsize=8)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=5.5, loc="upper left", frameon=False)
        self.save(fig, "F4a_coverage_vs_level", "F4a - empirical coverage against the attainable level, per space; one line per checkpoint key (colour = arm, line style = role); "
                  "grey diagonal = perfect calibration; axes [0, 1]")
        fig, axes = self.plt.subplots(len(SPACES), len(LEVELS), figsize=(12.5, 6.4), squeeze=False)
        vis = {}
        for i, sp in enumerate(SPACES):
            for j, lv in enumerate(LEVELS):
                ax, cell = axes[i][j], "%s_%s" % (sp, lv)
                ws = []
                for k in K:
                    arm, role = k.split("/")
                    w, cal = Cv[k]["width"][cell]["point"], Cv[k]["calibration"]["%s|%s" % (sp, lv)]
                    ws.append(w)
                    ax.plot([w], [cal["coverage"]], marker="o", ms=4, mfc="none", mec=ARM_COLOUR[arm], mew=1.0)
                    ax.annotate(role, (w, cal["coverage"]), fontsize=4.5, color=ARM_COLOUR[arm], xytext=(2, 2), textcoords="offset points")
                    pi = cal["pi"]
                ax.axhline(pi, color=REFERENCE, lw=0.8)
                ax.set_xlim(*pad_limits(ws))
                ax.set_ylim(0, 1)
                ax.set_title("space %s, level %s" % (sp, lv), fontsize=7.5)
                ax.tick_params(labelsize=6)
                ax.set_xlabel("mean interval width", fontsize=6.5)
                if j == 0:
                    ax.set_ylabel("empirical coverage", fontsize=6.5)
                vis[cell] = list(pad_limits(ws))
        self.save(fig, "F4b_coverage_vs_width", "F4b - empirical coverage against mean interval width, one panel per cell (space x level), one point per checkpoint key (colour = arm, "
                  "text = role); grey line = the attainable level", {"x_limits": vis})

    # ---- F5
    def f5(self):
        D, C = self.S["FD"]["fd_crps"]["data"], self.S["contrasts"]
        ms = ("ucrps", "mcrps", "energy")
        fig, axes = self.plt.subplots(1, 3, figsize=(15.0, 0.33 * len(C) + 3.6), squeeze=False)
        vis = {}
        for j, m in enumerate(ms):
            lim = sym_limit([D[c][m]["effect"] for c in C])
            vis[m] = lim
            forest(axes[0][j], C, [D[c][m]["effect"] for c in C], lim, unstable=[D[c][m]["mc_stability"] == UNSTABLE for c in C], show_names=(j == 0),
                   title={"ucrps": "U-CRPS", "mcrps": "M-CRPS", "energy": "energy score"}[m])
        self.save(fig, "F5_crps_effects", "F5 - paired effects of the fair CRPS in U and M and of the fair energy score per contrast; each panel has its own axis (different units); "
                  "label = sign label of the 95 % interval, with the Monte Carlo stability mark", {"half_range": vis})

    # ---- F6
    def f6(self):
        D, C = self.S["FD"]["fd_band"]["data"], self.S["contrasts"]
        lim_c = sym_limit([D[c]["band_crps"][b] for c in C for b in BANDS])
        lim_e = sym_limit([e for c in C for r in ("band_err_r0", "band_err_r1") for e in D[c]["band_error"][r]])
        fig, axes = self.plt.subplots(1, 9, figsize=(27.0, 0.33 * len(C) + 3.6), squeeze=False)
        for j, b in enumerate(BANDS):
            forest(axes[0][j], C, [D[c]["band_crps"][b] for c in C], lim_c, show_names=(j == 0), title="band U-CRPS, %s" % b)
        j = 3
        for r in ("band_err_r0", "band_err_r1"):
            for bi, b in enumerate(BANDS):
                forest(axes[0][j], C, [D[c]["band_error"][r][bi] for c in C], lim_e, show_names=False, title="matched band error %s, %s" % (r[-2:].upper(), b))
                j += 1
        self.save(fig, "F6_band_panel", "F6 - band-resolved U-CRPS effects (three panels, one shared axis) and matched band error for R0 and R1 as stored (six panels, one shared axis), per contrast",
                  {"half_range_crps": lim_c, "half_range_error": lim_e})

    # ---- F7
    def f7(self, np):
        S = self.S
        side = S["side"]
        for pos in S["fixed"]:
            ref = np.frombuffer(S["refs"][pos][0], dtype="<f4").reshape(side, side)
            need(bool(np.isfinite(ref).all()), "reference image of position %d holds a non-finite value" % pos)
            P = float(np.percentile(ref, MAP_PERCENTILE))
            need(math.isfinite(P) and P > 0, "reference image of position %d: the %.1fth percentile is not positive" % (pos, MAP_PERCENTILE))
            lim = {k: MAP_FRACTION[k] * P for k in MAP_FRACTION}
            n = S["n_states"]
            fig, axes = self.plt.subplots(n, 4, figsize=(7.6, 1.62 * n + 3.2), squeeze=False)
            clip = {}
            for si in range(n):
                arr = np.frombuffer(S["maps"][(si, pos)], dtype="<f4").reshape(3, side, side)
                need(bool(np.isfinite(arr).all()), "map of state %d, position %d holds a non-finite value" % (si, pos))
                panels = ((ref, "reference", "reference magnitude"), (arr[0], "sd", "M-space SD"), (arr[1], "error", "|m_R0 - t|"), (arr[2], "error", "||mean x| - t|"))
                for j, (img, kind, ttl) in enumerate(panels):
                    ax = axes[si][j]
                    ax.imshow(img, cmap="gray", vmin=0.0, vmax=lim[kind], interpolation="nearest")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    frac, ratio = float((img > lim[kind]).mean()), float(img.max() / lim[kind])
                    clip["s%02d_%d" % (si, j)] = [frac, ratio]
                    ax.set_xlabel("clipped %.2g; max/limit %.2g" % (frac, ratio), fontsize=4.5, labelpad=1)
                    if si == 0:
                        ax.set_title(ttl, fontsize=7)
                    if j == 0:
                        ax.set_ylabel("\n".join(r.replace("|", " ") for r in S["roles"][si]), fontsize=4.8, rotation=0, ha="right", va="center", labelpad=4)
            self.save(fig, "F7_maps_position_%02d" % pos, "F7 - reference magnitude, M-space SD and two absolute error maps for fixed slice position %d, all %d model states in the registered "
                      "processing order (rows: arm, seed, role). Linear grey scales fixed from the reference image only: P = its %.1fth percentile; reference [0, P], error maps [0, %.2f P], "
                      "SD map [0, %.2f P]; values above the limit are clipped; each panel prints the clipped fraction and (panel maximum / limit)"
                      % (pos, n, MAP_PERCENTILE, MAP_FRACTION["error"], MAP_FRACTION["sd"]), {"P": P, "limits": lim, "clipping": clip})

    # ---- F8
    def f8(self):
        D, C = self.S["FD"]["fd_convergence"]["data"], self.S["contrasts"]
        for m, ttl in (("ucrps", "U-CRPS"), ("mcrps", "M-CRPS"), ("energy", "energy score")):
            for c in C:
                need(set(D[c][m]) <= set(SUBSETS), "fd_convergence: unknown subset for %s, %s" % (c, m))
            lim = sym_limit([D[c][m][s] for c in C for s in SUBSETS if s in D[c][m]])
            ncol = 7
            nrow = math.ceil(len(C) / ncol)
            fig, axes = self.plt.subplots(nrow, ncol, figsize=(2.55 * ncol, 2.2 * nrow + 2.6), squeeze=False, sharey=True)
            for i, c in enumerate(C):
                ax = axes[i // ncol][i % ncol]
                ax.axhline(0.0, color=REFERENCE, lw=0.8)
                for x, s in enumerate(SUBSETS):
                    ef = D[c][m].get(s)
                    if s not in D[c][m]:
                        ax.text(x, 0.0, "n/s", ha="center", va="center", fontsize=4.5, color=REFERENCE)
                    elif not computed(ef):
                        ax.axvspan(x - 0.4, x + 0.4, facecolor="none", edgecolor=REFERENCE, hatch="////", lw=0.0)
                    else:
                        ax.plot([x, x], ef["ci"], color=NEUTRAL, lw=1.0)
                        ax.plot([x], [ef["point"]], marker="D", ms=3.5, color=NEUTRAL)
                ax.set_xlim(-0.6, len(SUBSETS) - 0.4)
                ax.set_ylim(-lim, lim)
                ax.set_xticks(range(len(SUBSETS)))
                ax.set_xticklabels(SUBSETS, fontsize=4.8, rotation=60)
                ax.tick_params(axis="y", labelsize=5.5)
                ax.set_title(c, fontsize=6.5)
            for k in range(len(C), nrow * ncol):
                axes[k // ncol][k % ncol].axis("off")
            self.save(fig, "F8_convergence_%s" % m, "F8 - Monte Carlo sensitivity of the %s paired effect: the stored sample subsets in fixed order (prefixes K = 16, 32, 64; full K = 128; "
                      "disjoint halves A and B); one panel per contrast in the registered order; common y-axis; n/s = subset not stored. The eight batch-means groups are stored in table t08b, "
                      "not in the figure data, and are not drawn" % ttl, {"half_range": lim})

    # ---- F9
    def f9(self):
        D, K = self.S["FD"]["fd_deciles"]["data"], self.S["checkpoints"]
        cols = D[K[0]]["columns"]
        ylim = pad_limits([v for k in K for b in D[k]["bins"] for v in b])
        fig, axes = self.plt.subplots(2, 5, figsize=(14.0, 6.6), squeeze=False, sharey=True)
        styles = ("-", "--", ":")
        for i, k in enumerate(K):
            ax = axes[i // 5][i % 5]
            need(D[k]["columns"] == cols, "fd_deciles: checkpoints differ in columns")
            for ci, col in enumerate(cols):
                ax.plot(range(1, len(D[k]["bins"]) + 1), [b[ci] for b in D[k]["bins"]], color=ARM_COLOUR[k.split("/")[0]], ls=styles[ci], lw=1.1, marker="o", ms=2.2, label=col)
            ax.set_xlim(0.5, len(D[k]["bins"]) + 0.5)
            ax.set_ylim(*ylim)
            ax.set_title(k, fontsize=7.5)
            ax.tick_params(labelsize=6)
            ax.set_xlabel("decile of predicted SD", fontsize=6.5)
            if i == 0:
                ax.legend(fontsize=5.5, frameon=False)
        self.save(fig, "F9_uncertainty_vs_error", "F9 - uncertainty against error: per decile of predicted SD, the mean SD, mean R0 error and mean R1 error as stored, one panel per checkpoint key; common axes",
                  {"y_limits": list(ylim)})


# ----------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=VERSION)
    ap.add_argument("--g-run", required=True)
    ap.add_argument("--manifest-g-sha", required=True)
    ap.add_argument("--figs-sha", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    run = os.path.dirname(here)
    if os.path.basename(here) != "code":
        print("FAIL the script must be run as <run>/code/diag_unc_fig.py", flush=True)
        return 1
    log = Log(os.path.join(run, "fig.log"))
    result, reason, extra = "FAIL", "", {}
    t0 = time.time()
    try:
        log("INFO", "%s | %s%s" % (VERSION, SPEC, " | SELF-TEST -- NOT G EVIDENCE" if args.selftest else ""))
        log("INFO", "start_utc %s" % time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        log("INFO", "script sha256 %s" % sha_file(os.path.abspath(__file__)))
        outdir = os.path.join(run, "out")
        need(sorted(os.listdir(run)) == ["code"], "the run directory must hold only code/ at start")
        need(args.figs_sha == FIGS_SHA, "--figs-sha is missing or is not the registered sha256 of the frozen figure specification")
        need(len(args.manifest_g_sha) == 64 and all(ch in "0123456789abcdef" for ch in args.manifest_g_sha), "--manifest-g-sha is not a sha256")
        S = verify_inputs(args, log)
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 0.6, "pdf.fonttype": 42, "svg.hashsalt": "dunc", "hatch.linewidth": 0.4})
        log("INFO", "python %s | numpy %s | matplotlib %s | backend %s" % (sys.version.split()[0], np.__version__, matplotlib.__version__, matplotlib.get_backend()))
        os.mkdir(outdir)
        F = Figures(S, outdir, log, plt)
        F.f1(); F.f2(); F.f3(); F.f4(); F.f5(); F.f6(); F.f7(np); F.f8(); F.f9()
        for name, digest in sorted(F.written.items()):                    # re-read every output before the manifest
            need(sha_file(os.path.join(outdir, name)) == digest, "output changed after writing: %s" % name)
        man = {"schema": "seqref-dunc-fig-manifest/1", "version": VERSION, "specification": SPEC, "specification_sha256": args.figs_sha, "selftest": bool(args.selftest),
               "g_run": os.path.basename(S["g"]), "manifest_g_sha256": args.manifest_g_sha,
               "inputs": {"figdata": {("figdata/%s.json" % n): S["M"]["outputs"]["figdata/%s.json" % n] for n in FIGDATA}, "stage1_index_sha256": S["M"]["stage1_index_sha256"],
                          "manifest_f2_sha256": S["M"]["manifest_f2_sha256"], "map_files": len(S["maps"]), "reference_files": len(S["refs"])},
               "outputs": dict(sorted(F.written.items())), "visualisation_only": F.visual, "script_sha256": sha_file(os.path.abspath(__file__)),
               "interpreter": {"python": sys.version.split()[0], "numpy": np.__version__, "matplotlib": matplotlib.__version__},
               "note": "visualisation_only holds axis limits, the reference percentile, map limits and clipping figures; they are not results and affect no result or label"}
        mdata = (json.dumps(man, sort_keys=True, indent=1, allow_nan=False) + "\n").encode("utf-8")
        write_once(os.path.join(outdir, "manifest_fig.json"), mdata)
        write_once(os.path.join(outdir, "manifest_fig.json.sha256"), (sha_bytes(mdata) + "\n").encode("ascii"))
        extra["manifest_sha256"] = sha_bytes(mdata)
        log("INFO", "[END] PASS  %d output files (%d figures x png, pdf) and manifest_fig.json written once; elapsed %.1f s" % (len(F.written), len(F.written) // 2, time.time() - t0))
        result, reason = "PASS", "all figures written"
    except Fail as exc:
        reason = str(exc)
        log("ERROR", "[FAIL] %s" % reason)
    except Exception as exc:                                              # logged, never swallowed: the run FAILS
        reason = "unexpected %s" % type(exc).__name__
        log("ERROR", "[FAIL] %s: %s" % (reason, exc))
        for line in traceback.format_exc().splitlines():
            log("ERROR", "    " + line)
    tag = "G_FIG_SELFTEST_RESULT" if args.selftest else "G_FIG_RESULT"
    log("INFO", "end_utc %s" % time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    log("INFO", "%s_CHECKS: %s (%s)" % (tag[:-7], result, reason))
    try:
        log_sha = log.close()
        lines = ["%s: %s" % (tag, result), VERSION, "reason " + reason, "utc " + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "log_sha256 " + log_sha]
        if "manifest_sha256" in extra:
            lines.append("manifest_sha256 " + extra["manifest_sha256"])
        write_once(os.path.join(run, "RESULT.txt"), ("\n".join(lines) + "\n").encode("utf-8"))
    except Exception as exc:
        print("ERROR closing evidence could not be written: %s: %s" % (type(exc).__name__, exc), flush=True)
        result = "FAIL"
    print("%s: %s" % (tag, result), flush=True)
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

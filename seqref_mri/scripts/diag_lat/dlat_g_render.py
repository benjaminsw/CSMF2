# SEQREF-DLAT-GREN v0.2 -- D-LAT step G: render figures, Table 1 and S1-S7 from the hashed result files
# LIFETIME: KEEP
"""Presentation only (SEQREF-DLAT-G-REPORT-PLAN v0.1, steps C1-C5).

Reads g_report.json, figdata/*.json, g_main_table.json and g_eg1.json of ONE G run.
Refuses any file whose sha256 differs from the one named in g_report.json.
Computes no D-LAT measure, effect, interval, class or tag. The only arithmetic is:
  * Figure 1: mean over the stored per-slice values of a state, and the mean of the three
    seed means ("means by checkpoint role are formed at rendering", fig1.json);
  * S4: histogram bin counts of the stored tau and q_tail values.
Both are listed in render_report.json under "derived_at_rendering".
No fallback, no placeholder: any error is logged and ends the run with RENDER: FAIL.

Changelog:
  v0.2 (2026-11-24) presentation only, for three readability defects of the first rendering (LudWiki 258 LE109):
       Figure 1 (a) draws NLL, B and J on three stacked axes instead of one; every legend is one legend
       below the figure instead of legends inside the axes; Figure 5 uses larger text and draws panel (c)
       with one row of axes per seed. No value read, no value derived, no table and no caption changed.
  v0.1 (2026-11-24) first version.
"""
import argparse
import hashlib
import json
import logging
import math
import os
import sys
import traceback

VERSION = "SEQREF-DLAT-GREN v0.2"
LOG = logging.getLogger("dlat.render")

MEASURES = ["L1", "L2", "L3a", "L3b", "L4", "L5"]  # fixed row order (figure plan rule 11)
GROUPS = [("gap", ["S-GAP0", "C-TIME0", "F-TRAJ0(10→30)"]),
          ("controls", ["S-CTL0", "C-TIME2"]),
          ("interactions", ["X-TIME", "X-FIX(10→30)"])]
READ = [c for _, cs in GROUPS for c in cs]
ROLES = ["e3", "e10", "e30", "sel", "fin"]
ARMS = [("lam0", "λ₀", "#1f4e9c", "-"), ("lam2", "λ₂", "#c75b12", "--")]
SEEDS = [("2026", "o"), ("2027", "s"), ("2028", "^")]
FLAG = "numerical adequacy unresolved"
DAGGER = "†"
RESULT_FILES = ["figdata/fig%d.json" % i for i in range(1, 6)] + \
               ["figdata/s%d.json" % i for i in range(1, 8)] + ["g_main_table.json", "g_eg1.json"]


class RenderError(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_inputs(g_out, expect):
    rp = os.path.join(g_out, "g_report.json")
    got = sha256(rp)
    if got != expect:
        raise RenderError("g_report.json sha256 %s differs from the expected %s" % (got, expect))
    with open(rp, encoding="utf-8") as f:
        report = json.load(f)
    if report.get("verdict") != "G: COMPLETE":
        raise RenderError("g_report.json verdict is %r, not 'G: COMPLETE'" % report.get("verdict"))
    named = report["outputs"]["result_files"]
    data, hashes = {}, {"g_report.json": got}
    for rel in RESULT_FILES:
        if rel not in named:
            raise RenderError("%s is not named in g_report.json outputs.result_files" % rel)
        p = os.path.join(g_out, rel)
        h = sha256(p)
        if h != named[rel]:
            raise RenderError("%s sha256 %s differs from g_report.json (%s)" % (rel, h, named[rel]))
        with open(p, encoding="utf-8") as f:
            data[rel] = json.load(f)
        hashes[rel] = h
    LOG.info("inputs verified: g_report.json and %d result files at their sha256", len(RESULT_FILES))
    return report, data, hashes


def panel(doc, name):
    for p in doc["panels"]:
        if p.get("panel") == name:
            return p
    raise RenderError("panel %r not found in %s" % (name, doc.get("id")))


def isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


class Checks:
    """Every plotted series is compared with the JSON values it was taken from."""

    def __init__(self):
        self.rows, self.derived = [], []

    def same(self, where, plotted, source):
        a, b = [float(x) for x in plotted], [float(x) for x in source]
        ok = len(a) == len(b) and all(x == y for x, y in zip(a, b))
        self.rows.append({"where": where, "n": len(b), "equal": ok})
        if not ok:
            raise RenderError("plotted values differ from the JSON values at %s" % where)

    def note_derived(self, where, what, values):
        self.derived.append({"where": where, "what": what, "values": values})


def tag(p):
    return "[%s]" % p["provenance"]


# ---------------------------------------------------------------- drawing helpers
def contrast_axes(ax, pdata, chk, where, title, unit, flags=None):
    """Rule 1: pooled effect, 95 % interval, three per-seed effects, zero line. Rule 2: X-TIME next to X-FIX."""
    xs, pts, lo, hi, missing = [], [], [], [], []
    for i, c in enumerate(READ):
        e = pdata[c]
        if not isnum(e.get("point")):
            missing.append(i)
            continue
        xs.append(i)
        pts.append(e["point"])
        lo.append(e["ci"][0])
        hi.append(e["ci"][1])
    ax.axhline(0.0, color="0.4", lw=0.8)
    if xs:
        cont = ax.errorbar(xs, pts, yerr=[[p - a for p, a in zip(pts, lo)], [b - p for p, b in zip(pts, hi)]],
                           fmt="D", color="black", ms=5, capsize=3, lw=1.2, label="pooled, 95 % interval")
        chk.same(where + " pooled", cont.lines[0].get_ydata(), pts)
    for s, (seed, mk) in enumerate(SEEDS):
        sx, sy = [], []
        for i, c in enumerate(READ):
            ps = pdata[c].get("per_seed")
            if ps is not None and isnum(ps[s]):
                sx.append(i + (s - 1) * 0.18)
                sy.append(ps[s])
        if sx:
            ln, = ax.plot(sx, sy, mk, mfc="none", color="0.35", ms=5, label="seed %s" % seed)
            chk.same("%s seed %s" % (where, seed), ln.get_ydata(), sy)
    for i in missing:  # rule 6: explicit empty marker with a note, never zero
        ax.plot([i], [0.95], "o", mfc="none", mec="red", ms=9, transform=ax.get_xaxis_transform(), clip_on=False)
        ax.text(i, 0.91, "not computed", rotation=90, ha="center", va="top", fontsize=7, color="red",
                transform=ax.get_xaxis_transform())
    edge = 0
    for g, cs in GROUPS:
        ax.text(edge + (len(cs) - 1) / 2.0, 1.01, g, ha="center", va="bottom", fontsize=8,
                transform=ax.get_xaxis_transform())
        edge += len(cs)
        if edge < len(READ):
            ax.axvline(edge - 0.5, color="0.8", lw=0.8)
    labels = []
    for c in READ:
        f = flags is not None and flags.get(c) == FLAG
        labels.append(c + (" " + DAGGER if f else ""))
    ax.set_xticks(range(len(READ)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_xlim(-0.6, len(READ) - 0.4)
    ax.set_ylabel("effect (%s)" % unit, fontsize=8)
    ax.set_title(title, fontsize=9, loc="left", pad=14)
    ax.tick_params(labelsize=8)


def epochs_text(roles_map):
    parts = []
    for r in ("sel", "fin"):
        parts.append("%s: %s" % ({"sel": "selected", "fin": "final"}[r], ", ".join(
            "%s %s" % (sym, "/".join(str(roles_map["%s/%s" % (arm, r)][s]["epoch"]) for s, _ in SEEDS))
            for arm, sym, _, _ in ARMS)))
    return "epochs per seed 2026/2027/2028 -- " + "; ".join(parts)


def role_ticks(ax, roles_map, compact=False):
    if compact:
        ax.set_xticks(range(len(ROLES)))
        ax.set_xticklabels(["e3", "e10", "e30", "sel", "fin"], fontsize=7)
        ax.set_xlim(-0.4, len(ROLES) - 0.6)
        return
    labs = []
    for r in ROLES:
        if r in ("sel", "fin"):
            ep = []
            for arm, sym, _, _ in ARMS:
                m = roles_map["%s/%s" % (arm, r)]
                ep.append("%s %s" % (sym, "/".join(str(m[s]["epoch"]) for s, _ in SEEDS)))
            labs.append("%s\n%s" % ({"sel": "selected", "fin": "final"}[r], "\n".join(ep)))
        else:
            labs.append("epoch %s" % r[1:])
    ax.set_xticks(range(len(ROLES)))
    ax.set_xticklabels(labs, fontsize=7)
    ax.set_xlim(-0.4, len(ROLES) - 0.6)


def traj_axes(ax, entry, roles_map, chk, where, ylabel, title, compact=False):
    """Rule 3: descriptive trajectory by checkpoint role (categorical axis, epochs printed)."""
    for arm, sym, col, ls in ARMS:
        xs, ys = [], []
        for i, r in enumerate(ROLES):
            v = entry["roles"]["%s/%s" % (arm, r)]["pooled"]
            if isnum(v):
                xs.append(i)
                ys.append(v)
        if xs:
            ln, = ax.plot(xs, ys, ls, color=col, lw=1.6, label="%s pooled" % sym)
            chk.same("%s %s pooled" % (where, arm), ln.get_ydata(), ys)
        for s, (seed, mk) in enumerate(SEEDS):
            sx, sy = [], []
            for i, r in enumerate(ROLES):
                v = entry["roles"]["%s/%s" % (arm, r)]["per_seed"].get(seed)
                if isnum(v):
                    sx.append(i + (s - 1) * 0.08)
                    sy.append(v)
                else:
                    ax.plot([i], [0.95], "o", mfc="none", mec="red", ms=8, transform=ax.get_xaxis_transform())
                    ax.text(i, 0.91, "not computed", rotation=90, ha="center", va="top", fontsize=6,
                            color="red", transform=ax.get_xaxis_transform())
            if sx:
                ln, = ax.plot(sx, sy, mk, mfc="none", color=col, ms=4, label="%s seed %s" % (sym, seed))
                chk.same("%s %s seed %s" % (where, arm, seed), ln.get_ydata(), sy)
    role_ticks(ax, roles_map, compact)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, loc="left")
    ax.tick_params(axis="y", labelsize=8)


def traj_panel(sub, p, roles_map, chk, where, letter, extra=None):
    keys = p["keys"]
    axs = sub.subplots(1, len(keys), squeeze=False)[0]
    for ax, k in zip(axs, keys):
        traj_axes(ax, p["data"][k], roles_map, chk, "%s %s" % (where, k), "%s (%s)" % (k, p["units"][k]),
                  "(%s) %s\n%s" % (letter, k, tag(p)), compact=len(keys) > 2)
        if extra:
            extra(ax, k)
    foot = "checkpoint role (descriptive; interpretation rests on the registered contrasts only)"
    if len(keys) > 2:
        foot += "\n" + epochs_text(roles_map)
    sub.supxlabel(foot, fontsize=7)


def fig_legend(fig, kinds, extra=()):
    """One legend below the figure (v0.2): nothing is drawn over the data."""
    from matplotlib.lines import Line2D
    h = []
    if "arms" in kinds:
        for _, sym, col, ls in ARMS:
            h.append(Line2D([], [], color=col, ls=ls, lw=1.8, label="%s pooled over seeds" % sym))
    if "seeds" in kinds:
        for seed, mk in SEEDS:
            h.append(Line2D([], [], color="0.35", marker=mk, mfc="none", ls="", ms=6,
                            label="seed %s%s" % (seed, " (in the colour of its arm)" if "arms" in kinds and seed == "2026" else "")))
    if "pooled" in kinds:
        h.append(Line2D([], [], color="black", marker="D", ls="-", lw=1.2, ms=5,
                        label="pooled effect with 95 % interval"))
    h.extend(extra)
    fig.legend(handles=h, loc="outside lower center", ncol=min(len(h), 7), fontsize=8, frameon=False)


def save(fig, out, name, written):
    for ext in ("png", "pdf"):
        p = os.path.join(out, "%s.%s" % (name, ext))
        fig.savefig(p, dpi=170)
        written.append(p)
    LOG.info("figure written: %s", name)


# ---------------------------------------------------------------- figures
def fig1(plt, d, out, chk, written):
    doc = d["figdata/fig1.json"]
    roles_map = doc["checkpoint_roles"]
    specs = [("a", ["NLL", "B", "J"]), ("b", ["R0_PSNR"]), ("c", ["R1_PSNR"])]
    fig = plt.figure(figsize=(15, 6.4), constrained_layout=True)
    subs = fig.subfigures(1, 3)
    for sub, (letter, keys) in zip(subs, specs):
        p = panel(doc, letter)
        if len(keys) > 1:  # v0.2: one axis per quantity; B and J stay thin lines beside NLL
            axs = sub.subplots(len(keys), 1, sharex=True, height_ratios=[3] + [1] * (len(keys) - 1))
        else:
            axs = [sub.subplots()]
        for ax, k in zip(axs, keys):
            thin = k in ("B", "J")
            for arm, sym, col, ls in ARMS:
                pooled = []
                per_seed = {seed: [] for seed, _ in SEEDS}
                for r in ROLES:
                    means = []
                    for seed, _ in SEEDS:
                        vals = p["data"][roles_map["%s/%s" % (arm, r)][seed]["state"]][k]
                        m = math.fsum(vals) / len(vals)
                        per_seed[seed].append(m)
                        means.append(m)
                    pooled.append(math.fsum(means) / len(means))
                chk.note_derived("fig1 (%s) %s %s" % (letter, k, arm),
                                 "mean over the stored per-slice values of each state; pooled = mean of the three seed means",
                                 {"roles": ROLES, "pooled": pooled, "per_seed": per_seed})
                ax.plot(range(len(ROLES)), pooled, ls, color=col, lw=0.9 if thin else 1.8)
                if not thin:
                    for s, (seed, mk) in enumerate(SEEDS):
                        ax.plot([i + (s - 1) * 0.08 for i in range(len(ROLES))], per_seed[seed], mk, mfc="none",
                                color=col, ms=4)
            if thin:
                ax.set_ylabel(k, fontsize=8)
            elif len(keys) > 1:
                ax.set_ylabel("%s (%s; %s in the same unit)" % (k, p["units"][k], " and ".join(keys[1:])), fontsize=8)
            else:
                ax.set_ylabel("%s (%s)" % (k.replace("_", " "), p["units"][k]), fontsize=8)
            ax.tick_params(axis="y", labelsize=8)
            if ax is axs[0]:
                ax.set_title("(%s) %s %s" % (letter, ", ".join(x.replace("_", " ") for x in keys), tag(p)),
                             fontsize=9, loc="left")
        role_ticks(axs[-1], roles_map)
    fig.suptitle("Figure 1. %s" % doc["title"], fontsize=10)
    fig_legend(fig, ("arms", "seeds"))
    save(fig, out, "fig1", written)
    plt.close(fig)


def contrast_and_traj_figure(plt, doc, number, layout, out, chk, written, flags=None, legend_extra=()):
    fig = plt.figure(figsize=(15, 5.0 * len(layout)), constrained_layout=True)
    subs = fig.subfigures(len(layout), 1, squeeze=False)
    roles_map = doc["checkpoint_roles"]
    for row, items in enumerate(layout):
        cols = subs[row][0].subfigures(1, len(items), squeeze=False)[0]
        for sub, (letter, kind, extra) in zip(cols, items):
            p = panel(doc, letter)
            where = "fig%d (%s)" % (number, letter)
            if kind == "contrast":
                ax = sub.subplots()
                fl = flags if p.get("measure") == "L4" else None
                title = "(%s) %s %s %s" % (letter, p["measure"], p["key"], tag(p))
                if p.get("label"):
                    title += "\n" + p["label"]
                contrast_axes(ax, p["data"], chk, where, title, p["unit"], fl)
                if fl is not None:
                    sub.supxlabel("%s %s (rule E-G1): the annotation changes no class, tag or label" % (DAGGER, FLAG),
                                  fontsize=7)
            else:
                traj_panel(sub, p, roles_map, chk, where, letter, extra)
    fig.suptitle("Figure %d. %s" % (number, doc["title"]), fontsize=10)
    fig_legend(fig, ("arms", "seeds", "pooled"), legend_extra)
    save(fig, out, "fig%d" % number, written)
    plt.close(fig)


def forest_rows(sub, pdata, contrasts, units, chk, where, title, eg1):
    axs = sub.subplots(len(MEASURES), 1, squeeze=False)[:, 0]
    for ax, m in zip(axs, MEASURES):
        ys, pts, lo, hi = [], [], [], []
        for j, c in enumerate(contrasts):
            e = pdata[m][c]
            y = len(contrasts) - 1 - j
            if not isnum(e.get("point")):
                ax.plot([0.95], [y], "o", mfc="none", mec="red", ms=8, transform=ax.get_yaxis_transform())
                ax.text(0.93, y, "not computed", fontsize=6, color="red", va="center", ha="right",
                        transform=ax.get_yaxis_transform())
                continue
            ys.append(y)
            pts.append(e["point"])
            lo.append(e["ci"][0])
            hi.append(e["ci"][1])
            for s, (seed, mk) in enumerate(SEEDS):
                if isnum(e["per_seed"][s]):
                    ax.plot([e["per_seed"][s]], [y + (s - 1) * 0.18], mk, mfc="none", color="0.35", ms=4)
        ax.axvline(0.0, color="0.4", lw=0.8)
        if ys:
            cont = ax.errorbar(pts, ys, xerr=[[p - a for p, a in zip(pts, lo)], [b - p for p, b in zip(pts, hi)]],
                               fmt="D", color="black", ms=4, capsize=2, lw=1.1)
            chk.same("%s %s pooled" % (where, m), cont.lines[0].get_xdata(), pts)
        labs = []
        for c in contrasts:
            f = m == "L4" and eg1["contrasts"][c]["label"] == FLAG
            labs.append(c + (" " + DAGGER if f else ""))
        ax.set_yticks(range(len(contrasts)))
        ax.set_yticklabels(list(reversed(labs)), fontsize=9)
        ax.set_ylim(-0.6, len(contrasts) - 0.4)
        ax.set_ylabel(m, fontsize=10, rotation=0, labelpad=18)
        ax.tick_params(axis="x", labelsize=9)
        if units:
            ax.set_xlabel("effect (%s)" % units[m], fontsize=8)
    sub.suptitle(title, fontsize=10, x=0.02, ha="left")
    sub.supxlabel("own axis per measure, fixed row order, no common scale (rule 11); %s %s (E-G1)" % (DAGGER, FLAG),
                  fontsize=9)


def fig5(plt, d, out, chk, written):
    doc, eg1 = d["figdata/fig5.json"], d["g_eg1.json"]
    fig = plt.figure(figsize=(17, 34), constrained_layout=True)
    top, mid, bot = fig.subfigures(3, 1, height_ratios=[1.3, 1.0, 1.3])
    sa, sb = top.subfigures(1, 2)
    pa, pb, pc, pd = (panel(doc, x) for x in "abcd")
    forest_rows(sa, pa["data"], pa["contrasts"], pa.get("units"), chk, "fig5 (a)",
                "(a) six classified measures on the gap contrasts %s" % tag(pa), eg1)
    forest_rows(sb, pb["data"], pb["contrasts"], pa.get("units"), chk, "fig5 (b)",
                "(b) X-TIME with X-FIX(10→30), same axis per measure %s" % tag(pb), eg1)
    axc = mid.subplots(len(SEEDS), len(READ), squeeze=False)
    for r, (seed, mk) in enumerate(SEEDS):
        for k, c in enumerate(READ):
            ax = axc[r][k]
            e = pc["data"][c][seed]
            ln, = ax.plot(e["d_R0_PSNR"], e["d_NLL"], mk, mfc="none", color="0.25", ms=3)
            chk.same("fig5 (c) %s seed %s d_NLL" % (c, seed), ln.get_ydata(), e["d_NLL"])
            chk.same("fig5 (c) %s seed %s d_R0_PSNR" % (c, seed), ln.get_xdata(), e["d_R0_PSNR"])
            ax.axhline(0, color="0.6", lw=0.6)
            ax.axvline(0, color="0.6", lw=0.6)
            ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title(c, fontsize=9)
            if k == 0:
                ax.set_ylabel("seed %s" % seed, fontsize=9)
    mid.suptitle("(c) per-volume change of NLL against change of R0 PSNR, per read contrast and seed %s" % tag(pc),
                 fontsize=10, x=0.02, ha="left")
    mid.supxlabel("change of R0 PSNR (%s)" % pc["units"]["d_R0_PSNR"], fontsize=9)
    mid.supylabel("change of NLL (%s)" % pc["units"]["d_NLL"], fontsize=9)
    axd = bot.subplots(len(MEASURES), 2, squeeze=False, sharex=True)
    for i, m in enumerate(MEASURES):
        for j, (vs, name) in enumerate((("vs_nll", "change of NLL"), ("vs_psnr", "change of R0 PSNR"))):
            ax = axd[i][j]
            ax.axhline(0, color="0.4", lw=0.8)
            for s, (seed, mk) in enumerate(SEEDS):
                xs, rho, lo, hi = [], [], [], []
                for k, c in enumerate(READ):
                    e = pd["data"][c][seed]["measures"][m][vs]
                    x = k + (s - 1) * 0.2
                    if not isnum(e.get("rho")):
                        ax.plot([x], [0.95], "o", mfc="none", mec="red", ms=7, transform=ax.get_xaxis_transform())
                        ax.text(x, 0.91, "not computed", rotation=90, fontsize=5, color="red", ha="center",
                                va="top", transform=ax.get_xaxis_transform())
                        continue
                    xs.append(x)
                    rho.append(e["rho"])
                    lo.append(e["ci"][0])
                    hi.append(e["ci"][1])
                if xs:
                    cont = ax.errorbar(xs, rho, yerr=[[r - a for r, a in zip(rho, lo)], [b - r for r, b in zip(rho, hi)]],
                                       fmt=mk, mfc="none", color="0.25", ms=4, capsize=2, lw=0.8,
                                       label="seed %s" % seed)
                    chk.same("fig5 (d) %s %s seed %s" % (m, vs, seed), cont.lines[0].get_ydata(), rho)
            ax.set_ylim(-1, 1)
            ax.set_ylabel("%s\nρ" % m, fontsize=10)
            ax.tick_params(labelsize=8)
            if i == 0:
                ax.set_title("Spearman ρ with the %s" % name, fontsize=10)
            if i == len(MEASURES) - 1:
                ax.set_xticks(range(len(READ)))
                ax.set_xticklabels(READ, rotation=30, ha="right", fontsize=9)
    bot.suptitle("(d) L6 Spearman ρ with intervals (descriptive; no class) %s" % tag(pd), fontsize=10, x=0.02,
                 ha="left")
    fig.suptitle("Figure 5. %s" % doc["title"], fontsize=11)
    fig_legend(fig, ("seeds", "pooled"))
    save(fig, out, "fig5", written)
    plt.close(fig)


# ---------------------------------------------------------------- tables
def num(v):
    return "not computed" if not isnum(v) else repr(float(v))


def r6(v):
    return "not computed" if not isnum(v) else "%+.6f" % v


def eff(e):
    if not isnum(e.get("point")):
        return "not computed"
    return "%s [%s, %s]" % (r6(e["point"]), r6(e["ci"][0]), r6(e["ci"][1]))


def seeds(e):
    ps = e.get("per_seed")
    return "not computed" if ps is None else ", ".join(r6(x) for x in ps)


def md(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def head(title, doc, hashes, rel):
    return ("<!-- %s -- %s -->\n<!-- LIFETIME: KEEP -->\n# %s\n\nSource: `%s`, sha256 `%s`. Rendered by %s; "
            "values are copied from the source, none is recomputed.\n\n" % (VERSION, title, title, rel, hashes[rel],
                                                                              VERSION))


def write(out, name, text, written):
    p = os.path.join(out, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    written.append(p)
    LOG.info("table written: %s", name)


def table1(d, hashes, out, written):
    t, eg1 = d["g_main_table.json"], d["g_eg1.json"]
    mt = t["main_table"]
    rows = []
    for r in mt["rows"]:
        e = r["effects"]
        ann = "not applicable"
        if r["measure"] == "L4":
            ann = "; ".join("%s: %s" % (c, eg1["contrasts"][c]["label"]) for c in
                            ("S-GAP0", "C-TIME0", "X-TIME", "X-FIX(10→30)"))
        rows.append([r["measure"] + " " + r["key"], eff(e["S-GAP0"]), seeds(e["S-GAP0"]), eff(e["C-TIME0"]),
                     seeds(e["C-TIME0"]), r["class"], r["fixed_epoch_tag"], r["seed_tag"], r["stability_summary"],
                     eff(e["X-TIME"]), eff(e["X-FIX(10→30)"]), r["coordinate_tag"], ann])
    text = head("Table 1 -- one row per classified measure", t, hashes, "g_main_table.json")
    text += md(rows, ["Measure", "S-GAP0 effect [95 % interval]", "S-GAP0 per seed", "C-TIME0 effect [95 % interval]",
                      "C-TIME0 per seed", "Class", "Fixed-epoch tag", "Seed tag", "Monte Carlo stability",
                      "X-TIME", "X-FIX(10→30)", "Coordinate tag", "E-G1 annotation"])
    o = mt["outcome"]
    text += ("\nOutcome: **%s** (%s; open measures: %s; open contrasts: %s). Reading rules: mini-spec 235 section 5 "
             "(%s).\n\nExploratory status: %s.\n\nReview status: %s\n\nE-G1 reading: %s\n\nValues are rounded to six "
             "decimals here; S1 gives them at full precision.\n"
             % (o["outcome"], "provisional" if o["provisional"] else "not provisional", o["open_measures"] or "none",
                o["open_contrasts"] or "none", mt["rules_version"], t["exploratory_status"], t["review_status"],
                eg1["reading"]))
    write(out, "table1.md", text, written)


def supp(plt, d, hashes, out, chk, written):
    # S1
    doc = d["figdata/s1.json"]
    text = head("S1 -- every classified and reported measure on all 21 contrasts", doc, hashes, "figdata/s1.json")
    pc, pr = panel(doc, "classified")["data"], panel(doc, "reported")
    cons = list(pc["L1"].keys())
    text += "## Classified measures %s\n\n" % tag(panel(doc, "classified"))
    text += md([[m, c, num(pc[m][c].get("point")), num((pc[m][c].get("ci") or [None, None])[0]),
                 num((pc[m][c].get("ci") or [None, None])[1]),
                 ", ".join(num(x) for x in (pc[m][c].get("per_seed") or [])), pc[m][c].get("label")]
                for m in MEASURES for c in cons],
               ["Measure", "Contrast", "Effect", "Interval low", "Interval high", "Per seed", "Direction label"])
    text += "\n## Reported measures (not classified) %s\n\n" % tag(pr)
    text += md([[k, c, num(v["contrasts"][c].get("point")), num((v["contrasts"][c].get("ci") or [None, None])[0]),
                 num((v["contrasts"][c].get("ci") or [None, None])[1]),
                 ", ".join(num(x) for x in (v["contrasts"][c].get("per_seed") or []))]
                for k, v in pr["data"].items() for c in v["contrasts"]],
               ["Quantity", "Contrast", "Effect", "Interval low", "Interval high", "Per seed"])
    write(out, "s1.md", text, written)
    # S2
    doc = d["figdata/s2.json"]
    text = head("S2 -- Monte Carlo stability: direction labels at K = 64, K = 128 and both halves", doc, hashes,
                "figdata/s2.json")
    for p in doc["panels"]:
        text += "## %s (measure-level summary: %s) %s\n\n" % (p["panel"], p["summary"], tag(p))
        text += md([[c, v["labels"]["pre64"], v["labels"]["full"], v["labels"]["halfA"], v["labels"]["halfB"],
                     v["stability"]] for c, v in p["data"].items()],
                   ["Contrast", "K = 64", "K = 128", "Half A", "Half B", "Stability"]) + "\n"
    write(out, "s2.md", text, written)
    # S3
    doc = d["figdata/s3.json"]
    text = head("S3 -- physical complements beside their standardised versions (reported, not classified)", doc,
                hashes, "figdata/s3.json")
    for p in doc["panels"]:
        text += "## %s %s\n\n" % (p["panel"], tag(p))
        if "coordinate_tag" in p:
            text += "Coordinate tag: %s. Physical complement: reported, not classified.\n\n" % p["coordinate_tag"]
            text += md([[ver, c, eff(e), seeds(e), e.get("label")] for ver in ("standardised", "physical")
                        for c, e in p["data"][ver].items()],
                       ["Version", "Contrast", "Effect [95 % interval]", "Per seed", "Direction label"]) + "\n"
        else:
            for k, v in p["data"].items():
                text += "### %s\n\n" % k
                text += md([[c, eff(e), seeds(e)] for c, e in v["contrasts"].items()],
                           ["Contrast", "Effect [95 % interval]", "Per seed"]) + "\n"
                text += md([[r, num(x["pooled"])] + [num(x["per_seed"].get(s)) for s, _ in SEEDS]
                            for r, x in v["state_means"]["roles"].items()],
                           ["Role", "Pooled", "Seed 2026", "Seed 2027", "Seed 2028"]) + "\n"
    write(out, "s3.md", text, written)
    # S4
    doc = d["figdata/s4.json"]
    pt, pl = panel(doc, "tail"), panel(doc, "L6 tail correlations")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for ax, k in zip(axs, ("tau", "q_tail")):
        vals = pt["data"][k]
        counts, edges, _ = ax.hist(vals, bins=20, color="0.5")
        chk.note_derived("s4 %s" % k, "histogram bin counts of the stored per-slice values (20 bins)",
                         {"counts": [int(c) for c in counts], "edges": [float(e) for e in edges], "n": len(vals)})
        ax.set_xlabel("%s over the %d slices" % (k, len(vals)), fontsize=8)
        ax.set_ylabel("number of slices", fontsize=8)
        ax.tick_params(labelsize=8)
    fig.suptitle("S4. Tail covariate: distribution over the slices %s" % tag(pt), fontsize=9)
    save(fig, out, "s4", written)
    plt.close(fig)
    text = head("S4 -- tail covariate over the slices and its L6 correlations", doc, hashes, "figdata/s4.json")
    text += "tau summary: %s; tau constant: %s. Distribution: `s4.png`.\n\n" % (
        json.dumps(pt["data"]["tau_summary"], sort_keys=True), pt["data"]["tau_constant"])
    text += md([[c, seed, k, num(e.get("rho")), num((e.get("ci") or [None, None])[0]),
                 num((e.get("ci") or [None, None])[1]), e.get("reason")]
                for c, v in pl["data"].items() for seed, _ in SEEDS for k, e in v[seed].items()],
               ["Contrast", "Seed", "Correlation", "rho", "Interval low", "Interval high", "Reason if not computed"])
    write(out, "s4.md", text, written)
    # S5
    doc = d["figdata/s5.json"]
    text = head("S5 -- L1: skewness, excess kurtosis and E_cov beside the N(0, I) reference", doc, hashes,
                "figdata/s5.json")
    for name in ("skewness", "excess_kurtosis"):
        p = panel(doc, name)
        text += "## %s (N(0, I) reference: 0) %s\n\n" % (name, tag(p))
        text += md([[r, num(x["pooled"])] + [num(x["per_seed"].get(s)) for s, _ in SEEDS]
                    for r, x in p["data"]["state_means"]["roles"].items()],
                   ["Role", "Pooled", "Seed 2026", "Seed 2027", "Seed 2028"]) + "\n"
        text += md([[c, eff(e), seeds(e)] for c, e in p["data"]["contrasts"].items()],
                   ["Contrast", "Effect [95 % interval]", "Per seed"]) + "\n"
    p = panel(doc, "E_cov")
    text += "## E_cov (m = %s, %s volumes; reference %s) %s\n\n" % (p["data"]["m"], p["data"]["n_volumes"],
                                                                    num(p["data"]["reference"]), tag(p))
    text += md([[r] + [num(x.get(s)) for s, _ in SEEDS] for r, x in p["data"]["by_role_and_seed"].items()],
               ["Role", "Seed 2026", "Seed 2027", "Seed 2028"])
    write(out, "s5.md", text, written)
    # S6
    doc = d["figdata/s6.json"]
    text = head("S6 -- fixtures", doc, hashes, "figdata/s6.json")
    for p in doc["panels"]:
        text += "## %s %s\n\nSource: %s\n\n" % (p["panel"], tag(p), json.dumps(p["source"], ensure_ascii=False))
        dat = p["data"]
        if "banner" in dat:
            text += "**%s**\n\n" % dat["banner"]
        if "cited" in dat:
            text += md([[k, v] for k, v in dat["cited"].items()], ["Item", "Value"]) + "\n"
        fx = dat.get("fixtures", dat if p["panel"] == "step G" else None)
        if isinstance(fx, dict):
            text += md([[k, json.dumps(v, ensure_ascii=False)] for k, v in fx.items() if k != "S3"],
                       ["Fixture", "Observed against tolerance"]) + "\n"
            if "S3" in fx:
                text += "S3 (donor map): %d pairs recorded in the source file.\n\n" % len(fx["S3"].get("pairs", []))
        elif isinstance(fx, list):
            text += md([[f.get("id"), f.get("title"), f.get("pass", f.get("state")), c.get("check"),
                         json.dumps(c.get("observed")), json.dumps(c.get("expected")), json.dumps(c.get("residual")),
                         json.dumps(c.get("tolerance")), c.get("pass")] for f in fx for c in f["checks"]],
                       ["Fixture", "Title", "Fixture result", "Check", "Observed", "Expected", "Residual",
                        "Tolerance", "Pass"]) + "\n"
        else:
            text += md([[k, json.dumps(v, ensure_ascii=False)] for k, v in dat.items()], ["Item", "Value"]) + "\n"
    write(out, "s6.md", text, written)
    # S7
    doc = d["figdata/s7.json"]
    p = panel(doc, "events")
    text = head("S7 -- events: every 'not computed' case and typed non-fatal event", doc, hashes, "figdata/s7.json")
    text += "Adapter events: %d. States with a 'not computed' case: %d. %s\n\n" % (
        p["data"]["n_adapter_events"], len(p["data"]["not_computed_states"]), tag(p))
    text += "```json\n%s\n```\n" % json.dumps(p["data"], indent=1, ensure_ascii=False, sort_keys=True)
    write(out, "s7.md", text, written)


def captions(d, hashes, out, written):
    text = "<!-- %s -- captions of Figures 1-5 -->\n<!-- LIFETIME: KEEP -->\n# Captions\n\n" % VERSION
    for i in range(1, 6):
        rel = "figdata/fig%d.json" % i
        doc = d[rel]
        text += "**Figure %d. %s.** Data file `%s`, sha256 `%s`.\n\n" % (i, doc["title"], rel, hashes[rel])
        for p in doc["panels"]:
            text += "- (%s) %s. Provenance: **%s**; source: %s\n" % (
                p["panel"], p.get("kind"), p["provenance"], json.dumps(p["source"], ensure_ascii=False))
        text += "\n"
    text += ("Contrast panels: pooled effect, 95 %% percentile bootstrap interval and the three per-seed effects; "
             "zero line; no significance stars. Trajectory panels are descriptive; interpretation rests on the "
             "registered contrasts only. Figure 1 consists of existing results only; none is a D-LAT finding. "
             "%s on an L4 contrast: %s (rule E-G1); the annotation changes no class, tag or label.\n" % (DAGGER, FLAG))
    write(out, "captions.md", text, written)


# ---------------------------------------------------------------- main
def run(args):
    report, d, hashes = load_inputs(args.g_out, args.g_report_sha256)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy
    chk, written = Checks(), []
    flags = panel(d["figdata/fig4.json"], "a")["numerical_annotation"]["contrasts"]
    fig1(plt, d, args.out, chk, written)
    contrast_and_traj_figure(plt, d["figdata/fig2.json"], 2, [[("a", "contrast", None), ("b", "contrast", None)],
                                                               [("c", "traj", None), ("d", "traj", None)]],
                             args.out, chk, written)
    contrast_and_traj_figure(plt, d["figdata/fig3.json"], 3, [[("a", "contrast", None), ("b", "contrast", None)],
                                                               [("c", "traj", None)]], args.out, chk, written)
    ref = panel(d["figdata/fig4.json"], "c")["isotropic_reference"]

    def pr_ref(ax, k):
        ax.axhline(ref, color="green", lw=1.0, ls=":")

    contrast_and_traj_figure(plt, d["figdata/fig4.json"], 4, [[("a", "contrast", None), ("b", "contrast", None)],
                                                               [("c", "traj", pr_ref), ("d", "traj", None)]],
                             args.out, chk, written, flags,
                             legend_extra=(Line2D([], [], color="green", lw=1.0, ls=":", label="panel (c): isotropic reference"),))
    fig5(plt, d, args.out, chk, written)
    table1(d, hashes, args.out, written)
    supp(plt, d, hashes, args.out, chk, written)
    captions(d, hashes, args.out, written)
    rep = {"schema": "seqref-dlat-g-render-report/1", "version": VERSION, "verdict": "RENDER: COMPLETE",
           "g_run_manifest_sha256": report["manifest"]["sha256"], "inputs_sha256": hashes,
           "outputs_sha256": {os.path.basename(p): sha256(p) for p in written},
           "libraries": {"python": sys.version.split()[0], "matplotlib": matplotlib.__version__,
                         "numpy": numpy.__version__},
           "value_checks": {"n": len(chk.rows), "all_equal": all(r["equal"] for r in chk.rows), "rows": chk.rows},
           "derived_at_rendering": chk.derived,
           "figure_plan": d["figdata/fig1.json"]["figure_plan"],
           "review_status": d["g_main_table.json"]["review_status"]}
    with open(os.path.join(args.out, "render_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, sort_keys=True, ensure_ascii=False)
    LOG.info("value checks: %d series, all equal; outputs: %d files", len(chk.rows), len(written))


def main(argv=None):
    ap = argparse.ArgumentParser(description=VERSION)
    ap.add_argument("--g-out", required=True, help="the out/ folder of the G run")
    ap.add_argument("--g-report-sha256", required=True, help="registered sha256 of g_report.json")
    ap.add_argument("--out", required=True, help="new, empty output folder")
    args = ap.parse_args(argv)
    if os.path.exists(args.out) and os.listdir(args.out):
        print("RENDER: FAIL (output folder is not empty)")
        return 2
    os.makedirs(args.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.FileHandler(os.path.join(args.out, "render.log"), encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)])
    LOG.info("%s; g-out %s", VERSION, args.g_out)
    try:
        run(args)
    except Exception as exc:  # logged, never swallowed: the run fails
        LOG.error("render failed: %s\n%s", exc, traceback.format_exc())
        with open(os.path.join(args.out, "render_verdict.txt"), "w", encoding="utf-8") as f:
            f.write("RENDER: FAIL\n%s\n" % exc)
        print("RENDER: FAIL")
        return 1
    with open(os.path.join(args.out, "render_verdict.txt"), "w", encoding="utf-8") as f:
        f.write("RENDER: COMPLETE\n")
    print("RENDER: COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())

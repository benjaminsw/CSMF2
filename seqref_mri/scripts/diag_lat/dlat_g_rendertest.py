# SEQREF-DLAT-GRENT v0.1 -- synthetic tests of the D-LAT G renderer (no real value is rendered)
# LIFETIME: KEEP
"""Builds a synthetic G output folder with the STRUCTURE of a real one and random numbers in place of
every real value, then runs the renderer on it. No figure of the real result is produced by this test."""
import argparse, copy, hashlib, json, os, random, shutil, subprocess, sys

RES = ["figdata/fig%d.json" % i for i in range(1, 6)] + ["figdata/s%d.json" % i for i in range(1, 8)] + \
      ["g_main_table.json", "g_eg1.json"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def scramble(x, rng):
    if isinstance(x, dict):
        y = {k: scramble(v, rng) for k, v in x.items()}
        for c in ("point", "rho", "share"):
            if c in x and isinstance(x[c], float) and isinstance(x.get("ci"), list):
                v = rng.uniform(-0.5, 0.5)
                y[c] = v
                y["ci"] = [v - rng.uniform(0.01, 0.2), v + rng.uniform(0.01, 0.2)]
        return y
    if isinstance(x, list):
        return [scramble(v, rng) for v in x]
    if isinstance(x, float):
        return rng.uniform(-1.0, 1.0)
    return x


def build(template, dst, rng, holes=False, verdict="G: COMPLETE"):
    os.makedirs(os.path.join(dst, "figdata"))
    named = {}
    for rel in RES:
        doc = scramble(json.load(open(os.path.join(template, rel), encoding="utf-8")), rng)
        if holes and rel == "figdata/fig2.json":
            doc["panels"][0]["data"]["S-GAP0"].update(point=None, ci=None, per_seed=None, label="not computed")
            doc["panels"][2]["data"]["log_gamma_bank"]["roles"]["lam0/e3"]["per_seed"]["2027"] = None
        if holes and rel == "figdata/fig5.json":
            doc["panels"][0]["data"]["L3a"]["C-TIME0"].update(point=None, ci=None, per_seed=None)
            doc["panels"][3]["data"]["X-TIME"]["2026"]["measures"]["L2"]["vs_nll"].update(rho=None, ci=None)
        p = os.path.join(dst, rel)
        json.dump(doc, open(p, "w", encoding="utf-8"), sort_keys=True)
        named[rel] = sha(p)
    rep = {"verdict": verdict, "outputs": {"result_files": named}, "manifest": {"sha256": "synthetic"}}
    rp = os.path.join(dst, "g_report.json")
    json.dump(rep, open(rp, "w", encoding="utf-8"), sort_keys=True)
    return sha(rp)


def render(py, script, g_out, h, out):
    r = subprocess.run([py, "-I", "-B", script, "--g-out", g_out, "--g-report-sha256", h, "--out", out],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--renderer", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    rng = random.Random(20261124)
    shutil.rmtree(a.work, ignore_errors=True)
    checks = []

    def ck(name, ok, note=""):
        checks.append({"check": name, "pass": bool(ok), "note": note})
        print("%s  %s %s" % ("PASS" if ok else "FAIL", name, note))

    want = ["fig%d.%s" % (i, e) for i in range(1, 6) for e in ("png", "pdf")] + ["s4.png", "s4.pdf", "table1.md",
            "captions.md", "render_report.json", "render_verdict.txt", "render.log"] + ["s%d.md" % i for i in range(1, 8)]
    # T1 intact
    g1 = os.path.join(a.work, "g1"); h1 = build(a.template, g1, rng); o1 = os.path.join(a.work, "o1")
    rc, txt = render(sys.executable, a.renderer, g1, h1, o1)
    ck("T1 intact synthetic input: exit 0 and RENDER: COMPLETE", rc == 0 and "RENDER: COMPLETE" in txt, txt[-300:] if rc else "")
    ck("T1 all 24 expected output files exist", all(os.path.exists(os.path.join(o1, f)) for f in want))
    if os.path.exists(os.path.join(o1, "render_report.json")):
        rep = json.load(open(os.path.join(o1, "render_report.json"), encoding="utf-8"))
        ck("T1 value checks: every plotted series equals its JSON values", rep["value_checks"]["all_equal"] and rep["value_checks"]["n"] > 100, "n=%d" % rep["value_checks"]["n"])
        ck("T1 every output hash in the report equals the file", all(sha(os.path.join(o1, k)) == v for k, v in rep["outputs_sha256"].items()))
        ck("T1 derived-at-rendering list names only Figure 1 means and S4 histograms", all(x["where"].startswith(("fig1", "s4")) for x in rep["derived_at_rendering"]))
        t1 = open(os.path.join(o1, "table1.md"), encoding="utf-8").read()
        src = json.load(open(os.path.join(g1, "g_main_table.json"), encoding="utf-8"))
        ck("T1 Table 1 holds the six classes and the S-GAP0 effects of the source", all(("%+.6f" % r["effects"]["S-GAP0"]["point"]) in t1 and r["class"] in t1 for r in src["main_table"]["rows"]))
        ck("T1 Table 1 carries the E-G1 annotation for L4", "numerical adequacy unresolved" in t1)
    # T2 not-computed values
    g2 = os.path.join(a.work, "g2"); h2 = build(a.template, g2, rng, holes=True); o2 = os.path.join(a.work, "o2")
    rc, txt = render(sys.executable, a.renderer, g2, h2, o2)
    ck("T2 'not computed' values are rendered as such: exit 0", rc == 0, txt[-300:] if rc else "")
    # T3 tampered result file
    g3 = os.path.join(a.work, "g3"); h3 = build(a.template, g3, rng); o3 = os.path.join(a.work, "o3")
    with open(os.path.join(g3, "figdata/fig3.json"), "a") as f: f.write(" ")
    rc, txt = render(sys.executable, a.renderer, g3, h3, o3)
    ck("T3 a changed figure data file is refused: RENDER: FAIL, no figure", rc == 1 and "differs from g_report.json" in txt and not any(f.endswith(".png") for f in os.listdir(o3)))
    # T4 wrong report hash
    o4 = os.path.join(a.work, "o4")
    rc, txt = render(sys.executable, a.renderer, g1, "0" * 64, o4)
    ck("T4 a g_report.json other than the registered one is refused", rc == 1 and "differs from the expected" in txt and not any(f.endswith(".png") for f in os.listdir(o4)))
    # T5 verdict
    g5 = os.path.join(a.work, "g5"); h5 = build(a.template, g5, rng, verdict="G: FAIL"); o5 = os.path.join(a.work, "o5")
    rc, txt = render(sys.executable, a.renderer, g5, h5, o5)
    ck("T5 a run that is not G: COMPLETE is refused", rc == 1 and "not 'G: COMPLETE'" in txt)
    # T6 non-empty output folder
    rc, txt = render(sys.executable, a.renderer, g1, h1, o1)
    ck("T6 a non-empty output folder is refused (exit 2)", rc == 2)
    # T7 missing panel
    g7 = os.path.join(a.work, "g7"); build(a.template, g7, rng)
    d = json.load(open(os.path.join(g7, "figdata/fig4.json"))); d["panels"] = d["panels"][:3]
    json.dump(d, open(os.path.join(g7, "figdata/fig4.json"), "w"), sort_keys=True)
    rp = json.load(open(os.path.join(g7, "g_report.json"))); rp["outputs"]["result_files"]["figdata/fig4.json"] = sha(os.path.join(g7, "figdata/fig4.json"))
    json.dump(rp, open(os.path.join(g7, "g_report.json"), "w"), sort_keys=True)
    rc, txt = render(sys.executable, a.renderer, g7, sha(os.path.join(g7, "g_report.json")), os.path.join(a.work, "o7"))
    ck("T7 a missing panel fails the run (no placeholder panel)", rc == 1 and "not found" in txt)
    src = open(a.renderer, encoding="utf-8").read()
    ck("T8 the renderer imports no torch, scipy or statistics code", not any(w in src for w in ("import torch", "import scipy", "spearman", "bootstrap(", "percentile(")))
    n_fail = sum(1 for c in checks if not c["pass"])
    out = {"schema": "seqref-dlat-g-rendertest/1", "renderer_sha256": sha(a.renderer), "n": len(checks), "failed": n_fail,
           "verdict": "RENDERTEST: PASS" if n_fail == 0 else "RENDERTEST: FAIL", "checks": checks,
           "note": "synthetic numbers in the structure of a real run; no real value rendered"}
    json.dump(out, open(a.report, "w", encoding="utf-8"), indent=1)
    print(out["verdict"], "%d checks, %d failed" % (len(checks), n_fail))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

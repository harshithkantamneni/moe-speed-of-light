"""Job 113 (jobs/113_heldset@vast.sh on the gpu branch): the two-read arms with the in-step fetches off, so that they
hold the scheduled set. Scores the registered predictions H1-H6 per valid host from the raw rows and counters, and
writes prereg/job113.json, prereg/scorecard_113.json, paper/wsg_job113.tex and paper/tab_job113.tex.

    python scripts/job113.py
"""
import glob
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from scripts import job109 as J  # noqa: E402

P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
GLOB = os.environ.get("MOSL_JOB113_GLOB", f"{J.RES}/113[a-e]_heldset@vast")
CELLS = (14, 32)
G = {14: "g11", 32: "g25"}
NM = {14: "Low", 32: "Mid"}
NAMES = ("base", "base0", "bypass0", "bypassplan", "bypassplan0", "fetchplan")
word = J.word


def load113(d):
    J.A_NAMES = NAMES   # this job's process A
    h = J.load(d, CELLS, 0.5, need_b=False, keep_pp=True)
    v0 = J.read(d, "v0.txt")
    h["known"] = "one of job 109's machines" in v0
    for C, c in h["cells"].items():
        dr = c.get("_draws", {})
        c["rel"], c["rel_ci"] = {}, {}
        # ratios between two configurations of the same process: speed of X over speed of Y, paired draws
        for x, y in (("bypassplan0", "base0"), ("bypass0", "base0"), ("base0", "base"), ("bypassplan0", "base"),
                     ("fetchplan", "base"), ("bypassplan", "base"), ("bypass0", "base")):
            k = f"{x}/{y}"
            rx = c["ratio"].get(x) if x != "base" else 1.0
            ry = c["ratio"].get(y) if y != "base" else 1.0
            if rx and ry:
                c["rel"][k] = rx / ry
                dx = dr.get(x) if x != "base" else None
                dy = dr.get(y) if y != "base" else None
                if dx is not None and (dy is not None or y == "base"):
                    v = dx / (dy if dy is not None else 1.0)
                    c["rel_ci"][k] = tuple(float(q) for q in np.percentile(v, [2.5, 97.5]))
        if "fetchplan" in dr and "bypassplan0" in dr:
            v = dr["fetchplan"] - dr["bypassplan0"]
            c["rel_ci"]["fetchplan-bypassplan0"] = tuple(float(q) for q in np.percentile(v, [2.5, 97.5]))
        ct = c["counters"]
        c["misses"] = {n: float(np.mean([x["misses"] for x in v])) for n, v in ct.items()}
        c["admits"] = {n: float(np.mean([x["admits"] for x in v])) for n, v in ct.items()}
        c["fetches"] = {n: float(np.mean([x.get("fetches", 0) for x in v])) for n, v in ct.items()}
        c["hostreads"] = {n: c["misses"][n] + c["admits"][n] for n in ct}
        c.pop("_pp", None); c.pop("_draws", None)
    return h


def clauses113(V):
    out = []

    def add(cid, short, meas, ok, thr, host, ci=None):
        out.append(dict(id=cid, short=short, type="band", measured=None if meas is None else round(float(meas), 4),
                        ci=None if ci is None else [round(float(x), 4) for x in ci], threshold=thr,
                        status=J.status(meas, ok, ci), why="" if ci is not None else "no interval", host=host))
    for h in V:
        nm = f"{h['job']} {h['cpu']}"; c = h["cells"]
        for C in CELLS:
            cc = c[C]
            if cc["rounds"] and "bypassplan0" in cc["misses"] and "bypassplan" in cc["misses"]:
                add(f"{h['job']}-H1-{G[C]}", f"bypassplan0 misses <= 0.88x bypassplan's ({nm}, {G[C]})",
                    cc["misses"]["bypassplan0"] / cc["misses"]["bypassplan"], lambda v: v <= 0.88, "<= 0.88", nm)
        lo = c[14]
        if not lo["rounds"]:
            continue
        if "bypassplan0" in lo["hostreads"] and "base0" in lo["hostreads"]:
            add(f"{h['job']}-H2", f"bypassplan0 host reads <= 0.90x base0's at 11% ({nm})",
                lo["hostreads"]["bypassplan0"] / lo["hostreads"]["base0"], lambda v: v <= 0.90, "<= 0.90", nm)
        r, ci = lo["rel"], lo["rel_ci"]
        if "base0/base" in r:
            add(f"{h['job']}-H3", f"base0/base within [0.80, 1.00] at 11% ({nm})", r["base0/base"], lambda v: 0.80 <= v <= 1.00,
                "0.80 to 1.00", nm, ci.get("base0/base"))
        if "bypassplan0/base0" in r:
            add(f"{h['job']}-H4", f"bypassplan0/base0 >= 1.10 at 11% ({nm})", r["bypassplan0/base0"], lambda v: v >= 1.10, ">= 1.10", nm,
                ci.get("bypassplan0/base0"))
        if "bypass0/base0" in r:
            add(f"{h['job']}-H5", f"bypass0/base0 >= 1.03 at 11% ({nm})", r["bypass0/base0"], lambda v: v >= 1.03, ">= 1.03", nm,
                ci.get("bypass0/base0"))
        if "fetchplan/base" in r and "bypassplan0/base" in r:
            add(f"{h['job']}-H6", f"fetchplan/base >= bypassplan0/base + 0.05 at 11% ({nm})", r["fetchplan/base"] - r["bypassplan0/base"],
                lambda v: v >= 0.05, ">= 0.05", nm, ci.get("fetchplan-bypassplan0"))
    out.append(dict(id="113-valid", short="at least two valid hosts (else single-machine results)", type="condition",
                    measured=len(V), ci=None, threshold=">= 2", status="met" if len(V) >= 2 else "not met", why="", host="pooled"))
    return out


def macros(H, V, cl):
    M = {}

    def rng(k, vals, fmt="{:.2f}"):
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if vals:
            a, b = (fmt.format(x).replace("-", "$-$") for x in (min(vals), max(vals)))
            M[k + "Min"], M[k + "Max"] = a, b
            M[k + "Rng"] = a if a == b else f"{a}--{b}"
    M["hsStarted"] = str(len(H)); M["hsStartedWord"] = word(len(H)); M["hsN"] = str(len(V)); M["hsNWord"] = word(len(V))
    M["hsGated"] = word(sum(1 for h in H if h["gate"]))
    rng("hsRatio", [h["ratio"] for h in V])
    for C in CELLS:
        nm = NM[C]
        sel = [h["cells"][C] for h in V if h["cells"][C]["rounds"]]
        for k, key in (("BaseZero", "base0/base"), ("PlanZero", "bypassplan0/base0"), ("GreedyZero", "bypass0/base0"),
                       ("PlanZeroVsBase", "bypassplan0/base"), ("FewOne", "fetchplan/base"), ("FewTwo", "bypassplan/base"),
                       ("GreedyZeroVsBase", "bypass0/base")):
            rng(f"hs{k}{nm}", [c["rel"].get(key) for c in sel])
        for k, n in (("PlanZero", "bypassplan0"), ("Plan", "bypassplan"), ("BaseZero", "base0"), ("GreedyZero", "bypass0"),
                     ("Base", "base"), ("FewOne", "fetchplan")):
            rng(f"hsMiss{k}{nm}", [c["misses"].get(n) for c in sel], "{:.1f}")
            rng(f"hsAdm{k}{nm}", [c["admits"].get(n) for c in sel], "{:.1f}")
            rng(f"hsReads{k}{nm}", [c["hostreads"].get(n) for c in sel], "{:.1f}")
            rng(f"hsFetch{k}{nm}", [c["fetches"].get(n) for c in sel], "{:.1f}")
        rng(f"hsMissRatio{nm}", [c["misses"]["bypassplan0"] / c["misses"]["bypassplan"] for c in sel
                                 if "bypassplan0" in c["misses"] and "bypassplan" in c["misses"]])
        rng(f"hsReadsRatio{nm}", [c["hostreads"]["bypassplan0"] / c["hostreads"]["base0"] for c in sel
                                  if "bypassplan0" in c["hostreads"] and "base0" in c["hostreads"]])
        rng(f"hsOneReadMargin{nm}", [c["rel"]["fetchplan/base"] - c["rel"]["bypassplan0/base"] for c in sel
                                     if "fetchplan/base" in c["rel"] and "bypassplan0/base" in c["rel"]])
        # the share of reading once's gain that the fetch-free two-read arm gets: (Few-2R CPU only - 1) / (Few-1R - 1)
        rng(f"hsShareOfOne{nm}", [100 * (c["rel"]["bypassplan0/base"] - 1) / (c["rel"]["fetchplan/base"] - 1) for c in sel
                                  if "fetchplan/base" in c["rel"] and "bypassplan0/base" in c["rel"] and c["rel"]["fetchplan/base"] > 1], "{:.0f}")
        rng(f"hsReadsRstar{nm}", [c["hostreads"]["bypassplan0"] / J.RSTAR[C] for c in sel if "bypassplan0" in c["hostreads"]])
    # every clause, the pooled count included (as the tally of Table 10 and the scorecard count them)
    pcl = [c for c in cl if not c["id"].endswith("-valid")]   # the predictions; the validity condition apart
    M["hsClauses"] = str(len(pcl)); M["hsClausesHeld"] = str(sum(c["status"] == "held" for c in pcl))
    M["hsClausesPoint"] = str(sum(c["status"] == "held (point)" for c in pcl)); M["hsClausesFailed"] = str(sum(c["status"] == "failed" for c in pcl))
    for t, w in (("One", "H1"), ("Two", "H2"), ("Three", "H3"), ("Four", "H4"), ("Five", "H5"), ("Six", "H6")):
        cs = [c for c in cl if f"-{w}" in c["id"] and c["host"] != "pooled"]
        if cs:
            M[f"hsH{t}N"] = str(len(cs)); M[f"hsH{t}Failed"] = str(sum(c["status"] == "failed" for c in cs))
            M[f"hsH{t}FailedOn"] = ", ".join(c["host"].split(" ", 1)[1] for c in cs if c["status"] == "failed") or "none"
            M[f"hsH{t}Status"] = "failed" if any(c["status"] == "failed" for c in cs) else (
                "held" if all(c["status"] == "held" for c in cs) else "held (point)")
    return M


def table(V):
    """per valid host: the ratios at 11% with half-widths of their 95% intervals over problems, and the misses"""
    hw = lambda c, k: (c["rel_ci"][k][1] - c["rel_ci"][k][0]) / 2 if k in c["rel_ci"] else None  # noqa: E731
    with open(P("paper", "tab_job113.tex"), "w") as f:
        f.write("% generated by scripts/job113.py\n\\begin{table*}[t]\\centering\\footnotesize\n")
        f.write("\\caption{The two-read path with the in-step fetches off (registered; gpt-oss 11\\%). \\emph{CPU only}: the "
                "configuration with no in-step fetches, every miss served by the CPU, each admission copied in the background. "
                "\\emph{Dep.}: the deployed cache. Speeds are relative to the configuration named above them, on the same "
                "machine; subscripts are half-widths of 95\\% intervals over problems. Misses: \\FewTwo's per token with the "
                "fetch table, and CPU only.}\\label{tab:job113}\n")
        f.write("\\setlength\\tabcolsep{5pt}\n\\begin{tabular}{@{}lrrrrrrr@{}}\\toprule\n")
        f.write(" & & \\multicolumn{1}{c}{vs Dep.} & \\multicolumn{2}{c}{vs Dep.\\ CPU only} & \\multicolumn{2}{c}{vs Dep.} & \\\\\n")
        f.write("\\cmidrule(lr){3-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n")
        f.write(" & & Dep. & \\MinTwo & \\FewTwo & \\FewTwo & & \\FewTwo \\\\\n")
        f.write("Machine & Ratio & CPU only & CPU only & CPU only & CPU only & \\FewOne & misses \\\\\\midrule\n")
        for h in V:
            c = h["cells"][14]
            if not c["rounds"]:
                continue
            r = c["rel"]

            def cell(k):
                if k not in r:
                    return "--"
                w = hw(c, k)
                return f"{r[k]:.2f}" + (f"$_{{\\pm{w:.2f}}}$".replace("0.", ".", 1) if w is not None else "")
            m = c["misses"]
            f.write(f"{h['cpu']} & {h['ratio']:.2f} & {cell('base0/base')} & {cell('bypass0/base0')} & {cell('bypassplan0/base0')} & "
                    f"{cell('bypassplan0/base')} & {cell('fetchplan/base')} & {m.get('bypassplan', float('nan')):.1f} / "
                    f"{m.get('bypassplan0', float('nan')):.1f} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table*}\n")
    from scripts.tabnote import split_caption
    split_caption(P("paper", "tab_job113.tex"))


def main():
    H = [load113(d) for d in sorted(glob.glob(GLOB))]
    V = [h for h in H if h["valid"]]
    for h in H:
        print(f"{h['job']} {h['cpu'][:24]:24s} known={h['known']} ratio {h.get('ratio', float('nan')):.2f} valid {h['valid']} "
              f"gate '{h['gate']}' {h['V2_line']} V3 {h['V3_fail']}")
        for C, c in h["cells"].items():
            if c.get("rounds"):
                print(f"   C{C} rounds {c['rounds']} " + " ".join(f"{k} {v:.3f}" for k, v in sorted(c["rel"].items())))
                print("      misses " + " ".join(f"{k} {v:.1f}" for k, v in sorted(c["misses"].items())))
                print("      admits " + " ".join(f"{k} {v:.1f}" for k, v in sorted(c["admits"].items())))
                print("      fetches " + " ".join(f"{k} {v:.1f}" for k, v in sorted(c["fetches"].items())))
    cl = clauses113(V)
    for c in cl:
        print(c["id"], c["measured"], c["ci"], c["threshold"], c["status"])
    json.dump(dict(hosts=H, valid=[h["job"] for h in V]), open(P("prereg", "job113.json"), "w"), indent=1, default=float)
    json.dump(dict(job="113", script="jobs/113_heldset@vast.sh", commit="f174e0a", scored_by="scripts/job113.py (machine)",
                   clauses=cl), open(P("prereg", "scorecard_113.json"), "w"), indent=1)
    M = macros(H, V, cl)
    with open(P("paper", "wsg_job113.tex"), "w") as f:
        f.write("% generated by scripts/job113.py from job 113\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    table(V)
    for k in sorted(M):
        print(k, M[k])


if __name__ == "__main__":
    main()

"""Job 105: the sum law with the GPU's profiled compute and nothing fitted, the engine's layer-ahead PREFETCH across
machines, and MIN's fewest-admission set on new machines. Scores the predictions of jobs/105_sumlaw@vast.sh by machine
and writes prereg/job105.json, prereg/scorecard_105.json, paper/wsg_job105.tex and paper/tab_job105.tex.

    python scripts/job105.py
"""
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from scripts.job100 import load_host  # noqa: E402
from scripts.panel_099 import _status  # noqa: E402
from scripts.speed_limit import host_rates  # noqa: E402

RES = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
S = 13253760
GL = {14: "g11", 32: "g25"}
NAMES = {"105a": "EPYC 7402P", "105b": "Threadripper 9960X", "105e": "Pf again", "105f": "O4 again (9950X)",
         "105g": "Threadripper PRO 7000"}


def main():
    hosts = {}
    for d in sorted(glob.glob(f"{RES}/105?_sumlaw@vast")):
        if not os.path.exists(f"{d}/ec_g_C14.jsonl"):
            continue
        h = load_host(d)
        h["link_over_cpu"] = h["B_p"] / h["B_c"]
        h["B_host"] = host_rates(open(f"{d}/concur.txt").read()) / 1e9
        gp = json.load(open(f"{d}/g_prof.json")) if os.path.exists(f"{d}/g_prof.json") else {}
        h["G_prof"] = {C: gp.get(f"C{C}", {}).get("G_prof_ms") for C in (14, 32)}
        h["planned"] = {C: os.path.exists(f"{d}/st_g_C{C}_fetchplan.json")
                        and json.load(open(f"{d}/st_g_C{C}_fetchplan.json")).get("oracle_plan", 0) == 1 for C in (14, 32)}
        hosts[os.path.basename(d)[:4]] = h
    clauses = []

    def add(cid, short, typ, meas, ci, ok, thr, host):
        stt, why = _status(meas, ci, ok)
        clauses.append(dict(id=cid, short=short, type=typ, measured=None if meas is None else round(float(meas), 4),
                            ci=None if ci is None else [round(float(x), 4) for x in ci], threshold=thr, status=stt, why=why, host=host))
    errs, rows = [], []
    for job, h in hosts.items():
        nm = NAMES.get(job, job)
        r = h["link_over_cpu"]
        for C, c in h["cells"].items():
            g = GL[C]
            G = h["G_prof"].get(C)
            add(f"{job}-P1-{g}", f"profiled GPU compute 4.0-5.0 ms ({nm}, {g})", "band", G, None, lambda v: 4.0 <= v <= 5.0, "4.0 to 5.0", nm)
            ct = c["counters"].get("base")
            if G is not None and ct:
                reads = ct["misses"] + ct["admits"]
                pred = G + reads * S / (h["B_host"] * 1e9) * 1e3
                e = pred / c["ms"]["base"] - 1
                c["sum_law"] = dict(pred_ms=pred, meas_ms=c["ms"]["base"], err=e, reads=reads, G_prof=G)
                errs.append(e)
                add(f"{job}-P2-{g}", f"deployed time within 6% of G_prof + R S / B_host ({nm}, {g})", "band", abs(e), None,
                    lambda v: v <= 0.06, "<= 0.06", nm)
            sp = c["speed"]
            if "pf" in sp and C == 14:
                if r >= 0.8:
                    add(f"{job}-P3-{g}", f"PREFETCH gains >= 3% (ratio {r:.2f}; {nm}, {g})", "band", sp["pf"][0], sp["pf"][1:],
                        lambda v: v >= 1.03, ">= 1.03", nm)
                elif r < 0.4:
                    add(f"{job}-P3-{g}", f"PREFETCH loses (ratio {r:.2f}; {nm}, {g})", "sign", sp["pf"][0], sp["pf"][1:],
                        lambda v: v < 1, "< 1", nm)
            if "fetchplan" in sp and C == 14 and not h["planned"].get(C):
                add(f"{job}-P4-{g}", f"P4 ({nm}, {g}): the plan did not load", "sign", None, None, lambda v: True, "> 1", nm)
            elif "fetchplan" in sp and C == 14:
                add(f"{job}-P4-{g}", f"fewest-admission set copied in the step beats the deployed cache ({nm}, {g})", "sign",
                    sp["fetchplan"][0], sp["fetchplan"][1:], lambda v: v > 1, "> 1", nm)
                if r < 0.7 and "fetch" in sp:
                    add(f"{job}-P4g-{g}", f"fewest-admission set beats the greedy set by >= 0.03 (ratio {r:.2f}; {nm}, {g})", "band",
                        sp["fetchplan"][0] - sp["fetch"][0], None, lambda v: v >= 0.03, ">= 0.03", nm)
            ctp, ctg = c["counters"].get("fetchplan"), c["counters"].get("fetch")
            if ctp and ctg and C == 14 and h["planned"].get(C):
                add(f"{job}-P5-{g}", f"fewest-admission copies <= 0.65x the greedy set's ({nm}, {g})", "band",
                    ctp["fetches"] / ctg["fetches"], None, lambda v: v <= 0.65, "<= 0.65", nm)
            rows.append((job, nm, r, C, c))
    if errs:
        add("105-P2-median", "median |error| of the sum law at most 4%", "band", float(np.median(np.abs(errs))), None,
            lambda v: v <= 0.04, "<= 0.04", "pooled")
    out = []
    for job, h in hosts.items():
        hh = {k: v for k, v in h.items() if k != "cells"}
        hh["job"] = job
        hh["cells"] = {str(C): {k: v for k, v in c.items() if k != "arr"} for C, c in h["cells"].items()}
        out.append(hh)
    json.dump(dict(hosts=out), open(P("prereg", "job105.json"), "w"), indent=1, default=float)
    json.dump(dict(job="105", script="jobs/105_sumlaw@vast.sh", commit="92ff9fd", scored_by="scripts/job105.py (machine)", clauses=clauses),
              open(P("prereg", "scorecard_105.json"), "w"), indent=1)
    st = Counter(c["status"] for c in clauses)
    M = dict(jgHosts=str(len(hosts)), jgClauses=str(len(clauses)), jgHeld=str(st.get("held", 0)), jgPoint=str(st.get("held (point)", 0)),
             jgFailed=str(st.get("failed", 0)), jgUntested=str(st.get("untested", 0)))

    def rng(key, vals, fmt="{:.2f}"):
        vals = [v for v in vals if v is not None]
        if vals:
            M[key + "Min"] = fmt.format(min(vals)).replace("-", "$-$"); M[key + "Max"] = fmt.format(max(vals)).replace("-", "$-$")
    if errs:
        M["jgErrMed"] = f"{100 * np.median(np.abs(errs)):.1f}"; M["jgErrMax"] = f"{100 * np.max(np.abs(errs)):.1f}"
        M["jgCells"] = str(len(errs))
    rng("jgGprof", [g for h in hosts.values() for g in h["G_prof"].values()], "{:.1f}")
    rng("jgRatio", [h["link_over_cpu"] for h in hosts.values()])
    for C, nmC in ((14, "Low"), (32, "Mid")):
        sel = [x for x in rows if x[3] == C]
        rng(f"jgPf{nmC}", [c["speed"]["pf"][0] for *_, c in sel if "pf" in c["speed"]])
        rng(f"jgFetchPlan{nmC}", [c["speed"]["fetchplan"][0] for *_, c in sel if "fetchplan" in c["speed"]])
        rng(f"jgFetch{nmC}", [c["speed"]["fetch"][0] for *_, c in sel if "fetch" in c["speed"]])
    # the layer-ahead copy by link class, at 11%; and the slow-link machines (ratio < 0.4) for the two MIN sets
    for nmC, sel in (("Fast", lambda r: r >= 0.8), ("Mid", lambda r: 0.4 <= r < 0.8), ("Slow", lambda r: r < 0.4)):
        v = [c["speed"]["pf"][0] for _, _, r, C, c in rows if C == 14 and sel(r) and "pf" in c["speed"]]
        if v:
            rng(f"jgPfLow{nmC}", v)
            M[f"jgPfLow{nmC}N"] = str(len(v))
    slow = [(r, c) for _, _, r, C, c in rows if C == 14 and r < 0.4 and "fetchplan" in c["speed"] and "fetch" in c["speed"]]
    if slow:
        rng("jgSlowFetch", [c["speed"]["fetch"][0] for _, c in slow])
        rng("jgSlowFetchPlan", [c["speed"]["fetchplan"][0] for _, c in slow])
        rng("jgSlowBypassPlan", [c["speed"]["bypassplan"][0] for _, c in slow if "bypassplan" in c["speed"]])
        rng("jgSlowRatio", [r for r, _ in slow])
        M["jgSlowN"] = str(len(slow))
    rng("jgBypassPlanLow", [c["speed"]["bypassplan"][0] for *_, C, c in rows if C == 14 and "bypassplan" in c["speed"]])
    rng("jgBypassPlanMid", [c["speed"]["bypassplan"][0] for *_, C, c in rows if C == 32 and "bypassplan" in c["speed"]])
    cp = [c["counters"]["fetchplan"]["fetches"] / c["counters"]["fetch"]["fetches"] for job, _, r, C, c in rows
          if hosts[job]["planned"].get(C) and "fetchplan" in c["counters"] and "fetch" in c["counters"]]
    rng("jgCopyRatio", cp)
    errs_by = {C: [c["sum_law"]["err"] for *_, CC, c in rows if CC == C and "sum_law" in c] for C in (14, 32)}
    for C, nmC in ((14, "Low"), (32, "Mid")):
        if errs_by[C]:
            M[f"jgErr{nmC}Med"] = f"{100 * np.median(errs_by[C]):.1f}".replace("-", "$-$")
            rng(f"jgErr{nmC}", [100 * e for e in errs_by[C]], "{:.1f}")
            fl = [100 * e for e in errs_by[C] if abs(e) > 0.06]
            M[f"jgErr{nmC}FailN"] = str(len(fl))
            if fl:
                rng(f"jgErr{nmC}Fail", fl, "{:.1f}")
            ok = [100 * e for e in errs_by[C] if abs(e) <= 0.06]
            if ok and C == 32:
                rng(f"jgErr{nmC}Held", ok, "{:.1f}")
    M["jgPfLoseN"] = str(sum(1 for *_, C, c in rows if C == 14 and "pf" in c["speed"] and c["speed"]["pf"][2] < 1))
    # the layer-ahead copy's counters: the share of its copies that the next layer used
    use = {C: [] for C in (14, 32)}
    for d in sorted(glob.glob(f"{RES}/105?_sumlaw@vast")):
        for C in (14, 32):
            f = f"{d}/st_g_C{C}_pf.json"
            if os.path.exists(f):
                st = json.load(open(f))
                if st.get("prefetches"):
                    use[C].append(st["prefetch_useful"] / st["prefetches"])
    for C, nmC in ((14, "Low"), (32, "Mid")):
        rng(f"jgPfUseful{nmC}", [100 * u for u in use[C]], "{:.0f}")
    # MIN's fewest-admission set over every launch that ran it (jobs 104 and 105), at gpt-oss 11%
    machine = {"104a": "Pf", "105e": "Pf", "104b": "285K", "104c": "5950X", "105a": "EPYC 7402P", "105b": "TR 9960X", "105f": "O4"}
    pl = []
    for pj in (P("prereg", "job104.json"), P("prereg", "job105.json")):
        for h in json.load(open(pj))["hosts"]:
            c = h["cells"].get("14")
            if c and h["planned"].get("14") and "fetchplan" in c["speed"] and "fetch" in c["speed"]:
                pl.append((h["job"], h["link_over_cpu"], c["speed"]["fetch"][0], c["speed"]["fetchplan"][0]))
    M["jhLaunches"] = str(len(pl)); M["jhMachines"] = str(len({machine.get(j, j) for j, *_ in pl}))
    rng("jhPlanLow", [x[3] for x in pl]); rng("jhRatio", [x[1] for x in pl])
    bal = [x for x in pl if x[1] < 0.7]
    rng("jhGainBal", [x[3] - x[2] for x in bal]); M["jhBalN"] = str(len(bal))
    rng("jhBalRatio", [x[1] for x in bal]); M["jhBalMachines"] = str(len({machine.get(j, j) for j, *_ in bal}))
    fast = [x for x in pl if x[1] >= 0.7]
    if fast:
        M["jhGainFast"] = f"{fast[0][3] - fast[0][2]:.2f}".replace("-", "$-$"); M["jhFastRatio"] = f"{fast[0][1]:.2f}"
    slow = [x for x in pl if x[1] < 0.4]
    rng("jhSlowFetch", [x[2] for x in slow]); rng("jhSlowPlan", [x[3] for x in slow]); rng("jhSlowRatio", [x[1] for x in slow])
    M["jhSlowLaunches"] = str(len(slow)); M["jhSlowMachines"] = str(len({machine.get(j, j) for j, *_ in slow}))
    f5 = next((h for j, h in hosts.items() if j == "105f"), None)
    if f5 and "32" in {str(k) for k in f5["cells"]}:
        sp = f5["cells"][32]["speed"]
        if "bypassplan" in sp and "bypass" in sp:
            M["jhFastBypassMidBehind"] = f"{sp['bypass'][0] - sp['bypassplan'][0]:.2f}".replace("-", "$-$")
    with open(P("paper", "wsg_job105.tex"), "w") as f:
        f.write("% generated by scripts/job105.py from the job 105 hosts (results/105?_sumlaw@vast)\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    with open(P("paper", "tab_job105.tex"), "w") as f:
        f.write("% generated by scripts/job105.py\n\\begin{table}[t]\\centering\\footnotesize\n")
        f.write("\\caption{Job 105, registered before the runs, nothing fitted: the deployed cache's time per token against "
                "the plain form of \\cref{eq:sum} (without the overlap term, which came after this job), the GPU's compute profiled on the same host before the timed runs plus the run's counted host "
                "reads at the machine's best rate; "
                "and the speed, relative to the deployed cache, of the deployed cache with the engine's layer-ahead copy (each "
                "layer's top predicted expert of the next layer copied ahead of use) and of MIN's greedy and fewest-admission sets "
                "copied in the step. Hosts sorted by the probe's link-to-CPU ratio.}\\label{tab:job105}\n")
        f.write("\\setlength\\tabcolsep{2.5pt}\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{@{}lrlrrrrrr@{}}\\toprule\n")
        f.write(" & Link/ & & $G_{\\text{prof}}$ & \\multicolumn{2}{c}{Deployed (ms)} & Layer- & \\multicolumn{2}{c}{MIN, in step} \\\\\n")
        f.write("Host & CPU & Budget & (ms) & plain Eq.~3 & measured & ahead & greedy & fewest \\\\\\midrule\n")
        for job, nm, r, C, c in sorted(rows, key=lambda x: (x[2], x[3])):
            sl = c.get("sum_law", {})
            sp = c["speed"]
            fmt = lambda k: f"{sp[k][0]:.2f}" if k in sp else "--"  # noqa: E731
            f.write(f"{nm} & {r:.2f} & gpt-oss {'11' if C == 14 else '25'}\\% & {sl.get('G_prof', float('nan')):.1f} & "
                    f"{sl.get('pred_ms', float('nan')):.1f} & {sl.get('meas_ms', float('nan')):.1f} & {fmt('pf')} & {fmt('fetch')} & {fmt('fetchplan')} \\\\\n")
        f.write("\\bottomrule\\end{tabular}}\\end{table}\n")
    print(dict(st))
    for c in clauses:
        print(c["status"], c["id"], c["short"], c["measured"], c["ci"])
    for job, nm, r, C, c in rows:
        print(job, nm, f"ratio {r:.2f} Bhost {hosts[job]['B_host']:.1f}", C, {k: round(v[0], 3) for k, v in c["speed"].items()}, c.get("sum_law"))
    print(M)


if __name__ == "__main__":
    main()

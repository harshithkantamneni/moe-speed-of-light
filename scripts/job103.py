"""Job 103: the minimum-admission MIN set against the greedy one. Scores the predictions of jobs/103_minadm@vast.sh by
machine and writes prereg/job103.json, prereg/scorecard_103.json, paper/wsg_job103.tex and paper/tab_job103.tex.

States per cell: base (deployed set, CPU then background copy), foa (deployed set, copied in the step), bypass /
bypassplan (MIN's set, greedy / fewest admissions, CPU then background copy), fetch / fetchplan (the same sets copied in
the step). The 2x2 accounting (scripts/factorial_shapley.py: effects) is computed with each MIN set against the bound
of each host (limit1_of: Eq. 1 at the probe's highest rate).

    python scripts/job103.py [103|104]      # job 104 is job 103 again with the plan in its own environment
"""
import glob
import json
import os
import sys
from collections import Counter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from scripts.job100 import load_host  # noqa: E402
from scripts.panel_099 import _status, cell_data  # noqa: E402
from scripts.factorial_shapley import limit1_of  # noqa: E402

RES = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
GL = {14: "g11", 32: "g25"}
CELL = {14: "gpt-oss 11%", 32: "gpt-oss 25%"}
NAMES = {"103a": "Pf again", "103b": "285K, 40 GB/s link", "103c": "7800X3D", "103d": "Pg again (5950X)", "103e": "3970X",
         "104a": "Pf again", "104b": "285K, 40 GB/s link", "104c": "Pg again (5950X)"}
RELAUNCHES = {"103a", "103d", "104a", "104c"}   # Pf and Pg of the panel (same offer and GPU as 099f and 099g)
PREFIX = {"103": "je", "104": "jf"}
COMMIT = {"103": "c19e186", "104": "dbabcc1"}


def acc(ms, limit, mset):
    """the 2x2 with MIN's set taken as `mset` ('' greedy, 'plan' fewest admissions); shares of the gap"""
    b, fo = ms["base"], ms["foa"]
    by, fe = ms["bypass" + mset], ms["fetch" + mset]
    gap = b - limit
    e = dict(gap=gap, set_two=b - by, set_one=fo - fe, reads_online=b - fo, reads_min=by - fe, both=b - fe)
    e["phi_set"] = 0.5 * (e["set_two"] + e["set_one"])
    e["phi_reads"] = 0.5 * (e["reads_online"] + e["reads_min"])
    e["interaction"] = e["set_one"] - e["set_two"]
    return {k: v / gap for k, v in e.items() if k != "gap"} | {"gap_ms": gap}


def planned(d, C, k):
    """did the engine follow a plan in configuration k (its stats file says so)?"""
    f = f"{d}/st_g_C{C}_{k}.json"
    return os.path.exists(f) and json.load(open(f)).get("oracle_plan", 0) == 1


def main(jobid="103"):
    pre = PREFIX[jobid]
    hosts = {}
    for d in sorted(glob.glob(f"{RES}/{jobid}?_minadm@vast")):
        if not os.path.exists(f"{d}/ec_g_C14.jsonl"):
            continue
        h = load_host(d)
        h["link_over_cpu"] = h["B_p"] / h["B_c"]
        h["limit"] = limit1_of(d)
        h["plan"] = {}
        h["planned"] = {C: planned(d, C, "fetchplan") and planned(d, C, "bypassplan") for C in (14, 32)}
        for C in (14, 32):
            f = f"{d}/plan_g{C}.txt"
            if os.path.exists(f):
                txt = open(f).read()
                try:
                    h["plan"][C] = json.loads(txt[txt.index("{"):txt.rindex("}") + 1])
                except ValueError:
                    pass
        hosts[os.path.basename(d)[:4]] = h
    clauses = []

    def add(cid, short, typ, meas, ci, ok, thr, host):
        stt, why = _status(meas, ci, ok)
        clauses.append(dict(id=cid, short=short, type=typ, measured=None if meas is None else round(float(meas), 4),
                            ci=None if ci is None else [round(float(x), 4) for x in ci], threshold=thr, status=stt, why=why, host=host))
    rows = []
    for job, h in hosts.items():
        nm = NAMES.get(job, job)
        r = h["link_over_cpu"]
        for C, c in h["cells"].items():
            g = GL[C]
            ct, sp = c["counters"], c["speed"]
            if not h["planned"].get(C):   # the plan configurations ran greedy MIN: every plan clause is untested
                ks = ["P1-copies", "P1-misses", "P2-reads", "P6"] + (["P3"] if r < 0.35 and C == 14 else []) + \
                     (["P4"] if r < 0.7 and C == 14 else []) + (["P5"] if r >= 0.85 else []) + (["P7"] if C == 14 else [])
                for k in ks:
                    add(f"{job}-{k}-{g}", f"{k} ({nm}, {g}): no plan on this host", "band", None, None, lambda v: True, "", nm)
                continue
            # 1. counters
            if "fetch" in ct and "fetchplan" in ct:
                add(f"{job}-P1-copies-{g}", f"fetchplan's in-step copies <= 0.70x fetch's ({nm}, {g})", "band",
                    ct["fetchplan"]["fetches"] / ct["fetch"]["fetches"], None, lambda v: v <= 0.70, "<= 0.70", nm)
                add(f"{job}-P1-misses-{g}", f"fetchplan's misses within 3% of fetch's ({nm}, {g})", "band",
                    abs(ct["fetchplan"]["misses"] / ct["fetch"]["misses"] - 1), None, lambda v: v <= 0.03, "<= 0.03", nm)
                pl = h["plan"].get(C, {}).get("sim")
                if pl:
                    for k, sk in (("fetch", "greedy_misses"), ("fetchplan", "plan_misses")):
                        add(f"{job}-P1-replay-{k}-{g}", f"{k}: engine misses within 2% of the host's replay ({nm}, {g})", "band",
                            abs(ct[k]["misses"] / pl[sk] - 1), None, lambda v: v <= 0.02, "<= 0.02", nm)
            # 2. two reads
            if "bypass" in ct and "bypassplan" in ct:
                rd = lambda k: ct[k]["misses"] + ct[k]["admits"]  # noqa: E731
                add(f"{job}-P2-reads-{g}", f"bypassplan's host reads <= 0.90x bypass's ({nm}, {g})", "band",
                    rd("bypassplan") / rd("bypass"), None, lambda v: v <= 0.90, "<= 0.90", nm)
            # 3-5. speed of the copy in the step
            if "fetchplan" in sp and "fetch" in sp:
                if r < 0.35 and C == 14:
                    f = sp["fetchplan"]
                    add(f"{job}-P3-{g}", f"fetchplan/base above 1 ({nm}, {g})", "sign", f[0], f[1:], lambda v: v > 1, "> 1", nm)
                if r < 0.7 and C == 14:
                    add(f"{job}-P4-{g}", f"fetchplan/base exceeds fetch/base by >= 0.03 ({nm}, ratio {r:.2f}, {g})", "band",
                        sp["fetchplan"][0] - sp["fetch"][0], None, lambda v: v >= 0.03, ">= 0.03", nm)
                if r >= 0.85:
                    add(f"{job}-P5-{g}", f"fetchplan/base within 0.05 of fetch/base ({nm}, ratio {r:.2f}, {g})", "band",
                        abs(sp["fetchplan"][0] - sp["fetch"][0]), None, lambda v: v <= 0.05, "<= 0.05", nm)
            if "bypassplan" in sp and "bypass" in sp:
                add(f"{job}-P6-{g}", f"bypassplan/base >= bypass/base - 0.01 ({nm}, {g})", "band",
                    sp["bypassplan"][0] - sp["bypass"][0], None, lambda v: v >= -0.01, ">= -0.01", nm)
            # 7. the interaction, with each MIN set
            lim = h["limit"].get(CELL[C])
            if lim and all(k in c["ms"] for k in ("base", "foa", "bypass", "fetch", "bypassplan", "fetchplan")):
                ag, ap = acc(c["ms"], lim, ""), acc(c["ms"], lim, "plan")
                c["acc_greedy"], c["acc_plan"] = ag, ap
                if C == 14:
                    add(f"{job}-P7-{g}", f"interaction share smaller with the plan's set ({nm}, {g})", "sign",
                        ag["interaction"] - ap["interaction"], None, lambda v: v > 0, "> 0", nm)
                rows.append((job, nm, r, C, c))
        # 8. Pf relaunched
        if job in ("103a", "104a"):
            for C, c in h["cells"].items():
                cd = cell_data(f"{RES}/101a_crossover@vast", "g", C)
                if cd:
                    b0 = cd["arr"]["base"].mean()
                    add(f"{job}-P8-base-{GL[C]}", f"base time within 3% of job 101a ({GL[C]})", "band", abs(c["ms"]["base"] / b0 - 1), None,
                        lambda v: v <= 0.03, "<= 0.03", nm)
                    add(f"{job}-P8-fetch-{GL[C]}", f"fetch/base within 0.03 of job 101a ({GL[C]})", "band",
                        abs(c["speed"]["fetch"][0] - b0 / cd["arr"]["fetch"].mean()), None, lambda v: v <= 0.03, "<= 0.03", nm)
    out = []
    for job, h in hosts.items():
        hh = {k: v for k, v in h.items() if k not in ("cells",)}
        hh["job"] = job
        hh["cells"] = {str(C): {k: v for k, v in c.items() if k != "arr"} for C, c in h["cells"].items()}
        out.append(hh)
    json.dump(dict(hosts=out), open(P("prereg", f"job{jobid}.json"), "w"), indent=1, default=float)
    json.dump(dict(job=jobid, script=f"jobs/{jobid}_minadm@vast.sh", commit=COMMIT[jobid], scored_by="scripts/job103.py (machine)", clauses=clauses),
              open(P("prereg", f"scorecard_{jobid}.json"), "w"), indent=1)
    st = Counter(c["status"] for c in clauses)
    M = {pre + "Hosts": str(len(hosts)), pre + "Clauses": str(len(clauses)), pre + "Held": str(st.get("held", 0)),
         pre + "Point": str(st.get("held (point)", 0)), pre + "Failed": str(st.get("failed", 0)), pre + "Untested": str(st.get("untested", 0))}

    def rng(key, vals, fmt="{:.2f}"):
        key = pre + key[2:]
        vals = [v for v in vals if v is not None]
        if vals:
            M[key + "Min"] = fmt.format(min(vals)).replace("-", "$-$"); M[key + "Max"] = fmt.format(max(vals)).replace("-", "$-$")
    cells = [x for x in rows if "acc_plan" in x[4]]
    rng("jeCopyRatio", [c["counters"]["fetchplan"]["fetches"] / c["counters"]["fetch"]["fetches"] for *_, c in cells])
    if cells:
        rng("jeRatio", [r for _, _, r, _, _ in cells])
        M[pre + "PlanAdm"] = f"{min(c['counters']['fetchplan']['fetches'] for *_, c in cells if c is not None):.1f}"
    rng("jeMissDev", [100 * (c["counters"]["fetchplan"]["misses"] / c["counters"]["fetch"]["misses"] - 1) for *_, c in cells], "{:.1f}")
    for C, nmC in ((14, "Low"), (32, "Mid")):
        sel = [x for x in cells if x[3] == C]
        rng(f"jeFetch{nmC}", [c["speed"]["fetch"][0] for *_, c in sel])
        rng(f"jeFetchPlan{nmC}", [c["speed"]["fetchplan"][0] for *_, c in sel])
        rng(f"jeBypassPlan{nmC}", [c["speed"]["bypassplan"][0] for *_, c in sel])
        rng(f"jeGain{nmC}", [c["speed"]["fetchplan"][0] - c["speed"]["fetch"][0] for *_, c in sel])
        rng(f"jeInterGreedy{nmC}", [100 * c["acc_greedy"]["interaction"] for *_, c in sel], "{:.0f}")
        rng(f"jeInterPlan{nmC}", [100 * c["acc_plan"]["interaction"] for *_, c in sel], "{:.0f}")
        rng(f"jeBothGreedy{nmC}", [100 * c["acc_greedy"]["both"] for *_, c in sel], "{:.0f}")
        rng(f"jeBothPlan{nmC}", [100 * c["acc_plan"]["both"] for *_, c in sel], "{:.0f}")
        rng(f"jeSetAlonePlan{nmC}", [100 * c["acc_plan"]["set_two"] for *_, c in sel], "{:.0f}")
    pf = hosts.get(jobid + "a")
    if pf and 14 in pf["cells"] and pf["planned"].get(14):
        c = pf["cells"][14]
        M[pre + "PfFetch"] = f"{c['speed']['fetch'][0]:.2f}"
        f = c["speed"]["fetchplan"]
        M[pre + "PfPlan"] = f"{f[0]:.2f}"; M[pre + "PfPlanLo"] = f"{f[1]:.2f}"; M[pre + "PfPlanHi"] = f"{f[2]:.2f}"
        M[pre + "PfRatio"] = f"{pf['link_over_cpu']:.2f}"
    if not cells:   # no plan anywhere (job 103): only the greedy states, on the new machines (103a is Pf again)
        new = [h for j, h in hosts.items() if j not in RELAUNCHES]
        rng("jeFetchAll", [h["cells"][C]["speed"]["fetch"][0] for h in hosts.values() for C in h["cells"]])
        rng("jeNewFetchLow", [h["cells"][14]["speed"]["fetch"][0] for h in new if 14 in h["cells"]])
        rng("jeNewRatio", [h["link_over_cpu"] for h in new])
        M[pre + "NewN"] = str(len(new))
        # the 2x2 on the new machines below the half line: does loading in the step cost time (Shapley value below 0)?
        sl = []
        for h in new:
            if h["link_over_cpu"] >= 0.5:
                continue
            for C, c in h["cells"].items():
                lim = h["limit"].get(CELL[C])
                if lim and all(k in c["ms"] for k in ("base", "foa", "bypass", "fetch")):
                    sl.append(acc(c["ms"] | {"bypassplan": c["ms"]["bypass"], "fetchplan": c["ms"]["fetch"]}, lim, "")["phi_reads"])
        M[pre + "SlowReadsLoseN"] = str(sum(v < 0 for v in sl)); M[pre + "SlowCells"] = str(len(sl))
        M[pre + "SlowN"] = str(sum(1 for h in new if h["link_over_cpu"] < 0.5))
    with open(P("paper", f"wsg_job{jobid}.tex"), "w") as fo:
        fo.write(f"% generated by scripts/job103.py from the job {jobid} hosts (results/{jobid}?_minadm@vast)\n")
        for k in sorted(M):
            fo.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    with open(P("paper", f"tab_job{jobid}.tex"), "w") as fo:
        fo.write("% generated by scripts/job103.py\n\\begin{table}[t]\\centering\\footnotesize\n")
        fo.write(f"\\caption{{Which of MIN's hit-optimal sets the oracle follows (job {jobid}): speed relative to the deployed cache on the same "
                 "host, with MIN's set loaded by the CPU (two reads) or in the step (one read), as the greedy rule picks it or as the "
                 "schedule with the fewest admissions does; and the copies per token each makes in the step. Hosts sorted by the "
                 f"probe's link-to-CPU ratio.}}\\label{{tab:job{jobid}}}\n")
        fo.write("\\setlength\\tabcolsep{2.5pt}\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{@{}lrlrrrrrr@{}}\\toprule\n")
        fo.write(" & Link/ & & \\multicolumn{2}{c}{By the CPU} & \\multicolumn{2}{c}{In the step} & \\multicolumn{2}{c}{Copies/token} \\\\\n")
        fo.write("Host & CPU & Budget & greedy & fewest & greedy & fewest & greedy & fewest \\\\\\midrule\n")
        for job, nm, r, C, c in sorted(cells, key=lambda x: (x[2], x[3])):
            sp, ct = c["speed"], c["counters"]
            fo.write(f"{nm} & {r:.2f} & gpt-oss {'11' if C == 14 else '25'}\\% & {sp['bypass'][0]:.2f} & {sp['bypassplan'][0]:.2f} & "
                     f"{sp['fetch'][0]:.2f} & {sp['fetchplan'][0]:.2f} & {ct['fetch']['fetches']:.1f} & {ct['fetchplan']['fetches']:.1f} \\\\\n")
        fo.write("\\bottomrule\\end{tabular}}\\end{table}\n")
    print(dict(st))
    for c in clauses:
        print(c["status"], c["id"], c["short"], c["measured"], c["ci"])
    for job, nm, r, C, c in cells:
        print(job, nm, f"ratio {r:.2f}", C, {k: round(v[0], 3) for k, v in c["speed"].items()},
              "copies", round(c["counters"]["fetch"]["fetches"], 2), round(c["counters"]["fetchplan"]["fetches"], 2),
              "inter", round(c["acc_greedy"]["interaction"], 3), round(c["acc_plan"]["interaction"], 3),
              "both", round(c["acc_greedy"]["both"], 3), round(c["acc_plan"]["both"], 3))
    print(M)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "103")

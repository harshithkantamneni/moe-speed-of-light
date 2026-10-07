"""Job 108: the time relation (eq. sum with its overlap term) registered on machines never rented before, restricted in
advance to desktop-class machines (one NUMA node, at most 32 usable cores; gated before any download); gpt-oss-120b,
the deployed cache only, at 11% and 25% in two rounds each. Scores the predictions in the header of
jobs/108_smallhosts@vast.sh and writes prereg/job108.json, prereg/scorecard_108.json, paper/wsg_job108.tex and
paper/tab_job108.tex.

    python scripts/job108.py
"""
import glob
import json
import os
import re
import sys
from collections import Counter

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from scripts.panel_099 import _status  # noqa: E402
import scripts.job106 as j106  # noqa: E402
from scripts.job107 import short_cpu, losses, machine_classes  # noqa: E402

RES = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
S = 13253760
CELLS = (("g", 14, 2), ("g", 32, 2))
LAB = {14: "11\\%", 32: "25\\%"}
DAG = "$^{\\dagger}$"
REQUIRED, BAND, BAND6, MEDMAX, EFF = 3, 0.08, 0.06, 0.04, (0.85, 1.25)
num = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight"]
word = lambda n: num[n] if n < len(num) else str(n)  # noqa: E731


def gate(d):
    txt = open(f"{d}/v0.txt").read() if os.path.exists(f"{d}/v0.txt") else ""
    m = re.search(r"GPU UUID (\S+); host memory in use (\d+) GB", txt)
    used = int(m.group(2)) if m else None
    for tag in ("V0 FAIL:", "V0c FAIL:"):
        if tag in txt:
            return False, txt.split(tag, 1)[1].split("\n")[0].strip(), used
    if not os.path.exists(f"{d}/concur.txt") or not glob.glob(f"{d}/ec_g_C14_r*.jsonl"):
        return False, "stopped before the timed runs", used
    return True, "", used


def fallback_G():
    """median profiled G per budget over the desktop-class hosts profiled before job 108 (jobs 105-107; keys C14/G14)"""
    out = {}
    for C in (14, 32):
        v = []
        for f in sorted(glob.glob(f"{RES}/10[5-7]?_*@vast/g_prof.json")):
            j = os.path.basename(os.path.dirname(f))[:4]
            if j in ("106c", "106d", "107c", "107d", "107e"):   # servers
                continue
            gp = json.load(open(f)); g = (gp.get(f"G{C}") or gp.get(f"C{C}") or {}).get("G_prof_ms")
            if g and g > 1.0:
                v.append(g)
        out[C] = (float(np.median(v)), len(v), min(v), max(v))
    return out


def main():
    j106.CELLS = CELLS
    FB = fallback_G()
    j106.RNG = np.random.default_rng(108)
    hosts, gated = {}, {}
    for d in sorted(glob.glob(f"{RES}/108?_smallhosts@vast")):
        job = os.path.basename(d)[:4]
        ok, why, used = gate(d)
        cpu = ""
        if os.path.exists(f"{d}/cpu.txt"):
            m = re.search(r"Model name:\s*(.+)", open(f"{d}/cpu.txt").read())
            cpu = m.group(1).strip() if m else ""
        if not ok:
            gated[job] = dict(dir=os.path.basename(d), reason=why, used_gb=used, cpu=short_cpu(cpu) if cpu else "")
            continue
        h = j106.load(d)
        h["name"] = short_cpu(h.get("cpu"))
        # a profile whose trace holds no decode kernels (driver 570 with this nsys) gives no G: the median profiled G
        # of the earlier desktop-class hosts stands in, reported as a deviation from the registration
        h["G_fallback"] = False
        for c in h["cells"].values():
            if c.get("G") is None or c["G"] < 1.0:
                c["G_prof_bad"] = c.get("G"); c["G"] = FB[c["C"]][0]; h["G_fallback"] = True
                pred = {}
                for k, ct in c["counters"].items():
                    pred[k] = dict(olap=j106.law(c["G"], ct["misses"], ct["admits"], h["b_host"], S),
                                   plain=j106.law(c["G"], ct["misses"], ct["admits"], h["b_host"], S, False))
                c["pred"] = pred
                c["err"] = {k: {f: pred[k][f] / c["ms"][k] - 1 for f in ("olap", "plain")} for k in pred}
        ls = losses(d)
        h["loss"] = {f"{t}{C}_r{r}": v for (t, C, r), v in ls.items()}
        h["V1"] = bool(ls) and max(abs(v / 0.190 - 1) for v in ls.values()) <= 0.02
        b = h["cells"].get("g14", {}).get("base_rounds", [])
        h["V2_spread"] = (max(b) / min(b) - 1) if len(b) >= 2 else None
        h["V2"] = h["V2_spread"] is not None and h["V2_spread"] <= 0.02
        h["valid"] = h["V1"] and h["V2"] and "g32" in h["cells"]
        for c in h["cells"].values():
            if c.get("G") is not None and "base" in c["counters"]:
                ct = c["counters"]["base"]; T = c["ms"]["base"]; G = c["G"]
                c["eff"] = (ct["misses"] + ct["admits"] * (1 - G / T)) * S / ((T - G) * 1e-3) / 1e9 / h["b_host"]
        hosts[job] = h
    valid = {j: h for j, h in hosts.items() if h["valid"]}
    clauses = []

    def add(cid, short, typ, meas, ci, ok, thr, host):
        stt, why = _status(meas, ci, ok)
        clauses.append(dict(id=cid, short=short, type=typ, measured=None if meas is None else round(float(meas), 4),
                            ci=None if ci is None else [round(float(x), 4) for x in ci], threshold=thr, status=stt, why=why, host=host))
    errs, plain, effs = [], [], []
    for job, h in valid.items():
        nm = f"{job} {h['name']}"
        for key, c in h["cells"].items():
            g = {14: "g11", 32: "g25"}[c["C"]]
            add(f"{job}-P1-{g}", f"profiled GPU compute 4.0-5.0 ms ({nm}, {g})", "band", None if "G_prof_bad" in c else c["G"], None,
                lambda v: 4.0 <= v <= 5.0, "4.0 to 5.0", nm)
            e = c["err"]["base"]["olap"]; errs.append(e); plain.append(c["err"]["base"]["plain"])
            add(f"{job}-P2-{g}", f"relation within 8% of base ({nm}, {g})", "band", abs(e), None, lambda v: v <= BAND, "<= 0.08", nm)
            effs.append(c["eff"])
            add(f"{job}-P3-{g}", f"implied read rate 0.85-1.25 of B_host ({nm}, {g})", "band", c["eff"], None,
                lambda v: EFF[0] <= v <= EFF[1], "0.85 to 1.25", nm)
    if errs:
        a = np.abs(errs)
        add("108-P2-six", "at most one cell beyond 6%", "band", int(np.sum(a > BAND6)), None, lambda v: v <= 1, "<= 1", "pooled")
        add("108-P2-median", "median |error| at most 4%", "band", float(np.median(a)), None, lambda v: v <= MEDMAX, "<= 0.04", "pooled")
    add("108-valid", "at least 3 valid machines", "band", len(valid), None, lambda v: v >= REQUIRED, ">= 3", "pooled")
    json.dump(dict(gated=gated, hosts=[dict(**{k: v for k, v in h.items() if k != "cells"}, cells=h["cells"]) for h in hosts.values()]),
              open(P("prereg", "job108.json"), "w"), indent=1, default=float)
    json.dump(dict(job="108", script="jobs/108_smallhosts@vast.sh", commit="88e954c", scored_by="scripts/job108.py (machine)", clauses=clauses),
              open(P("prereg", "scorecard_108.json"), "w"), indent=1)
    st = Counter(c["status"] for c in clauses)
    M = dict(jmLaunched=str(len(hosts) + len(gated)), jmGated=str(len(gated)), jmRan=str(len(hosts)), jmValid=str(len(valid)),
             jmInvalid=str(len(hosts) - len(valid)), jmClauses=str(len(clauses)), jmHeld=str(st.get("held", 0)),
             jmPoint=str(st.get("held (point)", 0)), jmFailed=str(st.get("failed", 0)))
    M["jmValidWord"] = word(len(valid)); M["jmGatedWord"] = word(len(gated)); M["jmRequiredWord"] = word(REQUIRED)
    M["jmLaunchedWord"] = word(len(hosts) + len(gated))
    M["jmGatedClass"] = str(sum("desktop-class" in g["reason"] for g in gated.values()))
    M["jmGatedMem"] = str(sum("memory in use" in g["reason"] for g in gated.values()))
    M["jmGatedUuid"] = str(sum("rented before" in g["reason"] for g in gated.values()))
    M["jmValidNames"] = ", ".join(h["name"] for h in sorted(valid.values(), key=lambda h: h["ratio"]))
    allok = all(c["status"] != "failed" for c in clauses)
    fbh = [h for h in valid.values() if h["G_fallback"]]
    M["jmFallbackN"] = word(len(fbh)); M["jmFallbackNames"] = ", ".join(h["name"] for h in fbh)
    M["jmFallbackGLow"] = f"{FB[14][0]:.2f}"; M["jmFallbackGMid"] = f"{FB[32][0]:.2f}"; M["jmFallbackHosts"] = str(FB[14][1])
    # the verdict without the hosts whose G was not profiled
    ex = [abs(c["err"]["base"]["olap"]) for h in valid.values() if not h["G_fallback"] for c in h["cells"].values()]
    if ex:
        M["jmProfOnlyCells"] = str(len(ex)); M["jmProfOnlyWithinEight"] = str(sum(e <= BAND for e in ex))
        M["jmProfOnlyMed"] = f"{100 * np.median(ex):.1f}"
    # sensitivity of the fallback hosts' error to G over the range of the earlier profiles
    sens = []
    for h in fbh:
        for c in h["cells"].values():
            ct = c["counters"]["base"]
            for G in (FB[c["C"]][2], FB[c["C"]][3]):
                sens.append(100 * abs(j106.law(G, ct["misses"], ct["admits"], h["b_host"], S) / c["ms"]["base"] - 1))
    if sens:
        M["jmFallbackErrMin"] = f"{min(sens):.1f}"; M["jmFallbackErrMax"] = f"{max(sens):.1f}"
    M["jmVerdict"] = "held" if allok else "failed"

    def rng(k, vals, fmt="{:.2f}"):
        vals = [v for v in vals if v is not None]
        if vals:
            M[k + "Min"] = fmt.format(min(vals)).replace("-", "$-$"); M[k + "Max"] = fmt.format(max(vals)).replace("-", "$-$")
    if errs:
        a = np.abs(errs)
        M["jmCells"] = str(len(a)); M["jmWithinEight"] = str(int(np.sum(a <= BAND))); M["jmWithinSix"] = str(int(np.sum(a <= BAND6)))
        M["jmErrMed"] = f"{100 * np.median(a):.1f}"; M["jmErrAbsMax"] = f"{100 * a.max():.1f}"
        rng("jmErr", [100 * e for e in errs], "{:.1f}")
        M["jmPlainMed"] = f"{100 * np.median(np.abs(plain)):.1f}"; M["jmPlainWithinSix"] = str(int(np.sum(np.abs(plain) <= BAND6)))
        rng("jmEff", effs)
    rng("jmG", [c["G"] for h in valid.values() for c in h["cells"].values()], "{:.1f}")
    rng("jmRatio", [h["ratio"] for h in valid.values()]); rng("jmBhost", [h["b_host"] for h in valid.values()], "{:.0f}")
    rng("jmCores", [h["cores"] for h in valid.values()], "{:.0f}")
    rng("jmSpread", [100 * h["V2_spread"] for h in valid.values()], "{:.1f}")
    for j, h in hosts.items():   # per machine, by job letter: the relation's |error| at each budget, its implied rate
        for key, nmb in (("g14", "Low"), ("g32", "Mid")):
            c = h["cells"].get(key)
            if c and "err" in c:
                M[f"jmErr{j[-1]}{nmb}"] = f"{100 * abs(c['err']['base']['olap']):.1f}"; M[f"jmEff{j[-1]}{nmb}"] = f"{c['eff']:.2f}"
        M[f"jmCores{j[-1]}"] = str(h["cores"]); M[f"jmUsed{j[-1]}"] = str(h.get("used_gb", ""))
        inv = [h for h in hosts.values() if not h["valid"]]
    rng("jmInvSpread", [100 * h["V2_spread"] for h in inv if h["V2_spread"] is not None], "{:.1f}")
    rng("jmUsedGate", [g["used_gb"] for g in gated.values() if "memory in use" in g["reason"]], "{:.0f}")
    # after the fact (not registered): every launch of jobs 093-108 at gpt-oss 11% by CPU family, server parts
    # (EPYC, Xeon, engineering samples) against consumer parts; implied read rate over the probe's best rate
    E = machine_classes(r"(09[3-9]|10[0-8])")
    for e in E:
        e["family"] = "server" if re.search(r"EPYC|Xeon|Eng Sample", e["cpu"]) else "consumer"
    json.dump(E, open(P("prereg", "job108_families.json"), "w"), indent=1)
    for fam, nm in (("consumer", "Cons"), ("server", "Serv")):
        sel = [e for e in E if e["family"] == fam and e["valid"]]
        M[f"jm{nm}N"] = str(len(sel)); M[f"jm{nm}Machines"] = str(len({e["uuid"] for e in sel}))
        rng(f"jm{nm}Eff", [e["eff"] for e in sel]); rng(f"jm{nm}Err", [100 * e["err"] for e in sel], "{:.0f}")
        M[f"jm{nm}WithinEight"] = str(sum(abs(e["err"]) <= BAND for e in sel)); M[f"jm{nm}WithinSix"] = str(sum(abs(e["err"]) <= BAND6 for e in sel))
        M[f"jm{nm}Word"] = word(len(sel))
    sv = sorted((e for e in E if e["family"] == "server" and e["valid"]), key=lambda e: e["eff"])
    M["jmServList"] = ", ".join(f"{short_cpu(e['cpu'])} {e['eff']:.2f}" for e in sv)
    M["jmServAll"] = str(sum(e["family"] == "server" for e in E)); M["jmServInvalid"] = str(sum(e["family"] == "server" and not e["valid"] for e in E))
    # new consumer machines in registered tests (jobs 107-108): the relation's errors at gpt-oss 11% and 25%
    newc = [h for h in valid.values() if not re.search(r"EPYC|Xeon|engineering", h["name"])]
    j107 = json.load(open(P("prereg", "job107.json")))
    ne = [c["err"]["base"]["olap"] for h in newc for c in h["cells"].values()]
    ne += [h["cells"][k]["err"]["base"]["olap"] for h in j107["hosts"] if h["valid"] and not re.search(r"engineering", h["name"])
           for k in ("g14", "g32") if k in h["cells"]]
    if ne:
        rng("jmNewConsErr", [100 * e for e in ne], "{:.1f}"); M["jmNewConsCells"] = str(len(ne))
        M["jmNewConsWithinEight"] = str(sum(abs(e) <= BAND for e in ne)); M["jmNewConsWord"] = word(len(newc) + sum(1 for h in j107["hosts"] if h["valid"] and not re.search(r"engineering", h["name"]))); M["jmNewConsWithinSix"] = str(sum(abs(e) <= BAND6 for e in ne))
    with open(P("paper", "wsg_job108.tex"), "w") as f:
        f.write("% generated by scripts/job108.py from the job 108 hosts (results/108?_smallhosts@vast)\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    with open(P("paper", "tab_job108.tex"), "w") as f:
        f.write("% generated by scripts/job108.py\n\\begin{table}[t]\\centering\\footnotesize\n")
        f.write("\\caption{Job 108, registered before the runs and restricted in advance to desktop-class machines never rented "
                "before: the deployed cache's time per token (gpt-oss, mean of two rounds) against \\cref{eq:sum}, with $G$ "
                "profiled on the same machine before the timed runs, the run's own counters and the probe's best rate, and the "
                "read rate the time implies as a share of that rate. $^\\dagger$: rounds further apart than the registered "
                "2\\%; outside the predictions.}\\label{tab:job108}\n")
        f.write("\\setlength\\tabcolsep{3pt}\\begin{tabular}{@{}lrrlrrrr@{}}\\toprule\n")
        f.write("Machine & Cores & $B_{\\mathrm{host}}$ & Budget & $G$ (ms) & \\cref{eq:sum} & Measured & Implied/best \\\\\\midrule\n")
        for j, h in sorted(hosts.items(), key=lambda x: x[1]["b_host"]):
            for key in ("g14", "g32"):
                c = h["cells"].get(key)
                if not c:
                    continue
                f.write(f"{h['name']}{'' if h['valid'] else DAG} & {h['cores']} & {h['b_host']:.0f} & {LAB[c['C']]} & {c['G']:.1f} & "
                        f"{c['pred']['base']['olap']:.1f} & {c['ms']['base']:.1f} & {c['eff']:.2f} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")
    print("gated:", gated)
    print({j: (h["name"], h["V1"], h["V2_spread"]) for j, h in hosts.items()})
    print(dict(st))
    for c in clauses:
        print(c["status"], c["id"], c["short"], c["measured"], c["ci"])
    print(M)


if __name__ == "__main__":
    main()

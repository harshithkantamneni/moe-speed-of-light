"""Robustness of the paper's machine-level claims, for the review round-14 complaints:
(1) the probe's best rate B_host is the single highest reading; how the bound's share and eq. (sum)'s error move with
    more robust rates (the second-highest reading; the best CPU-only reading), and how often the engine reads faster
    than each;
(2) machine-level intervals: bootstrap over machines (one value per machine, the mean over its launches) for the
    fewest-admission schedule's gain, the admission margin's gain and eq. (sum)'s error;
(3) how many of the consumer launches have their own profiled G, and the sign of eq. (sum)'s error on machines new to
    a registered test.
Writes paper/wsg_robust.tex and paper/tab_robust.tex.

    python scripts/robustness.py
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from scripts.job106 import law, host_info  # noqa: E402

RES = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
S = 13253760
RNG = np.random.default_rng(14)
M = {}


def rng(k, vals, fmt="{:.2f}"):
    vals = [v for v in vals if v is not None]
    if vals:
        M[k + "Min"] = fmt.format(min(vals)).replace("-", "$-$"); M[k + "Max"] = fmt.format(max(vals)).replace("-", "$-$")


def readings(txt):
    cpu = [float(x) for x in re.findall(r"^cpu_read_gbs t=\d+ ([\d.]+)", txt, re.M)]
    pcie = [float(x) for x in re.findall(r"^pcie_\w+_gbs [^\n]*? ([\d.]+)$", txt, re.M)]
    sums = [float(x) for x in re.findall(r"sum ([\d.]+)", txt)]
    return cpu, pcie, sums


def boot_machines(vals, stat=np.mean):
    """the summary over machines and its 95% t-interval (on the log scale for a geometric mean); the name is kept from
    the percentile bootstrap it replaced, which is too narrow with a handful of machines"""
    lo, hi = t_interval(vals, geo=stat is GMEAN)
    return float(stat(np.array(vals, float))), lo, hi


def t_interval(vals, geo=False):
    """the 95% t-interval of the mean over machines (the percentile bootstrap is narrow at a handful of machines); for
    ratios, on the log scale (the geometric mean's interval)"""
    v = np.log(np.array(vals, float)) if geo else np.array(vals, float)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    lo, hi = float(v.mean() - h), float(v.mean() + h)
    return (float(np.exp(lo)), float(np.exp(hi))) if geo else (lo, hi)


GMEAN = stats.gmean   # ratios are summarised by geometric means (the checklist's rule on summarising ratios)


def launch_rows():
    """every launch of jobs 093-108 that ran the deployed cache at gpt-oss 11% and 25% (first round), with its counters"""
    ra = json.load(open(P("prereg", "reanalysis.json")))
    fam = {e["dir"]: e for e in json.load(open(P("prereg", "job108_families.json")))}
    rows = []
    for d in sorted(glob.glob(f"{RES}/*@vast")):
        j = os.path.basename(d)
        if j not in fam or not os.path.exists(f"{d}/concur.txt"):
            continue
        cpu, pcie, sums = readings(open(f"{d}/concur.txt").read())
        allr = sorted(cpu + pcie + sums, reverse=True)
        for C in (14, 32):
            st = [f for f in (f"{d}/st_g_C{C}_base.json", f"{d}/st_g_C{C}_r1_base.json") if os.path.exists(f)]
            if not st:
                continue
            ec = st[0].replace("/st_g_C", "/ec_g_C").replace("_base.json", ".jsonl")
            if not os.path.exists(ec):
                continue
            s = json.load(open(st[0])); n = max(1, s["steps"]); Mi, A = s["misses"] / n, s["admits"] / n
            T = float(np.mean([r["decode_ms"] / r["n_decode"] for r in map(json.loads, filter(str.strip, open(ec)))
                               if r["config"].rsplit("stats=", 1)[-1].endswith("_base.json")]))
            G0 = ra["sum_law"][str(C)]["G"]; G, prof = G0, False
            if os.path.exists(f"{d}/g_prof.json"):
                gp = json.load(open(f"{d}/g_prof.json")); v = (gp.get(f"G{C}") or gp.get(f"C{C}") or {}).get("G_prof_ms")
                if v and v > 1.0:
                    G, prof = v, True
            rows.append(dict(dir=j, C=C, T=T, M=Mi, A=A, G=G, prof=prof, fam=fam[j]["family"], valid=fam[j]["valid"],
                             uuid=fam[j]["uuid"], Bmax=allr[0], B2=allr[1], Bcpu=max(cpu)))
    return rows


def main():
    R = launch_rows()
    good = [r for r in R if r["valid"]]
    # (1) the probe's rate
    for key, nm in (("Bmax", "Max"), ("B2", "Second"), ("Bcpu", "Cpu")):
        for r in good:
            B = r[key]
            r[f"err_{nm}"] = law(r["G"], r["M"], r["A"], B, S) / r["T"] - 1
            r[f"eff_{nm}"] = (r["M"] + r["A"] * (1 - r["G"] / r["T"])) * S / ((r["T"] - r["G"]) * 1e-3) / 1e9 / B
    cons = [r for r in good if r["fam"] == "consumer"]
    gap = [1 - r["B2"] / r["Bmax"] for r in {r["dir"]: r for r in good}.values()]
    M["rbLaunchBudgets"] = str(len(good))
    M["rbSecondGapMed"] = f"{100 * np.median(gap):.1f}"; M["rbSecondGapMax"] = f"{100 * max(gap):.0f}"
    M["rbSecondGapOverFive"] = str(sum(g > 0.05 for g in gap)); M["rbLaunches"] = str(len(gap))
    byl = {r["dir"]: r for r in good}
    M["rbSecondGapLaunch"] = max(byl, key=lambda d: byl[d]["Bmax"] / byl[d]["B2"])[:4]
    # the server machine new to job 107: its error with the probe's second-highest reading
    sv = [abs(r["err_Second"]) for r in good if r["fam"] == "server" and r["dir"][:3] == "107"]
    sm = [abs(r["err_Max"]) for r in good if r["fam"] == "server" and r["dir"][:3] == "107"]
    M["rbServSecondMin"] = f"{100 * min(sv):.0f}"; M["rbServSecondMax"] = f"{100 * max(sv):.0f}"
    M["rbServMaxMax"] = f"{100 * max(sm):.0f}"
    M["rbLaunchMachines"] = str(len({r["uuid"] for r in good}))
    for nm in ("Max", "Second", "Cpu"):
        M[f"rbAbove{nm}"] = str(sum(r[f"eff_{nm}"] > 1.05 for r in good))
        M[f"rbAbove{nm}Max"] = f"{max(r[f'eff_{nm}'] for r in good):.2f}"
        e = [r[f"err_{nm}"] for r in cons if r["C"] == 14]
        M[f"rbWithinSix{nm}"] = str(sum(abs(x) <= 0.06 for x in e)); M[f"rbWithinEight{nm}"] = str(sum(abs(x) <= 0.08 for x in e))
        M[f"rbErrMed{nm}"] = f"{100 * np.median(np.abs(e)):.1f}"
    M["rbConsLow"] = str(sum(r["C"] == 14 for r in cons))
    # the bound's share of the time with each rate (at gpt-oss 11%: the bound is MIN's read time R* S / B)
    # R* per launch is the cell's MIN reads; use reanalysis limit share where present: share = R* S / B / T
    lim = {}
    for f in (P("prereg", "reanalysis_hosts.json"),):
        for h in json.load(open(f))["launches"]:
            c = h.get("cells", {}).get("14")
            if c and c.get("limit_ms"):
                lim[h["dir"]] = c["limit_ms"]
    # (2) machine-level intervals
    # fewest-admission schedule copied in the step, gpt-oss 11%, per machine (mean over its launches)
    fp = {}
    for pj in ("job104.json", "job105.json"):
        for h in json.load(open(P("prereg", pj)))["hosts"]:
            c = h["cells"].get("14")
            if c and h.get("planned", {}).get("14") and c["speed"].get("fetchplan"):
                fp.setdefault(uuid_of(h["dir"]), []).append((h["dir"], c["speed"]["fetchplan"][0]))
    for pj, keep in (("job106.json", ("106a", "106b", "106e")), ("job107.json", None)):
        for h in json.load(open(P("prereg", pj)))["hosts"]:
            if (keep and h["dir"][:4] not in keep) or (keep is None and not h.get("valid")):
                continue
            c = h["cells"].get("g14")
            if c and c["speed"].get("fetchplan"):
                fp.setdefault(uuid_of(h["dir"]), []).append((h["dir"], c["speed"]["fetchplan"][0]))
    # job 112's valid machines (two RTX 4090s and two relaunched RTX 5090s of job 109) ran the same schedule in the step
    pj = P("prereg", "job112.json")
    if os.path.exists(pj):
        for h in json.load(open(pj))["hosts"]:
            c = h["cells"].get("14", {})
            if h.get("valid") and c.get("ratio", {}).get("fetchplan"):
                fp.setdefault(uuid_of(h["dir"]), []).append((h["dir"], c["ratio"]["fetchplan"]))
    per = [float(GMEAN([x for _, x in v])) for v in fp.values()]
    # what the machines are: RTX 4090s (from nvidia-smi) and server processors (the family table, else the CPU's name)
    def card_of(d):
        f = f"{RES}/{d}/nvidia-smi-q.txt"
        return next((ln.split(":", 1)[1].strip() for ln in open(f) if "Product Name" in ln), "") if os.path.exists(f) else ""

    def server_of(d):
        e = fam.get(d)
        if e:
            return e["family"] == "server"
        f = f"{RES}/{d}/lscpu.txt"
        txt = open(f).read() if os.path.exists(f) else ""
        return "EPYC" in txt or "Xeon" in txt
    fam = {e["dir"]: e for e in json.load(open(P("prereg", "job108_families.json")))}
    word = lambda n: ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"][n] if n < 10 else str(n)  # noqa: E731
    M["rbPlanCardTwo"] = word(sum(any("4090" in card_of(d) for d, _ in v) for v in fp.values()))
    M["rbPlanServer"] = word(sum(any(server_of(d) for d, _ in v) for v in fp.values()))
    pt, lo, hi = boot_machines(per, stat=GMEAN)
    M["rbPlanMachines"] = str(len(per)); M["rbPlanMean"] = f"{pt:.2f}"; M["rbPlanLo"] = f"{lo:.2f}"; M["rbPlanHi"] = f"{hi:.2f}"
    M["rbPlanMin"] = f"{min(per):.2f}"
    tl, th = t_interval(per, geo=True); M["rbPlanTLo"] = f"{tl:.2f}"; M["rbPlanTHi"] = f"{th:.2f}"
    plan_stable = list(per)
    # "beats the deployed cache" was a registered clause on job 104's slow-link host (104a) and in jobs 105-107; on
    # 104b and 104c job 104 registered only the comparison with the greedy set
    direct = sum(any(j[:4] == "104a" or j[:3] in ("105", "106", "107") for j, _ in v) for v in fp.values())
    # the link-to-CPU ratio of each machine (its probe's best link rate over its best CPU rate)
    ratio = lambda d: (lambda hi: hi["B_p"] / hi["B_c"])(host_info(f"{RES}/{d}"))  # noqa: E731
    rat = [min(ratio(j) for j, _ in v) for v in fp.values()]
    M["rbPlanRatioMin"] = f"{min(rat):.2f}"
    M["rbPlanRegMachines"] = str(direct); M["rbPlanImplied"] = (["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"] + [str(i) for i in range(10, 40)])[len(per) - direct]
    # the machines that ran unsteadily (or computed wrong outputs) and ran the same schedule
    un = []
    for pj, keep in (("job106.json", ("106c", "106d")), ("job107.json", None)):
        for h in json.load(open(P("prereg", pj)))["hosts"]:
            if (keep and h["dir"][:4] not in keep) or (keep is None and h.get("valid")):
                continue
            c = h["cells"].get("g14")
            if c and c["speed"].get("fetchplan"):
                un.append((c["speed"]["fetchplan"][0], h["ratio"]))
    num = ["no", "one", "two", "three", "four", "five", "six"]
    M["rbPlanUnstN"] = num[len(un)]; M["rbPlanUnstLost"] = num[sum(v < 1 for v, _ in un)]
    lost = sorted(r for v, r in un if v < 1)
    # every machine that ran it, stable or not (intention to treat)
    pt, lo, hi = boot_machines(plan_stable + [v for v, _ in un], stat=GMEAN)
    M["rbPlanAllN"] = str(len(plan_stable) + len(un)); M["rbPlanAllMean"] = f"{pt:.2f}"
    M["rbPlanAllLo"] = f"{lo:.2f}"; M["rbPlanAllHi"] = f"{hi:.2f}"
    M["rbPlanLostRatios"] = " and ".join(f"{r:.2f}" for r in lost)
    M["rbPlanGainedSlow"] = f"{min(r for v, r in un if v >= 1):.2f}"
    # the admission margin at gpt-oss, per machine (mean over its gpt-oss cells)
    dk, dkc = {}, {"g14": {}, "g32": {}}
    for pj, keep in (("job106.json", ("106a", "106b", "106e")), ("job107.json", None)):
        for h in json.load(open(P("prereg", pj)))["hosts"]:
            if (keep and h["dir"][:4] not in keep) or (keep is None and not h.get("valid")):
                continue
            for k in ("g14", "g32"):
                if h["cells"].get(k, {}).get("speed", {}).get("dk"):
                    x = h["cells"][k]["speed"]["dk"][0]
                    dk.setdefault(uuid_of(h["dir"]), []).append(x); dkc[k].setdefault(uuid_of(h["dir"]), []).append(x)
    for k, nm in (("g14", "Low"), ("g32", "Mid")):
        pc = [float(GMEAN(v)) for v in dkc[k].values()]
        a, b, c = boot_machines(pc, stat=GMEAN)
        M[f"rbDk{nm}"] = f"{a:.3f}"; M[f"rbDk{nm}Lo"] = f"{b:.3f}"; M[f"rbDk{nm}Hi"] = f"{c:.3f}"
    per = [float(GMEAN(v)) for v in dk.values()]
    pt, lo, hi = boot_machines(per, stat=GMEAN)
    M["rbDkMachines"] = str(len(per)); M["rbDkMean"] = f"{pt:.3f}"; M["rbDkLo"] = f"{lo:.3f}"; M["rbDkHi"] = f"{hi:.3f}"
    tl, th = t_interval(per, geo=True); M["rbDkTLo"] = f"{tl:.3f}"; M["rbDkTHi"] = f"{th:.3f}"
    # eq. (sum)'s error per consumer machine at gpt-oss 11% (mean over its launches)
    em = {}
    for r in cons:
        if r["C"] == 14:
            em.setdefault(r["uuid"], []).append(r["err_Max"])
    per = [float(np.mean(v)) for v in em.values()]
    pt, lo, hi = boot_machines(per)
    M["rbErrMachines"] = str(len(per)); M["rbErrMean"] = f"{100 * pt:.1f}".replace("-", "$-$")
    M["rbErrLo"] = f"{100 * lo:.1f}".replace("-", "$-$"); M["rbErrHi"] = f"{100 * hi:.1f}".replace("-", "$-$")
    # the same over the machines it was found on (jobs 093-104), their found-on launches only
    FOUND = ("093", "094", "095", "096", "097", "099", "100", "101", "102", "103", "104")
    ef = {}
    for r in cons:
        if r["C"] == 14 and r["dir"][:3] in FOUND:
            ef.setdefault(r["uuid"], []).append(r["err_Max"])
    per = [float(np.mean(v)) for v in ef.values()]
    pt, lo, hi = boot_machines(per)
    M["rbErrFoundMachines"] = str(len(per)); M["rbErrFoundMean"] = f"{100 * pt:.1f}".replace("-", "$-$")
    M["rbErrFoundLo"] = f"{100 * lo:.1f}".replace("-", "$-$"); M["rbErrFoundHi"] = f"{100 * hi:.1f}".replace("-", "$-$")
    # the same at gpt-oss 25%
    em = {}
    for r in cons:
        if r["C"] == 32:
            em.setdefault(r["uuid"], []).append(r["err_Max"])
    per = [float(np.mean(v)) for v in em.values()]
    pt, lo, hi = boot_machines(per)
    M["rbErrMidMachines"] = str(len(per)); M["rbErrMidMean"] = f"{100 * pt:.1f}".replace("-", "$-$")
    M["rbErrMidLo"] = f"{100 * lo:.1f}".replace("-", "$-$"); M["rbErrMidHi"] = f"{100 * hi:.1f}".replace("-", "$-$")
    # (3) profiled G among the consumer launches; sign of the error on machines new to a registered test
    cl = [r for r in cons if r["C"] == 14]
    M["rbConsProf"] = str(sum(r["prof"] for r in cl)); M["rbConsAll"] = str(len(cl))
    prof_err = [abs(r["err_Max"]) for r in cl if r["prof"]]; med_err = [abs(r["err_Max"]) for r in cl if not r["prof"]]
    M["rbProfErrMed"] = f"{100 * np.median(prof_err):.1f}"; M["rbMedGErrMed"] = f"{100 * np.median(med_err):.1f}"
    found = [r["err_Max"] for r in cl if r["dir"][:3] in ("093", "094", "095", "096", "097", "099", "100", "101", "102", "103", "104")]
    M["rbFoundSignedMed"] = f"{100 * np.median(found):.1f}".replace("-", "$-$")
    M["rbFoundUnder"] = str(sum(e < 0 for e in found)); M["rbFoundN"] = str(len(found))
    fm = [r["err_Max"] for r in cons if r["C"] == 32 and r["dir"][:3] in ("093", "094", "095", "096", "097", "099", "100", "101", "102", "103", "104")]
    M["rbFoundUnderMid"] = str(sum(e < 0 for e in fm)); M["rbFoundNMid"] = str(len(fm))
    M["rbFoundOverMid"] = str(sum(e > 0 for e in fm))
    with open(P("paper", "wsg_robust.tex"), "w") as f:
        f.write("% generated by scripts/robustness.py\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    with open(P("paper", "tab_robust.tex"), "w") as f:
        f.write("% generated by scripts/robustness.py\n\\begin{table}[t]\\centering\\footnotesize\n")
        f.write("\\caption{How the probe's rate changes the picture, over the " + M["rbLaunchBudgets"] + " launch-budgets at gpt-oss "
                "11\\% and 25\\% of the machines that passed our checks: \\cref{eq:sum} with the highest probe reading (as in the "
                "paper), the second-highest, and the best CPU-only reading. \\emph{Faster than the rate}: launch-budgets whose "
                "implied read rate exceeds the rate by more than 5\\%. The error columns are the consumer launches at 11\\%.}"
                "\\label{tab:robust}\n\\setlength\\tabcolsep{3pt}\\begin{tabular}{@{}lrrrr@{}}\\toprule\n")
        f.write("Rate & Faster than the rate (max) & Within 6\\% & Within 8\\% & Median $|$error$|$ \\\\\\midrule\n")
        for nm, lab in (("Max", "highest reading"), ("Second", "second-highest"), ("Cpu", "best CPU-only")):
            f.write(f"{lab} & {M['rbAbove' + nm]} ({M['rbAbove' + nm + 'Max']}$\\times$) & {M['rbWithinSix' + nm]}/{M['rbConsLow']} & "
                    f"{M['rbWithinEight' + nm]}/{M['rbConsLow']} & {M['rbErrMed' + nm]}\\% \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")
    print(M)


def uuid_of(dirname):
    f = f"{RES}/{dirname}/nvidia-smi-q.txt"
    if os.path.exists(f):
        for ln in open(f):
            if "GPU UUID" in ln:
                return ln.split(":", 1)[1].strip()
    return dirname


if __name__ == "__main__":
    main()

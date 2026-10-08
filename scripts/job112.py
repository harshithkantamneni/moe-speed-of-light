"""Job 112: the last measurements (jobs/112_closing@vast.sh on the gpu branch). Scores its registered predictions and
writes prereg/job112.json, prereg/scorecard_112.json, paper/wsg_job112.tex and paper/tab_job112.tex.

  (a) RTX 4090 machines at higher link-to-CPU ratios: job 110's predictions Q1-Q6 (the frozen RTX 5090 trend);
  (b) the 2x2's timing: Few-2R-early (bypassplanS) against Few-2R and Few-1R (T1-T4);
  (c) a second probe after the timed runs (T5);
  (d) relaunches of job 109's RTX 5090 machines (T6, T7).

    python scripts/job112.py
"""
import glob
import json
import math
import os
import re
import sys
from collections import Counter

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from scripts.job109 import (load, predictions110, clauses110, status, tint, trend_dev, trend_pred, RSTAR,  # noqa: E402
                            word, RES)
from scripts.speed_limit import host_rates  # noqa: E402

GLOB = os.environ.get("MOSL_JOB112_GLOB", f"{RES}/112[a-g]_closing@vast")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
CELLS = (14, 32)
G = {14: "g11", 32: "g25"}
NM = {14: "Low", 32: "Mid"}


def card_of(d):
    m = re.search(r"CARD=(\d+)", open(f"{d}/mode.txt").read()) if os.path.exists(f"{d}/mode.txt") else None
    return m.group(1) if m else None


def load112(d):
    card = card_of(d)
    h = load(d, CELLS, 0.45 if card == "4090" else 0.5, need_b=(card == "4090"), keep_pp=True)
    h["card"] = card
    t2 = f"{d}/concur2.txt"
    txt = open(t2).read() if os.path.exists(t2) else ""
    h["B_host2"] = host_rates(txt) / 1e9 if "concurrent t=" in txt or "cpu_read_gbs" in txt else None
    ft2 = f"{d}/fetch_table2_law_gptoss.json"
    if os.path.exists(ft2):
        j = json.load(open(ft2))
        h["ratio2"] = j["B_p"] / j["B_c"]
    for C, c in h["cells"].items():
        dr = c.get("_draws", {})
        # the timing cell: the late landing's part and the second read's part, with paired intervals over problems
        if all(k in dr for k in ("fetchplan", "bypassplan", "bypassplanS")):
            lag = dr["bypassplanS"] - dr["bypassplan"]
            read = dr["fetchplan"] - dr["bypassplanS"]
            c["lag_part"] = c["ratio"]["bypassplanS"] - c["ratio"]["bypassplan"]
            c["read_part"] = c["ratio"]["fetchplan"] - c["ratio"]["bypassplanS"]
            c["ci"]["lag_part"] = tuple(float(x) for x in np.percentile(lag, [2.5, 97.5]))
            c["ci"]["read_part"] = tuple(float(x) for x in np.percentile(read, [2.5, 97.5]))
            c["ci"]["read_minus_lag"] = tuple(float(x) for x in np.percentile(read - lag, [2.5, 97.5]))
            c["ci"]["early_minus_two"] = tuple(float(x) for x in np.percentile(lag, [2.5, 97.5]))
        c["misses"] = {n: float(np.mean([x["misses"] for x in v])) for n, v in c["counters"].items()}
        c["admits"] = {n: float(np.mean([x["admits"] for x in v])) for n, v in c["counters"].items()}
        c["fetches"] = {n: float(np.mean([x.get("fetches", 0) for x in v])) for n, v in c["counters"].items()}
        c.pop("_pp", None); c.pop("_draws", None)
    return h


def base109(uuid):
    """job 109's process-A base time per token at C = 14 (mean over rounds) on the machine with this GPU"""
    p = P("prereg", "job109.json")
    if not os.path.exists(p):
        return None
    for h in json.load(open(p))["hosts"]:
        if h.get("uuid") == uuid and h["cells"].get("14", {}).get("t"):
            return h["cells"]["14"]["t"]["base"]
    return None


def clauses112(V, V4, V5):
    out = []

    def add(cid, short, meas, ok, thr, host, ci=None):
        out.append(dict(id=cid, short=short, type="band", measured=None if meas is None else round(float(meas), 4),
                        ci=None if ci is None else [round(float(x), 4) for x in ci], threshold=thr, status=status(meas, ok, ci),
                        why="" if ci is not None else "no interval", host=host))
    for h in V:
        nm = f"{h['job']} {h['cpu']}"; c = h["cells"]
        for C in CELLS:
            cc = c[C]
            if not cc["rounds"] or "bypassplanS" not in cc["ratio"]:
                continue
            m = cc["misses"]
            add(f"{h['job']}-T1-{G[C]}", f"Few-2R-early misses <= 0.95x Few-2R's ({nm}, {G[C]})",
                m.get("bypassplanS", float("nan")) / m.get("bypassplan", float("nan")), lambda v: v <= 0.95, "<= 0.95", nm)
            add(f"{h['job']}-T2-{G[C]}", f"Few-2R-early/base >= Few-2R/base - 0.01 ({nm}, {G[C]})", cc["lag_part"],
                lambda v: v >= -0.01, ">= -0.01", nm, cc["ci"].get("lag_part"))
        lo = c[14]
        if h["ratio"] >= 0.5 and "read_part" in lo:
            add(f"{h['job']}-T3", f"second read costs more than the late landing at 11% ({nm})", lo["read_part"] - lo["lag_part"],
                lambda v: v > 0, "> 0", nm, lo["ci"].get("read_minus_lag"))
            add(f"{h['job']}-T4", f"Few-2R-early/base <= 1.10 at 11% ({nm})", lo["ratio"]["bypassplanS"], lambda v: v <= 1.10, "<= 1.10", nm,
                lo["ci"].get("bypassplanS"))
        if h.get("B_host") and h.get("B_host2"):
            add(f"{h['job']}-T5", f"second probe within 10% of the first ({nm})", h["B_host2"] / h["B_host"] - 1,
                lambda v: abs(v) <= 0.10, "|x| <= 0.10", nm)
    for h in V5:
        nm = f"{h['job']} {h['cpu']}"; c = h["cells"]
        b = base109(h["uuid"])
        if b:
            add(f"{h['job']}-T6", f"base within 5% of job 109's on this machine ({nm})", c[14]["t"]["base"] / b - 1,
                lambda v: abs(v) <= 0.05, "|x| <= 0.05", nm)
        r = c[14]["ratio"]
        if all(k in r for k in ("fetch", "bypass", "foa")):
            add(f"{h['job']}-T7", f"MIN-1R beats both single changes by 0.04 at 11% ({nm})", r["fetch"] - max(r["bypass"], r["foa"]),
                lambda v: v >= 0.04, ">= 0.04", nm)
    add("112-valid", "at least three valid hosts over the two cards for T1-T5", len(V), lambda x: x >= 3, ">= 3", "pooled")
    return out


def main():
    H = [load112(d) for d in sorted(glob.glob(GLOB))]
    V = [h for h in H if h["valid"]]
    V4 = [h for h in V if h["card"] == "4090"]
    V5 = [h for h in V if h["card"] == "5090"]
    for h in H:
        print(f"{h['job']} {h['card']} {h['cpu'][:22]:22s} ratio {h.get('ratio', float('nan')):.2f} ratio2 {h.get('ratio2', float('nan')):.2f} "
              f"B_host {h.get('B_host') or 0:.1f} B_host2 {h.get('B_host2') or 0:.1f} valid {h['valid']} {h['gate']} {h['V2_line']}")
        for C, c in h["cells"].items():
            if c.get("rounds"):
                print(f"   C{C} rounds {c['rounds']} " + " ".join(f"{k} {v:.3f}" for k, v in sorted(c["ratio"].items())))
                print("      misses " + " ".join(f"{k} {v:.1f}" for k, v in sorted(c["misses"].items())))
    pr4 = predictions110(V4, CELLS) if V4 else {}
    # job 110's clauses on this job's RTX 4090s; its "at least three valid hosts" clause is not part of job 112's
    # registration (Q1-Q6 are tested per host here), so it is left out
    cl4 = [c for c in (clauses110(V4, CELLS) if V4 else []) if not c["id"].endswith("-valid")]
    cl = clauses112(V, V4, V5)
    for c in cl4 + cl:
        print(c["id"], c["measured"], c["ci"], c["status"])
    json.dump(dict(hosts=H, predictions110={k: dict(passed=ok, detail=det) for k, (ok, det) in pr4.items()},
                   valid=[h["job"] for h in V]), open(P("prereg", "job112.json"), "w"), indent=1, default=float)
    json.dump(dict(job="112", script="jobs/112_closing@vast.sh", commit="92849de", scored_by="scripts/job112.py (machine)",
                   clauses=[dict(c, id=c["id"].replace("110-", "112-")) for c in cl4] + cl),
              open(P("prereg", "scorecard_112.json"), "w"), indent=1)
    M = macros(H, V, V4, V5, pr4, cl4, cl)
    with open(P("paper", "wsg_job112.tex"), "w") as f:
        f.write("% generated by scripts/job112.py from job 112\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    table(V)
    from scripts.tabnote import split_caption
    split_caption(P("paper", "tab_job112.tex"))   # short caption, the rest as a note below the table
    for k in sorted(M):
        print(k, M[k])


def macros(H, V, V4, V5, pr4, cl4, cl):
    M = {}
    pc = lambda x: str(int(round(100 * x))).replace("-", "$-$")  # noqa: E731

    def rng(k, vals, fmt="{:.2f}"):
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if vals:
            M[k + "Min"] = fmt.format(min(vals)).replace("-", "$-$"); M[k + "Max"] = fmt.format(max(vals)).replace("-", "$-$")
            M[k + "Rng"] = M[k + "Min"] if M[k + "Min"] == M[k + "Max"] else M[k + "Min"] + "--" + M[k + "Max"]
    M["tcStarted"] = str(len(H)); M["tcStartedWord"] = word(len(H)); M["tcN"] = str(len(V)); M["tcNWord"] = word(len(V))
    M["tcFourN"] = str(len(V4)); M["tcFourNWord"] = word(len(V4)); M["tcFiveN"] = str(len(V5)); M["tcFiveNWord"] = word(len(V5))
    M["tcGated"] = word(sum(1 for h in H if h["gate"])); M["tcInvalid"] = word(sum(1 for h in H if not h["gate"] and not h["valid"]))
    rng("tcRatioFour", [h["ratio"] for h in V4]); rng("tcRatioFive", [h["ratio"] for h in V5]); rng("tcRatio", [h["ratio"] for h in V])
    for C in CELLS:
        nm = NM[C]
        sel = [h["cells"][C] for h in V if h["cells"][C]["rounds"] and "bypassplanS" in h["cells"][C]["ratio"]]
        if not sel:
            continue
        M[f"tcCells{nm}"] = str(len(sel))
        for n, k in (("bypassplanS", "Early"), ("bypassplan", "Two"), ("fetchplan", "One")):
            rng(f"tc{k}{nm}", [c["ratio"][n] for c in sel])
            rng(f"tcMiss{k}{nm}", [c["misses"][n] for c in sel], "{:.1f}")
        rng(f"tcMissRatio{nm}", [c["misses"]["bypassplanS"] / c["misses"]["bypassplan"] for c in sel])
        rng(f"tcLag{nm}", [c["lag_part"] for c in sel]); rng(f"tcRead{nm}", [c["read_part"] for c in sel])
        rng(f"tcSlow{nm}", [-c["lag_part"] for c in sel])
        rng(f"tcEarlyMinusFetchMiss{nm}", [c["misses"]["bypassplanS"] - c["misses"]["fetchplan"] for c in sel], "{:.1f}")
        rng(f"tcFetchesEarly{nm}", [c["fetches"]["bypassplanS"] for c in sel], "{:.1f}")
        # admissions per token of the two two-read arms, and the early arm's host reads (misses + admissions) over Few-2R's
        rng(f"tcAdmEarly{nm}", [c["admits"]["bypassplanS"] for c in sel], "{:.1f}")
        rng(f"tcAdmTwo{nm}", [c["admits"]["bypassplan"] for c in sel], "{:.1f}")
        hr = lambda c, n: c["misses"][n] + c["admits"][n]  # noqa: E731
        rng(f"tcHostEarlyPct{nm}", [100 * (hr(c, "bypassplanS") / hr(c, "bypassplan") - 1) for c in sel], "{:.0f}")
    # the second probe
    dv = [h["B_host2"] / h["B_host"] - 1 for h in V if h.get("B_host") and h.get("B_host2")]
    if dv:
        M["tcProbeN"] = str(len(dv)); M["tcProbeDevMax"] = f"{100 * max(abs(x) for x in dv):.1f}"
        M["tcProbeDevMean"] = f"{100 * float(np.mean(dv)):+.1f}".replace("-", "$-$")
    rd = [h["ratio2"] / h["ratio"] - 1 for h in V if h.get("ratio2") and h.get("ratio")]
    if rd:
        M["tcRatioDevMax"] = f"{100 * max(abs(x) for x in rd):.1f}"
    # the relaunches of job 109's machines
    j109 = {x["uuid"]: x for x in json.load(open(P("prereg", "job109.json")))["hosts"]} if os.path.exists(P("prereg", "job109.json")) else {}
    rr = [h["ratio"] / j109[h["uuid"]]["ratio"] - 1 for h in V5 if h["uuid"] in j109]
    if rr:
        M["tcRelaunchRatioDevMax"] = f"{100 * max(abs(x) for x in rr):.0f}"
    rel = [h["cells"][14]["t"]["base"] / base109(h["uuid"]) - 1 for h in V5 if base109(h["uuid"])]
    if rel:
        M["tcRelaunchMax"] = f"{100 * max(abs(x) for x in rel):.1f}"
    # the RTX 4090s at higher ratios: the frozen trend
    for C in CELLS:
        nm = NM[C]
        sel = [h for h in V4 if h["cells"][C]["rounds"]]
        d = [abs(math.exp(trend_dev(h, C, n)) - 1) for h in sel for n in ("fetch", "both3p")]
        if d:
            M[f"tcFourDev{nm}Max"] = pc(max(d))
        d2 = [abs(math.exp(trend_dev(h, C, n)) - 1) for h in sel for n in ("foa", "bypass")]
        if d2:
            M[f"tcFourDevSmall{nm}Max"] = pc(max(d2))
        rng(f"tcFourFetch{nm}", [h["cells"][C]["ratio"]["fetch"] for h in sel])
        rng(f"tcFourOracle{nm}", [h["cells"][C]["ratio"]["both3p"] for h in sel])
        rng(f"tcFourBest{nm}", [h["cells"][C]["best"] for h in sel if "best" in h["cells"][C]], "{:.3f}")
        cp = [h["cells"][C]["capture"] for h in sel if "capture" in h["cells"][C]]
        if cp:
            M[f"tcFourCap{nm}Max"] = pc(max(cp))
    fast4 = [h for h in V4 if h["ratio"] >= 0.5]
    M["tcFourFastN"] = word(len(fast4))
    M["tcFourInteractionPos"] = word(sum(1 for h in fast4 if h["cells"][14].get("interaction_ms", -1) > 0))
    q_held = [k for k, (ok, _) in pr4.items() if k.startswith("Q") and ok]
    q_failed = [k for k, (ok, _) in pr4.items() if k.startswith("Q") and ok is False]
    M["tcQN"] = str(sum(1 for k in pr4 if k.startswith("Q") and pr4[k][0] is not None)); M["tcQHeld"] = str(len(q_held))
    M["tcQFailed"] = str(len(q_failed)); M["tcQFailedList"] = ", ".join(q_failed) if q_failed else "none"
    st = Counter(c["status"] for c in cl4 + cl)
    M["tcClauses"] = str(len(cl4) + len(cl)); M["tcClausesHeld"] = str(st.get("held", 0)); M["tcClausesPoint"] = str(st.get("held (point)", 0))
    M["tcClausesFailed"] = str(st.get("failed", 0)); M["tcClausesUntested"] = str(st.get("untested", 0))
    failed = sorted({c["id"].split("-")[1] for c in cl4 + cl if c["status"] == "failed"})
    M["tcFailedList"] = ", ".join(failed) if failed else "none"
    for t, tw in (("T1", "TOne"), ("T2", "TTwo"), ("T3", "TThree"), ("T4", "TFour"), ("T5", "TFive"), ("T6", "TSix"), ("T7", "TSeven")):
        cc = [c for c in cl if c["id"].split("-")[1] == t]
        M[f"tc{tw}Status"] = ("untested" if not cc else "failed" if any(c["status"] == "failed" for c in cc)
                              else "held" if all(c["status"] == "held" for c in cc) else "held (point)")
        M[f"tc{tw}N"] = str(len(cc)); M[f"tc{tw}Failed"] = str(sum(c["status"] == "failed" for c in cc))
    return M


def table(V):
    """the timing cell, one row per valid machine: speed vs the deployed cache and misses per token at 11%"""
    hw = [(c["ci"][n][1] - c["ci"][n][0]) / 2 for h in V for c in (h["cells"][14],) if "bypassplanS" in c["ratio"]
          for n in ("fetchplan", "bypassplan", "bypassplanS") if c["ci"].get(n)]
    hwmax = f"{max(hw):.2f}" if hw else "--"
    with open(P("paper", "tab_job112.tex"), "w") as f:
        f.write("% generated by scripts/job112.py\n\\begin{table}[t]\\centering\\footnotesize\n")
        f.write("\\caption{When the copy lands, against how often the expert is read: MIN's fewest-admission set read once "
                "(\\FewOne), read twice with the copy published two steps after the admission (\\FewTwo), and read twice with "
                "each copy issued a step earlier so that it lands for the next use (\\FewEarly). Speed relative to the deployed "
                "cache on the same machine (every 95\\% interval over problems within $\\pm$" + hwmax + ") and host misses per token, at "
                "gpt-oss 11\\%; registered before the machines were rented.}\\label{tab:timing}\n")
        f.write("\\setlength\\tabcolsep{2.5pt}\\begin{tabular}{@{}lrrrrrrr@{}}\\toprule\n")
        f.write(" & & \\multicolumn{3}{c}{Speed vs deployed} & \\multicolumn{3}{c}{Misses per token} \\\\\\cmidrule(lr){3-5}\\cmidrule(l){6-8}\n")
        f.write("Machine & Ratio & \\cfg{1R} & \\cfg{2R} & \\cfg{early} & \\cfg{1R} & \\cfg{2R} & \\cfg{early} \\\\\\midrule\n")
        for card in ("5090", "4090"):
            rows = [h for h in V if h["card"] == card and "bypassplanS" in h["cells"][14]["ratio"]]
            if not rows:
                continue
            f.write("\\multicolumn{8}{@{}l}{\\emph{RTX " + card + "}} \\\\\n")
            for h in sorted(rows, key=lambda x: x["ratio"]):
                c = h["cells"][14]
                row = [h["cpu"], f"{h['ratio']:.2f}"]
                for n in ("fetchplan", "bypassplan", "bypassplanS"):
                    row.append(f"{c['ratio'][n]:.2f}")
                for n in ("fetchplan", "bypassplan", "bypassplanS"):
                    row.append(f"{c['misses'][n]:.1f}")
                f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")


if __name__ == "__main__":
    main()

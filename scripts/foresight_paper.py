"""Macros and the figure for the measured-foresight paragraph (job 093), from prereg/foresight_093.json.

Writes paper/wsg_foresight.tex (macros fsm*; also \\fsMeasuredAbstract and \\fsMeasuredContrib, the clauses the abstract
and the contributions list carry) and paper/figs/foresight.pdf (per host-bound cell: speed as a fraction of this host's
limit against the oracle's window W, with base, the model's foresight-only term and the limit as references).

    python scripts/foresight_paper.py
"""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
MACROS = {}


def M(k, v):
    MACROS[k] = v


def bytes_oracle(tagof):
    """Macros for job 094 (the MIN-with-bypass oracle beside the hit-optimal one on a fourth host) and the copy-latency
    simulation (prereg/foresight_latency_sim.json): fsb* the bypass oracle, fsp* the prefetch oracle on that host, fsl*
    the simulation."""
    d = json.load(open(P("prereg", "foresight_094.json")))
    sim = json.load(open(P("prereg", "foresight_latency_sim.json")))
    hb = d["host"]
    M("fsbHostCpu", f"{hb['B_c']:.0f}"); M("fsbHostLink", f"{hb['B_p']:.0f}"); M("fsbHostBoth", f"{hb['B_cp']:.0f}"); M("fsbHostMax", f"{hb['b_host_max']:.1f}")
    high, low, lawerr, rr, recov, phigh, plow, plaw, fits_host, fits_gpu, w16, ideal = [], [], [], [], [], [], [], [], [], [], [], []
    lawgain, lowany, rr_host, pbytes, pvsb, simadm, closed_high, pgpu = [], [], [], [], [], [], [], []
    rows = []
    for c in d["cells"]:
        tag = tagof[(c["model"], c["C"])]
        r = c["runs"]; b, o, p, o16 = r["base"], r["bypass_w0"], r["prefetch_w0"], r["bypass_w16"]
        hostb = c["model_terms"]["c_at_limit"] <= 0
        M(f"fsbRatio{tag}", f"{o['ratio_to_base'][0]:.2f}"); M(f"fsbRatioLo{tag}", f"{o['ratio_to_base'][1]:.2f}"); M(f"fsbRatioHi{tag}", f"{o['ratio_to_base'][2]:.2f}")
        M(f"fsbGain{tag}", f"{100 * o['gain']:.0f}"); M(f"fsbHit{tag}", f"{100 * o['hit_rate']:.0f}"); M(f"fsbHitBase{tag}", f"{100 * b['hit_rate']:.0f}")
        M(f"fsbMisses{tag}", f"{o['misses_per_token']:.0f}"); M(f"fsbMissesBase{tag}", f"{b['misses_per_token']:.0f}"); M(f"fsbAdmits{tag}", f"{o['admits_per_token']:.0f}")
        M(f"fsbReadsRatio{tag}", f"{o['misses_per_token'] / c['opt_reads_per_token']:.2f}")
        M(f"fsbLawErr{tag}", f"{100 * (o['law_ms'] / o['ms'] - 1):+.0f}"); M(f"fsbClosed{tag}", f"{100 * o['gap_closed']:.0f}"); M(f"fsbRecovered{tag}", f"{100 * o['foresight_recovered']:.0f}")
        M(f"fspRatio{tag}", f"{p['ratio_to_base'][0]:.2f}"); M(f"fspRatioLo{tag}", f"{p['ratio_to_base'][1]:.2f}"); M(f"fspRatioHi{tag}", f"{p['ratio_to_base'][2]:.2f}")
        M(f"fspGain{tag}", f"{100 * p['gain']:.0f}"); M(f"fspLawErr{tag}", f"{100 * (p['law_ms'] / p['ms'] - 1):+.0f}")
        M(f"fspReads{tag}", f"{p['misses_per_token'] + p['admits_per_token']:.0f}"); M(f"fsbReadsBase{tag}", f"{b['misses_per_token'] + b['admits_per_token']:.0f}")
        M(f"fsbSixteenRatio{tag}", f"{o16['ratio_to_base'][0]:.2f}")
        lawerr.append(o["law_ms"] / o["ms"] - 1); rr.append(o["misses_per_token"] / c["opt_reads_per_token"])
        lg = b["ms"] / o["law_ms"] - 1                       # the gain the law predicts from the oracle's own counters
        M(f"fsbLawGain{tag}", f"{100 * lg:.0f}")
        if c["C"] in (14, 16):
            low.append(o["gain"]); plow.append(-p["gain"]); lowany += [o["gain"], o16["gain"]]
        else:
            high.append(o["gain"]); recov.append(o["foresight_recovered"]); phigh.append(p["gain"]); plaw.append(p["law_ms"] / p["ms"] - 1)
            lawgain.append(lg); closed_high.append(o["gap_closed"])
            pbytes.append((p["misses_per_token"] + p["admits_per_token"]) / (b["misses_per_token"] + b["admits_per_token"]) - 1)
            pvsb.append((p["misses_per_token"] + p["admits_per_token"]) / (o["misses_per_token"] + o["admits_per_token"]))
        if hostb:
            rr_host.append(o["misses_per_token"] / c["opt_reads_per_token"])
        else:
            pgpu.append(p["ratio_to_base"][0])
        sc = sim["models"][c["model"]]["cells"][str(c["C"])]
        M(f"fslIdeal{tag}", f"{sc['by_d']['1']['misses_per_token']:.0f}"); M(f"fslThree{tag}", f"{sc['by_d']['3']['misses_per_token']:.0f}"); M(f"fslTwo{tag}", f"{sc['by_d']['2']['misses_per_token']:.0f}")
        fit = sc.get("engine", {}).get("d_fit")
        M(f"fslFit{tag}", f"{fit:.1f}" if fit is not None else "below the ideal")
        if fit is not None:
            (fits_host if hostb else fits_gpu).append(fit)
            lo_d, hi_d = int(fit), int(fit) + 1
            sa = sc["by_d"][str(lo_d)]["admissions_per_token"] + (fit - lo_d) * (sc["by_d"][str(hi_d)]["admissions_per_token"] - sc["by_d"][str(lo_d)]["admissions_per_token"])
            simadm.append(sa / o["admits_per_token"])
        if hostb:
            ideal.append(sc["by_d"]["1"]["misses_per_token"] / c["opt_reads_per_token"] - 1)
        if o["ratio_to_base"][1] > 1:
            w16.append(o16["gain"] / o["gain"])
        rows.append((tag, b, o, o16, p, c, sc))
    M("fsbGainHighMin", f"{100 * min(high):.0f}"); M("fsbGainHighMax", f"{100 * max(high):.0f}")
    M("fsbGainLowMin", f"{100 * min(low):.0f}"); M("fsbGainLowMax", f"{100 * max(low):.0f}")
    M("fsbLawErrMin", f"{100 * min(lawerr):.0f}"); M("fsbLawErrMax", f"{100 * max(lawerr):.0f}")
    M("fsbReadsRatioMin", f"{min(rr):.2f}"); M("fsbReadsRatioMax", f"{max(rr):.2f}")
    M("fsbRecovHighMin", f"{100 * min(recov):.0f}"); M("fsbRecovHighMax", f"{100 * max(recov):.0f}")
    M("fspGainHighMin", f"{100 * min(phigh):.0f}"); M("fspGainHighMax", f"{100 * max(phigh):.0f}")
    M("fspLossMin", f"{100 * min(plow):.0f}"); M("fspLossMax", f"{100 * max(plow):.0f}")
    M("fspLawErrHighMin", f"{100 * min(plaw):.0f}"); M("fspLawErrHighMax", f"{100 * max(plaw):.0f}")
    # the hit-optimal oracle's gain at the four cells of 25% and above over both hosts (job 093's cells carry fsmGain*)
    both = phigh + [float(MACROS[k]) / 100 for k in ("fsmGainGMid", "fsmGainGHigh", "fsmGainQMid", "fsmGainQHigh")]
    M("fsHitGainMin", f"{100 * min(both):.0f}"); M("fsHitGainMax", f"{100 * max(both):.0f}")
    gpu_both = pgpu + [float(MACROS[k]) for k in ("fsmRatioGHigh", "fsmRatioQHigh")]
    M("fsHitGpuMin", f"{min(gpu_both):.2f}"); M("fsHitGpuMax", f"{max(gpu_both):.2f}")
    loss_both = plow + [-float(MACROS[k]) / 100 for k in ("fsmGainGLow", "fsmGainQLow")]
    M("fsHitLossMin", f"{100 * min(loss_both):.0f}"); M("fsHitLossMax", f"{100 * max(loss_both):.0f}")
    M("fsbLawGainHighMin", f"{100 * min(lawgain):.0f}"); M("fsbLawGainHighMax", f"{100 * max(lawgain):.0f}")
    M("fsbGainLowMaxAny", f"{100 * max(lowany):.0f}")
    M("fsbReadsRatioHostMin", f"{min(rr_host):.2f}"); M("fsbReadsRatioHostMax", f"{max(rr_host):.2f}")
    M("fspBytesFewerMin", f"{-100 * max(pbytes):.0f}"); M("fspBytesFewerMax", f"{-100 * min(pbytes):.0f}")
    M("fspVsBypassBytesMin", f"{min(pvsb):.2f}"); M("fspVsBypassBytesMax", f"{max(pvsb):.2f}")
    M("fsbSimAdmitMin", f"{min(simadm):.1f}"); M("fsbSimAdmitMax", f"{max(simadm):.1f}")
    M("fsbClosedHighMin", f"{100 * min(closed_high):.0f}"); M("fsbClosedHighMax", f"{100 * max(closed_high):.0f}")
    M("fslFitHostMin", f"{min(fits_host):.1f}"); M("fslFitHostMax", f"{max(fits_host):.1f}")
    M("fslIdealOverMin", f"{100 * min(ideal):.0f}"); M("fslIdealOverMax", f"{100 * max(ideal):.0f}")
    if fits_gpu:
        M("fslFitGpuMax", f"{max(fits_gpu):.1f}")
    M("fsbSixteenShareMin", f"{100 * min(w16):.0f}"); M("fsbSixteenShareMax", f"{100 * max(w16):.0f}")
    # the appendix table
    names = {"GLow": "gpt-oss 11\\%", "GMid": "gpt-oss 25\\%", "GHigh": "gpt-oss 40\\%", "QLow": "Qwen3 12.5\\%", "QMid": "Qwen3 25\\%", "QHigh": "Qwen3 43.75\\%"}
    with open(P("paper", "tab_foresight_bytes.tex"), "w") as f:
        f.write("% generated by scripts/foresight_paper.py from prereg/foresight_094.json and prereg/foresight_latency_sim.json\n")
        f.write("\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{1.7pt}\n")
        f.write("\\caption{The bytes-optimal oracle (MIN with bypass) beside the hit-optimal one on the second oracle host (RTX 5090, Ryzen 9 7950X, host memory "
                "\\fsbHostMax\\,GB/s at best). Per cell: the online policy's speed; each oracle's ratio to it with the whole sequence in view (paired, 95\\% interval), "
                "its hit rate, its misses per token (experts read from host memory by the CPU or by a demand copy; its background admissions, each read a second time by the copy, in parentheses) "
                "and the law on its own counters against its measured time; the optimum's reads and hit rate from the trace; "
                "and MIN with bypass simulated on the trace with the engine's lookahead (within the sequence) under a copy latency of 1 (ideal), 2 and 3 steps, "
                "with the latency at which the simulation meets the engine's misses.}\n")
        f.write("\\label{tab:foresight_bytes}\n")
        f.write("\\begin{tabular}{@{}lr|lrrr|lrrr|rr|rrrr@{}}\\toprule\n")
        f.write(" & & \\multicolumn{4}{c|}{bypass oracle, $W{=}$all} & \\multicolumn{4}{c|}{hit-optimal oracle, $W{=}$all} & \\multicolumn{2}{c|}{optimum} & \\multicolumn{4}{c}{simulated misses/token} \\\\\n")
        f.write("Cell & online & ratio & hits & misses (adm.) & law & ratio & hits & misses (adm.) & law & reads & hits & $d{=}1$ & 2 & 3 & fit \\\\\\midrule\n")
        for tag, b, o, o16, p, c, sc in rows:
            fit = sc.get("engine", {}).get("d_fit")
            f.write("%s & %.1f & %.2f [%.2f, %.2f] & %.0f & %.0f (%.0f) & %+.0f\\%% & %.2f [%.2f, %.2f] & %.0f & %.0f (%.0f) & %+.0f\\%% & %.1f & %.0f & %.0f & %.0f & %.0f & %s \\\\\n" % (
                names[tag], b["mean"], o["ratio_to_base"][0], o["ratio_to_base"][1], o["ratio_to_base"][2], 100 * o["hit_rate"], o["misses_per_token"], o["admits_per_token"], 100 * (o["law_ms"] / o["ms"] - 1),
                p["ratio_to_base"][0], p["ratio_to_base"][1], p["ratio_to_base"][2], 100 * p["hit_rate"], p["misses_per_token"], p["admits_per_token"], 100 * (p["law_ms"] / p["ms"] - 1),
                c["opt_reads_per_token"], c["opt_hit_rate"], sc["by_d"]["1"]["misses_per_token"], sc["by_d"]["2"]["misses_per_token"], sc["by_d"]["3"]["misses_per_token"],
                ("%.1f" % fit) if fit is not None else "below ideal"))
        f.write("\\bottomrule\\end{tabular}\\end{table*}\n")


def main():
    d = json.load(open(P("prereg", "foresight_093.json")))
    cells = d["cells"]
    host = [c for c in cells if c["model_terms"]["c_at_limit"] <= 0]        # host-bound at the limit
    gpu = [c for c in cells if c["model_terms"]["c_at_limit"] > 0]
    tagof = {("gpt-oss-120b", 14): "GLow", ("gpt-oss-120b", 32): "GMid", ("gpt-oss-120b", 51): "GHigh",
             ("qwen3-30b-a3b-bf16", 16): "QLow", ("qwen3-30b-a3b-bf16", 32): "QMid", ("qwen3-30b-a3b-bf16", 56): "QHigh"}
    gains, recov, closed, fr0, frb, hit0, hitb, admits, lawerr, noovl, allcpu, w50m = [], [], [], [], [], [], [], [], [], [], [], []
    for c in cells:
        tag = tagof[(c["model"], c["C"])]
        r = c["runs"]
        if "oracle_w0" not in r:
            continue
        o, b = r["oracle_w0"], r["base"]
        M(f"fsmGain{tag}", f"{100 * o['gain']:.0f}"); M(f"fsmGainLo{tag}", f"{100 * (o['ratio_to_base'][1] - 1):.0f}"); M(f"fsmGainHi{tag}", f"{100 * (o['ratio_to_base'][2] - 1):.0f}")
        M(f"fsmBase{tag}", f"{b['mean']:.1f}"); M(f"fsmOracle{tag}", f"{o['mean']:.1f}")
        M(f"fsmFracBase{tag}", f"{100 * b['frac_of_limit']:.0f}"); M(f"fsmFracOracle{tag}", f"{100 * o['frac_of_limit']:.0f}")
        M(f"fsmHitBase{tag}", f"{100 * (b['hit_rate'] or 0):.0f}"); M(f"fsmHitOracle{tag}", f"{100 * (o['hit_rate'] or 0):.0f}")
        M(f"fsmHitOpt{tag}", f"{c['opt_hit_rate']:.0f}")
        M(f"fsmModelF{tag}", f"{1e3 / c['model_terms']['v_F_ms']:.1f}")
        if o["foresight_recovered"] is not None:
            M(f"fsmRecovered{tag}", f"{100 * o['foresight_recovered']:.0f}")
        M(f"fsmClosed{tag}", f"{100 * o['gap_closed']:.0f}")
        M(f"fsmReadsBase{tag}", f"{b['misses_per_token'] + b['admits_per_token']:.0f}")
        M(f"fsmReadsOracle{tag}", f"{o['misses_per_token'] + o['admits_per_token']:.0f}")
        M(f"fsmReadsOpt{tag}", f"{c['opt_reads_per_token']:.0f}")
        M(f"fsmRatio{tag}", f"{o['ratio_to_base'][0]:.2f}"); M(f"fsmRatioLo{tag}", f"{o['ratio_to_base'][1]:.2f}"); M(f"fsmRatioHi{tag}", f"{o['ratio_to_base'][2]:.2f}")
        M(f"fsmLawErr{tag}", f"{100 * (o['law_ms'] / o['ms'] - 1):+.0f}")
        gains.append(o["gain"]); closed.append(o["gap_closed"]); fr0.append(o["frac_of_limit"]); frb.append(b["frac_of_limit"])
        hit0.append(o["hit_rate"] or 0); hitb.append(b["hit_rate"] or 0); admits.append(o["admits_per_token"] / max(1e-9, c["opt_reads_per_token"]))
        lawerr.append(o["law_ms"] / o["ms"] - 1)
        if o["foresight_recovered"] is not None:
            recov.append(o["foresight_recovered"])
        if "noovl" in r:
            noovl.append(1 - r["noovl"]["ratio_to_base"][0])
        if "allcpu" in r:
            allcpu.append(1 - r["allcpu"]["ratio_to_base"][0])
        if c.get("w50_measured") is not None and c["w50_measured"] != float("inf"):
            w50m.append((tag, c["w50_measured"], c.get("w50_trace")))
    hostcells = [c for c in host if "oracle_w0" in c["runs"]]
    hg = [c["runs"]["oracle_w0"]["gain"] for c in hostcells]
    hc = [c["runs"]["oracle_w0"]["gap_closed"] for c in hostcells]
    hrec = [c["runs"]["oracle_w0"]["foresight_recovered"] for c in hostcells if c["runs"]["oracle_w0"]["foresight_recovered"] is not None]
    hf0 = [c["runs"]["oracle_w0"]["frac_of_limit"] for c in hostcells]; hfb = [c["runs"]["base"]["frac_of_limit"] for c in hostcells]
    M("fsmHostGainMin", f"{100 * min(hg):.0f}"); M("fsmHostGainMax", f"{100 * max(hg):.0f}")
    M("fsmHostClosedMin", f"{100 * min(hc):.0f}"); M("fsmHostClosedMax", f"{100 * max(hc):.0f}")
    if hrec:
        M("fsmHostRecovMin", f"{100 * min(hrec):.0f}"); M("fsmHostRecovMax", f"{100 * max(hrec):.0f}")
    M("fsmHostFracMin", f"{100 * min(hf0):.0f}"); M("fsmHostFracMax", f"{100 * max(hf0):.0f}")
    M("fsmHostFracBaseMin", f"{100 * min(hfb):.0f}"); M("fsmHostFracBaseMax", f"{100 * max(hfb):.0f}")
    gg = [c["runs"]["oracle_w0"]["gain"] for c in gpu if "oracle_w0" in c["runs"]]
    if gg:
        M("fsmGpuGainMin", f"{100 * min(gg):.0f}"); M("fsmGpuGainMax", f"{100 * max(gg):.0f}")
    if noovl:
        M("fsmNoovlMin", f"{100 * min(noovl):.0f}"); M("fsmNoovlMax", f"{100 * max(noovl):.0f}")
    paced = [c["runs"]["paced"]["ratio_to_base"][0] for c in cells if "paced" in c["runs"]]
    if paced:
        M("fsmPacedMin", f"{min(paced):.3f}"); M("fsmPacedMax", f"{max(paced):.3f}")
    if allcpu:
        M("fsmAllcpuMin", f"{100 * min(allcpu):.0f}"); M("fsmAllcpuMax", f"{100 * max(allcpu):.0f}")
    M("fsmLawErrMin", f"{100 * min(lawerr):+.0f}"); M("fsmLawErrMax", f"{100 * max(lawerr):+.0f}")
    pos = [c for c in cells if "oracle_w0" in c["runs"] and c["runs"]["oracle_w0"]["gain"] > 0]
    neg = [c for c in cells if "oracle_w0" in c["runs"] and c["runs"]["oracle_w0"]["gain"] <= 0]
    M("fsmPosCells", str(len(pos))); M("fsmNegCells", str(len(neg)))
    M("fsmPosGainMin", f"{100 * min(c['runs']['oracle_w0']['gain'] for c in pos):.0f}"); M("fsmPosGainMax", f"{100 * max(c['runs']['oracle_w0']['gain'] for c in pos):.0f}")
    if neg:
        M("fsmLossMin", f"{-100 * max(c['runs']['oracle_w0']['gain'] for c in neg):.0f}"); M("fsmLossMax", f"{-100 * min(c['runs']['oracle_w0']['gain'] for c in neg):.0f}")
        M("fsmLossRange", MACROS["fsmLossMax"] if MACROS["fsmLossMin"] == MACROS["fsmLossMax"] else MACROS["fsmLossMin"] + "--" + MACROS["fsmLossMax"])
    pl = [c["runs"]["oracle_w0"]["law_ms"] / c["runs"]["oracle_w0"]["ms"] - 1 for c in pos]
    M("fsmLawErrPosMin", f"{100 * min(pl):.0f}"); M("fsmLawErrPosMax", f"{100 * max(pl):.0f}")
    hpos = [c for c in hostcells if c["runs"]["oracle_w0"]["gain"] > 0]
    M("fsmHostPosClosedMin", f"{100 * min(c['runs']['oracle_w0']['gap_closed'] for c in hpos):.0f}"); M("fsmHostPosClosedMax", f"{100 * max(c['runs']['oracle_w0']['gap_closed'] for c in hpos):.0f}")
    M("fsmHostPosRecovMin", f"{100 * min(c['runs']['oracle_w0']['foresight_recovered'] for c in hpos):.0f}"); M("fsmHostPosRecovMax", f"{100 * max(c['runs']['oracle_w0']['foresight_recovered'] for c in hpos):.0f}")
    # every window at the losing cells
    wl = [r["ratio_to_base"][0] for c in neg for k, r in c["runs"].items() if k.startswith("oracle_w")]
    if wl:
        M("fsmNegWinMin", f"{min(wl):.2f}"); M("fsmNegWinMax", f"{max(wl):.2f}")
    M("fsmAdmitRatioMin", f"{min(admits):.2f}"); M("fsmAdmitRatioMax", f"{max(admits):.2f}")
    for tag, w, wt in w50m:
        M(f"fsmWfifty{tag}", f"{w:.1f}")
    if w50m:
        M("fsmWfiftyMin", f"{min(w for _, w, _ in w50m):.1f}"); M("fsmWfiftyMax", f"{max(w for _, w, _ in w50m):.1f}")
    hb = d["host"]
    M("fsmHostCpu", f"{hb['B_c']:.0f}"); M("fsmHostLink", f"{hb['B_p']:.0f}"); M("fsmHostBoth", f"{hb['B_cp']:.0f}"); M("fsmHostMax", f"{hb['b_host_max']:.1f}")
    M("fsmCells", str(len([c for c in cells if "oracle_w0" in c["runs"]])))
    # the abstract's and the contributions' clauses
    M("fsMeasuredContrib", " (worth \\fsbGainHighMin--\\fsbGainHighMax\\%, mostly on the bytes it saves, and \\fsHitGainMin--\\fsHitGainMax\\% with the overlap "
                           "it allows, at budgets of 25\\% and above; at most \\fsbGainLowMaxAny\\% below)")
    bytes_oracle(tagof)
    with open(P("paper", "wsg_foresight.tex"), "w") as f:
        f.write("% generated by scripts/foresight_paper.py from prereg/foresight_093.json\n")
        for k in sorted(MACROS):
            f.write("\\newcommand{\\%s}{%s}\n" % (k, MACROS[k]))
    # the appendix table
    names = {"GLow": "gpt-oss 11\\%", "GMid": "gpt-oss 25\\%", "GHigh": "gpt-oss 40\\%", "QLow": "Qwen3 12.5\\%", "QMid": "Qwen3 25\\%", "QHigh": "Qwen3 43.75\\%"}
    rows = []
    for c in cells:
        tag = tagof[(c["model"], c["C"])]
        r = c["runs"]
        if "oracle_w0" not in r:
            continue
        o, b = r["oracle_w0"], r["base"]
        neg = lambda x: ("$-$%.0f" % -x) if x < 0 else ("%.0f" % x)  # noqa: E731
        rec = neg(100 * o["foresight_recovered"]) if o["foresight_recovered"] is not None else "--"
        rows.append("%s & %.1f & %.1f & %.2f [%.2f, %.2f] & %.0f $\\to$ %.0f (%.0f) & %.0f $\\to$ %.0f (%.0f) & %.0f & %.1f $\\to$ %.1f & %.1f & %s & %s \\\\" % (
            names[tag], b["mean"], o["mean"], o["ratio_to_base"][0], o["ratio_to_base"][1], o["ratio_to_base"][2],
            100 * (b["hit_rate"] or 0), 100 * (o["hit_rate"] or 0), c["opt_hit_rate"],
            b["misses_per_token"], o["misses_per_token"], c["opt_reads_per_token"], o["admits_per_token"],
            100 * b["frac_of_limit"], 100 * o["frac_of_limit"], 1e3 / c["model_terms"]["v_F_ms"], rec, neg(100 * o["gap_closed"])))
    with open(P("paper", "tab_foresight.tex"), "w") as f:
        f.write("% generated by scripts/foresight_paper.py from prereg/foresight_093.json\n")
        f.write("\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n")
        f.write("\\caption{The hit-optimal oracle in the engine (job 093; RTX 5090, Ryzen 9 9950X, host memory \\fsmHostMax\\,GB/s at best). "
                "tok/s of the online policy and of the oracle with the whole sequence in view; their ratio, paired by sequence, with its 95\\% interval; "
                "hit rate and misses per token of each (experts read from host memory by the CPU or by a demand copy), the optimum's reads in parentheses, and the oracle's background admissions, each read a second time by the copy; speed as a percentage of this "
                "host's limit; the accounting's foresight-only term $v(F)$ for the cell in tok/s; the share of that term the oracle recovered and the share of the gap to the limit it closed.}\n")
        f.write("\\label{tab:foresight}\n")
        f.write("\\begin{tabular}{@{}lrrlrrrrrrr@{}}\\toprule\n")
        f.write("Cell & online & oracle & ratio & hits \\% (opt.) & misses/token (opt.) & oracle adm. & \\% of limit & $v(F)$ & recovered \\% & closed \\% \\\\\\midrule\n")
        for r in rows:
            f.write(r + "\n")
        f.write("\\bottomrule\\end{tabular}\\end{table*}\n")
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    cols = {"GLow": "#1f5fa8", "GMid": "#5b9bd5", "QLow": "#c0572b", "QMid": "#e39a6b"}
    names = {"GLow": "gpt-oss 11%", "GMid": "gpt-oss 25%", "QLow": "Qwen3 12.5%", "QMid": "Qwen3 25%"}
    xs_all = []
    for c in hostcells:
        tag = tagof[(c["model"], c["C"])]
        r = c["runs"]
        pts = [(0.7, r["base"]["frac_of_limit"])]
        for w in (2, 4, 16, 64):
            if f"oracle_w{w}" in r:
                pts.append((w, r[f"oracle_w{w}"]["frac_of_limit"]))
        pts.append((256, r["oracle_w0"]["frac_of_limit"]))
        xs, ys = zip(*pts)
        ax.plot(xs, [100 * y for y in ys], "-o", color=cols[tag], ms=3.5, lw=1, label=names[tag])
        ax.plot([256], [100 * (1 / c["model_terms"]["v_F_ms"]) * (c["model_terms"]["limit_ms"])], marker="_", color=cols[tag], ms=9, mew=1.5, ls="")
        xs_all += list(xs)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks([0.7, 2, 4, 16, 64, 256])
    ax.set_xticklabels(["none", "2", "4", "16", "64", "all"], fontsize=6)
    ax.set_xlabel("tokens of routing foresight W", fontsize=6.5)
    ax.set_ylabel("% of this host's limit", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.set_ylim(0, 85)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    from matplotlib.lines import Line2D
    h = ax.get_legend_handles_labels()[0] + [Line2D([], [], marker="_", color="k", ls="", ms=9, mew=1.5, label="model: foresight fix alone")]
    ax.legend(handles=h, fontsize=5.2, frameon=False, loc="lower left", ncol=2, columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(P("paper", "figs", "foresight.pdf"))
    print("macros:", ", ".join(f"{k}={v}" for k, v in sorted(MACROS.items()) if not k.startswith("fsMeasured")))


if __name__ == "__main__":
    main()

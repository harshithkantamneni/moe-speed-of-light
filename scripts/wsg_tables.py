"""Tables, figures and number macros for the paper "Where the Seconds Go" (paper/paper.tex), generated from the
outcome files so that no number is typed by hand.

Inputs (prereg/): headline_081.json, split_080.json, run1_082.json, homepc/law_crosshost.json, and when present
speed_limit_084.json (scripts/speed_limit.py on the job 084 traces), vram_084.json, hard_083.json, halfpcie_085.json,
qwen36_086.json. Missing inputs print as "--" and the macro \\pend marks them in the PDF.

    python scripts/wsg_tables.py
"""
import json
import os
import statistics as st

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
OUT = P("paper")


def load(name):
    p = P("prereg", name)
    return json.load(open(p)) if os.path.exists(p) else None


def ci(t, nd=2):
    m, lo, hi = t
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def pct(x):
    return f"{100 * x:.0f}"


macros = {}
PENDING = ["limitPctMin", "limitPctMax", "hardAllA", "hardAllB", "hardAllC", "hardLateA", "hardLateB", "hardLateC",
           "hardAnsMax"]


def M(name, val):
    macros[name] = val


def headline():
    h, s, r1 = load("headline_081.json"), load("split_080.json"), load("run1_082.json")
    sl, vr = load("speed_limit_084.json"), load("vram_084.json")
    rows = []
    for r in h["rows"]:
        rows.append(dict(model=r["model"], budget=r["budget"], llama=r["llama"], ft=r["ft_L2"], ftb=r["ft_L2_variant"],
                         ours=r["ours_L2"], ratio=r["ours_over_ft_L2"], xl=r["ours_over_llama"][0], job="081", note=""))
    m = s["means"]
    ll = r1["llama"]
    rows.append(dict(model="Qwen3-30B-A3B", budget="25%", llama=ll["q_stock_n36"]["measured"], ft=m["ft_hybrid_r0.25 L1"],
                     ftb="hybrid", ours=m["ours_C32_law L1"], ratio=s["ratios"]["C32 best ours_C32_law / ft_hybrid_r0.25 (L1)"],
                     xl=m["ours_C32_law L1"] / ll["q_stock_n36"]["measured"], job="080", note="$^\\dagger$"))
    rows.append(dict(model="Qwen3-30B-A3B", budget="43.75%", llama=ll["q_stock_n27"]["measured"], ft=m["ft_offload_r0.4375 L2"],
                     ftb="offload", ours=m["ours_C56_law L2"], ratio=s["ratios"]["ours_C56_law / ft_offload_r0.4375 (L2)"],
                     xl=m["ours_C56_law L2"] / ll["q_stock_n27"]["measured"], job="080", note=""))
    C = {("gpt-oss-120b", "11%"): 14, ("gpt-oss-120b", "25%"): 32, ("gpt-oss-120b", "40%"): 51,
         ("Qwen3-30B-A3B", "12.5%"): 16, ("Qwen3-30B-A3B", "25%"): 32, ("Qwen3-30B-A3B", "43.75%"): 56}
    lines = []
    for x in rows:
        key = "gpt-oss-120b" if x["model"].startswith("gpt") else "qwen3-30b-a3b-bf16"
        c = C[(x["model"], x["budget"])]
        lim = sl and sl.get(key, {}).get("rows", {}).get(str(c), {}).get("opt", {}).get("tok_s")
        vram = vr and vr.get("vram", {}).get(key)
        x["limit"] = lim
        lines.append(
            f"{'gpt-oss-120b' if key.startswith('gpt') else 'Qwen3-30B-A3B'} & {x['budget'].replace('%', chr(92) + '%')} & "
            f"{x['llama']:.1f} & {x['ft']:.1f} {{\\scriptsize({x['ftb']})}} & \\textbf{{{x['ours']:.1f}}} & "
            f"{ci(x['ratio'], 3)}{x['note']} & {x['xl']:.2f}$\\times$ & "
            + (f"{lim:.0f} & {100 * x['ours'] / lim:.0f}\\%" if lim else "\\pend & \\pend")
            + (f" & {vram:.0f}" if vram else " & \\pend") + " \\\\")
    body = "\n".join(lines[:3]) + "\n\\midrule\n" + "\n".join(lines[3:])
    tex = r"""\begin{table*}[t]\centering\small
\caption{Decode speed at equal GPU expert memory on one machine (RTX 5090 + Ryzen 9 9950X3D; host memory read by the
CPU at 71.6\,GB/s and over PCIe at 53.2\,GB/s). 30 AIME-25 problems, first 256 decode tokens, greedy, session.
Ours uses the FETCH split computed from this machine's bandwidth probe; FreeToken uses its faster backend per budget.
Ratios are of mean speeds, paired by problem, with 95\% bootstrap intervals; the comparison launch is separate from
the launch that picked each system's variant. \emph{Speed limit}: the fastest any exact-routing system with the same
slots per layer can decode on this machine (\cref{sec:limit}). \emph{All in VRAM}: stock llama.cpp with every weight
on an RTX PRO 6000 (same memory bandwidth as the RTX 5090, 96\,GB).
$^\dagger$FreeToken's backend was picked on the launch it is scored on, which favours it.}\label{tab:headline}
\begin{tabular}{llrrrcrrrr}\toprule
Model & Experts & llama.cpp & FreeToken & Ours & Ours $\div$ FreeToken & Ours $\div$ & Speed & Ours, \% & All in \\
 & on GPU & (tok/s) & (tok/s) & (tok/s) & (95\% CI) & llama.cpp & limit & of limit & VRAM \\\midrule
""" + body + r"""
\bottomrule\end{tabular}\end{table*}
"""
    open(P("paper", "tab_headline.tex"), "w").write(tex)
    g = [x for x in rows if x["model"].startswith("gpt")]
    q = [x for x in rows if not x["model"].startswith("gpt")]
    M("ftLeadGptMin", f"{min(100 * (x['ratio'][0] - 1) for x in g):.0f}")
    M("ftLeadGptMax", f"{max(100 * (x['ratio'][0] - 1) for x in g):.0f}")
    M("ftLeadQwenMin", f"{min(100 * (x['ratio'][0] - 1) for x in q):.0f}")
    M("ftLeadQwenMax", f"{max(100 * (x['ratio'][0] - 1) for x in q):.0f}")
    M("llamaXMin", f"{min(x['xl'] for x in rows):.1f}")
    M("llamaXMax", f"{max(x['xl'] for x in rows):.1f}")
    lims = [x["ours"] / x["limit"] for x in rows if x.get("limit")]
    if lims:
        M("limitPctMin", pct(min(lims)))
        M("limitPctMax", pct(max(lims)))
    law = [r["law_over_cur_L1"][0] for r in h["rows"]] + [s["ratios"]["C32 law / cur (L1)"][0], s["ratios"]["law / cur (L1)"][0]]
    M("lawGainMin", f"{100 * (min(law) - 1):.1f}")
    M("lawGainMax", f"{100 * (max(law) - 1):.1f}")
    return rows


def ablation():
    r1 = load("run1_082.json")
    names = ["Stock llama.cpp", "Static expert cache", "+ LRU replacement", "+ decayed-frequency policy",
             "+ GPU-signalled CPU helpers", "+ slot maps on the GPU", "+ GPU-side sampling", "+ FETCH, fixed split",
             "+ FETCH, the machine's split"]
    g, q = r1["ladder"]["g"], r1["ladder"]["q"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    y = np.arange(len(names))[::-1]
    for d, lab, col, off in ((g, "gpt-oss-120b", "#1f5fa8", 0.2), (q, "Qwen3-30B-A3B", "#c0572b", -0.2)):
        v = [s["tok_s"] for s in d]
        ax.barh(y + off, v, height=0.38, color=col, label=lab)
        for yy, vv in zip(y + off, v):
            ax.text(vv + 1.5, yy, f"{vv:.0f}", va="center", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.5)
    ax.set_xlabel("tok/s, 25% of experts on the GPU", fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.legend(fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=2)
    ax.set_xlim(0, 128)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    os.makedirs(P("paper", "figs"), exist_ok=True)
    fig.savefig(P("paper", "figs", "ablation.pdf"))
    M("ablGptStock", f"{g[0]['tok_s']:.1f}")
    M("ablGptOurs", f"{g[-1]['tok_s']:.1f}")
    M("ablQwenStock", f"{q[0]['tok_s']:.1f}")
    M("ablQwenOurs", f"{q[-1]['tok_s']:.1f}")
    M("ablGptX", f"{g[-1]['over_stock'][0]:.2f}")
    M("ablQwenX", f"{q[-1]['over_stock'][0]:.2f}")
    M("ablLruGpt", f"{g[2]['over_prev'][0]:.2f}")
    M("ablLruQwen", f"{q[2]['over_prev'][0]:.2f}")
    M("ablDfaGpt", f"{100 * (g[3]['over_prev'][0] - 1):.0f}")
    M("ablDfaQwen", f"{100 * (q[3]['over_prev'][0] - 1):.0f}")
    M("ablStaticGpt", f"{g[1]['over_prev'][0]:.2f}")
    M("ablStaticQwen", f"{q[1]['over_prev'][0]:.2f}")
    par = r1["parity"]
    M("parTopGpt", f"{100 * par['g']['ours']['top1_agree']:.1f}")
    M("parTopQwen", f"{100 * par['q']['ours']['top1_agree']:.1f}")
    M("parStockTopGpt", f"{100 * par['g']['stock_n22']['top1_agree']:.1f}")
    M("parStockTopQwen", f"{100 * par['q']['stock_n27']['top1_agree']:.1f}")
    M("parNllGpt", f"{100 * par['g']['ours']['dnll']:+.2f}")
    M("parNllQwen", f"{100 * par['q']['ours']['dnll']:+.2f}")
    lg = {x["budget"]: x for x in r1["long"] if x["model"].startswith("Qwen")}
    for b, key in (("12.5%", "A"), ("25%", "B"), ("43.75%", "C")):
        M(f"longQwen{key}", ci(lg[b]["ours_over_ft_257-end"], 3))


def hard():
    h = load("hard_083.json")
    if not h:
        return
    for r, key in zip(h["rows"], ("A", "B", "C")):
        M(f"hardAll{key}", ci(r["ours_over_ft_all"], 2))
        M(f"hardLate{key}", ci(r["ours_over_ft_257-end"], 2))
        M(f"hardOurs{key}", f"{r['ours']['all']:.1f}")
        M(f"hardFt{key}", f"{r['freetoken']['all']:.1f}")
    M("hardAnsMax", str(max(max(r["ours"]["answered"], r["freetoken"]["answered"]) for r in h["rows"])))


def law():
    rows = json.load(open(P("prereg", "homepc", "law_crosshost.json")))
    extra = []
    r1 = load("run1_082.json")
    for lab, v in r1["llama"].items():
        extra.append(dict(job="082_review_run1@vast", cpu="AMD Ryzen 9 9950X3D 16-Core Processor", config=lab,
                          predicted=v["predicted"], measured=v["measured"], err=v["err"], engine="llama.cpp"))
    rows = rows + extra
    hosts = {}
    for r in rows:
        job = r["job"].replace("073a_samehost_v2_attempt1", "073_samehost_v2")   # two launches on one machine (offer 51051777)
        hosts.setdefault((r["cpu"], job), []).append(r)
    short = {"Intel(R) Core(TM) Ultra 7 270K Plus": "Core Ultra 7 270K (8P+16E)", "AMD EPYC 7352 24-Core Processor": "EPYC 7352 (Zen 2)",
             "AMD EPYC 9655 96-Core Processor": "EPYC 9655 (12-ch.\\ DDR5)", "AMD Ryzen 9 7900 12-Core Processor": "Ryzen 9 7900",
             "AMD Ryzen 9 9950X3D 16-Core Processor": "Ryzen 9 9950X3D", "AMD Ryzen 9 7950X 16-Core Processor": "Ryzen 9 7950X",
             "AMD Ryzen 9 9950X 16-Core Processor": "Ryzen 9 9950X"}
    lines, cache_err, desk_err = [], [], []
    for (cpu, job), v in sorted(hosts.items(), key=lambda kv: kv[0][1]):
        c = [abs(r["err"]) for r in v if r["engine"] == "cache"]
        l = [r["err"] for r in v if r["engine"] != "cache"]
        if not c and not l:
            continue
        cache_err += c
        if "9655" not in cpu:
            desk_err += c
        name = short.get(cpu, cpu) + (" (A)" if job.startswith("073") else " (B)" if job.startswith("082") else "")
        lines.append(f"{name} & {job.split('_')[0]} & {len(c)} & "
                     + (f"{100 * st.median(c):.1f} & {100 * max(c):.1f}" if c else "-- & --") + " & "
                     + (", ".join(f"{100 * e:+.0f}" for e in l) if l else "--") + " \\\\")
    tex = r"""\begin{table}[t]\centering\small
\caption{The law predicted blind: each machine wrote its prediction from its own bandwidth probe, with constants frozen
in the public repository, before any model run. Error of predicted over measured speed, gpt-oss-120b (cache rows) and
llama.cpp \texttt{--n-cpu-moe} (last column, \%).}\label{tab:law}
\setlength\tabcolsep{3pt}
\begin{tabular}{llrrrl}\toprule
Host CPU & Job & $n$ & Median & Max & llama.cpp \\
 & & & $|$err$|$ \% & $|$err$|$ \% & err \% \\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}\end{table}
"""
    open(P("paper", "tab_law.tex"), "w").write(tex)
    M("lawN", str(len(cache_err)))
    M("lawHosts", str(len({k for k, v in hosts.items() if any(r["engine"] == "cache" for r in v)})))
    M("lawHostsAll", str(len(hosts)))
    gains = {}
    for job in ("069c_profile_270k", "069c_profile_epyc7352", "072_defer", "073_samehost_v2", "074_fix40", "076_table_5090",
                "077_competitors_gptoss"):
        d = json.load(open(f"/home/claude/gpu-branch/results/{job}@vast/law_prediction.json"))
        gains[job] = d["B_both"] / max(d["B_c"], d["B_p"]) - 1
    ry = [v for k, v in gains.items() if "069c" not in k]
    M("secondMin", f"{100 * min(ry):.0f}")
    M("secondMax", f"{100 * max(ry):.0f}")
    M("secondUltra", f"{100 * gains['069c_profile_270k']:.0f}")
    M("secondEpyc", f"{100 * gains['069c_profile_epyc7352']:.0f}")
    M("lawMedian", f"{100 * st.median(cache_err):.1f}")
    M("lawDeskN", str(len(desk_err)))
    M("lawDeskMedian", f"{100 * st.median(desk_err):.1f}")
    M("lawDeskPninety", f"{100 * np.percentile(desk_err, 90):.1f}")
    M("lawDeskMax", f"{100 * max(desk_err):.1f}")


def split():
    s = load("split_080.json")
    m, rt = s["means"], s["ratios"]
    rows = [("Lite", "0,0,1,1,1,2,2,3,3", "ours_C56_lite"), ("Law", "0,0,1,1,2,3,3,4,5", "ours_C56_law"),
            ("Law, fetch a single miss", "0,1,1,1,2,3,3,4,5", "ours_C56_law_f1"), ("Fixed", "0,1,1,2,3,3,4,5,6", "ours_C56_cur"),
            ("None (CPU only)", "0,\\ldots,0", "ours_C56_none"), ("All (fetch every miss)", "0,1,\\ldots,8", "ours_C56_all")]
    lines = []
    for name, t, k in rows:
        r = rt[f"{k} / cur (L1)"]
        lines.append(f"{name} & \\texttt{{{t}}} & {m[k + ' L1']:.1f} & " + ("1" if k.endswith("cur") else ci(r, 3)) + " \\\\")
    tex = r"""\begin{table}[t]\centering\small
\caption{FETCH tables at 43.75\% on Qwen3-30B-A3B (launch 1): entry $n$ is how many of a layer's $n$ missed experts are
copied over PCIe; the rest run on the CPU. Ratio to the fixed table, paired, 95\% CI.}\label{tab:split}
\setlength\tabcolsep{3pt}
\begin{tabular}{llrc}\toprule
Table & $f(1..8)$ & tok/s & vs fixed \\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}\end{table}
"""
    open(P("paper", "tab_split.tex"), "w").write(tex)
    M("splitFOne", f"{100 * (m['ours_C56_law L1'] / m['ours_C56_law_f1 L1'] - 1):.1f}")


def main():
    headline()
    ablation()
    law()
    split()
    hard()
    for k in PENDING:   # results of jobs still running print as a red marker
        macros.setdefault(k, "\\pend{}")
    with open(P("paper", "wsg_numbers.tex"), "w") as f:
        f.write("% generated by scripts/wsg_tables.py; do not edit\n")
        for k, v in sorted(macros.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    for k, v in sorted(macros.items()):
        print(f"{k:18s} {v}")


if __name__ == "__main__":
    main()

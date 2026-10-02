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
PENDING = ["hardAllA", "hardAllB", "hardAllC", "hardLateA", "hardLateB", "hardLateC",
           "hardAnsMax"]


def M(name, val):
    macros[name] = val


def headline():
    """Table 1: the six cells on both RTX 5090 hosts (listing B, jobs 080-082; the stock-clock 9950X host, job 089)."""
    h, s, r1 = load("headline_081.json"), load("split_080.json"), load("run1_082.json")
    sl, vr = load("speed_limit_084.json"), load("vram_084.json")
    sl2 = load("speed_limit_v2.json")   # the exact optimum (mosl.cachesim MIN with bypass); supersedes the 084 optimum
    sc = load("stockclock_089.json")
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
    for x in rows:
        key = "gpt-oss-120b" if x["model"].startswith("gpt") else "qwen3-30b-a3b-bf16"
        c = C[(x["model"], x["budget"])]
        lim = sl and sl.get(key, {}).get("rows", {}).get(str(c), {}).get("opt", {}).get("tok_s")
        if sl2:
            lim = sl2["models"][key]["rows"][str(c)]["limits"]["exact"]["tok_s"]
        x["limit"] = lim
    # the second host: the same six cells, FreeToken's backend carried over, the limit from that host's own probe
    rows2 = []
    for r in sc["rows"]:
        rows2.append(dict(model=r["model"], budget=r["budget"], llama=r["llama"], ft=r["ft"], ftb=r["ft_variant"], ours=r["ours"],
                          ratio=r["ours_over_ft_L2"], xl=r["ours_over_llama"][0], limit=r["limit"]["tok_s"], note=""))

    def line(x):
        key = "gpt-oss-120b" if x["model"].startswith("gpt") else "Qwen3-30B-A3B"
        lim = x["limit"]
        lead = x["ratio"][0] >= 1   # the faster of the two in bold
        ft_s = (f"\\textbf{{{x['ft']:.1f}}}" if not lead else f"{x['ft']:.1f}") + f" {{\\scriptsize({x['ftb']})}}"
        ours_s = f"\\textbf{{{x['ours']:.1f}}}" if lead else f"{x['ours']:.1f}"
        return (f"{key} & {x['budget'].replace('%', chr(92) + '%')} & {x['llama']:.1f} & {ft_s} & "
                f"{ours_s} & {ci(x['ratio'], 3)}{x['note']} & {x['xl']:.2f}$\\times$ & "
                + (f"{lim:.0f} & {100 * x['ours'] / lim:.0f}\\%" if lim else "\\pend & \\pend") + " \\\\")
    hb = sc["hosts"]["listing B (081)"]; ha = sc["hosts"]["089 (9950X, stock clock)"]
    head1 = (r"\multicolumn{9}{l}{\textbf{Host B}: Ryzen 9 9950X3D, card at %d\,MHz memory clock, CPU %.0f\,GB/s, link %.0f, together %.0f} \\" %
             (hb["mem_clock_mhz"], hb["B_c"], hb["B_p"], hb["B_both"]))
    head2 = (r"\multicolumn{9}{l}{\textbf{Host S}: Ryzen 9 9950X at the stock clock, card at %d\,MHz, CPU %.0f\,GB/s, link %.0f, together %.0f; FreeToken's backend carried over from host B} \\" %
             (ha["mem_clock_mhz"], ha["B_c"], ha["B_p"], ha["B_both"]))
    body = head1 + "\n" + "\n".join(line(x) for x in rows[:3]) + "\n" + "\n".join(line(x) for x in rows[3:]) + "\n\\midrule\n" + head2 + "\n" + "\n".join(line(x) for x in rows2)
    vram = vr["vram"]
    tex = r"""\begin{table*}[t]\centering\small
\caption{Decode speed at equal GPU expert memory on two RTX 5090 hosts. 30 AIME-25 problems, first 256 decode tokens,
greedy, session; our cache uses the FETCH split computed from each machine's bandwidth probe; FreeToken uses its faster
backend per budget, picked on host B on a separate launch and carried over to the second host. Ratios are of mean
speeds, paired by problem, with 95\% bootstrap intervals. \emph{Speed limit}: the ceiling of \cref{sec:limit} for an
exact-routing system with the same slots per layer on that machine (exact optimum, that host's highest probed rate,
datasheet GPU rate, which host B's card exceeds by 3\%; \cref{tab:limit} tightens it). FreeToken keeps one pooled
cache and llama.cpp pins whole layers: against the pooled bound their designs allow they stand at
\ftPoolPctMin--\ftPoolPctMax\% and \llPoolPctMin--\llPoolPctMax\%. With every weight in the VRAM of an RTX PRO 6000 (the RTX 5090's
datasheet bandwidth, 96\,GB), stock llama.cpp decodes gpt-oss at VRAMG and Qwen3 at VRAMQ\,tok/s.
$^\dagger$Launch 1 for both systems: FreeToken's backend was picked on the launch it is scored on, which favours it,
and the row has no confirmation launch.}\label{tab:headline}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrcrrr}\toprule
Model & Experts & llama.cpp & FreeToken & Ours & Ours $\div$ FreeToken & Ours $\div$ & Speed & Ours, \% \\
 & on GPU & (tok/s) & (tok/s) & (tok/s) & (95\% CI) & llama.cpp & limit & of limit \\\midrule
""".replace("VRAMG", f"{vram['gpt-oss-120b']:.0f}").replace("VRAMQ", f"{vram['qwen3-30b-a3b-bf16']:.0f}") + body + r"""
\bottomrule\end{tabular}}\end{table*}
"""
    open(P("paper", "tab_headline.tex"), "w").write(tex)
    g = [x for x in rows if x["model"].startswith("gpt")]
    q = [x for x in rows if not x["model"].startswith("gpt")]
    M("ftLeadGptMin", f"{min(100 * (x['ratio'][0] - 1) for x in g):.0f}")
    M("ftLeadGptMax", f"{max(100 * (x['ratio'][0] - 1) for x in g):.0f}")
    M("ftLeadQwenMin", f"{min(100 * (x['ratio'][0] - 1) for x in q):.0f}")
    M("ftLeadQwenMax", f"{max(100 * (x['ratio'][0] - 1) for x in q):.0f}")
    M("ftRatioGptMin", f"{min(x['ratio'][0] for x in g):.2f}"); M("ftRatioGptMax", f"{max(x['ratio'][0] for x in g):.2f}")
    M("ftRatioQwenMin", f"{min(x['ratio'][0] for x in q):.2f}"); M("ftRatioQwenMax", f"{max(x['ratio'][0] for x in q):.2f}")
    M("llamaXMin", f"{min(x['xl'] for x in rows):.1f}")
    M("llamaXMax", f"{max(x['xl'] for x in rows):.1f}")
    # both hosts together (the abstract's ranges)
    both = rows + rows2
    gb = [x for x in both if x["model"].startswith("gpt")]; qb = [x for x in both if not x["model"].startswith("gpt")]
    sgn = lambda v: f"${v:.0f}$" if v < 0 else f"+{v:.0f}"  # noqa: E731
    M("bothLeadGptMin", f"{min(100 * (x['ratio'][0] - 1) for x in gb):.0f}"); M("bothLeadGptMax", f"{max(100 * (x['ratio'][0] - 1) for x in gb):.0f}")
    M("bothLeadQwenMin", sgn(min(100 * (x['ratio'][0] - 1) for x in qb))); M("bothLeadQwenMax", sgn(max(100 * (x['ratio'][0] - 1) for x in qb)))
    M("bothLlamaXMin", f"{min(x['xl'] for x in both):.1f}"); M("bothLlamaXMax", f"{max(x['xl'] for x in both):.1f}")
    M("bothCells", str(len(both))); M("bothLeadCells", str(sum(1 for x in both if x["ratio"][1] > 1)))
    lims = [x["ours"] / x["limit"] for x in rows if x.get("limit")]
    if lims:
        M("limitPctMin", pct(min(lims)))
        M("limitPctMax", pct(max(lims)))
    lims2 = [x["ours"] / x["limit"] for x in both if x.get("limit")]
    M("bothLimitPctMin", pct(min(lims2))); M("bothLimitPctMax", pct(max(lims2)))
    M("llLimBothMin", pct(min(x["llama"] / x["limit"] for x in both))); M("llLimBothMax", pct(max(x["llama"] / x["limit"] for x in both)))
    M("ftLimBothMin", pct(min(x["ft"] / x["limit"] for x in both))); M("ftLimBothMax", pct(max(x["ft"] / x["limit"] for x in both)))
    # the pooled bound (one pool of L*C slots; the optimum's reads 'global' in speed_limit_v2.json) on each host, for the
    # systems without a per-layer budget: FreeToken keeps one LRU over all layers, llama.cpp pins whole layers
    import sys as _sys
    _sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from speed_limit import MODELS as _MODELS, limit as _limit, host_rates as _host_rates  # noqa: E402
    pooled = []
    for host_rows, b_host in ((rows, sl2["B_host"]["max"]), (rows2, 1e9 * sc["b_host_max_gbs"])):
        for x in host_rows:
            key = "gpt-oss-120b" if x["model"].startswith("gpt") else "qwen3-30b-a3b-bf16"
            c = {"11%": 14, "25%": 32, "40%": 51, "12.5%": 16, "43.75%": 56}[x["budget"]] if key == "gpt-oss-120b" or x["budget"] != "25%" else 32
            m2 = sl2["models"][key]; mm = _MODELS[key]
            r_pool = m2["rows"][str(c)]["reads_per_token"]["global"]
            t_pool, _ = _limit(r_pool, m2["L"] * m2["k"], mm["S"], mm["D"], b_host, sl2["B_gpu_datasheet"])
            x["limit_pooled"] = 1 / t_pool
            pooled.append(x)
    M("ftPoolPctMin", pct(min(x["ft"] / x["limit_pooled"] for x in pooled))); M("ftPoolPctMax", pct(max(x["ft"] / x["limit_pooled"] for x in pooled)))
    M("llPoolPctMin", pct(min(x["llama"] / x["limit_pooled"] for x in pooled))); M("llPoolPctMax", pct(max(x["llama"] / x["limit_pooled"] for x in pooled)))
    M("oursPoolPctMin", pct(min(x["ours"] / x["limit_pooled"] for x in pooled))); M("oursPoolPctMax", pct(max(x["ours"] / x["limit_pooled"] for x in pooled)))
    M("poolOverLayerMin", f"{100 * (min(x['limit_pooled'] / x['limit'] for x in pooled) - 1):.0f}"); M("poolOverLayerMax", f"{100 * (max(x['limit_pooled'] / x['limit'] for x in pooled) - 1):.0f}")
    json.dump([{k: v for k, v in x.items()} for x in pooled], open(P("prereg", "pooled_fractions.json"), "w"), indent=1)
    law = [r["law_over_cur_L1"][0] for r in h["rows"]] + [s["ratios"]["C32 law / cur (L1)"][0], s["ratios"]["law / cur (L1)"][0]]
    M("lawGainMin", f"{100 * (min(law) - 1):.1f}")
    M("lawGainMax", f"{100 * (max(law) - 1):.1f}")
    return rows


def ablation():
    ab, r1 = load("ablation_087.json"), load("run1_082.json")
    names = ["Stock llama.cpp", "Static cache (profiled on other text)", "+ LRU replacement", "+ decayed-frequency policy",
             "+ GPU-signalled CPU helpers", "+ slot maps on the GPU", "+ GPU-side sampling", "+ FETCH, fixed split",
             "+ FETCH, the machine's split"]
    g, q = ab["ladder"]["g"], ab["ladder"]["q"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    rows = names[:2] + ["(static, profiled in hindsight)"] + names[2:]
    y = np.arange(len(rows))[::-1]
    for d, lab, col, off, key in ((g, "gpt-oss-120b", "#1f5fa8", 0.2, "g"), (q, "Qwen3-30B-A3B", "#c0572b", -0.2, "q")):
        v = [s_["tok_s"] for s_ in d]
        v = v[:2] + [ab["extra"][key]["abl_1h_static_hindsight"]["tok_s"]] + v[2:]
        for i, (yy, vv) in enumerate(zip(y + off, v)):
            ax.barh(yy, vv, height=0.38, color="white" if i == 2 else col, edgecolor=col, hatch="////" if i == 2 else None,
                    lw=0.6, label=lab if i == 0 else None)
            ax.text(vv + 1.5, yy, f"{vv:.0f}", va="center", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels(rows, fontsize=6.3)
    ax.get_yticklabels()[2].set_color("0.35")
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
    M("ablStaticGpt", f"{g[1]['over_prev'][0]:.2f}")
    M("ablStaticQwen", f"{q[1]['over_prev'][0]:.2f}")
    M("ablStaticHitGpt", f"{100 * g[1]['hit']:.0f}")
    M("ablStaticHitQwen", f"{100 * q[1]['hit']:.0f}")
    M("ablLruGpt", f"{g[2]['over_prev'][0]:.2f}")
    M("ablLruQwen", f"{q[2]['over_prev'][0]:.2f}")
    M("ablDfaGpt", f"{100 * (g[3]['over_prev'][0] - 1):.0f}")
    M("ablDfaQwen", f"{100 * (q[3]['over_prev'][0] - 1):.0f}")
    M("ablDfaStaticGpt", f"{g[3]['tok_s'] / g[1]['tok_s']:.2f}")
    M("ablDfaStaticQwen", f"{q[3]['tok_s'] / q[1]['tok_s']:.2f}")
    for key, nm in (("g", "Gpt"), ("q", "Qwen")):
        ex = ab["extra"][key]
        M(f"ablHind{nm}", ci(ex["abl_1h_static_hindsight"]["over_dfa"], 2))
        M(f"ablHindHit{nm}", f"{100 * ex['abl_1h_static_hindsight']['hit']:.0f}")
        M(f"ablNone{nm}", f"{ex['abl_1a_nocache']['over_stock'][0]:.2f}")
    for key, nm in (("g", "Gpt"), ("q", "Qwen")):
        for step, sn in (("abl_1_static", "Static"), ("abl_2_lru", "Lru"), ("abl_3_dfa", "Dfa")):
            st_ = json.load(open(os.path.join(os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results"), f"087_ablation_static@vast/srv_{key}_{step}.json")))
            M(f"rd{sn}{nm}", f"{(st_['misses'] + st_['admits']) / st_['steps']:.0f}")
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


def simcheck():
    v = load("vram_084.json")
    sl = load("speed_limit_084.json")
    if not v or not sl:
        return
    g = [abs(x["diff"]) for k, x in v["check"].items() if k.startswith("gpt")]
    q = [abs(x["diff"]) for k, x in v["check"].items() if k.startswith("qwen")]
    M("simHitGptMax", f"{100 * max(g):.1f}")
    M("simHitQwenMin", f"{100 * min(q):.1f}")
    M("simHitQwenMax", f"{100 * max(q):.1f}")
    M("bHost", f"{sl['gpt-oss-120b']['B_host'] / 1e9:.1f}")
    M("cOneTwoEight", f"{v['ours_c128']['gpt-oss-120b']['over_stock'][0]:.3f}")
    M("cOneTwoEightQ", f"{v['ours_c128']['qwen3-30b-a3b-bf16']['over_stock'][0]:.3f}")
    M("vramGpt", f"{v['vram']['gpt-oss-120b']:.1f}")
    M("vramQwen", f"{v['vram']['qwen3-30b-a3b-bf16']:.1f}")
    for key, nm in (("gpt-oss-120b", "Gpt"), ("qwen3-30b-a3b-bf16", "Qwen")):
        row = next(iter(sl[key]["rows"].values()))
        M(f"vramFrac{nm}", f"{100 * v['vram'][key] * row['gpu_only_ms'] / 1e3:.0f}")


def foresight():
    sys_path = os.path.dirname(__file__)
    import sys as _s
    _s.path.insert(0, sys_path)
    from fig_foresight import curves, w_at
    fits = {}
    for arm in ("S", "G", "D"):
        fn = f"foresight_{arm}_exact.json" if arm == "S" else f"foresight_{arm}.json"   # S arm: exact optimum
        rows = [r for r in curves(P("prereg", "foresight", fn)) if "job 063" not in r["model"]]
        x = np.array([r["ck"] for r in rows])
        y = np.array([w_at(r["g"], 0.5) for r in rows])
        ok = np.isfinite(y)
        b, c0 = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
        rr = np.corrcoef(np.log(x[ok]), np.log(y[ok]))[0, 1]
        fits[arm] = (np.exp(c0), b, rr, int(ok.sum()), rows)
    a0, b0, r0, n0, rows = fits["S"]
    save = [1 - r["ratio"] for r in rows]
    M("fsSaveMin", f"{100 * min(save):.0f}")
    M("fsSaveMax", f"{100 * max(save):.0f}")
    M("fsSaveMed", f"{100 * float(np.median(save)):.0f}")
    M("fsPoints", str(len(rows)))
    M("fsModels", str(len({r["model"] for r in rows})))
    M("fsPre", f"{a0:.2f}")
    M("fsExp", f"{b0:.2f}")
    M("fsR", f"{r0:.3f}")
    M("fsFitN", str(n0))
    M("fsExpG", f"{fits['G'][1]:.2f}")
    M("fsExpD", f"{fits['D'][1]:.2f}")
    w90 = [w_at(r["g"], 0.9) / r["ck"] for r in rows]
    w90 = [w for w in w90 if np.isfinite(w)]
    M("fsNinety", f"{float(np.median(w90)):.1f}")
    for r in rows:
        if r["model"] == "gpt-oss-120b" and r["C"] == 32:
            M("fsGptQuarter", f"{w_at(r['g'], 0.5):.0f}")
        if r["model"] == "gpt-oss-120b" and r["C"] == 64:
            M("fsGptHalf", f"{w_at(r['g'], 0.5):.0f}")
    g = load("gap_listingb.json")
    if g:
        sp = [c["measured_ms"] / (c["measured_ms"] - dict(c["steps_ms"])["no foresight"]) for c in g["cells"]]
        M("fsSpeedMin", f"{min(sp):.2f}")
        M("fsSpeedMax", f"{max(sp):.2f}")


def qwen36():
    q = load("qwen36_086.json")
    if not q:
        return
    for r, key in zip(q["rows"], ("A", "B", "C")):
        M(f"qsixFt{key}", ci(r["ours_over_ft_L2"], 3))
        M(f"qsixLl{key}", f"{r['ours_over_llama'][0]:.2f}")
        M(f"qsixLaw{key}", f"{100 * (r['law_over_fixed_L1'][0] - 1):.1f}")
    M("qsixAuto", f"{q['ft_auto']:.1f}")


def halfpcie():
    a, b = load("halfpcie_085.json"), load("halfpcie_085b.json")
    if not a or not b:
        return
    g, q = a["rows"], b["rows"]
    M("hpGptLawMin", f"{100 * (min(r['law_over_fixed_L1'][0] for r in g) - 1):.0f}")
    M("hpGptLawMax", f"{100 * (max(r['law_over_fixed_L1'][0] for r in g) - 1):.0f}")
    M("hpQwenLawMin", f"{100 * (min(r['law_over_fixed_L1'][0] for r in q) - 1):.0f}")
    M("hpQwenLawMax", f"{100 * (max(r['law_over_fixed_L1'][0] for r in q) - 1):.0f}")
    M("hpGptFtMin", f"{min(r['ours_over_ft_L2'][0] for r in g):.2f}")
    M("hpGptFtMax", f"{max(r['ours_over_ft_L2'][0] for r in g):.2f}")
    M("hpQwenFtMin", f"{min(r['ours_over_ft_L2'][0] for r in q):.2f}")
    M("hpQwenFtMax", f"{max(r['ours_over_ft_L2'][0] for r in q):.2f}")
    import json as _j
    la = _j.loads(a["law_llama.json"])
    lb = _j.loads(b["law_llama.json"])
    mg = next(r for r in g if "llama" in r)["llama"]
    mq = next(r for r in q if "llama" in r)["llama"]
    M("hpLlamaGpt", f"{100 * (la['predicted_tok_s']['gptoss_llama_n27'] / mg - 1):+.0f}")
    M("hpLlamaQwen", f"{100 * (lb['predicted_tok_s']['qwen3_llama_n36'] / mq - 1):+.1f}")
    M("hpTableGpt", a["tables.txt"].split("gpt-oss ")[1].split(" ")[0])
    M("hpTableQwen", b["tables.txt"].split("qwen3 ")[1].split(" ")[0])


def gap():
    g = load("gap_listingb.json")
    if not g:
        return
    share = {}
    for c in g["cells"]:
        T = c["measured_ms"]
        for k, v in c["steps_ms"]:
            share.setdefault(k, []).append(v / T)
    for k, name in (("no overlap", "Ovl"), ("no foresight", "Fs"), ("policy and read paths", "Pol"), ("host work", "Host"),
                    ("GPU below datasheet (net)", "Gpu"), ("speed limit", "Lim")):
        M(f"gap{name}Min", f"{100 * min(share[k]):.0f}")
        M(f"gap{name}Max", f"{100 * max(share[k]):.0f}")
    big = sum(1 for c in g["cells"] if max(v for k, v in c["steps_ms"][1:]) - dict(c["steps_ms"])["no foresight"] < 0.05)
    M("gapFsLargest", str(big))
    M("gapCells", str(len(g["cells"])))
    c = {x["cell"]: x for x in g["cells"]}
    q = c["Qwen3 12.5%"]
    M("gapQlowReads", f"{q['engine_reads']:.0f}")
    bw = q["engine_reads"] * 9437184 * q["measured_tok_s"] / 1e9
    M("qlowBW", f"{bw:.0f}")
    M("qlowBWpct", f"{100 * bw / g['B'][2]:.0f}")


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
    per_cfg = {}
    for (cpu, job), v in sorted(hosts.items(), key=lambda kv: kv[0][1]):
        c = [abs(r["err"]) for r in v if r["engine"] == "cache"]
        l = [r["err"] for r in v if r["engine"] != "cache"]
        if not c and not l:
            continue
        cache_err += c
        for r in v:
            if r["engine"] == "cache":
                per_cfg.setdefault((cpu, job, r["config"]), []).append(r["err"])
        name = short.get(cpu, cpu) + (" (A)" if job.startswith("073") else " (B)" if job.startswith("082") else "")
        lines.append(f"{name} & {job.split('_')[0]} & {len(c)} & "
                     + (f"{100 * st.median(c):.1f} & {100 * max(c):.1f}" if c else "-- & --") + " & "
                     + (", ".join(f"{100 * e:+.0f}" for e in l) if l else "--") + " \\\\")
    tex = r"""\begin{table}[t]\centering\small
\caption{The law predicted blind: each machine wrote its prediction from its own bandwidth probe, with constants frozen
in the public repository, before any model run. Error of predicted over measured speed, gpt-oss-120b (cache rows) and
llama.cpp \texttt{--n-cpu-moe} (last column, \%).}\label{tab:law}
\setlength\tabcolsep{3pt}\resizebox{\linewidth}{!}{%
\begin{tabular}{llrrrl}\toprule
Host CPU & Job & $n$ & Median & Max & llama.cpp \\
 & & & $|$err$|$ \% & $|$err$|$ \% & err \% \\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}}\end{table}
"""
    open(P("paper", "tab_law.tex"), "w").write(tex)
    cfg = {k: float(np.mean(v)) for k, v in per_cfg.items()}
    allc = [abs(e) for e in cfg.values()]
    desk_err = [abs(e) for (cpu, job, c), e in cfg.items() if "9655" not in cpu]
    M("lawMeas", str(len(cache_err)))
    M("lawCfg", str(len(allc)))
    M("lawCfgMedian", f"{100 * st.median(allc):.1f}")
    M("lawN", str(len(cache_err)))
    M("lawHosts", str(len({k for k, v in hosts.items() if any(r["engine"] == "cache" for r in v)})))
    M("lawHostsAll", str(len(hosts)))
    gains = {}
    for job in ("069c_profile_270k", "069c_profile_epyc7352", "072_defer", "073_samehost_v2", "074_fix40", "076_table_5090",
                "077_competitors_gptoss"):
        d = json.load(open(os.path.join(os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results"), f"{job}@vast/law_prediction.json")))
        gains[job] = d["B_both"] / max(d["B_c"], d["B_p"]) - 1
    ry = [v for k, v in gains.items() if "069c" not in k]
    M("secondMin", f"{100 * min(ry):.0f}")
    M("secondMax", f"{100 * max(ry):.0f}")
    M("secondUltra", f"{100 * gains['069c_profile_270k']:.0f}")
    M("secondEpyc", f"{100 * gains['069c_profile_epyc7352']:.0f}")
    M("lawMedian", f"{100 * st.median(allc):.1f}")
    ll_off = [r["err"] for (cpu, job), v in hosts.items() for r in v if r["engine"] != "cache" and "Ryzen" not in short.get(cpu, cpu)]
    if ll_off:
        M("lawLlamaOffMin", f"{100 * min(ll_off):+.0f}"); M("lawLlamaOffMax", f"{100 * max(ll_off):+.0f}")
    M("lawRowsMedian", f"{100 * st.median([abs(e) for e in cache_err]):.1f}")   # per measurement, the LOHO note's basis
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
\setlength\tabcolsep{3pt}\resizebox{\linewidth}{!}{%
\begin{tabular}{llrc}\toprule
Table & $f(1..8)$ & tok/s & vs fixed \\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}}\end{table}
"""
    open(P("paper", "tab_split.tex"), "w").write(tex)
    M("splitFOne", f"{100 * (m['ours_C56_law L1'] / m['ours_C56_law_f1 L1'] - 1):.1f}")


def main():
    headline()
    ablation()
    law()
    split()
    hard()
    qwen36()
    halfpcie()
    gap()
    foresight()
    simcheck()
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

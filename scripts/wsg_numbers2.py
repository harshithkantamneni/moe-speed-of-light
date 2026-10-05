"""Number macros and tables for the accounting revision of "Where the Seconds Go" (paper/paper.tex), generated from the
analysis files added after the persona reviews, so that no number is typed by hand. Complements scripts/wsg_tables.py.

Inputs (prereg/): perlayer_model.json (leave-one-host-out model comparison), speed_limit_v2.json (tightened limit),
shapley_gap.json (order-independent gap attribution), policy_study.json (nine-model policy study, drift),
foresight/w50_exact.json (W50 fit with intervals), batchk_trace.json (batched verification on traces),
audit/audit_sol.json (published measurements against their own speed of light), scorecard.json (prediction clauses),
and the headline rentals' gpu.csv (memory clocks).

Outputs: paper/wsg_numbers2.tex (macros), paper/tab_limit.tex (limit variants per cell), paper/tab_policy.tex
(reads relative to the optimum by policy and budget class).

    python scripts/wsg_numbers2.py
"""
import csv
import json
import os
import statistics as st

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
RES = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")

macros = {}


def M(name, val):
    macros[name] = val


def load(*p):
    return json.load(open(P("prereg", *p)))


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}"


CELLS = [("gpt-oss-120b", "gpt-oss-120b", 14, "11\\%"), ("gpt-oss-120b", "gpt-oss-120b", 32, "25\\%"),
         ("gpt-oss-120b", "gpt-oss-120b", 51, "40\\%"), ("qwen3-30b-a3b-bf16", "Qwen3-30B-A3B", 16, "12.5\\%"),
         ("qwen3-30b-a3b-bf16", "Qwen3-30B-A3B", 32, "25\\%"), ("qwen3-30b-a3b-bf16", "Qwen3-30B-A3B", 56, "43.75\\%")]


def perlayer():
    d = load("perlayer_model.json")
    m = d["models"]
    for key, nm in (("A", "Max"), ("B", "Add"), ("C", "Layer"), ("E", "Asym"), ("D", "LayerAdd")):
        M(f"loho{nm}Med", f"{m[key]['loho']['median']:.1f}")
        M(f"loho{nm}Pninety", f"{m[key]['loho']['p90']:.1f}")
        M(f"loho{nm}Max", f"{m[key]['loho']['max']:.1f}")
        M(f"ins{nm}Med", f"{m[key]['insample']['median']:.1f}")
    M("lohoFrozenMed", f"{m['A frozen G=4.819']['insample']['median']:.1f}")
    M("lawGRefit", f"{m['A']['params']['G']:.2f}")
    M("lawGAdd", f"{m['B']['params']['G']:.2f}")
    M("vramMs", f"{d['vram_ms']:.2f}")
    rows = d["rows"]
    err = lambda key, i: 100 * (m[key]["pred_loho"][i] / rows[i]["measured"] - 1)  # noqa: E731
    fast_fetch = [i for i, r in enumerate(rows) if r["config"].endswith("_fetch") and r["B_p"] > 45]
    slow_fetch = [i for i, r in enumerate(rows) if r["config"].endswith("_fetch") and r["B_p"] <= 45 and "9655" not in r["cpu"]]
    epyc = [i for i, r in enumerate(rows) if "9655" in r["cpu"] and r["config"] in ("C14", "C32", "C56")]
    pf = [i for i, r in enumerate(rows) if "_pf" in r["config"]]
    M("lohoAddFastFetchMin", f"{min(err('B', i) for i in fast_fetch):+.0f}")
    M("lohoAddFastFetchMax", f"{max(err('B', i) for i in fast_fetch):+.0f}")
    M("lohoMaxFastFetchMin", f"{min(err('A', i) for i in fast_fetch):+.1f}")
    M("lohoMaxFastFetchMax", f"{max(err('A', i) for i in fast_fetch):+.1f}")
    M("lohoMaxSlowFetchMin", f"{min(err('A', i) for i in slow_fetch):+.1f}")
    M("lohoMaxSlowFetchMax", f"{max(err('A', i) for i in slow_fetch):+.1f}")
    M("lohoAsymSlowFetchMin", f"{min(err('E', i) for i in slow_fetch):+.1f}")
    M("lohoAsymSlowFetchMax", f"{max(err('E', i) for i in slow_fetch):+.1f}")
    for host, tag in (("270K", "Ultra"), ("EPYC7352", "Epyc")):
        i = next(i for i, r in enumerate(rows) if r["host"] == host and r["config"] == "C32_fetch")
        M(f"lohoMaxFetch{tag}", f"{err('A', i):+.1f}")
        M(f"lohoAsymFetch{tag}", f"{err('E', i):+.1f}")
        M(f"lohoFrozenFetch{tag}", f"{100 * (m['A frozen G=4.819']['pred'][i] / rows[i]['measured'] - 1):+.1f}")
    M("lohoMaxEpycMin", f"{min(err('A', i) for i in epyc):+.0f}")
    M("lohoMaxEpycMax", f"{max(err('A', i) for i in epyc):+.0f}")
    M("lohoMaxPfMin", f"{min(err('A', i) for i in pf):+.1f}")
    M("lohoMaxPfMax", f"{max(err('A', i) for i in pf):+.1f}")
    M("lohoFastFetchN", str(len(fast_fetch)))
    M("lawRows", str(len(rows)))
    M("lawHostsLoho", str(len(d["hosts"])))
    M("lawHostsLohoOther", {6: "six", 7: "seven", 8: "eight"}.get(len(d["hosts"]) - 1, str(len(d["hosts"]) - 1)))
    hf = [h for h in d["helper_fit"] if "slope_us" in h and "_pf" not in h["config"]]   # plain configurations only
    desk = [h for h in hf if "EPYC" not in h["host"]]
    ep = [h for h in hf if "9655" in h["host"]]
    M("helperRateMin", f"{min(h['slope_us'] / h['S_over_Bc_us'] for h in desk):.2f}")
    M("helperRateMax", f"{max(h['slope_us'] / h['S_over_Bc_us'] for h in desk):.2f}")
    M("helperFloorDeskMax", f"{max(h['intercept_us'] for h in desk):.0f}")
    M("epycFloorMin", f"{min(h['intercept_us'] for h in ep):.0f}")
    M("epycFloorMax", f"{max(h['intercept_us'] for h in ep):.0f}")
    fe = [h for h in d["helper_fit"] if "c1_us" in h and h["config"] == "C32_fetch" and "9655" not in h["host"]]
    M("fetchBesideMin", f"{100 * min(h['c1_us'] / h['c1_nofetch_us'] - 1 for h in fe):.0f}")
    M("fetchBesideMax", f"{100 * max(h['c1_us'] / h['c1_nofetch_us'] - 1 for h in fe):.0f}")


def limit():
    d = load("speed_limit_v2.json")
    bh = d["B_host"]["stats"]["headline"]["concurrent_sum"]
    M("bHostMed", f"{bh['median']:.1f}")
    M("bHostMean", f"{bh['mean']:.1f}")
    M("bHostMeanLo", f"{bh['mean_ci95'][0]:.1f}")
    M("bHostMeanHi", f"{bh['mean_ci95'][1]:.1f}")
    M("bHostN", str(bh["n"]))
    M("bHostZc", f"{d['B_host']['median_concurrent_zerocopy64'] / 1e9:.1f}")
    M("warmSteps", str(d["warm_steps"]))
    lines = []
    ex_ours, all_ours, ex_ft, ex_ll, all_ft, all_ll = [], [], [], [], [], []
    exact_cut, glob_cut, twoforms, twoforms_seq, l3min, l3max, optbytes = [], [], [], [], [], [], []
    for key, name, c, bud in CELLS:
        mod = d["models"][key]
        r = mod["rows"][str(c)]
        L = r["limits"]
        f = r["frac_of_limit"]
        t_meas = 1e3 / r["measured_tok_s"]["ours"]
        lines.append(f"{name} & {bud} & {L['current_084']['tok_s']:.0f} & {L['exact']['tok_s']:.0f} & {L['global']['tok_s']:.0f} & "
                     f"{L['median_bhost']['tok_s']:.0f} & {L['measured_gpu']['tok_s']:.0f} & {L['layer_sum']['tok_s']:.0f} & "
                     f"{L['all_changes']['tok_s']:.0f} & {100 * f['exact']['ours']:.0f} & {100 * f['all_changes']['ours']:.0f} \\\\")
        ex_ours.append(f["exact"]["ours"]); all_ours.append(f["all_changes"]["ours"])
        ex_ft.append(f["exact"]["freetoken"]); all_ft.append(f["all_changes"]["freetoken"])
        if f["exact"].get("llama"):
            ex_ll.append(f["exact"]["llama"]); all_ll.append(f["all_changes"]["llama"])
        rp = r["reads_per_token"]
        exact_cut.append(1 - rp["exact"] / rp["pol_084"])
        glob_cut.append(1 - rp["global"] / rp["exact"])
        twoforms.append(r["foresight_overlap_ms"]["layer_sum"] / t_meas)
        twoforms_seq.append(r["foresight_overlap_ms"]["layer_sum_dense_seq"] / t_meas)
        l3min.append(r["l3"]["reuse_distance_bytes"]["min"] / 1e9)
        optbytes.append(r["l3"]["bytes_per_token"] / 1e6)
        M(f"limEx{ {14: 'Low', 32: 'Mid', 51: 'High', 16: 'Low', 56: 'High'}[c] }{'G' if key.startswith('gpt') else 'Q'}", f"{L['exact']['tok_s']:.0f}")
    tex = r"""\begin{table}[t]\centering\small
\caption{The speed limit under each tightening, tokens per second on host B, and our cache as a
percentage of the limit as published (exact optimum, highest probed host rate, datasheet GPU rate) and with every
tightening at once. \emph{no-evict}: Belady's MIN with bypass that keeps a served expert for the step; \emph{exact}:
the same, free to evict an expert after serving it; \emph{pool}: one pool of $LC$ slots shared across layers; \emph{median}:
$B_{\mathrm{host}}$ the median of six concurrent probe samples instead of the highest; \emph{GPU}: the GPU term from
the measured all-in-VRAM time instead of the datasheet rate; \emph{layers}: the per-layer sum, which no system without
cross-layer prefetch can beat; \emph{all}: exact, pool, median and GPU together (a warm start, not included, changes
the limit by at most 1.4\%).}\label{tab:limit}
\setlength\tabcolsep{2.5pt}\resizebox{\linewidth}{!}{%
\begin{tabular}{llrrrrrrrrr}\toprule
Model & Experts & no-evict & exact & pool & median & GPU & layers & all & \multicolumn{2}{c}{Ours, \% of} \\
 & on GPU & & & & & & & & exact & all \\\midrule
""" + "\n".join(lines[:3]) + "\n\\midrule\n" + "\n".join(lines[3:]) + r"""
\bottomrule\end{tabular}}\end{table}
"""
    open(P("paper", "tab_limit.tex"), "w").write(tex)
    M("limExPctMin", pct(min(ex_ours))); M("limExPctMax", pct(max(ex_ours)))
    M("limAllPctMin", pct(min(all_ours))); M("limAllPctMax", pct(max(all_ours)))
    M("ftLimExPctMin", pct(min(ex_ft))); M("ftLimExPctMax", pct(max(ex_ft)))
    M("ftLimAllPctMin", pct(min(all_ft))); M("ftLimAllPctMax", pct(max(all_ft)))
    M("llLimExPctMin", pct(min(ex_ll))); M("llLimExPctMax", pct(max(ex_ll)))
    M("llLimAllPctMin", pct(min(all_ll))); M("llLimAllPctMax", pct(max(all_ll)))
    M("exactReadsMin", f"{100 * min(exact_cut):.1f}"); M("exactReadsMax", f"{100 * max(exact_cut):.1f}")
    M("exactReadsMoreMin", f"{100 * min(1 / (1 - x) - 1 for x in exact_cut):.1f}")
    M("exactReadsMoreMax", f"{100 * max(1 / (1 - x) - 1 for x in exact_cut):.1f}")
    M("globalReadsMin", f"{100 * min(glob_cut):.1f}"); M("globalReadsMax", f"{100 * max(glob_cut):.1f}")
    M("twoFormsMin", pct(min(twoforms))); M("twoFormsMax", pct(max(twoforms)))
    M("twoFormsSeqMin", pct(min(twoforms_seq))); M("twoFormsSeqMax", pct(max(twoforms_seq)))
    M("lthreeReuseMin", f"{min(l3min):.1f}"); M("optBytesMin", f"{min(optbytes):.0f}"); M("optBytesMax", f"{max(optbytes):.0f}")
    for key, nm in (("gpt-oss-120b", "Gpt"), ("qwen3-30b-a3b-bf16", "Qwen")):
        M(f"gpuEffPct{nm}", pct(d["models"][key]["B_gpu_effective_frac"]))
    # the aggregate line under the measured GPU rate, the ceiling the GPU-bound cells share
    for key, nm, c in (("gpt-oss-120b", "Gpt", 51), ("qwen3-30b-a3b-bf16", "Qwen", 56)):
        M(f"limMeasGpu{nm}", f"{d['models'][key]['rows'][str(c)]['limits']['measured_gpu']['tok_s']:.0f}")


def largest_counts(cell, fixes):
    """How many of the orders of applying the fixes make each fix the largest marginal cost (from values_ms)."""
    import itertools
    vals = cell["values_ms"]
    n = len(fixes)
    counts = {f: 0 for f in fixes}
    for order in itertools.permutations(range(n)):
        key = ["0"] * n
        marg = {}
        prev = vals["".join(key)]
        for i in order:
            key[i] = "1"
            cur = vals["".join(key)]
            marg[fixes[i]] = prev - cur
            prev = cur
        counts[max(marg, key=marg.get)] += 1
    return counts


def shapley():
    d = load("shapley_gap.json")
    v = d["variants"]["exact_max"]
    fs_host, ovl_gpu, gpu_sh, pol_sh, host_sh, fs_gpu, ovl_host = [], [], [], [], [], [], []
    fs_host_tok, ovl_gpu_tok = [], []
    for c in v["cells"]:
        s = c["shapley_share"]
        regime = d["safe_claim"]["exact_max"]["regime"][c["cell"]]
        (fs_host if regime == "host" else fs_gpu).append(s["foresight"])
        (ovl_gpu if regime == "GPU" else ovl_host).append(s["overlap"])
        if regime == "host":
            fs_host_tok.append(c["shapley_ms"]["foresight"] / c["measured_ms"])
        else:
            ovl_gpu_tok.append(c["shapley_ms"]["overlap"] / c["measured_ms"])
        gpu_sh.append(s["GPU efficiency"]); pol_sh.append(s["policy and read paths"]); host_sh.append(s["host work"])
        tag = {"gpt-oss 11%": "GLow", "gpt-oss 25%": "GMid", "gpt-oss 40%": "GHigh", "Qwen3 12.5%": "QLow",
               "Qwen3 25%": "QMid", "Qwen3 43.75%": "QHigh"}[c["cell"]]
        M(f"shOrders{tag}", str(c["foresight_largest_orders"]))
        M(f"shFs{tag}", pct(s["foresight"])); M(f"shOvl{tag}", pct(s["overlap"]))
    M("shFsHostTokMin", pct(min(fs_host_tok))); M("shFsHostTokMax", pct(max(fs_host_tok)))
    M("shOvlGpuTokMin", pct(min(ovl_gpu_tok))); M("shOvlGpuTokMax", pct(max(ovl_gpu_tok)))
    gg_cells = {c["cell"]: c for c in d["variants"]["gpu_gross"]["cells"]}
    gross_orders = {}
    for cell, c in gg_cells.items():
        cnt = largest_counts(c, d["variants"]["gpu_gross"]["fixes"])
        gross_orders[cell] = cnt["GPU efficiency"]
    M("shGpuGrossOrdersGHigh", str(gross_orders["gpt-oss 40%"])); M("shGpuGrossOrdersQHigh", str(gross_orders["Qwen3 43.75%"]))
    M("shGpuGrossOrdersHostMax", str(max(v_ for k_, v_ in gross_orders.items() if d["safe_claim"]["gpu_gross"]["regime"][k_] == "host")))
    net_never = all(largest_counts(c, v["fixes"])[f] == 0 for c in v["cells"] for f in ("GPU efficiency", "policy and read paths", "host work"))
    M("shNetNever", "yes" if net_never else "no")
    M("shFsHostMin", pct(min(fs_host))); M("shFsHostMax", pct(max(fs_host)))
    M("shFsGpuMin", pct(min(fs_gpu))); M("shFsGpuMax", pct(max(fs_gpu)))
    M("shOvlGpuMin", pct(min(ovl_gpu))); M("shOvlGpuMax", pct(max(ovl_gpu)))
    M("shOvlHostMin", pct(min(ovl_host))); M("shOvlHostMax", pct(max(ovl_host)))
    M("shGpuMin", pct(min(gpu_sh))); M("shGpuMax", pct(max(gpu_sh)))
    M("shPolMin", pct(min(pol_sh))); M("shPolMax", pct(max(pol_sh)))
    M("shHostMin", pct(min(host_sh))); M("shHostMax", pct(max(host_sh)))
    sc = d["safe_claim"]["exact_max"]
    M("shFsEveryOrder", str(len(sc["foresight_largest_every_order"])))
    M("shFsLargestCells", str(len(sc["foresight_largest_shapley"])))
    tie = next(c for c in v["cells"] if c["cell"] == "gpt-oss 25%")
    M("shTieMs", f"{tie['foresight_minus_next_ms']:.2f}")
    M("shOrders", str(tie["orders"]))
    vm = d["variants"]["exact_median"]
    pol_med = [c["shapley_share"]["policy and read paths"] for c in vm["cells"]]
    M("shPolMedMin", pct(min(pol_med))); M("shPolMedMax", pct(max(pol_med)))
    M("shFsEveryOrderMed", str(len(d["safe_claim"]["exact_median"]["foresight_largest_every_order"])))
    M("shBhostMed", f"{vm['B_host_gbs']:.1f}")
    vz = d["variants"]["exact_median_zc"]
    pol_zc = [c["shapley_share"]["policy and read paths"] for c in vz["cells"]]
    M("shPolZcMin", pct(min(pol_zc))); M("shPolZcMax", pct(max(pol_zc)))
    M("shBhostZc", f"{vz['B_host_gbs']:.1f}")
    M("shFsEveryOrderZc", str(len(d["safe_claim"]["exact_median_zc"]["foresight_largest_every_order"])))
    # the paper's fixed-order foresight step for comparison
    fo = [c["fixed_order_ms"]["foresight"] / c["measured_ms"] for c in v["cells"]]
    M("fixFsMin", pct(min(fo))); M("fixFsMax", pct(max(fo)))
    gg = d["variants"]["gpu_gross"]
    gross = [c["shapley_share"]["GPU efficiency"] for c in gg["cells"] if d["safe_claim"]["gpu_gross"]["regime"][c["cell"]] == "GPU"]
    M("shGpuGrossMin", pct(min(gross))); M("shGpuGrossMax", pct(max(gross)))


def policy():
    d = load("policy_study.json")
    s = d["summary"]
    cls = s["rel_opt_by_budget_class"]
    order = [("lru", "LRU"), ("lfu", "LFU"), ("df0", "Decayed frequency"), ("dfk", "\\quad with hysteresis $\\kappa$"),
             ("arc", "ARC"), ("s3fifo", "S3-FIFO"), ("static", "Static, profiled"), ("W1", "Belady, $W{=}1$"),
             ("W4", "Belady, $W{=}4$"), ("W16", "Belady, $W{=}16$")]
    lines = []
    for key, name in order:
        cells = [f"{cls[b][key]['median']:.2f}" for b in ("E/8", "E/4", "3E/8")]
        rng = f"{cls['all'][key]['min']:.2f}--{cls['all'][key]['max']:.2f}"
        lines.append(f"{name} & " + " & ".join(cells) + f" & {rng} \\\\")
    tex = r"""\begin{table}[t]\centering\small
\caption{Host reads per token relative to the exact optimum (Belady's MIN with bypass over the whole future), median
over nine models at three budgets, and the range over all 27 model--budget cells. Event-atomic replay of each model's
own sampled text; every online policy is scored on the last 90\% of the trace. Mixtral (8 experts) uses $C=2$, 3 and 4.}\label{tab:policy}
\setlength\tabcolsep{3pt}\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrc}\toprule
Policy & $C{=}E/8$ & $E/4$ & $3E/8$ & range \\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}}\end{table}
"""
    open(P("paper", "tab_policy.tex"), "w").write(tex)
    for b, tag in (("E/8", "Eeighth"), ("E/4", "Equarter"), ("3E/8", "Ethree")):
        for key, nm in (("lru", "Lru"), ("df0", "Df"), ("s3fifo", "Ssf"), ("lfu", "Lfu"), ("static", "Static"), ("W4", "Wfour"),
                        ("W16", "Wsixteen"), ("W1", "Wone")):
            M(f"pol{nm}{tag}", f"{cls[b][key]['median']:.2f}")
    M("polStaticMin", f"{cls['all']['static']['min']:.1f}"); M("polStaticMax", f"{cls['all']['static']['max']:.1f}")
    M("polLfuMax", f"{cls['all']['lfu']['max']:.1f}")
    M("polWfourMax", f"{cls['all']['W4']['max']:.2f}")
    # per-cell best online policy vs the optimum, and LRU vs decayed frequency by regime
    best, lru_more_hi, lru_more_lo, qwen2, low_spread = [], [], [], [], []
    kappa_cost, kappa_cost_q = [], []
    drift100, drift1000, skew, union = [], [], [], []
    for mname, mod in d["models"].items():
        for C, cell in mod["cells"].items():
            pol = cell["policies"]
            online = [pol[k]["rel_opt"] for k in ("lru", "lfu", "df0", "dfk", "arc", "s3fifo")]
            best.append(min(online))
            diff = 100 * (pol["lru"]["reads"] / pol["df0"]["reads"] - 1)
            spread = 100 * (max(online) / min(online) - 1)
            if cell["C_over_k"] >= 2:
                (qwen2 if mname == "qwen2-57b" else lru_more_hi).append(diff)
            else:
                lru_more_lo.append(diff)
                low_spread.append(spread)
            kc = 100 * (pol["dfk"]["reads"] / pol["df0"]["reads"] - 1)
            (kappa_cost_q if mname == "qwen3-30b-a3b" else kappa_cost).append(kc)
            drift100.append(cell["drift"]["w100"]); drift1000.append(cell["drift"]["w1000"])
            skew.append(cell["topC_share"]); union.append(cell["union_to_capacity"])
    M("polBestOnlineMin", f"{100 * (min(best) - 1):.0f}"); M("polBestOnlineMax", f"{100 * (max(best) - 1):.0f}")
    M("polLruMoreMin", f"{min(lru_more_hi):.0f}"); M("polLruMoreMax", f"{max(lru_more_hi):.0f}")
    M("polLruLowAbsMax", f"{max(abs(x) for x in lru_more_lo):.0f}")
    M("polLowSpreadMax", f"{max(low_spread):.0f}")
    M("polLruLowCells", str(len(lru_more_lo)))
    M("polQwenTwoMin", f"{min(qwen2):+.0f}"); M("polQwenTwoMax", f"{max(qwen2):+.0f}")
    M("polKappaMin", f"{min(kappa_cost):.0f}"); M("polKappaMax", f"{max(kappa_cost):.0f}")
    M("polKappaQwenMax", f"{max(kappa_cost_q):.0f}")
    M("polDfBest", str(s["best_online_policy_cells"].get("df0", 0)))
    M("polSsfBest", str(s["best_online_policy_cells"].get("s3fifo", 0)))
    M("polLruBest", str(s["best_online_policy_cells"].get("lru", 0)))
    M("polDfNear", str(s["online_within_1pct_of_best_cells"].get("df0", 0)))
    M("polSsfNear", str(s["online_within_1pct_of_best_cells"].get("s3fifo", 0)))
    M("polCells", str(len(best)))
    M("driftHundredMin", pct(min(drift100))); M("driftHundredMax", pct(max(drift100)))
    M("driftThousandMin", pct(min(drift1000))); M("driftThousandMax", pct(max(drift1000)))
    M("driftThousandMed", pct(float(np.median(drift1000))))
    d4 = s["drift_at_E4"]
    rising = sum(1 for m in d4.values() if m["w1000"] > m["w100"])
    M("driftRisingModels", str(rising))
    M("driftQwenTwoHundred", pct(d4["qwen2-57b"]["w100"])); M("driftQwenTwoThousand", pct(d4["qwen2-57b"]["w1000"]))
    eq = [cell["topC_share"] for mod in d["models"].values() for C, cell in mod["cells"].items() if C == str(mod["budgets"][1])]
    M("skewQuarterMin", pct(min(eq))); M("skewQuarterMax", pct(max(eq)))
    M("unionMin", f"{min(union):.1f}"); M("unionMax", f"{max(union):.1f}")
    M("polRuntimeMin", f"{d['meta']['runtime_s'] / 60:.0f}")
    M("polModels", str(len(d["models"])))


def foresight():
    w = load("foresight", "w50_exact.json")
    M("fsPreX", f"{w['full']['prefactor']:.2f}"); M("fsExpX", f"{w['full']['exponent']:.2f}")
    M("fsRX", f"{w['full']['r']:.3f}"); M("fsFitNX", str(w["full"]["n"])); M("fsInterp", str(w["n_interpolated"]))
    lo, hi = w["bootstrap_models"]["exponent_ci95"]
    M("fsExpLo", f"{lo:.2f}"); M("fsExpHi", f"{hi:.2f}")
    lomo = [f["exponent"] for f in w["leave_one_model_out"]["fits"].values()]
    M("fsExpLomoMin", f"{min(lomo):.2f}"); M("fsExpLomoMax", f"{max(lomo):.2f}")
    M("fsExpNoInterp", f"{w['no_interpolated']['exponent']:.2f}")
    lo2, hi2 = w["no_interpolated"]["bootstrap_models"]["exponent_ci95"]
    M("fsExpNoInterpLo", f"{lo2:.2f}"); M("fsExpNoInterpHi", f"{hi2:.2f}")
    sv = w["versus_original"]["summary"]["saving_new"]
    M("fsSaveMinX", pct(sv["min"])); M("fsSaveMaxX", pct(sv["max"])); M("fsSaveMedX", pct(sv["median"]))
    M("fsWfiftyShiftMax", f"{w['versus_original']['summary']['w50_new_over_old']['max']:.2f}")


def batchk():
    d = load("batchk_trace.json")
    s = d["summary"]
    def med(b, K, a, var, pol="online"):
        return s[f"{b}|K{K}|alpha{a}|{var}|{pol}"]["median"]
    free8 = [1 - med(b, 8, a, "free") for b in ("E/8", "E/4", "3E/8") for a in ("0.6", "0.8", "0.9")]
    free8_best = [1 - s[f"{b}|K8|alpha{a}|free|online"]["min"] for b in ("E/8", "E/4", "3E/8") for a in ("0.6", "0.8", "0.9")]
    M("bkFreeModelMax", pct(max(free8_best)))
    M("bkFreeMin", pct(min(free8))); M("bkFreeMax", pct(max(free8)))
    free4 = [1 - med(b, 4, a, "free") for b in ("E/8", "E/4", "3E/8") for a in ("0.6", "0.8", "0.9")]
    M("bkFreeFourMin", pct(min(free4))); M("bkFreeFourMax", pct(max(free4)))
    same9 = [1 - med(b, 8, "0.9", "same") for b in ("E/8", "E/4", "3E/8")]
    M("bkSameNineMin", pct(min(same9))); M("bkSameNineMax", pct(max(same9)))
    same8 = [1 - med(b, 8, "0.8", "same") for b in ("E/8", "E/4", "3E/8")]
    M("bkSameEightMin", pct(min(same8))); M("bkSameEightMax", pct(max(same8)))
    M("bkSameEightLossMin", pct(-max(same8))); M("bkSameEightLossMax", pct(-min(same8)))
    rnd8 = [med(b, 8, "0.8", "random") for b in ("E/8", "E/4", "3E/8")]
    M("bkRandomEightMin", f"{min(rnd8):.1f}"); M("bkRandomEightMax", f"{max(rnd8):.1f}")
    bel = [1 - med(b, K, a, "free", "belady") for b in ("E/8", "E/4", "3E/8") for a in ("0.6", "0.8", "0.9") for K in (4, 8)]
    M("bkBeladyMin", pct(min(bel))); M("bkBeladyMax", pct(max(bel)))
    # union of K positions' expert sets relative to K*k at K = 8
    un = []
    for mod in d["models"].values():
        for b in mod["budgets"].values():
            for r in b["rows"]:
                if r["K"] == 8 and r.get("union_over_Kk") is not None:
                    un.append(r["union_over_Kk"])
    M("bkUnionMin", pct(min(un))); M("bkUnionMax", pct(max(un)))
    M("bkTokens", f"{d['token_cap'] // 1000}")


def audit():
    a = load("audit", "audit_sol.json")
    cs = a["class_stats"]
    for key, nm in (("trace", "Trace"), ("iid", "Iid"), ("nocap", "Nocap"), ("allfit", "Allfit"), ("all", "All")):
        M(f"aud{nm}Med", f"{cs[key]['median']:.1f}"); M(f"aud{nm}N", str(cs[key]["n"]))
        M(f"aud{nm}Qone", f"{cs[key]['q1']:.1f}"); M(f"aud{nm}Qthree", f"{cs[key]['q3']:.1f}")
        M(f"aud{nm}Max", f"{cs[key]['max']:.1f}")
    srcs = {r.get("source", r.get("paper")) for r in a["rows"]} if isinstance(a["rows"], list) else set(a["rows"])
    M("audSources", str(len(a["by_source"]) if isinstance(a["by_source"], (dict, list)) else len(srcs)))
    # the trace-scored rows inside the bound's class (exact routing, one token per pass): no speculative or lossy systems;
    # and the rows whose per-layer capacity was imputed as an upper bound from the card's memory (audit.json budget_imputed)
    full = load("audit", "audit.json")["rows"]
    byid = {r["id"]: r for r in full}
    inclass = [r["pct_of_sol"] for r in a["rows"] if r["cls"] == "trace" and not byid[r["id"]].get("speculative") and not byid[r["id"]].get("lossy")]
    inclass.sort()
    import statistics
    M("audInclassN", str(len(inclass))); M("audInclassMed", f"{statistics.median(inclass):.1f}")
    M("audInclassQone", f"{inclass[len(inclass) // 4]:.1f}"); M("audInclassQthree", f"{inclass[(3 * len(inclass)) // 4]:.1f}")
    M("audInclassBelowFifth", str(sum(1 for x in inclass if x < 20)))
    M("audImputedN", str(sum(1 for r in full if r.get("budget_imputed"))))
    M("audImputedTraceN", str(sum(1 for r in a["rows"] if r["cls"] == "trace" and byid[r["id"]].get("budget_imputed"))))
    M("audSpecN", str(sum(1 for r in a["rows"] if r["cls"] == "trace" and byid[r["id"]].get("speculative"))))
    M("audLossyN", str(sum(1 for r in a["rows"] if r["cls"] == "trace" and byid[r["id"]].get("lossy"))))


def scorecard():
    s = load("scorecard.json")
    t = s["total"]
    M("scHeld", str(t["held"])); M("scHeldPoint", str(t["held (point)"])); M("scFailed", str(t["failed"]))
    M("scUntested", str(t["untested"])); M("scVoid", str(t["void"])); M("scClauses", str(t["clauses"]))
    M("scHeldAnyPct", pct(t["held_any_share_of_scored"])); M("scScored", str(t["scored"])); M("scHeldStrictPct", pct(t["held_share_of_scored"]))
    rev = next(k for k in s["by_era"] if k.startswith("088"))         # the review-era key grows with every job
    M("scRevEra", rev.replace("-", "--")); M("scLastJob", rev.split("-")[1])
    for era, nm in (("073-075", "Early"), ("076-081", "Mid"), ("082-087", "Late"), (rev, "Rev")):
        if era not in s["by_era"]:
            continue
        e = s["by_era"][era]
        M(f"sc{nm}Held", str(e["held"])); M(f"sc{nm}HeldPoint", str(e["held (point)"])); M(f"sc{nm}Failed", str(e["failed"]))
        M(f"sc{nm}Clauses", str(e["clauses"])); M(f"sc{nm}Scored", str(e["scored"]))
    M("scSignHeldPct", pct(s["by_type"]["sign"]["held_any_share_of_scored"]))
    M("scMidSign", str(s["by_era"]["076-081"]["by_type"]["sign"]["clauses"]))
    bt = s["by_era"][rev]["by_type"]
    M("scRevFailBand", str(bt["band"]["failed"])); M("scRevFailThr", str(bt["threshold"]["failed"]))
    M("scRevFailEq", str(bt["equality"]["failed"])); M("scRevFailSign", str(bt["sign"]["failed"]))
    M("scRevSignHeld", str(bt["sign"]["held"])); M("scRevSignHeldAny", str(bt["sign"]["held"] + bt["sign"]["held (point)"])); M("scRevSignClauses", str(bt["sign"]["clauses"]))
    M("scRevBandClauses", str(bt["band"]["clauses"]))
    M("scSignClauses", str(s["by_type"]["sign"]["clauses"]))


def law_frozen():
    """The frozen-constant law on the RTX 5090 hosts rented after the blind test (prereg/law_frozen_later.json)."""
    d = load("law_frozen_later.json")
    s = d["summary"]
    M("lawFrozenN", str(s["n"])); M("lawFrozenHosts", str(len({r["host"] for r in d["rows"]})))
    M("lawFrozenMed", f"{s['median_abs_pct']:.1f}"); M("lawFrozenPninety", f"{s['p90_abs_pct']:.1f}"); M("lawFrozenMax", f"{s['max_abs_pct']:.0f}")
    M("lawFrozenWithinTen", str(s["within_10pct"]))
    worst = max(d["rows"], key=lambda r: abs(r["error"]))
    hostname = {"089": "host S", "093": "oracle host O1", "094": "oracle host O2", "095": "oracle host O3"}[worst['host'].split(' ')[0]]
    M("lawFrozenWorstCell", f"{worst['model']} {worst['budget'].replace('%', chr(92) + '%')} on {hostname}")


def cost():
    """The ledger's rentals, distinct machines (Vast offers) and GPU-hour cost for jobs 058 up to the last finished job."""
    import re
    led = json.load(open(os.path.join(ROOT, "gpu", "vast_ledger.json")))
    led = led if isinstance(led, list) else led.get("instances", led)
    items = led if isinstance(led, list) else list(led.values())
    num = lambda j: int(re.match(r"(\d+)", j).group(1)) if re.match(r"(\d+)", j) else None  # noqa: E731
    done = [it for it in items if it.get("job") and num(it["job"]) and num(it["job"]) >= 58 and it.get("stop")]
    hi = max(num(it["job"]) for it in done)
    sel = [it for it in done if num(it["job"]) <= hi]
    M("costRentals", str(len(sel))); M("costMachines", str(len({it["offer"] for it in sel})))
    M("costUsd", f"{sum((it['stop'] - it['start']) / 3600 * it['dph'] for it in sel):.1f}"); M("costJobHi", f"{hi:03d}")


def crossrental():
    pts = load("ratio_points.json")
    r = {p["job"][:3]: p["ratio"] for p in pts if p["model"] == "gpt-oss-120b" and p["budget"] == "25%" and p["table"] == "fixed"}
    M("fixedRatioA", f"{r['076']:.3f}"); M("fixedRatioB", f"{r['077']:.3f}"); M("fixedRatioC", f"{r['081']:.3f}")
    # distinct hosts behind the points: one per Vast offer in the ledger (jobs 080 and 081 ran on the same host)
    led = json.load(open(os.path.join(ROOT, "gpu", "vast_ledger.json")))
    led = led if isinstance(led, list) else led.get("instances", led)
    offer = {}
    for it in (led if isinstance(led, list) else led.values()):
        if it.get("job") and it.get("offer"):
            offer[it["job"]] = it["offer"]
    jobs = {p["job"] for p in pts}
    hosts = {offer[j] for j in jobs if j in offer}
    assert len(hosts) == len([j for j in jobs if j in offer]) - 1, (jobs, hosts)   # 080 and 081 share one offer
    M("ratioHostsN", {7: "seven", 8: "eight", 9: "nine", 10: "ten"}.get(len(hosts), str(len(hosts))))
    M("ratioJobsN", str(len(jobs)))
    # the gpt-oss lead at the hosts whose CPU-to-link ratio is near 1 (x <= 1.15), the headline machine, and the slow link
    g = [p for p in pts if p["model"] == "gpt-oss-120b"]
    near = [p["ratio"] for p in g if p["x"] <= 1.15]
    M("ratioNearMin", f"{min(near):.2f}"); M("ratioNearMax", f"{max(near):.2f}")
    M("ratioNearXMin", f"{min(p['x'] for p in g if p['x'] <= 1.15):.2f}"); M("ratioNearXMax", f"{max(p['x'] for p in g if p['x'] <= 1.15):.2f}")
    M("ratioNearHostsN", {2: "two", 3: "three", 4: "four", 5: "five"}[len({p["job"] for p in g if p["x"] <= 1.15})])
    qq = [p for p in pts if p["model"] == "Qwen3-30B-A3B"]
    M("ratioQwenLo", f"{min(p['ratio'] for p in qq):.2f}"); M("ratioQwenHi", f"{max(p['ratio'] for p in qq):.2f}")
    M("ratioQwenHostsN", {4: "four", 5: "five", 6: "six"}[len({offer[p["job"]] for p in qq if p["job"] in offer})])
    M("ratioGptLawHostsN", {3: "three", 4: "four"}[len({offer[p["job"]] for p in g if p["table"] == "law" and p["job"] in offer})])
    M("ratioXB", f"{[p['x'] for p in g if p['job'].startswith('081')][0]:.2f}")
    M("ratioXSlow", f"{[p['x'] for p in g if p['job'].startswith('088')][0]:.2f}")
    M("ratioXStock", f"{[p['x'] for p in g if p['job'].startswith('089')][0]:.2f}")
    h = load("halfpcie_085.json")["means"]
    hy = [h[f"g_ft_hybrid_r{b} L1"] for b in ("0.111", "0.25", "0.40")]
    of = [h[f"g_ft_offload_r{b} L1"] for b in ("0.111", "0.25", "0.40")]
    M("iNineHybridMin", f"{min(hy):.0f}"); M("iNineHybridMax", f"{max(hy):.0f}")
    M("iNineOffloadMin", f"{min(of):.0f}"); M("iNineOffloadMax", f"{max(of):.0f}")


def slowlink():
    d = load("slowlink_088.json")
    rows = {r["budget"]: r for r in d["rows"]}
    rat = [rows[b]["ours_over_ft_L2"] for b in ("11%", "25%", "40%")]
    M("slFtMin", f"{min(r[0] for r in rat):.2f}"); M("slFtMax", f"{max(r[0] for r in rat):.2f}")
    M("slFtLoMin", f"{min(r[1] for r in rat):.2f}")
    for b, tag in (("11%", "Low"), ("25%", "Mid"), ("40%", "High")):
        r = rows[b]
        M(f"slFt{tag}", f"{r['ours_over_ft_L2'][0]:.2f}"); M(f"slFtVar{tag}", r["ft_variant"].replace("hybrid8", "hybrid"))
        M(f"slOurs{tag}", f"{r['ours']:.1f}"); M(f"slFtSpeed{tag}", f"{r['ft']:.1f}")
        M(f"slHyb{tag}", f"{r['launch1']['hybrid8']:.1f}"); M(f"slOff{tag}", f"{r['launch1']['offload']:.1f}")
        M(f"slFzero{tag}", f"{r['launch1']['hybrid8f0']:.1f}")
    hy = [rows[b]["ft_hybrid8_over_085_hybrid"] for b in ("11%", "25%", "40%")]
    M("slHybGainMin", f"{min(hy):.1f}"); M("slHybGainMax", f"{max(hy):.1f}")
    fb = [rows[b]["ft_best_over_085_best"] for b in ("11%", "25%", "40%")]
    M("slBestGainMin", f"{min(fb):.2f}"); M("slBestGainMax", f"{max(fb):.2f}")
    ll = d["llama"]
    M("slLlamaTeight", f"{ll['t8']:.1f}"); M("slLlamaTall", f"{ll['tall']:.1f}")
    M("slLlamaErrTeight", f"{100 * ll['err_t8']:+.1f}"); M("slLlamaErrTall", f"{100 * ll['err_tall']:+.1f}")
    M("slLlamaTratio", f"{ll['t8_over_tall'][0]:.2f}")
    M("slOursHostGain", f"{100 * (rows['11%']['ours'] / rows['11%']['job085']['ours'] - 1):.0f}")
    import re
    fl = d.get("ft_lines.txt", "")
    m = re.search(r"fetching ([\d.]+)% of each decode step", fl)
    M("slFetchFrac", m.group(1) if m else "--")
    M("slFtProbeCpu", f"{d['ft_probe']['cpu']:.1f}"); M("slFtProbePcie", f"{d['ft_probe']['pcie']:.1f}")


def stockclock():
    d = load("stockclock_089.json")
    rows = {(r["model"][:3], r["budget"]): r for r in d["rows"]}
    g = [rows[("gpt", b)] for b in ("11%", "25%", "40%")]
    q = [rows[("Qwe", b)] for b in ("12.5%", "25%", "43.75%")]
    M("scLeadGptMin", f"{100 * (min(r['ours_over_ft_L2'][0] for r in g) - 1):.0f}")
    M("scLeadGptMax", f"{100 * (max(r['ours_over_ft_L2'][0] for r in g) - 1):.0f}")
    def signed(v):   # a signed percentage with a real minus sign
        return f"${v:.0f}$" if v < 0 else f"+{v:.0f}"
    M("scLeadQwenMin", signed(100 * (min(r['ours_over_ft_L2'][0] for r in q) - 1)))
    M("scLeadQwenMax", signed(100 * (max(r['ours_over_ft_L2'][0] for r in q) - 1)))
    for r, tag in zip(g + q, ("GLow", "GMid", "GHigh", "QLow", "QMid", "QHigh")):
        M(f"scFt{tag}", f"{r['ours_over_ft_L2'][0]:.2f}")
        M(f"scFtLo{tag}", f"{r['ours_over_ft_L2'][1]:.3f}"); M(f"scFtHi{tag}", f"{r['ours_over_ft_L2'][2]:.3f}")
        M(f"scOurs{tag}", f"{r['ours']:.1f}"); M(f"scFtSpeed{tag}", f"{r['ft']:.1f}"); M(f"scLlama{tag}", f"{r['llama']:.1f}")
    allr = g + q
    M("scRatioDiffMin", f"{min(r['vs_b']['ratio_diff'] for r in allr):+.2f}"); M("scRatioDiffMax", f"{max(r['vs_b']['ratio_diff'] for r in allr):+.2f}")
    M("scOursDropMin", f"{100 * -max(r['vs_b']['ours'] for r in allr):.0f}"); M("scOursDropMax", f"{100 * -min(r['vs_b']['ours'] for r in allr):.0f}")
    M("scFtDropMin", f"{100 * -max(r['vs_b']['ft'] for r in allr):.0f}"); M("scFtDropMax", f"{100 * -min(r['vs_b']['ft'] for r in allr):.0f}")
    M("scLlamaDropMin", f"{100 * -max(r['vs_b']['llama'] for r in allr):.0f}"); M("scLlamaDropMax", f"{100 * -min(r['vs_b']['llama'] for r in allr):.0f}")
    M("scOrderMax", f"{100 * max(abs(r['order_diff']) for r in allr):.1f}")
    M("scLruSlowMin", f"{100 * (1 - max(r['lru_over_law'][0] for r in allr)):.0f}"); M("scLruSlowMax", f"{100 * (1 - min(r['lru_over_law'][0] for r in allr)):.0f}")
    M("scLruFtMin", f"{min(r['lru_over_ft'][0] for r in allr):.2f}"); M("scLruFtMax", f"{max(r['lru_over_ft'][0] for r in allr):.2f}")
    trail = [r["lru_over_ft"][0] for r in allr if r["lru_over_ft"][2] < 1]
    M("scLruTrailMin", f"{min(trail):.2f}"); M("scLruTrailMax", f"{max(trail):.2f}")
    # the cells where the +-0.06 band failed, and where the LRU 10-30% band failed
    rf = [abs(r["vs_b"]["ratio_diff"]) for r in allr if abs(r["vs_b"]["ratio_diff"]) > 0.06]
    M("scRatioFailMin", f"{min(rf):.2f}"); M("scRatioFailMax", f"{max(rf):.2f}")
    M("scRatioFailCells", {3: "three", 4: "four", 5: "five", 6: "six"}[len(rf)])
    lf = [100 * (1 - r["lru_over_law"][0]) for r in allr if not 10 <= 100 * (1 - r["lru_over_law"][0]) <= 30]
    M("scLruFailMin", f"{min(lf):.0f}"); M("scLruFailMax", f"{max(lf):.0f}")
    M("scLruFtGLow", f"{rows[('gpt', '11%')]['lru_over_ft'][0]:.2f}"); M("scLruFtGMid", f"{rows[('gpt', '25%')]['lru_over_ft'][0]:.2f}")
    M("scLruFtQLow", f"{rows[('Qwe', '12.5%')]['lru_over_ft'][0]:.2f}")
    M("scLlamaXMin", f"{min(r['ours_over_llama'][0] for r in allr):.1f}"); M("scLlamaXMax", f"{max(r['ours_over_llama'][0] for r in allr):.1f}")
    # the two hosts: the law's own rates (B_c at the helper count, zero-copy B_p, median B_cp) and tables, and the cards
    ha, hb = d["hosts"]["089 (9950X, stock clock)"], d["hosts"]["listing B (081)"]
    M("scCpuBw", f"{ha['B_c']:.0f}"); M("scCpuBwB", f"{hb['B_c']:.0f}")
    M("scPcieBw", f"{ha['B_p']:.0f}"); M("scPcieBwB", f"{hb['B_p']:.0f}")
    M("scBothBw", f"{ha['B_both']:.0f}"); M("scBothBwB", f"{hb['B_both']:.0f}")
    M("scCpuBwDrop", f"{100 * -d['hosts']['089_over_B']['B_c']:.0f}"); M("scBothBwDrop", f"{100 * -d['hosts']['089_over_B']['B_both']:.0f}")
    M("scPowerW", f"{ha['power_limit_w']:.0f}"); M("scPowerWB", f"{hb['power_limit_w']:.0f}")
    M("scMemClk", str(ha["mem_clock_mhz"])); M("scMemClkB", str(hb["mem_clock_mhz"]))
    M("scTableGptB", ",".join(str(x) for x in hb["table_gpt"])); M("scTableQwenB", ",".join(str(x) for x in hb["table_qwen"]))
    # cells led with the interval rule, and the drops split by FreeToken's backend (offload runs no expert on the CPU)
    ld = d["leads"]
    W = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    M("scOursLeadCells", W[len(ld["ours_leads_ft"])]); M("scFtLeadCells", W[len(ld["ft_leads_ours"])])
    M("scLruTrailCells", W[len(ld["lru_trails_ft"])]); M("scLruLeadCells", W[len(ld["lru_leads_ft"])])
    M("scRatioDropMin", f"{min(abs(r['vs_b']['ratio_diff']) for r in allr):.2f}"); M("scRatioDropMax", f"{max(abs(r['vs_b']['ratio_diff']) for r in allr):.2f}")
    off = [r for r in allr if r["ft_variant"] == "offload"]; hyb = [r for r in allr if r["ft_variant"] == "hybrid"]
    M("scFtOffDropMin", f"{100 * -max(r['vs_b']['ft'] for r in off):.0f}"); M("scFtOffDropMax", f"{100 * -min(r['vs_b']['ft'] for r in off):.0f}")
    M("scFtHybDropMin", f"{100 * -max(r['vs_b']['ft'] for r in hyb):.0f}"); M("scFtHybDropMax", f"{100 * -min(r['vs_b']['ft'] for r in hyb):.0f}")
    M("scOursOffDropMin", f"{100 * -max(r['vs_b']['ours'] for r in off):.0f}"); M("scOursOffDropMax", f"{100 * -min(r['vs_b']['ours'] for r in off):.0f}")
    M("scOursHybDropMin", f"{100 * -max(r['vs_b']['ours'] for r in hyb):.0f}"); M("scOursHybDropMax", f"{100 * -min(r['vs_b']['ours'] for r in hyb):.0f}")
    fr = [r["limit"]["frac"]["ours"] for r in allr]
    M("scLimFracMin", pct(min(fr))); M("scLimFracMax", pct(max(fr)))
    hb = [r["limit"]["headline"]["frac"]["ours"] for r in allr]
    M("scLimFracBMin", pct(min(hb))); M("scLimFracBMax", pct(max(hb)))
    M("scLimHostBw", f"{d['b_host_max_gbs']:.1f}")
    hostb = [r for r in allr if r["budget"] in ("11%", "25%", "12.5%")]
    ld_ = [1 - r["limit"]["tok_s"] / r["limit"]["headline"]["tok_s"] for r in hostb]
    M("scLimDropMin", pct(min(ld_))); M("scLimDropMax", pct(max(ld_)))
    gsh = [ld["policy_share_of_lead"][f"gpt-oss-120b {b}"] for b in ("25%", "40%")]
    M("scPolicyShareGptMin", f"{100 * min(gsh):.0f}"); M("scPolicyShareGptMax", f"{100 * max(gsh):.0f}")
    bb = d["batched_bench_ratios"]
    pp = [v["pp_ratio"] for m in bb.values() for v in m.values()]
    tg = [v["tg_ratio"] for m in bb.values() for k, v in m.items() if not k.endswith("pl1")]
    tg1 = [v["tg_ratio"] for m in bb.values() for k, v in m.items() if k.endswith("pl1")]
    M("bbPrefillMin", pct(min(pp))); M("bbPrefillMax", pct(max(pp)))
    M("bbBatchMin", pct(min(tg))); M("bbBatchMax", pct(max(tg)))
    M("bbSingleMin", f"{min(tg1):.1f}"); M("bbSingleMax", f"{max(tg1):.1f}")
    import json as _j
    tg_ = _j.loads(d["fetch_table_law_gptoss.json"])["table"]; tq_ = _j.loads(d["fetch_table_law_qwen3.json"])["table"]
    M("scTableGpt", ",".join(str(x) for x in tg_)); M("scTableQwen", ",".join(str(x) for x in tq_))


def parity():
    d = load("parity_090.json")
    for m, nm in (("g", "Gpt"), ("q", "Qwen")):
        sy = d["models"][m]["systems"]
        o, st = sy["ours"], sy["n27" if m == "g" else "n36"]
        M(f"klMean{nm}", f"{o['kl_nats']['mean']:.4f}"); M(f"klNinetyNine{nm}", f"{o['kl_nats']['p999']:.3f}")
        M(f"klMax{nm}", f"{o['kl_nats']['max']:.2f}"); M(f"klTop{nm}", f"{100 * o['top1_agreement']:.1f}")
        M(f"klNll{nm}", f"{o['dnll_pct']:+.2f}"); M(f"klStockMean{nm}", f"{st['kl_nats']['mean']:.4f}")
        M(f"klStockTop{nm}", f"{100 * st['top1_agreement']:.1f}"); M(f"klSteps{nm}", str(o["n_steps"]))
        M(f"klAboveHundredth{nm}", f"{100 * o['kl_gt_0.01_frac']:.1f}")
    M("klVocabGpt", f"{d['models']['g']['systems']['ours']['n_vocab']:,}")


def clocks():
    def memclk(job):
        with open(f"{RES}/{job}/gpu.csv") as f:
            rows = list(csv.reader(f))
        hdr = [h.strip() for h in rows[0]]
        i = hdr.index("clocks.max.memory [MHz]")
        return rows[1][i].strip().split()[0]
    M("memClkHeadline", memclk("081_headline_law@vast"))
    M("memClkOther", memclk("087_ablation_static@vast"))
    import glob
    others = {}
    for path in sorted(glob.glob(f"{RES}/0[6-8]*/gpu.csv")):
        job = os.path.basename(os.path.dirname(path))
        if job.startswith(("080", "081", "082")):
            continue
        with open(path) as f:
            rows = list(csv.reader(f))
        hdr = [h.strip() for h in rows[0]]
        if "clocks.max.memory [MHz]" not in hdr or len(rows) < 2 or "5090" not in rows[1][0]:
            continue
        clk = rows[1][hdr.index("clocks.max.memory [MHz]")].strip().split()[0]
        others.setdefault(clk, []).append(job)
    M("memClkOthers", ", ".join(sorted(others)))
    M("memClkOtherRentals", str(sum(len(v) for v in others.values())))
    common = max(others, key=lambda k: len(others[k]))
    M("memClkCommon", common); M("memClkCommonN", str(len(others[common])))
    odd = {k: v for k, v in others.items() if k != common}
    M("memClkOdd", "; ".join(f"{k} ({len(v)})" for k, v in sorted(odd.items())) or "none")
    M("memClkRatio", f"{int(memclk('081_headline_law@vast')) / int(memclk('087_ablation_static@vast')):.2f}")


def main():
    perlayer()
    limit()
    shapley()
    policy()
    foresight()
    batchk()
    audit()
    scorecard()
    crossrental()
    law_frozen()
    cost()
    slowlink()
    stockclock()
    parity()
    clocks()
    with open(P("paper", "wsg_numbers2.tex"), "w") as f:
        f.write("% generated by scripts/wsg_numbers2.py; do not edit\n")
        for k, v in sorted(macros.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    for k, v in sorted(macros.items()):
        print(f"{k:22s} {v}")


if __name__ == "__main__":
    main()

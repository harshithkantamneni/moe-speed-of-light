"""Emit every number and generated table of the speed-of-light paper (paper/paper.tex) from the committed outputs.

    python scripts/paper_numbers4.py --results /home/claude/gpu-branch/results --out paper

Inputs (all in git): data/audit/*, prereg/{audit,audit_sens_a10,provenance,provenance_ext,harmony,a10_windows_038,
global,bound,anchors}/, prereg/provenance/exactness_gpu_vs_cpu.txt, the job-038 control runs, scripts/a10_rows.py.
Writes paper/numbers4.tex, paper/table_{prov,a10,audit,audit_full,contract}.tex, paper/prereg_log.tex and
paper/numbers4.json (the same values, for the fact-check).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

A10 = ["gpt-oss-20b-MXFP4", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M", "Qwen3-30B-A3B-Instruct-2507-Q8_0", "gpt-oss-120b-MXFP4"]
A10_SHORT = {"gpt-oss-20b-MXFP4": "gpt-oss-20b", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M": "Qwen3-30B Q4\\_K\\_M",
             "Qwen3-30B-A3B-Instruct-2507-Q8_0": "Qwen3-30B Q8\\_0", "gpt-oss-120b-MXFP4": "gpt-oss-120b"}
PROV_NAMES = {"olmoe": "OLMoE-1B-7B", "gpt-oss-20b": "gpt-oss-20b", "qwen3-30b-a3b": "Qwen3-30B-A3B",
              "gpt-oss-120b": "gpt-oss-120b", "mixtral-8x7b": "Mixtral-8x7B", "deepseek-v2-lite": "DeepSeek-V2-Lite",
              "qwen1.5-moe": "Qwen1.5-MoE-A2.7B", "qwen2-57b": "Qwen2-57B-A14B", "phi3.5-moe": "Phi-3.5-MoE"}
REGISTERED = ["olmoe", "gpt-oss-20b", "qwen3-30b-a3b", "gpt-oss-120b"]
EXTENSION = ["mixtral-8x7b", "deepseek-v2-lite", "qwen1.5-moe", "qwen2-57b", "phi3.5-moe"]


def tex(s):
    s = str(s)
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_"), ("$", "\\$"),
                 ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}")):
        s = s.replace(a, b)
    return s.replace("\\textbackslash\\{\\}", "\\textbackslash{}")


def rate(rows):
    return sum(r["n_decode"] for r in rows) / (sum(r["decode_ms"] for r in rows) / 1000)


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def short_system(s):
    s = re.sub(r"\s*\(.*$", "", s).strip()
    s = s.replace("llama.cpp GPU-resident LRU expert cache PR #27861", "llama.cpp PR 27861")
    return s if len(s) <= 26 else s[:25] + "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="paper")
    a = ap.parse_args()
    N, T = {}, {}   # macros; tables

    # ---------------- audit ----------------
    raw = [r for f in sorted(glob.glob("data/audit/rows_group*.jsonl")) for r in jl(f)]
    community = lambda r: r.get("venue", "").startswith("llama.cpp PR")
    N["auditRows"] = len(raw)
    N["auditSystems"] = len({r["system"] for r in raw})
    N["auditPaperSystems"] = len({r["system"] for r in raw if not community(r)})
    N["auditCommunitySystems"] = len({r["system"] for r in raw if community(r)})
    N["auditNormalized"] = len(jl("data/audit/normalized.jsonl"))
    ver = [r for b in (1, 2, 3) for r in jl(f"data/audit/verify_batch{b}.jsonl")]
    N["auditChecked"] = len(ver)
    N["auditCorrected"] = sum(v["verdict"] == "corrections" for v in ver)
    A = json.load(open("prereg/audit/audit.json"))
    S_ = A["summary"]
    adj = [r for r in A["rows"] if not any(l.startswith("not adjudicated") for l in r["labels"])]
    N["auditModelled"] = S_["n_rows"]
    N["auditAdj"] = S_["n_adjudicated"]
    N["auditWeak"] = S_["weak_baselines"]
    N["auditLlamaRows"] = S_["llama_rows"]
    N["auditSurvive"] = S_["gain_survives"]
    N["auditMedianSol"] = round(100 * S_["median_system_of_sol"])
    N["auditBestSol"] = round(100 * max(r["system_of_sol"] for r in adj))
    N["auditWorstSol"] = round(100 * min(r["system_of_sol"] for r in adj))
    N["auditTraced"] = sum("trace" in r["mstar_source"] for r in A["rows"])
    N["auditBandLo"] = f"{S_['band_q10_q90'][0]:.2f}"
    N["auditBandHi"] = f"{S_['band_q10_q90'][1]:.2f}"
    wk = [r["reported_baseline"]["tok_s"] / r["pred_baseline_tok_s"][1] for r in adj
          if "weak baseline" in r["labels"]]
    N["auditWeakMedianRatio"] = f"{np.median(wk):.2f}" if wk else "--"
    N["auditWeakHalfOrLess"] = sum(x <= 0.5 for x in wk)
    gl = [r["system_tok_s"] / r["sol_global_tok_s"] for r in adj if r.get("sol_global_tok_s")]
    N["auditGlobalN"] = len(gl)
    sens = json.load(open("prereg/audit_sens_a10/audit.json"))["summary"]
    N["auditSensSurvive"] = sens["gain_survives"]
    N["auditSensWeak"] = sens["weak_baselines"]
    N["auditSensAdj"] = sens["n_adjudicated"]
    N["auditSensLlama"] = sens["llama_rows"]
    N["auditSkipped"] = S_["n_skipped"]

    # ---------------- provenance ----------------
    P = json.load(open("prereg/provenance/provenance.json"))
    PX = json.load(open("prereg/provenance_ext/provenance.json")) if os.path.exists("prereg/provenance_ext/provenance.json") else {}
    G = json.load(open("prereg/global/global.json"))
    allp = {**P, **PX}
    d = lambda k, m: allp[k]["diffs"]["S-D"][m]
    reg = [k for k in REGISTERED if k in P]
    ext = [k for k in EXTENSION if k in PX and "S-D" in PX[k]["diffs"]]
    N["provNModels"] = len(reg)
    N["provNExt"] = len(ext)
    N["provNAll"] = len(reg) + len(ext)
    N["provDropMin"] = f"{-100 * max(d(k, 'hit_DFA_1')[0] for k in reg):.1f}"
    N["provDropMax"] = f"{-100 * min(d(k, 'hit_DFA_1')[0] for k in reg):.1f}"
    N["provMinDropMin"] = f"{-100 * max(d(k, 'hit_MIN-bypass_1')[0] for k in reg):.1f}"
    N["provMinDropMax"] = f"{-100 * min(d(k, 'hit_MIN-bypass_1')[0] for k in reg):.1f}"
    N["provHalfMax"] = f"{100 * max(abs(d(k, 'hit_DFA_4')[0]) for k in reg):.1f}"
    N["provAllCIExclude"] = "yes" if all(d(k, "hit_DFA_1")[2] < 0 and d(k, "hit_MIN-bypass_1")[2] < 0 for k in reg) else "no"
    if ext:
        xs = [d(k, "hit_DFA_1")[0] for k in ext]
        N["provExtDropMin"] = f"{-100 * max(xs):.1f}"
        N["provExtDropMax"] = f"{-100 * min(xs):.1f}"
        N["provExtNeg"] = sum(x < 0 for x in xs)
        N["provExtNegCI"] = sum(d(k, "hit_DFA_1")[2] < 0 for k in ext)
        ms = [d(k, "hit_MIN-bypass_1")[0] for k in ext]
        N["provExtMinDropMin"] = f"{-100 * max(ms):.1f}"
        N["provExtMinDropMax"] = f"{-100 * min(ms):.1f}"
    N["provPTwoFail"] = sum(not P[k]["predictions"].get("P2", True) for k in reg)
    N["provFlags"] = sum(bool(P[k]["flags"].get("F1")) or P[k]["flags"].get("F2") or P[k]["flags"].get("F3_fires", False) for k in reg)
    N["provNArch"] = 7
    toks = 0
    for key in list(P) + list(PX):
        for arm in "DGS":
            n = allp[key]["arms"].get(arm, {}).get("decode_steps")
            toks += n or 0
    N["provNTokensM"] = f"{toks / 1e6:.2f}"
    N["provNTraceModels"] = len(set(P) | set(PX))
    # F4: the mailbox cache's fraction of the bound, S vs D arm, same instance
    solD, solS = (json.load(open(f"prereg/a10_windows_038/sol_{x}.json")) for x in "DS")
    f4 = [abs(solS[t]["budgets"][q]["mailbox_of_sol"] - solD[t]["budgets"][q]["mailbox_of_sol"])
          for t in solD for q in solD[t]["budgets"] if t in solS and q in solS[t]["budgets"]]
    N["provFFour"] = f"{100 * max(f4):.1f}"
    gr = [G[k]["q1"]["reduction"] for k in G]
    N["globalGainMin"] = f"{100 * min(gr):.1f}"
    N["globalGainMax"] = f"{100 * max(gr):.1f}"
    N["globalNModels"] = len(G)
    txt = open("prereg/bound/test_bound_output.txt").read()
    N["boundTight"] = re.search(r"bound holds; tightest policy reaches ([\d.]+)%", txt).group(1)
    N["boundTightGlobal"] = re.search(r"global bound holds; tightest policy reaches ([\d.]+)%", txt).group(1)
    ex = [float(x) for x in re.findall(r"differing selections ([\d.]+)%", open("prereg/provenance/exactness_gpu_vs_cpu.txt").read())]
    N["gpuCpuMaxPct"] = f"{max(ex):.2f}"

    # ---------------- 120b ----------------
    H = json.load(open("prereg/harmony/harmony.json"))["summary"]
    b = H["gpt-oss-120b"]
    N["oneTwentyVz"] = f"{b['resp']:.2f}"
    N["oneTwentyCol"] = f"{b['collector_resp']:.2f}"
    N["oneTwentyVone"] = f"{b['v1_resp']:.2f}"
    N["oneTwentyVtwo"] = f"{b['v2_resp']:.2f}"
    N["oneTwentyImpl"] = f"{b['impl_absdiff_mean']:.2f}"
    N["oneTwentyRatioVz"] = f"{H['ratios']['resp_V0']:.2f}"
    N["oneTwentyRatioVone"] = f"{H['ratios']['resp_V1_same']:.2f}"
    N["oneTwentyN"] = b["n_conv"]
    scD = json.load(open("prereg/a10_windows_038/scored_D.json"))
    scS = json.load(open("prereg/a10_windows_038/scored_S.json"))
    N["oneTwentyLlama"] = f"{scD['results']['gpt-oss-120b-MXFP4']['budgets']['1']['llama.cpp static layers']['nll']:.2f}"
    N["oneTwentyLlamaS"] = f"{scS['results']['gpt-oss-120b-MXFP4']['budgets']['1']['llama.cpp static layers']['nll']:.2f}"

    # ---------------- validation / A10 ----------------
    N["hSeventeenMed"] = f"{100 * max(scD['H17']['median_ape'], scS['H17']['median_ape']):.1f}"
    N["hSeventeenN"] = scD["H17"]["n"]
    N["hSeventeenMax"] = f"{100 * max(scD['H17']['max_ape'], scS['H17']['max_ape']):.1f}"
    from scripts.a10_rows import A10_ROWS
    from scripts.validate import VARIANTS, fit, predict
    from mosl.validation_set import ROWS
    p4 = fit(ROWS, VARIANTS["M4 M1 + per-CPU-expert latency"])
    under = [r["tok_s"] / predict(r, p4)[0] - 1 for r in A10_ROWS]
    N["underMin"] = round(100 * min(under))
    N["underMax"] = round(100 * max(under))
    sp = [bb["speedup"] for sc in (scD, scS) for t in sc["results"].values() for bb in t["budgets"].values() if "speedup" in bb]
    N["caseMin"] = f"{min(sp):.2f}"
    N["caseMax"] = f"{max(sp):.2f}"
    mbs = [x[t]["budgets"][q][f] for x in (solD, solS) for t in x for q in x[t]["budgets"] for f in ("mailbox_of_sol",)]
    lls = [x[t]["budgets"][q]["llama_of_sol"] for x in (solD, solS) for t in x for q in x[t]["budgets"]]
    N["caseMbSolMin"], N["caseMbSolMax"] = round(100 * min(mbs)), round(100 * max(mbs))
    N["caseLlamaSolMin"], N["caseLlamaSolMax"] = round(100 * min(lls)), round(100 * max(lls))
    # same-instance control with the phase-3 windows (job 038) against the phase-3 instance
    from scripts.mb_model import MODELS as MB
    ctrl, llam = {}, {}
    d038 = glob.glob(os.path.join(a.results, "038_*"))[0]
    for tag in A10:
        L = MB[tag][0]
        n25 = L - L * 2 // 8
        c = jl(os.path.join(d038, f"{tag}_ctrl_ec.jsonl"))
        s = jl(os.path.join(d038, f"{tag}_ctrl_static_n{n25}.jsonl"))
        ph = scD["results"][tag]["budgets"]["2"]["phase3"]
        ctrl[tag] = (rate(c) / rate(s), ph["mailbox cache"] / ph["llama.cpp static layers"])
        llam[tag] = rate(s) / ph["llama.cpp static layers"] - 1
    N["caseCtrlSmallThis"], N["caseCtrlSmallPhase"] = (f"{x:.2f}" for x in ctrl["gpt-oss-20b-MXFP4"])
    N["caseCtrlBigThis"], N["caseCtrlBigPhase"] = (f"{x:.2f}" for x in ctrl["gpt-oss-120b-MXFP4"])
    N["instLlamaMax"] = round(100 * max(llam.values()))
    N["instSpeedupMax"] = f"{max(abs(x - y) for x, y in ctrl.values()):.2f}"
    bw = [float(x) for x in re.findall(r"read\s+n=\d+ threads=30:\s+([\d.]+) GB/s", open(os.path.join(d038, "cpubench.txt")).read())]
    from scripts.sol_a10 import B_C
    N["bwThis"], N["bwPhase"] = f"{bw[0]:.0f}", f"{B_C / 1e9:.0f}"
    N["bwDiffPct"] = round(100 * (bw[0] / (B_C / 1e9) - 1))
    win = {tag: scD["results"][tag]["budgets"]["2"]["speedup"] - ctrl[tag][0] for tag in A10}
    N["winEffectMax"] = f"{max(abs(v) for v in win.values()):.2f}"
    # share of phase-2/3 "decode" steps inside the user prompt (prefill capped at 128, 192 steps)
    from mosl.ecsim import decode_windows
    for tag, key in (("gpt-oss-20b-MXFP4", "winGptPct"), ("Qwen3-30B-A3B-Instruct-2507-Q4_K_M", "winQwenPct")):
        rows = jl(MB[tag][6])
        w = decode_windows(rows, MB[tag][7], 128, 192)
        inside = sum(max(0, min(b_, rows[si]["prompt_len"]) - a_) for si, a_, b_ in w)
        N[key] = round(100 * inside / sum(b_ - a_ for _, a_, b_ in w))

    # ---------------- anchors ----------------
    anc = {}
    for nm in ("a100", "gh200"):
        pth = f"prereg/anchors/{nm}.json"
        if os.path.exists(pth):
            anc[nm] = json.load(open(pth))
    N["anchorParagraph"] = anchor_paragraph(anc)
    for nm, x in anc.items():
        s = x["summary"]
        up = nm.capitalize()
        N[f"anchor{up}N"] = s["n"]
        N[f"anchor{up}Med"] = f"{100 * s['median_ape_m4']:.0f}"

    # ---------------- write macros ----------------
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "numbers4.tex"), "w") as f:
        f.write("% generated by scripts/paper_numbers4.py; do not edit\n")
        for k, v in N.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    json.dump(N, open(os.path.join(a.out, "numbers4.json"), "w"), indent=1, default=str)

    # ---------------- table: provenance ----------------
    L = ["\\begin{table*}[t]\\centering\\small",
         "\\caption{Trace provenance: own sampled text (S) against dataset text (D), response tokens only, paired bootstrap 95\\% CIs over prompts. "
         "Hit rates in points at 12.5\\% / 25\\% of experts; DFA is our deployed admission policy, MIN-bypass the oracle. "
         "$\\Delta M^\\star_{\\mathrm{glob}}$: reduction of MIN-bypass misses (S) when the per-layer budget is pooled over all layers, at 12.5\\%. "
         "Rows below the rule extend the registered study to the audit's models.}\\label{tab:prov}",
         "\\begin{tabular}{lrrrrrrrl}\\toprule",
         "model & $E$/$k$ & NLL D & NLL S & $\\Delta$DFA 12.5\\% & $\\Delta$DFA 25\\% & $\\Delta$MIN-b. 12.5\\% & $\\Delta M^\\star_{\\mathrm{glob}}$ & F1--F3 \\\\ \\midrule"]
    ci = lambda t: f"{100 * t[0]:+.1f} [{100 * t[1]:+.1f}, {100 * t[2]:+.1f}]"
    for grp in (reg, ext):
        for k in grp:
            r = allp[k]
            fl = r.get("flags", {})
            fires = [n for n, v in (("F1", bool(fl.get("F1"))), ("F2", fl.get("F2")), ("F3", fl.get("F3_fires", False))) if v]
            nd, ns = r["arms"]["D"].get("nll"), r["arms"]["S"].get("nll")
            L.append(f"{PROV_NAMES[k]} & {r['E']}/{r['k']} & {nd:.2f} & {ns:.2f} & {ci(d(k, 'hit_DFA_1'))} & {ci(d(k, 'hit_DFA_2'))} & "
                     f"{ci(d(k, 'hit_MIN-bypass_1'))} & {100 * G[k]['q1']['reduction']:.1f}\\% & {', '.join(fires) or 'none'} \\\\"
                     if k in G else "")
        if grp is reg and ext:
            L.append("\\midrule")
    L += ["\\bottomrule\\end{tabular}\\end{table*}"]
    open(os.path.join(a.out, "table_prov.tex"), "w").write("\n".join(x for x in L if x) + "\n")

    # ---------------- table: A10 case study ----------------
    L = ["\\begin{table}[t]\\centering\\footnotesize\\setlength\\tabcolsep{3pt}",
         "\\caption{Our cache (``mailbox'') against llama.cpp \\texttt{--n-cpu-moe} at equal expert memory on one A10 instance, "
         "response windows. tok/s on dataset text (D); speed-ups on D and on the models' own text (S); fractions of the "
         "implementation-relative speed-of-light where an all-GPU run exists.}\\label{tab:a10}",
         "\\begin{tabular}{llrrrrrr}\\toprule",
         "model & budget & llama & cache & $\\times$ D & $\\times$ S & cache/SoL & llama/SoL \\\\ \\midrule"]
    for tag in A10:
        for q, bb in scD["results"][tag]["budgets"].items():
            bs = scS["results"][tag]["budgets"].get(q, {})
            so = solD.get(tag, {}).get("budgets", {}).get(q)
            L.append(f"{A10_SHORT[tag]} & {100 * int(q) / 8:.1f}\\% & {bb['llama.cpp static layers']['tok_s']:.1f} & {bb['mailbox cache']['tok_s']:.1f} & "
                     f"{bb['speedup']:.2f} & {bs.get('speedup', float('nan')):.2f} & "
                     + (f"{100 * so['mailbox_of_sol']:.0f}\\% & {100 * so['llama_of_sol']:.0f}\\%" if so else "-- & --") + " \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    open(os.path.join(a.out, "table_a10.tex"), "w").write("\n".join(L) + "\n")

    # ---------------- tables: audit ----------------
    lab = lambda r: "; ".join({"weak baseline": "weak", "at strength": "at strength", "baseline above prediction": "above pred.",
                               "gain survives": "survives", "not established": "not est."}.get(x, x) for x in r["labels"])
    rb = lambda r: f"{r['reported_baseline']['tok_s']:.1f}" if r["reported_baseline"] else "--"
    islc = lambda r: bool(r["reported_baseline"]) and r["reported_baseline"].get("class", "").startswith("llama.cpp")
    L = ["\\begin{table*}[t]\\centering\\scriptsize\\setlength\\tabcolsep{3.5pt}",
         f"\\caption{{The {N['auditAdj']} adjudicated rows. Claimed: system over its strongest reported baseline (llama.cpp where one is reported, marked $^\\ell$). "
         "Predicted: equal-memory llama.cpp \\texttt{--n-cpu-moe} [band]. $S_n$: system over predicted. SoL: physical speed-of-light, $M^\\star$ from our traces (t) "
         "or independent routing (i). All rows in \\cref{app:audit}.}\\label{tab:audit}",
         "\\begin{tabular}{lllrrrrrrl}\\toprule",
         "system & model & GPU & sys.\\ tok/s & reported base & claimed & predicted [band] & $S_n$ [band] & sys./SoL & label \\\\ \\midrule"]
    for r in sorted(adj, key=lambda r: (r["system"], r["model"] or "")):
        L.append(f"{tex(short_system(r['system']))} & {tex((r['model'] or '')[:22])} & {tex(r['gpu'][:16])} & {r['system_tok_s']:.1f} & "
                 f"{rb(r)}{'$^\\ell$' if islc(r) else ''} & {r['claimed_speedup']:.2f} & "
                 f"{r['pred_baseline_tok_s'][1]:.1f} [{r['pred_baseline_tok_s'][0]:.1f}, {r['pred_baseline_tok_s'][2]:.1f}] & "
                 f"{r['normalized_speedup'][1]:.2f} [{r['normalized_speedup'][0]:.2f}, {r['normalized_speedup'][2]:.2f}] & "
                 f"{100 * r['system_of_sol']:.0f}\\% ({'t' if 'trace' in r['mstar_source'] else 'i'}) & {lab(r)} \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table*}"]
    open(os.path.join(a.out, "table_audit.tex"), "w").write("\n".join(L) + "\n")

    L = ["{\\scriptsize\\setlength\\tabcolsep{3pt}",
         "\\begin{longtable}{p{2.6cm}p{2.3cm}p{1.6cm}rrrrrrrp{2.4cm}}",
         f"\\caption{{All {N['auditModelled']} modelled rows (of {N['auditNormalized']} normalized, after the re-check). Budget: GPU bytes for routed experts "
         "($n_{\\mathrm{cpu}}/L$ MoE layers on the CPU in the predicted baseline; $^\\ast$ imputed upper bound). Other columns as in \\cref{tab:audit}; "
         "SoL$_g$: the global-budget variant.}\\label{tab:auditfull}\\\\",
         "\\toprule system & model & GPU & budget GB ($n/L$) & sys. & base & claim & pred. & $S_n$ & SoL (SoL$_g$) & label \\\\ \\midrule\\endfirsthead",
         "\\toprule system & model & GPU & budget GB ($n/L$) & sys. & base & claim & pred. & $S_n$ & SoL (SoL$_g$) & label \\\\ \\midrule\\endhead",
         "\\bottomrule\\endfoot"]
    for r in sorted(A["rows"], key=lambda r: (r["system"], r["model"] or "")):
        cl = f"{r['claimed_speedup']:.2f}" if r["claimed_speedup"] else "--"
        sg = f" ({r['sol_global_tok_s']:.1f})" if r.get("sol_global_tok_s") else ""
        L.append(f"{tex(short_system(r['system']))} & {tex((r['model'] or '')[:24])} & {tex(r['gpu'][:18])} & "
                 f"{r['budget_gb']:.1f}{'$^\\ast$' if r['budget_imputed'] else ''} ({r['n_cpu_layers']}/{r['L']}) & {r['system_tok_s']:.1f} & "
                 f"{rb(r)}{'$^\\ell$' if islc(r) else ''} & {cl} & {r['pred_baseline_tok_s'][1]:.1f} & {r['normalized_speedup'][1]:.2f} & "
                 f"{r['sol_tok_s']:.1f}{sg} {'t' if 'trace' in r['mstar_source'] else 'i'} & {tex(lab(r).replace('not adjudicated: ', 'n.a.: '))} \\\\")
    L += ["\\end{longtable}}", "Not modelled: " + "; ".join(f"\\texttt{{{tex(x['id'])}}} ({tex(x['reason'])})" for x in A["skipped"]) + "."]
    open(os.path.join(a.out, "table_audit_full.tex"), "w").write("\n".join(L) + "\n")

    # ---------------- figure data for the audit figure ----------------
    json.dump([dict(claimed=r["claimed_speedup"], sn=r["normalized_speedup"], labels=r["labels"], llama=islc(r),
                    system=r["system"]) for r in adj], open(os.path.join(a.out, "figs", "audit_points.json"), "w"), indent=1)

    # ---------------- table: contract ----------------
    C = [("Host", "CPU, DRAM type/channels/speed, and STREAM Triad (or read) bandwidth at the thread count used", "$B_c$"),
         ("GPU and link", "GPU model and clocks; measured device-read bandwidth; PCIe generation/width and measured pinned H2D", "$B_g$, $B_p$"),
         ("Bytes", "Model, exact quantisation per tensor class; dense bytes per token and bytes per expert (e.g.\\ from the GGUF header)", "$D$, $X$"),
         ("Budget", "GPU bytes given to routed experts, per layer or pooled, and what else shares GPU memory", "$C$"),
         ("Text", "Evaluated prompts and answers; whether the answer is the model's own; full prompt prefilled; decode = answer tokens", "trace"),
         ("Routing trace", "The routing of the evaluated text, released, so $M^\\star$ can be recomputed", "$M^\\star$"),
         ("Strong baseline", "llama.cpp commit and full command line, with \\texttt{--n-cpu-moe} chosen to fill the same GPU memory", "--"),
         ("Result", "Decode tok/s (single request), repeats and spread, and the fraction of the speed-of-light it implies", "$\\bar T$")]
    L = ["\\begin{table}[t]\\centering\\footnotesize",
         "\\caption{A reporting contract in seconds: the fields that make a batch-1 offloading result checkable against the bound and a strong baseline.}\\label{tab:contract}",
         "\\begin{tabular}{p{1.6cm}p{5.0cm}l}\\toprule field & report & feeds \\\\ \\midrule"]
    L += [f"{x} & {y} & {z} \\\\" for x, y, z in C]
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    open(os.path.join(a.out, "table_contract.tex"), "w").write("\n".join(L) + "\n")

    # ---------------- pre-registration log ----------------
    open(os.path.join(a.out, "prereg_log.tex"), "w").write(prereg_log(N, anc) + "\n")
    print(json.dumps(N, indent=1, default=str))


def anchor_paragraph(anc):
    if not anc:
        return "Anchor runs are pending."
    parts = []
    if "a100" in anc:
        s = anc["a100"]["summary"]
        parts.append(
            f"On an A100-SXM4-40GB x86 VM (STREAM Triad {s['platform']['bw_cpu']:.0f}\\,GB/s) we registered predictions for "
            f"{s['n']} llama.cpp \\texttt{{--n-cpu-moe}} configurations of six GGUF models (Mixtral-8x7B Q4\\_K\\_M and Q8\\_0, Phi-3.5-MoE, "
            f"Qwen2-57B-A14B, DeepSeek-V2-Lite, Qwen3-30B-A3B), committed them, then measured. The model's median error is "
            f"{100 * s['median_ape_m4']:.0f}\\% (registered: $\\le$25\\%; {'holds' if s['A2'] else 'fails'}), it under-predicts "
            f"{s['under_predicted']} of {s['n']} ({'as registered' if s['A3'] else 'fewer than the registered two thirds'}), measured/predicted "
            f"{s['ratio_range'][0]:.2f}--{s['ratio_range'][1]:.2f}; no configuration exceeds the physical floor"
            + (f" (the closest reaches {100 * s['max_of_floor']:.0f}\\%)" if s['A1'] else " (A1 fails)")
            + (f"; time per token is affine in the CPU layers ($R^2\\ge{s['min_r2']:.3f}$)" if s.get("min_r2") else "") + ".")
    if "gh200" in anc:
        s = anc["gh200"]["summary"]
        parts.append(
            f"On a GH200 (72-core Grace, LPDDR5X, NVLink-C2C; Triad {s['platform']['bw_cpu']:.0f}\\,GB/s), outside the fitted envelope, "
            f"the same predictions for {s['n']} configurations have {100 * s['median_ape_m4']:.0f}\\% median error "
            f"(measured/predicted {s['ratio_range'][0]:.2f}--{s['ratio_range'][1]:.2f}), "
            + ("and none exceeds the floor" if s["A1"] else f"and {s['floor_violations']} exceed the floor") + ".")
    if "a100" not in anc:
        parts.append("No A100 had capacity during the study; the registered A100 anchors remain to be run.")
    return " ".join(parts)


def prereg_log(N, anc):
    rows = [
        ("H0", "1 (A10)", "llama.cpp never beats the roofline", "holds, 31/31"),
        ("H1", "1", "ncu DRAM bytes within $\\pm$10\\% of GGUF bytes", "fails: +12--15\\% (inline ECC over-read)"),
        ("H2", "1", "M4 median APE $\\le$20\\%", "fails: 27.4\\% (every prediction too slow)"),
        ("H3", "1", "1/tok\\_s affine in \\texttt{--n-cpu-moe}, $R^2\\ge$0.98", "holds, 0.9965--0.9987"),
        ("H4", "1", "intercept/slope decomposition", "intercept over-predicted 25--55\\%"),
        ("H5, H1$'$", "1b", "two-run calibration; counter calibration", "not run (no second ncu platform)"),
        ("H6", "2", "cache top-1 agreement $\\ge$98\\% vs all-CPU", "fails narrowly: 97.4--97.9\\%"),
        ("H7", "2", "cache $\\ge$1.2$\\times$ llama.cpp at 25\\%", "fails: cache ran at all-CPU speed"),
        ("H9", "2", "measured hit rates within $\\pm$5 points of simulated", "holds on job 009 (0.679 vs 0.677)"),
        ("H8, H10", "2", "prediction; policy ordering", "not scored: system redesigned (phase 3)"),
        ("H11", "3", "mailbox $\\ge$1.30$\\times$ llama.cpp at 25\\%", "holds, 1.50--1.81$\\times$ (windows defect, see H16)"),
        ("H12", "3", "hand-off alone $\\ge$1.05$\\times$", "fails on Q8\\_0 (1.022, 1.047$\\times$)"),
        ("H13", "3", "NLL within 1\\%, top-1 $\\ge$95\\%", "fails on gpt-oss-120b (89.5--90.1\\%)"),
        ("H14", "3", "step-time model median APE $\\le$10\\%", "holds, 2.9\\%"),
        ("H15", "3", "own text: $\\ge$1.30$\\times$", "holds, 1.54--1.94$\\times$"),
        ("F1--F4", "4", "provenance changes a conclusion", "none fires on any model"),
        ("P1", "4", "own text has lower NLL", "holds on every model"),
        ("P2", "4", "greedy text is more reused", f"fails on {N['provPTwoFail']} of {N['provNModels']}"),
        ("P3", "4", "$|\\Delta$DFA$|\\ge$2 points on gpt-oss", "holds"),
        ("P4", "4", "DFA beats LRU in S", "fails on Qwen3 (both arms)"),
        ("4.4", "4", "classify the 120b anomaly", "class 3, format"),
        ("H16", "4.5", "$\\ge$1.30$\\times$ at 25\\%, correct windows", f"holds, {N['caseMin']}--{N['caseMax']}$\\times$ over all budgets"),
        ("H17", "4.5", "step model within 10\\% median", f"holds, {N['hSeventeenMed']}\\% (max {N['hSeventeenMax']}\\%)"),
        ("P5", "4.6", "$\\ge$1/3 of llama.cpp baselines weak", f"holds, {N['auditWeak']} of {N['auditLlamaRows']}"),
        ("P6", "4.6", "median system $\\le$60\\% of SoL", f"holds, {N['auditMedianSol']}\\%"),
    ]
    if "a100" in anc:
        s = anc["a100"]["summary"]
        rows += [("A1", "4.7", "no anchor beats the floor", "holds" if s["A1"] else "fails"),
                 ("A2", "4.7", "A100 anchor median APE $\\le$25\\%", ("holds" if s["A2"] else "fails") + f", {100 * s['median_ape_m4']:.0f}\\%"),
                 ("A3", "4.7", "model under-predicts $\\ge$2/3 of anchors", ("holds" if s["A3"] else "fails") + f", {s['under_predicted']} of {s['n']}")]
    elif "gh200" in anc:
        s = anc["gh200"]["summary"]
        rows += [("A1", "4.7", "no anchor beats the floor", ("holds" if s["A1"] else "fails") + " (GH200)"),
                 ("A2, A3", "4.7", "A100 anchors", "not run: no A100 capacity")]
    L = ["{\\small\\begin{longtable}{llp{6.5cm}p{6.5cm}}\\toprule id & phase & registered claim & outcome \\\\ \\midrule\\endhead\\bottomrule\\endfoot"]
    L += [f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in rows]
    L += ["\\end{longtable}}",
          "Every registration is a section of \\texttt{prereg/PROTOCOL.md} committed before the runs it concerns; the deviation logs there record every change."]
    return "\n".join(L)


if __name__ == "__main__":
    main()

"""Emit paper/numbers3.tex and paper/table_eval.tex from the phase-3 results (scored.json, sol.json, job 020c).

    python scripts/paper_numbers3.py --scored prereg/a10_mb/scored.json --sol prereg/a10_mb/sol.json \
        --gen results/firstparty/a10/020c --out paper
"""
import argparse
import glob
import json
import os

SHORT = {"gpt-oss-20b-MXFP4": "gpt-oss-20b", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M": "Qwen3 Q4\\_K\\_M",
         "Qwen3-30B-A3B-Instruct-2507-Q8_0": "Qwen3 Q8\\_0", "gpt-oss-120b-MXFP4": "gpt-oss-120b"}
ORDER = ["gpt-oss-20b-MXFP4", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M", "Qwen3-30B-A3B-Instruct-2507-Q8_0", "gpt-oss-120b-MXFP4"]


def load_runs(path):
    by = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            by.setdefault(r.get("config", ""), []).append(r)
    return by


def tok_s(rows):
    return 1000 * sum(r["n_decode"] for r in rows) / sum(r["decode_ms"] for r in rows)


def nll(rows):
    return sum(r["nll_sum"] for r in rows) / max(1, sum(r["nll_n"] for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--sol", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--nommap", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    S = json.load(open(args.scored))
    res, H = S["results"], S["hypotheses"]
    sol = json.load(open(args.sol)) if os.path.exists(args.sol) else {}
    N = {}

    def f2(x):
        return f"{x:.2f}"

    spd, hand, mbsplit, nllrel, agree, agag = [], [], [], [], [], []
    rows = []
    for tag in ORDER:
        if tag not in res:
            continue
        r = res[tag]
        first = True
        for q in sorted(r["budgets"], key=lambda x: int(x)):
            b = r["budgets"][q]
            mc, ml, sc, st = b["mailbox_cache"], b["mailbox_layers"], b["split_cache"], b["llama_static"]
            spd.append(mc["speedup_vs_llama"])
            hand.append(ml["speedup_vs_llama"])
            mbsplit.append(mc["tok_s"] / sc["tok_s"])
            nllrel.append(abs(mc["nll_rel_vs_llama"]))
            if mc.get("top1_agree_vs_llama") is not None:
                agree.append(mc["top1_agree_vs_llama"])
            if b.get("all_gpu_top1_agree_vs_llama") is not None:
                agag.append(b["all_gpu_top1_agree_vs_llama"])
            name = SHORT[tag] if first else ""
            first = False
            rows.append(f"{name} & {int(q)/8*100:.1f}\\% & {st['tok_s']:.1f} & {ml['tok_s']:.1f} & {sc['tok_s']:.1f} & \\textbf{{{mc['tok_s']:.1f}}} & "
                        f"{mc['speedup_vs_llama']:.2f}$\\times$ & {mc.get('pred_tok_s', float('nan')):.1f} & {mc.get('hit_rate', float('nan')):.3f} & "
                        f"{mc['nll_rel_vs_llama']*100:+.2f}\\% & {mc.get('top1_agree_vs_llama', float('nan'))*100:.1f}\\%\\\\")
        if "all_gpu" in r:
            rows.append(f" & all on GPU & {r['all_gpu']['tok_s']:.1f} & & & & & & & & \\\\")
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    head = ("\\begin{tabular}{llrrrrrrrrr}\\toprule\n"
            "model & budget & llama.cpp & helpers & split cache & mailbox cache & speed-up & predicted & hit & NLL $\\Delta$ & top-1\\\\\\midrule\n")
    open(os.path.join(args.out, "table_eval.tex"), "w").write(head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

    N["spdMin"], N["spdMax"] = f2(min(spd)), f2(max(spd))
    N["handoffMin"], N["handoffMax"] = f2(min(hand)), f2(max(hand))
    N["mbOverSplitMin"], N["mbOverSplitMax"] = f2(min(mbsplit)), f2(max(mbsplit))
    N["nllMaxRel"] = f"{max(nllrel)*100:.1f}"
    N["agreeMin"], N["agreeMax"] = f"{min(agree)*100:.1f}", f"{max(agree)*100:.1f}"
    small = [b["mailbox_cache"]["top1_agree_vs_llama"] for t in ORDER[:3] for b in res[t]["budgets"].values()]
    big = [b["mailbox_cache"]["top1_agree_vs_llama"] for b in res[ORDER[3]]["budgets"].values()]
    N["agreeSmallMin"], N["agreeSmallMax"] = f"{min(small)*100:.1f}", f"{max(small)*100:.1f}"
    N["agreeBigMin"], N["agreeBigMax"] = f"{min(big)*100:.1f}", f"{max(big)*100:.1f}"
    fails = [(t, q, b["mailbox_layers"]["speedup_vs_llama"]) for t in ORDER for q, b in res[t]["budgets"].items()
             if b["mailbox_layers"]["speedup_vs_llama"] < 1.05]
    N["hTwelveDetail"] = ("on " + " and ".join(sorted({SHORT[t] for t, _, _ in fails})) + " the gain is only " +
                          ", ".join(f"{v:.3f}$\\times$ at {int(q)/8*100:.1f}\\%" for t, q, v in fails)) if fails else "every configuration clears it"
    N["agGptSmall"] = f"{res[ORDER[0]]['all_gpu']['tok_s']:.0f}"
    N["agQwen"] = f"{res[ORDER[1]]['all_gpu']['tok_s']:.0f}"
    N["agAgreeMin"], N["agAgreeMax"] = (f"{min(agag)*100:.1f}", f"{max(agag)*100:.1f}") if agag else ("--", "--")
    q2 = lambda t: res[t]["budgets"]["2"]
    N["spdQuarterGptSmall"] = f2(q2(ORDER[0])["mailbox_cache"]["speedup_vs_llama"])
    N["spdQuarterQwenQ"] = f2(q2(ORDER[1])["mailbox_cache"]["speedup_vs_llama"])
    N["spdQuarterQwenE"] = f2(q2(ORDER[2])["mailbox_cache"]["speedup_vs_llama"])
    N["spdQuarterGptBig"] = f2(q2(ORDER[3])["mailbox_cache"]["speedup_vs_llama"])
    N["bigTokS"] = f"{q2(ORDER[3])['mailbox_cache']['tok_s']:.1f}"
    N["bigLlamaTokS"] = f"{q2(ORDER[3])['llama_static']['tok_s']:.1f}"
    verdict = lambda h: "holds" if H[h]["holds"] else "does not hold"
    for h, k in [("H11", "hElevenVerdict"), ("H12", "hTwelveVerdict"), ("H13", "hThirteenVerdict"), ("H14", "hFourteenVerdict")]:
        N[k] = verdict(h)
    N["predMedAPE"] = f"{H['H14']['median_ape']*100:.1f}"
    N["predMaxAPE"] = f"{H['H14']['max_ape']*100:.1f}"
    N["nPred"] = str(H["H14"]["n"])
    pred = json.load(open(os.path.join(os.path.dirname(args.scored), "predictions.json")))["fits"]
    N["aOneGpt"] = f"{pred[ORDER[0]]['a1']*1000:.0f}"
    N["aOneGptBw"] = f"{13.25e6 / (pred[ORDER[0]]['a1'] * 1e-3) / 1e9:.0f}"
    N["aOneQwen"] = f"{pred[ORDER[1]]['a1']*1000:.0f}"
    N["handoffUs"] = "84"
    sol = {t: o for t, o in sol.items() if t in ORDER[:2]}   # models with a measured all-GPU step
    if sol:
        allb = [b for o in sol.values() for b in o["budgets"].values()]
        few = [1 - o["budgets"]["2"]["minbypass_miss_per_layer_step"] / o["budgets"]["2"]["dfa_miss_per_layer_step"] for o in sol.values()]
        N["minFewerMin"], N["minFewerMax"] = f"{min(few)*100:.0f}", f"{max(few)*100:.0f}"
        N["solLo"] = f"{min(b['sol_tok_s'] for b in allb):.0f}"
        N["solHi"] = f"{max(b['sol_tok_s'] for b in allb):.0f}"
        N["llamaSolMin"] = f"{min(b['llama_of_sol'] for b in allb)*100:.0f}"
        N["llamaSolMax"] = f"{max(b['llama_of_sol'] for b in allb)*100:.0f}"
        N["mbSolMin"] = f"{min(b['mailbox_of_sol'] for b in allb)*100:.0f}"
        N["mbSolMax"] = f"{max(b['mailbox_of_sol'] for b in allb)*100:.0f}"
        N["mbIdealMin"] = f"{min(b['mailbox_of_dfa_ideal'] for b in allb)*100:.0f}"
        N["mbIdealMax"] = f"{max(b['mailbox_of_dfa_ideal'] for b in allb)*100:.0f}"
    # H15: own text
    g = {}
    for tag in ORDER:
        pe = os.path.join(args.gen, f"{tag}_gen_ec.jsonl")
        L = {"gpt-oss-20b-MXFP4": 24, "Qwen3-30B-A3B-Instruct-2507-Q4_K_M": 48, "Qwen3-30B-A3B-Instruct-2507-Q8_0": 48, "gpt-oss-120b-MXFP4": 36}[tag]
        E = {"gpt-oss-20b-MXFP4": 32}.get(tag, 128)
        ps = os.path.join(args.gen, f"{tag}_gen_static_n{L - L // 4}.jsonl")
        if not (os.path.exists(pe) and os.path.exists(ps)):
            continue
        by = load_runs(pe)
        mb = [v for k, v in by.items() if f"slots={E // 4}:" in k][0]
        st = list(load_runs(ps).values())[0]
        g[tag] = dict(speedup=tok_s(mb) / tok_s(st), mb_tok_s=tok_s(mb), llama_tok_s=tok_s(st), nll=nll(st))
    if g:
        N["genSpdMin"] = f2(min(v["speedup"] for v in g.values()))
        N["genSpdMax"] = f2(max(v["speedup"] for v in g.values()))
        N["hFifteenVerdict"] = "holds" if len(g) == 4 and all(v["speedup"] >= 1.30 for v in g.values()) else "does not hold"
        if ORDER[3] in g:
            N["genBigNll"] = f"{g[ORDER[3]]['nll']:.2f}"
        N["dataBigNll"] = f"{res[ORDER[3]]['budgets']['2']['llama_static']['nll']:.2f}"
        json.dump(g, open(os.path.join(os.path.dirname(args.scored), "own_text.json"), "w"), indent=1)
    if args.nommap:
        ch, best = [], []
        for tag in ORDER:
            L = {"gpt-oss-20b-MXFP4": 24, "gpt-oss-120b-MXFP4": 36}.get(tag, 48)
            for q, b in res[tag]["budgets"].items():
                n = L - L * int(q) // 8
                p = os.path.join(args.nommap, f"{tag}_nommap_n{n}.jsonl")
                if not os.path.exists(p):
                    continue
                t_nm = tok_s(list(load_runs(p).values())[0])
                t_mm = b["llama_static"]["tok_s"]
                ch.append(t_nm / t_mm - 1)
                best.append(b["mailbox_cache"]["tok_s"] / max(t_nm, t_mm))
        if ch:
            N["noMmapMin"], N["noMmapMax"] = f"{min(ch)*100:+.1f}", f"{max(ch)*100:+.1f}"
            N["spdBestMin"], N["spdBestMax"] = f2(min(best)), f2(max(best))
    with open(os.path.join(args.out, "numbers3.tex"), "w") as f:
        for k, v in N.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(open(os.path.join(args.out, "numbers3.tex")).read())


if __name__ == "__main__":
    main()

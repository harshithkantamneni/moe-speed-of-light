"""Statistics for job 089: Table 1's protocol on a second host with a stock-clock card (RTX 5090 at 14001 MHz memory
clock next to a Ryzen 9 9950X; listing B's card reported 17001 MHz), with the LRU attribution and the prefill/batch
timings on the same rental.

Launch 1 ran, per cell, ours with the law's table, FreeToken with the backend carried over from job 081, and stock
llama.cpp; launch 2 reran FreeToken and ours in the reversed order (the comparison uses launch 2; launch 1 against
launch 2 is the order check). Then ours with the LRU policy and the law's table at every cell, and llama-batched-bench
for ours and stock at 25%. Ratios of mean speeds, paired by problem, 95% percentile interval over 10,000 bootstrap
resamples (seed 0). Listing B's numbers come from prereg/headline_081.json (and split_080.json for Qwen3 25 / 43.75%).

    python scripts/stockclock_stats.py --out prereg/stockclock_089.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from samehost_stats import boot_ratio  # noqa: E402
from scripts.speed_limit import MODELS, host_rates, limit  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = "/home/claude/gpu-branch/results"
D = f"{R}/089_headline_stockclock@vast"
CELLS = [("gpt-oss-120b", "11%", "g_ours_C14", "g_ft_hybrid_r0.111", "g_llama_n32"),
         ("gpt-oss-120b", "25%", "g_ours_C32", "g_ft_hybrid_r0.25", "g_llama_n27"),
         ("gpt-oss-120b", "40%", "g_ours_C51", "g_ft_offload_r0.40", "g_llama_n22"),
         ("Qwen3-30B-A3B", "12.5%", "q_ours_C16", "q_ft_hybrid_r0.125", "q_llama_n42"),
         ("Qwen3-30B-A3B", "25%", "q_ours_C32", "q_ft_hybrid_r0.25", "q_llama_n36"),
         ("Qwen3-30B-A3B", "43.75%", "q_ours_C56", "q_ft_offload_r0.4375", "q_llama_n27")]


def listing_b():
    h = json.load(open(os.path.join(ROOT, "prereg", "headline_081.json")))
    s = json.load(open(os.path.join(ROOT, "prereg", "split_080.json")))
    r1 = json.load(open(os.path.join(ROOT, "prereg", "run1_082.json")))
    out = {}
    for r in h["rows"]:
        out[(r["model"], r["budget"])] = dict(ours=r["ours_L2"], ft=r["ft_L2"], llama=r["llama"], ratio=r["ours_over_ft_L2"][0])
    m = s["means"]
    out[("Qwen3-30B-A3B", "25%")] = dict(ours=m["ours_C32_law L1"], ft=m["ft_hybrid_r0.25 L1"], llama=r1["llama"]["q_stock_n36"]["measured"],
                                        ratio=s["ratios"]["C32 best ours_C32_law / ft_hybrid_r0.25 (L1)"][0])
    out[("Qwen3-30B-A3B", "43.75%")] = dict(ours=m["ours_C56_law L2"], ft=m["ft_offload_r0.4375 L2"], llama=r1["llama"]["q_stock_n27"]["measured"],
                                           ratio=s["ratios"]["ours_C56_law / ft_offload_r0.4375 (L2)"][0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    by = {}
    for line in open(os.path.join(D, "bs1.jsonl")):
        if line.strip():
            r = json.loads(line)
            by.setdefault((r["label"], r["launch"]), {})[r["problem"]] = r["decode_tok_s"]
    mean = {f"{k[0]} L{k[1]}": float(np.mean(list(v.values()))) for k, v in by.items()}

    def ratio(x, y):
        P = sorted(set(by[x]) & set(by[y]))
        return boot_ratio(np.array([by[x][p] for p in P]), np.array([by[y][p] for p in P]))
    B = listing_b()
    rows = []
    for model, budget, op, ft, ll in CELLS:
        law, lru = f"{op}_law", f"{op}_lru"
        row = dict(model=model, budget=budget, ft_variant=ft.split("_")[2],
                   ours=mean[f"{law} L2"], ft=mean[f"{ft} L2"], llama=mean[f"{ll} L1"], lru=mean[f"{lru} L1"],
                   ours_over_ft_L2=ratio((law, 2), (ft, 2)), ours_over_ft_L1=ratio((law, 1), (ft, 1)),
                   ours_over_llama=ratio((law, 2), (ll, 1)), lru_over_law=ratio((lru, 1), (law, 1)),
                   lru_over_ft=ratio((lru, 1), (ft, 1)))
        b = B[(model, budget)]
        row["listing_b"] = b
        row["vs_b"] = dict(ours=row["ours"] / b["ours"] - 1, ft=row["ft"] / b["ft"] - 1, llama=row["llama"] / b["llama"] - 1,
                           ratio_diff=row["ours_over_ft_L2"][0] - b["ratio"])
        row["order_diff"] = row["ours_over_ft_L1"][0] / row["ours_over_ft_L2"][0] - 1
        rows.append(row)
        m, lo, hi = row["ours_over_ft_L2"]
        print(f"{model} {budget}: ours {row['ours']:.1f} / FreeToken ({row['ft_variant']}) {row['ft']:.1f} = {m:.3f} [{lo:.3f}, {hi:.3f}] "
              f"(B {b['ratio']:.3f}, diff {row['vs_b']['ratio_diff']:+.3f}); vs B: ours {100 * row['vs_b']['ours']:+.0f}% FT {100 * row['vs_b']['ft']:+.0f}% "
              f"llama {100 * row['vs_b']['llama']:+.0f}%; order {100 * row['order_diff']:+.1f}%; LRU/law {row['lru_over_law'][0]:.3f}; "
              f"LRU/FT {row['lru_over_ft'][0]:.3f}; ours/llama {row['ours_over_llama'][0]:.2f}")
    bb = {}
    for key in ("g_ours", "g_stock", "q_ours", "q_stock"):
        p = os.path.join(D, f"bb_{key}.jsonl")
        if os.path.exists(p):
            bb[key] = [dict(pp=r["pp"], pl=r["pl"], speed_pp=r["speed_pp"], speed_tg=r["speed_tg"]) for r in (json.loads(l) for l in open(p) if l.strip())]
    bbr = {}
    for m_ in ("g", "q"):
        if f"{m_}_ours" in bb and f"{m_}_stock" in bb:
            o = {(r["pp"], r["pl"]): r for r in bb[f"{m_}_ours"]}
            s_ = {(r["pp"], r["pl"]): r for r in bb[f"{m_}_stock"]}
            bbr[m_] = {f"pp{pp}_pl{pl}": dict(pp_ratio=o[(pp, pl)]["speed_pp"] / s_[(pp, pl)]["speed_pp"],
                                             tg_ratio=o[(pp, pl)]["speed_tg"] / s_[(pp, pl)]["speed_tg"]) for (pp, pl) in o if (pp, pl) in s_}
            for k, v in sorted(bbr[m_].items()):
                print(f"  batched-bench {m_} {k}: prefill ours/stock {v['pp_ratio']:.2f}, decode ours/stock {v['tg_ratio']:.2f}")
    probes = {}
    txt = open(os.path.join(D, "concur.txt")).read()
    import re
    cpu = re.findall(r"cpu_read_gbs t=(\d+) ([\d.]+)", txt)
    probes["cpu_by_threads"] = {int(t): float(v) for t, v in cpu}
    zc = re.findall(r"pcie_zerocopy_gbs chunk=16MB blocks=64 ([\d.]+)", txt)
    probes["pcie_zerocopy"] = float(zc[0]) if zc else None
    both = [float(x) for x in re.findall(r"concurrent t=\d+ pcie=zerocopy64 cpu_gbs [\d.]+ pcie_gbs [\d.]+ sum ([\d.]+)", txt)]
    probes["both_zerocopy"] = both
    # the two hosts side by side: the law's probe rates and tables (fetch_table_law_*.json) and the card (gpu.csv)
    def host(job):
        d_ = f"{R}/{job}"
        g_ = json.load(open(os.path.join(d_, "fetch_table_law_gptoss.json")))
        q_ = json.load(open(os.path.join(d_, "fetch_table_law_qwen3.json")))
        import csv
        rows_ = list(csv.reader(open(os.path.join(d_, "gpu.csv"))))
        hdr = [h.strip() for h in rows_[0]]
        val = {h: rows_[1][i].strip() for i, h in enumerate(hdr)}
        ft = {}
        import glob
        for f in glob.glob(f"{d_}/GPU-*.json"):
            c = json.load(open(f))["ceilings"]
            ft = dict(cpu=c["cpu_stream_read_gbs"], pcie=c["pcie_linear_h2d_gbs"])
        return dict(B_c=g_["B_c"], B_p=g_["B_p"], B_both=g_["B_both"], table_gpt=g_["table"], table_qwen=q_["table"],
                    mem_clock_mhz=int(val["clocks.max.memory [MHz]"].split()[0]), sm_clock_mhz=int(val["clocks.max.sm [MHz]"].split()[0]),
                    power_limit_w=float(val["power.limit [W]"].split()[0]), ft_probe=ft)
    hosts = {"089 (9950X, stock clock)": host("089_headline_stockclock@vast"), "listing B (081)": host("081_headline_law@vast")}
    a_, b_ = hosts["089 (9950X, stock clock)"], hosts["listing B (081)"]
    hosts["089_over_B"] = dict(B_c=a_["B_c"] / b_["B_c"] - 1, B_p=a_["B_p"] / b_["B_p"] - 1, B_both=a_["B_both"] / b_["B_both"] - 1,
                               mem_clock=a_["mem_clock_mhz"] / b_["mem_clock_mhz"] - 1)
    print(f"host: B_c {a_['B_c']} vs {b_['B_c']} ({100 * hosts['089_over_B']['B_c']:+.0f}%), B_p {a_['B_p']} vs {b_['B_p']}, "
          f"B_both {a_['B_both']} vs {b_['B_both']} ({100 * hosts['089_over_B']['B_both']:+.0f}%); memory clock {a_['mem_clock_mhz']} vs "
          f"{b_['mem_clock_mhz']}; power {a_['power_limit_w']} vs {b_['power_limit_w']} W; tables {a_['table_gpt']} / {a_['table_qwen']} "
          f"vs {b_['table_gpt']} / {b_['table_qwen']}")
    # which cells each system leads, with the interval rule of the scorecard
    lead = dict(ours_leads_ft=[f"{r['model']} {r['budget']}" for r in rows if r["ours_over_ft_L2"][1] > 1],
                ft_leads_ours=[f"{r['model']} {r['budget']}" for r in rows if r["ours_over_ft_L2"][2] < 1],
                lru_trails_ft=[f"{r['model']} {r['budget']}" for r in rows if r["lru_over_ft"][2] < 1],
                lru_leads_ft=[f"{r['model']} {r['budget']}" for r in rows if r["lru_over_ft"][1] > 1],
                policy_share_of_lead={f"{r['model']} {r['budget']}": (r["ours_over_ft_L2"][0] - r["lru_over_ft"][0]) / (r["ours_over_ft_L2"][0] - 1)
                                      for r in rows if r["ours_over_ft_L2"][0] > 1})
    print("leads:", json.dumps(lead, indent=1))
    # the speed limit on this host: the exact optimum's reads (prereg/speed_limit_v2.json, the same trace), this host's
    # best probed rate (scripts/speed_limit.py's rule) and the 5090's datasheet; every system placed against it
    sl = json.load(open(os.path.join(ROOT, "prereg", "speed_limit_v2.json")))
    b_host = host_rates(txt)
    LM = {"gpt-oss-120b": "gpt-oss-120b", "Qwen3-30B-A3B": "qwen3-30b-a3b-bf16"}
    for row, (model, budget, op, ft, ll) in zip(rows, CELLS):
        C = int(op.split("_C")[1]); lm = LM[model]; srow = sl["models"][lm]["rows"][str(C)]; m_ = MODELS[lm]
        L, k = sl["models"][lm]["L"], sl["models"][lm]["k"]
        reads = srow["reads_per_token"]["exact"]
        t_, c_at = limit(reads, L * k, m_["S"], m_["D"], b_host, sl["B_gpu_datasheet"] if isinstance(sl["B_gpu_datasheet"], (int, float)) else 1792e9)
        row["limit"] = dict(reads_per_token=reads, b_host_gbs=b_host / 1e9, t_ms=1e3 * t_, tok_s=1 / t_, cpu_at_limit=c_at,
                            frac=dict(ours=row["ours"] * t_, ft=row["ft"] * t_, llama=row["llama"] * t_, lru=row["lru"] * t_),
                            headline=dict(tok_s=srow["limits"]["exact"]["tok_s"], frac=srow["frac_of_limit"]["exact"]))
        print(f"limit {model} {budget}: {1 / t_:.0f} tok/s here (host {b_host / 1e9:.0f} GB/s) vs {srow['limits']['exact']['tok_s']:.0f} on listing B; "
              f"ours {100 * row['ours'] * t_:.0f}% of it here vs {100 * srow['frac_of_limit']['exact']['ours']:.0f}% there; "
              f"FreeToken {100 * row['ft'] * t_:.0f}% vs {100 * srow['frac_of_limit']['exact']['freetoken']:.0f}%")
    out = dict(means=mean, rows=rows, batched_bench=bb, batched_bench_ratios=bbr, probes=probes, hosts=hosts, leads=lead,
               b_host_max_gbs=b_host / 1e9)
    for f in ("gate.txt", "oneline.txt", "fetch_table_law_gptoss.json", "fetch_table_law_qwen3.json"):
        p = os.path.join(D, f)
        if os.path.exists(p):
            out[f] = open(p).read()
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

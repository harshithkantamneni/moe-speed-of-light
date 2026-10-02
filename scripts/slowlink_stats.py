"""Statistics for job 088, the slow-link comparison done fairly (gpt-oss-120b next to a Core i9-14900K, RTX 5090 in a
PCIe 4.0-class slot; FreeToken's CPU executor on 8 pinned performance-core threads).

Launch 1 ran, per budget, FreeToken offload, FreeToken hybrid on 8 threads with its automatic fetch cap (hybrid8), the
same with no fetching (hybrid8f0), and ours with the law's table; then stock llama.cpp at 25% with -t 8 (one thread per
P-core) and with -t 24. Launch 2 reran FreeToken's faster configuration and ours per budget in the reversed order; the
comparison uses launch 2. Ratios of mean speeds, paired by problem, 95% percentile interval over 10,000 bootstrap
resamples (seed 0). Job 085 (the same CPU model, FreeToken at its default 23 threads) is the comparison point.

    python scripts/slowlink_stats.py --out prereg/slowlink_088.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from samehost_stats import boot_ratio  # noqa: E402

R = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
D = f"{R}/088_slowlink_fair@vast"
CELLS = [("11%", "g_ours_C14", "g_ft_{}_r0.111"), ("25%", "g_ours_C32", "g_ft_{}_r0.25"), ("40%", "g_ours_C51", "g_ft_{}_r0.40")]
J085 = {"11%": dict(hybrid=12.0, offload=24.6, ours=49.9, ratio=2.030), "25%": dict(hybrid=30.5, offload=48.2, ours=81.9, ratio=1.699),
        "40%": dict(hybrid=66.0, offload=86.0, ours=122.9, ratio=1.428)}   # prereg/halfpcie_085.json means and launch-2 ratios


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
    for k in sorted(mean):
        print(f"  {k:28s} {mean[k]:7.2f} tok/s  n={len(by[(k.rsplit(' L', 1)[0], int(k.rsplit(' L', 1)[1]))])}")

    def ratio(x, y):
        P = sorted(set(by[x]) & set(by[y]))
        return boot_ratio(np.array([by[x][p] for p in P]), np.array([by[y][p] for p in P]))
    rows = []
    for budget, op, fs in CELLS:
        row = dict(model="gpt-oss-120b", budget=budget, ours_variant="law")
        row["launch1"] = {v: mean.get(f"{fs.format(v)} L1") for v in ("offload", "hybrid8", "hybrid8f0")}
        row["launch1"]["ours"] = mean[f"{op}_law L1"]
        best = max((v for v in ("offload", "hybrid8") if (fs.format(v), 2) in by), key=lambda v: mean[f"{fs.format(v)} L2"])
        row.update(ft_variant=best, ft=mean[f"{fs.format(best)} L2"], ours=mean[f"{op}_law L2"],
                   ours_over_ft_L2=ratio((f"{op}_law", 2), (fs.format(best), 2)))
        row["ft_hybrid8_over_085_hybrid"] = row["launch1"]["hybrid8"] / J085[budget]["hybrid"]
        row["ft_best_over_085_best"] = row["ft"] / max(J085[budget]["hybrid"], J085[budget]["offload"])
        row["job085"] = J085[budget]
        rows.append(row)
        m, lo, hi = row["ours_over_ft_L2"]
        print(f"gpt-oss {budget}: ours {row['ours']:.1f} / FreeToken ({best}) {row['ft']:.1f} = {m:.3f} [{lo:.3f}, {hi:.3f}] (L2); "
              f"hybrid8 vs 085 hybrid {row['ft_hybrid8_over_085_hybrid']:.2f}x; FT best vs 085 best {row['ft_best_over_085_best']:.2f}x; "
              f"085 ratio {J085[budget]['ratio']}")
    law = json.load(open(os.path.join(D, "law_llama.json")))
    llama = {"t8": mean["g_llama_n27_t8 L1"], "tall": mean["g_llama_n27_tall L1"],
             "law_t8": law["predicted_tok_s"]["gptoss_llama_n27_t8"], "law_tall": law["predicted_tok_s"]["gptoss_llama_n27_t24"]}
    llama["err_t8"] = llama["law_t8"] / llama["t8"] - 1
    llama["err_tall"] = llama["law_tall"] / llama["tall"] - 1
    llama["t8_over_tall"] = ratio(("g_llama_n27_t8", 1), ("g_llama_n27_tall", 1))
    print(f"llama.cpp -t 8 {llama['t8']:.1f} (law {llama['law_t8']:.1f}, {100 * llama['err_t8']:+.1f}%), -t 24 {llama['tall']:.1f} "
          f"(law {llama['law_tall']:.1f}, {100 * llama['err_tall']:+.1f}%); t8/t24 {llama['t8_over_tall'][0]:.3f}")
    out = dict(means=mean, rows=rows, llama=llama)
    for f in ("tables.txt", "selection.txt", "ft_lines.txt", "pmask.txt", "ft_affinity.txt", "oneline.txt"):
        p = os.path.join(D, f)
        if os.path.exists(p):
            out[f] = open(p).read()
    g = glob.glob(f"{D}/GPU-*.json")
    if g:
        c = json.load(open(g[0]))["ceilings"]
        out["ft_probe"] = dict(cpu=c["cpu_stream_read_gbs"], pcie=c["pcie_linear_h2d_gbs"])
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

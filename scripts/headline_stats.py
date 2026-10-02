"""Statistics for job 081: the headline table with the law's per-host FETCH split (gpt-oss-120b at 11/25/40%,
Qwen3-30B-A3B BF16 at 12.5%), one RTX 5090 + 9950X3D host.

Launch 1 runs every variant; each system's faster variant per budget (ours: current or law table; FreeToken: offload
or hybrid) is picked on launch 1 and rerun in launch 2, and the headline comparison uses launch 2 only. llama.cpp runs
once (launch 1). Ratios of mean speeds, paired by problem, 95% percentile interval over 10,000 bootstrap resamples.

    python scripts/headline_stats.py --dir /home/claude/gpu-branch/results/081_headline_law@vast --out prereg/headline_081.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from samehost_stats import boot_ratio  # noqa: E402

ROWS = [  # model, budget, ours C, FreeToken rate, llama.cpp -ncmoe, prefix
    ("gpt-oss-120b", "11%", 14, "0.111", 32, "g"),
    ("gpt-oss-120b", "25%", 32, "0.25", 27, "g"),
    ("gpt-oss-120b", "40%", 51, "0.40", 22, "g"),
    ("Qwen3-30B-A3B", "12.5%", 16, "0.125", 42, "q"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results"), "081_headline_law@vast"))
    ap.add_argument("--out")
    a = ap.parse_args()
    by, vram = {}, {}
    for line in open(os.path.join(a.dir, "bs1.jsonl")):
        if line.strip():
            r = json.loads(line)
            by.setdefault((r["label"], r["launch"]), {})[r["problem"]] = r["decode_tok_s"]
            vram.setdefault(r["label"], set()).add(round(r.get("vram_used_gib") or 0, 1))
    mean = {k: float(np.mean(list(v.values()))) for k, v in by.items()}

    def ratio(x, y):
        P = sorted(set(by[x]) & set(by[y]))
        return boot_ratio(np.array([by[x][p] for p in P]), np.array([by[y][p] for p in P]))
    print(open(os.path.join(a.dir, "tables.txt")).read().strip())
    out = dict(rows=[], predictions={})
    p1, p2, p3, p4 = [], [], [], []
    for model, pct, C, rate, n, pre in ROWS:
        cur, law = (f"{pre}_ours_C{C}_cur", 1), (f"{pre}_ours_C{C}_law", 1)
        ftl = [(f"{pre}_ft_{b}_r{rate}", 1) for b in ("offload", "hybrid")]
        lc = (f"{pre}_llama_n{n}", 1)
        if cur not in by or law not in by:
            print(f"{model} {pct}: missing")
            continue
        row = dict(model=model, budget=pct, ours_cur_L1=mean[cur], ours_law_L1=mean[law],
                   law_over_cur_L1=ratio(law, cur), llama=mean.get(lc),
                   ft_L1={k[0]: mean[k] for k in ftl if k in mean})
        o2 = [k for k in by if k[1] == 2 and k[0].startswith(f"{pre}_ours_C{C}_")]
        f2 = [k for k in by if k[1] == 2 and k[0].startswith(f"{pre}_ft_") and k[0].endswith(f"r{rate}")]
        if o2 and f2:
            row.update(ours_L2_variant=o2[0][0].split("_")[-1], ours_L2=mean[o2[0]], ft_L2_variant=f2[0][0].split("_")[2],
                       ft_L2=mean[f2[0]], ours_over_ft_L2=ratio(o2[0], f2[0]))
            if lc in by:
                row["ours_over_llama"] = ratio(o2[0], lc)
        row["vram"] = {k: sorted(v) for k, v in vram.items() if k.startswith(f"{pre}_") and (f"C{C}_" in k or f"r{rate}" in k or f"n{n}" in k)}
        out["rows"].append(row)
        m, lo, hi = row["law_over_cur_L1"]
        print(f"{model} {pct}: L1 ours cur {mean[cur]:.1f}, law {mean[law]:.1f} (law/cur {m:.3f} [{lo:.3f}, {hi:.3f}]); "
              f"FreeToken {', '.join(f'{k.split(chr(95))[2]} {v:.1f}' for k, v in row['ft_L1'].items())}; llama.cpp {row['llama'] or float('nan'):.1f}")
        if "ours_over_ft_L2" in row:
            m2, lo2, hi2 = row["ours_over_ft_L2"]
            ml, lol, hil = row.get("ours_over_llama", (float("nan"),) * 3)
            print(f"   L2: ours ({row['ours_L2_variant']}) {row['ours_L2']:.1f} vs FreeToken ({row['ft_L2_variant']}) {row['ft_L2']:.1f}: "
                  f"{m2:.3f} [{lo2:.3f}, {hi2:.3f}]; ours / llama.cpp {ml:.2f} [{lol:.2f}, {hil:.2f}]")
        if pre == "g":
            p1.append(lo > 1.0)
            if "ours_over_ft_L2" in row:
                p2.append(row["ours_over_ft_L2"][1] > 1.0)
            if "ours_over_llama" in row:
                p4.append(row["ours_over_llama"][0] >= 1.8)
        else:
            p3.append(lo > 1.0)
            if "ours_over_ft_L2" in row:
                p3.append(row["ours_over_ft_L2"][1] > 1.0)
            if "ours_over_llama" in row:
                p4.append(row["ours_over_llama"][0] >= 2.0)
    out["predictions"]["1 gpt-oss: law table beats current at every budget (L1, CI > 1)"] = all(p1) if len(p1) == 3 else None
    out["predictions"]["2 gpt-oss: ours leads FreeToken at 11/25/40% (L2, CI > 1)"] = all(p2) if len(p2) == 3 else None
    out["predictions"]["3 Qwen3 12.5%: law beats current (L1) and ours leads FreeToken (L2), CIs > 1"] = all(p3) if len(p3) == 2 else None
    out["predictions"]["4 ours >= 1.8x llama.cpp (gpt-oss), >= 2x (Qwen3 12.5%)"] = all(p4) if len(p4) == 4 else None
    for k, v in out["predictions"].items():
        print(f"prediction {k}: {'HELD' if v else 'FAILED' if v is not None else 'n/a'}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

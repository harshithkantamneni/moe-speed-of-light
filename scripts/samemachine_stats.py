"""Statistics for the same-machine comparison jobs that follow the Table 1 protocol (jobs 085 and 086).

Launch 1 runs every variant; each system's faster variant per budget (ours: law or fixed FETCH table; FreeToken:
offload or hybrid) is picked on launch 1 and rerun in launch 2, which the comparison uses. llama.cpp and the extra
variants run once. Ratios of mean speeds, paired by problem, 95% percentile interval over 10,000 bootstrap resamples
(seed 0).

    python scripts/samemachine_stats.py --job 085 --out prereg/halfpcie_085.json
    python scripts/samemachine_stats.py --job 086 --out prereg/qwen36_086.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from samehost_stats import boot_ratio  # noqa: E402

R = "/home/claude/gpu-branch/results"
JOBS = {
    "085": dict(dir=f"{R}/085_half_pcie@vast", cells=[
        # model, budget, ours prefix+C, FreeToken label stem, llama label or None
        ("gpt-oss-120b", "11%", "g_ours_C14", "g_ft_{}_r0.111", None),
        ("gpt-oss-120b", "25%", "g_ours_C32", "g_ft_{}_r0.25", "g_llama_n27"),
        ("gpt-oss-120b", "40%", "g_ours_C51", "g_ft_{}_r0.40", None),
        ("Qwen3-30B-A3B", "12.5%", "q_ours_C16", "q_ft_{}_r0.125", None),
        ("Qwen3-30B-A3B", "25%", "q_ours_C32", "q_ft_{}_r0.25", "q_llama_n36"),
        ("Qwen3-30B-A3B", "43.75%", "q_ours_C56", "q_ft_{}_r0.4375", None)]),
    "086": dict(dir=f"{R}/086_qwen36@vast", cells=[
        ("Qwen3.6-35B-A3B", "12.5%", "ours_C32", "ft_{}_r0.125", "llama_n35"),
        ("Qwen3.6-35B-A3B", "25%", "ours_C64", "ft_{}_r0.25", "llama_n30"),
        ("Qwen3.6-35B-A3B", "37.5%", "ours_C96", "ft_{}_r0.375", "llama_n25")]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=list(JOBS))
    ap.add_argument("--dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    J = JOBS[a.job]
    d = a.dir or J["dir"]
    by, vram = {}, {}
    for line in open(os.path.join(d, "bs1.jsonl")):
        if line.strip():
            r = json.loads(line)
            by.setdefault((r["label"], r["launch"]), {})[r["problem"]] = r["decode_tok_s"]
            vram.setdefault(r["label"], set()).add(round(r.get("vram_used_gib") or 0, 1))
    mean = {f"{k[0]} L{k[1]}": float(np.mean(list(v.values()))) for k, v in by.items()}
    for k in sorted(mean):
        print(f"  {k:28s} {mean[k]:7.2f} tok/s  n={len(by[(k.rsplit(' L', 1)[0], int(k.rsplit(' L', 1)[1]))])}")

    def ratio(x, y):
        P = sorted(set(by[x]) & set(by[y]))
        return boot_ratio(np.array([by[x][p] for p in P]), np.array([by[y][p] for p in P]))
    rows = []
    for model, budget, op, fs, ll in J["cells"]:
        row = dict(model=model, budget=budget)
        law, cur = (f"{op}_law", 1), (f"{op}_cur", 1)
        if law in by and cur in by:
            row["law_over_fixed_L1"] = ratio(law, cur)
        cands = [(f"{op}_{t}", 2) for t in ("law", "cur") if (f"{op}_{t}", 2) in by]
        fcands = [(fs.format(b), 2) for b in ("offload", "hybrid") if (fs.format(b), 2) in by]
        if cands and fcands:
            o, f = cands[0], fcands[0]
            row.update(ours=mean[f"{o[0]} L2"], ours_variant=o[0].rsplit("_", 1)[1], ft=mean[f"{f[0]} L2"],
                       ft_variant=f[0].split("_")[-2], ours_over_ft_L2=ratio(o, f))
            if ll and (ll, 1) in by:
                row["llama"] = mean[f"{ll} L1"]
                row["ours_over_llama"] = ratio(o, (ll, 1))
        row["vram"] = {k: sorted(v) for k, v in vram.items() if k.startswith(op) or k.startswith(fs.split("{")[0]) or k == ll}
        rows.append(row)
        s = f"{model} {budget}:"
        if "law_over_fixed_L1" in row:
            s += f" law/fixed (L1) {row['law_over_fixed_L1'][0]:.3f} [{row['law_over_fixed_L1'][1]:.3f}, {row['law_over_fixed_L1'][2]:.3f}]"
        if "ours_over_ft_L2" in row:
            m, lo, hi = row["ours_over_ft_L2"]
            s += f"; ours ({row['ours_variant']}) {row['ours']:.1f} / FreeToken ({row['ft_variant']}) {row['ft']:.1f} = {m:.3f} [{lo:.3f}, {hi:.3f}] (L2)"
        if "ours_over_llama" in row:
            s += f"; llama.cpp {row['llama']:.1f}, ours/llama {row['ours_over_llama'][0]:.2f}x"
        print(s)
    out = dict(means=mean, rows=rows)
    if ("ft_auto", 1) in by:
        out["ft_auto"] = mean["ft_auto L1"]
        print(f"FreeToken as shipped: {mean['ft_auto L1']:.1f} tok/s")
    for f in ("tables.txt", "law_llama.json", "selection.txt"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            out[f] = open(p).read()
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

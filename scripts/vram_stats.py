"""Statistics for job 084 (RTX PRO 6000): the all-in-VRAM column, ours with every expert in its slots, and the speed
limit from the AIME-25 own-text routing traces.

A. Stock llama.cpp with every weight on the GPU (launches 1 and 2) and ours with 128 slots per layer (launch 1), 30
   AIME-25 problems x 256 tokens; ours / stock paired by problem (launch 1 vs launch 1).
B. scripts/speed_limit.py on the traces (la_aime25_{gptoss,qwen3}.bin) with the headline machine's probe (job 081), at
   the Table 1 budgets; the trace-simulated deployed policy's hit rate against the engine's measured hit rate in the
   Table 1 runs (jobs 080/081).

    python scripts/vram_stats.py --out prereg/vram_084.json --limit-out prereg/speed_limit_084.json
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from samehost_stats import boot_ratio  # noqa: E402

R = "/home/claude/gpu-branch/results"
ENGINE = {  # Table 1 run whose counters give the engine's hit rate
    ("gpt-oss-120b", 14): f"{R}/081_headline_law@vast/srv_g_ours_C14_law_L2.json",
    ("gpt-oss-120b", 32): f"{R}/081_headline_law@vast/srv_g_ours_C32_law_L2.json",
    ("gpt-oss-120b", 51): f"{R}/081_headline_law@vast/srv_g_ours_C51_law_L2.json",
    ("qwen3-30b-a3b-bf16", 16): f"{R}/081_headline_law@vast/srv_q_ours_C16_law_L2.json",
    ("qwen3-30b-a3b-bf16", 32): f"{R}/080_fetch_split@vast/srv_ours_C32_law_L1.json",
    ("qwen3-30b-a3b-bf16", 56): f"{R}/080_fetch_split@vast/srv_ours_C56_law_L2.json",
}
TABLE1 = {("gpt-oss-120b", 14): 69.9, ("gpt-oss-120b", 32): 109.2, ("gpt-oss-120b", 51): 152.5,
          ("qwen3-30b-a3b-bf16", 16): 40.0, ("qwen3-30b-a3b-bf16", 32): 63.1, ("qwen3-30b-a3b-bf16", 56): 108.2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=f"{R}/084b_vram_rerun@vast")
    ap.add_argument("--trace-dir", default=f"{R}/084c_gptoss_trace@vast", help="where the gpt-oss trace is (job 084c)")
    ap.add_argument("--out")
    ap.add_argument("--limit-out")
    a = ap.parse_args()
    by = {}
    for line in open(os.path.join(a.dir, "bs1.jsonl")):
        if line.strip():
            r = json.loads(line)
            by.setdefault((r["label"], r["launch"]), {})[r["problem"]] = r
    out = dict(vram={}, ours_c128={}, check={})
    for key, tag in (("gpt-oss-120b", "g"), ("qwen3-30b-a3b-bf16", "q")):
        L = [by[(f"{tag}_vram", L)] for L in (1, 2) if (f"{tag}_vram", L) in by]
        if not L:
            continue
        means = [float(np.mean([r["decode_tok_s"] for r in x.values()])) for x in L]
        out["vram"][key] = means[-1]
        out.setdefault("vram_launches", {})[key] = means
        print(f"{key}: all in VRAM {' / '.join(f'{m:.1f}' for m in means)} tok/s (launches), "
              f"VRAM {sorted({r.get('vram_used_gib') for r in L[0].values()})}")
        oc = by.get((f"{tag}_ours_C128", 1))
        if oc:
            P = sorted(set(oc) & set(L[0]))
            rt = boot_ratio(np.array([oc[p]["decode_tok_s"] for p in P]), np.array([L[0][p]["decode_tok_s"] for p in P]))
            out["ours_c128"][key] = dict(tok_s=float(np.mean([oc[p]["decode_tok_s"] for p in P])), over_stock=rt)
            st = os.path.join(a.dir, f"srv_{tag}_ours_C128.json")
            if os.path.exists(st):
                s = json.load(open(st))
                out["ours_c128"][key]["hit_rate"] = s["hit_rate"]
            print(f"   ours with 128 slots: {out['ours_c128'][key]['tok_s']:.1f} tok/s, / stock {rt[0]:.3f} [{rt[1]:.3f}, {rt[2]:.3f}]"
                  f", hit rate {out['ours_c128'][key].get('hit_rate')}")
    lim = {}
    for key, tag, budgets in (("gpt-oss-120b", "gptoss", "14,32,51"), ("qwen3-30b-a3b-bf16", "qwen3", "16,32,56")):
        la = next((p for p in (os.path.join(dd, f"route_aime25_{tag}.npz") for dd in (a.dir, a.trace_dir)) if p and os.path.exists(p)), None)
        if not la:
            print(f"{key}: no trace")
            continue
        tmp = os.path.join(os.path.dirname(la), f"speed_limit_{tag}.json")
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "speed_limit.py"), "--npz", la, "--model", key,
                        "--budgets", budgets, "--concur", f"{R}/081_headline_law@vast/concur.txt", "--json", tmp], check=True)
        lim[key] = json.load(open(tmp))
        for C in [int(x) for x in budgets.split(",")]:
            row = lim[key]["rows"][str(C)]
            eng = json.load(open(ENGINE[(key, C)]))
            d = row["deployed"]["hit_rate"] - eng["hit_rate"]
            frac = TABLE1[(key, C)] / row["opt"]["tok_s"]
            out["check"][f"{key} C{C}"] = dict(sim_hit=row["deployed"]["hit_rate"], engine_hit=eng["hit_rate"], diff=d,
                                                limit_tok_s=row["opt"]["tok_s"], ours_frac_of_limit=frac)
            print(f"   {key} C{C}: simulated deployed hit {row['deployed']['hit_rate']:.3f} vs engine {eng['hit_rate']:.3f} "
                  f"({100 * d:+.1f} points); limit {row['opt']['tok_s']:.1f} tok/s, ours {100 * frac:.0f}% of it")
    P = {}
    if "qwen3-30b-a3b-bf16" in out["vram"]:
        P["1 Qwen3 BF16 all in VRAM at 120-190 tok/s"] = 120 <= out["vram"]["qwen3-30b-a3b-bf16"] <= 190
    if "gpt-oss-120b" in out["vram"]:
        P["2 gpt-oss all in VRAM within 5% of 261.4"] = abs(out["vram"]["gpt-oss-120b"] / 261.4 - 1) <= 0.05
    if len(out["ours_c128"]) == 2:
        P["3 ours with 128 slots within 5% of stock all in VRAM, both models"] = all(abs(v["over_stock"][0] - 1) <= 0.05 for v in out["ours_c128"].values())
    if len(out["check"]) == 6:
        P["4a simulated hit rate within 5 points of the engine's at every budget"] = all(abs(v["diff"]) <= 0.05 for v in out["check"].values())
        P["4b ours at 20-50% of the speed limit at every budget"] = all(0.2 <= v["ours_frac_of_limit"] <= 0.5 for v in out["check"].values())
    out["predictions"] = P
    for k, x in P.items():
        print(f"prediction {k}: {'HELD' if x else 'FAILED'}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    if a.limit_out and lim:
        json.dump(lim, open(a.limit_out, "w"), indent=1)


if __name__ == "__main__":
    main()

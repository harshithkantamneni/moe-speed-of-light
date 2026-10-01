"""Statistics for the grid, jobs 091 (RTX 4090) and 092 (RTX 3090 + Mixtral-8x7B): Table 1's protocol on 24 GB cards.

Per cell that fit the card: ours with the law's table, FreeToken (backend carried over from job 081) and stock llama.cpp
at matched expert count, ours -> FreeToken -> llama.cpp in launch 1 and FreeToken -> ours in launch 2 (the comparison
uses launch 2; launch 1 against launch 2 is the order check). Ratios of mean speeds, paired by problem, 95% percentile
interval over 10,000 bootstrap resamples (seed 0). The speed limit of each cell is recomputed for the card: the exact
per-layer optimum's reads per token (prereg/speed_limit_v2.json, the same trace as Table 1), the host rate from this
machine's probe (the highest rate any method reached, as scripts/speed_limit.py) and the card's datasheet bandwidth.
Mixtral (092 only): ours against llama.cpp at C = 2 and C = 4, launch 1.

    python scripts/grid_stats.py --job 091 --out prereg/grid_091.json
    python scripts/grid_stats.py --job 092 --out prereg/grid_092.json
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from samehost_stats import boot_ratio  # noqa: E402
from scripts.speed_limit import MODELS, host_rates, limit  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = "/home/claude/gpu-branch/results"
JOBS = {"091": dict(dir="091_grid_4090@vast", card="RTX 4090", b_gpu=1008e9),
        "092": dict(dir="092_grid_3090@vast", card="RTX 3090", b_gpu=936e9)}
CELLS = [("gpt-oss-120b", "11%", "g_ours_C14", "g_ft_hybrid_r0.111", "g_llama_n32", 14),
         ("gpt-oss-120b", "25%", "g_ours_C32", "g_ft_hybrid_r0.25", "g_llama_n27", 32),
         ("gpt-oss-120b", "40%", "g_ours_C51", "g_ft_offload_r0.40", "g_llama_n22", 51),
         ("Qwen3-30B-A3B", "12.5%", "q_ours_C16", "q_ft_hybrid_r0.125", "q_llama_n42", 16),
         ("Qwen3-30B-A3B", "25%", "q_ours_C32", "q_ft_hybrid_r0.25", "q_llama_n36", 32),
         ("Qwen3-30B-A3B", "43.75%", "q_ours_C56", "q_ft_offload_r0.4375", "q_llama_n27", 56)]
MIXTRAL = [("Mixtral-8x7B", "C2", "m_ours_C2", "m_llama_n24"), ("Mixtral-8x7B", "C4", "m_ours_C4", "m_llama_n16")]
LIMIT_MODEL = {"gpt-oss-120b": "gpt-oss-120b", "Qwen3-30B-A3B": "qwen3-30b-a3b-bf16"}


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


def card(D):
    rows = list(csv.reader(open(os.path.join(D, "gpu.csv"))))
    hdr = [h.strip() for h in rows[0]]
    val = {h: rows[1][i].strip() for i, h in enumerate(hdr)}
    num = lambda s: float(s.split()[0])  # noqa: E731
    out = dict(name=val.get("name"), memory_mib=num(val["memory.total [MiB]"]), mem_clock_mhz=num(val["clocks.max.memory [MHz]"]),
               sm_clock_mhz=num(val["clocks.max.sm [MHz]"]), power_limit_w=num(val["power.limit [W]"]),
               pcie_gen=val.get("pcie.link.gen.current"), pcie_width=val.get("pcie.link.width.current"))
    g = os.path.join(D, "bw_gate.txt")
    if os.path.exists(g):
        m = re.search(r"device read 1 GiB: *([\d.]+)", open(g).read())
        out["device_read_gbs"] = float(m.group(1)) if m else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=list(JOBS))
    ap.add_argument("--out")
    a = ap.parse_args()
    J = JOBS[a.job]
    D = f"{R}/{J['dir']}"
    by = {}
    for line in open(os.path.join(D, "bs1.jsonl")):
        if line.strip():
            r = json.loads(line)
            by.setdefault((r["label"], r["launch"]), {})[r["problem"]] = r["decode_tok_s"]
    mean = {f"{k[0]} L{k[1]}": float(np.mean(list(v.values()))) for k, v in by.items()}
    n = {f"{k[0]} L{k[1]}": len(v) for k, v in by.items()}
    for k in sorted(mean):
        print(f"  {k:30s} {mean[k]:7.2f} tok/s  n={n[k]}")

    def ratio(x, y):
        P = sorted(set(by[x]) & set(by[y]))
        return boot_ratio(np.array([by[x][p] for p in P]), np.array([by[y][p] for p in P]))
    # the machine: probe, card, the law's tables
    txt = open(os.path.join(D, "concur.txt")).read()
    b_host = host_rates(txt)
    probes = dict(cpu_by_threads={int(t): float(v) for t, v in re.findall(r"cpu_read_gbs t=(\d+) ([\d.]+)", txt)},
                  pcie_zerocopy=[float(x) for x in re.findall(r"pcie_zerocopy_gbs chunk=16MB blocks=64 ([\d.]+)", txt)],
                  pcie_copyengine=[float(x) for x in re.findall(r"pcie_copyengine_gbs chunk=16MB ([\d.]+)", txt)],
                  both=[float(x) for x in re.findall(r"sum ([\d.]+)", txt)], b_host_max=b_host / 1e9)
    tables = {}
    for name in ("gptoss", "qwen3", "mixtral"):
        p = os.path.join(D, f"fetch_table_law_{name}.json")
        if os.path.exists(p):
            tables[name] = json.load(open(p))
    sl = json.load(open(os.path.join(ROOT, "prereg", "speed_limit_v2.json")))
    B = listing_b()
    rows = []
    for model, budget, op, ft, ll, C in CELLS:
        law = f"{op}_law"
        if (law, 2) not in by:
            if (law, 1) in by:
                print(f"{model} {budget}: launch 1 only")
            continue
        row = dict(model=model, budget=budget, C=C, ft_variant=ft.split("_")[2],
                   ours=mean[f"{law} L2"], ours_L1=mean[f"{law} L1"], ours_over_ft_L2=None, ours_over_ft_L1=None)
        if (ft, 2) in by:
            row.update(ft=mean[f"{ft} L2"], ft_L1=mean.get(f"{ft} L1"), ours_over_ft_L2=ratio((law, 2), (ft, 2)))
            if (ft, 1) in by:
                row["ours_over_ft_L1"] = ratio((law, 1), (ft, 1))
                row["order_diff"] = row["ours_over_ft_L1"][0] / row["ours_over_ft_L2"][0] - 1
        if (ll, 1) in by:
            row.update(llama=mean[f"{ll} L1"], ours_over_llama=ratio((law, 2), (ll, 1)))
        hp = os.path.join(D, f"srv_{law}_L2.json")
        if os.path.exists(hp):
            e = json.load(open(hp))
            row["cache"] = dict(hit_rate=e["hit_rate"], misses_per_token=e["misses"] / e["steps"], admits_per_token=e["admits"] / e["steps"], table=e.get("fetch_table"))
        b = B[(model, budget)]
        row["listing_b"] = b
        row["vs_b"] = dict(ours=row["ours"] / b["ours"] - 1, llama=row["llama"] / b["llama"] - 1 if "llama" in row else None,
                           ft=row["ft"] / b["ft"] - 1 if "ft" in row else None,
                           ratio_diff=row["ours_over_ft_L2"][0] - b["ratio"] if row["ours_over_ft_L2"] else None)
        # the limit on this card: exact optimum reads (same trace), this host's best rate, the card's datasheet
        lm = LIMIT_MODEL[model]
        srow = sl["models"][lm]["rows"][str(C)]
        m = MODELS[lm]
        L, k = sl["models"][lm]["L"], sl["models"][lm]["k"]
        reads = srow["reads_per_token"]["exact"]
        t, c_at = limit(reads, L * k, m["S"], m["D"], b_host, J["b_gpu"])
        v = np.array(list(by[(law, 2)].values()))
        rng = np.random.default_rng(0)
        bm = v[rng.integers(0, len(v), size=(10000, len(v)))].mean(1)
        row["ours_ci"] = [float(v.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]
        row["limit"] = dict(reads_per_token=reads, b_host_gbs=b_host / 1e9, b_gpu_gbs=J["b_gpu"] / 1e9, t_ms=1e3 * t, tok_s=1 / t,
                            cpu_at_limit=c_at, gpu_only_ms=1e3 * (m["D"] + L * k * m["S"]) / J["b_gpu"],
                            frac=dict(ours=row["ours"] * t, ft=row["ft"] * t if "ft" in row else None, llama=row["llama"] * t if "llama" in row else None),
                            frac_ours_ci=[row["ours_ci"][1] * t, row["ours_ci"][2] * t],
                            headline_frac_ours=srow["frac_of_limit"]["exact"]["ours"])
        rows.append(row)
        rf = row["ours_over_ft_L2"]
        print(f"{model} {budget}: ours {row['ours']:.1f} / FreeToken ({row['ft_variant']}) {row.get('ft', float('nan')):.1f} = "
              + (f"{rf[0]:.3f} [{rf[1]:.3f}, {rf[2]:.3f}]" if rf else "--")
              + f" (B {b['ratio']:.3f}); ours/llama {row['ours_over_llama'][0]:.2f} [{row['ours_over_llama'][1]:.2f}, {row['ours_over_llama'][2]:.2f}]"
              if "ours_over_llama" in row else "")
        print(f"    vs B: ours {100 * row['vs_b']['ours']:+.0f}%; order {100 * row.get('order_diff', float('nan')):+.1f}%; "
              f"limit {row['limit']['tok_s']:.0f} tok/s (host {b_host / 1e9:.0f} GB/s, GPU {J['b_gpu'] / 1e9:.0f}), ours {100 * row['limit']['frac']['ours']:.0f}% "
              f"of it (headline machine {100 * row['limit']['headline_frac_ours']:.0f}%)")
    mix = []
    for model, cell, op, ll in MIXTRAL:
        law = f"{op}_law"
        if (law, 1) in by and (ll, 1) in by:
            r = dict(model=model, cell=cell, C=int(cell[1:]), ours=mean[f"{law} L1"], llama=mean[f"{ll} L1"], ours_over_llama=ratio((law, 1), (ll, 1)))
            hp = os.path.join(D, f"srv_{law}_L1.json")
            if os.path.exists(hp):
                e = json.load(open(hp))
                r["cache"] = dict(hit_rate=e["hit_rate"], misses_per_token=e["misses"] / e["steps"], admits_per_token=e["admits"] / e["steps"], table=e.get("fetch_table"),
                                  pinned_hit_rate=r["C"] / 8)
            mix.append(r)
            print(f"Mixtral {cell}: ours {r['ours']:.1f} / llama.cpp {r['llama']:.1f} = {r['ours_over_llama'][0]:.3f} "
                  f"[{r['ours_over_llama'][1]:.3f}, {r['ours_over_llama'][2]:.3f}]")
    out = dict(job=a.job, card=J["card"], means=mean, n=n, rows=rows, mixtral=mix, probes=probes, tables=tables, gpu=card(D))
    for f in ("gate.txt", "oneline.txt", "tables.txt", "skipped.txt", "runs.txt", "ram.txt"):
        p = os.path.join(D, f)
        if os.path.exists(p):
            out[f] = open(p).read()
    for f in glob.glob(f"{D}/GPU-*.json"):
        c = json.load(open(f))["ceilings"]
        out["ft_probe"] = dict(cpu=c["cpu_stream_read_gbs"], pcie=c["pcie_linear_h2d_gbs"])
    print("card:", out["gpu"], "| tables:", {k: v["table"] for k, v in tables.items()})
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

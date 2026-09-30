"""Where the seconds go on the headline machine (RTX 5090 + Ryzen 9 9950X3D, Vast listing B): for every Table 1 cell,
from the speed limit to the measured time per token, one cause at a time.

  0 speed limit      min over c of max(GPU datasheet time, host reads of the optimum at the highest measured host rate)
                     (scripts/speed_limit.py, eq. 2 of the paper)
  1 no overlap       GPU datasheet time + the optimum's host time, in sequence
  2 no foresight     the best online policy's reads instead of the optimum's
  3 policy, paths    the engine's own reads per token (its counters: misses run on the CPU, fetches and background
                     admissions over PCIe) through the host-memory law with the machine's probe (B_c, B_p, B_cp)
  4 host work        per-token host time of the server, from the engine's host-side timer
  5 GPU at batch 1   the rest: measured time minus all of the above (GPU kernels below datasheet rate, idle gaps,
                     latencies the law leaves out)

Inputs: prereg/speed_limit_084.json (traces of the measured text), the engine counters of the Table 1 runs (jobs
080/081) and the machine's probe (job 081 concur.txt).

    python scripts/gap_listingb.py [--json prereg/gap_listingb.json] [--fig paper/figs/gap.pdf]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/home/claude/gpu-branch/jobs/ec2")
from fetch_table import bandwidths  # noqa: E402

R = "/home/claude/gpu-branch/results"
ROOT = os.path.join(os.path.dirname(__file__), "..")
CELLS = [  # model key, label, C, engine counters of the Table 1 run, measured tok/s key
    ("gpt-oss-120b", "gpt-oss 11%", 14, f"{R}/081_headline_law@vast/srv_g_ours_C14_law_L2.json"),
    ("gpt-oss-120b", "gpt-oss 25%", 32, f"{R}/081_headline_law@vast/srv_g_ours_C32_law_L2.json"),
    ("gpt-oss-120b", "gpt-oss 40%", 51, f"{R}/081_headline_law@vast/srv_g_ours_C51_law_L2.json"),
    ("qwen3-30b-a3b-bf16", "Qwen3 12.5%", 16, f"{R}/081_headline_law@vast/srv_q_ours_C16_law_L2.json"),
    ("qwen3-30b-a3b-bf16", "Qwen3 25%", 32, f"{R}/080_fetch_split@vast/srv_ours_C32_law_L1.json"),
    ("qwen3-30b-a3b-bf16", "Qwen3 43.75%", 56, f"{R}/080_fetch_split@vast/srv_ours_C56_law_L2.json"),
]


def measured():
    h = json.load(open(os.path.join(ROOT, "prereg", "headline_081.json")))
    s = json.load(open(os.path.join(ROOT, "prereg", "split_080.json")))["means"]
    m = {f"{'gpt-oss' if r['model'].startswith('gpt') else 'Qwen3'} {r['budget']}": r["ours_L2"] for r in h["rows"]}
    m["Qwen3 25%"] = s["ours_C32_law L1"]
    m["Qwen3 43.75%"] = s["ours_C56_law L2"]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", default=os.path.join(ROOT, "prereg", "speed_limit_084.json"))
    ap.add_argument("--json")
    ap.add_argument("--fig")
    a = ap.parse_args()
    sl = json.load(open(a.limit))
    txt = open(f"{R}/081_headline_law@vast/concur.txt").read()
    bc, bp, bb = [x * 1e9 for x in bandwidths(txt, 14)]
    meas = measured()
    out = []
    for key, label, C, stats in CELLS:
        L = sl[key]
        row = L["rows"][str(C)]
        S, D, Lk, bh, bg = L["S"], L["D"], L["L"] * L["k"], L["B_host"], L["B_gpu"]
        gpu = lambda c: (D + (Lk - c) * S) / bg  # noqa: E731
        st = json.load(open(stats))
        n = st["steps"]
        cpu = (st["misses"] - st["fetches"]) / n
        link = (st["fetches"] + st["admits"] + st.get("prefetches", 0)) / n
        host = max(cpu * S / bc, link * S / bp, (cpu + link) * S / bb)
        hu = st["host_us_per_step"]
        hw = (hu["app"] + hu["pre"] + hu["inputs"] + hu["launch"] + hu["post"]) * 1e-6
        t = [row["opt"]["t_ms"] * 1e-3,
             gpu(row["opt"]["cpu"]) + row["opt"]["reads"] * S / bh,
             gpu(row["online"]["cpu"]) + row["online"]["reads"] * S / bh,
             gpu(cpu) + host]
        t.append(t[-1] + hw)
        T = 1 / meas[label]
        steps = [("speed limit", t[0]), ("no overlap", t[1] - t[0]), ("no foresight", t[2] - t[1]),
                 ("policy and read paths", t[3] - t[2]), ("host work", hw), ("GPU below datasheet (net)", T - t[4])]
        out.append(dict(cell=label, C=C, measured_tok_s=meas[label], measured_ms=T * 1e3, engine_reads=cpu + link,
                        engine_cpu=cpu, engine_link=link, steps_ms=[(k, v * 1e3) for k, v in steps]))
        print(f"{label:13s} measured {meas[label]:6.1f} tok/s = {T * 1e3:6.2f} ms; reads/token engine {cpu + link:6.1f} "
              f"(cpu {cpu:5.1f}, link {link:5.1f}), best online {row['online']['reads']:6.1f}, optimum {row['opt']['reads']:6.1f}")
        acc = 0.0
        for k, v in steps:
            acc += v
            print(f"    {k:24s} {v * 1e3:+6.2f} ms -> {acc * 1e3:6.2f} ms ({1 / acc:6.1f} tok/s)")
    if a.json:
        json.dump(dict(B=[bc / 1e9, bp / 1e9, bb / 1e9], cells=out), open(a.json, "w"), indent=1)
    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = ["#2b6cb0", "#90cdf4", "#f6ad55", "#e53e3e", "#9f7aea", "#718096"]
        fig, ax = plt.subplots(figsize=(3.4, 2.5))
        for i, r in enumerate(out[::-1]):
            left = 0.0
            for j, (k, v) in enumerate(r["steps_ms"]):
                ax.barh(i, v, left=left, color=cols[j], height=0.62, label=k if i == 0 else None)
                left += v
            ax.text(left + 0.3, i, f"{r['measured_tok_s']:.0f} tok/s", va="center", fontsize=6)
        ax.set_yticks(range(len(out)))
        ax.set_yticklabels([r["cell"] for r in out[::-1]], fontsize=6.5)
        ax.set_xlabel("ms per token", fontsize=7)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.legend(fontsize=5.5, frameon=False, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3, columnspacing=0.8, handlelength=1.2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlim(0, max(r["measured_ms"] for r in out) * 1.22)
        fig.tight_layout()
        fig.savefig(a.fig)


if __name__ == "__main__":
    main()

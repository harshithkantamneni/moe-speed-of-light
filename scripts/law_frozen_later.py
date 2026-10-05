"""The host-memory law with its constants frozen (G = 4.819 ms for gpt-oss-120b, 4.3 ms for Qwen3-30B-A3B; the engine's
probe rates at the helper count) scored on the RTX 5090 hosts measured after the blind test's jobs (069c-077): the
stock-clock host S (job 089, both launches of our cache at the six cells of Table 1) and the oracle hosts O1-O3 (jobs
093-095, the online policy's teacher-forced runs at the same six cells). The engine's own counters give the reads: on
the CPU = misses - fetches. Two definitions of the link's bytes are scored:

  blind  link = fetches (+ prefetches): the blind test's registered definition (jobs/ec2/law_predict.py), which treats
         the background admission copies as off the critical path;
  adm    link = fetches + admissions: every host read that crosses the link.

and a naive baseline with one read path, every host read at the best single probed rate (bytes over bandwidth):
T = G + (cpu + fetches) / max(B_c, B_p). Errors are predicted over measured
speed, minus one (the convention of the blind test's table). These are predictions in the sense that the constants
were frozen before the hosts were rented, not in the sense of the blind test (the job did not write them before the
run).

    python scripts/law_frozen_later.py --out prereg/law_frozen_later.json
"""
import argparse
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(os.environ.get("MOSL_GPU_BRANCH", os.environ.get("MOSL_GPU_BRANCH", "/home/claude/gpu-branch")), "jobs", "ec2"))
from fetch_table import bandwidths  # noqa: E402

R_ = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
G = {"g": 4.819346e-3, "q": 4.3e-3}
S = {"g": 13253760, "q": 9437184}
CELLS = {"g": [14, 32, 51], "q": [16, 32, 56]}
NAME = {"g": "gpt-oss-120b", "q": "Qwen3-30B-A3B"}
BUDGET = {("g", 14): "11%", ("g", 32): "25%", ("g", 51): "40%", ("q", 16): "12.5%", ("q", 32): "25%", ("q", 56): "43.75%"}


def law(m, cpu, link, bc, bp, bb):
    return G[m] + max(cpu * S[m] / (bc * 1e9), link * S[m] / (bp * 1e9), (cpu + link) * S[m] / (bb * 1e9))


def host_rates(job):
    txt = open(f"{R_}/{job}/concur.txt").read()
    cores = int(re.search(r"usable physical cores (\d+)", open(f"{R_}/{job}/cores.txt").read()).group(1))
    H = cores - 2 if cores > 4 else 2
    return bandwidths(txt, H), H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = []

    def add(host, m, budget, launch, cpu, fetches, admits, B, meas):
        bc, bp, bb = B
        t_blind = law(m, cpu, fetches, bc, bp, bb)
        t_adm = law(m, cpu, fetches + admits, bc, bp, bb)
        t_naive = G[m] + (cpu + fetches) * S[m] / (max(bc, bp) * 1e9)   # one path: every host read at the best single rate
        rows.append(dict(host=host, model=NAME[m], budget=budget, launch=launch, cpu=cpu, link=fetches, admits=admits, B=B,
                         predicted_tok_s=1 / t_blind, measured_tok_s=meas, error=(1 / t_blind) / meas - 1,
                         predicted_adm_tok_s=1 / t_adm, error_adm=(1 / t_adm) / meas - 1,
                         predicted_naive_tok_s=1 / t_naive, error_naive=(1 / t_naive) / meas - 1))

    # the oracle hosts: the online policy's base runs (ec-bench timing, teacher-forced)
    for job, tag in (("093_foresight@vast", "093 (9950X)"), ("094_foresight_bytes@vast", "094 (7950X)"), ("095_single_read@vast", "095 (9950X)")):
        if not os.path.exists(f"{R_}/{job}/concur.txt"):
            continue
        B, H = host_rates(job)
        f = json.load(open(os.path.join(ROOT, "prereg", "foresight_" + job[:3] + ".json")))
        for c in f["cells"]:
            m = "g" if c["model"].startswith("gpt") else "q"
            b = c["runs"]["base"]
            add(tag, m, BUDGET[(m, c["C"])], "ec-bench", b["misses_per_token"] - b["fetches_per_token"], b["fetches_per_token"],
                b["admits_per_token"], B, b["mean"])
    # the stock-clock host: server runs, both launches, counters from the server's stats files
    job = "089_headline_stockclock@vast"
    B, H = host_rates(job)
    sc = json.load(open(os.path.join(ROOT, "prereg", "stockclock_089.json")))
    for m in ("g", "q"):
        for C in CELLS[m]:
            for L in ("L1", "L2"):
                key = f"{m}_ours_C{C}_law {L}"
                if key not in sc["means"]:
                    continue
                st = json.load(open(f"{R_}/{job}/srv_{m}_ours_C{C}_law_{L}.json"))
                n = st["steps"]
                meas = sc["means"][key] if not isinstance(sc["means"][key], dict) else sc["means"][key].get("mean")
                add("089 (9950X, stock clock)", m, BUDGET[(m, C)], L, (st["misses"] - st["fetches"]) / n, st["fetches"] / n,
                    st["admits"] / n, B, meas)

    def summ(key):
        e = np.array([abs(r[key]) for r in rows]); sg = np.array([r[key] for r in rows])
        return dict(n=len(rows), median_abs_pct=float(100 * np.median(e)), p90_abs_pct=float(100 * np.percentile(e, 90)),
                    max_abs_pct=float(100 * e.max()), median_signed_pct=float(100 * np.median(sg)),
                    within_5pct=int((e <= 0.05).sum()), within_10pct=int((e <= 0.10).sum()))
    summary = summ("error")
    summary_adm = summ("error_adm")
    summary_naive = summ("error_naive")
    # the signed error by model and budget (the bias the reviews asked for)
    bias = {}
    for r in rows:
        bias.setdefault(f"{r['model']} {r['budget']}", []).append(r["error"])
    bias = {k: dict(median_pct=float(100 * np.median(v)), min_pct=float(100 * min(v)), max_pct=float(100 * max(v)), n=len(v)) for k, v in bias.items()}
    for r in rows:
        print(f"{r['host']:26s} {r['model']:14s} {r['budget']:7s} {r['launch']:8s} cpu {r['cpu']:6.1f} fetch {r['link']:5.1f} adm {r['admits']:4.1f}  "
              f"pred {r['predicted_tok_s']:6.1f} meas {r['measured_tok_s']:6.1f}  err {100 * r['error']:+5.1f}%  adm {100 * r['error_adm']:+5.1f}%  naive {100 * r['error_naive']:+5.1f}%")
    print("blind definition", summary); print("with admissions", summary_adm); print("naive", summary_naive)
    for k, v in bias.items():
        print(f"  {k:22s} median {v['median_pct']:+5.1f}%  [{v['min_pct']:+5.1f}, {v['max_pct']:+5.1f}]  n={v['n']}")
    if a.out:
        json.dump(dict(rows=rows, summary=summary, summary_adm=summary_adm, summary_naive=summary_naive, bias=bias,
                       G_ms={k: 1e3 * v for k, v in G.items()}, note=__doc__), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

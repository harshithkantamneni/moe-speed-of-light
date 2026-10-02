"""The host-memory law with its constants frozen (G = 4.819 ms for gpt-oss-120b, 4.3 ms for Qwen3-30B-A3B; the engine's
probe rates at the helper count) scored on the RTX 5090 hosts measured after the blind test's jobs (069c-077): the
stock-clock host (job 089, both launches of our cache at the six cells of Table 1) and the two oracle hosts (jobs 093
and 094, the online policy's teacher-forced runs at the same six cells). The counters are the engine's own: experts run
on the CPU = misses - fetches, over the link = fetches + admissions. These are predictions in the sense that the
constants were frozen before the hosts were rented, not in the sense of the blind test (the job did not write them
before the run).

    python scripts/law_frozen_later.py --out prereg/law_frozen_later.json
"""
import argparse
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, "/home/claude/gpu-branch/jobs/ec2")
from fetch_table import bandwidths  # noqa: E402

R_ = "/home/claude/gpu-branch/results"
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
    # the oracle hosts: the online policy's base runs (ec-bench timing, teacher-forced)
    for job, tag in (("093_foresight@vast", "093 (9950X)"), ("094_foresight_bytes@vast", "094 (7950X)")):
        (bc, bp, bb), H = host_rates(job)
        f = json.load(open(os.path.join(ROOT, "prereg", "foresight_" + job[:3] + ".json")))
        for c in f["cells"]:
            m = "g" if c["model"].startswith("gpt") else "q"
            b = c["runs"]["base"]
            cpu, link = b["misses_per_token"] - b["fetches_per_token"], b["fetches_per_token"] + b["admits_per_token"]
            t = law(m, cpu, link, bc, bp, bb)
            rows.append(dict(host=tag, model=NAME[m], budget=BUDGET[(m, c["C"])], launch="ec-bench", cpu=cpu, link=link, B=(bc, bp, bb),
                             predicted_tok_s=1 / t, measured_tok_s=b["mean"], error=t / (1e3 / b["mean"] * 1e-3) - 1))
    # the stock-clock host: server runs, both launches, counters from the server's stats files
    job = "089_headline_stockclock@vast"
    (bc, bp, bb), H = host_rates(job)
    sc = json.load(open(os.path.join(ROOT, "prereg", "stockclock_089.json")))
    for m in ("g", "q"):
        for C in CELLS[m]:
            for L in ("L1", "L2"):
                key = f"{m}_ours_C{C}_law {L}"
                if key not in sc["means"]:
                    continue
                st = json.load(open(f"{R_}/{job}/srv_{m}_ours_C{C}_law_{L}.json"))
                n = st["steps"]
                cpu, link = (st["misses"] - st["fetches"]) / n, (st["fetches"] + st["admits"]) / n
                meas = sc["means"][key] if not isinstance(sc["means"][key], dict) else sc["means"][key].get("mean")
                t = law(m, cpu, link, bc, bp, bb)
                rows.append(dict(host="089 (9950X, stock clock)", model=NAME[m], budget=BUDGET[(m, C)], launch=L, cpu=cpu, link=link, B=(bc, bp, bb),
                                 predicted_tok_s=1 / t, measured_tok_s=meas, error=t / (1 / meas) - 1))
    errs = np.array([abs(r["error"]) for r in rows])
    signed = np.array([r["error"] for r in rows])
    summary = dict(n=len(rows), median_abs_pct=float(100 * np.median(errs)), p90_abs_pct=float(100 * np.percentile(errs, 90)),
                   max_abs_pct=float(100 * errs.max()), median_signed_pct=float(100 * np.median(signed)),
                   within_5pct=int((errs <= 0.05).sum()), within_10pct=int((errs <= 0.10).sum()))
    for r in rows:
        print(f"{r['host']:26s} {r['model']:14s} {r['budget']:7s} {r['launch']:8s} cpu {r['cpu']:6.1f} link {r['link']:5.1f}  predicted {r['predicted_tok_s']:6.1f}  measured {r['measured_tok_s']:6.1f}  error {100 * r['error']:+5.1f}%")
    print(summary)
    if a.out:
        json.dump(dict(rows=rows, summary=summary, G_ms={k: 1e3 * v for k, v in G.items()}, note=__doc__), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

"""Why the bytes-optimal oracle's reads stay above the optimum's (job 094): MIN with bypass simulated per layer on the
measured routing traces under a copy latency of d decode steps. An admitted expert is resident from step t + d; until
then a request for it is a miss (the CPU runs it), it is not re-admitted, and its slot cannot be evicted. The oracle
looks ahead within the sequence only and residents carry over between sequences, as the engine's does. d = 1 is the
ideal (admitted after serving step t on the CPU, resident at t + 1); the engine publishes a landed copy to the slot
map two steps after issuing it, and pacing adds about one more.

    python scripts/foresight_latency_sim.py --out prereg/foresight_latency_sim.json
"""
import argparse
import json
import os

import numpy as np
from numba import njit

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R_ = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
TRACES = {"gpt-oss-120b": (f"{R_}/084c_gptoss_trace@vast/route_aime25_gptoss.npz", [14, 32, 51], {14: "11%", 32: "25%", 51: "40%"}),
          "qwen3-30b-a3b-bf16": (f"{R_}/084b_vram_rerun@vast/route_aime25_qwen3.npz", [16, 32, 56], {16: "12.5%", 32: "25%", 56: "43.75%"})}
INF = 1 << 60


@njit(cache=True)
def next_use(R, E):
    T, k = R.shape
    last = np.full(E, INF, dtype=np.int64)
    nxt = np.empty((T, k), dtype=np.int64)
    for t in range(T - 1, -1, -1):
        for j in range(k):
            nxt[t, j] = last[R[t, j]]
        for j in range(k):
            last[R[t, j]] = t
    return nxt


@njit(cache=True)
def sim(R, E, cap, d, seqs):
    T, k = R.shape
    state = np.zeros(E, dtype=np.int64)        # 0 absent, 1 resident, 2 loading
    ready = np.zeros(E, dtype=np.int64)
    nu = np.full(E, INF, dtype=np.int64)
    nxt = next_use(R, E)
    miss = np.zeros(T, dtype=np.int64); adm = np.zeros(T, dtype=np.int64); inflight = np.zeros(T, dtype=np.int64)
    size = 0
    missed = np.zeros(k, dtype=np.int64); missnu = np.zeros(k, dtype=np.int64)
    for t in range(T):
        for e in range(E):
            if state[e] == 2 and ready[e] <= t:
                state[e] = 1
        for j in range(k):
            e = R[t, j]
            n = nxt[t, j]
            if n < INF and seqs[n] != seqs[t]:
                n = INF
            nu[e] = n
        nm = 0
        for j in range(k):
            e = R[t, j]
            if state[e] == 1:
                continue
            if state[e] == 2:
                inflight[t] += 1
            missed[nm] = e; missnu[nm] = nu[e]; nm += 1
        miss[t] = nm
        order = np.argsort(missnu[:nm])
        for q in range(nm):
            e = missed[order[q]]
            if state[e] != 0 or nu[e] >= INF:
                continue
            if size < cap:
                state[e] = 2 if d > 0 else 1; ready[e] = t + d; size += 1; adm[t] += 1
                continue
            victim = -1; best = -1
            for c in range(E):
                if state[c] == 1 and (victim == -1 or nu[c] > best):
                    best, victim = nu[c], c
            if victim == -1 or best <= nu[e]:
                continue
            state[victim] = 0; size -= 1
            state[e] = 2 if d > 0 else 1; ready[e] = t + d; size += 1; adm[t] += 1
    return miss, adm, inflight


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--engine", default=os.path.join(ROOT, "prereg", "foresight_094.json"))
    a = ap.parse_args()
    eng = json.load(open(a.engine)) if os.path.exists(a.engine) else None
    out = {"models": {}, "note": __doc__}
    for model, (path, caps, budget) in TRACES.items():
        z = np.load(path); act = z["act"]; seqs = z["seq"].astype(np.int64); E = int(z["n_expert"]); L = act.shape[1]; T = act.shape[0]
        out["models"][model] = {"trace": path, "T": int(T), "L": int(L), "E": E, "cells": {}}
        for cap in caps:
            row = {"budget": budget[cap], "by_d": {}}
            for d in (0, 1, 2, 3, 4, 5):
                m = ad = im = 0
                for il in range(L):
                    R = np.ascontiguousarray(act[:, il, :].astype(np.int64))
                    miss, adm, inf = sim(R, E, cap, d, seqs)
                    m += int(miss.sum()); ad += int(adm.sum()); im += int(inf.sum())
                row["by_d"][str(d)] = dict(misses_per_token=m / T, admissions_per_token=ad / T, inflight_misses_per_token=im / T)
            if eng:
                for c in eng["cells"]:
                    if c["model"] == model and c["C"] == cap and "bypass_w0" in c["runs"]:
                        r = c["runs"]["bypass_w0"]
                        row["engine"] = dict(misses_per_token=r["misses_per_token"], admissions_per_token=r["admits_per_token"], opt_reads_per_token=c["opt_reads_per_token"])
                        # the latency whose simulated misses are nearest the engine's (log interpolation between integers)
                        ms = [row["by_d"][str(d)]["misses_per_token"] for d in range(6)]
                        fit = None
                        for d in range(1, 5):
                            if ms[d] <= r["misses_per_token"] <= ms[d + 1]:
                                fit = d + (r["misses_per_token"] - ms[d]) / (ms[d + 1] - ms[d]); break
                        row["engine"]["d_fit"] = fit
            out["models"][model]["cells"][str(cap)] = row
            e = row.get("engine", {})
            print(f"{model} C={cap} ({budget[cap]}): ideal (d<=1) {row['by_d']['1']['misses_per_token']:.1f}  d=2 {row['by_d']['2']['misses_per_token']:.1f}  "
                  f"d=3 {row['by_d']['3']['misses_per_token']:.1f}  d=4 {row['by_d']['4']['misses_per_token']:.1f}  "
                  f"engine {e.get('misses_per_token', float('nan')):.1f} (optimum {e.get('opt_reads_per_token', float('nan')):.1f}; d fit {e.get('d_fit')})")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

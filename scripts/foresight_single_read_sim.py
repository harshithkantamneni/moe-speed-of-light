"""Predictions for job 095: the single-read oracles simulated per layer on the measured routing traces with the engine's
semantics (lookahead within the sequence; residents carried over between sequences).

  fetch    MIN with bypass, each admitted miss fetched into the victim's slot and served from it this step (one read on
           the critical path); bypassed misses run on the CPU. Reads per token = misses (= the optimum's reads up to
           the within-sequence lookahead).
  lead-d   Belady prefetch of experts whose next use is at least d steps away (the copy lands d steps after issue), the
           MIN eviction rule (victim = furthest next use; refuse if it is needed sooner than the candidate); an expert
           needed sooner than d steps misses on the CPU and is not admitted then. Reads = CPU misses + copies.
  both-d   lead-d plus fetch for this step's misses (MIN rule).

    python scripts/foresight_single_read_sim.py --out prereg/foresight_single_read_sim.json
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
OPT = {("gpt-oss-120b", 14): 38.3, ("gpt-oss-120b", 32): 15.3, ("gpt-oss-120b", 51): 6.8,
       ("qwen3-30b-a3b-bf16", 16): 96.5, ("qwen3-30b-a3b-bf16", 32): 43.4, ("qwen3-30b-a3b-bf16", 56): 13.7}
INF = 1 << 60


@njit(cache=True)
def next_use_table(R, E, seqs):
    """nxt[t, j]: next step > t within the same sequence at which R[t, j] is requested (INF if none)."""
    T, k = R.shape
    last = np.full(E, INF, dtype=np.int64)
    nxt = np.empty((T, k), dtype=np.int64)
    for t in range(T - 1, -1, -1):
        for j in range(k):
            n = last[R[t, j]]
            nxt[t, j] = n if (n < INF and seqs[n] == seqs[t]) else INF
        for j in range(k):
            last[R[t, j]] = t
    return nxt


@njit(cache=True)
def sim(R, E, cap, seqs, do_fetch, lead, land):
    """Returns per-step: misses served by the CPU, fetches (misses served by a fetch into a slot), copies (prefetch
    admissions), in-flight misses. land = steps after issue at which a prefetched expert is resident."""
    T, k = R.shape
    nxt = next_use_table(R, E, seqs)
    state = np.zeros(E, dtype=np.int64)        # 0 absent, 1 resident, 2 loading
    ready = np.zeros(E, dtype=np.int64)
    nu = np.full(E, INF, dtype=np.int64)       # next use of every expert as of the current step (updated incrementally)
    cpu_miss = np.zeros(T, dtype=np.int64); fetched = np.zeros(T, dtype=np.int64); copies = np.zeros(T, dtype=np.int64); inflight = np.zeros(T, dtype=np.int64)
    size = 0
    missed = np.zeros(k, dtype=np.int64); missnu = np.zeros(k, dtype=np.int64)
    selected = np.zeros(E, dtype=np.bool_)
    # future requests, to find prefetch candidates: we scan forward up to a horizon each step (bounded for speed)
    horizon = 64
    for t in range(T):
        for e in range(E):
            if state[e] == 2 and ready[e] <= t:
                state[e] = 1
        for j in range(k):
            selected[R[t, j]] = True
            nu[R[t, j]] = nxt[t, j]
        # this step's misses
        nm = 0
        for j in range(k):
            e = R[t, j]
            if state[e] == 1:
                continue
            if state[e] == 2:
                inflight[t] += 1
            missed[nm] = e; missnu[nm] = nu[e]; nm += 1
        # fetch-on-admit: MIN with bypass among this step's misses, soonest next use first
        if do_fetch:
            order = np.argsort(missnu[:nm])
            for q in range(nm):
                e = missed[order[q]]
                if state[e] != 0 or nu[e] >= INF:
                    continue
                slot_ok = False
                if size < cap:
                    slot_ok = True
                else:
                    victim = -1; best = -1
                    for c in range(E):
                        if state[c] == 1 and not selected[c] and nu[c] > best:
                            best, victim = nu[c], c
                    if victim >= 0 and best > nu[e]:
                        state[victim] = 0; size -= 1; slot_ok = True
                if not slot_ok:
                    break
                state[e] = 1; size += 1; fetched[t] += 1
        for j in range(k):
            e = R[t, j]
            if state[e] != 1:
                cpu_miss[t] += 1
        # scheduled single-read prefetch: experts whose FIRST use after t is at least `lead` steps away (so a copy
        # issued now lands in time), soonest first; admitted only if MIN with bypass would admit them at that use:
        # the victim (resident, not selected now, furthest next use) is needed later than the candidate's use AFTER
        # that first use (or never again). An expert used sooner than `lead` steps is left to miss (and, with fetch, to
        # be fetched then); a once-used expert is admitted only into a slot nobody needs again.
        if lead > 0:
            cand_e = np.full(horizon * k, -1, dtype=np.int64); cand_f = np.full(horizon * k, INF, dtype=np.int64); cand_s = np.full(horizon * k, INF, dtype=np.int64); nc = 0
            first = np.full(E, INF, dtype=np.int64); second = np.full(E, INF, dtype=np.int64)
            for tt in range(t + 1, min(T, t + horizon)):
                if seqs[tt] != seqs[t]:
                    break
                for j in range(k):
                    e = R[tt, j]
                    if first[e] >= INF:
                        first[e] = tt
                    elif second[e] >= INF:
                        second[e] = tt
            for e in range(E):
                if first[e] < INF and first[e] >= t + lead and state[e] == 0:
                    cand_e[nc] = e; cand_f[nc] = first[e]; cand_s[nc] = second[e]; nc += 1
            order = np.argsort(cand_f[:nc])
            for q in range(nc):
                e = cand_e[order[q]]; se = cand_s[order[q]]
                if state[e] != 0:
                    continue
                if size < cap:
                    state[e] = 2; ready[e] = t + land; size += 1; copies[t] += 1
                    continue
                victim = -1; best = -1
                for c in range(E):
                    if state[c] == 1 and not selected[c]:
                        v = first[c]   # the resident's next use after t (INF: never again in the sequence)
                        if v > best:
                            best, victim = v, c
                if victim < 0:
                    break
                if best < INF and best <= se:
                    continue   # the victim is needed before this expert's reuse: MIN would bypass it
                state[victim] = 0; size -= 1
                state[e] = 2; ready[e] = t + land; size += 1; copies[t] += 1
        for j in range(k):
            selected[R[t, j]] = False
    return cpu_miss, fetched, copies, inflight


def run(act, seqs, E, cap, do_fetch, lead, land):
    L, T = act.shape[1], act.shape[0]
    cm = fe = co = inf = 0
    for il in range(L):
        R = np.ascontiguousarray(act[:, il, :].astype(np.int64))
        a, b, c, d = sim(R, E, cap, seqs, do_fetch, lead, land)
        cm += int(a.sum()); fe += int(b.sum()); co += int(c.sum()); inf += int(d.sum())
    req = T * L * act.shape[2]
    return dict(cpu_misses_per_token=cm / T, fetches_per_token=fe / T, copies_per_token=co / T, reads_per_token=(cm + fe + co) / T,
                hit_rate=1 - (cm + fe) / req, inflight_misses_per_token=inf / T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    out = {"note": __doc__, "models": {}}
    for model, (path, caps, budget) in TRACES.items():
        z = np.load(path); act = z["act"]; seqs = z["seq"].astype(np.int64); E = int(z["n_expert"])
        out["models"][model] = {"trace": path, "cells": {}}
        for cap in caps:
            cell = {"budget": budget[cap], "opt_reads_per_token": OPT[(model, cap)], "modes": {}}
            cell["modes"]["fetch"] = run(act, seqs, E, cap, True, 0, 0)
            for d in (2, 3, 4):
                cell["modes"][f"lead-{d}"] = run(act, seqs, E, cap, False, d, d)
                cell["modes"][f"both-{d}"] = run(act, seqs, E, cap, True, d, d)
            out["models"][model]["cells"][str(cap)] = cell
            print(f"{model} C={cap} ({budget[cap]}), optimum {OPT[(model, cap)]:.1f} reads/token:")
            for m, r in cell["modes"].items():
                print(f"   {m:7s} reads {r['reads_per_token']:6.1f} (cpu {r['cpu_misses_per_token']:5.1f} fetch {r['fetches_per_token']:5.1f} copies {r['copies_per_token']:5.1f})  hit {100 * r['hit_rate']:.1f}%  in-flight misses {r['inflight_misses_per_token']:.1f}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

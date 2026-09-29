"""How far is the deployed cache policy from the optimum, in host-memory reads? (gpt-oss-120b, own text, job 063 trace)

Per layer, on the decode steps recorded by ec-bench --lookahead (12 sequences x 128 tokens, the same text as the
engine's own-text runs), count per token:
  cpu   = misses executed on the CPU (read from host memory by the helpers; the law's critical-path term)
  copy  = expert copies to the GPU (admissions, which cost host-memory reads too but run in the background)
for these policies with C slots per layer:
  dfa          the deployed policy (decayed frequency, half-life 16 steps, kappa 1, admission lands 2 steps later)
  lru          admit every miss, evict least recently used
  static       hindsight top-C experts by frequency over the whole trace, never admits
  opt          Belady MIN with bypass (future known), cold start: the fewest host reads any exact policy can make
  opt-W        MIN with bypass that sees only the next W decode steps (W = 1, 2, 4, 8, 16, 32, 128)
  lru-fetch    the same rule with no lookahead: every miss is copied, the least recently used is evicted
In opt and opt-W a miss is either bypassed (run on the CPU, cpu += 1) or admitted (copied, copy += 1; the copy serves
the current step, as with FETCH). The cache is carried across sequences, as in the engine.

    python scripts/policy_gap.py --la /home/claude/gpu-branch/results/063_lookahead@vast/la_gpt-oss-120b.bin
"""
import argparse
import json
import os
import sys

import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl import ecsim_fast  # noqa: E402
from scripts.sim_prefetch import load  # noqa: E402


@njit(cache=True)
def opt_window(R, E, C, W):
    """MIN with bypass and a lookahead of W steps (W = 0: the whole future; W < 0: none, i.e. LRU that fetches every
    miss). Returns (cpu misses, copies) per step.
    Next use beyond the window counts as infinite; ties among infinite ones are broken by the most recent use
    (keep the most recently used)."""
    T, k = R.shape
    # next occurrence index per (t, j): the next step > t that requests R[t, j]
    nxt_step = np.full(E, T + 1, np.int64)
    nextuse = np.empty((T, k), np.int64)
    for t in range(T - 1, -1, -1):
        for j in range(k):
            nextuse[t, j] = nxt_step[R[t, j]]
        for j in range(k):
            nxt_step[R[t, j]] = t
    resident = np.full(C, -1, np.int64)
    slot_of = np.full(E, -1, np.int64)
    nu = np.full(E, T + 1, np.int64)   # next use of each expert as of the current step
    last = np.full(E, -1, np.int64)
    cpu = np.zeros(T, np.int32)
    copy = np.zeros(T, np.int32)
    for t in range(T):
        for j in range(k):
            e = R[t, j]
            nu[e] = nextuse[t, j]
        for j in range(k):
            e = R[t, j]
            if slot_of[e] >= 0:
                last[e] = t
                continue
            # candidate: admit e, evicting the resident with the farthest (windowed) next use, unless e's own next
            # use is farther than all of them (then bypass)
            lim = T + 1 if W == 0 else (t if W < 0 else t + W)
            s_free = -1
            for s in range(C):
                if resident[s] < 0:
                    s_free = s
                    break
            if s_free >= 0:
                resident[s_free] = e
                slot_of[e] = s_free
                copy[t] += 1
                last[e] = t
                continue
            victim = -1
            vkey = -1.0
            for s in range(C):
                c = resident[s]
                inrow = False
                for jj in range(k):
                    if R[t, jj] == c:
                        inrow = True
                        break
                if inrow:
                    continue
                d = nu[c] if nu[c] <= lim else T + 2
                key = float(d) * 1e7 - float(last[c]) if d > T + 1 else float(d) * 1e7
                if key > vkey:
                    vkey = key
                    victim = s
            de = nu[e] if nu[e] <= lim else T + 2
            ekey = float(de) * 1e7 - float(t) if de > T + 1 else float(de) * 1e7
            if victim < 0 or ekey >= vkey:
                cpu[t] += 1
                last[e] = t
                continue
            v = resident[victim]
            slot_of[v] = -1
            resident[victim] = e
            slot_of[e] = victim
            copy[t] += 1
            last[e] = t
    return cpu, copy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--la", required=True)
    ap.add_argument("--budgets", default="14,32,56")
    ap.add_argument("--out")
    a = ap.parse_args()
    d = load(a.la)
    L, E, k = d["L"], d["E"], d["k"]
    act = d["act"]                      # [T, L, k], rows in engine order (seq, step)
    T = act.shape[0]
    res = {}
    for C in [int(x) for x in a.budgets.split(",")]:
        row = {}
        tot = {}
        for l in range(L):
            R = np.ascontiguousarray(act[:, l, :])
            h, m, ad = ecsim_fast.simulate_layer(R, E, C, "dfa", half_life=16.0, kappa=1.0)
            tot.setdefault("dfa", [0, 0])
            tot["dfa"][0] += m.sum(); tot["dfa"][1] += ad.sum()
            h, m, ad = ecsim_fast.simulate_layer(R, E, C, "lru")
            tot.setdefault("lru", [0, 0])
            tot["lru"][0] += m.sum(); tot["lru"][1] += ad.sum()
            cnt = np.bincount(R.ravel(), minlength=E)
            top = set(np.argsort(-cnt, kind="stable")[:C].tolist())
            ms = sum(1 for e in R.ravel() if int(e) not in top)
            tot.setdefault("static", [0, 0])
            tot["static"][0] += ms; tot["static"][1] += 0
            for W in (0, -1, 1, 2, 4, 8, 16, 32, 128):
                c, cp = opt_window(R, E, C, W)
                key = "opt" if W == 0 else ("lru-fetch" if W < 0 else f"opt-{W}")
                tot.setdefault(key, [0, 0])
                tot[key][0] += c.sum(); tot[key][1] += cp.sum()
        for key, (m, cp) in tot.items():
            row[key] = dict(cpu=m / T, copy=cp / T, reads=(m + cp) / T, hit=1 - (m + cp) / (T * L * k),
                            hit_incl_copies_served=1 - m / (T * L * k))
        res[C] = row
        print(f"C = {C}: per token (of {L * k} routed experts)")
        print(f"   {'policy':8s} {'cpu':>7s} {'copy':>7s} {'reads':>7s}")
        for key, v in row.items():
            print(f"   {key:8s} {v['cpu']:7.2f} {v['copy']:7.2f} {v['reads']:7.2f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

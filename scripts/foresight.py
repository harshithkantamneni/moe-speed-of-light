"""The value of routing foresight for offloaded MoE decode: host-memory reads per token of online cache policies,
of the offline optimum (Belady MIN with bypass), and of policies that see the next W decode steps' routing.

Traces: own sampled text (arm S) of 9 MoE models (jobs 032-048, `routes` packs), response tokens of every
conversation back to back with the cache carried across conversations (one user, one session). Also the job 063
gpt-oss-120b trace (12 x 128 tokens, the engine's own-text runs) with --la.

Per layer and budget C (slots per layer = E/8, E/4, E/2), counted per decode token:
  dfa        deployed policy: CPU runs every miss; background admission (decayed frequency, half-life 16, kappa per
             model, lands 2 steps later) copies the expert again. reads = cpu + copies; critical = cpu
  dfa-fetch  same scores, but an admitted miss is copied and run on the GPU (one read); others run on the CPU
             (this family is run with kappa 0 and the model's kappa, and the one with fewer reads is kept)
  W = 1..    foresight: Belady within the next W steps, decayed frequency beyond them (see _pol)
  opt        W = infinity: MIN with bypass, the fewest reads of any exact policy
  seq-static per conversation, the C experts it uses most (hindsight), loaded at its start; no admissions

    python scripts/foresight.py --results /home/claude/gpu-branch/results --out prereg/foresight
"""
import argparse
import json
import os
import sys

import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl import ecsim_fast  # noqa: E402
from mosl.traces import load_pack  # noqa: E402
from scripts.provenance import MODELS, find, per_seq  # noqa: E402

WINDOWS = (1, 2, 4, 8, 16, 32, 64, 256)
INF = 1 << 40


@njit(cache=True)
def _pol(R, E, C, W, hl, kappa):
    """W-step foresight (W < 0: none; W >= INF: whole future). A miss is copied into a slot (and served by it) or
    bypassed to the CPU. Victim: a resident not requested now; among those not reused within the window, the lowest
    decayed score; if all are reused within the window, the farthest next use. A miss reused within the window is
    admitted unless the victim is reused sooner; a miss not reused within the window is admitted only if its score
    beats a beyond-window victim's by kappa. Returns (cpu reads, copies)."""
    T, k = R.shape
    decay = 0.5 ** (1.0 / hl)
    nxt = np.full(E, INF, np.int64)
    nu_t = np.empty((T, k), np.int64)
    for t in range(T - 1, -1, -1):
        for j in range(k):
            nu_t[t, j] = nxt[R[t, j]]
        for j in range(k):
            nxt[R[t, j]] = t
    res = np.full(C, -1, np.int64)
    slot = np.full(E, -1, np.int64)
    nu = np.full(E, INF, np.int64)
    sc = np.zeros(E)
    last = np.zeros(E, np.int64)
    cpu = 0
    cp = 0
    for t in range(T):
        for j in range(k):
            e = R[t, j]
            nu[e] = nu_t[t, j]
            sc[e] = sc[e] * decay ** (t - last[e]) + 1.0
            last[e] = t
        lim = t + W if W >= 0 else t
        for j in range(k):
            e = R[t, j]
            if slot[e] >= 0:
                continue
            fr = -1
            for s in range(C):
                if res[s] < 0:
                    fr = s
                    break
            if fr >= 0:
                res[fr] = e
                slot[e] = fr
                cp += 1
                continue
            v = -1
            vbeyond = False
            vkey = 0.0
            for s in range(C):
                c = res[s]
                inrow = False
                for jj in range(k):
                    if R[t, jj] == c:
                        inrow = True
                        break
                if inrow:
                    continue
                beyond = W < 0 or nu[c] > lim
                if beyond:
                    key = sc[c] * decay ** (t - last[c])
                    if (not vbeyond) or key < vkey:
                        v = s
                        vbeyond = True
                        vkey = key
                elif not vbeyond:
                    key = float(nu[c])
                    if v < 0 or key > vkey:
                        v = s
                        vkey = key
            if v < 0:
                cpu += 1
                continue
            if W < 0 or nu[e] > lim:
                if not (vbeyond and sc[e] > vkey + kappa):
                    cpu += 1
                    continue
            elif (not vbeyond) and float(nu[e]) >= vkey:
                cpu += 1
                continue
            old = res[v]
            slot[old] = -1
            res[v] = e
            slot[e] = v
            cp += 1
    return cpu, cp


def seq_static(R, segs, E, C):
    miss = cp = 0
    prev = set()
    for a, b in segs:
        r = R[a:b]
        cnt = np.bincount(r.ravel(), minlength=E)
        top = set(np.argsort(-cnt, kind="stable")[:C].tolist())
        miss += int(np.isin(r, list(top), invert=True).sum())
        cp += len(top - prev)
        prev = top
    return miss, cp


def analyse(routes, segs, E, k, kappa, budgets):
    """routes [L, T, k] (response tokens back to back); segs: (start, stop) of each conversation in T"""
    L, T, _ = routes.shape
    out = {}
    for C in budgets:
        acc = {}

        def add(key, c, p):
            a = acc.setdefault(key, [0, 0])
            a[0] += c
            a[1] += p
        for l in range(L):
            R = np.ascontiguousarray(routes[l], dtype=np.int64)
            h, m, ad = ecsim_fast.simulate_layer(R, E, C, "dfa", half_life=16.0, kappa=kappa)
            add("dfa", int(m.sum()), int(ad.sum()))
            for key, W in [("dfa-fetch", -1)] + [(f"W{w}", w) for w in WINDOWS] + [("opt", INF)]:
                for kap in sorted({0.0, kappa}):
                    add(f"{key}@k{kap:g}", *_pol(R, E, C, W, 16.0, kap))
            add("seq-static", *seq_static(R, segs, E, C))
        row = {key: dict(cpu=c / T, copy=p / T, reads=(c + p) / T) for key, (c, p) in acc.items()}
        # for the fetch-admit family keep, per window, the kappa (0 or the model's) with fewer reads over all layers
        for key in ["dfa-fetch"] + [f"W{w}" for w in WINDOWS] + ["opt"]:
            cands = {kk: v for kk, v in row.items() if kk.startswith(key + "@k")}
            best = min(cands, key=lambda kk: cands[kk]["reads"])
            row[key] = dict(cands[best], kappa=float(best.split("@k")[1]))
        row["dfa"]["critical"] = row["dfa"]["cpu"]
        for key in row:
            row[key].setdefault("critical", row[key]["reads"])
        out[C] = row
    return out, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/home/claude/gpu-branch/results")
    ap.add_argument("--la", default="/home/claude/gpu-branch/results/063_lookahead@vast/la_gpt-oss-120b.bin")
    ap.add_argument("--arm", default="S")
    ap.add_argument("--out", default="prereg/foresight")
    ap.add_argument("--models", default=",".join(MODELS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    allres = {}
    jobs = []
    if a.la:
        from scripts.sim_prefetch import load
        d = load(a.la)
        routes = np.transpose(d["act"], (1, 0, 2))
        segs, s0 = [], 0
        for s in np.unique(d["seq"]):
            n = int((d["seq"] == s).sum())
            segs.append((s0, s0 + n))
            s0 += n
        jobs.append(("gpt-oss-120b (job 063, engine text)", routes, segs, d["E"], d["k"], 1.0))
    for key in a.models.split(","):
        ck, kappa, _ = MODELS[key]
        p = find(a.results, f"{ck}_{a.arm}.npz")
        if not p:
            print(f"{key}: no {a.arm} pack")
            continue
        pk = load_pack(p)
        win = per_seq(pk)
        idx = np.concatenate([np.arange(x, y) for _, x, y in win])
        segs, s0 = [], 0
        for _, x, y in win:
            segs.append((s0, s0 + (y - x)))
            s0 += y - x
        routes = pk["routes"][:, idx]
        jobs.append((key, routes, segs, pk["E"], routes.shape[2], kappa))
    for name, routes, segs, E, k, kappa in jobs:
        budgets = sorted({max(k, E // 8), E // 4, E // 2})
        res, T = analyse(routes, segs, int(E), int(k), float(kappa), budgets)
        allres[name] = dict(E=int(E), k=int(k), L=int(routes.shape[0]), T=int(T), convs=len(segs), budgets=res)
        print(f"== {name}: E={E} k={k} L={routes.shape[0]} T={T} conversations={len(segs)}")
        for C, row in res.items():
            opt = row["opt"]["reads"]
            onl = row["dfa-fetch"]["reads"]
            line = f"   C={C:3d} ({100 * C / E:.0f}%) reads/token: dfa {row['dfa']['reads']:6.1f} (critical {row['dfa']['critical']:5.1f})  dfa-fetch {onl:6.1f}  opt {opt:6.1f}  seq-static {row['seq-static']['reads']:6.1f} | gap closed by W:"
            for W in WINDOWS:
                r = row[f"W{W}"]["reads"]
                line += f" {W}:{100 * (onl - r) / max(1e-9, onl - opt):3.0f}%"
            print(line)
        json.dump(allres, open(os.path.join(a.out, f"foresight_{a.arm}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

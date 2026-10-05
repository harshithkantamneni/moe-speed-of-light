"""A realisable foresight policy: learned admission and eviction from routing history alone (no future routing).

The oracles of job 095 show that foresight pays when it is spent as MIN with bypass spends it: admit a missed expert
(fetched once into its slot) only if it will be reused before the resident it would evict, otherwise serve it on the
CPU. Belady needs the future; this policy replaces each expert's next use by a learned estimate of how likely it is to
be requested within the next H decode steps, from features the engine has at the end of every step:

  decayed request counts at half-lives 2, 8, 32 and 128 steps; log(1 + steps since the last request); and a
  transition score, the mean over this step's requested experts i of P(e requested within H | i requested now),
  counted on the training conversations of the same layer.

A logistic model per model and budget (pooled over layers) is trained on the first half of the conversations and the
policy is simulated on the second half, with the single-read semantics of fetch-on-admit: each miss is read once,
either fetched into a slot (admitted) or run on the CPU (bypassed). Victim: the resident not requested this step with
the lowest predicted probability; a miss is admitted if its probability exceeds the victim's by the margin m (chosen
on the training half). Reads per token are compared, on the same held-out half, with decayed frequency under the same
single-read semantics (dfa-fetch, kappa 0 or the model's, the better kept), Belady within the next 1 and 4 steps, and
MIN with bypass (the optimum).

    python scripts/learned_policy.py --models gpt-oss-120b,qwen3-30b-a3b --out prereg/learned_policy.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl.traces import load_pack  # noqa: E402
from scripts.provenance import MODELS, find, per_seq  # noqa: E402
from scripts.foresight import _pol, INF  # noqa: E402

HL = np.array([2.0, 8.0, 32.0, 128.0])
NF = 7   # 4 decayed counts, recency, transition, bias


@njit(cache=True)
def _features(sc, last, t, e, HLd, trans_row_sum, k):
    f = np.empty(NF)
    for h in range(4):
        f[h] = sc[h, e] * HLd[h] ** (t - last[e]) if last[e] >= 0 else 0.0
    f[4] = np.log1p(t - last[e]) if last[e] >= 0 else 8.0
    f[5] = trans_row_sum / k
    f[6] = 1.0
    return f


@njit(cache=True)
def _collect(R, E, H, M, seed, per_step):
    """Training samples on one layer: at each step (after its requests), the requested experts and per_step random
    others; label = requested within the next H steps."""
    T, k = R.shape
    HLd = np.empty(4)
    for h in range(4):
        HLd[h] = 0.5 ** (1.0 / HL[h])
    sc = np.zeros((4, E)); last = np.full(E, -1, np.int64)
    n_out = T * (k + per_step)
    X = np.empty((n_out, NF)); Y = np.empty(n_out); n = 0
    # next-use table per (t, expert) only for sampled experts: precompute per-expert sorted request times
    cnt = np.zeros(E, np.int64)
    for t in range(T):
        for j in range(k):
            cnt[R[t, j]] += 1
    start = np.zeros(E + 1, np.int64)
    for e in range(E):
        start[e + 1] = start[e] + cnt[e]
    pos = start[:-1].copy()
    tl = np.empty(start[E], np.int64)
    for t in range(T):
        for j in range(k):
            e = R[t, j]; tl[pos[e]] = t; pos[e] += 1
    ptr = start[:-1].copy()
    np.random.seed(seed)
    for t in range(T):
        for j in range(k):
            e = R[t, j]
            for h in range(4):
                sc[h, e] = (sc[h, e] * HLd[h] ** (t - last[e]) if last[e] >= 0 else 0.0) + 1.0
            last[e] = t
            while ptr[e] < start[e + 1] and tl[ptr[e]] <= t:
                ptr[e] += 1
        cand = np.empty(k + per_step, np.int64)
        for j in range(k):
            cand[j] = R[t, j]
        for q in range(per_step):
            cand[k + q] = np.random.randint(0, E)
        for q in range(k + per_step):
            e = cand[q]
            while ptr[e] < start[e + 1] and tl[ptr[e]] <= t:
                ptr[e] += 1
            nx = tl[ptr[e]] if ptr[e] < start[e + 1] else INF
            ts = 0.0
            for j in range(k):
                ts += M[R[t, j], e]
            X[n] = _features(sc, last, t, e, HLd, ts, k)
            Y[n] = 1.0 if nx - t <= H else 0.0
            n += 1
    return X[:n], Y[:n]


@njit(cache=True)
def _trans(R, E, H):
    """M[i, e] = P(e requested within the next H steps | i requested now)."""
    T, k = R.shape
    num = np.zeros((E, E)); den = np.zeros(E)
    seen = np.zeros(E, np.int64)
    for t in range(T - 1):
        stamp = t + 1
        for tt in range(t + 1, min(T, t + 1 + H)):
            for j in range(k):
                seen[R[tt, j]] = stamp
        for j in range(k):
            i = R[t, j]
            den[i] += 1
            for e in range(E):
                if seen[e] == stamp:
                    num[i, e] += 1
    for i in range(E):
        if den[i] > 0:
            for e in range(E):
                num[i, e] /= den[i]
    return num


@njit(cache=True)
def _sim(R, E, C, w, M, margin):
    """Single-read cache with learned admission and eviction. Returns (cpu reads, fetches)."""
    T, k = R.shape
    HLd = np.empty(4)
    for h in range(4):
        HLd[h] = 0.5 ** (1.0 / HL[h])
    sc = np.zeros((4, E)); last = np.full(E, -1, np.int64)
    res = np.full(C, -1, np.int64); slot = np.full(E, -1, np.int64)
    cpu = 0; fe = 0
    req = np.zeros(E, np.bool_)
    p = np.zeros(E)
    for t in range(T):
        for j in range(k):
            e = R[t, j]
            for h in range(4):
                sc[h, e] = (sc[h, e] * HLd[h] ** (t - last[e]) if last[e] >= 0 else 0.0) + 1.0
            last[e] = t
            req[e] = True
        # transition sums for every expert from this step's requests
        ts = np.zeros(E)
        for j in range(k):
            i = R[t, j]
            for e in range(E):
                ts[e] += M[i, e]
        # probabilities of the residents and of this step's misses
        for s in range(C):
            c = res[s]
            if c >= 0:
                f = _features(sc, last, t, c, HLd, ts[c], k)
                z = 0.0
                for q in range(NF):
                    z += w[q] * f[q]
                p[c] = 1.0 / (1.0 + np.exp(-z))
        for j in range(k):
            e = R[t, j]
            if slot[e] >= 0:
                continue
            f = _features(sc, last, t, e, HLd, ts[e], k)
            z = 0.0
            for q in range(NF):
                z += w[q] * f[q]
            p[e] = 1.0 / (1.0 + np.exp(-z))
        # misses in order of probability, highest first
        miss = np.empty(k, np.int64); nm = 0
        for j in range(k):
            if slot[R[t, j]] < 0:
                miss[nm] = R[t, j]; nm += 1
        order = np.argsort(-p[miss[:nm]])
        for qi in range(nm):
            e = miss[order[qi]]
            fr = -1
            for s in range(C):
                if res[s] < 0:
                    fr = s
                    break
            if fr >= 0:
                res[fr] = e; slot[e] = fr; fe += 1
                continue
            v = -1; vp = 2.0
            for s in range(C):
                c = res[s]
                if req[c]:
                    continue
                if p[c] < vp:
                    vp = p[c]; v = s
            if v < 0 or not (p[e] > vp + margin):
                cpu += 1
                continue
            old = res[v]
            slot[old] = -1
            res[v] = e; slot[e] = v; fe += 1
        for j in range(k):
            req[R[t, j]] = False
    return cpu, fe


def fit_logistic(X, Y):
    from sklearn.linear_model import LogisticRegression
    Xs = X[:, :NF - 1]
    clf = LogisticRegression(C=1.0, max_iter=400)
    clf.fit(Xs, Y)
    return np.concatenate([clf.coef_[0], clf.intercept_])


def split_routes(routes, segs, frac=0.5):
    n = len(segs)
    cut = segs[int(n * frac)][0]
    return routes[:, :cut], routes[:, cut:]


def study(name, routes, segs, E, k, kappa, budgets, seed=0):
    L = routes.shape[0]
    tr, te = split_routes(routes, segs)
    Tte = te.shape[1]
    out = {}
    for C in budgets:
        t0 = time.time()
        best = None
        for H in sorted({max(1, int(round(C / k))), max(2, int(round(2 * C / k))), max(4, int(round(4 * C / k)))}):
            Ms = [_trans(np.ascontiguousarray(tr[l], dtype=np.int64), E, H) for l in range(L)]
            Xs, Ys = [], []
            for l in range(L):
                X, Y = _collect(np.ascontiguousarray(tr[l], dtype=np.int64), E, H, Ms[l], seed + l, 4)
                idx = np.random.default_rng(seed + l).choice(len(Y), size=min(len(Y), 40000), replace=False)
                Xs.append(X[idx]); Ys.append(Y[idx])
            w = fit_logistic(np.concatenate(Xs), np.concatenate(Ys))
            # margin chosen on the training half (reads summed over layers)
            for margin in (0.0, 0.05, 0.1, 0.2):
                tr_reads = sum(sum(_sim(np.ascontiguousarray(tr[l], dtype=np.int64), E, C, w, Ms[l], margin)) for l in range(L))
                if best is None or tr_reads < best[0]:
                    best = (tr_reads, H, margin, w, Ms)
        _, H, margin, w, Ms = best
        # held-out half: the learned policy, dfa-fetch, Belady within 1 and 4 steps, and MIN with bypass
        acc = {"learned": [0, 0]}
        for l in range(L):
            R = np.ascontiguousarray(te[l], dtype=np.int64)
            c, f = _sim(R, E, C, w, Ms[l], margin)
            acc["learned"][0] += c; acc["learned"][1] += f
            for key, W in (("dfa-fetch", -1), ("W1", 1), ("W4", 4), ("opt", INF)):
                for kap in sorted({0.0, kappa}):
                    a = acc.setdefault(f"{key}@k{kap:g}", [0, 0])
                    cc, pp = _pol(R, E, C, W, 16.0, kap)
                    a[0] += cc; a[1] += pp
        row = {}
        for key, (c, f) in acc.items():
            row[key] = dict(cpu=c / Tte, copy=f / Tte, reads=(c + f) / Tte)
        for key in ("dfa-fetch", "W1", "W4", "opt"):
            cands = {kk: v for kk, v in row.items() if kk.startswith(key + "@k")}
            bk = min(cands, key=lambda kk: cands[kk]["reads"])
            row[key] = dict(cands[bk], kappa=float(bk.split("@k")[1]))
        onl, opt, lr = row["dfa-fetch"]["reads"], row["opt"]["reads"], row["learned"]["reads"]
        row["learned"].update(H=int(H), margin=float(margin), weights=[float(x) for x in w])
        row["gap_closed"] = (onl - lr) / max(1e-9, onl - opt)
        row["gap_closed_W1"] = (onl - row["W1"]["reads"]) / max(1e-9, onl - opt)
        row["gap_closed_W4"] = (onl - row["W4"]["reads"]) / max(1e-9, onl - opt)
        out[int(C)] = row
        print(f"   {name} C={C} ({100 * C / E:.1f}%): dfa-fetch {onl:7.2f}  learned {lr:7.2f} (H {H}, m {margin})  W1 {row['W1']['reads']:7.2f}  W4 {row['W4']['reads']:7.2f}  opt {opt:7.2f}"
              f"  | gap closed: learned {100 * row['gap_closed']:5.1f}%  W1 {100 * row['gap_closed_W1']:5.1f}%  W4 {100 * row['gap_closed_W4']:5.1f}%  ({time.time() - t0:.0f} s)", flush=True)
    return out, int(Tte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results"))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--budgets", default="8,4")   # E/8 and E/4 (as divisors of E)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = json.load(open(a.out)) if a.out and os.path.exists(a.out) else {}
    for key in a.models.split(","):
        ck, kappa, _ = MODELS[key]
        p = find(a.results, f"{ck}_S.npz")
        pk = load_pack(p)
        win = per_seq(pk)
        idx = np.concatenate([np.arange(x, y) for _, x, y in win])
        segs, s0 = [], 0
        for _, x, y in win:
            segs.append((s0, s0 + (y - x)))
            s0 += y - x
        routes = pk["routes"][:, idx]
        E, k = int(pk["E"]), int(routes.shape[2])
        budgets = sorted({max(k, E // int(d)) for d in a.budgets.split(",")})
        print(f"== {key}: E={E} k={k} L={routes.shape[0]} T={routes.shape[1]} conversations={len(segs)}", flush=True)
        r, T = study(key, routes, segs, E, k, float(kappa), budgets)
        res[key] = dict(E=E, k=k, L=int(routes.shape[0]), T_test=T, budgets=r)
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

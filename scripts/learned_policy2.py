"""Learned admission and eviction from routing history (see scripts/learned_policy.py), with the predicted probability
of reuse computed for every (step, expert) of a layer from features that depend only on the request history, so any
model (logistic, gradient-boosted trees, a small MLP) can be compared under the same single-read cache simulation.

Features of expert e after step t's requests: log(1 + decayed count) at half-lives 1, 4, 16, 64, 256; log(1 + steps
since its last request); requested at t; its transition score sum_i P(e within H | i now) / k over this step's experts
i (training conversations of the layer); its log training popularity; the number of distinct experts requested in the
last 4 steps that co-occurred with it... (kept small: 9 features). Label: requested within the next H steps.

    python scripts/learned_policy2.py --models gpt-oss-120b --budgets 8,4 --model gbdt
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
from scripts.learned_policy import _trans  # noqa: E402

HL = np.array([1.0, 4.0, 16.0, 64.0, 256.0])
NF = 9


@njit(cache=True)
def features_all(R, E, M, pop, t0, t1):
    """F[t - t0, e, :] for t in [t0, t1) after step t's requests (the history from step 0)."""
    T, k = R.shape
    nh = HL.shape[0]
    d = np.empty(nh)
    for h in range(nh):
        d[h] = 0.5 ** (1.0 / HL[h])
    sc = np.zeros((nh, E)); last = np.full(E, -1, np.int64)
    F = np.zeros((t1 - t0, E, NF), np.float32)
    for t in range(t1):
        for j in range(k):
            e = R[t, j]
            for h in range(nh):
                sc[h, e] = (sc[h, e] * d[h] ** (t - last[e]) if last[e] >= 0 else 0.0) + 1.0
            last[e] = t
        if t < t0:
            continue
        ts = np.zeros(E)
        for j in range(k):
            i = R[t, j]
            for e in range(E):
                ts[e] += M[i, e]
        for e in range(E):
            for h in range(nh):
                F[t - t0, e, h] = np.log1p(sc[h, e] * d[h] ** (t - last[e])) if last[e] >= 0 else 0.0
            F[t - t0, e, nh] = np.log1p(t - last[e]) if last[e] >= 0 else 12.0
            F[t - t0, e, nh + 1] = 1.0 if last[e] == t else 0.0
            F[t - t0, e, nh + 2] = ts[e] / k
            F[t - t0, e, nh + 3] = pop[e]
    return F


@njit(cache=True)
def labels_all(R, E, H, t0, t1):
    """Y[t - t0, e] = 1 if e is requested within (t, t + H]."""
    T, k = R.shape
    nxt = np.full(E, INF, np.int64)
    Y = np.zeros((t1 - t0, E), np.float32)
    for t in range(T - 1, t0 - 1, -1):
        if t < t1:
            for e in range(E):
                Y[t - t0, e] = 1.0 if nxt[e] - t <= H else 0.0
        for j in range(k):
            nxt[R[t, j]] = t
    return Y


@njit(cache=True)
def sim_p(R, E, C, P, margin, t_off):
    """Single-read cache driven by P[t - t_off, e] (probability of reuse); steps t >= t_off are counted."""
    T, k = R.shape
    res = np.full(C, -1, np.int64); slot = np.full(E, -1, np.int64)
    req = np.zeros(E, np.bool_)
    cpu = 0; fe = 0
    for t in range(t_off, T):
        p = P[t - t_off]
        for j in range(k):
            req[R[t, j]] = True
        miss = np.empty(k, np.int64); nm = 0
        for j in range(k):
            if slot[R[t, j]] < 0:
                miss[nm] = R[t, j]; nm += 1
        pm = np.empty(nm)
        for q in range(nm):
            pm[q] = -p[miss[q]]
        order = np.argsort(pm)
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


def make_model(kind):
    if kind == "logit":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=1.0, max_iter=500)
    if kind == "gbdt":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=31, learning_rate=0.1)
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(32,), max_iter=60, early_stopping=True)
    raise ValueError(kind)


def run_model(name, routes, segs, E, k, kappa, C, kind, H_mult, seed=0, n_train=300000):
    L, T, _ = routes.shape
    cut = segs[len(segs) // 2][0]
    rng = np.random.default_rng(seed)
    H = max(1, int(round(H_mult * C / k)))
    Ms, pops, Xs, Ys = [], [], [], []
    per_layer = max(2000, n_train // L)
    for l in range(L):
        R = np.ascontiguousarray(routes[l], dtype=np.int64)
        Rtr = R[:cut]
        M = _trans(Rtr, E, H)
        cnt = np.bincount(Rtr.ravel(), minlength=E).astype(np.float64)
        pop = np.log1p(cnt / max(1, cut)).astype(np.float64)
        Ms.append(M); pops.append(pop)
        # training samples: random (t, e) of the training half, half of them requested-now experts
        ts = rng.integers(64, cut - 1, size=per_layer // 64 + 1)
        F = features_all(Rtr, E, M, pop, 0, cut)
        Y = labels_all(Rtr, E, H, 0, cut)
        tt = rng.integers(64, cut - 1, size=per_layer)
        ee = rng.integers(0, E, size=per_layer)
        tq = rng.integers(64, cut - 1, size=per_layer // 2)
        eq = Rtr[tq, rng.integers(0, k, size=per_layer // 2)]
        tt = np.concatenate([tt, tq]); ee = np.concatenate([ee, eq])
        Xs.append(F[tt, ee]); Ys.append(Y[tt, ee])
        del F, Y
    X = np.concatenate(Xs); Yv = np.concatenate(Ys)
    t0 = time.time()
    model = make_model(kind)
    model.fit(X, Yv)
    fit_s = time.time() - t0
    # held-out half: predicted probabilities for every (t, e), then the cache
    lr_c = lr_f = 0
    acc = {}
    for l in range(L):
        R = np.ascontiguousarray(routes[l], dtype=np.int64)
        P = np.empty((T - cut, E), np.float32)
        for a in range(cut, T, 4000):
            b = min(T, a + 4000)
            F = features_all(R, E, Ms[l], pops[l], a, b)
            P[a - cut:b - cut] = model.predict_proba(F.reshape(-1, NF))[:, 1].reshape(b - a, E)
        best = None
        for margin in (0.0,):
            c, f = sim_p(R, E, C, P, margin, cut)
            best = (c, f)
        lr_c += best[0]; lr_f += best[1]
        Rte = np.ascontiguousarray(R[cut:])
        for key, W in (("dfa-fetch", -1), ("W1", 1), ("W4", 4), ("opt", INF)):
            for kap in sorted({0.0, kappa}):
                a_ = acc.setdefault(f"{key}@k{kap:g}", [0, 0])
                cc, pp = _pol(Rte, E, C, W, 16.0, kap)
                a_[0] += cc; a_[1] += pp
    Tte = T - cut
    row = {key: (c + f) / Tte for key, (c, f) in acc.items()}
    best = {key: min(v for kk, v in row.items() if kk.startswith(key + "@k")) for key in ("dfa-fetch", "W1", "W4", "opt")}
    lr = (lr_c + lr_f) / Tte
    gap = (best["dfa-fetch"] - lr) / max(1e-9, best["dfa-fetch"] - best["opt"])
    print(f"   {name} C={C} {kind} H={H}: dfa-fetch {best['dfa-fetch']:7.2f} learned {lr:7.2f} W1 {best['W1']:7.2f} W4 {best['W4']:7.2f} opt {best['opt']:7.2f}"
          f" | closed {100 * gap:5.1f}% (fit {fit_s:.0f} s)", flush=True)
    return dict(learned=lr, H=H, kind=kind, gap_closed=gap, **{k_: v for k_, v in best.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results"))
    ap.add_argument("--models", default="gpt-oss-120b")
    ap.add_argument("--budgets", default="8,4")
    ap.add_argument("--model", default="logit,gbdt")
    ap.add_argument("--hmult", default="1,2")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = {}
    for key in a.models.split(","):
        ck, kappa, _ = MODELS[key]
        pk = load_pack(find(a.results, f"{ck}_S.npz"))
        win = per_seq(pk)
        idx = np.concatenate([np.arange(x, y) for _, x, y in win])
        segs, s0 = [], 0
        for _, x, y in win:
            segs.append((s0, s0 + (y - x))); s0 += y - x
        routes = pk["routes"][:, idx]
        E, k = int(pk["E"]), int(routes.shape[2])
        for d in a.budgets.split(","):
            C = max(k, E // int(d))
            for kind in a.model.split(","):
                for hm in a.hmult.split(","):
                    out[f"{key} C{C} {kind} H{hm}"] = run_model(key, routes, segs, E, k, float(kappa), C, kind, float(hm))
        if a.out:
            json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

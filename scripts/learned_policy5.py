"""The realisable form of the learned reuse predictor (scripts/learned_policy3.py): every feature of expert e used at
step t's decisions is known to the engine when step t's fetch plan runs. History features are the decayed counts and
recency up to step t-1 (the counts include step t's request when e is requested at t, which the plan knows); the
same-layer and cross-layer transition scores come from step t-1's routing at every layer (the end of the previous
token); a flag says whether e is requested at t. Label: e is requested within (t, t+H]. The cache (scripts/
learned_policy2.sim_p) fetches a miss into the slot of the resident with the lowest predicted reuse when the miss's
prediction is higher, else runs it on the CPU; one read per admitted expert.

This is the policy the engine would run: at the end of step t-1 the host computes, per layer and expert, the logit
with the flag off (victims) and on (misses), sorts the residents, and uploads per-expert ranks as the fetch-on-admit
path already does for the decayed-frequency score.

    python scripts/learned_policy5.py --models gpt-oss-120b,qwen3-30b-a3b --budgets 9,4 --model logit
"""
import argparse
import json
import os
import sys

import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl.traces import load_pack  # noqa: E402
from scripts.provenance import MODELS, find, per_seq  # noqa: E402
from scripts.foresight import _pol, INF  # noqa: E402
from scripts.learned_policy import _trans  # noqa: E402
from scripts.learned_policy2 import labels_all, sim_p, make_model  # noqa: E402
from scripts.learned_policy3 import trans_all  # noqa: E402
from mosl import ecsim_fast  # noqa: E402

HL = np.array([1.0, 4.0, 16.0, 64.0, 256.0])
NF = 11


@njit(cache=True)
def features_r(routes, l, E, M, PX, pop, t0, t1):
    """F[t - t0, e, :] for the decisions of step t in [t0, t1): history to t - 1, the request flag of t."""
    L, T, k = routes.shape
    nh = HL.shape[0]
    d = np.empty(nh)
    for h in range(nh):
        d[h] = 0.5 ** (1.0 / HL[h])
    sc = np.zeros((nh, E)); last = np.full(E, -1, np.int64)
    F = np.zeros((t1 - t0, E, NF), np.float32)
    req = np.zeros(E, np.bool_)
    for t in range(t1):
        if t >= t0 and t >= 1:
            ts = np.zeros(E); tx = np.zeros(E); tmax = np.zeros(E)
            for j in range(k):
                i = routes[l, t - 1, j]
                for e in range(E):
                    ts[e] += M[i, e]
            for lp in range(L):
                for j in range(k):
                    i = routes[lp, t - 1, j]
                    for e in range(E):
                        v = PX[lp, i, l, e]
                        tx[e] += v
                        if v > tmax[e]:
                            tmax[e] = v
            for j in range(k):
                req[routes[l, t, j]] = True
            for e in range(E):
                for h in range(nh):
                    c = sc[h, e] * d[h] ** (t - last[e]) if last[e] >= 0 else 0.0
                    F[t - t0, e, h] = np.log1p(c + (1.0 if req[e] else 0.0))
                F[t - t0, e, nh] = np.log1p(t - last[e]) if last[e] >= 0 else 12.0
                F[t - t0, e, nh + 1] = 1.0 if req[e] else 0.0
                F[t - t0, e, nh + 2] = ts[e] / k
                F[t - t0, e, nh + 3] = pop[e]
                F[t - t0, e, nh + 4] = tx[e] / (L * k)
                F[t - t0, e, nh + 5] = tmax[e]
            for j in range(k):
                req[routes[l, t, j]] = False
        for j in range(k):
            e = routes[l, t, j]
            for h in range(nh):
                sc[h, e] = (sc[h, e] * d[h] ** (t - last[e]) if last[e] >= 0 else 0.0) + 1.0
            last[e] = t
    return F


def run_model(name, routes, segs, E, k, kappa, C, kind, H_mult=1.0, seed=0, n_train=300000):
    L, T, _ = routes.shape
    cut = segs[len(segs) // 2][0]
    rng = np.random.default_rng(seed)
    H = max(1, int(round(H_mult * C / k)))
    R3 = np.ascontiguousarray(routes, dtype=np.int64)
    PX = trans_all(R3, E, H, cut, 1)
    Ms, pops, Xs, Ys = [], [], [], []
    per_layer = max(2000, n_train // L)
    for l in range(L):
        Rtr = np.ascontiguousarray(R3[l, :cut])
        M = _trans(Rtr, E, H)
        cnt = np.bincount(Rtr.ravel(), minlength=E).astype(np.float64)
        pop = np.log1p(cnt / max(1, cut))
        Ms.append(M); pops.append(pop)
        F = features_r(R3[:, :cut], l, E, M, PX, pop, 0, cut)
        Y = labels_all(Rtr, E, H, 0, cut)
        tt = rng.integers(64, cut - 1, size=per_layer); ee = rng.integers(0, E, size=per_layer)
        tq = rng.integers(64, cut - 1, size=per_layer // 2); eq = Rtr[tq, rng.integers(0, k, size=per_layer // 2)]
        tt = np.concatenate([tt, tq]); ee = np.concatenate([ee, eq])
        Xs.append(F[tt, ee]); Ys.append(Y[tt, ee])
        del F, Y
    model = make_model(kind)
    model.fit(np.concatenate(Xs), np.concatenate(Ys))
    lr = 0; acc = {}; per_layer_gain = []
    for l in range(L):
        R = np.ascontiguousarray(R3[l])
        P = np.empty((T - cut, E), np.float32)
        for a in range(cut, T, 4000):
            b = min(T, a + 4000)
            F = features_r(R3[:, :b], l, E, Ms[l], PX, pops[l], a, b)
            P[a - cut:b - cut] = model.predict_proba(F.reshape(-1, NF))[:, 1].reshape(b - a, E)
        c, f = sim_p(R, E, C, P, 0.0, cut)
        lr += c + f
        Rte = np.ascontiguousarray(R[cut:])
        _, m_, ad_ = ecsim_fast.simulate_layer(Rte, E, C, "dfa", half_life=16.0, kappa=kappa)
        a_ = acc.setdefault("dfa@k", [0, 0]); a_[0] += int(m_.sum()); a_[1] += int(ad_.sum())
        for key, W in (("dfa-fetch", -1), ("W1", 1), ("W2", 2), ("W4", 4), ("opt", INF)):
            for kap in sorted({0.0, kappa}):
                a_ = acc.setdefault(f"{key}@k{kap:g}", [0, 0])
                cc, pp = _pol(Rte, E, C, W, 16.0, kap)
                a_[0] += cc; a_[1] += pp
    Tte = T - cut
    row = {key: (c + f) / Tte for key, (c, f) in acc.items()}
    best = {key: min(v for kk, v in row.items() if kk.startswith(key + "@k")) for key in ("dfa", "dfa-fetch", "W1", "W2", "W4", "opt")}
    lr /= Tte
    gap_dep = (best["dfa"] - lr) / max(1e-9, best["dfa"] - best["opt"])
    gap_foa = (best["dfa"] - best["dfa-fetch"]) / max(1e-9, best["dfa"] - best["opt"])
    gap = (best["dfa-fetch"] - lr) / max(1e-9, best["dfa-fetch"] - best["opt"])
    coef = getattr(model, "coef_", None)
    print(f"   {name} C={C} {kind} H={H} (realisable): dfa-fetch {best['dfa-fetch']:7.2f} learned {lr:7.2f} W1 {best['W1']:7.2f} W2 {best['W2']:7.2f}"
          f" W4 {best['W4']:7.2f} opt {best['opt']:7.2f} deployed {best['dfa']:7.2f} | closed {100 * gap:5.1f}% of dfa-fetch->opt;"
          f" of deployed->opt: single read {100 * gap_foa:5.1f}%, + learned {100 * gap_dep:5.1f}%{'' if coef is None else ' coef ' + str(np.round(coef[0], 3).tolist()) + ' b ' + str(np.round(model.intercept_, 3).tolist())}",
          flush=True)
    return dict(learned=lr, H=H, kind=kind, gap_closed=gap, gap_closed_deployed=gap_dep, gap_closed_deployed_single_read=gap_foa, coef=None if coef is None else coef[0].tolist(),
                intercept=None if coef is None else model.intercept_.tolist(), **best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results"))
    ap.add_argument("--models", default="gpt-oss-120b")
    ap.add_argument("--budgets", default="9,4")
    ap.add_argument("--model", default="logit")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = json.load(open(a.out)) if a.out and os.path.exists(a.out) else {}
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
                out[f"{key} C{C} {kind} realisable"] = run_model(key, routes, segs, E, k, float(kappa), C, kind)
                if a.out:
                    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

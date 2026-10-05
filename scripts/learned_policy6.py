"""The realisable learned reuse predictor (scripts/learned_policy5.py) trained on one corpus and tested on another:
trained on the model's mixed-domain trace (chat, code, math, multilingual; the nine-model trace study's corpus) and
replayed on the routing of the measured workload (the 30 AIME-25 problems x 256 decode steps that the engine runs and
that the speed limit's R* is computed from, cache carried across problems, cold at the start). Nothing of the test
text is seen in training. Optionally exports the model for the engine (LLAMA_EC_LEARNED=<file>).

Per cell: the deployed policy's reads (decayed frequency, serve-then-copy: misses + admissions), the same scores
with single-read admission (dfa-fetch), the learned policy, Belady within W = 1, 2, 4 tokens, and MIN with bypass.

    python scripts/learned_policy6.py --models gpt-oss-120b,qwen3-30b-a3b --export-dir <dir>
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl.traces import load_pack  # noqa: E402
from mosl import ecsim_fast  # noqa: E402
from scripts.provenance import MODELS, find, per_seq  # noqa: E402
from scripts.foresight import _pol, INF  # noqa: E402
from scripts.learned_policy import _trans  # noqa: E402
from scripts.learned_policy2 import labels_all, sim_p, make_model  # noqa: E402
from scripts.learned_policy3 import trans_all  # noqa: E402
from scripts.learned_policy5 import features_r, HL, NF  # noqa: E402

RES = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
AIME = {"gpt-oss-120b": f"{RES}/084c_gptoss_trace@vast/route_aime25_gptoss.npz",
        "qwen3-30b-a3b": f"{RES}/084b_vram_rerun@vast/route_aime25_qwen3.npz"}
# the paper's cells: C per model
CELLS = {"gpt-oss-120b": (14, 32, 51), "qwen3-30b-a3b": (16, 32, 56)}


@njit(cache=True)
def features_r_at(routes, l, E, M, PX, pop, times, extra, seed):
    """features_r at the given sorted times only, for the k experts requested at each time plus `extra` random others;
    returns (X, t_idx, e_idx)."""
    L, T, k = routes.shape
    nh = HL.shape[0]
    d = np.empty(nh)
    for h in range(nh):
        d[h] = 0.5 ** (1.0 / HL[h])
    sc = np.zeros((nh, E)); last = np.full(E, -1, np.int64)
    n = times.shape[0] * (k + extra)
    X = np.zeros((n, NF), np.float32)
    ti = np.empty(n, np.int64); ei = np.empty(n, np.int64)
    np.random.seed(seed)
    q = 0; w = 0
    req = np.zeros(E, np.bool_)
    for t in range(T):
        if w < times.shape[0] and times[w] == t:
            w += 1
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
            for jj in range(k + extra):
                e = routes[l, t, jj] if jj < k else np.random.randint(0, E)
                for h in range(nh):
                    c = sc[h, e] * d[h] ** (t - last[e]) if last[e] >= 0 else 0.0
                    X[q, h] = np.log1p(c + (1.0 if req[e] else 0.0))
                X[q, nh] = np.log1p(t - last[e]) if last[e] >= 0 else 12.0
                X[q, nh + 1] = 1.0 if req[e] else 0.0
                X[q, nh + 2] = ts[e] / k
                X[q, nh + 3] = pop[e]
                X[q, nh + 4] = tx[e] / (L * k)
                X[q, nh + 5] = tmax[e]
                ti[q] = t; ei[q] = e; q += 1
            for j in range(k):
                req[routes[l, t, j]] = False
        for j in range(k):
            e = routes[l, t, j]
            for h in range(nh):
                sc[h, e] = (sc[h, e] * d[h] ** (t - last[e]) if last[e] >= 0 else 0.0) + 1.0
            last[e] = t
    return X[:q], ti[:q], ei[:q]


def train(key, C, kind="logit", n_times=3000, extra=12, stride=4, seed=0, cross=True):
    ck, kappa, _ = MODELS[key]
    pk = load_pack(find(RES, f"{ck}_S.npz"))
    win = per_seq(pk)
    idx = np.concatenate([np.arange(x, y) for _, x, y in win])
    R3 = np.ascontiguousarray(pk["routes"][:, idx], dtype=np.int64)
    L, T, k = R3.shape
    E = int(pk["E"])
    H = max(1, int(round(C / k)))
    t0 = time.time()
    PX = trans_all(R3, E, H, T, stride) if cross else np.zeros((L, E, L, E), np.float32)
    rng = np.random.default_rng(seed)
    Ms, pops, Xs, Ys = [], [], [], []
    for l in range(L):
        Rl = np.ascontiguousarray(R3[l])
        M = _trans(Rl, E, H)
        cnt = np.bincount(Rl.ravel(), minlength=E).astype(np.float64)
        pop = np.log1p(cnt / T)
        Ms.append(M); pops.append(pop)
        times = np.sort(rng.choice(np.arange(64, T - H - 1), size=min(n_times, T - H - 66), replace=False)).astype(np.int64)
        X, ti, ei = features_r_at(R3, l, E, M, PX, pop, times, extra, seed + l)
        Y = labels_all(Rl, E, H, 0, T)
        Xs.append(X); Ys.append(Y[ti, ei])
    model = make_model(kind)
    model.fit(np.concatenate(Xs), np.concatenate(Ys))
    print(f"   {key} C={C}: trained on {T} tokens of the mixed-domain trace in {time.time() - t0:.0f} s", flush=True)
    return dict(model=model, Ms=np.stack(Ms), pops=np.stack(pops), PX=PX, H=H, L=L, E=E, k=k, kappa=float(kappa), cross=cross)


def test(key, C, tr):
    z = np.load(AIME[key])
    R3 = np.ascontiguousarray(z["act"].astype(np.int64).transpose(1, 0, 2))
    L, T, k = R3.shape
    E = int(z["n_expert"])
    assert (L, E, k) == (tr["L"], tr["E"], tr["k"])
    acc = {}
    lr = 0
    for l in range(L):
        R = np.ascontiguousarray(R3[l])
        F = features_r(R3, l, E, tr["Ms"][l], tr["PX"], tr["pops"][l], 0, T)
        P = tr["model"].predict_proba(F.reshape(-1, NF))[:, 1].reshape(T, E).astype(np.float32)
        c, f = sim_p(R, E, C, P, 0.0, 0)
        lr += c + f
        _, m_, ad_ = ecsim_fast.simulate_layer(R, E, C, "dfa", half_life=16.0, kappa=tr["kappa"])
        a_ = acc.setdefault("dfa@k", [0, 0]); a_[0] += int(m_.sum()); a_[1] += int(ad_.sum())
        for key_, W in (("dfa-fetch", -1), ("W1", 1), ("W2", 2), ("W4", 4), ("opt", INF)):
            for kap in sorted({0.0, tr["kappa"]}):
                a_ = acc.setdefault(f"{key_}@k{kap:g}", [0, 0])
                cc, pp = _pol(R, E, C, W, 16.0, kap)
                a_[0] += cc; a_[1] += pp
    row = {kk: (c + f) / T for kk, (c, f) in acc.items()}
    best = {kk: min(v for k2, v in row.items() if k2.startswith(kk + "@k")) for kk in ("dfa", "dfa-fetch", "W1", "W2", "W4", "opt")}
    lr /= T
    out = dict(C=C, learned=lr, **best)
    out["closed_fetch_to_opt"] = (best["dfa-fetch"] - lr) / (best["dfa-fetch"] - best["opt"])
    out["closed_deployed_to_opt"] = (best["dfa"] - lr) / (best["dfa"] - best["opt"])
    out["closed_deployed_to_opt_single_read"] = (best["dfa"] - best["dfa-fetch"]) / (best["dfa"] - best["opt"])
    for w in ("W1", "W2", "W4"):
        out[f"closed_{w}"] = (best["dfa-fetch"] - best[w]) / (best["dfa-fetch"] - best["opt"])
    print(f"   {key} C={C} on AIME: deployed {best['dfa']:7.2f} dfa-fetch {best['dfa-fetch']:7.2f} learned {lr:7.2f} W1 {best['W1']:7.2f}"
          f" W2 {best['W2']:7.2f} W4 {best['W4']:7.2f} opt {best['opt']:7.2f} | of fetch->opt: learned {100 * out['closed_fetch_to_opt']:.1f}%"
          f" (W1 {100 * out['closed_W1']:.1f}, W2 {100 * out['closed_W2']:.1f}); of deployed->opt: single read"
          f" {100 * out['closed_deployed_to_opt_single_read']:.1f}%, + learned {100 * out['closed_deployed_to_opt']:.1f}%", flush=True)
    return out


def export(path, tr):
    """Binary for the engine: magic, version, L, E, k, NF, H, cross; half-lives[5]; weights[NF]; bias; pop[L][E];
    M[L][E][E]; PX[L][E][L][E] if cross (float32, little endian)."""
    m = tr["model"]
    with open(path, "wb") as f:
        f.write(b"ECLP")
        f.write(struct.pack("<7i", 1, tr["L"], tr["E"], tr["k"], NF, tr["H"], int(tr["cross"])))
        f.write(np.asarray(HL, np.float32).tobytes())
        f.write(np.asarray(m.coef_[0], np.float32).tobytes())
        f.write(np.asarray([m.intercept_[0]], np.float32).tobytes())
        f.write(np.asarray(tr["pops"], np.float32).tobytes())
        f.write(np.asarray(tr["Ms"], np.float32).tobytes())
        if tr["cross"]:
            f.write(np.asarray(tr["PX"], np.float32).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gpt-oss-120b,qwen3-30b-a3b")
    ap.add_argument("--cells", default="")
    ap.add_argument("--model", default="logit")
    ap.add_argument("--export-dir")
    ap.add_argument("--no-cross", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = json.load(open(a.out)) if a.out and os.path.exists(a.out) else {}
    for key in a.models.split(","):
        cells = [int(c) for c in a.cells.split(",")] if a.cells else CELLS[key]
        for C in cells:
            tr = train(key, C, a.model, cross=not a.no_cross)
            r = test(key, C, tr)
            r["coef"] = tr["model"].coef_[0].tolist() if hasattr(tr["model"], "coef_") else None
            out[f"{key} C{C}" + (" nocross" if a.no_cross else "")] = r
            if a.out:
                json.dump(out, open(a.out, "w"), indent=1)
            if a.export_dir and hasattr(tr["model"], "coef_"):
                os.makedirs(a.export_dir, exist_ok=True)
                export(os.path.join(a.export_dir, f"learned_{key}_C{C}{'_nocross' if a.no_cross else ''}.bin"), tr)


if __name__ == "__main__":
    main()

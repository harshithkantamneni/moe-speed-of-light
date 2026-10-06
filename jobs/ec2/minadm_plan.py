"""The minimum-admission MIN plan: among the schedules with the most hits, the one that admits fewest experts.

Belady's MIN with bypass has many hit-optimal schedules; the greedy one the engine's oracle follows admits a miss
whenever some resident is needed later, and may evict it again before its next use. Per layer this script selects
reuse intervals (an expert held in a slot from one use to its next, within one sequence, as the engine's foresight
ends at the sequence) under the engine's rule for a copy in the step: at step g the slots hold every expert whose
selected interval covers g (s < g <= e) plus every expert admitted at g, at most C:

    maximise  sum x_j  -  mu * sum a_j      x_j, a_j in [0, 1],  a_j >= x_j - x_prev(j),  mu = 1 / (n + 1)

so hits come first and admissions break ties. The summary also replays both schedules under the engine's rules
(simulate: the greedy MIN of the engine's oracle_plan_fetch, and the plan) for misses and admissions per token. The LP is solved with HiGHS; if a solution is fractional the layer is
re-solved as a MILP. Output, for the engine (LLAMA_EC_ORACLE_PLAN): one record per admission, int32 step, int16 layer,
int16 expert, int32 hold (the last step of its chain of hits), sorted by step; plus a JSON summary.

    python minadm_plan.py la.bin C out.plan [procs]
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog, milp, LinearConstraint, Bounds



def load_la(path):
    """the routing records of ec-bench --lookahead (as value_map.load_la; kept here so that this script needs only numpy
    and scipy): act [T, L, k], the sequence id of each record, the JSON beside the file"""
    meta = json.load(open(path + ".json"))
    L, k = int(meta["n_layer"]), int(meta["k"])
    raw = np.fromfile(path, np.int16)
    stride = 2 + L * (2 * k + 24)
    rec = raw[: len(raw) // stride * stride].reshape(-1, stride)
    act = np.stack([rec[:, 2 + l * (2 * k + 24): 2 + l * (2 * k + 24) + k] for l in range(L)], axis=1).astype(np.int64)
    return act, rec[:, 0].astype(np.int64), meta


def intervals(act_l, seqs):
    """reuse intervals (s, e, expert) of one layer within each sequence; prev[j] = the same expert's interval ending at s,
    or -1"""
    last, lastj, iv, prev = {}, {}, [], []
    for t in range(act_l.shape[0]):
        for e in dict.fromkeys(int(x) for x in act_l[t] if x >= 0):
            if e in last and seqs[last[e]] == seqs[t]:
                p = lastj.get(e, -1)
                prev.append(p if p >= 0 and iv[p][1] == last[e] else -1)
                iv.append((last[e], t, e))
                lastj[e] = len(iv) - 1
            else:
                lastj[e] = -1
            last[e] = t
    return iv, prev


def solve_layer(args):
    act_l, seqs, C = args
    T = act_l.shape[0]
    iv, prev = intervals(act_l, seqs)
    n = len(iv)
    if n == 0:
        return [], dict(n=0)
    # occupancy at step g: x_j for s_j < g <= e_j, a_j for s_j == g
    rows, cols = [], []
    for j, (s, e, _) in enumerate(iv):
        rows.extend(range(s + 1, e + 1)); cols.extend([j] * (e - s))
    Gx = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(T, n))
    Ga = sp.csr_matrix((np.ones(n), ([s for s, _, _ in iv], range(n))), shape=(T, n))
    r2, c2, v2 = [], [], []
    for j in range(n):
        r2 += [j, j]; c2 += [j, n + j]; v2 += [1.0, -1.0]
        if prev[j] >= 0:
            r2.append(j); c2.append(prev[j]); v2.append(-1.0)
    A = sp.vstack([sp.hstack([Gx, Ga]), sp.csr_matrix((v2, (r2, c2)), shape=(n, 2 * n))]).tocsr()
    b = np.concatenate([np.full(T, float(C)), np.zeros(n)])
    c = np.concatenate([-np.ones(n), np.full(n, 1.0 / (n + 1))])
    res = linprog(c, A_ub=A, b_ub=b, bounds=(0, 1), method="highs")
    z = res.x
    frac = int(np.sum((z > 1e-6) & (z < 1 - 1e-6)))
    used_milp = False
    if frac:
        r = milp(c, constraints=LinearConstraint(A, -np.inf, b), integrality=np.ones(2 * n), bounds=Bounds(0, 1))
        z = r.x
        used_milp = True
    x = z[:n] > 0.5
    # chains: an admission starts a chain of selected intervals; hold = the end of its last interval
    nxt = {p: j for j, p in enumerate(prev) if p >= 0}
    out = []
    for j in range(n):
        if x[j] and not (prev[j] >= 0 and x[prev[j]]):
            k = j
            while k in nxt and x[nxt[k]]:
                k = nxt[k]
            out.append((iv[j][0], iv[j][2], iv[k][1]))
    return out, dict(n=n, hits=int(x.sum()), admissions=len(out), fractional=frac, milp=used_milp)


INF = 1 << 60


def _next_uses(a, sq, E):
    """nu[t, e]: the first use of e after step t within the sequence (INF if none)"""
    T = a.shape[0]
    nu = np.full((T, E), INF, np.int64)
    cur = {}
    for t in range(T - 1, -1, -1):
        if t + 1 < T and sq[t + 1] != sq[t]:
            cur = {}
        for e, v in cur.items():
            nu[t, e] = v
        for e in a[t]:
            if e >= 0:
                cur[int(e)] = t
    return nu


def simulate(args):
    """one layer under the engine's rules for a copy in the step: the greedy MIN of oracle_plan_fetch, and the plan
    (victim: a resident whose hold has passed, not requested this step, needed furthest); misses and admissions"""
    a, sq, C, E, plan = args
    nu = _next_uses(a, sq, E)
    res, gm, ga = {}, 0, 0
    for t in range(a.shape[0]):
        req = list(dict.fromkeys(int(e) for e in a[t] if e >= 0)); sel = set(req)
        ms = [e for e in req if e not in res]; gm += len(ms)
        for e in sorted((e for e in ms if nu[t, e] < INF), key=lambda e: nu[t, e]):
            if len(res) < C:
                res[e] = 1; ga += 1; continue
            vs = [r for r in res if r not in sel]
            if not vs:
                break
            v = max(vs, key=lambda r: nu[t, r])
            if nu[t, v] <= nu[t, e]:
                break
            del res[v]; res[e] = 1; ga += 1
    by = {}
    for s_, e, h in plan:
        by.setdefault(s_, []).append((e, h))
    res, pm, pa, ref = {}, 0, 0, 0
    for t in range(a.shape[0]):
        req = list(dict.fromkeys(int(e) for e in a[t] if e >= 0)); sel = set(req)
        pm += sum(1 for e in req if e not in res)
        for e, h in by.get(t, []):
            if e in res:
                res[e] = max(res[e], h); continue
            if len(res) < C:
                res[e] = h; pa += 1; continue
            vs = [r for r in res if r not in sel and res[r] < t]
            if not vs:
                ref += 1; continue
            v = max(vs, key=lambda r: nu[t, r])
            del res[v]; res[e] = h; pa += 1
    return gm, ga, pm, pa, ref


def main():
    la, C, path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    procs = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    act, seqs, meta = load_la(la)
    T, L, k = act.shape
    t0 = time.time()
    with Pool(procs) as pool:
        res = pool.map(solve_layer, [(act[:, l, :], seqs, C) for l in range(L)])
    recs = []
    for l, (out, _) in enumerate(res):
        recs.extend((s, l, e, h) for s, e, h in out)
    recs.sort()
    dt = np.dtype([("step", "<i4"), ("layer", "<i2"), ("expert", "<i2"), ("hold", "<i4")])
    arr = np.array(recs, dtype=dt)
    arr.tofile(path)
    info = [r[1] for r in res]
    E = int(meta.get("n_expert", 0)) or int(act.max()) + 1
    with Pool(procs) as pool:
        sims = pool.map(simulate, [(act[:, l, :], seqs, C, E, res[l][0]) for l in range(L)])
    sm = np.array(sims, dtype=np.float64).sum(axis=0) / T
    summ = dict(la=la, C=C, records=int(T), layers=int(L), admissions=int(len(arr)), admissions_per_token=len(arr) / T,
                hits_per_token=sum(i.get("hits", 0) for i in info) / T, fractional_layers=sum(1 for i in info if i.get("fractional")),
                milp_layers=sum(1 for i in info if i.get("milp")),
                sim=dict(greedy_misses=sm[0], greedy_admissions=sm[1], plan_misses=sm[2], plan_admissions=sm[3], plan_refused=sm[4]),
                seconds=round(time.time() - t0, 1))
    json.dump(summ, open(path + ".json", "w"))
    print(json.dumps(summ))


if __name__ == "__main__":
    main()

"""How many admissions does a hit-optimal schedule need, and does sending every admission over the link move the bound?

Belady's MIN with bypass has many hit-optimal schedules. The greedy one in mosl.cachesim admits eagerly: some of its
admissions are evicted before their next use. Read once, every admission must cross the PCIe link to reach its slot, so
the number of admissions decides how much a one-read schedule loads the link. Per layer this script solves the
interval formulation of caching with bypass as a linear program:

    x_j in [0, 1]   expert held in a slot from one use to its next (a hit at the next use)
    sum of x_j over the intervals spanning a step boundary <= C
    a_j >= x_j - x_prev(j)   an admission starts each residency chain

minimising misses + mu * admissions, for a sweep of mu. At mu -> 0 it gives the fewest admissions among hit-optimal
schedules; the sweep traces the misses-versus-admissions frontier. The LP is a relaxation, so a bound computed from its
frontier is valid for every integral schedule. The link-aware host term is

    T_link = min over the frontier (M, A), and over x in [A, M], of  max(x S / B_link, (M - x) S / B_c, M S / B_host)

with B_link the probe's highest link rate and B_c its highest CPU rate: every admission crosses the link, bypassed
misses may take either path. Writes prereg/linkaware.json and paper/wsg_linkaware.tex.

    python scripts/linkaware.py            # about 25 minutes on two cores
"""
import json
import os
import re
import sys
import time
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
RES = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
TRACE = f"{RES}/084c_gptoss_trace@vast/route_aime25_gptoss.npz"
S = 13253760
MUS = (1e-4, 0.5, 1.0, 2.0, 4.0)


def layer_frontier(args):
    l, C = args
    act = np.load(TRACE)["act"].astype(np.int64)[:, l, :]
    T = act.shape[0]
    last, lastj, iv, prev = {}, {}, [], []
    for t in range(T):
        for e in act[t]:
            e = int(e)
            if e in last:
                iv.append((last[e], t))
                prev.append(lastj.get(e, -1))
                lastj[e] = len(iv) - 1
            last[e] = t
    n = len(iv)
    rows, cols = [], []
    for j, (s, e) in enumerate(iv):
        rows.extend(range(s, e)); cols.extend([j] * (e - s))
    G = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(T - 1, n))
    r2, c2, v2 = [], [], []
    for j in range(n):
        r2 += [j, j]; c2 += [j, n + j]; v2 += [1.0, -1.0]
        if prev[j] >= 0:
            r2.append(j); c2.append(prev[j]); v2.append(-1.0)
    A = sp.vstack([sp.hstack([G, sp.csr_matrix((T - 1, n))]), sp.csr_matrix((v2, (r2, c2)), shape=(n, 2 * n))]).tocsr()
    b = np.concatenate([np.full(T - 1, float(C)), np.zeros(n)])
    N = T * act.shape[1]
    out = []
    for mu in MUS:
        res = linprog(np.concatenate([-np.ones(n), mu * np.ones(n)]), A_ub=A, b_ub=b, bounds=(0, 1), method="highs")
        x, a = res.x[:n], res.x[n:]
        frac = float(np.mean((x > 1e-6) & (x < 1 - 1e-6)))
        out.append(dict(mu=mu, misses=float(N - x.sum()), adm=float(a.sum()), frac_fractional=frac))
    return l, out


def t_link(front, bc, bl, bh):
    """link-aware host term (s per token) over the convex hull of the frontier"""
    pts = sorted(front, key=lambda p: p[0])
    best = np.inf
    for (m1, a1), (m2, a2) in zip(pts, pts[1:] or pts):
        for w in np.linspace(0, 1, 51):
            M, Ad = m1 + (m2 - m1) * w, a1 + (a2 - a1) * w
            xs = np.linspace(Ad, M, 2001)
            best = min(best, float(np.min(np.maximum.reduce([xs * S / bl, (M - xs) * S / bc, np.full_like(xs, M * S / bh)]))))
    return best


def main():
    t0 = time.time()
    with Pool(2) as pool:
        res = pool.map(layer_frontier, [(l, C) for C in (14, 32) for l in range(36)])
    T = np.load(TRACE)["act"].shape[0]
    fr = {}
    for i, (l, out) in enumerate(res):
        C = 14 if i < 36 else 32
        for o in out:
            d = fr.setdefault(C, {}).setdefault(o["mu"], dict(misses=0.0, adm=0.0, fracmax=0.0))
            d["misses"] += o["misses"] / T; d["adm"] += o["adm"] / T; d["fracmax"] = max(d["fracmax"], o["frac_fractional"])
    print("frontier", {C: {mu: (round(v["misses"], 3), round(v["adm"], 3), round(v["fracmax"], 4)) for mu, v in d.items()} for C, d in fr.items()},
          f"{time.time() - t0:.0f}s")
    rs = {r["host"]: r for r in json.load(open(P("prereg", "readsched.json")))["hosts"]}
    hosts = []
    for name, r in rs.items():
        if not name.startswith(("095", "096", "097", "099", "100", "101", "102")):
            continue
        txt = open(f"{RES}/{name}/concur.txt").read()
        bcm = max(float(x) for x in re.findall(r"cpu_read_gbs t=\d+ ([\d.]+)", txt)) * 1e9
        h = dict(host=name, link_over_cpu=r["C14"]["link_over_cpu"])
        for C in (14, 32):
            c = r[f"C{C}"]
            bl, bh = c["B_link_max"] * 1e9, r["B_host"] * 1e9
            tb = c["R_star"] * S / bh
            front = [(v["misses"], v["adm"]) for v in fr[C].values()]
            amin = fr[C][MUS[0]]["adm"]
            h[f"C{C}"] = dict(t_bound_host_ms=1e3 * tb, t_link_ms=1e3 * t_link(front, bcm, bl, bh),
                              cap_greedy=min(1.0, c["R_star"] / c["A_star"] * bl / bh), cap_minadm=min(1.0, c["R_star"] / amin * bl / bh))
            h[f"C{C}"]["link_over_bound"] = h[f"C{C}"]["t_link_ms"] / h[f"C{C}"]["t_bound_host_ms"]
        hosts.append(h)
    greedy = {C: rs[next(iter(rs))][f"C{C}"]["A_star"] for C in (14, 32)}
    json.dump(dict(mus=MUS, frontier={C: d for C, d in fr.items()}, greedy_adm=greedy, hosts=hosts), open(P("prereg", "linkaware.json"), "w"), indent=1)
    M = {}
    for C, nm in ((14, "Low"), (32, "Mid")):
        a0 = fr[C][MUS[0]]
        M[f"laAdmMin{nm}"] = f"{a0['adm']:.1f}"; M[f"laAdmGreedy{nm}"] = f"{greedy[C]:.1f}"
        M[f"laAdmShareMin{nm}"] = f"{100 * a0['adm'] / a0['misses']:.0f}"
        M[f"laAdmShareGreedy{nm}"] = f"{100 * greedy[C] / a0['misses']:.0f}"
        M[f"laFrac{nm}"] = f"{100 * a0['fracmax']:.1f}"
        lo = [h[f"C{C}"]["link_over_bound"] for h in hosts]
        M[f"laOver{nm}Max"] = f"{100 * (max(lo) - 1):.0f}"
        cm = [h[f"C{C}"]["cap_minadm"] for h in hosts]
        M[f"laCapMin{nm}"] = f"{100 * min(cm):.0f}"
    M["laHosts"] = str(len(hosts))
    with open(P("paper", "wsg_linkaware.tex"), "w") as f:
        f.write("% generated by scripts/linkaware.py\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    for h in sorted(hosts, key=lambda h: h["link_over_cpu"]):
        print(h["host"][:14], round(h["link_over_cpu"], 2), {C: (round(h[f"C{C}"]["cap_greedy"], 2), round(h[f"C{C}"]["cap_minadm"], 2),
                                                                  round(h[f"C{C}"]["link_over_bound"], 4)) for C in (14, 32)})
    print(M)


if __name__ == "__main__":
    main()

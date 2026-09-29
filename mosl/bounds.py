"""Lower bounds on batch-1 decode time for offloaded MoE, in the corrected forms of 29 Sep 2026 (the reviews of
28 Sep found that the draft's Proposition 1 assumes each layer's dense work precedes and does not overlap its expert
work, which shared-expert overlap, CPU-side prefetch into the LLC or a CPU share of the dense work can violate).

Notation per decode token: D dense bytes read from GPU memory (attention, routers, norms, shared experts, LM head,
KV cache), s bytes per routed expert, k routed experts per MoE layer, L MoE layers, M = M*/N the misses per
layer-step of Belady's MIN with bypass under the budget (per-layer or pooled) on the measured text's routing.
Bandwidths in bytes/s: B_g GPU memory, B_c the CPU's read path, B_p the host-to-GPU link, B_h host DRAM (joint cap on
everything read from host memory: CPU reads plus link transfers). For a PHYSICAL bound use datasheet peaks; a bound on
measured bandwidths is a reference, not a floor (a measured run can beat it).

resource_bound      Proposition 1, resource form (any exact-routing policy): variables m (routed experts executed from
                    host memory per layer-step, by the CPU or by a zero-copy GPU read, which also crosses the link)
                    and x (expert loads crossing the link per layer-step); by the charging argument every GPU hit is
                    a MIN-bypass bridged gap or charged to a load, so m + x >= M. Optionally the CPU also runs a share
                    D_c of the dense work from host copies.
layered_bound       the refinement under (S1) the CPU reads no routed-expert bytes of layer l before layer l's router
                    output exists and (S2) no other GPU work of layer l overlaps the CPU's routed-expert work of
                    layer l: T >= max(T_D + L*max((k-m)a, m b), L x p, L(m+x) s / B_h), minimised over m + x >= M.
demand_bound        the corollary for demand-only policies (an expert is loaded only after it is requested at that
                    layer): no clairvoyant loads, so per layer-step the misses mu >= M are either run from host
                    memory or fetched in the step (f), and a policy may also run resident experts on the CPU (h):
                    T >= T_D + L * max((k-mu+f-h) a, (mu-f+h) b, f p, (mu+h) s / B_h) minimised over f, h.
All three are small linear programs (minimise the max of linear terms), solved exactly with scipy's HiGHS.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from . import cachesim


@dataclass
class Bw:
    """Bandwidths in GB/s. bw_host=None drops the joint host-DRAM term."""
    gpu: float
    cpu: float
    link: float
    host: float | None = None


def _lp(c, A, b, bounds):
    r = linprog(c, A_ub=np.array(A, float), b_ub=np.array(b, float), bounds=bounds, method="highs")
    if not r.success:
        raise RuntimeError(r.message)
    return r


def resource_bound(D, s, k, L, M, bw: Bw, dense_split=False):
    """Seconds per token (float) and the optimum (dict). Variables: t, m, x, dc (dc only with dense_split)."""
    Bg, Bc, Bp = bw.gpu * 1e9, bw.cpu * 1e9, bw.link * 1e9
    Bh = bw.host * 1e9 if bw.host else None
    # t >= (D - dc + L (k - m) s) / Bg ; t >= (L m s + dc) / Bc ; t >= L x s / Bp ; t >= (L (m + x) s + dc) / Bh
    A, b = [], []
    A.append([-1, -L * s / Bg, 0, -1 / Bg]); b.append(-(D + L * k * s) / Bg)
    A.append([-1, L * s / Bc, 0, 1 / Bc]); b.append(0)
    A.append([-1, 0, L * s / Bp, 0]); b.append(0)
    if Bh:
        A.append([-1, L * s / Bh, L * s / Bh, 1 / Bh]); b.append(0)
    A.append([0, -1, -1, 0]); b.append(-M)                   # m + x >= M
    bounds = [(0, None), (0, k), (0, None), (0, D if dense_split else 0)]
    r = _lp([1, 0, 0, 0], A, b, bounds)
    t, m, x, dc = r.x
    return float(t), dict(m=float(m), x=float(x), dc=float(dc))


def layered_bound(D, s, k, L, M, bw: Bw):
    """Refinement under (S1)-(S2). Variables t, v (per-layer-step expert time), m, x."""
    a, b_, p = s / (bw.gpu * 1e9), s / (bw.cpu * 1e9), s / (bw.link * 1e9)
    TD = D / (bw.gpu * 1e9)
    A, b = [], []
    A.append([0, -1, -a, 0]); b.append(-k * a)               # v >= (k - m) a
    A.append([0, -1, b_, 0]); b.append(0)                    # v >= m b
    A.append([-1, L, 0, 0]); b.append(-TD)                   # t >= TD + L v
    A.append([-1, 0, 0, L * p]); b.append(0)                 # t >= L x p
    if bw.host:
        h = s / (bw.host * 1e9)
        A.append([-1, 0, L * h, L * h]); b.append(0)         # t >= L (m + x) s / Bh
    A.append([0, 0, -1, -1]); b.append(-M)                   # m + x >= M
    r = _lp([1, 0, 0, 0], A, b, [(0, None), (0, None), (0, k), (0, None)])
    t, v, m, x = r.x
    return float(t), dict(m=float(m), x=float(x))


def demand_bound(D, s, k, L, M, bw: Bw):
    """Demand-only corollary. Variables u (per-layer-step time), mu, f, h."""
    a, b_, p = s / (bw.gpu * 1e9), s / (bw.cpu * 1e9), s / (bw.link * 1e9)
    TD = D / (bw.gpu * 1e9)
    A, b = [], []
    A.append([-1, -a, a, -a]); b.append(-k * a)              # u >= (k - mu + f - h) a
    A.append([-1, b_, -b_, b_]); b.append(0)                 # u >= (mu - f + h) b
    A.append([-1, 0, p, 0]); b.append(0)                     # u >= f p
    if bw.host:
        hh = s / (bw.host * 1e9)
        A.append([-1, hh, 0, hh]); b.append(0)               # u >= (mu + h) s / Bh
    A.append([0, -1, 1, 0]); b.append(0)                     # f <= mu
    A.append([0, 1, 0, 1]); b.append(k)                      # mu + h <= k
    r = _lp([1, 0, 0, 0], A, b, [(0, None), (M, k), (0, None), (0, None)])
    u, mu, f, h = r.x
    return float(TD + L * u), dict(mu=float(mu), f=float(f), h=float(h))


def mstar_segments(R, E, C, segments=None, pooled=False):
    """MIN-with-bypass misses per layer-step on routes R [L, T, k], with the cache contents chosen freely (optimally)
    at the start of every segment (a list of (start, end) step ranges; None = one segment). The free initial set is
    realised as a synthetic prefix requesting the C experts with the earliest first use, whose misses are not counted.
    pooled=True uses one LC-slot cache over the layer-interleaved stream."""
    L, T, k = R.shape
    segs = segments or [(0, T)]
    if C <= 0:
        return float(k)
    if C >= E:
        return 0.0
    total, n = 0.0, 0
    for s0, s1 in segs:
        seg = R[:, s0:s1]
        if pooled:
            # interleave layers, experts renamed l*E + e; initial set: the LC earliest-first-use (layer, expert) pairs
            flat = [(t, l, int(e)) for t in range(seg.shape[1]) for l in range(L) for e in seg[l, t]]
            first = {}
            for t, l, e in flat:
                first.setdefault(l * E + e, t)
            init = sorted(first, key=first.get)[: L * C]
            stream = np.array([[l * E + int(e) for e in seg[l, t]] for t in range(seg.shape[1]) for l in range(L)])
            pre = _prefix(init, k, E * L)
            miss, *_ = cachesim.simulate(np.concatenate([pre, stream]), E * L, L * C, "min", bypass=True)
            total += float(miss[len(pre):].sum()); n += stream.shape[0]
        else:
            for l in range(L):
                r = seg[l]
                first = {}
                for t in range(r.shape[0]):
                    for e in r[t]:
                        first.setdefault(int(e), t)
                init = sorted(first, key=first.get)[:C]
                pre = _prefix(init, k, E)
                miss, *_ = cachesim.simulate(np.concatenate([pre, r]), E, C, "min", bypass=True)
                total += float(miss[len(pre):].sum()); n += r.shape[0]
    return total / n


def _prefix(experts, k, E):
    """Rows of k distinct experts listing `experts`, padded with filler experts that are never requested again is
    impossible in general; instead pad with repeats of the listed ones (a repeat within a row is avoided by cycling)."""
    ex = list(experts)
    if not ex:
        return np.zeros((0, k), np.int64)
    rows = []
    for i in range(0, len(ex), k):
        row = ex[i:i + k]
        j = 0
        while len(row) < k:          # fill with other listed experts (all will be resident anyway)
            cand = ex[j % len(ex)]
            if cand not in row:
                row.append(cand)
            j += 1
            if j > 4 * len(ex) + k:
                break
        if len(row) < k:             # fewer listed experts than k: cannot build a full row
            row = row + [e for e in range(E) if e not in row][: k - len(row)]
        rows.append(row)
    return np.array(rows, np.int64)

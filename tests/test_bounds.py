"""mosl.bounds: each LP equals a brute-force grid minimum of its own statement; the layered form without the joint
DRAM term reproduces the draft's speed_of_light_time; the resource form is never above the layered one (it drops
assumptions S1-S2); the demand-only bound is never below the layered one; the joint DRAM term only raises bounds;
warm-start M* is never above the cold M*."""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl import bounds, cachesim  # noqa: E402
from mosl.bounds import Bw  # noqa: E402

CASES = [  # D bytes, s bytes, k, L, M, bandwidths GB/s
    (1.69e9, 13.25e6, 4, 36, 0.95, Bw(1792, 64, 63, 76.8)),
    (1.69e9, 13.25e6, 4, 36, 0.39, Bw(1558, 46.6, 47.6, None)),
    (0.4e9, 9.4e6, 8, 48, 2.5, Bw(1008, 60, 25, 89.6)),
    (0.9e9, 13.25e6, 4, 24, 1.44, Bw(600, 100, 25, 150)),
]


def grid_resource(D, s, k, L, M, bw, dense_split, n=161):
    Bg, Bc, Bp = bw.gpu * 1e9, bw.cpu * 1e9, bw.link * 1e9
    best = np.inf
    for x in np.linspace(0, max(M, 1e-9) * 1.2 + 0.5, n):
        for m in np.linspace(max(0, M - x), k, n):
            for dc in (np.linspace(0, D, 21) if dense_split else [0.0]):
                t = max((D - dc + L * (k - m) * s) / Bg, (L * m * s + dc) / Bc, L * x * s / Bp,
                        (L * (m + x) * s + dc) / (bw.host * 1e9) if bw.host else 0)
                best = min(best, t)
    return best


def test_resource_lp_matches_grid():
    for D, s, k, L, M, bw in CASES:
        for ds in (False, True):
            t, _ = bounds.resource_bound(D, s, k, L, M, bw, dense_split=ds)
            g = grid_resource(D, s, k, L, M, bw, ds)
            assert t <= g * (1 + 1e-9) and t >= g * 0.97, (t, g)


def test_layered_matches_draft_without_joint_term():
    from mosl.perfmodel import speed_of_light_time, Workload, HW
    from mosl.calc import PHYS, shape
    w = Workload(shape("openai/gpt-oss-120b"), 4.25, 16, ctx=640)
    for M in (0.1, 0.39, 0.95, 2.0):
        old = speed_of_light_time(w, HW(1558.4, 46.6, 47.6), PHYS, M)
        new, _ = bounds.layered_bound(w.dense_bytes, w.expert_bytes, w.k, w.shape.n_moe_layers, M, Bw(1558.4, 46.6, 47.6))
        assert abs(new - old) / old < 5e-3, (M, old, new)   # the draft minimised on an 801x801 grid


def test_orderings():
    for D, s, k, L, M, bw in CASES:
        r, _ = bounds.resource_bound(D, s, k, L, M, bw)
        y, _ = bounds.layered_bound(D, s, k, L, M, bw)
        d, _ = bounds.demand_bound(D, s, k, L, M, bw)
        assert r <= y * (1 + 1e-9) <= d * (1 + 1e-9) * (1 + 1e-9), (r, y, d)
        nb = Bw(bw.gpu, bw.cpu, bw.link, None)
        assert bounds.resource_bound(D, s, k, L, M, nb)[0] <= r * (1 + 1e-9)
        assert bounds.resource_bound(D, s, k, L, M, bw, dense_split=True)[0] <= r * (1 + 1e-9)


def test_demand_lp_matches_grid():
    for D, s, k, L, M, bw in CASES:
        a, b, p = s / (bw.gpu * 1e9), s / (bw.cpu * 1e9), s / (bw.link * 1e9)
        best = np.inf
        for mu in np.linspace(M, k, 61):
            for f in np.linspace(0, mu, 41):
                for h in np.linspace(0, k - mu, 41):
                    u = max((k - mu + f - h) * a, (mu - f + h) * b, f * p, (mu + h) * s / (bw.host * 1e9) if bw.host else 0)
                    best = min(best, u)
        g = D / (bw.gpu * 1e9) + L * best
        t, _ = bounds.demand_bound(D, s, k, L, M, bw)
        assert t <= g * (1 + 1e-9) and t >= g * 0.97, (t, g)


def test_warm_start_mstar():
    rng = np.random.default_rng(3)
    E, k, Lr, T = 32, 4, 3, 120
    p = 1.0 / np.arange(1, E + 1) ** 1.1; p /= p.sum()
    R = np.stack([np.stack([rng.choice(E, k, replace=False, p=p) for _ in range(T)]) for _ in range(Lr)])
    for C in (4, 8, 16):
        cold = np.mean([cachesim.simulate(R[l], E, C, "min", bypass=True)[0].mean() for l in range(Lr)])
        warm = bounds.mstar_segments(R, E, C)
        warm4 = bounds.mstar_segments(R, E, C, segments=[(0, 30), (30, 60), (60, 90), (90, 120)])
        assert warm <= cold + 1e-12 and warm4 <= warm + 1e-12, (C, cold, warm, warm4)
        pooled = bounds.mstar_segments(R, E, C, pooled=True)
        assert pooled <= warm + 1e-9, (C, pooled, warm)

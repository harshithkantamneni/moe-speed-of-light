"""sim_prefetch: with no prefetch it is the deployed-cache simulation (ecsim_fast, policy dfa); an oracle prefetch never
increases misses on a stream where the cache holds every selected expert; useful prefetches never exceed prefetches."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl import ecsim_fast  # noqa: E402
from sim_prefetch import _layer_pf  # noqa: E402


def _routes(T, E, k, rng, skew=1.2):
    p = 1.0 / np.arange(1, E + 1) ** skew
    p /= p.sum()
    return np.stack([rng.choice(E, k, replace=False, p=p) for _ in range(T)]).astype(np.int64)


def test_no_prefetch_equals_ecsim_fast():
    rng = np.random.default_rng(0)
    for E, k, C in ((32, 4, 4), (128, 4, 14), (128, 8, 32)):
        R = _routes(400, E, k, rng)
        P = np.full((len(R), 8), -1, np.int64)
        h, m, a = ecsim_fast.simulate_layer(R, E, C, "dfa", kappa=1.0, delay=2)
        h2, m2, a2, p2, u2 = _layer_pf(R, P, E, C, 16.0, 1.0, 2, 0, 0, 0.0)
        assert (h == h2).all() and (m == m2).all() and (a == a2).all() and p2.sum() == 0


def test_oracle_prefetch_bounds():
    rng = np.random.default_rng(1)
    E, k, C = 64, 4, 8
    R = _routes(500, E, k, rng)
    P = np.concatenate([R, np.full((len(R), 4), -1)], 1)
    h0, m0, *_ = _layer_pf(R, np.full_like(P, -1), E, C, 16.0, 1.0, 2, 0, 0, 0.0)
    h, m, a, p, u = _layer_pf(R, P, E, C, 16.0, 1.0, 2, k, k, 0.0)
    assert m.sum() == 0              # q = k, perfect prediction, C >= k: every selected expert is brought in
    assert (u <= p).all() and m.sum() <= m0.sum()
    h1, m1, a1, p1, u1 = _layer_pf(R, P, E, C, 16.0, 1.0, 2, 1, k, 0.0)
    assert m1.sum() <= m0.sum() and (u1 <= p1).all() and (p1 <= 1).all()

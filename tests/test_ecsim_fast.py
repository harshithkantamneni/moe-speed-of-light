"""The numba port of the expert-cache policy simulator must match the reference exactly."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mosl import ecsim, ecsim_fast  # noqa: E402

rng = np.random.default_rng(0)
n = 0
for trial in range(60):
    E = int(rng.choice([8, 32, 64, 128]))
    k = int(rng.choice([1, 2, 4, 8]))
    T = int(rng.integers(50, 400))
    C = int(rng.integers(1, E))
    # skewed, temporally correlated routing
    p = rng.dirichlet(np.full(E, 0.3))
    R = np.stack([rng.choice(E, k, replace=False, p=p) for _ in range(T)])
    R[1::3] = R[0:-1:3][: len(R[1::3])]
    R.sort(1)
    for pol in ("dfa", "lru", "lfu", "static"):
        kw = dict(kappa=float(rng.choice([0.0, 1.0, 2.0])), half_life=float(rng.choice([4.0, 16.0])), delay=int(rng.choice([1, 2])))
        init = list(rng.choice(E, C, replace=False)) if pol == "static" or rng.random() < 0.3 else None
        a = ecsim.simulate_layer(R, E, C, pol, init=init, **kw)
        b = ecsim_fast.simulate_layer(R, E, C, pol, init=init, **kw)
        for x, y in zip(a, b):
            assert np.array_equal(np.asarray(x), np.asarray(y)), (trial, pol, E, k, C, kw)
        n += 1
print(f"OK ecsim_fast == ecsim on {n} random traces")

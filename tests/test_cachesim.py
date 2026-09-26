import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mosl.cachesim import simulate, brute_force_opt, top_frequency_set


def rand_trace(rng, T, E, k):
    return np.stack([rng.choice(E, k, replace=False) for _ in range(T)])


def test_hand_lru():
    # E=3, k=1, cap=2: A B A C B -> misses A,B,(hit A),C evicts B, B miss
    R = np.array([[0], [1], [0], [2], [1]])
    miss, adm = simulate(R, 3, 2, "lru")
    assert miss.tolist() == [1, 1, 0, 1, 1], miss


def test_hand_min():
    # same trace, MIN evicts A at step 3 (A never reused) -> B hits
    R = np.array([[0], [1], [0], [2], [1]])
    miss, _ = simulate(R, 3, 2, "min")
    assert miss.tolist() == [1, 1, 0, 1, 0], miss


def test_min_is_optimal_fetch_and_bypass():
    rng = np.random.default_rng(0)
    for trial in range(300):
        E = int(rng.integers(3, 7)); k = int(rng.integers(1, 3)); cap = int(rng.integers(k, E))
        T = int(rng.integers(3, 9))
        R = rand_trace(rng, T, E, k)
        for bypass in (False, True):
            got = simulate(R, E, cap, "min", bypass=bypass)[0].sum()
            opt = brute_force_opt(R, E, cap, bypass)
            assert got == opt, (trial, bypass, got, opt, R.tolist(), cap)


def test_policy_ordering():
    rng = np.random.default_rng(1)
    R = rand_trace(rng, 2000, 32, 4)
    for cap in (4, 8, 16, 24):
        m = {p: simulate(R, 32, cap, p)[0].sum() for p in ("lru", "lfu", "min")}
        mb = simulate(R, 32, cap, "min", bypass=True)[0].sum()
        assert mb <= m["min"] <= min(m["lru"], m["lfu"]), (cap, m, mb)


def test_static():
    R = np.array([[0, 1], [0, 2], [0, 1], [3, 1]])
    s = top_frequency_set(R, 4, 2)
    assert s.tolist() == [True, True, False, False]
    miss, adm = simulate(R, 4, 2, "static", static_set=s)
    assert miss.tolist() == [0, 1, 0, 1] and adm.sum() == 0


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name)

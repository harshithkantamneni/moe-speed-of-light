"""The speed-of-light bound must not be beaten by any evaluated policy in its
class (per-layer capacity), including an optimistic oracle prefetcher, on any
platform - in particular the RTX 5090 + DDR4 + PCIe5 configuration that broke
an earlier, miss-only version of the bound."""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from mosl.archs import shape
from mosl.cachesim import simulate, simulate_rstar
from mosl.perfmodel import HW, Params, Workload, dynamic_time, speed_of_light_time
from mosl.traces import Trace

PLATS = [HW(272, 89.6, 15.75), HW(1008, 96.0, 31.5), HW(1792, 102.4, 63.0), HW(1792, 51.2, 63.0),
         HW(1792, 51.2, 126.0), HW(936, 204.8, 31.5)]
PARAMS = [Params(0.38, 0.60, 0.86, tau_us=0.0, tau_e_us=20.7), Params(0.39, 0.53, 0.86, tau_us=0.0), Params(0.8, 0.8, 0.9, tau_us=0.0), Params(0.3, 0.9, 0.95, tau_us=0.0)]


def test_bound_holds(model="qwen3-30b-a3b", tok="data/tok_qwen3_30b.jsonl", repo="Qwen/Qwen3-30B-A3B-Instruct-2507", layers=12):
    tr = Trace(f"data/traces/{model}", tok)
    tr.R = tr.R[:layers]
    s = shape(repo)
    w = Workload(s, 4.5, 8.5, ctx=1024)
    worst = 0.0
    for frac in (0.0625, 0.125, 0.25, 0.5, 0.75):
        cap = max(1, int(frac * tr.E))
        runs = {}
        for pol, byp in (("lru", False), ("min", False), ("min", True), ("lfu", False)):
            mm = np.stack([simulate(R, tr.E, cap, pol, bypass=byp)[0] for R in tr.R], 1)
            aa = np.stack([simulate(R, tr.E, cap, pol, bypass=byp)[1] for R in tr.R], 1)
            runs[(pol, byp)] = (mm, aa)
        runs[("dfa", True)] = (np.stack([simulate_rstar(R, tr.E, cap)[0] for R in tr.R], 1),
                               np.stack([simulate_rstar(R, tr.E, cap)[1] for R in tr.R], 1))
        M = runs[("min", True)][0].sum() / mm.size
        # scale layer count: evaluate per-layer quantities on `layers` layers, tile to the model's L
        rep = int(np.ceil(s.n_moe_layers / layers))
        for hw in PLATS:
            for p in PARAMS:
                lb = speed_of_light_time(w, hw, p, M)
                lbs = speed_of_light_time(w, hw, p, M, concurrent=False)
                for (pol, byp), (mm, aa) in runs.items():
                    m2, a2 = np.tile(mm, rep)[:, : s.n_moe_layers], np.tile(aa, rep)[:, : s.n_moe_layers]
                    cands = [dynamic_time(w, hw, p, m2, a2, "cpu")[0], dynamic_time(w, hw, p, m2, a2, "fetch")[0]]
                    if pol == "min" and byp:  # oracle prefetch: admitted misses become hits, loads still paid
                        cands.append(dynamic_time(w, hw, p, m2 - a2, a2, "cpu")[0])
                    for t in cands:
                        assert t >= lb * (1 - 1e-9), (frac, hw, p, pol, byp, t, lb)
                        worst = max(worst, lb / t)
                    ts = dynamic_time(w, hw, p, m2, a2, "cpu_seq")[0]
                    assert ts >= lbs * (1 - 1e-9), ("serial", frac, hw, p, pol, byp, ts, lbs)
    print(f"bound holds; tightest policy reaches {100*worst:.1f}% of the bound")


if __name__ == "__main__":
    test_bound_holds()

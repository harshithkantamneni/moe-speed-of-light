"""Speed-of-light and strong-baseline calculator for one batch-1 offloaded-MoE configuration.

Given a model, its quantisation, the hardware bandwidths and the GPU bytes given to routed experts, prints
  * the physical speed-of-light (Proposition 1: every efficiency 1, no latencies), per-layer and, with a trace,
    for a budget pooled over all layers;
  * the predicted equal-memory llama.cpp baseline (--n-cpu-moe n, n = MoE layers whose experts do not fit), from
    the decode model (M4) fitted on the third-party validation set, with its 10th-90th percentile error band.

    python -m mosl.calc --repo Qwen/Qwen3-30B-A3B-Instruct-2507 --b-exp 4.85 --b-dense 4.85 \
        --bw-gpu 1008 --bw-cpu 89.6 --bw-pcie 31.5 --budget-gb 10 [--trace qwen3-30b-a3b_fp8_S.npz] [--ctx 4096]

Bandwidths in GB/s (datasheet GPU, host DRAM peak or STREAM, PCIe peak); bits per weight include scales
(Q4_K_M ~4.85, Q8_0 8.5, MXFP4 4.25, bf16 16). Without --trace, misses come from independent uniform routing,
which overstates misses of real routing (a conservative, i.e. lower, speed-of-light).
"""
import argparse
import csv
import os

import numpy as np

from mosl import cachesim
from mosl.archs import shape
from mosl.perfmodel import HW, Params, Workload, speed_of_light_time, static_offload_time

PHYS = Params(eta_g=1.0, eta_c=1.0, eta_p=1.0, tau_us=0.0, tau_x_us=0.0, tau_e_us=0.0)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def m4_params():
    import sys
    sys.path.insert(0, ROOT)
    from mosl.validation_set import ROWS
    from scripts.validate import VARIANTS, fit
    return fit(ROWS, VARIANTS["M4 M1 + per-CPU-expert latency"])


def band():
    p = os.path.join(ROOT, "results", "validation.csv")
    rows = list(csv.DictReader(open(p)))
    r = [1 / (1 + float(x["rel_err_loso"])) for x in rows if x["kind"] == "static"]
    return float(np.percentile(r, 10)), float(np.percentile(r, 90))


def mstar(R, E, k, C, glob_cap=False, rng=None):
    if C <= 0:
        return float(k)
    if C >= E:
        return 0.0
    if R is None:
        rng = rng or np.random.default_rng(0)
        Rr = np.stack([rng.choice(E, k, replace=False) for _ in range(20000)])
        return float(cachesim.simulate(Rr, E, C, "min", bypass=True)[0].mean())
    if glob_cap:
        return float(cachesim.simulate_global([R[l] for l in range(R.shape[0])], E, C, "min", bypass=True)[0].mean())
    return float(np.mean([cachesim.simulate(R[l], E, C, "min", bypass=True)[0].mean() for l in range(R.shape[0])]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--b-exp", type=float, required=True)
    ap.add_argument("--b-dense", type=float, required=True)
    ap.add_argument("--bw-gpu", type=float, required=True)
    ap.add_argument("--bw-cpu", type=float, required=True)
    ap.add_argument("--bw-pcie", type=float, default=31.5)
    ap.add_argument("--budget-gb", type=float, help="GPU bytes for routed experts (GB)")
    ap.add_argument("--experts-per-layer", type=int, help="alternatively, resident experts per MoE layer")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--trace", help="trace pack (.npz) of the evaluated text; response tokens are used")
    a = ap.parse_args()
    s = shape(a.repo)
    w = Workload(s, a.b_exp, a.b_dense, ctx=a.ctx)
    E, k, L = s.n_experts, w.k, s.n_moe_layers
    B = a.experts_per_layer * L * w.expert_bytes if a.experts_per_layer is not None else a.budget_gb * 1e9
    C = int(min(E, B // (L * w.expert_bytes)))
    n_gpu = int(min(L, B // (E * w.expert_bytes)))
    hw = HW(a.bw_gpu, a.bw_cpu, a.bw_pcie)
    R = None
    if a.trace:
        from mosl.traces import load_pack
        pk = load_pack(a.trace)
        idx = np.concatenate([np.arange(st + P, st + n) for st, n, P in zip(pk["starts"], pk["seq_lens"], pk["prompt_lens"]) if n - P > 1])
        R = pk["routes"][:, idx]
    m = mstar(R, E, k, C)
    sol = 1 / speed_of_light_time(w, hw, PHYS, m)
    p = m4_params()
    base = 1 / static_offload_time(w, hw, p, L - n_gpu)[0]
    lo, hi = band()
    print(f"{a.repo}: L={L} MoE layers, E={E}, k={k}; expert {w.expert_bytes / 1e6:.2f} MB, dense {w.dense_bytes / 1e9:.2f} GB/token")
    print(f"budget {B / 1e9:.2f} GB -> {C} experts per layer (per-layer budget); llama.cpp --n-cpu-moe {L - n_gpu} "
          f"(first MoE layer index may shift for models whose first blocks are dense)")
    print(f"M* (MIN-bypass misses per layer-step) = {m:.3f} [{'trace' if R is not None else 'independent routing'}]")
    print(f"physical speed-of-light: {sol:.1f} tok/s (per-layer budget)")
    if R is not None and 0 < C < E:
        mg = mstar(R, E, k, C, glob_cap=True)
        print(f"physical speed-of-light, budget pooled over layers: {1 / speed_of_light_time(w, hw, PHYS, mg):.1f} tok/s (M* {mg:.3f})")
    print(f"predicted llama.cpp --n-cpu-moe {L - n_gpu}: {base:.1f} tok/s, band [{base * lo:.1f}, {base * hi:.1f}] "
          f"(M4, third-party fit; conservative for current llama.cpp)")


if __name__ == "__main__":
    main()

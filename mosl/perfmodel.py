"""Analytical batch-1 decode model for MoE LLMs with experts in host memory.

Per decode step (one token), with bytes moved as the only first-order cost:

  T = [D + X_gpu] / (eta_g * BW_gpu)            GPU-resident weights + KV cache
    + X_cpu / (eta_c * BW_cpu)                  experts executed on the CPU
    + X_fetch / (eta_p * BW_pcie)               experts copied host->GPU on demand
    + n_switch * tau                            GPU<->CPU hand-offs (per layer)

D is the per-token dense traffic (attention, shared experts, dense MLPs,
routers, norms, LM head) plus KV-cache reads; X_* are routed-expert bytes by
where they execute. Peak bandwidths come from datasheets; the eta's are
implementation efficiencies fitted once (per engine family) and tau is a fixed
per-layer synchronisation latency. The model is deliberately small enough to
derive on a whiteboard: every term is bytes / bandwidth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .archs import Shape, kv_bytes


@dataclass
class HW:
    bw_gpu: float        # GB/s, device memory peak
    bw_cpu: float        # GB/s, host DRAM peak (channels * MT/s * 8B)
    bw_pcie: float = 0.  # GB/s, host->device link peak


@dataclass
class Params:
    eta_g: float = 0.7
    eta_c: float = 0.6
    eta_p: float = 0.7
    tau_us: float = 20.0     # per GPU<->CPU hand-off, microseconds
    tau_g_us: float = 0.0    # fixed GPU cost per decoder layer (kernel launches), us
    tau_c_us: float = 0.0    # fixed CPU cost per layer executed on the CPU (thread barriers), us
    tau_x_us: float = 0.0    # fixed cost per host->device transfer, us
    tau_e_us: float = 0.0    # fixed cost per routed expert executed on the CPU (per-op sync), us
    extra: dict = field(default_factory=dict)


@dataclass
class Workload:
    shape: Shape
    b_exp: float                 # bits per routed-expert weight (incl. scales)
    b_dense: float               # bits per dense weight
    ctx: int = 512
    kv_bits: float = 16
    top_k: int | None = None     # override (e.g. KTransformers top-6 of 8)
    dense_on: str = "gpu"        # "gpu" | "cpu"

    @property
    def k(self):
        return self.top_k or self.shape.top_k

    @property
    def expert_bytes(self):
        return self.shape.expert_params * self.b_exp / 8

    @property
    def dense_bytes(self):
        s = self.shape
        return (s.dense_params_per_token + s.lm_head_params) * self.b_dense / 8 + kv_bytes(s, self.ctx, self.kv_bits)


def static_offload_time(w: Workload, hw: HW, p: Params, n_cpu_layers: int, n_switch: int | None = None):
    """llama.cpp --n-cpu-moe / -ot style: all routed experts of `n_cpu_layers`
    MoE layers live and execute on the CPU; everything else on the GPU.
    Returns seconds per token and a breakdown."""
    s = w.shape
    L = s.n_moe_layers
    x_cpu = n_cpu_layers * w.k * w.expert_bytes
    x_gpu = (L - n_cpu_layers) * w.k * w.expert_bytes
    if w.dense_on == "cpu":  # CPU-only execution
        t_gpu = 0.0
        t_cpu = ((w.dense_bytes + x_cpu + x_gpu) / (p.eta_c * hw.bw_cpu * 1e9) + s.n_layers * p.tau_c_us * 1e-6
                 + L * w.k * p.tau_e_us * 1e-6)
        t_sync = 0.0
    else:
        t_gpu = (w.dense_bytes + x_gpu) / (p.eta_g * hw.bw_gpu * 1e9) + s.n_layers * p.tau_g_us * 1e-6
        t_cpu = (x_cpu / (p.eta_c * hw.bw_cpu * 1e9) + n_cpu_layers * p.tau_c_us * 1e-6
                 + n_cpu_layers * w.k * p.tau_e_us * 1e-6)
        t_sync = (n_cpu_layers if n_switch is None else n_switch) * p.tau_us * 1e-6
    return t_gpu + t_cpu + t_sync, dict(gpu=t_gpu, cpu=t_cpu, sync=t_sync)


def fetch_time(w: Workload, hw: HW, p: Params, fetched_per_token: float, gpu_expert_execs: float | None = None,
               transfers_per_token: float | None = None):
    """Demand paging into a GPU expert cache: every miss is copied over PCIe
    and then executed on the GPU (Mixtral-offloading / MoE-Infinity style).
    Each transfer also pays a fixed latency tau_x (driver + framework)."""
    s = w.shape
    execs = s.n_moe_layers * w.k if gpu_expert_execs is None else gpu_expert_execs
    nx = fetched_per_token if transfers_per_token is None else transfers_per_token
    t_gpu = (w.dense_bytes + execs * w.expert_bytes) / (p.eta_g * hw.bw_gpu * 1e9) + s.n_layers * p.tau_g_us * 1e-6
    t_pcie = fetched_per_token * w.expert_bytes / (p.eta_p * hw.bw_pcie * 1e9) + nx * p.tau_x_us * 1e-6
    return t_gpu + t_pcie, dict(gpu=t_gpu, pcie=t_pcie)


# ---------------------------------------------------------------------------
# Dynamic caches driven by simulated miss histograms.
#
# miss[t, l] = misses at step t, layer l; adm[t, l] = admissions (PCIe copies).
# Strategies:
#   "fetch"   : misses copied over PCIe, then run on the GPU, serially.
#   "cpu"     : misses run on the CPU concurrently with the GPU's hits; the layer
#               takes the slower path, plus one hand-off if any miss.
#   "cpu_seq" : as "cpu" but CPU and GPU work serialise (llama.cpp-style graph).
# For "cpu"/"cpu_seq", admissions are asynchronous copies; the link must carry
# them within the token time (otherwise PCIe is the bottleneck).
# ---------------------------------------------------------------------------

def _rates(w, hw, p):
    s = w.expert_bytes
    return (s / (p.eta_g * hw.bw_gpu * 1e9), s / (p.eta_c * hw.bw_cpu * 1e9), s / (p.eta_p * hw.bw_pcie * 1e9))


def dense_time(w, hw, p):
    return w.dense_bytes / (p.eta_g * hw.bw_gpu * 1e9) + w.shape.n_layers * p.tau_g_us * 1e-6


def dynamic_time(w: Workload, hw: HW, p: Params, miss, adm, strategy: str):
    a, b, pf = _rates(w, hw, p)
    k = w.k
    hits = k - miss
    dense = dense_time(w, hw, p)
    if strategy == "fetch":
        per_layer = hits * a + miss * (pf + a + p.tau_x_us * 1e-6)
        t = dense + per_layer.sum(1)
        return float(t.mean()), dict(dense=dense, experts=float(per_layer.sum(1).mean()))
    if strategy in ("cpu", "cpu_seq"):
        comb = np.maximum if strategy == "cpu" else np.add
        per_layer = comb(hits * a, miss * (b + p.tau_e_us * 1e-6)) + (miss > 0) * p.tau_us * 1e-6
        t = dense + per_layer.sum(1)
        t_adm = adm.sum(1) * pf
        t = np.maximum(t, t_adm)
        return float(t.mean()), dict(dense=dense, experts=float(per_layer.sum(1).mean()), adm=float(t_adm.mean()))
    raise ValueError(strategy)


def speed_of_light_time(w: Workload, hw: HW, p: Params, min_misses_per_layer_step: float,
                        concurrent: bool = True, grid: int = 801):
    """Lower bound on seconds/token for ANY policy (static or dynamic, demand or
    prefetching) that keeps at most `cap` experts per layer resident on the GPU.

    Accounting: every routed-expert execution runs either on the GPU from a
    resident copy (cost a) or on the CPU (cost b). A resident copy needs a load
    over PCIe (cost pf, which may be overlapped with anything). Group each
    expert's requests into maximal runs served by one resident copy: a run of
    r requests needs one load and bridges r-1 reuse gaps. Belady-with-bypass
    (MIN-bypass) maximises the number of bridged gaps under the capacity limit
    (its hits are exactly bridged gaps), so for any policy
        GPU executions <= G* + loads,   i.e.   CPU executions >= M* - loads,
    with M* the MIN-bypass miss count. Per layer-step, time is at least
    max((k-m_c) a, m_c b) (concurrent) or (k-m_c) a + m_c b (serial), convex in
    m_c, so by Jensen the mean is bounded at the mean m_c >= max(0, M*/N - x),
    where x = loads per layer-step; the link must also carry the loads:
    time per token >= L x pf. We minimise over x >= 0. Hand-offs, per-transfer
    latency and the capacity cost of prefetching are dropped, so the result is
    a valid lower bound under the model."""
    a, b, pf = _rates(w, hw, p)
    k, L = w.k, w.shape.n_moe_layers
    dense = dense_time(w, hw, p)
    xs = np.linspace(0.0, max(min_misses_per_layer_step, 1e-9), grid)
    best = np.inf
    for x in xs:
        lo = max(0.0, min_misses_per_layer_step - x)
        mc = np.linspace(lo, k, grid)
        lay = np.maximum((k - mc) * a, mc * b) if concurrent else (k - mc) * a + mc * b
        t = max(dense + L * float(lay.min()), L * x * pf)
        best = min(best, t)
    return best

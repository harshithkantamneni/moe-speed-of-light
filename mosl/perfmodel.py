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
        t_cpu = (w.dense_bytes + x_cpu + x_gpu) / (p.eta_c * hw.bw_cpu * 1e9) + s.n_layers * p.tau_c_us * 1e-6
        t_sync = 0.0
    else:
        t_gpu = (w.dense_bytes + x_gpu) / (p.eta_g * hw.bw_gpu * 1e9) + s.n_layers * p.tau_g_us * 1e-6
        t_cpu = x_cpu / (p.eta_c * hw.bw_cpu * 1e9) + n_cpu_layers * p.tau_c_us * 1e-6
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
# miss_counts[t, l] = misses at step t, layer l; adm[t, l] = admissions.
# Strategies:
#   "fetch"  : misses copied over PCIe then run on GPU, serial per layer.
#   "cpu"    : misses run on CPU; GPU runs hits; per layer the slower of the two
#              paths (they execute concurrently), plus a hand-off if any miss.
#              Admissions are asynchronous PCIe copies that must fit in the
#              token's time budget (otherwise PCIe becomes the bottleneck).
# ---------------------------------------------------------------------------

def dynamic_time(w: Workload, hw: HW, p: Params, miss, adm, strategy: str, overlap_admissions: bool = True):
    s, k, eb = w.shape, w.k, w.expert_bytes
    T = miss.shape[0]
    g = p.eta_g * hw.bw_gpu * 1e9
    c = p.eta_c * hw.bw_cpu * 1e9
    pc = p.eta_p * hw.bw_pcie * 1e9
    dense = w.dense_bytes / g
    hits = k - miss
    if strategy == "fetch":
        per_layer = hits * eb / g + miss * (eb / pc + eb / g)
        t = dense + per_layer.sum(1)
        return float(t.mean()), dict(dense=dense, experts=float(per_layer.sum(1).mean()))
    if strategy == "cpu":
        gpu_path = hits * eb / g
        cpu_path = miss * eb / c + (miss > 0) * p.tau_us * 1e-6
        per_layer = np.maximum(gpu_path, cpu_path)
        t = dense + per_layer.sum(1)
        t_adm = adm.sum(1) * eb / pc
        if overlap_admissions:
            t = np.maximum(t, t_adm)
        else:
            t = t + t_adm
        return float(t.mean()), dict(dense=dense, experts=float(per_layer.sum(1).mean()), adm=float(t_adm.mean()))
    raise ValueError(strategy)


def speed_of_light_time(w: Workload, hw: HW, p: Params, min_misses_per_layer_step: float, grid: int = 2001):
    """Lower bound on seconds/token for ANY expert-placement policy that keeps at
    most `cap` experts per layer on the GPU, given the Belady-with-bypass
    minimum number of misses M (expressed per layer-step, M/N).

    Each routed-expert execution is (i) GPU-resident: cost a, (ii) fetched then
    run on the GPU, serially: cost pf + a, or (iii) run on the CPU concurrently
    with the GPU: cost b. Per layer-step, time >= max((k - m_c) a + m_f pf, m_c b)
    — a convex function of (m_f, m_c) — and any policy has mean m_f + m_c >= M/N.
    By Jensen's inequality the mean layer time is at least the minimum of that
    function at the mean, which we find on a grid. Admission traffic, hand-off
    latency and per-transfer latency are dropped (they only add time), so this
    is a valid bound under the model."""
    k = w.k
    s = w.expert_bytes
    a = s / (p.eta_g * hw.bw_gpu * 1e9)
    b = s / (p.eta_c * hw.bw_cpu * 1e9)
    pf = s / (p.eta_p * hw.bw_pcie * 1e9)
    mbar = min_misses_per_layer_step
    mc = np.linspace(0, k, grid)
    mf = np.maximum(0.0, mbar - mc)
    layer = np.maximum((k - mc) * a + mf * pf, mc * b)
    dense = w.dense_bytes / (p.eta_g * hw.bw_gpu * 1e9) + w.shape.n_layers * p.tau_g_us * 1e-6
    return dense + w.shape.n_moe_layers * float(layer.min())

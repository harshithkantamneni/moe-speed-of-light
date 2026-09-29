"""Trace-driven admission / miss-service simulation for the home PC (RTX 5090 + Ryzen 9 9950X, dual-channel DDR5,
PCIe 5.0 x16). CPU only. Every tok/s this prints is SIMULATED: routing traces drive an exact cache simulation, and an
explicit per-token time model with ESTIMATED constants (DEF, MODELS below; none measured on the target machine)
turns the per-(token, layer) hit/miss/admission counts into time.

Time model per decode token (all constants in DEF / MODELS; vary with --sens and --mc):

    T = TD + sum_l [ O + max(G_l, C_l, F_l, Dr_l) ] + n_adm * (tau_host + theta * s / B_P)
    TD   = D / (eta_g B_G) + L t_lay + t_tok                  dense GPU work, per-layer non-GEMV kernels, per-token host
    a    = s / (eta_g B_G) + t_eg                             GPU time per expert
    G_l  = (k - mc) a                                         hits + fetched experts on the GPU
    C_l  = [mc > 0] (f + mc (s / B_h' + tau_e))               helpers: per-request fixed + per-expert bytes + fixed
    F_l  = [mf > 0] (mf (s / B_P' + tau_x) + a)               critical fetches, pipelined with their GEMVs
    Dr_l = (mc + mf) s / B_D'                                 CPU reads and PCIe DMA share one DRAM
    mc = misses executed on the CPU, mf = misses fetched and run on the GPU (both from the cache simulation).
    Background admissions (bytes A per token) are spread uniformly over the token ("uniform"): they take r = A / T of
    DRAM and link bandwidth, B_D' = B_D - r, B_h' = min(B_h, B_D'), B_P' = min(B_P - r, B_D'). "optimistic" assumes
    they fit in idle DRAM time (r = 0 inside layers). Either way T >= link bytes / B_P and T >= DRAM bytes / B_D.

Cache engine (numba, lazy heaps): per-layer (C slots per layer) or pooled (L*C slots shared by all layers, the stream
interleaved layer by layer). Policies: LRU, DFA (decayed frequency, half-life h, admit iff score(e) > score(victim)
+ kappa), MIN (next use; bypass = MIN-bypass bound, fetch = Belady). Modes: 0 = misses on the CPU, admissions are
background copies published `delay` steps later (the deployed system: delay 2); 1 = per-step split, mf_table[m] of
the m misses are fetched (highest decayed score first) and admitted at once, the rest run on the CPU and are not
cached (fetch-all is mf_table[m] = m; FreeToken-style hybrid is the balanced table); 2 = as 1, but the CPU-run misses
are admitted in the background by the DFA rule (kappa, delay), i.e. the deployed path plus a synchronous fetch.

    python scripts/sim_homepc.py --models gpt-oss-120b  --out homepc_120b.json    # ~4 min on 2 cores
    python scripts/sim_homepc.py --models qwen3-30b-a3b --out homepc_qwen.json    # ~16 min (40k-token prefixes)
(run the two in parallel; one process for both takes ~20 min). Findings: research_notes/MoE offload system race plan/
admission_simulation.md.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl import cachesim, ecsim_fast  # noqa: E402
from mosl.traces import load_pack  # noqa: E402

RES = "/home/claude/gpu-branch/results"
PACKS = {("gpt-oss-120b", "S"): "036_trace_retry@40gb/gpt-oss-120b_S.npz",
         ("gpt-oss-120b", "G"): "035_trace_rest@40gb/gpt-oss-120b_G.npz",
         ("gpt-oss-120b", "D"): "035_trace_rest@40gb/gpt-oss-120b_D.npz",
         ("qwen3-30b-a3b", "S"): "035_trace_rest@40gb/qwen3-30b-a3b_fp8_S.npz",
         ("qwen3-30b-a3b", "G"): "035_trace_rest@40gb/qwen3-30b-a3b_fp8_G.npz",
         ("qwen3-30b-a3b", "D"): "035_trace_rest@40gb/qwen3-30b-a3b_fp8_D.npz"}

# Model constants. Bytes: R:data/gguf_bytes.json (gpt-oss-120b MXFP4; Qwen3-30B-A3B Q4_K_M). t_lay, t_eg: A10 nsys
# x 0.7 (BASE 5.4 assumption). f: A10 helper per-request fixed (50 / 19-26 us) x ~0.6 for a 16-core 5+ GHz desktop
# (estimate). Budgets: llama.cpp -ncmoe 32 / 27 / 20 of 36 -> 4 / 9 / 16 layers' worth -> C = 128 * n / 36.
MODELS = {
    "gpt-oss-120b": dict(L=36, E=128, k=4, s=61073326080 / (36 * 128), D=1070339328 + 615329280, t_lay=43e-6,
                         t_eg=2.9e-6, f=30e-6, C=(14, 32, 57), budget_names=("ncmoe32", "ncmoe27", "ncmoe20"),
                         kappa_now=1.0, max_tokens=0),
    "qwen3-30b-a3b": dict(L=48, E=128, k=8, s=17553162240 / (48 * 128), D=567271424 + 255252480, t_lay=34e-6,
                          t_eg=0.4e-6, f=15e-6, C=(14, 32, 57), budget_names=("11%", "25%", "44%"),
                          kappa_now=2.0, max_tokens=40000),
}

# Machine constants (estimates, not measured). B_h = eta_h * B_D unless given.
DEF = dict(B_G=1792e9, eta_g=0.5, B_D=57e9, eta_h=0.8, B_h=None, B_P=50e9, tau_e=10e-6, O=24e-6, tau_x=5e-6,
           tau_host=3e-6, t_tok=150e-6, theta=0.0, f=None)

SENS = {  # one-at-a-time ranges (task item 4, plus the other estimated constants)
    "B_D": [40e9, 50e9, 57e9, 70e9], "B_P": [30e9, 40e9, 50e9, 52e9], "tau_e": [5e-6, 10e-6, 21e-6],
    "O": [10e-6, 24e-6, 38e-6], "f": [10e-6, 30e-6, 50e-6], "eta_g": [0.4, 0.5, 0.6], "theta": [0.0, 0.25, 0.5],
}

INF = np.int64(1 << 62)


# ----------------------------------------------------------------------------------------------------- cache engine
@njit(cache=False)
def _hpush(hk, hg, hn, p, key, g):
    i = hn[p]
    hn[p] = i + 1
    hk[p, i] = key
    hg[p, i] = g
    while i > 0:
        par = (i - 1) >> 1
        if hk[p, par] <= hk[p, i]:
            break
        hk[p, par], hk[p, i] = hk[p, i], hk[p, par]
        hg[p, par], hg[p, i] = hg[p, i], hg[p, par]
        i = par


@njit(cache=False)
def _hpop(hk, hg, hn, p):
    key = hk[p, 0]
    g = hg[p, 0]
    n = hn[p] - 1
    hn[p] = n
    if n > 0:
        hk[p, 0] = hk[p, n]
        hg[p, 0] = hg[p, n]
        i = 0
        while True:
            c = 2 * i + 1
            if c >= n:
                break
            if c + 1 < n and hk[p, c + 1] < hk[p, c]:
                c += 1
            if hk[p, i] <= hk[p, c]:
                break
            hk[p, c], hk[p, i] = hk[p, i], hk[p, c]
            hg[p, c], hg[p, i] = hg[p, i], hg[p, c]
            i = c
    return key, g


@njit(cache=False)
def _push(hk, hg, hn, p, key, g, slots, size, curkey, state):
    if hn[p] >= hk.shape[1]:  # compact: one entry per resident
        hn[p] = 0
        for s in range(size[p]):
            x = slots[p, s]
            if x >= 0 and state[x] != 0:
                _hpush(hk, hg, hn, p, curkey[x], x)
    _hpush(hk, hg, hn, p, key, g)


@njit(cache=False)
def _victim(hk, hg, hn, p, r, excl_req, reqrow, state, curkey, aside, slots, size):
    """Pop the valid resident with the smallest key, skipping (and restoring) excluded ones."""
    na = 0
    v = -1
    vkey = 0.0
    while hn[p] > 0:
        key, g = _hpop(hk, hg, hn, p)
        if state[g] == 0 or key != curkey[g]:
            continue  # stale
        if (excl_req and reqrow[g] == r) or state[g] == 2:
            aside[na] = g
            na += 1
            continue
        v = g
        vkey = key
        break
    for i in range(na):
        _push(hk, hg, hn, p, curkey[aside[i]], aside[i], slots, size, curkey, state)
    return v, vkey


@njit(cache=False)
def _engine(Rg, nxt, L, pooled, cap, pol, mode, kappa, lam, delay, mf_table):
    """Rg [T*L, k] global expert ids (row t*L + l). pol 0 LRU, 1 DFA, 2 MIN. mode 0 CPU + background admission,
    1 split by mf_table. Returns mc, mf, adm as [T, L] int8."""
    TL, k = Rg.shape
    T = TL // L
    G = 0
    for r in range(TL):
        for j in range(k):
            if Rg[r, j] + 1 > G:
                G = Rg[r, j] + 1
    P = 1 if pooled else L
    HM = 4 * cap + 64
    hk = np.empty((P, HM))
    hg = np.empty((P, HM), np.int64)
    hn = np.zeros(P, np.int64)
    slots = np.full((P, max(cap, 1)), -1, np.int64)
    slot_of = np.full(G, -1, np.int64)
    size = np.zeros(P, np.int64)
    state = np.zeros(G, np.int8)          # 0 absent, 1 resident, 2 loading
    curkey = np.zeros(G)
    score = np.zeros(G)
    last = np.zeros(G, np.int64)
    nu = np.full(G, INF, np.int64)
    reqrow = np.full(G, -1, np.int64)
    fq_g = np.empty(G + 1, np.int64)
    fq_r = np.empty(G + 1, np.int64)
    fh = 0
    fn = 0
    aside = np.empty(G + 1, np.int64)
    gs = np.empty(k, np.int64)
    miss = np.empty(k, np.int64)
    mc = np.zeros((T, L), np.int8)
    mf = np.zeros((T, L), np.int8)
    adm = np.zeros((T, L), np.int8)
    for r in range(TL):
        t = r // L
        l = r - t * L
        p = 0 if pooled else l
        while fn > 0 and fq_r[fh] <= r:          # publish completed admissions
            g = fq_g[fh]
            fh = (fh + 1) % (G + 1)
            fn -= 1
            if state[g] == 2:
                state[g] = 1
        for j in range(k):
            g = Rg[r, j]
            gs[j] = g
            reqrow[g] = r
            score[g] = score[g] * np.exp(-lam * (t - last[g])) + 1.0
            last[g] = t
            if pol == 0:
                curkey[g] = float(r)
            elif pol == 1:
                curkey[g] = np.log(score[g]) + lam * t
            else:
                nu[g] = nxt[r, j]
                curkey[g] = -float(nu[g])
            if state[g] != 0:
                _push(hk, hg, hn, p, curkey[g], g, slots, size, curkey, state)
        nm = 0
        for j in range(k):
            if state[gs[j]] != 1:
                miss[nm] = gs[j]
                nm += 1
        nf = 0
        if mode != 0:
            # fetch the mf_table[nm] misses with the highest decayed score (never a loading one; selection sort)
            nc = 0
            for a_ in range(nm):
                b_ = a_
                for c_ in range(a_ + 1, nm):
                    pb = -1.0 if state[miss[b_]] == 2 else score[miss[b_]]
                    pc = -1.0 if state[miss[c_]] == 2 else score[miss[c_]]
                    if pc > pb:
                        b_ = c_
                miss[a_], miss[b_] = miss[b_], miss[a_]
                if state[miss[a_]] != 2:
                    nc += 1
            nf = min(mf_table[nm], nc)
            for q in range(nf):
                e = miss[q]
                if cap == 0:
                    continue
                if size[p] < cap:
                    s_ = size[p]
                    size[p] += 1
                else:
                    v, vkey = _victim(hk, hg, hn, p, r, True, reqrow, state, curkey, aside, slots, size)
                    if v < 0:
                        continue
                    if mode == 1 and pol == 1 and not (score[e] > np.exp(vkey - lam * t) + kappa):
                        _push(hk, hg, hn, p, vkey, v, slots, size, curkey, state)
                        continue  # run from a staging buffer, not cached
                    state[v] = 0
                    s_ = slot_of[v]
                    slot_of[v] = -1
                slots[p, s_] = e
                slot_of[e] = s_
                state[e] = 1
                _push(hk, hg, hn, p, curkey[e], e, slots, size, curkey, state)
        mc[t, l] = nm - nf
        mf[t, l] = nf
        if mode != 1:   # background admission of CPU-run misses (mode 0: all misses; mode 2: the unfetched ones)
            na = 0
            for q in range(nf, nm):
                e = miss[q]
                if state[e] == 2 or cap == 0:
                    continue
                if size[p] < cap:
                    if pol == 2 and nu[e] >= INF:
                        continue
                    s_ = size[p]
                    size[p] += 1
                else:
                    v, vkey = _victim(hk, hg, hn, p, r, pol != 2, reqrow, state, curkey, aside, slots, size)
                    if v < 0:
                        continue
                    ok = True
                    if pol == 1:
                        ok = score[e] > np.exp(vkey - lam * t) + kappa
                    elif pol == 2:
                        ok = nu[e] < nu[v]
                    if not ok:
                        _push(hk, hg, hn, p, vkey, v, slots, size, curkey, state)
                        continue
                    state[v] = 0
                    s_ = slot_of[v]
                    slot_of[v] = -1
                slots[p, s_] = e
                slot_of[e] = s_
                if delay > 0:
                    state[e] = 2
                    fq_g[(fh + fn) % (G + 1)] = e
                    fq_r[(fh + fn) % (G + 1)] = r + delay * L
                    fn += 1
                else:
                    state[e] = 1
                _push(hk, hg, hn, p, curkey[e], e, slots, size, curkey, state)
                na += 1
            adm[t, l] = na
    return mc, mf, adm


# ----------------------------------------------------------------------------------------------------- traces
def load_stream(model, arm, max_tokens):
    pk = load_pack(os.path.join(RES, PACKS[(model, arm)]))
    idx, n = [], 0
    for st, ln, P in zip(pk["starts"], pk["seq_lens"], pk["prompt_lens"]):
        if ln - P <= 1:
            continue
        idx.append(np.arange(st + P, st + ln))
        n += ln - P
        if max_tokens and n >= max_tokens:
            break
    R = pk["routes"][:, np.concatenate(idx)]              # [L, T, k]
    L, T, k = R.shape
    E = int(pk["E"])
    Rg = np.ascontiguousarray((R + (np.arange(L) * E)[:, None, None]).transpose(1, 0, 2).reshape(T * L, k))
    return dict(R=R, Rg=Rg, nxt=cachesim.next_use(Rg, L * E), L=L, T=T, k=k, E=E, convs=len(idx))


# ----------------------------------------------------------------------------------------------------- time model
def consts(**over):
    c = dict(DEF)
    c.update(over)
    return c


def _derived(M, c):
    s, k = M["s"], M["k"]
    f = M["f"] if c["f"] is None else c["f"]
    Bh = c["eta_h"] * c["B_D"] if c["B_h"] is None else min(c["B_h"], c["B_D"])
    a = s / (c["eta_g"] * c["B_G"]) + M["t_eg"]
    TD = M["D"] / (c["eta_g"] * c["B_G"]) + M["L"] * M["t_lay"] + c["t_tok"]
    return s, k, f, Bh, a, TD


def summarize(mc, mf, adm, k):
    T, L = mc.shape
    N = np.bincount(mc.ravel().astype(np.int64) * (k + 1) + mf.ravel(), minlength=(k + 1) ** 2).reshape(k + 1, k + 1)
    return dict(N=N, T=T, L=L, adm_tok=float(adm.sum()) / T,
                hit=1.0 - float((N * (np.arange(k + 1)[:, None] + np.arange(k + 1)[None, :])).sum()) / (T * L * k),
                cpu_tok=float((N * np.arange(k + 1)[:, None]).sum()) / T,
                fetch_tok=float((N * np.arange(k + 1)[None, :]).sum()) / T)


def tok_time(opt, M, c, interf="uniform"):
    """Mean seconds per token from the (mc, mf) histogram (exact for the mean: layers add, max is per row)."""
    s, k, f, Bh0, a, TD = _derived(M, c)
    N, Tn = opt["N"], opt["T"]
    mc = np.arange(k + 1)[:, None].astype(float)
    mf = np.arange(k + 1)[None, :].astype(float)
    A = opt["adm_tok"] * s
    r = 0.0
    Tm = None
    for _ in range(200):
        BDr = max(c["B_D"] - r, 0.05 * c["B_D"])
        Bh = min(Bh0, BDr)
        BPr = max(min(c["B_P"] - r, BDr), 0.05 * c["B_P"])
        G = (k - mc) * a + 0 * mf
        C = np.where(mc > 0, f + mc * (s / Bh + c["tau_e"]), 0.0) + 0 * mf
        F = np.where(mf > 0, mf * (s / BPr + c["tau_x"]) + a, 0.0) + 0 * mc
        Dr = (mc + mf) * s / BDr
        lay = c["O"] + np.maximum(np.maximum(G, C), np.maximum(F, Dr))
        Tm = TD + float((N * lay).sum()) / Tn + opt["adm_tok"] * (c["tau_host"] + c["theta"] * s / c["B_P"])
        if interf == "optimistic" or A == 0:
            break
        rn = A / Tm
        if abs(rn - r) < 1e-4 * max(rn, 1.0):
            break
        r = 0.5 * r + 0.5 * rn
    link = (A + opt["fetch_tok"] * s) / c["B_P"]
    dram = (A + (opt["fetch_tok"] + opt["cpu_tok"]) * s) / c["B_D"]
    return max(Tm, link, dram)


def sol_time(Mstar, M, c, grid=201):
    """Speed-of-light (sol_a10.bound) with these constants, no fixed CPU/link costs, plus the shared-DRAM term
    L (mc + x) s / B_D: x PCIe loads and mc CPU experts per layer-step, mc >= M* - x."""
    s, k, f, Bh, a, TD = _derived(M, c)
    L = M["L"]
    b, p = s / Bh, s / min(c["B_P"], c["B_D"])
    xs = np.linspace(0.0, max(Mstar, 1e-9), grid)[:, None]
    lo = np.maximum(0.0, Mstar - xs)
    mcs = lo + (k - lo) * np.linspace(0, 1, grid)[None, :]
    lay = np.maximum((k - mcs) * a, mcs * b)
    T = np.maximum(np.maximum(TD + L * lay, L * xs * p), L * (mcs + xs) * s / c["B_D"])
    return float(T.min())


def split_table(M, c):
    """m -> number of misses to fetch so that CPU and PCIe finish together (argmin of the layer time)."""
    s, k, f, Bh, a, TD = _derived(M, c)
    ce = s / Bh + c["tau_e"]
    pe = s / min(c["B_P"], c["B_D"]) + c["tau_x"]
    tab = np.zeros(k + 1, np.int64)
    for m in range(k + 1):
        best = None
        for x in range(m + 1):
            y = m - x
            t = max((k - y) * a, (f + y * ce) if y else 0.0, (x * pe + a) if x else 0.0, m * s / c["B_D"])
            if best is None or t <= best * (1 + 1e-9):   # ties -> fetch more (a fetched expert is cached for free)
                best, tab[m] = t, x
    return tab


def rstar(M, c):
    s, k, f, Bh, a, TD = _derived(M, c)
    b = s / Bh + c["tau_e"]
    p = s / min(c["B_P"], c["B_D"]) + c["tau_x"]
    return dict(a_us=a * 1e6, b_us=b * 1e6, p_us=p * 1e6, rstar_link=p / (b - a), rstar_dram=(s / c["B_D"]) / (b - a))


# ----------------------------------------------------------------------------------------------------- options
LAM16 = np.log(2) / 16


def run(tr, C, pooled, pol, mode, kappa=-1e30, hl=16.0, delay=2, table=None):
    cap = C * tr["L"] if pooled else C
    tab = np.zeros(tr["k"] + 1, np.int64) if table is None else np.asarray(table, np.int64)
    mc, mf, adm = _engine(tr["Rg"], tr["nxt"], tr["L"], pooled, cap, pol, mode, float(kappa), np.log(2) / hl,
                          delay, tab)
    return summarize(mc, mf, adm, tr["k"])


def run_cachesim(tr, C, policy):
    """Per-layer LFU (cumulative counts, always admit, bypass, no publication delay) or hindsight static set."""
    ms, ads = [], []
    for R in tr["R"]:
        st = cachesim.top_frequency_set(R, tr["E"], C) if policy == "static" else None
        m, a = cachesim.simulate(R, tr["E"], C, policy, bypass=True, static_set=st)
        ms.append(m)
        ads.append(a)
    mc = np.stack(ms, 1).astype(np.int8)
    return summarize(mc, np.zeros_like(mc), np.stack(ads, 1), tr["k"])


def option_set(tr, M, C, full=True, kappas=(0.0, 0.5, 1.0, 2.0, 4.0)):
    o = {}
    o["DFA k=now (current)"] = run(tr, C, False, 1, 0, M["kappa_now"])
    if full:
        o["DFA always-admit"] = run(tr, C, False, 1, 0, -1e30)
        for kp in kappas:
            o[f"DFA k={kp:g}"] = run(tr, C, False, 1, 0, kp)
        for hl in (8.0, 32.0):
            o[f"DFA k=0.5 hl={hl:g}"] = run(tr, C, False, 1, 0, 0.5, hl=hl)
        o["LRU"] = run(tr, C, False, 0, 0)
        o["LFU (cum., no delay)"] = run_cachesim(tr, C, "lfu")
        o["static top-freq (hindsight)"] = run_cachesim(tr, C, "static")
        o["pooled LRU"] = run(tr, C, True, 0, 0)
        for kp in (0.0, 0.5, 1.0):
            o[f"pooled DFA k={kp:g}"] = run(tr, C, True, 1, 0, kp)
        o["pooled MIN-bypass"] = run(tr, C, True, 2, 0, delay=0)
        full_tab = np.arange(tr["k"] + 1)
        o["fetch-all LRU"] = run(tr, C, False, 0, 1, table=full_tab)
        o["fetch-all DFA-evict"] = run(tr, C, False, 1, 1, table=full_tab)
        o["fetch-all Belady"] = run(tr, C, False, 2, 1, table=full_tab)
        o["pooled fetch-all LRU"] = run(tr, C, True, 0, 1, table=full_tab)
        o["pooled fetch-all Belady"] = run(tr, C, True, 2, 1, table=full_tab)
    o["MIN-bypass"] = run(tr, C, False, 2, 0, delay=0)
    o["DFA k=0"] = o.get("DFA k=0") or run(tr, C, False, 1, 0, 0.0)
    return o


SPLITS = {  # name -> (pooled, pol, kappa, mode). mode 1: CPU-run misses are not cached; mode 2: they are admitted in
    # the background by the DFA rule with this kappa (the deployed path), fetched ones are cached at once.
    "split DFA-evict": (False, 1, -1e30, 1),
    "split DFA k=0": (False, 1, 0.0, 1),
    "split LRU": (False, 0, -1e30, 1),
    "split+bg DFA k=1": (False, 1, 1.0, 2),
    "pooled split LRU (FreeToken-like)": (True, 0, -1e30, 1),
    "pooled split DFA-evict": (True, 1, -1e30, 1),
    "pooled split+bg DFA k=1": (True, 1, 1.0, 2),
}


MC_SPLITS = ("split DFA-evict", "split+bg DFA k=1", "pooled split LRU (FreeToken-like)")


def split_opts(tr, M, C, c, cache, names=None):
    tab = tuple(split_table(M, c).tolist())
    out = {}
    for n, (pooled, pol, kp, mode) in SPLITS.items():
        if names is not None and n not in names:
            continue
        key = (n, C, tab)
        if key not in cache:
            cache[key] = run(tr, C, pooled, pol, mode, kp, table=np.array(tab))
        out[n] = cache[key]
    return out, tab


def evaluate(opts, M, c, Mstar, Mstar_pool, interf="uniform"):
    sol = sol_time(Mstar, M, c)
    solp = sol_time(Mstar_pool, M, c)
    rows = {}
    for n, o in opts.items():
        T = tok_time(o, M, c, interf)
        rows[n] = dict(tok_s=1 / T, of_sol=sol / T, hit=o["hit"], adm_tok=o["adm_tok"], cpu_tok=o["cpu_tok"],
                       fetch_tok=o["fetch_tok"])
    return dict(sol_tok_s=1 / sol, sol_pooled_tok_s=1 / solp, rows=rows)


# ----------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="gpt-oss-120b,qwen3-30b-a3b")
    ap.add_argument("--arms", default="S,G,D")
    ap.add_argument("--mc", type=int, default=200, help="Monte Carlo draws over the plausible constant ranges")
    ap.add_argument("--max-tokens", type=int, default=None, help="override per-model decode-token cap")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a DEF constant with a measured value (SI units), e.g. --set B_D=52e9")
    a = ap.parse_args()
    for kv in a.set:
        k, v = kv.split("=", 1)
        DEF[k] = float(v)
    out = {"DEF": DEF, "MODELS": {m: {k: v for k, v in MODELS[m].items()} for m in MODELS}, "results": {}}
    t0 = time.time()
    for model in a.models.split(","):
        M = MODELS[model]
        cap_tok = M["max_tokens"] if a.max_tokens is None else a.max_tokens
        res = out["results"][model] = {}
        c0 = consts()
        res["rstar_default"] = rstar(M, c0)
        res["rstar_grid"] = [dict(B_D=bd, B_P=bp, tau_e=te, **rstar(M, consts(B_D=bd, B_P=bp, tau_e=te)))
                             for bd in SENS["B_D"] for bp in SENS["B_P"] for te in SENS["tau_e"]]
        for arm in a.arms.split(","):
            tr = load_stream(model, arm, cap_tok)
            print(f"[{time.time() - t0:6.0f}s] {model} {arm}: {tr['T']} decode tokens, {tr['convs']} conversations",
                  flush=True)
            ra = res[arm] = dict(tokens=tr["T"], convs=tr["convs"], budgets={})
            full = arm == "S"
            for C, bname in zip(M["C"], M["budget_names"]):
                opts = option_set(tr, M, C, full=full)
                Mstar = (1 - opts["MIN-bypass"]["hit"]) * M["k"]
                pool = opts.get("pooled MIN-bypass") or run(tr, C, True, 2, 0, delay=0)
                Mstar_pool = (1 - pool["hit"]) * M["k"]
                cache = {}
                sp, tab = split_opts(tr, M, C, c0, cache, None if full else list(MC_SPLITS))
                opts.update(sp)
                if not full:
                    opts["fetch-all LRU"] = run(tr, C, False, 0, 1, table=np.arange(M["k"] + 1))
                    opts["pooled LRU"] = run(tr, C, True, 0, 0)
                b = ra["budgets"][bname] = dict(C=C, Mstar=Mstar, Mstar_pool=Mstar_pool, split_table=list(tab))
                b["default"] = evaluate(opts, M, c0, Mstar, Mstar_pool)
                b["optimistic_interference"] = evaluate(opts, M, c0, Mstar, Mstar_pool, "optimistic")
                print(f"[{time.time() - t0:6.0f}s]   C={C}: SoL {b['default']['sol_tok_s']:.1f}, "
                      + ", ".join(f"{n} {r['tok_s']:.1f}" for n, r in b["default"]["rows"].items()
                                  if n in ("DFA k=now (current)", "DFA k=0", "LRU", "fetch-all LRU",
                                           "split DFA-evict", "pooled split LRU (FreeToken-like)")), flush=True)
                if not full:
                    continue
                # one-at-a-time sensitivity
                sens = b["sensitivity"] = {}
                for par, vals in SENS.items():
                    for v in vals:
                        c = consts(**{par: v})
                        so, tb = split_opts(tr, M, C, c, cache)
                        oo = dict(opts)
                        oo.update(so)
                        e = evaluate(oo, M, c, Mstar, Mstar_pool)
                        e["split_table"] = list(tb)
                        sens[f"{par}={v:g}"] = e
                # Monte Carlo over the expected ranges for the target machine
                rng = np.random.default_rng(a.seed)
                draws = []
                for _ in range(a.mc):
                    BD = rng.uniform(52e9, 62e9)
                    c = consts(B_D=BD, B_h=min(rng.uniform(40e9, 50e9), BD), B_P=rng.uniform(45e9, 52e9),
                               tau_e=rng.uniform(5e-6, 21e-6), O=rng.uniform(10e-6, 38e-6),
                               f=rng.uniform(0.33, 1.67) * M["f"], eta_g=rng.uniform(0.4, 0.6),
                               tau_x=rng.uniform(2e-6, 10e-6), theta=rng.uniform(0.0, 0.25))
                    so, tb = split_opts(tr, M, C, c, cache, MC_SPLITS)
                    oo = dict(opts)
                    oo.update(so)
                    e = evaluate(oo, M, c, Mstar, Mstar_pool, "optimistic" if rng.uniform() < 0.5 else "uniform")
                    draws.append({n: r["tok_s"] for n, r in e["rows"].items()} | {"SoL": e["sol_tok_s"]})
                b["mc"] = draws
                print(f"[{time.time() - t0:6.0f}s]   sensitivity + {a.mc} MC draws done ({len(cache)} split sims)",
                      flush=True)
            del tr
    json.dump(out, open(a.out, "w"), default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    print(f"wrote {a.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

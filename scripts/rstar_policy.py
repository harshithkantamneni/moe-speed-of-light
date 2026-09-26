"""Tune the r*-aware admission policy on OLMoE (dev model) and evaluate it on
held-out models (Qwen3-30B-A3B, gpt-oss-20b) against LRU, MIN-bypass and the
speed-of-light bound, using the validated decode model."""
import json, os, sys, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mosl.archs import shape
from mosl.cachesim import simulate, simulate_rstar
from mosl.perfmodel import Params, Workload, dynamic_time, speed_of_light_time, static_offload_time
from mosl.traces import Trace
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import MODELS, PLATFORMS, model_params, per_layer

P0, _ = model_params()
FR = [0.125, 0.25, 0.5]


def rstar(w, hw, p):
    eb = w.expert_bytes
    a = eb / (p.eta_g * hw.bw_gpu * 1e9); b = eb / (p.eta_c * hw.bw_cpu * 1e9) + p.tau_e_us * 1e-6
    pf = eb / (p.eta_p * hw.bw_pcie * 1e9)
    return pf / (b - a) if b > a else 1e9


def run_policy(tr, cap, hl, theta, lfu):
    m = np.zeros((tr.T, tr.L), np.int32); a = np.zeros_like(m)
    for i, R in enumerate(tr.R):
        m[:, i], a[:, i] = simulate_rstar(R, tr.E, cap, hl, theta, lfu)
    return m, a


def evaluate(name, grid=None, fixed=None):
    repo, tok, b_exp, b_dense = MODELS[name]
    tr = Trace(f"data/traces/{name}", tok); s = shape(repo); w = Workload(s, b_exp, b_dense, ctx=1024)
    res = {}
    for pname, hw in PLATFORMS.items():
        rs = rstar(w, hw, P0)
        for f in FR:
            cap = int(f * tr.E)
            lm, la = per_layer(tr, cap, "lru"); bm, ba = per_layer(tr, cap, "min", bypass=True)
            M = bm.sum() / (tr.T * tr.L)
            base = dict(lru=1 / dynamic_time(w, hw, P0, lm, la, "cpu")[0], bypass=1 / dynamic_time(w, hw, P0, bm, ba, "cpu")[0],
                        sol=1 / speed_of_light_time(w, hw, P0, M), static=1 / static_offload_time(w, hw, P0, tr.L - int(f * tr.L))[0])
            cands = grid if grid is not None else [fixed]
            best = None
            for hl, kappa, lfu in cands:
                m, a = run_policy(tr, cap, hl, kappa * rs, lfu)
                v = 1 / dynamic_time(w, hw, P0, m, a, "cpu")[0]
                if best is None or v > best[0]:
                    best = (v, (hl, kappa, lfu), 1 - m.sum() / (tr.T * tr.L * tr.k), a.sum(1).mean(), la.sum(1).mean())
            res[f"{pname}|{f}"] = dict(**base, rstar_policy=best[0], cfg=best[1], hit=best[2], adm_tok=best[3], lru_adm_tok=best[4], r_star=rs)
    return res


if __name__ == "__main__":
    grid = list(itertools.product([4.0, 16.0, 64.0], [0.0, 0.5, 1.0, 2.0], [True, False]))
    dev = evaluate("olmoe-1b-7b", grid=grid)
    # choose the single config with the best mean ratio to LRU across dev platforms/fractions
    scores = {}
    for cfg in grid:
        pass
    from collections import Counter
    # re-score every grid config on dev to pick one global setting
    repo, tok, b_exp, b_dense = MODELS["olmoe-1b-7b"]
    tr = Trace("data/traces/olmoe-1b-7b", tok); s = shape(repo); w = Workload(s, b_exp, b_dense, ctx=1024)
    agg = {}
    for cfg in grid:
        r = []
        for pname, hw in PLATFORMS.items():
            rs = rstar(w, hw, P0)
            for f in FR:
                cap = int(f * tr.E)
                m, a = run_policy(tr, cap, cfg[0], cfg[1] * rs, cfg[2])
                r.append(np.log((1 / dynamic_time(w, hw, P0, m, a, "cpu")[0]) / dev[f"{pname}|{f}"]["lru"]))
        agg[cfg] = float(np.mean(r))
    chosen = max(agg, key=agg.get)
    print("chosen on OLMoE:", chosen, "mean log-gain vs LRU %.3f" % agg[chosen])
    out = {"chosen": chosen, "dev_gain": agg[chosen]}
    for name in ("qwen3-30b-a3b", "gpt-oss-20b"):
        out[name] = evaluate(name, fixed=chosen)
        for k, d in out[name].items():
            print(name[:8], k[:9], k.split("|")[1], "static %.0f lru %.0f rstar %.0f bypass %.0f sol %.0f | gain vs LRU %.2fx  adm/tok %.1f vs %.1f" % (
                d["static"], d["lru"], d["rstar_policy"], d["bypass"], d["sol"], d["rstar_policy"] / d["lru"], d["adm_tok"], d["lru_adm_tok"]))
    json.dump(out, open("results/rstar_policy.json", "w"), indent=1, default=lambda x: list(x) if isinstance(x, tuple) else float(x))

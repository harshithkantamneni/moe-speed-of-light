"""scripts/foresight.py with the exact optimum: the same sweep (windows W = 1..256, budgets 12.5/25/50% of E per
layer, own sampled text of 9 models with the cache carried across conversations, cold start, plus the job 063
gpt-oss-120b trace), the same output schema (prereg/foresight/foresight_S_exact.json), with two changes:
  opt      the exact Belady MIN with bypass of mosl/cachesim.py (verified against exhaustive search), which serves a
           token's hits first and may evict an expert already served this step; scripts/foresight._pol at W = infinity
           never evicts a resident of the current token and reads more
  W = 1..  the windowed policies (Belady within the window, decayed frequency beyond it) with the same freedom, from
           scripts/policy_study._pol_steps(atomic=False); the W = infinity member of this family reproduces the exact
           optimum
The online policies (dfa, dfa-fetch) and seq-static are unchanged: they keep the atomic rule a real cache obeys.
Then the W50 law is refitted as in scripts/fig_foresight.py, with the bootstrap over models, leave-one-model-out and
no-interpolated-point intervals of scripts/policy_study.w50_intervals (prereg/foresight/w50_exact.json), and the 26
W50 values are compared with the original file's.

    python scripts/foresight_exact.py --results /home/claude/gpu-branch/results --out prereg/foresight
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl import cachesim, ecsim_fast  # noqa: E402
from mosl.traces import load_pack  # noqa: E402
from scripts.fig_foresight import curves, w_at  # noqa: E402
from scripts.foresight import WINDOWS, seq_static  # noqa: E402
from scripts.policy_study import _pol_steps, w50_intervals  # noqa: E402
from scripts.provenance import MODELS, find, per_seq  # noqa: E402

HALF_LIFE = 16.0


def analyse(routes, segs, E, k, kappa, budgets, log=None):
    """scripts.foresight.analyse with the exact optimum and relaxed windowed policies; same keys and semantics."""
    L, T, _ = routes.shape
    out = {}
    for C in budgets:
        tt = time.time()
        acc = {}

        def add(key, c, p):
            a = acc.setdefault(key, [0, 0])
            a[0] += int(c)
            a[1] += int(p)
        for l in range(L):
            R = np.ascontiguousarray(routes[l], dtype=np.int64)
            h, m, ad = ecsim_fast.simulate_layer(R, E, C, "dfa", half_life=HALF_LIFE, kappa=kappa)
            add("dfa", m.sum(), ad.sum())
            for kap in sorted({0.0, kappa}):
                c, cp = _pol_steps(R, E, C, -1, HALF_LIFE, kap, True)          # online: atomic, as in foresight.py
                add(f"dfa-fetch@k{kap:g}", c.sum(), cp.sum())
                for w in WINDOWS:
                    c, cp = _pol_steps(R, E, C, w, HALF_LIFE, kap, False)      # foresight: the optimum's freedom
                    add(f"W{w}@k{kap:g}", c.sum(), cp.sum())
            m, adm = cachesim.simulate(R, E, C, "min", bypass=True)            # exact optimum; kappa irrelevant
            for kap in sorted({0.0, kappa}):
                add(f"opt@k{kap:g}", m.sum() - adm.sum(), adm.sum())           # cpu = bypassed misses, copy = admitted
            add("seq-static", *seq_static(R, segs, E, C))
        order = ["dfa"] + [f"{key}@k{kap:g}" for key in ["dfa-fetch"] + [f"W{w}" for w in WINDOWS] + ["opt"]
                           for kap in sorted({0.0, kappa})] + ["seq-static"]            # foresight_S.json's key order
        row = {key: dict(cpu=acc[key][0] / T, copy=acc[key][1] / T, reads=sum(acc[key]) / T) for key in order}
        for key in ["dfa-fetch"] + [f"W{w}" for w in WINDOWS] + ["opt"]:
            cands = {kk: v for kk, v in row.items() if kk.startswith(key + "@k")}
            best = min(cands, key=lambda kk: cands[kk]["reads"])
            row[key] = dict(cands[best], kappa=float(best.split("@k")[1]))
        row["dfa"]["critical"] = row["dfa"]["cpu"]
        for key in row:
            row[key].setdefault("critical", row[key]["reads"])
        out[C] = row
        if log:
            log(f"   C={C:3d}: dfa-fetch {row['dfa-fetch']['reads']:.2f}  W1 {row['W1']['reads']:.2f}  W16 {row['W16']['reads']:.2f}  "
                f"W256 {row['W256']['reads']:.2f}  opt {row['opt']['reads']:.2f}  [{time.time() - tt:.0f}s]")
    return out, T


def jobs_from(results, la, arm, models):
    """the job list of scripts.foresight.main: the job 063 trace first, then each model's arm pack"""
    jobs = []
    if la and os.path.exists(la):
        from scripts.sim_prefetch import load
        d = load(la)
        routes = np.transpose(d["act"], (1, 0, 2))
        segs, s0 = [], 0
        for s in np.unique(d["seq"]):
            n = int((d["seq"] == s).sum())
            segs.append((s0, s0 + n))
            s0 += n
        jobs.append(("gpt-oss-120b (job 063, engine text)", routes, segs, d["E"], d["k"], 1.0))
    for key in models:
        ck, kappa, _ = MODELS[key]
        p = find(results, f"{ck}_{arm}.npz")
        if not p:
            print(f"{key}: no {arm} pack")
            continue
        pk = load_pack(p)
        win = per_seq(pk)
        idx = np.concatenate([np.arange(x, y) for _, x, y in win])
        segs, s0 = [], 0
        for _, x, y in win:
            segs.append((s0, s0 + (y - x)))
            s0 += y - x
        routes = pk["routes"][:, idx]
        jobs.append((key, routes, segs, pk["E"], routes.shape[2], kappa))
    return jobs


def compare(old_path, new_path):
    """per-point W50 and savings, old against new"""
    old = {(r["model"], r["C"]): r for r in curves(old_path) if "job 063" not in r["model"]}
    new = {(r["model"], r["C"]): r for r in curves(new_path) if "job 063" not in r["model"]}
    rows = []
    for key in old:
        o, n = old[key], new[key]
        rows.append(dict(model=key[0], C=key[1], C_over_k=n["ck"],
                         w50_old=w_at(o["g"], 0.5), w50_new=w_at(n["g"], 0.5),
                         saving_old=1 - o["ratio"], saving_new=1 - n["ratio"],
                         interpolated_old=bool(o["g"][0] >= 0.5), interpolated_new=bool(n["g"][0] >= 0.5)))
    for r in rows:
        r["w50_ratio"] = r["w50_new"] / r["w50_old"] if np.isfinite(r["w50_old"]) and np.isfinite(r["w50_new"]) else None
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results"))
    ap.add_argument("--la", default=os.path.join(os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results"), "063_lookahead@vast/la_gpt-oss-120b.bin"))
    ap.add_argument("--arm", default="S")
    ap.add_argument("--out", default="prereg/foresight")
    ap.add_argument("--models", default=",".join(MODELS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t_all = time.time()
    allres = {}
    path = os.path.join(a.out, f"foresight_{a.arm}_exact.json")
    for name, routes, segs, E, k, kappa in jobs_from(a.results, a.la, a.arm, a.models.split(",")):
        budgets = sorted({max(k, E // 8), E // 4, E // 2})
        print(f"== {name}: E={E} k={k} L={routes.shape[0]} T={routes.shape[1]} conversations={len(segs)} budgets={budgets}", flush=True)
        res, T = analyse(routes, segs, int(E), int(k), float(kappa), budgets, log=lambda s: print(s, flush=True))
        allres[name] = dict(E=int(E), k=int(k), L=int(routes.shape[0]), T=int(T), convs=len(segs), budgets=res)
        json.dump(allres, open(path, "w"), indent=1)
    fit = w50_intervals(path)
    old_path = os.path.join(a.out, f"foresight_{a.arm}.json")
    if os.path.exists(old_path):
        fit["versus_original"] = dict(source=os.path.relpath(old_path), points=compare(old_path, path))
        rat = [r["w50_ratio"] for r in fit["versus_original"]["points"] if r["w50_ratio"]]
        sv = [r["saving_new"] for r in fit["versus_original"]["points"]]
        so = [r["saving_old"] for r in fit["versus_original"]["points"]]
        fit["versus_original"]["summary"] = dict(
            w50_new_over_old=dict(min=float(min(rat)), median=float(np.median(rat)), max=float(max(rat)), n=len(rat)),
            saving_new=dict(min=float(min(sv)), median=float(np.median(sv)), max=float(max(sv))),
            saving_old=dict(min=float(min(so)), median=float(np.median(so)), max=float(max(so))),
            interpolated_old=int(sum(r["interpolated_old"] for r in fit["versus_original"]["points"])),
            interpolated_new=int(sum(r["interpolated_new"] for r in fit["versus_original"]["points"])))
    fit["runtime_s"] = time.time() - t_all
    json.dump(fit, open(os.path.join(a.out, "w50_exact.json"), "w"), indent=1)
    f = fit["full"]
    print(f"W50 = {f['prefactor']:.2f} (C/k)^{f['exponent']:.2f}, r = {f['r']:.3f}, n = {f['n']}, interpolated {fit['n_interpolated']}; "
          f"bootstrap over models 95% {[round(x, 2) for x in fit['bootstrap_models']['exponent_ci95']]}; "
          f"LOO {[round(x, 2) for x in fit['leave_one_model_out']['exponent_range']]}; "
          f"no interpolated {fit['no_interpolated']['exponent']:.2f} (n={fit['no_interpolated']['n']})")
    if "versus_original" in fit:
        print("versus original:", json.dumps(fit["versus_original"]["summary"]))
    print(f"done in {time.time() - t_all:.0f}s -> {path}, {os.path.join(a.out, 'w50_exact.json')}")


if __name__ == "__main__":
    main()

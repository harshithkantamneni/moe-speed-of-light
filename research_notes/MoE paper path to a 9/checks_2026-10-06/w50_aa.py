import json, os, sys, time, copy
import numpy as np
sys.path.insert(0, "/home/claude/moe-speed-of-light")
from scripts.foresight import WINDOWS
from scripts.policy_study import _pol_steps, w50_intervals
from scripts.foresight_exact import jobs_from
S = os.path.dirname(os.path.abspath(__file__))
base = json.load(open("/home/claude/moe-speed-of-light/prereg/foresight/foresight_S_exact.json"))
KAP = -1e18
aa = {}
t0 = time.time()
for name, routes, segs, E, k, kappa in jobs_from("/home/claude/gpu-branch/results", None, "S", list(__import__("scripts.provenance", fromlist=["MODELS"]).MODELS)):
    L, T, _ = routes.shape
    budgets = sorted({max(k, E // 8), E // 4, E // 2})
    aa[name] = {}
    for C in budgets:
        acc = {}
        for l in range(L):
            R = np.ascontiguousarray(routes[l], dtype=np.int64)
            c, cp = _pol_steps(R, int(E), C, -1, 16.0, KAP, True)
            acc["dfa-fetch"] = acc.get("dfa-fetch", 0) + int(c.sum()) + int(cp.sum())
            for w in WINDOWS:
                c, cp = _pol_steps(R, int(E), C, w, 16.0, KAP, False)
                acc[f"W{w}"] = acc.get(f"W{w}", 0) + int(c.sum()) + int(cp.sum())
        aa[name][str(C)] = {kk: v / T for kk, v in acc.items()}
        print(name, C, {kk: round(v, 2) for kk, v in aa[name][str(C)].items()}, f"[{time.time()-t0:.0f}s]", flush=True)
json.dump(aa, open(os.path.join(S, "aa_reads.json"), "w"), indent=1)
# variant A: kappa set extended with -inf (min over {0, model kappa, -inf}) for every key
# variant B: always-admit everywhere (online and beyond-window rule)
for variant in ("A_min", "B_aa"):
    d = copy.deepcopy(base)
    d = {m: v for m, v in d.items() if "job 063" not in m}
    for m, v in d.items():
        for C, row in v["budgets"].items():
            for key in ["dfa-fetch"] + [f"W{w}" for w in WINDOWS]:
                new = aa[m][C][key]
                if variant == "B_aa" or new < row[key]["reads"]:
                    row[key] = dict(reads=new, cpu=None, copy=None, kappa=KAP)
    p = os.path.join(S, f"foresight_S_exact_{variant}.json")
    json.dump(d, open(p, "w"), indent=1)
    fit = w50_intervals(p)
    print(variant, "full", fit["full"], "boot", fit["bootstrap_models"]["exponent_ci95"], "noint", {k_: fit["no_interpolated"][k_] for k_ in ("exponent", "prefactor", "n")}, flush=True)
    json.dump(fit, open(os.path.join(S, f"w50_{variant}.json"), "w"), indent=1, default=float)
orig = json.load(open("/home/claude/moe-speed-of-light/prereg/foresight/w50_exact.json"))
print("orig full", orig["full"], "boot", orig["bootstrap_models"]["exponent_ci95"])

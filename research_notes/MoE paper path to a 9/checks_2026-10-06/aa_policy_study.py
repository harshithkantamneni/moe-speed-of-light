"""Always-admit (decayed-frequency victim, every miss copied) in the policy study's own frame: S-arm traces, the
study's budgets, reads per token over the last 90% of each trace. Compared with the best online policy the study
reports (lru, lfu, df0, dfk, arc, s3fifo) and with the exact optimum."""
import json, os, sys
import numpy as np
sys.path.insert(0, "/home/claude/moe-speed-of-light")
from scripts.policy_study import _pol_steps, budgets_for
from scripts.provenance import MODELS, find, per_seq
from mosl.traces import load_pack
ps = json.load(open("/home/claude/moe-speed-of-light/prereg/policy_study.json"))
out = {}
for key in MODELS:
    ck, kappa, _ = MODELS[key]
    pk = load_pack(find("/home/claude/gpu-branch/results", f"{ck}_S.npz")); win = per_seq(pk)
    idx = np.concatenate([np.arange(x, y) for _, x, y in win]); routes = pk["routes"][:, idx]
    E, L, T = int(pk["E"]), routes.shape[0], routes.shape[1]
    ref = ps["models"][key]; assert ref["T"] == T, (key, ref["T"], T)
    Te = ref["T_scored"]; t0 = T - Te
    for C in ref["budgets"]:
        s = 0
        for l in range(L):
            c, cp = _pol_steps(np.ascontiguousarray(routes[l], dtype=np.int64), E, C, -1, 16.0, -1e18, True)
            s += int((c + cp)[t0:].sum())
        aa = s / Te
        cell = ref["cells"][str(C)]["policies"]
        onl = {n: cell[n]["reads"] for n in ("lru", "lfu", "df0", "dfk", "arc", "s3fifo")}
        best = min(onl, key=onl.get); opt = cell["opt"]["reads"]
        out[f"{key}/{C}"] = dict(C_over_k=C / routes.shape[2], aa=aa, best=best, best_reads=onl[best], df0=onl["df0"], opt=opt,
                                 aa_rel_opt=aa / opt, best_rel_opt=onl[best] / opt, aa_vs_best_pct=100 * (onl[best] - aa) / onl[best])
        print(f"{key:18s} C={C:3d} best online {best:6s} {onl[best]:7.2f} ({onl[best]/opt:.3f}x opt) | always-admit {aa:7.2f} ({aa/opt:.3f}x opt) | AA vs best {100*(onl[best]-aa)/onl[best]:+5.1f}%", flush=True)
r_b = [v["best_rel_opt"] for v in out.values()]; r_a = [min(v["aa_rel_opt"], v["best_rel_opt"]) for v in out.values()]
print("best-online/opt range %.3f-%.3f ; with always-admit added %.3f-%.3f ; AA beats best in %d of %d cells" % (min(r_b), max(r_b), min(r_a), max(r_a), sum(v["aa_vs_best_pct"] > 0 for v in out.values()), len(out)))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "aa_policy_study.json"), "w"), indent=1)

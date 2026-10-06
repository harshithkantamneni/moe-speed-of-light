import json, os, sys
import numpy as np
sys.path.insert(0, "/home/claude/moe-speed-of-light")
from scripts.foresight import _pol, INF
from scripts.provenance import MODELS, find, per_seq
from mosl.traces import load_pack
res_dir = "/home/claude/gpu-branch/results"
ref = json.load(open("/home/claude/moe-speed-of-light/prereg/foresight/foresight_S.json"))
out = {}
for key in MODELS:
    ck, kappa, _ = MODELS[key]
    p = find(res_dir, f"{ck}_S.npz")
    if not p:
        print(key, "no pack"); continue
    pk = load_pack(p); win = per_seq(pk)
    idx = np.concatenate([np.arange(x, y) for _, x, y in win])
    routes = pk["routes"][:, idx]; E = int(pk["E"]); k = routes.shape[2]; L, T, _ = routes.shape
    budgets = sorted({max(k, E // 8), E // 4, E // 2})
    for C in budgets:
        c = cp = 0
        for l in range(L):
            R = np.ascontiguousarray(routes[l], dtype=np.int64)
            a, b = _pol(R, E, C, -1, 16.0, -1e18)
            c += a; cp += b
        aa = (c + cp) / T
        row = ref[key]["budgets"][str(C)]
        onl = row["dfa-fetch"]["reads"]; opt = row["opt"]["reads"]
        w = {W: row[f"W{W}"]["reads"] for W in (1, 2, 4, 8, 16)}
        out[f"{key}/{C}"] = dict(E=E, k=k, C=C, Ck=C / k, online=onl, online_kappa=row["dfa-fetch"]["kappa"], always_admit=aa, opt=opt,
                                 aa_fewer_pct=100 * (onl - aa) / onl, aa_gap_closed=(onl - aa) / (onl - opt),
                                 online_excess=onl / opt - 1, aa_excess=aa / opt - 1, W=w)
        print(f"{key:18s} C={C:3d} C/k={C/k:5.2f} online {onl:7.2f} (k{row['dfa-fetch']['kappa']:g}) always-admit {aa:7.2f} opt {opt:7.2f} | AA fewer {100*(onl-aa)/onl:5.1f}% closes {100*(onl-aa)/(onl-opt):5.1f}% of gap | excess online {100*(onl/opt-1):5.1f}% AA {100*(aa/opt-1):5.1f}%", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "always_admit_S.json"), "w"), indent=1)

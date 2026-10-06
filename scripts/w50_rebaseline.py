"""Re-baselining the trace study with always-admit, and the horizon restated in distinct experts.

1. Always-admit (every miss copied into the slot of the lowest decayed-score resident not requested now; kappa = -inf)
   is added to the online policies: (a) to the foresight study's baseline and to the beyond-window rule of every
   window W (the kappa set the study minimises over becomes {0, the model's kappa, -inf}), giving
   prereg/foresight/foresight_S_exact_aa.json and the refitted W50 law prereg/foresight/w50_aa.json; (b) to the policy
   study (LRU, LFU, two decayed-frequency variants, ARC, S3-FIFO), on its own frame (last 90% of each trace),
   prereg/policy_aa.json.
2. D(W): the mean number of distinct experts per layer in a window of W steps (including the current one), per model
   on the S traces; D(W50) / C at every point of the refitted law; and leave-one-model-out prediction of a held-out
   model's W50 by three rules: W50 proportional to C/k (one constant), the power law in C/k (two), and D(W50) = c C
   (one constant plus the model's own D(W) curve). prereg/foresight/w50_distinct.json.
Macros: paper/wsg_rebase.tex.

    python scripts/w50_rebaseline.py
"""
import copy
import json
import os
import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl.traces import load_pack  # noqa: E402
from scripts.foresight import WINDOWS  # noqa: E402
from scripts.foresight_exact import jobs_from  # noqa: E402
from scripts.policy_study import _pol_steps, w50_intervals  # noqa: E402
from scripts.provenance import MODELS, find, per_seq  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
RES = "/home/claude/gpu-branch/results"
KAP = -1e18
MAC = {}


def M(k, v):
    assert not any(c.isdigit() for c in k), k
    MAC[k] = v


def foresight_aa():
    base = json.load(open(P("prereg", "foresight", "foresight_S_exact.json")))
    aa = {}
    t0 = time.time()
    for name, routes, segs, E, k, kappa in jobs_from(RES, None, "S", list(MODELS)):
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
            print(name, C, f"always-admit {aa[name][str(C)]['dfa-fetch']:.2f}", f"[{time.time() - t0:.0f}s]", flush=True)
    d = {m: v for m, v in copy.deepcopy(base).items() if "job 063" not in m}
    changed = 0
    for m, v in d.items():
        for C, row in v["budgets"].items():
            for key in ["dfa-fetch"] + [f"W{w}" for w in WINDOWS]:
                new = aa[m][C][key]
                if new < row[key]["reads"]:
                    row[key] = dict(reads=new, cpu=None, copy=None, kappa=KAP)
                    changed += 1
    out = P("prereg", "foresight", "foresight_S_exact_aa.json")
    json.dump(d, open(out, "w"), indent=1)
    fit = w50_intervals(out)
    json.dump(fit, open(P("prereg", "foresight", "w50_aa.json"), "w"), indent=1, default=float)
    orig = json.load(open(P("prereg", "foresight", "w50_exact.json")))
    # baseline change per point: always-admit's reads against the study's online single-read baseline
    sav = []
    for m, v in base.items():
        if "job 063" in m:
            continue
        for C, row in v["budgets"].items():
            sav.append(100 * (1 - aa[m][C]["dfa-fetch"] / row["dfa-fetch"]["reads"]))
    M("rbAaSaveMax", f"{max(sav):.1f}")
    M("rbAaBeats", str(sum(s > 0 for s in sav)))
    M("rbAaCells", str(len(sav)))
    M("rbPre", f"{fit['full']['prefactor']:.2f}")
    M("rbExp", f"{fit['full']['exponent']:.2f}")
    M("rbR", f"{fit['full']['r']:.3f}")
    lo, hi = fit["bootstrap_models"]["exponent_ci95"]
    M("rbExpLo", f"{lo:.2f}"); M("rbExpHi", f"{hi:.2f}")
    lomo = [f["exponent"] for f in fit["leave_one_model_out"]["fits"].values()]
    M("rbExpLomoMin", f"{min(lomo):.2f}"); M("rbExpLomoMax", f"{max(lomo):.2f}")
    M("rbExpNoInterp", f"{fit['no_interpolated']['exponent']:.2f}")
    M("rbExpNoInterpOrig", f"{orig['no_interpolated']['exponent']:.2f}")
    po = {(p["model"], p["C"]): p["w50"] for p in orig["points"]}
    rise = [p["w50"] / po[(p["model"], p["C"])] for p in fit["points"] if (p["model"], p["C"]) in po and np.isfinite(p["w50"])]
    M("rbWRiseMed", f"{100 * (np.median(rise) - 1):.0f}"); M("rbWRiseMax", f"{100 * (max(rise) - 1):.0f}")
    return fit


def policy_aa():
    ps = json.load(open(P("prereg", "policy_study.json")))
    out = {}
    for key in MODELS:
        ck, kappa, _ = MODELS[key]
        pk = load_pack(find(RES, f"{ck}_S.npz"))
        win = per_seq(pk)
        idx = np.concatenate([np.arange(x, y) for _, x, y in win])
        routes = pk["routes"][:, idx]
        E, L, T = int(pk["E"]), routes.shape[0], routes.shape[1]
        if key not in ps["models"]:
            continue
        ref = ps["models"][key]
        assert ref["T"] == T, (key, ref["T"], T)
        Te = ref["T_scored"]
        t0 = T - Te
        for C in ref["budgets"]:
            s = 0
            for l in range(L):
                c, cp = _pol_steps(np.ascontiguousarray(routes[l], dtype=np.int64), E, C, -1, 16.0, KAP, True)
                s += int((c + cp)[t0:].sum())
            a = s / Te
            cell = ref["cells"][str(C)]["policies"]
            onl = {n: cell[n]["reads"] for n in ("lru", "lfu", "df0", "dfk", "arc", "s3fifo")}
            best = min(onl, key=onl.get)
            opt = cell["opt"]["reads"]
            out[f"{key}/{C}"] = dict(C_over_k=C / routes.shape[2], aa=a, best=best, best_reads=onl[best], opt=opt,
                                     aa_rel_opt=a / opt, best_rel_opt=onl[best] / opt, aa_vs_best_pct=100 * (onl[best] - a) / onl[best])
            print(f"{key:18s} C={C:3d} best online {best:6s} {onl[best] / opt:.3f}x opt | always-admit {a / opt:.3f}x", flush=True)
    json.dump(out, open(P("prereg", "policy_aa.json"), "w"), indent=1)
    r_new = [min(v["aa_rel_opt"], v["best_rel_opt"]) for v in out.values()]
    M("rbPolBestMin", f"{100 * (min(r_new) - 1):.0f}"); M("rbPolBestMax", f"{100 * (max(r_new) - 1):.0f}")
    M("rbPolAaWins", str(sum(v["aa_vs_best_pct"] > 0.5 for v in out.values())))
    M("rbPolAaTies", str(sum(abs(v["aa_vs_best_pct"]) <= 0.5 for v in out.values())))
    M("rbPolAaLossMax", f"{max(0.0, -min(v['aa_vs_best_pct'] for v in out.values())):.1f}")
    M("rbPolAaGainMax", f"{max(v['aa_vs_best_pct'] for v in out.values()):.1f}")
    M("rbPolCells", str(len(out)))


WS = np.array([1, 2, 4, 8, 16, 32, 64])


@njit(cache=True)
def _distinct(R, E, Wmax):
    T, k = R.shape
    out = np.zeros(Wmax + 1)
    seen = np.zeros(E, np.int64)
    stamp = 0
    n = 0
    for t in range(0, T - Wmax, 3):
        stamp += 1
        d = 0
        for w in range(1, Wmax + 1):
            for j in range(k):
                e = R[t + w - 1, j]
                if seen[e] != stamp:
                    seen[e] = stamp
                    d += 1
            out[w] += d
        n += 1
    return out / max(n, 1)


def distinct_rule(fit):
    pts = {(p["model"], p["C"]): p["w50"] for p in fit["points"]}
    res = {}
    for name, routes, segs, E, k, kappa in jobs_from(RES, None, "S", list(MODELS)):
        L = routes.shape[0]
        D = np.zeros(65)
        for l in range(L):
            D += _distinct(np.ascontiguousarray(routes[l], dtype=np.int64), int(E), 64)
        D /= L
        beta = float(np.polyfit(np.log(WS[:5]), np.log(D[WS[:5]]), 1)[0])
        res[name] = dict(E=int(E), k=int(k), beta=beta, D=[float(D[w]) for w in WS], pts=[])
        for (m, C), w in pts.items():
            if m != name or not np.isfinite(w):
                continue
            d = k * w if w < 1 else float(np.exp(np.interp(np.log(w), np.log(np.arange(1, 65)), np.log(D[1:65]))))
            res[name]["pts"].append(dict(C=C, Ck=C / k, W50=w, D_at_W50=d, D_over_C=d / C))
    rows = [(m, p) for m, v in res.items() for p in v["pts"]]

    def inv(v, target):
        if target <= v["k"]:
            return target / v["k"]
        return float(np.exp(np.interp(np.log(target), np.log(v["D"]), np.log(WS))))
    errs = {"proportional": [], "power": [], "distinct": []}
    for hold in res:
        tr = [p for m, p in rows if m != hold]
        te = [p for m, p in rows if m == hold]
        if not te:
            continue
        b, a = np.polyfit(np.log([p["Ck"] for p in tr]), np.log([p["W50"] for p in tr]), 1)
        a1 = np.median([p["W50"] / p["Ck"] for p in tr])
        c = np.median([p["D_over_C"] for p in tr])
        for p in te:
            errs["power"].append(abs(np.log(np.exp(a) * p["Ck"] ** b / p["W50"])))
            errs["proportional"].append(abs(np.log(a1 * p["Ck"] / p["W50"])))
            errs["distinct"].append(abs(np.log(inv(res[hold], c * p["C"]) / p["W50"])))
    summ = {k: dict(median=float(np.exp(np.median(v))), p90=float(np.exp(np.percentile(v, 90))), max=float(np.exp(np.max(v))), n=len(v)) for k, v in errs.items()}
    dc = [p["D_over_C"] for _, p in rows]
    w = [p["W50"] for _, p in rows]
    out = dict(models=res, lomo=summ, D_over_C=dict(median=float(np.median(dc)), q25=float(np.percentile(dc, 25)), q75=float(np.percentile(dc, 75)),
                                                    cv=float(np.std(dc) / np.mean(dc))), W50_cv=float(np.std(w) / np.mean(w)),
               W50_range=[float(min(w)), float(max(w))])
    json.dump(out, open(P("prereg", "foresight", "w50_distinct.json"), "w"), indent=1)
    M("dcMed", f"{out['D_over_C']['median']:.2f}"); M("dcLo", f"{out['D_over_C']['q25']:.2f}"); M("dcHi", f"{out['D_over_C']['q75']:.2f}")
    M("dcCv", f"{out['D_over_C']['cv']:.2f}")
    M("dcWMin", f"{min(w):.1f}"); M("dcWMax", f"{max(w):.0f}")
    betas = [v["beta"] for v in res.values()]
    M("dcBetaMin", f"{min(betas):.2f}"); M("dcBetaMax", f"{max(betas):.2f}")
    for k, nm in (("proportional", "Prop"), ("power", "Pow"), ("distinct", "Dist")):
        M(f"dc{nm}Med", f"{summ[k]['median']:.2f}"); M(f"dc{nm}Pninety", f"{summ[k]['p90']:.2f}"); M(f"dc{nm}Max", f"{summ[k]['max']:.2f}")
    with open(P("paper", "tab_lomo.tex"), "w") as f:
        f.write("% generated by scripts/w50_rebaseline.py\n")
        f.write(r"""\begin{table}[t]\centering\footnotesize
\caption{Predicting a held-out model's horizon $W_{50}$ (fitted on the other eight models, tested on the ninth, over all
26 budget points): error as a factor, $\max(\hat W/W, W/\hat W)$.}\label{tab:lomo}
\begin{tabular}{@{}lrrrr@{}}\toprule
Rule & Constants & Median & 90th pct. & Max \\\midrule
""")
        for k, nm, npar in (("proportional", r"$W_{50}\propto C/k$", "1"), ("power", r"$W_{50}=a\,(C/k)^b$", "2"),
                            ("distinct", r"$D(W_{50})=c\,C$", "1 + the model's $D(W)$")):
            v = summ[k]
            f.write(f"{nm} & {npar} & {v['median']:.2f} & {v['p90']:.2f} & {v['max']:.2f} \\\\\n")
        f.write(r"\bottomrule\end{tabular}\end{table}" + "\n")
    print("D(W50)/C", out["D_over_C"], "LOMO", summ, flush=True)


def main():
    fit = foresight_aa()
    policy_aa()
    distinct_rule(fit)
    with open(P("paper", "wsg_rebase.tex"), "w") as f:
        f.write("% generated by scripts/w50_rebaseline.py\n")
        for k in sorted(MAC):
            f.write(f"\\newcommand{{\\{k}}}{{{MAC[k]}}}\n")
    print({k: MAC[k] for k in sorted(MAC)})


if __name__ == "__main__":
    main()

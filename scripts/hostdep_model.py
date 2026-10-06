"""Does the calibrated time model (Eq. law) predict which way of spending foresight pays on each machine?

For every host and cell of the factorial (jobs 095-097 and the job 099 panel), each state's time per token is predicted
from that state's own counters and that host's probe: the CPU reads X_c = (misses - in-step fetches) S, the link reads
in the step X_p = fetches S, and T = G + max(X_c / B_c, X_p / B_p, (X_c + X_p) / B_cp), with G calibrated once per
host-cell on the deployed state (whose background copies, like every other state's background copies and prefetches,
are taken as hidden). Compared with the measured speed ratio to the deployed state. Writes prereg/hostdep_model.json
and paper/wsg_hostdepmodel.tex.

    python scripts/hostdep_model.py
"""
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(os.environ.get("MOSL_GPU_BRANCH", "/home/claude/gpu-branch"), "jobs", "ec2"))
from fetch_table import bandwidths  # noqa: E402

P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
RES = "/home/claude/gpu-branch/results"
S = {"g": 13253760, "q": 9437184}
CELLS = {"g14": ("g", 14, "gpt-oss 11%"), "g32": ("g", 32, "gpt-oss 25%"), "q16": ("q", 16, "Qwen3 12.5%"), "q32": ("q", 32, "Qwen3 25%")}
STATES = ("foa", "aa", "fetch", "w1", "w4", "w16", "w8r5", "allr5", "bypass", "both3p")


def host_rates(d):
    txt = open(f"{d}/concur.txt").read()
    cores = int(re.search(r"usable physical cores (\d+)", open(f"{d}/cores.txt").read()).group(1))
    H = cores - 2 if cores > 4 else 2
    return bandwidths(txt, H)


def ms_of(d, tag, C):
    out = {}
    p = f"{d}/ec_{tag}_C{C}.jsonl"
    if not os.path.exists(p):
        return out
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            lab = [x for x in r["config"].split(":") if x.startswith("stats=")][0]
            st = os.path.basename(lab[6:]).replace(".json", "").split(f"_C{C}_", 1)[1]
            out.setdefault(st, []).append(r["decode_ms"] / r["n_decode"])
    return {k: float(np.mean(v)) for k, v in out.items()}


def main():
    # every machine once: 097a re-ran O4's machine (096a), so it is left out here
    dirs = [d for d in sorted(glob.glob(f"{RES}/09[5-9]*@vast")) if os.path.exists(f"{d}/concur.txt") and "freetoken" not in d
            and "097a" not in d]
    rows = []
    for d in dirs:
        bc, bp, bb = host_rates(d)
        for key, (tag, C, lab) in CELLS.items():
            ms = ms_of(d, tag, C)
            if "base" not in ms:
                continue

            def X(st):
                f = f"{d}/st_{tag}_C{C}_{st}.json"
                s = json.load(open(f))
                n = s["steps"]
                fe = s["fetches"] / n
                bg = (s["admits"] + s.get("prefetches", 0)) / n
                return (s["misses"] / n - fe) * S[tag], fe * S[tag], bg * S[tag]

            def tmodel(xc, xp, xg, G, bgmode):
                if bgmode:   # background copies share the link and host memory over the step
                    return G + max(xc / (bc * 1e9), (xp + xg) / (bp * 1e9), (xc + xp + xg) / (bb * 1e9))
                return G + max(xc / (bc * 1e9), xp / (bp * 1e9), (xc + xp) / (bb * 1e9))
            xc, xp, xg = X("base")
            Gs = {m: ms["base"] * 1e-3 - tmodel(xc, xp, xg, 0.0, m) for m in (False, True)}
            for st in STATES:
                if st not in ms or not os.path.exists(f"{d}/st_{tag}_C{C}_{st}.json"):
                    continue
                xc, xp, xg = X(st)
                pred = tmodel(xc, xp, xg, Gs[False], False)
                pred_bg = tmodel(xc, xp, xg, Gs[True], True)
                meas = ms["base"] / ms[st]
                rows.append(dict(host=os.path.basename(d), cell=lab, state=st, link_over_cpu=bp / bc, G_ms=1e3 * Gs[False],
                                 B=[bc, bp, bb], X=[xc, xp, xg], X_base=list(X("base")), t_base_ms=ms["base"], t_ms=ms[st],
                                 measured_ratio=meas, predicted_ratio=ms["base"] * 1e-3 / pred,
                                 err=(ms["base"] * 1e-3 / pred) / meas - 1,
                                 predicted_ratio_bg=ms["base"] * 1e-3 / pred_bg, err_bg=(ms["base"] * 1e-3 / pred_bg) / meas - 1))
    # frozen G, out of sample: for each host, G of each cell is the median of the other hosts' fits at that cell; every
    # state's time (the deployed state's included) is then predicted from the probe and the counters alone
    Gcell = {}
    for r in rows:
        Gcell.setdefault(r["cell"], {})[r["host"]] = r["G_ms"] * 1e-3
    one = lambda xc, xp, B, G: G + (xc + xp) / (max(B) * 1e9)   # one path: every host byte at the best probed rate
    for r in rows:
        bc, bp, bb = r["B"]
        G = float(np.median([g for h, g in Gcell[r["cell"]].items() if h != r["host"]]))
        xc, xp, _ = r["X"]
        xcb, xpb, _ = r["X_base"]
        tb = G + max(xcb / (bc * 1e9), xpb / (bp * 1e9), (xcb + xpb) / (bb * 1e9))
        tx = G + max(xc / (bc * 1e9), xp / (bp * 1e9), (xc + xp) / (bb * 1e9))
        r["G_frozen_ms"] = 1e3 * G
        r["err_frozen"] = tx / (r["t_ms"] * 1e-3) - 1
        r["err_frozen_base"] = tb / (r["t_base_ms"] * 1e-3) - 1
        r["predicted_ratio_frozen"] = tb / tx
        # the one-path model, G calibrated on the deployed state like the two-path one
        G1 = r["t_base_ms"] * 1e-3 - (xcb + xpb) / (max(r["B"]) * 1e9)
        r["predicted_ratio_onepath"] = (r["t_base_ms"] * 1e-3) / one(xc, xp, r["B"], G1)
    json.dump(rows, open(P("prereg", "hostdep_model.json"), "w"), indent=1)
    M = {}
    for grp, sts in (("Step", ("foa", "aa", "fetch", "w1", "w4", "w16", "w8r5", "allr5")), ("Bg", ("bypass", "both3p"))):
        for ek, en in (("err", ""), ("err_bg", "Shared")):
            e = [abs(r[ek]) for r in rows if r["state"] in sts]
            if e:
                M[f"hmErr{grp}{en}Med"] = f"{100 * np.median(e):.1f}"
                M[f"hmErr{grp}{en}Pninety"] = f"{100 * np.percentile(e, 90):.0f}"
                M[f"hmErr{grp}N"] = str(len(e))
    # does the model get the sign of fetch / base right on every host-cell?
    f = [r for r in rows if r["state"] == "fetch"]
    if f:
        M["hmFetchSignRight"] = str(sum((r["measured_ratio"] > 1) == (r["predicted_ratio"] > 1) for r in f))
        M["hmFetchN"] = str(len(f))
        M["hmFetchSignFrozen"] = str(sum((r["measured_ratio"] > 1) == (r["predicted_ratio_frozen"] > 1) for r in f))
        M["hmFetchSignOnePath"] = str(sum((r["measured_ratio"] > 1) == (r["predicted_ratio_onepath"] > 1) for r in f))
        M["hmFetchLosses"] = str(sum(r["measured_ratio"] < 1 for r in f))
        M["hmFetchLossesOnePath"] = str(sum(r["measured_ratio"] < 1 and r["predicted_ratio_onepath"] < 1 for r in f))
    # frozen G (leave one host out): error of every in-step state's time, the deployed state's included
    fe = [abs(r["err_frozen"]) for r in rows if r["state"] in ("foa", "aa", "fetch", "w1", "w4", "w16", "w8r5", "allr5")]
    fb = {(r["host"], r["cell"]): abs(r["err_frozen_base"]) for r in rows}
    if fe:
        fe_all = fe + list(fb.values())   # the deployed state's own error once per host-budget, with the in-step states
        M["hmFrozenStepMed"] = f"{100 * np.median(fe_all):.1f}"
        M["hmFrozenStepPninety"] = f"{100 * np.percentile(fe_all, 90):.0f}"
        M["hmFrozenStepN"] = str(len(fe_all))
        M["hmFrozenBaseMed"] = f"{100 * np.median(list(fb.values())):.1f}"
        M["hmFrozenBaseMax"] = f"{100 * max(fb.values()):.0f}"
        gs = sorted({(r["host"], r["cell"]): r["G_ms"] for r in rows}.values())
        M["hmGMin"] = f"{min(gs):.1f}"; M["hmGMax"] = f"{max(gs):.1f}"
    fg = [r["err_frozen"] for r in rows if r["state"] in ("bypass", "both3p")]
    if fg:
        M["hmFrozenBgMed"] = f"{100 * np.median(np.abs(fg)):.0f}"
        M["hmFrozenBgFaster"] = str(sum(e < 0 for e in fg)); M["hmFrozenBgN"] = str(len(fg))
    M["hmHostCells"] = str(len({(r["host"], r["cell"]) for r in rows}))
    M["hmHostsN"] = str(len({r["host"] for r in rows}))
    # the share of MIN's time gain over admit-every-miss that each exact window recovers: predicted against measured
    by = {}
    for r in rows:
        by.setdefault((r["host"], r["cell"]), {})[r["state"]] = r
    werr, aerr = [], []
    for (h, c), s in by.items():
        if not all(k in s for k in ("aa", "fetch")):
            continue
        for x in ("w1", "w4", "w16", "allr5"):
            if x not in s:
                continue
            sh = {}
            for kind in ("measured_ratio", "predicted_ratio"):
                t = {k: 1 / s[k][kind] for k in ("aa", "fetch", x)}
                sh[kind] = (t["aa"] - t[x]) / (t["aa"] - t["fetch"])
            (aerr if x == "allr5" else werr).append(sh["predicted_ratio"] - sh["measured_ratio"])
    if werr:
        M["hmWinShareErrMed"] = f"{np.median(np.abs(werr)):.2f}"
        M["hmWinShareErrMax"] = f"{np.max(np.abs(werr)):.2f}"
        M["hmWinShareN"] = str(len(werr))
    if aerr:
        M["hmAllHalfOverMax"] = f"{max(aerr):.2f}"
    with open(P("paper", "wsg_hostdepmodel.tex"), "w") as fo:
        fo.write("% generated by scripts/hostdep_model.py\n")
        for k in sorted(M):
            fo.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    for r in rows:
        if r["state"] in ("fetch", "aa", "w4", "bypass", "both3p"):
            print(f"{r['host'][:26]:26s} {r['cell']:12s} {r['state']:7s} link/cpu {r['link_over_cpu']:.2f} measured {r['measured_ratio']:.3f} predicted {r['predicted_ratio']:.3f} err {100 * r['err']:+.1f}%")
    print(M)


if __name__ == "__main__":
    main()

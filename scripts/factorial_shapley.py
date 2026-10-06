"""Order-free accounting of the gap between the deployed cache and the speed limit, from the engine's own states.

Two factors are crossed in the engine at every host-bound cell (jobs 096 on hosts O4 and O5, job 099 on its panel):
  admission set   the online policy's (decayed frequency, kappa 1)  vs  MIN with bypass's (oracle)
  reads           two (the CPU serves the miss, the admission is copied in the background)  vs  one (fetched in the step)
giving the four states base (online, two), foa (online, one), bypass (MIN, two), fetch (MIN, one). Their effects on
the time per token are order-free: the Shapley value of each factor is the mean of its effect at the two levels of
the other, and the interaction is the difference of those two effects. Pacing (prefetching MIN's admissions a few steps
ahead, fetch -> both3p) needs foresight by construction, so it is nested under the MIN set rather than crossed with the
online set. The rest is both3p -> the limit.

Time per token is the arithmetic mean over problems of decode ms / decoded tokens (costs are averaged arithmetically;
speeds quoted elsewhere are the harmonic means this implies). Intervals: paired bootstrap over problems within a host
(10,000 resamples); with several hosts, a two-stage bootstrap (hosts, then problems within each drawn host).

    python scripts/factorial_shapley.py            # prereg/factorial_shapley.json, paper/wsg_shapley.tex
"""
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = "/home/claude/gpu-branch/results"
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
STATES = ("base", "foa", "bypass", "fetch", "both3p")
HB = {("g", 14): "gpt-oss 11%", ("g", 32): "gpt-oss 25%", ("q", 16): "Qwen3 12.5%", ("q", 32): "Qwen3 25%"}
RNG = np.random.default_rng(20261006)
NB = 10000


def label_of(cfg):
    for part in cfg.split(":"):
        if part.startswith("stats="):
            return os.path.basename(part[6:]).replace(".json", "")
    return None


def per_problem_ms(path, tag, C):
    """{state: {seq: ms per token}} from one ec_<tag>_C<C>.jsonl."""
    out = {}
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        lab = label_of(r["config"])
        if lab is None:
            continue
        st = lab.split(f"_C{C}_", 1)[1] if f"_C{C}_" in lab else lab
        out.setdefault(st, {})[r["seq"]] = r["decode_ms"] / r["n_decode"]
    return out


def effects(ms, limit):
    """ms: {state: mean ms per token}. Effects in ms (positive = time saved)."""
    b, fo, by, fe, bo = (ms[s] for s in STATES)
    e = dict(gap=b - limit, set_two=b - by, set_one=fo - fe, reads_online=b - fo, reads_min=by - fe, pacing=fe - bo, rest=bo - limit)
    e["both"] = b - fe
    e["phi_set"] = 0.5 * (e["set_two"] + e["set_one"])
    e["phi_reads"] = 0.5 * (e["reads_online"] + e["reads_min"])
    e["interaction"] = e["set_one"] - e["set_two"]
    for k in ("phi_set", "phi_reads", "interaction", "pacing", "rest", "set_two", "set_one", "reads_online", "reads_min", "both"):
        e[k + "_share"] = e[k] / e["gap"]
    return e


def hosts():
    """(host label, results dir, foresight json with the limit) for every host with the four crossed states."""
    out = [("O4", f"{RES}/096a_factorial_9950x@vast", P("prereg", "foresight_096a.json")),
           ("O5", f"{RES}/096b_factorial_9950x3d@vast", P("prereg", "foresight_096b.json"))]
    pl = P("prereg", "panel_limits.json")   # written by scripts/panel_099.py from each panel host's probe
    panel = json.load(open(pl)) if os.path.exists(pl) else {}
    for d in sorted(glob.glob(f"{RES}/099?_panel@vast")):
        name = "P" + os.path.basename(d)[3]
        out.append((name, d, panel.get(name)))
    # the new machines of jobs 100 and 101 (their relaunches of panel machines are left out: those machines are here)
    for d in sorted(glob.glob(f"{RES}/10[01]?_*@vast")):
        b = os.path.basename(d)
        if b[:4] in ("100a", "100c", "101a") or not os.path.exists(f"{d}/ec_g_C14.jsonl"):
            continue
        out.append(("N" + b[:4], d, limit1_of(d)))
    return out


def limit1_of(d):
    """the bound of the paper (one host rate, the probe's highest) at each gpt-oss host-bound cell, from the host's probe"""
    import sys
    sys.path.insert(0, ROOT)
    from scripts.speed_limit import MODELS, host_rates, limit
    v2 = json.load(open(P("prereg", "speed_limit_v2.json")))
    bh = host_rates(open(f"{d}/concur.txt").read())
    out = {}
    for (tag, C), cell in HB.items():
        if tag != "g":
            continue
        m = v2["models"]["gpt-oss-120b"]
        mm = MODELS["gpt-oss-120b"]
        t, _ = limit(m["rows"][str(C)]["reads_per_token"]["exact"], m["L"] * m["k"], mm["S"], mm["D"], bh, v2["B_gpu_datasheet"])
        out[cell] = 1e3 * t
    return out


def limit2_of(d):
    """the two-path bound (scripts/speed_limit.py, limit_two_path) at each host-bound cell, from the host's own probe"""
    import sys
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(os.environ.get("MOSL_GPU_BRANCH", "/home/claude/gpu-branch"), "jobs", "ec2"))
    from fetch_table import bandwidths
    from scripts.speed_limit import MODELS, limit_two_path
    txt = open(f"{d}/concur.txt").read()
    cores = int(re.search(r"usable physical cores (\d+)", open(f"{d}/cores.txt").read()).group(1))
    bc, bp, bb = bandwidths(txt, cores - 2 if cores > 4 else 2)
    v2 = json.load(open(P("prereg", "speed_limit_v2.json")))
    out = {}
    for (tag, C), cell in HB.items():
        key = "gpt-oss-120b" if tag == "g" else "qwen3-30b-a3b-bf16"
        m = v2["models"][key]
        row = m["rows"].get(str(C))
        if not row:
            continue
        mm = MODELS[key]
        t, _ = limit_two_path(row["reads_per_token"]["exact"], m["L"] * m["k"], mm["S"], mm["D"], bc * 1e9, bp * 1e9, bb * 1e9,
                              v2["B_gpu_datasheet"])
        out[cell] = 1e3 * t
    return out


def limits_of(x):
    if isinstance(x, dict):
        return x
    d = json.load(open(x))
    return {c["label"]: c["model_terms"]["limit_ms"] for c in d["cells"]}


def main():
    res = {"hosts": [], "pooled": {}}
    draws = {}   # cell -> list of (host, {state: per-problem arrays})
    for hname, d, limp in hosts():
        if limp is None:
            continue
        lim = limits_of(limp)
        lim2 = limit2_of(d)
        h = {"host": hname, "dir": os.path.basename(d), "cells": {}}
        for (tag, C), cell in HB.items():
            p = f"{d}/ec_{tag}_C{C}.jsonl"
            if not os.path.exists(p) or cell not in lim:
                continue
            pp = per_problem_ms(p, tag, C)
            if not all(s in pp for s in STATES):
                continue
            seqs = sorted(set.intersection(*(set(pp[s]) for s in STATES)))
            arr = {s: np.array([pp[s][q] for q in seqs]) for s in STATES}
            point = effects({s: arr[s].mean() for s in STATES}, lim[cell])
            idx = RNG.integers(0, len(seqs), (NB, len(seqs)))
            bs = {s: arr[s][idx].mean(axis=1) for s in STATES}
            boot = effects(bs, lim[cell])
            ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}
            h["cells"][cell] = {"n": len(seqs), "limit_ms": lim[cell], "ms": {s: float(arr[s].mean()) for s in STATES},
                                "effects": {k: float(v) for k, v in point.items()}, "ci": ci}
            if cell in lim2:   # how much of the gap the two-path bound explains as slack in the one-path bound
                h["cells"][cell]["limit2_ms"] = lim2[cell]
                h["cells"][cell]["effects"]["slack_share"] = (lim2[cell] - lim[cell]) / point["gap"]
            draws.setdefault(cell, []).append(arr)
        res["hosts"].append(h)
    # pooled over hosts: two-stage bootstrap (hosts, then problems within the drawn host); the gap uses each host's limit
    for cell, arrs in draws.items():
        if len(arrs) < 3:
            continue
        lims = [h["cells"][cell]["limit_ms"] for h in res["hosts"] if cell in h["cells"]]
        H = len(arrs)
        pts = [effects({s: a[s].mean() for s in STATES}, l) for a, l in zip(arrs, lims)]
        mean_pt = {k: float(np.mean([p[k] for p in pts])) for k in pts[0]}
        bt = {k: [] for k in pts[0]}
        for _ in range(2000):
            hs = RNG.integers(0, H, H)
            acc = {k: 0.0 for k in pts[0]}
            for hi in hs:
                a = arrs[hi]
                q = RNG.integers(0, len(a["base"]), len(a["base"]))
                e = effects({s: a[s][q].mean() for s in STATES}, lims[hi])
                for k in acc:
                    acc[k] += e[k] / H
            for k in acc:
                bt[k].append(acc[k])
        res["pooled"][cell] = {"hosts": H, "mean": mean_pt, "ci": {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bt.items()}}
    json.dump(res, open(P("prereg", "factorial_shapley.json"), "w"), indent=1)
    # macros: shares over the host-cells measured, as percentages of the gap
    M = {}
    # each host's link-to-CPU read ratio (its own probe)
    ratio = {}
    for hn, job in (("O4", "096a"), ("O5", "096b")):
        hh = json.load(open(P("prereg", f"foresight_{job}.json")))["host"]
        ratio[hn] = hh["B_p"] / hh["B_c"]
    pj = P("prereg", "panel_099.json")
    if os.path.exists(pj):
        for hh in json.load(open(pj))["hosts"]:
            ratio[hh["host"]] = hh["B_p"] / hh["B_c"]
    for h in res["hosts"]:
        if h["host"].startswith("N"):   # jobs 100/101 machines: their own probe
            import sys
            sys.path.insert(0, os.path.join(os.environ.get("MOSL_GPU_BRANCH", "/home/claude/gpu-branch"), "jobs", "ec2"))
            from fetch_table import bandwidths
            d = f"{RES}/{h['dir']}"
            cores = int(re.search(r"usable physical cores (\d+)", open(f"{d}/cores.txt").read()).group(1))
            bc, bp, _ = bandwidths(open(f"{d}/concur.txt").read(), cores - 2 if cores > 4 else 2)
            ratio[h["host"]] = bp / bc
        h["link_over_cpu"] = ratio.get(h["host"])
    json.dump(res, open(P("prereg", "factorial_shapley.json"), "w"), indent=1)
    allc_all = [(h["host"], c, v) for h in res["hosts"] for c, v in h["cells"].items()]
    allc = allc_all

    def rng(name, key, fmt="{:.0f}", scale=100.0):
        vals = [scale * v["effects"][key] for _, _, v in allc]
        def f(x):
            t = fmt.format(x)
            return ("$-$" + t[1:]) if t.startswith("-") and t.strip("-0.") else t.lstrip("-") if t.startswith("-") else t
        M[name + "Min"] = f(min(vals))
        M[name + "Max"] = f(max(vals))
    rng("fsSet", "phi_set_share")
    rng("fsReads", "phi_reads_share")
    rng("fsInter", "interaction_share")
    rng("fsPacing", "pacing_share")
    rng("fsRest", "rest_share")
    rng("fsReadsOnline", "reads_online_share")
    rng("fsReadsMin", "reads_min_share")
    rng("fsSetTwo", "set_two_share")
    rng("fsSetOne", "set_one_share")
    rng("fsBoth", "both_share")
    # the same on hosts whose link reads at least half as fast as their CPU, and on the others
    for sub, test in (("fsBal", lambda r: r is not None and r >= 0.5), ("fsSlow", lambda r: r is not None and r < 0.5)):
        allc = [(hn, c, v) for hn, c, v in allc_all if test(ratio.get(hn))]
        if not allc:
            continue
        M[sub + "Hosts"] = str(len({hn for hn, _, _ in allc}))
        for nm, key in (("Set", "phi_set_share"), ("Reads", "phi_reads_share"), ("Inter", "interaction_share"), ("Pacing", "pacing_share"),
                        ("Rest", "rest_share"), ("ReadsOnline", "reads_online_share"), ("SetTwo", "set_two_share"), ("SetOne", "set_one_share"),
                        ("Both", "both_share")):
            rng(sub + nm, key)
    allc = allc_all
    # either choice alone (cache alone or load alone) on the fast-link hosts
    al = [100 * v["effects"][k] for hn, c, v in allc_all if ratio.get(hn) is not None and ratio[hn] >= 0.5
          for k in ("set_two_share", "reads_online_share")]
    if al:
        f = lambda x: f"{x:.0f}".replace("-", "$-$")  # noqa: E731
        M["fsBalAloneMin"] = f(min(al)); M["fsBalAloneMax"] = f(max(al))
    # the two-path bound's share of the gap, and the rest beyond it, by link class
    for sub, test in (("Bal", lambda r: r is not None and r >= 0.5), ("Slow", lambda r: r is not None and r < 0.5)):
        sl = [v["effects"]["slack_share"] for hn, c, v in allc_all if test(ratio.get(hn)) and "slack_share" in v["effects"]]
        rb = [v["effects"]["rest_share"] - v["effects"]["slack_share"] for hn, c, v in allc_all if test(ratio.get(hn)) and "slack_share" in v["effects"]]
        if sl:
            M[f"fsSlack{sub}Min"] = f"{100 * min(sl):.0f}"; M[f"fsSlack{sub}Max"] = f"{100 * max(sl):.0f}"
            M[f"fsRestTwo{sub}Min"] = f"{100 * min(rb):.0f}"; M[f"fsRestTwo{sub}Max"] = f"{100 * max(rb):.0f}"
    # the worked example of the text: O4 at gpt-oss 11%
    ex = next(h for h in res["hosts"] if h["host"] == "O4")["cells"]["gpt-oss 11%"]["effects"]
    M["fsExGap"] = f"{ex['gap']:.1f}"
    for nm, key in (("ReadsOnline", "reads_online_share"), ("SetTwo", "set_two_share"), ("Both", "both_share"),
                    ("Inter", "interaction_share"), ("Set", "phi_set_share"), ("Reads", "phi_reads_share"),
                    ("Pacing", "pacing_share"), ("Rest", "rest_share"), ("SetOne", "set_one_share")):
        M["fsEx" + nm] = f"{100 * ex[key]:.1f}"
    # on the slow-link hosts: at how many host-cells the single read in the step costs time (Shapley value below 0)
    slow = [(hn, c, v) for hn, c, v in allc_all if ratio.get(hn) is not None and ratio[hn] < 0.5]
    M["fsSlowReadsLoseN"] = str(sum(v["effects"]["phi_reads_share"] < 0 for _, _, v in slow))
    M["fsSlowCells"] = str(len(slow))
    M["fsAccHostCells"] = str(len(allc))
    M["fsAccHosts"] = str(len(res["hosts"]))
    M["fsAccHostsTwo"] = str(sum(not h["host"].startswith("O") for h in res["hosts"]))   # hosts at the two gpt-oss budgets only
    # the table: O4 and O5 per cell, the panel's hosts summarised per cell (mean, and range over hosts)
    def pc(x):
        v = int(round(100 * x))
        return f"$-${-v}" if v < 0 else f"{v}"
    cols = ("reads_online_share", "set_two_share", "interaction_share", "phi_set_share", "phi_reads_share", "pacing_share", "rest_share")
    rows = []
    for h in res["hosts"]:
        if not h["host"].startswith("O"):
            continue
        for c, v in h["cells"].items():
            e = v["effects"]
            rows.append(f"{h['host']} & {c.replace('%', chr(92) + '%')} & {e['gap']:.1f} & " + " & ".join(pc(e[k]) for k in cols) + r" \\")
    pan = [h for h in res["hosts"] if h["host"].startswith("P")]
    if pan:
        rows.append(r"\midrule")
        for grp, test in (("Panel, fast link", lambda r: r is not None and r >= 0.5), ("Panel, slow link", lambda r: r is not None and r < 0.5)):
            for c in HB.values():
                hs = [h["cells"][c]["effects"] for h in pan if c in h["cells"] and test(h.get("link_over_cpu"))]
                if not hs:
                    continue
                cells = []
                for k in cols:
                    vals = [e[k] for e in hs]
                    cells.append(f"{pc(np.mean(vals))}" + (f" {{\\scriptsize({pc(min(vals))} to {pc(max(vals))})}}" if len(vals) > 1 else ""))
                rows.append(f"{grp} ({len(hs)}) & {c.replace('%', chr(92) + '%')} & {np.mean([e['gap'] for e in hs]):.1f} & " + " & ".join(cells) + r" \\")
    new = [h for h in res["hosts"] if h["host"].startswith("N")]
    names = {"N100b": "5700X3D", "N100f": "9950X, x8", "N101b": "9800X3D, slower link"}
    if new:
        rows.append(r"\midrule")
        for h in new:
            for c, v in h["cells"].items():
                e = v["effects"]
                rows.append(f"{names.get(h['host'], h['host'])} ({h['link_over_cpu']:.2f}) & {c.replace('%', chr(92) + '%')} & {e['gap']:.1f} & "
                            + " & ".join(pc(e[k]) for k in cols) + r" \\")
    with open(P("paper", "tab_shapley.tex"), "w") as f:
        f.write("% generated by scripts/factorial_shapley.py\n")
        f.write(r"""\begin{table*}[t]\centering\footnotesize
\caption{An order-free accounting of the gap between the deployed cache and the bound at the host-bound budgets, as
percentages of the gap (ms per token). The engine ran both choices of what to cache (the deployed policy's admissions
or MIN's) with both ways to load a cached expert (two reads: the CPU serves it, then a copy; one read: a copy in the
step). \emph{Load alone} and \emph{cache alone} are each choice's effect with the other at the deployed level; the
\emph{interaction} is how much more MIN's set is worth loaded once than loaded twice. The Shapley values average each
choice's effect over the other's two levels and add up, with the prefetch (nested under MIN's set, since it needs
foresight) and the rest, to 100\%. Panel: mean over its hosts (number
in parentheses) and, in small type, the range over its hosts (not an interval); a fast link reads at least half as fast as
the host's CPU, a slow one less. Last rows: the new machines of the follow-up launches, with their link-to-CPU ratio.}\label{tab:shapley}
\setlength\tabcolsep{3pt}\resizebox{\textwidth}{!}{%
\begin{tabular}{llrrrrrrrr}\toprule
 & & Gap & Load & Cache & Inter- & \multicolumn{2}{c}{Shapley value} & Prefetch & Rest \\
Host & Budget & (ms) & alone & alone & action & cache & load & & \\\midrule
""")
        f.write("\n".join(rows) + "\n\\bottomrule\\end{tabular}}\\end{table*}\n")
    with open(P("paper", "wsg_shapley.tex"), "w") as f:
        f.write("% generated by scripts/factorial_shapley.py from prereg/factorial_shapley.json\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    for h in res["hosts"]:
        for c, v in h["cells"].items():
            e = v["effects"]
            print(f"{h['host']:4s} {c:12s} gap {e['gap']:6.2f} ms | set {100*e['phi_set_share']:5.0f}%  reads {100*e['phi_reads_share']:5.0f}%  "
                  f"(interaction {100*e['interaction_share']:5.0f}%; reads|online {100*e['reads_online_share']:4.0f}%, reads|MIN {100*e['reads_min_share']:4.0f}%; "
                  f"set|two {100*e['set_two_share']:4.0f}%, set|one {100*e['set_one_share']:4.0f}%) pacing {100*e['pacing_share']:4.0f}%  rest {100*e['rest_share']:4.0f}%")
    for c, v in res["pooled"].items():
        print("pooled", c, v["hosts"], {k: round(100 * v["mean"][k], 0) for k in v["mean"] if k.endswith("_share")})
    print({k: M[k] for k in sorted(M)})


if __name__ == "__main__":
    main()

"""Job 100: the window on the deployed read path, and a probe-only test of the calibrated time model, on four hosts
(results/100?_window@vast on the gpu branch: 100a = Pd of job 099 relaunched, 100b = Ryzen 7 5700X3D, 100c = Ph of job
099 relaunched, 100d = Core i5-12400F). Scores the eight predictions of jobs/100_window@vast.sh by machine under the
interval rule of the scorecard, and writes prereg/job100.json, prereg/scorecard_100.json, paper/wsg_job100.tex and
paper/tab_window100.tex.

    python scripts/job100.py
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from scripts.panel_099 import cell_data, host_info, _status  # noqa: E402

RES = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
S = 13253760
G = {14: 4.56e-3, 32: 4.53e-3}                       # frozen in jobs/ec2/predict_100.py before launch
COUNT = {14: {"aa": (58.365, 58.365), "w4": (48.679, 47.103), "w16": (40.198, 29.456), "fetch": (39.738, 19.799)},
         32: {"aa": (28.080, 28.080), "w4": (25.244, 24.347), "w16": (20.476, 19.598), "fetch": (16.442, 9.683)}}
NAMES = {"100a": "Pd again", "100b": "5700X3D", "100c": "Ph again", "100d": "12400F", "100e": "13900KF, faster link",
         "100f": "9950X, slower link"}
RELAUNCH = {"100a": "099d", "100c": "099h"}
CELLS = {14: "gpt-oss 11%", 32: "gpt-oss 25%"}
GL = {14: "g11", 32: "g25"}
RNG = np.random.default_rng(100)


def tmodel(xc, xp, h, C):
    return G[C] + max(xc / (h["B_c"] * 1e9), xp / (h["B_p"] * 1e9), (xc + xp) / (h["B_cp"] * 1e9))


def ratio_ci(a, b, n=10000):
    """speed of b relative to a (t_a / t_b), paired by problem, 95% bootstrap"""
    idx = RNG.integers(0, len(a), size=(n, len(a)))
    r = a[idx].mean(1) / b[idx].mean(1)
    return [float(a.mean() / b.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def load_host(d):
    h = dict(dir=os.path.basename(d), job=os.path.basename(d)[:4], **host_info(d), cells={})
    pp = f"{d}/predict_probe.json"
    if os.path.exists(pp):   # the job's shell trace precedes the JSON in this file
        txt = open(pp).read()
        h["predict"] = json.loads(txt[txt.index("{"):])
    else:
        h["predict"] = None
    rp = f"{d}/value_map_replay.jsonl"
    h["replay"] = {int(json.loads(l)["C"]): json.loads(l) for l in open(rp) if l.strip()} if os.path.exists(rp) else {}
    for C in (14, 32):
        cd = cell_data(d, "g", C)
        if not cd:
            continue
        t = {k: float(v.mean()) for k, v in cd["arr"].items()}
        st = {}
        for k in cd["arr"]:
            f = f"{d}/st_g_C{C}_{k}.json"
            if os.path.exists(f):
                s = json.load(open(f))
                n = max(1, s["steps"])
                st[k] = dict(misses=s["misses"] / n, fetches=s["fetches"] / n, admits=s["admits"] / n,
                             plan_us=s.get("oracle_plan_us_per_step", 0.0))
        sp = {k: ratio_ci(cd["arr"]["base"], cd["arr"][k]) for k in cd["arr"] if k != "base"}
        h["cells"][C] = dict(ms=t, counters=st, speed=sp, n=len(cd["seqs"]), arr=cd["arr"])
    return h


def main():
    hosts = [load_host(d) for d in sorted(glob.glob(f"{RES}/100?_window@vast")) if os.path.exists(f"{d}/ec_g_C14.jsonl")]
    old = {}
    for j in set(RELAUNCH.values()):
        d = f"{RES}/{j}_panel@vast"
        if os.path.exists(d):
            old[j] = {C: cell_data(d, "g", C) for C in (14, 32)}
    clauses = []

    def add(cid, short, typ, meas, ci, ok, thr, host):
        stt, why = _status(meas, ci, ok)
        clauses.append(dict(id=cid, short=short, type=typ, measured=None if meas is None else round(float(meas), 4),
                            ci=None if ci is None else [round(float(x), 4) for x in ci], threshold=thr, status=stt, why=why, host=host))

    probe_err = []
    for h in hosts:
        nm = NAMES.get(h["job"], h["job"])
        h["link_over_cpu"] = h["B_p"] / h["B_c"]
        for C, c in h["cells"].items():
            lab = f"{CELLS[C]}, {nm}"
            # 1. counters of the host-independent states
            for k, (mi, fe) in COUNT[C].items():
                if k in c["counters"]:
                    dev = max(abs(c["counters"][k]["misses"] / mi - 1), abs(c["counters"][k]["fetches"] / fe - 1))
                    add(f"100{h['job'][3]}-P1-{k}-{GL[C]}", f"{k} counters equal job 099's ({lab})", "band", dev, None, lambda v: v <= 0.005, "<= 0.5%", nm)
            # 2. probe-only predictions
            if h["predict"]:
                for k in COUNT[C]:
                    key = f"C{C}_{k}"
                    if key in h["predict"]["pred_ms"] and k in c["ms"]:
                        e = h["predict"]["pred_ms"][key] / c["ms"][k] - 1
                        probe_err.append(dict(host=nm, C=C, state=k, err=e, ratio=h["link_over_cpu"]))
                        if k in ("aa", "w4"):
                            add(f"100{h['job'][3]}-P2-{k}-{GL[C]}", f"probe-only {k} time within 12% ({lab})", "band", abs(e), None, lambda v: v <= 0.12, "<= 12%", nm)
                        if k == "fetch" and h["link_over_cpu"] < 0.7:
                            add(f"100{h['job'][3]}-P2-fetchsign-{GL[C]}", f"probe-only fetch time under-predicted ({lab})", "sign", e, None, lambda v: v < 0, "< 0", nm)
            # 3. frozen G with the run's counters: the sign of fetch/base, foa/base, aa/base
            pt = {}
            for k in ("base", "foa", "aa", "fetch", "w4", "w16"):
                if k in c["counters"]:
                    m = c["counters"][k]
                    pt[k] = tmodel((m["misses"] - m["fetches"]) * S, m["fetches"] * S, h, C)
            c["pred_frozen_ms"] = {k: 1e3 * v for k, v in pt.items()}
            for k in ("fetch", "foa", "aa"):
                if k in pt and "base" in pt and k in c["speed"]:
                    pr = pt["base"] / pt[k]
                    meas = c["speed"][k]
                    if abs(pr - 1) >= 0.05:
                        add(f"100{h['job'][3]}-P3-{k}-{GL[C]}", f"model sign of {k}/base, predicted {pr:.2f} ({lab})", "sign",
                            meas[0], meas[1:], (lambda v, pr=pr: (v > 1) == (pr > 1)), ">1" if pr > 1 else "<1", nm)
            # 4. the window on the deployed path
            sp = c["speed"]
            for k, lo, hi in (("b16", 1.00, 1.25), ("b4", 0.98, 1.12), ("b8r5", 0.95, 1.08)):
                if k in sp:
                    add(f"100{h['job'][3]}-P4-{k}-{GL[C]}", f"{k}/base in [{lo}, {hi}] ({lab})", "band", sp[k][0], sp[k][1:],
                        (lambda v, lo=lo, hi=hi: lo <= v <= hi), f"{lo}-{hi}", nm)
            if "b16" in sp and "bypass" in sp:
                add(f"100{h['job'][3]}-P4-b16bypass-{GL[C]}", f"b16/base <= bypass/base + 0.03 ({lab})", "threshold",
                    sp["b16"][0] - sp["bypass"][0], None, lambda v: v <= 0.03, "<= 0.03", nm)
            if "b4" in c["ms"] and "b16" in c["ms"]:
                a4, a16 = c["arr"]["b4"], c["arr"]["b16"]
                add(f"100{h['job'][3]}-P4-mono-{GL[C]}", f"b16 at least as fast as b4 ({lab})", "sign", *(lambda r: (r[0], r[1:]))(ratio_ci(a4, a16)),
                    lambda v: v >= 1.0, ">= 1", nm)
            # 5. the path decides (11% only)
            if C == 14 and "b16" in sp and "w16" in sp:
                r = h["link_over_cpu"]
                diff = sp["b16"][0] - sp["w16"][0]
                if r < 0.5:
                    add(f"100{h['job'][3]}-P5-g11", f"b16/base > w16/base at link/CPU {r:.2f} ({lab})", "sign", diff, None, lambda v: v > 0, "> 0", nm)
                elif r >= 0.9:
                    add(f"100{h['job'][3]}-P5-g11", f"w16/base > b16/base at link/CPU {r:.2f} ({lab})", "sign", -diff, None, lambda v: v > 0, "> 0", nm)
            # 7. plan time
            for k in ("w4", "w16", "b4", "b16", "b8r5", "fetch"):
                if k in c["counters"]:
                    # the deployed-path windows plan after the step, where the engine does not time the plan: untested
                    v = None if k.startswith("b") else c["counters"][k]["plan_us"]
                    add(f"100{h['job'][3]}-P7-{k}-{GL[C]}", f"host plan of {k} at most 150 us per step ({lab})", "threshold",
                        v, None, lambda v: v <= 150, "<= 150 us", nm)
            # 8. engine misses of the deployed-path windows against the replay's
            rp = h["replay"].get(C, {})
            for k in ("b4", "b16", "b8r5"):
                if k in rp and k in c["counters"]:
                    dev = c["counters"][k]["misses"] / rp[k]["misses"] - 1
                    c.setdefault("replay_dev", {})[k] = dev
                    add(f"100{h['job'][3]}-P8-{k}-{GL[C]}", f"engine misses of {k} within 10% of the replay ({lab})", "band", abs(dev), None,
                        lambda v: v <= 0.10, "<= 10%", nm)
        # 6. relaunch
        if h["job"] in RELAUNCH and RELAUNCH[h["job"]] in old:
            for C, c in h["cells"].items():
                o = old[RELAUNCH[h["job"]]][C]
                if not o:
                    continue
                lab = f"{CELLS[C]}, {nm}"
                ob = o["arr"]["base"].mean()
                c["relaunch"] = {"base_ms_ratio": c["ms"]["base"] / ob}
                add(f"100{h['job'][3]}-P6-base-{GL[C]}", f"base time within 3% of job {RELAUNCH[h['job']]} ({lab})", "band",
                    abs(c["ms"]["base"] / ob - 1), None, lambda v: v <= 0.03, "<= 3%", nm)
                for k in ("foa", "aa", "fetch", "both3p", "w4", "w16"):
                    if k in c["speed"] and k in o["arr"]:
                        d0 = ob / o["arr"][k].mean()
                        dv = c["speed"][k][0] - d0
                        c["relaunch"][k] = dv
                        add(f"100{h['job'][3]}-P6-{k}-{GL[C]}", f"{k}/base within 0.03 of job {RELAUNCH[h['job']]} ({lab})", "band",
                            abs(dv), None, lambda v: v <= 0.03, "<= 0.03", nm)
    # 9. the same CPU model behind a different link: fetch/base at gpt-oss 11%
    def fb(job_dir):
        cd = cell_data(f"{RES}/{job_dir}", "g", 14)
        return float(cd["arr"]["base"].mean() / cd["arr"]["fetch"].mean()) if cd and "fetch" in cd["arr"] else None
    byjob = {h["job"]: h for h in hosts}
    for job, refs, sign, nm in (("100e", ("099d_panel@vast", "100a_window@vast"), 1, "13900KF, faster link"),
                                ("100f", ("099e_panel@vast", "099j_panel@vast"), -1, "9950X, slower link")):
        have = job in byjob and 14 in byjob[job]["cells"]
        v = byjob[job]["cells"][14]["speed"]["fetch"][0] if have else None
        for r in refs:
            rv = fb(r)
            add(f"{job}-P9-{r[:4]}", f"fetch/base at 11%: {nm} {'above' if sign > 0 else 'below'} {r[:4]}", "sign",
                None if v is None or rv is None else sign * (v - rv), None, lambda x: x > 0, "> 0", nm)
    # 2. pooled: the median probe-only error
    if probe_err:
        med = float(np.median([abs(e["err"]) for e in probe_err]))
        add("100-P2-median", "median |error| of the probe-only predictions over all new host-cells", "threshold", med, None, lambda v: v <= 0.05, "<= 5%", "pooled")
    # outputs
    out = []
    for h in hosts:
        hh = {k: v for k, v in h.items() if k != "cells"}
        hh["cells"] = {str(C): {k: v for k, v in c.items() if k != "arr"} for C, c in h["cells"].items()}
        out.append(hh)
    json.dump(dict(hosts=out, probe_err=probe_err), open(P("prereg", "job100.json"), "w"), indent=1)
    json.dump(dict(job="100", script="jobs/100_window@vast.sh", commit="4fda930 (predictions), 752e1c3 (hosts named)",
                   scored_by="scripts/job100.py (machine)", clauses=clauses), open(P("prereg", "scorecard_100.json"), "w"), indent=1)
    from collections import Counter
    st = Counter(c["status"] for c in clauses)
    print("job 100 clauses:", dict(st))
    for c in clauses:
        if c["status"] in ("failed", "untested"):
            print("  ", c["status"], c["id"], c["short"], c["measured"], c["ci"], c["threshold"])
    M = {"jaHosts": str(len(hosts)), "jaClauses": str(len(clauses)), "jaHeld": str(st.get("held", 0)),
         "jaPoint": str(st.get("held (point)", 0)), "jaFailed": str(st.get("failed", 0)), "jaUntested": str(st.get("untested", 0))}
    if probe_err:
        e = [abs(x["err"]) for x in probe_err]
        M["jaProbeMed"] = f"{100 * np.median(e):.1f}"; M["jaProbeMax"] = f"{100 * max(e):.0f}"; M["jaProbeN"] = str(len(e))
        ea = [abs(x["err"]) for x in probe_err if x["state"] in ("aa", "w4")]
        M["jaProbeAaMax"] = f"{100 * max(ea):.0f}"
        ef = [x["err"] for x in probe_err if x["state"] in ("fetch", "w16")]
        M["jaProbeFetchMin"] = f"{100 * min(ef):.0f}".replace("-", "$-$"); M["jaProbeFetchMax"] = f"{100 * max(ef):.0f}".replace("-", "$-$")
        M["jaProbeFetchMinAbs"] = f"{100 * abs(min(ef)):.0f}"
    sg = [c for c in clauses if "-P3-" in c["id"]]
    M["jaSignRight"] = str(sum(c["status"] != "failed" for c in sg)); M["jaSignN"] = str(len(sg))
    rl = [abs(c["relaunch"][k]) for h in hosts for c in h["cells"].values() if "relaunch" in c for k in c["relaunch"] if k != "base_ms_ratio"]
    rb = [abs(c["relaunch"]["base_ms_ratio"] - 1) for h in hosts for c in h["cells"].values() if "relaunch" in c]
    if rl:
        M["jaRelaunchRatioMax"] = f"{max(rl):.3f}"; M["jaRelaunchBaseMax"] = f"{100 * max(rb):.1f}"
    for C, nm in ((14, "Low"), (32, "Mid")):
        for k, kn in (("b4", "BFour"), ("b16", "BSixteen"), ("b8r5", "BEightHalf"), ("w4", "WFour"), ("w16", "WSixteen"), ("bypass", "Bypass")):
            v = [h["cells"][C]["speed"][k][0] for h in hosts if C in h["cells"] and k in h["cells"][C]["speed"]]
            if v:
                M[f"ja{kn}{nm}Min"] = f"{min(v):.2f}"; M[f"ja{kn}{nm}Max"] = f"{max(v):.2f}"
                M[f"ja{kn}{nm}GainMax"] = f"{100 * (max(v) - 1):.0f}"
    bl = [h["cells"][C]["speed"][k][0] for h in hosts for C in h["cells"] for k in ("b4", "b16", "b8r5") if k in h["cells"][C]["speed"]]
    if bl:
        M["jaWorstBLossPct"] = f"{100 * (1 - min(bl)):.0f}"
    bd = []
    for h in hosts:
        for C, c in h["cells"].items():
            rp = h["replay"].get(C, {})
            if "online_k1" in rp and "base" in c["counters"]:
                bd.append(abs(c["counters"]["base"]["misses"] / rp["online_k1"] - 1))
    if bd:
        M["jaBaseReplayDevMax"] = f"{100 * max(bd):.1f}"
    rd = [v for h in hosts for c in h["cells"].values() for v in c.get("replay_dev", {}).values()]
    if rd:
        M["jaReplayDevMin"] = f"{100 * min(rd):.0f}"; M["jaReplayDevMax"] = f"{100 * max(rd):.0f}"
    # the path comparison at 11%: deployed-path 16-token window against the in-step one, by host
    pc = [(h["link_over_cpu"], h["cells"][14]["speed"]["b16"][0] - h["cells"][14]["speed"]["w16"][0]) for h in hosts
          if 14 in h["cells"] and "b16" in h["cells"][14]["speed"] and "w16" in h["cells"][14]["speed"]]
    if pc:
        lo = [(r, d) for r, d in pc if r < 0.9]
        hi = [(r, d) for r, d in pc if r >= 0.9]
        M["jaPathLowWins"] = str(sum(d > 0 for _, d in lo)); M["jaPathLowN"] = str(len(lo))
        M["jaPathHighWins"] = str(sum(d < 0 for _, d in hi)); M["jaPathHighN"] = str(len(hi))
        if lo:
            M["jaPathLowRatioMax"] = f"{max(r for r, _ in lo):.2f}"
        if hi:
            M["jaPathHighRatioMin"] = f"{min(r for r, _ in hi):.2f}"
    gap = [h["cells"][C]["speed"]["b16"][0] - h["cells"][C]["speed"]["bypass"][0] for h in hosts for C in h["cells"]
           if "b16" in h["cells"][C]["speed"] and "bypass" in h["cells"][C]["speed"]]
    if gap:
        M["jaBSixteenOverBypassMax"] = f"{max(gap):.2f}"
    fo = [h["cells"][C]["speed"]["foa"][0] for h in hosts for C in h["cells"] if "foa" in h["cells"][C]["speed"]]
    if fo:
        M["jaFoaMin"] = f"{min(fo):.2f}"; M["jaFoaMax"] = f"{max(fo):.2f}"
        pf = [h["cells"][C]["pred_frozen_ms"] for h in hosts for C in h["cells"] if "foa" in h["cells"][C].get("pred_frozen_ms", {})]
        pr = [p["base"] / p["foa"] for p in pf]
        M["jaFoaPredDevMax"] = f"{100 * max(abs(x - 1) for x in pr):.0f}"
    rr = [h["link_over_cpu"] for h in hosts]
    if rr:
        M["jaRatioMin"] = f"{min(rr):.2f}"; M["jaRatioMax"] = f"{max(rr):.2f}"
    with open(P("paper", "wsg_job100.tex"), "w") as f:
        f.write("% generated by scripts/job100.py from the job 100 hosts (results/100?_window@vast)\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    # the table: per host, speed of each window relative to the deployed cache
    rows = []
    for h in sorted(hosts, key=lambda x: x["link_over_cpu"]):
        for C in (14, 32):
            if C not in h["cells"]:
                continue
            sp = h["cells"][C]["speed"]
            g = lambda k: f"{sp[k][0]:.2f}" if k in sp else "--"  # noqa: E731
            rows.append(f"{NAMES.get(h['job'], h['job'])} & {h['link_over_cpu']:.2f} & {CELLS[C].replace('%', chr(92) + '%')} & "
                        f"{g('w4')} & {g('w16')} & {g('b4')} & {g('b16')} & {g('b8r5')} & {g('bypass')} & {g('fetch')} & {g('both3p')} \\\\")
    with open(P("paper", "tab_window100.tex"), "w") as f:
        f.write("% generated by scripts/job100.py\n")
        f.write(r"""\begin{table}[t]\centering\footnotesize
\caption{Foresight on each read path (job 100): speed relative to the deployed cache on the same machine. \emph{In step}:
admit every miss with a window of $W$ exact tokens, each admission copied in the step (as on the panel). \emph{Deployed
path}: the deployed policy with the same window, its misses split between the CPU and an in-step copy by the machine's
fetch table and its admissions copied in the background;
$W{=}8$ at recall 0.5 in the column marked $r$. The last three columns: \MinTwo, \MinOne{} and the read-ahead oracle.
Machines sorted by the probe's link-to-CPU ratio.}\label{tab:window100}
\setlength\tabcolsep{2.5pt}\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lrlrrrrrrrr@{}}\toprule
 & Link/ & & \multicolumn{2}{c}{In step} & \multicolumn{3}{c}{Deployed path} & & & Read- \\
Machine & CPU & Budget & $W{=}4$ & 16 & 4 & 16 & $r$ & \MinTwo & \MinOne & ahead \\\midrule
""" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}}\\end{table}\n")
    for h in hosts:
        print(h["job"], NAMES.get(h["job"]), h["cpu"], f"link {h['B_p']:.1f} cpu {h['B_c']:.1f} both {h['B_cp']:.1f} ratio {h['link_over_cpu']:.2f}")
        for C, c in h["cells"].items():
            print("   C", C, {k: round(v[0], 3) for k, v in c["speed"].items()})
            print("      replay dev", {k: round(v, 3) for k, v in c.get("replay_dev", {}).items()}, "relaunch", {k: round(v, 3) for k, v in c.get("relaunch", {}).items()})
    print(M)


if __name__ == "__main__":
    main()

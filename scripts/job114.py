"""Job 114 (jobs/114_decomp0@vast.sh on the gpu branch): Table 4's 2x2 on the CPU-only read path (the in-step fetches
off), beside Table 4's own cells, in one process per round. Scores the registered predictions H1-H6 per valid host from
the raw rows and counters, computes the decomposition of each host's gap on the CPU-only path, and writes
prereg/job114.json, prereg/scorecard_114.json, paper/wsg_job114.tex and paper/tab_job114.tex.

    python scripts/job114.py
"""
import glob
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from scripts import job109 as J  # noqa: E402

P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
GLOB = os.environ.get("MOSL_JOB114_GLOB", f"{J.RES}/114[a-z]_decomp0@vast")
CELLS = (14, 32)
G = {14: "g11", 32: "g25"}
NM = {14: "Low", 32: "Mid"}
NAMES = ("base", "base0", "foa", "foa0", "bypass", "bypass0", "fetch", "both3p")
# the counters of the earlier machines (jobs 109 and 112 for MIN-1R; job 113 for the CPU-only arms), per token
REF = {"fetch": {14: (39.7, 19.8), 32: (16.4, 9.7)},            # misses, fetches
       "base0": {14: (62.1, 8.2)}, "bypass0": {14: (43.5, 20.4)}}  # misses, admissions
word = J.word


def gap_parts(t, eq1):
    """the decomposition on the CPU-only path, in ms per token, and as shares of the gap (base - Eq. (1))"""
    gap = t["base"] - eq1
    ms = dict(table=t["base"] - t["base0"], setalone0=t["base0"] - t["bypass0"], oncealone0=t["base0"] - t["foa0"],
              together0=t["base0"] - t["fetch"], ahead=t["fetch"] - t["both3p"], left=t["both3p"] - eq1,
              setalone=t["base"] - t["bypass"], oncealone=t["base"] - t["foa"], together=t["base"] - t["fetch"])
    ms["interaction0"] = ms["together0"] - ms["setalone0"] - ms["oncealone0"]
    ms["interaction"] = ms["together"] - ms["setalone"] - ms["oncealone"]
    assert abs(ms["table"] + ms["together0"] + ms["ahead"] + ms["left"] - gap) < 1e-9
    return gap, ms, {k: v / gap for k, v in ms.items()}


def load114(d):
    J.A_NAMES = NAMES   # this job's process A
    h = J.load(d, CELLS, 0.5, need_b=False, keep_pp=True)
    v0 = J.read(d, "v0.txt")
    h["known"] = "V0: one of job 109's machines" in v0
    # the machine's fetch table (from its probe): all zero means its deployed path already had no in-step fetches
    import re
    m = re.search(r"gpt-oss ([0-9,]+)", J.read(d, "tables.txt"))
    h["table"] = m.group(1) if m else ""
    h["has_table"] = any(int(x) for x in h["table"].split(",") if x) if h["table"] else None
    for C, c in h["cells"].items():
        dr = c.get("_draws", {})
        c["rel"], c["rel_ci"] = {}, {}
        for x, y in (("bypass0", "bypass"), ("bypass0", "base0"), ("foa0", "base0"), ("base0", "base"), ("fetch", "base0"),
                     ("foa0", "foa"), ("bypass", "base"), ("foa", "base"), ("fetch", "base"), ("both3p", "base"),
                     ("bypass0", "base"), ("foa0", "base")):
            k = f"{x}/{y}"
            rx = c["ratio"].get(x) if x != "base" else 1.0
            ry = c["ratio"].get(y) if y != "base" else 1.0
            if rx and ry:
                c["rel"][k] = rx / ry
                dx = dr.get(x) if x != "base" else None
                dy = dr.get(y) if y != "base" else None
                if dx is not None and (dy is not None or y == "base"):
                    v = dx / (dy if dy is not None else 1.0)
                    c["rel_ci"][k] = tuple(float(q) for q in np.percentile(v, [2.5, 97.5]))
        # the interaction in ms per token, with a paired bootstrap over problems (each round's per-problem times
        # resampled together, the same draw for every configuration): t_bypass0 + t_foa0 - t_base0 - t_fetch
        pp = c.get("_pp", {})
        if pp and all(n in pp[r][0] for r in pp for n in ("base0", "foa0", "bypass0", "fetch")):
            rng = np.random.default_rng(114 + C)
            rounds = sorted(pp)
            seqs = sorted(set.intersection(*(set(pp[r][0][n]) for r in rounds for n in ("base0", "foa0", "bypass0", "fetch"))))
            arr = {r: {n: np.array([pp[r][0][n][s][0] for s in seqs]) for n in ("base0", "foa0", "bypass0", "fetch")}
                   for r in rounds}
            point = np.mean([arr[r]["bypass0"].mean() + arr[r]["foa0"].mean() - arr[r]["base0"].mean()
                             - arr[r]["fetch"].mean() for r in rounds])
            draws = []
            for _ in range(4000):
                i = rng.integers(0, len(seqs), len(seqs))   # the same problems in every round and configuration
                draws.append(np.mean([arr[r]["bypass0"][i].mean() + arr[r]["foa0"][i].mean() - arr[r]["base0"][i].mean()
                                      - arr[r]["fetch"][i].mean() for r in rounds]))
            c["inter_ms"] = float(point)
            c["inter_rounds"] = [float(arr[r]["bypass0"].mean() + arr[r]["foa0"].mean() - arr[r]["base0"].mean()
                                       - arr[r]["fetch"].mean()) for r in rounds]
            c["inter_ms_ci"] = tuple(float(q) for q in np.percentile(draws, [2.5, 97.5]))
        ct = c["counters"]
        c["misses"] = {n: float(np.mean([x["misses"] for x in v])) for n, v in ct.items()}
        c["admits"] = {n: float(np.mean([x["admits"] for x in v])) for n, v in ct.items()}
        c["fetches"] = {n: float(np.mean([x.get("fetches", 0) for x in v])) for n, v in ct.items()}
        c["hostreads"] = {n: c["misses"][n] + c["admits"][n] for n in ct}
        t = c.get("t", {})
        if c["rounds"] and "eq1" in c and all(k in t for k in NAMES):
            c["gap_ms"], c["parts_ms"], c["parts"] = gap_parts(t, c["eq1"])
        c.pop("_pp", None); c.pop("_draws", None)
    return h


def clauses114(V):
    out = []

    def add(cid, short, meas, ok, thr, host, ci=None):
        out.append(dict(id=cid, short=short, type="band", measured=None if meas is None else round(float(meas), 4),
                        ci=None if ci is None else [round(float(x), 4) for x in ci], threshold=thr,
                        status=J.status(meas, ok, ci), why="" if ci is not None else "no interval", host=host))
    for h in V:
        nm = f"{h['job']} {h['cpu']}"; c = h["cells"]
        # H1, the manipulation checks: counters only (deterministic given the routing), so no interval
        for C in CELLS:
            cc = c[C]
            if not cc["rounds"]:
                continue
            if "fetch" in cc["misses"]:
                m0, f0 = REF["fetch"][C]
                dev = max(abs(cc["misses"]["fetch"] / m0 - 1), abs(cc["fetches"]["fetch"] / f0 - 1))
                add(f"{h['job']}-H1a-{G[C]}", f"MIN-1R misses and fetches within 1% of jobs 109/112 ({nm}, {G[C]})", dev,
                    lambda v: v <= 0.01, "<= 0.01 (largest relative deviation)", nm)
        lo = c[14]
        if not lo["rounds"]:
            continue
        for n in ("base0", "bypass0"):
            if n in lo["misses"]:
                m0, a0 = REF[n][14]
                dev = max(abs(lo["misses"][n] / m0 - 1), abs(lo["admits"][n] / a0 - 1))
                add(f"{h['job']}-H1b-{n}", f"{n}: no in-step fetch; misses and admissions within 1% of job 113's at 11% ({nm})",
                    dev if lo["fetches"][n] == 0 else 9.99, lambda v: v <= 0.01, "<= 0.01 and no fetch", nm)
        if "foa0" in lo["fetches"] and "foa" in lo["fetches"]:
            add(f"{h['job']}-H1c", f"foa0's in-step fetches <= 0.5x foa's at 11% ({nm})", lo["fetches"]["foa0"] / lo["fetches"]["foa"],
                lambda v: v <= 0.5, "<= 0.5", nm)
        r, ci = lo["rel"], lo["rel_ci"]
        if "bypass0/bypass" in r:
            add(f"{h['job']}-H2", f"bypass0/bypass >= 1.03 at 11% ({nm})", r["bypass0/bypass"], lambda v: v >= 1.03, ">= 1.03", nm,
                ci.get("bypass0/bypass"))
        for C, thr in ((14, 1.05), (32, 1.15)):
            rc, cic = c[C]["rel"], c[C]["rel_ci"]
            if c[C]["rounds"] and "bypass0/base0" in rc:
                add(f"{h['job']}-H3-{G[C]}", f"bypass0/base0 >= {thr:.2f} at {G[C]} ({nm})", rc["bypass0/base0"],
                    lambda v, thr=thr: v >= thr, f">= {thr:.2f}", nm, cic.get("bypass0/base0"))
        if "foa0/base0" in r:
            add(f"{h['job']}-H4", f"foa0/base0 within [0.95, 1.10] at 11% ({nm})", r["foa0/base0"], lambda v: 0.95 <= v <= 1.10,
                "0.95 to 1.10", nm, ci.get("foa0/base0"))
        if "inter_ms" in lo:
            add(f"{h['job']}-H5", f"interaction > 0 on the CPU-only path at 11% ({nm}; ms per token)", lo["inter_ms"],
                lambda v: v > 0, "> 0", nm, lo.get("inter_ms_ci"))
        if "base0/base" in r:
            add(f"{h['job']}-H6", f"base0/base within [0.92, 1.10] at 11% ({nm})", r["base0/base"], lambda v: 0.92 <= v <= 1.10,
                "0.92 to 1.10", nm, ci.get("base0/base"))
    return out


def macros(H, V, cl):
    M = {}

    def rng(k, vals, fmt="{:.2f}"):
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if vals:
            a, b = (fmt.format(x) for x in (min(vals), max(vals)))
            a, b = ("0" if v in ("-0", "-0.0", "-0.00") else v for v in (a, b))   # no negative zero
            a, b = (v.replace("-", "$-$") for v in (a, b))
            M[k + "Min"], M[k + "Max"] = a, b
            M[k + "Rng"] = a if a == b else f"{a}--{b}"
    pct = lambda x: str(int(round(100 * x))).replace("-", "$-$")  # noqa: E731
    M["dzStarted"] = str(len(H)); M["dzStartedWord"] = word(len(H)); M["dzN"] = str(len(V)); M["dzNWord"] = word(len(V))
    M["dzGated"] = word(sum(1 for h in H if h["gate"])); M["dzGatedNum"] = str(sum(1 for h in H if h["gate"]))
    M["dzKnown"] = word(sum(1 for h in V if h["known"]))
    # machines that passed the gates but failed the registered round check (V2): not scored
    import re
    RF = [h for h in H if not h["gate"] and h["V2_fail"]]
    M["dzRoundFailN"] = word(len(RF)); M["dzRoundFailCpu"] = ", ".join(h["cpu"] for h in RF) or "none"
    sp = [float(m.group(1)) for h in RF for m in [re.search(r"spread ([0-9.]+)%", h["V2_line"])] if m]
    M["dzRoundFailSpread"] = f"{max(sp):.1f}" if sp else "--"
    rng("dzRatio", [h["ratio"] for h in V])
    T = [h for h in V if h["has_table"]]
    M["dzTabN"] = str(len(T)); M["dzTabNWord"] = word(len(T))
    Z = [h for h in V if h["has_table"] is False]
    M["dzNoTabN"] = str(len(Z)); M["dzNoTabNWord"] = word(len(Z))
    M["dzNoTabCpu"] = ", ".join(h["cpu"] for h in Z) or "none"
    if Z:   # the failed checks of the fetches' removal on the machine with an all-zero table
        c = Z[0]["cells"][14]
        if c["fetches"].get("foa"):
            M["dzNoTabFetchRatio"] = f"{c['fetches']['foa0'] / c['fetches']['foa']:.2f}"
        M["dzNoTabGreedyHeld"] = f"{c['rel'].get('bypass0/bypass', float('nan')):.2f}"
        if "parts" in c:
            M["dzShInteractionZeroNoTabLow"] = f"{100 * c['parts']['interaction0']:.0f}"
    for C in CELLS:
        nm = NM[C]
        selt = [h["cells"][C] for h in T if h["cells"][C]["rounds"]]
        for k, key in (("GreedyHeld", "bypass0/bypass"), ("BaseZero", "base0/base"), ("OnceHeld", "foa0/foa"),
                       ("Greedy", "bypass/base"), ("Once", "foa/base")):
            rng(f"dz{k}Tab{nm}", [c["rel"].get(key) for c in selt])
        rng(f"dzMissGreedyTab{nm}", [c["misses"].get("bypass") for c in selt], "{:.1f}")
        for k in ("interaction", "interaction0", "setalone", "setalone0", "together", "together0"):
            K = (k[0].upper() + k[1:]).replace("0", "Zero")
            rng(f"dzSh{K}Tab{nm}", [100 * c["parts"][k] for c in selt if "parts" in c], "{:.0f}")
        rng(f"dzFetchOnceTab{nm}", [c["fetches"].get("foa") for c in selt], "{:.1f}")
    for C in CELLS:
        nm = NM[C]
        sel = [h["cells"][C] for h in V if h["cells"][C]["rounds"]]
        M[f"dzN{nm}"] = str(len(sel))
        for k, key in (("GreedyZero", "bypass0/base0"), ("GreedyHeld", "bypass0/bypass"), ("OnceZero", "foa0/base0"),
                       ("BaseZero", "base0/base"), ("TogetherZero", "fetch/base0"), ("Greedy", "bypass/base"),
                       ("Once", "foa/base"), ("Together", "fetch/base"), ("Ahead", "both3p/base"), ("OnceHeld", "foa0/foa"),
                       ("GreedyZeroVsBase", "bypass0/base"), ("OnceZeroVsBase", "foa0/base")):
            rng(f"dz{k}{nm}", [c["rel"].get(key) for c in sel])
        for k, n in (("Base", "base"), ("BaseZero", "base0"), ("Once", "foa"), ("OnceZero", "foa0"), ("Greedy", "bypass"),
                     ("GreedyZero", "bypass0"), ("Together", "fetch"), ("Ahead", "both3p")):
            rng(f"dzMiss{k}{nm}", [c["misses"].get(n) for c in sel], "{:.1f}")
            rng(f"dzAdm{k}{nm}", [c["admits"].get(n) for c in sel], "{:.1f}")
            rng(f"dzFetch{k}{nm}", [c["fetches"].get(n) for c in sel], "{:.1f}")
        rng(f"dzInterMs{nm}", [c.get("inter_ms") for c in sel], "{:.1f}")
        M[f"dzInterPos{nm}"] = word(sum(1 for c in sel if c.get("inter_ms", 0) > 0))
        M[f"dzInterPosCI{nm}"] = word(sum(1 for c in sel if c.get("inter_ms_ci") and c["inter_ms_ci"][0] > 0))
        # shares of the gap, per host (range) and their mean with a t-interval over hosts
        parts = [c["parts"] for c in sel if "parts" in c]
        for k in ("table", "setalone0", "oncealone0", "together0", "interaction0", "ahead", "left", "setalone", "oncealone",
                  "together", "interaction"):
            v = [p[k] for p in parts]
            if not v:
                continue
            K = k[0].upper() + k[1:]
            K = K.replace("0", "Zero")
            # per-host range here; the mean over hosts with its interval is Table 4's (scripts/decomp_measured.py, dmCpu*)
            M[f"dzSh{K}{nm}Min"], M[f"dzSh{K}{nm}Max"] = pct(min(v)), pct(max(v))
            M[f"dzSh{K}{nm}Rng"] = M[f"dzSh{K}{nm}Min"] if M[f"dzSh{K}{nm}Min"] == M[f"dzSh{K}{nm}Max"] else f"{M[f'dzSh{K}{nm}Min']}--{M[f'dzSh{K}{nm}Max']}"
        # the share of "together" (on the CPU-only path) that MIN's set alone, read twice, gets
        rng(f"dzSetShareOfTogether{nm}", [100 * c["parts_ms"]["setalone0"] / c["parts_ms"]["together0"] for c in sel
                                         if "parts_ms" in c and c["parts_ms"]["together0"] > 0], "{:.0f}")
    pcl = [c for c in cl if not c["id"].endswith("-valid")]   # the predictions; the validity condition apart
    M["dzClauses"] = str(len(pcl)); M["dzClausesHeld"] = str(sum(c["status"] == "held" for c in pcl))
    M["dzClausesPoint"] = str(sum(c["status"] == "held (point)" for c in pcl)); M["dzClausesFailed"] = str(sum(c["status"] == "failed" for c in pcl))
    for t, w in (("One", "H1"), ("Two", "H2"), ("Three", "H3"), ("Four", "H4"), ("Five", "H5"), ("Six", "H6")):
        cs = [c for c in cl if f"-{w}" in c["id"] and c["host"] != "pooled"]
        if cs:
            M[f"dzH{t}N"] = str(len(cs)); M[f"dzH{t}Failed"] = str(sum(c["status"] == "failed" for c in cs))
            M[f"dzH{t}FailedWord"] = word(sum(c["status"] == "failed" for c in cs))
            M[f"dzH{t}FailedOn"] = ", ".join(sorted({c["host"].split(" ", 1)[1] for c in cs if c["status"] == "failed"})) or "none"
            M[f"dzH{t}Status"] = "failed" if any(c["status"] == "failed" for c in cs) else (
                "held" if all(c["status"] == "held" for c in cs) else "held (point)")
    return M


def table(V):
    """per valid host, at 11%: each change's speed on both read paths, and the interaction in ms"""
    hw = lambda c, k: (c["rel_ci"][k][1] - c["rel_ci"][k][0]) / 2 if k in c["rel_ci"] else None  # noqa: E731
    with open(P("paper", "tab_job114.tex"), "w") as f:
        f.write("% generated by scripts/job114.py\n\\begin{table*}[t]\\centering\\footnotesize\n")
        f.write("\\caption{The 2$\\times$2 on both read paths, registered (gpt-oss 11\\%). Speeds relative to the deployed "
                "cache on the same machine and read path (\\emph{fetch table}: deployed; \\emph{CPU only}: in-step fetches off); "
                "\\MinOne{} ignores the table and is given against the CPU-only deployed cache. Subscripts: half-widths of 95\\% "
                "intervals over problems. Interaction: \\MinTwo's plus \\DepOne's time minus the deployed cache's and \\MinOne's "
                "(CPU only, ms per token, 95\\% interval). $^\\ddagger$An all-zero fetch table: no in-step fetches on either "
                "path.}\\label{tab:job114}\n")
        f.write("\\setlength\\tabcolsep{4.5pt}\n\\begin{tabular}{@{}lrrrrrrrr@{}}\\toprule\n")
        f.write(" & & & \\multicolumn{2}{c}{\\MinTwo} & \\multicolumn{2}{c}{\\DepOne} & & \\\\\n")
        f.write("\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n")
        f.write(" & & CPU only & fetch & CPU & fetch & CPU & & Interaction \\\\\n")
        f.write("Machine & Ratio & vs table & table & only & table & only & \\MinOne & (ms) \\\\\\midrule\n")
        for h in V:
            c = h["cells"][14]
            if not c["rounds"]:
                continue
            r = c["rel"]

            def cell(k):
                if k not in r:
                    return "--"
                w = hw(c, k)
                return f"{r[k]:.2f}" + (f"$_{{\\pm{w:.2f}}}$".replace("0.", ".", 1) if w is not None else "")
            im = c.get("inter_ms"); ic = c.get("inter_ms_ci")
            inter = "--" if im is None else (f"{im:.1f}".replace("-", "$-$") + (f" [{ic[0]:.1f}, {ic[1]:.1f}]".replace("-", "$-$") if ic else ""))
            mark = "$^\\ddagger$" if h["has_table"] is False else ""
            f.write(f"{h['cpu']}{mark} & {h['ratio']:.2f} & {cell('base0/base')} & {cell('bypass/base')} & {cell('bypass0/base0')} & "
                    f"{cell('foa/base')} & {cell('foa0/base0')} & {cell('fetch/base0')} & {inter} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table*}\n")
    from scripts.tabnote import split_caption
    split_caption(P("paper", "tab_job114.tex"))


def main():
    H = [load114(d) for d in sorted(glob.glob(GLOB))]
    V = [h for h in H if h["valid"]]
    for h in H:
        print(f"{h['job']} {h['cpu'][:24]:24s} known={h['known']} ratio {h.get('ratio', float('nan')):.2f} valid {h['valid']} "
              f"gate '{h['gate']}' {h['V2_line']} V3 {h['V3_fail']}")
        for C, c in h["cells"].items():
            if c.get("rounds"):
                print(f"   C{C} rounds {c['rounds']} " + " ".join(f"{k} {v:.3f}" for k, v in sorted(c["rel"].items())))
                for q in ("misses", "admits", "fetches"):
                    print(f"      {q} " + " ".join(f"{k} {v:.1f}" for k, v in sorted(c[q].items())))
                if "parts" in c:
                    print("      shares " + " ".join(f"{k} {100 * v:.0f}" for k, v in c["parts"].items()))
                    print(f"      interaction {c.get('inter_ms')} {c.get('inter_ms_ci')} gap {c['gap_ms']:.2f} ms eq1 {c['eq1']:.2f}")
    cl = clauses114(V)
    cl.append(dict(id="114-valid", short="at least two valid hosts (else single-machine results)", type="condition",
                   measured=len(V), ci=None, threshold=">= 2", status="met" if len(V) >= 2 else "not met", why="", host="pooled"))
    for c in cl:
        print(c["id"], c["measured"], c["ci"], c["threshold"], c["status"])
    json.dump(dict(hosts=H, valid=[h["job"] for h in V]), open(P("prereg", "job114.json"), "w"), indent=1, default=float)
    json.dump(dict(job="114", script="jobs/114_decomp0@vast.sh", commit="cf0112c", scored_by="scripts/job114.py (machine)",
                   clauses=cl), open(P("prereg", "scorecard_114.json"), "w"), indent=1)
    M = macros(H, V, cl)
    with open(P("paper", "wsg_job114.tex"), "w") as f:
        f.write("% generated by scripts/job114.py from job 114\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    table(V)
    for k in sorted(M):
        print(k, M[k])


if __name__ == "__main__":
    main()

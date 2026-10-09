"""Jobs 114 and 115 (jobs/114_decomp0@vast.sh and jobs/115_moremachines@vast.sh on the gpu branch): Table 4's 2x2 on
the CPU-only read path (the in-step fetches off), beside Table 4's own cells, in one process per round; job 115 adds
three idle probe runs after the download. Scores each job's registered predictions per valid host from the raw rows and
counters (job 114: H1-H6; job 115: H1-H8, its own thresholds), computes the decomposition of each host's gap on the
CPU-only path, and writes prereg/job114.json (both jobs' hosts), prereg/scorecard_114.json, prereg/scorecard_115.json,
paper/wsg_job114.tex (macros dz* for job 114, dq* for job 115, dp* pooled over both jobs' valid machines) and
paper/tab_job114.tex.

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
GLOB = os.environ.get("MOSL_JOB114_GLOB", f"{J.RES}/11[45][a-z]_*@vast")
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
    h["jobno"] = h["job"][:3]
    # job 115: the first probe's highest reading against the three idle runs' (probe_compare.txt), and the fetch table
    # the best idle run would give (recorded only)
    h["probe1_dl"] = "download running: yes" in J.read(d, "probe1_when.txt")
    pc = J.read(d, "probe_compare.txt")
    m = re.search(r"first probe highest ([\d.]+) GB/s; idle runs ([\d. ]+); idle best / first ([\d.]+); idle spread ([\d.]+)", pc)
    if m:
        h["probe"] = dict(first=float(m.group(1)), idle=[float(x) for x in m.group(2).split()], ratio=float(m.group(3)),
                          spread=float(m.group(4)))
        mi = re.search(r"idle law table \(from [^)]*\): gpt-oss ([0-9,]+)", J.read(d, "tables.txt"))
        h["probe"]["idle_table"] = mi.group(1) if mi else ""
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
            c["inter_names"] = ("bypass0", "foa0", "base0", "fetch")
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


TH = {"114": dict(ref1="jobs 109/112", ref="job 113's", h1c_all=True, h2=1.03, h2_all=True, h3=(1.05, 1.15), h4=(0.95, 1.10), h6=(0.92, 1.10), h7=False),
      "115": dict(ref1="jobs 109-114", ref="job 114's", h1c_all=False, h2=1.05, h2_all=False, h3=(1.07, 1.20), h4=(0.98, 1.10), h6=(0.92, 1.10), h7=True)}


def clauses114(V, job="114"):
    out = []
    th = TH[job]

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
                add(f"{h['job']}-H1a-{G[C]}", f"MIN-1R misses and fetches within 1% of {th['ref1']} ({nm}, {G[C]})", dev,
                    lambda v: v <= 0.01, "<= 0.01 (largest relative deviation)", nm)
        lo = c[14]
        if not lo["rounds"]:
            continue
        for n in ("base0", "bypass0"):
            if n in lo["misses"]:
                m0, a0 = REF[n][14]
                dev = max(abs(lo["misses"][n] / m0 - 1), abs(lo["admits"][n] / a0 - 1))
                add(f"{h['job']}-H1b-{n}", f"{n}: no in-step fetch; misses and admissions within 1% of {th['ref']} at 11% ({nm})",
                    dev if lo["fetches"][n] == 0 else 9.99, lambda v: v <= 0.01, "<= 0.01 and no fetch", nm)
        if "foa0" in lo["fetches"] and "foa" in lo["fetches"] and (th["h1c_all"] or h["has_table"]):
            add(f"{h['job']}-H1c", f"foa0's in-step fetches <= 0.5x foa's at 11% ({nm})", lo["fetches"]["foa0"] / lo["fetches"]["foa"],
                lambda v: v <= 0.5, "<= 0.5", nm)
        r, ci = lo["rel"], lo["rel_ci"]
        if "bypass0/bypass" in r and (th["h2_all"] or h["has_table"]):
            t2 = th["h2"]
            add(f"{h['job']}-H2", f"bypass0/bypass >= {t2:.2f} at 11% ({nm})", r["bypass0/bypass"], lambda v: v >= t2, f">= {t2:.2f}", nm,
                ci.get("bypass0/bypass"))
        for C, thr in zip((14, 32), th["h3"]):
            rc, cic = c[C]["rel"], c[C]["rel_ci"]
            if c[C]["rounds"] and "bypass0/base0" in rc:
                add(f"{h['job']}-H3-{G[C]}", f"bypass0/base0 >= {thr:.2f} at {G[C]} ({nm})", rc["bypass0/base0"],
                    lambda v, thr=thr: v >= thr, f">= {thr:.2f}", nm, cic.get("bypass0/base0"))
        if "foa0/base0" in r:
            a4, b4 = th["h4"]
            add(f"{h['job']}-H4", f"foa0/base0 within [{a4:.2f}, {b4:.2f}] at 11% ({nm})", r["foa0/base0"], lambda v: a4 <= v <= b4,
                f"{a4:.2f} to {b4:.2f}", nm, ci.get("foa0/base0"))
        if "inter_ms" in lo:
            add(f"{h['job']}-H5", f"interaction > 0 on the CPU-only path at 11% ({nm}; ms per token)", lo["inter_ms"],
                lambda v: v > 0, "> 0", nm, lo.get("inter_ms_ci"))
        if "base0/base" in r:
            a6, b6 = th["h6"]
            add(f"{h['job']}-H6", f"base0/base within [{a6:.2f}, {b6:.2f}] at 11% ({nm})", r["base0/base"], lambda v: a6 <= v <= b6,
                f"{a6:.2f} to {b6:.2f}", nm, ci.get("base0/base"))
        if th["h7"] and h.get("probe"):
            pr = h["probe"]
            add(f"{h['job']}-H7a", f"best idle probe reading within [0.95, 1.15] of the first probe's ({nm})", pr["ratio"],
                lambda v: 0.95 <= v <= 1.15, "0.95 to 1.15", nm)
            add(f"{h['job']}-H7b", f"the three idle runs' highest readings within 5% of each other ({nm})", pr["spread"],
                lambda v: v <= 1.05, "<= 1.05", nm)
    return out


def clause_pooled(Vall):
    """job 115's H8: MIN's set alone (CPU-only path, read twice) closes a mean share of the gap at 11% of at least 0.10
    over the valid machines of jobs 114 and 115, with its 95% t-interval over machines above zero"""
    from scipy.stats import t as tdist
    v = [h["cells"][14]["parts"]["setalone0"] for h in Vall if "parts" in h["cells"][14]]
    if len(v) < 2:
        return []
    m = float(np.mean(v)); hw = float(tdist.ppf(0.975, len(v) - 1) * np.std(v, ddof=1) / math.sqrt(len(v)))
    ci = (m - hw, m + hw)   # the t-interval over machines, as job 109's pooled clauses
    return [dict(id="115-H8a", short=f"mean share of the gap closed by MIN's set alone, CPU only, 11%, over {len(v)} machines >= 0.10",
                 type="band", measured=round(m, 4), ci=[round(x, 4) for x in ci], threshold=">= 0.10",
                 status=J.status(m, lambda x: x >= 0.10, ci), why="pooled over machines (t-interval)", host="pooled"),
            dict(id="115-H8b", short="its 95% t-interval over machines above zero (lower end)", type="band", measured=round(m - hw, 4),
                 ci=None, threshold="> 0", status="held (point)" if m - hw > 0 else "failed", why="pooled over machines", host="pooled")]


def macros(H, V, cl, pre="dz"):
    """the macros of one set of machines, written with the prefix pre (dz job 114, dq job 115, dp both jobs' valid ones)"""
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
    # the gate failures by gate: V0 (a GPU rented before) and VR (link/CPU ratio below 0.5, with the ratios)
    import re
    vz = [h for h in H if h["gate"].startswith("V0 FAIL")]
    vr = [h for h in H if h["gate"].startswith("VR FAIL")]
    M["dzVZeroFailN"] = word(len(vz)); M["dzVRFailN"] = word(len(vr))
    M["dzVRFailCpu"] = ", ".join(sorted({h["cpu"] for h in vr})) or "none"
    # the same as a phrase with counts and each model's ratios: "two Ryzen Threadripper 3970Xs (0.39) and an EPYC 7352 (0.26)"
    from collections import Counter
    cnt = Counter(h["cpu"] for h in vr)
    parts = []
    for cpu, n in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
        rs = sorted(float(m.group(1)) for h in vr if h["cpu"] == cpu for m in [re.search(r"ratio ([0-9.]+)", h["gate"])] if m)
        rr = f"{rs[0]:.2f}" if f"{rs[0]:.2f}" == f"{rs[-1]:.2f}" else f"{rs[0]:.2f}--{rs[-1]:.2f}"
        art = ("an" if cpu[0] in "AEIOU" else "a") if n == 1 else word(n)
        parts.append(f"{art} {cpu}{'s' if n > 1 else ''} ({rr})")
    M["dzVRFailList"] = " and ".join([", ".join(parts[:-1]), parts[-1]] if len(parts) > 1 else parts) or "none"
    rng("dzVRFailRatio", [float(m.group(1)) for h in vr for m in [re.search(r"ratio ([0-9.]+)", h["gate"])] if m])
    # machines that passed the gates but failed the registered round check (V2): not scored
    RF = [h for h in H if not h["gate"] and h["V2_fail"]]
    M["dzRoundFailN"] = word(len(RF)); M["dzRoundFailCpu"] = ", ".join(h["cpu"] for h in RF) or "none"
    sp = [float(m.group(1)) for h in RF for m in [re.search(r"spread ([0-9.]+)%", h["V2_line"])] if m]
    M["dzRoundFailSpread"] = f"{max(sp):.1f}" if sp else "--"
    # its interaction in each round at 11% (not scored), reported as the excluded machines of other jobs are
    ir = [x for h in RF for x in h["cells"][14].get("inter_rounds", [])]
    if ir:
        M["dzRoundFailInterMin"] = f"{min(ir):.1f}".replace("-", "$-$"); M["dzRoundFailInterMax"] = f"{max(ir):.1f}".replace("-", "$-$")
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
        rng("dzShInteractionZeroNoTabLow", [100 * h["cells"][14]["parts"]["interaction0"] for h in Z if "parts" in h["cells"][14]], "{:.0f}")
        rng("dzShInteractionNoTabLow", [100 * h["cells"][14]["parts"]["interaction"] for h in Z if "parts" in h["cells"][14]], "{:.0f}")
        M["dzNoTabCpuUniq"] = ", ".join(sorted({h["cpu"] for h in Z}))
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
    # post hoc: the interaction at 11% on the CPU-only path by the link-to-CPU ratio (split at 0.7), and the machines
    # where it was not above zero
    for tag, sel in (("HiRatio", [h for h in V if h["ratio"] >= 0.7]), ("LoRatio", [h for h in V if h["ratio"] < 0.7])):
        cs = [h["cells"][14] for h in sel if "parts" in h["cells"][14]]
        M[f"dz{tag}N"] = word(len(cs))
        rng(f"dzRatio{tag}", [h["ratio"] for h in sel])
        rng(f"dzInterMs{tag}", [c.get("inter_ms") for c in cs], "{:.1f}")
        rng(f"dzShInteractionZero{tag}", [100 * c["parts"]["interaction0"] for c in cs], "{:.0f}")
    NP = [h for h in V if h["cells"][14].get("inter_ms") is not None and h["cells"][14]["inter_ms"] <= 0]
    M["dzInterNegN"] = word(len(NP)); M["dzInterNegCpu"] = ", ".join(h["cpu"] for h in NP) or "none"
    rng("dzInterNegMs", [h["cells"][14]["inter_ms"] for h in NP], "{:.1f}")
    for h in NP:
        lo, hi = h["cells"][14]["inter_ms_ci"]
        M["dzInterNegCI"] = f"[{lo:.1f}, {hi:.1f}]".replace("-", "$-$")
        # its two rounds, and how far the fetch-free deployed cache's speed moved between them
        ir = h["cells"][14].get("inter_rounds", [])
        M["dzInterNegRounds"] = " and ".join(f"{x:+.1f}".replace("-", "$-$") for x in ir)
        rr = h["cells"][14].get("ratio_rounds", {}).get("base0")
        if rr and len(rr) == 2:
            M["dzInterNegBaseZeroMove"] = f"{100 * abs(rr[1] / rr[0] - 1):.0f}"
    pcl = [c for c in cl if not c["id"].endswith("-valid")]   # the predictions; the validity condition apart
    M["dzClauses"] = str(len(pcl)); M["dzClausesHeld"] = str(sum(c["status"] == "held" for c in pcl))
    M["dzClausesPoint"] = str(sum(c["status"] == "held (point)" for c in pcl)); M["dzClausesFailed"] = str(sum(c["status"] == "failed" for c in pcl))
    M["dzVerdict"] = "mixed" if any(c["status"] == "failed" for c in pcl) else "held"
    vc = [c for c in cl if c["id"].endswith("-valid")]
    if vc:
        M["dzValidStatus"] = vc[0]["status"]
    # the probe (job 115): the best idle run over the first probe, the idle runs' spread, how often the idle table differs
    PR = [h for h in V if h.get("probe")]
    if PR:
        rng("dzProbeIdleRatio", [h["probe"]["ratio"] for h in PR], "{:.3f}")
        rng("dzProbeIdleSpread", [h["probe"]["spread"] for h in PR], "{:.3f}")
        M["dzProbeN"] = word(len(PR))
        M["dzProbeDuringDlN"] = word(sum(1 for h in PR if h["probe1_dl"]))
        M["dzProbeIdleDevMax"] = f"{100 * max(abs(h['probe']['ratio'] - 1) for h in PR):.1f}"
        M["dzProbeIdleSpreadPctMax"] = f"{100 * max(h['probe']['spread'] - 1 for h in PR):.1f}"
        M["dzProbeTableChanged"] = word(sum(1 for h in PR if h["probe"].get("idle_table") and h["probe"]["idle_table"] != h["table"]))
    # job 115's H8 (pooled over both jobs' valid machines): the mean share and its t-interval's lower end, in percent
    for c in cl:
        if c["id"] == "115-H8a":
            M["dzHEightMean"] = pct(c["measured"]); M["dzHEightNum"] = re.search(r"over (\d+) machines", c["short"]).group(1)
            M["dzHEightN"] = word(int(M["dzHEightNum"]))
        if c["id"] == "115-H8b":
            M["dzHEightLo"] = pct(c["measured"])
    for t, w in (("One", "H1"), ("Two", "H2"), ("Three", "H3"), ("Four", "H4"), ("Five", "H5"), ("Six", "H6"), ("Seven", "H7"), ("Eight", "H8")):
        cs = [c for c in cl if f"-{w}" in c["id"] and c["host"] != "pooled"]
        if cs:
            M[f"dzH{t}N"] = str(len(cs)); M[f"dzH{t}Failed"] = str(sum(c["status"] == "failed" for c in cs))
            M[f"dzH{t}FailedWord"] = word(sum(c["status"] == "failed" for c in cs))
            M[f"dzH{t}FailedOn"] = ", ".join(sorted({c["host"].split(" ", 1)[1] for c in cs if c["status"] == "failed"})) or "none"
            M[f"dzH{t}Status"] = "failed" if any(c["status"] == "failed" for c in cs) else (
                "held" if all(c["status"] == "held" for c in cs) else "held (point)")
    return {pre + k[2:]: v for k, v in M.items()}


def table(V):
    """per valid host, at 11%: each change's speed on both read paths, and the interaction in ms; grouped by test"""
    hw = lambda c, k: (c["rel_ci"][k][1] - c["rel_ci"][k][0]) / 2 if k in c["rel_ci"] else None  # noqa: E731
    with open(P("paper", "tab_job114.tex"), "w") as f:
        f.write("% generated by scripts/job114.py\n\\begin{table*}[t]\\centering\\footnotesize\n")
        f.write("\\caption{The 2$\\times$2 with the fetch table on and with the in-step fetches off, registered (gpt-oss 11\\%). "
                "Speeds relative to the deployed cache on the same machine and read path (\\emph{on}: the deployed read path, "
                "with the fetch table; \\emph{off}: the in-step fetches off). \\MinOne's forced plans replace the table, so it "
                "is the same on both paths; its column is against the deployed cache with the fetches off. \\emph{Dep.}: the deployed cache with its fetch table (on); Eq.~(1): the bound. Subscripts: "
                "half-widths of 95\\% intervals over problems. Interaction: \\MinTwo's plus \\DepOne's time minus the deployed "
                "cache's and \\MinOne's, fetches off, in ms per token, with its 95\\% interval. $^\\ddagger$An all-zero fetch table: "
                "no in-step fetches on either path. $^\\S$The second test.}\\label{tab:job114}\n")
        f.write("\\setlength\\tabcolsep{4.5pt}\n\\begin{tabular}{@{}lrrrrrrrrrr@{}}\\toprule\n")
        f.write(" & & \\multicolumn{2}{c}{ms per token} & \\multicolumn{1}{c}{Deployed} & \\multicolumn{2}{c}{\\MinTwo} & \\multicolumn{2}{c}{\\DepOne} & & \\\\\n")
        f.write("\\cmidrule(lr){3-4}\\cmidrule(lr){5-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n")
        f.write("Machine & Ratio & Dep. & Eq.\\,(1) & off\\,/\\,on & on & off & on & off & \\MinOne & Interaction (ms) \\\\\\midrule\n")
        groups = (("114", "First test"), ("115", "Second test (three idle probe runs; job 109's machines not admitted)"))
        for job, lab in groups:
            VV = [h for h in V if h["jobno"] == job and h["cells"][14]["rounds"]]
            if not VV:
                continue
            for h in VV:
                c = h["cells"][14]
                r = c["rel"]

                def cell(k):
                    if k not in r:
                        return "--"
                    w = hw(c, k)
                    return f"{r[k]:.2f}" + (f"$_{{\\pm{w:.2f}}}$".replace("0.", ".", 1) if w is not None else "")
                im = c.get("inter_ms"); ic = c.get("inter_ms_ci")
                inter = "--" if im is None else (f"{im:.1f}".replace("-", "$-$") + (f" [{ic[0]:.1f}, {ic[1]:.1f}]".replace("-", "$-$") if ic else ""))
                mark = ("$^\\ddagger$" if h["has_table"] is False else "") + ("$^\\S$" if job == "115" else "")
                f.write(f"{h['cpu']}{mark} & {h['ratio']:.2f} & {c['t']['base']:.1f} & {c['eq1']:.1f} & {cell('base0/base')} & {cell('bypass/base')} & {cell('bypass0/base0')} & "
                        f"{cell('foa/base')} & {cell('foa0/base0')} & {cell('fetch/base0')} & {inter} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table*}\n")
    from scripts.tabnote import split_caption
    split_caption(P("paper", "tab_job114.tex"))


def main():
    H = [load114(d) for d in sorted(glob.glob(GLOB))]
    for h in H:
        print(f"{h['job']} {h['cpu'][:24]:24s} known={h['known']} ratio {h.get('ratio', float('nan')):.2f} valid {h['valid']} "
              f"gate '{h['gate']}' {h['V2_line']} V3 {h['V3_fail']} table {h['table']} probe {h.get('probe')}")
        for C, c in h["cells"].items():
            if c.get("rounds"):
                print(f"   C{C} rounds {c['rounds']} " + " ".join(f"{k} {v:.3f}" for k, v in sorted(c["rel"].items())))
                for q in ("misses", "admits", "fetches"):
                    print(f"      {q} " + " ".join(f"{k} {v:.1f}" for k, v in sorted(c[q].items())))
                if "parts" in c:
                    print("      shares " + " ".join(f"{k} {100 * v:.0f}" for k, v in c["parts"].items()))
                    print(f"      interaction {c.get('inter_ms')} {c.get('inter_ms_ci')} gap {c['gap_ms']:.2f} ms eq1 {c['eq1']:.2f}")
    Vall = [h for h in H if h["valid"]]
    M = {}
    for job, pre, commit, script in (("114", "dz", "cf0112c", "jobs/114_decomp0@vast.sh"), ("115", "dq", "4b69328", "jobs/115_moremachines@vast.sh")):
        Hj = [h for h in H if h["jobno"] == job]
        if not Hj:
            continue
        Vj = [h for h in Hj if h["valid"]]
        cl = clauses114(Vj, job) + (clause_pooled(Vall) if job == "115" else [])
        cl.append(dict(id=f"{job}-valid", short="at least two valid hosts (else single-machine results)", type="condition",
                       measured=len(Vj), ci=None, threshold=">= 2", status="met" if len(Vj) >= 2 else "not met", why="", host="pooled"))
        for c in cl:
            print(c["id"], c["measured"], c["ci"], c["threshold"], c["status"])
        json.dump(dict(job=job, script=script, commit=commit, scored_by="scripts/job114.py (machine)", clauses=cl),
                  open(P("prereg", f"scorecard_{job}.json"), "w"), indent=1)
        M.update(macros(Hj, Vj, cl, pre))
    # pooled over both jobs' valid machines: the paper's CPU-only results (no clauses of their own)
    clall = [c for job in ("114", "115") if os.path.exists(P("prereg", f"scorecard_{job}.json"))
             for c in json.load(open(P("prereg", f"scorecard_{job}.json")))["clauses"]]
    M.update(macros(H, Vall, clall, "dp"))
    json.dump(dict(hosts=H, valid=[h["job"] for h in Vall]), open(P("prereg", "job114.json"), "w"), indent=1, default=float)
    with open(P("paper", "wsg_job114.tex"), "w") as f:
        f.write("% generated by scripts/job114.py from jobs 114 and 115\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    table(Vall)
    for k in sorted(M):
        print(k, M[k])


if __name__ == "__main__":
    main()

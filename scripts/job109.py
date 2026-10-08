"""Job 109: an online policy built from the engine's own mechanisms (read once and admit by the layer-ahead
prediction, R1; the same admitting less, R2) against the engine's oracles (the 2x2 of MIN's set and one read, and MIN
prefetched), on desktop-class machines never rented before whose link-to-CPU ratio is at least 0.5 (the population
fixed before launch). Scores the predictions in the header of jobs/109_onlinepolicy@vast.sh and writes prereg/job109.json,
paper/wsg_job109.tex and paper/tab_job109.tex.

    python scripts/job109.py            (MOSL_JOB109_GLOB overrides the result directories, e.g. for the smoke run)
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from scripts.speed_limit import host_rates  # noqa: E402

RES = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
GLOB = os.environ.get("MOSL_JOB109_GLOB", f"{RES}/109[a-j]_onlinepolicy@vast")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
S = 13253760
TGPU = 2.9402065        # the smallest profiled T_GPU, ms (scripts/decomp_measured.py), as registered
RSTAR = {14: 38.315364583333334, 32: 15.340364583333333}
LAB = {14: "11\\%", 32: "25\\%"}
REQUIRED, WANTED = 4, 5
A_NAMES = ("base", "foa", "bypass", "fetch", "both3p", "dk", "pf", "fetchplan", "bypassplan", "bypassplanS")
B_NAMES = ("base", "R1", "R2")
REAL = ("dk", "pf", "R1", "R2")
num = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
word = lambda n: num[n] if n < len(num) else str(n)  # noqa: E731


def short_cpu(s):
    s = re.sub(r"\(R\)|\(TM\)|™|®|CPU|Processor|with Radeon Graphics|\d+-Core", "", s or "")
    s = re.sub(r"1[2-4]th Gen\s+", "", s.strip())
    s = re.sub(r"^(AMD|Intel)\s+", "", s.strip())
    return re.sub(r"\s+", " ", s).strip()


def read(d, f):
    p = f"{d}/{f}"
    return open(p).read() if os.path.exists(p) else ""


def rows_by_config(f):
    by = {}
    for line in open(f):
        if line.strip():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:   # a process cut by the job's deadline can leave a partial last line
                continue
            lab = os.path.basename(r["config"].rsplit("stats=", 1)[-1]).replace(".json", "").rsplit("_", 1)[-1]
            by.setdefault(lab, {})[r["seq"]] = (r["decode_ms"] / r["n_decode"], r["nll_sum"] / r["nll_n"])
    return by


def counters(d, C, rd, pr, n):
    p = f"{d}/st_g_C{C}_r{rd}{pr}_{n}.json"
    if not os.path.exists(p):
        return None
    x = json.load(open(p))
    s = max(1, x.get("steps", 1))
    return dict(reads=(x["misses"] + x["admits"] + x.get("prefetches", 0)) / s, misses=x["misses"] / s,
                admits=x["admits"] / s, prefetches=x.get("prefetches", 0) / s,
                useful=x.get("prefetch_useful", 0) / s, fetch_on_admit=x.get("fetch_on_admit"),
                prefetch_q=x.get("prefetch_q"), kappa=x.get("kappa"))


def boot_cell(pp, nb=2000, seed=109):
    """95% intervals over problems (resampled once per draw, paired across configurations and rounds) of each speed
    ratio vs its process's base (geometric mean over rounds), of the best online policy's ratio, and of its capture.
    Vectorised over the draws (the same draws as the loop it replaced)."""
    draws = boot_draws(pp, nb, seed)
    return {n: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for n, v in draws.items() if len(v)}


def boot_draws(pp, nb=2000, seed=109):
    """the bootstrap draws behind boot_cell: {configuration: array of nb speed ratios}, plus best and capture"""
    rng = np.random.default_rng(seed)
    seqs = sorted(set.intersection(*[set(by["base"]) for ba, bb in pp.values() for by in (ba, bb)]))
    if len(seqs) < 2:
        return {}
    arr = {}
    for rd, (ba, bb) in pp.items():
        for by, names in ((ba, A_NAMES), (bb, B_NAMES)):
            base = np.array([by["base"][q][0] for q in seqs])
            for n in names:
                if n != "base" and n in by and all(q in by[n] for q in seqs):
                    arr.setdefault(n, []).append((base, np.array([by[n][q][0] for q in seqs])))
    idx = rng.integers(0, len(seqs), size=(nb, len(seqs)))
    draws = {n: np.exp(np.mean([np.log(b[idx].mean(axis=1) / x[idx].mean(axis=1)) for b, x in v], axis=0))
             for n, v in arr.items()}
    real = [draws[k] for k in REAL if k in draws]
    if real and "both3p" in draws:
        draws["best"] = np.max(real, axis=0)
        draws["capture"] = (draws["best"] - 1) / (draws["both3p"] - 1)
    return draws


def tint(v, level=0.95):
    """mean over machines and its t-interval (the machine is the unit; with 3-15 machines a percentile bootstrap is too
    narrow); None for fewer than two values"""
    from scipy.stats import t as tdist
    v = np.asarray([x for x in v if x is not None], float)
    if len(v) < 2:
        return None
    h = tdist.ppf(0.5 + level / 2, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))
    return float(v.mean() - h), float(v.mean() + h)


def load(d, cells=(14, 32), min_ratio=0.5, need_b=True, keep_pp=False):
    """one host: its gates, its probe, and per budget the within-process speed ratios, times and counters"""
    v0, val = read(d, "v0.txt"), read(d, "validity.txt")
    m = re.search(r"Model name:\s*(.+)", read(d, "cpu.txt"))
    h = dict(dir=os.path.basename(d), job=os.path.basename(d)[:4], cpu=short_cpu(m.group(1)) if m else "")
    m = re.search(r"GPU UUID (\S+); host memory in use (\d+) GB", v0)
    h["uuid"], h["used_gb"] = (m.group(1), int(m.group(2))) if m else (None, None)
    m = re.search(r"usable physical cores (\d+).*NUMA nodes (\S+)", read(d, "cores.txt"))
    h["cores"], h["numa"] = (int(m.group(1)), m.group(2)) if m else (None, None)
    h["gate"] = next((t + " " + v0.split(t, 1)[1].split("\n")[0].strip() for t in ("V0 FAIL:", "V0c FAIL:", "VR FAIL:")
                      if t in v0), "")
    ft = f"{d}/fetch_table_law_gptoss.json"
    if os.path.exists(ft):
        j = json.load(open(ft))
        h["B_p"], h["B_c"] = j["B_p"], j["B_c"]
        h["ratio"] = j["B_p"] / j["B_c"]
    txt = read(d, "concur.txt")
    h["B_host"] = host_rates(txt) / 1e9 if "concurrent t=" in txt or "cpu_read_gbs" in txt else None
    h["V1_fail"] = "V1" in val and "FAIL" in "\n".join(l for l in val.splitlines() if l.startswith("V1"))
    h["V2_fail"] = any(l.startswith("V2") and "FAIL" in l for l in val.splitlines())
    h["V2_line"] = next((l for l in val.splitlines() if l.startswith("V2")), "")
    h["V3_fail"] = sorted({(l.split()[1], l.split()[2].rstrip(":")) for l in val.splitlines()
                           if l.startswith("V3") and l.rstrip().endswith("FAIL")})
    gp = read(d, "g_prof.json")
    h["G_prof"] = {k: v.get("G_prof_ms") for k, v in json.loads(gp).items()} if gp.strip() else {}
    h["cells"] = {}
    for C in cells:
        c = dict(C=C, rounds=[], ratio={}, ratio_rounds={}, t={}, reads={}, counters={}, loss={})
        for rd in (1, 2):
            fa, fb = f"{d}/ec_g_C{C}_r{rd}A.jsonl", f"{d}/ec_g_C{C}_r{rd}B.jsonl"
            if not (os.path.exists(fa) and (os.path.exists(fb) or not need_b)):
                continue
            ba = rows_by_config(fa)
            bb = rows_by_config(fb) if os.path.exists(fb) else {"base": ba.get("base", {})}
            if "base" not in ba or "base" not in bb:
                continue
            c["rounds"].append(rd)
            c.setdefault("_pp", {})[rd] = (ba, bb)
            for by, names in ((ba, A_NAMES), (bb, B_NAMES)):
                base = by["base"]
                for n in names:
                    if n not in by or n == "base":
                        continue
                    seqs = [s for s in by[n] if s in base]
                    r = np.mean([base[s][0] for s in seqs]) / np.mean([by[n][s][0] for s in seqs])
                    c["ratio_rounds"].setdefault(n, []).append(float(r))
            for n in A_NAMES:
                if n in ba:
                    c["t"].setdefault(n, []).append(float(np.mean([x[0] for x in ba[n].values()])))
            c["t"].setdefault("baseB", []).append(float(np.mean([x[0] for x in bb["base"].values()])))
            for pr, names in (("A", A_NAMES), ("B", B_NAMES)):
                for n in names:
                    ct = counters(d, C, rd, pr, n)
                    if ct:
                        c["counters"].setdefault(n if not (pr == "B" and n == "base") else "baseB", []).append(ct)
        for n, v in c["ratio_rounds"].items():
            c["ratio"][n] = float(math.exp(np.mean(np.log(v))))
        if c["rounds"]:
            c["_draws"] = boot_draws(c["_pp"])
            c["ci"] = {n: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for n, v in c["_draws"].items() if len(v)}
            if not keep_pp:
                c.pop("_pp"); c.pop("_draws")
        if c["t"].get("base"):
            c["t_r1_base"] = c["t"]["base"][0]   # process A's base in the first round run
        c["t"] = {n: float(np.mean(v)) for n, v in c["t"].items()}
        c["reads"] = {n: float(np.mean([x["reads"] for x in v])) for n, v in c["counters"].items()}
        if c["rounds"] and h.get("B_host"):
            t = c["t"]
            eq1 = RSTAR.get(C, float('nan')) * S / (h["B_host"] * 1e9) * 1e3
            gap = t["base"] - eq1
            c["eq1"], c["gap"] = eq1, gap
            if all(k in t for k in ("foa", "bypass", "fetch", "both3p")):
                c["together"] = (t["base"] - t["fetch"]) / gap
                c["setalone"] = (t["base"] - t["bypass"]) / gap
                c["oncealone"] = (t["base"] - t["foa"]) / gap
                c["interaction_ms"] = (t["base"] - t["fetch"]) - (t["base"] - t["bypass"]) - (t["base"] - t["foa"])
                c["ahead"] = (t["fetch"] - t["both3p"]) / gap
                extra = (c["reads"]["both3p"] - RSTAR.get(C, float('nan'))) * S / (h["B_host"] * 1e9) * 1e3
                c["extra"] = extra / gap
                c["tgpu"] = TGPU / gap
                c["residual"] = (t["both3p"] - eq1 - TGPU - extra) / gap
            r = c["ratio"]
            avail = [k for k in REAL if k in r]
            if avail and "both3p" in r:
                c["best_name"] = max(avail, key=lambda k: r[k])
                c["best"] = r[c["best_name"]]
                c["capture"] = (c["best"] - 1) / (r["both3p"] - 1)
                c["capture_each"] = {k: (r[k] - 1) / (r["both3p"] - 1) for k in avail}
            if "R1" in r and "R2" in r:
                c["R2_over_R1"] = r["R2"] / r["R1"]
        h["cells"][C] = c
    h["ran"] = bool(h["cells"].get(cells[0], {}).get("rounds"))
    fa = f"{d}/ec_g_C{cells[0]}_r1A.jsonl"
    if os.path.exists(fa):
        b = rows_by_config(fa).get("base", {})
        h["loss_base"] = float(np.mean([x[1] for x in b.values()])) if b else None
    h["valid"] = (not h["gate"]) and h["ran"] and h.get("ratio", 0) >= min_ratio and not h["V1_fail"] and not h["V2_fail"] \
        and len(h["cells"][cells[0]]["rounds"]) == 2
    return h


def predictions(V, cells=(14, 32)):
    """the registered predictions over the valid machines V; returns {name: (passed or None, detail)}"""
    lo, mid = cells
    out = {}
    if len(V) < REQUIRED:
        out["_conclusive"] = (False, f"{len(V)} valid machines, fewer than {REQUIRED}")
    def every(f):
        vals = [f(h) for h in V]
        return all(vals), vals
    c = lambda h, C: h["cells"][C]  # noqa: E731
    r = lambda h, C, n: c(h, C)["ratio"].get(n, float("nan"))  # noqa: E731
    ok1a, v1a = every(lambda h: r(h, lo, "fetch") >= max(r(h, lo, "bypass"), r(h, lo, "foa")) + 0.04)
    ok1b, _ = every(lambda h: all(c(h, C).get("interaction_ms", -1) > 0 for C in cells if c(h, C)["rounds"]))
    out["P1"] = (ok1a and ok1b, dict(margin=[r(h, lo, "fetch") - max(r(h, lo, "bypass"), r(h, lo, "foa")) for h in V],
                                     interaction_ms={C: [c(h, C).get("interaction_ms") for h in V] for C in cells}))
    ok2, _ = every(lambda h: 0.96 <= r(h, lo, "foa") <= 1.05 and 0.96 <= r(h, lo, "bypass") <= 1.05)
    out["P2"] = (ok2, dict(foa=[r(h, lo, "foa") for h in V], bypass=[r(h, lo, "bypass") for h in V]))
    tog = {C: [c(h, C)["together"] for h in V if "together" in c(h, C)] for C in cells}
    mt = {C: float(np.mean(v)) if v else float("nan") for C, v in tog.items()}
    out["P3"] = (0.25 <= mt[lo] <= 0.55 and 0.20 <= mt[mid] <= 0.50, dict(mean=mt, each=tog))
    beat = {C: [r(h, C, "both3p") > r(h, C, "fetch") for h in V if c(h, C)["rounds"]] for C in cells}
    out["P4"] = (all(beat[mid]) and sum(not x for x in beat[lo]) <= 1, beat)
    ok5, v5 = every(lambda h: r(h, mid, "bypass") >= 1.08 if c(h, mid)["rounds"] else True)
    out["P5"] = (ok5, [r(h, mid, "bypass") for h in V])
    res = {C: [c(h, C)["residual"] for h in V if "residual" in c(h, C)] for C in cells}
    mr = {C: float(np.mean(v)) if v else float("nan") for C, v in res.items()}
    out["P6"] = (abs(mr[lo]) <= 0.10 and abs(mr[mid]) <= 0.15, dict(mean=mr, each=res))
    cap = {C: [c(h, C)["capture"] for h in V if "capture" in c(h, C)] for C in cells}
    best = {C: [c(h, C)["best"] for h in V if "best" in c(h, C)] for C in cells}
    ok7 = all(b <= 1.10 for C in cells for b in best[C]) and all(x <= 0.35 for C in cells for x in cap[C]) \
        and (float(np.mean(cap[lo])) <= 0.25 if cap[lo] else False)
    out["P7"] = (ok7, dict(best=best, capture=cap, mean_capture={C: float(np.mean(v)) if v else None for C, v in cap.items()}))
    low = [(h["ratio"], r(h, lo, "pf")) for h in V if h["ratio"] < 0.75]
    high = [(h["ratio"], r(h, lo, "pf")) for h in V if h["ratio"] >= 0.9]
    ok8 = all(p < 1.0 for _, p in low) and all(p >= 0.98 for _, p in high)
    out["P8"] = (ok8 if (low or high) else None, dict(below_075=low, at_least_09=high))
    ok9, _ = every(lambda h: abs(r(h, lo, "R1") - r(h, lo, "pf")) <= 0.04)
    out["P9"] = (ok9, [r(h, lo, "R1") - r(h, lo, "pf") for h in V])
    ok10, _ = every(lambda h: all(0.98 <= r(h, C, "dk") <= 1.06 and 0.97 <= c(h, C).get("R2_over_R1", 0) <= 1.06
                                  for C in cells if c(h, C)["rounds"]))
    out["P10"] = (ok10, dict(dk={C: [r(h, C, "dk") for h in V] for C in cells},
                             R2_over_R1={C: [c(h, C).get("R2_over_R1") for h in V] for C in cells}))
    ok11, _ = every(lambda h: all(c(h, C)["reads"].get(n, 0) >= 1.3 * RSTAR.get(C, float('nan')) for C in cells if c(h, C)["rounds"]
                                  for n in ("R1", "R2")))
    out["P11"] = (ok11, {C: {n: [c(h, C)["reads"].get(n) for h in V] for n in ("R1", "R2", "base", "pf", "both3p")}
                         for C in cells})
    return out


# job 110's frozen RTX 5090 trend: ln(X/base) = a + b ln(ratio) (its header)
TREND = {14: dict(foa=(0.0133, 0.0401), bypass=(0.0189, 0.0462), fetch=(0.3053, 0.3184), both3p=(0.3935, 0.2901)),
         32: dict(foa=(0.0242, 0.1299), bypass=(0.1574, 0.0105), fetch=(0.3457, 0.2809), both3p=(0.5533, 0.2993))}
GLOB110 = os.environ.get("MOSL_JOB110_GLOB", f"{RES}/110[a-h]_secondcard@vast")
GLOB111 = os.environ.get("MOSL_JOB111_GLOB", f"{RES}/111[b-g]_secondcard@vast")


def trend_dev(h, C, n):
    a, b = TREND[C][n]
    return math.log(h["cells"][C]["ratio"][n]) - (a + b * math.log(h["ratio"]))


def trend_dev_ci(h, C, n):
    """the deviation's 95% interval over problems (the ratio's paired bootstrap; the trend is fixed)"""
    ci = h["cells"][C].get("ci", {}).get(n)
    if not ci:
        return None
    a, b = TREND[C][n]
    return tuple(math.log(x) - (a + b * math.log(h["ratio"])) for x in ci)


def predictions110(V, cells=(14, 32)):
    """job 110's registered predictions over its valid hosts (RTX 4090)"""
    lo, mid = cells
    out = {}
    if len(V) < 3:
        out["_conclusive"] = (False, f"{len(V)} valid hosts, fewer than 3")
    c = lambda h, C: h["cells"][C]  # noqa: E731
    r = lambda h, C, n: c(h, C)["ratio"].get(n, float("nan"))  # noqa: E731
    dev = {C: {n: [trend_dev(h, C, n) for h in V if c(h, C)["rounds"]] for n in TREND[C]} for C in cells}
    ok_lo = all(abs(x) <= 0.10 for n in ("fetch", "both3p") for x in dev[lo][n]) and \
        all(abs(x) <= 0.06 for n in ("foa", "bypass") for x in dev[lo][n])
    miss_mid = sum(abs(x) > 0.10 for n in ("fetch", "both3p") for x in dev[mid][n])
    out["Q1"] = (ok_lo and miss_mid <= 1, dict(dev=dev, misses_mid=miss_mid))
    slow = [h for h in V if h["ratio"] < 0.75]
    out["Q2"] = (all(r(h, lo, "pf") < 1 and r(h, lo, "R1") < 1 for h in slow) if slow else None,
                 dict(pf=[r(h, lo, "pf") for h in slow], R1=[r(h, lo, "R1") for h in slow]))
    best = {C: [c(h, C)["best"] for h in V if "best" in c(h, C)] for C in cells}
    out["Q3"] = (all(b <= 1.10 for C in cells for b in best[C]), best)
    dist = {C: [c(h, C)["t"]["base"] / c(h, C)["eq1"] for h in V if "eq1" in c(h, C)] for C in cells}
    out["Q4"] = (all(1.8 <= x <= 3.5 for x in dist[lo]) and all(2.5 <= x <= 6.5 for x in dist[mid]), dist)
    rd = {C: {n: [c(h, C)["reads"].get(n) for h in V if c(h, C)["rounds"]] for n in ("R1", "R2")} for C in cells}
    out["Q5"] = (all((x or 0) >= 1.3 * RSTAR.get(C, float("nan")) for C in cells for n in ("R1", "R2") for x in rd[C][n]), rd)
    fast = [h for h in V if h["ratio"] >= 0.5]
    out["Q6"] = (all(c(h, lo).get("interaction_ms", -1) > 0 for h in fast) if fast else None,
                 [c(h, lo).get("interaction_ms") for h in fast])
    return out


def status(point, ok, ci=None):
    """held: the whole 95% interval (over problems) satisfies the clause; held (point): only the point estimate does,
    or there is no interval; failed: the point estimate does not"""
    if point is None or (isinstance(point, float) and math.isnan(point)):
        return "untested"
    if not ok(point):
        return "failed"
    if ci is not None and ok(ci[0]) and ok(ci[1]):
        return "held"
    return "held (point)"


def clauses109(V, cells=(14, 32)):
    """job 109's predictions as clauses, per machine where the prediction is per machine (scorecard format)"""
    out = []
    lo, mid = cells
    G = {14: "g11", 32: "g25"}

    def add(cid, short, meas, ok, thr, host, ci=None):
        out.append(dict(id=cid, short=short, type="band", measured=None if meas is None else round(float(meas), 4),
                        ci=None if ci is None else [round(float(x), 4) for x in ci], threshold=thr, status=status(meas, ok, ci),
                        why="" if ci is not None else "no interval", host=host))
    for h in V:
        nm = f"{h['job']} {h['cpu']}"; c = h["cells"]; r = lambda C, n: c[C]["ratio"].get(n)  # noqa: E731
        add(f"{h['job']}-P1-g11", f"MIN read once beats both single changes by 0.04 ({nm})",
            r(lo, "fetch") - max(r(lo, "bypass"), r(lo, "foa")), lambda v: v >= 0.04, ">= 0.04", nm)
        for C in cells:
            if c[C]["rounds"]:
                add(f"{h['job']}-P1-int-{G[C]}", f"interaction positive ({nm}, {G[C]})", c[C].get("interaction_ms"), lambda v: v > 0, "> 0 ms", nm)
        for n in ("foa", "bypass"):
            add(f"{h['job']}-P2-{n}", f"{n}/base within [0.96, 1.05] at 11% ({nm})", r(lo, n), lambda v: 0.96 <= v <= 1.05, "0.96 to 1.05", nm,
                c[lo]["ci"].get(n))
        if c[mid]["rounds"]:
            add(f"{h['job']}-P4-g25", f"both3p beats fetch at 25% ({nm})", r(mid, "both3p") - r(mid, "fetch"), lambda v: v > 0, "> 0", nm)
            add(f"{h['job']}-P5", f"bypass/base >= 1.08 at 25% ({nm})", r(mid, "bypass"), lambda v: v >= 1.08, ">= 1.08", nm,
                c[mid]["ci"].get("bypass"))
        for C in cells:
            if "best" in c[C]:
                add(f"{h['job']}-P7-best-{G[C]}", f"best online policy <= 1.10 ({nm}, {G[C]})", c[C]["best"], lambda v: v <= 1.10, "<= 1.10", nm,
                    c[C]["ci"].get("best"))
                add(f"{h['job']}-P7-cap-{G[C]}", f"capture <= 0.35 ({nm}, {G[C]})", c[C]["capture"], lambda v: v <= 0.35, "<= 0.35", nm,
                    c[C]["ci"].get("capture"))
        if h["ratio"] < 0.75:
            add(f"{h['job']}-P8", f"pf/base below 1 at ratio < 0.75 ({nm})", r(lo, "pf"), lambda v: v < 1.0, "< 1.00", nm, c[lo]["ci"].get("pf"))
        if h["ratio"] >= 0.9:
            add(f"{h['job']}-P8", f"pf/base >= 0.98 at ratio >= 0.9 ({nm})", r(lo, "pf"), lambda v: v >= 0.98, ">= 0.98", nm, c[lo]["ci"].get("pf"))
        add(f"{h['job']}-P9", f"|R1 - pf| <= 0.04 at 11% ({nm})", abs(r(lo, "R1") - r(lo, "pf")), lambda v: v <= 0.04, "<= 0.04", nm)
        for C in cells:
            if c[C]["rounds"]:
                add(f"{h['job']}-P10-dk-{G[C]}", f"dk/base within [0.98, 1.06] ({nm}, {G[C]})", r(C, "dk"), lambda v: 0.98 <= v <= 1.06, "0.98 to 1.06", nm,
                    c[C]["ci"].get("dk"))
                add(f"{h['job']}-P10-r2-{G[C]}", f"R2/R1 within [0.97, 1.06] ({nm}, {G[C]})", c[C].get("R2_over_R1"), lambda v: 0.97 <= v <= 1.06, "0.97 to 1.06", nm)
                add(f"{h['job']}-P11-{G[C]}", f"R1 and R2 read >= 1.3 R* ({nm}, {G[C]})",
                    min(c[C]["reads"].get("R1", 0), c[C]["reads"].get("R2", 0)) / RSTAR[C], lambda v: v >= 1.3, ">= 1.3", nm)
    beat_lo = [h["cells"][lo]["ratio"]["both3p"] > h["cells"][lo]["ratio"]["fetch"] for h in V]
    add("109-P4-g11", "both3p beats fetch at 11% on all but at most one", sum(not b for b in beat_lo), lambda v: v <= 1, "<= 1 miss", "pooled")
    for C, (a, b) in ((lo, (0.25, 0.55)), (mid, (0.20, 0.50))):
        v = [h["cells"][C]["together"] for h in V if "together" in h["cells"][C]]
        add(f"109-P3-{G[C]}", f"mean together share in [{a}, {b}]", float(np.mean(v)) if v else None, lambda x, a=a, b=b: a <= x <= b, f"{a} to {b}", "pooled",
            tint(v))
    for C, lim in ((lo, 0.10), (mid, 0.15)):
        v = [h["cells"][C]["residual"] for h in V if "residual" in h["cells"][C]]
        add(f"109-P6-{G[C]}", f"mean residual within +-{lim}", float(np.mean(v)) if v else None, lambda x, lim=lim: abs(x) <= lim, f"|x| <= {lim}", "pooled",
            tint(v))
    v = [h["cells"][lo]["capture"] for h in V if "capture" in h["cells"][lo]]
    add("109-P7-mean", "mean capture at 11% at most 0.25", float(np.mean(v)) if v else None, lambda x: x <= 0.25, "<= 0.25", "pooled", tint(v))
    add("109-valid", f"at least {REQUIRED} valid machines", len(V), lambda x: x >= REQUIRED, f">= {REQUIRED}", "pooled")
    return out


def clauses110(V, cells=(14, 32)):
    out = []
    lo, mid = cells
    G = {14: "g11", 32: "g25"}

    def add(cid, short, meas, ok, thr, host, ci=None):
        out.append(dict(id=cid, short=short, type="band", measured=None if meas is None else round(float(meas), 4),
                        ci=None if ci is None else [round(float(x), 4) for x in ci], threshold=thr, status=status(meas, ok, ci),
                        why="" if ci is not None else "no interval", host=host))
    for h in V:
        nm = f"{h['job']} {h['cpu']}"; c = h["cells"]
        for n in ("fetch", "both3p"):
            add(f"{h['job']}-Q1-{n}-g11", f"{n}/base within 0.10 (log) of the 5090 trend at 11% ({nm})", trend_dev(h, lo, n),
                lambda v: abs(v) <= 0.10, "|x| <= 0.10", nm, trend_dev_ci(h, lo, n))
        for n in ("foa", "bypass"):
            add(f"{h['job']}-Q1-{n}-g11", f"{n}/base within 0.06 (log) of the 5090 trend at 11% ({nm})", trend_dev(h, lo, n),
                lambda v: abs(v) <= 0.06, "|x| <= 0.06", nm, trend_dev_ci(h, lo, n))
        if h["ratio"] < 0.75:
            for n in ("pf", "R1"):
                add(f"{h['job']}-Q2-{n}", f"{n}/base below 1 at 11% ({nm})", c[lo]["ratio"].get(n), lambda v: v < 1.0, "< 1.00", nm,
                    c[lo]["ci"].get(n))
        for C in cells:
            if "best" in c[C]:
                add(f"{h['job']}-Q3-{G[C]}", f"best online policy <= 1.10 ({nm}, {G[C]})", c[C]["best"], lambda v: v <= 1.10, "<= 1.10", nm,
                    c[C]["ci"].get("best"))
            if "eq1" in c[C]:
                a, b = (1.8, 3.5) if C == lo else (2.5, 6.5)
                add(f"{h['job']}-Q4-{G[C]}", f"base / Eq. (1) within [{a}, {b}] ({nm}, {G[C]})", c[C]["t"]["base"] / c[C]["eq1"],
                    lambda v, a=a, b=b: a <= v <= b, f"{a} to {b}", nm)
            if c[C]["rounds"]:
                add(f"{h['job']}-Q5-{G[C]}", f"R1 and R2 read >= 1.3 R* ({nm}, {G[C]})",
                    min(c[C]["reads"].get("R1", 0), c[C]["reads"].get("R2", 0)) / RSTAR[C], lambda v: v >= 1.3, ">= 1.3", nm)
        if h["ratio"] >= 0.5:
            add(f"{h['job']}-Q6", f"interaction positive at 11% ({nm})", c[lo].get("interaction_ms"), lambda v: v > 0, "> 0 ms", nm)
    have = [h for h in V if h["cells"][mid]["rounds"]]
    miss = sum(abs(trend_dev(h, mid, n)) > 0.10 for h in have for n in ("fetch", "both3p")) if have else None
    add("110-Q1-g25", "fetch and both3p within 0.10 (log) of the trend at 25% on all but one host-configuration", miss,
        lambda v: v <= 1, "<= 1 miss", "pooled")
    add("110-valid", "at least 3 valid hosts", len(V), lambda x: x >= 3, ">= 3", "pooled")
    return out


def show(H, pr):
    for h in H:
        print(f"{h['job']} {h['cpu'][:24]:24s} ratio {h.get('ratio', float('nan')):.2f} B_host {h.get('B_host') or 0:.1f} "
              f"valid {h['valid']} {h['gate']} {h['V2_line']} V3 fails {h['V3_fail']}")
        for C, c in h["cells"].items():
            if c.get("rounds"):
                print(f"   C{C} rounds {c['rounds']} " + " ".join(f"{k} {v:.3f}" for k, v in sorted(c["ratio"].items()))
                      + f" | together {c.get('together', float('nan')):.2f} resid {c.get('residual', float('nan')):.2f} "
                      f"best {c.get('best_name')} {c.get('best', float('nan')):.3f} capture {c.get('capture', float('nan')):.2f} "
                      f"reads R1 {c['reads'].get('R1', 0):.1f} base {c['reads'].get('base', 0):.1f} "
                      f"base/eq1 {c['t'].get('base', float('nan')) / c.get('eq1', float('nan')):.2f}")
    for k, (ok, det) in pr.items():
        print(k, "PASS" if ok else ("n/a" if ok is None else "FAIL"), json.dumps(det, default=float)[:300])


def trend_pred(h, C, n):
    a, b = TREND[C][n]
    return math.exp(a + b * math.log(h["ratio"]))


def write_paper(H, V, pr, H2, V2, H3, V3, pr3, cells=(14, 32)):
    """macros (paper/wsg_job109.tex) and the table of both jobs' machines (paper/tab_job109.tex)"""
    from collections import Counter
    M = {}
    lo, mid = cells
    NM = {lo: "Low", mid: "Mid"}

    def rng(k, vals, fmt="{:.2f}"):
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if vals:
            M[k + "Min"] = fmt.format(min(vals)).replace("-", "$-$"); M[k + "Max"] = fmt.format(max(vals)).replace("-", "$-$")
    pc = lambda x: str(int(round(100 * x))).replace("-", "$-$")  # noqa: E731
    # job 109
    M["olN"] = str(len(V)); M["olNWord"] = word(len(V)); M["olStarted"] = str(len(H)); M["olStartedWord"] = word(len(H))
    nz = lambda n: "none" if n == 0 else word(n)  # noqa: E731
    M["olGated"] = nz(sum(1 for h in H if h["gate"])); M["olUnstable"] = nz(sum(1 for h in H if not h["gate"] and not h["valid"]))
    rng("olRatio", [h["ratio"] for h in V])
    for C in cells:
        nm = NM[C]
        sel = [h["cells"][C] for h in V if h["cells"][C]["rounds"]]
        rng(f"olOracle{nm}", [c["ratio"]["both3p"] for c in sel]); rng(f"olFetch{nm}", [c["ratio"]["fetch"] for c in sel])
        rng(f"olBest{nm}", [c["best"] for c in sel], "{:.3f}")
        rng(f"olRone{nm}", [c["ratio"]["R1"] for c in sel], "{:.3f}"); rng(f"olRtwo{nm}", [c["ratio"]["R2"] for c in sel], "{:.3f}")
        rng(f"olPf{nm}", [c["ratio"]["pf"] for c in sel], "{:.3f}"); rng(f"olDk{nm}", [c["ratio"]["dk"] for c in sel], "{:.3f}")
        caps = [c["capture"] for c in sel]
        if caps:
            M[f"olCap{nm}Mean"] = pc(np.mean(caps)); M[f"olCap{nm}Max"] = pc(max(caps)); M[f"olCap{nm}Min"] = pc(min(caps))
        rng(f"olReadsRone{nm}", [min(c["reads"]["R1"], c["reads"]["R2"]) / RSTAR[C] for c in sel])
        rng(f"olReadsBase{nm}", [c["reads"]["base"] / RSTAR[C] for c in sel])
        M[f"olBestName{nm}"] = ", ".join(sorted(Counter(c["best_name"] for c in sel)))
    # the frozen RTX 5090 trend of job 110 against job 109's own RTX 5090s (out of sample, same card; not registered)
    tdv = [trend_dev(h, C, n) for h in V for C in cells if h["cells"][C]["rounds"] for n in ("fetch", "both3p")]
    if tdv:
        M["olTrendDevMax"] = pc(max(abs(math.exp(x) - 1) for x in tdv)); M["olTrendCells"] = str(len(tdv))
        M["olTrendMiss"] = str(sum(abs(x) > 0.10 for x in tdv)); M["olTrendMissWord"] = word(sum(abs(x) > 0.10 for x in tdv))
    # reads of both online policies (R1 and R2) over R*
    for C in cells:
        nm = NM[C]
        rr = [h["cells"][C]["reads"][n] / RSTAR[C] for h in V if h["cells"][C]["rounds"] for n in ("R1", "R2")]
        if rr:
            M[f"olReadsOnline{nm}Min"] = f"{min(rr):.2f}"; M[f"olReadsOnline{nm}Max"] = f"{max(rr):.2f}"
        # the deployed cache's share of Eq. (1)'s speed on these machines
        sh = [h["cells"][C]["eq1"] / h["cells"][C]["t"]["base"] for h in V if h["cells"][C]["rounds"]]
        if sh:
            M[f"olShare{nm}Min"] = pc(min(sh)); M[f"olShare{nm}Max"] = pc(max(sh))
    held = [k for k, (ok, _) in pr.items() if k.startswith("P") and ok]
    failed = [k for k, (ok, _) in pr.items() if k.startswith("P") and ok is False]
    M["olPredN"] = str(sum(1 for k in pr if k.startswith("P"))); M["olPredHeld"] = str(len(held)); M["olPredFailed"] = str(len(failed))
    M["olPredFailedList"] = ", ".join(failed) if failed else "none"
    cl = clauses109(V, cells); st = Counter(c["status"] for c in cl)
    M["olClauses"] = str(len(cl)); M["olClausesHeld"] = str(st.get("held (point)", 0) + st.get("held", 0)); M["olClausesFailed"] = str(st.get("failed", 0))
    # the RTX 4090s: job 111 (registered predictions of job 110, relaunched with V1 corrected) and job 110's first rounds
    M["cxN"] = str(len(V3)); M["cxNWord"] = word(len(V3)); M["cxStarted"] = str(len(H3))
    M["cxFirstN"] = str(sum(1 for h in H2 if h["ran"])); M["cxFirstNWord"] = word(sum(1 for h in H2 if h["ran"]))
    rng("cxRatio", [h["ratio"] for h in V3])
    for C in cells:
        nm = NM[C]
        sel = [h for h in V3 if h["cells"][C]["rounds"]]
        dv = [abs(math.exp(trend_dev(h, C, n)) - 1) for h in sel for n in ("fetch", "both3p")]
        if dv:
            M[f"cxDev{nm}Max"] = pc(max(dv)); M[f"cxDev{nm}Med"] = pc(float(np.median(dv)))
        dv2 = [abs(math.exp(trend_dev(h, C, n)) - 1) for h in sel for n in ("foa", "bypass")]
        if dv2:
            M[f"cxDevSmall{nm}Max"] = pc(max(dv2))
        rng(f"cxBest{nm}", [h["cells"][C]["best"] for h in sel if "best" in h["cells"][C]], "{:.3f}")
        rng(f"cxDist{nm}", [h["cells"][C]["t"]["base"] / h["cells"][C]["eq1"] for h in sel], "{:.1f}")
        rng(f"cxOracle{nm}", [h["cells"][C]["ratio"]["both3p"] for h in sel])
        rng(f"cxFetch{nm}", [h["cells"][C]["ratio"]["fetch"] for h in sel])
        rs = [h["cells"][C]["residual"] for h in sel if "residual" in h["cells"][C]]
        if rs:
            M[f"cxResid{nm}Min"] = pc(min(rs)); M[f"cxResid{nm}Max"] = pc(max(rs))
        rng(f"cxPf{nm}", [h["cells"][C]["ratio"]["pf"] for h in sel]); rng(f"cxRone{nm}", [h["cells"][C]["ratio"]["R1"] for h in sel])
        rng(f"cxLayer{nm}", [h["cells"][C]["ratio"][n] for h in sel for n in ("pf", "R1", "R2")])
        cp = [h["cells"][C]["capture"] for h in sel if "capture" in h["cells"][C]]
        if cp:
            M[f"cxCap{nm}Max"] = pc(max(cp)); M[f"cxCap{nm}Min"] = pc(min(cp))
    # simpler predictors on the same RTX 4090s: no change (1.0x), and the mean of the slow-link panel machines
    dmj = P("prereg", "decomp_measured.json")
    if os.path.exists(dmj):
        dm = json.load(open(dmj))["machines"]
        for C in cells:
            nm = NM[C]
            slow = [m for m in dm[nm] if m["ratio"] < 0.5]
            sel = [h for h in V3 if h["cells"][C]["rounds"]]
            if not slow or not sel:
                continue
            mean = {n: float(np.mean([m["times"]["base"] / m["times"][n] for m in slow])) for n in ("fetch", "both3p")}
            M[f"cxNoChangeDev{nm}Max"] = pc(max(abs(h["cells"][C]["ratio"][n] - 1) for h in sel for n in ("fetch", "both3p")))
            M[f"cxSlowMeanDev{nm}Max"] = pc(max(abs(h["cells"][C]["ratio"][n] / mean[n] - 1) for h in sel for n in ("fetch", "both3p")))
            M["cxSlowPanelN"] = word(len(slow))
    gr = [h.get("ratio") for h in H2 if h["gate"].startswith("VR") and h.get("ratio")]
    M["cxTenGatedRatio"] = f"{gr[0]:.4f}" if gr else "--"
    M["cxTenStarted"] = word(len(H2)); M["cxTenGated"] = word(sum(1 for h in H2 if h["gate"]))
    M["cxElevenStarted"] = word(len(H3)); M["cxElevenNoModel"] = word(sum(1 for h in H3 if not h["gate"] and not h["ran"]))
    cl3 = clauses110(V3, cells); st3 = Counter(c["status"] for c in cl3)
    M["cxClauses"] = str(len(cl3)); M["cxClausesHeld"] = str(st3.get("held", 0)); M["cxClausesPoint"] = str(st3.get("held (point)", 0))
    M["cxClausesFailed"] = str(st3.get("failed", 0))
    M["olClausesHeldCI"] = str(st.get("held", 0)); M["olClausesPoint"] = str(st.get("held (point)", 0))
    first = [h for h in H2 if h["ran"]]
    dvf = [abs(math.exp(trend_dev(h, lo, n)) - 1) for h in first for n in ("fetch", "both3p")]
    if dvf:
        M["cxFirstDevLowMax"] = pc(max(dvf))
    # relaunch agreement: job 111's first round against job 110's first round on the same GPU (deployed cache's time)
    rel = []
    for h in V3:
        g = next((x for x in H2 if x["uuid"] == h["uuid"] and x["ran"]), None)
        if g:
            rel.append(abs(h["cells"][lo]["t_r1_base"] / g["cells"][lo]["t_r1_base"] - 1))
    if rel:
        M["cxRelaunchMax"] = f"{100 * max(rel):.1f}"
    held3 = [k for k, (ok, _) in pr3.items() if k.startswith("Q") and ok]
    failed3 = [k for k, (ok, _) in pr3.items() if k.startswith("Q") and ok is False]
    M["cxPredN"] = str(sum(1 for k in pr3 if k.startswith("Q") and pr3[k][0] is not None)); M["cxPredHeld"] = str(len(held3))
    M["cxPredFailed"] = str(len(failed3)); M["cxPredFailedList"] = ", ".join(failed3) if failed3 else "none"
    lf = [h["loss_base"] for h in H2 + H3 if h.get("loss_base")]
    M["cxLossDiff"] = pc(np.mean(lf) / 0.190 - 1) if lf else "--"
    M["cxLossFour"] = f"{np.mean([h['loss_base'] for h in H2 + H3 if h.get('loss_base')]):.3f}" if any(h.get("loss_base") for h in H2 + H3) else "--"
    for k in ("cxRatioMin", "cxRatioMax", "cxDevLowMax", "cxDevMidMax", "cxDevLowMed", "cxDevMidMed", "cxDevSmallLowMax",
              "cxBestLowMax", "cxBestMidMax", "cxDistLowMin", "cxDistLowMax", "cxDistMidMin", "cxDistMidMax", "cxRelaunchMax",
              "cxFirstDevLowMax", "olBestLowMin", "olBestLowMax", "olCapLowMax", "olCapLowMean", "olCapMidMax", "olCapMidMean",
              "olReadsRoneLowMin", "olReadsRoneLowMax", "olRatioMin", "olRatioMax"):
        M.setdefault(k, "--")
    with open(P("paper", "wsg_job109.tex"), "w") as f:
        f.write("% generated by scripts/job109.py from jobs 109 (RTX 5090), 110 and 111 (RTX 4090)\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    def hw(c, n, pct=False):
        """the 95% interval's half-width over problems, as a subscript"""
        ci = c.get("ci", {}).get(n)
        if not ci:
            return ""
        w = (ci[1] - ci[0]) / 2
        return f"$_{{\\pm{int(round(100 * w))}}}$" if pct else f"$_{{\\pm{w:.2f}}}$".replace("0.", ".", 1)
    # the table: one row per valid machine, both cards
    with open(P("paper", "tab_job109.tex"), "w") as f:
        f.write("% generated by scripts/job109.py\n\\begin{table*}[t]\\centering\\footnotesize\n")
        f.write("\\caption{The registered tests on machines rented for them: speed relative to the deployed cache on the same "
                "machine (geometric mean of its rounds) of \\MinOne, the read-ahead oracle "
                "(\\emph{ahead}), and the best of the four variants without foresight (\\emph{none}: \\Margin, \\LA, "
                "\\LAOne, \\LAOneM; "
                "\\cref{sec:online}), and that variant's capture, its share of the read-ahead oracle's gain. \\emph{Ratio}: link-to-CPU. RTX 5090: job 109, "
                "whose population (desktop-class, link-to-CPU ratio at least 0.5) was fixed before any machine started. RTX "
                "4090: job 111 (job 110's predictions, committed before any RTX 4090 ran); in brackets, what the RTX 5090 "
                "machines' trend in the link-to-CPU ratio predicted for that machine. Subscripts: half-width of the 95\\% interval over "
                "problems (paired bootstrap within the machine).}\\label{tab:job109}\n")
        f.write("\\setlength\\tabcolsep{3pt}\\begin{tabular}{@{}lr" + "rrrr" * 2 + "@{}}\\toprule\n")
        f.write(" & & \\multicolumn{4}{c}{gpt-oss 11\\%} & \\multicolumn{4}{c}{gpt-oss 25\\%} \\\\\\cmidrule(lr){3-6}\\cmidrule(l){7-10}\n")
        f.write("Machine & Ratio & \\MinOne & Ahead & None & Capture & \\MinOne & Ahead & None & Capture \\\\\\midrule\n")
        for card, VV, pred in (("RTX 5090, job 109", V, False), ("RTX 4090, job 111", V3, True)):
            if not VV:
                continue
            f.write("\\multicolumn{10}{@{}l}{\\emph{" + card + "}} \\\\\n")
            for h in sorted(VV, key=lambda x: x["ratio"]):
                row = [h["cpu"], f"{h['ratio']:.2f}"]
                for C in cells:
                    c = h["cells"][C]
                    if not c["rounds"]:
                        row += ["--"] * 4
                        continue
                    for n in ("fetch", "both3p"):
                        v = f"{c['ratio'][n]:.2f}" + hw(c, n)
                        if pred:
                            v += f" {{\\scriptsize[{trend_pred(h, C, n):.2f}]}}"
                        row.append(v)
                    row.append(f"{c['best']:.2f}" + hw(c, "best")); row.append(pc(c["capture"]) + hw(c, "capture", pct=True) + "\\%")
                f.write(" & ".join(row) + " \\\\\n")
            if card.startswith("RTX 5090"):
                f.write("\\addlinespace\n")
        f.write("\\bottomrule\\end{tabular}\\end{table*}\n")
    return M


def main():
    cells = tuple(int(x) for x in os.environ.get("MOSL_JOB109_CELLS", "14,32").split(","))
    H = [load(d, cells, 0.5) for d in sorted(glob.glob(GLOB))]
    V = [h for h in H if h["valid"]]
    pr = predictions(V, cells)
    print("== job 109"); show(H, pr)
    H2 = [load(d, cells, 0.25) for d in sorted(glob.glob(GLOB110))]
    V2 = [h for h in H2 if h["valid"]]
    pr2 = predictions110(V2, cells)
    print("== job 110 (as registered: V1 against the RTX 5090's 0.190)"); show(H2, pr2)
    H3 = [load(d, cells, 0.25) for d in sorted(glob.glob(GLOB111))]
    V3 = [h for h in H3 if h["valid"]]
    pr3 = predictions110(V3, cells)
    print("== job 111 (job 110's predictions on the relaunched RTX 4090s, V1 against 0.1963)"); show(H3, pr3)
    if os.environ.get("MOSL_JOB109_GLOB"):
        return H, pr
    for name, hh, vv, pp in (("job109", H, V, pr), ("job110", H2, V2, pr2), ("job111", H3, V3, pr3)):
        json.dump(dict(hosts=hh, predictions={k: dict(passed=ok, detail=det) for k, (ok, det) in pp.items()},
                       valid=[h["job"] for h in vv]), open(P("prereg", f"{name}.json"), "w"), indent=1, default=float)
    sc = [("109", "jobs/109_onlinepolicy@vast.sh", "02ff39d", clauses109(V, cells)),
          ("110", "jobs/110_secondcard@vast.sh", "ca719f2", clauses110(V2, cells)),
          ("111", "jobs/111_secondcard@vast.sh (job 110's predictions)", "443bcba", clauses110(V3, cells))]
    for job, script, commit, cl in sc:
        json.dump(dict(job=job, script=script, commit=commit, scored_by="scripts/job109.py (machine)", clauses=cl),
                  open(P("prereg", f"scorecard_{job}.json"), "w"), indent=1)
    M = write_paper(H, V, pr, H2, V2, H3, V3, pr3, cells)
    for k in sorted(M):
        print(k, M[k])
    return (H, pr), (H2, pr2), (H3, pr3)


if __name__ == "__main__":
    main()

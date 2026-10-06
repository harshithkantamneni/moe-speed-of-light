"""First-principles re-analysis of every engine launch (jobs 093-104): one table of machines (relaunches merged by GPU
UUID), their probe features and the engine's outcomes at gpt-oss 11% and 25%, and rank correlations between them
(Spearman with a bootstrap interval over machines and a permutation p-value; Benjamini-Hochberg across the family).

    python scripts/reanalysis.py          # writes prereg/reanalysis_hosts.json and prints the tables
"""
import glob
import json
import os
import re
import statistics
import sys

import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
GPU = os.environ.get("MOSL_GPU_BRANCH", "/home/claude/gpu-branch")
sys.path.insert(0, os.path.join(GPU, "jobs", "ec2"))
from fetch_table import bandwidths  # noqa: E402
from scripts.panel_099 import cell_data  # noqa: E402
from scripts.factorial_shapley import limit1_of  # noqa: E402
from scripts.speed_limit import host_rates  # noqa: E402

RES = os.environ.get("MOSL_RESULTS", f"{GPU}/results")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
RNG = np.random.default_rng(7)
JOBS = ("093", "094", "095", "096a", "096b", "097a", "097b", "099", "100", "101", "103", "104")
CELL = {14: "gpt-oss 11%", 32: "gpt-oss 25%"}


def num(pat, txt, f=float, default=None):
    m = re.search(pat, txt)
    return f(m.group(1)) if m else default


def features(d):
    txt = open(f"{d}/concur.txt").read()
    cores = num(r"usable physical cores (\d+)", open(f"{d}/cores.txt").read(), int)
    H = cores - 2 if cores > 4 else 2
    bc, bp, bcp = bandwidths(txt, H)
    f = dict(B_c=bc, B_p=bp, B_cp=bcp, B_host=host_rates(txt) / 1e9, cores=cores, helpers=H)
    ce = [float(x) for x in re.findall(r"^pcie_copyengine_gbs [^\n]*? ([\d.]+)$", txt, re.M)]
    zc = [float(x) for x in re.findall(r"^pcie_zerocopy_gbs [^\n]*? ([\d.]+)$", txt, re.M)]
    f["B_ce"] = max(ce) if ce else None
    f["B_link_max"] = max(ce + zc) if ce or zc else None
    c1 = num(r"cpu_read_gbs t=1 ([\d.]+)", txt)
    f["B_c1"] = c1
    allc = [float(x) for x in re.findall(r"cpu_read_gbs t=\d+ ([\d.]+)", txt)]
    f["B_cmax"] = max(allc) if allc else None
    f["cpu_scaling"] = (max(allc) / c1) if c1 and allc else None
    # per-copy latency: the probe's fetch_us for k copies of one size; slope and intercept by mode
    fu = re.findall(r"fetch_us size=([\d.]+)MB k=(\d+) mode=copyengine ([\d.]+)", txt)
    if fu:
        size = float(fu[0][0])
        ks = np.array([int(k) for s, k, v in fu if float(s) == size], float)
        us = np.array([float(v) for s, k, v in fu if float(s) == size])
        if len(ks) >= 2:
            sl, ic = np.polyfit(ks, us, 1)
            f["copy_us_per_mb"] = sl / size
            f["copy_fixed_us"] = ic
    # the concurrent split: how the two paths share memory (min over thread counts of pcie share when both run)
    conc = re.findall(r"concurrent t=(\d+) pcie=copyengine cpu_gbs ([\d.]+) pcie_gbs ([\d.]+) sum ([\d.]+)", txt)
    if conc:
        f["conc_pcie_min"] = min(float(p) for _, _, p, _ in conc)
        f["conc_sum_max"] = max(float(s) for _, _, _, s in conc)
    st = open(f"{d}/stream.txt").read() if os.path.exists(f"{d}/stream.txt") else ""
    tri = [float(x) for x in re.findall(r"Triad:\s+([\d.]+)", st)]
    f["stream_triad"] = max(tri) / 1e3 if tri else None
    fr = open(f"{d}/free.txt").read() if os.path.exists(f"{d}/free.txt") else ""
    f["ram_gb"] = num(r"Mem:\s+(\d+)", fr, int)
    g = open(f"{d}/gpu.csv").read().splitlines() if os.path.exists(f"{d}/gpu.csv") else []
    if len(g) > 1:
        v = [x.strip() for x in g[1].split(",")]
        f["pcie_gen_max"] = int(v[4]) if v[4].isdigit() else None
        f["pcie_width"] = int(v[7]) if v[7].isdigit() else None
    bw = open(f"{d}/bw_gate.txt").read() if os.path.exists(f"{d}/bw_gate.txt") else (open(f"{d}/bw.txt").read() if os.path.exists(f"{d}/bw.txt") else "")
    f["gpu_read"] = num(r"device read 1 GiB: *([\d.]+)", bw)
    for fn in ("cpu.txt", "lscpu.txt"):
        if os.path.exists(f"{d}/{fn}"):
            m = re.search(r"Model name:\s*(.+)", open(f"{d}/{fn}").read())
            if m:
                f["cpu"] = m.group(1).strip()
                break
    f["ratio"] = bp / bc
    f["ratio_ce"] = (f["B_ce"] or bp) / bc
    f["cp_over_c"] = bcp / bc
    q = open(f"{d}/nvidia-smi-q.txt").read() if os.path.exists(f"{d}/nvidia-smi-q.txt") else ""
    f["gpu_uuid"] = num(r"GPU UUID\s*:\s*(\S+)", q, str)
    return f


def outcomes(d):
    out = {}
    lim = limit1_of(d)
    for C in (14, 32):
        cd = cell_data(d, "g", C)
        if not cd or "base" not in cd["arr"]:
            continue
        b = cd["arr"]["base"]
        o = dict(base_ms=float(b.mean()), n=len(b))
        for k, v in cd["arr"].items():
            if k != "base":
                o[f"{k}/base"] = float(b.mean() / v.mean())
            o[f"ms_{k}"] = float(v.mean())
        if CELL[C] in lim:
            o["limit_ms"] = lim[CELL[C]]
            o["eff_base"] = lim[CELL[C]] / o["base_ms"]
            best = min(v.mean() for v in cd["arr"].values())
            o["eff_best"] = lim[CELL[C]] / float(best)
        # problems: per-problem ms for each state (paired by sequence)
        o["per_problem"] = {k: {int(s): float(x) for s, x in zip(cd["seqs"], v)} for k, v in cd["arr"].items()}
        out[C] = o
    return out


def dirs():
    out = []
    for d in sorted(glob.glob(f"{RES}/*@vast")):
        b = os.path.basename(d)
        if not b.startswith(JOBS) or not os.path.exists(f"{d}/concur.txt") or not os.path.exists(f"{d}/ec_g_C14.jsonl"):
            continue
        if "attempt" in b:
            continue
        out.append(d)
    return out


def table():
    rows = []
    for d in dirs():
        try:
            f = features(d)
            o = outcomes(d)
        except Exception as e:  # noqa: BLE001
            print("skip", os.path.basename(d), e)
            continue
        if o:
            rows.append(dict(dir=os.path.basename(d), **f, cells=o))
    # machines: launches grouped by GPU UUID (falling back to the directory)
    mach = {}
    for r in rows:
        mach.setdefault(r["gpu_uuid"] or r["dir"], []).append(r)
    return rows, mach


def spearman_ci(x, y, nb=4000, nperm=4000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 5:
        return None
    rho = stats.spearmanr(x, y).statistic
    idx = RNG.integers(0, n, (nb, n))
    bs = [stats.spearmanr(x[i], y[i]).statistic for i in idx]
    bs = np.array([b for b in bs if not np.isnan(b)])
    perm = np.array([stats.spearmanr(x, RNG.permutation(y)).statistic for _ in range(nperm)])
    p = (1 + np.sum(np.abs(perm) >= abs(rho))) / (nperm + 1)
    return dict(rho=float(rho), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), p=float(p), n=int(n),
                tau=float(stats.kendalltau(x, y).statistic))


def bh(ps):
    ps = np.asarray(ps)
    o = np.argsort(ps)
    q = np.empty_like(ps, dtype=float)
    m = len(ps)
    prev = 1.0
    for rank, i in reversed(list(enumerate(o, 1))):
        prev = min(prev, ps[i] * m / rank)
        q[i] = prev
    return q


def machine_rows(mach):
    """one row per machine: features of its first launch, outcomes averaged over launches that ran the state"""
    out = []
    for key, launches in mach.items():
        r0 = launches[0]
        row = {k: v for k, v in r0.items() if k != "cells"}
        row["launches"] = [r["dir"] for r in launches]
        for C in (14, 32):
            vals = {}
            for r in launches:
                for k, v in r["cells"].get(C, {}).items():
                    if isinstance(v, (int, float)):
                        vals.setdefault(k, []).append(v)
            for k, v in vals.items():
                row[f"{k}@{C}"] = float(np.mean(v))
        out.append(row)
    return out


FEATS = ["ratio", "ratio_ce", "B_p", "B_ce", "B_c", "B_c1", "B_cp", "B_host", "cp_over_c", "cores", "cpu_scaling",
         "copy_fixed_us", "copy_us_per_mb", "conc_pcie_min", "stream_triad", "ram_gb"]


def main():
    rows, mach = table()
    M = machine_rows(mach)
    print(f"{len(rows)} launches, {len(M)} machines")
    for m in sorted(M, key=lambda r: r["ratio"]):
        print(f"{m['dir'][:22]:22s} {str(m.get('cpu', ''))[:26]:26s} ratio {m['ratio']:.2f} Bc {m['B_c']:5.1f} Bp {m['B_p']:5.1f} "
              f"Bce {m['B_ce'] or 0:5.1f} Bcp {m['B_cp']:5.1f} Bh {m['B_host']:5.1f} cores {m['cores']:3d} "
              f"base11 {m.get('base_ms@14', float('nan')):6.2f} eff11 {m.get('eff_base@14', float('nan')):.2f} "
              f"fetch11 {m.get('fetch/base@14', float('nan')):.3f} launches {len(m['launches'])}")
    json.dump(dict(launches=[{k: v for k, v in r.items() if k != "cells"} | {"cells": {str(C): {k: v for k, v in c.items() if k != "per_problem"}
                                                                                      for C, c in r["cells"].items()}} for r in rows],
                   machines=M), open(P("prereg", "reanalysis_hosts.json"), "w"), indent=1, default=float)
    return rows, M


def sum_law(rows, G={14: 4.56, 32: 4.53}, S=13253760):
    """time per token = G + host reads x S / B_host? Elasticity of (T - G) to B_host, and the fixed cost each state
    implies, T - reads x S / B_host, over every launch (G: the model's constants, the GPU's own compute per token in
    the profiles of job 069c)"""
    out = {}
    for C in (14, 32):
        x = np.array([r["B_host"] for r in rows if C in r["cells"]])
        y = np.array([r["cells"][C]["base_ms"] for r in rows if C in r["cells"]])
        e = {}
        for nm, yy in (("T", y), ("T_minus_G", y - G[C])):
            sl = stats.linregress(np.log(x), np.log(yy)).slope
            bs = []
            for _ in range(4000):
                i = RNG.integers(0, len(x), len(x))
                bs.append(stats.linregress(np.log(x[i]), np.log(yy[i])).slope)
            e[nm] = dict(slope=float(sl), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
        implied = {}
        for r in rows:
            c = r["cells"].get(C)
            if not c:
                continue
            for k in [x_[3:] for x_ in c if x_.startswith("ms_")]:
                f = f"{RES}/{r['dir']}/st_g_C{C}_{k}.json"
                if not os.path.exists(f):
                    continue
                s_ = json.load(open(f))
                n = max(1, s_["steps"])
                reads = (s_["misses"] + s_["admits"] + s_.get("prefetches", 0)) / n
                implied.setdefault(k, []).append(dict(dir=r["dir"], ratio=r["ratio"], reads=reads,
                                                      G=c["ms_" + k] - reads * S / (r["B_host"] * 1e9) * 1e3))
        summ = {}
        for k, v in implied.items():
            g = np.array([q["G"] for q in v])
            summ[k] = dict(n=len(v), median=float(np.median(g)), q1=float(np.percentile(g, 25)), q3=float(np.percentile(g, 75)),
                           lo=float(g.min()), hi=float(g.max()), rho_ratio=float(stats.spearmanr([q["ratio"] for q in v], g).statistic) if len(v) > 4 else None,
                           reads=float(np.median([q["reads"] for q in v])))
        base = implied.get("base", [])
        q = np.array([(r_["G"]) for r_ in base])
        effc = [((rr["cells"][C]["base_ms"] - G[C]) * 1e-3 * rr["B_host"] * 1e9 / S) /
                next(b["reads"] for b in base if b["dir"] == rr["dir"]) for rr in rows if C in rr["cells"] and any(b["dir"] == rr["dir"] for b in base)]
        lim = np.array([r["cells"][C]["limit_ms"] for r in rows if C in r["cells"] and "limit_ms" in r["cells"][C]])
        yb = np.array([r["cells"][C]["base_ms"] for r in rows if C in r["cells"] and "limit_ms" in r["cells"][C]])
        out[C] = dict(elasticity=e, implied_G=summ, eff_over_counted=dict(median=float(np.median(effc)), q1=float(np.percentile(effc, 25)),
                      q3=float(np.percentile(effc, 75)), lo=float(min(effc)), hi=float(max(effc)), n=len(effc)),
                      G_share_of_gap=dict(lo=float(np.min(G[C] / (yb - lim))), hi=float(np.max(G[C] / (yb - lim)))),
                      G_share_of_time=dict(lo=float(np.min(G[C] / yb)), hi=float(np.max(G[C] / yb))))
    return out


def per_problem(rows):
    """do problems rank the same on every machine (Kendall's W), and which trace statistic predicts a problem's gain?"""
    from mosl.cachesim import simulate, simulate_rstar
    first = {}
    for r in rows:
        first.setdefault(r["gpu_uuid"] or r["dir"], r)
    z = np.load(f"{RES}/084c_gptoss_trace@vast/route_aime25_gptoss.npz")
    act = z["act"].astype(np.int64); seq = z["seq"]; E = int(z["n_expert"])
    useq = list(dict.fromkeys(seq.tolist()))
    out = {}
    for C in (14, 32):
        T, L, k = act.shape
        mn = np.zeros((T, L)); dp = np.zeros((T, L))
        for l in range(L):
            m, _ = simulate(act[:, l, :], E, C, "min", bypass=True); mn[:, l] = m
            mm, aa = simulate_rstar(act[:, l, :], E, C, 16.0, 1.0, True); dp[:, l] = mm + aa
        per = {i: dict(excess_rel=float(dp[seq == s].sum(1).mean() / mn[seq == s].sum(1).mean()),
                       excess=float(dp[seq == s].sum(1).mean() - mn[seq == s].sum(1).mean()),
                       min_reads=float(mn[seq == s].sum(1).mean())) for i, s in enumerate(useq)}
        for state in ("fetch", "both3p", "bypass", "foa"):
            mats = []
            for r in first.values():
                c = r["cells"].get(C)
                if not c or state not in c["per_problem"]:
                    continue
                b, v = c["per_problem"]["base"], c["per_problem"][state]
                pr = sorted(set(b) & set(v))
                if len(pr) >= 15:
                    mats.append({p_: b[p_] / v[p_] for p_ in pr})
            if len(mats) < 3:
                continue
            common = sorted(set.intersection(*[set(m) for m in mats]))
            X = np.array([[m[p_] for p_ in common] for m in mats])
            R = np.array([stats.rankdata(x_) for x_ in X]); m_, n_ = R.shape
            W = 12 * ((R.sum(0) - m_ * (n_ + 1) / 2) ** 2).sum() / (m_ ** 2 * (n_ ** 3 - n_))
            lg = np.log(X); tot = lg.var()
            mg = X.mean(0)
            out[f"{state}@{C}"] = dict(machines=m_, problems=n_, kendall_W=float(W),
                                      rho_excess_rel=float(stats.spearmanr(mg, [per[p_]["excess_rel"] for p_ in common]).statistic),
                                      rho_excess=float(stats.spearmanr(mg, [per[p_]["excess"] for p_ in common]).statistic),
                                      var_hosts=float(lg.mean(1).var() / tot), var_problems=float(lg.mean(0).var() / tot),
                                      gain_lo=float(mg.min()), gain_hi=float(mg.max()))
    return out


def run_all():
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        rows, M = main()
    outs = [f"{o}@{C}" for C in (14, 32) for o in ("base_ms", "eff_base", "eff_best", "fetch/base", "both3p/base", "bypass/base", "foa/base", "aa/base")]
    res = correlations(M, outs)
    sl = sum_law(rows)
    pp = per_problem(rows)
    json.dump(dict(machines=len(M), launches=len(rows), correlations=res, sum_law=sl, per_problem=pp),
              open(P("prereg", "reanalysis.json"), "w"), indent=1, default=float)
    print(json.dumps(dict(sum_law=sl, per_problem=pp), indent=1, default=float)[:6000])
    for o in outs:
        show(res, top=3, outcome=o)





def correlations(M, outs, feats=FEATS):
    res = []
    for o in outs:
        for f in feats:
            x = [m.get(f) for m in M]
            y = [m.get(o) for m in M]
            pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None and not np.isnan(a) and not np.isnan(b)]
            if len(pairs) < 6:
                continue
            r = spearman_ci(*zip(*pairs))
            if r:
                res.append(dict(outcome=o, feature=f, **r))
    q = bh([r["p"] for r in res])
    for r, qq in zip(res, q):
        r["q"] = float(qq)
    return res


def show(res, top=60, outcome=None):
    sel = [r for r in res if outcome is None or r["outcome"] == outcome]
    for r in sorted(sel, key=lambda r: -abs(r["rho"]))[:top]:
        print(f"  {r['outcome']:18s} ~ {r['feature']:15s} rho {r['rho']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}] tau {r['tau']:+.2f} p {r['p']:.4f} q {r['q']:.3f} n {r['n']}")


if __name__ == "__main__":
    run_all() if len(sys.argv) < 2 else main()

"""Job 099: the host panel. Per host (results/099?_panel@vast on the gpu branch) and cell, the time per token of every
configuration, its reads, the engine's reads against this machine's replay (value_map_replay.jsonl), the value of
foresight by horizon and accuracy as a share of the aa -> fetch gap in time and in reads, the 2 x 2 ratios, and the
limit on that host (datasheet and measured GPU rate, the probe's highest host rate). Pooled over hosts with a two-stage
bootstrap (hosts, then problems within each drawn host); the machine is the unit. Writes prereg/panel_099.json,
prereg/panel_limits.json (read by scripts/factorial_shapley.py) and paper/wsg_panel.tex.

Time per token = decode ms / decoded tokens per problem, averaged arithmetically over problems; speed ratios are
ratios of those means (t_ref / t_x), paired by problem.

    python scripts/panel_099.py
"""
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.environ.get("MOSL_GPU_BRANCH", "/home/claude/gpu-branch"), "jobs", "ec2"))
from fetch_table import bandwidths  # noqa: E402
from scripts.speed_limit import MODELS, host_rates, limit  # noqa: E402

RES = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
CELLS = {("g", 14): ("gpt-oss 11%", "gpt-oss-120b"), ("g", 32): ("gpt-oss 25%", "gpt-oss-120b"), ("q", 16): ("Qwen3 12.5%", "qwen3-30b-a3b-bf16")}
HYB = ("w1", "w2", "w4", "w16", "w8r5", "allr5")
RNG = np.random.default_rng(99)


def label_of(cfg):
    for part in cfg.split(":"):
        if part.startswith("stats="):
            return os.path.basename(part[6:]).replace(".json", "")
    return None


def host_info(d):
    txt = open(f"{d}/concur.txt").read()
    cores = int(re.search(r"usable physical cores (\d+)", open(f"{d}/cores.txt").read()).group(1))
    H = cores - 2 if cores > 4 else 2
    bc, bp, bb = bandwidths(txt, H)
    cpu = ""
    if os.path.exists(f"{d}/cpu.txt"):
        m = re.search(r"Model name:\s*(.+)", open(f"{d}/cpu.txt").read())
        cpu = m.group(1).strip() if m else ""
    dr = None
    if os.path.exists(f"{d}/bw_gate.txt"):
        m = re.search(r"device read 1 GiB: *([\d.]+)", open(f"{d}/bw_gate.txt").read())
        dr = float(m.group(1)) if m else None
    return dict(cpu=cpu, cores=cores, helpers=H, B_c=bc, B_p=bp, B_cp=bb, b_host=host_rates(txt) / 1e9, device_read=dr)


def cell_data(d, tag, C):
    p = f"{d}/ec_{tag}_C{C}.jsonl"
    if not os.path.exists(p):
        return None
    ms, reads, plan = {}, {}, {}
    for line in open(p):
        if not line.strip():
            continue
        r = json.loads(line)
        lab = label_of(r["config"])
        st = lab.split(f"_C{C}_", 1)[1]
        ms.setdefault(st, {})[r["seq"]] = r["decode_ms"] / r["n_decode"]
    for st in ms:
        f = f"{d}/st_{tag}_C{C}_{st}.json"
        if os.path.exists(f):
            s = json.load(open(f))
            n = max(1, s["steps"])
            reads[st] = (s["misses"] + s["admits"] + s.get("prefetches", 0)) / n
            plan[st] = s.get("oracle_plan_us_per_step", 0.0)
    seqs = sorted(set.intersection(*(set(v) for v in ms.values())))
    arr = {k: np.array([v[q] for q in seqs]) for k, v in ms.items()}
    return dict(seqs=seqs, arr=arr, reads=reads, plan=plan)


def replay_of(d):
    out = {}
    p = f"{d}/value_map_replay.jsonl"
    if os.path.exists(p):
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                tag = "g" if "la_g" in r["la"] else "q"
                out[(tag, r["C"])] = r
    return out


def share(t, a, b, x):
    return (t[a] - t[x]) / (t[a] - t[b])


def main():
    dirs = sorted(glob.glob(f"{RES}/099?_panel@vast"))
    v2 = json.load(open(P("prereg", "speed_limit_v2.json")))
    hosts, limits = [], {}
    for d in dirs:
        if not os.path.exists(f"{d}/concur.txt"):
            continue
        name = "P" + os.path.basename(d)[3]
        hi = host_info(d)
        rp = replay_of(d)
        h = dict(host=name, dir=os.path.basename(d), **hi, cells={})
        limits[name] = {}
        for (tag, C), (lab, key) in CELLS.items():
            cd = cell_data(d, tag, C)
            if cd is None or "base" not in cd["arr"]:
                continue
            m = v2["models"][key]
            mm = MODELS[key]
            L, k, S, Dd = m["L"], m["k"], mm["S"], mm["D"]
            Rs = m["rows"][str(C)]["reads_per_token"]["exact"]
            lim, c_at = limit(Rs, L * k, S, Dd, hi["b_host"] * 1e9, v2["B_gpu_datasheet"])
            lim_meas, _ = limit(Rs, L * k, S, Dd, hi["b_host"] * 1e9, m["B_gpu_effective"])
            limits[name][lab] = 1e3 * lim
            t = {k_: float(v.mean()) for k_, v in cd["arr"].items()}
            c = dict(n=len(cd["seqs"]), ms=t, reads=cd["reads"], plan_us=cd["plan"], limit_ms=1e3 * lim, limit_meas_ms=1e3 * lim_meas,
                     host_bound=bool(c_at <= 0))
            # engine reads against this machine's replay
            r = rp.get((tag, C))
            if r:
                c["replay"] = {k_: r[k_] for k_ in r if k_ in HYB or k_ in ("aa", "min", "online_k1")}
                c["engine_over_replay"] = {k_: cd["reads"][k_] / r[k_] - 1 for k_ in HYB + ("aa",) if k_ in cd["reads"] and k_ in r}
                if "fetch" in cd["reads"]:
                    c["engine_over_replay"]["fetch_vs_min"] = cd["reads"]["fetch"] / r["min"] - 1
            # speed ratios and shares with within-host bootstrap (paired by problem)
            A = cd["arr"]
            n = len(cd["seqs"])
            idx = RNG.integers(0, n, (10000, n))
            if "aa" in t and "fetch" in t:
                c["time_share"] = {x: share(t, "aa", "fetch", x) for x in HYB if x in t}
                c["read_share"] = {x: share(cd["reads"], "aa", "fetch", x) for x in HYB if x in cd["reads"]}
                bm = {k_: A[k_][idx].mean(1) for k_ in A}
                c["time_share_ci"] = {x: [float(np.percentile((bm["aa"] - bm[x]) / (bm["aa"] - bm["fetch"]), q)) for q in (2.5, 97.5)]
                                      for x in HYB if x in bm}
                if all(k_ in bm for k_ in ("base", "foa", "bypass", "fetch")):
                    inter = (bm["foa"] - bm["fetch"]) - (bm["base"] - bm["bypass"])
                    c["interaction_ms"] = [float(((t["foa"] - t["fetch"]) - (t["base"] - t["bypass"]))),
                                           float(np.percentile(inter, 2.5)), float(np.percentile(inter, 97.5))]

            def ratio(ref, x):
                if ref not in A or x not in A:
                    return None
                bs = A[ref][idx].mean(1) / A[x][idx].mean(1)
                return [float(A[ref].mean() / A[x].mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            c["speed"] = {f"{x}/{ref}": ratio(ref, x) for ref, x in (("base", "foa"), ("base", "aa"), ("foa", "aa"), ("base", "bypass"), ("base", "fetch"),
                                                                     ("base", "both3p"), ("aa", "fetch"), ("foa", "fetch")) if ratio(ref, x)}
            c["frac_of_limit"] = {x: c["limit_ms"] / t[x] for x in t}
            c["frac_of_limit_meas"] = {x: c["limit_meas_ms"] / t[x] for x in t}
            h["cells"][lab] = c
            h.setdefault("_arr", {})[lab] = A
        hosts.append(h)
    # pooled: two-stage bootstrap over hosts and problems
    pooled = {}
    for (tag, C), (lab, key) in CELLS.items():
        hs = [h for h in hosts if lab in h["cells"]]
        if len(hs) < 2:
            continue
        stats = {}
        keys = [("fetch/base", "base", "fetch"), ("both3p/base", "base", "both3p"), ("foa/base", "base", "foa"), ("aa/foa", "foa", "aa"),
                ("bypass/base", "base", "bypass")]
        for nm, ref, x in keys:
            per = [h["cells"][lab]["speed"].get(f"{x}/{ref}") for h in hs]
            per = [p for p in per if p]
            if len(per) < 2:
                continue
            vals = np.array([p[0] for p in per])
            hw = np.array([0.5 * (p[2] - p[1]) for p in per])
            bt = []
            arrs = [h["_arr"][lab] for h in hs if ref in h["_arr"][lab] and x in h["_arr"][lab]]
            for _ in range(2000):
                pick = RNG.integers(0, len(arrs), len(arrs))
                v = []
                for i in pick:
                    a = arrs[i]
                    q = RNG.integers(0, len(a[ref]), len(a[ref]))
                    v.append(a[ref][q].mean() / a[x][q].mean())
                bt.append(np.mean(v))
            stats[nm] = dict(hosts=len(per), mean=float(vals.mean()), min=float(vals.min()), max=float(vals.max()), sd=float(vals.std(ddof=1)),
                             ci=[float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))], within_halfwidth_median=float(np.median(hw)))
        for x in HYB:
            ts = [h["cells"][lab]["time_share"][x] for h in hs if x in h["cells"][lab].get("time_share", {})]
            rs = [h["cells"][lab]["read_share"][x] for h in hs if x in h["cells"][lab].get("read_share", {})]
            if ts:
                stats[f"share_{x}"] = dict(hosts=len(ts), time_mean=float(np.mean(ts)), time_min=float(min(ts)), time_max=float(max(ts)),
                                           read_mean=float(np.mean(rs)) if rs else None)
        pooled[lab] = stats
    for h in hosts:
        h.pop("_arr", None)
    json.dump(dict(hosts=hosts, pooled=pooled), open(P("prereg", "panel_099.json"), "w"), indent=1)
    score_clauses(hosts, pooled)
    json.dump(limits, open(P("prereg", "panel_limits.json"), "w"), indent=1)
    # macros
    MC = {}

    def rng(name, vals, fmt="{:.2f}"):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if vals:
            MC[name + "Min"] = fmt.format(min(vals)); MC[name + "Max"] = fmt.format(max(vals))
    MC["pnHosts"] = str(len(hosts))
    for lab, nm in (("gpt-oss 11%", "Low"), ("gpt-oss 25%", "Mid"), ("Qwen3 12.5%", "Qlow")):
        hs = [h for h in hosts if lab in h["cells"]]
        if not hs:
            continue
        MC[f"pnHosts{nm}"] = str(len(hs))
        for x, xn in (("w1", "One"), ("w2", "Two"), ("w4", "Four"), ("w16", "Sixteen"), ("w8r5", "EightHalf"), ("allr5", "AllHalf")):
            rng(f"pnShare{xn}{nm}", [h["cells"][lab].get("time_share", {}).get(x) for h in hs])
            rng(f"pnRead{xn}{nm}", [h["cells"][lab].get("read_share", {}).get(x) for h in hs])
        for k_, kn in (("fetch/base", "Fetch"), ("both3p/base", "Paced"), ("foa/base", "Foa"), ("aa/foa", "AaFoa"), ("aa/base", "Aa")):
            rng(f"pn{kn}{nm}", [h["cells"][lab]["speed"].get(k_, [None])[0] for h in hs])
        if lab in pooled:
            for k_, kn in (("fetch/base", "Fetch"), ("both3p/base", "Paced")):
                if k_ in pooled[lab]:
                    s = pooled[lab][k_]
                    MC[f"pn{kn}{nm}Mean"] = f"{s['mean']:.2f}"; MC[f"pn{kn}{nm}Lo"] = f"{s['ci'][0]:.2f}"; MC[f"pn{kn}{nm}Hi"] = f"{s['ci'][1]:.2f}"
                    MC[f"pn{kn}{nm}Sd"] = f"{s['sd']:.3f}"; MC[f"pn{kn}{nm}Hw"] = f"{s['within_halfwidth_median']:.3f}"
        rng(f"pnBest{nm}", [100 * h["cells"][lab]["frac_of_limit"].get("both3p", np.nan) for h in hs], "{:.0f}")
        rng(f"pnBestMeas{nm}", [100 * h["cells"][lab]["frac_of_limit_meas"].get("both3p", np.nan) for h in hs], "{:.0f}")
        rng(f"pnBase{nm}", [100 * h["cells"][lab]["frac_of_limit"].get("base", np.nan) for h in hs], "{:.0f}")
        rng(f"pnBaseMeas{nm}", [100 * h["cells"][lab]["frac_of_limit_meas"].get("base", np.nan) for h in hs], "{:.0f}")
        eo = [abs(v) for h in hs for k_, v in h["cells"][lab].get("engine_over_replay", {}).items() if k_ != "fetch_vs_min"]
        if eo:
            MC[f"pnReplayMax{nm}"] = f"{100 * max(eo):.1f}"
        pl = [v for h in hs for k_, v in h["cells"][lab].get("plan_us", {}).items() if v]
        if pl:
            MC[f"pnPlanMax{nm}"] = f"{max(pl):.0f}"
    # time share against read share for the exact windows at gpt-oss 11%: the gap (read - time) and how it tracks the
    # host's link/CPU ratio; and the share of each state's host reads that go over the link in the step
    lab = "gpt-oss 11%"
    hs = [h for h in hosts if lab in h["cells"] and "w4" in h["cells"][lab].get("time_share", {})]
    if len(hs) >= 4:
        g4 = [h["cells"][lab]["read_share"]["w4"] - h["cells"][lab]["time_share"]["w4"] for h in hs]
        gx = [h["cells"][lab]["read_share"][x] - h["cells"][lab]["time_share"][x] for h in hs for x in ("w1", "w4", "w16")]
        rr = [np.log(h["B_p"] / h["B_c"]) for h in hs]
        MC["pnGapFourLowMin"] = f"{min(g4):.2f}"; MC["pnGapFourLowMax"] = f"{max(g4):.2f}"
        MC["pnGapExactLowMax"] = f"{max(gx):.2f}"
        MC["pnGapFourCorr"] = f"{np.corrcoef(rr, g4)[0, 1]:.2f}".replace("-", "$-$")
        MC["pnGapRatioAtMax"] = f"{hs[int(np.argmax(g4))]['B_p'] / hs[int(np.argmax(g4))]['B_c']:.2f}"
        for x, xn in (("w4", "Four"), ("w16", "Sixteen"), ("allr5", "AllHalf"), ("fetch", "Min"), ("aa", "Aa")):
            fr = []
            for h in hs:
                f_ = f"{RES}/{h['dir']}/st_g_C14_{x}.json"
                if os.path.exists(f_):
                    s_ = json.load(open(f_))
                    fr.append(s_["fetches"] / max(1, s_["misses"]))
            if fr:
                MC[f"pnLinkPct{xn}"] = f"{100 * np.mean(fr):.0f}"
    rng("pnLink", [h["B_p"] for h in hosts], "{:.0f}")
    rng("pnCpu", [h["B_c"] for h in hosts], "{:.0f}")
    rng("pnHostRate", [h["b_host"] for h in hosts], "{:.0f}")
    with open(P("paper", "wsg_panel.tex"), "w") as f:
        f.write("% generated by scripts/panel_099.py from the job 099 panel (results/099?_panel@vast)\n")
        for k_ in sorted(MC):
            f.write(f"\\newcommand{{\\{k_}}}{{{MC[k_]}}}\n")
    for h in hosts:
        print(h["host"], h["cpu"], f"link {h['B_p']:.0f} cpu {h['B_c']:.0f} both {h['B_cp']:.0f} host {h['b_host']:.0f}")
        for lab, c in h["cells"].items():
            print("  ", lab, "speed", {k_: round(v[0], 3) for k_, v in c["speed"].items()})
            print("     time share", {k_: round(v, 2) for k_, v in c.get("time_share", {}).items()}, "read share", {k_: round(v, 2) for k_, v in c.get("read_share", {}).items()})
            print("     engine/replay-1", {k_: round(100 * v, 1) for k_, v in c.get("engine_over_replay", {}).items()}, "plan us", c["plan_us"])
    print(json.dumps(pooled, indent=1)[:3000])



def _status(point, ci, ok):
    """machine scoring under the interval rule: held if the whole interval satisfies the clause, held (point) if only
    the point does (or no interval exists: 'no interval'), failed otherwise"""
    if point is None:
        return "untested", ""
    if not ok(point):
        return "failed", ""
    if ci is None:
        return "held (point)", "no interval"
    if ok(ci[0]) and ok(ci[1]):
        return "held", ""
    return "held (point)", "interval crosses"


def score_clauses(hosts, pooled):
    """job 099's predictions (header of jobs/099_panel@vast.sh, gpu commit 7b8327a), scored by machine"""
    out = []

    def add(cid, short, ptype, point, ci, ok, threshold, host="pooled"):
        st, why = _status(point, ci, ok)
        out.append(dict(id=cid, short=short, type=ptype, measured=None if point is None else round(float(point), 4),
                        ci=None if ci is None else [round(float(ci[0]), 4), round(float(ci[1]), 4)], threshold=threshold,
                        status=st, why=why, host=host))
    band = lambda lo, hi: (lambda v: lo <= v <= hi)  # noqa: E731
    for h in hosts:
        H = h["host"]
        for lab, cn in (("gpt-oss 11%", "g11"), ("gpt-oss 25%", "g25")):
            c = h["cells"].get(lab)
            if not c:
                continue
            eo = c.get("engine_over_replay", {})
            mx = max((abs(v) for k_, v in eo.items() if k_ != "fetch_vs_min"), default=None)
            add(f"099{H}-P1-{cn}", f"engine reads within 4% of the replay ({lab}, {H})", "band", mx, None, lambda v: v <= 0.04, "<= 0.04", H)
            ts, tc = c.get("time_share", {}), c.get("time_share_ci", {})
            lo11 = lab == "gpt-oss 11%"
            for x, rule, thr in (("w16", (lambda v: v >= 0.80) if lo11 else band(0.45, 0.85), ">= 0.80" if lo11 else "0.45-0.85"),
                                 ("w4", band(0.35, 0.70) if lo11 else band(0.12, 0.40), "0.35-0.70" if lo11 else "0.12-0.40"),
                                 ("w1", lambda v: v <= 0.30, "<= 0.30"),
                                 ("w8r5", band(0.15, 0.45) if lo11 else (lambda v: v <= 0.30), "0.15-0.45" if lo11 else "<= 0.30"),
                                 ("allr5", band(0.30, 0.60), "0.30-0.60")):
                add(f"099{H}-P{3 if x in ('w16', 'w4', 'w1') else 4}-{x}-{cn}", f"{x} time share ({lab}, {H})", "band" if "-" in thr else "threshold",
                    ts.get(x), tc.get(x), rule, thr, H)
            it = c.get("interaction_ms")
            add(f"099{H}-P5a-{cn}", f"MIN's set worth more with one read than two ({lab}, {H})", "sign", it[0] if it else None,
                it[1:] if it else None, lambda v: v > 0, "> 0 ms", H)
            sp = c["speed"]
            fb = sp.get("foa/base")
            add(f"099{H}-P5b-{cn}", f"single read within 5% of deployed ({lab}, {H})", "band", fb[0] if fb else None, fb[1:] if fb else None,
                band(0.95, 1.05), "0.95-1.05", H)
            for k_, lo, hi in (("fetch/base", 1.10, 1.55), ("both3p/base", 1.25, 1.90)):
                v = sp.get(k_)
                add(f"099{H}-P6-{k_.split('/')[0]}-{cn}", f"{k_} ({lab}, {H})", "band", v[0] if v else None, v[1:] if v else None, band(lo, hi), f"{lo}-{hi}", H)
            ms = c["ms"]
            okord = all(k_ in ms for k_ in ("both3p", "fetch", "foa")) and ms["both3p"] < ms["fetch"] < ms["foa"]
            add(f"099{H}-P6-order-{cn}", f"both3p > fetch > foa ({lab}, {H})", "sign", 1.0 if okord else 0.0, None, lambda v: v > 0.5, "ordering", H)
            pl = [v for k_, v in c.get("plan_us", {}).items() if v]
            add(f"099{H}-P9-{cn}", f"host plan at most 150 us per step ({lab}, {H})", "threshold", max(pl) if pl else None, None,
                lambda v: v <= 150, "<= 150 us", H)
            af = sp.get("aa/foa")
            if af and h["B_p"] >= 40:
                add(f"099{H}-P7-{cn}", f"aa / foa in [0.92, 1.04] on a >= 40 GB/s link ({lab}, {H})", "band", af[0], af[1:], band(0.92, 1.04), "0.92-1.04", H)
            elif af and h["B_p"] < 32:
                add(f"099{H}-P7-{cn}", f"aa / foa below 0.95 on a < 32 GB/s link ({lab}, {H})", "threshold", af[0], af[1:], lambda v: v < 0.95, "< 0.95", H)
        c = h["cells"].get("Qwen3 12.5%")
        if c:
            ts, tc = c.get("time_share", {}), c.get("time_share_ci", {})
            add(f"099{H}-P10-w2", f"Qwen3 w2 time share ({H})", "band", ts.get("w2"), tc.get("w2"), band(0.30, 0.70), "0.30-0.70", H)
            add(f"099{H}-P10-w8r5", f"Qwen3 w8r5 time share ({H})", "band", ts.get("w8r5"), tc.get("w8r5"), band(0.25, 0.60), "0.25-0.60", H)
            v = c["speed"].get("fetch/base")
            add(f"099{H}-P10-fetch", f"Qwen3 fetch/base ({H})", "band", v[0] if v else None, v[1:] if v else None, band(1.15, 1.55), "1.15-1.55", H)
    # pooled clauses
    pairs = [(h["cells"][lab]["time_share"][x], h["cells"][lab]["read_share"][x]) for h in hosts for lab in ("gpt-oss 11%", "gpt-oss 25%")
             if lab in h["cells"] for x in HYB if x in h["cells"][lab].get("time_share", {}) and x in h["cells"][lab].get("read_share", {})]
    if pairs:
        frac = float(np.mean([abs(a - b) <= 0.15 for a, b in pairs]))
        add("099-P2a", "time share within 0.15 of read share at >= 80% of hybrid host-cells", "threshold", frac, None, lambda v: v >= 0.8, ">= 0.80")
    mono = [h["cells"][lab]["time_share"]["w1"] < h["cells"][lab]["time_share"]["w4"] < h["cells"][lab]["time_share"]["w16"]
            for h in hosts for lab in ("gpt-oss 11%", "gpt-oss 25%") if lab in h["cells"] and all(x in h["cells"][lab].get("time_share", {}) for x in ("w1", "w4", "w16"))]
    if mono:
        add("099-P2b", "w1 < w4 < w16 at >= 80% of host-cells", "threshold", float(np.mean(mono)), None, lambda v: v >= 0.8, ">= 0.80")
    for lab, cn in (("gpt-oss 11%", "g11"), ("gpt-oss 25%", "g25")):
        st = pooled.get(lab, {}).get("fetch/base")
        if st:
            add(f"099-P8-{cn}", f"spread of fetch/base across hosts > 3x the median within-host half-width ({lab})", "threshold",
                (st["max"] - st["min"]) / max(st["within_halfwidth_median"], 1e-9), None, lambda v: v > 3, "> 3")
    json.dump(dict(job="099", script="jobs/099_panel@vast.sh", commit="7b8327a", scored_by="scripts/panel_099.py (machine)", clauses=out),
              open(P("prereg", "scorecard_099.json"), "w"), indent=1)
    from collections import Counter
    print("job 099 clauses:", Counter(c["status"] for c in out), Counter(c["why"] for c in out if c["why"]))


if __name__ == "__main__":
    main()

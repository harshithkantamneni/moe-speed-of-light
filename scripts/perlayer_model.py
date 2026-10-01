"""Per-layer bytes-over-rate models of offloaded MoE decode against the token-level host-memory law (paper Eq. 1), on
the blind cross-host measurements of Table 2 (jobs 069c x3, 072, 073, 073a, 074, 076: 33 measurements, 29
configurations, 7 hosts; the 9950X3D ran twice), refitted leave-one-host-out.

Per layer l and decode step, the engine's stats file records how many of the layer's 4 routed experts were not
resident (miss_hist[n]); of those, f = table(n) are copied over PCIe into slots (FETCH) and c = n - f run on the CPU
helpers; a prefetch copy (p in {0, 1}; q_l = prefetches into layer l+1 / steps) is issued in layer l's window. S is
the bytes of one expert; B_c, B_p, B_cp are the host's probed rates (CPU read at the helper count, zero-copy link,
median of both at once), never fitted; s_c = S/B_c etc. X_c, X_p are the per-token expert reads (CPU, link).
Models (T = time per token; N_c = layer-steps per token with c > 0):
  A     token-max, Eq. 1:            T = G + max(X_c s_c, X_p s_p, (X_c + X_p) s_cp)
  A+h   token-max + hand-off:         T = G + h N_c + max(...)
  B     additive:                     T = G + X_c s_c + X_p s_p
  C     per-layer max (Eq. 3 form):   T = G0 + sum_l E[ max( (f+p) s_p + g, [c>0](h + c s_c), [c>0][f+p>0](c+f+p) s_cp ) ]
        with h = 0; C+h with h free; C(g=37) with g frozen at the profile's 37 us (paper Eq. 3), C(g=37)+h likewise;
        Cu applies the shared term to every layer that reads host memory, as Eq. 1 does per token
  D     per-layer additive:           T = G0 + sum_l E[ g + [c>0](h + c s_c) + (f+p) s_p ]  =  B + h N_c (g merges into G)
  E     per-layer, asymmetric share:  as C(g=37)+h, but when a layer reads through both paths at once each runs at the
        probe's concurrent rate (cpu_gbs, pcie_gbs of concur.txt) until the first finishes, then the other alone
  A*k   token-max, CPU rate factor:   T = G + max(k X_c s_c, X_p s_p, (k X_c + X_p) s_cp): the helpers read at B_c / k
  C*k+h per-layer max with k and h, g frozen at 37 us
The free constants (G or G0 in ms, g and h in us, k dimensionless, all >= 0) minimise the sum of |ln(predicted /
measured)| on the training rows; leave-one-host-out refits them on the other six hosts and scores the held-out host,
and the in-sample fit on all rows is given for reference, as is Eq. 1 with the paper's frozen G = 4.819 ms. The
mailbox counters (mb_busy_us / mb_count: helper wall time per layer-step by CPU-expert count) are printed against
S / B_c as a direct check of the per-layer CPU term.

    python scripts/perlayer_model.py [--out prereg/perlayer_model.json]
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from scipy.optimize import minimize

RES = "/home/claude/gpu-branch/results"
S = 13253760.0    # bytes per gpt-oss-120b expert (GGUF)
L = 36
K = 4
G_FROZEN = 4.819346      # ms, paper Eq. 1 (law_predict.py)
G_PROFILE_US = 37.0      # GPU expert products per layer, gpt-oss (paper Eq. 3, fetch_table.py)
VRAM_MS = 1000.0 / 255.27938471870277   # all-in-VRAM token time, RTX 5090 (prereg/vram_084.json)
MAP = [   # stats file name -> configuration (as in law_crosshost.py)
    (r"ref_slots_14\.json|ref_C14_r\d\.json", "C14"),
    (r"ref_slots_32\.json|ref_C32_r\d\.json", "C32"),
    (r"ref_slots_56\.json|ref_C56_r\d\.json", "C56"),
    (r"ref_slots_32_f\.json|ref_C32f_r\d\.json", "C32_fetch"),
    (r"ref_slots_32_f_pf\.json", "C32_fetch_pf"),
    (r"ref_C32pf_r\d\.json", "C32_pf"),
]
HOST = {  # results dir -> host id (the 9950X3D machine of job 073 ran twice)
    "069c_profile_270k@vast": "270K", "069c_profile_epyc7352@vast": "EPYC7352", "069c_profile_epyc9655@vast": "EPYC9655",
    "072_defer@vast": "7900", "073_samehost_v2@vast": "9950X3D", "073a_samehost_v2_attempt1@vast": "9950X3D",
    "074_fix40@vast": "7950X", "076_table_5090@vast": "9950X",
}
FETCH_CFG = ("C32_fetch", "C32_fetch_pf")
PF_CFG = ("C32_fetch_pf", "C32_pf")


def rates(p):
    by = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            by.setdefault(r.get("config", ""), []).append(r)
    return {k: sum(r["n_decode"] for r in v) / (sum(r["decode_ms"] for r in v) / 1000) for k, v in by.items()}


def concurrent_rates(d):
    """Per-path rates while both run (GB/s): medians over the zero-copy lines, the reading behind B_both."""
    t = open(f"{RES}/{d}/concur.txt").read()
    m = re.findall(r"concurrent t=\d+ pcie=zerocopy64 cpu_gbs ([\d.]+) pcie_gbs ([\d.]+) sum ([\d.]+)", t)
    return float(np.median([float(a) for a, _, _ in m])), float(np.median([float(b) for _, b, _ in m]))


def events(st):
    """Expected number of layer-steps per token with c CPU experts and f' = f + p link experts: {(c, f'): weight}, sum L."""
    steps = st["steps"]
    tab = [int(x) for x in st["fetch_table"].split(",")] if st.get("fetch_table") else [0] * (K + 1)
    layers = st["layers"]
    w = {}
    for l, lay in enumerate(layers):
        q = layers[l + 1]["prefetches"] / steps if l + 1 < len(layers) else 0.0   # prefetch copy issued in this window
        for n, cnt in enumerate(lay["miss_hist"]):
            if not cnt:
                continue
            f, c = tab[n], n - tab[n]
            for p, pp in ((0, 1.0 - q), (1, q)):
                if pp > 0:
                    w[(c, f + p)] = w.get((c, f + p), 0.0) + cnt / steps * pp
    return w


def helper_time(st):
    """Mean helper busy time per layer-step by CPU-expert count (us), from the mailbox counters."""
    return {c: (b / n if n else None) for c, (b, n) in enumerate(zip(st["mb_busy_us"], st["mb_count"])) if c > 0}


def load_rows():
    rows = []
    for d in sorted(HOST):
        pred = json.load(open(f"{RES}/{d}/law_prediction.json"))
        cpu = re.search(r"Model name:\s*(.+)", open(f"{RES}/{d}/lscpu.txt").read()).group(1).strip()
        rc, rp = concurrent_rates(d)
        meas, stats = {}, {}
        for k, v in rates(f"{RES}/{d}/ref_ec.jsonl").items():
            if "stats=" not in k:
                continue
            name = k.split("stats=")[-1].split("/")[-1]
            for pat, key in MAP:
                if re.fullmatch(pat, name):
                    meas.setdefault(key, []).append(v)
                    stats.setdefault(key, []).append(name)
        for key in ("C14", "C32", "C56", "C32_fetch", "C32_fetch_pf", "C32_pf"):
            if key not in meas:
                continue
            sts = [json.load(open(f"{RES}/{d}/{n}")) for n in stats[key]]
            st = sts[0]
            ev = events(st)
            for o in sts[1:]:
                assert events(o) == ev, (d, key)   # repeated launches on one host record the same routing
            Xc = sum(w * c for (c, f), w in ev.items())
            Xp = sum(w * f for (c, f), w in ev.items())
            assert abs(Xc - (st["misses"] - st["fetches"]) / st["steps"]) < 1e-6
            assert abs(Xp - (st["fetches"] + st.get("prefetches", 0)) / st["steps"]) < 1e-6
            measured = float(np.mean(meas[key]))
            helper_ms = float(np.mean([sum(o["mb_busy_us"]) / o["steps"] / 1e3 for o in sts]))   # helpers' critical path per token
            rows.append(dict(
                host=HOST[d], job=d, cpu=cpu, config=key, launch=len(sts), stats=stats[key], measured=measured,
                helpers=pred["helpers"], B_c=pred["B_c"], B_p=pred["B_p"], B_cp=pred["B_both"], r_c=rc, r_p=rp,
                X_c=Xc, X_p=Xp, N_c=sum(w for (c, f), w in ev.items() if c > 0), N_f=sum(w for (c, f), w in ev.items() if f > 0),
                N_0=ev.get((0, 0), 0.0), events={f"{c},{f}": w for (c, f), w in sorted(ev.items())},
                helper_us=helper_time(st), helper_ms=helper_ms, rest_ms=1000.0 / measured - helper_ms,
                fetch_table=st.get("fetch_table", ""), law_prediction=pred["predicted_tok_s"].get(key)))
    return rows


# ---------------------------------------------------------------- models: T in ms; params in ms (G) and us (g, h)
def per_layer(r, g_us, h_us, form, k=1.0):
    sc, sp, sb = k * S / r["B_c"] / 1e6, S / r["B_p"] / 1e6, S / r["B_cp"] / 1e6   # ms per expert
    g, h = g_us / 1e3, h_us / 1e3
    t = 0.0
    for key, w in r["events"].items():
        c, f = map(int, key.split(","))
        pcie = f * sp + g                          # copy, then the GPU's expert products
        cpu = h + c * sc if c > 0 else 0.0         # helpers, started before the copy
        if form == "C":
            shared = (k * c + f) * sb if (c > 0 and f > 0) else 0.0
            tl = max(pcie, cpu, shared)
        elif form == "Cu":
            shared = (k * c + f) * sb
            tl = max(pcie, cpu, shared)
        elif form == "D":
            tl = pcie + cpu
        elif form == "E":
            if c > 0 and f > 0:
                # both paths read host DRAM at once at the probe's concurrent rates until the first finishes
                s1, s2 = k * S / r["r_c"] / 1e6, S / r["r_p"] / 1e6
                tc, tp = h + c * s1, f * s2
                if tc <= tp:      # CPU done first: the rest of the copy at the link's own rate
                    tl = tc + (1.0 - tc / tp) * f * sp + g
                else:             # copy done first: the rest of the CPU read at its own rate
                    tl = max(tp + (1.0 - tp / tc) * c * sc, tp + g)
            else:
                tl = max(pcie, cpu)
        else:
            raise ValueError(form)
        t += w * tl
    return t


def token_max(r, k=1.0):
    sc, sp, sb = k * S / r["B_c"] / 1e6, S / r["B_p"] / 1e6, S / r["B_cp"] / 1e6
    return max(r["X_c"] * sc, r["X_p"] * sp, (k * r["X_c"] + r["X_p"]) * sb)


def token_add(r):
    return r["X_c"] * S / r["B_c"] / 1e6 + r["X_p"] * S / r["B_p"] / 1e6


MODELS = {  # name: (free parameter names, predictor(row, dict of params) -> T ms)
    "A":        (["G"], lambda r, p: p["G"] + token_max(r)),
    "A+h":      (["G", "h"], lambda r, p: p["G"] + p["h"] / 1e3 * r["N_c"] + token_max(r)),
    "B":        (["G"], lambda r, p: p["G"] + token_add(r)),
    "C":        (["G", "g"], lambda r, p: p["G"] + per_layer(r, p["g"], 0.0, "C")),
    "C+h":      (["G", "g", "h"], lambda r, p: p["G"] + per_layer(r, p["g"], p["h"], "C")),
    "C(g=37)":  (["G"], lambda r, p: p["G"] + per_layer(r, G_PROFILE_US, 0.0, "C")),
    "C(g=37)+h": (["G", "h"], lambda r, p: p["G"] + per_layer(r, G_PROFILE_US, p["h"], "C")),
    "Cu":       (["G", "g"], lambda r, p: p["G"] + per_layer(r, p["g"], 0.0, "Cu")),
    "D":        (["G", "h"], lambda r, p: p["G"] + per_layer(r, 0.0, p["h"], "D")),
    "E":        (["G", "h"], lambda r, p: p["G"] + per_layer(r, G_PROFILE_US, p["h"], "E")),
    "A*k":      (["G", "k"], lambda r, p: p["G"] + token_max(r, p["k"])),
    "C*k+h":    (["G", "k", "h"], lambda r, p: p["G"] + per_layer(r, G_PROFILE_US, p["h"], "C", p["k"])),
}
STARTS = {"G": (2.0, 3.5, 5.0), "g": (0.0, 40.0, 120.0), "h": (0.0, 30.0, 80.0), "k": (1.0, 1.25)}


def fit(model, rows):
    names, f = MODELS[model]

    def obj(x):
        p = {n: abs(v) for n, v in zip(names, x)}
        return sum(abs(np.log(f(r, p) / (1000.0 / r["measured"]))) for r in rows)

    best = None
    for x0 in np.array(np.meshgrid(*[STARTS[n] for n in names])).T.reshape(-1, len(names)):
        res = minimize(obj, x0, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-9, maxiter=4000))
        if best is None or res.fun < best.fun:
            best = res
    res = minimize(obj, best.x, method="Powell", options=dict(xtol=1e-7, ftol=1e-10))
    if res.fun < best.fun:
        best = res
    return {n: float(abs(v)) for n, v in zip(names, best.x)}


def predict(model, rows, p):
    return [1000.0 / MODELS[model][1](r, p) for r in rows]


def summary(rows, pred):
    e = np.array([q / r["measured"] - 1 for r, q in zip(rows, pred)])
    a = np.abs(e)

    def sel(cond):
        return {f"{r['host']} {r['config']}" + (f" #{i}" if r["job"].startswith("073a") else ""): round(100 * x, 1)
                for i, (r, x) in enumerate(zip(rows, e)) if cond(r)}
    return dict(median=float(np.median(a)) * 100, p90=float(np.percentile(a, 90)) * 100, max=float(a.max()) * 100,
                mean=float(a.mean()) * 100, n=len(rows),
                fetch=sel(lambda r: r["config"] in FETCH_CFG), prefetch=sel(lambda r: r["config"] in PF_CFG),
                epyc9655=sel(lambda r: r["host"] == "EPYC9655"), epyc7352=sel(lambda r: r["host"] == "EPYC7352"),
                median_desktop=float(np.median([x for r, x in zip(rows, a) if r["host"] != "EPYC9655"])) * 100,
                median_nofetch=float(np.median([x for r, x in zip(rows, a) if r["config"] not in FETCH_CFG + PF_CFG])) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = load_rows()
    hosts = sorted({r["host"] for r in rows}, key=lambda h: [r["job"] for r in rows if r["host"] == h][0])
    n_cfg = len({(r["host"], r["config"]) for r in rows})
    print(f"{len(rows)} measurements, {n_cfg} configurations, {len(hosts)} hosts; "
          f"{sum(1 for r in rows if r['events'])} rows with per-layer histograms (36 layers each)")
    print("measured and law = tok/s (law: the job's blind prediction, frozen G); B in GB/s; X = expert reads per token; N = layer-steps per\n"
          "token with CPU work / with a copy / with neither; helper = the helpers' wall time per token (mailbox counters), rest = the\n"
          "token time outside it; helper us = mean helper wall time per layer-step by CPU-expert count")
    print(f"{'host':9s} {'job':28s} {'config':13s} {'meas':>6s} {'law':>6s} {'B_c':>6s} {'B_p':>5s} {'B_cp':>6s} {'hlp':>3s} "
          f"{'X_c':>6s} {'X_p':>6s} {'N_c':>5s} {'N_f':>5s} {'N_0':>5s} {'helper':>7s} {'rest':>6s}  helper us by c (1..4)")
    for r in rows:
        hu = " ".join(f"{r['helper_us'][c]:5.0f}" if r["helper_us"].get(c) else "    -" for c in (1, 2, 3, 4))
        print(f"{r['host']:9s} {r['job'][:28]:28s} {r['config']:13s} {r['measured']:6.1f} {r['law_prediction'] or 0:6.1f} {r['B_c']:6.1f} "
              f"{r['B_p']:5.1f} {r['B_cp']:6.1f} {r['helpers']:3d} {r['X_c']:6.2f} {r['X_p']:6.2f} {r['N_c']:5.1f} {r['N_f']:5.1f} {r['N_0']:5.1f} "
              f"{r['helper_ms']:5.2f}ms {r['rest_ms']:4.2f}ms  {hu}" + (f"  x{r['launch']} runs" if r["launch"] > 1 else ""))

    # helper busy time against S/B_c: slope and intercept over c = 1..4 where all four counts exist
    print("\nhelper busy time per layer-step (us): fitted a + b*c against the probe's S/B_c")
    helper_fit = []
    for r in rows:
        hu = r["helper_us"]
        cs = [c for c in (1, 2, 3, 4) if hu.get(c)]
        sc_us = S / r["B_c"] / 1e3
        if len(cs) >= 2:
            b, a0 = np.polyfit(cs, [hu[c] for c in cs], 1)
            helper_fit.append(dict(host=r["host"], config=r["config"], intercept_us=float(a0), slope_us=float(b), S_over_Bc_us=sc_us))
            print(f"   {r['host']:9s} {r['config']:13s} intercept {a0:6.0f}  slope {b:6.0f} per expert  (S/B_c {sc_us:5.0f}, slope/(S/B_c) {b / sc_us:4.2f})")
        else:
            helper_fit.append(dict(host=r["host"], config=r["config"], c1_us=hu[1], S_over_Bc_us=sc_us,
                                   c1_nofetch_us=next((q["helper_us"][1] for q in rows if q["host"] == r["host"] and q["config"] == "C32"), None)))
            print(f"   {r['host']:9s} {r['config']:13s} c=1 only: {hu[1]:6.0f} us  (S/B_c {sc_us:5.0f}; C32 without fetch on this host: "
                  f"{helper_fit[-1]['c1_nofetch_us']:.0f})")

    # the Jensen gap itself: sum over layers of the per-layer maxima minus the maximum of the per-token sums (g = h = 0),
    # against the time the frozen law leaves unexplained
    print("\nJensen gap (per-layer max summed - token max, same rates, g = h = 0) against the frozen law's residual, ms per token:")
    print(f"   {'row':26s} {'measured':>9s} {'frozen law':>11s} {'residual':>9s} {'Jensen':>7s}")
    for r in rows:
        tm = token_max(r)
        r["jensen_ms"] = per_layer(r, 0.0, 0.0, "C") - tm
        r["residual_frozen_ms"] = 1000.0 / r["measured"] - (G_FROZEN + tm)
        if r["config"] in FETCH_CFG + PF_CFG or r["host"] == "EPYC9655":
            print(f"   {r['host'] + ' ' + r['config']:26s} {1000 / r['measured']:8.2f}ms {G_FROZEN + tm:10.2f}ms {r['residual_frozen_ms']:+8.2f}ms {r['jensen_ms']:+6.2f}ms")

    out = dict(rows=rows, hosts=hosts, models={}, helper_fit=helper_fit, vram_ms=VRAM_MS)
    # the paper's frozen law
    pf = predict("A", rows, {"G": G_FROZEN})
    out["models"]["A frozen G=4.819"] = dict(params={"G": G_FROZEN}, insample=summary(rows, pf), pred=pf)
    for m in MODELS:
        p_all = fit(m, rows)
        ins = predict(m, rows, p_all)
        loho_pred = [None] * len(rows)
        folds = {}
        for h in hosts:
            tr = [r for r in rows if r["host"] != h]
            p = fit(m, tr)
            folds[h] = p
            for i, r in enumerate(rows):
                if r["host"] == h:
                    loho_pred[i] = 1000.0 / MODELS[m][1](r, p)
        out["models"][m] = dict(params=p_all, insample=summary(rows, ins), loho=summary(rows, loho_pred), folds=folds,
                                pred_insample=ins, pred_loho=loho_pred)

    def fmt(p):
        return ", ".join(f"{k} {v:.3f} ms" if k == "G" else f"{k} {v:.3f}" if k == "k" else f"{k} {v:.0f} us" for k, v in p.items())
    def table(which):
        print(f"{'model':18s} {'median':>7s} {'p90':>6s} {'max':>6s} | {'FETCH rows (270K f, fpf; 7352 f, fpf; 9655 f, fpf; 7900; X3D; X3D; 9950X)':<58s} "
              f"| {'prefetch (270K, 7352, 9655, 7900)':<34s} | EPYC 9655 C14/C32/C56")
        for m, v in out["models"].items():
            s = v.get(which) or v["insample"]
            fe = " ".join(f"{x:+.1f}" for x in s["fetch"].values())
            pf_ = " ".join(f"{x:+.1f}" for x in s["prefetch"].values())
            ep = " ".join(f"{x:+.1f}" for k, x in s["epyc9655"].items() if k.split()[1] in ("C14", "C32", "C56"))
            print(f"{m:18s} {s['median']:6.1f}% {s['p90']:5.1f}% {s['max']:5.1f}% | {fe:<58s} | {pf_:<34s} | {ep}")
    print("\n=== leave-one-host-out: constants refitted on the other six hosts, error = predicted / measured - 1 (%) ===")
    table("loho")
    print("\n=== in-sample: constants fitted on all 33 rows (reference only) ===")
    table("insample")
    print("\nfitted constants (all rows) and the leave-one-host-out range:")
    for m, v in out["models"].items():
        if "folds" in v:
            rng = {k: (min(f[k] for f in v["folds"].values()), max(f[k] for f in v["folds"].values())) for k in v["params"]}
            print(f"   {m:18s} {fmt(v['params'])};  folds: " + ", ".join(
                f"{k} {lo:.3f}-{hi:.3f}" if k in ("G", "k") else f"{k} {lo:.0f}-{hi:.0f}" for k, (lo, hi) in rng.items()))
        else:
            print(f"   {m:18s} {fmt(v['params'])}")
    print(f"   all-in-VRAM token time (RTX 5090, prereg/vram_084.json): {VRAM_MS:.2f} ms")
    print("\nper-row leave-one-host-out errors (predicted / measured - 1, %):")
    ms = [m for m in out["models"] if "loho" in out["models"][m]]
    print(f"{'row':28s}" + "".join(f"{m:>11s}" for m in ["frozen"] + ms))
    for i, r in enumerate(rows):
        lab = f"{r['host']} {r['config']}" + (" (2nd)" if r["job"].startswith("073a") else "")
        vals = [out["models"]["A frozen G=4.819"]["pred"][i]] + [out["models"][m]["pred_loho"][i] for m in ms]
        print(f"{lab:28s}" + "".join(f"{100 * (v / r['measured'] - 1):+10.1f}%" for v in vals))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1, default=float)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()

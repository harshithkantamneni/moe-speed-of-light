"""Step-time model of the mailbox expert cache, fitted on the calibration run (profile sequences, job 019) and
used to predict the test run (fresh sequences, job 020) before it is made.

Per decode step, with r layer-steps that hand experts to the CPU helpers, mu CPU experts and alpha admissions:

    T = T0 + a0 * r + a1 * mu + beta * alpha                     (mailbox cache and mailbox static-layer layout)
    T = T0s + s * n                                              (llama.cpp static layers, n CPU layers)

T0 is the step with every expert on the GPU, a0 the fixed cost of one hand-off (helper start-up, phase
synchronisation and the GPU wait, net of the GPU expert work it replaces), a1 the helpers' time per expert
(bytes over their bandwidth), beta the critical-path cost of one admission copy. A static-layer layout on the
helpers has r = n and mu = n k. For the cache, r, mu and alpha on the test sequences come from replaying the HF
routing traces through the exact policy simulator (mosl/ecsim.py; gpt-oss-20b and Qwen3, whose traces we have);
gpt-oss-120b has no trace, so its per-step counts are carried over from its calibration run.

    python scripts/mb_model.py --calib <019 results dir> --out prereg/a10_mb
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl.ecsim import load_steps, simulate  # noqa: E402

MODELS = {  # tag -> (L, E, k, kappa, budgets in 1/8, trace dir, corpus, test sequences)
    "gpt-oss-20b-MXFP4": (24, 32, 4, 1, [1, 2, 4], "data/traces/gpt-oss-20b", "data/tok_gpt-oss-20b.jsonl",
                          [7, 10, 12, 15, 16, 18, 19, 20, 25, 26, 27, 31]),
    "Qwen3-30B-A3B-Instruct-2507-Q4_K_M": (48, 128, 8, 2, [1, 2, 4], "data/traces/qwen3-30b-a3b", "data/tok_qwen3_30b.jsonl",
                                           [7, 10, 12, 15, 16, 18, 19, 20, 25, 26, 27, 32]),
    "Qwen3-30B-A3B-Instruct-2507-Q8_0": (48, 128, 8, 2, [1, 2, 4], "data/traces/qwen3-30b-a3b", "data/tok_qwen3_30b.jsonl",
                                         [7, 10, 12, 15, 16, 18, 19, 20, 25, 26, 27, 32]),
    "gpt-oss-120b-MXFP4": (36, 128, 4, 1, [1, 2], None, "data/tok_gpt-oss-120b.jsonl",
                           [7, 10, 12, 15, 16, 18, 19, 20, 25, 26, 27, 31]),
}
N_PREFILL, N_DECODE_CAL, N_DECODE_TEST = 128, 128, 192


def runs(path):
    """config label -> (step_ms mean, rows)"""
    by = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            by.setdefault(r.get("config", ""), []).append(r)
    return {k: (sum(r["decode_ms"] for r in v) / sum(r["n_decode"] for r in v), v) for k, v in by.items()}


def stats_of(cfg, calib):
    if "stats=" not in cfg:
        return None
    p = os.path.join(calib, os.path.basename(cfg.split("stats=")[-1].split(":")[0]))
    return json.load(open(p)) if os.path.exists(p) else None


def counts_from_stats(s, k):
    """per-step hand-offs r, CPU experts mu, admissions alpha, from a stats file (layers with 0 slots hand off
    all k experts every step; the others per their miss histogram)"""
    steps = s["steps"]
    r = mu = 0.0
    for ly in s["layers"]:
        if ly["C"] == 0:
            r += 1.0
            mu += k
        else:
            h = np.array(ly["miss_hist"], float)
            r += h[1:].sum() / steps
            mu += (np.arange(len(h)) * h).sum() / steps
    return r, mu, s["admits"] / steps


def fit(calib):
    out = {}
    for tag, (L, E, k, K, FR, *_ ) in MODELS.items():
        pe = os.path.join(calib, f"{tag}_ec.jsonl")
        if not os.path.exists(pe):
            continue
        X, y, lab = [], [], []
        for cfg, (ms, _) in runs(pe).items():
            if "mailbox=1" not in cfg:
                continue
            s = stats_of(cfg, calib)
            r, mu, al = counts_from_stats(s, k)
            X.append([1.0, r, mu, al])
            y.append(ms)
            lab.append(cfg.split(":stats")[0])
        X, y = np.array(X), np.array(y)
        # admissions only occur in cache runs; with few points keep beta >= 0 via a bounded refit
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        if coef[3] < 0:
            c2, *_ = np.linalg.lstsq(X[:, :3], y, rcond=None)
            coef = np.array([*c2, 0.0])
        pred = X @ coef
        stock = {}
        for p in glob.glob(os.path.join(calib, f"{tag}_static_n*.jsonl")):
            n = int(p.rsplit("_n", 1)[1].split(".")[0])
            stock[n] = list(runs(p).values())[0][0]
        ns = np.array(sorted(stock))
        A = np.stack([np.ones(len(ns)), ns], 1)
        sc, *_ = np.linalg.lstsq(A, np.array([stock[n] for n in ns]), rcond=None)
        out[tag] = dict(T0=coef[0], a0=coef[1], a1=coef[2], beta=coef[3],
                        fit=[dict(cfg=l, ms=float(a), pred=float(b)) for l, a, b in zip(lab, y, pred)],
                        stock_T0=sc[0], stock_s=sc[1], stock_pts={int(n): stock[n] for n in ns})
    return out


def test_counts(tag, calib):
    """(r, mu, alpha) per cache budget on the test sequences: trace simulation, or the calibration run's counts"""
    L, E, k, K, FR, tdir, corp, test = MODELS[tag]
    res = {}
    if tdir is not None:
        R, E2, _ = load_steps(tdir, corp, test, N_PREFILL, N_DECODE_TEST)
        for q in FR:
            C = E * q // 8
            h, m, a = simulate(R, E2, C, "dfa", kappa=float(K))
            res[q] = (float((m > 0).sum(1).mean()), float(m.sum(1).mean()), float(a.sum(1).mean()), "trace simulation")
    else:
        for q in FR:
            C = E * q // 8
            s = json.load(open(os.path.join(calib, f"{tag}_mb_C{C}.json")))
            res[q] = (*counts_from_stats(s, k), "calibration-run counts (no trace)")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fits = fit(args.calib)
    preds = []
    for tag, f in fits.items():
        L, E, k, K, FR, *_ = MODELS[tag]
        tc = test_counts(tag, args.calib)
        for q in FR:
            C, n = E * q // 8, L - L * q // 8
            r, mu, al, src = tc[q]
            t_mb = f["T0"] + f["a0"] * r + f["a1"] * mu + f["beta"] * al
            t_ml = f["T0"] + f["a0"] * n + f["a1"] * n * k
            t_st = f["stock_T0"] + f["stock_s"] * n
            preds += [dict(model=tag, budget=q / 8, system="mailbox cache", C=C, tok_s=1000 / t_mb, counts=[r, mu, al], counts_from=src),
                      dict(model=tag, budget=q / 8, system="mailbox static layers", n=n, tok_s=1000 / t_ml),
                      dict(model=tag, budget=q / 8, system="llama.cpp static layers", n=n, tok_s=1000 / t_st),
                      dict(model=tag, budget=q / 8, system="speed-up (cache / llama.cpp)", ratio=t_st / t_mb)]
    os.makedirs(args.out, exist_ok=True)
    json.dump(dict(fits=fits, predictions=preds), open(os.path.join(args.out, "predictions.json"), "w"), indent=1, default=float)
    with open(os.path.join(args.out, "predictions.md"), "w") as fo:
        fo.write("# Phase-3 predictions (mailbox cache), from the calibration run on the profile sequences\n\n")
        fo.write("| model | T0 ms | a0 us | a1 us | beta us | stock T0 ms | stock s us | calib fit max APE |\n|---|---|---|---|---|---|---|---|\n")
        for tag, f in fits.items():
            ape = max(abs(p["pred"] - p["ms"]) / p["ms"] for p in f["fit"]) * 100
            fo.write(f"| {tag} | {f['T0']:.2f} | {f['a0']*1000:.0f} | {f['a1']*1000:.0f} | {f['beta']*1000:.0f} | {f['stock_T0']:.2f} | {f['stock_s']*1000:.0f} | {ape:.1f}% |\n")
        fo.write("\n| model | budget | system | predicted |\n|---|---|---|---|\n")
        for p in preds:
            v = f"{p['ratio']:.2f}x" if "ratio" in p else f"{p['tok_s']:.1f} tok/s"
            fo.write(f"| {p['model']} | {p['budget']:.3f} | {p['system']} | {v} |\n")
    print(open(os.path.join(args.out, "predictions.md")).read())


if __name__ == "__main__":
    main()

"""Score the pre-registered predictions against first-party measurements.

    python scripts/analyze_firstparty.py <prereg dir> <sweep results dir> [<counters results dir>]

Implements the analysis fixed in prereg/PROTOCOL.md: H0 (roofline), H1 (ncu
bytes), H2 (median APE of M4), H3 (1/tok_s affine in n_cpu_moe), H4
(intercept/slope decomposition). Writes <prereg dir>/scored.json and .md.
"""
import json
import os
import statistics as st
import sys

import numpy as np


def load_sweep(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "sweep.jsonl")) if l.strip()]
    order = [l.split() for l in open(os.path.join(d, "order.txt")) if l.strip()]
    n_main = len(order)
    meas, drift, failed = {}, [], []
    for r in rows:
        key = (r["model"], str(r["n_cpu_moe"]))
        ts = r.get("samples_ts")
        if not ts:
            failed.append(dict(model=key[0], n_cpu_moe=key[1], rc=r.get("rc"), error=r.get("error", "")[-300:]))
            continue
        v = dict(median=st.median(ts), samples=ts, seq=r["seq"], n_threads=r.get("n_threads"))
        if r["seq"] <= n_main:
            meas[key] = v
        else:
            drift.append(dict(model=key[0], n_cpu_moe=key[1], repeat=v["median"]))
    for d_ in drift:
        first = meas.get((d_["model"], d_["n_cpu_moe"]))
        d_["first"] = first["median"] if first else None
        d_["change"] = (d_["repeat"] / first["median"] - 1) if first else None
    return meas, drift, failed


def main(pdir, sdir, cdir=None):
    pred = json.load(open(os.path.join(pdir, "predictions.json")))
    meas, drift, failed = load_sweep(sdir)
    rows = []
    for p in pred["sweep"]:
        key = (p["model"], str(p["n_cpu_moe"]))
        m = meas.get(key)
        r = dict(model=key[0], n_cpu_moe=key[1], pred=p["pred_tok_s"], pred_M0=p["pred_tok_s_M0"],
                 roofline=p["roofline_tok_s"], meas=m["median"] if m else None)
        if m:
            r["ape"] = abs(r["pred"] - r["meas"]) / r["meas"]
            r["ape_M0"] = abs(r["pred_M0"] - r["meas"]) / r["meas"]
            r["signed_err"] = r["pred"] / r["meas"] - 1
            r["beats_roofline"] = r["meas"] > r["roofline"]
        rows.append(r)
    ok = [r for r in rows if r["meas"] is not None]
    apes = [r["ape"] for r in ok]
    h2 = dict(n=len(ok), median_ape=st.median(apes), mape=float(np.mean(apes)), max_ape=max(apes),
              median_ape_M0=st.median([r["ape_M0"] for r in ok]),
              within_25=float(np.mean([a <= 0.25 for a in apes])), pass_=st.median(apes) <= 0.20,
              per_model={m: st.median([r["ape"] for r in ok if r["model"] == m]) for m in sorted({r["model"] for r in ok})})
    h0 = dict(violations=[(r["model"], r["n_cpu_moe"]) for r in ok if r["beats_roofline"]])
    h0["pass_"] = not h0["violations"]
    # H3/H4: affine fit of 1/tok_s (seconds per token) vs n_cpu_moe, GPU-offload points only
    h3, h4 = {}, {}
    for m in sorted({r["model"] for r in ok}):
        pts = [(int(r["n_cpu_moe"]), 1 / r["meas"], 1 / r["pred"]) for r in ok if r["model"] == m and r["n_cpu_moe"] != "cpu"]
        if len(pts) < 3:
            continue
        x = np.array([p[0] for p in pts], float); y = np.array([p[1] for p in pts]); yp = np.array([p[2] for p in pts])
        A = np.vstack([np.ones_like(x), x]).T
        (b0, b1), *_ = np.linalg.lstsq(A, y, rcond=None)
        (c0, c1), *_ = np.linalg.lstsq(A, yp, rcond=None)
        r2 = 1 - np.sum((y - A @ [b0, b1]) ** 2) / np.sum((y - y.mean()) ** 2)
        h3[m] = dict(r2=float(r2), pass_=bool(r2 >= 0.98), n=len(pts))
        h4[m] = dict(meas_intercept_ms=b0 * 1e3, pred_intercept_ms=c0 * 1e3, intercept_err=c0 / b0 - 1,
                     meas_slope_us_per_layer=b1 * 1e6, pred_slope_us_per_layer=c1 * 1e6, slope_err=c1 / b1 - 1)
    out = dict(platform=pred["platform"], H0=h0, H2=h2, H3=h3, H4=h4, drift=drift, failed=failed, rows=rows)
    if cdir and os.path.exists(os.path.join(cdir, "ncu_summary.json")):
        nc = json.load(open(os.path.join(cdir, "ncu_summary.json")))["per_token"]
        h1 = []
        for p in pred["ncu"]:
            tag = f"{p['model']}_ncmoe{p['n_cpu_moe']}"
            if tag in nc:
                mb = nc[tag]["read_bytes_per_token"]
                h1.append(dict(config=tag, pred_GB=p["pred_gpu_dram_read_bytes_per_token"] / 1e9, meas_GB=mb / 1e9,
                               err=p["pred_gpu_dram_read_bytes_per_token"] / mb - 1,
                               kernels_per_token=nc[tag]["kernels_per_token"]))
        out["H1"] = dict(rows=h1, pass_=bool(h1) and all(abs(r["err"]) <= 0.10 for r in h1))
    json.dump(out, open(os.path.join(pdir, "scored.json"), "w"), indent=1)
    with open(os.path.join(pdir, "scored.md"), "w") as f:
        f.write(f"# Scored against first-party measurements: {pred['platform']['gpu']}\n\n")
        f.write(f"- H0 roofline never beaten: {'PASS' if h0['pass_'] else 'FAIL ' + str(h0['violations'])}\n")
        if "H1" in out:
            f.write(f"- H1 ncu bytes within ±10%: {'PASS' if out['H1']['pass_'] else 'FAIL'}\n")
            for r in out["H1"]["rows"]:
                f.write(f"  - {r['config']}: predicted {r['pred_GB']:.3f} GB, measured {r['meas_GB']:.3f} GB ({100*r['err']:+.1f}%)\n")
        f.write(f"- H2 median APE (M4) {100*h2['median_ape']:.1f}% (≤20%: {'PASS' if h2['pass_'] else 'FAIL'}); "
                f"MAPE {100*h2['mape']:.1f}%, max {100*h2['max_ape']:.1f}%, within 25% {100*h2['within_25']:.0f}%; "
                f"M0 median APE {100*h2['median_ape_M0']:.1f}%\n")
        for m, v in h2["per_model"].items():
            f.write(f"  - {m}: median APE {100*v:.1f}%\n")
        for m, v in h3.items():
            f.write(f"- H3 {m}: R² {v['r2']:.4f} ({'PASS' if v['pass_'] else 'FAIL'})\n")
        for m, v in h4.items():
            f.write(f"- H4 {m}: intercept {v['meas_intercept_ms']:.2f} ms measured vs {v['pred_intercept_ms']:.2f} predicted "
                    f"({100*v['intercept_err']:+.0f}%); slope {v['meas_slope_us_per_layer']:.0f} vs {v['pred_slope_us_per_layer']:.0f} µs/layer "
                    f"({100*v['slope_err']:+.0f}%)\n")
        for d_ in drift:
            if d_["change"] is not None:
                f.write(f"- drift {d_['model']} n={d_['n_cpu_moe']}: {100*d_['change']:+.1f}%\n")
        for x in failed:
            f.write(f"- FAILED {x['model']} n={x['n_cpu_moe']} rc={x['rc']}\n")
        f.write("\n| model | n_cpu_moe | measured tok/s | predicted (M4) | error | M0 | roofline |\n|---|---|---|---|---|---|---|\n")
        for r in rows:
            if r["meas"] is None:
                f.write(f"| {r['model']} | {r['n_cpu_moe']} | — | {r['pred']:.1f} | | {r['pred_M0']:.1f} | {r['roofline']:.1f} |\n")
            else:
                f.write(f"| {r['model']} | {r['n_cpu_moe']} | {r['meas']:.1f} | {r['pred']:.1f} | {100*r['signed_err']:+.0f}% | "
                        f"{r['pred_M0']:.1f} | {r['roofline']:.1f} |\n")
    print(open(os.path.join(pdir, "scored.md")).read())


if __name__ == "__main__":
    main(*sys.argv[1:])

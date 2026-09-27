"""Score the audit anchors (prereg/PROTOCOL.md 4.7): llama.cpp --n-cpu-moe measured by job 051/053 against the
predictions job 050/052 committed first.

A1: no configuration beats the physical floor. A2 (A100): median APE of M4 <= 25 %. A3 (A100): M4 under-predicts
(measured > predicted) on >= 2/3 of configurations. GH200: exploratory. Also: affine fit of time per token in the
number of CPU MoE layers (R^2) and drift of the two repeated configurations.

    python scripts/anchor_score.py --pred <050 dir> --meas <051 dir> --name a100 --out prereg/anchors
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--meas", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = json.load(open(os.path.join(a.pred, "anchor_predictions.json")))
    pred = {(r["model"], int(r["n_cpu_moe"])): r for r in P["sweep"]}
    rows = [json.loads(l) for l in open(os.path.join(a.meas, "sweep.jsonl")) if l.strip()]
    first, repeat, errors = {}, {}, []
    for r in sorted(rows, key=lambda r: r["seq"]):
        key = (r["model"], int(r["n_cpu_moe"]))
        if "avg_ts" not in r:
            errors.append(dict(model=key[0], n_cpu_moe=key[1], error=(r.get("error") or "")[-300:]))
            continue
        (repeat if key in first else first)[key] = r
    cfg = []
    for key, r in first.items():
        p = pred.get(key)
        if not p:
            continue
        m = float(r["avg_ts"])
        cfg.append(dict(model=key[0], n_cpu_moe=key[1], moe_layers_on_cpu=p["moe_layers_on_cpu"], measured=m,
                        measured_sd=float(r.get("stddev_ts", 0)), threads=r.get("n_threads"),
                        pred_m4=p["pred_tok_s"], pred_a10=p["pred_tok_s_a10refit"], floor=p["roofline_tok_s"],
                        ape_m4=abs(p["pred_tok_s"] - m) / m, ape_a10=abs(p["pred_tok_s_a10refit"] - m) / m,
                        ratio_m4=m / p["pred_tok_s"]))
    drift = [dict(model=k[0], n_cpu_moe=k[1], first=float(first[k]["avg_ts"]), repeat=float(v["avg_ts"]),
                  change=float(v["avg_ts"]) / float(first[k]["avg_ts"]) - 1) for k, v in repeat.items()]
    aff = {}
    for model in sorted({c["model"] for c in cfg}):
        cs = [c for c in cfg if c["model"] == model]
        if len(cs) >= 3:
            x = np.array([c["moe_layers_on_cpu"] for c in cs], float)
            y = np.array([1 / c["measured"] for c in cs])
            A = np.vstack([x, np.ones_like(x)]).T
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            yh = A @ coef
            aff[model] = dict(r2=float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                              slope_ms_per_layer=float(coef[0] * 1e3), intercept_ms=float(coef[1] * 1e3), n=len(cs))
    ape = np.array([c["ape_m4"] for c in cfg]) if cfg else np.array([])
    s = dict(platform=P["platform"], n=len(cfg), n_errors=len(errors),
             median_ape_m4=float(np.median(ape)) if cfg else None, max_ape_m4=float(ape.max()) if cfg else None,
             median_ape_a10=float(np.median([c["ape_a10"] for c in cfg])) if cfg else None,
             under_predicted=sum(c["measured"] > c["pred_m4"] for c in cfg),
             ratio_range=[min(c["ratio_m4"] for c in cfg), max(c["ratio_m4"] for c in cfg)] if cfg else None,
             floor_violations=sum(c["measured"] > c["floor"] for c in cfg),
             max_of_floor=max(c["measured"] / c["floor"] for c in cfg) if cfg else None,
             min_r2=min(v["r2"] for v in aff.values()) if aff else None,
             max_drift=max(abs(d["change"]) for d in drift) if drift else None)
    s["A1"] = s["floor_violations"] == 0
    if a.name != "gh200":
        s["A2"] = s["median_ape_m4"] is not None and s["median_ape_m4"] <= 0.25
        s["A3"] = s["under_predicted"] >= 2 * s["n"] / 3
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(summary=s, configs=cfg, affine=aff, drift=drift, errors=errors),
              open(os.path.join(a.out, f"{a.name}.json"), "w"), indent=1)
    L = [f"# Anchors, {a.name}: {P['platform']['gpu']}, B_c {P['platform']['bw_cpu']:.1f} GB/s (STREAM Triad)\n",
         "| model | --n-cpu-moe | measured tok/s | M4 | M4+A10 | floor | measured / M4 |", "|---|---|---|---|---|---|---|"]
    for c in sorted(cfg, key=lambda c: (c["model"], c["n_cpu_moe"])):
        L.append(f"| {c['model']} | {c['n_cpu_moe']} | {c['measured']:.1f} ± {c['measured_sd']:.1f} | {c['pred_m4']:.1f} | "
                 f"{c['pred_a10']:.1f} | {c['floor']:.1f} | {c['ratio_m4']:.2f} |")
    L += ["", "Affine fits (time per token vs CPU MoE layers): " + "; ".join(f"{m}: R^2 {v['r2']:.4f}" for m, v in aff.items()),
          "", "Summary: " + json.dumps(s), "", "Errors: " + json.dumps(errors)[:2000]]
    open(os.path.join(a.out, f"{a.name}.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

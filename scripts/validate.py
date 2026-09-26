"""Fit the model's efficiency/overhead parameters and retrodict published
measurements, with an ablation over model terms.

Cross-validation groups: CPU-offload rows are held out one *source* at a time
(a source = one paper/thread, i.e. one lab's hardware and methodology);
the demand-fetch rows all come from one paper, so they are held out one
*GPU* at a time. Writes results/validation.json and results/validation.csv.
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mosl.archs import shape
from mosl.perfmodel import HW, Params, Workload, static_offload_time, fetch_time
from mosl.validation_set import ROWS, NUMA_ROWS

MEAS = {x["id"]: x for x in json.load(open("data/published_measurements.json"))}
SERVER_BW = 150.0  # GB/s: 2-channel desktops are <= ~110, servers >= 230


def group_of(r):
    if r["kind"] == "fetch":
        return f"fetch:{r['bw_gpu']}:{r['bw_pcie']}"
    return MEAS[r.get("src", r["id"])]["source_url"].split("#")[0]


# name, lower, upper, init
TERMS = {
    "eta_g": (0.05, 1.0, 0.7), "eta_c": (0.05, 1.0, 0.6), "eta_c_srv": (0.05, 1.0, 0.6),
    "eta_p": (0.02, 1.0, 0.7), "tau_us": (0.0, 2000.0, 50.0), "tau_g_us": (0.0, 500.0, 20.0),
    "tau_c_us": (0.0, 2000.0, 50.0), "tau_x_us": (0.0, 50000.0, 1000.0),
}
VARIANTS = {
    "M0 bytes/bandwidth + hand-off": ["eta_g", "eta_c", "eta_p", "tau_us"],
    "M1 + per-transfer latency": ["eta_g", "eta_c", "eta_p", "tau_us", "tau_x_us"],
    "M2 + per-layer fixed costs": ["eta_g", "eta_c", "eta_p", "tau_us", "tau_x_us", "tau_g_us", "tau_c_us"],
    "M3 + server/desktop CPU efficiency": ["eta_g", "eta_c", "eta_c_srv", "eta_p", "tau_us", "tau_x_us", "tau_g_us", "tau_c_us"],
}


def to_params(names, x):
    d = dict(zip(names, x))
    p = Params(**{k: v for k, v in d.items() if k in Params.__dataclass_fields__})
    if "eta_c_srv" in d:
        p.extra["eta_c_srv"] = d["eta_c_srv"]
    return p


def predict(r, p: Params):
    s = shape(r["repo"])
    w = Workload(s, r["b_exp"], r["b_dense"], ctx=r.get("ctx", 512), top_k=r.get("top_k"),
                 dense_on="cpu" if r["kind"] == "cpu_only" else "gpu")
    hw = HW(bw_gpu=r.get("bw_gpu", 1), bw_cpu=r.get("bw_cpu", 1), bw_pcie=r.get("bw_pcie", 1))
    q = Params(**{k: getattr(p, k) for k in Params.__dataclass_fields__ if k != "extra"})
    if "eta_c_srv" in p.extra and r.get("bw_cpu", 0) > SERVER_BW:
        q.eta_c = p.extra["eta_c_srv"]
    if r["kind"] in ("cpu_only", "static"):
        t, parts = static_offload_time(w, hw, q, r.get("n_cpu", s.n_moe_layers))
    else:
        t, parts = fetch_time(w, hw, q, r["fetched"], transfers_per_token=r.get("transfers"))
    return 1.0 / t, parts


def fit(rows, names):
    lo = [TERMS[n][0] for n in names]
    hi = [TERMS[n][1] for n in names]
    x0 = [TERMS[n][2] for n in names]
    res = lambda x: [np.log(predict(r, to_params(names, x))[0] / r["tok_s"]) for r in rows]
    return to_params(names, least_squares(res, x0, bounds=(lo, hi)).x)


def errs(rows, p):
    out = []
    for r in rows:
        pred, parts = predict(r, p)
        out.append(dict(id=r["id"], engine=r["engine"], kind=r["kind"], repo=r["repo"], measured=r["tok_s"],
                        predicted=pred, rel_err=(pred - r["tok_s"]) / r["tok_s"],
                        **{f"t_{k}_ms": v * 1e3 for k, v in parts.items()}))
    return out


def summary(e):
    a = np.abs([x["rel_err"] for x in e])
    return dict(n=len(a), mape=float(a.mean()), median_ape=float(np.median(a)),
                within_15=float((a <= 0.15).mean()), within_25=float((a <= 0.25).mean()), max_ape=float(a.max()))


def loso(rows, names):
    groups = defaultdict(list)
    for r in rows:
        groups[group_of(r)].append(r)
    out = []
    for g, held in groups.items():
        train = [r for r in rows if group_of(r) != g]
        out += errs(held, fit(train, names))
    return out, len(groups)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    report = {"variants": {}}
    best = None
    for vname, names in VARIANTS.items():
        p = fit(ROWS, names)
        e_in = errs(ROWS, p)
        e_cv, ng = loso(ROWS, names)
        rep = dict(params={n: (p.extra[n] if n in p.extra else getattr(p, n)) for n in names},
                   in_sample=summary(e_in), loso=summary(e_cv),
                   loso_by_kind={k: summary([x for x in e_cv if x["kind"] == k]) for k in ("cpu_only", "static", "fetch")},
                   numa_holdout=summary(errs(NUMA_ROWS, p)))
        report["variants"][vname] = rep
        print(f"{vname:40s} k={len(names)}  in-sample median {rep['in_sample']['median_ape']:.3f}  "
              f"LOSO median {rep['loso']['median_ape']:.3f} mape {rep['loso']['mape']:.3f} within25 {rep['loso']['within_25']:.2f}  "
              f"[cpu {rep['loso_by_kind']['cpu_only']['median_ape']:.2f} static {rep['loso_by_kind']['static']['median_ape']:.2f} "
              f"fetch {rep['loso_by_kind']['fetch']['median_ape']:.2f}]")
        if best is None or rep["loso"]["mape"] < best[1]["loso"]["mape"] - 1e-9:
            best = (vname, rep, p, e_in, e_cv)
    vname, rep, p, e_in, e_cv = best
    report["selected"] = vname
    report["n_rows"] = len(ROWS)
    report["n_groups"] = len({group_of(r) for r in ROWS})
    # per-engine CPU efficiency under the selected variant (explanatory only)
    names = VARIANTS[vname]
    eng = {}
    for e in sorted({r["engine"] for r in ROWS if r["kind"] != "fetch"}):
        sub = [r for r in ROWS if r["engine"] == e]
        base = dict(zip(names, [rep["params"][n] for n in names]))
        def res(x):
            q = to_params(names, [base[n] for n in names]); q.eta_c = x[0]; q.extra.pop("eta_c_srv", None)
            return [np.log(predict(r, q)[0] / r["tok_s"]) for r in sub]
        eng[e] = dict(n=len(sub), eta_c=float(least_squares(res, [0.6], bounds=([0.05], [1.0])).x[0]))
    report["per_engine_eta_c"] = eng
    json.dump(report, open("results/validation.json", "w"), indent=1, default=float)
    e_numa = errs(NUMA_ROWS, p)
    cv = {x["id"]: x["predicted"] for x in e_cv}
    keys = []
    for x in e_in + e_numa:
        keys += [k for k in x if k not in keys]
    with open("results/validation.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys + ["predicted_loso", "rel_err_loso"], restval="")
        wr.writeheader()
        for x in e_in:
            wr.writerow({**x, "predicted_loso": cv[x["id"]], "rel_err_loso": (cv[x["id"]] - x["measured"]) / x["measured"]})
        for x in e_numa:
            wr.writerow({**x, "kind": "numa_holdout"})
    print("selected:", vname, json.dumps(rep["params"], default=lambda v: round(float(v), 4)))
    print("per-engine eta_c:", {k: round(v["eta_c"], 3) for k, v in eng.items()})
    print("numa holdout:", rep["numa_holdout"])

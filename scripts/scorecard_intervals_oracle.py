"""Intervals for the timed clauses of the oracle jobs (094, 095) that prereg/scorecard_clauses.json carried without one.

Paired bootstrap over the 30 sequences of each run (10,000 resamples, seed 0) of the per-sequence token rates in
prereg/foresight_094.json and prereg/foresight_095.json:

    ratio of means between two runs            (both2 / hitopt; both3p / both2)
    a run's share of the limit                 (mean tok/s over the limit's tok/s)
    the law's error on a run's own counters    (law_ms over 1000 / mean tok/s, minus one; the counters are fixed)
    the share of the W=all gain kept at W=16   ((mean16 / base - 1) / (meanall / base - 1))

Writes the interval into each clause's "ci" field, re-scores the clause under the scorecard's rule, and records the
source. Run before scripts/scorecard.py.

    python scripts/scorecard_intervals_oracle.py
"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
N_BOOT, SEED = 10000, 0


def seqs(run):
    return np.array([run["tok_s"][k] for k in sorted(run["tok_s"], key=lambda x: int(x) if str(x).isdigit() else x)])


def boot(stat, arrays):
    rng = np.random.default_rng(SEED)
    n = len(arrays[0])
    vals = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        vals.append(stat(*[a[idx] for a in arrays]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cell_by_label(d, label):
    return next(c for c in d["cells"] if c["label"] == label)


def main():
    d94 = json.load(open(P("prereg", "foresight_094.json")))
    d95 = json.load(open(P("prereg", "foresight_095.json")))
    cl = json.load(open(P("prereg", "scorecard_clauses.json")))
    # clause id -> (job data, cell label, function of the resampled sequences -> the clause's quantity)
    spec = {}
    for lab in ("gpt-oss 11%", "Qwen3 25%"):
        cid = {"gpt-oss 11%": "094-P6a", "Qwen3 25%": "094-P6e"}[lab]
        spec[cid] = (d94, lab, ("base", "bypass_w16", "bypass_w0"), lambda b, w16, wa: 100 * (w16.mean() / b.mean() - 1) / (wa.mean() / b.mean() - 1))
    for cid, lab in (("095-P2g", "gpt-oss 11%"), ("095-P2j", "Qwen3 12.5%")):
        spec[cid] = (d95, lab, ("fetch",), None)          # the law's error: handled below (needs law_ms)
    for cid, lab in (("095-P4a", "gpt-oss 11%"), ("095-P4c", "Qwen3 12.5%")):
        spec[cid] = (d95, lab, ("both2",), None)          # share of the limit: handled below (needs the limit)
    for cid, lab in (("095-P4e", "gpt-oss 11%"), ("095-P4f", "gpt-oss 25%"), ("095-P4g", "Qwen3 12.5%"), ("095-P4h", "Qwen3 25%")):
        spec[cid] = (d95, lab, ("both2", "hitopt"), lambda a, b: a.mean() / b.mean())
    for cid, lab in (("095-P5b", "gpt-oss 25%"), ("095-P5d", "Qwen3 25%")):
        spec[cid] = (d95, lab, ("both2",), None)          # the law's error on both2
    for cid, lab in (("095-P7a", "gpt-oss 11%"), ("095-P7c", "gpt-oss 40%"), ("095-P7d", "Qwen3 12.5%"), ("095-P7f", "Qwen3 43.75%")):
        spec[cid] = (d95, lab, ("both3p", "both2"), lambda a, b: 100 * (a.mean() / b.mean() - 1))
    out = {}
    for cid, (d, lab, runs, fn) in spec.items():
        c = cell_by_label(d, lab)
        arrays = [seqs(c["runs"][r]) for r in runs]
        if fn is None and (cid.startswith("095-P2") or cid.startswith("095-P5")):
            law = c["runs"][runs[0]]["law_ms"]
            fn = lambda a, law=law: 100 * (law / (1e3 / a.mean()) - 1)        # noqa: E731
        elif fn is None and cid.startswith("095-P4"):
            lim = c["model_terms"]["limit_tok_s"]
            fn = lambda a, lim=lim: 100 * a.mean() / lim                     # noqa: E731
        point = fn(*arrays)
        lo, hi = boot(fn, arrays)
        out[cid] = (point, lo, hi)
    # write into the clause file and re-score under the rule
    import re

    def inside(lo, hi, thr):
        t = thr.replace("%", "").replace(" ", "")
        if t.startswith(">="):
            return lo >= float(t[2:])
        if t.startswith(">"):
            return lo > float(t[1:])
        m = re.match(r"^([+-]?[0-9.]+)(?:to|-)([+-]?[0-9.]+)$", t)
        a, b = float(m.group(1)), float(m.group(2))
        return a <= lo and hi <= b
    changed = []
    for job in cl["jobs"]:
        for c in job["clauses"]:
            if c["id"] in out:
                point, lo, hi = out[c["id"]]
                dec = c.get("decimals", 1)
                c["ci"] = [round(lo, dec), round(hi, dec)]
                c["source"] = c["source"] + "; interval by scripts/scorecard_intervals_oracle.py (paired bootstrap over the 30 sequences, 10,000 resamples, seed 0)"
                old = c["status"]
                held_point = old in ("held", "held (point)")
                if held_point:
                    c["status"] = "held" if inside(lo, hi, c["threshold"]) else "held (point)"
                changed.append((c["id"], round(point, 2), c["ci"], c["threshold"], old, c["status"]))
    json.dump(cl, open(P("prereg", "scorecard_clauses.json"), "w"), indent=1)
    for row in changed:
        print(row)


if __name__ == "__main__":
    main()

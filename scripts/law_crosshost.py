"""Score the pre-registered per-host predictions of the host-DRAM law (jobs 069c, 072, ...: law_prediction.json written
on the machine from its own bandwidth probe before any model run) against the speeds measured afterwards.

    python scripts/law_crosshost.py [--out prereg/homepc/law_crosshost.json]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

RES = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
# measured stats file (by prefix) -> prediction key
MAP = [
    (r"ref_slots_14\.json|ref_C14_r\d\.json", "C14"),
    (r"ref_slots_32\.json|ref_C32_r\d\.json", "C32"),
    (r"ref_slots_56\.json|ref_C56_r\d\.json", "C56"),
    (r"ref_slots_32_f\.json|ref_C32f_r\d\.json", "C32_fetch"),
    (r"ref_slots_32_f_pf\.json", "C32_fetch_pf"),
    (r"ref_C32pf_r\d\.json", "C32_pf"),
]


def rates(p):
    by = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            by.setdefault(r.get("config", ""), []).append(r)
    return {k: sum(r["n_decode"] for r in v) / (sum(r["decode_ms"] for r in v) / 1000) for k, v in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    allrows = []
    for d in sorted(glob.glob(f"{RES}/*")):
        pp = os.path.join(d, "law_prediction.json")
        if not os.path.exists(pp):
            continue
        pred = json.load(open(pp))
        cpu = ""
        if os.path.exists(os.path.join(d, "lscpu.txt")):
            m = re.search(r"Model name:\s*(.+)", open(os.path.join(d, "lscpu.txt")).read())
            cpu = m.group(1).strip() if m else ""
        meas = {}
        if os.path.exists(os.path.join(d, "ref_ec.jsonl")):
            for k, v in rates(os.path.join(d, "ref_ec.jsonl")).items():
                name = k.split("stats=")[-1].split("/")[-1] if "stats=" in k else ""
                for pat, key in MAP:
                    if re.fullmatch(pat, name):
                        meas.setdefault(key, []).append(v)
        if os.path.exists(os.path.join(d, "ref_static27.jsonl")):
            v = list(rates(os.path.join(d, "ref_static27.jsonl")).values())
            if v:
                meas["static27"] = v
        print(f"== {os.path.basename(d)}: {cpu}; B_c {pred['B_c']} B_p {pred['B_p']} B_both {pred['B_both']} GB/s, "
              f"helpers {pred['helpers']}, predicted {pred['time_utc']}")
        errs = []
        for key, vs in meas.items():
            p = pred["predicted_tok_s"].get(key)
            if p is None:
                continue
            m = float(np.mean(vs))
            e = p / m - 1
            errs.append((key, e))
            allrows.append(dict(job=os.path.basename(d), cpu=cpu, config=key, predicted=p, measured=m, err=e,
                                engine="llama.cpp" if key.startswith("static") else "cache"))
            print(f"   {key:14s} predicted {p:6.1f}  measured {m:6.1f}  err {100 * e:+6.1f}%")
        ce = [abs(e) for k, e in errs if not k.startswith("static")]
        if ce:
            print(f"   cache configurations: median |err| {100 * np.median(ce):.1f}%, max {100 * max(ce):.1f}% (n={len(ce)})")
    ce = [abs(r["err"]) for r in allrows if r["engine"] == "cache"]
    if ce:
        print(f"ALL cache rows: n={len(ce)} median |err| {100 * np.median(ce):.1f}%, 90th pct {100 * np.percentile(ce, 90):.1f}%, max {100 * max(ce):.1f}%")
    if a.out:
        json.dump(allrows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

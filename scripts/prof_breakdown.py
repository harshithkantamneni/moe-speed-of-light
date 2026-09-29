"""Per-token time breakdown of offloaded MoE decode from an Nsight Systems export (jobs 069 / 070: `prof_<label>_cuda_gpu_trace.csv.gz`
and `prof_<label>_cuda_api_trace.csv.gz`, CUDA graphs traced per node).

Decode tokens are delimited by the host's graph launches (cudaGraphLaunch; eager launches otherwise). For the last
`--tokens` decode tokens of the run, every GPU kernel and memcpy is put in one category by name, and the token's wall
time (launch to next launch) is split into: GPU busy per category, of which `ec_wait` is the GPU spinning while the
CPU helpers compute (the CPU-miss critical path), and idle (no GPU activity at all: host-side gaps, e.g. llama.cpp's
CPU splits). Intervals are merged per token, so overlapping kernels (the prefetch side stream) are not double counted
in `busy`; the category sums are per-kernel sums.

    python scripts/prof_breakdown.py --dir /home/claude/gpu-branch/results/069_profile_5090@vast --label C32
"""
import argparse
import csv
import gzip
import json
import os
import re

import numpy as np

CATS = [
    ("ec_wait", r"ec_wait_kernel"),
    ("ec_copy", r"ec_fetch_copy_kernel|zc_copy|ec_zc|copy_kernel"),
    ("ec_ctl", r"ec_request_kernel|ec_fetch_plan|ec_prefetch_plan"),
    ("expert_gemv", r"(mul_mat_vec|mmvq|mmvf|mul_mat_q|mmq|mmf).*(_id|ids)|mul_mat_id|moe"),
    ("attention", r"flash_attn|fattn|soft_max|rope"),
    ("dense_gemv", r"mul_mat_vec|mmvq|mmvf|gemv|gemm|cutlass|cublas|mul_mat"),
    ("topk_router", r"argsort|topk|top_k"),
    ("small_ops", r"rms_norm|norm|add|mul|glu|swiglu|scale|cpy|get_rows|set_rows|concat|unary|bin_bcast|k_"),
    ("memcpy", r"\[CUDA memcpy|\[CUDA memset|memcpy|memset"),
]


def load_csv(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return list(csv.DictReader(f))


def col(row, *names):
    for n in names:
        for k in row:
            if k.strip().lower().startswith(n.lower()):
                return row[k]
    raise KeyError(names)


def category(name):
    for c, pat in CATS:
        if re.search(pat, name, re.I):
            return c
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tokens", type=int, default=20)
    ap.add_argument("--json")
    ap.add_argument("--gap-us", type=float, default=40.0)
    a = ap.parse_args()
    gpu = load_csv(os.path.join(a.dir, f"prof_{a.label}_cuda_gpu_trace.csv.gz"))
    ev = []
    for r in gpu:
        s = int(col(r, "Start")); d = int(col(r, "Duration")); n = col(r, "Name")
        ev.append((s, s + d, n))
    ev.sort()
    api_path = os.path.join(a.dir, f"prof_{a.label}_cuda_api_trace.csv.gz")
    launches = []
    if os.path.exists(api_path):
        api = load_csv(api_path)
        launches = sorted(int(col(r, "Start")) for r in api if re.search(r"cudaGraphLaunch", col(r, "Name")))
        kind = "graph launches"
    if len(launches) < a.tokens + 2:
        # no API trace: a decode token starts at the first GPU activity after a host gap longer than --gap-us
        kind = f"GPU gaps > {a.gap_us} us"
        launches, last_end = [], None
        for s, e, n in ev:
            if last_end is not None and s - last_end > a.gap_us * 1000:
                launches.append(s)
            last_end = e if last_end is None else max(last_end, e)
    # token boundaries: use the last tokens+1 launches (the decode tail of the last sequence)
    bounds = launches[-(a.tokens + 1):]
    per = []
    for t0, t1 in zip(bounds[:-1], bounds[1:]):
        cats = {}
        iv = []
        for s, e, n in ev:
            if e <= t0 or s >= t1:
                continue
            s2, e2 = max(s, t0), min(e, t1)
            c = category(n)
            cats[c] = cats.get(c, 0) + (e2 - s2)
            iv.append((s2, e2))
        busy, cur_s, cur_e = 0, None, None
        for s, e in sorted(iv):
            if cur_e is None or s > cur_e:
                if cur_e is not None:
                    busy += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        if cur_e is not None:
            busy += cur_e - cur_s
        per.append(dict(wall=t1 - t0, busy=busy, idle=(t1 - t0) - busy, **cats))
    keys = ["wall", "busy", "idle"] + [c for c, _ in CATS] + ["other"]
    med = {k: float(np.median([p.get(k, 0) for p in per])) / 1e6 for k in keys}
    print(f"{a.label}: {len(per)} tokens ({kind}); medians in ms per token")
    for k in keys:
        print(f"   {k:12s} {med[k]:7.3f}")
    names = {}
    t0, t1 = bounds[0], bounds[-1]
    for s, e, n in ev:
        if s >= t0 and e <= t1:
            short = re.sub(r"<.*", "", n)[:60]
            names[short] = names.get(short, 0) + (e - s)
    top = sorted(names.items(), key=lambda x: -x[1])[:15]
    print("   top kernels (ms per token):")
    for n, d in top:
        print(f"     {d / 1e6 / len(per):7.3f}  {n}")
    if a.json:
        json.dump(dict(label=a.label, kind=kind, median_ms=med, top=[(n, d / 1e6 / len(per)) for n, d in top]), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

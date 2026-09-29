"""Checkpoint 2 (job 059, RTX 5090 + Ryzen 9 9950X): measured host constants, each system's decode speed, and its
fraction of the speed-of-light on this host, from the job's results directory.

    python scripts/analyze_homepc.py --res /home/claude/gpu-branch/results/059_ckpt2_homepc@vast [--out prereg/homepc/ckpt2.json]

Speed-of-light (Proposition 1, per-layer budget, MIN-bypass misses from the gpt-oss-120b own-text trace): physical
(every efficiency 1) with this host's *measured* bandwidths: GPU device read (bw.txt), host read at the helpers'
thread count (concur.txt), pinned host-to-device copy (concur.txt).
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl.calc import PHYS, mstar, shape  # noqa: E402
from mosl.perfmodel import HW, Workload, speed_of_light_time  # noqa: E402
from mosl.traces import load_pack  # noqa: E402

TRACE = "/home/claude/gpu-branch/results/036_trace_retry@40gb/gpt-oss-120b_S.npz"
BUDGETS = ((32, 14), (27, 32), (20, 56))   # llama.cpp -ncmoe n  <->  cache slots per layer C (of 128)


def read(p):
    return open(p, errors="replace").read() if os.path.exists(p) else ""


def constants(res):
    c = {}
    t = read(f"{res}/concur.txt")
    cpu = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"cpu_read_gbs t=(\d+) ([\d.]+)", t)}
    c["cpu_read_gbs"] = cpu
    ce = [float(x) for x in re.findall(r"pcie_copyengine_gbs chunk=16MB ([\d.]+)", t)]
    zc = [float(x) for x in re.findall(r"pcie_zerocopy_gbs chunk=16MB blocks=\d+ ([\d.]+)", t)]
    c["pcie_copyengine_gbs"] = max(ce) if ce else None
    c["pcie_zerocopy_gbs"] = max(zc) if zc else None
    c["concurrent"] = [dict(t=int(a), mode=b, cpu=float(x), pcie=float(y), total=float(z)) for a, b, x, y, z in
                       re.findall(r"concurrent t=(\d+) pcie=(\S+) cpu_gbs ([\d.]+) pcie_gbs ([\d.]+) sum ([\d.]+)", t)]
    c["fetch_us"] = [dict(size_mb=float(a), k=int(b), mode=m, us=float(u)) for a, b, m, u in
                     re.findall(r"fetch_us size=([\d.]+)MB k=(\d+) mode=(\S+) ([\d.]+)", t)]
    b = read(f"{res}/bw.txt")
    m = re.search(r"device read 1 GiB:\s+([\d.]+) GB/s", b)
    c["gpu_read_gbs"] = float(m.group(1)) if m else None
    s = read(f"{res}/stream.txt")
    tri = [float(x) for x in re.findall(r"Triad:\s+([\d.]+)", s)]
    c["stream_triad_best_mbs"] = max(tri) if tri else None
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    res = a.res
    c = constants(res)
    summ = read(f"{res}/summary.txt")
    rows = {}
    for n, C in BUDGETS:
        m = re.search(rf"ncmoe {n} / C {C}: (.*)", summ)
        line = m.group(1) if m else ""
        def num(pat):
            mm = re.search(pat, line)
            return float(mm.group(1)) if mm else None
        rows[n] = dict(C=C, ours=num(r"ours\s+([\d.]+) tok/s"), static=num(r"static\(ec-bench\)\s+([\d.]+)"),
                       llamabench=num(r"llama-bench best\s+([\d.]+)"))
        mf = re.search(rf"ncmoe {n} / C {C}:.*\n\s+\+ fetch:\s+([\d.]+) tok/s", summ)
        rows[n]["fetch"] = float(mf.group(1)) if mf else None
    # speed-of-light on this host (measured bandwidths)
    helpers = max(c["cpu_read_gbs"]) if c["cpu_read_gbs"] else None
    bw_cpu = max(c["cpu_read_gbs"].values()) if c["cpu_read_gbs"] else None
    bw_pcie = max(x for x in (c["pcie_copyengine_gbs"], c["pcie_zerocopy_gbs"]) if x) if (c["pcie_copyengine_gbs"] or c["pcie_zerocopy_gbs"]) else None
    out = dict(constants=c, rows=rows)
    if c["gpu_read_gbs"] and bw_cpu and bw_pcie:
        s = shape("openai/gpt-oss-120b")
        w = Workload(s, 4.25, 16, ctx=640)
        hw = HW(c["gpu_read_gbs"], bw_cpu, bw_pcie)
        pk = load_pack(TRACE)
        idx = np.concatenate([np.arange(st + P, st + nn) for st, nn, P in zip(pk["starts"], pk["seq_lens"], pk["prompt_lens"]) if nn - P > 1])
        R = pk["routes"][:, idx]
        for n, C in BUDGETS:
            ms = mstar(R, s.n_experts, w.k, C)
            sol = 1 / speed_of_light_time(w, hw, PHYS, ms)
            r = rows[n]
            r["sol_tok_s"] = sol
            for k in ("ours", "fetch", "static", "llamabench"):
                if r.get(k):
                    r[f"{k}_of_sol"] = r[k] / sol
        out["sol_basis"] = dict(bw_gpu=c["gpu_read_gbs"], bw_cpu=bw_cpu, bw_pcie=bw_pcie, trace=TRACE, helpers_threads=helpers)
    print(json.dumps(c, indent=1)[:3000])
    print(f"{'ncmoe':>6} {'C':>3} {'ours':>7} {'+fetch':>7} {'static':>7} {'llama-bench':>11} {'SoL':>7}   fractions of SoL (ours / fetch / best llama.cpp)")
    for n, C in BUDGETS:
        r = rows[n]
        best = max(x for x in (r.get("static") or 0, r.get("llamabench") or 0))
        f = lambda x: f"{x:7.1f}" if x else "     --"
        fr = lambda x: f"{100 * x / r['sol_tok_s']:.0f}%" if x and r.get("sol_tok_s") else "--"
        print(f"{n:>6} {C:>3} {f(r.get('ours'))} {f(r.get('fetch'))} {f(r.get('static'))} {f(r.get('llamabench')):>11} {f(r.get('sol_tok_s'))}   "
              f"{fr(r.get('ours'))} / {fr(r.get('fetch'))} / {fr(best)}")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

"""The host-DRAM law for offloaded MoE decode on a PC, tested on every gpt-oss-120b configuration measured on the RTX
5090 hosts (jobs 059, 064, 066, 068).

    T_token = G + max( cpu_bytes / B_c , link_bytes / B_p , (cpu_bytes + link_bytes) / B_both )     (overlapped)
    T_token = G + cpu_bytes / B_c + link_bytes / B_p                                                 (serial FETCH)

cpu_bytes: routed-expert bytes the CPU reads from host DRAM per token (misses run by the helpers, or every expert of the
CPU layers for llama.cpp --n-cpu-moe); link_bytes: expert bytes copied to the GPU in the token (FETCH and PREFETCH;
background admissions are left out: they are issued after the step and overlap the next step's GPU work). The
bandwidths are NOT fitted: they are the host's own microbenchmarks (concur.cu: CPU read at the helpers' thread count,
the zero-copy kernel's link rate, and the sum when both run). G, the GPU-side time per token that nothing overlaps, is
the only fitted constant: one value for the expert cache and one for stock llama.cpp, fitted on the 9950X host (job
066 + 064) and then used unchanged on the other two hosts.

    python scripts/law_hostdram.py [--out prereg/homepc/law_hostdram.json]
"""
import argparse
import json
import os
import re

import numpy as np

RES = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
S = 13253760.0    # bytes per gpt-oss-120b expert (GGUF)
L = 36


def concur(d):
    t = open(f"{RES}/{d}/concur.txt").read()
    cpu = {int(a): float(b) for a, b in re.findall(r"cpu_read_gbs t=(\d+) ([\d.]+)", t)}
    zc = [float(x) for x in re.findall(r"pcie_zerocopy_gbs chunk=16MB blocks=64 ([\d.]+)", t)]
    both = [float(z) for z in re.findall(r"concurrent t=\d+ pcie=zerocopy64 cpu_gbs [\d.]+ pcie_gbs [\d.]+ sum ([\d.]+)", t)]
    return cpu, max(zc), float(np.median(both))


def rates(p):
    by = {}
    for line in open(p):
        if line.strip():
            r = json.loads(line)
            by.setdefault(r.get("config", ""), []).append(r)
    return {k: sum(r["n_decode"] for r in v) / (sum(r["decode_ms"] for r in v) / 1000) for k, v in by.items()}


def cache_rows(d, jsonl, host):
    out = []
    for k, tok in rates(f"{RES}/{d}/{jsonl}").items():
        if "stats=" not in k:
            continue
        name = k.split("stats=")[1].split(":")[0].split("/")[-1]
        st = json.load(open(f"{RES}/{d}/{name}"))
        serial = "fetch_overlap=0" in k or (d.startswith("059") and "fetch=" in k)
        out.append(dict(host=host, job=d, name=name[:-5], tok=tok, kind="cache",
                        cpu=(st["misses"] - st["fetches"]) / st["steps"] * S, link=(st["fetches"] + st.get("prefetches", 0)) / st["steps"] * S,
                        serial=serial))
    return out


def static_rows(host, d, pairs):
    return [dict(host=host, job=d, name=f"static_n{n}", tok=tok, kind="static", cpu=n * 4 * S, link=0.0, serial=False) for n, tok in pairs]


def dram_time(r, bw):
    bc, bp, bb = bw
    if r["serial"]:
        return r["cpu"] / bc + r["link"] / bp
    return max(r["cpu"] / bc, r["link"] / bp, (r["cpu"] + r["link"]) / bb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    a = ap.parse_args()
    hosts = {}
    c1, p1, b1 = concur("059_ckpt2_homepc@vast")
    hosts["9950X #1 (job 059)"] = (np.mean([c1[12], c1[16]]) * 1e9, p1 * 1e9, b1 * 1e9)
    c2, p2, b2 = concur("066_prefetch_homepc@vast")
    hosts["9950X #2 (jobs 064/066)"] = (np.mean([c2[12], c2[16]]) * 1e9, p2 * 1e9, b2 * 1e9)
    c3, p3, b3 = concur("068_ours_9800x3d@vast")
    hosts["9800X3D (job 068)"] = (np.mean([c3[4], c3[8]]) * 1e9, p3 * 1e9, b3 * 1e9)   # 6 helper threads
    rows = []
    rows += cache_rows("066_prefetch_homepc@vast", "ab_C14.jsonl", "9950X #2 (jobs 064/066)")
    rows += cache_rows("066_prefetch_homepc@vast", "ab_C32.jsonl", "9950X #2 (jobs 064/066)")
    rows += cache_rows("066_prefetch_homepc@vast", "ab_C56.jsonl", "9950X #2 (jobs 064/066)")
    rows += static_rows("9950X #2 (jobs 064/066)", "064_samehost_rerun@vast", ((32, 24.1), (27, 28.0), (20, 36.4)))
    rows += cache_rows("059_ckpt2_homepc@vast", "ec.jsonl", "9950X #1 (job 059)")
    rows += cache_rows("059_ckpt2_homepc@vast", "ecf.jsonl", "9950X #1 (job 059)")
    rows += static_rows("9950X #1 (job 059)", "059_ckpt2_homepc@vast", ((32, 30.4), (27, 36.2), (20, 46.4)))
    rows += cache_rows("068_ours_9800x3d@vast", "ot_ec.jsonl", "9800X3D (job 068)")
    for r in rows:
        r["T"] = 1.0 / r["tok"]
        r["dram"] = dram_time(r, hosts[r["host"]])
    fit_host = "9950X #2 (jobs 064/066)"
    G = {}
    for kind in ("cache", "static"):
        sel = [r for r in rows if r["host"] == fit_host and r["kind"] == kind]
        G[kind] = float(np.median([r["T"] - r["dram"] for r in sel]))
    print("bandwidths (GB/s, measured, not fitted): " + "; ".join(f"{h}: CPU {v[0]/1e9:.1f}, link {v[1]/1e9:.1f}, both {v[2]/1e9:.1f}" for h, v in hosts.items()))
    print(f"fitted on {fit_host}: G_cache {G['cache']*1e3:.2f} ms, G_llama.cpp {G['static']*1e3:.2f} ms")
    for h in hosts:
        errs = []
        print(f"== {h}" + ("  (fit host)" if h == fit_host else "  (out of sample)"))
        for r in [r for r in rows if r["host"] == h]:
            pred = 1.0 / (G[r["kind"]] + r["dram"])
            r["pred"] = pred
            errs.append(pred / r["tok"] - 1)
            print(f"   {r['name']:22s} measured {r['tok']:6.1f}  law {pred:6.1f} tok/s  err {100*errs[-1]:+5.1f}%   host-DRAM MB/token cpu {r['cpu']/1e6:5.0f} link {r['link']/1e6:5.0f}{'  (serial)' if r['serial'] else ''}")
        e = np.abs(errs)
        print(f"   n={len(e)} median |err| {100*np.median(e):.1f}%  max {100*e.max():.1f}%")
    if a.out:
        json.dump(dict(hosts={h: [x / 1e9 for x in v] for h, v in hosts.items()}, G=G, rows=rows), open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

"""The speed limit for Table 1: the fastest any exact system can decode on this machine with C expert slots per layer.

Inputs: a routing trace of the measured workload (ec-bench --lookahead on the model's own greedy text; the cache is
carried across problems, cold start), the machine's measured host-memory read rates (concur.txt), and the GPU's
datasheet memory bandwidth.

Per decode token, a system with C slots per layer serves each of the L*k routed experts either from a slot (the GPU
reads S bytes from VRAM) or not (the expert's S bytes are read from host memory, by the CPU that runs it or by the
copy that brings it to the GPU). Two facts bound the time per token:
  host   every exact policy reads at least R_opt experts per token from host memory, where R_opt is the per-layer
         Belady MIN with bypass on the trace (the fewest reads of any policy, even one that knows the future); and an
         expert run on the CPU is always one of those reads, so host reads >= max(R_opt, c) with c experts on the CPU;
  GPU    the GPU reads the dense weights, output head and KV cache (D bytes) plus every expert it runs, (L*k - c) * S.
With perfect overlap of everything, T >= min over c of max((D + (L*k - c) * S) / B_gpu, max(R_opt, c) * S / B_host).
B_host is the highest host-memory read rate measured on the machine by any method (CPU threads, zero-copy PCIe,
copy engine, or two at once), B_gpu the datasheet rate. Assumptions: an expert run on the CPU is read from DRAM
(its reuse distance is hundreds of MB, far beyond the CPU caches); tokens are decoded one at a time.

The same formula with the reads of the best online policy we know (decayed frequency, a miss is either copied and
admitted or run on the CPU, one read per miss; kappa 0 or 1, fewer reads kept) gives the best-online reference: what
perfect overlap would allow without foresight. The deployed policy's reads (decayed frequency, CPU runs every miss,
background admissions) are reported as a check of the trace against the engine's own counters.

    python scripts/speed_limit.py --la LA.bin --model gpt-oss-120b --budgets 14,32,51 --concur concur.txt [--json out]
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl import ecsim_fast  # noqa: E402
from scripts.foresight import INF, _pol  # noqa: E402
from scripts.sim_prefetch import load  # noqa: E402

MODELS = {
    # S: bytes per expert; D: dense + head + KV bytes read per decode token; B_gpu: RTX 5090 datasheet
    "gpt-oss-120b": dict(S=13253760.0, D=1070339328.0 + 615329280.0 + 28311552.0, kappa=1.0),
    # Qwen3-30B-A3B BF16: attention (q 2048x4096, k/v 2048x512, o 4096x2048) + router (2048x128) per layer x 48,
    # head 151936x2048, 2 bytes each; KV 48 x 2 x 512 x 2 bytes x ~260 positions
    "qwen3-30b-a3b-bf16": dict(S=9437184.0, D=2 * (48 * (8388608 + 2 * 1048576 + 8388608 + 262144) + 311164928)
                               + 48 * 2 * 512 * 2 * 260, kappa=1.0),
}
B_GPU = 1792e9


def host_rates(txt):
    v = [float(x) for x in re.findall(r"cpu_read_gbs t=\d+ ([\d.]+)", txt)]
    v += [float(x) for x in re.findall(r"pcie_\w+_gbs [^\n]*? ([\d.]+)\n", txt)]
    v += [float(x) for x in re.findall(r"sum ([\d.]+)", txt)]
    return max(v) * 1e9


def limit_two_path(R, Lk, S, D, b_c, b_p, b_cp, b_gpu=B_GPU):
    """the bound with the host reads split feasibly between the two paths: the CPU reads the c experts it runs (at most
    b_c), the link carries the rest of MIN's reads (at most b_p), both together at most b_cp; seconds per token"""
    c = np.linspace(0, Lk, 20001)
    host = np.maximum.reduce([c * S / b_c, np.maximum(R - c, 0) * S / b_p, np.maximum(R, c) * S / b_cp])
    t = np.maximum((D + (Lk - c) * S) / b_gpu, host)
    i = int(np.argmin(t))
    return float(t[i]), float(c[i])


def limit(R, Lk, S, D, b_host, b_gpu=B_GPU):
    """min over c of max(GPU time, host time), seconds per token"""
    c = np.linspace(0, Lk, 20001)
    t = np.maximum((D + (Lk - c) * S) / b_gpu, np.maximum(R, c) * S / b_host)
    i = int(np.argmin(t))
    return float(t[i]), float(c[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--la", help="ec-bench --lookahead file")
    ap.add_argument("--npy-dir", help="or a trace directory of layerNNN.npy [T, k] arrays (preliminary checks only)")
    ap.add_argument("--npz", help="or the router's choices kept by jobs 084b/084c (route_aime25_*.npz: act [T, L, k] uint8)")
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--budgets", required=True)
    ap.add_argument("--concur", required=True)
    ap.add_argument("--seqs", help="first N sequences only")
    ap.add_argument("--json")
    a = ap.parse_args()
    m = MODELS[a.model]
    if a.npz:
        z = np.load(a.npz)
        d = dict(L=int(z["n_layer"]), k=int(z["k"]), E=int(z["n_expert"]), act=z["act"].astype(np.int64), seq=z["seq"].astype(int))
    elif a.la:
        d = load(a.la)
    else:
        fs = sorted(f for f in os.listdir(a.npy_dir) if re.fullmatch(r"layer\d+\.npy", f))
        act = np.stack([np.load(os.path.join(a.npy_dir, f)).astype(np.int64) for f in fs], axis=1)
        meta = json.load(open(os.path.join(a.npy_dir, "meta.json")))
        lay = meta["layers"][min(meta["layers"], key=int)]
        d = dict(L=act.shape[1], k=act.shape[2], E=int(lay["num_experts"]), act=act, seq=np.zeros(len(act), int))
    L, E, k = d["L"], d["E"], d["k"]
    act = d["act"]
    if a.seqs:
        keep = np.isin(d["seq"], np.unique(d["seq"])[:int(a.seqs)])
        act = act[keep]
    T = act.shape[0]
    b_host = host_rates(open(a.concur).read())
    out = dict(model=a.model, trace=a.npz or a.la or a.npy_dir, T=T, L=L, E=E, k=k, S=m["S"], D=m["D"], B_host=b_host, B_gpu=B_GPU, rows={})
    print(f"{a.model}: {T} decode steps, {L} layers x top-{k} of {E}; B_host {b_host / 1e9:.1f} GB/s (highest measured), "
          f"B_gpu {B_GPU / 1e9:.0f} GB/s; D {m['D'] / 1e9:.3f} GB, S {m['S'] / 1e6:.2f} MB")
    for C in [int(x) for x in a.budgets.split(",")]:
        acc = {}
        for l in range(L):
            R = np.ascontiguousarray(act[:, l, :])
            _, mi, ad = ecsim_fast.simulate_layer(R, E, C, "dfa", half_life=16.0, kappa=m["kappa"])
            x = acc.setdefault("deployed", [0, 0, 0])
            x[0] += int(mi.sum()); x[1] += int(ad.sum()); x[2] += int(mi.sum())
            for key, W in (("online", -1), ("opt", INF)):
                for kap in sorted({0.0, m["kappa"]}):
                    c, p = _pol(R, E, C, W, 16.0, kap)
                    x = acc.setdefault(f"{key}@{kap:g}", [0, 0, 0])
                    x[0] += int(c); x[1] += int(p); x[2] += int(c) + int(p)
        row = {}
        for key in ("online", "opt"):
            best = min((kk for kk in acc if kk.startswith(key + "@")), key=lambda kk: acc[kk][0] + acc[kk][1])
            c, p, _ = acc[best]
            row[key] = dict(cpu=c / T, copy=p / T, reads=(c + p) / T, kappa=float(best.split("@")[1]))
        mi, ad, _ = acc["deployed"]
        row["deployed"] = dict(misses=mi / T, admits=ad / T, reads=(mi + ad) / T, hit_rate=1 - mi / (T * L * k))
        for key in ("opt", "online"):
            t, c = limit(row[key]["reads"], L * k, m["S"], m["D"], b_host)
            row[key].update(t_ms=t * 1e3, tok_s=1 / t, cpu_at_limit=c)
        row["gpu_only_ms"] = (m["D"] + L * k * m["S"]) / B_GPU * 1e3
        out["rows"][C] = row
        print(f"  C {C:3d} ({C / E:.1%}): reads/token opt {row['opt']['reads']:6.2f}  best online {row['online']['reads']:6.2f}"
              f"  deployed {row['deployed']['reads']:6.2f} (hit {row['deployed']['hit_rate']:.3f})"
              f"  ->  speed limit {row['opt']['tok_s']:6.1f} tok/s, best-online reference {row['online']['tok_s']:6.1f} tok/s"
              f"  (all in VRAM at datasheet rate {1e3 / row['gpu_only_ms']:.0f})")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

"""Pre-registered predictions for a first-party GPU platform.

Committed BEFORE any throughput or counter measurement exists. Given only the
platform microbenchmarks of job 001 (STREAM, copy bandwidth, nvidia-smi), it
deterministically produces the model's predictions for every configuration
that jobs 003 (llama-bench sweep) and 004 (Nsight Compute bytes) will measure.
Nothing here is fitted to the new platform: the efficiencies and latencies are
the ones selected on the third-party validation set (results/validation.json).

    python scripts/preregister.py <results/001_platform dir> <out dir>

Inputs and their rules (fixed in advance, see prereg/PROTOCOL.md):
  B_g  GPU memory peak from the datasheet (by nvidia-smi name).
  B_p  PCIe peak from the negotiated maximum link generation and width.
  B_c  host DRAM: the best STREAM Triad over thread counts. A VM exposes no
       DIMM configuration and a slice of a socket, so no datasheet peak applies;
       using a measured number where the model expects a peak biases CPU-side
       predictions SLOW by the unknown Triad/peak ratio (typically 0.7-0.85).
       We register that direction now.
  bytes  exact, from each GGUF header (mosl/gguf_bytes.py).
"""
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mosl.archs import shape, kv_bytes
from mosl.gguf_bytes import GGUFS, account

DATASHEET_GBS = {"NVIDIA A10": 600, "NVIDIA A10G": 600, "NVIDIA RTX A6000": 768, "Quadro RTX 6000": 672,
                 "NVIDIA A100-SXM4-40GB": 1555, "NVIDIA A100-SXM4-80GB": 2039, "NVIDIA A100 80GB PCIe": 1935,
                 "NVIDIA H100 PCIe": 2000, "NVIDIA H100 80GB HBM3": 3350, "NVIDIA GH200 480GB": 4000,
                 "NVIDIA GeForce RTX 4090": 1008, "NVIDIA GeForce RTX 5090": 1792}
PCIE_GBS = {3: 15.75, 4: 31.5, 5: 63.0}
HF_REPO = {"gpt-oss-20b-mxfp4": "openai/gpt-oss-20b", "gpt-oss-120b-mxfp4": "openai/gpt-oss-120b",
           "qwen3-30b-a3b-q4_k_m": "Qwen/Qwen3-30B-A3B-Instruct-2507", "qwen3-30b-a3b-q8_0": "Qwen/Qwen3-30B-A3B-Instruct-2507"}

# The sweep job 003 runs (n_cpu_moe values; "cpu" = -ngl 0). Fixed here, before measurement.
SWEEP = {
    "gpt-oss-20b-mxfp4": [0, 3, 6, 9, 12, 15, 18, 21, 24, "cpu"],
    "qwen3-30b-a3b-q4_k_m": [0, 6, 12, 18, 24, 30, 36, 42, 48, "cpu"],
    "qwen3-30b-a3b-q8_0": [18, 24, 30, 36, 42, 48],
    "gpt-oss-120b-mxfp4": [25, 28, 31, 34, 36],
}
NCU = {"gpt-oss-20b-mxfp4": [0, 12], "qwen3-30b-a3b-q4_k_m": [0, 24]}
CTX = 64          # llama-bench -p 0 -n 128: decode positions 0..127, mean ~64
KV_BITS = 16      # llama.cpp default f16 KV cache


def platform(d):
    g = next(csv.DictReader(open(os.path.join(d, "gpu.csv")), skipinitialspace=True))
    g = {k.strip(): v.strip() for k, v in g.items()}
    name = g["name"]
    try:
        gen = int(g["pcie.link.gen.max"]); width = int(g["pcie.link.width.max"])
    except ValueError:   # GH200: the GPU hangs off NVLink-C2C, nvidia-smi reports no PCIe link; unused by --n-cpu-moe
        gen, width = 5, 16
    triad = {}
    t = None
    for line in open(os.path.join(d, "stream.txt")):
        m = re.match(r"== OMP_NUM_THREADS=(\d+)", line)
        if m:
            t = int(m.group(1))
        m = re.match(r"Triad:\s+([\d.]+)", line)
        if m and t is not None:
            triad[t] = float(m.group(1)) / 1000
    bw = open(os.path.join(d, "bw.txt")).read()
    h2d = [float(x) for x in re.findall(r"H2D pinned\s+268\.4 MB\s+([\d.]+) GB/s", bw)]
    dev = [float(x) for x in re.findall(r"device read 1 GiB:\s+([\d.]+) GB/s", bw)]
    return dict(gpu=name, bw_gpu=DATASHEET_GBS[name], bw_pcie=PCIE_GBS[gen] * width / 16, pcie_gen=gen,
                pcie_width=width, bw_cpu=max(triad.values()), stream_triad_by_threads=triad,
                measured_h2d=h2d[0] if h2d else None, measured_dev_read=dev[0] if dev else None)


def params():
    v = json.load(open("results/validation.json"))["variants"]
    return {"M4": v["M4 M1 + per-CPU-expert latency"]["params"], "M0": v["M0 bytes/bandwidth + hand-off"]["params"]}


def step_time(g, n_cpu, plat, p, ctx=CTX):
    """Seconds per decode token under llama.cpp --n-cpu-moe n_cpu (experts of the
    first n_cpu MoE layers on the CPU), or CPU-only when n_cpu == 'cpu'."""
    s = shape(HF_REPO[g.file_key])
    kv = kv_bytes(s, ctx, KV_BITS)
    dense = g.dense_bytes + g.head_bytes + kv
    e = [g.top_k * x for x in g.expert_bytes_per_layer]
    tau_e = p.get("tau_e_us", 0) * 1e-6
    tau = p.get("tau_us", 0) * 1e-6
    Bg, Bc = p["eta_g"] * plat["bw_gpu"] * 1e9, p["eta_c"] * plat["bw_cpu"] * 1e9
    if n_cpu == "cpu":
        return (dense + sum(e)) / Bc + len(e) * g.top_k * tau_e
    xc, xg = sum(e[:n_cpu]), sum(e[n_cpu:])
    return (dense + xg) / Bg + xc / Bc + n_cpu * g.top_k * tau_e + n_cpu * tau


def roofline_time(g, n_cpu, plat, ctx=CTX):
    """Physical floor with all efficiencies = 1 and measured bandwidths
    (device read for the GPU, STREAM for the CPU): no run may beat it."""
    p = dict(eta_g=1.0, eta_c=1.0)
    q = dict(plat, bw_gpu=plat["measured_dev_read"] or plat["bw_gpu"])
    return step_time(g, n_cpu, q, p, ctx)


def gpu_bytes(g, n_cpu, ctx=CTX):
    s = shape(HF_REPO[g.file_key])
    e = [g.top_k * x for x in g.expert_bytes_per_layer]
    return g.dense_bytes + g.head_bytes + kv_bytes(s, ctx, KV_BITS) + sum(e[n_cpu:])


def main(pdir, out):
    plat = platform(pdir)
    P = params()
    rows = []
    for name, sweep in SWEEP.items():
        g = account(*GGUFS[name]); g.file_key = name
        for n in sweep:
            r = dict(model=name, n_cpu_moe=n, pred_tok_s=1 / step_time(g, n, plat, P["M4"]),
                     pred_tok_s_M0=1 / step_time(g, n, plat, P["M0"]), roofline_tok_s=1 / roofline_time(g, n, plat))
            if n != "cpu":
                r["pred_gpu_dram_read_bytes_per_token"] = gpu_bytes(g, n)
            rows.append(r)
    ncu = []
    for name, ns in NCU.items():
        g = account(*GGUFS[name]); g.file_key = name
        for n in ns:
            # ncu runs decode at positions 0..7: KV is negligible, use ctx=4
            ncu.append(dict(model=name, n_cpu_moe=n, pred_gpu_dram_read_bytes_per_token=gpu_bytes(g, n, ctx=4)))
    os.makedirs(out, exist_ok=True)
    doc = dict(platform=plat, params=P, ctx=CTX, sweep=rows, ncu=ncu,
               note="Generated by scripts/preregister.py from job-001 microbenchmarks only.")
    json.dump(doc, open(os.path.join(out, "predictions.json"), "w"), indent=1)
    with open(os.path.join(out, "predictions.md"), "w") as f:
        f.write(f"# Pre-registered predictions: {plat['gpu']}\n\n")
        f.write(f"B_g={plat['bw_gpu']} GB/s (datasheet), B_c={plat['bw_cpu']:.1f} GB/s (STREAM Triad max), "
                f"B_p={plat['bw_pcie']} GB/s (PCIe Gen{plat['pcie_gen']} x{plat['pcie_width']})\n\n")
        f.write("| model | n_cpu_moe | pred tok/s (M4) | pred tok/s (M0) | roofline tok/s | GPU bytes/token |\n|---|---|---|---|---|---|\n")
        for r in rows:
            gb = r.get("pred_gpu_dram_read_bytes_per_token")
            f.write(f"| {r['model']} | {r['n_cpu_moe']} | {r['pred_tok_s']:.1f} | {r['pred_tok_s_M0']:.1f} | "
                    f"{r['roofline_tok_s']:.1f} | {'' if gb is None else f'{gb/1e9:.3f} GB'} |\n")
        f.write("\n| ncu model | n_cpu_moe | predicted DRAM read bytes / token |\n|---|---|---|\n")
        for r in ncu:
            f.write(f"| {r['model']} | {r['n_cpu_moe']} | {r['pred_gpu_dram_read_bytes_per_token']/1e9:.3f} GB |\n")
    print(open(os.path.join(out, "predictions.md")).read())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

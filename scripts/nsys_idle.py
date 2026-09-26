"""GPU busy/idle split per decode token from Nsight Systems kernel traces.

    python scripts/nsys_idle.py results/firstparty/a10/007_counters

For each configuration: kernel-busy time, idle time between kernels, number of
idle gaps > 20 us (one per GPU->CPU hand-off), and host<->device copies, all per
decode token (32 tokens per trace). nsys itself slows the run by ~20%, so the
split, not the absolute time, is the quantity of interest.
"""
import csv, glob, gzip, json, os, sys


def split(path, n_tok=32):
    rows = list(csv.DictReader(gzip.open(path, "rt")))
    for r in rows:
        r["s"], r["d"] = float(r["Start (ns)"]), float(r["Duration (ns)"])
    k = sorted((r for r in rows if not r["Name"].startswith("[CUDA")), key=lambda r: r["s"])
    t0, t1 = k[0]["s"], max(r["s"] + r["d"] for r in k)
    gaps, end = [], k[0]["s"]
    for r in k:
        if r["s"] > end:
            gaps.append(r["s"] - end)
        end = max(end, r["s"] + r["d"])
    mc = [r for r in rows if r["Name"].startswith("[CUDA memcpy") and t0 <= r["s"] <= t1]
    big = [g for g in gaps if g > 20e3]
    return dict(span_ms=(t1 - t0) / n_tok / 1e6, busy_ms=sum(r["d"] for r in k) / n_tok / 1e6,
                idle_ms=sum(gaps) / n_tok / 1e6, handoff_gaps=len(big) / n_tok, kernels=len(k) / n_tok,
                idle_frac=sum(gaps) / (t1 - t0),
                h2d_kb=sum(float(r["Bytes (MB)"]) for r in mc if "Host-to-Device" in r["Name"]) / n_tok * 1e3,
                d2h_kb=sum(float(r["Bytes (MB)"]) for r in mc if "Device-to-Host" in r["Name"]) / n_tok * 1e3)


if __name__ == "__main__":
    d = sys.argv[1]
    out = {os.path.basename(f)[5:-len("_gpu_trace.csv.gz")]: split(f) for f in sorted(glob.glob(os.path.join(d, "nsys_*_gpu_trace.csv.gz")))}
    json.dump(out, open(os.path.join(d, "nsys_summary.json"), "w"), indent=1)
    for k, v in out.items():
        print(f"{k:32s} busy {v['busy_ms']:5.2f} ms  idle {v['idle_ms']:6.2f} ms ({100*v['idle_frac']:4.1f}%)  hand-offs {v['handoff_gaps']:5.1f}/tok  kernels {v['kernels']:.0f}/tok")

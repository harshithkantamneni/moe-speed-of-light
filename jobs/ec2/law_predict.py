"""Pre-registered prediction of gpt-oss-120b decode speed on this machine, from the host-DRAM law (main repo:
scripts/law_hostdram.py), computed on the machine from its own bandwidth probe (concur.cu) BEFORE any model run.

    T_token = G + max(cpu / B_c, link / B_p, (cpu + link) / B_both)

Frozen here (committed before launch, never refitted per host):
  - G per engine, fitted on the 9950X #2 host (jobs 064/066): expert cache 4.819 ms, llama.cpp 5.063 ms (RTX 5090);
  - expert reads per token for each configuration (own text, seqs 0-11 x 128 tokens), from the engine's counters,
    which are identical on every host (jobs 066/068/071): cpu = misses run on the CPU, link = fetches + prefetches;
    llama.cpp --n-cpu-moe n reads all 4 routed experts of n layers on the CPU.
Bandwidths come from this host's concur.txt: B_c = CPU read interpolated at the helper thread count, B_p = the
zero-copy link rate (16 MB chunks, 64 blocks), B_both = median CPU + zero-copy sum when both run.
Only valid for an RTX 5090 (G is the 5090's); on another GPU it is reported but flagged.

    python3 law_predict.py concur.txt HELPERS GPU_NAME > law_prediction.json
"""
import json
import re
import statistics
import sys
import time

S = 13253760.0
G = {"cache": 4.819346e-3, "static": 5.063192e-3}
READS = {  # expert reads per token (cpu, link)
    "C14": (66.0423, 0.0), "C32": (34.8770, 0.0), "C56": (17.9388, 0.0),
    "C14_fetch": (18.6979, 40.8704), "C32_fetch": (7.9193, 22.5924), "C56_fetch": (3.1087, 12.1107),
    "C14_pf": (36.5547, 29.4674), "C32_pf": (16.3418, 19.6842), "C56_pf": (7.5723, 11.4512),
    "C32_fetch_pf": (3.1986, 31.1198),
}
STATIC = {"static20": 20, "static27": 27, "static32": 32}


def main():
    txt = open(sys.argv[1]).read()
    helpers = int(sys.argv[2])
    gpu = sys.argv[3] if len(sys.argv) > 3 else ""
    cpu = sorted((int(t), float(v)) for t, v in re.findall(r"cpu_read_gbs t=(\d+) ([\d.]+)", txt))
    # several lines may share a thread count: average them
    by = {}
    for t, v in cpu:
        by.setdefault(t, []).append(v)
    pts = sorted((t, statistics.mean(v)) for t, v in by.items())
    if helpers <= pts[0][0]:
        bc = pts[0][1]
    elif helpers >= pts[-1][0]:
        bc = pts[-1][1]
    else:
        for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
            if t0 <= helpers <= t1:
                bc = v0 + (v1 - v0) * (helpers - t0) / (t1 - t0)
                break
    zc = [float(x) for x in re.findall(r"pcie_zerocopy_gbs chunk=16MB blocks=64 ([\d.]+)", txt)]
    both = [float(z) for z in re.findall(r"concurrent t=\d+ pcie=zerocopy64 cpu_gbs [\d.]+ pcie_gbs [\d.]+ sum ([\d.]+)", txt)]
    bp = max(zc) if zc else float("nan")
    bb = statistics.median(both) if both else float("nan")
    pred = {}
    for name, (c, l) in READS.items():
        t = G["cache"] + max(c * S / (bc * 1e9), l * S / (bp * 1e9), (c + l) * S / (bb * 1e9))
        pred[name] = round(1.0 / t, 2)
    for name, n in STATIC.items():
        t = G["static"] + n * 4 * S / (bc * 1e9)
        pred[name] = round(1.0 / t, 2)
    json.dump(dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), gpu=gpu,
                   valid="RTX 5090" in gpu, helpers=helpers, B_c=round(bc, 2), B_p=round(bp, 2), B_both=round(bb, 2),
                   G_ms={k: round(v * 1e3, 3) for k, v in G.items()}, predicted_tok_s=pred), sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()

"""Per-host FETCH table from the host-DRAM law, computed on the machine from its own bandwidth probe (concur.txt) before
any model run (job 080+).

For a layer with n missed experts, the cache fetches f of them over PCIe into slots (the GPU then runs all its
experts) and hands the other n - f to the CPU helpers, which run while the GPU copies and computes. The layer's MoE
time is modelled as

    T(n, f) = max( f * S / B_p + g,                     # PCIe copy, then the GPU's expert products
                   [n - f > 0] * (L_c + (n - f) * S / B_c),   # CPU helpers (started before the copy)
                   [0 < f < n] * n * S / B_both )        # both paths read host DRAM at once

and the table picks, for each n, the f with the smallest T (ties to the larger f: a fetched expert is admitted at
once, which saves later misses). B_c = CPU read at the helper count, B_p = zero-copy link rate, B_both = median
CPU + zero-copy sum (the same reading of concur.txt as law_predict.py). g and L_c are frozen: g = 48 us (Qwen3-30B-A3B
BF16 expert products per layer on an RTX 5090, job 079 profile), L_c = 20 us (helper hand-off).

    python3 fetch_table.py concur.txt HELPERS EXPERT_BYTES TOPK  ->  prints the table, e.g. 0,0,1,1,2,2,3,4,4
"""
import json
import re
import statistics
import sys

G_US, LC_US = 48.0, 20.0


def bandwidths(txt, helpers):
    by = {}
    for t, v in re.findall(r"cpu_read_gbs t=(\d+) ([\d.]+)", txt):
        by.setdefault(int(t), []).append(float(v))
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
    return bc, max(zc), statistics.median(both)


def table(S, k, bc, bp, bb, g=G_US, lc=LC_US):
    sp, sc, sb = S / bp / 1e3, S / bc / 1e3, S / bb / 1e3   # us per expert (B in GB/s)
    out, cost = [], []
    for n in range(k + 1):
        best = None
        for f in range(n + 1):
            t = f * sp + g
            if n - f > 0:
                t = max(t, lc + (n - f) * sc)
            if 0 < f < n:
                t = max(t, n * sb)
            if best is None or t <= best[0] + 1e-9:
                best = (t, f)
        out.append(best[1])
        cost.append(round(best[0], 1))
    return out, cost


def main():
    txt = open(sys.argv[1]).read()
    helpers, S, k = int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    bc, bp, bb = bandwidths(txt, helpers)
    t, c = table(S, k, bc, bp, bb)
    print(",".join(map(str, t)))
    json.dump(dict(B_c=round(bc, 2), B_p=round(bp, 2), B_both=round(bb, 2), g_us=G_US, L_c_us=LC_US, table=t,
                   modelled_layer_us=c), open(sys.argv[5] if len(sys.argv) > 5 else "/dev/stderr", "w"), indent=1)


if __name__ == "__main__":
    main()

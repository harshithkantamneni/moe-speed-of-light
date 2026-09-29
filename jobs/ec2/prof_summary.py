"""On-host summary of one Nsight Systems profile (prof.sh exports), so that only small files leave the machine.

Input: $WORK/prof_<label>_cuda_gpu_trace.csv and _cuda_api_trace.csv (nsys stats, CUDA graphs traced per node).
Output (in --out):
  prof_<label>.json       per-token medians over the last --tokens decode tokens: wall, GPU busy (union of
                          kernels and copies), GPU idle split into short gaps (< --short-us, dependent-launch bubbles
                          inside a graph) and long gaps (host-side), per-category kernel time, kernel count, the
                          ec_wait distribution (the GPU spinning while the CPU helpers run misses), top kernels
  prof_<label>_tail.csv.gz  (with --keep-tail) the GPU trace of those tokens only, for offline inspection

Token boundaries: cudaGraphLaunch calls in the API trace (llama.cpp launches one or two graphs per token; with two,
pairs are merged by --launches-per-token), otherwise GPU gaps longer than --gap-us.
Standard library only.
"""
import argparse
import csv
import gzip
import json
import os
import re
import statistics as st

CATS = [
    ("ec_wait", r"ec_wait_kernel"),
    ("ec_copy", r"ec_fetch_copy_kernel|zc_copy|ec_zc|copy_kernel"),
    ("ec_ctl", r"\bec_"),
    ("expert_gemv", r"\(ggml_type\)39\b|mul_mat_id|moe_"),
    ("attention", r"flash_attn|fattn|soft_max|rope"),
    ("gemv", r"mul_mat_vec|mmvq|mmvf|gemv|gemm|cutlass|cublas|mul_mat"),
    ("quantize", r"quantize"),
    ("topk_router", r"argsort|topk|top_k"),
    ("norm", r"norm"),
    ("small_ops", r"add|mul|glu|swiglu|scale|cpy|get_rows|set_rows|concat|unary|bin_bcast|k_"),
    ("memcpy", r"\[CUDA memcpy|\[CUDA memset|memcpy|memset"),
]


def category(name, grdx=0):
    if grdx > 100000 and re.search(r"mul_mat_vec|mmvq|mmvf", name):
        return "head"
    for c, pat in CATS:
        if re.search(pat, name, re.I):
            return c
    return "other"


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def col(r, *names):
    for n in names:
        for k in r:
            if k.strip().lower().startswith(n.lower()):
                return r[k]
    raise KeyError(names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=20)
    ap.add_argument("--gap-us", type=float, default=40.0)
    ap.add_argument("--short-us", type=float, default=20.0)
    ap.add_argument("--launches-per-token", type=int, default=0, help="0: infer (1 or 2) from launch spacing")
    ap.add_argument("--keep-tail", action="store_true")
    a = ap.parse_args()
    gp = os.path.join(a.work, f"prof_{a.label}_cuda_gpu_trace.csv")
    apip = os.path.join(a.work, f"prof_{a.label}_cuda_api_trace.csv")
    g = rows(gp)
    ev = []
    for r in g:
        try:
            s = int(float(col(r, "Start"))); d = int(float(col(r, "Duration"))); n = col(r, "Name")
        except (KeyError, ValueError):
            continue
        try:
            gx, gy = int(float(r.get("GrdX") or 0)), int(float(r.get("GrdY") or 0))
        except ValueError:
            gx, gy = 0, 0
        ev.append((s, s + d, n, gx, gy))
    ev.sort()
    launches, kind = [], ""
    if os.path.exists(apip):
        launches = sorted(int(float(col(r, "Start"))) for r in rows(apip) if "cudaGraphLaunch" in col(r, "Name"))
        kind = "graph launches"
    lpt = a.launches_per_token
    if launches and lpt == 0:
        # two launches per token show as alternating short/long spacings; compare the two interleaved medians
        dl = [b - x for x, b in zip(launches[-81:-1], launches[-80:])]
        m0, m1 = st.median(dl[0::2]), st.median(dl[1::2])
        lpt = 2 if min(m0, m1) < 0.5 * max(m0, m1) else 1
    starts = launches
    if launches and lpt == 2:
        thr = (m0 + m1) / 2
        starts = [launches[i] for i in range(1, len(launches)) if launches[i] - launches[i - 1] > thr]
    if len(starts) >= a.tokens + 1 and lpt > 0:
        bounds = starts[-(a.tokens + 1):]
    else:
        kind = f"GPU gaps > {a.gap_us} us"
        bounds, last_end = [], None
        for s, e, n, _, _ in ev:
            if last_end is not None and s - last_end > a.gap_us * 1000:
                bounds.append(s)
            last_end = e if last_end is None else max(last_end, e)
        bounds = bounds[-(a.tokens + 1):]
    per = []
    names = {}
    tail_rows = []
    for t0, t1 in zip(bounds[:-1], bounds[1:]):
        cats, iv, nk, waits = {}, [], 0, []
        for s, e, n, gx, gy in ev:
            if e <= t0 or s >= t1:
                continue
            s2, e2 = max(s, t0), min(e, t1)
            c = category(n, gx)
            cats[c] = cats.get(c, 0) + (e2 - s2)
            iv.append((s2, e2)); nk += 1
            if c == "ec_wait":
                waits.append(e - s)
            ty = re.search(r"\(ggml_type\)(\d+)", n)
            short = re.sub(r"<.*|\(.*", "", n)[:50] + (f" t{ty.group(1)}" if ty else "") + f" g{gx}x{gy}"
            nm = names.setdefault(short, [0, 0])
            nm[0] += e2 - s2; nm[1] += 1
            if a.keep_tail:
                tail_rows.append((s, e - s, n))
        iv.sort()
        busy, gaps_short, gaps_long, n_short = 0, 0, 0, 0
        cs, ce = None, None
        for s, e in iv:
            if ce is None:
                cs, ce = s, e
                if s > t0:
                    g0 = s - t0
                    gaps_long += g0 if g0 >= a.short_us * 1000 else 0
                    gaps_short += g0 if g0 < a.short_us * 1000 else 0
            elif s > ce:
                busy += ce - cs
                gap = s - ce
                if gap < a.short_us * 1000:
                    gaps_short += gap; n_short += 1
                else:
                    gaps_long += gap
                cs, ce = s, e
            else:
                ce = max(ce, e)
        if ce is not None:
            busy += ce - cs
            if t1 > ce:
                g1 = t1 - ce
                gaps_long += g1 if g1 >= a.short_us * 1000 else 0
                gaps_short += g1 if g1 < a.short_us * 1000 else 0
        wsorted = sorted(waits)
        per.append(dict(wall=t1 - t0, busy=busy, gaps_short=gaps_short, gaps_long=gaps_long, n_kernels=nk,
                        n_short_gaps=n_short, n_wait=len(waits), wait_total=sum(waits),
                        wait_median=wsorted[len(wsorted) // 2] if waits else 0,
                        wait_under_5us=sum(1 for w in waits if w < 5000),
                        wait_min=wsorted[0] if waits else 0, **{"cat_" + k: v for k, v in cats.items()}))
    keys = sorted({k for p in per for k in p})
    med = {k: st.median([p.get(k, 0) for p in per]) for k in keys}
    ms = {k: (v / 1e6 if not k.startswith("n_") and k != "wait_under_5us" else v) for k, v in med.items()}
    top = sorted(names.items(), key=lambda x: -x[1][0])[:30]
    out = dict(label=a.label, kind=kind, launches_per_token=lpt, tokens=len(per), median_ms=ms,
               mean_ms={k: (sum(p.get(k, 0) for p in per) / len(per) / (1e6 if not k.startswith("n_") and k != "wait_under_5us" else 1)) for k in keys} if per else {},
               top_ms_per_token=[(n, d / 1e6 / max(1, len(per)), c / max(1, len(per)), d / 1e3 / max(1, c)) for n, (d, c) in top],
               top_fields=["kernel type grid", "ms per token", "calls per token", "us per call"])
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, f"prof_{a.label}.json"), "w"), indent=1)
    if a.keep_tail and tail_rows:
        with gzip.open(os.path.join(a.out, f"prof_{a.label}_tail.csv.gz"), "wt") as f:
            w = csv.writer(f)
            w.writerow(["start_ns", "dur_ns", "name"])
            t00 = tail_rows[0][0]
            for s, d, n in tail_rows:  # noqa
                w.writerow([s - t00, d, n[:120]])
    print(f"{a.label}: {len(per)} tokens ({kind}, {lpt} launch/token); wall {ms.get('wall', 0):.3f} ms, busy "
          f"{ms.get('busy', 0):.3f}, short gaps {ms.get('gaps_short', 0):.3f} ({med.get('n_short_gaps', 0):.0f}), long gaps "
          f"{ms.get('gaps_long', 0):.3f}, ec_wait {ms.get('wait_total', 0):.3f} over {med.get('n_wait', 0):.0f} calls, "
          f"kernels {med.get('n_kernels', 0):.0f}")


if __name__ == "__main__":
    main()

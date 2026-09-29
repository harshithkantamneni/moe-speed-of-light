"""Where the seconds go: gpt-oss-120b decode on one RTX 5090 host, from a speed-of-light reference to the measured
time per token, one cause at a time (job 072, Ryzen 9 7900 host, own text, 12 x 128 tokens).

Every step adds one cost to the previous time (ms per token):
  0 reference   max(GPU physical, host reads of the optimum)   the per-layer optimum (Belady with bypass, cold start)
                                                               with its reads split over both host paths at the
                                                               host's measured both-paths bandwidth, overlapped with
                                                               the GPU's physical time (datasheet 1792 GB/s)
  1 no overlap  GPU physical + optimum's host time             no exact-routing schedule on this engine overlaps
                                                               a layer's host reads with the GPU work they depend on
  2 foresight   best online policy without foresight           decayed frequency, fetch-admit (one read per miss)
  3 policy      the deployed policy (CPU runs every miss at B_c; admissions in the background); with FETCH instead
  4 GPU         the GPU work measured in the profiles (36 layer phases + head) instead of the physical time
  5 host        graph launch, pre/post-processing, measured by the host-side timer (job 072)
  6 residual    measured minus the above (admission reads competing with the CPU phases, latency not modelled)

    python scripts/gap_decomposition.py [--json prereg/homepc/gap_decomposition_072.json]
"""
import argparse
import json

S = 13253760.0                      # bytes per expert
D = 1713980160.0                    # dense + head + KV bytes per token (GGUF)
B_GPU = 1792e9                      # RTX 5090 datasheet
RES = "/home/claude/gpu-branch/results/072_defer@vast"
PHASE_US, HEAD_US = 90.0, 350.0     # profiles (069c, 072): median GPU phase per layer, output head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    law = json.load(open(f"{RES}/law_prediction.json"))
    Bc, Bp, Bb = law["B_c"] * 1e9, law["B_p"] * 1e9, law["B_both"] * 1e9
    pol = json.load(open("prereg/homepc/policy_bounds_engine_text.json"))
    runs = {}
    for line in open(f"{RES}/ref_ec.jsonl"):
        r = json.loads(line)
        k = r["config"].split("stats=")[-1].split("/")[-1]
        runs.setdefault(k, []).append(r)
    out = {}
    for C in (14, 32, 56):
        opt, onl = pol[str(C)]["opt"], pol[str(C)]["online"]
        gpu_phys = (D + (144 - opt["cpu"]) * S) / B_GPU
        host_opt = opt["reads"] * S / Bb
        steps = []
        t = max(gpu_phys, host_opt)
        steps.append(("reference: optimum, overlapped", t))
        t1 = gpu_phys + host_opt
        steps.append(("no overlap", t1 - t)); t = t1
        t2 = gpu_phys + onl["reads"] * S / Bb
        steps.append(("no foresight (best online policy)", t2 - t)); t = t2
        res = {}
        for mode, key in (("deployed", f"ref_C{C}_r"), ("deployed+FETCH", f"ref_C{C}f_r")):
            files = [k for k in runs if k.startswith(key) and k.endswith(".json") and (mode == "deployed+FETCH") == ("f_r" in k)]
            if not files:
                continue
            tok = sum(r["n_decode"] for k in files for r in runs[k]) / (sum(r["decode_ms"] for k in files for r in runs[k]) / 1000)
            st = json.load(open(f"{RES}/{files[0]}"))
            n = st["steps"]
            cpu = (st["misses"] - st["fetches"]) / n * S
            link = (st["fetches"] + st.get("prefetches", 0)) / n * S
            host = max(cpu / Bc, link / Bp, (cpu + link) / Bb)
            h = st["host_us_per_step"]
            host_us = (h["launch"] + h["post"] + h["pre"] + h["inputs"] + h["issue"]) / 1e6
            gpu_meas = (36 * PHASE_US + HEAD_US) / 1e6
            s2 = list(steps)
            tt = t
            t3 = gpu_phys + host
            s2.append((f"policy as deployed ({mode})", t3 - tt)); tt = t3
            t4 = gpu_meas + host
            s2.append(("GPU work at batch 1 (measured phases)", t4 - tt)); tt = t4
            t5 = t4 + host_us
            s2.append(("host launch and bookkeeping", t5 - tt)); tt = t5
            s2.append(("residual (admission contention, unmodelled latency)", 1 / tok - tt))
            res[mode] = dict(measured_tok_s=tok, steps_ms=[(k, v * 1e3) for k, v in s2])
            print(f"C = {C} ({mode}): measured {tok:.1f} tok/s = {1e3 / tok:.2f} ms")
            acc = 0.0
            for k, v in s2:
                acc += v
                print(f"   {k:52s} {v * 1e3:+6.2f} ms  -> {acc * 1e3:6.2f} ms ({1 / acc:6.1f} tok/s)")
        out[C] = res
    if a.json:
        json.dump(dict(host=RES, B=[Bc / 1e9, Bp / 1e9, Bb / 1e9], budgets=out), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

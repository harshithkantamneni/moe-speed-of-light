"""Order-independent attribution of the gap between measured time per token and the speed limit (Shapley values over
every order of the fixes), for the six Table 1 cells of scripts/gap_listingb.py (RTX 5090 + Ryzen 9 9950X3D, jobs
080/081).

Model. gap_listingb.py adds one cost at a time in a fixed order; the same quantities define a value function over
subsets of five "fixes" (each either applied, as in the limit, or not, as measured):

  O overlap     applied: GPU work and host reads run concurrently and the CPU may run resident experts to balance the
                two, T = min over c of max(GPU(c), host(c, max(R - c, 0)))  (the limit's form, eq. 2 of the paper);
                not: GPU(c) + host(c, p) with the policy's own CPU/copy split, in sequence
  F foresight   applied: the optimum's reads (Belady with bypass; cpu, copy per token); not: an online policy's reads
  P policy and  applied: the best online policy's reads (when F is not applied) and every read at the limit's single
    read paths  rate B_host; not: the engine's own reads (misses on the CPU; fetches and admissions over PCIe) through
                the host-memory law with the machine's probe, max(cpu S/B_c, link S/B_p, (cpu + link) S/B_cp)
  H host work   applied: none; not: the engine's per-token host time (app + pre + inputs + launch + post)
  G GPU         applied: datasheet rate; not: the GPU's effective rate at batch 1, B_gpu / eta, with eta calibrated per
    efficiency  cell so that the measured time is reproduced with no fix applied (eta is net of the overlap the engine
                achieves, exactly as the paper's "GPU below datasheet (net)" step is)

v(S) = core(S) + host work; v(none) = measured time, v(all) = the speed limit, and breaking the fixes in the paper's
order (overlap, foresight, policy and paths, host work, GPU) reproduces prereg/gap_listingb.json step for step. The
Shapley value of a fix is its marginal saving averaged over the 5! = 120 orders; we also keep the smallest and largest
marginal of each fix over the orders, and in how many orders foresight is the largest.

Variants (all written to the JSON in one run; --limit and --bhost pick the one reported as `selected`):
  the optimum's reads R*: 084 (scripts/foresight._pol, prereg/speed_limit_084.json), exact (mosl.cachesim MIN with
    bypass per layer, 0.5-3.9% fewer reads, prereg/speed_limit_v2.json) or global (one pool of L*C slots); for exact
    and global the optimum's CPU/copy split (bypassed misses / admissions) is recomputed from the routing traces
  B_host: max (87.5 GB/s, the highest probe rate, the limit's), median (80.7, the median of the six concurrent CPU +
    PCIe sums) or median_zc (77.7, the median of the three zero-copy pairs, the B_cp the engine's law sees)
  the GPU shortfall as a constant instead of a rate; a gross GPU shortfall (the RTX PRO 6000 all-in-VRAM run's rate,
    same datasheet bandwidth, with the overlap the engine achieves credited to every non-overlapped state); and the
    policy-and-paths fix split in two (6 fixes, 720 orders): these three on the 084 reads at B_host max.

    python scripts/shapley_gap.py [--limit 084|exact|global] [--bhost max|median|median_zc] [--json prereg/shapley_gap.json]
"""
import argparse
import itertools
import json
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, "/home/claude/gpu-branch/jobs/ec2")
from fetch_table import bandwidths  # noqa: E402
from mosl import cachesim  # noqa: E402
from scripts.gap_listingb import CELLS, R, measured  # noqa: E402

FIX5 = ["overlap", "foresight", "policy and read paths", "host work", "GPU efficiency"]
FIX6 = ["overlap", "foresight", "policy", "read paths", "host work", "GPU efficiency"]
PAPER_STEPS = ["no overlap", "no foresight", "policy and read paths", "host work", "GPU below datasheet (net)"]
VRAM_DIR = f"{R}/084b_vram_rerun@vast"      # ours with 128 slots per layer, every expert resident (RTX PRO 6000)
GRID = 20001                                 # the limit's grid over c (scripts/speed_limit.py)
LIMITS = ("084", "exact", "global")
BHOSTS = ("max", "median", "median_zc")


def concurrent_sums(txt):
    """the probe's concurrent CPU + zero-copy PCIe read sums (fetch_table.bandwidths takes their median as B_cp)"""
    return [float(z) for z in re.findall(r"concurrent t=\d+ pcie=zerocopy64 cpu_gbs [\d.]+ pcie_gbs [\d.]+ sum ([\d.]+)", txt)]


def optimum_reads(trace, C, pooled):
    """The exact optimum's reads per token on the routing trace (MIN with bypass, mosl.cachesim): per layer with C
    slots, or one pool of L*C slots on the layer-interleaved stream. Returns (cpu, copy): bypassed misses run on the
    CPU, admitted misses are copied into a slot."""
    z = np.load(trace)
    act = z["act"].astype(np.int64)
    E = int(z["n_expert"])
    T, L, _ = act.shape
    if pooled:
        mi, ad = cachesim.simulate_global([act[:, l, :] for l in range(L)], E, C, "min", bypass=True)
        miss, adm = int(mi.sum()), int(ad.sum())
    else:
        miss = adm = 0
        for l in range(L):
            mi, ad = cachesim.simulate(act[:, l, :], E, C, "min", bypass=True)
            miss += int(mi.sum())
            adm += int(ad.sum())
    return (miss - adm) / T, adm / T


def cell_inputs(key, label, C, stats, sl, meas, v2=None):
    """Per-cell inputs; the optimum's reads from job 084 (`opt`) and, when speed_limit_v2.json is given, the exact
    per-layer and global-pool optima (`exact`, `global`), their totals checked against that file."""
    L = sl[key]
    row = L["rows"][str(C)]
    st = json.load(open(stats))
    n = st["steps"]
    hu = st["host_us_per_step"]
    reads = dict(opt=(row["opt"]["cpu"], row["opt"]["copy"]), online=(row["online"]["cpu"], row["online"]["copy"]),
                 engine=((st["misses"] - st["fetches"]) / n, (st["fetches"] + st["admits"] + st.get("prefetches", 0)) / n))
    if v2 is not None:
        m = v2["models"][key]
        for name, pooled in (("exact", False), ("global", True)):
            c, p = optimum_reads(m["trace"], C, pooled)
            ref = m["rows"][str(C)]["reads_per_token"][name]
            assert abs(c + p - ref) < 1e-9, f"{label} {name}: {c + p} reads vs {ref} in speed_limit_v2.json"
            reads[name] = (c, p)
    return dict(
        cell=label, model=key, C=C, S=L["S"], D=L["D"], Lk=L["L"] * L["k"], B_gpu=L["B_gpu"], reads=reads,
        hw=(hu["app"] + hu["pre"] + hu["inputs"] + hu["launch"] + hu["post"]) * 1e-6,
        T=1 / meas[label], limit_json=row["opt"]["t_ms"] * 1e-3, c_at_limit=row["opt"]["cpu_at_limit"],
        online_limit_json=row["online"]["t_ms"] * 1e-3)


def value_function(ci, b_host, probe, gpu_model="rate", eta_gross=None, split=False, opt="opt"):
    """Returns (v, info): v(flags) = predicted seconds per token with the fixes in `flags` applied (1) or not (0);
    flags are (O, F, P, H, G), or (O, F, Pol, Path, H, G) with split=True. `opt` names the optimum's reads in
    ci["reads"]: "opt" (job 084), "exact" or "global"."""
    S, D, Lk, bg = ci["S"], ci["D"], ci["Lk"], ci["B_gpu"]
    bc, bp, bb = probe
    c_eng, p_eng = ci["reads"]["engine"]
    grid = np.linspace(0, Lk, GRID)

    def gpu_ds(c):
        return (D + (Lk - c) * S) / bg

    def host(c, p, path):
        if path:
            return (c + p) * S / b_host
        return np.maximum(np.maximum(c * S / bc, p * S / bp), (c + p) * S / bb)

    law_eng = float(host(c_eng, p_eng, 0))
    resid = ci["T"] - ci["hw"] - law_eng - gpu_ds(c_eng)      # the paper's "GPU below datasheet (net)" step
    if gpu_model == "gross":
        eta = eta_gross
        delta = eta * gpu_ds(c_eng) + law_eng + ci["hw"] - ci["T"]   # overlap the engine achieves, by this accounting
    else:
        eta = 1 + resid / gpu_ds(c_eng)
        delta = 0.0

    def gpu(c, G):
        if G:
            return gpu_ds(c)
        if gpu_model == "const":
            return gpu_ds(c) + resid
        return gpu_ds(c) * eta

    def v(flags):
        if split:
            O, F, Pol, Path, H, G = flags
        else:
            O, F, P, H, G = flags
            Pol = Path = P
        c, p = ci["reads"][opt] if F else (ci["reads"]["online"] if Pol else ci["reads"]["engine"])
        if O:
            core = float(np.min(np.maximum(gpu(grid, G), host(grid, np.maximum(c + p - grid, 0.0), Path))))
        else:
            core = float(gpu(c, G) + host(c, p, Path)) - delta
        return core + (0.0 if H else ci["hw"])

    def c_at_limit(c, p):
        """the CPU experts per token at the limit's minimum (the grid's argmin, as scripts/speed_limit.limit)"""
        return float(grid[int(np.argmin(np.maximum(gpu_ds(grid), host(grid, np.maximum(c + p - grid, 0.0), 1))))])

    return v, dict(eta=eta, resid_ms=resid * 1e3, delta_ms=delta * 1e3, law_engine_ms=law_eng * 1e3,
                   gpu_datasheet_engine_ms=gpu_ds(c_eng) * 1e3, gpu_datasheet_c0_ms=gpu_ds(0) * 1e3,
                   c_at_limit=c_at_limit(*ci["reads"][opt]), optimum=opt,
                   opt_reads=dict(cpu=ci["reads"][opt][0], copy=ci["reads"][opt][1], total=sum(ci["reads"][opt])))


def shapley(v, n):
    """Shapley values (seconds saved by each fix), the smallest and largest marginal of each fix over all n! orders,
    every order's marginals, and the value table."""
    vals = {f: v(f) for f in itertools.product((0, 1), repeat=n)}
    phi = np.zeros(n)
    orders = []
    for perm in itertools.permutations(range(n)):
        cur = [0] * n
        prev = vals[tuple(cur)]
        marg = np.zeros(n)
        for i in perm:
            cur[i] = 1
            nxt = vals[tuple(cur)]
            marg[i] = prev - nxt
            prev = nxt
        phi += marg
        orders.append((perm, marg))
    phi /= math.factorial(n)
    M = np.array([m for _, m in orders])
    return phi, M.min(axis=0), M.max(axis=0), orders, vals


def analyse(ci, b_host, probe, gpu_model="rate", eta_gross=None, split=False, opt="opt"):
    names = FIX6 if split else FIX5
    n = len(names)
    v, info = value_function(ci, b_host, probe, gpu_model, eta_gross, split, opt)
    phi, mn, mx, orders, vals = shapley(v, n)
    none, full = (0,) * n, (1,) * n
    T, lim = vals[none], vals[full]
    gap = T - lim
    # the paper's order, from the limit: break overlap, foresight, policy (then paths), host work, GPU
    path = [list(full)]
    for i in range(n):
        s = list(path[-1])
        s[i] = 0
        path.append(s)
    pv = [vals[tuple(s)] for s in path]
    fixed = [pv[i + 1] - pv[i] for i in range(n)]
    iF = names.index("foresight")
    fs_first = sum(1 for _, m in orders if m[iF] >= m.max() - 1e-12 and (m > m[iF] - 1e-12).sum() == 1)
    fs_rank_worst = max(int((m > m[iF] + 1e-12).sum()) + 1 for _, m in orders)
    largest = names[int(np.argmax(phi))]
    order_phi = sorted(range(n), key=lambda i: -phi[i])
    out = dict(cell=ci["cell"], C=ci["C"], measured_ms=T * 1e3, limit_ms=lim * 1e3, gap_ms=gap * 1e3, **info,
               shapley_ms={k: phi[i] * 1e3 for i, k in enumerate(names)},
               shapley_share={k: phi[i] / gap for i, k in enumerate(names)},
               fixed_order_ms={k: fixed[i] * 1e3 for i, k in enumerate(names)},
               marginal_min_ms={k: mn[i] * 1e3 for i, k in enumerate(names)},
               marginal_max_ms={k: mx[i] * 1e3 for i, k in enumerate(names)},
               largest=largest, second=names[order_phi[1]],
               foresight_minus_next_ms=(phi[iF] - max(phi[i] for i in range(n) if i != iF)) * 1e3,
               foresight_largest_orders=fs_first, orders=len(orders),
               foresight_largest_every_order=fs_first == len(orders),
               foresight_worst_rank=fs_rank_worst,
               foresight_min_beats_all_max=bool(mn[iF] > max(mx[i] for i in range(n) if i != iF)),
               values_ms={"".join(map(str, f)): x * 1e3 for f, x in vals.items()})
    if not split:
        # the two accountings reviewers quoted: foresight costed under overlap (everything else fixed), and the GPU
        # shortfall costed first from the limit (everything else fixed) or first from the measurement (nothing fixed)
        out["foresight_under_overlap_ms"] = (vals[(1, 0, 1, 1, 1)] - vals[full]) * 1e3
        out["gpu_first_from_limit_ms"] = (vals[(1, 1, 1, 1, 0)] - vals[full]) * 1e3
        out["gpu_first_from_measured_ms"] = (vals[none] - vals[(0, 0, 0, 0, 1)]) * 1e3
    out["regime"] = regime(ci, b_host, opt, info["c_at_limit"])
    return out


def regime(ci, b_host, opt, c):
    """which term of the limit's max binds: host reads of the optimum, or the GPU's datasheet time (balanced by c CPU
    experts); and whether the best online policy's reads also sit under the GPU term (foresight worth 0 with overlap)"""
    S, D, Lk, bg = ci["S"], ci["D"], ci["Lk"], ci["B_gpu"]
    R_opt, R_onl = sum(ci["reads"][opt]), sum(ci["reads"]["online"])
    gpu0 = (D + Lk * S) / bg
    host_opt = R_opt * S / b_host
    return dict(host_reads_opt_ms=host_opt * 1e3, gpu_c0_ms=gpu0 * 1e3, c_at_limit=c, R_opt=R_opt, R_online=R_onl,
                R_engine=sum(ci["reads"]["engine"]),
                bound="host" if host_opt >= gpu0 else "GPU",
                online_reads_under_gpu_term=bool(R_onl <= max(c, 0) + 1e-9) if host_opt < gpu0 else False)


def print_variant(name, rows, names):
    print(f"\n== {name} ==")
    for r in rows:
        print(f"{r['cell']:13s} measured {r['measured_ms']:6.2f} ms, limit {r['limit_ms']:6.2f}, gap {r['gap_ms']:6.2f}; "
              f"eta {r['eta']:.3f}; largest: {r['largest']}; foresight largest in {r['foresight_largest_orders']}/{r['orders']} orders")
        for k in names:
            print(f"    {k:22s} Shapley {r['shapley_ms'][k]:6.3f} ms ({100 * r['shapley_share'][k]:5.1f}%)  fixed order "
                  f"{r['fixed_order_ms'][k]:6.3f}  marginal [{r['marginal_min_ms'][k]:6.3f}, {r['marginal_max_ms'][k]:6.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", choices=LIMITS, default="084", help="the optimum's reads of the `selected` variant")
    ap.add_argument("--bhost", choices=BHOSTS, default="max", help="B_host of the `selected` variant")
    ap.add_argument("--limit-084", default=os.path.join(ROOT, "prereg", "speed_limit_084.json"))
    ap.add_argument("--limit-v2", default=os.path.join(ROOT, "prereg", "speed_limit_v2.json"))
    ap.add_argument("--json")
    a = ap.parse_args()
    sl = json.load(open(a.limit_084))
    v2 = json.load(open(a.limit_v2)) if os.path.exists(a.limit_v2) else None
    if v2 is None and a.limit != "084":
        sys.exit(f"--limit {a.limit} needs {a.limit_v2}")
    txt = open(f"{R}/081_headline_law@vast/concur.txt").read()
    probe = tuple(x * 1e9 for x in bandwidths(txt, 14))
    sums_zc = concurrent_sums(txt)
    sums_all = [float(z) for z in re.findall(r"concurrent t=\d+ pcie=\w+ cpu_gbs [\d.]+ pcie_gbs [\d.]+ sum ([\d.]+)", txt)]
    b_host = {"max": sl["gpt-oss-120b"]["B_host"], "median": float(np.median(sums_all)) * 1e9,
              "median_zc": float(np.median(sums_zc)) * 1e9}
    if v2 is not None:   # the same statistics, as speed_limit_v2.py computed them
        for k, v in (("max", v2["B_host"]["max"]), ("median", v2["B_host"]["median_concurrent"]),
                     ("median_zc", v2["B_host"]["median_concurrent_zerocopy64"])):
            assert abs(b_host[k] - v) < 1e6, (k, b_host[k], v)
    meas = measured()
    cells = [cell_inputs(key, label, C, stats, sl, meas, v2) for key, label, C, stats in CELLS]
    vram = json.load(open(os.path.join(ROOT, "prereg", "vram_084.json")))
    eta_gross = {}
    for key, tag in (("gpt-oss-120b", "g"), ("qwen3-30b-a3b-bf16", "q")):
        st = json.load(open(f"{VRAM_DIR}/srv_{tag}_ours_C128.json"))
        hu = st["host_us_per_step"]
        hw = (hu["app"] + hu["pre"] + hu["inputs"] + hu["launch"] + hu["post"]) * 1e-6
        first = next(c for c in cells if c["model"] == key)
        eta_gross[key] = (1 / vram["ours_c128"][key]["tok_s"] - hw) / ((first["D"] + first["Lk"] * first["S"]) / first["B_gpu"])
    print(f"probe B_c {probe[0] / 1e9:.3f}, B_p {probe[1] / 1e9:.1f}, B_cp {probe[2] / 1e9:.1f} GB/s; concurrent sums {sums_all} "
          f"(zero-copy pairs {sums_zc}): B_host max {b_host['max'] / 1e9:.1f} (the limit's), median {b_host['median'] / 1e9:.1f}, "
          f"zero-copy median {b_host['median_zc'] / 1e9:.1f}")
    print("gross GPU rate factors (RTX PRO 6000, ours with 128 slots): " + ", ".join(f"{k} {v:.3f}" for k, v in eta_gross.items()))
    for c in cells:
        print(f"{c['cell']:13s} reads/token (cpu + copy): " + "; ".join(f"{k} {v[0]:.2f} + {v[1]:.2f} = {v[0] + v[1]:.2f}" for k, v in c["reads"].items()))

    opt_key = {"084": "opt", "exact": "exact", "global": "global"}
    variants = {
        "main": dict(b_host="max", opt="084", gpu_model="rate", split=False,
                     note="084 optimum (scripts/foresight._pol), B_host = highest probe rate (the limit's); GPU shortfall as a "
                          "rate, net, calibrated per cell"),
        "bhost_median": dict(b_host="median_zc", opt="084", gpu_model="rate", split=False,
                             note="084 optimum, B_host = median of the three zero-copy concurrent pairs (77.7 GB/s, the B_cp the "
                                  "engine's law sees for both paths)"),
        "gpu_const": dict(b_host="max", opt="084", gpu_model="const", split=False,
                          note="GPU shortfall as a constant per token instead of a rate"),
        "gpu_gross": dict(b_host="max", opt="084", gpu_model="gross", split=False,
                          note="GPU rate from the all-in-VRAM run (RTX PRO 6000, same datasheet bandwidth); the overlap the "
                               "engine achieves (delta) is credited to every non-overlapped state, so the paper's overlap "
                               "and GPU steps shift by delta"),
        "split6": dict(b_host="max", opt="084", gpu_model="rate", split=True,
                       note="policy (engine's reads -> best online's) and read paths (probe law -> B_host) as separate fixes"),
    }
    if v2 is not None:
        for opt in ("exact", "global"):
            for bh in BHOSTS:
                variants[f"{opt}_{bh}"] = dict(
                    b_host=bh, opt=opt, gpu_model="rate", split=False,
                    note=f"{'exact per-layer' if opt == 'exact' else 'global-pool (L*C slots)'} optimum (mosl.cachesim MIN with "
                         f"bypass, speed_limit_v2.json), B_host = {bh} ({b_host[bh] / 1e9:.1f} GB/s); CPU/copy split of the "
                         f"optimum from the trace; GPU shortfall as a rate, net")
    selected = "main" if (a.limit, a.bhost) == ("084", "max") else f"{a.limit}_{a.bhost}"
    if selected not in variants:
        variants[selected] = dict(b_host=a.bhost, opt=a.limit, gpu_model="rate", split=False,
                                  note=f"084 optimum, B_host = {a.bhost} ({b_host[a.bhost] / 1e9:.1f} GB/s)")
    out = dict(
        inputs=dict(B_probe_gbs=[x / 1e9 for x in probe], concurrent_sums_gbs=sums_all, concurrent_sums_zerocopy_gbs=sums_zc,
                    B_host_gbs={k: v / 1e9 for k, v in b_host.items()}, B_host_max_gbs=b_host["max"] / 1e9,
                    B_host_median_gbs=b_host["median"] / 1e9, B_host_median_zerocopy_gbs=b_host["median_zc"] / 1e9,
                    eta_gross=eta_gross, speed_limit_v2=a.limit_v2 if v2 is not None else None, cells=[
                        dict(cell=c["cell"], C=c["C"], measured_ms=c["T"] * 1e3, host_work_ms=c["hw"] * 1e3,
                             reads_per_token={k: dict(cpu=v[0], copy=v[1], total=v[0] + v[1]) for k, v in c["reads"].items()})
                        for c in cells]),
        fixes=FIX5, paper_steps=PAPER_STEPS, selected=selected, variants={})
    for name, spec in variants.items():
        names = FIX6 if spec["split"] else FIX5
        rows = [analyse(ci, b_host[spec["b_host"]], probe, spec["gpu_model"], eta_gross[ci["model"]], spec["split"],
                        opt_key[spec["opt"]]) for ci in cells]
        if v2 is not None and spec["opt"] != "084" and not spec["split"] and spec["gpu_model"] == "rate":
            # v(all) must be the limit speed_limit_v2.py computed for this optimum and B_host
            key = {("exact", "max"): "exact", ("exact", "median"): "median_bhost", ("exact", "median_zc"): "median_bhost_zerocopy",
                   ("global", "max"): "global"}.get((spec["opt"], spec["b_host"]))
            for r, ci in zip(rows, cells):
                if key:
                    ref = v2["models"][ci["model"]]["rows"][str(ci["C"])]["limits"][key]["t_ms"]
                    assert abs(r["limit_ms"] - ref) < 1e-9, (name, ci["cell"], r["limit_ms"], ref)
        out["variants"][name] = dict(note=spec["note"], B_host=spec["b_host"], B_host_gbs=b_host[spec["b_host"]] / 1e9,
                                     optimum=spec["opt"], gpu_model=spec["gpu_model"], fixes=names, cells=rows)
        print_variant(name, rows, names)
    # regime map and the safe claim, on the selected variant
    sel = out["variants"][selected]["cells"]
    print(f"\n== regime map ({selected}) ==")
    for r in sel:
        g = r["regime"]
        print(f"{r['cell']:13s} {g['bound']}-bound (host reads of optimum {g['host_reads_opt_ms']:.2f} ms vs GPU at c=0 "
              f"{g['gpu_c0_ms']:.2f}; c* {g['c_at_limit']:.1f}, reads opt {g['R_opt']:.1f} / online {g['R_online']:.1f} / "
              f"engine {g['R_engine']:.1f}); largest under Shapley: {r['largest']} (foresight {r['shapley_share']['foresight']:.0%}, "
              f"largest in {r['foresight_largest_orders']}/120 orders, marginal {r['marginal_min_ms']['foresight']:.2f}-"
              f"{r['marginal_max_ms']['foresight']:.2f} ms); foresight under overlap {r['foresight_under_overlap_ms']:.2f} ms")
    out["safe_claim"] = {}
    for name, var in out["variants"].items():
        rows = var["cells"]
        out["safe_claim"][name] = dict(foresight_largest_every_order=[r["cell"] for r in rows if r["foresight_largest_every_order"]],
                                       foresight_largest_shapley=[r["cell"] for r in rows if r["largest"] == "foresight"],
                                       overlap_largest_shapley=[r["cell"] for r in rows if r["largest"] == "overlap"],
                                       regime={r["cell"]: r["regime"]["bound"] for r in rows})
        print(f"{name:16s} foresight largest in every order: {out['safe_claim'][name]['foresight_largest_every_order']}; "
              f"largest Shapley component: foresight in {out['safe_claim'][name]['foresight_largest_shapley']}, "
              f"overlap in {out['safe_claim'][name]['overlap_largest_shapley']}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

"""Can the bound's host term be reached? The bound (Eq. 1) lets a token's host reads stream at the host's best probed
rate, as if all of a token's MIN reads were one transfer. A real engine reads them layer by layer, a whole expert at a
time, and a layer cannot start until its experts are in. This script computes, from MIN's per-layer reads on the AIME
routing of gpt-oss-120b (mosl.cachesim, MIN with bypass, cache carried across problems), a granular host-read bound:
per layer, the n missed experts are split, whole, between the CPU path (c experts at B_c) and the link (n - c at B_p),
both at most B_cp together, and the layers are read one after another:

    T_gran = sum over layers of  min_c max(c S / B_c, (n - c) S / B_p, n S / B_cp)

against the bound's host term R* S / B_host. It writes the per-step per-layer read counts for the GPU microbenchmark
(jobs/ec2/readsched.cu, job 102), prereg/readsched.json and paper/wsg_readsched.tex.

    python scripts/readsched.py            # analysis on every probed host + the count files for job 102
"""
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.environ.get("MOSL_GPU_BRANCH", "/home/claude/gpu-branch"), "jobs", "ec2"))
from fetch_table import bandwidths  # noqa: E402
from mosl.cachesim import simulate  # noqa: E402
from scripts.speed_limit import host_rates  # noqa: E402

RES = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
S = 13253760
TRACE = f"{RES}/084c_gptoss_trace@vast/route_aime25_gptoss.npz"


def min_reads(C, with_adm=False):
    z = np.load(TRACE)
    act = z["act"].astype(np.int64)          # [T, L, k]
    E = int(z["n_expert"])
    T, L, k = act.shape
    out = np.zeros((T, L), np.int32)
    adm = np.zeros((T, L), np.int32)
    for l in range(L):
        miss, a = simulate(act[:, l, :], E, C, "min", bypass=True)
        out[:, l] = miss
        adm[:, l] = a
    return (out, adm) if with_adm else out


def t_gran(n, bc, bp, bb):
    """per-layer whole-expert split, layers in sequence; n [T, L] counts; seconds per token"""
    best = {}
    for m in range(int(n.max()) + 1):
        best[m] = min(max(c * S / bc, (m - c) * S / bp, m * S / bb) for c in range(m + 1))
    tab = np.array([best[m] for m in range(int(n.max()) + 1)])
    return float(tab[n].sum(axis=1).mean())


def t_gran_adm(n, a, bc, bp, bb):
    """the same, with every admission over the link (read once, into its slot) and only bypasses free to go to the CPU"""
    nm = int(n.max()) + 1
    tab = np.full((nm, nm), np.inf)
    for m in range(nm):
        for q in range(m + 1):
            tab[m, q] = min(max(c * S / bc, (m - c) * S / bp, m * S / bb) for c in range(m - q + 1))
    return float(tab[n, a].sum(axis=1).mean())


def main():
    both = {C: min_reads(C, True) for C in (14, 32)}
    counts = {C: v[0] for C, v in both.items()}
    adms = {C: v[1] for C, v in both.items()}
    for C, n in counts.items():
        n.astype(np.uint8).tofile(os.path.join(os.environ.get("MOSL_GPU_BRANCH", "/home/claude/gpu-branch"), "jobs", "ec2", f"minreads_g{C}.bin"))
    rows = []
    for d in sorted(glob.glob(f"{RES}/*@vast")):
        if not os.path.exists(f"{d}/concur.txt") or not os.path.exists(f"{d}/cores.txt"):
            continue
        txt = open(f"{d}/concur.txt").read()
        try:
            cores = int(re.search(r"usable physical cores (\d+)", open(f"{d}/cores.txt").read()).group(1))
            bc, bp, bb = bandwidths(txt, cores - 2 if cores > 4 else 2)
            bh = host_rates(txt) / 1e9
        except Exception:
            continue
        r = dict(host=os.path.basename(d), B_c=bc, B_p=bp, B_cp=bb, B_host=bh)
        for C, n in counts.items():
            Rs = float(n.sum(axis=1).mean())
            tb = Rs * S / (bh * 1e9)
            tg = t_gran(n, bc * 1e9, bp * 1e9, bb * 1e9)
            ta = t_gran_adm(n, adms[C], bc * 1e9, bp * 1e9, bb * 1e9)
            A = float(adms[C].sum(axis=1).mean())
            cap = min(bb / bh, Rs / A * bp / bh, 1.0)   # read once, every admission crosses the link: T >= max(A/B_p, R/B_cp) S
            r[f"C{C}"] = dict(R_star=Rs, A_star=A, t_bound_host_ms=1e3 * tb, t_gran_ms=1e3 * tg, frac=tb / tg, t_gran_adm_ms=1e3 * ta,
                              frac_adm=tb / ta, cap_once=cap, link_over_cpu=bp / bc)
        rows.append(r)
    json.dump(dict(counts_mean={C: float(n.sum(axis=1).mean()) for C, n in counts.items()},
                   adm_share={C: float(adms[C].sum() / counts[C].sum()) for C in counts},
                   per_layer_hist={C: np.bincount(n.ravel()).tolist() for C, n in counts.items()}, hosts=rows),
              open(P("prereg", "readsched.json"), "w"), indent=1)
    M = {}
    panel = [r for r in rows if r["host"].startswith(("099", "100", "101", "096", "095", "097"))]
    for C, nm in ((14, "Low"), (32, "Mid")):
        f = [r[f"C{C}"]["frac"] for r in panel]
        M[f"rsFrac{nm}Min"] = f"{100 * min(f):.0f}"; M[f"rsFrac{nm}Max"] = f"{100 * max(f):.0f}"
        M[f"rsRstar{nm}"] = f"{float(counts[C].sum(axis=1).mean()):.1f}"
        M[f"rsPerLayer{nm}"] = f"{float(counts[C].mean()):.2f}"
        M[f"rsAdmShare{nm}"] = f"{100 * adms[C].sum() / counts[C].sum():.0f}"
        fa = [r[f"C{C}"]["frac_adm"] for r in panel]
        M[f"rsAdm{nm}Min"] = f"{100 * min(fa):.0f}"; M[f"rsAdm{nm}Max"] = f"{100 * max(fa):.0f}"
        fs = [r[f"C{C}"]["frac_adm"] for r in panel if r[f"C{C}"]["link_over_cpu"] < 0.5]
        fb = [r[f"C{C}"]["frac_adm"] for r in panel if r[f"C{C}"]["link_over_cpu"] >= 0.5]
        if fs:
            M[f"rsAdmSlow{nm}Min"] = f"{100 * min(fs):.0f}"; M[f"rsAdmSlow{nm}Max"] = f"{100 * max(fs):.0f}"
        if fb:
            M[f"rsAdmFast{nm}Min"] = f"{100 * min(fb):.0f}"; M[f"rsAdmFast{nm}Max"] = f"{100 * max(fb):.0f}"
    M["rsHosts"] = str(len(panel))
    # the cap on MIN read once, as a share of the gap between the deployed cache and the bound (factorial hosts)
    fsj = P("prereg", "factorial_shapley.json")
    if os.path.exists(fsj):
        byd = {r["host"]: r for r in rows}
        sh = {"slow": [], "fast": []}
        for h in json.load(open(fsj))["hosts"]:
            r = byd.get(h["dir"])
            for cell, C in (("gpt-oss 11%", 14), ("gpt-oss 25%", 32)):
                c = h["cells"].get(cell)
                if not r or not c:
                    continue
                lim, gap = c["limit_ms"], c["effects"]["gap"]
                share = (lim / r[f"C{C}"]["cap_once"] - lim) / gap
                sh["slow" if r[f"C{C}"]["link_over_cpu"] < 0.5 else "fast"].append(
                    dict(host=h["host"], cell=cell, share=share, rest=c["effects"]["rest_share"]))
        if sh["fast"]:
            M["rsCapGapFastMed"] = f"{100 * float(np.median([x['share'] for x in sh['fast']])):.0f}"
        for k, v in sh.items():
            if v:
                M[f"rsCapGap{k.title()}Min"] = f"{100 * min(x['share'] for x in v):.0f}"
                M[f"rsCapGap{k.title()}Max"] = f"{100 * max(x['share'] for x in v):.0f}"
        json.dump(sh, open(P("prereg", "readsched_gap.json"), "w"), indent=1)
        print("cap share of the gap", {k: [(x["host"], x["cell"][-3:], round(x["share"], 2), round(x["rest"], 2)) for x in v] for k, v in sh.items()})
    for C, nm in ((14, "Low"), (32, "Mid")):
        cs = [r[f"C{C}"]["cap_once"] for r in panel if r[f"C{C}"]["link_over_cpu"] < 0.5]
        cf = [r[f"C{C}"]["cap_once"] for r in panel if r[f"C{C}"]["link_over_cpu"] >= 0.5]
        M[f"rsCapSlow{nm}Min"] = f"{100 * min(cs):.0f}"; M[f"rsCapSlow{nm}Max"] = f"{100 * max(cs):.0f}"
        M[f"rsCapFast{nm}Min"] = f"{100 * min(cf):.0f}"; M[f"rsCapFast{nm}Max"] = f"{100 * max(cf):.0f}"
        M[f"rsCapBelow{nm}"] = str(sum(1 for r in panel if r[f"C{C}"]["cap_once"] < 0.995))
    with open(P("paper", "wsg_readsched.tex"), "w") as fo:
        fo.write("% generated by scripts/readsched.py\n")
        for k_ in sorted(M):
            fo.write(f"\\newcommand{{\\{k_}}}{{{M[k_]}}}\n")
    for r in panel:
        print(r["host"][:24], f"Bc {r['B_c']:.0f} Bp {r['B_p']:.0f} Bcp {r['B_cp']:.0f} Bhost {r['B_host']:.0f}",
              {C: (round(r[f"C{C}"]["frac"], 3), round(r[f"C{C}"]["frac_adm"], 3), round(r[f"C{C}"]["cap_once"], 3)) for C in (14, 32)})
    print(M)
    for C, n in counts.items():
        print(C, "per-layer histogram", np.bincount(n.ravel()).tolist(), "R*", round(float(n.sum(axis=1).mean()), 2))


if __name__ == "__main__":
    main()

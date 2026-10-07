"""Where the seconds go, machine by machine: the deployed cache's time per token at a gpt-oss budget split by the law
(eq. sum, with its overlap term) into the GPU's own compute (the median profile at that budget), MIN's host reads at the machine's best rate
(the bound of eq. limit at a host-bound budget) and the reads it makes beyond MIN's; the dot is the measured time.
One launch per machine (the first, by GPU UUID), from jobs 093-104 (prereg/reanalysis_hosts.json) and the new machines
of jobs 105-108 (server processors, EPYC and Xeon, are drawn but left out of the shares).
Writes paper/figs/decomp.pdf and paper/wsg_decomp.tex.

    python scripts/fig_decomp.py
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from scripts.speed_limit import host_rates  # noqa: E402
from scripts.factorial_shapley import limit1_of  # noqa: E402

RES = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
S = 13253760
CELL = {14: "gpt-oss 11%", 32: "gpt-oss 25%"}
UNSTABLE = ("106c", "106d", "107c", "107e", "108b")
NEW_IN_TEST = ("105a", "105b", "107b", "107d", "108a", "108d", "108f")   # machines first rented for a registered test of eq. sum   # wrong outputs, or rounds further apart than 2%


def uuid_of(d):
    for ln in open(f"{d}/nvidia-smi-q.txt"):
        if "GPU UUID" in ln:
            return ln.split(":", 1)[1].strip()
    return d


def cpu_of(d):
    import re
    for f in ("cpu.txt", "lscpu.txt"):
        if os.path.exists(f"{d}/{f}"):
            m = re.search(r"Model name:\s*(.+)", open(f"{d}/{f}").read())
            if m:
                return m.group(1).strip()
    return ""


def short(cpu):
    import re
    s = re.sub(r"\s+\d+-Cores?(\s+Processor)?", "", cpu)
    s = re.sub(r"\s+Processor|\s+CPU.*$|\(R\)|\(TM\)|™|®", "", s)
    s = s.replace("AMD ", "").replace("Ryzen Threadripper PRO", "TR PRO").replace("Ryzen Threadripper", "TR").replace("Ryzen ", "R")
    if "Eng Sample" in s:
        return "AMD ES, 2 sockets"
    s = s.replace("Intel ", "").replace("13th Gen ", "").replace("Core Ultra 9", "U9").replace("Core ", "").replace("Xeon Platinum", "Xeon")
    return re.sub(r"\s+", " ", s).strip()[:20]


def base_of(d, C):
    """(base ms, counted reads per token) of a launch: one file per cell (jobs <= 105) or per round (job 106)"""
    fs = [f"{d}/ec_g_C{C}.jsonl"] if os.path.exists(f"{d}/ec_g_C{C}.jsonl") else sorted(glob.glob(f"{d}/ec_g_C{C}_r*.jsonl"))
    ms, mi, ad = [], [], []
    for f in fs:
        tag = os.path.basename(f)[len("ec_g_"):-len(".jsonl")]
        rows = [json.loads(l) for l in open(f) if l.strip()]
        v = [r["decode_ms"] / r["n_decode"] for r in rows if r["config"].rsplit("stats=", 1)[-1].endswith(f"st_g_{tag}_base.json")]
        st = f"{d}/st_g_{tag}_base.json"
        if v and os.path.exists(st):
            s = json.load(open(st)); n = max(1, s["steps"])
            ms.append(np.mean(v)); mi.append(s["misses"] / n); ad.append(s["admits"] / n)
    return (float(np.mean(ms)), float(np.mean(mi)), float(np.mean(ad))) if ms else (None, None, None)


def best_rates(txt):
    """the probe's best CPU-only and best link-only readings, GB/s"""
    import re
    cpu = [float(x) for x in re.findall(r"^cpu_read_gbs t=\d+ ([\d.]+)", txt, re.M)]
    pcie = [float(x) for x in re.findall(r"^pcie_\w+_gbs [^\n]*? ([\d.]+)$", txt, re.M)]
    return max(cpu), max(pcie)


def dep_bound(lim, B, Bc, Bp, tgpu):
    """the dependency-aware bound (ms): reads copied over the link can be issued ahead and overlap everything, but an
    expert run on the CPU needs its layer's attention output and the next layer's attention needs its result, so CPU
    reads sit in series with the GPU's non-expert work. With a share f of MIN's reads on the CPU:
    max((1-f) R*S/B_p, T_GPU + f R*S/B_c), minimised over f, and never below Eq. (1)."""
    a, b = lim * B / Bp, lim * B / Bc
    f = min(1.0, max(0.0, (a - tgpu) / (a + b)))
    return max(lim, max((1 - f) * a, tgpu + f * b))


def machines(C):
    from scripts.sumlaw_paper import profiles
    tgpu = min(p["nonexpert"] for p in profiles())
    ra = json.load(open(P("prereg", "reanalysis.json")))
    G = ra["sum_law"][str(C)]["G"]
    seen, rows = set(), []
    hosts = json.load(open(P("prereg", "reanalysis_hosts.json")))["launches"]
    dirs = [f"{RES}/{h['dir']}" for h in hosts] + sorted(glob.glob(f"{RES}/105?_sumlaw@vast")) + sorted(glob.glob(f"{RES}/106?_onlineadmit@vast")) + sorted(glob.glob(f"{RES}/107?_newhosts@vast")) + sorted(glob.glob(f"{RES}/108?_smallhosts@vast"))
    for d in dirs:
        if not os.path.exists(f"{d}/concur.txt") or not os.path.exists(f"{d}/nvidia-smi-q.txt"):
            continue
        u = uuid_of(d)
        if u in seen:
            continue
        T, Mi, A = base_of(d, C)
        if T is None:
            continue
        if os.path.basename(d)[:4] in UNSTABLE:   # job 106's hosts whose deployed cache varied > 2% between rounds
            continue
        seen.add(u)
        B = host_rates(open(f"{d}/concur.txt").read()) / 1e9
        lim = limit1_of(d).get(CELL[C])
        Gh, prof = G, False   # the host's own profile where one was taken and captured the decode kernels
        if os.path.exists(f"{d}/g_prof.json"):
            gp = json.load(open(f"{d}/g_prof.json")); v = (gp.get(f"G{C}") or gp.get(f"C{C}") or {}).get("G_prof_ms")
            if v and v > 1.0:
                Gh, prof = v, True
        tm = lambda r: r * S / (B * 1e9) * 1e3  # noqa: E731
        tl = Gh + tm(Mi + A)
        for _ in range(100):   # the relation with its overlap term (eq. sum)
            tl = Gh + tm(Mi + A * (1 - Gh / tl))
        import re
        cores = int(re.search(r"usable physical cores (\d+)", open(f"{d}/cores.txt").read()).group(1))
        m = re.search(r"available: (\d+) nodes", open(f"{d}/numa.txt").read()) if os.path.exists(f"{d}/numa.txt") else None
        server = bool(re.search(r"EPYC|Xeon|Eng Sample", cpu_of(d)))   # server processors (after jobs 107-108)
        new = os.path.basename(d)[:4] in NEW_IN_TEST
        Bc, Bp = best_rates(open(f"{d}/concur.txt").read())
        rows.append(dict(dir=os.path.basename(d), cpu=short(cpu_of(d)), B=B, T=T, G=Gh, prof=prof, min_ms=lim, law=tl,
                         excess_ms=tl - Gh - lim, server=server, new=new, dep=dep_bound(lim, B, Bc, Bp, tgpu),
                         eq2=tgpu + lim, ratio=Bp / Bc))
    return rows


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = {}
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 4.1), sharey=False)
    cols = ("#3a6ea5", "#7a7974", "#d9822b")
    for ax, C, nm in zip(axs, (14, 32), ("Low", "Mid")):
        rows = machines(C)
        rows = sorted([r for r in rows if not r["server"]], key=lambda r: -r["T"]) + sorted([r for r in rows if r["server"]], key=lambda r: -r["T"])
        # unique labels: repeated CPU models numbered in order; machines first rented for a registered test marked
        cnt, seenl = {}, {}
        for r in rows:
            cnt[r["cpu"]] = cnt.get(r["cpu"], 0) + 1
        for r in sorted(rows, key=lambda r: r["dir"]):   # numbered by launch order, the same in both panels
            lab = r["cpu"]
            if cnt[lab] > 1:
                seenl[lab] = seenl.get(lab, 0) + 1; lab = f"{lab} ({seenl[lab]})"
            r["label"] = lab + (" *" if r["new"] else "")
        y = np.arange(len(rows))
        g = np.array([r["G"] for r in rows]); mn = np.array([r["min_ms"] for r in rows]); ex = np.array([r["excess_ms"] for r in rows])
        ax.barh(y, g, color=cols[0], height=0.72, label="GPU compute (profiled)", edgecolor="white", linewidth=0.4)
        ax.barh(y, mn, left=g, color=cols[1], height=0.72, label="MIN's host reads (the bound)", edgecolor="white", linewidth=0.4)
        ax.barh(y, ex, left=g + mn, color=cols[2], height=0.72, label="reads beyond MIN's", edgecolor="white", linewidth=0.4)
        short_ = np.array([max(0.0, r["T"] - r["law"]) for r in rows])
        ax.barh(y, short_, left=g + mn + ex, color="white", height=0.72, hatch="////", edgecolor="#b5651d", linewidth=0.4,
                label="reading below the machine's rate")
        ax.plot([r["T"] for r in rows], y, "o", color="black", ms=3.2, label="measured", zorder=5)
        ax.set_yticks(y); ax.set_yticklabels([r["label"] for r in rows], fontsize=6)
        ns = sum(not r["server"] for r in rows)
        ax.axhline(ns - 0.5, color="#52514e", lw=0.6, ls="--")
        ax.invert_yaxis()
        ax.set_xlabel("ms per token", fontsize=8); ax.tick_params(axis="x", labelsize=7)
        ax.set_title(CELL[C].replace("%", "\\%") if False else CELL[C], fontsize=9)
        ax.grid(axis="x", color="#e4e4e2", lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        dk = np.array([not r["server"] for r in rows])
        sv = [r for r in rows if r["server"]]
        if sv:
            M[f"dcServ{nm}N"] = str(len(sv))
            M[f"dcServShort{nm}"] = "; ".join(f"{r['cpu']} {max(0, round(100 * (r['T'] - r['law']) / r['T']))}\\%" for r in sorted(sv, key=lambda r: r["T"] - r["law"]))
        g, mn, ex, rows = g[dk], mn[dk], ex[dk], [r for r, k in zip(rows, dk) if k]
        M[f"dcDeskShort{nm}Max"] = f"{100 * max(max(0.0, r['T'] - r['law']) / r['T'] for r in rows):.0f}"
        tot = g + mn + ex
        T = np.array([r["T"] for r in rows])
        M[f"dcMachines{nm}"] = str(len(rows))
        M[f"dcShareG{nm}Min"] = f"{100 * np.min(g / T):.0f}"; M[f"dcShareG{nm}Max"] = f"{100 * np.max(g / T):.0f}"
        M[f"dcShareMin{nm}Min"] = f"{100 * np.min(mn / T):.0f}"; M[f"dcShareMin{nm}Max"] = f"{100 * np.max(mn / T):.0f}"
        dep = np.array([r["dep"] for r in rows]); eq2 = np.array([r["eq2"] for r in rows])
        M[f"dcShareDep{nm}Min"] = f"{100 * np.min(dep / T):.0f}"; M[f"dcShareDep{nm}Max"] = f"{100 * np.max(dep / T):.0f}"
        M[f"dcShareDem{nm}Min"] = f"{100 * np.min(eq2 / T):.0f}"; M[f"dcShareDem{nm}Max"] = f"{100 * np.max(eq2 / T):.0f}"
        M[f"dcDepOverLim{nm}Max"] = f"{np.max(dep / mn):.2f}"
        M[f"dcDepOverLim{nm}Med"] = f"{np.median(dep / mn):.2f}"
        M[f"dcShareEx{nm}Min"] = f"{100 * np.min(ex / T):.0f}"; M[f"dcShareEx{nm}Max"] = f"{100 * np.max(ex / T):.0f}"
        gap = np.array([r["G"] / (r["T"] - r["min_ms"]) for r in rows])
        M[f"dcGapG{nm}Min"] = f"{100 * gap.min():.0f}"; M[f"dcGapG{nm}Max"] = f"{100 * gap.max():.0f}"
        M[f"dcLawErr{nm}Med"] = f"{100 * np.median(tot / T - 1):.0f}".replace("-", "$-$")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.90))
    os.makedirs(P("paper", "figs"), exist_ok=True)
    fig.savefig(P("paper", "figs", "decomp.pdf")); fig.savefig(P("paper", "figs", "decomp.png"), dpi=170)
    with open(P("paper", "wsg_decomp.tex"), "w") as f:
        f.write("% generated by scripts/fig_decomp.py\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    print(M)


if __name__ == "__main__":
    main()

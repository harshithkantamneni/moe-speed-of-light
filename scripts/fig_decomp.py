"""Where the seconds go, machine by machine: the deployed cache's time per token at a gpt-oss budget split by the law
(eq. sum, with its overlap term) into the GPU's own compute (the median profile at that budget), MIN's host reads at the machine's best rate
(the bound of eq. limit at a host-bound budget) and the reads it makes beyond MIN's; the dot is the measured time.
One launch per machine (the first, by GPU UUID), from jobs 093-104 (prereg/reanalysis_hosts.json) and the new machines
of jobs 105-106. Writes paper/figs/decomp.pdf and paper/wsg_decomp.tex.

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
UNSTABLE = ("106c", "106d")


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


def machines(C):
    ra = json.load(open(P("prereg", "reanalysis.json")))
    G = ra["sum_law"][str(C)]["G"]
    seen, rows = set(), []
    hosts = json.load(open(P("prereg", "reanalysis_hosts.json")))["launches"]
    dirs = [f"{RES}/{h['dir']}" for h in hosts] + sorted(glob.glob(f"{RES}/105?_sumlaw@vast")) + sorted(glob.glob(f"{RES}/106?_onlineadmit@vast"))
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
        tm = lambda r: r * S / (B * 1e9) * 1e3  # noqa: E731
        tl = G + tm(Mi + A)
        for _ in range(100):   # the law with its overlap term (eq. sum)
            tl = G + tm(Mi + A * (1 - G / tl))
        rows.append(dict(dir=os.path.basename(d), cpu=short(cpu_of(d)), B=B, T=T, G=G, min_ms=lim, law=tl, excess_ms=tl - G - lim))
    return rows


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = {}
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.9), sharey=False)
    cols = ("#3a6ea5", "#7a7974", "#d9822b")
    for ax, C, nm in zip(axs, (14, 32), ("Low", "Mid")):
        rows = sorted(machines(C), key=lambda r: -r["T"])
        y = np.arange(len(rows))
        g = np.array([r["G"] for r in rows]); mn = np.array([r["min_ms"] for r in rows]); ex = np.array([r["excess_ms"] for r in rows])
        ax.barh(y, g, color=cols[0], height=0.72, label="GPU compute (profiled)", edgecolor="white", linewidth=0.4)
        ax.barh(y, mn, left=g, color=cols[1], height=0.72, label="MIN's host reads (the bound)", edgecolor="white", linewidth=0.4)
        ax.barh(y, ex, left=g + mn, color=cols[2], height=0.72, label="reads beyond MIN's", edgecolor="white", linewidth=0.4)
        ax.plot([r["T"] for r in rows], y, "o", color="black", ms=3.2, label="measured", zorder=5)
        ax.set_yticks(y); ax.set_yticklabels([r["cpu"] for r in rows], fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("ms per token", fontsize=8); ax.tick_params(axis="x", labelsize=7)
        ax.set_title(CELL[C].replace("%", "\\%") if False else CELL[C], fontsize=9)
        ax.grid(axis="x", color="#e4e4e2", lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        tot = g + mn + ex
        T = np.array([r["T"] for r in rows])
        M[f"dcMachines{nm}"] = str(len(rows))
        M[f"dcShareG{nm}Min"] = f"{100 * np.min(g / T):.0f}"; M[f"dcShareG{nm}Max"] = f"{100 * np.max(g / T):.0f}"
        M[f"dcShareMin{nm}Min"] = f"{100 * np.min(mn / T):.0f}"; M[f"dcShareMin{nm}Max"] = f"{100 * np.max(mn / T):.0f}"
        M[f"dcShareEx{nm}Min"] = f"{100 * np.min(ex / T):.0f}"; M[f"dcShareEx{nm}Max"] = f"{100 * np.max(ex / T):.0f}"
        M[f"dcLawErr{nm}Med"] = f"{100 * np.median(tot / T - 1):.0f}".replace("-", "$-$")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.94))
    os.makedirs(P("paper", "figs"), exist_ok=True)
    fig.savefig(P("paper", "figs", "decomp.pdf")); fig.savefig(P("paper", "figs", "decomp.png"), dpi=170)
    with open(P("paper", "wsg_decomp.tex"), "w") as f:
        f.write("% generated by scripts/fig_decomp.py\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    print(M)


if __name__ == "__main__":
    main()

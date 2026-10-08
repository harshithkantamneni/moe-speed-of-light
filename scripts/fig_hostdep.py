"""Which way of spending foresight pays depends on the machine: the speed of each state relative to the deployed cache
against the host's link-to-CPU read-rate ratio (B_p / B_c from the job's own probe), at the two gpt-oss host-bound
budgets, on every host that ran the factorial (O3: job 095; O4, O5: job 096; O6: job 097, which ran fetch and the
paced prefetch) and on the job 099 panel. Writes paper/figs/hostdep.pdf and macros paper/wsg_hostdep.tex.

    python scripts/fig_hostdep.py
"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
SERIES = [("fetchplan", "Few-1R", "#b8860b", "*"), ("fetch", "MIN-1R", "#1f4e79", "o"), ("both3p", "read-ahead oracle", "#2e8b57", "s"),
          ("bypass", "MIN-2R", "#c55a11", "^"), ("foa", "Dep-1R", "#7a7974", "v"),
          ("aa", "admit every miss", "#4a3aa7", "D"), ("pf", "LA (layer-ahead copy)", "#a4243b", "P")]
CARD2 = "#7b2cbf"   # the RTX 4090 machines of the registered second-card tests (jobs 111 and 112)
CELLS = ["gpt-oss 11%", "gpt-oss 25%"]


def points():
    pts = []   # (host, ratio B_p/B_c, cell, state, speed ratio)
    for job, name in (("095", "O3"), ("096a", "O4"), ("096b", "O5"), ("097b", "O6")):
        p = P("prereg", f"foresight_{job}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        r = d["host"]["B_p"] / d["host"]["B_c"]
        for c in d["cells"]:
            if c["label"] not in CELLS:
                continue
            for k, _, _, _ in SERIES:
                if k in c["runs"]:
                    pts.append((name, r, c["label"], k, c["runs"][k]["ratio_to_base"][0]))
    pp = P("prereg", "panel_099.json")
    if os.path.exists(pp):
        for h in json.load(open(pp))["hosts"]:
            r = h["B_p"] / h["B_c"]
            for lab in CELLS:
                c = h["cells"].get(lab)
                if not c:
                    continue
                for k, _, _, _ in SERIES:
                    key = f"{k}/base"
                    if key in c["speed"]:
                        pts.append((h["host"], r, lab, k, c["speed"][key][0]))
    # job 100's new machines (its relaunches of Pd and Ph are left out: the same machines are already here)
    for pj in (P("prereg", "job100.json"), P("prereg", "job101.json"), P("prereg", "job103.json"), P("prereg", "job104.json"),
               P("prereg", "job105.json")):
        if not os.path.exists(pj):
            continue
        for h in json.load(open(pj))["hosts"]:
            if h["job"] in ("100a", "100c", "101a", "103a", "103d"):   # relaunches of panel machines (103d: Pg)
                continue
            # job 104 relaunches job 103's machines and Pf: only its plan states are new
            only = ("fetchplan",) if h["job"].startswith("104") else None
            # job 105's relaunches of Pf and O4: the layer-ahead copy is new on both, the fewest-admission set on O4
            if h["job"] == "105e":
                only = ("pf",)
            elif h["job"] == "105f":
                only = ("pf", "fetchplan")
            r = h["B_p"] / h["B_c"]
            for C, lab in (("14", "gpt-oss 11%"), ("32", "gpt-oss 25%")):
                c = h["cells"].get(C)
                if not c:
                    continue
                for k, _, _, _ in SERIES:
                    if only and k not in only:
                        continue
                    if k == "fetchplan" and not h.get("planned", {}).get(C, h.get("planned", {}).get(int(C), False)):
                        continue
                    if k in c["speed"]:
                        pts.append((h["job"], r, lab, k, c["speed"][k][0]))
    # job 106: 106e is panel host Pe again (same GPU), whose fewest-admission set is new; 106a and 106b relaunch the
    # 9960X and Pf (their plan states are already here); 106c computed wrong outputs and 106d varied by 39% between rounds
    pj = P("prereg", "job106.json")
    if os.path.exists(pj):
        for h in json.load(open(pj))["hosts"]:
            if h["job"] != "106e":
                continue
            c = h["cells"].get("g14")
            for k in ("fetchplan",):
                if c and c["speed"].get(k):
                    pts.append((h["job"], h["ratio"], "gpt-oss 11%", k, c["speed"][k][0]))
    # job 107: the two machines new to it that passed every validity check (the fewest-admission set copied in the step)
    pj = P("prereg", "job107.json")
    if os.path.exists(pj):
        for h in json.load(open(pj))["hosts"]:
            c = h["cells"].get("g14")
            if h.get("valid") and c and c["speed"].get("fetchplan"):
                pts.append((h["job"], h["ratio"], "gpt-oss 11%", "fetchplan", c["speed"]["fetchplan"][0]))
    # job 113: its two new RTX 5090 machines' fewest-admission set copied in the step (with the machine's fetch table)
    pj = P("prereg", "job113.json")
    if os.path.exists(pj):
        for h in json.load(open(pj))["hosts"]:
            for C, lab in (("14", "gpt-oss 11%"), ("32", "gpt-oss 25%")):
                c = h["cells"].get(C) or {}
                if h.get("valid") and c.get("ratio", {}).get("fetchplan"):
                    pts.append((h["job"], h["ratio"], lab, "fetchplan", c["ratio"]["fetchplan"]))
    return pts


PLOTTED = ("fetchplan", "fetch", "bypass", "pf")   # the series the text discusses; the macros use every series


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = points()
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, lab in zip(axs, CELLS):
        ax.axhline(1.0, color="#52514e", lw=0.8)
        for k, name, col, mk in SERIES:
            if k not in PLOTTED:
                continue
            xs = [p[1] for p in pts if p[2] == lab and p[3] == k]
            ys = [p[4] for p in pts if p[2] == lab and p[3] == k]
            if xs:
                ax.plot(xs, ys, mk, color=col, ms=11 if mk == "*" else 5.5, mec="black" if mk == "*" else "white", mew=0.5,
                        label=name, ls="none", zorder=5 if mk == "*" else 3)
                if len(xs) >= 4:
                    b, a = np.polyfit(np.log(xs), ys, 1)
                    xx = np.linspace(min(xs), max(xs), 50)
                    ax.plot(xx, a + b * np.log(xx), "-", color=col, lw=0.8, alpha=0.6)
        # the RTX 4090 machines of the registered second-card tests (jobs 111 and 112): the series' marker in purple, not
        # in the fits or the macros
        C = "14" if lab == "gpt-oss 11%" else "32"
        for pj in (P("prereg", "job111.json"), P("prereg", "job112.json")):
            if not os.path.exists(pj):
                continue
            for h in json.load(open(pj))["hosts"]:
                c = h["cells"].get(C) or h["cells"].get(int(C))
                if not h.get("valid") or h.get("card", "4090") != "4090" or not c or not c.get("rounds"):
                    continue
                for k, name, col, mk in SERIES:
                    if k in PLOTTED and k in c["ratio"]:
                        ax.plot([h["ratio"]], [c["ratio"][k]], mk, color=CARD2, ms=11 if mk == "*" else 5.5, mec="white",
                                mew=0.5, ls="none", zorder=6)
        ax.set_xscale("log")
        ax.set_xlim(0.22, 1.25)
        ax.set_xticks([0.25, 0.35, 0.5, 0.7, 1.0])
        ax.set_xticklabels(["0.25", "0.35", "0.5", "0.7", "1.0"])
        ax.minorticks_off()
        ax.set_title(lab + "$^\\dagger$", fontsize=9)
        ax.set_xlabel("link / CPU read rate (the host's probe)", fontsize=8)
        ax.tick_params(labelsize=8)
    axs[0].set_ylabel("speed relative to deployed", fontsize=8.5)
    from matplotlib.lines import Line2D
    hh, ll = axs[1].get_legend_handles_labels()
    if os.path.exists(P("prereg", "job111.json")):
        hh.append(Line2D([0], [0], marker="o", color=CARD2, mec="white", ls="none", ms=5.5, mew=0.5))
        ll.append("purple: RTX 4090 (registered tests)")
    axs[1].legend(hh, ll, fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout(pad=0.3)
    fig.savefig(P("paper", "figs", "hostdep.pdf"))
    fig.savefig(P("paper", "figs", "hostdep.png"), dpi=160)
    # macros: the link/CPU ratio range, and the in-step single read's range of speed ratios
    M = {}
    r = sorted({p[1] for p in pts})
    M["hdHosts"] = str(len({p[0] for p in pts}))
    M["hdRatioMin"] = f"{min(r):.2f}"; M["hdRatioMax"] = f"{max(r):.2f}"
    for k, nm in (("fetch", "Fetch"), ("both3p", "Paced"), ("bypass", "Bypass"), ("aa", "Aa"), ("foa", "Foa"), ("fetchplan", "FetchPlan")):
        v = [p[4] for p in pts if p[3] == k]
        if v:
            M[f"hd{nm}Min"] = f"{min(v):.2f}"; M[f"hd{nm}Max"] = f"{max(v):.2f}"
    # correlation of the in-step single read's ratio with log(B_p / B_c)
    fx = [(np.log(p[1]), p[4]) for p in pts if p[3] == "fetch"]
    if len(fx) >= 4:
        M["hdFetchCorr"] = f"{np.corrcoef(*zip(*fx))[0, 1]:.2f}"
    # per budget, and without the two hosts whose link reads at less than half their CPU's rate
    for lab, nm in (("gpt-oss 11%", "Low"), ("gpt-oss 25%", "Mid")):
        fx = [(np.log(p[1]), p[4]) for p in pts if p[3] == "fetch" and p[2] == lab]
        if len(fx) >= 4:
            M[f"hdFetchCorr{nm}"] = f"{np.corrcoef(*zip(*fx))[0, 1]:.2f}"
            M[f"hdFetchN{nm}"] = str(len(fx))
        fx = [(np.log(p[1]), p[4]) for p in pts if p[3] == "fetch" and p[2] == lab and p[1] >= 0.5]
        if len(fx) >= 4:
            M[f"hdFetchCorrFast{nm}"] = f"{np.corrcoef(*zip(*fx))[0, 1]:.2f}"
    with open(P("paper", "wsg_hostdep.tex"), "w") as f:
        f.write("% generated by scripts/fig_hostdep.py\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    print(M)


if __name__ == "__main__":
    main()

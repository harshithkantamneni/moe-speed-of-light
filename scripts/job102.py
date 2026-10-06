"""Job 102: can the bound's host term be reached? Reads the microbenchmark of jobs/ec2/readsched.cu (MIN's per-layer
host reads of gpt-oss-120b at C = 14 and 32, the first 4,000 steps, through CPU helper threads and the PCIe copy
engine), puts each mode's time per token against the bound's host term on the same host (reads per token x S / B_host,
B_host the probe's highest rate, as Eq. 1) and against the analytic per-layer schedule of scripts/readsched.py, scores
the predictions of jobs/102_readsched@vast.sh by machine, and writes prereg/job102.json, prereg/scorecard_102.json and
paper/wsg_job102.tex.

    python scripts/job102.py
"""
import glob
import json
import os
import re
import sys
from collections import Counter

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from scripts.panel_099 import _status  # noqa: E402
from scripts.readsched import S, t_gran  # noqa: E402
from scripts.speed_limit import host_rates  # noqa: E402

RES = os.environ.get("MOSL_RESULTS", "/home/claude/gpu-branch/results")
GPU = os.environ.get("MOSL_GPU_BRANCH", "/home/claude/gpu-branch")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
L = 36
NAMES = {"102a": "Pd again", "102b": "9950X", "102c": "9800X3D"}
MODES = ("layer", "token", "layer_link", "layer_cpu")


def parse(path):
    """{mode: [ms per token by repetition]}, reads per token, steps"""
    out, reads, steps = {}, None, None
    for m in re.finditer(r"readsched mode=(\w+) rep=(\d+) steps=(\d+) ms_per_token ([\d.]+) reads_per_token ([\d.]+)", open(path).read()):
        out.setdefault(m.group(1), []).append(float(m.group(4)))
        reads, steps = float(m.group(5)), int(m.group(3))
    return out, reads, steps


def analytic_modes(n, bc, bp, bb):
    """each mode's time per token (ms) from the probe's rates alone, no latency: the schedule readsched.cu runs"""
    g = 1e9 / S
    split = np.array([min(range(m + 1), key=lambda c: (max(c / bc, (m - c) / bp, m / bb), c)) for m in range(int(n.max()) + 1)])
    c = split[n]
    lay = np.maximum.reduce([c / bc, (n - c) / bp, n / bb]).sum(axis=1)
    ct, pt = c.sum(axis=1), (n - c).sum(axis=1)
    tok = np.maximum.reduce([ct / bc, pt / bp, (ct + pt) / bb])
    return {k: 1e3 * float(v.mean()) / g for k, v in
            dict(layer=lay, token=tok, layer_link=(n / bp).sum(axis=1), layer_cpu=(n / bc).sum(axis=1)).items()}


def load_host(d):
    rates = re.search(r"rates B_c B_p B_cp: ([\d.]+) ([\d.]+) ([\d.]+) \(helpers (\d+)\)", open(f"{d}/rates.txt").read())
    bc, bp, bb, H = float(rates.group(1)), float(rates.group(2)), float(rates.group(3)), int(rates.group(4))
    bh = host_rates(open(f"{d}/concur.txt").read()) / 1e9
    cpu = re.search(r"Model name:\s*(.+)", open(f"{d}/cpu.txt").read()) if os.path.exists(f"{d}/cpu.txt") else None
    h = dict(B_c=bc, B_p=bp, B_cp=bb, B_host=bh, helpers=H, cpu=cpu.group(1).strip() if cpu else "", cells={})
    for C in (14, 32):
        f = f"{d}/readsched_C{C}.txt"
        if not os.path.exists(f):
            continue
        ms, R, T = parse(f)
        if not ms or R is None:
            continue
        n = np.fromfile(f"{GPU}/jobs/ec2/minreads_g{C}.bin", dtype=np.uint8).reshape(-1, L)[:T].astype(np.int64)
        t_bound = 1e3 * R * S / (bh * 1e9)
        t_an = 1e3 * t_gran(n, bc * 1e9, bp * 1e9, bb * 1e9)
        an = analytic_modes(n, bc, bp, bb)
        c = dict(steps=T, reads_per_token=R, t_bound_host_ms=t_bound, t_analytic_ms=t_an, frac_analytic=t_bound / t_an, ms=ms,
                 frac={k: t_bound / float(np.mean(v)) for k, v in ms.items()},
                 rep_dev={k: abs(v[0] / v[1] - 1) for k, v in ms.items() if len(v) == 2},
                 analytic_ms=an, meas_over_an={k: float(np.mean(ms[k])) / an[k] for k in an if k in ms})
        h["cells"][C] = c
    return h


def main():
    hosts = {os.path.basename(d)[:4]: load_host(d) for d in sorted(glob.glob(f"{RES}/102?_readsched@vast"))
             if os.path.exists(f"{d}/rates.txt") and os.path.exists(f"{d}/concur.txt")}
    hosts = {k: v for k, v in hosts.items() if v["cells"]}
    clauses = []

    def add(cid, short, typ, meas, ok, thr, host):
        stt, why = _status(meas, None, ok)
        clauses.append(dict(id=cid, short=short, type=typ, measured=None if meas is None else round(float(meas), 4), ci=None,
                            threshold=thr, status=stt, why=why, host=host))
    for job, h in hosts.items():
        nm = NAMES.get(job, job)
        for C in (14, 32):
            c = h["cells"].get(C)
            g = f"C{C}"
            if c is None:
                for k in ("P1", "P2floor", "P2below", "P4", "P5"):
                    add(f"{job}-{k}-{g}", f"{k} at C = {C} ({nm})", "band", None, lambda v: True, "", nm)
                continue
            fr = c["frac"]
            add(f"{job}-P1-{g}", f"token mode reaches >= 0.80 of the bound's host term ({nm}, C = {C})", "band", fr.get("token"),
                lambda v: v >= 0.80, ">= 0.80", nm)
            add(f"{job}-P2floor-{g}", f"layer mode reaches >= 0.50 of it ({nm}, C = {C})", "band", fr.get("layer"),
                lambda v: v >= 0.50, ">= 0.50", nm)
            add(f"{job}-P2below-{g}", f"layer mode at least 0.03 below the analytic per-layer fraction {c['frac_analytic']:.2f} ({nm}, C = {C})",
                "band", None if "layer" not in fr else c["frac_analytic"] - fr["layer"], lambda v: v >= 0.03, ">= 0.03", nm)
            r = h["B_p"] / h["B_c"]
            if "layer_link" in c["ms"] and "layer" in c["ms"]:
                slow = np.mean(c["ms"]["layer_link"]) / np.mean(c["ms"]["layer"]) - 1
                if h["B_c"] >= 1.5 * h["B_p"]:
                    add(f"{job}-P4-{g}", f"layer_link > 10% slower than layer (CPU path {1 / r:.2f}x the link; {nm}, C = {C})", "band", slow,
                        lambda v: v > 0.10, "> 0.10", nm)
                elif 0.8 <= r <= 1.25:
                    add(f"{job}-P4-{g}", f"layer_link within 10% of layer (paths match, link/CPU {r:.2f}; {nm}, C = {C})", "band", abs(slow),
                        lambda v: v <= 0.10, "<= 0.10", nm)
            if c["rep_dev"]:
                add(f"{job}-P5-{g}", f"two repetitions agree within 3%, worst mode ({nm}, C = {C})", "band", max(c["rep_dev"].values()),
                    lambda v: v <= 0.03, "<= 0.03", nm)
        if 14 in h["cells"] and 32 in h["cells"]:
            a, b = h["cells"][14]["frac"].get("layer"), h["cells"][32]["frac"].get("layer")
            add(f"{job}-P3", f"layer mode's fraction lower at C = 32 than at 14 ({nm})", "sign", None if a is None or b is None else a - b,
                lambda v: v > 0, "> 0", nm)
    out = []
    for job, h in hosts.items():
        hh = dict(h, job=job, name=NAMES.get(job, job), cells={str(C): c for C, c in h["cells"].items()})
        out.append(hh)
    json.dump(dict(hosts=out), open(P("prereg", "job102.json"), "w"), indent=1)
    json.dump(dict(job="102", script="jobs/102_readsched@vast.sh", commit="e089377", scored_by="scripts/job102.py (machine)", clauses=clauses),
              open(P("prereg", "scorecard_102.json"), "w"), indent=1)
    st = Counter(c["status"] for c in clauses)
    M = dict(jdHosts=str(len(hosts)), jdClauses=str(len(clauses)), jdHeld=str(st.get("held", 0)), jdPoint=str(st.get("held (point)", 0)),
             jdFailed=str(st.get("failed", 0)), jdUntested=str(st.get("untested", 0)))
    cells = [(job, C, c) for job, h in hosts.items() for C, c in h["cells"].items()]
    if cells:
        def rng(key, f, fmt="{:.0f}", scale=100):
            v = [f(c) for _, _, c in cells if f(c) is not None]
            M[f"{key}Min"] = fmt.format(scale * min(v)); M[f"{key}Max"] = fmt.format(scale * max(v))
        rng("jdToken", lambda c: c["frac"].get("token"))
        rng("jdLayer", lambda c: c["frac"].get("layer"))
        rng("jdAnalytic", lambda c: c["frac_analytic"])
        rng("jdLayerOverAn", lambda c: c["frac"]["layer"] / c["frac_analytic"] if "layer" in c["frac"] else None)
        rng("jdLink", lambda c: c["frac"].get("layer_link"))
        rng("jdCpu", lambda c: c["frac"].get("layer_cpu"))
        rng("jdRep", lambda c: max(c["rep_dev"].values()) if c["rep_dev"] else None, "{:.1f}")
        mo = [v for _, _, c in cells for v in c["meas_over_an"].values()]
        M["jdMeasOverAnMin"] = f"{min(mo):.2f}"; M["jdMeasOverAnMax"] = f"{max(mo):.2f}"
        M["jdMeasOverAnN"] = str(len(mo))
        M["jdLayerCostMax"] = f"{100 - 100 * min(c['frac']['layer'] for _, _, c in cells):.0f}"
        M["jdWaitCostMax"] = f"{100 * max(c['frac']['token'] - c['frac']['layer'] for _, _, c in cells):.0f}"
        wa = [100 * (c['frac']['token'] - c['frac']['layer']) for job, _, c in cells if job != "102a"]
        M["jdWaitCostAmdMax"] = f"{max(wa):.0f}"
        ov = max(((c["frac_analytic"] - c["frac"]["layer"]), c) for _, _, c in cells)
        M["jdModelOverMax"] = f"{100 * ov[0]:.0f}"; M["jdModelOverAn"] = f"{100 * ov[1]['frac_analytic']:.0f}"
        M["jdModelOverMeas"] = f"{100 * ov[1]['frac']['layer']:.0f}"
        M["jdLayerMoreTimeMax"] = f"{100 * max(1 / c['frac']['layer'] - 1 for _, _, c in cells):.0f}"
        M["jdLinkSlow"] = f"{100 * min(c['frac']['layer_link'] for _, _, c in cells):.0f}"
        best = [max(c["frac"].values()) for _, _, c in cells]
        M["jdBestMin"] = f"{100 * min(best):.0f}"; M["jdBestMax"] = f"{100 * max(best):.0f}"
    with open(P("paper", "tab_readsched.tex"), "w") as f:
        f.write("% generated by scripts/job102.py\n\\begin{table}[t]\\centering\\footnotesize\n")
        f.write("\\caption{Can the bound's read time be reached (job 102)? MIN's per-layer host reads of gpt-oss (first 4{,}000 steps of the "
                "AIME routing), replayed without the model through CPU threads and the copy engine. Each mode's share of the bound's host term "
                "($R^\\star S/B_{\\mathrm{host}}$, ms per token in the column \\emph{Term}); \\emph{Model}: the per-layer mode predicted from "
                "the probe's rates. Two repetitions of each mode agree within "
                + M.get("jdRepMax", "--") + "\\%.}\\label{tab:readsched}\n")
        f.write("\\setlength\\tabcolsep{3pt}\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{@{}lrrrrrrrr@{}}\\toprule\n")
        f.write(" & Link/ & & Term & \\multicolumn{2}{c}{Split} & Link & CPU & \\\\\n")
        f.write("Host & CPU & $C$ & (ms) & per layer & per token & only & only & Model \\\\\\midrule\n")
        for job, h in hosts.items():
            for C in sorted(h["cells"]):
                c = h["cells"][C]
                fr = c["frac"]
                f.write(f"{NAMES.get(job, job)} & {h['B_p'] / h['B_c']:.2f} & {C} & {c['t_bound_host_ms']:.2f} & "
                        + " & ".join(f"{100 * fr[k]:.0f}\\%" if k in fr else "--" for k in ("layer", "token", "layer_link", "layer_cpu"))
                        + f" & {100 * c['frac_analytic']:.0f}\\% \\\\\n")
        f.write("\\bottomrule\\end{tabular}}\\end{table}\n")
    with open(P("paper", "wsg_job102.tex"), "w") as f:
        f.write("% generated by scripts/job102.py from the job 102 hosts (results/102?_readsched@vast)\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    print(dict(st))
    for c in clauses:
        print(c["status"], c["id"], c["short"], c["measured"])
    for job, h in hosts.items():
        print(job, h["cpu"], f"Bc {h['B_c']:.1f} Bp {h['B_p']:.1f} Bcp {h['B_cp']:.1f} Bhost {h['B_host']:.1f} helpers {h['helpers']}")
        for C, c in h["cells"].items():
            print("  ", C, "R", round(c["reads_per_token"], 2), "bound ms", round(c["t_bound_host_ms"], 2), "analytic", round(c["frac_analytic"], 3),
                  {k: round(v, 3) for k, v in c["frac"].items()}, {k: [round(x, 2) for x in v] for k, v in c["ms"].items()},
                  "meas/analytic", {k: round(v, 3) for k, v in c["meas_over_an"].items()})
    print(M)


if __name__ == "__main__":
    main()

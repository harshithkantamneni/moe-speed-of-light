"""The accounting, measured: the states of the engine that the oracle jobs realised, per host and cell, in ms per token,
beside the limit and the model's own values for the states it prices (prereg/foresight_093.json, foresight_094.json,
foresight_095.json). Writes prereg/accounting_measured.json and paper/tab_accounting_measured.tex, and the macros
paper/wsg_accounting.tex (am*).

States (ms per token; the measured ones are means over the 30 sequences):
  online        the deployed policy (decayed frequency, the law's FETCH table)                    all three hosts
  no-overlap    the same with the engine's within-layer overlap off                               093
  all-CPU       the same with no FETCH table                                                      093 (host-bound cells)
  hit-opt.      Belady prefetch of the coming steps' experts without bypass (admits what MIN bypasses) 093, 094, 095
  bypass        MIN with bypass with background admissions (two reads)                            094
  fetch         MIN with bypass, admitted misses fetched into their slot (one read; serialised)    095  = the model's F
  both          fetch + the scheduled single-read prefetch (one read; overlapped)                  095  = the model's F+O
  limit         the speed limit on that host
Model (shapley_gap.value_function on that host's probe): v(none), v(F), v(O), v(O,F), v(all).

    python scripts/accounting_measured.py
"""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
JOBS = [("093", "093_foresight@vast", "O1 (9950X, job 093)", {"online": "base", "no-overlap": "noovl", "all-CPU": "allcpu", "hit-opt. paced": "oracle_w0"}),
        ("094", "094_foresight_bytes@vast", "O2 (7950X, job 094)", {"online": "base", "bypass": "bypass_w0", "hit-opt. paced": "prefetch_w0"}),
        ("095", "095_single_read@vast", "O3 (9950X, job 095)", {"online": "base", "fetch": "fetch", "lead": "lead2", "both": "both2", "both, paced": "both3p", "hit-opt.": "hitopt"}),
        ("096a", "096a_factorial_9950x@vast", "O4 (9950X, job 096)", {"online": "base", "foa": "foa", "fetch": "fetch", "bypass": "bypass", "lead": "lead2", "nb2": "nb2", "both": "both2", "both, paced": "both3p", "hit-opt.": "hitopt", "hit-opt. paced": "hitoptp"}),
        ("096b", "096b_factorial_9950x3d@vast", "O5 (9950X3D, job 096)", {"online": "base", "foa": "foa", "fetch": "fetch", "bypass": "bypass", "lead": "lead2", "nb2": "nb2", "both": "both2", "both, paced": "both3p", "hit-opt.": "hitopt", "hit-opt. paced": "hitoptp"})]
MACROS = {}


def M(k, v):
    MACROS[k] = v


def main():
    out = {"hosts": []}
    lines = []
    for job, jdir, hname, states in JOBS:
        path = P("prereg", f"foresight_{job}.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        host = {"job": job, "host": hname, "B": d["host"], "cells": []}
        for c in d["cells"]:
            mt = c["model_terms"]
            cell = {"cell": c["label"], "limit_ms": mt["limit_ms"], "model": {"none": mt["v_none_ms"], "F": mt["v_F_ms"], "O": mt["v_O_ms"], "OF": mt["v_OF_ms"], "all": mt["v_all_ms"]},
                    "measured": {}, "host_bound": mt["c_at_limit"] <= 0}
            for name, lab in states.items():
                if lab in c["runs"]:
                    r = c["runs"][lab]
                    cell["measured"][name] = {"ms": r["ms"], "ratio": r["ratio_to_base"][0], "ci": r["ratio_to_base"][1:], "reads": r["misses_per_token"] + r["admits_per_token"],
                                              "hit": r["hit_rate"], "frac": r["frac_of_limit"]}
            base_ms = cell["measured"]["online"]["ms"]
            gap = base_ms - mt["limit_ms"]
            cell["gap_ms"] = gap
            # the measured decomposition where the single-read oracles exist: bytes (online -> fetch), the overlap foresight allows
            # (fetch -> both), the rest (both -> limit); the model's: none -> F, F -> OF, OF -> limit
            if "fetch" in cell["measured"] and "both" in cell["measured"]:
                f_ms = cell["measured"]["fetch"]["ms"]
                b_ms = min(cell["measured"][k]["ms"] for k in ("both", "both, paced") if k in cell["measured"])   # the faster overlapped state
                h_ms = cell["measured"]["hit-opt."]["ms"] if "hit-opt." in cell["measured"] else None
                a_ms = cell["measured"]["foa"]["ms"] if "foa" in cell["measured"] else None
                cell["decomposition"] = {"measured": {"bytes": (base_ms - f_ms) / gap, "overlap_given_foresight": (f_ms - b_ms) / gap, "rest": (b_ms - mt["limit_ms"]) / gap,
                                                      "single_read": ((base_ms - a_ms) / gap) if a_ms is not None else None,
                                                      "foresight_given_single_read": ((a_ms - f_ms) / gap) if a_ms is not None else None},
                                         # the other feasible order: overlap first (online -> the Belady prefetch at about the online policy's bytes), then the bytes given overlap (-> the faster single-read prefetch state)
                                         "measured_overlap_first": ({"overlap": (base_ms - h_ms) / gap, "bytes_given_overlap": (h_ms - b_ms) / gap, "rest": (b_ms - mt["limit_ms"]) / gap} if h_ms is not None else None),
                                         "model": {"F": (mt["v_none_ms"] - mt["v_F_ms"]) / (mt["v_none_ms"] - mt["limit_ms"]),
                                                   "O_given_F": (mt["v_F_ms"] - mt["v_OF_ms"]) / (mt["v_none_ms"] - mt["limit_ms"]),
                                                   "rest": (mt["v_OF_ms"] - mt["limit_ms"]) / (mt["v_none_ms"] - mt["limit_ms"])},
                                         "model_interaction_ms": (mt["v_none_ms"] - mt["v_O_ms"]) + (mt["v_none_ms"] - mt["v_F_ms"]) - (mt["v_none_ms"] - mt["v_OF_ms"])}
            host["cells"].append(cell)
            def ms(name):
                return f"{cell['measured'][name]['ms']:.2f}" if name in cell["measured"] else "--"
            lines.append(f"{hname.split(' ')[0]} & {c['label'].replace('%', chr(92) + '%')} & {base_ms:.2f} & {ms('foa')} & {ms('no-overlap')} & {ms('all-CPU')} & {ms('bypass')} & {ms('hit-opt.')} & {ms('hit-opt. paced')} & {ms('nb2')} & {ms('fetch')} & {ms('both')} & {ms('both, paced')} & "
                         f"{mt['limit_ms']:.2f} & {mt['v_F_ms']:.2f} & {mt['v_OF_ms']:.2f} & {mt['v_all_ms']:.2f} \\\\")
        out["hosts"].append(host)
    # macros from the measured decomposition (095) at the host-bound cells
    dec = [c for h in out["hosts"] for c in h["cells"] if "decomposition" in c]
    if dec:
        hb = [c for c in dec if c["host_bound"]]
        M("amHostsN", str(sum(1 for h in out["hosts"] if any("decomposition" in c for c in h["cells"]))))
        for key, px in (("bytes", "amBytes"), ("overlap_given_foresight", "amOverlap"), ("rest", "amRest")):
            vals = [100 * c["decomposition"]["measured"][key] for c in hb]
            M(f"{px}HostMin", f"{min(vals):.0f}"); M(f"{px}HostMax", f"{max(vals):.0f}")
            vals = [100 * c["decomposition"]["measured"][key] for c in dec]
            M(f"{px}Min", f"{min(vals):.0f}"); M(f"{px}Max", f"{max(vals):.0f}")
        for key, px in (("F", "amModelF"), ("O_given_F", "amModelO"), ("rest", "amModelRest")):
            vals = [100 * c["decomposition"]["model"][key] for c in hb]
            M(f"{px}HostMin", f"{min(vals):.0f}"); M(f"{px}HostMax", f"{max(vals):.0f}")
        # the model's share over the measured share (bytes, overlap) and the measured over the model's (the rest), host-bound cells
        for mkey, key, px in (("F", "bytes", "amBytesMult"), ("O_given_F", "overlap_given_foresight", "amOverlapMult")):
            vals = [c["decomposition"]["model"][mkey] / c["decomposition"]["measured"][key] for c in hb]
            M(f"{px}Min", f"{min(vals):.1f}"); M(f"{px}Max", f"{max(vals):.1f}")
        vals = [c["decomposition"]["measured"]["rest"] / c["decomposition"]["model"]["rest"] for c in hb]
        M("amRestMultMin", f"{min(vals):.1f}"); M("amRestMultMax", f"{max(vals):.1f}")
        for h in out["hosts"]:
            nm = {"095": "Three", "096a": "Four", "096b": "Five"}.get(h["job"])
            hc = [c for c in h["cells"] if "decomposition" in c and c["host_bound"]]
            if nm and hc:
                v = [100 * c["decomposition"]["measured"]["rest"] for c in hc]
                M(f"amRest{nm}Min", f"{min(v):.0f}"); M(f"amRest{nm}Max", f"{max(v):.0f}")
                v = [100 * (c["decomposition"]["measured"]["bytes"] + c["decomposition"]["measured"]["overlap_given_foresight"]) for c in hc]
                M(f"amRecov{nm}Min", f"{min(v):.0f}"); M(f"amRecov{nm}Max", f"{max(v):.0f}")
        hb_sr = [c for c in hb if c["decomposition"]["measured"].get("single_read") is not None]
        if hb_sr:
            for key, px in (("single_read", "amSingle"), ("foresight_given_single_read", "amFsGivenSingle")):
                vals = [100 * c["decomposition"]["measured"][key] for c in hb_sr]
                M(f"{px}HostMin", f"{min(vals):.0f}"); M(f"{px}HostMax", f"{max(vals):.0f}")
        for c in [c for h in out["hosts"] if h["job"] == "095" for c in h["cells"] if "decomposition" in c]:
            tag = {"gpt-oss 11%": "GLow", "gpt-oss 25%": "GMid", "gpt-oss 40%": "GHigh", "Qwen3 12.5%": "QLow", "Qwen3 25%": "QMid", "Qwen3 43.75%": "QHigh"}[c["cell"]]
            for key, px in (("bytes", "amBytes"), ("overlap_given_foresight", "amOverlap"), ("rest", "amRest")):
                M(f"{px}{tag}", f"{100 * c['decomposition']['measured'][key]:.0f}")
            of = c["decomposition"].get("measured_overlap_first")
            if of:
                M(f"amOvlFirst{tag}", f"{100 * of['overlap']:.0f}"); M(f"amBytesGivenOvl{tag}", f"{100 * of['bytes_given_overlap']:.0f}")
            M(f"amModelF{tag}", f"{100 * c['decomposition']['model']['F']:.0f}"); M(f"amModelO{tag}", f"{100 * c['decomposition']['model']['O_given_F']:.0f}")
    json.dump(out, open(P("prereg", "accounting_measured.json"), "w"), indent=1)
    with open(P("paper", "wsg_accounting.tex"), "w") as f:
        f.write("% generated by scripts/accounting_measured.py\n")
        for k in sorted(MACROS):
            f.write("\\newcommand{\\%s}{%s}\n" % (k, MACROS[k]))
    with open(P("paper", "tab_accounting_measured.tex"), "w") as f:
        f.write("% generated by scripts/accounting_measured.py\n")
        f.write("\\begin{table*}[t]\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n")
        f.write("\\caption{The accounting, measured: milliseconds per token of the engine's states that the oracle jobs realised, on each oracle host, "
                "beside that host's limit and the model's values for the states it prices; O1--O3 ran jobs 093--095 in a fixed order, O4 and O5 job 096 in a shuffled order. \\emph{online}: the deployed policy; \\emph{foa}: the same with each admission fetched into its slot in the step that misses it (one read, no foresight); \\emph{no ovl.}: its "
                "within-layer overlap off; \\emph{all CPU}: no FETCH table; \\emph{bypass}: MIN with bypass with background admissions (two reads per "
                "admitted expert); \\emph{hit-opt.}: Belady prefetch without bypass, which admits what the optimum bypasses, unpaced, and the same on the paced copy path (the first \\emph{paced} column; O1 and O2 ran only that form); \\emph{nb2}: Belady within a two-step lead with each admission read once; \\emph{fetch}: MIN with bypass, admitted misses fetched into their "
                "slot (one read, serialised: the model's foresight-only state $v(F)$); \\emph{both}: fetch and the scheduled single-read prefetch "
                "(one read, overlapped: the model's $v(O,F)$); the second \\emph{paced}: both on the paced copy path with a three-step lead; \\emph{limit}: the limit on that host; $v(F)$, $v(O,F)$, $v(\\mathrm{all})$: the model.}\n")
        f.write("\\label{tab:accounting_measured}\n")
        f.write("\\begin{tabular}{@{}llrr|rrrrrr|rrr|r|rrr@{}}\\toprule\n")
        f.write("Host & Cell & online & foa & no ovl. & all CPU & bypass & hit-opt. & paced & nb2 & fetch & both & paced & limit & $v(F)$ & $v(O,F)$ & $v(\\mathrm{all})$ \\\\\\midrule\n")
        f.write("\n".join(lines) + "\n")
        f.write("\\bottomrule\\end{tabular}\\end{table*}\n")
    print("hosts:", [h["host"] for h in out["hosts"]], "macros:", len(MACROS))


if __name__ == "__main__":
    main()

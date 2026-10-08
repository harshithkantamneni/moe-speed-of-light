"""Round-to-round and rental-to-rental variation of the speed ratios, and the registered clauses rescored with it.

Within a machine the paper's intervals are over problems (a paired bootstrap), so they leave out the variation between
rounds (separate processes on one rental) and between rentals of one machine. Here:

  * round-to-round: for every valid machine of jobs 109, 111 and 112, budget and configuration run in two rounds, the
    difference between the rounds' speed ratios (log scale);
  * rental-to-rental: machines (GPU UUIDs) measured in two launches with the same configuration and protocol
    (job 099's panel and its relaunches in jobs 100-103; job 109 and its relaunches in job 112; job 110's first rounds
    and job 111);
  * rescoring: each machine's interval for each configuration is widened on the log scale by that configuration's full
    round-to-round range on that machine, on both sides, and every per-machine clause of jobs 109, 111 and 112 (and job
    112's RTX 4090s under job 110's predictions) that held with an interval is scored again. A clause fails only on its
    point estimate, so widening can move "held" to "held (point)" but cannot make a clause fail.

Writes paper/wsg_noise.tex.

    python scripts/noise.py
"""
import copy
import glob
import json
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scripts import job109 as J  # noqa: E402
from scripts import job112 as K  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = lambda *a: os.path.join(ROOT, *a)  # noqa: E731
RES = J.RES
CELLS = (14, 32)


def uuid_of(d):
    f = f"{RES}/{d}/nvidia-smi-q.txt" if not d.startswith("/") else f"{d}/nvidia-smi-q.txt"
    if os.path.exists(f):
        for ln in open(f):
            if "GPU UUID" in ln:
                return ln.split(":", 1)[1].strip()
    return None


def spreads(h):
    """{(C, config): full range of the rounds' log ratios} for one machine"""
    out = {}
    for C, c in h["cells"].items():
        for n, v in c.get("ratio_rounds", {}).items():
            if len(v) >= 2:
                lv = np.log(v)
                out[(C, n)] = float(lv.max() - lv.min())
    return out


def widen(h, extra=0.0):
    """widen the machine's within-machine intervals (in place) by each configuration's round-to-round range on that
    machine (zero where it ran one round, as at 25% in jobs 109-112), plus extra on every interval"""
    sp = spreads(h)
    for C, c in h["cells"].items():
        ci = c.get("ci")
        if not ci:
            continue
        r = c.get("ratio", {})
        w = lambda n: sp.get((C, n), 0.0) + extra  # noqa: E731
        new = {}
        for n, (lo, hi) in ci.items():
            if n in r:
                new[n] = (lo * math.exp(-w(n)), hi * math.exp(w(n)))
        # the best policy without foresight: its own configuration's range
        if "best" in ci and c.get("best_name"):
            d = w(c["best_name"])
            new["best"] = (ci["best"][0] * math.exp(-d), ci["best"][1] * math.exp(d))
        # capture = (best - 1) / (oracle - 1): the extremes over the widened best and oracle intervals
        if "capture" in ci and "best" in new and "both3p" in new:
            b0, b1 = new["best"]; o0, o1 = new["both3p"]
            if o0 > 1:
                v = [(b - 1) / (o - 1) for b in (b0, b1) for o in (o0, o1)]
                new["capture"] = (min(min(v), ci["capture"][0]), max(max(v), ci["capture"][1]))
        # job 112's differences of ratios: add the ranges, converted to ratio units
        def dd(n):
            return r.get(n, 1.0) * (math.exp(w(n)) - 1)
        if "lag_part" in ci:
            u = dd("bypassplanS") + dd("bypassplan")
            new["lag_part"] = (ci["lag_part"][0] - u, ci["lag_part"][1] + u)
        if "read_minus_lag" in ci:
            u = dd("fetchplan") + 2 * dd("bypassplanS") + dd("bypassplan")
            new["read_minus_lag"] = (ci["read_minus_lag"][0] - u, ci["read_minus_lag"][1] + u)
        ci.update(new)


def widen_uniform(h, d):
    """widen every ratio interval by its round-to-round range plus a fixed log amount d (the rental-to-rental case)"""
    widen(h, extra=d)


def rescore(cl_before, cl_after):
    b = {c["id"]: c["status"] for c in cl_before}
    a = {c["id"]: c["status"] for c in cl_after}
    # per-machine clauses only: the pooled ones (intervals over machines) are not widened
    with_ci = [c["id"] for c in cl_before if c["ci"] is not None and b[c["id"]] == "held" and c["host"] != "pooled"]
    still = [k for k in with_ci if a.get(k) == "held"]
    moved = [k for k in with_ci if a.get(k) != "held"]
    changed_fail = [k for k in b if b[k] != "failed" and a.get(k) == "failed"]
    return with_ci, still, moved, changed_fail


def table_of(d):
    f = f"{RES}/{d}/fetch_table_law_gptoss.json"
    return json.load(open(f)).get("table") if os.path.exists(f) else None


def rental_pairs():
    """log differences of the speed ratio of one configuration between two launches of one machine, with whether the
    two launches used the same fetch table (each launch derives its table from its own probe)"""
    pairs = []   # (launch pair label, budget, config, log difference, same fetch table, uuid)
    # job 099's panel and its relaunches (jobs 100, 101, 103)
    pan = {uuid_of(h["dir"]): h for h in json.load(open(P("prereg", "panel_099.json")))["hosts"]}
    for pj in ("job100.json", "job101.json", "job103.json"):
        f = P("prereg", pj)
        if not os.path.exists(f):
            continue
        for h in json.load(open(f))["hosts"]:
            u = uuid_of(h["dir"])
            if u not in pan or u is None:
                continue
            p = pan[u]
            for C, lab in (("14", "gpt-oss 11%"), ("32", "gpt-oss 25%")):
                a, b = p["cells"].get(lab, {}).get("speed", {}), h["cells"].get(C, {}).get("speed", {})
                for n in ("foa", "bypass", "fetch", "both3p"):
                    if f"{n}/base" in a and n in b:
                        pairs.append((f"{p['host']}/{h['job']}", int(C), n, math.log(b[n][0]) - math.log(a[f"{n}/base"][0]),
                                      table_of(p["dir"]) == table_of(h["dir"]), u))
    # job 109 and its relaunches in job 112; job 110's first rounds and job 111
    h109 = {h["uuid"]: h for h in (J.load(d, CELLS, 0.5) for d in sorted(glob.glob(J.GLOB)))}
    h112 = [K.load112(d) for d in sorted(glob.glob(K.GLOB))]
    for h in h112:
        if h["card"] != "5090" or not h["valid"] or h["uuid"] not in h109:
            continue
        a = h109[h["uuid"]]
        for C in CELLS:
            for n in ("foa", "bypass", "fetch"):
                ra, rb = a["cells"][C]["ratio"].get(n), h["cells"][C]["ratio"].get(n)
                if ra and rb:
                    pairs.append((f"{a['job']}/{h['job']}", C, n, math.log(rb) - math.log(ra),
                                  table_of(os.path.basename(a["dir"])) == table_of(os.path.basename(h["dir"])), h["uuid"]))
    h110 = {h["uuid"]: h for h in (J.load(d, CELLS, 0.25) for d in sorted(glob.glob(J.GLOB110)))}
    for h in (J.load(d, CELLS, 0.25) for d in sorted(glob.glob(J.GLOB111))):
        if not h["valid"] or h["uuid"] not in h110:
            continue
        a = h110[h["uuid"]]
        for C in CELLS:
            for n in ("foa", "bypass", "fetch", "both3p", "pf", "dk", "R1", "R2"):
                ra, rb = a["cells"][C]["ratio"].get(n), h["cells"][C]["ratio"].get(n)
                if ra and rb:
                    pairs.append((f"{a['job']}/{h['job']}", C, n, math.log(rb) - math.log(ra),
                                  table_of(os.path.basename(a["dir"])) == table_of(os.path.basename(h["dir"])), h["uuid"]))
    return pairs, h112


def main():
    M = {}
    pc = lambda x: f"{100 * x:.1f}"  # noqa: E731
    # the valid machines of the registered jobs
    V109 = [h for h in (J.load(d, CELLS, 0.5) for d in sorted(glob.glob(J.GLOB))) if h["valid"]]
    V111 = [h for h in (J.load(d, CELLS, 0.25) for d in sorted(glob.glob(J.GLOB111))) if h["valid"]]
    pairs, H112 = rental_pairs()
    V112 = [h for h in H112 if h["valid"]]
    V4 = [h for h in V112 if h["card"] == "4090"]; V5 = [h for h in V112 if h["card"] == "5090"]
    # round-to-round
    rr = [(h["job"], C, n, d) for h in V109 + V111 + V112 for (C, n), d in spreads(h).items()]
    dl = np.array([d for *_, d in rr])
    M["nzRoundN"] = str(len(rr)); M["nzRoundMachines"] = str(len({j for j, *_ in rr}))
    M["nzRoundMed"] = pc(np.median(dl)); M["nzRoundMax"] = pc(dl.max())
    M["nzRoundOverOne"] = str(int((dl > 0.01).sum())); M["nzRoundOverTwo"] = str(int((dl > 0.02).sum()))
    # how the round-to-round range compares with the half-width of the interval over problems
    hw = []
    for h in V109 + V111 + V112:
        sp = spreads(h)
        for C, c in h["cells"].items():
            for n, (lo, hi) in c.get("ci", {}).items():
                if (C, n) in sp and lo > 0 and hi > 0:
                    hw.append((math.log(hi) - math.log(lo)) / 2)
    hw = np.array(hw)
    M["nzHalfWidthMed"] = pc(np.median(hw))
    # rental-to-rental
    rl = np.array([abs(p[3]) for p in pairs])
    M["nzRentalN"] = str(len(pairs)); M["nzRentalPairs"] = str(len({p[0] for p in pairs}))
    M["nzRentalMachines"] = str(len({p[5] for p in pairs}))
    same = np.array([abs(p[3]) for p in pairs if p[4]])
    M["nzRentalSameN"] = str(len(same)); M["nzRentalSameMax"] = pc(same.max()); M["nzRentalSameMed"] = pc(np.median(same))
    M["nzRentalTableDiff"] = str(len({p[0] for p in pairs if not p[4]}))
    M["nzRentalMed"] = pc(np.median(rl)); M["nzRentalMax"] = pc(rl.max())
    M["nzRentalOverFive"] = str(int((rl > 0.05).sum()))
    worst = max(pairs, key=lambda p: abs(p[3]))
    print("pairs with different fetch tables:", sorted({p[0] for p in pairs if not p[4]}))
    print("worst rental pair", worst)
    # rescoring
    before = J.clauses109(V109) + J.clauses110(V111) + K.clauses112(V112, V4, V5) + \
        [c for c in J.clauses110(V4) if not c["id"].endswith("-valid")]
    W109, W111, W112 = (copy.deepcopy(x) for x in (V109, V111, V112))
    for h in W109 + W111 + W112:
        widen(h)
    W4 = [h for h in W112 if h["card"] == "4090"]; W5 = [h for h in W112 if h["card"] == "5090"]
    after = J.clauses109(W109) + J.clauses110(W111) + K.clauses112(W112, W4, W5) + \
        [c for c in J.clauses110(W4) if not c["id"].endswith("-valid")]
    with_ci, still, moved, changed_fail = rescore(before, after)
    assert not changed_fail, changed_fail
    M["nzHeldCI"] = str(len(with_ci)); M["nzStillHeld"] = str(len(still)); M["nzMoved"] = str(len(moved))
    M["nzPooled"] = str(sum(1 for c in before if c["ci"] is not None and c["status"] == "held" and c["host"] == "pooled"))
    M["nzMovedWord"] = ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"][len(moved)] if len(moved) < 10 else str(len(moved))
    print("moved to held (point):", moved)
    # which predictions the moved clauses belong to
    kinds = Counter(k.split("-")[1] for k in moved)
    M["nzMovedKinds"] = ", ".join(f"{k} ({v})" for k, v in sorted(kinds.items())) if kinds else "none"
    # the same with every interval widened by the largest rental-to-rental difference instead
    dmax = float(rl.max())
    U109, U111, U112 = (copy.deepcopy(x) for x in (V109, V111, V112))
    for h in U109 + U111 + U112:
        widen_uniform(h, dmax)
    U4 = [h for h in U112 if h["card"] == "4090"]; U5 = [h for h in U112 if h["card"] == "5090"]
    after_u = J.clauses109(U109) + J.clauses110(U111) + K.clauses112(U112, U4, U5) + \
        [c for c in J.clauses110(U4) if not c["id"].endswith("-valid")]
    _, still_u, moved_u, fail_u = rescore(before, after_u)
    assert not fail_u, fail_u
    M["nzRentalStillHeld"] = str(len(still_u)); M["nzRentalMoved"] = str(len(moved_u))
    # of those, the bands of +-0.06 or narrower around configurations that barely change the speed (P2, P10, Q1 foa/bypass)
    narrow = [k for k in moved_u if "-P2-" in k or "-P10-" in k or "-Q1-foa" in k or "-Q1-bypass" in k]
    M["nzRentalMovedNarrow"] = str(len(narrow)); M["nzRentalMovedOther"] = str(len(moved_u) - len(narrow))
    print("moved, not narrow:", [k for k in moved_u if k not in narrow])
    print("rental widening moves:", moved_u)
    with open(P("paper", "wsg_noise.tex"), "w") as f:
        f.write("% generated by scripts/noise.py\n")
        for k in sorted(M):
            f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
    json.dump(dict(round_to_round=rr, rental_pairs=[dict(pair=p[0], C=p[1], config=p[2], log_diff=p[3], same_table=p[4])
                                                    for p in pairs], moved=moved, still=still),
              open(P("prereg", "noise.json"), "w"), indent=1)
    for k in sorted(M):
        print(k, M[k])


if __name__ == "__main__":
    main()

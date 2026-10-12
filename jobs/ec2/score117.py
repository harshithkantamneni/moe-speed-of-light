"""Job 117's scorer (registered with the job): speeds, shares, validity, room, the decision rule and the predictions.

    python score117.py OUT        writes OUT/score117.json and prints a summary
"""
import glob
import json
import os
import sys

import numpy as np

NAMES = ("base", "aa", "fetch", "w4", "f0", "f1", "f2", "r7")
ORACLE = ("aa", "fetch", "w4", "f0", "f1", "f2", "r7")
SHARED = ("w4", "r7", "f0", "f1", "f2")
GATE_SPEED, ROOM_TOKENS = 1.08, 4
V3_TOL, V4_TOL, V5_TOL = 0.03, 0.01, 0.03


def rows(out):
    """{name: {round: {problem: (ms per token, loss)}}}"""
    by = {}
    for f in sorted(glob.glob(f"{out}/ec_q_r*.jsonl")):
        rd = int(os.path.basename(f)[len("ec_q_r"):].split(".")[0])
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            lab = os.path.basename(r["config"].rsplit("stats=", 1)[-1]).replace(".json", "").rsplit("_", 1)[-1]
            by.setdefault(lab, {}).setdefault(rd, {})[r["seq"]] = (r["decode_ms"] / r["n_decode"], r["nll_sum"] / r["nll_n"])
    return by


def per_problem(by, name):
    """ms per token per problem, averaged over the rounds that ran it"""
    d = {}
    for rd, v in by.get(name, {}).items():
        for p, (ms, _) in v.items():
            d.setdefault(p, []).append(ms)
    return {p: float(np.mean(v)) for p, v in d.items()}


def speed(by, name, nb=4000, seed=117):
    b, x = per_problem(by, "base"), per_problem(by, name)
    ps = sorted(set(b) & set(x))
    if not ps:
        return None
    B, X = np.array([b[p] for p in ps]), np.array([x[p] for p in ps])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ps), (nb, len(ps)))
    draws = B[idx].mean(axis=1) / X[idx].mean(axis=1)
    return dict(ratio=float(B.mean() / X.mean()), lo=float(np.percentile(draws, 2.5)), hi=float(np.percentile(draws, 97.5)),
                ms=float(X.mean()), problems=len(ps))


def stats(out, name):
    s = [json.load(open(f)) for f in sorted(glob.glob(f"{out}/st_q_r*_{name}.json"))]
    if not s:
        return None
    reads = [(x["misses"] + x["admits"] + x.get("prefetches", 0)) / max(1, x["steps"]) for x in s]
    return dict(reads_per_token=float(np.mean(reads)), rounds=len(s), policy=[x["policy"] for x in s],
                forecast_rows=[x.get("forecast_rows") for x in s], forecast_lag=[x.get("forecast_lag") for x in s],
                forecast_seq_mismatch=[x.get("forecast_seq_mismatch") for x in s],
                plan_us=[x.get("oracle_plan_us_per_step") for x in s])


def g_prof(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))["median_ms"]
    g = sum(v for k, v in d.items() if k.startswith("cat_") and k not in ("cat_ec_wait", "cat_ec_copy"))
    return dict(wall=d["wall"], G_prof=g, room=d["wall"] - g, wait=d.get("cat_ec_wait", 0.0), copy=d.get("cat_ec_copy", 0.0),
                busy=d.get("busy"))


def main(out):
    by = rows(out)
    res = dict(speed={}, engine={}, validity={}, predictions={})
    for n in NAMES:
        res["speed"][n] = speed(by, n)
        res["engine"][n] = stats(out, n)
    rp = json.load(open(f"{out}/replay117.json")) if os.path.exists(f"{out}/replay117.json") else {}
    res["replay"] = rp
    ms = {n: (res["speed"][n] or {}).get("ms") for n in NAMES}
    # shares of aa -> fetch, in time and in reads
    ts, rs = {}, {}
    if ms["aa"] and ms["fetch"]:
        for n in SHARED:
            if ms[n]:
                ts[n] = (ms["aa"] - ms[n]) / (ms["aa"] - ms["fetch"])
    eng = {n: (res["engine"][n] or {}).get("reads_per_token") for n in NAMES}
    if eng["aa"] and eng["fetch"]:
        for n in SHARED:
            if eng[n]:
                rs[n] = (eng["aa"] - eng[n]) / (eng["aa"] - eng["fetch"])
    res["time_share"], res["read_share"] = ts, rs
    # validity
    v = res["validity"]
    rr = rp.get("reads_per_token", {})
    v3 = {n: (eng[n] / rr[n] - 1) if eng.get(n) and rr.get(n) else None for n in ORACLE}
    v["V3_engine_vs_replay"] = v3
    v["V3"] = all(x is not None and abs(x) <= V3_TOL for x in v3.values())
    fc_ok = all(res["engine"][n] and all(r == rp.get("records") for r in res["engine"][n]["forecast_rows"]) and
                all(m == 0 for m in res["engine"][n]["forecast_seq_mismatch"]) for n in ("f0", "f1", "f2"))
    v["forecast_loaded"] = fc_ok and bool(rp.get("forecast_seq_equal"))
    loss = {}
    for n in NAMES:
        for rd, d in by.get(n, {}).items():
            bd = by.get("base", {}).get(rd, {})
            ps = sorted(set(d) & set(bd))
            if ps:
                loss[f"{n}_r{rd}"] = float(np.mean([d[p][1] for p in ps]) / np.mean([bd[p][1] for p in ps]) - 1)
    v["V4_loss_vs_base"] = loss
    v["V4"] = bool(loss) and all(abs(x) <= V4_TOL for x in loss.values())
    bms = [float(np.mean([x[0] for x in d.values()])) for rd, d in sorted(by.get("base", {}).items())]
    v["V5_base_ms_rounds"] = bms
    v["V5"] = len(bms) == 2 and max(bms) / min(bms) - 1 <= V5_TOL
    sc = json.load(open(f"{out}/score.json")) if os.path.exists(f"{out}/score.json") else {}
    v["V1_trace_overlap"] = sc.get("V1_trace_overlap")
    v["V1"] = (sc.get("V1_trace_overlap") or 0) >= 0.90
    v["V2_greedy_match"] = sc.get("V2_greedy_match")
    v["V2"] = (sc.get("V2_greedy_match") or 0) >= 0.90
    res["forecast_quality"] = dict(precision={h: q["precision_at_k"] for h, q in sc.get("quality_self", {}).items() if h.isdigit()},
                                   token_accuracy=sc.get("self_token_accuracy"), shadow_shares_hf_routing=sc.get("C16", {}).get("shares"))
    # room
    tf = []
    if os.path.exists(f"{out}/tfull.jsonl"):
        tf = [json.loads(l) for l in open(f"{out}/tfull.jsonl") if l.strip()]
    t_full = float(np.mean([r["decode_ms"] / r["n_decode"] for r in tf])) if tf else None
    pf, pb = g_prof(f"{out}/prof_F1.json"), g_prof(f"{out}/prof_BASE.json")
    res["room"] = dict(T_full_ms=t_full, f1=pf, base=pb, shadow_tokens=ROOM_TOKENS,
                       need_ms=ROOM_TOKENS * t_full if t_full else None)
    # the decision rule
    s1 = (res["speed"]["f1"] or {}).get("ratio")
    a = s1 is not None and s1 >= GATE_SPEED
    b = None if not (pf and t_full) else pf["room"] >= ROOM_TOKENS * t_full
    res["gate"] = dict(a_speed_f1=s1, a=a, b_room_ms=pf["room"] if pf else None, b_need_ms=res["room"]["need_ms"], b=b,
                       decision="go" if (a and b) else ("go only with a cheaper shadow" if (a and b is False) else
                                                        ("no-go" if not a else "undecided: room not measured")))
    # predictions
    sp = {n: (res["speed"][n] or {}).get("ratio") for n in NAMES}
    P = res["predictions"]
    P["P1"] = None if None in (sp["f0"], sp["f1"], sp["f2"]) else sp["f0"] > sp["f1"] > sp["f2"]
    rng = dict(f0=(1.00, 1.15), f1=(0.97, 1.10), f2=(0.95, 1.06), w4=(1.08, 1.35), r7=(1.00, 1.15))
    P["P2"] = {n: (None if sp[n] is None else lo <= sp[n] <= hi) for n, (lo, hi) in rng.items()}
    P["P3"] = {n: (abs(ts[n] - rs[n]) <= 0.15 if n in ts and n in rs else None) for n in SHARED}
    P["P4"] = None if b is None else (not b)
    P["P5"] = res["gate"]["decision"] != "go"
    json.dump(res, open(f"{out}/score117.json", "w"), indent=1)
    for n in NAMES:
        s, e = res["speed"][n], res["engine"][n]
        if s:
            print(f"{n:6s} {s['ms']:7.2f} ms/token  vs base {s['ratio']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]  reads/token "
                  f"{e['reads_per_token'] if e else float('nan'):.2f} (replay {rr.get(n, float('nan')):.2f})  "
                  f"time share {ts.get(n, float('nan')):.3f} read share {rs.get(n, float('nan')):.3f}")
    print("room:", json.dumps(res["room"]))
    print("validity:", {k: v[k] for k in ("V1", "V2", "V3", "V4", "V5", "forecast_loaded")})
    print("gate:", json.dumps(res["gate"]))
    print("predictions:", json.dumps(P))
    return res


if __name__ == "__main__":
    main(sys.argv[1])

"""Online admission rules for the deployed path (serve each miss on the CPU, copy admissions in the background: every
admission is a second read). The fewest-admission lesson made online: admit a miss only when its predicted reuse beats
the victim's by a margin. Two rules, each with one scalar chosen on other text (the model's mixed-domain trace, no AIME):

  dfa-kappa   the deployed decayed-frequency score with its admission margin kappa (deployed: kappa = 1)
  learned-m   the learned reuse predictor of job 097 (scripts/learned_policy6.py, the engine's 8-bit cross-layer form),
              admitting when p(miss) > p(victim) + m

Both are replayed with the engine's semantics for background admissions (mosl.ecsim_fast: copies land two steps
later, a miss already on its way is not re-admitted, residents requested this step are never victims) and counted as
misses + admissions per token. Then on the AIME routing the engine runs (test): the deployed rule, both tuned rules and
MIN with bypass.

    python scripts/online_admit.py --out prereg/online_admit.json [--export-dir DIR]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl import ecsim_fast  # noqa: E402
from mosl.traces import load_pack  # noqa: E402
from scripts.provenance import MODELS, find, per_seq  # noqa: E402
from scripts.learned_policy5 import features_r, NF  # noqa: E402
from scripts.learned_policy6 import train, export, AIME, RES  # noqa: E402
from scripts.foresight import _pol, INF  # noqa: E402

CELLS = {"gpt-oss-120b": (14, 32), "qwen3-30b-a3b": (16, 32)}
KAPPAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
MARGINS = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
DEPLOYED_KAPPA = 1.0   # the engine's deployed setting for both models (jobs 093-105: kappa=1)


@njit(cache=True)
def sim_learned_bg(R, E, C, P, margin, delay):
    """the deployed path's background admissions (ecsim_fast._layer's semantics) with the learned order: misses
    sorted by predicted reuse, victim = lowest predicted reuse among residents not loading and not requested this
    step, admitted when p(miss) > p(victim) + margin. Returns misses and admissions (totals)."""
    T, k = R.shape
    slot_of = np.full(E, -1, np.int64)
    resident = np.full(C, -1, np.int64)
    loading = np.full(C, -1, np.int64)
    batch = np.zeros(C, np.int64)
    mis = 0
    adm = 0
    missed = np.empty(k, np.int64)
    pm = np.empty(k)
    for t in range(T):
        p = P[t]
        for s in range(C):
            if loading[s] >= 0 and t - batch[s] >= delay:
                e = loading[s]
                loading[s] = -1
                resident[s] = e
                slot_of[e] = s
        nm = 0
        for j in range(k):
            e = R[t, j]
            if slot_of[e] < 0:
                missed[nm] = e
                pm[nm] = -p[e]
                nm += 1
        mis += nm
        order = np.argsort(pm[:nm])
        for qi in range(nm):
            e = missed[order[qi]]
            busy = False
            for s2 in range(C):
                if loading[s2] == e:
                    busy = True
                    break
            if busy:
                continue
            s = -1
            for s2 in range(C):
                if resident[s2] < 0 and loading[s2] < 0:
                    s = s2
                    break
            if s < 0:
                victim = -1
                best = 0.0
                for s2 in range(C):
                    c = resident[s2]
                    if c < 0 or loading[s2] >= 0:
                        continue
                    insel = False
                    for j in range(k):
                        if R[t, j] == c:
                            insel = True
                            break
                    if insel:
                        continue
                    if victim < 0 or p[c] < best:
                        best = p[c]
                        victim = s2
                if victim < 0:
                    continue
                if not (p[e] > best + margin):
                    continue
                v = resident[victim]
                resident[victim] = -1
                slot_of[v] = -1
                s = victim
            loading[s] = e
            batch[s] = t
            adm += 1
    return mis, adm


def probs(R3, l, tr, t0, t1):
    F = features_r(R3, l, tr["E"], tr["Ms"][l], tr["PX"], tr["pops"][l], t0, t1)
    return tr["model"].predict_proba(F.reshape(-1, NF))[:, 1].reshape(t1 - t0, tr["E"]).astype(np.float32)


def evaluate(R3, E, C, tr, kappas, margins, t0=0):
    """per-token misses and admissions of dfa at each kappa and learned at each margin, over steps t0.."""
    L, T, k = R3.shape
    n = T - t0
    out = {f"dfa@{kp:g}": [0, 0] for kp in kappas} | {f"learned@{m:g}": [0, 0] for m in margins}
    opt = 0
    for l in range(L):
        R = np.ascontiguousarray(R3[l])
        for kp in kappas:
            _, m_, a_ = ecsim_fast.simulate_layer(R, E, C, "dfa", half_life=16.0, kappa=kp)
            out[f"dfa@{kp:g}"][0] += int(m_[t0:].sum()); out[f"dfa@{kp:g}"][1] += int(a_[t0:].sum())
        if tr is not None:
            P = probs(R3, l, tr, 0, T)
            for m in margins:
                mm, aa = sim_learned_bg(np.ascontiguousarray(R[t0:]), E, C, np.ascontiguousarray(P[t0:]), float(m), 2)
                out[f"learned@{m:g}"][0] += mm; out[f"learned@{m:g}"][1] += aa
        cc, pp = _pol(R, E, C, INF, 16.0, 0.0)
        opt += cc + pp
    res = {kk: dict(misses=v[0] / n, admits=v[1] / n, reads=(v[0] + v[1]) / n) for kk, v in out.items() if tr is not None or kk.startswith("dfa")}
    res["min"] = dict(reads=opt / T)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gpt-oss-120b,qwen3-30b-a3b")
    ap.add_argument("--tune-tokens", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--export-dir")
    a = ap.parse_args()
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for key in a.models.split(","):
        ck, _, _ = MODELS[key]
        pk = load_pack(find(RES, f"{ck}_S.npz"))
        win = per_seq(pk)
        idx = np.concatenate([np.arange(x, y) for _, x, y in win])[: a.tune_tokens]
        Rtr = np.ascontiguousarray(pk["routes"][:, idx], dtype=np.int64)
        E = int(pk["E"])
        z = np.load(AIME[key])
        Rte = np.ascontiguousarray(z["act"].astype(np.int64).transpose(1, 0, 2))
        for C in CELLS[key]:
            t0 = time.time()
            tr = train(key, C, cross=2)
            tune = evaluate(Rtr, E, C, tr, KAPPAS, MARGINS)
            kbest = min(KAPPAS, key=lambda kp: tune[f"dfa@{kp:g}"]["reads"])
            mbest = min(MARGINS, key=lambda m: tune[f"learned@{m:g}"]["reads"])
            test = evaluate(Rte, E, C, tr, sorted({DEPLOYED_KAPPA, kbest}), sorted({0.0, mbest}))
            dep = test[f"dfa@{DEPLOYED_KAPPA:g}"]
            row = dict(C=C, tune_tokens=int(Rtr.shape[1]), kappa_best=kbest, margin_best=mbest, tune=tune, test=test,
                       deployed=dep, dfa_tuned=test[f"dfa@{kbest:g}"], learned_tuned=test[f"learned@{mbest:g}"],
                       learned_m0=test["learned@0"], min_reads=test["min"]["reads"])
            for nm in ("dfa_tuned", "learned_tuned", "learned_m0"):
                row[nm + "_reads_rel"] = row[nm]["reads"] / dep["reads"]
                row[nm + "_closed"] = (dep["reads"] - row[nm]["reads"]) / (dep["reads"] - row["min_reads"])
            out[f"{key} C{C}"] = row
            json.dump(out, open(a.out, "w"), indent=1)
            print(f"{key} C{C} ({time.time() - t0:.0f} s): tuned kappa {kbest:g}, margin {mbest:g} | AIME reads/token: deployed "
                  f"{dep['reads']:.2f} (m {dep['misses']:.2f} a {dep['admits']:.2f}), dfa@{kbest:g} {row['dfa_tuned']['reads']:.2f} "
                  f"(m {row['dfa_tuned']['misses']:.2f} a {row['dfa_tuned']['admits']:.2f}), learned@{mbest:g} {row['learned_tuned']['reads']:.2f} "
                  f"(m {row['learned_tuned']['misses']:.2f} a {row['learned_tuned']['admits']:.2f}), learned@0 {row['learned_m0']['reads']:.2f}, "
                  f"MIN {row['min_reads']:.2f}", flush=True)
            if a.export_dir:
                os.makedirs(a.export_dir, exist_ok=True)
                export(os.path.join(a.export_dir, f"learned_{key}_C{C}_q8.bin"), tr)


if __name__ == "__main__":
    main()

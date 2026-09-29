"""Prefetch from router lookahead, simulated on recorded decode steps (job 063, ec-bench --lookahead).

For every layer, the deployed cache (per-layer DFA: decayed frequency, half-life 16 steps, admit iff score > victim
score + kappa, background admission published `delay` steps later; misses run on the CPU helpers) is simulated as in
mosl.ecsim_fast, plus a prefetch: before layer l runs at step t, up to q of the experts that layer l's router would
pick from the residual after layer l-1's attention (the `mid1` prediction recorded at layer l-1; ranks 0..npred-1,
in rank order) that are not resident are copied into the lowest-score slot (a free slot first; never a slot holding
an expert predicted for this step) and are resident for this step; one whose background admission is in flight is
copied at once instead. The copy is assumed to finish before layer l's expert GEMVs; the time model below charges
its DRAM traffic. Layer 0 has no prediction (no layer before it).

Output per budget C: misses per layer-step (CPU-executed), prefetches and useful prefetches per layer-step, and a
decode-time estimate from the measured home-PC constants:
    T(token) = T0 + (misses + w * prefetches) * s / B     (DRAM-bound misses; w = 0: prefetch hidden, w = 1: charged)
with T0 and s/B fitted on job 059's three cache runs (tok/s against measured misses per token).

    python scripts/sim_prefetch.py --la /home/claude/gpu-branch/results/063_lookahead@vast/la_gpt-oss-120b.bin
"""
import argparse
import json
import os
import sys

import numpy as np
from numba import njit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl import ecsim_fast  # noqa: E402


def load(path):
    h = json.load(open(path + ".json"))
    L, k = h["n_layer"], h["k"]
    per = 2 * k + 24
    a = np.fromfile(path, dtype=np.int16).reshape(-1, 2 + L * per)
    seq, step = a[:, 0].astype(int), a[:, 1].astype(int)
    b = a[:, 2:].reshape(-1, L, per).astype(np.int64)
    return dict(h=h, L=L, k=k, E=h["n_expert"], seq=seq, step=step, act=b[..., :k], self=b[..., k:2 * k],
                mid1=b[..., 2 * k:2 * k + 8], out1=b[..., 2 * k + 8:2 * k + 16], mid2=b[..., 2 * k + 16:])


@njit(cache=True)
def _layer_pf(R, P, E, C, half_life, kappa, delay, q, npred, gate):
    """ecsim_fast._layer (policy dfa) plus prefetch of up to q predicted experts per step (P[t, :npred], rank order).
    gate: prefetch e only if its decayed score >= gate * (victim score) (0 = always)."""
    T, k = R.shape
    decay = 0.5 ** (1.0 / half_life)
    slot_of = np.full(E, -1, np.int64)
    resident = np.full(C, -1, np.int64)
    loading = np.full(C, -1, np.int64)
    batch = np.zeros(C, np.int64)
    score = np.zeros(E)
    last = np.zeros(E, np.int64)
    hits = np.zeros(T, np.int32)
    misses = np.zeros(T, np.int32)
    admits = np.zeros(T, np.int32)
    pref = np.zeros(T, np.int32)
    useful = np.zeros(T, np.int32)
    missed = np.empty(k, np.int64)
    pf = np.empty(8, np.int64)
    for t in range(T):
        for s in range(C):
            if loading[s] >= 0 and t - batch[s] >= delay:
                e = loading[s]
                loading[s] = -1
                resident[s] = e
                slot_of[e] = s
        # prefetch before the layer runs (its selection is not known yet)
        npf = 0
        for i in range(npred):
            if npf >= q:
                break
            e = P[t, i]
            if e < 0 or slot_of[e] >= 0:
                continue
            busy = -1
            for s2 in range(C):
                if loading[s2] == e:
                    busy = s2
                    break
            if busy >= 0:   # a background admission in flight: the prefetch copies it now instead
                loading[busy] = -1
                resident[busy] = e
                slot_of[e] = busy
                pf[npf] = e
                npf += 1
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
                    # do not evict an expert predicted for this step (this includes this step's prefetches)
                    was = False
                    for j in range(npred):
                        if P[t, j] == c:
                            was = True
                            break
                    if was:
                        continue
                    key = score[c] * decay ** (t - last[c])
                    if victim < 0 or key < best:
                        best = key
                        victim = s2
                if victim < 0:
                    continue
                if gate > 0 and score[e] * decay ** (t - last[e]) < gate * best:
                    continue
                v = resident[victim]
                resident[victim] = -1
                slot_of[v] = -1
                s = victim
            resident[s] = e
            slot_of[e] = s
            pf[npf] = e
            npf += 1
        pref[t] = npf
        nm = 0
        for j in range(k):
            e = R[t, j]
            score[e] = score[e] * decay ** (t - last[e]) + 1.0
            last[e] = t
            if slot_of[e] >= 0:
                hits[t] += 1
                for i in range(npf):
                    if pf[i] == e:
                        useful[t] += 1
            else:
                misses[t] += 1
                missed[nm] = e
                nm += 1
        for i in range(nm):
            e = missed[i]
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
                    key = score[c] * decay ** (t - last[c])
                    if victim < 0 or key < best:
                        best = key
                        victim = s2
                if victim < 0:
                    continue
                if not (score[e] > best + kappa):
                    continue
                v = resident[victim]
                resident[victim] = -1
                slot_of[v] = -1
                s = victim
            loading[s] = e
            batch[s] = t
            admits[t] += 1
    return hits, misses, admits, pref, useful


def streams(d):
    """Per layer: the actual routes [T, k] and the prediction for it [T, 8] (mid1 recorded at the layer before);
    steps in recorded order (sequences back to back, as the deployed cache sees them)."""
    L = d["L"]
    R = [np.ascontiguousarray(d["act"][:, l]) for l in range(L)]
    P = [np.full((len(d["act"]), 8), -1, np.int64)] + [np.ascontiguousarray(d["mid1"][:, l - 1]) for l in range(1, L)]
    return R, P


def run(d, C, q, npred, kappa=1.0, delay=2, gate=0.0, pred="mid1"):
    L, E = d["L"], d["E"]
    R, P = streams(d)
    if pred == "out1":
        P = [P[0]] + [np.ascontiguousarray(d["out1"][:, l - 1]) for l in range(1, L)]
    if pred == "oracle":   # the true selection, as a ceiling
        P = [np.concatenate([r, np.full((len(r), 8 - r.shape[1]), -1)], 1) for r in R]
    tot = np.zeros(5)
    for l in range(L):
        h, m, a, p, u = _layer_pf(R[l], P[l], E, C, 16.0, kappa, delay, q, npred, gate)
        tot += [h.sum(), m.sum(), a.sum(), p.sum(), u.sum()]
    n = len(d["act"]) * L
    return dict(C=C, q=q, npred=npred, pred=pred, gate=gate, hit=tot[0] / (tot[0] + tot[1]), miss_ls=tot[1] / n,
                admit_ls=tot[2] / n, pref_ls=tot[3] / n, useful_ls=tot[4] / n)


def recall(d, key, n, shift=1):
    act, pr = d["act"], d[key]
    T, L, k = act.shape
    hit = 0
    for l in range(L - shift):
        for t in range(T):
            hit += len(set(act[t, l + shift]) & set(pr[t, l, :n]))
    return hit / (T * (L - shift) * k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--la", required=True)
    ap.add_argument("--budgets", default="14,32,56")
    ap.add_argument("--out")
    a = ap.parse_args()
    d = load(a.la)
    L, k = d["L"], d["k"]
    same = np.mean([set(d["act"][t, l]) == set(d["self"][t, l]) for t in range(len(d["act"])) for l in range(L)])
    print(f"{a.la}: {len(d['act'])} steps x {L} layers, top-{k} of {d['E']}; host self-check {same:.4f}")
    print(f"recall of layer l+1's experts: mid(l) top-{k} {recall(d, 'mid1', k):.3f} top-8 {recall(d, 'mid1', 8):.3f} | "
          f"out(l) top-{k} {recall(d, 'out1', k):.3f} top-8 {recall(d, 'out1', 8):.3f} | l+2 from mid(l) top-{k} "
          f"{recall(d, 'mid2', k, 2):.3f}")
    rows = []
    for C in [int(x) for x in a.budgets.split(",")]:
        base = run(d, C, 0, 0)
        rows.append(base)
        print(f"C={C}: base hit {base['hit']:.3f} misses/layer-step {base['miss_ls']:.3f}")
        for pred in ("mid1", "out1", "oracle"):
            for q in (1, 2, 3):
                for npred in ((k, 8) if pred != "oracle" else (k,)):
                    for gate in ((0.0, 0.5, 1.0) if pred == "mid1" else (0.0,)):
                        r = run(d, C, q, npred, gate=gate, pred=pred)
                        rows.append(r)
                        print(f"   {pred:6s} q={q} from top-{npred} gate {gate:.1f}: misses {r['miss_ls']:.3f} "
                              f"({100 * (r['miss_ls'] / base['miss_ls'] - 1):+.0f}%), prefetches {r['pref_ls']:.3f} "
                              f"(useful {r['useful_ls']:.3f}), admissions {r['admit_ls']:.3f}")
    if a.out:
        json.dump(dict(la=a.la, rows=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

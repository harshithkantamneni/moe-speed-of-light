"""Hit-rate bounds per layer on the HF routing traces (decode windows as ec-bench runs them):
Belady (demand admission with bypass, free and instant copies) and the best fixed set in hindsight,
next to the cache's DFA (with its 2-step publication delay) and LRU."""
import sys, json
import numpy as np
sys.path.insert(0, ".")
from mosl.ecsim import load_steps, simulate

def belady_layer(R, E, C):
    T, k = R.shape
    # next use of expert e after step t
    nxt = np.full((T + 1, E), T + 10**6, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        nxt[t] = nxt[t + 1]
        nxt[t, R[t]] = t
    res = set(); hits = 0
    for t in range(T):
        sel = list(R[t])
        for e in sel:
            if e in res:
                hits += 1
        for e in sel:
            if e in res:
                continue
            if len(res) < C:
                res.add(e); continue
            # evict the resident with the farthest next use (after this step), unless e's is farther
            cand = [c for c in res if c not in sel]
            if not cand:
                continue
            far = max(cand, key=lambda c: nxt[t + 1, c])
            if nxt[t + 1, e] < nxt[t + 1, far]:
                res.remove(far); res.add(e)
    return hits / (T * k)

def fixed_best(R, E, C):
    cnt = np.bincount(R.ravel(), minlength=E)
    return np.sort(cnt)[::-1][:C].sum() / R.size

if __name__ == "__main__":
    MODELS = {"gpt-oss-20b": ("data/traces/gpt-oss-20b", "data/tok_gpt-oss-20b.jsonl", [4, 8, 16]),
              "qwen3-30b-a3b": ("data/traces/qwen3-30b-a3b", "data/tok_qwen3_30b.jsonl", [16, 32, 64])}
    SEQ = [4, 8, 11, 14]
    out = {}
    for m, (td, corp, Cs) in MODELS.items():
        R, E, _ = load_steps(td, corp, SEQ, 128, 128)
        for C in Cs:
            bel = np.mean([belady_layer(r, E, C) for r in R])
            fix = np.mean([fixed_best(r, E, C) for r in R])
            h, mi, a = simulate(R, E, C, "dfa", kappa=1.0 if "gpt" in m else 2.0)
            dfa = h.sum() / (h.sum() + mi.sum())
            h2, mi2, _ = simulate(R, E, C, "lru")
            lru = h2.sum() / (h2.sum() + mi2.sum())
            print(f"{m:14s} C={C:3d}: fixed-hindsight {fix:.3f}  LRU {lru:.3f}  DFA+k {dfa:.3f} ({a.sum()/len(R[0]):.1f} adm/step)  Belady {bel:.3f}")
            out[f"{m}/C{C}"] = dict(fixed=fix, lru=lru, dfa=dfa, belady=bel)
    json.dump(out, open("results/oracle_hits.json", "w"), indent=1)

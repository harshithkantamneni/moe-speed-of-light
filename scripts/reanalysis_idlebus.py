"""Re-analysis: how much host-read time could a layer-ahead prefetch hide under the GPU's own compute? The engine
serialises the GPU's non-expert work (~3.3 ms per token) with the host reads; during it the host bus is idle. Layer
l's output predicts layer l+1's experts (job 063 records, field out1). Each layer's window of the GPU's compute can
prefetch W x B bytes of the predicted, non-resident experts; a correct prefetch saves its read time.

    python scripts/reanalysis_idlebus.py
"""
import numpy as np, json
S = 13253760
la = '/home/claude/gpu-branch/results/063_lookahead@vast/la_gpt-oss-120b.bin'
meta = json.load(open(la + '.json')); L, k = meta['n_layer'], meta['k']
raw = np.fromfile(la, np.int16); st = 2 + L * (2 * k + 24); rec = raw[: len(raw) // st * st].reshape(-1, st)
blk = lambda l: rec[:, 2 + l * (2 * k + 24): 2 + (l + 1) * (2 * k + 24)]
act = np.stack([blk(l)[:, :k] for l in range(L)], 1).astype(int)
out1 = np.stack([blk(l)[:, 2 * k + 8: 2 * k + 16] for l in range(L)], 1).astype(int)   # predicts layer l+1 from layer l's output
mid1 = np.stack([blk(l)[:, 2 * k: 2 * k + 8] for l in range(L)], 1).astype(int)
seq = rec[:, 0]
T = act.shape[0]; E = 128
for nm, pr in (('out1', out1), ('mid1', mid1)):
    rk = np.mean([(act[:, l + 1, :, None] == pr[:, l, None, :k]).any(-1).mean() for l in range(L - 1)])
    r8 = np.mean([(act[:, l + 1, :, None] == pr[:, l, None, :]).any(-1).mean() for l in range(L - 1)])
    print(f"{nm}: recall of layer l+1's experts from layer l: top-{k} {rk:.2f}, top-8 {r8:.2f}")
def deployed_misses(a, C, hl=16.0, theta=1.0):
    """the deployed policy (decayed count, LFU victim, kappa=theta), per step: the set of missed experts and the
    residency before the step"""
    decay = 0.5 ** (1 / hl); score = np.zeros(E); last = np.zeros(E, int); inc = np.zeros(E, bool); size = 0
    misses, resid = [], []
    for t in range(a.shape[0]):
        resid.append(inc.copy())
        req = set(a[t].tolist())
        for e in a[t]:
            score[e] = score[e] * decay ** (t - last[e]) + 1; last[e] = t
        ms = [e for e in a[t] if not inc[e]]; misses.append(ms)
        for e in ms:
            if size < C:
                inc[e] = True; size += 1; continue
            cand = [c for c in np.nonzero(inc)[0] if c not in req]
            if not cand: continue
            v = min(cand, key=lambda c: score[c] * decay ** (t - last[c]))
            if score[e] > score[v] * decay ** (t - last[v]) + theta:
                inc[v] = False; inc[e] = True
    return misses, resid
W_nonexp = 3.3e-3 / L    # the GPU's non-expert compute per layer (profiles: 4.3 ms per token minus resident experts and control)
for C in (14, 32):
    per_layer = [deployed_misses(act[:, l, :], C) for l in range(L)]
    R = np.mean([sum(len(per_layer[l][0][t]) for l in range(L)) for t in range(T)])
    for B in (30e9, 50e9, 90e9):
        cap = W_nonexp * B
        saved = np.zeros(T)
        for l in range(L - 1):
            ms_, rs_ = per_layer[l + 1]
            for t in range(T):
                miss = set(ms_[t]); resid = rs_[t]; budget = cap; sb = 0.0
                for c in out1[t, l]:
                    if budget <= 0: break
                    if resid[c]: continue
                    take = min(S, budget); budget -= take
                    if c in miss: sb += take
                saved[t] += sb / B
        Tdep = 4.3e-3 + R * S / B
        print(f"C={C} B={B/1e9:.0f} GB/s: deployed reads (misses) {R:.1f}/token, T ~ {1e3*Tdep:.1f} ms; idle-window prefetch hides {1e3*saved.mean():.2f} ms "
              f"(window {1e3*W_nonexp*L:.1f} ms) -> {100*saved.mean()/Tdep:.1f}% of the time")

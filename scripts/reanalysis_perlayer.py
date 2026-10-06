"""Re-analysis: would a non-uniform number of slots per layer (allocated by marginal reads on half the problems) read
less than uniform slots on the other half? Deployed policy and MIN, gpt-oss and Qwen3 AIME traces.

    python scripts/reanalysis_perlayer.py
"""
import sys, numpy as np
sys.path.insert(0, '/home/claude/moe-speed-of-light')
from scipy import stats
from mosl.cachesim import simulate, simulate_rstar
def curves(act, seqmask, E, Cs, pol):
    T, L, k = act.shape
    out = np.zeros((L, len(Cs)))
    a = act[seqmask]
    for l in range(L):
        for j, C in enumerate(Cs):
            if pol == 'min':
                m, ad = simulate(a[:, l, :], E, C, 'min', bypass=True); out[l, j] = m.mean()
            elif pol == 'dep':
                m, ad = simulate_rstar(a[:, l, :], E, C, 16.0, 1.0, True); out[l, j] = (m + ad).mean()   # two reads per admission
            elif pol == 'dep1':
                m, ad = simulate_rstar(a[:, l, :], E, C, 16.0, 1.0, True); out[l, j] = m.mean()          # one read per miss
    return out
def allocate(cur, Cs, total, cmin):
    # greedy marginal allocation (curves are close to convex): start at cmin per layer
    L = cur.shape[0]; idx = {C: j for j, C in enumerate(Cs)}
    alloc = np.full(L, cmin)
    left = total - cmin * L
    while left > 0:
        best, bl = -1e9, -1
        for l in range(L):
            if alloc[l] + 1 > Cs[-1]: continue
            g = cur[l, idx[alloc[l]]] - cur[l, idx[alloc[l] + 1]]
            if g > best: best, bl = g, l
        alloc[bl] += 1; left -= 1
    return alloc
for name, path in (('gpt-oss', '/home/claude/gpu-branch/results/084c_gptoss_trace@vast/route_aime25_gptoss.npz'),
                   ('Qwen3', '/home/claude/gpu-branch/results/084b_vram_rerun@vast/route_aime25_qwen3.npz')):
    z = np.load(path); act = z['act'].astype(np.int64); seq = z['seq']; E = int(z['n_expert'])
    T, L, k = act.shape
    us = list(dict.fromkeys(seq.tolist()))
    tr = np.isin(seq, us[0::2]); te = np.isin(seq, us[1::2])   # alternate problems: train / test
    budgets = {'gpt-oss': (14, 32), 'Qwen3': (16, 32)}[name]
    for Cu in budgets:
        Cs = list(range(max(2, Cu // 4), min(E, Cu * 3) + 1))
        for pol in ('dep', 'min'):
            ctr = curves(act, tr, E, Cs, pol); cte = curves(act, te, E, Cs, pol)
            al = allocate(ctr, Cs, Cu * L, Cs[0])
            j = {C: i for i, C in enumerate(Cs)}
            uni = sum(cte[l, j[Cu]] for l in range(L)); non = sum(cte[l, j[al[l]]] for l in range(L))
            unitr = sum(ctr[l, j[Cu]] for l in range(L)); nontr = sum(ctr[l, j[al[l]]] for l in range(L))
            per_layer = np.array([cte[l, j[Cu]] for l in range(L)])
            print(f"{name} C={Cu} {pol:4s}: reads/token uniform {uni:6.2f} -> per-layer {non:6.2f} on held-out problems ({100*(1-non/uni):.1f}% fewer; in-sample {100*(1-nontr/unitr):.1f}%); "
                  f"slots per layer {al.min()}-{al.max()}; per-layer reads at uniform C: {per_layer.min():.2f}-{per_layer.max():.2f}, spearman with layer index {stats.spearmanr(np.arange(L), per_layer).statistic:+.2f}")
            if pol == 'dep':
                print('      allocation:', ' '.join(map(str, al)))

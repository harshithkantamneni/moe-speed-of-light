import sys, numpy as np
sys.path.insert(0, ".")
from pilot_forecast_sim import pol_fc, opt_reads, load_pack
def run(model, pack, kap):
    p = load_pack(pack); routes = p["routes"]; L, Ttot, k = routes.shape; E = p["E"]; st = p["starts"]
    resp = np.concatenate([np.arange(s + pl, s + n) for s, pl, n in zip(st, p["prompt_lens"], p["seq_lens"])])
    for C in (E // 8, E // 4, E // 2):
        acc = {}
        for l in range(L):
            R = np.ascontiguousarray(routes[l, resp]); T = len(R)
            empty = np.full((T, 1, k), -1, np.int64)
            for kk in (0.0, kap, -1e9):
                c, cp = pol_fc(R, empty, E, C, 1, 16.0, kk); acc[f"W0k{kk:g}"] = acc.get(f"W0k{kk:g}", 0) + c + cp
            for W in (1, 4):
                Fp = np.ascontiguousarray(np.broadcast_to(R[:, None, :], (T, W, k)))
                c, cp = pol_fc(R, Fp, E, C, W, 16.0, kap); acc[f"persistW{W}"] = acc.get(f"persistW{W}", 0) + c + cp
                idx = np.minimum(np.arange(T)[:, None] + 1 + np.arange(W)[None, :], T - 1)
                Fe = np.ascontiguousarray(np.where((np.arange(T)[:, None] + 1 + np.arange(W)[None, :] < T)[:, :, None], R[idx], -1))
                c, cp = pol_fc(R, Fe, E, C, W, 16.0, kap); acc[f"exactW{W}"] = acc.get(f"exactW{W}", 0) + c + cp
            c, cp = opt_reads(R, E, C); acc["opt"] = acc.get("opt", 0) + c + cp
        acc = {a: round(v / len(resp), 2) for a, v in acc.items()}
        best0 = min(v for a, v in acc.items() if a.startswith("W0"))
        gap = best0 - acc["opt"]
        print(model, "C", C, acc, "| closed vs best no-foresight:", {a: round((best0 - v) / gap, 3) for a, v in acc.items() if a.startswith(("persist", "exact"))}, flush=True)
run("gpt-oss-120b", "/home/claude/gpu-branch/results/035_trace_rest@40gb/gpt-oss-120b_tok.npz", 1.0)
run("qwen3-30b-a3b", "/home/claude/gpu-branch/results/035_trace_rest@40gb/qwen3-30b-a3b_tok.npz", 2.0)

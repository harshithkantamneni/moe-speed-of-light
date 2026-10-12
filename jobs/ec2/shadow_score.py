"""Job 116's scorer: how good are the shadow run-ahead forecasts, and what are they worth in host reads.

From shadow.npz (shadow_runahead.py): per horizon h = 1..H, precision@k (share of the true top-k of step g+h among the
forecast's first k) and recall@m (among its first m = k+3), over every step and layer with a forecast; and the share
of MIN's gain over admitting every miss that the window policy closes when driven by the forecast's first k per step
(vmap.replay_fc, a copy of scripts/value_map.py: the paper's yardstick of Section 8, kappa = -1e9), for windows
W = 1, 2, 4 at the job's budget C, with the exact window, random-error windows (fill, precision = recall = r) and
persistence (the current step's experts) on the same routing for reference. Also the validity checks: the routing's
agreement with job 084b's llama.cpp trace (V1) and the greedy match (V2).

    python shadow_score.py OUTDIR [TRACE.npz]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vmap as V  # noqa: E402

GO_SHARE, GO_PREC = 1 / 3, 0.5


def overlap(R, F, h, seq, k, width):
    """mean |F[g, :, h-1, :width] ∩ R[g+h]| / k over valid (g, layer)"""
    T, L, _ = R.shape
    g = np.arange(T - h)
    g = g[(seq[g + h] == seq[g]) & (F[g, 0, h - 1, 0] != 255)]
    if len(g) == 0:
        return None, 0
    E = int(max(R.max(), F[F != 255].max() if (F != 255).any() else 0)) + 1
    truth = np.zeros((len(g), L, E), bool)
    np.put_along_axis(truth, R[g + h].astype(np.int64), True, axis=2)
    fc = F[g, :, h - 1, :width].astype(np.int64)
    hit = np.take_along_axis(truth, np.where(fc == 255, 0, fc), axis=2) & (fc != 255)
    # a forecast list holds distinct experts, so hits count each true expert once
    return float(hit.sum(axis=2).mean() / k), len(g)


def as_fc(F, width):
    Fx = F[:, :, :, :width].astype(np.int64)
    Fx[Fx == 255] = -1
    return np.ascontiguousarray(Fx.transpose(1, 0, 2, 3))     # [L, T, H, width]


def main(out, trace=None):
    z = np.load(os.path.join(out, "shadow.npz"))
    R, seq = z["R"].astype(np.int64), z["seq"].astype(np.int64)
    k, H, E, C = int(z["k"]), int(z["H"]), int(z["E"]), int(z["C"])
    T, L, _ = R.shape
    res = dict(steps=T, problems=int(seq.max()) + 1, C=C, k=k, H=H, E=E, L=L)
    # V2: the model's greedy choice at step g is the next input of its own text
    res["V2_greedy_match"] = float(z["match"].mean())
    # V1: agreement with the llama.cpp trace of job 084b (same ids, same weights)
    if trace and os.path.exists(trace):
        tz = np.load(trace)
        act, tseq = tz["act"].astype(np.int64), tz["seq"].astype(np.int64)
        n = int((tseq < res["problems"]).sum())
        if n == T:
            a = np.zeros((T, L, int(tz["n_expert"])), bool)
            np.put_along_axis(a, act[:T], True, axis=2)
            res["V1_trace_overlap"] = float(np.take_along_axis(a, R, axis=2).sum(axis=2).mean() / k)
            # the same one step shifted either way (a diagnosis of an off-by-one alignment, not a check)
            for d, nm in ((1, "next"), (-1, "prev")):
                g = np.arange(max(0, -d), T - max(0, d))
                g = g[tseq[g + d] == tseq[g]]
                res[f"V1_overlap_shift_{nm}"] = float(np.take_along_axis(a[g + d], R[g], axis=2).sum(axis=2).mean() / k)
        else:
            res["V1_trace_overlap"] = None
            res["V1_note"] = f"trace has {n} steps for these problems, the run {T}"
    # forecast quality
    for name in ("self", "orac"):
        F = z[f"F_{name}"]
        m = F.shape[3]
        q = {}
        for h in range(1, H + 1):
            p, n = overlap(R, F, h, seq, k, k)
            r, _ = overlap(R, F, h, seq, k, m)
            q[str(h)] = dict(precision_at_k=p, recall_at_m=r, n=n)
        q["precision_mean"] = float(np.mean([q[str(h)]["precision_at_k"] for h in range(1, H + 1)]))
        res[f"quality_{name}"] = q
    # the shadow's own tokens: its input at h is right when it equals the real input of step g+h
    tok, inp = z["tok_self"], z["inp"]
    acc = {}
    for h in range(1, H + 1):
        g = np.arange(T - h)
        g = g[(seq[g + h] == seq[g]) & (tok[g, h - 1] >= 0)]
        acc[str(h)] = float((tok[g, h - 1] == inp[g + h]).mean()) if len(g) else None
    res["self_token_accuracy"] = acc
    # host reads: shares of MIN's gain over admitting every miss
    for Cb in sorted({C, 2 * C}):
        online = V.replay(R, seq, E, Cb, -1, True, kappa=-1e9)[0]
        opt = V.replay(R, seq, E, Cb, 0, False)[0]
        gap = online - opt
        cell = dict(online=online, min=opt, shares={})
        sh = cell["shares"]
        for W in (1, 2, 4):
            sh[f"exact_W{W}"] = (online - V.replay(R, seq, E, Cb, W, True, kappa=-1e9)[0]) / gap
            for name in ("self", "orac"):
                Fk = as_fc(z[f"F_{name}"], k)
                sh[f"{name}_W{W}"] = (online - V.replay_fc(R, seq, E, Cb, Fk, W)) / gap
        for r in (0.5, 0.7, 0.9):
            v = [V.replay(R, seq, E, Cb, 4, True, r, True, s, kappa=-1e9)[0] for s in (1, 2, 3)]
            sh[f"random_r{r}_W4"] = (online - float(np.mean(v))) / gap
        Fp = np.full((T, L, H, k), 255, np.uint8)
        for h in range(1, H + 1):
            g = np.arange(T - h)
            g = g[seq[g + h] == seq[g]]
            Fp[g, :, h - 1, :] = R[g]
        sh["persistence_W4"] = (online - V.replay_fc(R, seq, E, Cb, as_fc(Fp, k), 4)) / gap
        res[f"C{Cb}"] = cell
    p = res[f"C{C}"]["shares"]
    res["go"] = dict(share_self_W4=p["self_W4"], precision_self_mean=res["quality_self"]["precision_mean"],
                     bar_share=GO_SHARE, bar_precision=GO_PREC,
                     decision="go" if (p["self_W4"] >= GO_SHARE and res["quality_self"]["precision_mean"] >= GO_PREC) else "no-go")
    json.dump(res, open(os.path.join(out, "score.json"), "w"), indent=1)
    print(json.dumps(res["go"]), flush=True)
    for name in ("self", "orac"):
        print(name, {h: (round(v["precision_at_k"], 3), round(v["recall_at_m"], 3)) for h, v in res[f"quality_{name}"].items()
                     if h.isdigit()}, flush=True)
    for Cb in sorted({C, 2 * C}):
        print(f"C{Cb}", {kk: round(v, 3) for kk, v in res[f"C{Cb}"]["shares"].items()}, flush=True)
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

"""Phase 4, sections 4.1-4.3 (prereg/PROTOCOL.md): does trace provenance change conclusions?

For each model and arm (D: dataset response teacher-forced; G: own greedy response; S: own sampled response),
over the response tokens of every conversation, replayed back to back:
  m1  NLL of the response tokens (vLLM prompt/sample logprobs)
  m2  temporal reuse: mean over layers and steps of |S_t & S_t-1| / k
  m3  regime ratio: distinct experts per layer in 16-step windows / C, at C = 25 % of E
  m4  hit rates at 12.5 / 25 / 50 %: LRU, LFU (decayed), DFA (as deployed), all with the cache's 2-step
      publication delay; and MIN-bypass (oracle, instant)
  m5  speed-of-light and DFA-ideal tok/s on the A10 constants (gpt-oss-20b, Qwen3-30B-A3B Q4_K_M)
95 % CIs: paired bootstrap over conversations (10 000 resamples, seed 0) of G-D and S-D. Then F1-F4 and P1-P4.

    python scripts/provenance.py --results /home/claude/gpu-branch/results --out prereg/provenance
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mosl import cachesim, ecsim_fast  # noqa: E402
from mosl.traces import load_pack  # noqa: E402
from scripts.sol_a10 import B_C, B_G, B_P, bound  # noqa: E402

BUDGETS = (1, 2, 4)                    # eighths of E
POLICIES = ("LRU", "LFU", "DFA", "MIN-bypass")
MODELS = {  # key -> (corpus key in the gen job, kappa, A10 constants: (expert bytes, dense bytes, all-GPU tag) or None)
    "olmoe": ("olmoe", 2.0, None),
    "gpt-oss-20b": ("gpt-oss-20b", 1.0, (13253760, 687012096, "gpt-oss-20b-MXFP4")),
    "qwen3-30b-a3b": ("qwen3-30b-a3b_fp8", 2.0, (2800000, 567271424, "Qwen3-30B-A3B-Instruct-2507-Q4_K_M")),
    "gpt-oss-120b": ("gpt-oss-120b", 1.0, None),
    # extension to the audit's models (not in the phase-4 registration; reported separately, DFA kappa = k/4 as above)
    "mixtral-8x7b": ("mixtral-8x7b", 0.5, None),
    "deepseek-v2-lite": ("deepseek-v2-lite", 1.5, None),
    "qwen1.5-moe": ("qwen1.5-moe", 1.0, None),
    "qwen2-57b": ("qwen2-57b", 2.0, None),
    "phi3.5-moe": ("phi3.5-moe", 0.5, None),
}
REGISTERED = ("olmoe", "gpt-oss-20b", "qwen3-30b-a3b", "gpt-oss-120b")
NBOOT = 10000


def find(results, name):
    hits = sorted(glob.glob(os.path.join(results, "*", name)))
    return hits[-1] if hits else None


def region_nll(score_path, packs_meta, arm):
    """per-conversation (sum, count) of response-token NLL from a vLLM score file"""
    out = {}
    if not score_path:
        return out
    pl = dict(zip(packs_meta["corpus_idx"].tolist(), packs_meta["prompt_lens"].tolist()))
    for line in open(score_path):
        r = json.loads(line)
        ci = r["corpus_idx"]
        if ci not in pl:
            continue
        v = [x for x in r["nll"][pl[ci] - 1:] if x is not None]
        out[ci] = (float(np.sum(v)), len(v))
    return out


def per_seq(pack):
    """response windows: list of (corpus_idx, start, stop) in the pack's concatenated token axis"""
    w = []
    for ci, s0, L, P in zip(pack["corpus_idx"], pack["starts"], pack["seq_lens"], pack["prompt_lens"]):
        if L - P >= 2:
            w.append((int(ci), int(s0 + P), int(s0 + L)))
    return w


def arm_stats(pack, kappa, E, k):
    """per-conversation statistics of one arm"""
    win = per_seq(pack)
    idx = np.concatenate([np.arange(a, b) for _, a, b in win])
    seg = np.concatenate([np.full(b - a, i) for i, (_, a, b) in enumerate(win)])
    first = np.concatenate([[True], seg[1:] != seg[:-1]])
    R = pack["routes"][:, idx]                                   # [L, T, k]
    Lyr, T, _ = R.shape
    n = len(win)
    st = {"corpus_idx": [c for c, _, _ in win], "steps": np.bincount(seg, minlength=n)}
    # m2 temporal reuse (within a conversation)
    same = (R[:, 1:, :, None] == R[:, :-1, None, :]).any(-1).sum(-1) / k    # [L, T-1]
    ok = ~first[1:]
    reuse = same.mean(0)
    st["reuse_sum"] = np.bincount(seg[1:][ok], weights=reuse[ok], minlength=n)
    st["reuse_n"] = np.bincount(seg[1:][ok], minlength=n)
    # m3 regime ratio at 25 %: distinct experts per layer in 16-step windows (within a conversation)
    C25 = E * 2 // 8
    rr_sum, rr_n = np.zeros(n), np.zeros(n)
    for i, (_, a, b) in enumerate(win):
        for w0 in range(a, b - 15, 16):
            blk = pack["routes"][:, w0:w0 + 16]
            d = np.mean([len(np.unique(blk[l])) for l in range(Lyr)])
            rr_sum[i] += d / C25
            rr_n[i] += 1
    st["rr_sum"], st["rr_n"] = rr_sum, rr_n
    # m4 hit rates, per step, attributed to conversations
    for q in BUDGETS:
        C = E * q // 8
        for pol in POLICIES:
            hits = np.zeros(T)
            miss_all, adm_all = np.zeros((T, Lyr), np.int8), np.zeros((T, Lyr), np.int8)
            for l in range(Lyr):
                if pol == "MIN-bypass":
                    m, adm = cachesim.simulate(R[l], E, C, "min", bypass=True)
                    h = k - m
                else:
                    h, m, adm = ecsim_fast.simulate_layer(R[l], E, C, {"LRU": "lru", "LFU": "lfu", "DFA": "dfa"}[pol],
                                                          kappa=kappa if pol == "DFA" else 0.0)
                hits += h
                miss_all[:, l], adm_all[:, l] = m, adm
            st[f"hit_{pol}_{q}"] = np.bincount(seg, weights=hits, minlength=n)
            if pol == "DFA":
                st[f"miss_{pol}_{q}"] = miss_all            # [T, L] for the DFA-ideal time
                st[f"adm_{pol}_{q}"] = adm_all
            if pol == "MIN-bypass":
                st[f"mstar_{q}"] = float(miss_all.mean())   # MIN-bypass misses per layer-step, for the bound
    st["den"] = st["steps"] * Lyr * k
    st["L"], st["k"], st["E"], st["seg"] = Lyr, k, E, seg
    return st


def sol_tokps(st, q, consts, eta):
    s, D, _ = consts
    k, L = st["k"], st["L"]
    a, b, p = s / (eta * B_G), s / B_C, s / B_P
    TD = D / (eta * B_G)
    Mstar = st[f"mstar_{q}"]                                          # MIN-bypass misses per layer-step
    t_sol = bound(TD, L, k, a, b, p, Mstar)
    m, adm = st[f"miss_DFA_{q}"].astype(float), st[f"adm_DFA_{q}"].astype(float)
    lay = np.maximum((k - m) * a, m * b)
    t_dfa = max(TD + float(lay.sum(1).mean()), float(adm.sum(1).mean()) * p)
    return 1 / t_sol, 1 / t_dfa


def boot_diff(fa, fb, ids_a, ids_b, rng, nboot=NBOOT):
    """paired bootstrap of f(arm b) - f(arm a) over shared conversations; f takes an index array into its arm"""
    shared = sorted(set(ids_a) & set(ids_b))
    ia = np.array([ids_a.index(c) for c in shared])
    ib = np.array([ids_b.index(c) for c in shared])
    point = fb(ib) - fa(ia)
    bs = np.empty(nboot)
    for i in range(nboot):
        r = rng.integers(0, len(shared), len(shared))
        bs[i] = fb(ib[r]) - fa(ia[r])
    return float(point), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(shared)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allgpu", default="prereg/a10_windows/scored_D.json")
    ap.add_argument("--models", default=",".join(REGISTERED))
    a = ap.parse_args()
    allgpu = json.load(open(a.allgpu))["results"] if os.path.exists(a.allgpu) else {}
    report, lines = {}, []
    for key in a.models.split(","):
        ck, kappa, consts = MODELS[key]
        arms = {}
        for arm in ("D", "G", "S"):
            p = find(a.results, f"{ck}_{arm}.npz")
            if p:
                arms[arm] = load_pack(p)
        if "D" not in arms or len(arms) < 2:
            print(f"{key}: packs {sorted(arms)} - skipped")
            continue
        E = arms["D"]["E"]
        k = arms["D"]["routes"].shape[2]
        eta = None
        if consts and consts[2] in allgpu and "all_gpu" in allgpu[consts[2]]:
            Lr = arms["D"]["routes"].shape[0]
            eta = (consts[1] + Lr * k * consts[0]) * allgpu[consts[2]]["all_gpu"]["tok_s"] / B_G
        S = {arm: arm_stats(pk, kappa, E, k) for arm, pk in arms.items()}
        nll = {arm: region_nll(find(a.results, f"score_{ck}_{arm}.jsonl"), arms[arm], arm) for arm in arms}
        rep = report[key] = {"arms": {}, "diffs": {}, "E": E, "k": k, "kappa": kappa, "eta_g": eta}
        for arm, st in S.items():
            ids = st["corpus_idx"]
            r = rep["arms"][arm] = {"n_conv": len(ids), "decode_steps": int(st["steps"].sum())}
            nl = [nll[arm][c] for c in ids if c in nll[arm]]
            r["nll"] = sum(x for x, _ in nl) / max(1, sum(n for _, n in nl)) if nl else None
            r["reuse"] = float(st["reuse_sum"].sum() / st["reuse_n"].sum())
            r["regime_ratio_25"] = float(st["rr_sum"].sum() / max(1, st["rr_n"].sum()))
            for q in BUDGETS:
                for pol in POLICIES:
                    r[f"hit_{pol}_{q}"] = float(st[f"hit_{pol}_{q}"].sum() / st["den"].sum())
                if eta:
                    r[f"sol_{q}"], r[f"dfa_ideal_{q}"] = sol_tokps(st, q, consts, eta)
        rng = np.random.default_rng(0)
        mets = {"reuse": lambda st: (lambda ix: st["reuse_sum"][ix].sum() / st["reuse_n"][ix].sum()),
                "regime_ratio_25": lambda st: (lambda ix: st["rr_sum"][ix].sum() / max(1, st["rr_n"][ix].sum()))}
        for q in BUDGETS:
            for pol in POLICIES:
                mets[f"hit_{pol}_{q}"] = (lambda q, pol: lambda st: (lambda ix: st[f"hit_{pol}_{q}"][ix].sum() / st["den"][ix].sum()))(q, pol)
            for p1, p2 in (("DFA", "LRU"), ("DFA", "LFU"), ("LFU", "LRU")):
                mets[f"gap_{p1}-{p2}_{q}"] = (lambda q, p1, p2: lambda st: (lambda ix: (st[f"hit_{p1}_{q}"][ix].sum() - st[f"hit_{p2}_{q}"][ix].sum()) / st["den"][ix].sum()))(q, p1, p2)
        def nll_f(arm):
            ids = S[arm]["corpus_idx"]
            v = np.array([nll[arm].get(c, (np.nan, 0))[0] for c in ids]); n = np.array([nll[arm].get(c, (0, 0))[1] for c in ids])
            return lambda ix: np.nansum(v[ix]) / max(1, n[ix].sum())
        for arm in ("G", "S"):
            if arm not in S:
                continue
            d = rep["diffs"][f"{arm}-D"] = {}
            for m, f in mets.items():
                d[m] = boot_diff(f(S["D"]), f(S[arm]), S["D"]["corpus_idx"], S[arm]["corpus_idx"], rng)
            if nll["D"] and nll[arm]:
                d["nll"] = boot_diff(nll_f("D"), nll_f(arm), S["D"]["corpus_idx"], S[arm]["corpus_idx"], rng)
            # within-arm policy gaps (for F1): CI of hit(p1) - hit(p2) in each arm separately
            # (D restricted to the conversations this arm also has, so both arms cover the same prompts)
            shared = set(S["D"]["corpus_idx"]) & set(S[arm]["corpus_idx"])
            for arm2, label in (("D", f"D|{arm}"), (arm, arm)):
                g = rep.setdefault("gaps", {}).setdefault(label, {})
                ids = S[arm2]["corpus_idx"]
                sel = np.array([i for i, c in enumerate(ids) if c in shared])
                for m in [m for m in mets if m.startswith("gap_")]:
                    if m in g:
                        continue
                    f = mets[m](S[arm2])
                    bs = [f(sel[rng.integers(0, len(sel), len(sel))]) for _ in range(2000)]
                    g[m] = (float(f(sel)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
        # F1-F3 and P1-P4 (S vs D); F4 needs the A10 S-arm run
        F = rep["flags"] = {}
        if "S" in S:
            dS = rep["diffs"]["S-D"]
            flips = []
            for q in BUDGETS:
                for p1, p2 in (("DFA", "LRU"), ("DFA", "LFU"), ("LFU", "LRU")):
                    gd, gs = rep["gaps"]["D|S"][f"gap_{p1}-{p2}_{q}"], rep["gaps"]["S"][f"gap_{p1}-{p2}_{q}"]
                    sig = lambda g: 1 if g[1] > 0 else (-1 if g[2] < 0 else 0)
                    if sig(gd) * sig(gs) == -1:
                        flips.append(f"{p1} vs {p2} at {q}/8")
            F["F1"] = flips
            h = dS["hit_DFA_2"]
            F["F2"] = bool(abs(h[0]) >= 0.05 and (h[1] > 0 or h[2] < 0))
            if eta:
                F["F3"] = {q: rep["arms"]["S"][f"sol_{q}"] / rep["arms"]["D"][f"sol_{q}"] - 1 for q in BUDGETS}
                F["F3_fires"] = bool(any(abs(v) >= 0.05 for v in F["F3"].values()))
            P = rep["predictions"] = {}
            if "nll" in dS:
                P["P1_S"] = dS["nll"][0] < 0
            if "G-D" in rep["diffs"]:
                if "nll" in rep["diffs"]["G-D"]:
                    P["P1_G"] = rep["diffs"]["G-D"]["nll"][0] < 0
                P["P2"] = rep["diffs"]["G-D"]["reuse"][0] > 0
            if key.startswith("gpt-oss"):
                P["P3"] = abs(dS["hit_DFA_2"][0]) >= 0.02
            P["P4"] = all(rep["gaps"]["S"][f"gap_DFA-LRU_{q}"][0] > 0 for q in BUDGETS)
        print(key, json.dumps({arm: {k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in r.items()} for arm, r in rep["arms"].items()})[:1500])
        print(key, "flags", rep.get("flags"), "predictions", rep.get("predictions"))
    os.makedirs(a.out, exist_ok=True)
    json.dump(report, open(os.path.join(a.out, "provenance.json"), "w"), indent=1, default=float)
    write_md(report, os.path.join(a.out, "provenance.md"))


def write_md(report, path):
    L = ["# Phase 4: trace provenance (D dataset text, G own greedy, S own sampled; response tokens only)\n"]
    for key, rep in report.items():
        L.append(f"## {key} (E = {rep['E']}, k = {rep['k']}, DFA kappa = {rep['kappa']})\n")
        arms = list(rep["arms"])
        L.append("| metric | " + " | ".join(arms) + " | " + " | ".join(f"{x} (95% CI)" for x in rep["diffs"]) + " |")
        L.append("|---" * (1 + len(arms) + len(rep["diffs"])) + "|")
        mets = ["nll", "reuse", "regime_ratio_25"] + [f"hit_{p}_{q}" for q in BUDGETS for p in POLICIES] + \
               [f"sol_{q}" for q in BUDGETS] + [f"dfa_ideal_{q}" for q in BUDGETS]
        for m in mets:
            vals = [rep["arms"][x].get(m) for x in arms]
            if all(v is None for v in vals):
                continue
            cell = lambda v: "–" if v is None else (f"{v:.1f}" if m.startswith(("sol", "dfa_ideal")) else f"{v:.3f}")
            dcell = []
            for dname, d in rep["diffs"].items():
                t = d.get(m)
                dcell.append("–" if t is None else f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]")
            L.append(f"| {m} | " + " | ".join(cell(v) for v in vals) + " | " + " | ".join(dcell) + " |")
        L.append(f"\nConversations and decode steps: " + ", ".join(f"{x}: {rep['arms'][x]['n_conv']} / {rep['arms'][x]['decode_steps']}" for x in arms))
        if rep.get("flags"):
            L.append(f"\nFlags (S vs D): F1 order reversals {rep['flags'].get('F1')}; F2 {rep['flags'].get('F2')}; "
                     f"F3 {rep['flags'].get('F3')} (fires: {rep['flags'].get('F3_fires')}).")
        if rep.get("predictions"):
            L.append(f"Predictions: {rep['predictions']}\n")
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

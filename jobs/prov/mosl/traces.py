"""Load collected routing traces as decode-only streams."""
import json
import os

import numpy as np


class Trace:
    """Decode-step routing for one model.

    R[l] : int64 [T, k] experts selected at decode step t in MoE layer l
    dom  : [T] domain label of each decode step
    Only assistant-response tokens are decode steps; prompt tokens are prefill.
    Conversations are concatenated in corpus order (one user, one session),
    so cache state carries across conversations as it would on a desktop.
    """

    def __init__(self, trace_dir, tok_file):
        meta = json.load(open(os.path.join(trace_dir, "meta.json")))
        self.meta = meta
        self.repo = meta["repo"]
        rows = [json.loads(l) for l in open(tok_file)]
        lens = np.load(os.path.join(trace_dir, "seq_lens.npy"))
        assert [len(r["ids"]) for r in rows] == lens.tolist(), "trace and tokenized corpus disagree"
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        keep = np.concatenate([np.arange(s + r["prompt_len"], s + len(r["ids"])) for s, r in zip(starts, rows)])
        self.dom = np.concatenate([[r["domain"]] * (len(r["ids"]) - r["prompt_len"]) for r in rows])
        self.conv = np.concatenate([[i] * (len(r["ids"]) - r["prompt_len"]) for i, r in enumerate(rows)])
        self.layers = sorted(int(k) for k in meta["layers"])
        self.E = int(meta["layers"][str(self.layers[0])]["num_experts"])
        self.k = int(meta["layers"][str(self.layers[0])]["top_k"])
        self.R = [np.load(os.path.join(trace_dir, f"layer{l:03d}.npy")).astype(np.int64)[keep] for l in self.layers]
        self.T = len(keep)

    @property
    def L(self):
        return len(self.layers)

    def domains(self):
        return sorted(set(self.dom.tolist()))


def locality_stats(tr: Trace, windows=(1, 16, 64)):
    """Per-layer-averaged routing locality statistics."""
    out = {}
    E, k = tr.E, tr.k
    reuse = {w: [] for w in windows}
    top10, top25, entropy_eff = [], [], []
    for R in tr.R:
        T = len(R)
        last = np.full(E, -10**9)
        hits = {w: 0 for w in windows}
        for t in range(T):
            for e in R[t]:
                gap = t - last[e]
                for w in windows:
                    if gap <= w:
                        hits[w] += 1
            last[R[t]] = t
        for w in windows:
            reuse[w].append(hits[w] / (T * k))
        cnt = np.bincount(R.ravel(), minlength=E).astype(float)
        srt = np.sort(cnt)[::-1] / cnt.sum()
        top10.append(srt[: max(1, E // 10)].sum())
        top25.append(srt[: max(1, E // 4)].sum())
        p = srt[srt > 0]
        entropy_eff.append(float(np.exp(-(p * np.log(p)).sum())) / E)
    for w in windows:
        out[f"reuse_within_{w}"] = float(np.mean(reuse[w]))
    out["uniform_reuse_within_1"] = k / E  # expected if routing were i.i.d. uniform
    out["top10pct_share"] = float(np.mean(top10))
    out["top25pct_share"] = float(np.mean(top25))
    out["effective_experts_frac"] = float(np.mean(entropy_eff))
    # domain shift: Jaccard of each domain's top-25% expert set, averaged over layers and pairs
    ds = tr.domains()
    c = max(1, E // 4)
    jac = []
    for R in tr.R:
        tops = {d: set(np.argsort(-np.bincount(R[tr.dom == d].ravel(), minlength=E))[:c]) for d in ds}
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                a, b = tops[ds[i]], tops[ds[j]]
                jac.append(len(a & b) / len(a | b))
    out["domain_top25_jaccard"] = float(np.mean(jac))
    out["random_top25_jaccard"] = (c / E) / (2 - c / E)
    return out


def load_pack(path):
    """Read a pack written by scripts/trace_corpus.py: dict with routes [L, T, k] (int64), seq_lens, prompt_lens,
    corpus_idx, user_start, domains, nll, layers, E and meta."""
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files if k not in ("meta", "num_experts")}
    out["routes"] = out["routes"].astype(np.int64)
    out["E"] = int(z["num_experts"])
    out["meta"] = json.loads(str(z["meta"]))
    out["starts"] = np.concatenate([[0], np.cumsum(out["seq_lens"])[:-1]])
    return out

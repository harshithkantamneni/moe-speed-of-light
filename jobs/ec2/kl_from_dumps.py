"""Output parity of a system against an all-VRAM reference from llama-ec-bench dumps (job 090).

Full mode (the --dump-full files of jobs/ec2/ec-bench-fulldump.patch: one float32 logit vector of n_vocab per decode
step, in run order; FILE.json holds n_vocab, FILE.steps lists "seq step target" per record):

    python3 kl_from_dumps.py --ref full_ref.bin --sys ours=full_ours.bin.1 --sys n27=full_n27.bin --out parity.json

Per step, over the whole vocabulary: KL(reference || system) in nats, from log-softmaxes computed in float64; top-1
agreement; the teacher-forced NLL of the forced next token for both. Reported: mean / median / 99th / 99.9th
percentile / max of the KL, top-1 agreement, mean NLL of system and reference and their difference. Steps are matched
by (seq, step) from the .steps files, so a run that lost a sequence is compared on the steps both have.

Top-k fallback (--topk: the plain --dump files, int32 top-5 ids then float32 top-5 logits per step): only the dumped
set is known and the normaliser is not, so the KL is computed over the union of both top-5 sets renormalised within
it, a token missing from one side's dump taking that side's 5th logit (an upper bound), and the result is marked
approximate; the lumped remainder's mass cannot be known from such dumps. The job uses the full mode.
"""
import argparse
import json
import os
import sys

import numpy as np


def read_steps(path):
    out = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                out.append((int(p[0]), int(p[1]), int(p[2])))
    return out


def open_full(path):
    meta = json.load(open(path + ".json"))
    n_vocab = int(meta["n_vocab"])
    n_bytes = os.path.getsize(path)
    n_steps = n_bytes // (4 * n_vocab)
    if n_steps * 4 * n_vocab != n_bytes:
        print(f"warning: {path}: {n_bytes} bytes is not a multiple of {4 * n_vocab}", file=sys.stderr)
    arr = np.memmap(path, dtype=np.float32, mode="r", shape=(n_steps, n_vocab))
    steps = read_steps(path + ".steps") if os.path.exists(path + ".steps") else [(0, i, -1) for i in range(n_steps)]
    if len(steps) != n_steps:
        print(f"warning: {path}: {len(steps)} index lines for {n_steps} records; using the first {min(len(steps), n_steps)}",
              file=sys.stderr)
        n = min(len(steps), n_steps)
        steps, arr = steps[:n], arr[:n]
    return arr, steps, n_vocab


def log_softmax64(x):
    x = x.astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def stats(v):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return {}
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p99": float(np.percentile(v, 99)),
            "p999": float(np.percentile(v, 99.9)), "max": float(v.max()), "n": int(v.size)}


def compare_full(ref, sys_path, chunk=32):
    ra, rsteps, nv = open_full(ref)
    sa, ssteps, nv2 = open_full(sys_path)
    if nv != nv2:
        return {"error": f"n_vocab differs: {nv} vs {nv2}"}
    si = {k[:2]: i for i, k in enumerate(ssteps)}
    pairs = [(i, si[k[:2]], k[2]) for i, k in enumerate(rsteps) if k[:2] in si]
    if not pairs:
        return {"error": "no common (seq, step)"}
    kl, agree, nll_r, nll_s, tgt_n = [], 0, 0.0, 0.0, 0
    kl_by_seq = {}
    for c0 in range(0, len(pairs), chunk):
        p = pairs[c0:c0 + chunk]
        ri = np.array([x[0] for x in p]); sj = np.array([x[1] for x in p]); tg = np.array([x[2] for x in p])
        lr = log_softmax64(ra[ri]); ls = log_softmax64(sa[sj])
        pr = np.exp(lr)
        k = (pr * (lr - ls)).sum(axis=1)
        kl.extend(k.tolist())
        for (i, _, _), v in zip(p, k.tolist()):
            kl_by_seq.setdefault(rsteps[i][0], []).append(v)
        agree += int((lr.argmax(axis=1) == ls.argmax(axis=1)).sum())
        ok = tg >= 0
        if ok.any():
            rows = np.arange(len(p))[ok]
            nll_r += float(-lr[rows, tg[ok]].sum()); nll_s += float(-ls[rows, tg[ok]].sum()); tgt_n += int(ok.sum())
    n = len(pairs)
    out = {"mode": "full_vocab", "n_vocab": nv, "n_steps": n, "n_ref_steps": len(rsteps), "n_sys_steps": len(ssteps),
           "kl_nats": stats(kl), "kl_gt_0.01_frac": float(np.mean(np.array(kl) > 0.01)),
           "kl_gt_0.1_frac": float(np.mean(np.array(kl) > 0.1)),
           "kl_mean_by_seq": {str(s): float(np.mean(v)) for s, v in sorted(kl_by_seq.items())},
           "top1_agreement": agree / n, "top1_disagreements": n - agree}
    if tgt_n:
        out.update({"nll_ref": nll_r / tgt_n, "nll_sys": nll_s / tgt_n, "nll_n": tgt_n,
                    "dnll_abs": (nll_s - nll_r) / tgt_n, "dnll_pct": 100.0 * (nll_s - nll_r) / nll_r if nll_r else None})
    return out


def read_topk(path):
    a = np.fromfile(path, dtype=np.int32).reshape(-1, 10)
    return a[:, :5], a[:, 5:].view(np.float32)


def compare_topk(ref, sys_path):
    rt, rv = read_topk(ref); st, sv = read_topk(sys_path)
    n = min(len(rt), len(st))
    rt, rv, st, sv = rt[:n], rv[:n], st[:n], sv[:n]
    kl = np.zeros(n)
    for i in range(n):
        u = sorted(set(rt[i].tolist()) | set(st[i].tolist()))
        rd = dict(zip(rt[i].tolist(), rv[i].astype(np.float64).tolist()))
        sd = dict(zip(st[i].tolist(), sv[i].astype(np.float64).tolist()))
        lr = np.array([rd.get(t, float(rv[i][4])) for t in u]); ls = np.array([sd.get(t, float(sv[i][4])) for t in u])
        lr -= lr.max(); ls -= ls.max()
        lr -= np.log(np.exp(lr).sum()); ls -= np.log(np.exp(ls).sum())
        kl[i] = float((np.exp(lr) * (lr - ls)).sum())
    return {"mode": "top5_approximate", "approximate": True, "n_steps": n, "kl_nats_top5_renormalised": stats(kl),
            "top1_agreement": float((rt[:, 0] == st[:, 0]).mean()), "top1_disagreements": int((rt[:, 0] != st[:, 0]).sum()),
            "top5_jaccard_mean": float(np.mean([len(set(rt[i]) & set(st[i])) / len(set(rt[i]) | set(st[i])) for i in range(n)])),
            "note": "only the top-5 logits were dumped: the normaliser and the remainder's mass are unknown"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--sys", action="append", default=[], help="name=path (repeatable)")
    ap.add_argument("--topk", action="store_true", help="the inputs are plain --dump (top-5) files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default="{}", help="JSON fields copied into the output")
    a = ap.parse_args()
    res = {"reference": a.ref, "systems": {}, **json.loads(a.meta)}
    for s in a.sys:
        name, path = s.split("=", 1)
        try:
            if not os.path.exists(path):
                res["systems"][name] = {"error": f"missing {path}"}
            else:
                res["systems"][name] = compare_topk(a.ref, path) if a.topk else compare_full(a.ref, path)
        except Exception as e:  # keep the other systems
            res["systems"][name] = {"error": repr(e)[:300]}
        r = res["systems"][name]
        k = r.get("kl_nats") or r.get("kl_nats_top5_renormalised") or {}
        print(f"{name:10s} steps {r.get('n_steps')} KL mean {k.get('mean', float('nan')):.5f} median {k.get('median', float('nan')):.5f} "
              f"p99 {k.get('p99', float('nan')):.4f} p99.9 {k.get('p999', float('nan')):.4f} max {k.get('max', float('nan')):.4f} "
              f"top-1 {r.get('top1_agreement', float('nan')):.4f} NLL {r.get('nll_sys', float('nan')):.4f} vs ref {r.get('nll_ref', float('nan')):.4f} "
              f"({r.get('dnll_pct', float('nan')):+.3f}%)" + (f"  ERROR {r['error']}" if "error" in r else ""), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

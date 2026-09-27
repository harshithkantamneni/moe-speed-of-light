"""Trace a token corpus with the layer-streamed collector and pack the result into one compressed file.

    python scripts/trace_corpus.py --repo /path/or/hub-id --hub-id openai/gpt-oss-20b --corpus corp.jsonl \
        --out traces/gpt-oss-20b_G.npz --device cuda

The pack holds routes [L, T, k] (uint8 when E <= 256, else int16; each row sorted), seq_lens, prompt_lens,
corpus_idx, user_start, domains, nll (collector's fp32 teacher-forced NLL, per sequence len - 1 values,
concatenated), the MoE layer ids and meta.json. `mosl.traces.load_pack` reads it back.
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mosl.collect import collect  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="hub id or local checkpoint directory")
    ap.add_argument("--hub-id", default=None, help="recorded in the pack (default: --repo)")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--moe-chunk", type=int, default=0, help="tokens per MoE sub-block call (0: all at once)")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.corpus) if l.strip()]
    seqs = [r["ids"] for r in rows]
    tmp = a.out + ".d"
    shutil.rmtree(tmp, ignore_errors=True)
    t0 = time.time()
    meta = collect(a.repo, seqs, tmp, dtype=getattr(torch, a.dtype), keep=True, device=a.device,
                   moe_chunk=a.moe_chunk or None, log=lambda s: print(s, flush=True))
    layers = sorted(int(k) for k in meta["layers"])
    E = max(int(v["num_experts"]) for v in meta["layers"].values())
    R = np.stack([np.load(os.path.join(tmp, f"layer{l:03d}.npy")) for l in layers])
    R = R.astype(np.uint8 if E <= 256 else np.int16)
    meta.update(hub_id=a.hub_id or a.repo, corpus=os.path.basename(a.corpus), device=a.device, seconds=time.time() - t0,
                torch=torch.__version__, dtype=a.dtype, moe_chunk=a.moe_chunk)
    np.savez_compressed(a.out, routes=R, layers=np.array(layers), num_experts=E,
                        seq_lens=np.array([len(s) for s in seqs]),
                        prompt_lens=np.array([r.get("prompt_len", 0) for r in rows]),
                        corpus_idx=np.array([r.get("corpus_idx", i) for i, r in enumerate(rows)]),
                        user_start=np.array([r.get("user_start", -1) for r in rows]),
                        domains=np.array([r.get("domain", "") for r in rows]),
                        nll=np.load(os.path.join(tmp, "nll.npy")) if os.path.exists(os.path.join(tmp, "nll.npy"))
                        else np.zeros(0, np.float32),
                        meta=json.dumps(meta))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"packed {a.out}: routes {R.shape} {R.dtype}, {os.path.getsize(a.out)/1e6:.1f} MB, "
          f"ppl {meta.get('teacher_forced_ppl', float('nan')):.3f}, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

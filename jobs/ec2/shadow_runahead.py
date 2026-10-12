"""Job 116: shadow run-ahead on Qwen3-30B-A3B (BF16, Hugging Face transformers).

The model's own AIME-25 text (the routing trace of job 084b) is replayed teacher-forced, one decode step at a time.
After real step g (input ids[P+g], P = prompt length), an online cache of C slots per layer (always-admit, the lowest
decayed score evicted: scripts/value_map.py's online policy) is advanced with step g's routing. Then the model is run
H more tokens ahead on a temporary extension of the KV cache, its MoE layers allowed to execute only the experts that
cache holds (the top-k among residents, weights renormalised); the routers' unmasked top-m choices for each shadow
token are the forecast of steps g+1 .. g+H. The extension is then cropped away. Two shadow inputs:
  self   the model's own greedy choices: x1 = argmax of real step g, x(h+1) = argmax of shadow step h
  orac   the true future inputs ids[P+g+h] (separates token errors from the resident-only approximation)

Outputs (OUT dir): shadow.npz with R [T, L, k] (real routing, uint8), seq [T], inp [T] (each step's real input), F_self / F_orac [T, L, H, m] (uint8,
255 = none), tok_self [T, H], match [T] (argmax of real step g == ids[P+g+1]), and run.json (timings, settings).

    python shadow_runahead.py --model DIR --corpus FILE --out DIR [--nprob N] [--budget-min M]
    python shadow_runahead.py --tiny --out DIR          # CPU smoke test with a small random Qwen3-MoE
"""
import argparse
import json
import os
import time
import types

import numpy as np
import torch
import torch.nn.functional as F


class Ctx:
    mode = None      # None (prefill), "real" or "shadow"
    mask = None      # bool [L, E] on the device: experts the shadow may execute
    log = None       # per MoE layer: the routers' top-m expert ids of the current token(s)
    k = 8
    m = 11


CTX = Ctx()


def experts_forward(ex, x, idx, w, keep):
    """x [T, H]; idx, w [T, k]; keep [T, k] bool or None. The experts' gate_up_proj [E, 2I, H], down_proj [E, H, I]."""
    out = torch.zeros_like(x)
    T, k = idx.shape
    if T == 1:   # decode: at most k experts, one host sync
        es = idx[0].tolist()
        ks = keep[0].tolist() if keep is not None else [True] * k
        for j, e in enumerate(es):
            if not ks[j]:
                continue
            g, u = F.linear(x, ex.gate_up_proj[e]).chunk(2, dim=-1)
            out += F.linear(ex.act_fn(g) * u, ex.down_proj[e]) * w[:, j:j + 1]
        return out
    flat = idx.reshape(-1)
    tok = torch.arange(T, device=x.device).repeat_interleave(k)
    wf = w.reshape(-1)
    if keep is not None:
        kf = keep.reshape(-1)
        flat, tok, wf = flat[kf], tok[kf], wf[kf]
    for e in torch.unique(flat).tolist():
        sel = (flat == e).nonzero(as_tuple=True)[0]
        t = tok[sel]
        g, u = F.linear(x[t], ex.gate_up_proj[e]).chunk(2, dim=-1)
        y = F.linear(ex.act_fn(g) * u, ex.down_proj[e]) * wf[sel, None]
        out.index_add_(0, t, y.to(out.dtype))
    return out


def moe_forward(self, hidden_states):
    B, S, H = hidden_states.shape
    x = hidden_states.reshape(-1, H)
    logits = F.linear(x, self.gate.weight)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    k = self.gate.top_k
    if CTX.mode in ("real", "shadow"):
        CTX.log[self._li] = torch.topk(probs, CTX.m, dim=-1).indices
    keep = None
    if CTX.mode == "shadow" and CTX.mask is not None:
        pm = probs.masked_fill(~CTX.mask[self._li], -1.0)
        w, idx = torch.topk(pm, k, dim=-1)
        keep = w >= 0                       # fewer than k residents: the missing picks are dropped
        w = torch.where(keep, w, torch.zeros_like(w))
    else:
        w, idx = torch.topk(probs, k, dim=-1)
    if self.gate.norm_topk_prob:
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    out = experts_forward(self.experts, x, idx, w.to(x.dtype), keep)
    return out.reshape(B, S, H)


def patch(model):
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
    n = 0
    for layer in model.model.layers:
        if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
            layer.mlp._li = n
            layer.mlp.forward = types.MethodType(moe_forward, layer.mlp)
            n += 1
    return n


class OnlineCache:
    """scripts/value_map.py's online policy (hybrid, no window, kappa = -1e9): every miss is admitted into a free slot
    or the slot of the resident with the lowest decayed score (half-life 16) not requested in this step; a slot filled
    in this step is not a victim again in it. Carried across problems; g is the global step."""

    def __init__(self, L, E, C, hl=16.0):
        self.L, self.E, self.C = L, E, C
        self.res = np.full((L, C), -1, np.int64)
        self.slot = np.full((L, E), -1, np.int64)
        self.score = np.zeros((L, E))
        self.last = np.zeros((L, E), np.int64)
        self.d = 0.5 ** (1.0 / hl)

    def step(self, g, R):
        for l in range(self.L):
            sel = set(int(e) for e in R[l])
            taken = np.zeros(self.C, bool)
            res = self.res[l]
            for e in R[l]:
                e = int(e)
                if self.slot[l, e] >= 0:
                    continue
                s = -1
                for c in range(self.C):
                    if res[c] < 0 and not taken[c]:
                        s = c
                        break
                if s < 0:
                    best = None
                    for c in range(self.C):
                        r = res[c]
                        if r < 0 or taken[c] or r in sel:
                            continue
                        key = self.score[l, r] * self.d ** (g - self.last[l, r])
                        if best is None or key < best:
                            best, s = key, c
                    if s < 0:
                        continue
                old = res[s]
                if old >= 0:
                    self.slot[l, old] = -1
                res[s] = e
                self.slot[l, e] = s
                taken[s] = True
            for e in sel:
                self.score[l, e] = self.score[l, e] * self.d ** (g - self.last[l, e]) + 1.0
                self.last[l, e] = g

    def mask(self, device):
        return torch.from_numpy(self.slot >= 0).to(device)


def tiny_model():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    cfg = Qwen3MoeConfig(vocab_size=97, hidden_size=64, intermediate_size=96, moe_intermediate_size=32, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=16, num_experts_per_tok=4,
                         norm_topk_prob=True, max_position_embeddings=512, decoder_sparse_step=1)
    torch.manual_seed(0)
    m = Qwen3MoeForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.08)
    rows = []
    rng = np.random.default_rng(0)
    for i in range(3):
        P = int(rng.integers(5, 12))
        rows.append(dict(ids=[int(v) for v in rng.integers(0, 97, P + 21)], prompt_len=P))
    return m, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--corpus"); ap.add_argument("--out", required=True)
    ap.add_argument("--tiny", action="store_true"); ap.add_argument("--nprob", type=int, default=30)
    ap.add_argument("--C", type=int, default=16); ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--extra", type=int, default=3, help="m = k + extra router choices logged")
    ap.add_argument("--budget-min", type=float, default=150.0, help="no new problem starts after this many minutes")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    if a.tiny:
        model, rows = tiny_model()
        dev = torch.device("cpu")
        C = 6
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map=a.device).eval()
        rows = [json.loads(l) for l in open(a.corpus)][: a.nprob]
        dev = torch.device(a.device)
        C = a.C
    t_load = time.time() - t0
    L = patch(model)
    k = model.config.num_experts_per_tok
    E = model.config.num_experts
    CTX.k, CTX.m = k, k + a.extra
    H, m = a.H, CTX.m
    cache = OnlineCache(L, E, C)
    Rs, seqs, Fs, Fo, toks, match, times, inps = [], [], [], [], [], [], [], []
    g_glob = 0
    done = 0

    def ids_t(v):
        return torch.tensor([v], dtype=torch.long, device=dev)

    with torch.inference_mode():
        for pi, row in enumerate(rows):
            if (time.time() - t0) / 60 > a.budget_min:
                break
            ids, P = row["ids"], row["prompt_len"]
            G = len(ids) - P - 1
            CTX.mode = None
            out = model(input_ids=ids_t(ids[:P]), use_cache=True, logits_to_keep=1)
            kv = out.past_key_values
            R = np.zeros((G, L, k), np.uint8)
            Fself = np.full((G, L, H, m), 255, np.uint8)
            Forac = np.full((G, L, H, m), 255, np.uint8)
            tk = np.full((G, H), -1, np.int64)
            mt = np.zeros(G, bool)
            inp = np.array(ids[P:P + G], np.int64)    # the real input of each step
            ts = time.time()
            for g in range(G):
                CTX.mode, CTX.log = "real", [None] * L
                out = model(input_ids=ids_t([ids[P + g]]), past_key_values=kv, use_cache=True)
                lg = torch.stack(CTX.log)[:, 0, :]            # [L, m], ranked
                R[g] = lg[:, :k].to("cpu").numpy()
                nxt = int(out.logits[0, -1].argmax())
                mt[g] = nxt == ids[P + g + 1]
                cache.step(g_glob, R[g])
                CTX.mask = cache.mask(dev)
                n = min(H, G - 1 - g)       # forecasts for steps g+1 .. g+n of this problem
                for variant, dst in (("self", Fself), ("orac", Forac)):
                    if n <= 0:
                        break
                    x = nxt if variant == "self" else ids[P + g + 1]
                    CTX.mode = "shadow"
                    for h in range(n):
                        CTX.log = [None] * L
                        o = model(input_ids=ids_t([x]), past_key_values=kv, use_cache=True)
                        dst[g, :, h, :] = torch.stack(CTX.log)[:, 0, :].to("cpu").numpy()
                        if variant == "self":
                            tk[g, h] = x
                            x = int(o.logits[0, -1].argmax())
                        else:
                            x = ids[P + g + 2 + h] if P + g + 2 + h < len(ids) else 0
                    kv.crop(-n)
                CTX.mode = None
                g_glob += 1
            times.append((time.time() - ts) / G)
            Rs.append(R); Fs.append(Fself); Fo.append(Forac); toks.append(tk); match.append(mt); inps.append(inp)
            seqs.append(np.full(G, pi, np.int64))
            done += 1
            print(f"problem {pi}: {G} steps, {times[-1] * 1e3:.0f} ms per step, greedy match {mt.mean():.3f}, "
                  f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    np.savez_compressed(os.path.join(a.out, "shadow.npz"), R=np.concatenate(Rs), seq=np.concatenate(seqs),
                        F_self=np.concatenate(Fs), F_orac=np.concatenate(Fo), tok_self=np.concatenate(toks),
                        match=np.concatenate(match), inp=np.concatenate(inps), E=E, C=C, H=H, k=k, m=m, L=L)
    json.dump(dict(problems=done, C=C, H=H, k=k, m=m, E=E, L=L, load_s=t_load, ms_per_step=[t * 1e3 for t in times],
                   minutes=(time.time() - t0) / 60, torch=torch.__version__,
                   transformers=__import__("transformers").__version__),
              open(os.path.join(a.out, "run.json"), "w"), indent=1)
    print(f"done: {done} problems in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

"""Layer-streaming, teacher-forced routing-trace collector.

Key idea: for a causal decoder, the experts a token is routed to during
autoregressive decode depend only on its prefix. Teacher-forced prefill over
the same token sequence therefore yields *exactly* the decode-time routing
trace (up to floating-point ties). We exploit this to collect traces for
models far larger than host RAM: weights are streamed one decoder layer at a
time, every sequence in the corpus is pushed through that layer, and the
layer is freed before the next is loaded. Each weight byte is read once per
corpus instead of once per token, so a 2-vCPU / 7 GB box can trace a 30B MoE.

Output: for every MoE layer l, an int16 array [n_tokens, top_k] of selected
expert ids, plus per-sequence offsets.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from collections import defaultdict

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

torch.set_grad_enabled(False)

EXPERT_RE = re.compile(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


class ShardStore:
    """Resolves tensor names to safetensors shards, downloading lazily and
    deleting shards once no remaining layer needs them (disk is limited)."""

    def __init__(self, repo: str, cache_dir: str, keep: bool = False):
        self.repo, self.cache_dir, self.keep = repo, cache_dir, keep
        self.local = os.path.isdir(repo)   # a local checkpoint directory: read in place, never delete
        if self.local:
            self.keep = True
        try:
            idx_path = self._file("model.safetensors.index.json")
            self.weight_map = json.load(open(idx_path))["weight_map"]
        except Exception:
            p = self._file("model.safetensors")
            with safe_open(p, "pt") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
        self.paths: dict[str, str] = {}
        self.remaining = defaultdict(set)  # shard -> tensor names not yet consumed
        for k, s in self.weight_map.items():
            self.remaining[s].add(k)

    def _file(self, f):
        if self.local:
            p = os.path.join(self.repo, f)
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            return p
        return hf_hub_download(self.repo, f, cache_dir=self.cache_dir)

    def names(self, prefix: str):
        return [k for k in self.weight_map if k.startswith(prefix)]

    def get(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        if shard not in self.paths:
            self.paths[shard] = self._file(shard)
        with safe_open(self.paths[shard], "pt") as f:
            t = f.get_tensor(name)
        self.remaining[shard].discard(name)
        if not self.remaining[shard] and not self.keep:
            real = os.path.realpath(self.paths[shard])
            for p in {self.paths[shard], real}:
                try:
                    os.remove(p)
                except OSError:
                    pass
        return t


class PackedExperts(torch.nn.Module):
    """gpt-oss experts kept in their MXFP4 checkpoint format and dequantized one
    expert at a time during the forward pass (a 120B layer would need ~13 GB
    in fp32). Math mirrors transformers' GptOssExperts exactly."""

    def __init__(self, gu_blocks, gu_scales, gu_bias, dn_blocks, dn_scales, dn_bias, dtype):
        super().__init__()
        self.t = dict(gu_blocks=gu_blocks, gu_scales=gu_scales, dn_blocks=dn_blocks, dn_scales=dn_scales)
        self.gu_bias, self.dn_bias = gu_bias.to(dtype=dtype), dn_bias.to(dtype=dtype)
        self.num_experts = gu_blocks.shape[0]
        self.dtype = dtype
        self.alpha, self.limit = 1.702, 7.0

    def _w(self, kind, e):
        from transformers.integrations.mxfp4 import convert_moe_packed_tensors
        return convert_moe_packed_tensors(self.t[f"{kind}_blocks"][e:e + 1], self.t[f"{kind}_scales"][e:e + 1],
                                          dtype=self.dtype)[0]

    def forward(self, hidden_states, router_indices=None, routing_weights=None):
        out = torch.zeros_like(hidden_states)
        for e in torch.unique(router_indices).tolist():
            tok, pos = torch.where(router_indices == e)
            x = hidden_states[tok]
            gate_up = x @ self._w("gu", e) + self.gu_bias[e]
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            h = (up + 1) * (gate * torch.sigmoid(gate * self.alpha))
            y = h @ self._w("dn", e) + self.dn_bias[e]
            out.index_add_(0, tok, (y * routing_weights[tok, pos, None]).to(out.dtype))
        return out


def is_packed(store, i):
    return f"model.layers.{i}.mlp.experts.gate_up_proj_blocks" in store.weight_map


class RangeStore:
    """Reads individual tensors straight out of remote safetensors files with
    HTTP range requests: no shard ever touches the disk, so models far larger
    than the disk can be traced. Same interface as ShardStore."""

    DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "U8": torch.uint8,
          "I8": torch.int8, "I32": torch.int32, "I64": torch.int64, "F8_E4M3": torch.float8_e4m3fn}

    def __init__(self, repo: str, cache_dir: str = None, keep: bool = False):
        import requests
        self.repo, self.sess = repo, requests.Session()
        try:
            idx_path = hf_hub_download(repo, "model.safetensors.index.json")
            self.weight_map = json.load(open(idx_path))["weight_map"]
        except Exception:
            self.weight_map = None
        self.headers = {}
        if self.weight_map is None:
            h = self._header("model.safetensors")
            self.weight_map = {k: "model.safetensors" for k in h if k != "__metadata__"}

    def _url(self, f):
        return f"https://huggingface.co/{self.repo}/resolve/main/{f}"

    def _range(self, f, a, b):
        for attempt in range(6):
            try:
                r = self.sess.get(self._url(f), headers={"Range": f"bytes={a}-{b}"}, timeout=600)
                if r.status_code == 206 and len(r.content) == b - a + 1:
                    return r.content
            except Exception:
                pass
            time.sleep(2 ** attempt)
        raise RuntimeError(f"range read failed: {f} {a}-{b}")

    def _header(self, f):
        if f not in self.headers:
            import struct
            n = struct.unpack("<Q", self._range(f, 0, 7))[0]
            self.headers[f] = (json.loads(self._range(f, 8, 8 + n - 1)), 8 + n)
        return self.headers[f][0]

    def names(self, prefix: str):
        return [k for k in self.weight_map if k.startswith(prefix)]

    def get(self, name: str) -> torch.Tensor:
        f = self.weight_map[name]
        h = self._header(f)
        base = self.headers[f][1]
        info = h[name]
        a, b = info["data_offsets"]
        buf = bytearray(self._range(f, base + a, base + b - 1))
        return torch.frombuffer(buf, dtype=self.DT[info["dtype"]]).reshape(info["shape"]).clone()


def load_layer(layer: torch.nn.Module, store: ShardStore, i: int, dtype=torch.float32) -> None:
    """Copy checkpoint tensors for layer i straight into the (materialized)
    layer's parameters, one tensor at a time, to keep peak RAM ~= layer size.
    Per-expert projections are written into transformers v5's fused layout:
    gate_up_proj[e] = [gate_proj; up_proj], down_proj[e] = down_proj."""
    prefix = f"model.layers.{i}."
    params = dict(layer.named_parameters())
    params.update(dict(layer.named_buffers()))
    seen = set()
    packed = {}
    for name in store.names(prefix):
        local = name[len(prefix):]
        if is_packed(store, i) and local.startswith("mlp.experts."):
            packed[local[len("mlp.experts."):]] = store.get(name)
            continue
        t = store.get(name)
        m = EXPERT_RE.match(local)
        if m:
            e, kind = int(m.group(1)), m.group(2)
            if kind == "down_proj":
                tgt, key = params["mlp.experts.down_proj"][e], "mlp.experts.down_proj"
            else:
                gu = params["mlp.experts.gate_up_proj"]
                half = gu.shape[1] // 2
                tgt = gu[e, :half] if kind == "gate_proj" else gu[e, half:]
                key = "mlp.experts.gate_up_proj"
        else:
            key = "mlp.gate.e_score_correction_bias" if local == "mlp.gate.bias" else local
            tgt = params[key]
        if tgt.shape != t.shape:
            raise RuntimeError(f"layer {i}: shape mismatch for {local}: {tuple(t.shape)} vs {tuple(tgt.shape)}")
        tgt.data.copy_(t) if tgt is params.get(key) else tgt.copy_(t)
        seen.add(key)
        del t
    if packed:
        dev = next(layer.parameters()).device
        packed = {k: v.to(dev) for k, v in packed.items()}
        layer.mlp.experts = PackedExperts(packed["gate_up_proj_blocks"], packed["gate_up_proj_scales"],
                                          packed["gate_up_proj_bias"], packed["down_proj_blocks"],
                                          packed["down_proj_scales"], packed["down_proj_bias"], dtype)
    missing = [k for k in params if k not in seen and not k.endswith("inv_freq")]
    if missing:
        raise RuntimeError(f"layer {i}: parameters not found in checkpoint: {missing}")


def causal_mask(T: int, dtype, window: int = 0, device="cpu") -> torch.Tensor:
    m = torch.full((T, T), float("-inf"), dtype=dtype, device=device).triu(1)
    if window:  # sliding-window attention: query i sees keys (i-window, i]
        m = m + torch.full((T, T), float("-inf"), dtype=dtype, device=device).tril(-window)
    return m[None, None]


def collect(repo, sequences, out_dir, dtype=torch.float32, cache_dir=None, keep=False, log=print, remote=False,
            device="cpu", moe_chunk=None):
    """device="cuda" runs every layer on the GPU in the same precision (TF32 disabled); selections can then differ
    from the CPU's only at floating-point near-ties."""
    os.makedirs(out_dir, exist_ok=True)
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    cfg = AutoConfig.from_pretrained(repo)
    cfg._attn_implementation = "eager" if cfg.model_type == "gpt_oss" else "sdpa"
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype)
    base = model.model
    store = (RangeStore if remote else ShardStore)(repo, cache_dir or os.path.join(out_dir, "_hf"), keep=keep)

    # embeddings
    emb = store.get("model.embed_tokens.weight").to(dtype)
    hs = [emb[torch.tensor(s)].unsqueeze(0).to(device) for s in sequences]
    tied_head = emb if getattr(cfg, "tie_word_embeddings", False) else None
    del emb
    rotary = type(base.rotary_emb)(config=cfg).to(device)  # real (non-meta) instance
    pos = [rotary(h, torch.arange(h.shape[1], device=device)[None]) for h in hs]
    masks = {}

    lens = np.array([len(s) for s in sequences])
    np.save(os.path.join(out_dir, "seq_lens.npy"), lens)
    meta = {"repo": repo, "n_seqs": len(sequences), "n_tokens": int(lens.sum()), "layers": {}}

    for i, layer in enumerate(base.layers):
        t0 = time.time()
        if is_packed(store, i):
            layer.mlp.experts = torch.nn.Module()  # replaced by PackedExperts in load_layer
        layer = layer.to_empty(device=device)
        load_layer(layer, store, i, dtype)
        rec = []
        gate = getattr(layer.mlp, "gate", None) or getattr(layer.mlp, "router", None)
        lt = getattr(cfg, "layer_types", None)
        window = int(cfg.sliding_window) if lt and "sliding" in lt[i] and getattr(cfg, "sliding_window", None) else 0
        hook = None
        if gate is not None and hasattr(gate, "top_k"):
            hook = gate.register_forward_hook(lambda mod, inp, out: rec.append(out[2].to(torch.int16).clone()))
        # Phase A (per sequence): attention sub-block. Phase B (all tokens at once):
        # the MoE sub-block is token-wise, so we run it once over the concatenated
        # corpus; each expert's weights are then touched once per layer.
        normed, lens_j = [], []
        for j, h in enumerate(hs):
            T = h.shape[1]
            if (T, window) not in masks:
                masks[(T, window)] = causal_mask(T, dtype, window, device)
            a = layer.self_attn(hidden_states=layer.input_layernorm(h), attention_mask=masks[(T, window)],
                                position_embeddings=pos[j], position_ids=torch.arange(T, device=device)[None])[0]
            hs[j] = h + a
            normed.append(layer.post_attention_layernorm(hs[j]))
            lens_j.append(T)
        x_all = torch.cat(normed, dim=1)
        del normed
        n_tok = x_all.shape[1]
        step = moe_chunk or n_tok       # the MoE sub-block is token-wise: chunking bounds its peak memory
        ys = []
        for c0 in range(0, n_tok, step):
            yc = layer.mlp(x_all[:, c0:c0 + step])
            ys.append(yc[0] if isinstance(yc, tuple) else yc)
        y = torch.cat(ys, dim=1) if len(ys) > 1 else ys[0]
        del ys, x_all
        for j, part in enumerate(torch.split(y, lens_j, dim=1)):
            hs[j] = hs[j] + part
        del y
        if hook is not None:
            hook.remove()
            idx = torch.cat(rec, 0).cpu().numpy()
            idx.sort(axis=1)
            np.save(os.path.join(out_dir, f"layer{i:03d}.npy"), idx)
            meta["layers"][i] = {"top_k": int(idx.shape[1]), "num_experts": int(gate.num_experts)}
        base.layers[i] = torch.nn.Module()  # drop weights
        del layer
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        log(f"layer {i:3d} done in {time.time()-t0:6.1f}s  moe={hook is not None}")
        json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)

    # final-norm + lm_head -> teacher-forced NLL as a correctness check
    masks.clear()
    norm = type(base.norm)(cfg.hidden_size, eps=cfg.rms_norm_eps).to(device)
    norm.weight.data = store.get("model.norm.weight").to(device=device, dtype=dtype)
    W = tied_head if tied_head is not None else (
        store.get("lm_head.weight").to(dtype) if "lm_head.weight" in store.weight_map else None)
    nll, cnt = 0.0, 0
    if W is not None:
        W = W.to(device)
        pers = []
        for j, h in enumerate(hs):
            logits = norm(h)[0, :-1] @ W.T
            tgt = torch.tensor(sequences[j][1:], device=device)
            per = torch.nn.functional.cross_entropy(logits.float(), tgt, reduction="none").cpu()
            pers.append(per.numpy().astype(np.float32))
            nll += per.sum().item()
            cnt += len(tgt)
            del logits
        # nll[i] = -log p(token i+1 | tokens <= i) within its sequence, sequences concatenated (len - 1 each)
        np.save(os.path.join(out_dir, "nll.npy"), np.concatenate(pers) if pers else np.zeros(0, np.float32))
        meta["teacher_forced_ppl"] = float(np.exp(nll / cnt))
        log(f"teacher-forced perplexity: {meta['teacher_forced_ppl']:.3f}")
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--corpus", required=True, help="jsonl with {'domain','ids'} per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--remote", action="store_true", help="read tensors via HTTP range requests (no disk)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cache-dir", default=None, help="HF cache for downloaded shards (default: <out>/_hf)")
    a = ap.parse_args()
    seqs, doms = [], []
    for line in open(a.corpus):
        r = json.loads(line)
        seqs.append(r["ids"])
        doms.append(r["domain"])
    os.makedirs(a.out, exist_ok=True)
    json.dump(doms, open(os.path.join(a.out, "domains.json"), "w"))
    collect(a.repo, seqs, a.out, dtype=getattr(torch, a.dtype), keep=a.keep, remote=a.remote, device=a.device,
            cache_dir=a.cache_dir, log=lambda s: print(s, flush=True))

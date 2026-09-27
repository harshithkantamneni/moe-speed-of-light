"""Exact byte accounting from a GGUF file's header.

Reads only the header (key/values and tensor descriptors) over HTTP range
requests, then sizes every tensor from its ggml type, so the bytes the
performance model charges for one decode step are exactly those the engine
stores: per-layer routed-expert bytes (one expert), dense bytes per token,
output head, and token embedding (read as one row per token, so ~0).
"""
from __future__ import annotations

import json
import os
import struct
import sys
from dataclasses import dataclass, asdict

# ggml type id -> (elements per block, bytes per block)
GGML = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
        10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
        16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82),
        23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2),
        34: (256, 54), 35: (256, 66), 39: (32, 17)}
GGML_NAME = {0: "F32", 1: "F16", 8: "Q8_0", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 30: "BF16", 39: "MXFP4"}
CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "gguf_bytes.json")


class _Remote:
    """Sequential reader over an HTTP URL, fetched in growing chunks."""

    def __init__(self, url, chunk=8 << 20):
        import requests
        self.url, self.s, self.buf, self.pos, self.chunk = url, requests.Session(), b"", 0, chunk

    def read(self, n):
        while self.pos + n > len(self.buf):
            a = len(self.buf)
            r = self.s.get(self.url, headers={"Range": f"bytes={a}-{a + self.chunk - 1}"}, timeout=300)
            r.raise_for_status()
            if not r.content:
                raise EOFError(self.url)
            self.buf += r.content
            self.chunk *= 2
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out


def _u(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]


def _str(f):
    return f.read(_u(f, "Q")).decode("utf-8", "replace")


_SC = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def _val(f, t):
    if t in _SC:
        return _u(f, _SC[t])
    if t == 8:
        return _str(f)
    if t == 9:
        et, n = _u(f, "I"), _u(f, "Q")
        if et in _SC:  # skip bulk scalar arrays cheaply
            f.read(n * struct.calcsize(_SC[et]))
            return f"<array {n}>"
        return [_val(f, et) for _ in range(n)] if n < 64 else [_val(f, et) for _ in range(n)] and f"<array {n}>"
    raise ValueError(f"gguf value type {t}")


def read_header(url):
    f = _Remote(url)
    assert f.read(4) == b"GGUF", "not a GGUF file"
    ver, nt, nkv = _u(f, "I"), _u(f, "Q"), _u(f, "Q")
    kv = {}
    for _ in range(nkv):
        k = _str(f)
        kv[k] = _val(f, _u(f, "I"))
    tensors = []
    for _ in range(nt):
        name = _str(f)
        nd = _u(f, "I")
        dims = [_u(f, "Q") for _ in range(nd)]
        t, off = _u(f, "I"), _u(f, "Q")
        n = 1
        for d in dims:
            n *= d
        be, bb = GGML[t]
        tensors.append(dict(name=name, dims=dims, type=t, nbytes=n // be * bb))
    return ver, kv, tensors


@dataclass
class GGUFBytes:
    file: str
    arch: str
    n_layers: int
    n_moe_layers: int
    n_experts: int
    top_k: int
    expert_bytes_per_layer: list    # bytes of ONE routed expert, per MoE layer (type mix can vary by layer)
    dense_bytes: float               # everything read per token except routed experts and the embedding table
    head_bytes: float
    embed_bytes: float
    routed_bytes_total: float
    file_tensor_bytes: float
    types: dict                      # tensor-class -> {ggml type: count}

    @property
    def mean_expert_bytes(self):
        return sum(self.expert_bytes_per_layer) / len(self.expert_bytes_per_layer)


def account(repo, fname):
    db = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    key = f"{repo}/{fname}"
    if key in db:
        return GGUFBytes(**db[key])
    ver, kv, ts = read_header(f"https://huggingface.co/{repo}/resolve/main/{fname}")
    arch = kv["general.architecture"]
    L = kv[f"{arch}.block_count"]
    E = kv[f"{arch}.expert_count"]
    k = kv[f"{arch}.expert_used_count"]
    per_layer, dense, head, emb, routed, types = {}, 0, 0, 0, 0, {}
    for t in ts:
        n = t["name"]
        cls = ("expert" if "_exps" in n else "head" if n.startswith("output.") else
               "embed" if n.startswith("token_embd") else "dense")
        types.setdefault(cls, {}).setdefault(GGML_NAME.get(t["type"], str(t["type"])), 0)
        types[cls][GGML_NAME.get(t["type"], str(t["type"]))] += 1
        if cls == "expert":
            li = int(n.split(".")[1])
            if t["type"] in (0, 1, 30) and "bias" in n:  # per-expert biases (gpt-oss) are read with the expert
                pass
            per_layer[li] = per_layer.get(li, 0) + t["nbytes"]
            routed += t["nbytes"]
        elif cls == "head":
            head += t["nbytes"]
        elif cls == "embed":
            emb += t["nbytes"]
        else:
            dense += t["nbytes"]
    if head == 0:  # tied embeddings: the head reads the whole table each token
        head = emb
    layers = sorted(per_layer)
    g = GGUFBytes(file=key, arch=arch, n_layers=L, n_moe_layers=len(layers), n_experts=E, top_k=k,
                  expert_bytes_per_layer=[per_layer[i] / E for i in layers], dense_bytes=float(dense),
                  head_bytes=float(head), embed_bytes=float(emb), routed_bytes_total=float(routed),
                  file_tensor_bytes=float(sum(t["nbytes"] for t in ts)), types=types)
    db[key] = asdict(g)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(db, open(CACHE, "w"), indent=1)
    return g


GGUFS = {
    "gpt-oss-20b-mxfp4": ("ggml-org/gpt-oss-20b-GGUF", "gpt-oss-20b-MXFP4.gguf"),
    "gpt-oss-120b-mxfp4": ("ggml-org/gpt-oss-120b-GGUF", "gpt-oss-120b-MXFP4.gguf"),
    "qwen3-30b-a3b-q4_k_m": ("unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"),
    "qwen3-30b-a3b-q8_0": ("unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF", "Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf"),
    # anchors for the audit (A100 40 GB)
    "mixtral-8x7b-q4_k_m": ("mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF", "Mixtral-8x7B-Instruct-v0.1.Q4_K_M.gguf"),
    "mixtral-8x7b-q8_0": ("mradermacher/Mixtral-8x7B-Instruct-v0.1-GGUF", "Mixtral-8x7B-Instruct-v0.1.Q8_0.gguf"),
    "phi3.5-moe-q4_k_m": ("bartowski/Phi-3.5-MoE-instruct-GGUF", "Phi-3.5-MoE-instruct-Q4_K_M.gguf"),
    "qwen2-57b-q4_k_m": ("Qwen/Qwen2-57B-A14B-Instruct-GGUF", "qwen2-57b-a14b-instruct-q4_k_m.gguf"),
    "dsv2lite-q8_0": ("mradermacher/DeepSeek-V2-Lite-Chat-GGUF", "DeepSeek-V2-Lite-Chat.Q8_0.gguf"),
}

if __name__ == "__main__":
    for name in sys.argv[1:] or GGUFS:
        g = account(*GGUFS[name])
        eb = g.expert_bytes_per_layer
        print(f"{name}: L={g.n_layers} moe={g.n_moe_layers} E={g.n_experts} k={g.top_k} "
              f"expert={min(eb)/1e6:.2f}-{max(eb)/1e6:.2f} MB dense={g.dense_bytes/1e9:.3f} GB "
              f"head={g.head_bytes/1e9:.3f} GB embed={g.embed_bytes/1e9:.3f} GB routed={g.routed_bytes_total/1e9:.2f} GB "
              f"file={g.file_tensor_bytes/1e9:.2f} GB types={g.types}")

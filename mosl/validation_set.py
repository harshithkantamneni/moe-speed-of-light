"""Annotated subset of published measurements used to validate the model.

Every row points to an id in data/published_measurements.json (which carries
the source URL and the quoted table/line). Here we add only what the model
needs: model repo, bits per weight, placement, and datasheet peak bandwidths.
Assumptions not stated by the source are listed in `assume`.

Peak bandwidths (GB/s): GPU memory from vendor datasheets; host DRAM =
channels x MT/s x 8 B; PCIe x16 = 15.75 (Gen3), 31.5 (Gen4), 63 (Gen5).
"""

# --- datasheet peaks -------------------------------------------------------
GPU = {"RTX 4090": 1008, "RTX 4090D": 1008, "RTX 5090": 1792, "RTX 3090": 936, "RTX 3090 Ti": 1008,
       "RTX 4080": 717, "RTX 4070": 504, "RTX 3060": 360, "RTX 5070": 672, "RX 7900 XT": 800,
       "RTX 3070": 448, "A100-80GB-SXM": 2039, "RTX 3080 Mobile 16GB": 448, "T4": 320}
PCIE = {3: 15.75, 4: 31.5, 5: 63.0}


def dram(ch, mts):
    return ch * mts * 8 / 1000


# GGUF / checkpoint bits-per-weight. File-size derived where the source gives a size.
BPW = {
    "Q8_0": 8.5, "Q6_K": 6.5625, "Q4_K_S": 4.58, "FP8": 8.03, "MXFP4": 4.25, "BF16": 16.0,
    "GGUF_GPTOSS_DENSE": 8.5,  # ggml-org gpt-oss GGUF: 12.1 GB for 20.9B params -> non-expert tensors ~Q8_0
    "Q4_K_M_DS": 404e9 * 8 / 671.03e9,        # DeepSeek-R1 Q4_K_M "404GB" (P054 text)
    "IQ4_XS_Q3_30B": 15.4 * 2**30 * 8 / 30.53e9,  # Qwen3-30B-A3B IQ4_XS 15.4 GiB (P087)
    "Q6_K_Q3_235B": 193e9 * 8 / 235.09e9,     # unsloth Q6_K ~193GB (P096)
    "IQ6_K_mix": 7.237, "IQ3_K_mix": 3.903,   # BPW printed in the posters' logs
    "HQQ2_g16": 2 + 16 / 16,                  # 2-bit, 8-bit scale+zero per 16 weights
    "HQQ3_g64": 3 + 16 / 64,
    "HQQ4_g64": 4 + 16 / 64,
}

DS3 = "deepseek-ai/DeepSeek-V3"
Q235 = "Qwen/Qwen3-235B-A22B"
Q30 = "Qwen/Qwen3-30B-A3B"
M22 = "mistralai/Mixtral-8x22B-v0.1"
M7 = "mistralai/Mixtral-8x7B-v0.1"
G20 = "openai/gpt-oss-20b"
G120 = "openai/gpt-oss-120b"
K2 = "moonshotai/Kimi-K2-Instruct"

# platform: "desktop" (2-channel DRAM) or "server" (8+ channels) is derived from bw_cpu.
# kind: cpu_only | static (n_cpu = MoE layers whose experts run on CPU) | fetch
# engine family is used for per-engine efficiency analysis.
ROWS = [
    # ---------------- CPU-only, single NUMA domain ----------------
    dict(id="P102", repo=M22, kind="cpu_only", b_exp=BPW["Q8_0"], b_dense=BPW["Q8_0"], bw_cpu=dram(6, 4800), engine="llama.cpp", tok_s=3.91, ctx=0),
    dict(id="P104", repo=M22, kind="cpu_only", b_exp=BPW["Q8_0"], b_dense=BPW["Q8_0"], bw_cpu=dram(8, 6400), engine="llama.cpp", tok_s=6.8, ctx=0),
    dict(id="P106", repo=M22, kind="cpu_only", b_exp=BPW["Q8_0"], b_dense=BPW["Q8_0"], bw_cpu=dram(12, 4800), engine="llama.cpp", tok_s=7.02, ctx=0),
    dict(id="P108", repo=DS3, kind="cpu_only", b_exp=BPW["Q4_K_S"], b_dense=BPW["Q4_K_S"], bw_cpu=dram(8, 6400), engine="llama.cpp", tok_s=9.08, ctx=0),
    dict(id="P110", repo=DS3, kind="cpu_only", b_exp=BPW["Q4_K_S"], b_dense=BPW["Q4_K_S"], bw_cpu=dram(12, 4800), engine="llama.cpp", tok_s=8.48, ctx=0),
    dict(id="P112", repo=DS3, kind="cpu_only", b_exp=BPW["Q8_0"], b_dense=BPW["Q8_0"], bw_cpu=dram(12, 4800), engine="llama.cpp", tok_s=6.03, ctx=0),
    dict(id="P094", repo=Q235, kind="cpu_only", b_exp=BPW["IQ6_K_mix"], b_dense=BPW["IQ6_K_mix"], bw_cpu=dram(12, 6000), engine="ik_llama.cpp", tok_s=14.26, ctx=0, assume="VM"),
    dict(id="P096", repo=Q235, kind="cpu_only", b_exp=BPW["Q6_K_Q3_235B"], b_dense=BPW["Q6_K_Q3_235B"], bw_cpu=dram(12, 6000), engine="llama.cpp", tok_s=14.13, ctx=0, assume="VM"),
    dict(id="P100", repo=Q235, kind="cpu_only", b_exp=BPW["IQ6_K_mix"], b_dense=BPW["IQ6_K_mix"], bw_cpu=dram(12, 6000), engine="ik_llama.cpp", tok_s=16.24, ctx=0, assume="VM"),
    dict(id="P123", repo=G120, kind="cpu_only", b_exp=BPW["MXFP4"], b_dense=BPW["GGUF_GPTOSS_DENSE"], bw_cpu=dram(2, 4800), engine="llama.cpp", tok_s=14.75, ctx=256),
    dict(id="P087", repo=Q30, kind="cpu_only", b_exp=BPW["IQ4_XS_Q3_30B"], b_dense=BPW["IQ4_XS_Q3_30B"], bw_cpu=dram(2, 6000), engine="ik_llama.cpp", tok_s=26.35, ctx=0, assume="RAM speed not stated: DDR5-6000 2ch assumed (AM5)"),
    dict(id="P088", repo=Q30, kind="cpu_only", b_exp=BPW["IQ4_XS_Q3_30B"], b_dense=BPW["IQ4_XS_Q3_30B"], bw_cpu=dram(2, 6000), engine="llama.cpp", tok_s=25.26, ctx=0, assume="RAM speed not stated: DDR5-6000 2ch assumed (AM5)"),
    # ---------------- static CPU offload, consumer desktops ----------------
    dict(id="P089", repo=Q30, kind="static", n_cpu=14, b_exp=BPW["IQ4_XS_Q3_30B"], b_dense=BPW["IQ4_XS_Q3_30B"], bw_gpu=GPU["RTX 4080"], bw_cpu=dram(2, 6000), engine="ik_llama.cpp", tok_s=76.74, ctx=0, assume="RAM speed not stated: DDR5-6000 2ch assumed"),
    dict(id="P090", repo=Q30, kind="static", n_cpu=14, b_exp=BPW["IQ4_XS_Q3_30B"], b_dense=BPW["IQ4_XS_Q3_30B"], bw_gpu=GPU["RTX 4080"], bw_cpu=dram(2, 6000), engine="llama.cpp", tok_s=68.68, ctx=0, assume="RAM speed not stated: DDR5-6000 2ch assumed"),
    dict(id="P091", repo=Q235, kind="static", n_cpu=82, b_exp=BPW["IQ3_K_mix"], b_dense=6.6, bw_gpu=GPU["RTX 3090 Ti"], bw_cpu=dram(2, 6400), engine="ik_llama.cpp", tok_s=10.73, ctx=0),
    dict(id="P114", repo=G20, kind="static", n_cpu=4, b_exp=BPW["MXFP4"], b_dense=BPW["GGUF_GPTOSS_DENSE"], bw_gpu=GPU["RX 7900 XT"], bw_cpu=dram(2, 3200), engine="llama.cpp", tok_s=60, ctx=256, assume="approximate web-UI speed"),
    dict(id="P115", repo=G20, kind="static", n_cpu=8, b_exp=BPW["MXFP4"], b_dense=BPW["GGUF_GPTOSS_DENSE"], bw_gpu=GPU["RX 7900 XT"], bw_cpu=dram(2, 3200), engine="llama.cpp", tok_s=38, ctx=256, assume="approximate web-UI speed"),
    dict(id="P116", repo=G20, kind="static", n_cpu=16, b_exp=BPW["MXFP4"], b_dense=BPW["GGUF_GPTOSS_DENSE"], bw_gpu=GPU["RX 7900 XT"], bw_cpu=dram(2, 3200), engine="llama.cpp", tok_s=26, ctx=256, assume="approximate web-UI speed"),
    dict(id="P117", repo=G20, kind="static", n_cpu=24, b_exp=BPW["MXFP4"], b_dense=BPW["GGUF_GPTOSS_DENSE"], bw_gpu=GPU["RX 7900 XT"], bw_cpu=dram(2, 3200), engine="llama.cpp", tok_s=20, ctx=256, assume="approximate web-UI speed"),
    dict(id="P120", repo=G20, kind="static", n_cpu=2, b_exp=BPW["MXFP4"], b_dense=BPW["GGUF_GPTOSS_DENSE"], bw_gpu=GPU["RTX 5070"], bw_cpu=dram(2, 6000), engine="llama.cpp", tok_s=62.21, ctx=256),
    dict(id="P121", repo=G120, kind="static", n_cpu=31, b_exp=BPW["MXFP4"], b_dense=BPW["GGUF_GPTOSS_DENSE"], bw_gpu=GPU["RTX 4070"], bw_cpu=dram(2, 6000), engine="llama.cpp", tok_s=28.0, ctx=512),
    # all-in-VRAM baselines reported in the same threads (identify GPU efficiency)
    dict(id="P114v", src="P114", repo=G20, kind="static", n_cpu=0, b_exp=BPW["MXFP4"], b_dense=BPW["GGUF_GPTOSS_DENSE"], bw_gpu=GPU["RX 7900 XT"], bw_cpu=dram(2, 3200), engine="llama.cpp", tok_s=94, ctx=256, assume="approximate; 'All-in-VRAM baseline ~94 tok/s' in thread"),
    dict(id="P120v", src="P120", repo=G20, kind="static", n_cpu=0, b_exp=BPW["MXFP4"], b_dense=BPW["GGUF_GPTOSS_DENSE"], bw_gpu=GPU["RTX 5070"], bw_cpu=dram(2, 6000), engine="llama.cpp", tok_s=128, ctx=1024, assume="'128 t/s fully in VRAM (2K ctx)'"),
    dict(id="P118v", src="P118", repo=G20, kind="static", n_cpu=0, b_exp=BPW["MXFP4"], b_dense=BPW["GGUF_GPTOSS_DENSE"], bw_gpu=GPU["RTX 3060"], bw_cpu=51.2, engine="llama.cpp", tok_s=75, ctx=1024, assume="'Fully in VRAM with 2K ctx: 75 tok/s'"),
    # ---------------- static CPU offload, servers ----------------
    dict(id="P038", repo=DS3, kind="static", n_cpu=58, b_exp=BPW["Q4_K_M_DS"], b_dense=BPW["Q4_K_M_DS"], bw_gpu=GPU["RTX 4090D"], bw_cpu=dram(16, 4800), engine="KTransformers", tok_s=12.208, ctx=500, assume="dual-socket, weights replicated per NUMA node"),
    dict(id="P037", repo=DS3, kind="static", n_cpu=58, top_k=6, b_exp=BPW["Q4_K_M_DS"], b_dense=BPW["Q4_K_M_DS"], bw_gpu=GPU["RTX 4090D"], bw_cpu=dram(16, 4800), engine="KTransformers", tok_s=13.69, ctx=500, assume="dual-socket; top-6 routing"),
    dict(id="P040", repo=DS3, kind="static", n_cpu=58, b_exp=BPW["Q4_K_M_DS"], b_dense=BPW["Q4_K_M_DS"], bw_gpu=GPU["RTX 4090D"], bw_cpu=dram(8, 4800), engine="KTransformers", tok_s=8.73, ctx=500),
    dict(id="P039", repo=DS3, kind="static", n_cpu=58, top_k=6, b_exp=BPW["Q4_K_M_DS"], b_dense=BPW["Q4_K_M_DS"], bw_gpu=GPU["RTX 4090D"], bw_cpu=dram(8, 4800), engine="KTransformers", tok_s=10.303, ctx=500, assume="top-6 routing"),
    dict(id="P052", repo=DS3, kind="static", n_cpu=58, b_exp=BPW["FP8"], b_dense=BPW["FP8"], bw_gpu=GPU["RTX 5090"], bw_cpu=dram(24, 6400), engine="OSDI26-engine", tok_s=21.5, ctx=512, assume="dual-socket"),
    dict(id="P053", repo=K2, kind="static", n_cpu=60, b_exp=BPW["FP8"], b_dense=BPW["FP8"], bw_gpu=GPU["RTX 5090"], bw_cpu=dram(24, 6400), engine="OSDI26-engine", tok_s=22.4, ctx=512, assume="dual-socket"),
    dict(id="P054", repo=DS3, kind="static", n_cpu=58, b_exp=BPW["Q4_K_M_DS"], b_dense=BPW["Q4_K_M_DS"], bw_gpu=GPU["RTX 5090"], bw_cpu=dram(24, 6400), engine="OSDI26-engine", tok_s=28, ctx=2048, assume="dual-socket"),
    dict(id="P056", repo=DS3, kind="static", n_cpu=58, b_exp=BPW["Q4_K_M_DS"], b_dense=BPW["Q4_K_M_DS"], bw_gpu=GPU["RTX 5090"], bw_cpu=dram(24, 6400), engine="KTransformers", tok_s=22, ctx=2048, assume="dual-socket"),
    dict(id="P095", repo=Q235, kind="static", n_cpu=89, b_exp=BPW["IQ6_K_mix"], b_dense=BPW["IQ6_K_mix"], bw_gpu=GPU["RTX 3090"], bw_cpu=dram(12, 6000), engine="ik_llama.cpp", tok_s=17.26, ctx=0, assume="VM"),
    dict(id="P097", repo=Q235, kind="static", n_cpu=89, b_exp=BPW["Q6_K_Q3_235B"], b_dense=BPW["Q6_K_Q3_235B"], bw_gpu=GPU["RTX 3090"], bw_cpu=dram(12, 6000), engine="llama.cpp", tok_s=12.2, ctx=0, assume="VM"),
    dict(id="P098", repo=Q235, kind="static", n_cpu=80, b_exp=BPW["IQ6_K_mix"], b_dense=BPW["IQ6_K_mix"], bw_gpu=GPU["RTX 3090"], bw_cpu=dram(12, 6000), engine="ik_llama.cpp", tok_s=17.75, ctx=0, assume="VM; two GPUs"),
    dict(id="P099", repo=Q235, kind="static", n_cpu=80, b_exp=BPW["Q6_K_Q3_235B"], b_dense=BPW["Q6_K_Q3_235B"], bw_gpu=GPU["RTX 3090"], bw_cpu=dram(12, 6000), engine="llama.cpp", tok_s=17.1, ctx=0, assume="VM; two GPUs"),
    dict(id="P101", repo=DS3, kind="static", n_cpu=58, b_exp=BPW["Q4_K_M_DS"], b_dense=BPW["Q4_K_M_DS"], bw_gpu=GPU["RTX 3070"], bw_cpu=dram(12, 4800), engine="ik_llama.cpp", tok_s=10.3, ctx=1024, assume="approximate server-reported speed"),
    # ---------------- demand fetch over PCIe (Mixtral-offloading, Table 2) ----------------
    # 'W/o LRU cache & pre-loading': every active expert fetched each token (2 per layer).
    # 'Naive offloading': whole MoE layer (8 experts) fetched each token.
]

_MIX = [("A100-80GB-SXM", 4, "P017", "P021", "P025", "P029", 2.265, 2.055, 1.392, 1.246),
        ("RTX 3080 Mobile 16GB", 4, "P018", "P022", "P026", "P030", 1.758, 1.595, 1.059, 0.914),
        ("RTX 3060", 3, "P019", "P023", "P027", None, 1.547, 1.346, 0.919, None),
        ("T4", 3, "P020", "P024", "P028", "P032", 1.168, 1.061, 0.661, 0.58)]
for gpu, gen, i2, i3, n2, n3, t2, t3, tn2, tn3 in _MIX:
    for rid, bits, fetched, tok in ((i2, "HQQ2_g16", 64, t2), (i3, "HQQ3_g64", 64, t3),
                                    (n2, "HQQ2_g16", 256, tn2), (n3, "HQQ3_g64", 256, tn3)):
        if rid is None:
            continue
        ROWS.append(dict(id=rid, repo=M7, kind="fetch", fetched=fetched, b_exp=BPW[bits], b_dense=BPW["HQQ4_g64"],
                         bw_gpu=GPU[gpu], bw_pcie=PCIE[gen], engine="Mixtral-offloading", tok_s=tok, ctx=256,
                         transfers=32 if fetched == 256 else 64,
                         assume="HQQ bpw from group size; A100/3080M assumed PCIe Gen4"))

# Rows deliberately held out of the fit and reported separately: llama.cpp
# with --numa distribute across two sockets (P103/P105/P107/P109/P111/P113,
# P041). The model assumes one NUMA domain; these quantify that limitation.
NUMA_ROWS = [
    dict(id="P055", repo=DS3, kind="static", n_cpu=58, b_exp=BPW["Q4_K_M_DS"], b_dense=BPW["Q4_K_M_DS"], bw_gpu=GPU["RTX 5090"], bw_cpu=dram(24, 6400), engine="ik_llama.cpp", tok_s=14, ctx=2048, assume="dual-socket"),
    dict(id="P103", repo=M22, kind="cpu_only", b_exp=BPW["Q8_0"], b_dense=BPW["Q8_0"], bw_cpu=dram(12, 4800), engine="llama.cpp", tok_s=7.23, ctx=0),
    dict(id="P105", repo=M22, kind="cpu_only", b_exp=BPW["Q8_0"], b_dense=BPW["Q8_0"], bw_cpu=dram(16, 6400), engine="llama.cpp", tok_s=10.58, ctx=0),
    dict(id="P107", repo=M22, kind="cpu_only", b_exp=BPW["Q8_0"], b_dense=BPW["Q8_0"], bw_cpu=dram(24, 4800), engine="llama.cpp", tok_s=10.23, ctx=0),
    dict(id="P109", repo=DS3, kind="cpu_only", b_exp=BPW["Q4_K_S"], b_dense=BPW["Q4_K_S"], bw_cpu=dram(16, 6400), engine="llama.cpp", tok_s=9.79, ctx=0),
    dict(id="P111", repo=DS3, kind="cpu_only", b_exp=BPW["Q4_K_S"], b_dense=BPW["Q4_K_S"], bw_cpu=dram(24, 4800), engine="llama.cpp", tok_s=8.67, ctx=0),
    dict(id="P113", repo=DS3, kind="cpu_only", b_exp=BPW["Q8_0"], b_dense=BPW["Q8_0"], bw_cpu=dram(24, 4800), engine="llama.cpp", tok_s=6.69, ctx=0),
    dict(id="P041", repo=DS3, kind="static", n_cpu=58, b_exp=BPW["Q4_K_M_DS"], b_dense=BPW["Q4_K_M_DS"], bw_gpu=GPU["RTX 4090D"], bw_cpu=dram(16, 4800), engine="llama.cpp", tok_s=4.51, ctx=500),
]

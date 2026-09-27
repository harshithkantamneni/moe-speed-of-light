"""Hardware and quantization tables for the audit rows (phase 4.6).

Extends the conventions of mosl/validation_set.py without editing it:
GPU memory bandwidth from vendor datasheets (GB/s), PCIe x16 peak per
generation (GB/s), host DRAM peak = channels x MT/s x 8 B, and bits per
weight including block scales. Every added entry cites its datasheet or
format definition.

CPU gives the platform facts used to impute a host-DRAM band when a paper
names the CPU but not the DIMM configuration (rules R1-R3 in
data/audit/normalized.md). 'rated' is the vendor's maximum JEDEC speed at
one DIMM per channel; 'lower' is the speed used for the band's lower edge
(the vendor's 2-DIMM-per-channel speed, or one JEDEC grade below 'rated').
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from mosl.validation_set import GPU as _GPU, PCIE as _PCIE, BPW as _BPW, dram  # noqa: E402

GPU = dict(_GPU)
GPU.update({
    "Quadro RTX 6000": 672,          # NVIDIA Quadro RTX 6000 datasheet: 24 GB GDDR6, 384-bit, 672 GB/s
    "RTX 6000 Ada": 960,             # NVIDIA RTX 6000 Ada datasheet: 48 GB GDDR6 ECC, 960 GB/s
    "A100-40GB": 1555,               # NVIDIA A100 datasheet: 40 GB HBM2 (PCIe and SXM4), 1,555 GB/s
    "RTX A6000": 768,                # NVIDIA RTX A6000 datasheet: 48 GB GDDR6, 768 GB/s
    "GTX 1080 Ti": 484,              # NVIDIA GTX 1080 Ti spec: 11 GB GDDR5X, 352-bit, 11 Gbps -> 484 GB/s
    "RTX 4060 Laptop": 256,          # NVIDIA GeForce RTX 4060 Laptop: 8 GB GDDR6, 128-bit, 16 Gbps -> 256 GB/s
    "RTX PRO 6000 Blackwell": 1792,  # NVIDIA RTX PRO 6000 Blackwell datasheet: 96 GB GDDR7, 512-bit, 1,792 GB/s
    "RTX 5070 Ti": 896,              # NVIDIA GeForce RTX 5070 Ti: 16 GB GDDR7, 256-bit, 28 Gbps -> 896 GB/s
    "RTX PRO 4500 Blackwell": 896,   # nvidia.com RTX PRO 4500 Blackwell product page: 32 GB GDDR7, 896 GB/s
    "RX 7600": 288,                  # AMD Radeon RX 7600: 8 GB GDDR6, 128-bit, 18 Gbps -> 288 GB/s
    "Radeon AI PRO R9700": 640,      # AMD Radeon AI PRO R9700: 32 GB GDDR6, 256-bit, 20 Gbps -> 640 GB/s
})

# Host link the GPU itself supports: (PCIe generation, lanes). Used when a paper
# does not state the link: the imputed link is min(GPU, CPU platform).
GPU_LINK = {
    "RTX 4090": (4, 16), "RTX 5090": (5, 16), "RTX A6000": (4, 16), "RTX 3090": (4, 16),
    "RTX PRO 4500 Blackwell": (5, 16),   # product page: PCIe Gen 5
    "Radeon AI PRO R9700": (5, 16),      # AMD spec: PCIe 5.0 x16
    "RX 7600": (4, 8),                   # AMD spec: PCIe 4.0 x8 (Navi 33)
    "RTX 4060 Laptop": (4, 8),
}

PCIE = dict(_PCIE)
PCIE[1] = 4.0  # PCIe 1.x: 2.5 GT/s x 16 lanes x 8b/10b / 8 bits = 4.0 GB/s (Fate states '1.0 x16 (4 GB/s)')


def pcie(gen, lanes=16):
    """Peak host->device GB/s for a link of `lanes` lanes."""
    return PCIE[gen] * lanes / 16


BPW = dict(_BPW)
BPW.update({
    "FP16": 16.0,
    "Q4_0": 18 * 8 / 32,          # ggml block_q4_0: 32 weights in 18 B (fp16 scale + 16 B of nibbles) = 4.5
    "Q2_K": 84 * 8 / 256,         # ggml block_q2_K: 256 weights in 84 B = 2.625
    "NVFP4": 4 + 8 / 16,          # NVFP4: E2M1 4-bit values + one FP8 (E4M3) scale per 16 values (+ per-tensor fp32, ~0)
    "GPTQ_INT4_G128": 4 + 16 / 128,  # symmetric int4 (Marlin) with one fp16 scale per 128 weights = 4.125
    # 'INT4'/'INT8' named without a format: assume the ggml-style block format with one
    # fp16 scale per 32 weights (Q4_0 / Q8_0 equivalents). The row's quant_note gives the band.
    "INT4_UNSPEC": 18 * 8 / 32,
    "INT8_UNSPEC": 34 * 8 / 32,   # ggml block_q8_0: 32 weights in 34 B = 8.5
    # File-size derived, as in mosl/validation_set.py:
    "Q4_K_M_Q3_30B": 18556685824 * 8 / 30.532e9,   # Qwen/Qwen3-30B-A3B-GGUF Q4_K_M file 18,556,685,824 B (HF listing)
    "UD_Q4_K_XL_Q38FN": (10946624 + 49859583136 + 49376141504 + 12087983520) * 8 / 176.94e9,
    # ^ unsloth/Qwen3.8-Flash-Next-GGUF UD-Q4_K_XL, 4 shards (HF listing; the PR's '104 GB' is GiB);
    #   176.94e9 = all parameters per mosl.archs, including the 51.2B-parameter per-layer n-gram table
    "Q6_K_LAGUNA_S21": 107e9 * 8 / 117.56e9,       # 'Q6_K (107 GB)' as reported, over 117.56e9 params (F16 file 235.2 GB)
})

# name: (channels per socket, dram type, rated MT/s, lower MT/s, CPU PCIe generation)
CPU = {
    "Xeon Gold 6126": (6, "DDR4", 2666, 2400, 3),        # Skylake-SP, 12 cores
    "Xeon Platinum 8480+": (8, "DDR5", 4800, 4400, 5),   # Sapphire Rapids, 56 cores; 2DPC DDR5-4400
    "Core i9-14900K": (2, "DDR5", 5600, 4000, 5),        # Raptor Lake; Intel 2DPC 2R spec DDR5-4000 (also DDR4-3200 boards)
    "Xeon Platinum 8452Y": (8, "DDR5", 4800, 4400, 5),   # Sapphire Rapids, 36 cores; 1DPC 4800 / 2DPC 4400
    "Xeon Gold 5220R": (6, "DDR4", 2666, 2400, 3),       # Cascade Lake refresh, 24 cores, PCIe 3.0
    "Core i9-9900X": (4, "DDR4", 2666, 2400, 3),         # Skylake-X HEDT, PCIe 3.0
    "Xeon E5-2650 v4": (4, "DDR4", 2400, 2133, 3),       # Broadwell-EP
    "Xeon Gold 6348": (8, "DDR4", 3200, 2933, 4),        # Ice Lake-SP, 28 cores
    "Xeon Gold 6338N": (8, "DDR4", 2666, 2400, 4),       # Ice Lake-SP NFV SKU, 32 cores, DDR4-2666 (Dell SKU listing)
    "Xeon Platinum 8358P": (8, "DDR4", 3200, 2933, 4),   # Ice Lake-SP, 32 cores
    "Xeon Silver 4310": (8, "DDR4", 2666, 2400, 4),      # Ice Lake-SP, 12 cores
    "Threadripper 7960X": (4, "DDR5", 5200, 4800, 5),    # TRX50, 4 channels
    "Xeon Gold 6459C": (8, "DDR5", 4800, 4400, 5),       # custom cloud SKU; assumed Sapphire Rapids Gold 64xx class
    "Ryzen 9 9950X3D": (2, "DDR5", 5600, 3600, 5),       # AM5; AMD 2DPC spec DDR5-3600
    "Core i9-13900H": (2, "LPDDR5", 6400, 4800, 4),      # 128-bit LPDDR5 bus = 2 x 64-bit channel equivalents
    "Xeon Platinum 8559C": (8, "DDR5", 5600, 4800, 5),   # custom cloud SKU; assumed Emerald Rapids Platinum 85xx class
    "Xeon Gold 6430": (8, "DDR5", 4400, 4000, 5),        # Sapphire Rapids, 32 cores, DDR5-4400
    "Xeon Platinum 8470Q": (8, "DDR5", 4800, 4400, 5),   # Sapphire Rapids, 52 cores
    "Core Ultra 7 270K Plus": (2, "DDR5", 6400, 6000, 5),  # Arrow Lake desktop; row states DDR5-6000
    "Ryzen 9 5900X": (2, "DDR4", 3200, 3200, 4),         # AM4; row states DDR4-3200
}


def cpu_band(name, sockets=1, lo_sockets=None):
    """[lo, hi] GB/s: all channels at the 'lower' speed .. all channels at 'rated'."""
    ch, _, rated, lower, _ = CPU[name]
    lo_s = sockets if lo_sockets is None else lo_sockets
    return [round(dram(ch * lo_s, lower), 1), round(dram(ch * sockets, rated), 1)]

# Phase 4.5, arm D: A10, whole prompt prefilled, response tokens decoded

| model | budget | llama.cpp | helpers, layers | mailbox cache | speed-up | phase 3 speed-up | DFA hit (sim) | pred. cache / llama |
|---|---|---|---|---|---|---|---|---|
| gpt-oss-20b-MXFP4 | 12.5% | 63.8 | 70.4 | 87.5 | 1.37x | 1.52x | 0.471 | 83.1 / 54.5 |
| gpt-oss-20b-MXFP4 | 25.0% | 69.4 | 76.3 | 100.8 | 1.45x | 1.63x | 0.676 | 96.0 / 59.6 |
| gpt-oss-20b-MXFP4 | 50.0% | 80.9 | 90.5 | 116.4 | 1.44x | 1.60x | 0.880 | 116.8 / 73.6 |
| gpt-oss-20b-MXFP4 | all-GPU | 138.1 | | | | | | |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | 12.5% | 68.2 | 77.3 | 96.8 | 1.42x | 1.46x | 0.604 | 96.7 / 63.1 |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | 25.0% | 72.0 | 83.0 | 107.3 | 1.49x | 1.52x | 0.773 | 107.0 / 68.5 |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | 50.0% | 85.4 | 97.6 | 118.0 | 1.38x | 1.41x | 0.908 | 125.3 / 82.8 |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | all-GPU | 146.7 | | | | | | |
| Qwen3-30B-A3B-Instruct-2507-Q8_0 | 12.5% | 48.3 | 51.6 | 69.5 | 1.44x | 1.41x | 0.604 | 67.9 / 44.0 |
| Qwen3-30B-A3B-Instruct-2507-Q8_0 | 25.0% | 51.7 | 55.8 | 79.3 | 1.53x | 1.50x | 0.773 | 77.7 / 47.8 |
| Qwen3-30B-A3B-Instruct-2507-Q8_0 | 50.0% | 61.6 | 66.5 | 88.7 | 1.44x | 1.45x | 0.908 | 91.7 / 58.1 |
| gpt-oss-120b-MXFP4 | 12.5% | 41.3 | 47.5 | 64.4 | 1.56x | 1.57x | – | – / – |
| gpt-oss-120b-MXFP4 | 25.0% | 44.8 | 52.0 | 71.5 | 1.60x | 1.81x | – | – / – |

**H16** (>= 1.30x at 25 % on every model): holds (gpt-oss-20b-MXFP4: 1.45x, Qwen3-30B-A3B-Instruct-2507-Q4_K_M: 1.49x, Qwen3-30B-A3B-Instruct-2507-Q8_0: 1.53x, gpt-oss-120b-MXFP4: 1.60x).
**H17** (median APE <= 10 %): holds — median 5.1 %, max 14.7 %, n = 27.

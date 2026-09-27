# Phase 4.5, arm D: A10, whole prompt prefilled, response tokens decoded

| model | budget | llama.cpp | helpers, layers | mailbox cache | speed-up | phase 3 speed-up | DFA hit (sim) | pred. cache / llama |
|---|---|---|---|---|---|---|---|---|
| gpt-oss-20b-MXFP4 | 12.5% | 58.9 | 69.4 | 86.2 | 1.46x | 1.52x | 0.471 | 83.1 / 54.5 |
| gpt-oss-20b-MXFP4 | 25.0% | 65.4 | 75.2 | 99.3 | 1.52x | 1.63x | 0.676 | 96.0 / 59.6 |
| gpt-oss-20b-MXFP4 | 50.0% | 77.7 | 89.5 | 114.3 | 1.47x | 1.60x | 0.880 | 116.8 / 73.6 |
| gpt-oss-20b-MXFP4 | all-GPU | 135.9 | | | | | | |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | 12.5% | 66.4 | 76.6 | 96.2 | 1.45x | 1.46x | 0.604 | 96.7 / 63.1 |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | 25.0% | 73.1 | 82.6 | 105.9 | 1.45x | 1.52x | 0.773 | 107.0 / 68.5 |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | 50.0% | 85.7 | 97.2 | 116.4 | 1.36x | 1.41x | 0.908 | 125.3 / 82.8 |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | all-GPU | 144.0 | | | | | | |
| Qwen3-30B-A3B-Instruct-2507-Q8_0 | 12.5% | 47.2 | 51.7 | 69.4 | 1.47x | 1.41x | 0.604 | 67.9 / 44.0 |
| Qwen3-30B-A3B-Instruct-2507-Q8_0 | 25.0% | 47.7 | 56.0 | 78.9 | 1.66x | 1.50x | 0.773 | 77.7 / 47.8 |
| Qwen3-30B-A3B-Instruct-2507-Q8_0 | 50.0% | 61.1 | 66.6 | 87.9 | 1.44x | 1.45x | 0.908 | 91.7 / 58.1 |
| gpt-oss-120b-MXFP4 | 12.5% | 40.8 | 46.1 | 63.7 | 1.56x | 1.57x | 0.625 | 61.8 / 35.4 |
| gpt-oss-120b-MXFP4 | 25.0% | 44.5 | 51.6 | 71.1 | 1.60x | 1.81x | 0.781 | 70.1 / 39.2 |

**H16** (>= 1.30x at 25 % on every model): holds (gpt-oss-20b-MXFP4: 1.52x, Qwen3-30B-A3B-Instruct-2507-Q4_K_M: 1.45x, Qwen3-30B-A3B-Instruct-2507-Q8_0: 1.66x, gpt-oss-120b-MXFP4: 1.60x).
**H17** (median APE <= 10 %): holds — median 5.0 %, max 13.2 %, n = 33.

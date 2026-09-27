# Phase 4.5, arm S: A10, whole prompt prefilled, response tokens decoded

| model | budget | llama.cpp | helpers, layers | mailbox cache | speed-up | phase 3 speed-up | DFA hit (sim) | pred. cache / llama |
|---|---|---|---|---|---|---|---|---|
| gpt-oss-20b-MXFP4 | 12.5% | 64.1 | 70.2 | 84.2 | 1.31x | 1.52x | 0.431 | 79.9 / 54.5 |
| gpt-oss-20b-MXFP4 | 25.0% | 69.6 | 75.6 | 99.5 | 1.43x | 1.63x | 0.662 | 95.5 / 59.6 |
| gpt-oss-20b-MXFP4 | 50.0% | 80.6 | 89.9 | 115.9 | 1.44x | 1.60x | 0.887 | 119.0 / 73.6 |
| gpt-oss-20b-MXFP4 | all-GPU | 135.8 | | | | | | |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | 12.5% | 68.7 | 76.2 | 93.3 | 1.36x | 1.46x | 0.575 | 93.0 / 63.1 |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | 25.0% | 74.5 | 82.0 | 104.6 | 1.40x | 1.52x | 0.755 | 103.9 / 68.5 |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | 50.0% | 83.0 | 96.6 | 116.4 | 1.40x | 1.41x | 0.908 | 125.2 / 82.8 |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | all-GPU | 143.8 | | | | | | |
| Qwen3-30B-A3B-Instruct-2507-Q8_0 | 12.5% | 46.4 | 50.7 | 65.5 | 1.41x | 1.41x | 0.575 | 65.4 / 44.0 |
| Qwen3-30B-A3B-Instruct-2507-Q8_0 | 25.0% | 50.1 | 52.7 | 76.0 | 1.52x | 1.50x | 0.755 | 75.7 / 47.8 |
| Qwen3-30B-A3B-Instruct-2507-Q8_0 | 50.0% | 60.1 | 66.9 | 88.2 | 1.47x | 1.45x | 0.908 | 91.7 / 58.1 |
| gpt-oss-120b-MXFP4 | 12.5% | 41.1 | 47.3 | 62.6 | 1.52x | 1.57x | 0.598 | 60.1 / 35.4 |
| gpt-oss-120b-MXFP4 | 25.0% | 43.8 | 51.8 | 70.6 | 1.61x | 1.81x | 0.775 | 69.6 / 39.2 |

**H16** (>= 1.30x at 25 % on every model): holds (gpt-oss-20b-MXFP4: 1.43x, Qwen3-30B-A3B-Instruct-2507-Q4_K_M: 1.40x, Qwen3-30B-A3B-Instruct-2507-Q8_0: 1.52x, gpt-oss-120b-MXFP4: 1.61x).
**H17** (median APE <= 10 %): holds — median 5.0 %, max 15.0 %, n = 33.

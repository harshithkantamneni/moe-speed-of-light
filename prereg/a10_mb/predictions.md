# Phase-3 predictions (mailbox cache), from the calibration run on the profile sequences

| model | T0 ms | a0 us | a1 us | beta us | stock T0 ms | stock s us | calib fit max APE |
|---|---|---|---|---|---|---|---|
| gpt-oss-20b-MXFP4 | 6.77 | -77 | 122 | 369 | 7.22 | 530 | 1.3% |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | 6.52 | -38 | 26 | 166 | 7.03 | 210 | 2.5% |
| Qwen3-30B-A3B-Instruct-2507-Q8_0 | 9.19 | -48 | 41 | 159 | 9.87 | 306 | 1.5% |
| gpt-oss-120b-MXFP4 | 9.74 | -103 | 130 | 247 | 10.85 | 544 | 0.1% |

| model | budget | system | predicted |
|---|---|---|---|
| gpt-oss-20b-MXFP4 | 0.125 | mailbox cache | 83.6 tok/s |
| gpt-oss-20b-MXFP4 | 0.125 | mailbox static layers | 64.9 tok/s |
| gpt-oss-20b-MXFP4 | 0.125 | llama.cpp static layers | 54.5 tok/s |
| gpt-oss-20b-MXFP4 | 0.125 | speed-up (cache / llama.cpp) | 1.53x |
| gpt-oss-20b-MXFP4 | 0.250 | mailbox cache | 96.7 tok/s |
| gpt-oss-20b-MXFP4 | 0.250 | mailbox static layers | 70.6 tok/s |
| gpt-oss-20b-MXFP4 | 0.250 | llama.cpp static layers | 59.6 tok/s |
| gpt-oss-20b-MXFP4 | 0.250 | speed-up (cache / llama.cpp) | 1.62x |
| gpt-oss-20b-MXFP4 | 0.500 | mailbox cache | 117.1 tok/s |
| gpt-oss-20b-MXFP4 | 0.500 | mailbox static layers | 85.4 tok/s |
| gpt-oss-20b-MXFP4 | 0.500 | llama.cpp static layers | 73.6 tok/s |
| gpt-oss-20b-MXFP4 | 0.500 | speed-up (cache / llama.cpp) | 1.59x |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | 0.125 | mailbox cache | 95.9 tok/s |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | 0.125 | mailbox static layers | 73.4 tok/s |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | 0.125 | llama.cpp static layers | 63.1 tok/s |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | 0.125 | speed-up (cache / llama.cpp) | 1.52x |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | 0.250 | mailbox cache | 106.9 tok/s |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | 0.250 | mailbox static layers | 79.3 tok/s |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | 0.250 | llama.cpp static layers | 68.5 tok/s |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | 0.250 | speed-up (cache / llama.cpp) | 1.56x |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | 0.500 | mailbox cache | 125.3 tok/s |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | 0.500 | mailbox static layers | 94.5 tok/s |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | 0.500 | llama.cpp static layers | 82.8 tok/s |
| Qwen3-30B-A3B-Instruct-2507-Q4_K_M | 0.500 | speed-up (cache / llama.cpp) | 1.51x |
| Qwen3-30B-A3B-Instruct-2507-Q8_0 | 0.125 | mailbox cache | 67.4 tok/s |
| Qwen3-30B-A3B-Instruct-2507-Q8_0 | 0.125 | mailbox static layers | 48.0 tok/s |
| Qwen3-30B-A3B-Instruct-2507-Q8_0 | 0.125 | llama.cpp static layers | 44.0 tok/s |
| Qwen3-30B-A3B-Instruct-2507-Q8_0 | 0.125 | speed-up (cache / llama.cpp) | 1.53x |
| Qwen3-30B-A3B-Instruct-2507-Q8_0 | 0.250 | mailbox cache | 77.7 tok/s |
| Qwen3-30B-A3B-Instruct-2507-Q8_0 | 0.250 | mailbox static layers | 52.2 tok/s |
| Qwen3-30B-A3B-Instruct-2507-Q8_0 | 0.250 | llama.cpp static layers | 47.8 tok/s |
| Qwen3-30B-A3B-Instruct-2507-Q8_0 | 0.250 | speed-up (cache / llama.cpp) | 1.62x |
| Qwen3-30B-A3B-Instruct-2507-Q8_0 | 0.500 | mailbox cache | 91.7 tok/s |
| Qwen3-30B-A3B-Instruct-2507-Q8_0 | 0.500 | mailbox static layers | 63.1 tok/s |
| Qwen3-30B-A3B-Instruct-2507-Q8_0 | 0.500 | llama.cpp static layers | 58.1 tok/s |
| Qwen3-30B-A3B-Instruct-2507-Q8_0 | 0.500 | speed-up (cache / llama.cpp) | 1.58x |
| gpt-oss-120b-MXFP4 | 0.125 | mailbox cache | 60.6 tok/s |
| gpt-oss-120b-MXFP4 | 0.125 | mailbox static layers | 43.3 tok/s |
| gpt-oss-120b-MXFP4 | 0.125 | llama.cpp static layers | 35.4 tok/s |
| gpt-oss-120b-MXFP4 | 0.125 | speed-up (cache / llama.cpp) | 1.71x |
| gpt-oss-120b-MXFP4 | 0.250 | mailbox cache | 67.7 tok/s |
| gpt-oss-120b-MXFP4 | 0.250 | mailbox static layers | 47.6 tok/s |
| gpt-oss-120b-MXFP4 | 0.250 | llama.cpp static layers | 39.2 tok/s |
| gpt-oss-120b-MXFP4 | 0.250 | speed-up (cache / llama.cpp) | 1.73x |

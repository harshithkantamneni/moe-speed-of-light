# Pre-registered predictions: NVIDIA A10

B_g=600 GB/s (datasheet), B_c=152.2 GB/s (STREAM Triad max), B_p=31.5 GB/s (PCIe Gen4 x16)

| model | n_cpu_moe | pred tok/s (M4) | pred tok/s (M0) | roofline tok/s | GPU bytes/token |
|---|---|---|---|---|---|
| gpt-oss-20b-mxfp4 | 0 | 89.0 | 87.4 | 199.5 | 2.578 GB |
| gpt-oss-20b-mxfp4 | 3 | 79.8 | 77.2 | 174.0 | 2.419 GB |
| gpt-oss-20b-mxfp4 | 6 | 72.3 | 69.1 | 154.3 | 2.260 GB |
| gpt-oss-20b-mxfp4 | 9 | 66.1 | 62.5 | 138.5 | 2.101 GB |
| gpt-oss-20b-mxfp4 | 12 | 60.8 | 57.1 | 125.7 | 1.942 GB |
| gpt-oss-20b-mxfp4 | 15 | 56.4 | 52.6 | 115.1 | 1.783 GB |
| gpt-oss-20b-mxfp4 | 18 | 52.5 | 48.7 | 106.1 | 1.624 GB |
| gpt-oss-20b-mxfp4 | 21 | 49.2 | 45.3 | 98.4 | 1.465 GB |
| gpt-oss-20b-mxfp4 | 24 | 46.2 | 42.4 | 91.8 | 1.305 GB |
| gpt-oss-20b-mxfp4 | cpu | 33.0 | 31.6 | 59.0 |  |
| qwen3-30b-a3b-q4_k_m | 0 | 119.2 | 117.0 | 267.1 | 1.926 GB |
| qwen3-30b-a3b-q4_k_m | 6 | 96.5 | 97.7 | 226.1 | 1.779 GB |
| qwen3-30b-a3b-q4_k_m | 12 | 81.7 | 84.6 | 198.3 | 1.645 GB |
| qwen3-30b-a3b-q4_k_m | 18 | 70.8 | 74.6 | 176.6 | 1.511 GB |
| qwen3-30b-a3b-q4_k_m | 24 | 62.5 | 66.7 | 159.2 | 1.377 GB |
| qwen3-30b-a3b-q4_k_m | 30 | 55.9 | 60.3 | 144.9 | 1.243 GB |
| qwen3-30b-a3b-q4_k_m | 36 | 50.6 | 55.1 | 133.0 | 1.110 GB |
| qwen3-30b-a3b-q4_k_m | 42 | 46.2 | 50.6 | 122.9 | 0.976 GB |
| qwen3-30b-a3b-q4_k_m | 48 | 42.4 | 46.6 | 113.4 | 0.829 GB |
| qwen3-30b-a3b-q4_k_m | cpu | 34.4 | 42.2 | 79.0 |  |
| qwen3-30b-a3b-q8_0 | 18 | 45.4 | 45.8 | 103.0 | 2.554 GB |
| qwen3-30b-a3b-q8_0 | 24 | 40.6 | 41.3 | 92.4 | 2.313 GB |
| qwen3-30b-a3b-q8_0 | 30 | 36.8 | 37.5 | 83.8 | 2.073 GB |
| qwen3-30b-a3b-q8_0 | 36 | 33.6 | 34.4 | 76.6 | 1.832 GB |
| qwen3-30b-a3b-q8_0 | 42 | 30.9 | 31.7 | 70.6 | 1.591 GB |
| qwen3-30b-a3b-q8_0 | 48 | 28.6 | 29.5 | 65.5 | 1.351 GB |
| gpt-oss-120b-mxfp4 | 25 | 37.7 | 35.0 | 76.2 | 2.274 GB |
| gpt-oss-120b-mxfp4 | 28 | 35.9 | 33.2 | 72.1 | 2.115 GB |
| gpt-oss-120b-mxfp4 | 31 | 34.3 | 31.6 | 68.5 | 1.955 GB |
| gpt-oss-120b-mxfp4 | 34 | 32.9 | 30.2 | 65.2 | 1.796 GB |
| gpt-oss-120b-mxfp4 | 36 | 32.0 | 29.3 | 63.2 | 1.690 GB |

| ncu model | n_cpu_moe | predicted DRAM read bytes / token |
|---|---|---|
| gpt-oss-20b-mxfp4 | 0 | 2.575 GB |
| gpt-oss-20b-mxfp4 | 12 | 1.939 GB |
| qwen3-30b-a3b-q4_k_m | 0 | 1.920 GB |
| qwen3-30b-a3b-q4_k_m | 24 | 1.371 GB |

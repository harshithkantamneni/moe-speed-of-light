# Scored against first-party measurements: NVIDIA A10

- H0 roofline never beaten: PASS
- H1 ncu bytes within ±10%: FAIL
  - gpt-oss-20b-mxfp4_ncmoe0: predicted 2.575 GB, measured 3.034 GB (-15.1%)
  - gpt-oss-20b-mxfp4_ncmoe12: predicted 1.939 GB, measured 2.278 GB (-14.9%)
  - qwen3-30b-a3b-q4_k_m_ncmoe0: predicted 1.920 GB, measured 2.178 GB (-11.9%)
  - qwen3-30b-a3b-q4_k_m_ncmoe24: predicted 1.371 GB, measured 1.575 GB (-12.9%)
- H2 median APE (M4) 27.4% (≤20%: FAIL); MAPE 27.5%, max 37.9%, within 25% 39%; M0 median APE 26.9%
  - gpt-oss-120b-mxfp4: median APE 19.1%
  - gpt-oss-20b-mxfp4: median APE 26.3%
  - qwen3-30b-a3b-q4_k_m: median APE 30.6%
  - qwen3-30b-a3b-q8_0: median APE 37.2%
- H3 gpt-oss-120b-mxfp4: R² 0.9965 (PASS)
- H3 gpt-oss-20b-mxfp4: R² 0.9983 (PASS)
- H3 qwen3-30b-a3b-q4_k_m: R² 0.9987 (PASS)
- H3 qwen3-30b-a3b-q8_0: R² 0.9966 (PASS)
- H4 gpt-oss-120b-mxfp4: intercept 10.12 ms measured vs 15.68 predicted (+55%); slope 437 vs 433 µs/layer (-1%)
- H4 gpt-oss-20b-mxfp4: intercept 7.40 ms measured vs 11.23 predicted (+52%); slope 395 vs 433 µs/layer (+10%)
- H4 qwen3-30b-a3b-q4_k_m: intercept 6.75 ms measured vs 8.43 predicted (+25%); slope 183 vs 315 µs/layer (+72%)
- H4 qwen3-30b-a3b-q8_0: intercept 9.81 ms measured vs 14.28 predicted (+46%); slope 248 vs 431 µs/layer (+73%)
- drift gpt-oss-20b-mxfp4 n=0: -0.4%
- drift qwen3-30b-a3b-q4_k_m n=24: +0.3%
- drift gpt-oss-120b-mxfp4 n=36: -0.9%

| model | n_cpu_moe | measured tok/s | predicted (M4) | error | M0 | roofline |
|---|---|---|---|---|---|---|
| gpt-oss-20b-mxfp4 | 0 | 134.3 | 89.0 | -34% | 87.4 | 199.5 |
| gpt-oss-20b-mxfp4 | 3 | 116.8 | 79.8 | -32% | 77.2 | 174.0 |
| gpt-oss-20b-mxfp4 | 6 | 101.0 | 72.3 | -28% | 69.1 | 154.3 |
| gpt-oss-20b-mxfp4 | 9 | 90.4 | 66.1 | -27% | 62.5 | 138.5 |
| gpt-oss-20b-mxfp4 | 12 | 83.8 | 60.8 | -27% | 57.1 | 125.7 |
| gpt-oss-20b-mxfp4 | 15 | 75.9 | 56.4 | -26% | 52.6 | 115.1 |
| gpt-oss-20b-mxfp4 | 18 | 69.4 | 52.5 | -24% | 48.7 | 106.1 |
| gpt-oss-20b-mxfp4 | 21 | 63.6 | 49.2 | -23% | 45.3 | 98.4 |
| gpt-oss-20b-mxfp4 | 24 | 58.6 | 46.2 | -21% | 42.4 | 91.8 |
| gpt-oss-20b-mxfp4 | cpu | 39.1 | 33.0 | -15% | 31.6 | 59.0 |
| qwen3-30b-a3b-q4_k_m | 0 | 145.6 | 119.2 | -18% | 117.0 | 267.1 |
| qwen3-30b-a3b-q4_k_m | 6 | 127.4 | 96.5 | -24% | 97.7 | 226.1 |
| qwen3-30b-a3b-q4_k_m | 12 | 110.5 | 81.7 | -26% | 84.6 | 198.3 |
| qwen3-30b-a3b-q4_k_m | 18 | 101.3 | 70.8 | -30% | 74.6 | 176.6 |
| qwen3-30b-a3b-q4_k_m | 24 | 90.7 | 62.5 | -31% | 66.7 | 159.2 |
| qwen3-30b-a3b-q4_k_m | 30 | 82.3 | 55.9 | -32% | 60.3 | 144.9 |
| qwen3-30b-a3b-q4_k_m | 36 | 75.2 | 50.6 | -33% | 55.1 | 133.0 |
| qwen3-30b-a3b-q4_k_m | 42 | 69.2 | 46.2 | -33% | 50.6 | 122.9 |
| qwen3-30b-a3b-q4_k_m | 48 | 63.8 | 42.4 | -34% | 46.6 | 113.4 |
| qwen3-30b-a3b-q4_k_m | cpu | 43.1 | 34.4 | -20% | 42.2 | 79.0 |
| qwen3-30b-a3b-q8_0 | 18 | 69.2 | 45.4 | -34% | 45.8 | 103.0 |
| qwen3-30b-a3b-q8_0 | 24 | 63.4 | 40.6 | -36% | 41.3 | 92.4 |
| qwen3-30b-a3b-q8_0 | 30 | 58.6 | 36.8 | -37% | 37.5 | 83.8 |
| qwen3-30b-a3b-q8_0 | 36 | 53.5 | 33.6 | -37% | 34.4 | 76.6 |
| qwen3-30b-a3b-q8_0 | 42 | 49.7 | 30.9 | -38% | 31.7 | 70.6 |
| qwen3-30b-a3b-q8_0 | 48 | 45.6 | 28.6 | -37% | 29.5 | 65.5 |
| gpt-oss-120b-mxfp4 | 25 | 47.3 | 37.7 | -20% | 35.0 | 76.2 |
| gpt-oss-120b-mxfp4 | 28 | 45.0 | 35.9 | -20% | 33.2 | 72.1 |
| gpt-oss-120b-mxfp4 | 31 | 42.5 | 34.3 | -19% | 31.6 | 68.5 |
| gpt-oss-120b-mxfp4 | 34 | 39.9 | 32.9 | -18% | 30.2 | 65.2 |
| gpt-oss-120b-mxfp4 | 36 | 38.7 | 32.0 | -17% | 29.3 | 63.2 |

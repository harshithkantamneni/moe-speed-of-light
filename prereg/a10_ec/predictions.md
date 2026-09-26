# Pre-registered expert-cache predictions (A10)

calibration from the static sweep: h = 42.6 us per layer hand-off, Bc = 108.7 GB/s

## gpt-oss-20b-mxfp4

T_dense 3.75 ms, a 38.1 us, b 121.9 us per expert; 3072 decode steps over test sequences [4, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 25, 26, 27, 31]

| budget | configuration | slots / n_cpu_moe | predicted hit rate | admits/step | predicted tok/s |
|---|---|---|---|---|---|
| 0.125 | llama.cpp static layers | 21 |  |  | 63.7 |
| 0.125 | ec static | 4 | 0.277 | 0.0 | 75.2 |
| 0.125 | ec lru | 4 | 0.334 | 32.0 | 57.6 |
| 0.125 | ec dfa | 4 | 0.467 | 4.3 | 89.0 |
| 0.125 | ec dfa serial | 4 | 0.467 | 4.3 | 78.5 |
| 0.250 | llama.cpp static layers | 18 |  |  | 68.9 |
| 0.250 | ec static | 8 | 0.454 | 0.0 | 86.7 |
| 0.250 | ec lru | 8 | 0.634 | 27.8 | 66.2 |
| 0.250 | ec dfa | 8 | 0.677 | 6.6 | 103.9 |
| 0.250 | ec dfa serial | 8 | 0.677 | 6.6 | 90.5 |
| 0.500 | llama.cpp static layers | 12 |  |  | 82.4 |
| 0.500 | ec static | 16 | 0.708 | 0.0 | 106.5 |
| 0.500 | ec lru | 16 | 0.879 | 9.6 | 110.8 |
| 0.500 | ec dfa | 16 | 0.884 | 5.8 | 116.3 |
| 0.500 | ec dfa serial | 16 | 0.884 | 5.8 | 106.5 |

## qwen3-30b-a3b-q4_k_m

T_dense 2.91 ms, a 10.0 us, b 26.3 us per expert; 3072 decode steps over test sequences [4, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 25, 26, 27, 32]

| budget | configuration | slots / n_cpu_moe | predicted hit rate | admits/step | predicted tok/s |
|---|---|---|---|---|---|
| 0.125 | llama.cpp static layers | 42 |  |  | 69.3 |
| 0.125 | ec static | 16 | 0.250 | 0.0 | 79.2 |
| 0.125 | ec lru | 16 | 0.568 | 132.4 | 65.1 |
| 0.125 | ec dfa | 16 | 0.605 | 32.9 | 103.6 |
| 0.125 | ec dfa serial | 16 | 0.605 | 32.9 | 88.5 |
| 0.250 | llama.cpp static layers | 36 |  |  | 75.0 |
| 0.250 | ec static | 32 | 0.416 | 0.0 | 89.9 |
| 0.250 | ec lru | 32 | 0.781 | 69.3 | 98.4 |
| 0.250 | ec dfa | 32 | 0.791 | 39.9 | 111.8 |
| 0.250 | ec dfa serial | 32 | 0.791 | 39.9 | 98.4 |
| 0.500 | llama.cpp static layers | 24 |  |  | 89.8 |
| 0.500 | ec static | 64 | 0.697 | 0.0 | 110.3 |
| 0.500 | ec lru | 64 | 0.918 | 26.3 | 113.7 |
| 0.500 | ec dfa | 64 | 0.939 | 19.1 | 113.3 |
| 0.500 | ec dfa serial | 64 | 0.939 | 19.1 | 108.0 |

## qwen3-30b-a3b-q8_0

T_dense 4.03 ms, a 15.0 us, b 46.1 us per expert; 3072 decode steps over test sequences [4, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 25, 26, 27, 32]

| budget | configuration | slots / n_cpu_moe | predicted hit rate | admits/step | predicted tok/s |
|---|---|---|---|---|---|
| 0.125 | llama.cpp static layers | 42 |  |  | 49.4 |
| 0.125 | ec static | 16 | 0.250 | 0.0 | 51.3 |
| 0.125 | ec lru | 16 | 0.568 | 132.4 | 37.9 |
| 0.125 | ec dfa | 16 | 0.605 | 32.9 | 71.5 |
| 0.125 | ec dfa serial | 16 | 0.605 | 32.9 | 60.2 |
| 0.250 | llama.cpp static layers | 36 |  |  | 53.3 |
| 0.250 | ec static | 32 | 0.416 | 0.0 | 59.6 |
| 0.250 | ec lru | 32 | 0.781 | 69.3 | 63.1 |
| 0.250 | ec dfa | 32 | 0.791 | 39.9 | 79.4 |
| 0.250 | ec dfa serial | 32 | 0.791 | 39.9 | 69.0 |
| 0.500 | llama.cpp static layers | 24 |  |  | 63.4 |
| 0.500 | ec static | 64 | 0.697 | 0.0 | 77.9 |
| 0.500 | ec lru | 64 | 0.918 | 26.3 | 82.1 |
| 0.500 | ec dfa | 64 | 0.939 | 19.1 | 82.6 |
| 0.500 | ec dfa serial | 64 | 0.939 | 19.1 | 78.2 |


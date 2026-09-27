# Phase 4.4: gpt-oss-120b NLL anomaly (35 traced conversations, nats per token)

| model | user (vLLM V0) | response V0 | response V1 (own analysis) | response V2 (reasoning low) | response, fp32 collector | mean abs vLLM-collector diff |
|---|---|---|---|---|---|---|
| gpt-oss-20b | 4.819 | 2.134 | 2.025 | 2.061 | 2.132 | 0.064 |
| gpt-oss-120b | 8.959 | 6.404 | 3.004 | 6.563 | 6.394 | 0.544 |

Ratios 120b / 20b: resp_V0 3.000, user_V0 1.859, resp_V1_same 1.483, resp_V2_same 3.184, resp_V1_vs_20b_V0 1.408, resp_V2_vs_20b_V0 3.075, collector_resp 2.999
Affected conversations (collector response NLL of 120b > 2x 20b's): 30; mean abs vLLM-collector difference on them: 0.616 nats.

**Class: 3 format**

# Reference check of the audit models' real traces (job 049)

Stock transformers 5.17 (`from_pretrained`, whole model, fp32, Grace CPU) against the layer-streamed collector (fp32, GPU) on the first conversations of each D pack.

| model | conversations | tokens | (token, layer) pairs with any differing expert | collector NLL - reference NLL (mean abs) | reference NLL (answer) | vLLM NLL (answer) |
|---|---|---|---|---|---|---|
| mixtral-8x7b | 2 | 2687 | 0.000 % | 6.40e-06 | 0.585 | 0.579 |
| phi3.5-moe | 2 | 2726 | 0.008 % | 2.12e-01 | 0.458 | 0.459 |
| qwen2-57b | 1 | 1304 | 0.345 % | 7.34e-05 | 0.017 | 0.026 |

Phi-3.5-MoE: the collector's NLL check omitted the final LayerNorm bias and the lm_head bias (fixed in `mosl/collect.py` after this run; routing is unaffected, the packs' `nll` field for Phi-3.5-MoE is wrong and is not used by any result). vLLM numbers are from the FP8 generation engine.

# Job 084: all in VRAM and AIME-25 routing traces — void, rerun as 084b and 084c

The predictions are in the header of `jobs/084_vram_traces@vast.sh` (gpu branch commit 995681b, pushed before launch).

**Host:** RTX PRO 6000 Blackwell Workstation (96 GB) + Core Ultra 9 285K, Vast offer 37184228.

**Why the job is void.**
- **The card was throttled.**
  - gpt-oss-120b with every weight in VRAM ran at 93.8 / 92.1 tok/s (launches 1 / 2). Job 075 measured 261.4 on the
    same card model under the same protocol.
  - Its device read was 1,304 GB/s, against 1,641 GB/s on job 075's card.
  - `nvidia-smi -q` shows application clocks set to 2,617 MHz and a large software power-capping counter.
  - None of its speeds are a ceiling, so none of predictions 1–3 is scored.
- **The Qwen3 trace failed.**
  - The corpus step started llama-server through `eval ... &`, so `kill $!` killed the subshell and not the server.
  - The gpt-oss tokenising server kept the port, and the Qwen3 prompts were tokenised by it.
  - llama-ec-bench then refused token 200006 (a gpt-oss id).
- **The gpt-oss trace was lost.**
  - The full lookahead record (17.7 MB) exceeded the 8 MB per-file limit of the result channel.
  - The generated text (30 × 257 tokens, all on the GPU) and its teacher-forced loss (0.198 nats/token) survived.

**Within this card only** (not scored): ours with 128 slots per layer ÷ stock all in VRAM, launch 1:
- gpt-oss: 92.95 / 93.84 = 0.99;
- Qwen3: 67.82 / 74.35 = 0.91.

**Reruns, predictions carried over unchanged:**
- **084b:** another RTX PRO 6000.
  - It stops at once if device read is below 1,500 GB/s, or if gpt-oss all in VRAM runs below 200 tok/s.
  - The server is started without `eval`.
  - Traces keep only the router's choices as uint8.
- **084c:** the gpt-oss trace on job 084's own greedy text, on an RTX 5090 host.

## Reruns 084b and 084c: outcome of job 084's predictions

Statistics: `scripts/vram_stats.py`; numbers in `prereg/vram_084.json` and `prereg/speed_limit_084.json`.

**Job 084b.**
- **Host:** RTX PRO 6000 Workstation + Core Ultra 9 285K, Vast offer 49074400. Device read 1,637 GB/s; both gates
  passed.
- **Stock llama.cpp, every weight in VRAM (launches 1 / 2):** gpt-oss-120b 259.0 / 255.3 tok/s; Qwen3-30B-A3B BF16
  178.3 / 177.8 tok/s.
- **Ours with 128 slots per layer ÷ stock (launch 1):**
  - gpt-oss 0.997 [0.991, 1.004];
  - Qwen3 1.047 [1.043, 1.052].
  - The hit rate is 1.0 in both.

**Traces.**
- Qwen3 from 084b, gpt-oss from 084c (on 084's text).
- Teacher-forced loss on the models' own greedy text: 0.090 (Qwen3) and 0.206 (gpt-oss) nats per token.

**Speed limit** (headline machine's probe: highest host read 87.5 GB/s; RTX 5090 datasheet 1,792 GB/s):

| Cell | Reads/token: optimum / best online / deployed (sim.) | Speed limit (tok/s) | Ours (Table 1) | Ours, % of limit | Sim. hit vs engine |
|---|---|---|---|---|---|
| gpt-oss 11% | 39.0 / 59.4 / 70.8 | 169.2 | 69.9 | 41% | 0.566 vs 0.580 |
| gpt-oss 25% | 15.5 / 28.3 / 40.0 | 426.9 | 109.2 | 26% | 0.785 vs 0.793 |
| gpt-oss 40% | 6.8 / 14.3 / 21.4 | 518.8 | 152.5 | 29% | 0.892 vs 0.892 |
| Qwen3 12.5% | 100.4 / 166.2 / 196.6 | 92.3 | 40.0 | 43% | 0.533 vs 0.586 |
| Qwen3 25% | 44.3 / 84.2 / 115.1 | 209.1 | 63.1 | 30% | 0.750 vs 0.775 |
| Qwen3 43.75% | 13.8 / 28.8 / 47.0 | 307.7 | 108.2 | 35% | 0.908 vs 0.915 |

**Predictions.**
1. **Held.** Qwen3 BF16 all in VRAM runs at 177.8 tok/s (120–190).
2. **Held.** gpt-oss all in VRAM runs at 255.3 tok/s, −2.3% from job 075's 261.4.
3. **Held.** Ours with every expert in its slots is within 5% of stock all in VRAM on both models (0.997, 1.047).
4. **Half held.**
   - Ours runs at 26–43% of the speed limit at every budget (20–50% predicted).
   - The simulated hit rate is within 5 points of the engine's at five of six budgets. It misses Qwen3 12.5% by 5.3
     points. The simulator models the deployed policy without FETCH, whose immediate admissions raise the engine's
     hit rate most where it fetches most (76 fetches per token at Qwen3 12.5%).

**Where the seconds go** (`scripts/gap_listingb.py`, `prereg/gap_listingb.json`):
- Missing foresight is the largest or tied-largest step in all six cells: 17–27% of each token.
- Serialised host reads cost 13–22%, the policy and read paths 9–17%, host work 2–7%, and GPU kernels below datasheet
  rate (net of the overlap achieved) 6–13%.

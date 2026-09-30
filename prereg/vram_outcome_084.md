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

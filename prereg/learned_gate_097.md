# Gate for job 097 (a realisable foresight mechanism in the engine)

Written 5 October 2026, 22:50 UTC, before the cross-corpus replay (`scripts/learned_policy6.py`) and before job 096's
results. Job 097 is launched only if all three conditions hold; otherwise no further GPU money is spent on it and the
paper reports the offline result as a negative or partial one.

## The mechanism

Fetch-on-admit (job 096's `foa`: an admitted miss is copied into its slot in the step that needs it, one read) with
the admission and victim order taken from a learned reuse predictor instead of the decayed-frequency score. The
predictor is a logistic regression on features the engine has when step t's fetch plan runs: decayed request counts
at half-lives 1, 4, 16, 64, 256 steps (including step t's request when the expert is requested at t), steps since its
last request before t, a requested-at-t flag, a same-layer transition score from step t-1's experts, the expert's
training popularity, and (variant `cross`) a cross-layer transition score from step t-1's experts at every layer.
Label: requested within the next H = C/k steps. A miss is fetched into the slot of the non-requested resident with the
lowest predicted reuse if its own prediction is higher; otherwise it runs on the CPU.

## The three conditions

1. **Offline, cross-corpus, at the four host-bound cells** (gpt-oss-120b C = 14 and 32, Qwen3-30B-A3B C = 16 and
   32): trained on the model's mixed-domain trace (chat, code, math, multilingual) and replayed on the AIME-25 routing
   the engine runs, the mechanism closes **at least 30% of the read gap between the deployed online policy
   (decayed frequency, serve-then-copy) and MIN with bypass** at every one of the four cells. (This is the gate as
   stated to the author: "it must close >= 30% of the read gap between our online policy and MIN in simulation".)
2. **The predictor itself must matter**: beyond single-read admission with the decayed-frequency score (`dfa-fetch`),
   the learned order closes at least 15% of the remaining dfa-fetch-to-MIN read gap at three of the four cells.
   (Recorded for honesty: on the held-out half of the mixed-domain trace, with features that also see step t's routing
   at every layer, it closed 23-25% at gpt-oss C = 14-16, 18-21% at C = 32 and 34% at Qwen3 C = 16, so the stricter
   ">= 30% of dfa-fetch-to-MIN" reading of the gate fails on gpt-oss.)
3. **In the engine, single-read admission must not lose**: in job 096, `foa` / `base` has a lower 95% bound of at least
   0.98 at three of the four host-bound cells on each host. If in-step fetches cost more than the second read they
   save, a better admission order cannot be measured through them.

## If the gate passes

Job 097 on one or two RTX 5090 hosts (Ryzen 9 9950X class), the six cells, configurations `base`, `foa`, `learned`
(foa with the learned order), `learnedx` (with cross-layer features, if condition 1 holds for it and its host cost is
under 0.3 ms per token), `fetch` (oracle) and `both3p` (oracle, the best measured state), seeded random order,
predictions in the job header from the replay, committed and pushed on `gpu` before launch. Budget at most $15.

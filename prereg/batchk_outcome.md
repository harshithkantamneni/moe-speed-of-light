# Batch-K verification on routing traces: what speculative decoding's batching buys an expert cache

Trace study, no engineering: `scripts/batchk_trace.py` (numbers in `prereg/batchk_trace.json`, figure
`scripts/fig_batchk.py` → `paper/figs/batchk.pdf`). Speculative decoding verifies several draft tokens in one target
forward pass, so at every MoE layer the expert sets of K positions are requested together and a missed expert is read
from host memory once for all of them. This converts foresight (the next tokens' routing) into batching. The question
before building anything: how many host reads per committed token does that save, and what do the rejected drafts cost?

**Setup.**
- Traces: the nine S-arm packs of `data/traces_manifest.json` (the models' own sampled text; OLMoE 64/8/16,
  gpt-oss-20b 32/4/24, Qwen3-30B-A3B 128/8/48, gpt-oss-120b 128/4/36, Mixtral-8x7B 8/2/32, DeepSeek-V2-Lite 64/6/26,
  Qwen1.5-MoE 60/4/24, Qwen2-57B 64/8/28, Phi-3.5-MoE 16/2/32 as E/k/L), response tokens back to back, the cache
  carried across conversations. **Subsampled to the first 40,000 response tokens of each trace** (gpt-oss-120b has
  32,601) to keep the run at 16 minutes; domains are interleaved in corpus order, so the prefix mixes all four.
- Budgets: C = E/8, E/4, 3E/8 slots per layer (floor; Mixtral 2, 3, 4). K ∈ {1, 2, 4, 8}, α ∈ {0.6, 0.8, 0.9}, seed 0.
- Policies, both event-atomic (no member of the step's union is evicted during the step; when every resident is in the
  union, a miss is bypassed): Belady's MIN with bypass at step granularity, and the decayed-frequency online policy
  (half-life 16 committed tokens, κ ∈ {0, the model's κ}, fewer reads kept). One host read per missed expert of the
  union (copied into a slot or run on the CPU: the accounting of the speed limit). At K = 1 both reproduce
  `foresight.py`'s `opt` and `dfa-fetch` rows exactly.
- Metric: host reads per committed token relative to K = 1 (bytes per token = reads × expert bytes; BF16 sizes from
  `data/shapes.json` and the served GGUF sizes for gpt-oss and Qwen3 are in the JSON). Also the union size per step
  as a fraction of K·k and of (positions in the batch)·k.

**Convention (stated).** K is the number of target-model positions in one verification forward pass: the token the
previous step committed (whose forward pass is due) plus K − 1 drafts; K = 1 is plain decoding (papers that count K as
draft tokens would call our K their K + 1). Drafts are accepted independently with probability α until the first
rejection: A = min(K − 1, G), G ~ Geometric with P(G = g) = α^g (1 − α). A step commits A + 1 tokens (the accepted drafts
plus the token resampled at the first rejection, or the bonus token), so the mean committed per step is
(1 − α^K)/(1 − α), i.e. Leviathan et al.'s (1 − α^(γ+1))/(1 − α) with γ = K − 1 drafts; the simulated means match it
(K = 8: 2.45 / 4.12 / 5.64 at α = 0.6 / 0.8 / 0.9 against 2.46 / 4.16 / 5.70, the shortfall being steps cut at
conversation ends). Every trace position is committed exactly once, so "per accepted token" = per committed token =
total reads / trace length. The committed positions carry the trace's true routing. The K − 1 − A rejected drafts are
wrong tokens whose routing the trace does not hold; three variants bracket it:
- **random** — each rejected draft routes like a token drawn from a uniformly random other position of the trace (the
  same position in every layer; no correlation with the true token). The requested pessimistic end.
- **same** — each rejected draft routes like the true token at the position it occupies (full correlation: the reads
  only come earlier, and the next step re-requests them). An intermediate reference added here.
- **free** — rejected drafts cost nothing (as if the verifier knew the outcome in advance). The requested optimistic end.

## Summary table

Median across the nine models of the host-read saving per committed token, 1 − reads(K)/reads(K = 1), in %, with the
inter-quartile range across models in brackets. Positive = fewer reads than plain decoding; negative = more.

| budget | α | K | online, free | online, same | online, random | Belady, free | Belady, same | Belady, random |
|---|---|---|---|---|---|---|---|---|
| E/8 | 0.6 | 4 | +13 [+13, +18] | −38 [−41, −31] | −76 [−81, −73] | −0 [−4, +3] | −65 [−73, −60] | −125 [−129, −113] |
| E/8 | 0.6 | 8 | +17 [+16, +22] | −100 [−107, −82] | −175 [−185, −172] | +3 [−3, +7] | −148 [−181, −133] | −268 [−274, −258] |
| E/8 | 0.8 | 4 | +17 [+17, +23] | −5 [−6, +2] | −19 [−25, −15] | +1 [−4, +5] | −25 [−36, −21] | −50 [−55, −50] |
| E/8 | 0.8 | 8 | +27 [+25, +33] | −19 [−23, −8] | −53 [−64, −47] | +11 [−2, +16] | −52 [−72, −38] | −98 [−114, −94] |
| E/8 | 0.9 | 4 | +19 [+18, +25] | +10 [+9, +16] | +6 [−5, +7] | +2 [−4, +7] | −9 [−18, −5] | −20 [−27, −18] |
| E/8 | 0.9 | 8 | +32 [+29, +38] | +12 [+8, +21] | −0 [−14, +5] | +15 [−0, +22] | −12 [−37, −2] | −29 [−50, −26] |
| E/4 | 0.6 | 4 | +13 [+9, +14] | −35 [−37, −32] | −106 [−124, −86] | −4 [−6, −2] | −68 [−69, −36] | −200 [−232, −129] |
| E/4 | 0.6 | 8 | +16 [+11, +18] | −96 [−105, −89] | −233 [−273, −208] | −4 [−6, −2] | −185 [−202, −128] | −435 [−502, −350] |
| E/4 | 0.8 | 4 | +17 [+11, +18] | −4 [−7, −2] | −35 [−40, −29] | −6 [−7, −3] | −33 [−34, −22] | −90 [−105, −60] |
| E/4 | 0.8 | 8 | +26 [+17, +27] | −18 [−25, −17] | −80 [−96, −71] | −4 [−6, −1] | −73 [−82, −55] | −184 [−213, −154] |
| E/4 | 0.9 | 4 | +18 [+12, +20] | +9 [+5, +12] | −6 [−7, −4] | −6 [−8, −3] | −17 [−20, −11] | −44 [−52, −32] |
| E/4 | 0.9 | 8 | +31 [+20, +32] | +12 [+2, +13] | −20 [−22, −15] | −2 [−8, +1] | −31 [−36, −27] | −79 [−94, −75] |
| 3E/8 | 0.6 | 4 | +12 [+7, +12] | −35 [−36, −23] | −137 [−187, −98] | −4 [−5, −3] | −36 [−43, −20] | −242 [−320, −138] |
| 3E/8 | 0.6 | 8 | +15 [+9, +15] | −98 [−100, −79] | −294 [−402, −232] | −5 [−6, −2] | −164 [−172, −117] | −592 [−769, −390] |
| 3E/8 | 0.8 | 4 | +15 [+9, +15] | −4 [−5, −3] | −52 [−73, −36] | −5 [−6, −5] | −21 [−23, −14] | −112 [−146, −64] |
| 3E/8 | 0.8 | 8 | +22 [+15, +23] | −20 [−21, −15] | −112 [−160, −87] | −6 [−9, −4] | −74 [−81, −52] | −260 [−338, −174] |
| 3E/8 | 0.9 | 4 | +16 [+10, +17] | +8 [+3, +9] | −16 [−25, −10] | −5 [−7, −5] | −13 [−14, −9] | −58 [−72, −34] |
| 3E/8 | 0.9 | 8 | +26 [+18, +28] | +8 [+5, +9] | −36 [−57, −25] | −7 [−11, −4] | −38 [−43, −14] | −122 [−158, −85] |

## Reading

- **The most batching can buy an online cache is a quarter to a third of its host reads.** With rejected drafts free,
  the decayed-frequency policy reads 12–19% fewer experts per committed token at K = 4 and 15–32% fewer at K = 8
  (medians; the gain grows with α and shrinks slowly with the budget: 32 / 31 / 26% at α = 0.9, K = 8, for E/8, E/4,
  3E/8). Per model at C = E/4, K = 8, α = 0.9 the ratio runs from 0.53 (Mixtral, C = 3 with k = 2) to 0.92
  (gpt-oss-120b, C = 32 with k = 4); the gain is largest where the cache turns over fastest (small C/k: 1.5 for
  Mixtral, 2 for OLMoE, gpt-oss-20b, Qwen2 and Phi at 0.66–0.69, 8 for gpt-oss-120b), the same variable that sets the
  foresight window W50 in `scripts/fig_foresight.py`. In bytes at C = E/4 (online, K = 1 → K = 8 at α = 0.9, free): Qwen3-30B-A3B 0.65 → 0.53 GB per token at BF16
  (0.20 → 0.16 GB at Q4_K_M), gpt-oss-120b 0.34 → 0.32 GB at MXFP4, Mixtral 11.4 → 6.1 GB at BF16.
- **It is foresight by another name, and it stops at Belady.** Measured as the fraction of the online-to-optimum gap
  (both at K = 1) closed, batching K = 8 at α = 0.9 closes 82% on OLMoE, 42% on Qwen3, 17% on gpt-oss-120b and 133% on
  Mixtral — what a foresight window of W ≈ 2–4 decode steps closes in `prereg/foresight/foresight_S.json` (the same
  ordering across models: the gap closes faster where C/k is small). Belady with the rejected drafts free stays within
  −7 to +15% of K = 1: a policy that already knows the future gains nothing from seeing it in batches (the small losses
  at E/4 and 3E/8 are the coarser step-level next-use information and the event-atomic constraint; Mixtral exceeds the
  K = 1 optimum because a union member is read once for every position that needs it in the step).
- **Rejected drafts eat the gain unless their routing is correlated with the true token's.** If a rejected draft
  routes like the true token at its position ('same': the reads come one step early and are re-requested), the online
  saving at α = 0.9 shrinks to 8–12% and at α = 0.8 turns into a loss of 4–5% (K = 4) or 18–20% (K = 8); at α = 0.6 the
  reads double at K = 8. With uncorrelated routing ('random') batching costs reads at every α ≤ 0.8 (K = 8: 1.5–3.9×
  the reads) and only reaches break-even at α = 0.9 on the smallest budget (+6% at K = 4, 0% at K = 8). The reason is
  capacity, not policy: a K-wide batch requests, per layer, 79–93% of K·k distinct experts at K = 2 and 42–73% at K = 8
  (α = 0.9, across models), several times C at E/8, so most of the union cannot be resident and the same experts are
  read again at the next step.
- **What an engine must do to collect the gain.** (i) Keep the verification batch's expert demand near the committed
  positions' demand: the free variant's union is 32–53% of K·k at K = 8 (45–74% of the batch's own k per position),
  the random variant's 42–73%. That means either high acceptance (α ≥ 0.9 with K ≤ 4 already gives +18–19% online,
  +6–10% even with rejected drafts costing), a draft-side filter that requests experts only for drafts likely to be
  accepted, or serving rejected drafts' misses on the CPU without admitting them. (ii) Serve every position of the
  batch from one host read per missed expert: the within-batch deduplication is the whole mechanism; an engine that
  reads a missed expert once per token gets none of it. (iii) Expect nothing from batching on top of a policy with
  real foresight; the budget for speculative-verification work in the cache is the online policy's foresight term of
  the decomposition (17–27% of a token in the Table 1 cells), not more.
- Two caveats. The step-granularity Belady is not proven optimal for set requests with bypass (ties in next use at
  step level, admission order within a step); it is below the online policy's reads in every cell and serves only as
  the foresight reference. The prefix subsample (40,000 tokens) makes the K = 1 absolute reads differ from the
  full-trace numbers of `foresight_S.json` by a few percent; the ratios are what this study reports.

**Assumptions made, in one place.** K counts positions per forward pass with K = 1 plain decoding; acceptance is
i.i.d. per draft with the same α at every position and independent of routing; rejected drafts' routing is unknown
and bracketed by the three variants (random = the requested assumption, free = the requested optimistic variant, same
= an added intermediate); the committed tokens' routing is the trace's; the online policy's clock is the committed-token
position and each requesting position of a batch adds 1 to an expert's score; steps do not straddle conversations;
the first 40,000 response tokens of each trace; bytes use BF16 expert sizes (and the served GGUF sizes where known).

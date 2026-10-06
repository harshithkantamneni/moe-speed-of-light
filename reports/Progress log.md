# Progress log: option C measurement study

*Started Monday 28 September 2026. Newest entries first. Numbers link to result folders on the `gpu` branch
(`results/<job>/`).*

## 6 October (afternoon, unattended): review round 7 (5/10 and 5/10), number check 7, jobs 100f and 101

**Status for Harshith (read this first).** Round 7: MLSys PC **5/10**, professor **5/10** (would take the student;
"reject as it stands, likely accept after a focused rewrite plus the crossover experiment"). Every artifact check
passed in both reviews (15 and 16 claims, scripts regenerate macros byte for byte). Both still rank clarity first
(2/5), then: the bound's tightness is unknown; the price of foresight is measured against a weak baseline; the
"model predicts which way pays" claim rests on one machine (both losses on Pf). Number check 7: 18 defects, the
largest that job 100's new machines were left out of the accounting ranges (fixed: both-together range is now 14-60%).
Reviewer flag for you: the PDF names the author on the title page and uses the plain article class (template and
anonymisation are your call).

- **Fixed after round 7:** abstract cut to ~7 numbers and plain words; finding (3) states its baseline; a Terms
  paragraph in Section 2; the model paragraph defines its symbols, says G is leave-one-host-out, says the sign test is
  weak (2 losses of 39, both on one machine; "always copy" and the one-path model get 37), and that the model cannot
  resolve the single read (predicted within 2%, measured 0.90-1.03); window policy as a 3-step list; worked example
  to one decimal (4.5 + 3.9 + 43.2 ~ 51.5) with its own admission ratio (5.9x); "the rest" no longer claims path-
  splitting slack (the per-path caps never bind; it is the probe's combined vs highest rate, 1-7% of the gap);
  deployed-path paragraph corrected (the fetch table still sends some misses over the link; 4-token and half-right
  windows lose up to 6%; replay excess 7-33% with the deployed state within 2.4%, cause not isolated); launch-to-launch
  shifts up to 0.03 exceed within-launch intervals, said so; MIN prefetched vs usual ways restricted to O4/O5
  (1.18-1.43x); Fig. 1 simplified (learned order and prefetched Belady moved out); accounting table gains the new
  machines and range notation.
- **Job 100f** (9950X behind an x8 link, ratio 0.54): in-step copy 1.07x (Pe/Pj 1.35/1.27, prediction 9 held);
  probe-only model over-predicts admit-every-miss and windows by up to 20% (probe read the link at 22, the engine's
  copies ran near 27). Outcome: `prereg/window_outcome_100.md` (41 held, 122 point, 27 failed, 26 untested).
- **Job 101** (predictions in gpu commit 8b6c67a before launch): Pf relaunched and a 9800X3D behind a slower link.

## 6 October (morning, unattended): review round 6 (5/10 and 6/10), number check 6, job 100

**Status for Harshith (read this first).** Round 6 reviews: MLSys PC **5/10**, professor **6/10 weak accept** ("Would I
take this student? Yes"); both rank clarity first (2/5). Number check 6 found 1 error and 5 overclaims (all fixed).
Job 100 (four launches so far, two more running) answers the reviews' two main technical asks. Main is committed as
you, not pushed; `gpu` is pushed. Reports: `reports/Review 6a MLSys PC (6 October, night).md`, `Review 6b professor
(6 October, night).md`, `Number check 6 (6 October, night).md`.

- **Clarity rewrite:** abstract and finding (2) in plain words (what to cache, how to load it); the order-free
  accounting now walks one worked example (O4, gpt-oss 11%: 4% + 4% alone, 52% together, interaction 43) before the
  ranges; the time-versus-reads passage rewritten; "law" and "speed limit" renamed (calibrated model, bound) in tables
  and appendix; the Shapley table relabelled (load alone, cache alone).
- **Fixes from number check 6:** the measured-GPU bound moves at gpt-oss 25% (was "host-bound unchanged"); "usual ways
  lose" -> "gain at most 15% or lose"; forecaster vs random window restated; "published systems less" -> the common
  procedure's medians; fill vs drop scoped; the two 285K hosts described correctly; 20 problems on the panel stated;
  standard audit quartiles (8.1-20.6); MIN prefetched (not MIN 1 read) beats the usual ways; same-CPU spread from the
  panel (16-29%); scoring-rule exception for deterministic counts stated; hard-coded numbers made macros.
- **Model checks (professor W2):** `hostdep_model.py` now counts O4 once (39 host-budgets, 14 hosts), freezes G at the
  median of the other hosts (median error 2.1%, sign of fetch/base right 39/39), and compares a one-path model (37/39:
  misses both losses). Background-copy states predicted faster than measured at all 64 (median 16%).
- **Statistics:** crossed two-stage bootstrap (same problem draw for every host); per-budget correlations (0.93, 0.88;
  0.71 and 0.56 without the two slow-link hosts); the line at half drawn after the data, said so.
- **Horizon rule:** bootstrap over models gives the power law's / rule's median-error ratio 0.95-1.09, so not
  distinguished; D(W50)/C falls with budget in 9/9 models. Paper now says this and argues the rule's value is the unit.
- **Engine:** new mode (patch oracle3) - the window on the deployed path (decayed frequency, kappa 1, background copies).
  CPU test: equals the deployed policy counter for counter with no window (one admission differs at the trace's last
  step at C=8).
- **Job 100** (gpu commits 4fda930, 752e1c3, 718bf41, predictions before each launch): 100a = Pd relaunched, 100b =
  5700X3D, 100c = Ph relaunched (100d, a 12400F, never finished its 65 GB download in 95 min and was destroyed without
  timed runs; its first attempt 100a died at setup on a broken package index, fixed by a retry in setup.sh). Results:
  relaunches reproduce every job 099 ratio within 0.030 and the deployed time within 1.9%; the probe-only model (G
  frozen, written on the host before any timed run) has median error 2.8% over 24 predictions (worst -19%, MIN 1 read
  on Pd); deployed-path windows are safe but small (b16 1.00-1.03x at 11%, 1.07-1.12x at 25%); the in-step windows
  win where the link matches the CPU and lose on Pd, as predicted; engine misses of the deployed-path windows exceed
  the instant-admission replay by 7-33% (prediction 8 failed: late-landing copies). 100e (13900KF behind a faster
  link) and 100f (9950X behind a slower link) running, prediction 9 added for them before launch.

## 6 October (night, unattended): research regroup, the clarity rewrite, job 099 (host panel)

**Status for Harshith (read this first).** Deep research is in `reports/MoE paper path to a 9.md`. The main text is
rewritten around one question per section and plain names (bound; where the seconds go; how much foresight is needed).
Job 099, a panel of ten rented RTX 5090 hosts with the machine as the statistical unit, is done, fetched and pushed on
`gpu`; every host was destroyed after its fetch. Remaining Vast credit: $13.46. Main is committed as you, not pushed.

- **Research** (5 threads + report): the bound in seconds and the in-engine oracle factorial are still unclaimed; the
  trace study and learned order are precedented (2505.16056 ICLR'26 lookahead-window oracle over 20 models; 2608.07911;
  FlashMoE; SeqMoE; MoE-SpAc). New concurrent work: llama.cpp PR #29887 (3 Oct, LRU expert cache admitting every miss).
  A 9 needs a realisable forecaster at ~0.7 precision/recall 4-8 tokens ahead; none exists. The paper's identity moved
  to: the bound, an order-free accounting, host dependence predicted by the time model, and the price of foresight.
- **Always-admit check** (found by the research): admitting every miss reads less than our online baseline at 23 of 26
  trace points (up to 6.7%) and is the best online policy at 18 of 27 policy-study points; the best online policy is still
  35-110% above MIN. W50 refit against it: 0.67 (C/k)^1.31 (exponent 1.35 without sub-token points, was 1.51).
- **Distinct-expert rule:** half the gap closes when the window holds 0.65 C distinct experts per layer (IQR 0.61-0.72,
  CV 0.16); leave-one-model-out it predicts a held-out model's W50 within 1.16x median (power law 1.19x, proportional
  1.22x). `scripts/w50_rebaseline.py`, `prereg/foresight/w50_distinct.json`.
- **Order-free accounting** (`scripts/factorial_shapley.py`, jobs 096 + 099): on the 10 hosts whose link reads at least
  half as fast as the CPU, reading the deployed policy's admissions once is worth -4 to 4% of the gap; MIN's set read
  twice -13 to 26%, read once 26-57%; interaction 10-58%; Shapley set 9-35%, reads 1-32%; pacing (nested) -4 to 23%;
  rest 30-62%. On the two link-starved hosts the single read in the step costs time (-21 to 2%) and the rest is 68-99%.
- **Like-for-like audit:** our cache scored by the audit's own procedure reaches 18-42% (median 27%) of its bound,
  against the published in-class median of 13.6% (`scripts/audit_ours.py`).
- **Engine: degraded foresight.** New oracle modes (patch oracle2): `oracle_hybrid` (a W-token window over decayed
  frequency, single reads), `oracle_recall/fill/seed` (each future expert kept with probability r, hashed per step).
  Engine equals `scripts/value_map.py`'s replay exactly on a CPU build (30 configurations, 3 budgets); admit-every-miss
  (kappa -1e9) equals the replay's W = -1.
- **Value map (reads, AIME routing):** half of MIN's gain over admit-every-miss needs 2-10 exact tokens at the four
  host-bound budgets; recall 0.5 over 8 tokens closes 0.14-0.42. Realisable forecasters fitted on other text: ridge
  (precision 0.48 one token ahead, 0.26 four ahead on gpt-oss) closes 3-15% (gpt-oss) and 6-19% (Qwen3); a GRU over the
  all-layer routing vector (SeqMoE-style; recall at k+3 0.53-0.54) closes 2-11% and 6-17%.
- **Job 099** (gpu commit 7b8327a, predictions in the header, pushed before launch; outcome
  `prereg/panel_outcome_099.md`): ten hosts (285K x2, 9800X3D, 7900, 13900KF, 9950X x2, 5950X, 7950X, 14700K), link
  19-49 GB/s, CPU 25-93 GB/s. Machine scoring: 190 held, 42 held (point), 39 failed, 2 untested (Pd's replay step
  failed: numba did not install). MIN with one read 1.22x the deployed cache pooled at gpt-oss 11% [1.12, 1.30], 0.87x
  on the 285K whose link reads at 0.29 of its CPU; the paced prefetch 1.01-1.71x. Exact windows of 4 and 16 tokens
  recover 0.32-0.47 and 0.78-0.93 of MIN's time gain over admit-every-miss at 11% (reads: 0.52, 0.98); the time share
  falls below the read share by 0.05-0.20 as the link slows relative to the CPU (correlation -0.91), because a short
  window still reads 97% of its misses over the link where MIN reads 50%. The calibrated time model predicts in-step
  states within 2.2% median over 14 hosts, the sign of fetch/base at 43/43 host-cells, and window time shares within
  0.03. Failures: P3 (3, Pd/Pf), P5-P6 (Pd/Pf, plus the ordering on Pi), P7 fast-link half (4: admit-every-miss costs
  10-15% on Pa, Ph, Pj), P9 (allr5 plans in 193-277 us per step, all 20).
- **Paper:** Section 5's engine paragraph now states the measured time-versus-read relation and the model check; the
  "gain on every host" claim for the paced prefetch is now "never lose (the 285K breaks even at 11%)".

## 6 October (later): job 097, number check 5, review round 5 (5/10 and 5/10)

- **Job 097** (learned order in the engine; $3.10 including two 9950X3D rentals that were given 57 GB of disk and died
  at setup): on the O4 machine the learned order runs 1.01-1.03x the single-read online policy and 1.03-1.05x the
  deployed one at the host-bound cells (7-13% of the fetch oracle's gain); on O6 (9950X3D, 69 GB/s host memory) it
  runs 0.90-0.94x and 0.90-0.94x. It reads 6-18% fewer experts on both hosts but buys them with 1.1-1.6x as many in-step
  fetches and 0.24-0.59 ms of host time per token. The realisable mechanism does not reach the reviews' bar (a third of
  the oracle's gain). A second launch on O4's machine reproduced job 096's ratios within 0.024. Outcome:
  `prereg/learned_outcome_097.md`.
- **Gate addendum:** condition 1's simulated deployed baseline admitted about twice as often as the engine (kappa 2 for
  Qwen3); against the engine's measured online reads it held at seven of eight host-cells, not all four cells on O4.
- **Number check 5** (29 defects, all fixed): the thesis was overclaimed (the usual ways also gain 25-63% at the 25%
  cells on fast hosts; MIN's paced prefetch beats every usual way by 1.18-2.03x at every host-bound cell); the
  learned-replay "deployed" column was a simulator that admits twice as often as the engine; O5's rest exceeds what
  the oracles recover; scope slips.
- **Review round 5** (`reports/Review 5a`, `5b`): **5/10 and 5/10, weak reject** (round 4: 4 and 4). Both verified
  13 claims each against the artifact with no data errors. Both rank clarity first (2/5), then uncertainty only within a
  launch, the order dependence of the accounting (now stated), and the absence of a realisable mechanism. The professor
  would take the student. Their factual errors are fixed (commit 62a451f); the scorecard table moved to
  `paper/supplement.pdf` (paper 26 pages, main text 8).
- **Spend this round:** $14.56 by the credit balance ($40.24 before job 096, $25.68 now); the ledger's estimate is
  $9.63 (096 $5.44, 097 $2.45, 098 $1.74), the difference being disk and bandwidth charges the ledger does not track.

## 6 October: the plan to a 9, executed: jobs 096 and 098 in, the learned order gated and in the engine (job 097 running)

- **Gate first** (`prereg/learned_gate_097.md`, commit d23d090, written before the decisive replay and before job 096's
  results): a realisable admission order is measured in the engine only if (1) offline it closes >= 30% of the deployed
  policy's read gap to MIN at the four host-bound cells, (2) it adds >= 15% of the single-read policy's gap at three of
  them, (3) job 096 shows fetch-on-admit not losing (lower bound >= 0.98 at three of four host-bound cells per host).
- **The learned order, offline** (`scripts/learned_policy2-6.py`): history-only features plateau at 18-25% of the
  single-read gap on gpt-oss (about one token of perfect foresight); MIN-imitation labels are worse (15-19%); cross-layer
  transition scores from the previous token add 2-3 points. The realisable form (features the engine has at plan time),
  fitted on each model's mixed-domain trace and replayed on the AIME routing (no test text seen): 6-12% fewer reads than
  single-read decayed frequency at the host-bound cells, 17-30% of its gap to MIN, 47-57% of the deployed policy's
  (`prereg/learned_offline.json`; Table in Section 5). Conditions 1 and 2 passed (2 only with the cross-layer scores,
  8-bit table).
- **Engine** (`LLAMA_EC_LEARNED`, patch on gpu 73bf256): host-computed logits per expert at the end of each step,
  victims and fetch ranks from them; on a CPU build its reads and fetches equal a Python replay of the same policy
  exactly in three variants; the model files are recomputed on the rented host from the committed traces and weights
  and must hash to the replay's.
- **Job 096** (two hosts, ten configurations per cell, shuffled, cold cache; $5.40): O4 (9950X, link 45 GB/s) and O5
  (9950X3D, link 27 GB/s). MIN with each admission fetched once 1.36-1.51x (O4) and 1.16-1.26x (O5) at the host-bound
  cells; with the paced prefetch 1.25-1.81x; Belady prefetch reads 1.6-2.3x R* even read once and is within 1-3% of its
  two-read form at gpt-oss; MIN served-then-copied 0.94-1.18x; the online policy with its admissions read once
  0.97-1.03x (its second read is not where its time goes; MIN admits 2-11x as often). Predictions: O4 61 of 66 held
  (13 strictly), O5 64 of 66 (16 strictly); between hosts 1 of 6 (fetch's gain differs by 0.13-0.26: the link).
  Condition 3 passed on both hosts. Outcome: `prereg/foresight_outcome_096.md`.
- **Job 098** (FreeToken tuned per cell, five settings, 9950X host; attempt 1 died on the CUDA 12.8 image, $0.40;
  attempt 2 $2.70): FreeToken's best is at most 1.04x its Table 1 setting; ours over its best 1.17, 1.17, 1.09, 1.02,
  1.04, 0.98 (within 0.04 of host S); VRAM FreeToken 28.4-30.1 GB, ours 9.5-27.3; CPU-only the engines are within 1.13x
  (prediction of >= 2x failed: FreeToken's CPU path is fine on this host). 37 of 44 clauses held.
- **Job 097** launched on the O4 machine and the O5 machine (the first 9950X3D rental died with a full disk after
  10 minutes and was replaced): base, foa, learned, fetch, both3p per cell, shuffled.
- **Paper:** front half rewritten (abstract, thesis, glossary, law folded into Section 2 with its details in the
  appendix, model accounting moved to an appendix), Section 4 rewritten around the three-host factorial with a new
  figure, Section 5 gains the learned order, related work on learned replacement and integrated prefetching,
  KTransformers and tuned FreeToken in the appendix, one stated error convention. Main text 8 pages.

## 5 October: number check 4 and review round 4 (both 4/10); the corrections; what the reviews ask for next

- **Number check 4** (`reports/Number check 4 (2 October).md`, an independent re-derivation of about 285 statements):
  22 defects, all in prose and captions, none in a macro or table. The serious one: the hit-optimal oracle is not a
  double-reading oracle. Its counters in job 095 show 1-3 CPU misses per token against 14-84 admissions at five of six
  cells (98-99% hits), so at most 2-6% of its admissions were read twice; its 1.77-2.47x R* is over-admission (Belady
  prefetch without bypass admits what the optimum bypasses and re-copies what it evicts; its admissions alone are
  1.57-2.16x R*). "Two reads per admitted expert" is right for the bytes-optimal oracle (094) and the online policy.
  Also: "slower than every single-read variant at every cell" false at the two GPU-bound cells (hit-opt ties fetch
  within a point); "44-45% of the limit at every window" false (what is constant is utilisation, 89-94%); the two
  oracle-gain ranges reversed; the regime claim wrong on O3; the contribution bullet's shares are the four host-bound
  cells; the running example's lead 1.27x not 1.29x and its table copying from two misses; power limits 350-600 W;
  prompt lengths 52-879 tokens; the 3090 host as a second exception to "a second path adds bandwidth"; 16 timed
  clauses scored held without an interval (intervals now computed by `scripts/scorecard_intervals_oracle.py`: all 16
  hold, the strict rate stays 54%). All fixed in the paper (main at 3ba8a49 and after).
- **Review round 4** (`reports/Review 4a MLSys PC (2 October).md`, `Review 4b professor (2 October).md`): **4/10 and
  4/10, weak reject** (round 3: 5 and 5; the structure is cleaner and the claims more visible, and both reviewers
  dug into the repository). The professor would still take the student and support the fellowship. Where they agree:
  (1) the oracle evidence is one host, one launch, six configurations in a fixed order; the same-host comparator is
  the *unpaced* Belady prefetch (the paced one, faster on O1 at three of four host-bound cells, was not run on O3);
  serve-then-copy was not run on O3, so "reads an admitted expert twice" is inferred across hosts; the two mechanisms
  (over-admission, the second read) are never separated on one host; an online single-read admission (decayed
  frequency with fetch-on-admit, no foresight) is the missing control for the "bytes foresight saves" step, since the
  online policy itself double-reads 3-7 admissions per token; (2) the Shapley accounting is a model whose endpoints
  are fitted and whose intermediate states no engine realises, and at the running example foresight edges overlap by
  0.14 ms of 6.84 (largest in 70 of 120 orders); it should be demoted to "what the model implies" with the
  measurement leading; (3) the writing: one thesis, a glossary, fewer numbers per sentence; (4) the audit: a
  different bound from Eq. 3 (per-layer, dense in sequence), 9 of 29 trace rows outside the class (speculative,
  lossy), 21 rows with an imputed upper-bound capacity, classes pooled; (5) the law: the blind predictor set the link
  bytes to fetches + prefetches (no admission copies) while the re-score counted admissions; G larger than the
  all-in-VRAM token; (6) W50 is a perfect-foresight horizon and its stated baseline is the single-read dfa, not the
  deployed policy; the speculation claim holds only once rejected drafts are charged; (7) FreeToken fairness (defaults,
  backend carried over, VRAM not reported, KTransformers absent); (8) a simulator bug (professor): the "warm cache"
  explanation of the two failed 095 reads clauses is impossible (fresh context per configuration); the simulators left
  a carried-over resident's next use at "never". Fixed: both simulators recompute next uses at each sequence start and
  now match the engine within 0-4% at every cell (outcome note 095 carries the correction; the clauses stay scored
  against the registered values).
- **Corrections applied today ($0):** the audit paragraph (in-class median 13.6% over 20 rows, speculative and lossy
  rows named, the imputed capacities disclosed, the audit bound's construction stated; the abstract and conclusion
  use the in-class figure); the W50 baseline named; the accounting paragraph gives the near tie and the overlap-first
  order on O3 (overlap 17 and 9% of the gap at the two 25% cells, bytes given it 40 and 52%); the running example's
  lead attributed to the admission policy (77%) and the CPU path; host B's card about 9% above the common-clock
  cards, host S's power limit and SM clock stated; VRAM reported (ours 9.5-25.9 GB, llama.cpp 9.5-25.3, FreeToken
  28.4-28.6); the 3090 FreeToken ratios no longer printed; Fig. 5's caption; "as the law predicts" removed; the
  launch-order 11% effect reported; Cao et al. 1995 cited; checklist rules 4 and 10 reworded; the scorecard's sign
  count strict.
- **Not done, and what each costs:** a confirmatory randomised, replicated oracle factorial on three hosts (online /
  MIN-bypass fetch / serve-then-copy / single-read prefetch unpaced and paced / Belady prefetch paced and unpaced /
  Belady single-read, two launches each, cold start per configuration; about $40 and a week, one engine flag to add);
  an online fetch-on-admit admission path for the deployed policy (2-3 days of engine work, ~$5), which both reviewers
  call the missing control and the practical result; a re-score of the law under one definition of the link bytes;
  FreeToken's thread count and fetch cap swept per host and KTransformers added (~$10); a realisable forecaster (4-6
  weeks, the step both reviewers tie to a 9). Both reviewers put items 1-5 of their lists at a 6-7.

## 2 October, midday: job 095 lands (read once, foresight pays everywhere); the paper rewritten to one thesis

- **Job 095 (`095_single_read@vast`, RTX 5090 + Ryzen 9 9950X, host memory 52.5 / 43.2 / 59.7 GB/s by CPU / link /
  both, 61.5 at best; limits 121 / 302 / 512 and 68 / 150 / 303 tok/s; $1.50; results on `gpu` at 8a77cbc).** All 36
  runs completed. Ratios to the online policy (paired, 30 sequences), gpt-oss 11 / 25 / 40% then Qwen3 12.5 / 25 /
  43.75%: **fetch** (MIN with bypass, admitted misses fetched into their slot, one read, serialised) 1.286 / 1.316 /
  1.254 / 1.345 / 1.379 / 1.264, reads 1.04-1.14x R*, hits within 1.1 points of the simulation, 32-64% of the limit,
  62-75% of the gain v(F) predicts, the law under-predicting its time by 8-17% (the fetch's per-layer latency);
  **lead-2** 1.26 / 1.45 / 1.38 / 1.22 / 1.41 / 1.40; **both-2** (unpaced) 1.394 / 1.489 / 1.392 / 1.421 / 1.476 /
  1.407, 35-68% of the limit; **both-3 paced** 1.453 / 1.664 / 1.493 / 1.414 / 1.667 / 1.503, 38-68% of the limit,
  56-61% of the gap closed at the host-bound cells, 85-110% of v(F), reads 1.18-1.58x R*, the law over-predicting by
  5-36%; **hit-optimal** (two reads) 0.910 / 1.137 / 1.266 / 0.695 / 1.063 / 1.274, slower than every single-read
  variant at every cell, reads 1.77-2.47x R*, host memory at 71-82% of the probe's best rate. Of 56 clauses 44 held
  and 12 failed, all of size (`prereg/foresight_outcome_095.md`; scorecard 299 clauses, 54% strict, 64% with
  held-on-point). Reading: read once, foresight pays everywhere, including the two lowest budgets where the
  double-read oracles of 093 and 094 lost or gained 0-3%; the double read, not the foresight, was the problem.
- **The accounting, measured (`scripts/accounting_measured.py`, `prereg/accounting_measured.json`).** On host O3 at
  the host-bound cells the four states online, fetch, the faster prefetch state and the limit split the gap into the
  bytes foresight saves (34-49%), the overlap foresight allows (8-23%) and the rest (39-43%); the model's shares are
  52-66, 27-36 and 8-13%: it over-prices the bytes by a third to a half and the overlap by 1.5-3x and under-prices the
  rest 3-5x (the prefetch's excess reads, the CPU misses still served in sequence, the GPU at 52-61% of datasheet).
- **The paper rewritten (`paper/paper.tex`, main text 8.7 pages; appendix 20).** One thesis: a limit in seconds, the
  accounting predicted then measured, and what the traces say would recover it. Abstract 150 words, 8 numbers; three
  contributions; the instrument in 1.5 pages of Section 2 (the ablation, other cards, scope and parity moved to the
  appendix with their figures); the law in Section 3 as the tool that places an engine in the limit's coordinates
  and picks the split (Eq. 2 moved there), scoped to serialised execution; the limit in Section 4 with the audit as
  one paragraph (figure in the appendix); Section 5 "Where the Seconds Go": the Shapley model as the prediction, the
  factorial of oracles as the measurement (new Fig. 2, `scripts/fig_factorial.py`), the double-read trap, the measured
  decomposition, and what remains; hosts named B, S, A and O1-O3 at first use instead of "listing B" and "the
  stock-clock host"; the running example (gpt-oss-120b at 25%) carried through law, limit, accounting and oracle.
  `paper/foresight_measured.tex` removed; Appendix E rewritten around the three oracle jobs with the single-read table
  and the measured-accounting table. Macros: fsf*/fslt*/fsbt*/fsbp*/fsh* (095), fsbTotalReadsOpt*/fspTotalReadsOpt*/
  fsmTotalReadsOpt* (total reads of the double-read oracles over R*), am* (the measured decomposition).

## 2 October, morning: two reviews at 5/10, the single-read oracles (job `095_single_read@vast`, running), the $0 corrections

- **Review round 3** (`reports/Review round 3 (2 October).md`): with both oracle jobs in the paper, an MLSys PC member
  and a performance-modeling professor each gave 5/10 (down from 6 and 6.5). The agreed diagnosis: both oracles as
  built read an admitted expert twice (the CPU serves the miss, the background copy reads it again), so their host
  bytes were 1.7-2.0x the optimum's R* and neither realised the foresight state the accounting prices; the hit-optimal
  oracle sat at the DRAM roof (45-52 GB/s of 50.5) at 44% of the limit = R*/reads x utilisation, which is the Fig. 2
  plateau; the main text blamed copy latency and the queue cap and left the double read in the table captions. The
  abstract overclaimed ("the limit's own policy", "mostly on the bytes it saves", the regime claim refuted by its own
  test at 11-12.5%), the bound was applied to pooled (FreeToken) and whole-layer (llama.cpp) systems, the audit median
  was the 29-row subset, the scorecard rate counted held-on-point, and the writing still hid the paper. Both: the
  decisive experiment is a single-read oracle (reads = R*), run as a factorial with prefetch overlap, worth about +2;
  with the rewrite, the bound scoped and the number fixes, a clear accept (8); the step to 9 is a realisable foresight
  mechanism (a learned Belady approximation or a routing forecaster at W = 2-4).
- **The single-read oracles, in the engine.** `LLAMA_EC_ORACLE_FETCH=1`: MIN with bypass decided before the step
  from the trace's record of it; each admitted miss is copied by the GPU's FETCH op into the victim's slot and run
  from there this step (one read, on the critical path; a forced plan the host writes per layer, the plan kernel
  executes it instead of the table); bypassed misses run on the CPU; reads = misses = the optimum's up to the
  within-sequence lookahead. `LLAMA_EC_ORACLE_LEAD=d`: the scheduled single-read prefetch: an expert whose first use
  is at least d steps away is copied ahead if MIN with bypass would admit it at that use (the victim is needed later
  than the expert's use after that one, or never); sooner uses miss (and, with FETCH, are fetched then). Simulated on
  the traces (`scripts/foresight_single_read_sim.py`, `prereg/foresight_single_read_sim.json`): fetch reads 40.0 /
  17.6 / 9.3 and 101.3 / 47.3 / 17.7 per token against R* 38.3 / 15.3 / 6.8 and 96.5 / 43.4 / 13.7, hits 72-95%;
  fetch + lead-2 reads 48.9 / 23.1 / 12.3 and 117.9 / 64.7 / 24.2 with hits 84-99%. The engine's modes match the
  simulation on the CPU toy model within 2% (a synchronous-copy fallback stands in for FETCH there).
- **Job 095 (RTX 5090 + Ryzen 9 9950X, offer 46402211, $0.50/h; running since 05:34 UTC; a first instance on a
  9950X3D stalled while loading its image, $0.16).** Six configurations per cell at all six cells: base, fetch,
  lead-2 (unpaced), both-2 (unpaced), both-3 paced, and the hit-optimal oracle unpaced. Seven predictions in the
  header (fetch's reads within 10% of the simulation and hits within 3 points; fetch faster than base everywhere with
  the law within 10%; both-2's reads within 25% and hits within 5 points of the simulation; both-2 at 55-85% of the
  limit at the host-bound cells and faster than the hit-optimal oracle; the law over-predicts both-2 by 15-50%;
  both-2 gains at least 25% at gpt-oss 11% and Qwen3 12.5%; paced within 10% of unpaced).
- **The $0 corrections (main at 272e0da):** the abstract's llama.cpp/FreeToken fractions over both hosts, and both
  scored against the pooled bound their designs allow (18-40% and 8-20%; `scripts/wsg_tables.py`,
  `prereg/pooled_fractions.json`); the audit median over all 52 rows (14.5%) beside the trace-scored 29 (9.5%) and
  the 52-row table with every source cited (Fate added) in a new appendix; the scorecard's strict rate (48%) beside
  the point-estimate rate (61%); the bound's class stated (exact routing, whole experts, C per layer, one token per
  pass) and listing B's probe reading 3% above the datasheet GPU rate; the law scoped to engines that read host
  memory in sequence with GPU work, the blind test's scope stated (our cache, gpt-oss, RTX 5090s), and the
  frozen-constant law scored on the three later RTX 5090 hosts (24 measurements: median 7.0%, p90 11.9%, worst 16%;
  `scripts/law_frozen_later.py`); the host-lottery figure in one unit; the trace study's baseline named; Table 3's
  "084" column renamed; draft and review references removed; Fig. 3's medians explained.

## 2 October, early: job 093 in the paper (hit-optimal foresight: +25-56% at 25% and above, -8% below); job `094_foresight_bytes@vast` running

- **Job 093 landed (RTX 5090 + Ryzen 9 9950X; a weak host memory: 43.7 / 45.8 / 49.4 GB/s by CPU / link / both,
  50.5 at best; all 46 runs completed, $2.43).** With the whole sequence in view the hit-optimal oracle runs 1.40x
  [1.38, 1.41] the online policy at gpt-oss 25% and 1.25x [1.24, 1.26] at Qwen3 25%, closing 41 and 31% of the gap to
  this host's limit and reaching 76 and 52% of what the accounting's foresight-only term predicts; 1.48x and 1.56x at
  the two GPU-bound cells. Half of the gpt-oss gain arrives by W = 9.2 tokens (trace study: 10.5 measured, 9.3 fit).
  At gpt-oss 11% and Qwen3 12.5% it *loses* 8% at every window: it reads more host bytes than the online policy (82
  and 195 experts per token against 63 and 160) because Belady admits experts used once where the optimum (38 and 97
  reads) bypasses them, and its copies land late (67 / 66% hits against the optimum's 73 / 75%). The unpaced path,
  with 98% hits at gpt-oss 11%, still loses: the bytes do it, not the timing. No-overlap within a layer: under 1%
  everywhere. The law on the oracle runs' own counters over-predicts them 24-39%: the admissions overlapped the step.
  Of 36 clauses 9 held, 2 on the point, 21 failed, 4 untested (`prereg/foresight_outcome_093.md`); the overall
  prereg hit rate is now 66% of 205 clauses.
- **In the paper (main at ba09b5e):** a "Foresight, measured" paragraph and figure in Section 4 (the four
  host-bound cells against W, with the model's foresight-only term as a tick), an appendix section with the per-cell
  table and the window sweep, the abstract (about 220 words) and the contributions list carry the result, the
  conclusion ends on it. An independent number check found 17 defects (the worst: "unpaced slower at all four
  cells" was wrong at gpt-oss 11%; the GPU-bound ranking of foresight differs per host; W50 should be compared with
  the trace-measured value, not only the fit); all fixed. Main text ends on page 10.
- **Job 094 (RTX 5090 + Ryzen 9 7950X, offer 53039899, $0.61/h; running since 00:11 UTC).** The bytes-optimal
  oracle: the same lookahead, but the policy admits only among this step's *missed* experts that have a next use
  (MIN with bypass), so its reads should match the optimum's (38.3 / 15.3 / 6.8 and 96.5 / 43.4 / 13.7 per token).
  Configurations at all six cells: base, bypass W = all, bypass W = 16, prefetch W = all. Six predictions in the
  header (reads within 15% of the optimum's; faster than base at every cell, by 10-45% at the four host-bound ones;
  hit rate within 4 points of the optimum's; the law over-predicts 10-40% at host-bound cells; bypass beats
  hit-optimal by at least 15% at 11 / 12.5%; W = 16 keeps at least 70% of the gain).

## 1 October, evening: two reviews (6 and 6.5 of 10), the rewrite, and the foresight measurement (job `093_foresight@vast`)

- **Two independent reviews of the grid draft** (`reports/Review round 2 (1 October).md`): 6 and 6.5 / 10. Agreed
  causes: the writing hides the contribution (430-word abstract, job IDs in prose, a three-page system section);
  two of the accounting's five terms are definitions (the GPU term closes the model on the measured time; "overlap"
  includes rebalancing onto the CPU) and the text did not say so; abstract-versus-body overclaims (the bound's class,
  "never largest", the 9.5%-versus-25-42% juxtaposition, the hit rate); and the largest term, foresight, was named and
  never measured. Both: fix the writing and the framing for an 8; measure foresight for a 9.
- **The rewrite (main at 4aed1e1).** Abstract to about 220 words with the prereg hit rate and the ceilings named; five
  contributions; Table 1 shows both RTX 5090 hosts (listing B and the stock-clock host) with the faster system in bold;
  the instrument moved after the audit and cut to about 1.7 pages (the split table, the half-link hosts, long outputs,
  FreeToken's own model, other systems, parity and prompts/batches now in a new appendix section); job IDs out of the
  main text; the accounting's two definitional terms stated and reconciled with the Nsight profile; the bound scoped to
  per-layer budgets; the law's evidence stated (three fast-link rows; FreeToken not predicted; per-measurement median
  3.4% beside the per-configuration 3.7%). Main text 9.1 pages before the foresight paragraph.
- **Job 093, foresight measured (RTX 5090 + Ryzen 9 9950X, offer 48822557, $0.70/h; running).** The engine gets an
  oracle policy (`llama.cpp-expert-cache-4da6337-oracle.patch`): the routing of the next W steps from a lookahead file
  recorded on the machine over the same teacher-forced text; Belady admissions within the window on the normal copy
  path (paced, 16 MB pieces). Tested on a CPU build with a 4-layer toy model (hit rate 81% dfa -> 98% oracle). The job
  runs base, paced, oracle W = 2 / 4 / 16 / 64 / all, oracle unpaced, no-overlap and all-CPU at the four host-bound
  cells, and base / oracle / no-overlap at the two GPU-bound ones, teacher-forced on the trace corpora (30 x 256
  steps), after a lookahead pass per model. Seven predictions in the header (the law's point values +38 / +34 / +65 /
  +76% at W = all; bands 20-80%; W = 16 captures half; 45-65% of the limit; hit rate within 3 points of the optimum's;
  no-overlap 5-25% slower; GPU-bound gain below 15%). `scripts/foresight_stats.py` and `foresight_paper.py` ready.

## 1 October, late morning: the grid lands (jobs `091_grid_4090@vast` and `092_grid_3090@vast`); tranche 1 complete

- **Job 091, RTX 4090 next to a Core i5-12400 (3.5 h, 1.9 USD).** The host reads 37 GB/s (CPU 34, link 25, both 37);
  ours leads FreeToken 1.37 / 1.30x on gpt-oss (11 / 25%) and 1.03 / 1.05x on Qwen3 (12.5 / 25%), runs 1.8-2.6x
  llama.cpp, at half the headline machine's absolute speeds. Every cell is host-bound at the limit; ours stands at
  30-46% of this machine's limit. Qwen3 43.75% skipped by the VRAM arithmetic (28.0 GB). Predictions: the gpt-oss 11%
  lead (37%) overshot the 10-35% band; 1.78x llama.cpp at Qwen3 12.5% missed 1.8 by 0.02; the law's tables copy one
  more expert than the headline machine's at one entry each (the prediction reasoned from the link alone; the CPU is
  slower in the same proportion, so the ratio is listing B's); the fraction of the limit at Qwen3 12.5% (45.7%) overshot
  25-45%; order within 2%. `prereg/grid_outcome_091.md`.
- **Job 092, RTX 3090 next to a Core i9-11900KF (3.9 h, 1.6 USD).** The host's CPU alone reads 45 GB/s and CPU + link
  41-43, so the law's tables copy nothing. Ours runs 1.64 / 1.56 / 2.18x llama.cpp (gpt-oss 11%, Qwen3 12.5 / 25%), at
  27-42% of the limit. FreeToken's carried-over hybrid backend ran at 8-11 tok/s, below llama.cpp (its own calibration
  recommends offload on this host, not run): reported, not counted. **Mixtral-8x7B** (8 experts, top-2, 26 GB of
  Q4_K_M experts): ours ties llama.cpp at C = 2 (1.015 [1.008, 1.024]); with 8 experts per layer a 25% cache hits
  31.5% of reads against pinning's 25%, so there is little locality to earn; the C = 4 cell was lost to the 4 h
  deadline. The noisiest host: the two launch orders differ by up to 11%. 2 of 5 predictions failed outright.
  `prereg/grid_outcome_092.md`.
- **The grid in the paper.** Abstract: "across four hosts and three cards the limit moves with the machine and our
  cache stays at 25-46% of it". An "Other cards" paragraph and Fig. 3 (ours and FreeToken as a fraction of each
  machine's limit, 19 cells) in Section 4; Table 8 (every cell against its own limit, with the term that binds it) and
  a grid section in the appendix; the Mixtral tie stated as the cache's scope. Limitations updated (three consumer
  cards). Main text ends on page 10.
- **Scorecard:** 169 clauses (jobs 073-092: 91 held, 25 on the point, 46 failed, 3 untested, 4 void); 090's ten
  clauses were missing and are added (all held); the "after the reviews" era is 83 clauses (41 / 14 / 27 / 1), 20 of
  the 27 failures bands. `scripts/grid_stats.py`, `scripts/grid_table.py`.
- **Spend:** balance 17.54 USD; ledger 40.6 of the 62 cap; all instances destroyed. Tranche 1 and the grid cost
  about 11 USD in total (088 1.6, 089 3.6 with the no-network host, 090 1.9, 091 1.9, 092 1.6).

## 1 October, morning: Table 1 on a second host (job `089_headline_stockclock@vast`; 091/092 running)

- **Job 089 (RTX 5090 at the common 14,001 MHz clock, 400 W, Ryzen 9 9950X; 3.0 h, 2.4 USD plus the 1.2 USD of the
  no-network host).** All six cells of Table 1 rerun with FreeToken's backend carried over from job 081 and both
  launch orders, then LRU at every cell and `llama-batched-bench` at 25%. 36 runs, 1,080 rows, nothing skipped.
- **The ratios shrink.** Ours / FreeToken 1.21 / 1.20 / 1.09 on gpt-oss (Table 1: 1.29 / 1.28 / 1.15) and
  1.03 / 1.05 / 0.97 on Qwen3 (1.03 / 1.15 / 1.05): ours leads at five cells, FreeToken at Qwen3 43.75% (CI
  0.962-0.987). Every ratio is 0.01-0.11 below Table 1's; the prediction was within +-0.06 (held at one cell).
  Order effect at most 1.0%.
- **The cause is the host, not the card.** This CPU reads 56 GB/s at the helper count against listing B's 72 (-22%;
  66 vs 78 combined), the link is the same 53 GB/s, and the card's memory clock is 18% lower. FreeToken's offload
  backend, which runs no expert on the CPU, lost only 5-8%; its hybrid lost 6-16%, ours 12-17%, llama.cpp 16-17%.
  The law's tables moved to 0,0,1,2,3 and 0,0,1,2,2,3,4,5,6 (more copying). Fig. 3 now has eight hosts and the
  gpt-oss lead reads as a function of the CPU-to-link ratio: 1.09-1.22x at 1.05-1.14, 1.15-1.29x at 1.28,
  1.51-1.58x at 2.96.
- **The limit moves with the host, the fraction does not.** Recomputed with this host's best probed rate (71 vs 87.5
  GB/s), the host-bound limits are 19% lower; ours stands at 26-43% of this host's limit against 25-42% on listing B.
- **LRU attribution.** LRU with the law's table runs 6-21% slower than decayed frequency across the six cells and
  trails FreeToken at four of them (0.81-0.97); it leads only at gpt-oss 25% (1.045). The admission policy is
  77-84% of the lead at gpt-oss 25 / 40% and more than the whole lead elsewhere. Prediction 5's 10-30% band failed
  at the two high budgets (7.8%, 6.4%), and LRU did not keep the lead at gpt-oss 11% as predicted.
- **Prefill and batch.** Ours runs 79-83% of stock llama.cpp's prefill at 512 / 2,048 tokens and 77-79% of its decode
  at 2 and 4 parallel sequences (both models, 25%); single-sequence decode in the same tool 3.7-5.4x stock. Now
  reported in the Design paragraph and Limitations instead of "not measured".
- **Scorecard:** 28 clauses for 089 (9 held, 7 on the point, 12 failed); the "after the reviews" era is now 39
  clauses (14 / 10 / 15). `prereg/stockclock_outcome_089.md`, `prereg/stockclock_089.json`.
- **Paper:** abstract and contributions carry both hosts (9-21% and -3 to +5% on the second); a "same table on a
  second host" paragraph in the head-to-head section; LRU attribution sentence in the ablation paragraph; prefill /
  batch cost in Design and Limitations; Fig. 3 caption; appendix log.
- **Spend:** balance 20.12 USD after 089 (ledger estimate 38.29 of the 62 cap). 091 (4090, 0.59/h) and 092 (3090,
  0.48/h) still running, 1.3 h in.

## 1 October, early: tranche 1 lands (jobs 088 and 090; 089 running; 091/092 launched)

- **Job 090, parity (RTX PRO 6000, 1.9 h, 1.8 USD).** Full-vocabulary KL of our cache (25%) against stock llama.cpp
  with every weight in VRAM: mean 0.0019 nats (gpt-oss) and 0.0005 (Qwen3), 99.9th percentile 0.12 / 0.06, top-1
  agreement 98.5% / 99.3%, loss +0.16% / +0.19%; stock's own `--n-cpu-moe` placement diverges by the same amount.
  3 of 3 predictions held. The first rental (offer 31632904) was a 40 GB A100 and the gate stopped it in a minute.
- **Job 088, the slow link done fairly (i9-14900K, 3.2 h, 1.6 USD).** FreeToken's hybrid on 8 P-core threads runs
  37.5 / 62.1 / 92.5 tok/s (3.1x / 2.0x / 1.4x job 085's 23-thread hybrid; its calibration copies 51.7% of misses
  instead of 65.5%); ours leads its better configuration 1.58 / 1.52 / 1.51x (CIs above 1.47). llama.cpp `-t 8`
  40.0 tok/s, within 10% of the law; `-t 24` 32.8 (+26% off): the hybrid-core failure is a thread-count choice.
  Prediction 3's band (1.10-1.50x) failed by 0.01-0.08; the other four held. `prereg/slowlink_outcome_088.md`.
- **Job 089, Table 1 on a stock-clock card.** The first host (offer 51871552) passed both gates and then had no
  network: every download failed, every run died at load, 1.2 USD lost. Relaunched on a verified 9950X (offer
  49539124, 0.79/h); running.
- **Jobs 091 (RTX 4090) and 092 (RTX 3090 + Mixtral-8x7B)** launched: the grid. Qwen3 43.75% does not fit a 24 GB
  card (28 GB of slots), so both cards run Qwen3 at 12.5 and 25% and gpt-oss at 11% (and 25% on the 4090).
- **Paper:** parity paragraph rewritten around the KL result; the slow-link paragraph reports 088 and marks 085's
  gpt-oss comparison as confounded; Fig. 3 uses 088's points; the abstract carries 1.51-1.58x; the scorecard has an
  "after the reviews" era (088 onward: 5 held, 3 held on the point, 3 failed).

## 30 September, late: six persona reviews, then the accounting revision (no GPU)

**Reviews.** Six persona reviews of the draft (benchmarking, MLSys PC, competitor author, llama.cpp maintainer,
memory-systems architect, statistician) scored it 4.8 of 10 on average (grade B-). Consolidated: the slow-link
headline was confounded (FreeToken's CPU backend failed on the i9-14900K's hybrid cores), the gap decomposition was
order-dependent and the bound loose and mis-scoped, Eq. 1 was not identified against an additive model, the CIs cover
problems only and the Table 1 card runs its memory at 17001 MHz against 14001 elsewhere, prior art was missing, and
several prediction clauses were scored in our favour. The reviews and a plan are in two Claude docs.

**Revision, all from existing data (scripts and outcome notes in `prereg/`):**

- **Model comparison** (`scripts/perlayer_model.py`, `prereg/perlayer_model_outcome.md`): leave-one-host-out over the
  33 law measurements. Token-level max with G refit 3.1% median, additive 4.5% (misses the three fast-link FETCH
  rows by -11%), per-layer max 3.2% and no explanation of the failures. Eq. 1 stays. The three failures get causes:
  a CPU miss beside a copy takes 20-62% longer than alone; the EPYC 9655 has a 55-70 us per-layer floor; prefetch
  overlaps attention. G > the all-in-VRAM token because the helpers read at 1.09-1.31x the probe's time per expert.
- **Tightened limit** (`scripts/speed_limit_v2.py`): the 084 optimum was not exact (foresight._pol never evicts an
  expert of the current token); the exact MIN-with-bypass reads 0.5-3.9% fewer. Global pool +5-10% at host-bound
  cells; median B_host (80.7 of six samples) -8%; measured GPU ceiling collapses the limit at >=25% to 279/192 tok/s;
  per-layer sum 1-5% of a token below the token max. Ours 25-42% of the limit as published, 33-57% with every
  tightening. Table 1's limit column now uses the exact optimum.
- **Shapley accounting** (`scripts/shapley_gap.py`, `paper/figs/shapley.pdf`): foresight largest in the four
  host-bound cells (37-53% of the gap), overlap in the two GPU-bound cells (39-46%); GPU efficiency, policy and host
  work never largest in any of the 120 orders (net accounting). Replaces the fixed-order figure.
- **Nine-model policy study** (`scripts/policy_study.py`, `paper/tab_policy.tex`): best online policy reads 35-110%
  more than the exact optimum; at C/k <= 1.5 all six online policies within 4%; from C/k = 2 LRU reads 2-11% more
  than decayed frequency on 8 of 9 models (Qwen2-57B the exception); S3-FIFO ties decayed frequency; LFU up to 6.2x;
  profiled static 1.4-13.2x; W = 4 beats every online policy in every cell. Drift: the top-C set changes 22-58% per
  100 tokens and 33-75% per 1,000. Our kappa costs 1-8% reads (up to 22% on Qwen3).
- **W50 with the exact optimum** (`scripts/foresight_exact.py`): 0.59 (C/k)^1.33, r 0.971, exponent CI [1.09, 1.47]
  over a bootstrap of models; savings of the optimum over the best online policy 26-56% (median 41%).
- **Batched verification on traces** (`scripts/batchk_trace.py`): a median 15-32% of reads saved at K = 8 with
  rejected drafts free (W ~ 2-4 equivalent); +8-12% at alpha 0.9 and a loss of 18-20% at alpha 0.8 once rejected
  drafts' routing counts. Speculation is not foresight; the stretch item "measured speculation point" is dropped.
- **Audit figure** (`scripts/fig_audit_sol.py`, `paper/figs/audit_sol.pdf`): 52 published measurements against their
  own speed of light (datasheet ceilings): trace-based median 9.5% (IQR 4.9-17.8, best 36.8), i.i.d. rows 34.8%.
- **Scorecard** (`scripts/scorecard.py`, `paper/tab_scorecard.tex`): 86 clauses under one CI rule: 50 held, 11 held
  on the point estimate, 19 failed, 2 untested, 4 void. Downgraded: 074b P2b (untested), 085b P4 (failed by its own
  "2 of 3"); 073 P4 scored on fully identical outputs (0-3%); 087 P1 held (point).
- **Bibliography**: WiSP, Budgeting Bytes, Paging the Experts, FlashMoE, Mira, MoE-SpAc, Cascade, ShapleyIQ, FlexGen
  added; SP-MoE, 2608.07911, 2608.12103, SeqMoE, SpecMD and the two llama.cpp PRs now cited.
- **Paper** (`paper/paper.tex`; the pre-review draft kept as `paper/paper_v1_prereview.tex`): reframed around the
  limit and the accounting; 2.03x dropped from the abstract and the i9 result reported as confounded; the memory
  clock disclosed; listing A/B defined; "a system the law implies" gone; KTransformers dropped from the main text;
  limitations expanded. 9 pages of main text plus references and appendix (15 in all). An independent check found
  15 wording defects (gap share vs token share, "every" vs "best" policy, an audit overclaim, a 15501 MHz law host),
  all fixed; `scratchpad/reviews/check_v2.md`.

**Still to do (needs Vast credits; balance 3.65 USD):** the fair slow-link rerun (FreeToken on 8 threads, fetch caps
0-2; llama.cpp -t 8), the stock-clock headline on two rentals with ABBA order, ours-with-LRU against FreeToken, KL
divergence against an all-VRAM reference, prefill and -np 2 timings, the RTX 4090 and 3090 grid, a fourth model.

## 30 September, night: the contrasting machines (jobs `085_half_pcie@vast` and `085b_half_pcie_qwen3@vast`)

**Setup:** RTX 5090s in PCIe 4.0-class slots (about 27 GB/s): gpt-oss next to a Core i9-14900K (085), Qwen3 next to a
Ryzen 9 7900 (085b). Job 085's Qwen3 conversion hung with the CPU idle until the 6-hour timeout, so its Qwen3 half ran
as 085b. `prereg/halfpcie_outcome_085.md` has the outcome.

- **The law's tables fetch less:** gpt-oss 0,0,0,1,1 and Qwen3 0,0,1,1,2,2,3,3,4.
- **The law's table beats the fixed one by more:** +21–34% (gpt-oss) and +9–18% (Qwen3), two to four times the gain on
  the headline machine.
- **Ours ÷ FreeToken:** 2.03 / 1.70 / 1.43 on gpt-oss, where FreeToken's offload sends every miss over the slow link
  and its hybrid backend ran at 11 tok/s; 1.07 / 1.08 / 1.04 on Qwen3.
- **With the fixed table** ours would have trailed FreeToken at Qwen3 12.5%.
- **llama.cpp against the law:** within 2.7% on the Ryzen; +22% on the Core i9 (hybrid cores, as in job 069c).
- **Predictions:** 4 of 5 held.

## 30 September, evening: the ablation ladder with a real static cache (job `087_ablation_static@vast`)

**Setup:** listing A, 25%, AIME-25 problems 0–14. `prereg/ablation_outcome_087.md` has the outcome.

- **Placement is the first gain.** A static cache of each layer's 32 most requested experts, profiled on the models'
  MATH-500 outputs, runs 1.31× (gpt-oss) and 1.89× (Qwen3) stock llama.cpp.
- **The decayed-frequency policy matches the hindsight-best static placement:** 0.98 [0.95, 1.01] and
  1.01 [0.98, 1.04] of it, online and without a profile. It is 1.58× / 1.12× the profiled static cache.
- **LRU is the wrong online policy.** It admits every miss, and on Qwen3 it is 0.88× the static cache.
- **The rest of the ladder is as in job 082:** 2–9% per step, and ours runs 2.68× / 2.81× stock.
- **Predictions:** 1 and 3 held; 2 failed on gpt-oss (the profile from other text matched poorly); 4 failed on Qwen3
  (the LRU step).

## 30 September, evening: FreeToken's own headline model (job `086_qwen36@vast`)

**Setup:** Qwen3.6-35B-A3B BF16 on listing A, Table 1 protocol. `prereg/qwen36_outcome_086.md` has the outcome.

- **Ours and FreeToken are close:** +3.0% [2.3, 3.8] at 12.5%, +1.0% [−0.2, 2.1] at 25%, +0.3% [−0.6, 1.3] at 37.5%
  (FreeToken hybrid at every budget).
- **Against llama.cpp:** 1.98 / 2.31 / 2.48×.
- **The law's table beats the fixed one** by +5.7–7.1%.
- **FreeToken as shipped runs 93.8 tok/s,** the top of its paper's 77–83. The comparison does not handicap it.
- **Our lead is model-dependent:** large for gpt-oss (few large experts), small or nil for this model (125 misses per
  token at 12.5%, experts half the size).
- **Predictions:** 1 and 4 held; 2 and 3 failed (3 by 1.98 against 2×).

## 30 September, afternoon: gpt-oss long outputs on hard problems (job `083_gptoss_hard_long@vast`)

**Setup:** listing A (job 079's 9950X3D machine; listing B was rented out). 10 held-out AIME 2022 problems (11–15 of
both exams) × 2,048 tokens. `prereg/hard_outcome_083.md` has the outcome.

- **Ours leads FreeToken on long outputs by more than in Table 1:** 1.36 / 1.32 / 1.19 at 11 / 25 / 40% over all
  tokens, and 1.37 / 1.33 / 1.19 over tokens 257–2,048 (every CI above 1).
- **The runs are valid:** both engines stream exactly one event per token, and at most 1 of 10 requests reaches an
  answer in 2,048 tokens.
- **Ours runs +3–5% faster than its Table 1 rate,** with no post-answer speed-up.
- **Predictions:** 4 of 4 held. This replaces job 082's void gpt-oss long runs.

## 30 September, afternoon: all in VRAM and routing traces, void (job `084_vram_traces@vast`)

- **The RTX PRO 6000 was throttled:**
  - gpt-oss-120b all in VRAM ran at 93 tok/s, against 261 in job 075 on the same card model;
  - device read was 1,304 against 1,641 GB/s.
- **Nothing is scored.** `prereg/vram_outcome_084.md` has the details.
- **Two harness bugs, fixed:**
  - The tokenising server was started through `eval`, so it survived `kill` and tokenised the next model's prompts.
  - A lookahead record over 8 MB is dropped by the result channel.
- **Reruns:**
  - 084b: another PRO 6000, with bandwidth and speed gates at start.
  - 084c: the gpt-oss trace on 084's own greedy text, which survived.

## 30 September, morning: the review's run 1 (job `082_review_run1@vast`)

**Setup:** the headline machine; `prereg/run1_outcome_082.md` has the outcome.

- **Long outputs (Qwen3, 2,048 tokens, held-out MATH-500):** ours leads FreeToken by +4.3 / +5.6 / +5.4% over tokens
  257–2048 (all CIs above 1). Both systems speed up after the first 256 tokens, ours more.
- **FreeToken's prefill overlap:** turning it off changes nothing, so the review's decode-window threat is refuted
  where it can be measured.
- **gpt-oss long runs are not comparable.** The easy prompts end early, `ignore_eos` then forces post-answer text,
  and ours' hit rate jumps from 58% to 77%. A rerun on hard held-out problems (AIME 2022–2024) is needed.
- **Ablation at 25%:** stock 40.5 → ours 107.6 tok/s (gpt-oss) and 22.0 → 61.7 (Qwen3).
  - Dynamic caching gives 2.2–2.3× over a static cache; the decayed-frequency policy +20–27%. **Erratum:** the
    "static" step loaded no experts (`LLAMA_EC_INIT` unset), so it measured "no expert resident". Job 087 reruns the
    ladder with a real static cache.
  - Mailbox, maps, GPU sampling, fixed FETCH and the law's table add 2–10% each.
  - The "static" step (in fact no expert resident) is slower than stock.
- **Teacher-forced parity:** top-1 agreement with stock is 98.6% (gpt-oss) and 99.3% (Qwen3); NLL moves −0.22% and
  +0.31%. Stock against its own second placement: 99.6% / 99.9%.
- **Missing Table 1 cells:** llama.cpp Qwen3 at 25% / 43.75% runs 22.1 / 28.4 tok/s, within 3.6 / 4.0% of the law's
  prediction.
- **Predictions:** 5 and 6 held; 2, 3 and 4 failed (3 favourably); 1 is not scorable.

## 30 September, night: the headline table with the law's split (job `081_headline_law@vast`)

**Setup:** RTX 5090 + 9950X3D (the job 080 listing), both models on one machine. The law's table was used at every
budget and confirmed on launch 2.

| Model | Experts on GPU | llama.cpp | FreeToken | ours | ours ÷ FreeToken |
|---|---|---|---|---|---|
| gpt-oss-120b | 11% | 34.9 | 54.0 | **69.9** | +29.4% [27.8, 31.2] |
| gpt-oss-120b | 25% | 40.0 | 85.6 | **109.2** | +27.5% [25.4, 29.5] |
| gpt-oss-120b | 40% | 47.7 | 132.2 | **152.5** | +15.4% [13.4, 17.2] |
| Qwen3 BF16 | 12.5% | 19.5 | 38.7 | **40.0** | +3.2% [2.2, 4.2] |
| Qwen3 BF16 | 25% (080) | — | 54.7 | **63.1** | +15.3% [13.4, 17.3] |
| Qwen3 BF16 | 43.75% (080) | — | 103.1 | **108.2** | +4.9% [3.6, 6.4] |

- **All four predictions held;** `prereg/headline_outcome_081.md` has the outcome.
- **The law's table against the fixed one:** +3.3% to +8.0% on every budget of both models.
- **Ours against llama.cpp:** 2.0–3.2× on gpt-oss, 2.05× on Qwen3 at 12.5%.
- **This is the candidate Table 1:** one machine, two models, every budget ahead of FreeToken.

## 30 September, early: a per-host FETCH split from the law (job `080_fetch_split@vast`)

**Setup:** RTX 5090 + 9950X3D (CPU 70.9 / PCIe 53.2 / both 80.7 GB/s), Qwen3 BF16. The law picks the FETCH table from
the machine's own probe: 0,0,1,1,2,3,3,4,5 against the fixed 0,1,1,2,3,3,4,5,6.

| Experts on GPU | ours, current table | ours, law table | FreeToken (better) | law vs current | ours (law) vs FreeToken |
|---|---|---|---|---|---|
| 43.75% | 103.1 | **108.0** | 103.2 | +4.8% [4.1, 5.5] | **+4.9% [3.6, 6.4]** (confirmation launch) |
| 25% | 59.2 | **63.1** | 54.7 | +6.7% [6.2, 7.1] | +15.3% [13.4, 17.3] |

- **All four predictions held;** `prereg/split_outcome_080.md` has the outcome.
- **The deciding entry:** a single missed expert runs on the CPU instead of being fetched, worth +3.7% by itself.
  - The best table in the sweep (lite) is within 1% of the law's table.
  - Fetching everything is the worst table.
- **The per-layer model ranks the nine runs with r = 0.96,** but overstates the differences by ~1.8×.
- **The 43.75% tie with FreeToken (job 079) is now a +4.9% lead.**

## 30 September, early: profile of Qwen3 at 43.75% (job `079_qwen3_profile@vast`)

**Setup:** RTX 5090 + 9950X3D (CPU 66.9 / PCIe 57.9 GB/s), Nsight per-token profiles plus the unprofiled
comparison.

- **Parity on this host:**
  - ours 100.1 tok/s, FreeToken's better backend 101.0; ours ÷ FreeToken = 0.991 [0.975, 1.009];
  - llama.cpp 27.5 tok/s.
  - Job 078's −5.2% depends on the host.
- **All three predictions held;** `prereg/qwen3_profile_outcome_079.md` has the outcome.
- **The 5.44 ms "fixed cost" is GPU work, not overhead:** attention, projections, router, resident experts and head
  take ~5.3 ms for us against ~5.0 ms for FreeToken.
  - Our extra ~0.5 ms: unfused batch-1 kernels (1.6× the kernels) and 0.3 ms more idle time between kernels.
  - Host side: 0.27 ms per token.
- **The miss path is the other half of the token in both systems:** 4.9–5.2 ms of PCIe copies, not overlapped with
  any compute in either system.
  - Overlapping each layer's resident experts with its copy is worth at most ~0.5–0.6 ms per token for ours (0.9–1.0 ms for FreeToken).
- **Our FETCH table is not tuned per host.** It sends 75% of misses over PCIe even where the CPU path is faster
  (+0.4% here, +16% on the 078 host).

## 29 September, late night: a second model, Qwen3-30B-A3B BF16 (job `078_qwen3_4way@vast`)

**Setup:** RTX 5090 + Ryzen 9 9950X with slow host DRAM (CPU 44.7 GB/s, link 46.7 GB/s), equal measured GPU memory,
one launch each.

| Experts on GPU | llama.cpp | KTransformers | FreeToken | ours v2 | ours ÷ FreeToken (95% CI) |
|---|---|---|---|---|---|
| 12.5% | 13.4 | 13.1 | 26.5 | **28.4** | +7.1% [+6.3, +8.0] |
| 25% | 15.4 | 15.3 | 43.9 | **46.8** | +6.7% [+4.3, +9.4] |
| 43.75% | 19.9 | 20.2 | **94.1** | 89.2 | −5.2% [−6.5, −3.8] |

- **All three predictions held.** `prereg/qwen3_outcome_078.md` has the outcome.
- **FreeToken leads at 43.75%.** That was not predicted, and it is the gpt-oss crossover again. Our fixed cost is
  5.44 ms per token, plus 0.194 ms per miss. At ~30 misses per token FreeToken's lower fixed cost wins, and on this host
  the CPU path is no faster than the link.
- **KTransformers (BF16 on Zen 5, no AMX) equals llama.cpp.** Both are static placements, and the CPU expert bytes over
  STREAM bandwidth explain 91–95% of their token time (post-hoc).
- **The BF16 CPU helper path works end to end:** coherent text, and the same opening as stock on 29–30 of 30 problems.

## 29 September, night: two more entrants on gpt-oss-120b (job `077_competitors_gptoss@vast`)

**Setup:** RTX 5090 + 9950X3D, equal measured GPU memory, one launch each.

| Experts on GPU | llama.cpp | Pipelined Sharding (MLSys'26) | leloch cache | FreeToken | ours v2 |
|---|---|---|---|---|---|
| 11% | 27.8 | 36.9 | 39.4 | 45.8 | **56.0** |
| 25% | 33.6 | 45.2 | 48.5 | 76.0 | **92.0** |
| 40% | 40.1 | 56.8 | 51.3 | 119.5 | **133.0** |

- **Both predictions held;** `prereg/competitors_outcome_077.md` has the outcome.
- **The two new entrants** run 1.3–1.45× stock llama.cpp. Ours runs 1.4–2.3× the better of them, and 11–22% ahead of
  FreeToken (paired CIs).

## 29 September, night: the clean table (job `076_table_5090@vast`)

**Setup:** RTX 5090 + Ryzen 9 9950X, fixed build (ours v2), GPU-side sampling, variant chosen on launch 1 and
confirmed on launches 2–3.

| Experts on GPU | ours v2 | FreeToken | Difference (95% CI) | llama.cpp |
|---|---|---|---|---|
| 11% | 60.5 | 50.2 | **+20.5% [+19.0, +22.0]** | 31.5 |
| 25% | 97.7 | 80.3 | **+21.7% [+20.1, +23.4]** | 36.4 |
| 40% | 138.7 | 124.1 | **+11.8% [+10.8, +12.9]** | 43.5 |

- All three predictions held; `prereg/table_outcome_076.md` has the outcome.
- **With the 60% result (job 075), the cache leads FreeToken at every budget from 11% to 60%** on the two
  desktop-class hosts measured with the fixed build.
- **Running now:**
  - job 077: Pipelined Sharding (MLSys'26) and leloch's llama.cpp cache on gpt-oss-120b;
  - job 078: Qwen3-30B-A3B BF16 with llama.cpp, ours, FreeToken and KTransformers.

## 29 September, evening: 40% and 60% on an RTX PRO 6000 (job `075_pro6000_40_60@vast`)

**Setup:** 9950X host; CPU / link / both = 62 / 53 / 71 GB/s. Each system's variant was chosen on launch 1 and
confirmed on launches 2–3.

| Experts on GPU | ours v2 | FreeToken | Difference (95% CI) | llama.cpp |
|---|---|---|---|---|
| 40% | 137.5 (FETCH) | 122.6 (offload) | **+12.2% [+10.7, +13.7]** | 43.7 |
| 60% | 189.7 | 169.7 (offload) | **+11.8% [+10.9, +12.7]** | 63.0 |

- **All in VRAM**, llama.cpp: 261.4 tok/s. With 60% of experts on the GPU, the cache reaches **72.7%** of that, in
  37.5 GiB against 59.6.
- **Predictions:** 2 of 3 held. The third ("within +10% at 60%") was exceeded. Outcome in
  `prereg/pro6000_outcome_075.md`.

**Where the comparison stands after the fixes** (maps on the GPU, GPU-side sampling, FETCH chosen per host). Ours
leads FreeToken:
- 7950X host (074b): +21% at 25%, +7% at 40%;
- PRO 6000 host (075): +12% at 40% and at 60%.

11% on these hosts has not been rerun with the fixes. Job 073's 11% and 25% results predate them.

## 29 September, evening: the server's sampler was the rest of the fixed cost (job `074b_server_overhead@vast`)

**Change tested:** `"backend_sampling": true`, a stock llama-server option that samples on the GPU. It cuts the
server's time between decode steps from 1.7–2.0 ms to 0.23–0.44 ms. The rest of that time is the CPU sampler chain
over 201k logits.

**Same 7950X host as 074, same session as FreeToken:**

| Experts on GPU | ours v2 + FETCH + GPU sampling | FreeToken | Difference (95% CI) |
|---|---|---|---|
| 40% | 122.8 tok/s | 114.4 | **+7.4% [+6.3, +8.5]** |
| 25% | 83.0 tok/s | 68.4 | **+21.3% [+19.9, +22.7]** |

- All three predictions held; `prereg/server_overhead_outcome_074b.md` has the outcome.
- **Settings across the jobs:** FETCH off on the 073 host, where the CPU path is faster than the link; FETCH on here,
  where they are equal. The host-memory law is what should pick.
- **Next:** 40% and 60% on an RTX PRO 6000 (job 075, running). Every llama-server system samples on the GPU there,
  and each system's better variant is chosen on launch 1 and confirmed on launches 2–3.

## 29 September, afternoon: the 40% fix, and why FreeToken still leads there (job `074_fix40@vast`)

**The data behind the 40% loss (job 073).** Speed per budget fits a line in misses per token, within 0.12 ms, for
both systems:

| | Fixed cost per token | Cost per miss |
|---|---|---|
| Ours (no FETCH) | 5.14 ms | 0.195 ms (68 GB/s, the host's CPU path) |
| FreeToken offload | 3.20 ms | 0.297 ms (45 GB/s over the link) |

This assumes FreeToken's miss counts equal ours. The lines cross near 19 misses per token, and at 40% there are 16.

**Fix 1: the slot maps on the GPU.** The profile's 81 small host-to-device copies per token (9 in stock llama.cpp)
were the maps input. Placed on the CPU by the scheduler, each per-layer view of it was copied synchronously before
every launch. With the maps on the GPU, launch time falls from 0.61–0.72 to 0.19–0.21 ms, a gain of +3.6% at C51.

**Admission tuning:** at most 1% overall in simulation, not pursued.

**Result on a 7950X host** (CPU 46 GB/s ≈ link 47 GB/s): ours v2 against FreeToken in the same session:
- **C14:** +2.3%
- **C32:** −4.3%
- **C51:** −16.9%, or −8.4% with FETCH, which helps on this host.

Two of the four predictions failed; `prereg/fix40_outcome_074.md` has the outcome. On a host where the CPU path is no
faster than the link, our per-miss advantage is gone and the fixed cost decides.

**The next item is the server's 2.0–2.8 ms per token between decode steps** (ec-bench spends 1.2 ms). Job 074b tests
GPU-side sampling.

**60% of experts needs 37 GB of slots,** so it is prepared for an RTX PRO 6000 (job 075, not yet launched).

## 29 September, midday: the same-machine comparison, redone (job `073_samehost_v2@vast`)

The protocol and outcome are in `prereg/samehost_v2_outcome.md`.

**Setup:**
- gpt-oss-120b on a 9950X3D + RTX 5090;
- 30 AIME-25 problems, 256 tokens, greedy decoding;
- a held-out warm-up, then a session over the 30 problems;
- 3 launches per configuration; paired-bootstrap CIs over problems.

**Result:**

| Experts on the GPU | ours+FETCH | FreeToken's best backend | Ratio (95% CI) | llama.cpp |
|---|---|---|---|---|
| 11% | 58.0 | 52.3 | **+10.9% [+9.8, +12.0]** | 31.7 |
| 25% | 88.0 | 81.1 | **+8.4% [+7.6, +9.2]** (+7.1% against FreeToken's best tuned setting) | 37.3 |
| 40% | 117.7 | 125.1 | **−5.9% [−6.7, −4.9]** (ours without FETCH: −2.2%) | 44.5 |

- **Both are 2–3× llama.cpp** at 25–40% of experts.
- **The earlier +18–21 / +9–11 / +2–3% (jobs 064/067) shrink or reverse** under the stricter protocol and on a host
  with faster memory. The claim is now "ahead at 11–25%, behind at 40%".
- **Launch-to-launch spread is 0.1–0.4%**, so the spread across problems carries the uncertainty.
- **Warming up on the measured problem made no difference** (−0.9 / −0.3%).
- **Greedy text is not shared across engines.** CPU and GPU kernels round differently, so each engine decodes its own
  continuation.

## 29 September, late morning: deferring admissions fails; where G really goes (job `072_defer@vast`)

**Host.** Ryzen 9 7900 (12 cores, 10 helpers) + RTX 5090. Host memory: CPU / link / both = 45.3 / 44.6 / 49.0 GB/s.

**The law, predicted blind on the machine:** C14 / C32 / C56 +4.9 / +8.2 / +5.8%, FETCH +4.4%, PREFETCH −3.2%,
llama.cpp `-ncmoe 27` +3.7% (26.3 tok/s).

**Deferring the admission copies** until after the next launch **loses 3–4%** at every budget (C32: 61.5 → 59.3
tok/s). The prediction failed; `prereg/homepc/defer_prediction_072.md` has the outcome.

**The new host-side timer explains why.** In steady state the boundary between tokens is ~1 ms, not the 2–5 ms the
profiler showed:
- graph launch 0.5–0.8 ms;
- the cache's post-processing 0.25 ms;
- inputs 0.03 ms.

Tracing ~1400 graph nodes and ~80 small copies per token had inflated it. Deferring moved ~0.8 ms of admission reads
into the CPU phases and saved only 0.2–0.4 ms at the boundary.

**G on this host.** It is 5.4–6.0 ms per token:
- ~3.65 ms of GPU work: 36 layers × ~90 µs, plus the 0.35 ms head;
- ~1 ms of host launch and bookkeeping;
- ~0.5–1 ms of admissions competing with the CPU phases (less with FETCH, which admits almost nothing).

Nothing in G is free to remove. The largest parts are batch-1 GPU efficiency and the per-token launch.

## 29 September, morning: profiles, and the law predicted on three new machines (job `069c`)

**Setup.** Job 069c ran on three RTX 5090 hosts. Before any model run, each machine wrote the law's prediction
(`law_prediction.json`) from its own bandwidth probe, using constants frozen in the public `gpu` branch. Profiles were
summarised on the machine: 069b's raw exports, 13 MB, were past the log channel's 20000-line limit and were lost.
Scores: `scripts/law_crosshost.py` → `prereg/homepc/law_crosshost.json`.

| Host | Host memory, CPU / link / both (GB/s) | Cache C14 / C32 / C56 (tok/s) | Law error | llama.cpp `-ncmoe 27` |
|---|---|---|---|---|
| Core Ultra 7 270K (8P+16E, DDR5) | 61.0 / 35.7 / 61.7 | 51.8 / 77.8 / 113.5 | +0.6 / +3.7 / +1.0% | 16.0 tok/s, law +119% |
| EPYC 7352 (Zen 2, DDR4, PCIe 4) | 39.6 / 26.2 / 52.7 | 38.2 / 59.5 / 89.6 | −2.8 / +1.9 / +3.2% | 22.0 tok/s, law +10% |
| EPYC 9655 (Zen 5, 12-channel DDR5) | 559 / 26.0 / 418 | 121.9 / 128.0 / 147.6 | +19 / +32 / +26% | 112.9 tok/s, law +16% |

**Where the law holds:** the two machines of desktop-like bandwidth, within 4% for the cache's base configurations,
predicted blind.

**Where it fails, each case explained:**
1. **FETCH is overpredicted by ~12% on every host with a slow link** (the 270K, the EPYC 7352, and earlier the X3D
   host at C14). The law has no term for FETCH's per-layer copy latency.
2. **llama.cpp on the hybrid-core Intel runs at 45% of the law.** Its CPU threads split each matrix evenly across P-
   and E-cores. The cache's helpers claim chunks dynamically and stay on the law.
3. **On the 12-channel EPYC, host bandwidth stops being the limit.**
   - A layer with one or two CPU misses takes 80–200 µs, about 100 GB/s effective, not 559 GB/s: the per-layer
     hand-off and the 44-thread split have a latency floor that the law lacks.
   - Even so, this machine runs gpt-oss-120b at 122–148 tok/s with 11–44% of experts on the GPU, and llama.cpp at
     113 tok/s. It is the "your RAM decides" thesis at its extreme.

**What the profiles show about G** (the GPU-side time, ~5 ms per token). Each C = 32 token splits into:
- ~3.3 ms of GPU work inside the layers (36 × 90 µs);
- 0.35 ms for the output head;
- a gap between tokens of 2.2 ms (270K), 3.4 ms (EPYC 9655) or 5.5 ms (EPYC 7352).

In that gap, the admission copies of the step just finished (zero-copy kernels) saturate the link. The next token's
input uploads then crawl behind them: 81 small host-to-device copies per token, against 9 in stock llama.cpp. After
that, 0.3–0.9 ms of host work comes before the next launch. Job 072 tests issuing the admissions after the next
launch.

## 29 September, morning: what routing foresight is worth (CPU only, 9 models)

**Question.** After FETCH, PREFETCH and overlap, the largest remaining byte gap is the cache policy itself. For
gpt-oss-120b at C = 32:
- the deployed policy reads 45.9 experts per token from host memory (34.9 on the critical path);
- the best online policy without foresight reads 29.6, all on the critical path (fetch-admit: an admitted miss is
  copied once and run on the GPU);
- Belady's optimum with bypass reads 16.4.

How much of that gap can knowledge of future routing close?

**Method.** `scripts/foresight.py` measures a policy that sees the next W decode steps: Belady within the window,
decayed frequency beyond it. Traces are own sampled text (arm S) of 9 models, cache carried across conversations, at
12.5 / 25 / 50% of experts per layer. It is a heuristic, so its curve is a lower bound on what W tokens of foresight
are worth. Output: `prereg/foresight/foresight_S.json`, figure `figures/foresight.pdf`.

**Results.**
- **The optimum reads 22–56% fewer experts than the best online policy** (median 40%) across the 29 model × budget
  points.
- **The foresight needed scales with the cache's turnover time C/k** (slots per layer ÷ experts per token). Half the
  gap closes at W50 ≈ 0.5 (C/k)^1.4 tokens: log-log r = 0.975 over 26 points, median W50 / (C/k) = 0.73. 90% of the
  gap needs about 2.5 C/k.
- **Small caches (C/k ≤ 2):** one or two tokens of foresight recover half or more. OLMoE, gpt-oss-20b, Mixtral,
  Phi-3.5 and Qwen2-57B at 12.5–25% are in this regime.
- **Large caches:** gpt-oss-120b at 25% (C/k = 8) needs about 10 tokens; at 50%, about 33.
- **Next-layer prediction gives zero tokens of cross-token foresight.** That is why PREFETCH raises the hit rate but
  never lowers host bytes.
- **Per-conversation hindsight placement** (the C experts each conversation uses most) captures part of the gap only
  for gpt-oss-120b at 25–50%. For the others it is no better than online.

**Robust to how the trace was made.** The same fit on the greedy own-text traces (arm G) and on teacher-forced
dataset text (arm D), 9 models and 26 points each:

| Arm | W50 fit | r | Optimum ÷ online reads |
|---|---|---|---|
| S (sampled own text) | 0.51 (C/k)^1.39 | 0.975 | median 0.61 |
| G (greedy own text) | 0.51 (C/k)^1.38 | 0.975 | median 0.61 |
| D (dataset text) | 0.51 (C/k)^1.43 | 0.978 | median 0.62 |

Files: `prereg/foresight/foresight_{S,G,D}.json`. Unlike hit rates (the provenance result), the foresight horizon
barely depends on provenance.

**In seconds, gpt-oss-120b at C = 32 on the 9950X #2 host** (law, both paths at 51.8 GB/s): the best online policy
takes 12.4 ms per token (81 tok/s), the optimum 9.0 ms (111 tok/s). Foresight is worth up to +37% there. No
engineering of paths or overlap can recover it.

**Prior art checked.**
- **2608.07911 (closest).** It already splits the online-to-optimal gap and finds 84–97% of it is knowing which
  resident expert is used furthest in the future. The gap grows with its regime ratio: per-step expert union ÷
  capacity, the inverse of C/k. A trained causal next-use predictor recovered −11% of the gap. It studies neither a
  limited lookahead nor the horizon's scaling (3 models).
- **Read-ME (NeurIPS'24)** makes routing known ahead by decoupling the router, then applies Belady.
- **ExpertFlow and SpecMD** predict one layer or one batch ahead.
- **Limited-lookahead caching in general:** Hasslinger et al. (2018, web caching) and Albers (1997, paging theory).

What is new here: the horizon curve, W50 ≈ 0.5 (C/k)^1.4 across 9 models, and its conversion to seconds.

## 29 September, morning: the overlap test failed (job `071_overlap_x3d@vast`)

**Host:** RTX 5090 + Ryzen 9 9950X3D2 (two CCDs, 192 MB L3). CPU read 69.6 GB/s; CPU + copy engine 86 GB/s.
**Run:** own text, 12 × 128 tokens. Predictions were recorded before the run
(`prereg/homepc/overlap_prediction_071.md`, outcome appended).

| Config | C = 14 | C = 32 | C = 56 |
|---|---|---|---|
| base (tok/s) | 55.8 | 84.0 | 121.2 |
| PREFETCH (early) | +13.9% | +11.0% | +3.7% |
| PREFETCH_LATE | +8.3% | +3.4% | −3.0% |
| LLC (warm the next layer's CPU experts into L3) | −1.8% | −2.5% | −5.0% |
| FETCH | +7.2% | +7.6% | +4.4% |
| LLC + FETCH | −8.4% | −10.1% | −13.6% |

- **The two overlap predictions failed.** Late copies are slower than early ones, and L3 warming loses. The other
  predictions held:
  - the law on the configurations it covers: median 1.8%, worst 11.2%;
  - no configuration beats its bound;
  - top-1 agreement ≥ 98.4%, NLL within 0.6%.
- **Why.** Host memory is not idle during the GPU's work:
  - background admissions read 100–145 MB per token in exactly those windows;
  - early PREFETCH already puts predicted copies there, and that is its whole gain.
  - What is left per layer is about 130 µs, half an expert. The new mechanisms only added competing reads.
- **Consequence for the paper.** The "+40–55% overlap dividend" of the entry below is withdrawn. The overlap stays in
  the paper as a bound-level quantity: the max-form bound, and what early PREFETCH realizes of it.
- **Next.** The profile (job 069b, running) splits G into its parts. G is now 40% of the token time at C = 32 and 58%
  at C = 56, the largest term the engine controls.

## 29 September, early: a law for offloaded decode on a PC

**The law** (`scripts/law_hostdram.py`, results in `prereg/homepc/law_hostdram.json`):

  time per token = G + max(CPU bytes / B_c, link bytes / B_p, (CPU + link bytes) / B_both)

- CPU bytes and link bytes are the expert bytes read from host memory per token, taken from the engine's own counters.
- The bandwidths are each host's own microbenchmarks (`concur.cu`), not fitted.
- G is one constant per engine, fitted on the 9950X #2 host: 4.82 ms for the cache, 5.06 ms for llama.cpp.

**Out-of-sample checks:**

| Host | CPU / link / both (GB/s) | Configurations | Median error | Worst |
|---|---|---|---|---|
| 9950X #1 (job 059) | 62.2 / 53.0 / 73.0 | 9 (cache, serial FETCH, llama.cpp) | 2.2% | 3.3% |
| 9800X3D (job 068) | 47.5 / 46.4 / 51.5 | 6 (cache, FETCH) | 3.2% | 5.5% |
| 9950X + RTX PRO 6000 (job 070, a different GPU board) | 64.7 / 53.0 / 74.2 | 3 | 5.7% | 7.3% |
| 9950X #2 (fit host) | 46.7 / 47.3 / 51.8 | 24 (cache, FETCH, PREFETCH, both, llama.cpp) | 3.1% | 8.4% |

**Readings:**
1. **Every mechanism acts only through host-memory bytes per token and the number of read paths in use.** A second
   path gets 11–21% more bandwidth; that is the whole gain of FETCH, PREFETCH and FreeToken-style splits. Prefetch
   never reduces the bytes.
2. **The host lottery is the RAM.** The 19–22% gap between "identical" 9950X rentals is 62 vs 47 GB/s of host memory
   bandwidth.
3. **G and the host-memory term add up.** *(Corrected after job 071, below.)* This entry claimed that host memory is
   idle while the GPU works and that overlapping would give +40–55% at C = 32. That was wrong: the cache's background
   admissions and early PREFETCH already use those windows.

**All in VRAM (job 070, RTX PRO 6000 Blackwell, the 5090's die with 96 GB):**
- gpt-oss-120b decodes at **265 tok/s** in llama.cpp (llama-bench 263). That is about 54% of the GPU's datasheet
  bandwidth for the bytes a token reads.
- Of its ~3.8 ms per token:
  - about 2.8 ms is the quantized matrix-vector kernel;
  - most of the rest is activation quantization (0.4 ms), norms (0.2 ms) and top-k by full argsort (0.2 ms);
  - plus a 0.15 ms host gap between two graph launches per token.
- Offloaded (C = 32) on the same machine: cache 79.0, +FETCH 89.0, llama.cpp `-ncmoe 27` 35.4 tok/s.

**Lost run:** the first profile of the offloaded configurations (job 069) was lost. Its machine was destroyed after a
failed log fetch. It is rerunning on a 9950X3D host, with the profiler exports fixed.

## Where things stand (end of 28 September)

- **Same machine, same client, equal GPU memory, gpt-oss-120b:** our cache with FETCH is ahead of FreeToken at all
  three shared budgets on two RTX 5090 machines: +18–21% at 11% of experts on the GPU, +9–11% at 25%, +2–3% at 40%.
  - The last is within the spread across prompts.
  - Against llama.cpp at its best setting it is 1.9× / 2.7× / 3.2–3.3×.
  - It also runs 44%, which FreeToken cannot fit.
- **PREFETCH works and behaves as simulated.** Its speed gain is smaller than FETCH's, and it only helps together with
  FETCH at small budgets on own text.
- **FETCH and PREFETCH both lose 16% on a PCIe 4.0 machine.** They should be switched on from measured bandwidths.
- **Spend:** $7.62 of the $25 tranche by Vast's own balance ($17.38 left). Our ledger estimated $4.40: it counts
  the hourly GPU price but not storage and download charges, so from now on the balance is the number to trust.
- **Open next:**
  1. Auto-select FETCH / PREFETCH from a start-up bandwidth probe.
  2. Why both gain less on short chat prompts than on long own-text runs.
  3. A second model (Qwen3-30B-A3B).
  4. A PCIe 5.0 machine with faster memory (like job 059's, 62 GB/s).

## 28 September, night: a second machine, RTX 5090 + Ryzen 7 9800X3D (jobs `067` and `068`)

**The machine:** a common gaming configuration, 8 cores on one CCD. CPU memory reads 47.6 GB/s with 8 threads; the
link 46.9 GB/s.
- Job 067 measured llama.cpp and FreeToken. Our patched tree failed to fetch llama.cpp (a network reset), so
  `setup.sh` now retries.
- Job 068 measured our cache on the same machine in a second session.
- The settings are unchanged from job 066. The FETCH table was chosen on the 9950X machine, so this run is out of
  sample for it.

Chat benchmark (FreeToken's method, AIME-25 problems 0–4 × 256 tokens, equal GPU memory for experts), tok/s:

| Experts on the GPU | llama.cpp | Our cache | **Our cache + FETCH** | FreeToken offload | FreeToken hybrid |
|---|---|---|---|---|---|
| 11% | 25.1 | 44.6 | **48.3** | 39.0 | 39.9 |
| 25% | 29.2 | 73.6 | **77.5** | 69.8 | 65.7 |
| 40% | – | 111.9 | **114.7** | 112.9 | 94.4 |
| 44% | 38.9 | 124.3 | **124.3** | does not fit | does not fit |

- **Same picture as the 9950X machine.** With FETCH, our cache is +21% / +11% / +2% against FreeToken's better mode;
  on the 9950X machine it was +18% / +9% / +3%.
- **Without FETCH:** +12% / +5% / −1%.
- **Against llama.cpp at its best:** 1.9× / 2.7× / 3.2×.
- **Paired prompts across machines.** The cache's hit rates are identical to the 9950X machine's to six digits (0.551265
  at C = 14). With a fixed seed on the same GPU model the sampled text, and so the routing, is the same, so the two
  machines saw exactly the same work.
- **Own text:** cache 42.1 / 65.2 / 98.6 tok/s; with FETCH 48.1 / 76.3 / 111.2 (+14% / +17% / +13%).

## 28 September, night: PREFETCH on the RTX 5090 host (job `066_prefetch_homepc@vast`)

Same machine as jobs 060 and 064 (RTX 5090 + Ryzen 9 9950X; CPU memory 46.5 GB/s, link 46.5 GB/s).

### Own text: 12 prompts × 128 tokens, every configuration in one process

| C of 128 | Cache | + PREFETCH q=1 | + PREFETCH q=2 | + FETCH | + PREFETCH q=1 + FETCH |
|---|---|---|---|---|---|
| 14 | 40.7 tok/s | 46.2–46.5 (**+14%**) | 45.3 (+11%) | 47.1 (+16%) | **49.7 (+22%)** |
| 32 | 62.7 | 72.7 (**+16%**) | 71.5 (+14%) | 74.7 (+19%) | **76.8 (+22%)** |
| 56 | 93.6 | 102.4–103.2 (**+10%**) | 102.0 (+9%) | **108.8 (+16%)** | 106.1 (+13%) |

- **The hit rates are as simulated:**
  - with q=1: 0.746 / 0.887 / 0.947, against simulated 0.746 / 0.886 / 0.948;
  - without prefetch: 0.541 / 0.758 / 0.875.
- **Prediction against measurement.** The prediction, recorded before the run in
  `prereg/homepc/prefetch_prediction_066.json`, was +17% / +17% / +14%. Measured is +14% / +16% / +10%: the timeline
  model is 1–4 points optimistic.
- **Repeats agree** within 1 point; the base is identical in both runs.
- **Accuracy:** 98.4–99.0% same next token as the cache alone; NLL within 0.6%.
- **What wins:**
  - On this host FETCH alone (+16–19%) beats PREFETCH alone (+10–16%).
  - Both together is best at the two smaller budgets (+22%). At the largest, FETCH alone is best (+16%).
  - Two prefetches per layer is worse than one everywhere, as the simulation said.

### The chat benchmark: job 064's client, AIME-25 problems 0–4 × 256 tokens, `llama-server`

| Budget | Cache (job 064) | + PREFETCH | + FETCH | FreeToken, better mode (job 064) |
|---|---|---|---|---|
| 11% (C = 14) | 42.9 tok/s | 44.6 (+4%) | **46.3 (+8%)** | 39.4 |
| 25% (C = 32) | 68.1 (68.1 again in job 066) | 72.5 (+6%) | **74.9 (+10%)** | 68.6 |
| 40% (C = 51) | 105.2 | 103.2 (−2%) | **112.3 (+7%)** | 108.9 |
| 44% (C = 56) | 114.8 | 111.6 (−3%) | **119.8 (+4%)** | does not fit |

- **With FETCH, our cache is ahead of FreeToken at all three budgets on this host:** +18% / +9% / +3%. The last is
  within the ±10% spread across problems.
- **Against llama.cpp at its best:** 1.9× / 2.7× / 3.3×.
- **The cache-only server run at C = 32 reproduced job 064 to 0.1%.**
- **Both options gain less on the chat benchmark than on the own-text runs.**
  - A likely reason: these prompts are short (109 tokens, against 640+ in the own-text runs), so each layer's
    attention is quicker. That leaves less GPU time for a copy to hide behind.
  - The hit rates with PREFETCH are similar to own text (0.734 / 0.886 / 0.955).

### Consequences

1. **PREFETCH is correct and does what the simulation says to the hit rate.** Its speed gain is real but smaller than
   FETCH's on this host, and it only pays together with FETCH at small budgets.
2. **Both options depend on the host.**
   - On a PCIe 4.0 host (job 065) both lost 16%.
   - A deployable version should pick them from measured bandwidths, as FreeToken's `ft bench bw` calibrates its
     hybrid mode.
3. **The chat-benchmark claim, on this host:** our cache with FETCH is 3–18% faster than FreeToken at equal GPU
   memory, and runs a budget FreeToken cannot fit.

## 28 September, late: same host, same client (job `064_samehost_rerun@vast`)

**Setup:**
- One machine: RTX 5090 + Ryzen 9 9950X, job 060's host.
  - Measured: CPU memory reads 46.6 GB/s with 16 threads (48.6 at best), the link 47.6 GB/s, both at once 52 GB/s in
    total; GPU memory 1,558 GB/s.
- **One client times every system.** It is FreeToken's own benchmark code (`benchmarks/bench_decode_moe.py`, commit
  0d652e7): its AIME-25 prompts, sampling, warm-up request and tok/s formula, over 5 problems × 256 tokens.
  - Every system runs as a server with the OpenAI chat API.
  - Our cache runs in `llama-server`.
- **Equal GPU memory for experts:** llama.cpp `-ncmoe 32 / 27 / 20`, cache C = 14 / 32 / 56 of 128 (and 51),
  FreeToken `--moe-cache-rate` 0.111 / 0.25 / 0.40.
- **Check on the client:** on the same problem (rate 0.25, problem 0), FreeToken's unmodified script gave 67.2 / 67.1
  tok/s for offload / hybrid, and our client 68.1 / 66.6. That is within 1.3%, as in job 062 (within 2%).

| Experts on the GPU | llama.cpp | **Our cache** | FreeToken offload | FreeToken hybrid |
|---|---|---|---|---|
| 11% | 24.3 tok/s | **42.9** | 38.5 | 39.4 |
| 25% | 28.2 | **68.1** | 68.6 | 66.2 |
| 40% (C = 51) | – | **105.2** | 108.9 | 99.2 |
| 44% (C = 56) | 36.4 | **114.8** | does not fit | does not fit |

- **Against FreeToken, we are even:** +9% at 11%, −1% at 25%, −3% at 40% against its better mode.
  - Every system varies by about ±10% across the five problems (sampled text routes differently), so only the 11%
    difference is clearly outside the noise.
  - FreeToken does not fit 44%: its KV cache runs out of room, even at `--memory-ratio 0.95`.
- **Against llama.cpp:** 1.8× / 2.4× / 3.2×.
- **Speed limit on this host (corrected 29 Sep).**
  - The first version (174 / 313 / 313 tok/s) counted the non-expert weights at 16 bits (3.13 GB instead of the GGUF's
    1.69 GB). That is the same kind of error already fixed for the A10 on 28 Sep, reintroduced in the new home-PC
    script.
  - It also used the draft's layer-structured bound, which the reviews showed is not valid for every policy.
  - Corrected: resource-form bound, pooled budget, GGUF byte counts.
    - Physical, on datasheet bandwidths (the RAM speed of the rental is unknown, so a DDR5-3600 to 5600 band):
      135–210 / 347–519 / 511–519 tok/s. All experts resident: 511–519.
    - On measured bandwidths (a reference, not a floor): 123 / 316 / 444.
  - Against the physical band, the cache with FETCH reaches 22–35% / 14–21% / 21% (own text); llama.cpp 11–18% /
    5–8% / 7%.
- **The host reproduces.** The own-text runs repeated job 060's on the same machine within 0.8%:
  - cache 41.0 / 62.7 / 94.0 tok/s, with FETCH 46.9 / 74.3 / 108.0;
  - llama.cpp inside our harness 24.1 / 28.0 / 36.4.
- **Missing:** our cache with FETCH in the server. It aborted at start-up, because the cache also started during
  llama-server's memory-fit pass, when the weights are not loaded. Fixed (commit 33290cf); job 066 runs it.

## 28 September, late: predicting the next layer's experts, and PREFETCH

Details: `research_notes/MoE offload system race plan/prefetch_lookahead.md`.

### Lookahead (job `063_lookahead@vast`)

**The next layer's experts are predictable from the current layer.** On gpt-oss-120b, layer l+1's router applied to
the residual after layer l's attention finds 84% of layer l+1's selected experts in its top 4 and 97% in its top 8
(gpt-oss-20b: 87% and 98%). A host recomputation of every layer's own routing matched 100%, which checks the
extraction.

### Simulated benefit

- **The cache simulation reproduces the measured hit rates of job 060** (0.542 / 0.757 / 0.876 against 0.541 / 0.758 /
  0.875).
- **With a prefetch of one predicted expert per layer, CPU misses fall 45% / 53% / 58%** at C = 14 / 32 / 56.
- **A layer timeline with the measured host-1 constants gives +19% / +18% / +13% tok/s**, against +2–6% measured for
  FETCH on that host. The timeline reproduces the measured base within 2.7%.
- **This needs the copy on a second GPU stream.** On the main stream the model predicts about +0%.

### PREFETCH, built and checked (job `065_prefetch_check@vast`)

- **Implementation** (llama.cpp branch commit 6dd2d1f): `LLAMA_EC_PREFETCH=q`. The next layer's router is precomputed
  with the norm ratio folded in, a plan op updates the next layer's maps, the copy runs on a side stream, and the next
  layer joins it just before its GPU experts.
- **First GPU run:** gpt-oss-20b, C = 8 of 32, RTX 4500 Ada + Ryzen 9 7900X, PCIe 4.0.

| | Cache | + PREFETCH q=1 | + PREFETCH q=2 | + FETCH | + PREFETCH q=1 + FETCH |
|---|---|---|---|---|---|
| Hit rate (simulated) | 0.639 (0.639) | **0.816 (0.816)** | **0.884 (0.885)** | 0.682 | 0.826 |
| Prefetches per layer-step, used (simulated) | – | 0.722, 0.606 (0.721, 0.605) | 1.087, 0.872 (1.086, 0.871) | – | 0.712, 0.595 |
| tok/s | 61.5 | 51.4 (−16%) | 43.2 (−30%) | 51.9 (−16%) | 41.1 (−33%) |
| Same next token as the cache | – | 99.0% | 99.1% | 99.0% | 99.3% |

- **Correct:**
  - With `LLAMA_EC_CHECK=1`, the ids the graph used equal the host's maps on all 1,536 steps.
  - CUDA graphs on and off give identical output.
  - `compute-sanitizer` memcheck reports 0 errors.
  - NLL is within 0.3%.
  - Hit rates and prefetch counts match the simulation to 0.001.
- **Slower on this host, as FETCH is.** Its GPU link is PCIe 4.0 (about 25 GB/s), half its memory bandwidth, so moving
  an expert over the link costs more than computing it on the CPU. Both FETCH and PREFETCH are for hosts whose link is
  about as fast as their memory (PCIe 5.0 with a desktop CPU). **Job 066** tests PREFETCH on the RTX 5090 host.

## 28 September, night: FETCH A/B, FreeToken attempt, same-host comparison started

### FETCH A/B (job `060_fetch_ab_homepc@vast`, another RTX 5090 + Ryzen 9 9950X host)

Own text, 12 prompts × 128 tokens, every configuration in one process on one loaded model:

| Budget (C of 128) | Cache, no FETCH | FETCH, copy overlapped with the CPU | FETCH, copy first (job 059's form) | Same next token as no FETCH |
|---|---|---|---|---|
| 14 | 40.8 tok/s | 46.6 (**+14.2%**) | 46.0 (+12.8%) | 98.8% |
| 32 | 62.5 | 73.7 (**+18.0%**) | 73.2 (+17.1%) | 98.8% |
| 56 | 93.8 | 107.3 (**+14.3%**) | 106.8 (+13.8%) | 98.7% |

- **Repeats agree within 0.4%** (each configuration run again in reverse order).
- **The overlap itself adds only 0.5–1.4 points.** The serial form, which gave +2–6% on job 059's host, gives +13–17% here.
  So the benefit of FETCH depends on the machine far more than on the overlap.
- **This host is 19–22% slower without FETCH** than job 059's (40.8 / 62.5 / 93.8 against 52.6 / 79.0 / 116.5 tok/s), on
  the same hardware class. Its bandwidths were not measured. A likely reading is slower main memory, which makes a
  CPU miss costlier and an over-the-link copy relatively cheaper; job 062 measures this.
- **Tables:** `0,1,1,2,3` and `0,1,1,2,2` are equal within 0.3%; `0,0,1,1,2` (fetch less) is weaker at every budget.
- **Accuracy:** NLL within ±0.4% of no FETCH.

**Consequence for the method:** numbers from different rented hosts are not comparable, even for the same CPU and
GPU model. From now on every job measures its own host (STREAM, `bw.cu`, `concur.cu`), and systems are compared only
within one host.

### FreeToken, first attempt (job `061_freetoken_homepc@vast`, RTX 5090 + Ryzen 9 9950X3D)

- **Failed:** the model folder was empty. `hf download … --exclude "original/*" "metal/*"` read `metal/*` as a file
  name. Fixed in job 062 (Python `snapshot_download` with `ignore_patterns`, and a check for `config.json`).
- **FreeToken's own link measurement ran:** CPU MoE read 36.6 GB/s for MXFP4, PCIe 27.0 GB/s. Its hybrid mode would
  fetch 53% of misses. This host's link is slow (28.9 GB/s, against 57 GB/s on job 059's host), so it was not used again.

### Running now

1. **Job 062 (same host, same client).**
   - Stock llama.cpp, our cache (with and without FETCH) and FreeToken (offload and hybrid) run on one RTX 5090 +
     9950X host (link 45.6 GB/s by Vast's figure), at equal GPU memory for experts.
   - All are timed by one client built on FreeToken's own benchmark code: their AIME-25 prompts, sampling, warm-up and
     tok/s formula, 5 problems × 256 tokens.
   - Their unmodified script runs once as a check on the client.
   - Our cache runs in `llama-server` for this, which needed `-ot` to accept pinned host memory (`CUDA_Host`).
2. **Job 063 (lookahead).** How well can a layer's experts be predicted from the hidden state one layer earlier?
   gpt-oss-20b and 120b, own text. `ec-bench --lookahead` records the actual routing and the top-8 predictions (the
   next layer's router applied to the residual after this layer's attention, or to this layer's output). A host
   recomputation of each layer's own routing checks the arithmetic; it matched 100% on two tiny models.
   `scripts/sim_prefetch.py` turns the recording into misses under the deployed cache plus a prefetch.

## 28 September, evening: first measurements on a home PC

### What was done

1. **Our system was updated to current llama.cpp** (commit 4da6337, 27 Sep) with no conflicts.
2. **Three bugs were fixed and tested:**
   - experts running in BF16 could leave stale numbers behind;
   - a rare timing slip in the GPU-to-CPU mailbox;
   - the paper's case-study speed limit left out the model's final output layer. Fixed in the paper: our system
     now reads 51–72% of the limit on the A10 (was 47–64%), llama.cpp 36–50% (was 33–45%). Erratum recorded.
3. **A new feature, FETCH:** when some of a layer's experts are missing from the GPU, copy a few of them over at
   once and run them on the GPU, while the CPU runs the rest. A trace simulation suggested this for home PCs,
   where the GPU link is about as fast as main memory.
4. **A safe way to rent machines:** jobs run on Vast.ai without ever putting your GitHub key on the machine.
   Results come back through Vast's log feature.
5. **"How our system differs"** was written up for you: `reports/How our system differs.md`.

### Results

**Checkpoint 1: does it build and give the same answers?** (RTX 5080 + Ryzen 9 7900X, gpt-oss-20b, 12 prompts ×
128 tokens, job `058_ckpt1_dev@vast`) — **passed.**

| | tok/s | Same next-token pick as llama.cpp | Error on the text (NLL) vs llama.cpp |
|---|---|---|---|
| llama.cpp, 25% of experts on the GPU (`--n-cpu-moe 18`) | 18.5 | — | — |
| **Our system, same GPU memory** | **40.3 (2.18×)** | 99.5% | +0.03% |
| Everything on the GPU (reference) | 246.6 | 99.2% | −0.05% |

Op tests: 929/929 standard expert matmul cases and, in job 059, 28/28 of our cases (all weight types including BF16),
with 0 memory errors.

**Checkpoint 2: where do we stand on a home PC?** (RTX 5090 + Ryzen 9 9950X, 126 GB DDR5, gpt-oss-120b, the model's
own text, 12 prompts × 128 tokens, job `059_ckpt2_homepc@vast`)

*Correction, 29 Sep: the "speed limit" column below and the percentages derived from it used overstated dense bytes and the
draft's layer-structured bound. On this host's measured bandwidths the corrected reference is 180 / 464 / 484 tok/s:
our system 29% / 17% / 24%, llama.cpp 17% / 8% / 10% (`scripts/analyze_homepc.py`).*

| GPU memory for experts (llama.cpp setting) | llama.cpp, best of `llama-bench` and our harness | **Our system** | Our system + FETCH (copy, then hand over) | Speed limit on this machine |
|---|---|---|---|---|
| Small (`-ncmoe 32`, 14 of 128 experts per layer) | 31.2 tok/s (15% of limit) | **52.6 (1.68×, 25%)** | 53.5 (+1.8%) | 210 |
| Middle (`-ncmoe 27`, 32 per layer) | 36.3 (11%) | **79.0 (2.17×, 23%)** | 83.8 (+6.0%) | 341 |
| Large (`-ncmoe 20`, 56 per layer) | 46.7 (14%) | **116.5 (2.49×, 34%)** | 120.3 (+3.2%) | 341 |

- **Same answers:** 98.6–99.0% of next-token picks match llama.cpp; NLL within −0.75% to +0.14%.
- **The harness agrees with stock llama.cpp:** llama.cpp measured inside our harness is within 1–3% of stock
  `llama-bench` (30.4/36.2/46.4 vs 31.2/36.3/46.7).
- **The paper's decode model held up on this new machine.** On the audit's datasheet basis it predicted llama.cpp at
  25.6/29.3/36.8 tok/s; measured is 1.22–1.27× that, inside its error band, and the same ratio as on the A10.
- **The machine:**
  - The CPU reads memory at 62 GB/s; the GPU link moves 57 GB/s (copy engine) or 53 GB/s (kernel reading host memory).
  - Both at once total 73–77 GB/s, so neither path alone saturates the memory.
  - One gpt-oss-120b expert (13.25 MB) takes ~230–250 µs to copy, about the same as the CPU needs to run it
    (~230–240 µs per expert, measured from the helpers).
- **For comparison:** a community run of FreeToken on a 5090 reported 127 tok/s with 40% of experts on the GPU (host
  not stated). Our 116–120 tok/s at 44% is in the same range; a same-machine comparison is still to do.

### What this means

- **Our system is well ahead of properly set-up llama.cpp on a home PC:** 1.7–2.5× on gpt-oss-120b, with the same
  answers. Checkpoint 2's rule (≥ 1.25× at every budget) is met.
- **Every system is still far from the limit (15–35%).** The biggest gap is the cache's hit rate. The best possible
  policy would miss 0.95 / 0.39 / 0.14 experts per layer and token at the three budgets, against our ~1.8 / 1.0 / 0.5.
- **FETCH gave only +2–6% as first built,** because the copy ran before the CPU started, so the two didn't overlap.
  It is now changed to overlap them (job 060, running).

### Spend

About $1.40 of the $25 so far (ledger estimate in `gpu/vast_ledger.json`).

### Next

1. Job 060: the overlapped FETCH A/B (three tables, plus the serial form, and repeats in reverse order).
2. FreeToken on the same class of machine. It needs the Hugging Face weights (another ~65 GB download) and a CUDA 13
   image.
3. Qwen3-30B-A3B in BF16 as the second model (61 GB).
4. Hit rate: the gap to the best possible policy is the largest remaining one.

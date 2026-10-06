# Review 6b: "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"

Reviewer: professor, performance modelling and scientific benchmarking of parallel and ML systems. This is a blind,
independent review, calibrated for MLSys. I read the paper (28 pages) and the supplement (23 pages). I re-ran the
central analysis scripts on a scratch copy of the repository against the GPU results on the `gpu` branch, and wrote
small checks of my own. I modified no file except this one.

---

## 1. Summary

The paper studies batch-1 decode of two Mixture-of-Experts models, gpt-oss-120b (MXFP4) and Qwen3-30B-A3B (BF16), on
rented RTX 5090 desktops. Most experts stay in host DRAM. Each layer has C GPU slots, and a miss is served either by
the CPU or by a PCIe copy into a slot. The paper makes three contributions.

1. **A bound** on time per token for any system that runs the exact routing with C slots per layer (Eq. 1). Its inputs
   are the host reads of Belady's MIN with bypass, computed on the model's own routing trace, and the machine's best
   probed host-read rate. It is applied to the authors' llama.cpp expert cache (25–43% of the bound), to FreeToken and
   llama.cpp, and to 52 published measurements (in-class median 13.6%).
2. **An in-engine oracle factorial.** Oracles that read a recorded routing trace run inside the real engine on 3–5
   hosts at six budgets, and on a 10-host panel at two budgets. The design crosses the *admission set* (deployed vs
   MIN) with the *number of reads* per admission (CPU-serve then background copy, vs one fetch in the step). The gap is
   split with two-factor Shapley values. Prefetching is nested under MIN's set, and the remainder is a residual. A
   calibrated two-path time model (Eq. 2) is said to predict which read path pays on each host.
3. **A "price of foresight".** Window policies that see the next W tokens' routing, either exactly or degraded to
   recall = precision = r, are measured in time on the panel and in reads on traces of nine models. A one-constant
   rule, D(W50) ≈ 0.65 C distinct experts per layer, is proposed for the horizon that closes half the read gap.
   Realisable forecasters, a learned admission order and speculative batching are shown to fall far short of it.

Two features make the artifact unusual. Every GPU job's predictions were committed before the machine started: 565
clauses for jobs 073–098 and 273 for job 099, all scored, failures included. Every number in the paper is generated
by a script. I could reproduce the central tables exactly.

## 2. Strengths

- **S1. The process is unusually honest and reproducible.** Predictions were committed before each run, and the git
  order confirms it: job 099's predictions were committed at 02:18, its first results at 04:14. Clauses are scored by
  rule, and 39% of 553 scored clauses were reported as strictly held. I re-ran `panel_099.py`,
  `factorial_shapley.py`, `hostdep_model.py` and `scorecard.py`. Each reproduced the committed JSON, tables and macros
  (Section 6). Few papers in this area could survive that test.
- **S2. The machine is treated as an experimental unit, and the paper draws the right conclusion from it.** The paper
  rented a 10-host panel and gave each configuration a cold cache in a seeded random order. It then observed that
  between-host spread dwarfs the within-host problem intervals. From this it moves its claims to "orderings that hold
  on every host of a stated class, and a model that predicts each host" rather than a pooled mean. That is the correct
  instinct, and it is rarely seen.
- **S3. The oracles run inside a real engine, not only a simulator.** Engine reads match the per-host replay within
  1.6–1.7% on every panel host-cell I checked. The oracle families (MIN with bypass, Belady, paced prefetch, windows,
  degraded windows) are well chosen.
- **S4. A substantive systems insight with a physical explanation.** Foresight pays only when the cache admits what MIN
  admits *and* reads each admission once. Whether the in-step single read pays depends on the host's link-to-CPU
  read-rate ratio. The two-path model gets this crossover right. In my check (Section 6), a naive one-path model with
  the same G predicts a +27% gain on the slow-link host, where the engine actually loses 13%. The two-path form gets
  the sign right there. Cache designers can use this result.
- **S5. A roofline-style normalisation of the literature.** Each published batch-1 number is scored against its own
  hardware's bound. This is a useful corrective to "speed-up over a baseline of our choosing". The result is robust to
  aggregation: the median over the 7 sources is 14.5%, against 13.6% over the 20 rows.
- **S6. Limitations are stated plainly**, including those that cut against the paper: the bound uses the highest probe
  sample, degraded windows use independent errors, and the slow-link class has only two panel hosts.

## 3. Weaknesses (ranked, each with a concrete fix)

**W1. The price of foresight is measured against a weak baseline and on the wrong read path, so readers will
misjudge it.** Every window policy in the engine is "admit every miss + W tokens of lookahead". It therefore inherits
the in-step PCIe fetch of every admission, the read path that Section 4 shows loses on slow links. Its value is
reported as a share of MIN's gain over *admit every miss*, and the deployed cache beats that baseline on most panel
hosts (aa/base 0.42–1.00). I recomputed the panel's states against the deployed cache from `prereg/panel_099.json`:

| window (gpt-oss) | slower than the deployed cache on |
|---|---|
| exact W=4, 11% | 6 of 10 hosts (Pc, Pd, Pf, Pg, Ph, Pi; down to 0.51×) |
| exact W=4, 25% | 8 of 10 hosts |
| exact W=16, 11% / 25% | 2 / 3 of 10 hosts (Pf: 0.71× / 0.68×) |
| W=8 at r=0.5, either budget | 8 of 10 hosts |

The abstract says "8 tokens buy about what 2 exact tokens do". In absolute terms, that forecaster makes the engine
slower than having no forecaster on 8 of 10 hosts. A second problem is the degraded window, whose errors are uniform
and independent. The paper's own linear forecaster (precision 0.48 one token ahead on gpt-oss) and GRU forecaster close
2–19% of the gap, against 14–49% for the synthetic windows at r = 0.5 (Table 14). So the accuracy axis of Fig. 3 is not calibrated to any realisable
predictor. The question "how much foresight, how accurate" is the right one, but as measured the answer confounds
horizon with read path, and accuracy with error structure.

*Fix:* (a) Report every window result as a speed ratio against the deployed cache on the same host. The data already
exist. (b) Build the window family on the deployed mechanism: the κ=1 margin, with the CPU serving the miss and a
background or paced copy. Table 14's κ=1 column already shows this closes 0.34–0.88 of its own (deployed-to-MIN) read gap at
W=4. Its denominator differs from the aa-based window's 0.25–0.77, which is one more reason to report everything
against the deployed cache. Run it on two or three hosts spanning the link/CPU ratio. (c) Define forecaster accuracy on the experts
that matter, those not already resident or those whose prediction changes an admission or eviction decision, and
redo the degraded-window and forecaster comparison on that metric. This needs traces only.

**W2. The panel validation of the calibrated model is described as the frozen model of Appendix C, but the script
fits G for every host and cell from a measured run.** Appendix C defines Eq. 2 with "one constant G per engine and
model, fitted once on one host and frozen". The panel claim reads: "a calibrated model … predicts the time of the
in-step states within a median 2.2% across 14 hosts, and on every host whether the single read in the step pays."
That claim comes from `scripts/hostdep_model.py`, which sets G = (measured time of the deployed state) − (model term)
for each of the 43 host-cells. The fitted G ranges from 4.1 to 9.2 ms; it is 9.22 ms on Pi at gpt-oss 11%. The script
also takes each state's misses and fetches from that state's own run. The paper does not disclose either fact. The
"43 host-budgets" also count O4 twice, once as job 096a and once as job 097a. In the sign test, 41 of 43 cells are
gains, usually large, and the only losses are two cells on one host (Pf). A constant "fetching in the step always
pays" predictor therefore scores 41/43. Finally, the model over-predicts the background and prefetching states by a
median of 18% (24.5% for the best state, both3p). So it cannot price the configuration a designer most wants.

*Fix:* Add a fitted-where / tested-where table. The substance survives a cleaner test. With G frozen at the Appendix C
values (4.82 ms gpt-oss, 4.3 ms Qwen3), my re-run gives a median absolute error of 3.2%, a 90th percentile of 10.6%, and
the sign right on 43/43. A naive one-path model with per-cell G gets 41/43 and misses both Pf losses (it predicts
1.29× and 1.27× where the engine measured 0.875× and 0.969×). Report that comparison as the evidence for the
two-path form. Take counters from the trace simulator rather than the run, so the model predicts a configuration
before it is run. Add an overlap term, or state plainly that Eq. 2 is a model of in-step reads only.

**W3. The bound is a valid lower bound, but the paper treats it as reachable, which it has not shown.** Eq. 1 is
sound: I checked the averaging argument and the 172 tok/s value at host B. It uses B_host = 87.5 GB/s, the maximum of
six concurrent probe samples ranging from 76.8 to 87.5 (median 80.7). It also uses the datasheet GPU rate, lets the
two paths share B_host freely, and drops all latency. The best oracle state, which knows the whole future and copies
ahead, reaches only 46–73% of the bound at the host-bound budgets. In Table 3 the residual "rest" is 30–62% of the gap
on fast-link hosts and 68–99% on slow-link hosts. The paper concludes that "most of the time a consumer machine could
save is not being saved". That holds only if that slack is attainable, and the paper never separates bound slack from
engine inefficiency.

*Fix:* Run an attainability microbenchmark. Replay MIN's exact per-step read schedule (no expert compute) through the
CPU-helper and PCIe paths, using the engine's synchronisation, on one or two hosts. That gives an achievable
host-read time. Report the bound at the median probe as the primary number and the maximum-sample bound as a
sensitivity. Call it a "probe bound" and avoid "speed of light", or else use the DRAM datasheet rate, which the audit
already uses for others.

**W4. The statistics treat the machine as the unit in spirit, but several details are wrong.**
(a) "Machines differ more than problems" compares the between-host SD (0.154) with a *paired-by-problem* within-host
interval (half-width 0.009). The comparison that matters is between-host against *between-launch on the same host*.
The panel has one launch per host, and the whole study has one relaunch (O4, within 0.024).
(b) The two-stage bootstrap resamples problems independently within each drawn host. But the same 20 problems run on
every host, so problems are crossed with hosts, not nested in them. A crossed bootstrap is needed: resample hosts and
problems, and apply the same problem draw to every host. Pooled intervals use 2,000 draws; Appendix A quotes 10,000, which
applies only to the within-host intervals.
(c) The 0.89 correlation with log(link/CPU) pools both budgets, so it has 28 points from 14 hosts, and two
high-leverage hosts drive it. Per budget the Pearson r is 0.93 and 0.88, and the Spearman ρ is 0.78 and 0.66. Without
Pd and Pf, ρ is 0.66 and 0.45. Above a ratio of 0.6 the host spread (1.16–1.36× at 11%) is mostly not explained by the
ratio.
(d) The fast/slow class threshold (link/CPU ≥ 0.5) was chosen after the data. Job 099 pre-registered absolute link
classes (≥ 40 / < 32 GB/s), and P7 failed on three "fast" hosts under them.
(e) Table 2 quotes 3-decimal problem intervals on two hosts, although §2 itself says rentals of one CPU model differ by
19–22%.
(f) The panel ran 20 AIME problems per host (`NSEQ=20`; n = 20 in `panel_099.json`), but §2 says 30.

*Fix:* Relaunch two or three panel hosts once each and fit a variance-components (random-effects) model over host,
launch and problem. Use a crossed bootstrap. Report the correlation per budget with leave-one-host-out. Label the
ratio classes exploratory. State n = 20.

**W5. The horizon rule in distinct experts is elegant but adds little prediction and leaves structured residuals.**
The D rule needs the held-out model's own D(W) curve, which is computed from its trace. With that trace one can
simulate W50 directly. Its leave-one-model-out gain is small (median factor 1.16 against 1.19 for the power law and
1.22 for C/k), comes from 26 points within 9 models, and carries no uncertainty. The residuals are not noise. In
`w50_distinct.json`, D(W50)/C at the largest budget is below its value at the smallest budget in **9 of 9** models,
and strictly decreasing in 7 of the 8 models that have three budgets: for example Qwen1.5 goes 0.99 → 0.79 → 0.61,
and DeepSeek-V2-Lite 0.84 → 0.76 → 0.63. Five of the 26 points have W50 < 1, where D is set to kW by construction.

*Fix:* Bootstrap over models for the comparison in Table 4. Fit and report the budget trend in D(W50)/C. State the
use case: what a designer knows before they have a trace.

**W6. The paper mixes a systems paper with a measurement paper.** About a fifth of the main text and Appendices D and
I argue that "our cache beats FreeToken". This draws attention and space away from the measurement science, which is
the real contribution. The scope is also narrow: two models, one GPU class for the oracle work, batch 1,
teacher-forced decode, and AIME prompts that were also used during development.

*Fix:* Move the system comparison to one compact subsection plus the appendix. State the scope in the title or
abstract (batch-1, consumer RTX 5090 hosts).

**W7. Clarity** (Section 5). This is the main obstacle to acceptance at a general venue.

## 4. Methodology and statistics

**Machine as the unit.** The panel and the decision to state results per host class and per host model are right.
The issues are in W4: launch-level variance is not estimated, the bootstrap assumes nesting where the design is
crossed, the class boundary was chosen after the data, and leverage is concentrated in two hosts. The panel was a
convenience sample of rented machines, so a pooled mean such as [1.12, 1.30] describes this sample of rentals, not a
population. The paper half-acknowledges this. It should report only per-host results and per-class ranges, and drop
pooled intervals for quantities that depend on host class.

**Order-free (Shapley) accounting.** The algebra is correct. φ_set + φ_reads = base − fetch, and with prefetch and the
residual the four terms add to the gap. Table 3 reproduces byte-for-byte. Three caveats should be stated.

1. Only the two-factor part is order-free. The prefetch is nested and the "rest" is a residual, so "an order-free
   accounting of the gap" overstates it.
2. The "number of reads" factor is not only a byte count. Two reads means the CPU serves the miss on the critical path
   and the copy happens off it. One read means a synchronous PCIe fetch on the critical path. The factor therefore
   changes the path and its criticality as well as the bytes, and that is why its effect depends on the host. Call it
   the *admission mechanism*.
3. The interaction is comparable to the main effects (10–58% of the gap, against −13 to 26% for either choice alone).
   Splitting it 50/50 is then a convention, not a finding. The 2×2 cell means with per-host ranges should be the
   primary display, and the Shapley values secondary. The explanation in the main text, one sentence citing Jain and
   Fawcett & Hoos, is adequate for specialists. Outsiders need the worked example in Section 5.

The five-factor model attribution in Appendix E (120 orders) values counterfactual states that no engine produced, and
one of its terms closes the model by definition. It is a model-based attribution and should be labelled as such in
Fig. 6.

**Is the time model a model in the predictive sense?** Partly. Eq. 2 is fitted once (G) on one host, and its
bandwidths come from each host's probe. It was tested blind on 7 hosts (33 rows: median 3.4%, 90th percentile 12%)
and frozen on 4 later hosts (30 rows: median 5.6%, 90th percentile 16%). On those tests it **ties** a naive one-path
model (6.1%, 11%), as the paper honestly says. Its predictive value is at the level of decisions. It picks the
per-machine FETCH split, worth 3.3–8% over the fixed table on host B and 9–34% on half-link hosts. It also predicts the sign of
the in-step single read across hosts, which the naive model gets wrong on the slow-link host. The paper should tell
this story directly. The panel validation needs the disclosure and the re-run described in W2. Qwen3's G was borrowed
from llama.cpp, and the result is a +5 to +21% bias that grows with budget. That is a failure to state in the main
text, not only in an appendix. "Law" is still used in Table 6, Fig. 5 and Table 16 for what Appendix C correctly calls
"a calibrated model, not a law".

**Is the bound a bound?** Under its stated assumptions, yes. The assumptions are exact routing, whole experts, per-layer
slots, one token per pass, and B_host equal to the highest probe. Averaging per-token bounds yields the min-max form
by linearity of the GPU term and convexity of max(R, c). MIN with bypass is checked against brute force in the tests
(15/15 pass). It is not a bound for pooled-slot designs (FreeToken, llama.cpp), and the paper handles this with a
pooled variant. It is also not a hardware bound, because a faster reader could exceed the probe. Tightness is the
open question (W3).

**Price of foresight.** The question is the right one, and the in-engine measurement is careful. The planner equals
the replay on a CPU build, and engine reads match within 1.6%. The answer is distorted by the baseline and path choice
and by the independent-error model (W1). The horizon rule is interesting as a reparameterisation from paging theory,
but weakly supported as a predictor (W5). It is useful today mainly as a negative result: no realisable predictor or
learned admission order comes close.

**Pre-registration.** This is excellent, and it could serve as a model for the field. One suggestion: report the
scorecard as a *calibration* result. For example, of the 301 band clauses, what fraction fell inside the band, and
were misses biased in one direction? Also state which bands were set after same-host pilots (11 sign clauses in
076–081). A reader currently gets 565 lines but no summary of what they say about the author's predictive skill.

## 5. Clarity

**Score: 2/5.** The prose is grammatical and the figures are mostly clean. But almost every sentence carries two or
three numbers and a term coined in the paper, and the argument has to be reconstructed from range macros. A reader
outside MoE offloading will not follow Sections 4–5 on a first read. The main recurring problems:

- **"read" carries three meanings.** It means host reads per token (bytes), reads per admission (one or two), and
  read path (CPU or link).
- **One quantity has five names.** "bound", "limit", "speed limit", "speed of light" (Fig. 9, repository) and "the law"
  appear to refer to one quantity, or two.
- **Too many identifiers.** There are host labels (A, B, S, O1–O6, Pa–Pj), job numbers, and configuration codes in
  the supplement (foa, aa, both3p, w8r5) that do not match the paper's names.
- **Stale numbering.** Scorecard clauses still refer to "Table 1" for what is now Table 2.
- **Unreadable tables.** Tables 8–11 are printed at an unreadable size.
- **Key evidence is in the appendix.** Model validation and tightness are there, while the main text spends space on
  the comparison with FreeToken.

The three hardest passages, with suggested rewrites:

**(1) Abstract, second finding.**
> "Foresight pays when two choices go together: admit only what MIN admits, and read each admitted expert from host
> memory once, on a path with room. Where the PCIe link reads at least half as fast as the CPU, MIN's admissions read
> once in the step recover 26–60% of the gap to the bound, while either choice alone recovers at most 26%."

*Rewrite:* "Knowing future routing helps only if the cache uses it the way the offline optimum (Belady's MIN) does:
it caches only the experts MIN would cache, and copies each from host memory exactly once, instead of first running it
on the CPU and then copying it. On hosts whose PCIe link is at least half as fast as the CPU's own memory reads, doing
both closes 26–60% of the gap between our cache and the bound. Doing either one alone closes at most 26%."

**(2) Section 4, "An order-free accounting".**
> "On the 10 hosts whose link reads at least half as fast as their CPU, reading the deployed policy's admissions once
> is worth −4 to 4% of the gap: the policy rations admissions to a few per token, and their background copies overlap
> later steps. MIN's admission set read twice is worth −13 to 26%; read once, 26–57%. The difference, the interaction,
> is 10–58% of the gap: MIN admits 2–11 times as many experts per token as the deployed policy, and with two reads each
> of those admissions is read again on the critical path."

*Rewrite (with a running example and a 2×2 figure):* "Take host O4 at gpt-oss 11%, where our cache is 10.6 ms per
token from the bound. Change only *which experts are cached* (MIN's choice instead of ours) and 4% of that gap closes.
Change only *how a cached expert is loaded* (one PCIe copy in the step instead of a CPU pass plus a background copy)
and another 4% closes. Change both and about 51% closes. The extra 43 points are an interaction. MIN caches 2–11 times
more experts per token than our policy, so if each of those is loaded twice, the extra loads land on the critical path
and cancel the savings. Because each choice's effect depends on whether the other has been made, we credit each choice
with its effect averaged over the two orders (its Shapley value): 26% and 25% here. Across the ten fast-link hosts the
interaction is 10–58% of the gap (Table 3)."

**(3) Section 5, "In the engine".**
> "Time follows reads, weighted by the path each read takes. At 11% a 4-token window makes 0.52 of MIN's cut in host
> reads, but still reads 97% of its misses over the link in the step, where MIN reads 50%. Where the link is the slower
> path, its share of the time gain is therefore smaller than its share of the read gain: by 0.05 on hosts whose link
> matches the CPU, and by up to 0.20 on the most link-starved (link/CPU 0.29; correlation of the gap with the log ratio
> −0.91)."

*Rewrite:* "A window saves host reads, but like the policy it extends (admit every miss) it still copies almost every
miss over PCIe during the step. At gpt-oss 11%, a 4-token window removes 52% of the reads that MIN removes, yet 97% of
its remaining misses still cross the PCIe link on the critical path, against 50% for MIN. Where PCIe is slower than the
CPU's own memory reads, the window therefore saves less time than its read savings suggest. The shortfall is 0.05 of
the gap on hosts whose two paths are equally fast, and 0.20 on the host whose link runs at 0.29 of its CPU rate."

Further suggestions: put a glossary of about eight terms on page 2 (budget, host-bound, miss, admission, fetch, in the
step, read path, MIN with bypass). Carry one running example (host B or O4, gpt-oss 11%) through Sections 3–5. Replace
Appendix B's wall of prose with a table by era × clause type.

## 6. Artifact checks

All re-runs used a scratch copy of the repository (not `reports/`) against `/home/claude/gpu-branch/results`.

| # | Claim | How checked | Result |
|---|---|---|---|
| 1 | Panel scorecard: 273 clauses, 190 held, 42 held (point), 39 failed, 2 untested | re-ran `scripts/panel_099.py` (results 099a–099j) | **Held.** Identical JSON. `wsg_panel.tex` identical except a stale, unused macro (`\pnMissCutFourPct`) that the script no longer writes. |
| 2 | Panel, gpt-oss 11%: within-host half-width 0.009, SD across hosts 0.154, two-stage interval [1.12, 1.30] for MIN-with-one-read | same run, pooled block | **Held** numerically (0.0085, 0.154, [1.119, 1.300]). Caveats in W4: problems are crossed with hosts, 2,000 draws, no launch-level variance. |
| 3 | Slowest-link host (Core Ultra 9 285K, link 27 / CPU 93 GB/s, ratio 0.29) loses 0.87× at gpt-oss 11%; the same CPU on a faster-link board gains | per-host output (Pf vs Pa) | **Held.** Pf 0.875 [0.855, 0.911]; Pa (link 49 / CPU 60) 1.334. |
| 4 | Prefetched MIN 1.01–1.81× (slow host breaks even at 11%); admit every miss 0.42–1.00× | panel + factorial outputs | **Held.** Pf both3p 1.008 [0.975, 1.056]; aa from 0.424 (Pf) to 1.001 (Pe). |
| 5 | Exact-window time shares at 11% (W=1: 0.14–0.17; W=4: 0.32–0.47; W=16: 0.78–0.93), at 25% (0.21–0.24, 0.49–0.63), W=8 r=0.5 0.20–0.26, whole future r=0.5 0.42–0.55 | panel output and supplement Table 2 | **Held.** But against the deployed cache, W=4 is slower on 6/10 hosts (11%) and 8/10 (25%); W=8 r=0.5 is slower on 8/10 at both (W1). |
| 6 | Gain tracks log(link/CPU), correlation 0.89 across 14 hosts | re-ran `fig_hostdep.points()` | **Partly held.** 0.885 reproduced, but over 28 points (two budgets). Per budget Pearson 0.93/0.88, Spearman 0.78/0.66; without Pd and Pf, Spearman 0.66/0.45. |
| 7 | Calibrated model: in-step states median 2.2% (90th percentile 9%, n=208); background 18.0%; sign of MIN-one-read gain right at 43/43; window-share error 0.03 (max 0.10, n=60) | re-ran `scripts/hostdep_model.py`; own re-runs with frozen G and a one-path baseline | **Held numerically, misdescribed.** G is re-fitted per host-cell on the deployed run (4.1–9.2 ms), not frozen as Appendix C says; counters come from each run; O4 is counted twice. Frozen G: median 3.2%, 90th percentile 10.6%, sign 43/43. Naive one-path: 41/43; it misses both Pf losses (predicts 1.29× and 1.27× against 0.875× and 0.969×). |
| 8 | Table 3 (Shapley accounting), including 10–58% interaction and 30–62% rest on fast-link hosts | re-ran `scripts/factorial_shapley.py` | **Held.** `tab_shapley.tex` and `wsg_shapley.tex` byte-identical; terms add to the gap by construction. |
| 9 | Table 2, host B, gpt-oss 11%: 69.9 vs 54.0 tok/s, 1.294 [1.278, 1.312] | recomputed from `081_headline_law@vast/bs1.jsonl` (launch 2, paired bootstrap) | **Held.** 1.293 [1.277, 1.310] from ms/token; the paper uses mean tok/s (1.294). Within the 0.003 stated in Appendix A. |
| 10 | Bound 172 tok/s at host B, gpt-oss 11% | R* = 38.315 reads/token × 13.25 MB at B_host = 87.5 GB/s | **Held** (172.3). B_host is the maximum of six samples (76.8–87.5; median 80.7). |
| 11 | Audit: 20 in-class rows, median 13.6% | `prereg/audit/audit_sol.json` filtered as in `wsg_numbers2.py` | **Held.** Robust to aggregating by source (7 sources, median 14.5%). |
| 12 | Scorecard: 565 clauses, 218 / 193 / 142 / 8 / 4 | re-ran `scripts/scorecard.py` | **Held.** "168 of 473" in 088–098 means scored clauses (479 − 6 untested). |
| 13 | Table 4 (leave-one-model-out) and D(W50)/C median 0.65 (IQR 0.61–0.72, CV 0.16) | `prereg/foresight/w50_distinct.json` and code in `w50_rebaseline.py` | **Held numerically.** Residuals are structured (falling with budget in 9/9 models); 5/26 points have W50 < 1 with D = kW by construction (W5). |
| 14 | Job 099 predictions committed before the machines started | `git log` on the gpu branch | **Held.** 7b8327a at 02:18; first results at 04:14. |
| 15 | Panel workload is "the 30 AIME-25 problems" (§2) | job header (`NSEQ=20`); n in `panel_099.json` | **Not as stated.** The panel ran 20 problems per host. |
| 16 | Simulator correctness: MIN with bypass optimal | `pytest tests` | **Held.** 15/15 pass, including MIN against brute force. |

A minor provenance note: some inputs are typed into scripts as constants, for example the corrected simulation reads
`SIMF` in `factorial_paper.py`. "Every number is generated by a script" is true, but those constants should point to
the script that produced them.

## 7. Scores

| | |
|---|---|
| Overall | **6 / 10** (weak accept: a strong measurement core; framing and clarity need a revision) |
| Confidence | **4 / 5** |
| Clarity | **2 / 5** |
| Soundness | **3 / 5** |
| Novelty | **3 / 5** |
| Significance | **3 / 5** |

## 8. Would I take this student?

Yes. Committing every prediction before the machine starts, scoring every failure in public, treating rented
machines as the experimental unit, and producing an artifact whose tables I could regenerate byte for byte is the
discipline my group tries to teach and seldom gets from incoming students. What I would work on with him is
distillation: writing for a reader outside the sub-area, saying exactly where a model was fitted and where it was
tested, and turning a hundred careful measurements into the three that change what a designer does.

## 9. The changes that would most raise the score (about three weeks, $13 of GPU)

RTX 5090 rentals cost $0.55–0.75 per hour in the ledger, and a panel-style launch takes at most 2.5–3.5 h, so about
$1.5–2.5 per launch. $13 buys about six launches.

1. **Re-price foresight against the deployed cache, on the right read path, with a meaningful accuracy measure.**
   - With no GPU spend: restate Fig. 3 and §5 as speed relative to the deployed cache, using the existing panel data.
   - Also with no GPU spend: redefine forecaster accuracy on the decision-relevant experts (not resident, or changing a
     decision), and redo the degraded-window and forecaster comparison on traces.
   - About $5–6: on three new hosts with link/CPU near 1.0, near 0.6 and below 0.4, run the κ=1 window family (CPU
     serve plus background or paced copy; W ∈ {2, 4, 8, 16}; r = 0.5 at W = 8) beside base, fetch and both3p, with
     predictions committed beforehand.
   - The output is the number a cache designer actually needs: how many tokens of foresight, at what accuracy, beat my
     cache by X% on my machine.
2. **Make the model predictive and show it.**
   - Write a fitted-where / tested-where table.
   - Use frozen G, and take counters from the trace simulator, not the run.
   - Report the naive one-path baseline beside the two-path model; the sign of the in-step read on slow links is the
     result that makes Eq. 2 worth having.
   - Pre-register the time of every configuration in change 1 from the probe and the simulator alone.
   - With the remaining $6–7, relaunch two of the three hosts once to estimate host, launch and problem variance
     components. Use them to replace the two-stage bootstrap with a crossed one.
3. **Rewrite for an outside reader.**
   - Add a glossary and one running example.
   - Use one name for the bound, and split "read" into bytes, loads per admission, and path.
   - Make the 2×2 the primary display of the accounting, with the worked example from Section 5.
   - Report the bound at the median probe, with the maximum sample as a sensitivity, and state that its attainability
     is untested. Better, add the read-schedule microbenchmark from W3 if budget remains.
   - Disclose n = 20 on the panel.
   - Move the comparison with FreeToken to one short subsection.

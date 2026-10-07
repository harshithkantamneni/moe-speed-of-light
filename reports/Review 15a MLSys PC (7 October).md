# Review 15a: MLSys 2027 main track, PC review (blind)

**Submission:** "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"
**Date:** 7 October 2026

## Materials read

- `paper/paper.pdf`: all 40 pages, read as extracted text, with pages 5–6 (Table 3, Table 4, Fig. 2) also rendered as images. I also read `paper/paper.tex` (the main-text source). From `paper/app_wsg.tex` and `paper/app_more.tex` I grepped only cross-references and one sentence (the G_fit wording).
- `paper/supplement.pdf`: the opening pages and the scoring rule. I skimmed the job-108 clause rows.
- `/home/claude/gpu-branch/jobs/`: the full headers of jobs 102, 103, 104, 105, 106, 107 and 108, and wrapper 105a. I ran `git log --format='%h %ad'` / `'%H %ct'` and `git diff` on the job scripts, with no commit messages read.
- `/home/claude/gpu-branch/results/`, raw files only:
  - jobs 093–108: `ec_*.jsonl`, `st_*_base/dk/fetch/fetchplan/....json`, `concur.txt`, `cores.txt`, `g_prof.json`/`q_prof.json`, `value_map_replay.jsonl`, `nvidia-smi-q.txt`, `cpu.txt`/`lscpu.txt`, `numa.txt`, `free.txt`, `gpu.csv`;
  - job 069c `prof_C14/C32.json`;
  - `bs1.jsonl` of jobs 080, 081 and 089;
  - job 102 `readsched_C*.txt`;
  - job 090 `par_*.jsonl` and its on-machine `parity_kl.json`;
  - the job 108c/108e gate outputs.
- `moe-speed-of-light/gpu/vast_ledger.json`.
- `moe-speed-of-light/prereg/job108_families.json`. I used it **only** for the list of launch directories of jobs 093–108 and their consumer/server and valid labels. For jobs 105–108 I re-derived validity from raw `nll_sum/nll_n` and round spreads.
- Authors' scripts, read **for definitions only**:
  - `scripts/panel_099.py` (`host_info`, `cell_data`, `_status`);
  - `scripts/job106.py` (`law`, `load`);
  - `scripts/reanalysis.py` (`profiled_G`, launch selection);
  - `scripts/robustness.py` (header and `launch_rows`);
  - `scripts/job108.py` (`fallback_G`);
  - `scripts/speed_limit.py` (`host_rates`, `limit`);
  - `scripts/wsg_tables.py` (`headline()`, which says which job feeds which Table 3 row);
  - the `scripts/headline_stats.py` docstring;
  - gpu-branch `jobs/ec2/fetch_table.py` (`bandwidths`).

## Independence statement

- **What I did not open:** anything under `reports/` (except to write this file), any `prereg/*outcome*.md`, `research_notes/`, any file named like a review, number check, plan or progress log, `paper/paper_v1_prereview.tex`, and `apply/`. Directory listings of `prereg/` and the repository root showed such file names, but I opened none of them.
- **Commit messages:** I read none.
- **My own code:** every number in the claims table below was recomputed from raw rows with my own scripts in `scratchpad/rev15a/` (`common.py`, `reg.py`, `reg2.py`, `plan.py`, `launches.py`, `agg.py`, `plainfound.py`, `decomp.py`, `fact.py`, `dk.py`, `j105.py`, `win.py`, `same.py`, `elast.py`, `spear.py`, `anova.py`, `headline.py`, `ledger*.py`). I re-implemented the probe parsing and Eq. (3) after reading the authors' definitions.
- **Repositories:** I modified nothing in either repository except writing this file, and committed nothing.

---

## Summary

The paper studies batch-1 decoding of MoE models whose experts mostly live in host DRAM, on rented RTX 5090 hosts, with gpt-oss-120b and Qwen3-30B-A3B. It makes four contributions.

1. **Two bounds.**
   - Eq. (1): Belady's MIN with bypass gives the fewest host reads per token (R*). R* times the expert size, divided by the machine's highest probed host read rate, lower-bounds the time per token of any exact-routing cache with C slots per layer.
   - Eq. (2): a system that reads only on demand also pays the GPU's non-expert compute in series.
   - A microbenchmark replays MIN's per-layer reads and reaches 86–96% of the bound's read time.
2. **An accounting of time (Eq. 3).** For the authors' llama.cpp expert cache, time per token is described as profiled GPU kernel time G plus counted host reads at the best probed rate, with an overlap discount on background admissions. On 25 machines with consumer processors this splits the deployed cache's time into GPU compute, MIN's reads and excess reads (Fig. 2). The authors are explicit that the relation was found on past data and failed all four registered prospective tests (Table 4), and that a processor-class split was drawn after the fact.
3. **In-engine oracles.** Oracles in the engine show three things:
   - Foresight pays when spent as MIN spends it (each admission read once): +16–51%, and up to +81% with prefetch.
   - Among MIN's hit-optimal schedules, the fewest-admission schedule beats the deployed cache at gpt-oss 11% on every machine that "ran stably" (1.22× mean over 9 machines).
   - The online analogue (a larger admission margin) gains about 2%.
4. **A price on foresight.** Half of MIN's read saving needs a window containing about 0.65·C distinct experts per layer, which the forecasters tested do not approach.

The work is accompanied by an unusually complete artifact: predictions committed to git before each rental, raw per-problem rows, counters and probes.

## Strengths

1. **Exemplary transparency and an artifact that reproduces.**
   - I recomputed 34 quantitative claims from raw rows (table below). Nearly all match to the printed precision, and the rest are within rounding or differ in scope and wording only.
   - Every job header among the 71 rentals of jobs 093–108 was committed at least 4 s before its rental was created.
   - The post-launch edits to headers (jobs 100, 102, 105, 106, 107) only append host amendments. None edits or removes a prediction line.
   - This is the most auditable empirical systems submission I have reviewed.
2. **Honest evidence labelling.** Table 1 maps each claim to derivation, registered, exploratory or after the fact. Paragraph headings repeat the label. Failures are reported in the main text (Table 4) rather than buried, and the limitations are concrete.
3. **A useful, sharp in-engine result on how foresight must be spent.** "One read versus two" and "fewest admissions versus greedy" are measured in a real engine across many machines. The link-to-CPU ratio dependence is substantial (Spearman ≈ 0.88 over 19 machines). These are actionable for designers of prefetching and offloading systems, and the fewest-admission schedule is a clean idea, properly credited to the interval-LP literature.
4. **Solid system baseline.**
   - The cache runs 2.0–4.0× stock llama.cpp at equal expert memory and beats FreeToken at 11 of 12 cells on two hosts (Table 3 reproduced exactly).
   - It is checked against a tuned FreeToken, two newer llama.cpp-based systems and KTransformers.
   - Output parity is measured by full-vocabulary KL against an all-VRAM reference.
5. **Machines treated as the unit.** The authors correctly observe that machines dominate variance: about 96–98% of the variance of the log gain, against 1–3% for problems. They give machine-level bootstrap intervals and t-intervals for the main pooled claims.

## Weaknesses (most important first)

### W1. The paper's titular thesis rests on a relation that never passed a prospective test, and whose distinctive term is not identified.

Eq. (3) and its decomposition (Fig. 2, §4.4) are what "where the seconds go" refers to. Yet:

- **It never passed.** All four registered tests failed, or were inconclusive and failed per cell (Table 4 reproduced exactly).
- **The overlap term is post hoc and unidentified.** The term (1−G/T)A was added after job 105 failed. On the only subset where it "held" (three previously rented machines in job 106), a constant half-discount passes 11 of 12 cells against 12 of 12. That 12/12 includes a cell at **+5.99%** against a 6% band (106a, Qwen3 25%).
- **Its residual is budget-dependent.** Over the 25 consumer machines the residual is −2.2% [−3.7, −0.4] at 11% and +1.6% [0.2, 3.1] at 25%. That is a systematic misspecification, not noise. On the consumer machines new to the last two tests, every gpt-oss cell is under-predicted.
- **It fits worse where G was measured on the machine.** For 30 of the 39 consumer launches, G is not measured on that machine but taken as a median of other machines' profiles. The relation fits better on those launches (median |error| 1.7%) than on the 9 with their own profile (3.2%).

The paper labels all of this "exploratory", which is honest. But the title, Fig. 2 and the "serialisation is 23–66% of the gap" statement present the decomposition as explanatory. As it stands, it is an accounting with a model-dependent residual on in-sample machines.

### W2. The "bound" is relative to a single, outlier-prone probe reading, and some framing overstates what is bounded.

- **B_host is a single maximum and is exceeded in practice.** B_host is the single highest of about 25 probe readings. By Eq. (3)'s own accounting, the engine reads more than 5% faster than B_host on 12 of 84 launch-budgets, up to 1.22× (reproduced). So Eq. (1) is not a bound on the machine, only on the probe. The "31–54% of the bound" fractions inherit per-machine uncertainty of the same order.
- **The dual-socket outlier.** On the dual-socket server (107d), B_host = 204.5 GB/s comes from one concurrent reading in which CPU threads read 181.4 GB/s, while the CPU-only maximum is 151.4 GB/s. With 151.4, the abstract's "missed by up to 59%" becomes 46–51%. The qualitative point stands, but the headline number is driven by an implausible reading.
- **Eq. (2) is presented as a bound for systems that read on demand.** Its T_GPU (2.9–3.4 ms) is the authors' own kernel time. A demand-reading system with faster attention or dense kernels would beat it. The Limitations section says so, but §3 does not.
- **Unlike-for-like comparison in §3.** The text places "our cache at 31–54% of Eq. (1)" (probed rates) next to "published systems … median 13.6%" (datasheet ceilings, which inflate the bound). These are not like for like. Either score the authors' own cache at datasheet ceilings in the same sentence, or drop the juxtaposition.

### W3. Narrow scope and small effective samples for the generalising claims.

- **Narrow setup.**
  - Hardware: one GPU model, on rented hosts with unlocked clocks and varying power limits.
  - Workload: two models on AIME-25, which was also used during development. Later experiments use 20 problems × 256 teacher-forced tokens, at batch 1.
- **Few machines per claim.** Each machine-level claim rests on 3–9 valid machines.
- **The consumer/server split is post hoc and confounded.** Most server rentals fail validity (5 of 8). The paper acknowledges that the split was drawn after the tests and is confounded with co-tenancy (40–130 GB of host memory in use), NUMA layout, core limits and driver version.
- **Consequence.** A reader cannot tell whether "consumer processor" describes a hardware property or simply "a machine without noisy neighbours". The 4090/3090 grid (App. E) covers only the system comparison, not the decomposition or the oracles.

### W4. The fewest-admission claim conflates "ran unstably" with "slow link", and its stability gate is partly post hoc.

- **The losses coincide with the slowest links.** The abstract says the schedule "gains on every machine that passed our checks". The two machines where it lost are the two with the slowest links: EPYC 7302 at 0.846×, with link/CPU 0.14 and 13 GB/s; EPYC 7663 at 0.974×, with link/CPU 0.21 and 26 GB/s. The paper's own crossover rule predicts that copying in the step should lose there. On the EPYC 7663, the same set loaded by the CPU gained 1.13×.
- **The stable sample is all faster-link machines.** Every stable machine has link/CPU ≥ 0.28. The "every stable machine" claim is therefore really "every machine with link/CPU ≥ 0.28".
- **The gate is partly post hoc.** The 2% round-spread gate was a registered *prediction* in job 106 that failed, and was then used as an exclusion criterion. It was applied after the fact to jobs 104–105, whose launches ran one round, so the criterion is vacuous there.
- **"By implication".** Two of the nine "held" machines rest on a registered greedy comparison plus an unregistered observation.

### W5. Practical payoff is modest, and the bound itself is incremental.

- **Small realisable gains.**
  - The only online improvement derived from the study is the admission margin: +2.0% [1.2, 3.1] over 5 machines.
  - The oracle gains need foresight that no forecaster tested comes close to (at most 15% of the read gap).
  - The lead over FreeToken is 0.97–1.29× and model-dependent: a tie on FreeToken's own headline model.
- **The bound is close to existing work.** Eq. (1) is a roofline-style "minimum bytes over best bandwidth" bound with MIN-with-bypass reads. Several cited concurrent works (WiSP, Budgeting Bytes, Zhang 2026b's gap to MIN with bypass) are close.
- **Where the novelty lies.** The novel parts are the in-engine oracle factorial, the fewest-admission schedule measured in an engine, and the distinct-experts horizon rule. The paper would be stronger if it were framed around them.

### W6. The registered record is weaker than "registered; held" suggests in places.

- **Point-estimate holds outnumber interval holds.** Across Table 8, 597 clauses held on the point estimate only, against 495 that held with an interval and 308 that failed. Most quantitative bands failed, and "held" usually refers to sign or threshold clauses.
- **Example: the read-time reachability claim.** Table 1's "registered; held (3 hosts)" rests on registered thresholds of ≥0.80 (token mode) and ≥0.50 (layer mode). None of the 33 clauses of that job carries an interval, and 5 failed. The 86–96% figure is observed, not predicted.
- **Example: the set-aside for job 108.** The EPYC 7543 is set aside in Table 4 because its G could not be profiled. But no plausible G explains a 34% miss: it would need G ≈ 12 ms. The set-aside rationale does not bear on the outcome, which still fails.

### W7. Minor factual and wording errors found while checking (all minor; listed for the camera-ready)

1. **The 107d probe reading.** App. K says 107d's second-highest reading is "more than 5% lower … by 35%". It is 26% lower (151.4 against 204.5); the highest is 35% *above* it.
2. **Same-CPU spread.** §2 says "two hosts with the same CPU model differ by up to 29%". That holds within the job-099 panel. Over all launches it is 33% (Core Ultra 9 285K, 18.14 against 13.68 ms at gpt-oss 11%).
3. **Machines versus offers.** App. N's "74 distinct machines" are 74 distinct *offers* in the ledger. Elsewhere machines are counted by GPU UUID.
4. **Spearman.** §5 prints 0.89 [0.63, 0.97]; I get 0.881 [0.634, 0.973] with one value per GPU UUID. This is a rounding or aggregation detail.
5. **Window ratios.** The deployed-path 16-token window runs 1.000–1.035×; the text says 1.00–1.03×.

---

## Table of checked claims

All values were recomputed by me from raw `ec_*.jsonl`, `st_*.json`, `concur.txt`, `g_prof.json`, `bs1.jsonl` and `readsched_*.txt`. Eq. (3) is solved as the larger root of the quadratic; B_host is the maximum probe reading; G falls back to the median of the 7 profiles of jobs 069c and 105. Verdicts: ✓ = matches; ≈ = matches within rounding, aggregation choice or launch choice; ✗ = wrong as stated.

| # | Claim | Location | My value | Verdict |
|---|---|---|---|---|
| 1 | Job 105 (plain form, 6%): 5/8 cells within; 2/4 on new machines; median 4.2%; failed | Table 4 | 5/8; 2/4; 4.24% | ✓ |
| 2 | Job 106 (overlap form): 12/20 within; 0/8 new; median 4.7%. Set aside: 12/12, 2.3% | Table 4, §4.2 | 12/20; 0/8; 4.67%. Set aside: 12/12, 2.34% (106a Qwen3 25% at +5.99%) | ✓ (fragile) |
| 3 | Job 107: 1/8 within; median 31.7%; 2 valid of 7 rented; inconclusive | Table 4, App. K | 1/8; 31.75%; valid b, d; 3 stopped at the gate; 2 failed the round check | ✓ |
| 4 | Job 108 (8% band): 4/6 within; median 6.7%. Without the EPYC 7543: 4/4, 5.3% | Table 4 | 4/6; 6.70%. Without the EPYC 7543: 4/4, 5.25% (2 of 4 cells beyond 6%) | ✓ |
| 5 | Predictions committed before each machine started; amendments did not change predictions | §1, App. C | Earliest header commit precedes rental by ≥4.0 s for all 71 rentals of jobs 093–108. Post-launch diffs only append host amendments | ✓ |
| 6 | Discounting admissions by half passes 11 of the 12 stable cells | §4.2 | 11/12 | ✓ |
| 7 | Job 106 server machines: Eq. (3) off by 10–52%; one ran its rounds 52% apart | §4.2 | 10.2–52.4%; spread 52.2% (106d) | ✓ |
| 8 | Dual-socket server under-predicted by 52–59% (abstract: "up to 59%") | §4.2, abstract | 52.2–58.9% with B_host = 204.5; 46.2–51.4% with the second-highest or CPU-only reading (151.4) | ✓ (driven by an outlier reading) |
| 9 | New consumer machines: every gpt-oss cell under-predicted by 0.9–7.1%; up to 11% (Qwen3) | §4.2, abstract | 0.9–7.1%; 11.3% (5900XT, Qwen3 12.5%) | ✓ |
| 10 | 39 valid consumer launches on 25 machines; implied rate 0.88–1.16; within 8% on 37; G profiled for 9 | §4.3 | 39 / 25; 0.880–1.164; 37; 9 | ✓ |
| 11 | Server: 5 of 8 launches invalid; valid ones read at 0.27, 0.58 and 1.04 of B_host | §4.3 | 5/8; 0.27, 0.58, 1.04 | ✓ |
| 12 | Eq. (3) mean error over 25 consumer machines at 11%: −2.2% [−3.7, −0.4]; on the 21 found-on machines: −1.5% [−3.2, 0.4]; at 25%: +1.6% [0.2, 3.1] | §4.2, App. K | −2.16 [−3.74, −0.46]; −1.54 [−3.25, 0.37]; +1.58 [0.22, 3.05] | ✓ |
| 13 | Plain form within 6% on 28/30 at 11%. Over 34 launches: plain 32/18, overlap 29/33 (11% / 25%) | §4.1 | 28/30; 32/18 and 29/33 with the 7-profile median G (16 and 32 at 25% with job 069c's profiles alone) | ✓ |
| 14 | Implied rate >5% above the rate on 12/15/46 of 84 launch-budgets (max 1.22/1.26/1.53); within 6% on 31/34/22 of 39; median \|error\| 1.7/1.5/5.6% | Table 26, §8 | identical | ✓ |
| 15 | Second-highest reading more than 5% lower on 1 of 42 launches, 107d, "by 35%"; median 0.8% | App. K | median 0.80%; 107d second-highest 26% below the highest | ✗ (wording) |
| 16 | gpt-oss 11% split: G 12–45%, MIN 31–54%, beyond MIN 22–35%; serialisation 23–66% of the gap; below-rate ≤11% on consumer machines; servers 0/34/59% | §4.4, Fig. 2 | 12–45, 31–55, 21–34, 23–64/67 (depends on which launch per machine); ≤10.6%; 0/34/59 | ≈ |
| 17 | Elasticity of (T−G) to B_host: −0.95 [−1.06, −0.84] | §4.4 | −0.950 [−1.066, −0.840] (30 launches, 21 machines, cluster bootstrap) | ✓ |
| 18 | Same-CPU hosts differ by up to 29%; relaunched machines within 3.2% | §2 | 29% within the panel; 33% over all launches; relaunches ≤3.1% | ≈ (scope) |
| 19 | G 4.1–4.7 ms; T_GPU 2.9–3.4 ms; median of 7 profiles | §3, §4.1, App. K | G 4.05–4.69; G − expert GEMV − cache control = 2.92–3.38; median 4.284 / 4.498 ms | ✓ |
| 20 | Table 3: all 12 ratios and CIs; leads at 11 of 12; 2.0–4.0× llama.cpp; host intervals overlap only at Qwen3 12.5% | Table 3, §3 | identical to 3 decimals (e.g. host B gpt-oss 11% 1.294 [1.278, 1.311]; host S Qwen3 43.75% 0.974 [0.961, 0.987]); 2.00–4.02× | ✓ |
| 21 | Microbenchmark, per-layer wait: 86–96% of the bound's read time | §3, Table 28 | 86, 89, 96, 96, 95, 93% | ✓ |
| 22 | MIN 1 read faster than deployed at every budget on 3 hosts by 16–51%; up to 81% prefetched; usual ways gain ≤15% or lose at the smallest budgets | §5, Fig. 3 | 1.160–1.514 (18/18 host-cells); 1.812; max 1.148 (Belady 1 read, O4, Qwen3 12.5%) | ✓ |
| 23 | Fewest-admission schedule: 1.22× [1.14, 1.31] over 9 stable machines (t-interval 1.11–1.33); lost on 2 of 4 unsteady ones | §5, App. K | 1.222 [1.136, 1.309]; t 1.113–1.331; losses 0.846 and 0.974 | ✓ (scope: W4) |
| 24 | Links at 0.28–0.32 of CPU rate: greedy 0.85–0.93×; fewest-admission loaded by the CPU 1.11–1.22× | §5 | 0.853–0.933; 1.112–1.217 | ✓ |
| 25 | Fewest-admission copies 0.60–0.73 of greedy's; 2.5–5.3% more misses | §5, §8 | 0.597–0.726; +2.5–5.3% | ✓ |
| 26 | Admission margin: 1.020× [1.012, 1.031] over 5 machines; +6% max; −1% at 1 of 10 cells; reads −4–6% at 11% | §5 | 1.0197 [1.0115, 1.0302]; 1.057; 0.987; 4.3–5.8% | ✓ |
| 27 | Layer-ahead copy: 1.04× where link ≈ CPU; down to 0.72× at 11% | §6, Table 22 | 1.042; 0.722 (0.691 at 25%) | ✓ |
| 28 | 16-token window on the deployed path: 1.00–1.03×; half-right windows lose up to 6%; exact 16-token window recovers 0.78–0.93 of the gain in time | §6 | 1.000–1.035; 0.943; 0.78–0.93 (10 hosts) | ≈ |
| 29 | Spearman of the MIN-1-read gain against link/CPU: 0.89 [0.63, 0.97] over 19 machines | §5 | 0.881 [0.634, 0.973] | ≈ |
| 30 | Machines carry 96% of the variance of the log gain; problems ≤3% | §5, App. K | 98.3% (11%), 96.5% (25%); problems 1.1–2.9% | ≈ |
| 31 | Running example (9950X, 11%): 10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms | §5 | O4 (096a): 20.64, 15.16, 13.81. 10.0 is consistent with the exact-MIN R* (≈38.4); the no-evict R* gives 10.2 | ✓ |
| 32 | Parity: loss +0.16% (gpt-oss), +0.19% (Qwen3); KL 0.0019 | §8, App. L | +0.16%, +0.19% from raw NLL; KL 0.00186 from the on-machine summary (full logits not in the artifact) | ✓ |
| 33 | 124 rentals, $87.2, "74 distinct machines" | App. N | 124; $87.2; 74 distinct offers | ✓ (wording) |
| 34 | In-class audit median 13.6% (20 rows); all trace rows 9.5% (29) | §3, App. M | 13.6 and 9.5, recomputed from Table 30's percentages (not re-scored from sources) | ✓ (internal) |

**Not checked:**
- the trace-study claims: the horizon rule D(W50) ≈ 0.65·C, pooling −4.5 to −18.6%, and speculative batches;
- the forecaster precision figures;
- the audit's per-row bounds.

---

## Clarity

Earlier rounds scored clarity 2/5, then 3/5. This revision is better organised than a lab log, but it is still much harder to read than its content requires. **My score is 3/5.**

### What works

- **Table 1** (claims × evidence class × section) is the single most helpful device in the paper. Each main-text paragraph heading carries the same label: registered, exploratory or after the fact.
- **Fig. 1** (one decode step) and **Fig. 2** (per-machine stacked decomposition, consumer above, server below, with the hatched "reading below the rate" part) communicate the core picture at a glance.
- **Table 2** (configurations, with reads per admission) and the **running example** in §5 anchor the many configuration names.
- **Table 4** compresses the four registered tests, and its "set aside" column is explicitly marked as after the fact.
- The **Limitations** section is specific rather than boilerplate.

### What still makes it hard to read

1. **Sentences carry too many numbers.** Many sentences carry 5–10 numbers, nested qualifiers and parenthetical exceptions. Examples are §4.2 "Machines never rented before: not passed" (about 15 numbers in one paragraph), §5 "Which of MIN's schedules is followed matters", and §4.3. The claim is hard to separate from the bookkeeping. Each paragraph should state one claim, give one number and an interval, and point to Table 4 or the appendix for the rest.
2. **Vocabulary overload.**
   - About ten configuration names: deployed, single read, admit every miss, MIN 1 read and MIN 2 reads, greedy and fewest-admission, prefetched, Belady, window W, deployed + window W.
   - Twenty appendix codes (Table 6).
   - Host codes: A, B, S, O1–O6, Pa–Pj, "Pf again".
   - Three GPU-time quantities: G, T_GPU and G_fit.
   - Four rates: B_c, B_p, B_cp and B_host.
   - Overlapping names for the oracles: "MIN", "Belady prefetch", "hit-optimal" and "bytes-optimal".
   - "Bound" in the main text, but "limit" and "speed of light" in the appendix (Fig. 11's axis label reads "% of own speed of light").

   Readers will confuse "Belady prefetch" with "Belady's MIN".
3. **The narrative of §4 does not tell the reader what to believe.** The paper is titled for the decomposition, yet §4 presents the relation first, then four failed tests, then a post hoc processor split, and only then the decomposition, labelled "exploratory". A reader finishes §4 unsure whether Fig. 2 is evidence or illustration. The Evidence column of Table 1 is cryptic, e.g. "registered on 7; held on all 9 stable ones (two by implication), lost on two of four unsteady ones". So is its caption: "Held is mostly on the point estimate".
4. **Stale cross-references and inconsistencies remain.**
   - App. A rule 10, App. C (job 093), App. D (MIN prefetched over-prediction) and App. H (single-read factorial) all point to §4 (`sec:gap`) for oracle results that are now in §5.
   - App. L points to §6 for trace-study results that are in App. I.
   - The supplement says the scoring rule is in "Appendix B"; it is in App. C.
   - App. D says "the profiled G_fit of Eq. (3)"; it should say G.
   - The introduction says Eq. (3) was registered "before four later experiments", while §4.2 says "before three more experiments". Both are reconcilable (plain form plus three overlap tests), but the reader must work it out.
   - App. K's "by 35%" is the wrong direction (W7).
5. **Appendix sprawl with legacy framing.** About 31 pages and 30 tables, written largely as a per-job log: job numbers, host codes, "launch 2", "listing B".
   - It keeps the vocabulary of an earlier framing: the calibrated "law" (Eqs. 4–5), "limit", v(F)/v(O,F), a Shapley "accounting as a model".
   - The main text no longer uses these, and they compete with Eq. (3) and its G.
   - Finding the evidence for a given main-text sentence usually needs a full-text search.

### Smaller points

- **Abstract.** "reaches 31–54% of that bound" is ambiguous, since the bound is on time. Say "runs at 31–54% of the bound's speed".
- **Eq. (3).** It is written implicitly (T on both sides) and then "solved". Give the closed form, and one sentence of physical justification for the G/T share.
- **Fig. 3.** Six panels with three host markers plus grey panel ticks are dense. Consider showing only the host-bound budgets in the main text.
- **"Ran stably".** The paper should say that for single-round launches (jobs 104–105) the definition reduces to "correct outputs".

---

## Questions for the authors

1. **The probe rate.** Why is B_host the single maximum reading rather than a robust statistic, such as the median of the concurrent sums? Is the 107d reading (CPU threads at 181 GB/s while concurrent, against 151 GB/s alone) an artifact? Please give the abstract's and Fig. 2's server numbers with the second-highest or median rate.
2. **Instability versus link speed.** Can you separate instability from link speed for the fewest-admission claim? Is there any stable machine with link/CPU < 0.25? Would you restate the abstract as "on every stable machine with link/CPU ≥ 0.28"?
3. **The sign flip in Eq. (3).** Its residual changes sign between budgets: −2.2% at 11%, +1.6% at 25%. What mechanism do you believe is missing? Is there a pre-specified, falsifiable version of Eq. (3) you would accept as passed on new consumer machines?
4. **Host-side work.** `host_us_per_step` shows about 0.8–1.0 ms of host-side work per token (app, launch and post) on the deployed path. How does Eq. (3) account for it? Is it hidden under GPU time, or absorbed into the residual?
5. **The scope of Eq. (2).** Would a demand-reading system with faster non-expert kernels beat Eq. (2) on the same machine? If so, should Eq. (2) be called a bound for your engine's kernel schedule rather than for "systems that read on demand"?
6. **Generality.** How do the oracle factorial and the fewest-admission result behave on a different GPU class (the 4090 and 3090 hosts exist for Table 10) and on a non-AIME workload, given that AIME was used during development?
7. **The horizon rule.** Does D(W50) ≈ 0.65·C hold on the engine's own AIME trace at both host-bound budgets, not only on the nine models' sampled text?

## What would raise my score

1. **Restructure around what is established.**
   - Lead with the bound, the system comparison and the foresight results, which are registered and robust.
   - Present Eq. (3) as an accounting identity with a measured residual: G, plus MIN's reads, plus excess reads, plus "unexplained".
   - Do not present it as a relation that explains the time, unless it passes a pre-specified prospective test on new consumer machines.
   - Consider retitling, or make the subtitle carry the paper's weight.
2. **Use a robust B_host** (median or second-highest concurrent sum) for every headline number, with the single maximum as a sensitivity check rather than the other way round.
3. **Scope the fewest-admission and Eq. (2) statements** as described in W2 and W4.
4. **Cut the appendix to the material the main text cites.**
   - Move the per-job narrative to the supplement.
   - Unify terminology (bound/limit; Belady/MIN; G/G_fit).
   - Fix the stale cross-references and errors listed under Clarity and W7.
5. **Add one out-of-family replication** of the oracle factorial: another GPU class or another workload, with machine-level intervals.

## Scores

| Criterion | Score |
|---|---|
| Overall (1–10; 6 = weak accept, 8 = strong accept) | **5** |
| Soundness (1–5) | **3** |
| Significance (1–5) | **3** |
| Novelty (1–5) | **3** |
| Clarity (1–5) | **3** |
| Confidence (1–5) | **4** |

**Rationale.**
- **What is solid.** The measurements are real, honestly labelled and reproducible to the printed digit. The system comparison and the in-engine finding on how foresight must be spent are solid, publishable contributions.
- **Why it is held back.** Three things keep it from a weak accept in its current form:
  - The explanatory decomposition the paper is named for failed every prospective test, and its distinctive term is unidentified.
  - The bound is anchored to an outlier-prone single probe reading.
  - The presentation still demands too much of the reader.
- **Where a revision could land.** A restructured version (the first three items above) would be a 6–7.

# Review 18a: "Where the Seconds Go" (MLSys 2027, main track)

## Materials read

- `paper/paper.pdf` (43 pages). The main text (Sections 1 to 10, Tables 1 to 4, Figures 1 to 5) was read in full, from extracted text and from page renders of pages 3, 5, 7 and 8. Appendices read: A, B, C (including the Table 8 caption and the failure notes), D (most of it), E (text and captions), F, G, H (most of it), I, J, K, L, M (most of it), N (caption and introduction) and O.
- LaTeX: `paper/paper.tex` (lines 86 to 280: Introduction, Table 1, Setting, Section 3), `app_wsg.tex` (full), `app_relation.tex`, `app_traces.tex`, `app_value.tex`, `app_more.tex` and `supplement.tex`.
- `paper/supplement.pdf`: the header, the clause rows for jobs 073 to 076, and targeted searches for particular clauses.
- Authors' scripts, read **for definitions only**: `decomp_measured.py`, `reanalysis.py` (features and outcomes), `speed_limit.py` (`host_rates`, `limit`, `MODELS`), `gpu-branch/jobs/ec2/fetch_table.py` (`bandwidths`), `factorial_shapley.py` (`limit1_of`), `fig_decomp.py`, `sumlaw_paper.py` (`profiles`), `value_map.py` (docstring and semantics) and `speed_limit_v2.py` (notes).
- `prereg/speed_limit_v2.json`. I opened it only after computing R\* myself, to confirm that my MIN simulator and the authors' agree.
- `gpu/vast_ledger.json` (rental start times). I looked only at the structure of `data/published_measurements.json` and `data/traces/`.
- Raw results in `/home/claude/gpu-branch`:
  - `results/*/ec_*.jsonl` (per-problem `decode_ms`, `n_decode`, and the config `stats=` path);
  - `st_*.json` counters;
  - `concur.txt`, `cores.txt`, `nvidia-smi-q.txt`, `cpu.txt`/`lscpu.txt` and `manifest.json` (start times);
  - the Nsight profiles `prof_C14/C32.json` and `g_prof.json`;
  - `readsched_C*.txt` (job 102);
  - `bs1.jsonl` (jobs 081, 089 and 098);
  - the AIME routing traces `084c/route_aime25_gptoss.npz` and `084b/route_aime25_qwen3.npz`.
- Job-script headers (registered predictions) for jobs 081, 096, 099, 102 and 104. I also used `git log --format='%h %ad'` dates for jobs 093 to 108, and `git diff` of file contents between commits for jobs 100, 102, 105, 106 and 107.

## Independence statement

- I did not open anything under `reports/` (except to write this file), any `prereg/*outcome*.md`, `research_notes/`, any review, number-check, plan or progress file, `paper/paper_v1_prereview.tex`, or `apply/`.
- I did not read commit messages. I used only commit dates and file contents and diffs.
- Every number in the verification table below comes from my own code, run from `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/rev18a/`, on raw rows, counters, probes, profiles and routing traces. The code includes:
  - my own per-layer Belady-with-bypass (MIN) simulator and a pooled-slot variant (`minsim.py`, `pooled.py`);
  - my own replay of the window policies (`window_sim.py`);
  - the gap decomposition (`decomp.py`, `decomp2.py`);
  - the machine-level analyses (`consumer.py`, `plan.py`, `spearman.py`, `varshare.py`, `implied.py`, `factorial.py`, `headline.py`, `hb.py`, `regtime.py`).
- I modified nothing in either repository and committed nothing.

---

## Summary

The paper studies batch-1 decode of MoE models (gpt-oss-120b, Qwen3-30B-A3B) whose experts mostly live in host DRAM, on rented RTX 5090 machines. It makes five contributions.

1. **A bound.** It proposes a per-machine bound, Eq. (1): MIN-with-bypass host reads per token times expert size, divided by the highest probed host read rate. It adds two tighter variants, Eq. (2) for systems that read on demand and Eq. (3), which respects layer dependencies. A microbenchmark reaches 86 to 96% of the bound's read time. The authors' llama.cpp expert cache runs at 31 to 54% of Eq. (1) at gpt-oss 11% on 25 consumer machines. It leads FreeToken at 11 of 12 configurations and runs 2.0 to 4.0 times as fast as stock llama.cpp.
2. **A decomposition of the gap.** Oracles built into the engine remove parts of the gap between the deployed cache and Eq. (1), in a 2x2 design: what is cached (deployed set or MIN's set) by how it is read (twice, served by the CPU and then copied, or once, in the step). A prefetching oracle is nested on top. On 15 machines, and in particular on the 13 whose link-to-CPU ratio is at least 0.5 (a subset chosen after the data), each change alone closes about 1% of the gap at gpt-oss 11%. Both together close 39%, prefetching closes another 15%, and 47% remains.
3. **How foresight must be spent.** Read each admitted expert once, admit as rarely as hit-optimality allows (a "fewest-admission" MIN schedule), and choose the read path by the link-to-CPU ratio.
4. **The price of foresight.** Next-token routing recovers at most 0.31 of MIN's read savings over admit-every-miss. Half of the savings needs about 0.65 C distinct experts of lookahead (across 9 models).
5. **An open record.** Every prediction was committed before its machine started, 565 clauses are scored, and every number is generated by a script.

## Strengths

1. **Reproducibility, which I verified extensively.** I re-derived 37 quantitative claims from raw rows, counters, probes and routing traces using my own code. Every number that carries the main argument reproduced, most to the stated rounding. This includes:
   - Table 3 cell by cell, with bootstrap intervals;
   - Eq. (1) to (3) shares on 25 machines;
   - Table 4 bounds and speeds;
   - the window-value numbers. My independent replay of the window policy matched the registered read counts to two decimals.

   This is far above the usual standard.
2. **Registration that holds up.** For jobs 093 to 108, all 71 rentals in the ledger started after their job script's first commit (at least 4.0 s by commit time). Mid-job amendments that I diffed (jobs 102, 105, 106 and 107) changed only host selection, as the paper says. Job 100's added ninth prediction is disclosed. Failed predictions are reported in full: 142 of 565 clauses failed. That is rare and valuable.
3. **Oracles inside a real engine, in time rather than only in reads.** Most prior work (trace-level MIN gaps, lookahead hit rates) stops at hit or read counts. This paper measures what oracles buy in seconds on many machines, with the machine as the statistical unit. The 2x2 shows a real interaction: MIN's set, read once, beats both single changes. The mechanism is visible in the counters. On panel host Pa (099a), MIN's set admits 15.8 experts per token against the deployed policy's 3.5, and with two reads each admission costs a second read.
4. **The machine as the unit.** The paper takes host heterogeneity seriously. The machine carries 96% of the variance of the read-once gain (I get 96.3%, with problems at 1.1%). The link-to-CPU ratio predicts which read path pays (Spearman 0.89 over 19 machines, which I reproduced). This is a useful, actionable finding for builders of consumer offloading systems.
5. **Honest scoping in the body.** The post-hoc 0.5 threshold is labelled in Section 4, Table 3 and Section 9. The fact that a probe can be exceeded is stated, and so is the use of the GPU's datasheet rate. The closed-form account is labelled exploratory after its registered tests failed.

## Verification against raw data

Notation: "own" means my code. Rentals are identified by result-directory prefix (for example 099f). Ratio is the link-to-CPU ratio Bp/Bc as defined in the authors' fetch-table code (CPU rate at the helper count, zero-copy link rate).

| # | Claim | Location | My value (from raw data) | Verdict |
|---|---|---|---|---|
| 1 | R\* (MIN with bypass, per layer, cold start, cache carried across problems), gpt-oss 11/25/40% | Sec. 3, Table 4 | own simulator: 38.32 / 15.34 / 6.78 reads per token (identical to the authors' recorded values) | consistent |
| 2 | R\* on the first 20 problems changes by −0.4% | Sec. 9 | −0.38% at 11% and at 25% | consistent |
| 3 | Pooling slots across layers lowers MIN's reads by 4.5 to 18.6% | Sec. 9 | own pooled simulation: 4.5% (Qwen3 12.5%), 5.5% (gpt-oss 11%), 18.6% (Qwen3 43.75%) | consistent |
| 4 | Host B bounds: 172/430/519/96/214/308 tok/s at the datasheet GPU rate; 172/279/279/96/192/192 at the measured GPU rate | Table 4 | identical (B_host 87.5 GB/s, own R\*) | consistent |
| 5 | Host B rows (gpt-oss 11/25/40%, Qwen3 12.5%) and host S rows (all six); e.g. host B gpt-oss 11%: 34.9 / 54.0 / 69.9 tok/s, 1.294 | Table 4 | job 081 L2: 34.9 / 54.0 / 69.9; job 089 rows all match (e.g. Qwen3 43.75%: 23.8 / 98.3 / 95.8) | consistent |
| 6 | Leads FreeToken at 11 of 12 configurations; 2.0 to 4.0 times llama.cpp | Sec. 3 | 11/12; 2.00 to 4.02 | consistent |
| 7 | FreeToken tuned still trails at 5 of 6 cells | Sec. 3, App. M | ours over FreeToken's best: 1.17, 1.17, 1.09, 1.02, 1.04, 0.98 | consistent |
| 8 | 25 consumer machines at gpt-oss 11%: 31 to 54% of Eq. (1), 40 to 54% of Eq. (3), 55 to 70% of Eq. (2) | Sec. 3, abstract | 25 machines; 31.2 to 54.0, 40.5 to 54.0, 55.2 to 70.3 | consistent |
| 9 | Eq. (3)/Eq. (1): median 1.00, up to 1.56 | Sec. 3 | 1.00, 1.56 (TR 9960X, ratio 0.32) | consistent |
| 10 | The probe's second-highest reading is a median 0.8% below the highest | Sec. 3 | 0.82% (consumer), 0.89% (all) | consistent |
| 11 | T_GPU = 2.9 ms, the smallest Nsight profile | Sec. 3 | 2.94 ms (range 2.94 to 3.38 over 14 profiles) | consistent |
| 12 | The microbenchmark reaches 86 to 96% of the bound's read time (per layer) | Sec. 3, Table 29 | 86.1 to 96.3% (token mode 92.9 to 97.4%) | consistent |
| 13 | Table 3, 11%: 1 [−1,3]; 1 [0,2]; 39 [32,45]; 15 [11,18]; 47 [42,51] | Table 3, Sec. 4, abstract | 1.2 [−0.7,3.0]; 0.9 [−0.1,2.0]; 38.8 [31.9,45.4]; 14.6 [10.5,18.3]; 46.5 [42.1,50.9] | consistent |
| 14 | Table 3, 25%: 21 [19,22]; −1 [−4,1]; 35 [30,40]; 20 [18,22]; 44 [40,48] | Table 3 | 20.8 [19.2,22.3]; −1.1 [−4.0,1.3]; 35.3 [30.1,40.4]; 20.5 [18.1,22.3]; 44.3 [40.1,48.4] | consistent |
| 15 | The best oracle closes a mean 54% (49 to 58) where the ratio is at least 0.5, and at most 17% on the two slow-link machines | Sec. 4 | 53.8 [49.5, 58.0]; 16.8 and 1.2 (ratios 0.40 and 0.29; the next ratio up is 0.54) | consistent |
| 16 | T_GPU is 27 to 31% of the gap at 11% and 33 to 38% at 25% | Table 3 caption, Sec. 4, Sec. 7 | 26.9 to 30.9; 33.4 to 38.4 | consistent |
| 17 | MIN prefetched runs at 1.07 to 1.38 times Eq. (2) at 11% | Sec. 4 | 1.07 to 1.38 (13 machines) | consistent |
| 18 | The prefetching oracle reads 1.22 to 1.31 times MIN's reads at 11% | Sec. 4 | 1.218 to 1.312 from counters; **1.42 to 1.44 at 25% (not stated)** | consistent (incomplete) |
| 19 | Running example (Ryzen 9 9950X, 11%): 10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms | Sec. 5 | 096a: 10.02 / 12.96 / 20.64 / 15.16 / 13.81 | consistent |
| 20 | Positive interaction "held on 11 of 12 machines" | Table 1 | 11 of 12 positive at 11% (point estimates); negative only on 099f (ratio 0.29) | consistent (2 of the 12 are job 096 hosts; see W5) |
| 21 | Fig. 2: 15 machines ran every state, one launch per GPU; 13 have ratio at least 0.5 | Sec. 4, Fig. 2 | 15 unique GPUs (096a/b, 099a to j, 100b, 100f, 101b), 13 at ratio at least 0.5 | consistent |
| 22 | MIN read once is 16 to 51% faster than deployed at every budget on the 3 factorial machines, and up to 81% faster read ahead | Sec. 5, Fig. 3 | 1.160 to 1.514; up to 1.812 | consistent |
| 23 | No-bypass prefetch and serve-then-copy gain at most 15% or lose at the smallest budget of each model | Sec. 5 | maximum 1.148 (no bypass, 1 read, Qwen3 12.5%, O4) | consistent |
| 24 | Fewest-admission, read once, at 11%: beats deployed on all 9 stable machines, geomean 1.21x; 1.17x over 13; loses on the two slowest links | Sec. 5, Table 1 | 9/9 gain, 1.215x; 1.166x over 13; losses at ratios 0.14 (106c) and 0.21 (107e) | consistent |
| 25 | Fewest-admission makes 0.60 to 0.73 of greedy's copies; in the engine it misses 2.5 to 5.3% more | Sec. 5, App. L | counters: 0.597 / 0.726 copies; 1.025 / 1.053 misses | consistent |
| 26 | At ratios 0.28 to 0.32, greedy read once loses (0.85 to 0.93x); fewest-admission served by the CPU gains 1.11 to 1.22x | Sec. 5 | 0.853 to 0.933; 1.112 to 1.217 | consistent |
| 27 | Spearman 0.89 [0.63, 0.97] over 19 machines | Sec. 5, Fig. 4 | 0.89 [0.62, 0.98] (19 machines, jobs 093 to 104) | consistent |
| 28 | The machine carries 96% of the gain's variance; problems at most 3% | Sec. 5 | 96.3% / 1.1% (log ratio, two-way decomposition) | consistent |
| 29 | A larger admission margin cuts reads 4 to 6% at 11% and runs 1.021x (11%) and 1.018x (25%) on 5 stable machines | Sec. 5 | 4.3 to 5.8%; 1.021x / 1.018x | consistent |
| 30 | The layer-ahead copy gains 1.04x on a fast link and falls to 0.72x elsewhere | Sec. 6 | 1.042 (105f) and 0.722 (105e) | consistent |
| 31 | Next-token window recovers 0.18 (11%) and 0.07 (25%), at most 0.31 at any host-bound budget; 2 exact tokens give 0.31; half needs 4 and 10 tokens | Sec. 6, abstract | own replay: aa 58.64, w1 55.07, w2 52.57, w4 48.58, w16 39.72, MIN 39.26 → 0.18 / 0.31 / 0.52 / 0.98; 25%: 0.07; Qwen3 12.5% w1: 0.308; W50 about 3.8 and 10.4 | consistent |
| 32 | Engine 16-token window recovers 0.78 to 0.93 at 11% (registered at 0.80 or more; the lowest machine missed) | Sec. 6 | 0.78 to 0.93 over 10 panel machines | consistent |
| 33 | Implied read rate 0.88 to 1.16 of B_host over 39 consumer launches; more than 5% above on 12 of 84 launch-budgets | Sec. 4, Sec. 9, App. K | 0.88 to 1.16 (39 launches); 1 (11%) + 11 (25%) = 12 of 84 | consistent |
| 34 | Same-CPU panel machines differ by up to 29%; a re-rented machine came within 3.2% | Sec. 2 | 29.0% (two Core Ultra 9 285K); 3.2% (TR 9960X, 105b vs 106a) | consistent |
| 35 | On the AIME routing, the window for half the gain holds 0.66 to 0.81 C distinct experts | Sec. 6, Sec. 10 | coarse own estimate 0.71 to 0.82 C | roughly consistent |
| 36 | "held at most cells on three machines with ratios near one" | Table 1, row 5 | O3 0.82, O4 1.05, **O5 0.57** (App. H itself gives O5 47/27 GB/s) | **inaccurate** |
| 37 | Every prediction was committed before its machine started (71 rentals, jobs 093 to 108) | App. C | 71 rentals; every rental started at least 4.0 s after its job's first commit (commit time; push times not checkable offline) | consistent |

Not checked:

- the audit of published systems (median 13.6%; it depends on curated rows and datasheet rates);
- the forecaster and speculative-batch results;
- the 9-model horizon rule;
- the engine source.

---

## Weaknesses (most important first)

**W1. The decomposition is a property of one engine design, but the builder rules are stated generally.**

- The "gap" is the authors' own deployed cache's time beyond Eq. (1). That cache serves admissions on the CPU and then copies them, so it reads every admission twice.
- The headline interaction ("pay only together") follows directly from that design:
  - MIN's set admits about 4.6 times as many experts per token as the deployed policy (15.8 vs 3.5 on 099a), so reading it twice cancels its hit gain.
  - The deployed policy admits so little that reading its admissions once changes little.
- For a system that already reads once (FreeToken, or the paper's own admit-every-miss), the 2x2 would look different. No such decomposition is shown.
- The decomposition covers one model (gpt-oss-120b) at two budgets on one GPU model. The prompts (AIME) were also used during development, and the machines are a convenience sample (13 after the post-hoc filter).
- Despite this, Section 7, rule 2 ("Change what is cached and how it is read together") and the abstract present the interaction as general guidance. It should be scoped to serve-then-copy caches, or shown on a second design or model. The Qwen3 panel cells lack the bypass and prefetched states needed for the 2x2.

**W2. The "other half no oracle removes" is mostly accounted for by named items, including the oracle's own inefficiency.**

The third step of the decomposition (MIN read once, then MIN prefetched) does not change only *when* reads happen:

- the prefetching oracle uses a different, paced and queue-capped copy path;
- its prefetch rule reads 1.22 to 1.31 times R\* at 11% and, by my count from the counters, **1.42 to 1.44 times R\* at 25%**. The main text gives only the 11% figure.

Charging those excess reads at the probe rate and adding the median T_GPU accounts for the whole residual. On the 13 machines:

| Budget | Excess oracle reads, share of gap | T_GPU, share of gap | Sum | "Left" |
|---|---|---|---|---|
| 11% | 22% (18 to 29%) | 29% | 51% | 47% |
| 25% | 20% | 36% | 56% | 44% |

Section 4 names both items. But the abstract ("no oracle we built removes the other half") and the conclusion ("The other half no oracle we built removes; it is larger than the GPU's own non-expert time") invite the reader to see an unexplained or irreducible half. The paper should report this accounting. Better, it should build an oracle whose prefetches read about R\*: unpaced or uncapped, or the fewest-admission schedule with a lead.

**W3. Post-hoc choices propagate into unlabelled headline numbers.**

- The 13-machine subset (ratio at least 0.5) is labelled in the body. The abstract, introduction and conclusion quote 39% / 15% / about half without saying the subset was chosen after the data.
- Over all 15 machines, "together" is 33% [22, 42] and the best oracle closes 48% [38, 56]. The shares also change continuously with the ratio: "together" is 42% at ratios of at least 0.6 and 47% at ratios of at least 0.7.
- The threshold sits in a natural gap (0.40 to 0.54), so I do not suspect tuning. But a fit against the ratio over all 15 machines would be more honest than a subset mean.
- The introduction states "where the PCIe link reads at 0.32 of the CPU rate or less, copies belong in the background" without hedging. That rests on two stably-run machines (Pf and the TR 9960X). Rule 4 in Section 7 hedges it properly.

**W4. The bound is relative to one probe run, and the abstract overstates its reach.**

- B_host is the maximum of one probe run, usually taken while the model was downloading.
- By the paper's own closed-form account, the engine's reads imply rates more than 5% above B_host on 12 of 84 launch-budgets, up to 1.22 times on 100f at 25%.
- The top of the "31 to 54%" range is that same machine. 100f's probe reads 52.0 GB/s, the same as the 9950X machines 099e, 096a and 105f (50.7 to 51.9). Yet its deployed cache runs at 18.1 ms per token against their 20.5 to 20.6. Its probe also shows the lowest CPU (40.3) and link (21.8) rates of any 9950X. Most likely the probe under-read this machine, which inflates its share.
- The abstract says MIN's reads at the probed rate "bound every system that executes the exact routing". The derivation covers per-layer caches of C whole experts at the probe's rate. Pooled caches (FreeToken) read 4.5 to 18.6% less, and a reader faster than the probe can beat it.
- The body states these caveats; the abstract should too. Re-probing after the download (or the median of repeated probes) would make the shares far more defensible. I checked that no measured configuration beats Eq. (1): the best is 1.33 times it.

**W5. Some evidence labels are inaccurate or generous.**

- Table 1, row 5 says the result held "on three machines with ratios near one". O5's ratio is 0.57, and O3's is 0.82.
- Row 4 counts 12 machines as registered. The two job 096 hosts registered a different 2x2 ({MIN, Belady} x reads), not the deployed-set x reads interaction. Only job 099's prediction 5 registered it, and that prediction's second clause (one read alone changes time by at most 5%) failed on the slow-link hosts.
- "Held" in Table 1 is mostly on point estimates: only 218 of 565 clauses held with an interval.
- The headline summaries are post-hoc aggregates of per-host directional predictions: geomean 1.21x over 9 machines, 1.021x over 5, and 39% over 13. That should be stated where they appear.

**W6. Novelty is moderate.**

- The bound is a roofline with Belady's MIN as the byte count.
- Trace-level MIN gaps, lookahead hit rates and stage-level perfect-knowledge studies exist (Zhang 2026b; Liang et al. 2026b; SAEM; SpecMD).
- Choosing among hit-optimal schedules by admission cost has precedents (Berger et al. 2018; Jain & Lin 2018).
- What is new is the in-engine, many-machine timing decomposition and the registered record. That is a solid but incremental systems contribution.

**W7. Smaller issues.**

- **Mixed run sets.** The decomposition pools runs of 20 and 30 problems from three engine versions (disclosed). R\* differs by only 0.4%, so this is minor.
- **Stability and the claims it supports.** "Ran stably" is defined by round-to-round spread, but most of the 15 decomposition machines ran one process. Stability was therefore not checked for them, yet it is used to exclude machines elsewhere. The two fewest-admission losses fall on exactly the two slowest-link machines, which were also the unsteady ones (106c computed wrong outputs). "Gains on every machine that ran it stably" is therefore partly an artifact of exclusion. The paper discloses this, but the Table 1 wording hides it.
- **The audit's medians are not like-for-like.** The comparison with published systems (13.6% vs 27%) mixes datasheet rates, uniform-routing stand-ins and pooled or speculative designs scored against a per-layer, one-token bound.
- **A GPU-bound 25% budget.** At the measured GPU rate, gpt-oss 25% is GPU-bound on 3 of the 13 machines. I checked that this moves the Table 3 shares by at most 1 point, so it is fine, but it is worth a sentence.

---

## Clarity

**Assessment: 3/5.** Earlier rounds scored clarity 2 and then 3. The restructuring helped, but the paper is still hard to read for a general MLSys audience.

**What works.**

- **Question-driven structure.** Sections 3 to 6 answer "how fast could it be / where do the seconds go / how must foresight be spent / how much is needed". Each section opens with its answer.
- **Table 1.** Mapping each claim to its evidence type is excellent and should be kept.
- **Table 2 and Figure 1.** Table 2 describes every configuration in words, with a "reads" column. Figure 1 makes the double read concrete.
- **The running example.** The 9950X example (10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms) gives the reader a fixed point.
- **The "Yardsticks" paragraph.** It names the four normalisations explicitly.
- **Sections 7 and 9.** Section 7's numbered rules and Section 9's itemised limitations are clear and honest.
- **Sentences.** They are short and declarative, and terms are defined before use.

**What still makes it hard to read** (roughly in order of cost to the reader):

1. **Numbers in almost every sentence.** Many sentences carry two to five quantities with ranges, intervals, machine counts and parenthetical qualifiers. For example, Section 3's "Where systems stand" packs three bounds, two systems, a probe caveat and a pointer to the closed-form account into one paragraph. The narrative is hard to pull out on first reading. The main text would read better with one number per claim and the ranges moved to tables.
2. **Machine sets and baselines shift from claim to claim.**
   - Claims are made over 25, 15, 13, 12, 9, 13, 19, 5, 3 and 10 machines, drawn from different jobs.
   - Host labels (A, B, S, O1 to O6, Pa to Pj, Pf) leak into captions and the appendix.
   - The four yardsticks use different denominators. Section 6's baseline (admit every miss) is the online policy that reads least, but in time it is up to 2.4 times *slower* than the deployed cache on slow-link machines (its speed is 0.42 of the deployed cache's on 099f and 0.56 on 099d). A reader moving from Section 5 to Section 6 will be confused.
   - A small "machine sets" table (set, size, jobs, used in) would help a lot.
3. **Symbols and terms are overloaded.**
   - Bc and Bp mean two different things. For the link-to-CPU ratio they are the CPU rate at the helper count and the zero-copy link rate; in Eq. (3) they are the best CPU and best link readings. The code confirms the two definitions.
   - There are three GPU-time constants with similar roles: T_GPU (2.9 ms, non-expert), G (4.1 to 4.7 ms, App. K) and G_fit (App. D).
   - "MIN" names the optimum, several oracle configurations ("MIN, 1 read", "MIN, prefetched") and "MIN's set", even though the paper stresses that many hit-optimal sets exist.
   - Fetch, copy, admit, read twice, late, ahead, budget, launch, cell and "ran stably" all need to be held in memory at once.
4. **Figures 2 and 3 are hard to decode, and Table 3 invites a wrong sum.**
   - Figure 2 puts two alternative single changes at one x position, with two different "median" paths, over 15 overlapping lines. The 2x2 has to be reconstructed from a staircase.
   - Table 3 lists two alternative rows above three additive rows. The caption explains this, but readers will try to add all five.
   - Figure 3 packs seven configurations by six budgets, plus panel ticks, into a small forest plot.
   - Table 4's caption runs about ten lines and carries essential definitions.
5. **Appendices carry essential content, and some numbers disagree.**
   - The PDF is 43 pages with about 30 tables. Main-text claims point to Tables 23, 27, 28 and 29 and to Appendices C, K, L and M.
   - Appendix K is summarised in the main text and then declared "not relied on".
   - The horizon rule is "about 0.65 C" (9 models) in the abstract and introduction but "0.66 to 0.81 C" (the AIME routing) in Section 7 and the conclusion, with no signpost that these are two different quantities.
   - Table 1, row 5 mislabels the machines (W5).
   - The abstract sentence "Oracles built into the engine then remove parts of the rest of its time, the gap" is awkward.

---

## Questions for the authors

1. Can you fit the decomposition shares against the link-to-CPU ratio over all 15 machines, instead of averaging over the post-hoc 13? Will you report the all-15 figures (33% together, 48% closed at 11%) next to the subset?
2. How much of the residual remains with a prefetching oracle whose reads equal R\* (unpaced, no queue cap, or the fewest-admission schedule with a lead)? By my accounting, T_GPU plus the current oracle's 22 to 44% excess reads covers the "left" share at both budgets.
3. Does "pay only together" hold for an engine that already reads admissions once? Could the 2x2 be run with admit-every-miss or a FreeToken-like design as the baseline? If not, will you scope rule 2 to serve-then-copy caches?
4. How sensitive are the 31 to 54% shares to the probe? 100f's probe matches the other 9950X machines (about 52 GB/s), yet its deployed cache is 12% faster than theirs. Would re-probing after the download remove the launches whose implied rate exceeds B_host?
5. Which three machines does Table 1, row 5 refer to, and what are their ratios?
6. Why are bypass and prefetched states missing from the Qwen3 panel cells? Is a second-model decomposition feasible?
7. The fewest-admission schedule misses 2.5 to 5.3% more in the engine than in replay. Is that copy latency, or a semantic difference in the engine's plan?

## What would raise my score

- **Clarity.** A main text with fewer numbers per sentence, one definition each of Bc and Bp, a machine-sets table, host IDs moved out of the main text, a redrawn Figure 2 (an explicit 2x2 panel plus a separate "ahead" bar), and one consistent horizon number with its scope.
- **Labelling.** The post-hoc subset labelled in the abstract and conclusion, with the all-15 numbers given; Table 1, row 5 corrected; and a note that the summary statistics aggregate per-host directional predictions after the fact.
- **The residual.** An explicit accounting of the 47% / 44% "left" (T_GPU, excess oracle reads, unexplained), or an R\*-reading prefetch oracle.
- **Scope.** Either a decomposition on a second engine design or model, or builder rules scoped explicitly to serve-then-copy caches on this hardware class.
- **The probe.** A probe-sensitivity analysis: repeated or post-download probes and the effect on the shares.

## Scores

- Overall: **6/10** (weak accept). The measurements are exceptionally reproducible and honestly registered, and the in-engine oracle decomposition is a useful method. The scope is narrow, the residual's framing overstates what is irreducible, and the paper is still hard to read.
- Soundness: **4/5**. Every number I checked reproduces from raw data. The remaining problems are framing, scoping and labelling (W2 to W5), not wrong numbers.
- Significance: **3/5**
- Novelty: **3/5**
- Clarity: **3/5**
- Confidence: **4/5**. I verified 37 claims from raw data. I did not verify the audit of published systems, the forecasters, or the engine code.

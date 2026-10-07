# Review 20a — MLSys 2027 main track, program committee

**Submission:** "Where the Seconds Go: Bounding Mixture-of-Experts Decode from Host Memory, and Measuring What Foresight Is Worth"
**Reviewer:** PC member, blind review, 7 October 2026

## Materials read

- `paper/paper.pdf`, 37 pages. I read the main text (pp. 1–10) in full, both as rendered pages and in `paper.tex`. I read the appendices from source: `app_wsg.tex` (Appendices A–E, J, K, L), `app_relation.tex` (H), `app_more.tex` (I), `app_value.tex` (G) and the start of `app_traces.tex` (F). I also read the generated tables `tab_dm`, `tab_headline`, `tab_configs`, `tab_names`, `tab_prereg` and `tab_readsched`.
- `paper/supplement.pdf`: I looked at its structure and at the job-099 clause rows (P5, the interaction clauses).
- Authors' scripts, read **only for definitions**:
  - `scripts/decomp_measured.py` (states, gap, parts);
  - `scripts/speed_limit.py` (`host_rates`, `limit`);
  - the header of `scripts/speed_limit_v2.py`;
  - `scripts/sumlaw_paper.py` (`profiles`, the T_GPU definition);
  - `scripts/fig_decomp.py` (machine selection, Eq. 3);
  - the slow-link selection in `scripts/job106.py`;
  - `mosl/cachesim.py` (`_sim`, the step semantics of MIN).
- `prereg/*.json`:
  - `speed_limit_v2.json`, only to compare my R\* with theirs;
  - `reanalysis.json`, for the pooled G;
  - `job104/105/106.json`, for how the ratio is defined.
- `gpu/vast_ledger.json`: the rental start times.
- Raw results in `/home/claude/gpu-branch/results/`:
  - `ec_*.jsonl` rows (decode_ms, n_decode, nll, the `stats=` path), `st_*.json` counters and `concur.txt` probes for jobs 093–108;
  - Nsight `prof_C*.json` and `g_prof.json` for 069c and 105–108;
  - `bs1.jsonl` for jobs 081, 089 and 098;
  - `readsched_C*.txt` for 102a–c;
  - the AIME routing trace `084c_gptoss_trace@vast/route_aime25_gptoss.npz`.
- `jobs/` on the gpu branch:
  - the headers of `099_panel`, `102_readsched` and `104_minadm`;
  - the content diffs of the commits made after launch to `102`, `105` and `106`;
  - the prefetch-oracle code in `jobs/ec2/llama.cpp-expert-cache-4da6337-oracle3.patch`.
- `git log --format='%h %ad'` / `%at` on the gpu branch, for commit times only.

All my analysis code is in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/rev20a/`:

| Script | What it checks |
|---|---|
| `rstar.py`, `rstar2.py` | R\* recomputed |
| `globalmin.py` | pooled-slot MIN |
| `catalog.py` | raw-data loader |
| `decomp.py` | Table 3 and Fig. 2 |
| `shares.py` | Section 3 shares |
| `reads.py` | counters relative to R\* |
| `factorial.py` | Fig. 3 |
| `plan.py` | fewest-admission schedule |
| `readpath.py` | link ratio and Spearman |
| `dk.py` | admission margin |
| `window.py` | Section 6 from the trace, my own re-implementation |
| `panelwin.py` | Section 6 in the engine |
| `implied.py` | closed-form implied rate |
| `headline.py` | Table 4 |
| `misc.py`, `second.py`, `regtime.py` | the other checks |

## Independence statement

**What I did not open:**
- anything under `reports/`, apart from writing this file;
- `prereg/*outcome*.md`;
- `research_notes/`;
- `paper/paper_v1_prereview.tex`;
- `apply/`;
- any file named like a review, a number check, a plan or a progress log.

**Git and outside help:**
- I read no commit messages. From git I used only hashes, dates, Unix times and file-content diffs (`git show --format= <hash> -- <file>`).
- I used no other reviewer's material.
- I did not use the parent scratchpad. A `find` for large files listed some of its file names; I opened none of them.

**How the numbers were produced:**
- Every number marked "my value" below comes from my own code run on the raw rows, counters, probes, profiles or the routing trace.
- I read the authors' scripts only to learn definitions: which configuration is "deployed", how B_host and T_GPU are defined, and which machines enter a set.
- My Section 6 window policy is written from the prose of Appendix G, not from `value_map.py`.

**Changes to the repositories:** none, apart from writing this file. I made no commits.

---

## Summary

The paper studies batch-1 decode of MoE models (gpt-oss-120b, Qwen3-30B-A3B) on rented RTX 5090 machines when most experts live in host DRAM. It makes four contributions.

**1. A bound (Eq. 1).** No policy with C slots per layer reads fewer experts than Belady's MIN with bypass (R\*). So R\*·S/B_host is a lower bound on time per token at host-bound budgets, where B_host is the highest reading of a bandwidth probe.
- Two tighter variants: Eq. 2 adds the engine's non-expert GPU time T_GPU in series, for systems that read on demand. Eq. 3 respects the layer order.
- A microbenchmark reaches 86–96% of the bound's read time.
- The authors' llama.cpp expert cache beats FreeToken at 11 of 12 cells and stock llama.cpp by 2–4×. On 25 consumer machines it runs at 31–54% of Eq. 1 at gpt-oss 11%.

**2. A decomposition of the gap**, the deployed cache's time beyond Eq. 1. Oracles built into the engine change, alone and together:
- *what* is cached: MIN's set instead of the deployed one;
- *how* it is read: once instead of twice;
- *when* it is read: copies issued ahead.

Results, on 15 machines:
- At gpt-oss 11%, each of the first two changes alone closes about 1% of the gap; together they close 39% on the 13 "fast-link" machines (a post hoc subset).
- Reading ahead closes another 15%.
- The remaining 47% is attributed to T_GPU (27 points) and to the prefetching oracle's reads beyond MIN's (22 points).

**3. How foresight must be spent:**
- read once;
- use the fewest-admission hit-optimal schedule, which wins on all 9 stable machines by 1.21×;
- choose the read path by the link-to-CPU bandwidth ratio.

**4. What foresight is worth, by horizon and accuracy.**
- Exact next-token routing recovers 0.18 / 0.07 of MIN's read savings over admit-every-miss.
- Half needs about 4 / 10 tokens, which a cross-model rule restates as ≈0.65 C distinct experts per layer.

**Registration.** Every experiment's predictions were committed to a public branch before its machine started, and failures are reported (Table 8: 495 clauses held with an interval, 597 held on the point estimate only, 308 failed).

## Strengths

1. **Exceptional reproducibility; the central numbers hold up.** From raw rows and counters, with my own code, I reproduced the entire measured decomposition (Table 3, both columns, every interval) to within 1 point. The same holds for:
   - Section 3's shares over 25 machines (31–54 / 40–54 / 55–70%) and its Eq. 3 / Eq. 1 ratios (1.00, 1.56);
   - Table 4's ratios and paired-bootstrap intervals;
   - the fewest-admission geometric means (1.21× over 9, 1.17× over 13);
   - the admission-margin results;
   - the engine's window shares.

   I also recomputed R\* from the routing trace and got the artifact's 38.315 and 15.340 reads per token exactly, and the pooled-slot values 36.21 and 13.93. A re-implementation of the Section 6 window policy written only from the appendix's prose gives 0.176 / 0.069 for next-token foresight, against the paper's 0.18 / 0.07. All 35 checks are in the table below; I found no number in the main text that the raw data contradicts.

2. **A real registration record that can be audited.** I checked against the rental ledger:
   - Every job script for jobs 093–108 was committed before its first rental, usually 3–30 s before.
   - Edits after launch (jobs 100, 102, 105, 106, 107) exist and are disclosed in Appendix D. The diffs I inspected for 102, 105 and 106 only substitute hosts and leave the predictions unchanged.
   - Failed predictions are reported, including in Table 1's evidence column.

3. **A useful yardstick.** The bound needs only a routing trace and a probe run. The read term is shown to be reachable on three machines. Reporting "distance to the machine's bound" instead of speed-up over a chosen baseline is a good norm for this sub-field.

4. **The interaction result is genuine and mechanistically explained.** The counters confirm the stated mechanism:
   - MIN's set read twice admits 16–20 experts per token and reads 1.68–1.75× R\* at 11%. That is more than the deployed cache reads on 13 of the 15 machines (1.65–1.82×; the two slow-link machines are the exceptions), because late copies miss again.
   - Reading the deployed set once removes only its 3–8 admissions per token.
   - Together, the reads fall to 1.04× R\*.

   The 2×2 design within each machine, with the machine as the statistical unit, is the right design.

5. **Treating the machine as the unit.** Machines with the same CPU model differ by up to 29% (verified: two Core Ultra 9 285K machines in the panel). Most papers in this area use one host. The paper's bootstrap over machines and its "every machine of a stated class" framing are appropriate.

6. **Honest limitations section.** It covers rented, unsteady machines; a single probe run; pooled engine versions; and teacher forcing.

## Weaknesses (most important first)

### W1. The decomposition's third step and its "what is left" split are partly attributed, not measured, and the "ahead" oracle is not MIN's schedule read ahead.

**The "ahead" oracle.**
- Table 2 describes "MIN, prefetched" as "MIN, 1 read, plus copies issued three steps ahead". But the configuration that ran (`both3p`) sets `oracle_bypass=0` and `oracle_lead=3`.
- In the patch, the copies issued ahead follow a local rule: admit if the furthest resident's next use is beyond the candidate's *second* use. They are not MIN's schedule.
- The counters show the effect. On every one of the 15 machines this oracle reads 1.22–1.32× R\* at 11% and 1.42–1.44× at 25%, while "MIN, 1 read" reads 1.04× and 1.07–1.08×.
- So the "Then reading ahead: 15% / 20%" row is the *net* of a timing gain and an 18–35% increase in reads. The text's phrase "Issuing MIN's copies a few steps ahead closes another 15%" misdescribes it. The paper then charges those extra reads to "what is left" (22% / 20% of the gap), which is internally consistent but blurs the what / how / when separation that is the section's premise.

**The split of what is left.**
- The 47% / 44% is split into T_GPU plus extra reads plus a residual. Neither named part is removed by an oracle.
- T_GPU is a constant: the minimum non-expert time over 14 Nsight profiles, taken on 7 *other* machines (I get 2.94 ms; median 3.18, maximum 3.38). It is assumed to be fully serial.
- The extra reads are charged serially at B_host, although the oracle issues most of them ahead.
- At 25% the residual is −9% [−13, −5], so the two attributed parts demonstrably over-account. With the median profile instead of the minimum it would be about −12%.

**Consequence for the abstract.** "Most of the rest is the GPU's own work, which waits in series with reads that depend on routing" is an attribution under an assumption, not a measurement. It does not appear in Table 1 with an evidence label. The limitations bullet ("the GPU's non-expert time is profiled, not removed") is accurate, but the abstract and Section 7 rule 5 read as if it were measured.

### W2. The bound is relative to one maximal probe reading that the engine sometimes exceeds.

- B_host is the single highest reading of one probe run, which on most machines ran while the model was downloading.
- By the paper's own closed-form account (Appendix H), the deployed cache's implied read rate exceeds B_host by more than 5% on 12 launch-budgets. I reproduced this count. Examples:
  - 100f: 1.16× at 11% and 1.22× at 25%. This machine sits at the top of the 31–54% range and is one of the 13 "fast" machines in Table 3.
  - 095, 096a, 097a, 099a, 099b, 103b, 104b, 105a, 105f: 1.05–1.11× at 25%.
- If those implied rates are right, Eq. 1 is not a lower bound on those machines: the "54%" top end is inflated, and that machine's gap is understated.
- The paper is candid that the bound is "relative to the probe". But the abstract's "bound every system that executes the exact routing" is stronger than the evidence, and no sensitivity of Table 3 to the choice of B_host is reported. Candidates: the second-highest reading, a repeated probe, or a probe run on an idle machine.
- Similarly, Eq. 2 and Eq. 3 use *this engine's* kernel time, so they are not bounds for other systems. Section 3 calls Eq. 2 "a tighter bound" and notes "every system we compare reads on demand", which invites applying it to them.

### W3. The read-path conclusions rest on very few distinct slow-link machines, and the link-to-CPU ratio has no single definition.

**Few machines.**
- The "slow-link" findings rest on essentially two GPUs: the Core Ultra 9 285K "Pf", rented five times, and the Threadripper 9960X, rented twice. Server machines below 0.3 were mostly unstable.
- These findings are: "where the link reads at 0.32 of the CPU rate or less, copies belong in the background" (introduction), the 0.28–0.32 range in Section 5, and builders' rule 4.
- Rule 4 hedges this; the introduction does not.

**No single definition of the ratio.** Section 2 defines it as best PCIe reading over best CPU reading. The scripts behind several thresholds use a different B_c (the rate at the helper count) and a different B_p. Examples:

| Machine | Section 2 definition | Value used elsewhere |
|---|---|---|
| Threadripper 9960X | 0.34 | 0.32 (jobs 105/106; inside the "0.28–0.32" range) |
| 285K, host 104b | 0.60 | 0.49 (quoted in Appendix D) |
| 9950X, host 102b | 0.95 | 1.02 (Table 26) |

The thresholds in the text (0.32, 0.40, 0.5, 1/3 in code) were drawn after the data and lie close to these values. So class membership, and hence statements like "on every machine of a stated class", depends on which definition is used.

### W4. Some evidence labels are more generous than the registered clauses.

**"What is cached and how it is read pay only together — registered on the panel."**
- What job 099 registered (prediction 5) was a *positive interaction on every host at both cells*, plus "base→foa within 5%".
- At 11% the interaction clause failed on one host (Pf). The base→foa clause failed on Pf at both budgets and on Pd at 25%.
- "Only together", meaning MIN's set alone ≈ 0, was not predicted. Neither were the 39% / 15% shares (the table does say "shares over 13 after the fact").
- The label should read "sign of the interaction registered (failed on 1 of 10); the 'alone ≈ 0' pattern and the shares are exploratory".

**"Held".** Table 1's caption notes that "held" is mostly on the point estimate. Over all jobs, 597 clauses held on the point estimate only, against 495 that held with an interval, and 308 failed. The forecasting record is weak in absolute terms. This belongs in the main text, not only in a caption.

### W5. Section 6's yardstick measures foresight from a baseline the deployed system already beats.

- The "value of a window" is its share of MIN's gain over **admit every miss**, "the online policy that reads least". I confirm this in reads: 1.53× R\* against the deployed cache's 1.65×.
- In time, admit-every-miss is slower than the deployed cache on every panel machine: 0.42–0.97× at 11%.
- An exact 16-token window on this baseline is still slower than the deployed cache on the two slow-link panel machines (0.71× and 0.89×). An exact 1-token window is slower on 8 of 10.
- Read as a statement about host reads, the price of foresight is correct. A practitioner reading "exact 16-token window recovers 0.78–0.93 of the gain in time" will assume it is a gain over the system they run, which it is not everywhere. The main text should say this.

**The horizon rule.** "≈0.65 C distinct experts" is a one-constant fit that Appendix F concedes cannot be told apart from a two-parameter power law under leave-one-model-out. On the AIME gpt-oss routing I estimate ≈0.84 C at 11% and 0.68–0.74 C at 25%, slightly outside the stated 0.66–0.81 C at 11%. It is a useful heuristic, presented a little too much like a rule.

### W6. Scope.

- One GPU model; two models; one prompt set (AIME-25, which the limitations note was also used while developing the system); batch 1; teacher-forced replay; 20 problems in the later jobs.
- The decomposition pools three engine versions and runs of 20 and 30 problems. It is at two budgets of one model, with the 0.5 split drawn after the data.
- The audit of published systems against datasheet bounds (13.6% median, against "ours 27%") compares across different hardware, quantisations and design classes (pooled caches, speculative systems), against a per-layer one-token bound. It is suggestive, not a comparison.
- I did not verify the audit rows.

### W7. Comparison fairness is mostly handled, but asymmetries remain.

- Our FETCH table is computed per machine (and was swept). FreeToken's backend was picked on host B and carried over to host S.
- Job 098's tuned FreeToken (best of 5 settings, selected on the measuring launch, which favours FreeToken) still trails at 5 of 6 cells. I verified this, so this is a minor concern.

## Clarity

**My score: 3/5.** The revision is well organised at the macro level and every number is generated from the data (I found none that is stale). Reading it is still hard work: the main obstacles are terminology load, numerical density and float placement, not missing content.

### What works

- **One question per section.** The four questions (how fast, where the time goes, how to spend foresight, how much is needed) map onto Sections 3–6, and the "Yardsticks" paragraph names the metric each uses. This is the best structural decision in the paper.
- **Table 1.** Claims next to evidence labels, with a section pointer, lets a reader find the epistemic status of each claim quickly.
- **The running example in Section 5.** 10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms on one 9950X grounds every ratio that follows. I verified it.
- **Figure 1.** It makes the "two reads" mechanism concrete in one glance.
- **Table 3.** "Together + ahead + left = gap; the last three rows sum to left" is stated in the caption, and the arithmetic holds.
- **Post hoc scoping in the abstract.** The 0.5 split is labelled after-the-fact in the abstract, Section 4, Table 3 and the limitations. Few papers are this careful.

### What still makes it hard to read

1. **Terminology load and near-synonyms.**
   - The main text defines or uses about 14 configuration names: deployed, single read, admit every miss, MIN 1/2 reads, fewest-admission 1/2 reads, prefetched, no bypass 1/2 reads, window W, window with recall r, deployed + window.
   - It also uses three bounds, two "gaps", "host-bound", "launch", "ran stably", hosts B/S/O3–O5/Pf, and several machine classes.
   - The verbs overlap: fetched / copied / admitted / read once / read twice / in the step / background / late / ahead.
   - The appendix adds the codes base, foa, aa, both3p, hitopt, nb2, fetchplan, dk, pf and lrn.
   - "Single read" (the deployed set fetched) and "MIN, 1 read" are easy to confuse. "Fetch" means an in-step copy in the engine but MIN-read-once in the code.
2. **A name that misleads.** "MIN, prefetched" is not MIN's schedule (W1). The "Then reading ahead" row in Table 3 therefore reads as a pure timing change. Renaming it, for example "lookahead prefetch oracle (reads 1.2–1.4× R\*)", would remove a real misreading.
3. **Float placement.**
   - Table 3, the decomposition, appears on page 4 inside Section 3, before Section 4 begins.
   - Table 4, Section 3's system comparison, appears on page 5 inside Section 4.
   - Figures 4–5 appear on page 8, after the Section 6 text that cites them.
   - The reader has to page back and forth at exactly the points where the argument is built.
4. **Numbers per sentence.**
   - Many sentences carry 3–6 numbers with ranges, brackets and qualifiers. For example: "makes 0.60–0.73 of the greedy one's copies for the same hits in replay (in the engine it misses 2.5–5.3% more)", and the paragraph on the closed-form account.
   - The abstract carries about 12 numbers, including "(0.66–0.81 C on our own)" and two post hoc caveats.
   - Main-text paragraphs should carry the one or two numbers that support the claim and push the rest to tables.
5. **Many machine sets, never shown together.** The main text uses:
   - 25 consumer machines (Section 3);
   - 15 / 13 / 2 (Section 4);
   - 3 factorial machines;
   - 10 panel machines;
   - 9 / 13 (fewest-admission);
   - 19 (Spearman);
   - 5 (admission margin);
   - 3 (microbenchmark);
   - 2 hosts (Table 4).

   No main-text table lists which machine is in which set, with its CPU and link-to-CPU ratio. I had to rebuild this to check anything.
6. **Shifting thresholds and directions.**
   - The link-to-CPU thresholds change: 0.32 in the introduction and rule 4, 0.40 and 0.5 in Section 4, 0.28–0.32 in Section 5. The definition also changes between scripts (W3).
   - The metrics change direction: time relative to deployed (lower is better) in Fig. 2; speed relative to deployed (higher is better) in Figs. 3–4; share of a gap; share of a gain.
7. **A sentence that needs decoding.** Section 3: "These shares are relative to one probe run: its second-highest reading is a median 0.8% below the highest, and by the closed-form account of Appendix H, on the machine at the top of the range the engine's reads imply a rate above the probe's best (Table 24)." The reader must work out the consequence alone: the top of the 31–54% range is likely overstated.
8. **The evidence column of Table 1 is cryptic.** "registered and held on host B; host S's registered bands failed" and "registered on 7, held; two losses on unsteady machines" can only be decoded via Appendix D. One plain clause per row stating *what* was predicted would help.
9. **Appendix and supplement sprawl.**
   - 27 pages of appendix plus a 48-page supplement.
   - Appendix D is a dense wall of failure narratives.
   - Several appendix paragraphs (for example the double-reading oracles in Appendix E) carry 20+ macro-generated numbers each.
   - A reviewer cannot audit this in reasonable time without code. The supplement's scorecard is good; the appendix prose should be cut back to what supports a main-text claim.
10. **Minor notation.** Eq. 1 uses lowercase c (experts run on the CPU) next to uppercase C (slots per layer). Eq. 3 is introduced but used only for one range and one median ratio.

## Questions for the authors

1. **The prefetch oracle.** Why does `both3p` use `oracle_bypass=0` with a second-use admission test rather than MIN's schedule with copies issued three steps ahead? Can you run a "MIN 1-read set, copies ahead" oracle whose reads match "MIN, 1 read" (about 1.04× R\*)? That would separate *when* from *what* cleanly.
2. **T_GPU.** Can T_GPU's serial share be measured on the decomposition machines rather than attributed? For example, profile those machines, or remove the router dependency with an oracle that stages CPU misses ahead. How do Table 3's last three rows change with the median or maximum profile?
3. **B_host.** How sensitive are the 31–54% shares and Table 3 to the choice of B_host: the second-highest reading, a repeated idle probe, or the engine-implied rate on 100f?
4. **The ratio definition.** Which link-to-CPU definition underlies each threshold (0.32, 0.40, 0.5)? Would any machine change class under Section 2's definition?
5. **Windows on the deployed path.** In time, what is the value of the windows relative to the *deployed* cache (job 100's deployed-path windows), not relative to admit-every-miss?
6. **Held-out prompts.** AIME-25 was used during development. Do the decomposition shares hold on held-out prompts (for example the MATH-500 or AIME-2022 sets already used in Appendix J)?
7. **Table 1.** Can each evidence entry state exactly what was registered (sign, threshold or band) and on which machines?

## What would raise my score

- **Fix the "ahead" step and the attribution (W1).**
  - Rename "MIN, prefetched", or add a true MIN-read-ahead oracle.
  - Label the T_GPU / extra-reads split in Table 3 and the abstract as an attribution, with sensitivity to the profile chosen.
  - Remove "most of the rest is the GPU's own work" from the abstract, or hedge it.
- **Bound sensitivity (W2).** Report Table 3 and the Section 3 shares under a second B_host choice, and state in the abstract that the bound can be exceeded where the probe under-reads.
- **One ratio, one roster (W3).** Use a single definition of the link-to-CPU ratio throughout, and give a main-text machine roster: machine, CPU, ratio, and which sets it is in.
- **More evidence on slow links (W3, W6).** More distinct slow-link consumer machines, or the decomposition on Qwen3 or a second GPU type.
- **Clarity.**
  - Fix float placement.
  - Cut the main-text configuration names to those Section 4 needs.
  - Keep one or two numbers per claim sentence.
  - Use one direction of metric in the figures where possible.

## Scores

| | Score |
|---|---|
| **Overall** (1–10) | **6** (weak accept) |
| Soundness (1–5) | 3 |
| Significance (1–5) | 3 |
| Novelty (1–5) | 3 |
| Clarity (1–5) | 3 |
| Confidence (1–5) | 4 |

**Why these scores.** The measurements are unusually trustworthy: every main-text number I checked reproduces from raw data, and the registration is real. The bound and the oracle-in-engine methodology are useful to the community.

Soundness is held at 3 by interpretation, not arithmetic:
- the third step of the decomposition is mislabelled, and its "left" split is assumed rather than measured (W1);
- the bound is relative to a probe the engine sometimes beats (W2);
- the slow-link generalisations rest on about two machines under an inconsistent ratio definition (W3).

Novelty is moderate. Belady-with-bypass bounds and oracle hit-rate studies for MoE caches exist (Zhang 2026; Liang 2026b; Angelopoulos 2025). The new part is the time domain inside a real engine and the read-once / fewest-admission findings.

---

## Claims checked against raw data

Values in **Paper** are as printed. **My value** comes from my own code on raw `ec_*.jsonl` rows (mean over problems of decode_ms / n_decode), `st_*.json` counters, `concur.txt` probes, Nsight `prof_*.json`, `bs1.jsonl` or the AIME routing trace. Bootstrap intervals use 10,000 resamples over machines (or problems, for Table 4).

| # | Claim | Location | Paper | My value | Verdict |
|---|---|---|---|---|---|
| 1 | R\*, MIN-with-bypass reads per token, gpt-oss AIME, cold start, carried across problems | §3, Eq. 1 / Table 4 | (38.32, 15.34 in artifact) | 38.315 (C=14), 15.340 (C=32) with step-granular semantics; a request-granular MIN gives 0.8% / 0.3% more | Verified |
| 2 | R\* on the first 20 problems changes by −0.4% | §9 | −0.4% | −0.38% / −0.37% | Verified |
| 3 | Pooling slots across layers lowers MIN's reads by 4.5–18.6% | §9 | 4.5–18.6% | gpt-oss: 5.5% (11%), 9.2% (25%); pooled R\* 36.21 / 13.93 | Consistent (other cells not checked) |
| 4 | T_GPU = 2.9 ms, the smallest non-expert Nsight profile | §3 Eq. 2 | 2.9 ms | 2.94 ms (min of 14 profiles; median 3.18, max 3.38) | Verified |
| 5 | Running example, 9950X, gpt-oss 11%: Eq. 1, Eq. 2, deployed, MIN 1 read, MIN prefetched | §5 | 10.0 / 13.0 / 20.6 / 15.2 / 13.8 ms | 10.02 / 12.96 / 20.64 / 15.16 / 13.81 (host O4, job 096a) | Verified |
| 6 | 25 consumer machines; share of Eq. 1 / Eq. 3 / Eq. 2 at gpt-oss 11% | Abstract, §3 | 31–54 / 40–54 / 55–70% | 31–54 / 40–54 / 55–70% (n=25, first launch per GPU) | Verified |
| 7 | Eq. 3 / Eq. 1 on consumer machines at 11%: median and maximum | §3 | 1.00, 1.56 | 1.00, 1.56 | Verified |
| 8 | The probe's second-highest reading is a median 0.8% below the highest | §3 | 0.8% | 0.82% (25 consumer machines; 0.96% over all 33) | Verified |
| 9 | The read-schedule microbenchmark reaches 86–96% of the bound's read time | §3, Table 26 | 86–96% | per-layer mode: 86.1 / 89 (i9), 96.3 / 96 (9950X), 95 / 93 (9800X3D) | Verified |
| 10 | Table 4 ratios and intervals | Table 4 | host B gpt-oss 11%: 1.294 [1.278, 1.312]; host S: 1.207 [1.195, 1.219]; host S Qwen3 43.75%: 0.974 [0.962, 0.987] | 1.294 [1.278, 1.311]; 1.207 [1.194, 1.219]; 0.974 [0.962, 0.987] | Verified |
| 11 | Ours leads FreeToken at 11 of 12 cells; 2.0–4.0× llama.cpp | §3 | 11/12; 2.0–4.0× | 11/12; 2.00–4.03× | Verified |
| 12 | Bound at gpt-oss 11%, host B / host S | Table 4 | 172 / 140 tok/s | 172.3 / 139.8 | Verified |
| 13 | Tuned FreeToken still trails at 5 of 6 cells | §3, App. J (job 098) | 5/6 | 5/6 (1.17, 1.17, 1.09, 1.02, 1.04, 0.98) | Verified |
| 14 | Machines that ran every state; ratio ≥ 0.5; slow | §4 | 15 / 13 / 2 | 15 / 13 / 2 (one launch per GPU UUID) | Verified |
| 15 | Table 3, gpt-oss 11%: set alone / once alone / together / ahead / left / T_GPU / extra reads / residual | Table 3 | 1 [−1, 3]; 1 [0, 2]; 39 [32, 45]; 15 [11, 18]; 47 [42, 51]; 27 [24, 29]; 22 [20, 24]; −2 [−6, 2] | 1.2 [−0.7, 3.0]; 0.9 [−0.2, 2.0]; 38.8 [31.8, 45.4]; 14.6 [10.5, 18.4]; 46.5 [42.1, 51.0]; 26.9 [24.0, 29.6]; 21.8 [20.2, 23.6]; −2.2 [−6.4, 2.4] | Verified |
| 16 | Table 3, gpt-oss 25% (same rows) | Table 3 | 21; −1; 35; 20; 44; 33; 20; −9 [−13, −5] | 20.8; −1.1; 35.3; 20.5; 44.3; 33.4; 20.0; −9.1 [−13.3, −4.9] | Verified (the residual's interval excludes 0: the attribution over-accounts, see W1) |
| 17 | "33% together on 15 machines" | Abstract | 33% | 32.6% | Verified |
| 18 | The prefetching oracle reads 1.22–1.31× MIN's reads at 11% (1.42–1.43 at 25%) | §4 | 1.22–1.31 / 1.42–1.43 | 1.22–1.32 / 1.42–1.44; "MIN, 1 read" is 1.04 / 1.07–1.08 | Verified; this is W1 |
| 19 | With all three changes, time is 1.07–1.38× Eq. 2 at 11% | §4 | 1.07–1.38 | 1.07–1.38 | Verified |
| 20 | Best oracle closes 54% [49, 58] on fast machines; at most 17% on slow ones | §4 | 54 [49, 58]; ≤17 | 53.8 [49.3, 58.0]; 16.8 | Verified |
| 21 | The interaction's sign held on 9 of 10 panel machines | Table 1 | 9/10 | 9/10 at 11% (fails on Pf, ratio 0.30); 10/10 at 25%. Registered clause said "every host" | Verified; label generous (W4) |
| 22 | Same-CPU panel machines differ by up to 29%; a re-rental came within 3.2% | §2 | 29%; 3.2% | 29.0% (285K Pa vs Pf); 3.2% (Threadripper 9960X, 105b vs 106a) | Verified |
| 23 | MIN 1 read beats deployed by 16–51% at every budget on O3–O5; up to 81% read ahead | §5, Fig. 3 | 16–51%; 81% | 1.160–1.514; 1.812 | Verified |
| 24 | No-bypass and 2-read variants gain at most 15%, or lose, at the smallest budgets | §5 | ≤15% | max 1.148 (no bypass 1 read, O4, Qwen3 12.5%); several < 1 | Verified |
| 25 | Fewest-admission, 1 read, gpt-oss 11%: beats deployed on all 9 stable machines, geometric mean 1.21×; 1.17× over 13; loses on the two slowest links | §5 | 1.21× [1.13–1.30]; 1.17×; 2 losses | 1.215 (all > 1, min 1.035); 1.166; losses on EPYC 7302 (0.85) and EPYC 7663 (0.97), ratios 0.13 and 0.22 | Verified |
| 26 | Spearman of the MIN-1-read gain against link ratio over 19 machines | §5 | 0.89 [0.63, 0.97] | 0.91 [0.68, 0.98] (same 19 GPUs, first launch, ratio of mean times) | Approximately verified |
| 27 | At ratio 0.28–0.32: greedy 1 read loses (0.85–0.93×); fewest-admission 2 reads gains 1.11–1.22× | §5 | 0.85–0.93; 1.11–1.22 | 0.853–0.933; 1.112–1.217 | Values verified; the Threadripper's ratio is 0.34 under §2's definition (W3) |
| 28 | Admission margin: reads cut 4–6%; 1.021× at 11% and 1.018× at 25% on 5 stable machines | §5 | 4–6%; 1.021; 1.018 | 4.3–5.8%; 1.021; 1.018 (106a/b/e, 107b/d) | Verified |
| 29 | Next-token share 0.18 / 0.07; half of the gain at 4 / 10 tokens; 2 tokens 0.31; 8 tokens at recall 0.5 gives 0.29 | §6 | as stated | my re-implementation from App. G prose: 0.176 / 0.069; 0.50 at W=4, 0.46 at W=10 (half at about 11); 0.30; 0.28 | Verified (W50 at 25% slightly higher) |
| 30 | Admit-every-miss "reads least" (trace) and its engine counters | §6 | — | trace: 1.531 R\* (11%), 1.840 R\* (25%); engine counters 1.53 / 1.84 | Verified; but in time it is 0.42–0.97× the deployed cache (W5) |
| 31 | Engine: an exact 16-token window recovers 0.78–0.93 of the gain in time at 11%; registered ≥ 0.80, missed by the lowest machine | §6 | 0.78–0.93 | 0.78–0.93 (lowest: Pf) | Verified |
| 32 | Layer-ahead copy: 1.04× where the link is as fast as the CPU; down to 0.72× at 11% | §6 | 1.04×; 0.72× | 1.042 (105f); 0.722 (105e) | Verified |
| 33 | Closed-form implied read rate 0.88–1.16 of B_host over 39 consumer launches; more than 5% above on 12 launch-budgets | §4, §9 | 0.88–1.16; 12 of 84 | 0.88–1.16 (39); 12 (of 91 in my set) | Verified; supports W2 |
| 34 | Half of the value needs 0.66–0.81 C distinct experts on the AIME routing | §6 | 0.66–0.81 C | gpt-oss: ≈0.84 C at 11%, 0.68–0.74 C at 25% | Approximately (11% slightly above the range) |
| 35 | Every prediction committed before its machine started; amendments disclosed | §1, App. D | — | every job-script commit precedes the first rental (by 3–30 s for most jobs); later commits to 100, 102, 105, 106, 107 are disclosed, and the 102/105/106 diffs only substitute hosts | Verified |

**Not verified:**
- the audit of published systems (13.6% / 27%);
- the Qwen3-specific numbers (for example "at most 0.31", the Qwen3 bounds);
- KL parity;
- the 96% / 3% variance split;
- the 9-model horizon rule.

# Number check 17 (7 October)

Independent check of the parts of `paper/paper.tex` changed in commit daf964f (review round 17): abstract and
introduction, Table 1, the Section 2 "Yardsticks" paragraph and per-machine split sentence, Section 3's Eq. (3) and its
shares, Section 4's 2x2 split, T_GPU band and residual, Section 7's rules, the Conclusion, and the Limitations bullets on
the bounds and the decomposition.

## Method

All scripts are in the scratch folder `.../scratchpad/nc17/` and were run from there. Repository files were only read.

- **2x2 split** (`split2x2.py`). For every `*@vast` result directory I read the raw `ec_g_C14.jsonl` and `ec_g_C32.jsonl`
  rows and picked each state by the `stats=` file name (base, bypass, foa, fetch, both3p). Time per token is the mean
  of `decode_ms / n_decode` over problems. I kept one launch per GPU UUID (the first by directory name, parsed from
  `nvidia-smi-q.txt`). I computed Eq. (1) myself as `R* S / B_host` and checked that the GPU term is smaller at both
  budgets on these machines. `R*` comes from `speed_limit_v2.json` (exact). `B_host` is the highest of the CPU, PCIe and
  concurrent-sum readings in `concur.txt`. I parsed the T_GPU profiles directly from the 14 `prof_C14/C32.json` files of
  jobs 069c and 105, summing the non-expert categories. I split fast from slow links at a link-to-CPU ratio of 0.5,
  using both the helper-interpolated CPU rate (as the engine does) and the probe's best CPU reading. Intervals: a
  percentile bootstrap over machines (20,000 resamples, my own seed) and a t-interval. I also read the engine counters
  (`st_g_*.json`) for misses, admissions and fetches.
- **Eq. (3)** (`eq3.py`). Same machine selection as `fig_decomp.py`, rebuilt independently: jobs 093–108, one launch per
  GPU, the unsteady launches excluded, consumer processors only. I minimised Eq. (3) over a grid of f rather than with
  the script's closed form, with and without the clamp to Eq. (1), and also with the median T_GPU and the
  helper-interpolated `B_c` as sensitivity checks. Finally I ran the script's own `machines()` read-only to compare,
  machine by machine.
- **Labels.** I read the headers of job scripts 096, 099, 100, 101 and 102 and the scorecards `tab_scorecard.tex`
  (job 096), `tab_scorecard_099.tex` and `tab_scorecard_102.tex`.
- **Build.** `git archive HEAD` into `nc17/build`, then `latexmk -pdf -g paper.tex`; I read the log and the PDF
  (pdftotext and a page render).
- Importing the repository's scripts read-only refreshed three git-ignored bytecode files in `scripts/__pycache__`
  (`fig_decomp`, `robustness`, `job106`). No tracked file changed.

## Defects, most severe first

### 1. "Most of what remains is the GPU's own work waiting in series with the reads": the data on the prefetched state do not support this (abstract, introduction, Section 7 rule 5, Conclusion)

**Where.**
- Abstract: "most of what remains is the GPU's own work waiting in series with the reads".
- Introduction: "the GPU's own work in series with the reads is most of the rest".
- Conclusion: the same claim.
- Section 7 rule 5: "A quarter to a third of the gap is the GPU waiting in series with reads".
- Section 4: "Issuing MIN's copies a few steps ahead closes another 15%".

**Evidence.** Two separate problems.

- *The prefetched state reads more than MIN.* By the engine counters, "MIN, prefetched" (both3p) reads, per token,
  1.22–1.31 times MIN's reads R\* at 11% and 1.42–1.44 times at 25% (fast-link machines). MIN read once in the step
  (fetch) reads 1.04x and 1.07–1.08x. Job 096's simulation predicted this (both2 46.3 and 22.3 reads per token against
  R\* 38.3 and 15.3). At B_host, these extra reads alone come to a mean 22% of the gap at 11% (range 18–29%) and 20% at
  25%. That is more than the whole residual (18% and 8%). The main text never says that the "ahead" state reads more
  than MIN; it calls them "MIN's copies".
- *T_GPU cannot be wholly in series in that state.* On 8 of the 13 fast-link machines at 11% (10 of 13 at 25%), the
  prefetched state's measured time is below even the smallest T_GPU (2.94 ms) plus its own host reads at B_host. For
  example, on 099e: 2.94 + 11.91 = 14.85 ms, but it measured 14.08 ms. So in this state either the GPU's non-expert
  work overlaps the reads, or the reads beat the probe.

The 29% "serial" versus 18% "residual" split of what remains is therefore an accounting convention, T_GPU assumed fully
serial, not a measurement. Table 3 does label T_GPU "(assumed)"; the abstract, introduction and conclusion state it as a
finding.

**Fix.**
- In the abstract, introduction and conclusion, say "if the GPU's non-expert work stays in series, it is 29% of the
  gap" or drop "most of what remains".
- In Section 4, state that MIN prefetched reads 1.22–1.31x (11%) and 1.42–1.44x (25%) MIN's reads, so the residual
  mixes extra reads with overlap.
- In rule 5, give the numbers with their scope and the assumption.

### 2. Table 1's label for "pay only together" leaves out registered tests of exactly this claim, and a failure on the slowest-link host

**Where.** Table 1: "states measured in registered runs; split and link subset after the fact (15 machines)". The
Limitations bullet on the decomposition has the same gap.

**Evidence.** Two jobs registered the pattern before launch:
- Job 099, prediction 5, on all 10 panel hosts: "a positive interaction at both cells on every host … and reading the
  online policy's admissions once (base -> foa) changes the time by at most 5% on every host".
- Job 096, predictions 3 and 7, on O4 and O5: foa 0.97–1.12x base, and "bypass gains at most 8%" at gpt-oss 11%.

Scorecards:
- At gpt-oss 11%, the interaction (P5a) and one-read-alone (P5b) clauses held on 9 of 10 panel hosts. Both failed on
  Pf (099f, ratio 0.29): interaction −0.48 ms [−1.06, −0.14], foa/base 0.947.
- At 25%, P5b failed on Pd and Pf (0.922, 0.853).
- Job 096's P3b and P7a held on both hosts.

So 12 of the 15 machines had the 2x2 registered. The claim held at 11% on 11 of those 12, and failed on the slowest-link
host. The after-the-fact subset of 13 machines drops the two hosts where the registered clauses failed. The current
label reads as though the claim were post hoc and does not disclose the failure. The "(15 machines)" in the label also
conflicts with the 13 machines behind the 39%.

**Fix.** Use a label like: "registered on 12 machines (interaction; one read alone within 5%): held at 11% on 11, failed
on the slowest-link one; shares and link subset after the fact (13 of 15)". Mention the registered failure in the
Limitations bullet.

### 3. "Fast/slow link" is used for the link-to-CPU ratio but read as link speed (abstract, Section 3, Conclusion, Section 7)

**Where.**
- Abstract: "on the 13 machines whose PCIe link is fast".
- Abstract: "only when … the link is fast".
- Section 3: "where the link is slow, it is up to 1.56 times as high".
- Conclusion: "Where the link is fast / slow".
- Section 7 rule 3: "gains where links are slow and ties where they are fast".

**Evidence.** The 13 "fast-link" machines are those whose link-to-CPU ratio is at least 0.5.
- Seven of them have PCIe readings of 19–29 GB/s: 099i 19.2, 100f 25.3, 100b 27.7, 099g 28.3, 096b/099c/101b 28.8.
  099i's 19.2 GB/s is the slowest link of all 15 machines.
- The two "slow-link" machines read 28.4 and 28.7 GB/s.
- The 1.56x in Section 3 is 105b, whose link reads 57.5 GB/s, the fastest measured; its ratio is low because its CPU
  reads 169 GB/s.
- Rule 3's slow-link gains (0.08–0.17) come from machines with ratios 0.28–0.63. Section 4 calls ratios of 0.5 and
  above fast.

**Fix.** Write "whose link reads at least half as fast as their CPU" (or "link-to-CPU ratio") wherever "fast/slow link"
appears, and use one threshold.

### 4. Eq. (3): the displayed formula is not the bound computed, and its premise overreaches (Sections 3, 4 and 7)

**(a) The clamp is a definition, not a property.** "…and never below Eq. (1)" reads as a property of the displayed
expression, but the script imposes it with a max. Without the clamp, the expression is *below* Eq. (1) on 21 of 25
consumer machines at 11% (e.g. 093: 7.07 against 10.06 ms). That is because the link and the CPU each run at their own
best rate, which together exceed B_host. The reported "40–54% of Eq. (3)" and "equals Eq. (1) on the median machine"
both depend on the clamp; without it the shares are 32–52%. Mathematically the clamp is sound: the min over f of a max
with a third term, R\*S/B_host. It should be in the displayed equation.

**(b) "Only copies over the link can be issued ahead" overreaches.** Sections 3 and 4 and rule 5 state that only copies
over the link can be issued ahead ("which only copies over the link allow (Eq. 3)"; "and only over the link"). But CPU
threads can read the next layer's experts into their caches ahead of use. The engine has exactly this mode:
`LLAMA_EC_LLC`, "the helpers read the next layer's predicted CPU experts into their cache while idle", profiled as
C32llc in job 069c. Eq. (3) bounds only systems whose CPU reads an expert's weights when it runs it. The Limitations
state this assumption, but Sections 3, 4 and 7 present it as a fact.

**(c) T_GPU is engine-specific.** It is the engine's smallest profiled kernel time (2.94 ms, profiles from 7 machines),
not a hardware quantity. So Eqs. (2) and (3) bound systems with these kernels. This is disclosed in the Limitations, but
the "\bar T ≥" in Section 3 has no qualifier.

**Verdict.** Under its stated assumptions (CPU reads at use, the probe's best rates, T_GPU a floor on the non-expert
work, the clamp), Eq. (3) is a valid lower bound. The algebra of the min–max is right, and the dependency argument (a
CPU expert needs its layer's attention; the next attention needs its output) is right.

**Fix.** Write Eq. (3) as min over f of max((1−f)R\*S/B_p, T_GPU + fR\*S/B_c, R\*S/B_host). In Section 3, state "for
systems whose CPU reads an expert when it runs it". Soften "only copies over the link" in Sections 4 and 7.

### 5. The 39% and the "quarter to a third" are stated without their scope (introduction, Section 7 rules 2 and 5, Conclusion)

**Where.**
- Introduction: "doing both closes 39%".
- Rule 2: "a better set read once closes 39%".
- Conclusion: "pay only together … most of what remains", with no subset given.
- Rule 5: "A quarter to a third of the gap".

**Evidence.**
- The 39% is the mean over the 13 fast-link machines. Over all 15 it is 33%, and on 099f "both" costs time (−23%).
- T_GPU's share is 29% [26, 32] at 11% (profile band 27–31%), but 36% [33, 39] at 25% (band 33–38%). That is above a
  third, and rule 5 names no budget.

**Fix.** Add "on the 13 machines whose link reads at least half their CPU rate" to the introduction, rule 2 and the
Conclusion. In rule 5, write "29% at 11% and 36% at 25%, if it stays in series" (see defect 1).

### 6. Stale "one part at a time" framing contradicts the new 2x2 (introduction, Contributions, Figure 2 caption)

**Where.**
- Introduction: "build oracles into a real engine that remove one part of the gap at a time".
- Contributions: "a measured decomposition of the gap into five parts, three of them removed one at a time by oracles".
- Figure 2 caption: "as the oracles remove one part of the gap after another".

**Evidence.** The new result is that at 11% neither of the first two parts closes anything alone (1% and 1%). The parts
that sum to the gap are now four: together, ahead, T_GPU and the residual. The figure shows "one change alone" and then
"both", not a sequence.

**Fix.** Reword all three to match the 2x2 ("change what is cached and how it is read alone and together, then read
ahead").

### 7. Table 1, "The bound's read time is nearly reachable": "an overhead clause failed" undercounts

**Evidence.** Job 102's scorecard has 33 clauses: 28 held, all on point estimates, and 5 failed.
- P2below ("layer mode at least 0.03 below the analytic fraction") failed at 4 of its 6 host-cells (hosts b and c,
  both budgets).
- P3 ("layer fraction lower at C=32 than at 14"), also a fixed-cost clause, failed on host a.

The read-time thresholds P1 and P2floor held at 6 of 6.

**Fix.** "registered; read-time thresholds held at 6/6 host-budgets; two overhead clauses failed (5 of 33)".

### 8. Table 3's "Residual above Eq. (2)" uses a different T_GPU than Eq. (2)

**Evidence.** The residual is computed as both3p − Eq. (1) − median T_GPU (3.16 ms). Eq. (2) as defined in Section 3
uses the smallest profile (2.94 ms). Measured above Eq. (2) itself, the residual is 20% at 11% and 11% at 25%, against
the 18% and 8% reported. The symbol T_GPU also takes two values in adjacent sections: 2.9 ms in Section 3, median 3.2 ms
in Section 4.

**Fix.** Either relabel the row "above Eq. (1) + median T_GPU", or use 2.94 ms. Give the median a different name, such
as T̂_GPU.

### 9. Section 3's probe-sensitivity sentence: true, but needs a qualifier

**Where.** "on the machine at the top of the range the engine's reads imply a rate above the probe's best
(tab:robust)".

**Evidence.** The top of the range is 100f (54.0% of Eqs. 1 and 3; 70.3% of Eq. 2). By `robustness.py` its implied rate
is 1.16 times the probe's best, so the sentence holds. But:
- the implied rate comes from the closed-form account, which the paper calls exploratory and which failed its
  registered tests;
- `tab:robust` gives only counts, so a reader cannot find this machine there.

**Fix.** Write "by the closed-form account (1.16x)", as the Limitations bullet does.

### 10. The "MIN's set, read twice" arm does not achieve MIN's hits; the mechanism sentence is incomplete

**Evidence.**
- At 11% the bypass state misses 1.10–1.28 times as often as MIN read once (1.08–1.25 at 25%). For example, on 099e it
  has 50.8 misses per token against 39.75.
- It admits 15.8 per token against the deployed cache's 3.5.
- Its total reads are 0.92–1.06 times the deployed cache's at 11%.

The sentence "each extra admission costs a second read" is right, but the arm also loses hits: its admissions land too
late for some next uses.

**Fix.** One clause in Section 4: "and the read-twice arm misses 10–28% more than MIN read once".

### 11. Script bug in `fig_decomp.dep_bound` at a GPU-bound cell (no effect on the text)

**Evidence.** `dep_bound` takes R\*S/B_p as `lim·B/B_p`, which assumes Eq. (1) is host-bound. On 105b at gpt-oss 25%,
Eq. (1) is GPU-bound: 1.84 ms against a host term of 1.14 ms at B_host = 178 GB/s. As a result:
- `\dcShareDepMidMax` is 53 where it should be 45;
- `\dcDepOverLimMidMax` is 1.98 where it should be 1.68;
- `eq2 = tgpu + lim` is also wrong there.

These macros are not used in the text. The same machine contradicts two unchanged statements: Section 2's "The two
smallest budgets of each model are host-bound", and Section 3's "[the GPU term] binds only at the largest budgets". Both
fail on 105b at 25%.

**Fix.** Compute a and b from R\*·S directly, and scope the host-bound statement.

## What reproduces

**The 2x2 split.** It reproduces exactly from the raw rows. 15 machines, one launch per GPU: 100a, 100c and 101a are
skipped as repeat GPUs. 13 are fast (by either CPU-rate definition): the lowest fast ratio is 0.54, the highest slow one
0.40 (099d, 099f). Problem counts are 30 on 096a/b and 20 on the rest, the same in all five states. The patches are
oracle, oracle2 and oracle3 ("three versions" holds).

Mean share of the gap over the 13 fast-link machines, with bootstrap 95% intervals and the macro value:

| Part | gpt-oss 11% | Macro | gpt-oss 25% | Macro |
|---|---|---|---|---|
| MIN's set alone | 1.2 [−0.7, 3.1] | 1 [−1, 3] | 20.8 [19.1, 22.3] | 21 [19, 22] |
| One read alone | 0.9 [−0.2, 2.0] | 1 [0, 2] | −1.1 [−4.0, 1.4] | −1 [−4, 1] |
| Together | 38.8 [31.9, 45.4] | 39 [32, 45] | 35.3 [30.0, 40.3] | 35 [30, 40] |
| Ahead | 14.6 [10.6, 18.3] | 15 [11, 18] | 20.5 [18.1, 22.4] | 20 [18, 22] |
| T_GPU (median) | 28.9 [25.8, 31.7] | 29 [26, 32] | 35.9 [32.7, 38.7] | 36 [33, 39] |
| Residual | 17.6 [12.5, 23.1] | 18 [12, 23] | 8.4 [4.1, 12.9] | 8 [4, 13] |
| Best oracle | 53.8 [49.3, 58.1] | 54 [49, 58] | 55.7 [51.6, 59.8] | 56 [52, 60] |

The t-intervals agree with the bootstrap within about 1 point.

Other numbers that reproduce:
- Interaction: 37 at 11% and 16 at 25%.
- The four parts sum to the gap on every machine.
- T_GPU profiles: minimum 2.94, median 3.16, maximum 3.38 ms (2.9, 3.2, 3.4). Its band at 11% is 26.9–30.9 (27–31).
- Slow-link best oracle: at most 16.8% (17).
- MIN prefetched against Eq. (2): 1.07–1.38 at 11%.
- MIN's admissions per token are 15.8 against the deployed cache's 3.5 at 11%, and 8.8 against 4.3 at 25%, so "MIN's
  set admits more" and "at 25%, where admissions are fewer" hold.

**The verbal quantities, on the fast-link mean.**
- "Almost none alone": 10 of 13 machines have both one-change shares under 5%; the largest is 6%.
- "Most of what remains": T_GPU is 64% of the remainder on the mean and exceeds the residual on 11 of 13 machines. This
  holds only under the serial assumption; see defect 1.
- "About half": 54% and 56%.
- "Roughly a third to a half": 31–54%.

**Eq. (3) at 11% over the 25 consumer machines.** It reproduces exactly; my selection matches the script's 25 machines
one for one. The speed shares of the bounds are 31.2–54.0% for Eq. (1), 40.5–54.0% for Eq. (3) and 55.2–70.3% for
Eq. (2). The ratio of Eq. (3) to Eq. (1) has median 1.000, with 21 of 25 machines at exactly 1 by the clamp. Its maximum
is 1.557, on 105b (ratio 0.34). The four machines above 1 are exactly the four with a link-to-CPU ratio below 0.5. The
result is robust to using the median T_GPU (maximum 1.62) or the helper-interpolated B_c (1.57).

**Other changed text.**
- The probe's second-highest reading is a median 0.8% below the highest.
- `\audOursMed` 27 is the median of 12 cells (26.6). I did not recompute `\audInclassMed` 13.6 over 20 rows.
- Ours leads FreeToken at gpt-oss 11% on both hosts (1.29x and 1.21x), so the Conclusion's "the other systems we
  measured run further from it" holds at that budget.
- `\jkSlowBypassPlan` 1.11–1.22 spans gpt-oss 11% and 25% on the hosts with a ratio below 1/3 ("over the two budgets"
  is correct).

**Section 2.**
- The per-machine split sentence is correct: every job from 093 to 108 computes the fetch table from the probe with
  `fetch_table.py`, and the configurations carry it (`fetch=0,0,1,2,3` on 099a; `0,0,1,1,2` on 101b).
- The Yardsticks paragraph matches each section's measure.
- The probe ran while the downloads were in the background in every job script from 093 to 106 that I checked.

**Table 1, other rows.** "Foresight pays when MIN's set is read once — registered; held at most cells on three
fast-link machines" is consistent with jobs 095 and 096. The fetch-over-base clauses held, except job 096's
Qwen3-25% band on O4 at 1.513. The three factorial machines have link-to-CPU ratios of 0.82, 1.05 and 0.58.

## Build

- `latexmk -pdf -g` on a clean `git archive HEAD` copy succeeds: 43 pages, anonymous author block.
- No undefined references or citations, and no multiply defined labels.
- The main text has no overfull boxes. The only overfull boxes are four `\hbox` lines, 70.4 pt too wide, in
  `tab_audit.tex` (appendix, page 41).
- The main text ends on **page 9, right column, about two-thirds down** (Conclusion). References start right after it on
  page 9.
- Main-text floats are all on pages 2–8: Table 1 on page 2; Figure 1 and Table 2 on page 3; Table 3 (the gap) on page 4;
  Table 4 (headline) and Figure 2 on page 5; Figures 3 and 4 on page 7; Figure 5 on page 8.
- Minor: Table 3 is numbered before Table 4, although Table 4 is cited first (Section 3).
- Minor: pdfTeX reports duplicate hyperref destinations (`figure.11`, `table.28`–`table.30`) in the appendix, so those
  links may land on the wrong float.

# Number check 20 (8 October)

## Scope

Every number and claim in the parts of the paper changed since commit beb8f9f (`git diff beb8f9f HEAD -- paper/ scripts/`, HEAD = 932384e). That covers:

- **Main text:** the abstract; Table 1 (the rewritten 2x2 row and the new consumer-scope row, with every other row re-read); Contributions; Yardsticks; the R* definition; the audit paragraph; the 2x2 and timing-control paragraphs of Section 4; "What is left"; the closed-form paragraph; the registered tags of the paragraph heads; Section 6.2; Section 7's deployed-cache sentence; the builder rules; Related Work; Limitations; the Conclusion.
- **Appendices:** the new app_related.tex and app_limits.tex, compared against the text removed from paper.tex; app_prereg.tex (the 110e and job 112 disclosures and the new claims-to-clauses index, tab:claimindex); app_wsg.tex (the traces and cost items, and the appendix map).
- **Captions:** Table 2 (tab_configs), Table 3 (tab_headline) and Table 4 (tab_dm).
- **New macros:** cxResidOwn\*, cxTgpu\* and cxNotRelaunchedDev (job109.py); dcShareServ\* (fig_decomp.py); dmProbe\* (decomp_measured.py); cost\* (wsg_numbers2.py).
- **Layout and build:** float positions and a clean build of both documents.

I modified nothing in either repository except this file.

## Method

I read the generating scripts only to get definitions. All values below are recomputed by my own code from the raw files on the gpu branch: per-problem `ec_g_*.jsonl` rows, `st_g_*.json` counters, `concur.txt` and `concur2.txt` probes, `fetch_table_law_gptoss.json`, `prof_G*.json` and `prof_C*.json`, `bs1.jsonl`, the routing traces, `gpu/vast_ledger.json`, and the job headers in `jobs/*.sh`. I also read the scorecards in `prereg/scorecard_*.json`.

The scripts are in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/nc20/`:

| Script | What it recomputes |
|---|---|
| `rstar2.py` | MIN with bypass per layer, step semantics, independent of `mosl.cachesim` |
| `rglobal.py` | The same with one pooled cache across layers (heap based) |
| `dcshare.py` | Share of the bound's speed, consumer and server machines |
| `rounds.py` + `cx.py` | Jobs 109–112 loader; RTX 4090 residuals and 110e |
| `dm.py`, `dm2.py`, `eq3.py` | Table 4, the interaction, probe sensitivity, Eq. (3) shares |
| `cap.py`, `ci.py`, `nochange.py` | Capture, trend deviations and their bootstrap intervals, the no-change predictor |
| `tc.py` | The timing control (job 112) |
| `probe2.py` | Second probe and relaunch drift |
| `j100.py`, `jk.py` | Job 100 windows; Few-2R at slow links |
| `cost.py` | Ledger rentals and cost |
| `tgpu.py` | RTX 5090 T_GPU |
| `cites.py` | Citations and macros before and after the move |
| `macros.py` | Macro table |

**Rules applied:**
- **Eq. (1):** min over m of max((D + (Lk − m)S)/B_gpu, max(R\*, m)S/B_host), with B_host the highest probe reading by any method.
- **Machine sets:** as defined in the text and scripts (first launch per GPU UUID; unstable set 106c, 106d, 107c, 107e and 108b; consumer or server by CPU name).
- **Intervals:** t-intervals over machines.

**Build:** `git archive HEAD paper` into a new directory, then `latexmk -pdf` for paper.tex and supplement.tex. I built beb8f9f the same way for comparison.

## Results table

Verdicts: OK = reproduced; DEFECT = see the numbered list below; NOTE = minor or advisory.

### Build and layout (items 10, 11)

| Location | Claim | My value | Verdict |
|---|---|---|---|
| Build | latexmk exit codes | paper.tex 0, supplement.tex 0 | OK |
| Build | Lines starting with "! " | 0 and 0 | OK |
| Build | Undefined references or citations | 0 and 0 | OK |
| Build | Overfull boxes | paper 0 (beb8f9f: 1); supplement 31 (beb8f9f: 31, unchanged) | OK |
| Build | Other warnings | "Text page 2 contains only floats" is new (Table 1 fills the right column of p. 2). 38 pdfTeX duplicate-destination warnings (37 at beb8f9f). | NOTE |
| Build | Committed PDFs match source | pdftotext of the clean builds is identical to the committed paper.pdf and supplement.pdf | OK |
| Layout | Floats in the main text | Table 1 p2; Table 2 and Fig 1 p3; Table 3 and Fig 2 p6; Table 4 and Fig 3 p7; Table 5 p8; Table 6 and Fig 4 p9; Fig 5 p10 | OK |
| Layout | Main text ends by p10, floats before the references | Conclusion heading p10; References heading p11; paper 44 pp, supplement 55 pp | OK |

### Abstract

| Location | Claim | My value | Verdict |
|---|---|---|---|
| Abstract | 31–54% of the bound's speed on consumer machines at the smallest budget | 31.2–54.0% (25 machines, gpt-oss 11%) | OK; NOTE that the model is not named (11% < Qwen3's 12.5%, so the wording is accurate) |
| Abstract | "at that budget, caching the experts MIN would keep pays only if … read once" | True on fast links. On slow links at 11%: 105b (ratio 0.32) MIN-2R 1.044× vs MIN-1R 0.912×; Few-2R 1.11× at ratios 0.28–0.32 | DEFECT 2 |
| Abstract | Together close 35% of the gap in the registered test | 35 [22, 48] (job 109, 5 machines) | OK |
| Abstract | Four policies without foresight recover at most 6% where the link is ≥ half the CPU | Max capture 5.7% (job 109, 11%); 25% max 4%; fast RTX 4090s 3–5% | OK |
| Abstract | Half of foresight's value needs ≈0.65 C distinct experts (in host reads) | Median of 26 stored points over 9 models = 0.6506 (recomputed from stored per-point D/C, not from raw traces) | OK (partial) |

### Introduction and Table 1 (item 5)

| Location | Claim | My value | Verdict |
|---|---|---|---|
| Intro | 33% on 15 panel machines; 35% on 5 new | 33 [21, 44]; 35 [22, 48]; panel = 096a, 096b, 099a–j, 100b, 100f, 101b | OK |
| Table 1 r2 | Read time ≥ 50% per layer; 5 of 33 clauses failed; 3 machines | 33 clauses, 5 failed, 3 hosts. All six P2floor clauses held (point); the failures are P2below (4) and P3 (1). | OK; NOTE N1 |
| Table 1 r3 | Host S ratios outside ±0.06 at five of six configurations; faster at 11 of 12 | 089-P2 a, b, c, e, f failed; 11 of 12 ratios ≥ 1 (host S Qwen3 43.75%: 0.974) | OK |
| Table 1 r4 | 2x2 interaction: failed on the panel (true on 9 of 10), held on the 5 new | 099 P5a at 11%: 9 of 10 (099f −0.48 ms); 109 P1-int: 5 of 5; my interaction values match the scorecard | OK |
| Table 1 r5 | Timing: T1 failed, T2 failed, T3 held; 4 machines | Scorecard 112: T1 failed 5 of 8, T2 failed 7 of 8, T3 held 4 of 4 | OK |
| Table 1 r6 | Capture ≤ 35% per machine; at most 6%; 5 machines | 5.7% max | OK |
| Table 1 r7 | Ratios 0.37–0.77; within 0.10 in log; held (7%); misses fast RTX 5090s by up to 18%; 5 machines | 0.371–0.771; max \|dev\| 6.57% (112a both3p); 17.6% (109e); 5 machines | OK |
| Table 1 r8 | Fewest-admission schedule: pre-specified on 7, held; 13 machines | 13 and 7 match the definition, but the clause also failed on registered machine 106c (and 106d round 1) | DEFECT 4 |
| Table 1 r9 (new) | 31–54% on 25 consumer machines; server machines 12–46% | 31.2–54.0 (n = 25); server 12.4–46.1 (n = 3: 105a, 107d, 108a); 25% server 13.6–28.6 → 14–29 | OK |
| Table 1 r10–12 | Median 13.6%; 9 models; three of four tests failed | Unchanged macros. 9 models confirmed. rtN four, three failed, one inconclusive (tab_regtests.py) | OK; NOTE N2 |

### Sections 2 and 3

| Location | Claim | My value | Verdict |
|---|---|---|---|
| §3 R\* definition | R\* on the routing trace the engine runs | Recomputed R\* (step-semantics MIN with bypass, 30 problems = 7680 tokens): gpt-oss C14 38.3154, C32 15.3404; Qwen3 C16 96.5150. All equal prereg/speed_limit_v2.json. | OK |
| §3 read-time paragraph | 86–96% of the bound's read time | P2floor measured 0.861–0.963 | OK |
| §3 audit paragraph | Caveats rewritten | No numbers changed. "the comparison with our cache crosses hardware" dropped (partly kept as "on our own hardware") | NOTE |

### Section 4

| Location | Claim | My value | Verdict |
|---|---|---|---|
| §4 2x2 | MIN-2R closes 0%, Dep-1R 0% at 11%; 21% at 25% | 0 [−2, 3], 0 [−2, 2]; 21 [19, 22] | OK |
| §4 timing control | Four machines at 11%: Few-2R-early 0.98–1.00×, Few-2R 1.03×, Few-1R 1.21–1.41× | 0.977–0.997; 1.027–1.030; 1.208–1.406 | OK |
| §4 timing control | Misses 0.97–0.98 of Few-2R's (registered ≤ 0.95); fetches 17.8–23.7 misses per token | 0.968–0.980; 17.8–23.7 | OK |
| §4 | Interaction held on 9 of 10 (panel) and 5 of 5 (new) | Same | OK |
| §4 What is left | Ahead 15%, left 52% | 15 [11, 19], 52 [43, 62] | OK |
| §4 What is left | "at 25% they slightly overshoot" | Residual −5% (Table 4); −9% on fast panel machines (min −21%); −5% on new machines | NOTE |
| §4 What is left | Slow-link panel residual 35–47%; RTX 4090 31–36% | 35.5, 47.4; 31.5–35.8 | OK |
| §4 What is left | cxResidOwn 27–31% (25%: 16–19); cxTgpu 3.5 ms | t4 = min over 111d, f, g profiles: C14 3.5077 ms, C32 3.4661 ms; residual with own T_GPU 26.9–30.8% and 16.4–19.2% | OK |
| §4 What is left | Eq. (3) left: 83–99% (slow panel), 73–81% (RTX 4090s), median 45% (fast) | 83.1, 98.6; 72.7–80.6; 45.5. With the RTX 4090's own T_GPU: 72–80%. | OK; NOTE |
| §4 closed form | Of four registered tests, three failed | Fourth was inconclusive, not held | NOTE N2 |

### Section 5 (registered tags)

| Location | Claim | My value | Verdict |
|---|---|---|---|
| §5 head | "gain bands per budget on two machines; held at most budgets" | 096 P2 (1.15–1.45× at host-bound cells) held at 7 of 8 cells (O4 Qwen3 25%: 1.513) | OK |
| §5 head | "faster than the deployed cache; held" | 106c-P6 (all rounds), 106c-P6-launch and 106c-P7-speed failed (0.85); 106d-P6-r1 failed | DEFECT 4 |

### Section 6.2

| Location | Claim | My value | Verdict |
|---|---|---|---|
| §6.2 | Three at 0.37–0.40; two at 0.57–0.77 | 0.371, 0.392, 0.396; 0.572, 0.771 | OK |
| §6.2 | Within 7% at 11% and 5% at 25%; every 95% interval inside the band | 6.57%, 4.99%. Bootstrap ends ≤ 7.7% and ≤ 6.3% (band ±0.10 in log). | OK |
| §6.2 | Interaction positive on the two fast RTX 4090s | 112a 6.604 ms, 112b 2.978 ms | OK |
| §6.2 footnote | Loss 3% higher; first rounds within 6% | 0.1963/0.190 − 1 = 3.3%; max 6.23% (110b both3p) | OK |
| §6.2 | No change misses by up to 46% and 69% | 46.49% and 68.93% (112a both3p) | OK |
| §6.2 | Trend misses five of 18 cells, up to 18% | 5 of 18; 17.59% | OK |
| §6.2 | Few-2R 1.11–1.22× at ratios 0.28–0.32 | 104a, 105b, 105e: 1.112–1.114 at 11%, 1.159–1.217 at 25%; MIN-1R 0.86–0.98 | OK; DEFECT 11 (the range spans both budgets and the qualifier was dropped) |

### Section 7, builder rules, Related Work, Limitations, Conclusion

| Location | Claim | My value | Verdict |
|---|---|---|---|
| §7 | 16-token window in the step 0.89–1.32× on 4 machines; below 1 on the 2 lowest-ratio machines | 0.891 (0.41), 0.892 (0.54), 1.205 (0.96), 1.320 (0.92) | OK |
| Rule 1 | "the best system" reaches 31–54% | Only our cache was measured on the 25 machines | NOTE |
| Rule 2 | A better set read twice closes almost none of the gap | 11% only. At 25% MIN-2R closes 21% (panel) and 20% (new); bypass/base 1.10–1.21. Pre-existing text. | DEFECT 6 |
| Rule 4 | Within 7% over 0.37–0.77; missed fast RTX 5090s by up to 18% | As above | OK |
| Related Work | Citations | No citation lost: every citation of the old paper.tex is in the new paper.tex or app_related.tex; app_related.tex is verbatim | OK |
| Limitations | Server launches 8, three passed | 105a, 106c, 106d, 107c, 107d, 107e, 108a, 108b; valid 105a, 107d, 108a | OK |
| Limitations | Pooling lowers MIN's reads by 4.5–18.6% | My pooled MIN, gpt-oss C14 36.2099 and C32 13.9251, equals the JSON; range over 6 cells 4.46–18.57% | OK |
| Limitations | 12 of 84 launch-budgets | Unchanged macros, not recomputed | NOTE |
| Limitations | Second probe within 4.9% on 4; ratio moved by up to 12% | 0.81, 4.90, −0.44, −0.28%; 112g vs 109f −11.8% | OK |
| Limitations | "T_GPU is our engine's kernel time on an RTX 5090" | Smallest of 069c and 105 prof_C14/C32 = 2.9402 ms (105e), all RTX 5090 | OK |
| Limitations | "Both RTX 4090 tests deviated from their registrations" | No deviation listed for job 112's RTX 4090s | DEFECT 5 |
| Limitations | KL 0.0019 and 0.0005 | Unchanged macros, not recomputed | NOTE |
| Limitations, moved text | Every macro survives | All of jlGated, jmGated, jlInvalid, jmInvalid, rsDiffLow, jfMissDev\*, globalReads\*, rbAboveMax, tcProbe\* are present. "so where the crossovers lie rests on few points" was dropped (not moved). "Prompts and batches above one bypass the cache" survives in app_wsg §"Prompts and batches". | NOTE |
| Conclusion | Caching MIN's set pays only if each admitted expert is read once | Unscoped. False at 25% (MIN-2R 1.10–1.21× on all 15 panel machines; 109 P5 ≥ 1.08 held 4 of 4) and on slow links | DEFECT 1 |

### Appendices (items 3, 4, 6, 7, 9)

| Location | Claim | My value | Verdict |
|---|---|---|---|
| app_limits | 3.5 ms; 27–31% | Same | OK |
| app_limits | rsDiffLow −0.4% | R\* on 20 problems 38.1697 vs 38.3154 → −0.38% (C32 also −0.38%) | OK |
| app_prereg 110e | cxNotRelaunchedDev −5.6% | 110e first round: fetch −3.56%, both3p −5.59% → −5.6 | OK (value) |
| app_prereg 110e | "deviated most … so leaving it out favours the test" | 110b both3p +6.23% and 110c fetch −5.72% exceed it; inside the band | DEFECT 3 |
| app_prereg 110e | Relaunched machines' first rounds were in the repository when the relaunch was committed | 110a–d results in commit 443bcba (job 111 script); 110e results later (58d5454) | OK |
| app_prereg job 112 | Core i5-12400 machine allowed by name | Header V0 CARD=4090 exempts GPU-406c0b8f…; ledger offer 49588631 = 112a | OK; DEFECT 10 (antecedent) |
| tab:claimindex r1 | 102 P2: 102x-P2floor-C14/C32 | Exist, all held (point). P2 also had a "≥ 0.03 below the analytic split" part (102x-P2below), which failed on 102b and 102c | NOTE N1 |
| tab:claimindex r2 | 081 (B), 089 (S), hand-scored | 081-P2/P3b/P4a/P4b held; 089-P2a–f in scorecard_clauses.json | OK |
| tab:claimindex r3 | 099 interaction sign; 109x-P1-int-g11/g25 | 109 IDs exist (all held (point)). The 099 prediction (P5, part a) and its IDs (099Px-P5a-g11/g25) are not given | DEFECT 8 (minor) |
| tab:claimindex r4 | 112x-T1…T4 | Exist (T1 and T2 carry -g11/-g25 suffixes) | OK |
| tab:claimindex r5 | 109x-P7-best, 109x-P7-cap; 111 and 112 Q3 | Exist (with -g11/-g25; pooled 109-P7-mean); Q3 in 111 and 112 | OK |
| tab:claimindex r6 | 111x-Q1-\*, 112x-Q1-\* (Q6) | Exist; Q6 only in job 112 (112a-Q6, 112b-Q6) | OK |
| tab:claimindex r7 | 104 (a), 105, 106 (P6), 107 (P5): 106x-P6-plan-\*, 107x-P5-fetchplan | 104a's clause is P3, 105's is P4, 106e's is P7 (P6 covers only ratio < 0.4). The listed 106x-P6-plan-\* failed on 106c and 106d | DEFECT 8 |
| tab:claimindex r8 | "The second probe and the relaunches" | Not a claim in Table 1. The closed-form row's tests (jobs 105–108) are not indexed | DEFECT 8 (minor) |
| app_wsg traces item | Trace paths exist; speed_limit_v2.py reads them; prereg/speed_limit_v2.json holds the output; data/traces/ exists | All true (TRACES dict, RES = gpu branch results; both .npz files committed on the gpu branch) | OK |
| app_wsg cost | 145 rentals, 84 offers, jobs 058–112, $105.8 | 145; 84; 112; 105.78 | OK |
| app_wsg map | Map of the appendices | Omits app:related (K) and app:limits (L) | DEFECT 7 |
| Related/limits move | Nothing lost | See the Related Work and Limitations rows above | OK / NOTE |

### Table captions

| Location | Claim | My value | Verdict |
|---|---|---|---|
| Table 2 caption | Window policies are in app:value | app_value "The window policies" covers fill, drop and the deployed path | OK |
| Table 3 caption | "Speeds and ratios are from the second of the two launch orders run for each cell" | Ours and FreeToken = launch 2 at every cell except † (Qwen3 25% host B: one selection launch, job 080 L1). llama.cpp has one launch. Host B's two launches both ran ours before FreeToken. | DEFECT 9 |
| Table 4 caption | Probe sensitivity: 5 (9) for Both, 6 (11) for left at 11% | 5.09 (9.27), 5.77 (10.52), fast panel and new (n = 18); identical including slow machines. 22% = rbAboveMaxMax 1.22 (hard-coded) | OK |
| Table 4 cells | All 24 cells | All reproduced exactly from raw data (my t-intervals) | OK |

## DEFECTS (most severe first)

**1. The Conclusion's 2x2 claim has no scope, and data contradict it** (paper.tex l.527–528).

- **Text:** "In the engine, caching MIN's set pays only if each admitted expert is read once;"
- **What the data show:**
  - At gpt-oss 25%, MIN's set read twice (MIN-2R) gains 1.10–1.21× on all 15 panel machines. It closes 21% [19, 22] of the gap on the panel and 20% on the new machines, and job 109's registered P5 (bypass/base ≥ 1.08 at 25%) held on 4 of 4 (1.16–1.18).
  - On slow links at 11%, MIN's fewest-admission set read twice runs 1.11× on the three machines at ratios 0.28–0.32. On 105b, MIN-2R runs 1.044× while MIN-1R runs 0.912×.
- **Fix:** "In the engine, at the smallest gpt-oss budget and where the link is fast, caching MIN's set pays only if each admitted expert is read once;"

**2. The abstract's 2x2 claim lacks the fast-link scope that Table 1 carries** (paper.tex l.78–79).

- **Text:** "at that budget, caching the experts MIN would keep pays only if each admitted expert is also read once instead of twice"
- **What the data show:** Table 1 scopes this claim "(gpt-oss 11%, fast links)". At 11% on slow links it fails: 105b (Threadripper 9960X, ratio 0.32) MIN-2R 1.044× vs MIN-1R 0.912×. Few-2R runs 1.11× at ratios 0.28–0.32, and §6.2 itself says Few-2R pays there.
- **Fix:** "at that budget and where the link reads at least half as fast as the CPU, caching …" (or move the existing "Where the link reads …" clause ahead of both sentences).

**3. 110e's first round did not deviate most from the trend** (app_prereg.tex l.139–140).

- **Text:** "Its first round deviated most from the trend, by \cxNotRelaunchedDev\% [−5.6] at 11\%, so leaving it out favours the test;"
- **What the data show:** 110e's largest deviation is −5.6% (both3p; fetch −3.6%). The relaunched machines' first rounds deviated more: 110b both3p +6.2% and 110c fetch −5.7%. The footnote in §6.2 itself says they came "within 6%", which contradicts "most". 110e's deviation (log −0.058) is inside the registered 0.10 band and below the reported cxAllDevLowMax of 7%, so including it would not have changed the outcome. Only by mean \|dev\| over the two configurations (110e 4.6% vs 3.2, 2.9 and 1.9%) is 110e the largest.
- **Fix:** "Its first round came within 5.6\% of the trend at 11\% (the three relaunched machines' first rounds within 6.2\%), inside the registered band, so leaving it out does not change the test's outcome;" Or name the metric (mean over fetch and both3p) if "most" is kept.

**4. The fewest-admission tag "held" relies on machines set aside after the fact** (paper.tex l.369; also Table 1 l.150 and the tab:claimindex row at app_prereg.tex l.36).

- **Text:** "\paragraph{Admit as rarely as optimality allows (registered: faster than the deployed cache; held).}"; Table 1: "pre-specified on \rbPlanRegMachines{} [7] machines: $>$\,1, held".
- **What the data show:** Job 106 registered "fetchplan/base > 1 at gpt-oss 11% on every host" (P7) and, for ratio < 0.4, in every round (P6). It failed on 106c (EPYC 7302: 0.85, all rounds) and P6 failed in round 1 on 106d (0.996). Job 106 had no registered validity gate. Its stability rule was prediction 8, and tab_regtests.py calls 106c and 106d "set aside after the fact". The registration therefore covered 9 machines and held on the 7 that ran stably. The claims index points readers to 106x-P6-plan-\*, which are exactly the failing clauses. The body and tab:failures do disclose the losses.
- **Fix:** Tag: "(registered: faster than the deployed cache; held on the machines that ran stably, failed on the EPYC 7302 of job 106, set aside after the fact)". Table 1: "pre-specified on 9 machines: > 1; held on the 7 that ran stably (failed on 106c, wrong outputs)".

**5. Limitations overstates the deviations** (paper.tex l.506–507).

- **Text:** "Both RTX 4090 tests deviated from their registrations in ways \cref{app:prereg} lists."
- **What the data show:** Appendix D lists deviations only for the first RTX 4090 test (jobs 110/111). For job 112's RTX 4090s:
  - the Core i5-12400's earlier rental was allowed by name in the header (V0, CARD=4090);
  - both machines were the first two offers in the registered order;
  - on both, the first probe ran while the model downloaded, as the header states (probe1_when.txt: "download running: yes").
  The 109d replacement concerns the RTX 5090 relaunches.
- **Fix:** "The first RTX 4090 test, and the RTX 5090 relaunches of the last, deviated from their registrations in ways \cref{app:prereg} lists."

**6. Builder rule 2 lacks a budget scope** (paper.tex l.469–471; pre-existing text, reviewed under item 8).

- **Text:** "A better set read twice, or the same set read once, closes almost none of the gap; where the link reads at least half as fast as the CPU, a better set read once closes about a third of it."
- **What the data show:** True at gpt-oss 11% only. At 25% MIN-2R closes 21% (panel) and 20% (new) of the gap.
- **Fix:** Prefix "At the smallest budget, …".

**7. The appendix map omits the two new appendices** (app_wsg.tex l.1–15).

- **What the data show:** "Map of the appendices" lists A–J, M and N but not K (app:related, "Related work in full") or L (app:limits, "Limitations in detail"), which this commit inserted.
- **Fix:** Add "\item \Cref{app:related}: related work in full." and "\item \Cref{app:limits}: the remaining limitations."

**8. The claims-to-clauses index is inaccurate or incomplete in four rows** (app_prereg.tex l.24–38).

- (a) The row "The fewest-admission schedule gains: jobs 104 (machine a), 105, 106 (P6) and 107 (P5): 106x-P6-plan-\*, 107x-P5-fetchplan" is incomplete. 104a's clause is P3 (104a-P3-g11) and job 105's are P4 (105x-P4-g11). Job 106's "> 1 on every host" is P7 (106x-P7-speed), and 106e, one of the 7 registered machines, was registered only by P7. The listed 106x-P6-plan-\* failed on 106c and 106d (see DEFECT 4). **Fix:** "jobs 104 (104a, P3), 105 (P4), 106 (P6, P7) and 107 (P5): 104a-P3-g11, 105x-P4-g11, 106x-P6-plan-\*, 106x-P7-speed, 107x-P5-fetchplan (106c, 106d set aside)".
- (b) The row "MIN's set and one read pay only together" gives no prediction or identifier for job 099. **Fix:** "job 099, P5 (part a): 099Px-P5a-g11/g25".
- (c) "The second probe and the relaunches" is not a claim of Table 1, while the caption says the table indexes Table 1's claims. **Fix:** drop the row or reword the caption.
- (d) Table 1's closed-form row cites four pre-specified tests (jobs 105–108) that are not indexed. **Fix:** add a row.

**9. Table 3's launch-order sentence is not true for every cell** (tab_headline.tex l.3; also scripts/wsg_tables.py).

- **Text:** "Speeds and ratios are from the second of the two launch orders run for each cell."
- **What the data show:** True for ours and FreeToken at 11 of 12 cells (launch 2 in bs1.jsonl reproduces every printed value). Three exceptions:
  - the † cell (Qwen3 25%, host B) has one selection launch (job 080, L1);
  - llama.cpp has a single launch at every cell, so the "Ours ÷ llama.cpp" column mixes launch 2 with launch 1;
  - on host B both launches ran ours before FreeToken (grouped, then interleaved), so the second is not a FreeToken-first order as on host S.
- **Fix:** "Our cache's and FreeToken's speeds, and their ratio, are from the second of the two launches of each cell (except †); llama.cpp ran once."

**10. An inserted sentence breaks an antecedent** (app_prereg.tex l.149–150).

- **Text:** "… whether two of job 109's RTX 5090 machines reproduce on a second rental. Its Core i5-12400 RTX 4090 machine had run job 091's system grid, … allowed by name in the header. One of those two, job 109d's machine, …"
- **What is wrong:** "One of those two" now follows the i5-12400 sentence.
- **Fix:** Move the i5-12400 sentence after the 109d/109f sentence, or write "One of the two RTX 5090 machines, job 109d's, …".

**11. The Few-2R range lost its budget qualifier** (paper.tex l.410–412).

- **Text:** "\FewTwo{} pays instead of \MinOne{} (\jkSlowBypassPlanMin--\jkSlowBypassPlanMax$\times$ [1.11–1.22] on RTX 5090s at ratios 0.28--0.32)."
- **What the data show:** The range spans both budgets: 1.11 at 11% (1.112–1.114) and 1.16–1.22 at 25%. The beb8f9f text said "over the two budgets"; the paragraph around it is mostly at 11%.
- **Fix:** "(1.11–1.22× over the two budgets on RTX 5090s at ratios 0.28–0.32)".

## NOTES (no fix required unless the authors wish)

- **N1 (read-time tag).** "(registered: 50% or more per layer; 5 of 33 clauses failed)" and Table 1 r2: the 5 failures are not the 50% floor, which held on all six host-cells (86–96%). Four are P2's second part (registered: layer mode at least 0.03 below the analytic split; on 102b and 102c the layer mode came closer than predicted) and one is P3 (102a). The index also describes P2 as only "layer mode at least 0.50". Consider "(registered: 50% or more per layer, held; 5 other clauses of the job failed)".
- **N2 (closed-form account).** "of four registered tests of it, three failed" (§4, Table 1): the fourth (job 107) was inconclusive by its registered rule, not held. The beb8f9f text said so (rtInconclN, no longer used).
- **N3 (overshoot).** "at 25% they slightly overshoot": the residual is −5% in Table 4, −9% mean on fast panel machines (down to −21%) and −5% on the new machines.
- **N4 (Eq. (3) and T_GPU).** The Eq. (3) shares next to the new own-T_GPU residual (73–81% on the RTX 4090s) still use the RTX 5090's 2.94 ms. With 3.5 ms they are 72–80%.
- **N5 (dropped caveats).** Dropped from the main text without a move: "so where the crossovers lie rests on few points" (old Limitations) and "And the comparison with our cache crosses hardware" (old audit paragraph; "on our own hardware" remains). Also dropped (not caveats): cxFetchLow\*, tcFourFetchLowRng, cxSlowMeanDev\* (the slow-panel-mean predictor), tcEarlyMinusFetchMissLowRng, tcMissEarlyLowRng and tcRatioRng.
- **N6 (index caption).** "P, Q, T numbers in each job's header": the headers of jobs 099–109 number their predictions 1., 2., …; the P prefix is the scorer's.
- **N7 (rule 1).** "the best system reaches 31–54%": only our cache was measured on the 25 consumer machines.
- **N8 (abstract model).** "at the smallest budget we test" does not name the model. It is accurate because gpt-oss 11% is below Qwen3's 12.5%.
- **N9 (not recomputed).** These unchanged macros used in rewritten sentences were not recomputed: rbAboveMax/rbLaunchBudgets (12 of 84), klMeanGpt/klMeanQwen, audInclassMed 13.6, audOursMed. dcMed 0.65 was reproduced only from stored per-point values.
- **N10 (build warning).** New LaTeX warning "Text page 2 contains only floats": Table 1 occupies the whole right column of p2. The pdfTeX duplicate-destination warnings (38) predate this commit.

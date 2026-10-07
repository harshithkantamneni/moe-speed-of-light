# Number check 13 (7 October)

## Scope

This check covers the text that changed between 181e55c and 0205c68 in `paper/paper.tex`, `paper/app_more.tex` and `paper/app_wsg.tex`, plus everything that text cites:
- the abstract, the Introduction's "Where does the time go?" paragraph, its "What is knowing the future worth?" sentence on the margin rule, and the contributions;
- the Setting's "desktop-class" definition and the Statistics paragraph;
- in Section 4, "Registered on new machines, it failed.", "What the failures share.", the two renamed paragraphs and the Fig. 1 caption;
- in Section 5, the job 107 sentences;
- the Limitations items "Hardware and sample" and "The relation", and the Conclusion;
- in the appendices, `app_more`'s "Job 106" edit and its "Job 107, every host" paragraph, `tab_job107.tex`, and `app_wsg`'s job 107 paragraph;
- all macros in `wsg_job107.tex` and `wsg_decomp.tex`, and `figs/decomp.pdf` (I looked at the rendered `decomp.png`).

I recomputed everything from the raw files on the GPU branch. I read the authors' scripts only for definitions; none of my code imports them.

- **Job 107 (hosts a–g).** Sources: `v0.txt`, `gate.txt`, `cpu.txt`, `numa.txt`, `cores.txt`, `validity.txt`, `concur.txt`, `g_prof.json`/`q_prof.json` (cross-checked against the `prof_*.json` medians), `ec_*_r*.jsonl` and `st_*.json`. From these:
  - time per token, as the mean over problems of `decode_ms/n_decode`;
  - speed ratios paired by problem within each round, with intervals that resample rounds and then problems;
  - B_c, B_p and B_host parsed from `concur.txt`;
  - Eq. (3) solved in closed form (the larger root of T² − (G + t_M + t_A)T + t_A·G = 0), and also its plain, half-admission and misses-only forms;
  - the dk ratio predicted from each configuration's counters;
  - loss per token and round spread.
- **Machine classes.** Every launch of jobs 093–107 that ran the deployed cache at gpt-oss 11% (the first round where there are rounds): the implied read rate over B_host, the class (one NUMA node and at most 32 usable cores), and the relation's error. I repeated this at gpt-oss 25% and on Qwen3, and for job 069c's older server rentals.
- **Fig. 1.** One launch per GPU UUID, the four invalid hosts left out, with the median G and MIN's exact reads from `speed_limit_v2.json`.
- **The dk aggregate.** Hosts 106a/b/e plus 107b/d, at gpt-oss.
- **The registration.** `git show` of 1a0616a, 9f01f98, f9ac8c7, ddec94f and 346bd34. I compared commit times with `vast_ledger.json` and the manifests' `start_utc`, and checked the GPU UUIDs against `known_gpu_uuids.txt` and against every earlier `nvidia-smi-q.txt`.
- **Build.** I built `paper.tex` and `supplement.tex` with latexmk in a scratch copy.

My scripts are in `/tmp/claude-0/-home-claude-moe-speed-of-light/68c27986-37d1-524e-96c8-48dc5afd384e/scratchpad/nc13/`: `common.py`, `j107.py`, `classes.py`, `classes25.py`, `decomp.py`, `dkall.py`, `nllcheck.py`, `bsens.py` and `old_servers.py`.

Every macro value in `wsg_job107.tex` and `wsg_decomp.tex`, and every cell of `tab_job107.tex`, reproduces (see the last section). The defects below are about scope, about validity, about how the registration is described, and about wording.

## Defects (most severe first)

### 1. Major: the relation is claimed for "desktop-class machines", but that class was drawn after the test, and the one new desktop tested in advance failed at 3 of its 4 cells

**Text.**
- Abstract: "On 24 rented RTX 5090 machines with one socket and at most 32 cores, our llama.cpp expert cache takes the GPU's profiled compute plus its host reads at the machine's best rate, with nothing fitted".
- Contributions: "failed when registered on new machines, which marks out where it holds".
- Section 4: "The relation describes desktop-class machines, where it was found, and even there a new machine missed by up to 11%."
- Section 4 paragraph title: "On desktops it reads at the machine's best rate."
- Limitations: "It assumes the engine reads at the machine's probed rate, which held on desktop-class machines and on no server. … the desktop-class boundary that separates its hits from its misses was drawn after that test".
- Conclusion: "On desktop-class machines our cache executes its reads at the machine's best rate".

**Evidence** (`j107.py`, `classes.py`).
- The 24 desktop-class machines in the abstract include 107b, the Ryzen 9 5900XT. The next sentence of the abstract reports the relation failing on that same machine.
  - On 107b, Eq. (3) under-predicted by 7.1%, 2.3%, 11.3% and 6.4% at gpt-oss 11%, gpt-oss 25%, Qwen3 12.5% and Qwen3 25%, so only 1 of 4 cells is within 6%.
  - Its implied read rate was 0.92, 0.97, 0.87 and 0.92 of B_host. At Qwen3 12.5% that is below the 0.88 floor that the paper gives for the class.
- 107b is the only desktop-class machine that was tested out of sample on a machine never rented before.
  - The class statistics ("within 6% on 32 of them") mix that machine with the 34 launches on which the relation was found and the three re-rented machines of job 106.
  - Six desktop-class launches, on 5 of the 24 machines, miss by more than 6% at gpt-oss 11%: 099d −7.5%, 099i −10.6%, 100a −6.2%, 100f +12.6%, 103e −7.0% and 107b −7.3%.
- The boundary therefore does not "separate its hits from its misses". The registered test on the new desktop failed for the same reason as on the server, under-prediction, only smaller.
- On 107b the plain form fits better than the registered form. The plain form puts 2 of 4 cells within 6% (+0.3%, −5.5%); the overlap form puts 1 of 4 within 6%. Their median |errors| are 6.0% and 6.75%.

**Fix.**
- Abstract: "On the 24 machines with one NUMA node and at most 32 usable cores on which it was found and re-run, the relation is within 6% on 32 of 38 launches at gpt-oss 11%. The one such machine new to the registered test missed at 3 of 4 cells, by up to 11%."
- Section 4: replace "The relation describes desktop-class machines" with "The relation was found on desktop-class machines and fits most of their launches; the one new desktop it was registered on missed at 3 of 4 cells".
- Contributions: drop "which marks out where it holds".
- Limitations: replace "separates its hits from its misses" with "separates its small misses from its large ones".
- Conclusion: "reads at close to the machine's best rate (0.88–1.16 of it at gpt-oss 11%)".

### 2. Major: "every server we rented read at only 0.27–0.77 of its best rate" rests on four launches that failed validity checks, one of which computed wrong outputs

**Text.**
- Abstract: "every server we rented read at only 0.27–0.77 of its best rate".
- Introduction: "Across all our launches, … and every server at 0.27–0.77".
- Section 4: "and 0.27–0.77 on all 5 launches on servers. Both of job 106's failures were servers."
- `app_more`: "We report their clauses … and draw nothing from them about the relation on those runs".

**Evidence** (`classes.py`, gpt-oss 11%, first round).

| Launch | CPU | Implied rate / B_host | Valid? |
|---|---|---|---|
| 107d | AMD engineering sample, 2 sockets | 0.27 | yes |
| 107c | EPYC 9754 | 0.58 | no: rounds 3.5% apart |
| 106c | EPYC 7302, 2 sockets | 0.62 | no: wrong outputs, NLL 0.208 |
| 107e | EPYC 7663 | 0.74 | no: rounds 6.0% apart |
| 106d | Xeon Platinum 8347C | 0.77 | no: rounds 39% apart |

- Every value of the range except 0.27 comes from a launch that failed a validity check. One of them is 106c, whose outputs the paper says are wrong and from which it says it draws nothing. The text does not say that 4 of the 5 failed. (`jlServValid` = 1 is computed but not used.)
- **The range holds at gpt-oss 11% only.** At Qwen3 25%, 106d read at 0.85 and 106c at 0.74. The abstract and Introduction give no cell.
- **"Every server we rented" is wider than the analysis.**
  - Job 069c rented an EPYC 9655 server (46 usable cores). With the older engine it read at about 0.49, which fits the claim but is not in it.
  - The same job rented an EPYC 7352 (24 cores, one NUMA node, so desktop-class by the rule), which read at about 0.83. That is below the stated 0.88–1.16 for desktop-class machines "across all our launches".
  - Different engine and text, so these are a matter of scope wording only.
- **The one valid server's "best rate" rests on one reading.**
  - On 107d, B_host = 204.5 GB/s comes from `concurrent t=128 pcie=zerocopy64 cpu_gbs 181.4 … sum 204.5`.
  - The probe's CPU-only rate at 128 threads was 117.6 GB/s, and its best CPU-only reading was 151.4 GB/s (16 threads).
  - With 151.4 GB/s as the rate, the error at gpt-oss 11% is −51% rather than −59%, and the implied rate is 0.36 rather than 0.27.
  - On 107c, B_host (145.8) is a CPU-plus-link sum; against its best CPU-only reading (101.7) the implied rate is 0.84 and the error −12%.
  - The failure on 107d is robust, but the quoted 52–59% and 0.27 depend on that one reading.
- **The appendix already gives a cause.** `app_wsg` (the blind-test paragraph) says: "Each failure has a measured cause … on a 12-channel EPYC host memory stops being the limit and a per-layer latency floor takes over". Section 4 now says the cause on servers is "not established". The two passages should refer to each other.

**Fix.**
- Abstract: "the five server launches of jobs 093–107, four of which failed a validity check, read at 0.27–0.77 of the probe's best rate at gpt-oss 11%; the one valid server read at 0.27".
- Introduction: replace "Across all our launches" with "Across the 43 launches of jobs 093–107 at gpt-oss 11%".
- `app_more`: either leave 106c out of the class analysis or change "draw nothing from them".
- Give 107d's sensitivity to its B_host reading.
- Cross-reference the EPYC 9655 latency-floor finding in `app_wsg`.

### 3. Moderate: the registration's own rules for too few valid hosts and for replacements are described inaccurately

**Text.**
- `app_more`: "The registered replacement list ran out after the first gate failure and two hosts that failed the round check; the two hosts added after that were chosen by a rule written into the job script before each started (cheapest verified offer not rented before, with fast downloads and cheap traffic)".
- Section 4 title and Abstract: "Registered on new machines, it failed."

**Evidence** (header of `jobs/107_newhosts@vast.sh` at 1a0616a, and the amendments).
- **What was registered.** The header said "A host that fails V0 stops and is replaced by the next offer in the list", and "A host that passes V0 but fails V1 or V2 is reported … and left out … The test needs 3 valid hosts; with fewer it is inconclusive."
- **What happened.**
  - Replacing the hosts that failed V2 (107c, 107e) was not in the registration. It was added in f9ac8c7.
  - The list had not run out: 54519248 was no longer offered, but 51600687 (Xeon E5-2699 v3) was skipped because it "would exceed the remaining credit".
- **The rules for f and g differed.**
  - Host f: download ≥ 1000 Mb/s, and its first pick could not be rented (ddec94f).
  - Host g: download ≥ 900 Mb/s, and machines already rented in this job were excluded (346bd34).
- **Timing is clean.** Every amendment was committed before the host it adds was rented, and every amendment changed comment lines only. In UTC:

| Host | Commit adding it | Rented (ledger) |
|---|---|---|
| a–d | 1a0616a, 05:17:31 | 05:17:44–47 |
| e | 9f01f98, 05:24:30 | 05:24:36 |
| f | f9ac8c7 06:36:28 and ddec94f 06:36:57 | 06:37:00 |
| g | 346bd34, 06:41:16 | 06:41:20 |

- **"Failed" is still logically right, but the paper does not say why.**
  - Prediction 2's per-cell band failed on both valid hosts.
  - A third host could not have brought the pooled median |error| under 4%: with 7 of the 8 known cells above 6%, the median of 12 cells is at least (6.4 + 7.1)/2 = 6.75%.

**Fix.**
- `app_more`: "Gate failures were to be replaced from the registered list. Replacing hosts that failed the round check was added during the job (f9ac8c7): one listed offer was gone and the other over the remaining credit, so hosts f and g were chosen by rules committed before each started (download ≥ 1000 and ≥ 900 Mb/s)."
- Section 4: "the registration called a test with fewer than three valid hosts inconclusive; the relation's per-cell clause failed on both valid hosts, and no third host could have brought the pooled median under 4%".

### 4. Moderate: "No other form fits either" quotes medians that sit between two very different hosts and hide that the plain form fits the desktop better

**Text (Section 4).** "No other form fits either: the plain form, half the admissions or misses alone leave median errors of 30.4, 31.9 and 33.4%."

**Evidence.**

The medians reproduce (30.4, 31.9, 33.4; within 6%: 2, 1, 0 of 8 cells). But each is the midpoint between one host near 6–13% and one host near 50–60%:

| Form | Ryzen 9 5900XT, median \|error\| | Within 6% | Server, median \|error\| |
|---|---|---|---|
| Overlap (registered) | 6.75% | 1 of 4 | 57.3% |
| Plain | 6.0% | 2 of 4 | 56.3% |
| Half admissions | 8.2% | 1 of 4 | 57.3% |
| Misses only | 10.35% | 0 of 4 | 58.3% |

The registered clause that the overlap form beats the plain form failed, but only the appendix says so.

**Fix.** Report per host. For example: "on the desktop the plain form came closest (median 6.0%, 2 of 4 cells within 6%; the registered overlap form 6.75%, 1 of 4, so its registered advantage over the plain form failed); on the server every form missed by 50–60%."

### 5. Moderate: the main text keeps the job 106 claim about predicting the margin rule's gain and omits job 107's failed clauses

**Text (Section 5).**
- "The gain is small, as Eq. (3) says it must be".
- "Predicted from each configuration's own counters, Eq. (3) gives these ratios within a median 0.013 (at most 0.045 …)".

**Evidence** (`j107.py`).
- On the server 107d, dk gained 1.049 at gpt-oss 11% and 1.074 at Qwen3 25%, where Eq. (3) predicted 1.008 and 1.0075.
- It lost at gpt-oss 25% (0.987) where Eq. (3) predicted a gain of 1.005.
- The registered tolerance of 0.03 failed at two cells (|errors| 0.041 and 0.067).
- Over the 8 job 107 cells the prediction's median error is 0.016, against 0.021 for predicting no change.
- Only `app_more` and `app_wsg` report this.

**Fix.** After the job 107 sentence, add: "on the server the relation's prediction of these ratios missed the registered 0.03 at two cells (by 0.041 and 0.067), and the gain was larger than it predicts".

### 6. Minor: "whose outputs and rounds matched every other host's"

**Evidence** (`nllcheck.py`).
- 107d's outputs match. Its per-problem base NLL differs from 106a's by at most 0.0045, against 0.0032–0.0047 for the other valid hosts, with means of 0.1899 and 0.0855–0.0856.
- Its rounds were 0.28% apart. That does not match 107c (3.5%), 107e (6.0%) or 106d (39%).

**Fix.** "every valid host's".

### 7. Minor: the abstract's class definition differs from the Setting's

**Text.**
- Abstract: "with one socket and at most 32 cores".
- Setting: "one NUMA node and at most 32 usable cores".

**Evidence.**
- The class rule in `job107.py` and `fig_decomp.py` tests NUMA nodes and usable physical cores.
- `099a/cpu.txt` reports "Socket(s): 24" (a 285K in a VM).
- 106c has 32 physical cores but 15 usable.
- The abstract and Introduction write the threshold as `\jlDeskCoresMax`, which is the largest core count found among the desktop launches (32), not the rule's threshold. The two coincide only by chance.
- The class also includes an EPYC 7402P (105a), so "desktop" is a label rather than a description.

**Fix.**
- Use "one NUMA node and at most 32 usable cores" everywhere.
- Define the threshold in one macro that both the scripts and the text use.
- Note in the Setting that the class includes a single-socket EPYC.

### 8. Minor: Fig. 1's hatched "reading below the machine's best rate" appears on most desktop machines, and the text attributes it to the server only

**Evidence** (`decomp.py`, and the rendered `decomp.png`).
- At gpt-oss 11%, 18 of the 24 desktop-class rows have a hatched segment, measured above Eq. (3). The largest are the i7-14700K (10.6% of its time), the 5900XT (7.7%), the i9-13900KF (7.5%) and the TR 3970X (7.0%).
- The text says "On the dual-socket server, reading below the machine's rate is a third part", and the paragraph title says desktops read "at the machine's best rate".
- "The error follows the class of machine (Fig. 1)": Fig. 1 holds one server, so the class evidence is in the text, not the figure.

**Fix.**
- Caption: "hatched … (also up to 11% of the time on some desktop-class machines)".
- Cite the class numbers, not Fig. 1, for "follows the class".

### 9. Minor: the Statistics paragraph says every configuration ran in rounds

**Text.** "in jobs 106 and 107 each configuration also ran in two or three rounds".

**Evidence.** The Qwen3 cells of both jobs ran one round (`runs.txt`). `tab_job107`'s caption says so correctly.

**Fix.** "each gpt-oss configuration".

### 10. Minor: the gate is described as more than it measured

**Text.**
- Section 4: "host memory not in use by other tenants … loss per token within 2% of every earlier host's".
- Limitations: "3 had other tenants' memory in use".

**Evidence.**
- V0 allowed up to 48 GB "used" (`free -g`). 107e passed with 39 GB in use and 107c with 13 GB.
- Whose memory it was is not measured.
- V1 compares against fixed references (0.190 and 0.0855). Jobs 093–097 used another text and show 0.207, so "every earlier host's" is not literally the gate.

**Fix.**
- "at most 48 GB of host memory in use".
- "loss within 2% of the reference (0.190 gpt-oss, 0.0855 Qwen3)".
- Limitations: "3 had 50–96 GB of host memory in use".

### 11. Minor: wording

- **"missed the 6% band at 7: by 2–11% on a Ryzen 9 5900XT".** The 2.3% cell is within the band. Say "under-predicted by 2–11% (missing the band at 3 of 4 cells)".
- **Abstract and Conclusion, "gains up to 6%".** They omit the one loss (0.987 on the server at gpt-oss 25%) that the Introduction reports. The pooled set also mixes job 106's re-rented hosts with job 107's new ones; the Introduction says "over 5 machines" but not which jobs.
- **`app_more`, "as on every earlier host: the GPU's own work does not depend on the CPU".** At 11%, G is 4.05–4.45 ms across hosts, about 10% apart. Say "varies little across hosts".
- **`app_more`, "The EPYC 9754 and the EPYC 7663 ran … 3.5% and 6.0% apart".** These are `jlInvalidSpreadMin`/`Max` matched to host names by position. That is correct today but breaks silently if the order changes. Emit one macro per host.

### 12. Minor: hand-typed counts in the changed text

Each of these should be a macro. The macros marked as existing are already in `wsg_job107.tex`.

| Where | Hand-typed | Macro |
|---|---|---|
| Abstract | "three of them" | `\jiStable` (exists) |
| Introduction | "three of those machines" | `\jiStable` (exists) |
| Introduction | "the two of 7" | `\jlValidWord` (exists) |
| Introduction | "16-core desktop" | from `cores.txt` |
| Introduction | "one cell lost" | `\jlDkAllLoss` (exists) |
| Section 5 | "below the registered 0.995 at one cell" | count of P3 failures |
| Limitations | "Only one server passed every check" | `\jlServValid` (exists) |
| `app_more` | "the four machines" | `\jlRan` (exists) |
| `app_more` | "Three more machines" | `\jlGated` (exists) |
| `app_more` | "two hosts that failed the round check" | `\jlInvalid` (exists) |
| `app_more` | "the two hosts added after that" | needs one |
| `app_more` | "On the two valid machines" | `\jlValidWord` (exists) |
| `app_more` | "job 107's one valid server" | `\jlServValid` (exists) |
| `app_wsg` | "at one cell" | count of failed clauses |
| `app_wsg` | "at two cells on the server" | count of failed clauses |

The registered constants (2%, 6%, 4%, 0.995, 0.4, "three it required") and the class threshold "32 usable cores" (see defect 7) are typed. That is acceptable if they are treated as constants of the registration.

### 13. Minor: the class script reads the wrong profile key for job 105

`job107.machine_classes` reads `g_prof.json["G14"]`. Job 105's files use the key `"C14"`, so 105a, b, e and f fall back to the median G (4.28 ms) instead of their own profiles (4.30, 4.14, 4.07 and 4.40 ms). I ran both versions. No macro changes: desktop 0.88–1.16, median 0.98, 32 of 38 within 6%. Fix the lookup anyway.

## What reproduces

**Gates and validity** (`v0.txt`, `validity.txt`).
- Host memory in use: a 50 GB, f 50 GB and g 96 GB, so all three failed V0; b 4, c 13, d 8 and e 39 GB.
- None of the 7 GPU UUIDs appears in `known_gpu_uuids.txt` or in any earlier `nvidia-smi-q.txt`, and that list covers all 48 earlier UUIDs.
- None of the 7 offers appears in the ledger before job 107.
- Loss per token in every round: b 0.1903 and 0.0854–0.0856; c 0.1899; d 0.1899 and 0.0855–0.0856; e 0.1894. All are within 0.32% of the references.
- Base round spread at gpt-oss 11%: b 0.30%, c 3.54%, d 0.28%, e 5.98%.

**Rates.**

| Host | B_c | B_p | Link/CPU | B_host |
|---|---|---|---|---|
| b | 34.8 | 26.2 | 0.75 | 36.1 |
| c | 101.0 | 51.7 | 0.51 | 145.8 |
| d | 118.2 | 51.6 | 0.44 | 204.5 |
| e | 126.1 | 25.9 | 0.21 | 126.7 |

G from the files equals G recomputed from the `prof_*.json` medians on every host and cell.

**`tab_job107.tex`.** Every G, law value, measured time, dk, in-the-step and by-the-CPU cell reproduces, intervals included.

**`wsg_job107.tex`.**
- Relation:
  - `jlCells` 8, `jlUnder` 8, `jlWithin` 1, `jlFailCells` 7;
  - `jlErrMed` 31.7 and `jlErrMin`/`Max` −58.9/−2.3;
  - `jlPlainMed` 30.4, `jlHalfMed` 31.9, `jlMissMed` 33.4, with 2, 1 and 0 within 6%;
  - `jlNewDeskErr` 2–11 and `jlNewServErr` 52–59;
  - the invalid hosts at 11%: `jlInvErr` 21–31.
- Admission margin:
  - `jlDk` 0.987–1.074, median 1.021;
  - `jlDkPredMed` 0.016 and `jlDkPredMax` 0.067 (on the server), against `jlDkNoChangeMed` 0.021.
- Fewest-admission set: `jlFetchPlan` 1.16–1.42 and `jlBypassPlan` 1.04–1.13. Copying in the step beat loading by the CPU on both valid hosts (1.419 > 1.038; 1.157 > 1.126).
- Hosts and gates:
  - `jlRatio` 0.44–0.75;
  - `jlG` 4.1–4.5 and `jlGQ` 5.6–5.9;
  - `jlInvalidSpread` 3.5–6.0 and `jlUsedGate` 50–96;
  - `jlLaunched` 7, `jlGated` 3, `jlInvalid` 2, `jlValid` 2.
- Scorecard: `jlClauses` 46, `jlHeld` 11, `jlPoint` 22, `jlFailed` 13. I counted the clauses from the header by hand, and the 13 failures are the ones `app_wsg` lists.

**Machine classes** (`jl*Desk*`, `jl*Serv*`).
- Desktop-class: 38 launches on 24 GPUs, implied rate 0.88–1.16 (median 0.98), errors −11 to +13%, 32 within 6%.
- Servers: 5 launches on 5 GPUs, 0.27–0.77 (median 0.62), errors −59 to −17%, none within 6%.
- Both of job 106's failures are servers (106c has 2 NUMA nodes; 106d has 36 usable cores), and only one server (107d) passed every check.
- Jobs 093–104 are all desktop-class.

**The dk aggregate** (`jlDkAll*`). 10 gpt-oss cells on 5 machines; median 1.019; largest gain 5.7% (Pf at 25%); one cell below 1, at −1.3% (107d at 25%).

**Fig. 1 and `wsg_decomp.tex`.**
- 25 machines at each budget: 24 desktop-class plus 107d. 107c and 107e are left out as invalid.
- Desktop-class shares at 11%: G 12–47%, MIN 31–54%, beyond MIN 22–35%, serialisation 23–68% of the gap.
- At 25%: 22–66%, 23–36%, 10–41% and 33–90%.
- Median law error −1% and +3%.
- 107d's measured time beyond Eq. (3): 58% of its time at 11% and 50% at 25%. The caption's counts (24 and 1) are right.

**Section 5 claims.** "No valid new machine had a slower link" is right: the valid ratios are 0.44 and 0.75. Copying in the step won on both valid hosts, as registered for ratios of 0.40 and above.

**Build.**
- `paper.tex` (37 pages) and `supplement.tex` (40 pages) build cleanly.
- There are no undefined references, citations or macros, and no `[pend.]`.
- The only overfull boxes are in `tab_audit.tex`.
- The Conclusion starts on page 9 and its last six lines run onto the top of page 10, where the References begin. The main text now ends on page 10, against page 9 in the Number check 12 build.

# Job 105 outcome: the sum law, the layer-ahead copy, and the fewest-admission set on new machines

Script: `jobs/105_sumlaw@vast.sh` on the gpu branch (commit 92ff9fd, amended a8e2f92 before 105f and 105g started; the
predictions were not changed). Scored by `scripts/job105.py` into `prereg/scorecard_105.json` and `prereg/job105.json`.

## Hosts

| Job | Machine | Link / CPU | B_host (GB/s) | Status |
|---|---|---|---|---|
| 105a | EPYC 7402P | 0.45 | 60.4 | ran |
| 105b | Threadripper 9960X | 0.32 | 178.2 | ran |
| 105c | Ryzen 7 7800X3D | — | — | image never pulled; destroyed without results |
| 105d | Threadripper PRO 7000 | — | — | card read 1072 GB/s; the gate stopped it |
| 105e | Pf again (Core Ultra 9 285K) | 0.28 | 96.1 | ran |
| 105f | O4 again (Ryzen 9 9950X) | 1.04 | 51.1 | ran |
| 105g | Threadripper PRO 7000, second | — | — | card read 1329 GB/s; the gate stopped it |

## Score

31 clauses: 7 held with an interval, 20 held on the point estimate, 4 failed, none untested.

## Prediction by prediction

1. **G_prof is 4.0–5.0 ms on every host at both budgets.** Held on all 8 cells, 4.07–4.50 ms. The profiled
   non-expert compute is 2.9–3.2 ms, lower than job 069c's 3.3–3.4 ms on the older engine. The demand bound uses the
   smallest profiled value, so it moves down (more conservative).
2. **The deployed cache's time is within 6% of G_prof + R·S/B_host at every cell; median error ≤ 4%.**
   - At 11% it held on all four hosts: errors −1.5% to +4.7%, median 1.7%.
   - At 25% it failed on three of four hosts: the law over-predicts by 3.8–9.3%, median 7.0%. That takes the pooled
     median (4.2%) past 4%.
   - The cause was visible in the exploratory data: at 25% the implied G was 3.85 ms against about 4.4 profiled.
     Background admissions overlap the GPU's work, and at 25% they are 13–22% of the counted reads against 5–11% at
     11%. The law has no term for that overlap. We registered the same 6% band for both budgets and it failed at the
     larger one; the paper says so in Section 4, Limitations and Appendix B.
3. **Layer-ahead copy (LLAMA_EC_PREFETCH=1): ≥ 1.03 at 11% where the ratio is ≥ 0.8; < 1 where it is < 0.4.** Held
   both ways:
   - O4 (ratio 1.04): 1.042 [1.034, 1.050];
   - Threadripper 9960X (0.32): 0.909;
   - Pf (0.28): 0.722.

   The EPYC 7402P (0.45, no prediction) lost too (0.81). At 25% the copy is 1.01 on O4 and 0.69–0.87 elsewhere. 84%
   of its copies at 11% are used by the next layer (73% at 25%).

   On O4 it gains a third of what job 066 measured on a 9950X (+14%), against today's deployed cache, which reads
   at the machine's best rate.
4. **The fewest-admission set copied in the step beats the deployed cache at 11% on every host, and beats the greedy
   set by ≥ 0.03 where the ratio is < 0.7.** Held:
   - against the deployed cache: 1.03–1.34, each interval above 1;
   - against the greedy set: +0.13 (9960X), +0.14 (7402P), +0.17 (Pf).

   On O4, where the link is as fast as the CPU, it is 0.01 behind the greedy set (no prediction). On the second
   slow-link machine (the 9960X) the greedy set loses (0.91) and the fewest-admission set gains (1.04), as on Pf. Loaded
   by the CPU, the fewest-admission set gains on every host at both budgets (1.02–1.22).
5. **Its in-step copies are ≤ 0.65× the greedy set's at 11%.** Held: 0.60 on every host. At 25% it is 0.73, not
   predicted.

## What changes in the paper

- **Section 4:** the law's test on new launches, and the fewest-admission set over jobs 104 and 105 (7 launches,
  6 machines, two slow-link machines).
- **Section 5:** the layer-ahead copy's cross-machine result.
- **Elsewhere:**
  - the job 105 table (`paper/tab_job105.tex`) and the host figure;
  - Limitations, where the law over-predicts at 25%;
  - Appendix B and the supplement's clause table.

# Job 106 outcome: the law with its overlap term, online admission rules, slow links launch by launch

Script: `jobs/106_onlineadmit@vast.sh` on the gpu branch (commit 525a9c9; amended in 89c5c3b before 106e started, to replace
its host; the predictions were not changed). Scored by `scripts/job106.py` into `prereg/scorecard_106.json` and
`prereg/job106.json`. Engine patch oracle5 (oracle4 plus the learned order allowed on the deployed path, with a margin).

## Hosts

| Job | Machine | Link / CPU | B_host (GB/s) | Deployed time, spread between rounds at gpt-oss 11% |
|---|---|---|---|---|
| 106a | Threadripper 9960X (105b's again) | 0.32 | 176 | 0.1% |
| 106b | Pf again (Core Ultra 9 285K, x8 link) | 0.29 | 94 | 0.3% |
| 106c | EPYC 7302, 13 GB/s link | 0.14 | 108 | **11%** |
| 106d | Xeon Platinum 8347C | 0.23 | 113 | **39%** |
| 106e | Ryzen 9 9950X | 1.02 | 52 | 0.2% |

- **106c and 106d were not stable.** Each failed the registered at-most-2% spread between rounds (prediction 8).
  - On 106d the deployed cache ran 14.5, 22.1 and 21.9 ms per token in its three rounds. Only 79 of the node's 257 GB
    were free, so other tenants held two-thirds of its memory.
  - On both hosts the probe's CPU read rate falls when more threads than physical cores are used. 106c's container
    showed 64 CPUs for 16 cores.
- **How they are treated.** Their clauses are scored as they came out: 36 clauses, 20 held and 16 failed. The paper's
  summaries leave both hosts out by the registered 2% threshold and say so. The leftover summary is 3 stable hosts.

## Score

181 clauses: 21 held with an interval, 112 held on the point estimate, 48 failed.

| Host | Held | Held (point) | Failed |
|---|---|---|---|
| a | 4 | 30 | 3 |
| b | 7 | 28 | 2 |
| c | 3 | 16 | 18 |
| d | 2 | 17 | 18 |
| e | 5 | 19 | 6 |
| pooled | 0 | 2 | 1 |

## Prediction by prediction

1. **Profiled GPU compute.** Held on all 20 cells.
   - gpt-oss: 4.1–4.7 ms.
   - Qwen3: 5.3–6.3 ms.
2. **The law with its overlap term within 6% at every host, model and budget.**
   - **Stable hosts:** held on all 12 cells. Errors were −3.3% to +6.0%, median |error| 2.3%, and the largest
     (9960X, Qwen3 25%) was just inside the band at 5.99%.
   - **Plain law on the same cells:** −4.1% to +10.1%. At gpt-oss 25% the overlap form's median |error| is 0.36 of the
     plain law's (registered: at most half), so that clause held.
   - **Unstable hosts:** the law under-predicts by 10–52%; the engine ran slower than the probe's best rate allows.
     Those 8 failures carry the pooled median |error| to 4.7%, so the pooled clause failed.
3. **Admitting less (dk: decayed count with kappa 3 at 11% and 2 at 25%, chosen on other text).**
   - **Reads:** fewer than base at every gpt-oss cell (held): 4–6% fewer at 11%, 6–10% at 25% on the stable hosts.
   - **Speed at gpt-oss:** registered at ≥ 1.01.
     - Held on Pf (1.024 and 1.057) and on 106c.
     - Failed on the 9960X at 11% (1.003) and on the 9950X (1.010 and 1.008).
   - **Pooled median:** registered ≥ 1.02, and it held at 1.04 over all hosts. That pass rests on the unstable hosts
     (106d at 1.28); on the three stable hosts the median is 1.01.
   - **Qwen3:** dk/base ≥ 1.00 held everywhere except 106d at 25%.
4. **The learned order with a margin (lrn).**
   - **Reads:** at most dk's at every gpt-oss cell (held).
   - **Speed:** lrn/base ≥ 1.00 held on Pf, the 9950X and 106c at 25%. It failed on the 9960X (0.98 and 0.99), where
     the host's memory is fast and the learned order's host time outweighs its saving, and on 106c at 11%.
   - **Host time:** the registered caps (400 µs on gpt-oss, 700 µs on Qwen3) failed on 106c, 106d and the 9950X. Its
     host time on the stable hosts is 270–440 µs on gpt-oss and 590–1030 µs on Qwen3 (up to 690 and 1230 µs on the
     unstable ones), and it tracks the CPU.
5. **The law's prediction of dk/base and lrn/base from each configuration's own counters, within 0.03.**
   - On the stable hosts it held except on Pf at 25%, where the errors were 0.030 and 0.045; there the law
     under-predicts the gain. Median error on the stable hosts: 0.013.
   - It failed on the unstable hosts.
6. **Slow links (ratio < 0.4) round by round.**
   - **9960X:** greedy MIN in the step lost in every round (0.93); the fewest-admission set gained in every round
     (1.05–1.06), launch-level interval [1.048, 1.064].
   - **Pf:** greedy 0.85; fewest-admission 1.018–1.022 in every round, launch-level interval [1.013, 1.026].
   - **106c (13 GB/s link, unstable):** greedy 0.65–0.71, and the fewest-admission set **lost** in every round
     (0.82–0.88).
     - Copying its 11.8 experts per token in the step over a 13 GB/s link takes about as long as the deployed cache's
       whole step.
     - Unlike the host's run-to-run spread, the loss is consistent across rounds.
     - Below some link rate even the fewest-admission set must not be copied in the step.
   - **106d:** inconclusive.
7. **fetchplan/base > 1 at gpt-oss 11%.** Held on a, b, d and e (9950X: 1.34); failed on 106c. Its copies are 0.60 of
   the greedy set's everywhere (held).
8. **Deployed cache within 2% between rounds.** Held on a, b and e (0.0–0.9%); failed on c (11%) and d (39%).

## What changes in the paper

- **The law (eq. sum) gains its overlap term.**
  - It is registered and holds within 6% on every stable host, at both models and both budgets.
  - The exploratory form and the job 105 test become history.
  - The unstable hosts go in Limitations: the law assumes the machine delivers its probed rate during the run.
- **Admitting less is realisable but small:** 0–6% here (median 1.01 on stable hosts). The law predicts each rule's
  effect from its counters (median error 0.013).
- **The learned order** helps only where host memory is slow relative to its host-side cost.
- **The fewest-admission set** keeps its lead on slow-but-not-too-slow links, with launch-level intervals. On a 13 GB/s
  link it loses too.

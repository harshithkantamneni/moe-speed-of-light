# Where the seconds go, order-independent: Shapley attribution of the gap to the speed limit

Replaces the fixed-order reading of `prereg/gap_listingb.json` (Fig. gap) for the six Table 1 cells on the headline
machine (RTX 5090 + Ryzen 9 9950X3D, jobs 080/081). Script: `scripts/shapley_gap.py`; numbers in
`prereg/shapley_gap.json` (variants `main` = job 084 optimum at B_host 87.5, `exact_max` / `exact_median` /
`exact_median_zc` = the exact optimum of `speed_limit_v2.json` at 87.5 / 80.7 / 77.7 GB/s, `global_*` likewise with
the global pool, and `bhost_median`, `gpu_const`, `gpu_gross`, `split6` on the 084 optimum); figure
`scripts/fig_shapley.py` → `paper/figs/shapley.pdf` (same style and size as `figs/gap.pdf`), drawn from `exact_max`.
No existing file was changed. The first tables below are the 084 optimum (`main`); the exact optimum follows under
"Exact optimum" and changes nothing qualitatively.

**Model.** The quantities of `gap_listingb.py` define a value function v(S) = predicted ms per token with a subset S
of five fixes applied: **overlap** (the limit's form, min over c of max(GPU(c), host reads), the CPU free to run
resident experts to balance the two; not applied: GPU time + host time in sequence with the policy's own CPU/copy
split), **foresight** (the optimum's reads instead of an online policy's), **policy and read paths** (the best online
policy's reads at the limit's single rate B_host; not applied: the engine's own reads through the host-memory law
with the probe's B_c, B_p, B_cp), **host work** (the engine's per-token host timer), **GPU efficiency** (datasheet
rate; not applied: the GPU's effective batch-1 rate, B_gpu/η, with η calibrated per cell so that no fix gives the
measured time; η is net of the overlap the engine achieves, exactly as the paper's "GPU below datasheet (net)" step
is). Checks: v(none) = measured, v(all) = the speed limit, and breaking the fixes in the paper's order reproduces
every step of `gap_listingb.json` to floating-point precision. The Shapley value of a fix is its marginal saving
averaged over the 5! = 120 orders; the closed-form and the permutation sums agree to 1e-14 ms and sum to the gap.

**Shapley values, ms per token (share of the gap), B_host = 87.5 GB/s (the limit's).**

| Cell | Limit binds on | Limit | Gap | Overlap | Foresight | Policy+paths | Host work | GPU (net) | Largest | Foresight largest in |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss 11% | host | 5.91 | 8.40 | 2.42 (29%) | **3.57 (43%)** | 1.33 (16%) | 0.46 (5%) | 0.61 (7%) | foresight | 115/120 orders |
| gpt-oss 25% | host | 2.34 | 6.82 | 2.36 (35%) | **2.50 (37%)** | 0.87 (13%) | 0.49 (7%) | 0.60 (9%) | foresight (by 0.14) | 65/120 |
| gpt-oss 40% | GPU | 1.93 | 4.63 | **1.83 (39%)** | 1.07 (23%) | 0.55 (12%) | 0.45 (10%) | 0.73 (16%) | overlap | 55/120 |
| Qwen3 12.5% | host | 10.83 | 14.20 | 3.85 (27%) | **7.27 (51%)** | 1.85 (13%) | 0.42 (3%) | 0.80 (6%) | foresight | 120/120 |
| Qwen3 25% | host | 4.78 | 11.06 | 3.78 (34%) | **4.92 (44%)** | 1.41 (13%) | 0.45 (4%) | 0.49 (4%) | foresight | 115/120 |
| Qwen3 43.75% | GPU | 3.25 | 5.99 | **2.74 (46%)** | 1.35 (23%) | 0.76 (13%) | 0.40 (7%) | 0.73 (12%) | overlap | 40/120 |

Host work is additive in every state, so its Shapley value is its fixed-order value. Shares of the measured token:
overlap 15–30%, foresight 15–31%, policy and paths 7–10%, host work 2–7%, GPU (net) 3–11%.

**Fixed order (paper) against Shapley, foresight and GPU; marginal range over the 120 orders.**

| Cell | Foresight: fixed | Shapley | marginal min–max | costed under overlap | Overlap: fixed | Shapley | GPU (net): fixed | Shapley | marginal min–max |
|---|---|---|---|---|---|---|---|---|---|
| gpt-oss 11% | 2.94 | 3.57 | 2.85–4.28 | 3.09 | 1.89 | 2.42 | 1.18 | 0.61 | 0.00–1.29 |
| gpt-oss 25% | 1.97 | 2.50 | 1.48–3.27 | 1.94 | 1.99 | 2.36 | 0.90 | 0.60 | 0.00–0.98 |
| gpt-oss 40% | 1.15 | 1.07 | 0.00–1.98 | 0.24 | 1.11 | 1.83 | 0.83 | 0.73 | 0.00–0.87 |
| Qwen3 12.5% | 6.64 | 7.27 | 6.40–7.85 | 7.10 | 3.20 | 3.85 | 1.59 | 0.80 | 0.00–1.71 |
| Qwen3 25% | 4.16 | 4.92 | 4.11–5.90 | 4.30 | 3.34 | 3.78 | 0.97 | 0.49 | 0.00–1.03 |
| Qwen3 43.75% | 1.63 | 1.35 | 0.00–2.72 | 0.00 | 1.63 | 2.74 | 0.81 | 0.73 | 0.00–0.84 |

"Costed under overlap" is foresight's marginal with every other fix applied (the online limit minus the optimum's
limit): 0.24 ms at gpt-oss 40% and 0 at Qwen3 43.75%, the reviewers' numbers. The GPU shortfall costed first from the
limit end is 0 at gpt-oss 11%, Qwen3 12.5% and Qwen3 25% (a slow GPU stays hidden under host reads), 0.46 ms at
gpt-oss 25% (host-bound by only 0.32 ms) and 0.78 / 0.76 ms in the two GPU-bound cells; costed last, from the
measured end, it is the paper's step (0.81–1.59 ms), near its largest marginal. Costing foresight before overlap, as
the paper does, gives it close to its smallest marginal in the host-bound cells (under-credited by 0.5–0.8 ms) and
more than its Shapley value in the GPU-bound cells (over-credited by 0.1–0.3 ms).

**Regime map.** The limit binds on host reads of the optimum (c* = 0) at gpt-oss 11% and 25% and Qwen3 12.5% and
25%; on the GPU's datasheet time, balanced by c* = 12.7 (gpt-oss 40%) and 30.1 (Qwen3 43.75%) experts run on the
CPU. In the host-bound cells the largest Shapley component is foresight (37–51% of the gap); in the GPU-bound cells
it is overlap (39–46%), with foresight second (23%) and GPU (net) third (12–16%). GPU efficiency is never the
largest, under any of the variants below. At Qwen3 43.75% the best online policy's reads (28.8/token) already fit
under the GPU term (c* = 30.1), so foresight is worth nothing there once overlap is perfect, and at gpt-oss 40% only
0.24 ms; that is why the fixed order, which costs foresight before overlap, made it look largest in those cells.

**Safe claim.** Foresight is the largest cost in every one of the 120 orders only at Qwen3 12.5% (its smallest
marginal, 6.40 ms, beats every other fix's largest marginal). At gpt-oss 11% and Qwen3 25% it is largest in 115 of
120 orders; it loses only in the five orders that cost overlap after both policy and foresight while the GPU is still
at its measured rate, where overlap hides the whole slow GPU (3.18 / 4.37 ms) under the optimum's reads. At gpt-oss
25% foresight and overlap are within 0.14 ms (37% vs 35%), which no tie rule should separate. In every order of every
cell the largest cost is foresight or overlap; policy, host work and GPU efficiency are never the largest.

Suggested wording (numbers of the exact optimum, variant (a) below; the 084 optimum gives 37–51%): "Averaged over
every order of the fixes (Shapley), missing foresight is the largest cost in the four cells where the limit binds on
host reads, 37–53% of the gap, and the second largest, after serialised host reads, in the two where it binds on the
GPU (23%); GPU kernels below datasheet rate are 4–16%, never the largest." The paper's "largest or tied-largest step
in all 6 cells" and the 0.05 ms tie rule of `wsg_tables.py` should go.

**B_host = the median concurrent probe rate (77.7 GB/s, the B_cp the engine's law sees) instead of the maximum
(87.5).** The limit rises in the host-bound cells (5.91 → 6.66, 2.34 → 2.64, 10.83 → 12.19, 4.78 → 5.39 ms) and
barely in the GPU-bound ones; the policy-and-paths component falls to 0.39 / 0.48 / 0.41 / 0.03 / 0.54 / 0.57 ms
(5 / 7 / 9 / 0 / 5 / 10% of the gap, from 12–16%); the rest moves into the limit and into foresight (now 41–60% in
the host-bound cells). Foresight is then largest in every order at gpt-oss 11%, Qwen3 12.5% and Qwen3 25%, in 90 of
120 at gpt-oss 25%; overlap stays largest in the two GPU-bound cells. So most of the paper's policy step is indeed
the 87.5 vs 77.7 GB/s difference: with the two fixes split (6 fixes, 720 orders, `split6`), read paths take 0.31–1.81
ms and the policy's read count −0.01 to 0.48 ms (at Qwen3 12.5% the engine reads fewer experts than the simulated
best online policy, 165 vs 166, and the best online policy's CPU/copy split sent through the engine's own paths is
slower than the engine's, so the policy fix can be negative before the paths are fixed).

**Exact optimum (`speed_limit_v2.json`).** The job 084 optimum (`scripts/foresight._pol`) reads 0.5–3.9% more than
the exact per-layer MIN with bypass (`mosl.cachesim`); the exact optimum's CPU/copy split (bypassed misses / admissions)
is recomputed from the routing traces, its totals and every limit checked against `speed_limit_v2.json`. The model's
checks hold as before (v(none) = measured; v(all) = that file's limit for `exact`, `median_bhost`,
`median_bhost_zerocopy`, `global`).

(a) Exact R* at B_host = 87.5 GB/s (`exact_max`, the figure):

| Cell | Limit binds on | Limit | Gap | Overlap | Foresight | Policy+paths | Host work | GPU (net) | Largest | Foresight largest in | Foresight marginal min–max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gpt-oss 11% | host | 5.80 | 8.51 | 2.43 (29%) | **3.68 (43%)** | 1.33 (16%) | 0.46 (5%) | 0.61 (7%) | foresight | 115/120 | 2.94–4.40 |
| gpt-oss 25% | host | 2.32 | 6.84 | 2.36 (35%) | **2.51 (37%)** | 0.87 (13%) | 0.49 (7%) | 0.61 (9%) | foresight (by 0.14) | 70/120 | 1.48–3.29 |
| gpt-oss 40% | GPU | 1.93 | 4.63 | **1.83 (39%)** | 1.07 (23%) | 0.55 (12%) | 0.45 (10%) | 0.73 (16%) | overlap | 55/120 | 0.00–1.97 |
| Qwen3 12.5% | host | 10.41 | 14.62 | 3.86 (26%) | **7.70 (53%)** | 1.82 (12%) | 0.42 (3%) | 0.81 (6%) | foresight | 120/120 | 6.78–8.33 |
| Qwen3 25% | host | 4.68 | 11.16 | 3.80 (34%) | **4.99 (45%)** | 1.43 (13%) | 0.45 (4%) | 0.49 (4%) | foresight | 115/120 | 4.21–6.02 |
| Qwen3 43.75% | GPU | 3.25 | 5.99 | **2.74 (46%)** | 1.35 (23%) | 0.77 (13%) | 0.40 (7%) | 0.73 (12%) | overlap | 55/120 | 0.00–2.72 |

Against `main`: the limit falls by 0.11 / 0.02 / 0 / 0.42 / 0.10 / 0 ms and the whole difference lands on foresight
(+0.11 / +0.01 / 0 / +0.43 / +0.07 / 0 ms); every other component moves by under 0.03 ms. Foresight costed under
overlap: 3.20 / 1.96 / 0.24 / 7.52 / 4.40 / 0.00 ms. The regime map is unchanged (c* = 0 in the four host-bound
cells, 12.7 and 30.1 in the GPU-bound ones) and so is the safe claim: foresight largest in every order only at Qwen3
12.5%, in 115/120 at gpt-oss 11% and Qwen3 25%, a 0.14 ms tie with overlap at gpt-oss 25%, overlap largest in the two
GPU-bound cells; in every order of every cell the largest is foresight or overlap.

(b) Exact R* at the median of the six concurrent probe sums, 80.7 GB/s (`exact_median`): the limit rises to 6.29 /
2.52 / 1.93 / 11.29 / 5.07 / 3.26 ms; policy and paths falls to 0.70 / 0.61 / 0.46 / 0.63 / 0.85 / 0.64 ms (9 / 9 /
10 / 5 / 8 / 11% of the gap); foresight is 3.82 / 2.60 / 1.13 / 8.02 / 5.17 / 1.40 ms (48 / 39 / 24 / 58 / 48 / 24%),
marginal ranges 3.21–4.40, 1.83–3.29, 0.00–1.97, 7.42–8.33, 4.58–6.02, 0.00–2.72; overlap 2.43 / 2.38 / 1.87 / 3.86 /
3.80 / 2.81. Foresight is largest in every order at gpt-oss 11%, Qwen3 12.5% and Qwen3 25%, in 90/120 at gpt-oss 25%;
the regime map and the largest component per cell are the same as in (a). At 77.7 GB/s (`exact_median_zc`) policy
and paths is 0.03–0.58 ms. The global pool (`global_max`: L×C slots shared, 4–19% fewer reads) lowers the host-bound
limits further (5.48 / 2.11 / 9.95 / 4.28 ms) and raises foresight to 4.01 / 2.65 / 1.12 / 8.18 / 5.32 / 1.49 ms
(45 / 38 / 24 / 54 / 46 / 25%), largest in every order at gpt-oss 11%, Qwen3 12.5% and Qwen3 25%; nothing else
changes. Which to draw: (a) keeps the paper's definition of the limit (eq. 2, the highest measured host rate, a
bound) and the "% of limit" of Table 1; (b) is the sensitivity to report in the text, where the 87.5 vs 80.7 GB/s
difference is named as what most of the policy-and-paths term is.

**Robustness of the GPU term.** With the shortfall as a constant per token instead of a rate (`gpu_const`) every
Shapley value moves by at most 0.05 ms. With a gross shortfall (`gpu_gross`: the batch-1 rate of ours with every
expert resident on the RTX PRO 6000, job 084b, host work removed, 57% / 65% of datasheet on gpt-oss / Qwen3, same
datasheet bandwidth as the 5090; the overlap the engine achieves by that accounting, up to 0.9 ms, is credited to
every non-overlapped state) GPU efficiency rises to 1.28 ms (28%) at gpt-oss 40% and 1.56 ms (26%) at Qwen3 43.75%,
ahead of foresight there (1.01, 1.29) but still behind overlap (1.39, 2.04); the four host-bound cells keep
foresight as the largest (35–51%; GPU 6–15%). That variant does not reproduce the paper's overlap and GPU steps
(they shift by the credited overlap); it is the accounting a reviewer who takes the all-in-VRAM rate at face value
would use.

**Reading.** The order matters because the fixes interact through the max of the limit: once host reads overlap
GPU work, fewer reads only help where the host term binds, and a slower GPU only costs where the GPU term binds.
Shapley resolves it: in the four host-bound cells the answer of the paper stands and is stronger than the fixed
order gave it (foresight 2.5–7.3 ms, 37–51% of the gap); in the two GPU-bound cells the largest cost is running host
reads and GPU work in sequence (39–46%), and foresight is worth 0–0.24 ms once that is fixed. GPU efficiency is a
4–16% item under the paper's net accounting and at most a second-place item under the gross one.

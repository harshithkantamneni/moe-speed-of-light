# Per-layer model test: does a sum of per-layer maxima fix Eq. 1's failures? — outcome

Reviewers proposed that Eq. 1's failures (FETCH over-predicted by 12% on the Core Ultra 7 270K and the EPYC 7352,
prefetch under-predicted, the 12-channel EPYC 9655 over-predicted by 19–32%, and a fitted G of 4.82 ms above the
3.92 ms all-in-VRAM token time) are a Jensen gap: a sum over layers of per-layer maxima exceeds the maximum of the
per-token sums. This note fits the per-layer model and its relatives on the same 33 blind measurements as Table 2 and
scores them leave-one-host-out. Script: `scripts/perlayer_model.py`; numbers in `prereg/perlayer_model.json`.

**Data.** The 33 measurements of Table 2 (jobs 069c ×3, 072, 073, 073a, 074, 076; 29 configurations, 7 hosts; the
9950X3D ran twice), reassembled from each job's `law_prediction.json`, `ref_ec.jsonl` and the engine's stats files.
Every row has the 36 per-layer histograms of misses per step (`miss_hist`) and the FETCH table, which give the per-layer
distribution of (c CPU experts, f copied experts); a prefetch copy (0 or 1, at the rate `prefetches/steps` of the
target layer) is charged to the window of the layer that issues it. The histogram totals reproduce the engine's
per-token counters exactly and the frozen READS table to 0.1%. Rates are the hosts' probes, never fitted.

**Models** (T per token; s_x = S/B_x; N_c = layer-steps per token with CPU work; free constants ≥ 0, fitted by least
absolute deviation of ln(predicted/measured)):
- A: Eq. 1, `G + max(X_c s_c, X_p s_p, (X_c+X_p) s_cp)`; B: additive, `G + X_c s_c + X_p s_p`;
- C: per-layer max in the form of Eq. 3, `G0 + Σ_l E[max((f+p) s_p + g, [c>0](h + c s_c), [c>0][f+p>0](c+f+p) s_cp)]`,
  with g free and h = 0; C+h with h free; C(g=37)+h with g frozen at the profile's 37 µs; Cu with the shared term on
  every layer that reads host memory (Eq. 1 per layer; differs from C only on the 9655, where B_cp < B_c);
- D: per-layer additive, which reduces to `G + h N_c + X_c s_c + X_p s_p` (g merges into G);
- E: C(g=37)+h with asymmetric DRAM sharing: when a layer reads through both paths, each runs at the probe's
  concurrent rate (`cpu_gbs`, `pcie_gbs` of concur.txt) until the first finishes, then the other alone;
- A×k: Eq. 1 with a fitted CPU-rate factor k (`k X_c s_c`); C×k+h: C(g=37) with k and h.

**Result, leave-one-host-out** (constants refitted on the other six hosts; error = predicted/measured − 1, %):

| Model | Free | Median | p90 | Max | FETCH 270K / 7352 | FETCH Ryzen (7900, X3D ×2, 9950X) | Prefetch (4 rows) | EPYC 9655 C14/C32/C56 |
|---|---|---|---|---|---|---|---|---|
| A, frozen G = 4.82 (the paper) | 0 | 3.3 | 11.6 | 31.9 | +11.7 / +11.4 | +4.4, +4.2, +4.5, +4.1 | −1.4 … −8.0 | +18.7 / +31.9 / +25.8 |
| A, G refit | 1 | **3.1** | 9.4 | 28.6 | +8.8 / +8.5 | +2.0, −0.2, +0.1, −0.0 | −3.4 … −8.7 | +16.1 / +28.6 / +22.3 |
| B, additive | 1 | 4.5 | 11.3 | 41.6 | −2.0 / −4.7 | −2.1, −11.3, −11.1, −11.1 | −6.1 … −9.3 | +31.1 / +41.6 / +32.4 |
| C, per-layer max | 2 | 3.2 | 10.2 | 36.1 | +7.9 / +10.6 | +1.2, −2.9, −2.6, −4.6 | −2.6 … −8.4 | +26.6 / +36.1 / +26.9 |
| Cu, shared term always | 2 | 3.2 | 10.2 | 29.8 | +7.9 / +10.6 | +1.2, −2.9, −2.6, −4.6 | −2.6 … −8.4 | +17.1 / +29.8 / +23.6 |
| C+h | 3 | 4.2 | 10.8 | 35.6 | +8.9 / +11.3 | +1.9, +2.6, +2.9, −2.6 | −1.9 … −8.3 | +25.6 / +35.6 / +26.9 |
| C(g=37)+h | 2 | 3.8 | 10.8 | 34.9 | +8.9 / +11.2 | +4.1, +1.9, +2.2, −1.9 | −2.8 … −9.0 | +26.2 / +34.9 / +25.0 |
| D, per-layer additive | 2 | 5.3 | 8.4 | 35.6 | +3.5 / −1.5 | +2.9, −6.2, −6.0, −6.5 | +1.4 … −6.9 | +19.2 / +35.6 / +34.4 |
| E, asymmetric sharing | 2 | 4.2 | **6.3** | 35.6 | **+3.1 / +4.1** | +2.9, −2.1, −1.8, −3.0 | −2.4 … −6.1 | +21.4 / +35.6 / +31.6 |
| A×k | 2 | 3.2 | 9.9 | **23.4** | +7.8 / +8.5 | +1.4, −3.7, −3.4, −0.4 | −4.2 … −9.9 | +12.4 / +23.4 / +16.7 |
| C×k+h | 3 | 3.5 | 10.4 | 47.5 | +8.9 / +10.3 | +5.4, +1.7, +2.0, −1.6 | −3.4 … −8.7 | +40.9 / +47.5 / +33.1 |

In-sample (all 33 rows, reference): A 2.8 / 24.6, B 3.3 / 35.9, C 2.9 / 35.6, C+h 3.0 / 31.2, D 4.0 / 34.0, E 3.1 / 28.3,
C×k+h 3.5 / 11.5 (median / max %). The frozen row reproduces Table 2 to 0.2 points (the jobs predicted from the
frozen READS table; here each host's own counts are used).

**Constants** (all rows; range over the seven folds): A: G 5.17 ms (4.97–5.28). B: G 4.93 (4.69–5.13). C: G0 4.94
(4.91–5.17), g 0 (0–0). C+h: G0 4.80, g 1 µs, h 15 µs (h 8–47). C(g=37)+h: G0 3.71 (3.50–3.71), h 41 µs (34–65).
D: G 3.64 (3.36–4.11), h 77 µs (39–98). E: G0 3.04 (3.00–3.23), h 79 µs (63–84). A×k: G 5.29, k 0.97 (0.95–0.98).
C×k+h: G0 3.77, k 0.87, h 94 µs (folds: k 0.86–1.08, h 9–98). All-in-VRAM: 3.92 ms.

**Reading.**
1. **No per-layer form beats Eq. 1 out of sample.** A, C, Cu, A×k and the frozen law sit within 0.2 points of median
   (3.1–3.3%): a tie within the noise of 33 rows. The per-layer max (C) trails on p90 and max. B (4.5%) and D (5.3%)
   are worse.
2. **The Jensen gap is too small to be the explanation.** With the same rates and no constants, the per-layer sum of
   maxima exceeds the token maximum by +0.31 ms of the 1.54 ms the frozen law leaves unexplained on the 270K's FETCH
   row, and by 0.00 of 1.86 ms on the 7352, where the copy is the largest term in every layer. On the 9655's C14/C32/C56
   it has the wrong sign (−0.5 to −0.1 ms against residuals of +1.3 to +1.9 ms). Under C the slow-link FETCH rows stay
   at +7.9/+10.6% and the 9655 at +27–36%.
3. **The additive model fails where the max is tested.** Its in-sample median is close (3.3 vs 2.8%), but it
   under-predicts the FETCH rows on the three Ryzen hosts with 53 GB/s links by 11% (the CPU and the copy do overlap
   there; A gets them within ±2% once G is refit). The reviewers' "fits as well" holds only for the 19 rows that read
   through one path.
4. **What the FETCH failure is: asymmetric DRAM sharing.** The mailbox counters (helper wall time per layer-step) show
   a single CPU miss beside a copy taking 20–62% longer than alone (270K 299 vs 244 µs, 7352 411 vs 342, 7900 537 vs
   331, 9950X3D 316 vs 217, 9950X 347 vs 242, 9655 110 vs 84), and the probe's concurrent split gives the link only
   12–24 GB/s while the CPU reads on the 270K, 7352, 7900 and 7950X (33–36 on the two 9950X hosts, whose FETCH rows
   Eq. 1 already gets right). Model E, which uses those per-path rates and no new constant beyond h, brings every
   FETCH row within ±6.1% (270K +3.1, 7352 +4.1) and has the lowest p90 (6.3%), but a worse median (4.2%) and h = 63–84 µs.
5. **The 9655 is a floor no other host has.** The counters put the helper time per layer-step at 55–70 µs + 27–31 µs per
   expert there (desktops: intercept −72 to −5 µs, slope 1.09–1.31 × S/B_c; 7352: −1 to +15 µs, 0.95–1.02 ×), and its
   token time outside the helpers' critical path is 4.7–5.7 ms against 3.3–4.6 ms on the desktops (4.7–5.1 on the 7352,
   whose helpers read at the probe's rate, so its G is not inflated). A hand-off term fitted without the 9655 comes out
   at 10–34 µs (C+h, C(g=37)+h) and leaves it at +25–36%; in-sample, h = 94 µs (C×k+h) brings it within 11.5% and costs
   the 9950X and 9950X3D C56 rows 8–10%, and held out it fails (+47.5%). Cu, the literal per-layer Eq. 1, matches A on
   the 9655 only because that host's B_cp (418 GB/s, the probe's median over 8–24 threads) is below its B_c at 44 threads.
6. **Prefetch is under-predicted by every model** (−1.9 to −11.1%; D's +1.4 on one row is the exception): the prefetch
   copy overlaps the next layer's attention, which no maximum over the MoE window represents. Charging it per layer
   moves those rows by under 2 points either way (C: −2.6 to −8.4%, A: −3.4 to −8.7%).
7. **G above the VRAM time is a rate shortfall, not a missing term.** On every desktop the helpers read at 1.1–1.3 ×
   S/B_c (rising with C), while the token time outside their critical path is 3.3–3.8 ms on the 9950X (C14/C32/C56),
   below the 3.92 ms all-in-VRAM time; the refit G (5.0–5.3 ms) absorbs the shortfall in proportion to X_c. The per-layer
   variants with a hand-off term bring G0 to 3.0–3.7 ms, but at a worse median, and A×k does not recover it (k = 0.97:
   a per-layer h and a per-expert k are nearly collinear on the desktops, so the fit cannot tell them apart).
8. With G refit per fold, the Ryzen FETCH rows are within ±2% under A: the "+4% on all hosts" of Table 2 is the frozen
   G being 0.2–0.4 ms low for these hosts, not a FETCH effect.

**Verdict.** Keep Eq. 1. The per-layer max is not the fix: it ties the token max on the median, explains at most a fifth
of the slow-link FETCH residual and none of the 9655's, and worsens prefetch. The three failures have three different
causes (asymmetric sharing of DRAM between the helpers and the copy; a per-layer floor specific to the 12-channel
EPYC; prefetch overlapping attention), and the excess of G over the VRAM time is the helpers reading 10–30% below the
probe. Any of these can be added as a named term; none is a Jensen gap.

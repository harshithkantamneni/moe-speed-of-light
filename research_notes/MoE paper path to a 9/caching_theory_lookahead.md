# Caching and prefetching with limited lookahead and imperfect predictions, applied to an MoE expert cache

Scope: theory and empirical work relevant to "Where the Seconds Go". The paper's model is batch-1 MoE decode with k-of-E experts per layer and C GPU slots per layer. A miss is either served by the CPU (bypass) or copied over PCIe (admission). MIN-with-bypass is the offline reference. Paper results: W50 ≈ 0.59 (C/k)^1.33, and Belady-style prefetch reads 1.6-2.5x MIN's bytes. A degraded-oracle experiment is planned.

Notes on sourcing:
- Every finding below has a link. Some come from companion notes in this repo (`research_notes/MoE offload paper novelty recheck/...`); those are marked "(companion notes)" and were not re-fetched in this pass.
- Several classical references were not fetched (Belady 1966, Mattson 1970, McFarling 1992, Yannakakis & Gavril 1987, Albers & Büttner). They are listed under Gaps, not as findings.

---

## Q1. Is MIN-with-bypass provably optimal in the unit-cost bypass model, what is the citation, and what changes when bypass cost differs from admission cost? (Includes integrated prefetching and caching.)

### Takeaway
I found no clean, standalone theorem paper titled "MIN with bypass is optimal".

The defensible citation chain has three links:
1. In the bypass model, each reuse interval is cached or not, independently. Maximizing hits is therefore exactly the maximum k-colorable subgraph problem on the interval graph of reuse intervals, which is solved exactly: greedy when unweighted, min-cost flow when weighted (Carlisle & Lloyd 1995, building on Yannakakis & Gavril 1987).
2. This interval formulation is how architecture work computes OPT. Hawkeye's OPTgen states explicitly that it "assumes that misses will bypass the cache".
3. FOO uses the same formulation with min-cost flow and supports non-uniform miss costs.

When bypass cost ≠ admission cost, farthest-next-use greedy is no longer guaranteed optimal. The exact offline optimum with uniform sizes is still polynomial: it is a min-cost-flow / interval LP (CHOPT 2020; FOO 2018; 2606.20539).

In the online setting, weighted costs break the value of next-arrival predictions entirely (Jiang, Panigrahi & Sun, ICALP 2020).

The integrated-prefetching literature (Cao et al. 1995; Kimbrel & Karlin; Jain & Lin 2018) directly mirrors the paper's finding:
- Using foresight "as MIN uses it" (same fetches as MIN) is near-optimal in time.
- Aggressive, Belady-style prefetch trades extra traffic for fewer stalls.

### Cited Findings
**Bypass-model optimality (interval / OPTgen formulation)**
- Hawkeye's OPTgen reconstructs OPT from "usage intervals" ("the time period that starts with a reference to X and proceeds up to (but not including) its next reference X'"). It declares a miss when "at any point in its usage interval the number of overlapping liveness intervals matches the cache's capacity". Figure 6 caption: "OPTgen assumes that misses will bypass the cache. If we wanted OPTgen to instead assume a cache with no bypassing, then the most recent entry ... would have been initialized to 1 instead of 0." — [Jain & Lin, "Back to the Future: Leveraging Belady's Algorithm for Improved Cache Replacement", ISCA 2016 (PDF)](https://cs.utexas.edu/~akanksha/isca16.pdf)
- Maximum-weight k-colorable subgraph of an interval graph ("find a k-coloring of the intervals of maximum total weight"):
  - The weighted case is solved exactly by min-cost flow ("a minimum cost flow of size k will provide a solution to weighted k-coloring of intervals") in O(k·S(n)).
  - The unweighted case builds on Yannakakis & Gavril (1987) with a greedy rule.
  - The paper notes the register-allocation (load-minimization) application.
  - [Carlisle & Lloyd, "On the k-coloring of intervals", Discrete Applied Mathematics 59 (1995)](https://www.martincarlisle.com/publications/kcoloring.pdf)
- FOO (Berger, Beckmann, Harchol-Balter, POMACS/SIGMETRICS 2018) models caching over the intervals between consecutive requests to the same object, as a min-cost flow.
  - It states: "For objects with equal sizes, computing OPT is simple (i.e., Belady)".
  - With variable sizes, OPT is "NP-hard" ("strongly NP-complete even with just three object sizes").
  - The formulation "easily supports non-uniform miss costs".
  - Its Assumption 1 is that an object can only enter the cache at its own request times (no prefetch).
  - [Berger et al., "Practical Bounds on Optimal Caching with Variable Object Sizes" (arXiv 1711.03709)](https://arxiv.org/pdf/1711.03709)
- Cao et al. (1995) state "optimal replacement" as "Every prefetch should discard the block whose next reference is furthest in the future". Their Conservative strategy "performs exactly the same number of fetches as the optimal offline demand paging algorithm (MIN)". — [Cao, Felten, Karlin, Li, "A Study of Integrated Prefetching and Caching Strategies", SIGMETRICS 1995 (PDF)](https://homes.cs.washington.edu/~karlin/papers/sigmetrics.pdf)
- Prior MoE use of MIN with bypass: 2608.07911 uses "Belady (Belady 1966) (offline optimum with bypass admission), and Belady-forced-admit", at per-layer and global scope, in block counts only. — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911) (companion notes)
- Michaud (ACM TACO 13(4), 2016) gives "an explicit formula ... giving OPT hits and misses as a function of past references", proves "OPT miss curves are always convex", and introduces "OPT tokens". The full text (HAL) was access-denied, so its treatment of bypass was not verified. — [Crossref record 10.1145/3017992](https://api.crossref.org/works/10.1145%2F3017992); [HAL hal-01411156](https://hal.archives-ouvertes.fr/hal-01411156)

**When bypass cost ≠ admission cost**
- CHOPT (Zhang, Karimi, Ahmad, Vigfusson, SIGMETRICS/POMACS 2020) asks "which data should be cached in faster memory if it could instead be served directly from slower memory?"
  - It presents "an offline algorithm for data placement across multiple tiers of memory with asymmetric read and write costs", shown to be optimal.
  - Optimal placement improves average request latency "by 8.2%-44.8% when compared with ... Belady and Mattson's offline, evict-farthest-in-the-future optimal algorithms".
  - So Belady is not latency-optimal when the slow tier can serve directly at a different cost.
  - [CHOPT PDF](https://geraldleizhang.com/publications/CHOPT_Sigmetrics20.pdf); [ACM 10.1145/3379472](https://dl.acm.org/doi/10.1145/3379472) (companion notes)
- Mandarapu & Kunkunuru (arXiv 2606.20539, 2026): "For uniform-size page caches with heterogeneous miss costs the offline dollar-optimum is exact in polynomial time via an integral interval linear program." — [arXiv 2606.20539](https://arxiv.org/abs/2606.20539) (companion notes)
- Jiang, Panigrahi, Sun (ICALP 2020), online weighted paging:
  - With per-request next-arrival predictions (PRP), "any deterministic algorithm is Ω(k)-competitive, and any randomized algorithm is Ω(log k)-competitive", i.e., no better than with no predictions.
  - Strong lookahead of ℓ ≤ n−k distinct pages also gives no improvement.
  - "Strong per-request predictions" (SPRP: next arrival plus all intervening requests) allow "a deterministic 2-competitive algorithm".
  - Under edit-distance error there is "no deterministic algorithm whose cost is o(k)·OPT + o(ℓ_ed)", but a randomized algorithm with cache k+1 gets O(OPT + ℓ_ed).
  - [Jiang, Panigrahi, Sun, "Online Algorithms for Weighted Paging with Predictions" (arXiv 2006.09509)](https://arxiv.org/pdf/2006.09509)
- Caching with rejection (Epstein et al.), as distinguished in Khare & Young: "a file can be rejected only at the time step when it is requested. Moreover, a rejected file can incur retrieval cost or rejection penalty again in the future". This is the online bypass-with-penalty model. — [Khare & Young, "Caching with rental cost and zapping" (arXiv 1208.2724)](https://arxiv.org/pdf/1208.2724); [Epstein et al., "Online File Caching with Rejection Penalties", Algorithmica](https://cris.haifa.ac.il/en/publications/online-file-caching-with-rejection-penalties/)
- An MoE-specific break-even: 2608.12103 computes "breakeven reuse count r* = ... = 1.06" for promoting an expert to HBM. — [arXiv 2608.12103](https://arxiv.org/abs/2608.12103) (companion notes)

**Integrated prefetching and caching: MIN-as-used vs aggressive prefetch**
- Cao et al. (1995) give four rules: optimal prefetching, optimal replacement, "Do No Harm" ("Never discard block A to prefetch block B when A will be referenced before B") and "First Opportunity".
  - Aggressive: elapsed time ≤ min(1 + F/K, 2) × optimal (F = fetch time, K = cache size in blocks).
  - Conservative: ≤ 2 × optimal (tight), with MIN's fetch count.
  - On traces, Aggressive was within 1.024× (Ultrix) and 1.02× (Sprite) of optimal, and integrated strategies cut running time "by up to 50%".
  - The paper assumes "full knowledge of file accesses".
  - [Cao et al. 1995 PDF](https://homes.cs.washington.edu/~karlin/papers/sigmetrics.pdf)
- Kimbrel & Karlin, parallel disks:
  - "Reverse aggressive requires at most 1 + dF/K times the optimal elapsed time".
  - Aggressive: ≤ d(1 + (F+1)/K) × optimal. Conservative: ≤ d+1 × optimal.
  - "more practical limited-lookahead versions of our algorithms do well in practice".
  - [Kimbrel & Karlin, "Near-optimal parallel prefetching and caching" (PDF)](https://homes.cs.washington.edu/~karlin/papers/tracy.pdf)
- Albers, Garg, Leonardi (JACM 2000): "an optimum prefetching/caching schedule for a single disk problem can be computed in polynomial time ... formulating the prefetching/caching problems as linear programs." — [JACM 10.1145/355541.355542](https://dl.acm.org/doi/10.1145/355541.355542) (companion notes; abstract only)
- Jain & Lin (ISCA 2018):
  - "Belady's algorithm ... minimizes the total number of misses, including those for lines brought in by prefetches, but it does not minimize the number of demand misses".
  - Demand-MIN: "Evict the line that will be prefetched furthest in the future, and if no such line exists, evict the line that will see a demand request furthest in the future".
  - SPEC 2006, 4-core: LRU MPKI 29.8, MIN 21.7, Demand-MIN 16.9. Traffic: MIN TPKI 45.4 vs Demand-MIN 79.4 (≈+75%). Flex-MIN: MPKI 17.7, TPKI 60.1.
  - [Jain & Lin, "Rethinking Belady's Algorithm to Accommodate Prefetching", ISCA 2018 (PDF)](https://cs.utexas.edu/~akanksha/isca18.pdf)

### Inferences
- **Proof sketch suitable for the paper (my derivation, built from the cited formulations).**
  - With bypass, a request at step t to expert e is a hit iff e was resident throughout the reuse interval [prev(t), t).
  - Residency in different intervals is independent, because bypass means no forced admission.
  - A feasible schedule is therefore a set of intervals with at most C overlapping at any time, and hits = number of selected intervals.
  - Maximizing hits is unweighted maximum C-colorable subgraph on an interval graph. Greedy is optimal for it (Yannakakis & Gavril; Carlisle & Lloyd).
  - Greedy is equivalent to: scan in time order; when more than C intervals overlap, drop the one with the latest right endpoint.
  - That is the farthest-next-use rule, extended to the incoming request (MIN-with-bypass).
  - Set-valued steps (k experts arriving at once) create ties in left endpoints, which do not affect interval-graph optimality.
  - OPTgen implements exactly this feasibility check with bypass.
- **Cost of admission vs bypass.** Let a = cost of an admitted miss (PCIe + GPU), b = bypassed miss (CPU), h = hit.
  - A residency chain starting with an admission and covering m later hits beats bypassing all m+1 requests iff a + m·h < (m+1)·b, i.e., m > (a−b)/(b−h).
  - When a ≠ b, the objective is weighted. The exact offline optimum is a min-cost flow / integral interval LP: FOO-style intervals, with an extra "admission" arc cost. It is polynomial for uniform expert sizes but is not the greedy MIN.
  - MIN-with-bypass then remains optimal for **hit count** and for **bytes moved under single-read admission**, but not for seconds.
  - This is consistent with the paper using MIN miss counts as an input to a bound rather than as a time-optimal policy.
  - The CHOPT result is the closest published analogue of "Belady is not latency-optimal when the slow tier can serve".
- The paper's two measurements map closely onto these classical results:
  - "Foresight as MIN uses it (single read)" vs "Belady-style prefetch over-admits and reads 1.6-2.5× MIN's bytes" corresponds to Cao's Conservative (MIN's fetch count, ≤2× time) vs Aggressive (earlier fetches, possibly more of them).
  - Jain & Lin's MIN vs Demand-MIN shows a similar traffic gap (+75% traffic for −22% demand MPKI).
  - Citing both makes the 1.6-2.5× result look expected, not anomalous. It also suggests a Flex-MIN-like middle ground as a named design point.
- The weighted-paging lower bound (Jiang et al.) implies:
  - If bypass and admission costs differ materially, a predictor that supplies only next-use times is theoretically insufficient.
  - The "window oracle" (all requests in the next W tokens) is SPRP-like, so it is the right kind of foresight.

### Gaps
- Classical primary sources were not fetched in this pass. Bibliographic details below come from my own knowledge and need checking before they go in refs.bib:
  - Belady (1966), IBM Systems Journal 5(2):78–101.
  - Mattson, Gecsei, Slutz, Traiger (1970), IBM Systems Journal 9(2):78–117.
  - McFarling, "Cache Replacement with Dynamic Exclusion", ISCA 1992.
  - Yannakakis & Gavril (1987), Information Processing Letters.
- I found no paper that states and proves "MIN with bypass is optimal for the bypass model" as a standalone theorem. Michaud's TACO 2016 may contain it, but the full text was inaccessible.
- I did not verify whether OPT with bypass at capacity C equals OPT without bypass at C+1. I believe this equivalence is folklore, but it is unverified.
- I did not verify the classical claim that "demand paging is optimal" (prefetching never reduces the fetch count) in a primary source. Cao et al.'s "Conservative = MIN fetch count" statement is the closest cited support.
- Albers & Büttner (integrated prefetching and caching in single/parallel disk systems) was not fetched. Exact results are unverified.

---

## Q2. What is known about competitive ratio or fault rate as a function of lookahead length, and how does it compare with the empirical W50 scaling?

### Takeaway
Worst-case theory says plain lookahead (the next l requests) does not help, because the adversary repeats requests. Lookahead helps only when measured in **distinct pages** (strong lookahead) or in faults (resource-bounded/natural lookahead), and then it trades roughly one-for-one with cache size:
- Albers: deterministic ratio k − l for strong lookahead l ≤ k−2; randomized between H(k−l) and 2H(k−l).
- Breslauer: "the competitive ratio is a function of k + l".

Torng shows that in a full-access-cost model, sufficient lookahead gives ratio 2. Patterson's "prefetch horizon" sets the lookahead beyond which latency hiding gains nothing.

No source gives an average-case W50-style law. The theory instead suggests a test: express W in distinct experts and check whether the half-gap lookahead is a fixed fraction of C. That would explain the super-linear (C/k)^1.33 exponent via sub-linear growth of distinct experts per window.

### Cited Findings
- Albers (Algorithmica 18:283–305, 1997): "A paging algorithm is on-line with strong lookahead l if it sees the present request and a sequence of future requests that contains l pairwise distinct pages ... strong lookahead has practical as well as theoretical importance and improves the competitive factors ... This is the first model of lookahead having such properties." — [Albers, "On the Influence of Lookahead in Competitive Paging Algorithms", Algorithmica 1997](https://link.springer.com/doi/10.1007/PL00009158)
- Albers' technical report (MPI-I-92-143):
  - Weak lookahead l = "sees the present request and the next l future requests". Strong lookahead l = future sequence containing "l pairwise distinct requests which also differ from the present request".
  - Theorem 1: LRU(l) is (k − l)-competitive for l ≤ k − 2.
  - Theorem 3: any deterministic algorithm with strong lookahead l ≤ k − 2 has c ≥ k − l.
  - Randomized: the extraction gave MARKER(l) 2H(k−1) and lower bound H(k−1). This conflicts with Breslauer's summary below (see Gaps).
  - [Albers, MPI-I-92-143 (PDF)](https://domino.mpi-inf.mpg.de/internet/reports.nsf/c125634c000710cec125613300585c64/470f3e826a5f094bc12560400053710f/$FILE/MPI-I-92-143.pdf)
- Breslauer ("On Competitive On-Line Paging with Lookahead", BRICS 1995; later TCS):
  - Summary of Albers: "there exists an on-line paging algorithm with K pages and strong lookahead l ... that has competitive ratio K − l and that this competitive ratio is the best possible"; randomized: "there exists an algorithm that is 2·H(K − l)-competitive and that no algorithm is better than H(K − l)-competitive".
  - Defines Young's resource-bounded lookahead ("never incur more than l + 1 page faults on the requests in the lookahead queue") and his own natural lookahead ("at no time ... more than l+1 distinct page requests that are not currently in fast memory").
  - Key conclusion: "the competitive ratio is a function of k + l. Thus, under these measures for the lookahead size, an on-line algorithm obtains the same benefit from using an extra page or knowing an extra bit of future requests."
  - [Breslauer, BRICS report (PDF)](https://tidsskrift.dk/brics/article/download/19951/17604/45315)
- Jiang, Panigrahi, Sun summarize Albers: ℓ-strong lookahead with ℓ > k − 2 yields a constant approximation for unweighted paging, but this does not transfer to weighted paging. — [arXiv 2006.09509](https://arxiv.org/pdf/2006.09509)
- Torng (Algorithmica 20:175–200, 1998) reintroduces miss penalty and page size. In this model "finite lookahead can be used to obtain algorithms with improved competitive ratios", and "modified marking algorithms with sufficient lookahead achieve competitive ratios of 2". — [Torng, "A Unified Analysis of Paging and Caching", Algorithmica 1998](https://link.springer.com/doi/10.1007/PL00009192)
- Koutsoupias & Papadimitriou (SIAM J. Comput. 2000) propose comparing "information regimes", used "to explore the power of lookahead in server and task systems". — [Koutsoupias & Papadimitriou, "Beyond competitive analysis"](https://cgi.di.uoa.gr/~elias/papers/paper-kp00.html)
- Patterson et al. (SOSP 1995) define the prefetch horizon P(T_CPU) = T_disk / (T_CPU + T_hit + T_driver): "there is no benefit from prefetching more deeply than the prefetch horizon". — [Patterson, Gibson, Ginting, Stodolsky, Zelenka, "Informed Prefetching and Caching", SOSP 1995 (PDF)](https://cs.uwaterloo.ca/~brecht/courses/epfl/Possible-Readings/prefetching-to-memory/informed-prefetching-and-caching-sosp-1995.pdf)
- Belady boundary (LRB): "the minimum of time to next request for all evicted objects by Belady's MIN algorithm".
  - Relaxed Belady evicts any object whose next request is beyond this boundary.
  - It incurs only "9-13% more misses" than Belady on CDN traces, against a "25–40%" gap from state-of-the-art heuristics to Belady.
  - [Song, Berger, Li, Lloyd, "Learning Relaxed Belady for Content Distribution Network Caching", NSDI 2020 (PDF)](https://www.cs.princeton.edu/~wlloyd/papers/lrb-nsdi20.pdf)
- Hawkeye/OPTgen (past-history analogue of lookahead): "a reuse window of 8× is necessary to generate OPT's solution accurately". With 64 sampled sets and quantum 4, "OPTgen can achieve 99% accuracy". — [Jain & Lin, ISCA 2016 (PDF)](https://cs.utexas.edu/~akanksha/isca16.pdf)

### Inferences
- **How to compare W50 with theory (proposed analysis, not from a source).**
  1. Let D(W) be the mean number of distinct experts per layer in a W-token window. A window of W tokens is a strong lookahead of l ≈ D(W) − (repeats of the current set).
  2. Albers/Breslauer predict that lookahead value depends on l relative to the cache size (k − l, or k + l in their notation).
  3. If routing has sub-linear working-set growth, D(W) ≈ k·W^β with β < 1, and the half-gap point sits at a fixed fraction of C, then W50 ∝ (C/k)^(1/β).
  4. The fitted exponent 1.33 implies β ≈ 0.75. This is testable directly on the traces by fitting D(W).
  - If it holds, the W50 law is "explained" by classical strong-lookahead theory plus the routing working-set curve.
  - It also suggests reporting W50 in distinct-expert units (l50/C), which may collapse across models better than tokens.
- Weak lookahead does no worst-case good because repeated requests carry no information. In MoE decode, consecutive tokens often reuse experts, so a W-token window carries fewer than W·k distinct experts. This is why tokens are a poor unit for lookahead.
- **Two distinct roles of lookahead should be separated in the paper:**
  - (a) Decision quality (which experts to admit or bypass). Its natural scale is the Belady boundary or reuse distances, and the relaxed-Belady result suggests a window ≈ boundary suffices for most of the benefit.
  - (b) Latency hiding (start the PCIe copy early). Its scale is the prefetch horizon, ≈ copy time per expert / compute time per layer-step.
  - W50 measures (a). The 16-51% engine gain mixes both.
- The relaxed-Belady result (+9–13% misses when you only know "beyond boundary or not") suggests a check: a window oracle whose W just covers the Belady boundary should get most of MIN's benefit. Measuring the Belady-boundary distribution (in tokens) vs C/k on the MoE traces would be a direct comparison to W50.
- Hawkeye's "8× cache" rule for past history translates to about 8C/k tokens per layer if every token brings k new experts, and more given reuse. The paper's W50 at C/k = 4 is about 3.7 tokens (0.59·4^1.33). So half the gap closes well inside the window an OPTgen-style reconstruction needs for ~99% fidelity. That is plausible, since half-gap is far from full fidelity.

### Gaps
- Albers' randomized bound: the MPI tech report extraction (2H(k−1) / H(k−1)) conflicts with Breslauer's summary (2H(K−l) / H(K−l)). The journal full text was not accessible. Treat H(k−l)..2H(k−l) as the likely journal statement, but verify.
- Breslauer's exact LRU-with-lookahead formula and Young's resource-bounded bound were garbled in extraction. Only the qualitative "function of k + l" statement is reliable here. Young's original paper ("On-line caching as cache size varies", SODA 1991, or a related tech report) was not fetched.
- Torng's exact lookahead size needed for ratio 2 (I believe it is expressed in terms of miss penalty p and k) was not extracted.
- I could not verify Koutsoupias–Papadimitriou's exact comparative-ratio formula for lookahead.
- I found no average-case (stochastic or IRM) result giving the fault rate as a function of lookahead length. W50-type empirical laws appear absent from the literature I checked.

---

## Q3. How do learning-augmented caching algorithms degrade with prediction error, and which prediction form (next-arrival time vs binary reuse) is most robust?

### Takeaway
With next-arrival-time predictions and ℓ1 error η, the best-known bounds are summarized below. The bounds are additive in η/OPT, so a few large time errors on decision-irrelevant pages can dominate η.

| Algorithm | Competitive ratio |
|---|---|
| Predictive Marker (Lykouris & Vassilvitskii) | min(2 + 2√(5η/OPT), 4H_k) |
| Rohatgi | O(1 + min((η/OPT)/k, 1)·log k), lower bound Ω(log min((η/OPT)/(k log k), k)) |
| BlindOracle (Wei) | min(1 + 2η/OPT, 2 + 4η/((k−1)OPT)), combined with LRU/Marker for robustness |

Binary, decision-level predictions behave better:
- **Discard bits** ("would Belady evict this page before its next request?") give **exact consistency** (ALG = OPT with perfect bits) and an additive cost per wrong bit (Antoniadis et al., ICML 2023).
- Empirically, imitating Belady's decisions beats regressing reuse distance (Parrot cache head vs reuse head).
- LRB and Hawkeye/Glider succeed with "beyond-the-boundary" or cache-friendly/averse labels.

For weighted costs, next-arrival predictions alone are provably useless (Jiang et al.).

### Cited Findings
- Lykouris & Vassilvitskii (ICML 2018; JACM 2021):
  - η = Σ_i |y(σ_i) − h(σ_i)| (ℓ1 error of predicted next arrival).
  - Theorem 3.3: Predictive Marker has cr ≤ min(2 + 2√(5η/OPT), 4H_k).
  - Blindly following predictions ("Algorithm B", evict predicted-furthest) has cr = Ω(η/OPT).
  - Theorem 3.4: deterministic marking algorithms using predictions for tie-breaking have cr = Ω(min(S_ℓ(ε), k)).
  - Experiments: BrightKite and CitiBike with lognormal noise σ. Predictive Marker "degrades gracefully" while BlindOracle "deteriorates rapidly". With the PLECO predictor: PM 1.266 vs LRU 1.280 (BK), 1.810 vs 1.859 (Citi).
  - [Lykouris & Vassilvitskii, "Competitive Caching with Machine Learned Advice" (arXiv 1802.05399)](https://arxiv.org/pdf/1802.05399)
- Rohatgi (SODA 2020): upper bound O(1 + min((η/OPT)/k, 1)·log k), improving O(1 + min(√(η/OPT), log k)). Lower bound Ω(log min((η/OPT)/(k log k), k)). — [Rohatgi, "Near-Optimal Bounds for Online Caching with Machine Learned Advice" (arXiv 1910.12172)](https://arxiv.org/abs/1910.12172v1)
- Wei (APPROX/RANDOM 2020):
  - Theorem 1.1: "BlindOracle obtains a competitive ratio of min(1 + 2η/OPT, 2 + 4η/((k−1)·OPT))".
  - Deterministic combination with LRU: 2·min(..., k). Randomized combination with Equitable: (1+ε)·min(..., H_k).
  - Theorem 1.4 (deterministic lower bound): 1 + Ω(min(η/(k·OPT), k)).
  - [Wei, "Better and Simpler Learning-Augmented Online Caching" (arXiv 2005.13716)](https://arxiv.org/pdf/2005.13716)
- Skachkov, Ponomaryov, Dorn, Demin (arXiv 2410.01760, 2024) improve BlindOracle to min(1 + η/OPT, 3 + (3/k)·η/OPT). They propose AlternatingOracle with an O(√((1/k)·η/OPT))-type bound, and its combination with PredictiveMarker achieving (1+γ)·min(H_k, ·). Their prior-work table quotes Wei's bound as "min{1+2η/OPT, 4+4/(k−1)·η/OPT}" (constant 4, not 2; see Gaps). — [Skachkov et al., "Learning-augmented online caching: new upper bounds" (arXiv 2410.01760)](https://arxiv.org/pdf/2410.01760)
- Antoniadis, Boyar, Eliáš, Favrholdt, Hoeksma, Larsen, Polak, Simon (ICML 2023), one bit per request:
  - **Discard predictions** ("whether LFD would evict the current page before it gets requested again"): deterministic "(1, k−1, 1)-competitive", randomized "(1, 2H_k, 1)-competitive". The first coordinate is the multiplier on OPT (exact consistency); the others multiply the counts of the two kinds of wrong predictions.
  - **Phase predictions** ("whether the current page will be requested in the following k-phase"): MARK&PREDICT is "(2, H_k, 1)-competitive" and O(log k)-robust.
  - Matching lower bounds: e.g., no deterministic algorithm with α + β < k or α + (k−1)γ < k.
  - No experiments.
  - [Antoniadis et al., "Paging with Succinct Predictions" (arXiv 2210.02775)](https://arxiv.org/abs/2210.02775); [full text](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICML/2023/paging_with_succinct_predictions__12689074.txt)
- Chłędowski, Polak, Szabucki, Żołna (ICML 2021):
  - First comprehensive evaluation of these algorithms on 13 SPEC CPU2006 traces (CRC2) with Parrot predictors.
  - "a straightforward method -- blindly following either a predictor or a classical robust algorithm, and switching whenever one becomes worse than the other -- has only a low overhead over a well-performing predictor".
  - PARROT-REUSE (reuse-distance prediction) "significantly underperforms" PARROT-CACHE (eviction-decision imitation).
  - With predictors trained on 1% of the data, the augmented algorithms "approach MARKER, even when coupled predictors are wildly inaccurate".
  - [arXiv 2106.14693](https://arxiv.org/abs/2106.14693v1); [full text](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICML/2021/robust_learningaugmented_caching_an_experimental_study__f79fe0a5.txt)
- Weighted paging (Jiang, Panigrahi, Sun, ICALP 2020): next-arrival predictions give Ω(k) deterministic / Ω(log k) randomized lower bounds. Next-arrival plus all intervening requests gives 2-competitive. — [arXiv 2006.09509](https://arxiv.org/pdf/2006.09509)
- LRB regresses log(time-to-next-request) but "primarily care[s] about which side of the boundary the next request is on". Its "good decision ratio" (evicted object's next request beyond the Belady boundary) "correlates strongly with the byte miss ratio". — [LRB NSDI 2020 (PDF)](https://www.cs.princeton.edu/~wlloyd/papers/lrb-nsdi20.pdf)
- Raven (CoNEXT 2022) predicts *distributions* of next-arrival times with a Mixture Density Network. It evicts by "the probability that an object's residual time is greater than the residual time of any other object in cache". — [Hu, Ramadan, Ye, Tian, Zhang, "Raven" (PDF)](https://par.nsf.gov/servlets/purl/10424505)

### Inferences
- **Most robust prediction form for the MoE oracle experiment: decision-level binary information.** Candidates are "will this expert be requested before MIN's eviction horizon / within W?" (discard or Belady-boundary bit), not exact next-use times. Reasons:
  - (i) Discard-bit algorithms have consistency exactly 1, and error cost is per wrong bit.
  - (ii) Next-arrival ℓ1 error charges heavily for errors on far-future requests that never change a decision.
  - (iii) Empirically, decision imitation beats reuse-distance regression (Parrot heads), and binary-boundary policies (LRB, Hawkeye) work.
- **Mapping the planned degradations onto theory:**
  - Window truncation (W < all) turns every reuse beyond W into "no reuse". That is a large ℓ1 error but often decision-irrelevant when W exceeds the Belady boundary, so ℓ1-based bounds will look pessimistic.
  - Randomly *dropping* future requests (recall < 1) produces false "will not be reused" bits, so the policy bypasses or evicts experts that will be needed.
  - *Corrupting* requests (replacing them with wrong experts) produces both false negatives and false positives. False positives keep or admit useless experts.
  - Antoniadis et al.'s asymmetric constants (k−1 vs 1 for the two error types) suggest these two arms should be reported **separately**. Drops and spurious insertions likely have very different marginal costs.
- Use the fraction of MIN decisions changed (LRB's good-decision ratio, computed against full-foresight MIN) as a second x-axis besides recall and W. LRB shows it tracks miss ratio, and it is the quantity the discard-prediction bounds are stated in.
- Consider a robustness arm: a degraded oracle plus "switch to LRU/LFU when worse" (Chłędowski's follow-the-better combiner). Theory and experiments both say this costs little when the oracle is good and caps the loss when it is bad.
- Because MoE bypass and admission costs differ, the Jiang et al. result implies a weighted variant. A "predicted next-use only" oracle (no intervening requests) is a qualitatively weaker form than the window oracle and could be a separate arm.

### Gaps
- Discrepancy in Wei's BlindOracle second term (2 + 4η/((k−1)OPT) in arXiv v1 vs "4 + 4/(k−1)·η/OPT" in Skachkov et al.'s table). The APPROX final version was not checked.
- Skachkov et al.'s AlternatingOracle bound as extracted ("O(√((1/k)·η/OPT))") lacks an additive constant and should be verified in the PDF.
- In Antoniadis et al., the exact mapping of the β and γ coefficients to false-positive vs false-negative counts was not verified verbatim.
- Not fetched:
  - Antoniadis, Coester, Eliáš, Polak, Simon, "Online metric algorithms with untrusted predictions" (ICML 2020; action predictions).
  - Im, Kumar, Petety, Purohit, "Parsimonious learning-augmented caching" (ICML 2022).
  - Bansal et al., "Learning-augmented weighted paging" (SODA 2022).
  - Sadek & Eliáš (ICLR 2024).
- The JACM 2021 version of Lykouris–Vassilvitskii may restate constants. The arXiv (Aug 2020) version was used.

---

## Q4. What empirical curves exist of cache benefit vs prediction accuracy or horizon in CPU caches, CDNs or storage that a degraded-oracle experiment should be compared against?

### Takeaway
There are few curves that sweep *oracle* horizon or recall directly. The comparable anchors are:
- **Synthetic-noise sweeps**: Lykouris & Vassilvitskii (lognormal σ on next-arrival); Chłędowski et al. (SPEC traces, full vs 1%-trained predictor).
- **Accuracy → miss-reduction pairs** for Belady-imitating CPU policies: Hawkeye 81% accuracy → 17.0% miss reduction vs LRU; Glider 88.8% vs Hawkeye 84.9% → 8.9% vs 7.1% miss reduction.
- **Gap-closure numbers** for learned CDN policies: relaxed Belady +9–13% misses; LRB 4–25% WAN reduction; Raven closes 37.2% of the OHR gap.
- **Integrated-prefetching results** with full knowledge: Cao aggressive within 2.4% of optimal; Jain & Lin MIN vs Demand-MIN traffic.

The common pattern: modest accuracy gains buy disproportionately small miss gains, and binary-decision accuracy, not time accuracy, predicts benefit.

### Cited Findings
- **CPU caches (Belady imitation):**
  - Hawkeye: predictor "81% accurate in predicting OPT's decisions"; "average miss reduction of 17.0% on the 20 memory-intensive SPEC benchmarks" vs LRU. OPTgen needs an 8× cache-size reuse window for accurate OPT reconstruction. — [Jain & Lin, ISCA 2016 (PDF)](https://cs.utexas.edu/~akanksha/isca16.pdf)
  - Glider (MICRO 2019): offline attention-LSTM 82.6% accuracy vs Hawkeye predictor 72.2%; online Glider 88.8% vs Hawkeye 84.9%. Miss reduction vs LRU: Glider 8.9%, Hawkeye 7.1%, SHiP++ 7.5%, MPPPB 6.5% (single-core). 4-core IPC +14.7% vs +13.6% (Hawkeye). Glider features: unordered history of last k = 5 unique PCs. — [Shi, Huang, Jain, Lin, "Applying Deep Learning to the Cache Replacement Problem", MICRO 2019 (PDF)](https://www.cs.utexas.edu/~lin/papers/micro19c.pdf)
  - Parrot (ICML 2020): imitation of Belady from past accesses only. On 13 memory-intensive SPEC applications it improves on the state of the art by 20%; on a web-search benchmark, by 61% over LRU. — [Liu, Hashemi, Swersky, Ranganathan, Ahn, "An Imitation Learning Approach for Cache Replacement" (arXiv 2006.16239)](https://arxiv.org/abs/2006.16239)
  - Chłędowski et al. (ICML 2021): learning-augmented algorithms plus Parrot on the same SPEC traces. Switching combiners have low overhead with a good predictor and approach Marker with a 1%-trained predictor. The reuse-distance head is worse than the eviction head. — [arXiv 2106.14693](https://arxiv.org/abs/2106.14693v1)
  - Jain & Lin (ISCA 2018), with prefetching: MIN MPKI 21.7 / TPKI 45.4; Demand-MIN 16.9 / 79.4; Flex-MIN 17.7 / 60.1; LRU MPKI 29.8. — [ISCA 2018 PDF](https://cs.utexas.edu/~akanksha/isca18.pdf)
- **CDNs:**
  - LRB: state-of-the-art heuristics leave "a gap of 25–40%" to Belady. Relaxed Belady costs "9-13% more misses". LRB "reduces WAN traffic by 4–25%" vs deployed B-LRU. The good-decision ratio correlates strongly with byte miss ratio. — [LRB NSDI 2020 (PDF)](https://www.cs.princeton.edu/~wlloyd/papers/lrb-nsdi20.pdf)
  - Raven: "reduces the OHR gap between SOTAs and Belady by 37.2% on average, and ... BHR gap ... by 29.2%"; up to 7.3% OHR, 18.8% traffic and 17.9% latency improvements. — [Raven, CoNEXT 2022 (PDF)](https://par.nsf.gov/servlets/purl/10424505)
- **Synthetic noise on next-arrival predictions:** Lykouris & Vassilvitskii sweep lognormal σ on BrightKite/CitiBike. Predictive Marker degrades gracefully; BlindOracle degrades rapidly. — [arXiv 1802.05399](https://arxiv.org/pdf/1802.05399)
- **Storage / file systems (full or hinted knowledge):**
  - Cao et al.: Aggressive within 1.024× and 1.02× of optimal elapsed time on Ultrix and Sprite traces; up to 50% runtime reduction. — [SIGMETRICS 1995 PDF](https://homes.cs.washington.edu/~karlin/papers/sigmetrics.pdf)
  - Patterson et al. (TIP): no benefit beyond the prefetch horizon. With hints: XDataSlice 6× speedup on 10 disks; Agrep −83% execution time; Gnuld −87% stall; Sphinx −17%; Postgres up to 75%. — [SOSP 1995 PDF](https://cs.uwaterloo.ca/~brecht/courses/epfl/Possible-Readings/prefetching-to-memory/informed-prefetching-and-caching-sosp-1995.pdf)
  - Kimbrel & Karlin note that limited-lookahead versions of their algorithms "do well in practice". — [PDF](https://homes.cs.washington.edu/~karlin/papers/tracy.pdf)
- **MoE-specific:** SpecMD (Apple, 2026) reports that "higher prediction accuracy does not guarantee better cache performance" for expert prefetch/caching. — [Hoang, Jaiswal, Samragh, Cho, "SpecMD" (arXiv 2602.03921)](https://www.arxiv.org/pdf/2602.03921)

### Inferences
- **Glider vs Hawkeye calibration point:** ~4 pp of online accuracy (84.9 → 88.8%) bought ~1.8 pp extra miss reduction vs LRU (7.1 → 8.9%). If the MoE engine shows a steep benefit-vs-recall curve near recall 1.0, that would contrast with CPU-cache experience. A flatter curve (most benefit retained at recall 0.8–0.9) would match it. Either is reportable.
- **Gap closure on a common axis.** The paper's W50 is a "fraction of online-to-MIN gap closed" metric. That is the same axis as:
  - Raven (37% of the OHR gap closed);
  - relaxed Belady (it forfeits roughly 9–13 points of the MIN advantage, measured as % more misses);
  - learned CPU policies (Hawkeye's 17% miss reduction vs a much larger OPT reduction).
  Reporting the degraded-oracle results as "% of gap closed" makes them directly comparable to these.
- The Jain & Lin MIN vs Demand-MIN traffic numbers (+75% traffic) are the closest published analogue of the paper's "Belady-style prefetch reads 1.6–2.5× MIN's bytes". Cao's Conservative vs Aggressive is the time-domain analogue.
- No published sweep of exact-oracle *window length* W vs hit rate was found for CPU, CDN or storage caches. The planned experiment fills a real gap. Torng's and Albers' theory, plus the Patterson horizon, are the nearest references.

### Gaps
- I did not find a CPU, CDN or storage paper that sweeps exact-future lookahead W (truncated Belady) vs hit rate. One may exist in the prefetching literature (e.g., Kimbrel et al. OSDI 1996, "A trace-driven comparison of algorithms for parallel prefetching and caching", cited by Kimbrel & Karlin as [28]), but it was not fetched.
- Exact per-σ numbers from the Lykouris–Vassilvitskii Figure 1 and per-trace numbers from Chłędowski et al. were not extracted, only qualitative trends.
- Parrot's ablations (history length, reuse vs decision heads) were not extracted from the full paper; only the abstract was read. The extraction tool's summary of Parrot's headline result was garbled ("increases cache miss rates"). Check the exact abstract wording before quoting.

---

## Q5. Is there 2024–2026 work applying Belady or learning-augmented caching to MoE experts or LLM KV caches?

### Takeaway
Yes, but only partially. Belady/MIN (sometimes with bypass) is used as a **block-count reference**:
- 2608.07911 and "Paging the Experts" use it as a miss reference.
- Angelopoulos et al. (Euro-Par 2025) give competitive-ratio lower bounds for layered expert paging.
- 2608.12103 uses Belady for miss headroom.
- FlashMoE trains a model to imitate Belady.

LLM prefix-cache work (LPC, NeurIPS 2025) is learning-augmented eviction but does not use Belady as a bound.

In the sources checked, I found **no** MoE or KV paper that degrades oracle lookahead or recall and measures the effect in a real engine. One paper (2609.04238) reports that a perfect-prediction trace oracle barely helps on its edge platform.

### Cited Findings
- Angelopoulos, Marchal, Obrecht, Simon, "Cache Management for Mixture-of-Experts LLMs" (Euro-Par 2025; arXiv 2509.02408):
  - Layered paging model.
  - "Fixed cache partitions lead to unbounded competitive ratio".
  - Deterministic lower bound k − ℓ + 1; randomized lower bound max(H_n, log(ℓ)/(6n)); LRU ≥ k.
  - Proposed LLRU policy (~15% fewer faults than LRU on Llama-MoE traces).
  - OPT-DIST (per-layer partition) suffers up to 3× more misses than shared OPT on synthetic Zipf traces.
  - No predictions or lookahead used.
  - [arXiv 2509.02408 (HTML)](https://arxiv.org/html/2509.02408v1)
- 2608.07911 (Yu Zhang, Aug 2026): MIN with bypass and forced-admit Belady on MoE traces, per-layer and global. "Miss bursts are reported in block counts, not in time." — [arXiv 2608.07911](https://arxiv.org/abs/2608.07911) (companion notes)
- 2608.12103 (Si, Lin, Li, Zhang, Aug 2026):
  - "Belady bounds the remaining miss-reduction potential at roughly one third of the misses LRU still incurs".
  - Global vs per-layer differ by ≤ 2.7 pp.
  - Its one-layer lookahead predictor PILOT "recalls 64.7% of decode-time selections, with a per-layer range of 1–84%".
  - Break-even reuse r* = 1.06.
  - [arXiv 2608.12103](https://arxiv.org/abs/2608.12103) (companion notes)
- "Paging the Experts" (2609.29032, Sep 2026): a Belady-style lower miss bound on a shared all-layer cache: "not an online speed prediction". — [arXiv 2609.29032](https://arxiv.org/pdf/2609.29032) (companion notes)
- 2609.04238 (edge MoE, RK3588 and M4): "trace-driven oracle (perfect prediction, 0.12 → 0.13)" does not help. — [arXiv 2609.04238](https://arxiv.org/abs/2609.04238) (companion notes)
- FlashMoE (arXiv 2601.17063, Jan 2026): trains a feed-forward network on recency/frequency features to approximate Belady for an SSD-backed expert cache.
  - Hit rate +21% (OLMoE) and +51% (Qwen3) vs LRU.
  - On OLMoE, Belady reaches ~86% hit rate vs LRU ~73%.
  - "LRU's evicted experts were reused 34.2% of the time" vs Belady's 0.1%.
  - 2.6× speedup vs existing MoE systems.
  - [arXiv 2601.17063](https://arxiv.org/html/2601.17063v1)
- ProMoE (arXiv 2410.22134): a learned predictor of next-layer experts; no Belady. Stride prefetching (one layer further ahead) costs only ~5% predictor accuracy; average accuracy 84.7% on one model. — [arXiv 2410.22134](https://arxiv.org/pdf/2410.22134)
- SpecMD (arXiv 2602.03921, Feb 2026): MoE cache-policy benchmark; Least-Stale policy; no Belady. "higher prediction accuracy does not guarantee better cache performance." — [arXiv 2602.03921](https://www.arxiv.org/pdf/2602.03921)
- KV/prefix caches:
  - LPC (NeurIPS 2025): "the prevalent least-recently-used (LRU) eviction algorithm has a large gap to the optimal algorithm". It learns conversation-continuation probabilities and gets "18-47% reductions in required cache sizes for equivalent hit ratios". — [Yang, Li, Li, Lloyd, "Learned Prefix Caching for Efficient LLM Inference"](https://neurips.cc/virtual/2025/poster/117662)
  - "Not All Tokens Are Worth Caching" (arXiv 2605.18825) learns semantic-aware prefix eviction (4.8–5.9 pp hit-ratio gains; 1.4–2.7× TTFT). It does not use Belady. — [arXiv 2605.18825](https://arxiv.org/pdf/2605.18825)

### Inferences
- For "Where the Seconds Go", the closest theory neighbour is Angelopoulos et al.:
  - Its per-layer-partition result (unbounded ratio; 3× OPT-DIST vs OPT on Zipf) bears directly on the per-layer C budget.
  - Its lower bounds are for no-lookahead online algorithms. Nobody has combined layered paging with strong lookahead or predictions; that is open.
- The MoE literature uses "prediction accuracy" (recall of next-layer experts: PILOT 64.7%, ProMoE 84.7%) as its metric. The degraded-oracle experiment gives the missing mapping from recall to seconds, so these published recall numbers can be read off the paper's curve.
  - Caveat: those predictors are one-layer-ahead (intra-token). The paper's W is inter-token, so the mapping applies only if the paper also sweeps intra-token (layer) lookahead, or states the distinction.
- FlashMoE's "Belady 86% vs LRU 73%" and 2608.12103's "Belady ≈ one third of LRU's remaining misses" are block-count headroom figures. The paper's 16–51% engine-measured value of foresight is the time-domain counterpart and should be positioned against them.

### Gaps
- Not re-fetched in this pass: 2608.12103, 2609.29032, 2608.07911 and 2609.04238. Their quotes come from companion notes in this repo.
- Did not check:
  - "Mixture of Cache-Conditional Experts" (arXiv 2412.00099);
  - SPICE (2608.21240), for which only an EmergentMind summary was seen, with no Belady or oracle content;
  - KV-cache papers such as KVFlow, Marconi and CacheWise (2606.16824), for Belady or oracle usage.
- I found no MoE or KV paper that sweeps oracle lookahead window or recall and reports engine-measured time. Absence in these searches is not proof of absence.

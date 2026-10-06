# Reviewer bar and statistical methods for a performance-bound measurement paper at MLSys (and by Hoefler/SPCL standards)

Context: "Where the Seconds Go" (batch-1 MoE decode on rented RTX 5090 + Ryzen 9950X-class hosts). Round-5 reviews were 5/10 with these complaints: clarity; intervals that cover problem resampling on one machine only; decomposition that depends on attribution order; a bound that mixes probed and datasheet ceilings; a "law" that does not beat a naive model on held-out data; and no realisable mechanism that recovers at least 1/3 of the oracle gain. Each section below ends with inferences tagged **[Fix: ...]** that map findings to those complaints.

---

## Q1. Hoefler & Belli SC15: the 12 rules verbatim (variability, experimental unit, ratios) and SPCL follow-ups

### Takeaway
Hoefler & Belli (SC15) give 12 rules. The ones the reviews are effectively invoking are Rule 5 (CIs for nondeterministic data), Rule 7 (statistically sound comparison, e.g. ANOVA), Rule 9 (document all varying factors and levels), Rule 11 (show upper performance bounds) and Rule 12 (plot as much as needed). Rules 3 and 4 say to summarise costs (seconds), not ratios, and to use the geometric mean only as a fallback. I found no 2022-2025 Hoefler paper that issues new ML-specific benchmarking rules. The ML follow-ups are Deep500 (IPDPS 2019), the ISC'19 reproducibility keynote, and the MLSys 2021 Outstanding Paper "Data Movement Is All You Need".

### Cited Findings
- **Citation:** Torsten Hoefler and Roberto Belli, "Scientific Benchmarking of Parallel Computing Systems: Twelve Ways to Tell the Masses when Reporting Performance Results", Proc. SC15, pp. 73:1–73:12, ISBN 978-1-4503-3723-6 — [unixer.de pub page](https://www.unixer.de/publications/index.php?pub=222); [title confirmed at evaluate.inf.usi.ch](https://evaluate.inf.usi.ch/node/366). The DOI was not shown on the page I fetched. I believe it is 10.1145/2807591.2807644, but that is UNVERIFIED.
- **Basis of the paper:** the authors "examine 120 papers from top conferences" and find practices insufficient for reproducibility — [unixer.de](https://www.unixer.de/publications/index.php?pub=222). The ISC'19 keynote describes this as a stratified random sample of HPDC, PPoPP and SC papers from 2011–2014, with "No statistically significant evidence for improvement over the years" — [Hoefler NRE19 keynote](https://www.cs.fsu.edu/~nre/nre-2019/presentations/01-Hoefler.isc19-nre19-neynote.pdf).
- **The 12 rules, verbatim.** Rules 1–6 and 8–12 come from Hoefler's ICS'17 invited slides. Rule 7 comes from a Uni Hamburg 2020 lecture that reproduces the slide. Both sources list the same 12 rule openings.
  - Rule 1: "When publishing parallel speedup, report if the base case is a single parallel process or best serial execution, as well as the absolute execution performance of the base case." — [Hoefler ICS'17 slides](https://blogs.fau.de/hager/files/2017/02/hoefler-scientific-benchmarking-isc-pe-invited.pdf)
  - Rule 2: "Specify the reason for only reporting subsets of standard benchmarks or applications or not using all system resources." — [slides](https://blogs.fau.de/hager/files/2017/02/hoefler-scientific-benchmarking-isc-pe-invited.pdf)
  - Rule 3: "Use the arithmetic mean only for summarizing costs. Use the harmonic mean for summarizing rates." — [slides](https://blogs.fau.de/hager/files/2017/02/hoefler-scientific-benchmarking-isc-pe-invited.pdf)
  - Rule 4: "Avoid summarizing ratios; summarize the costs or rates that the ratios base on instead. Only if these are not available use the geometric mean for summarizing ratios." — [slides](https://blogs.fau.de/hager/files/2017/02/hoefler-scientific-benchmarking-isc-pe-invited.pdf)
  - Rule 5: "Report if the measurement values are deterministic. For nondeterministic data, report confidence intervals of the measurement." — [slides](https://blogs.fau.de/hager/files/2017/02/hoefler-scientific-benchmarking-isc-pe-invited.pdf)
  - Rule 6: "Do not assume normality of collected data (e.g., based on the number of samples) without diagnostic checking." — [slides](https://blogs.fau.de/hager/files/2017/02/hoefler-scientific-benchmarking-isc-pe-invited.pdf)
  - Rule 7: "Compare nondeterministic data in a statistically sound way, e.g., using non-overlapping confidence intervals or ANOVA." — [Uni Hamburg SIW20 lecture, slide 17](https://wr.informatik.uni-hamburg.de/_media/teaching/sommersemester_2020/siw20-benchmarking-kordt.pdf)
  - Rule 8: "Carefully investigate if measures of central tendency such as mean or median are useful to report. Some problems, such as worst-case latency, may require other percentiles." — [slides](https://blogs.fau.de/hager/files/2017/02/hoefler-scientific-benchmarking-isc-pe-invited.pdf). The keynote version recommends quantile regression for this case — [NRE19 keynote](https://www.cs.fsu.edu/~nre/nre-2019/presentations/01-Hoefler.isc19-nre19-neynote.pdf).
  - Rule 9: "Document all varying factors and their levels as well as the complete experimental setup (e.g., software, hardware, techniques) to facilitate reproducibility and provide interpretability." — [slides](https://blogs.fau.de/hager/files/2017/02/hoefler-scientific-benchmarking-isc-pe-invited.pdf)
  - Rule 10: "For parallel time measurements, report all measurement, (optional) synchronization, and summarization techniques." — [slides](https://blogs.fau.de/hager/files/2017/02/hoefler-scientific-benchmarking-isc-pe-invited.pdf)
  - Rule 11: "If possible, show upper performance bounds to facilitate interpretability of the measured results." — [slides](https://blogs.fau.de/hager/files/2017/02/hoefler-scientific-benchmarking-isc-pe-invited.pdf)
  - Rule 12: "Plot as much information as needed to interpret the experimental results. Only connect measurements by lines if they indicate trends and the interpolation is valid." — [slides](https://blogs.fau.de/hager/files/2017/02/hoefler-scientific-benchmarking-isc-pe-invited.pdf)
- **On variability and stopping rules (slides):**
  - Rank-based, nonparametric measures ("no assumption about distribution") are "Essentially always better than assuming normality."
  - Recommended stopping rule: "Measure until the confidence interval has a certain acceptable width."
  - Example of good practice: "We collected measurements until the 99% confidence interval was within 5% of our reported means."
  - Source: [Hoefler ICS'17 slides](https://blogs.fau.de/hager/files/2017/02/hoefler-scientific-benchmarking-isc-pe-invited.pdf)
- **Comparison methods:** the Hamburg lecture glosses Rule 7 as ANOVA or t-tests that compare the variance between groups with the variance within groups against degrees-of-freedom-based thresholds — [Uni Hamburg lecture](https://wr.informatik.uni-hamburg.de/_media/teaching/sommersemester_2020/siw20-benchmarking-kordt.pdf)
- **SPCL ML follow-up 1 (ISC'19 / NRE2019 keynote, "Performance Reproducibility in HPC and Deep Learning"):**
  - Separates "reproducibility – identical results/conclusions with identical data and method" from "replicability – non-identical but similar results/conclusions with similar data and method".
  - Introduces Deep500 for DL benchmarking.
  - Lists "12 ways to floptimize", e.g. comparing outdated hardware with specialised hardware, or ignoring I/O.
  - Source: [NRE19 keynote](https://www.cs.fsu.edu/~nre/nre-2019/presentations/01-Hoefler.isc19-nre19-neynote.pdf)
- **SPCL ML follow-up 2:** Deep500 (Ben-Nun et al., IPDPS 2019), "A Modular Benchmarking Infrastructure for High-Performance and Reproducible Deep Learning" — [arXiv 1901.10183](https://arxiv.org/abs/1901.10183v1); [unixer.de pub 332](https://www.unixer.de/publications/index.php?pub=332)
- **SPCL ML follow-up 3:** Ivanov, Dryden, Ben-Nun, Li and Hoefler, "Data Movement Is All You Need: A Case Study on Optimizing Transformers", MLSys 2021, **Outstanding Paper Award** — [MLSys 2021 Best Papers](https://mlsys.org/Conferences/2021/BestPapers). The abstract claims:
  - "We find that data movement is the key bottleneck when training. Due to Amdahl's Law and massive improvements in compute performance, training has now become memory-bound."
  - Data movement reduced by up to 22.91%.
  - 1.30× speedup over state-of-the-art frameworks on a BERT encoder layer and 1.19× on full BERT.
  - Source: [arXiv 2007.00072](https://arxiv.org/abs/2007.00072v3)

### Inferences
- **[Fix: clarity / too many numbers per sentence]** Rule 12 ("plot as much information as needed") plus Rule 9 (document factors and levels) support moving dense numeric sentences into one factor/level table and one or two plots. The text would then carry one number per claim and point to the exhibit.
- **[Fix: summarising ratios]** Rules 3 and 4 imply the headline should be in seconds per token (a cost, arithmetic mean) or tokens per second (a rate, harmonic mean). Speedup ratios should be derived from those, and the geometric mean of ratios used only where costs are unavailable. Any geomean-of-speedups across the 6 cells should be replaced or justified.
- **[Fix: experimental unit]** Rule 5 asks for CIs "of the measurement", and Rule 7 asks for statistically sound comparison. Neither rule names the experimental unit. However, Rule 9 requires documenting "all varying factors and their levels", and the rental machine is a varying factor. The reviewers' demand to treat the machine as the statistical unit is consistent with Rules 7 and 9 together with the hierarchical-variance literature in Q3.
- **[Fix: bound]** Rule 11 makes the speed-of-light bound a positive under SPCL norms. The bound must be clearly defined and measured, though, not a mix of probed and datasheet numbers (see Q4: ERT, MoE-Lightning).
- The SPCL exemplar (Ivanov et al. 2021) won by pairing a diagnosis ("data movement is the bottleneck") with a realised optimisation recipe (1.30× / 1.19×). This is the same shape the reviewers ask for with "a realisable mechanism recovering at least 1/3 of the oracle gain".

### Gaps
- I could not fetch the SC15 PDF itself, so the rule wording comes from Hoefler's own invited slides (ICS'17) and a lecture reproduction. They may differ in minor punctuation from the printed paper. The DOI is unverified.
- I found no Hoefler/SPCL 2022–2025 paper or talk that issues new "benchmarking ML systems" rules beyond the 2015 rules, the 2019 keynote and Deep500. "Demystifying Parallel and Distributed Deep Learning" (Ben-Nun & Hoefler, ACM CSUR 2019) is a survey; I did not fetch it, and as far as I know it does not add statistical-reporting rules (unverified).

---

## Q2. Order-independent attribution: Shapley values, factorial designs with interaction terms, and what to do when only a subset of the 2^k states is measured

### Takeaway
A sequential ("one at a time, in one order") decomposition is the greedy ablation path that the algorithm-configuration literature explicitly calls path-dependent. The standard fixes are:
1. **Full 2^k(r) factorial with allocation of variation** (Jain), which reports main effects and interaction terms with CIs.
2. **Shapley decomposition** over all coalitions (Shorrocks 2013 in economics; HyperSHAP, AAAI 2026, for configuration "ablation games"), which is exact-additive and order-free and reports Shapley interaction indices.

When not all states can be measured, the literature uses pair-wise or fractional (Plackett–Burman) sampling with an interaction model (Siegmund et al. FSE 2015), or a validated surrogate model (Biedenkapp et al. AAAI 2017; HyperSHAP's 2ε error bound).

### Cited Findings
- **Path dependence of ablation:** Fawcett & Hoos call ablation "a greedy approach (not unlike forward selection)" that "may produce suboptimal results at any distance except 1 from θsource". Parameter interactions, including conditional ones, make important parameters appear later in the path. Brute-force ablation costs up to |I|·p·(p+1)/2 runs. Citation: C. Fawcett & H. H. Hoos, "Analysing differences between algorithm configurations through ablation", *Journal of Heuristics* 22(4), 2016, DOI 10.1007/s10732-014-9275-9 — [preprint PDF](https://www.cs.ubc.ca/labs/beta/Projects/Ablation/papers/FawcettHoos-joh2016-ablationAnalysis.pdf); [RePEc record](https://ideas.repec.org/a/spr/joheur/v22y2016i4d10.1007_s10732-014-9275-9.html)
- **Shapley-based decomposition (economics canon):** A. Shorrocks, "Decomposition procedures for distributional analysis: a unified framework based on the Shapley value", *Journal of Economic Inequality* 11(1):99–126, March 2013 — [RePEc](https://ideas.repec.org/a/kap/jecinq/v11y2013i1p99-126.html). (The abstract was not available on the page. That this paper's motivation is the order-dependence of sequential elimination, which the Shapley value resolves by averaging over all elimination orders, is my understanding and was not verified in this session.)
- **Shapley with interactions for configuration effects (HyperSHAP):** M. Wever, M. Muschalik, F. Fumagalli & M. Lindauer, "HyperSHAP: Shapley Values and Interactions for Explaining Hyperparameter Optimization", AAAI 2026 — [arXiv 2502.01276](https://arxiv.org/pdf/2502.01276)
  - **Ablation game** (verbatim): "λ∗ ⊕S λ0 := λ∗i, if i ∈ S, λ0i, else, and evaluate its value via νGA(S) := VALu(λ∗ ⊕S λ0, D)". The value of a coalition S is the performance with S's settings switched to the target and all others at the reference.
  - Reports Shapley values, Möbius interactions and Shapley interactions. "Positive values indicat[e] synergy and negative indicat[e] redundancy." Uses the Faithful Shapley Interaction Index (FSII).
  - With a surrogate of error ε, the "approximation error of Shapley values and interactions … is bounded by 2ϵ".
- **Factorial design with interaction terms (Jain):**
  - Model: "y = q0 + qA xA + qB xB + qAB xA xB + e"
  - Allocation of variation: "SST = SSA + SSB + SSAB + SSE", with each effect's share reported as SSj/SST. Worked example: A 78.88%, B 15.40%, AB 4.27%, error 1.45%.
  - Effect CIs: sqi = se/√(2²r) and qi ± t[1−α/2; 2²(r−1)]·sqi. A CI that excludes zero indicates a significant effect.
  - Multiplicative effects should be analysed on log(y), with antilogs read as ratios.
  - Sources: [Jain CSE567 slides, "2^k r Factorial Designs"](https://classes.engineering.wustl.edu/~jain/cse567-15/ftp/k_182kr.pdf); [lecture outline](https://classes.engineering.wustl.edu/~jain/cse567-17/k_182kr.htm). The underlying text is R. Jain, *The Art of Computer Systems Performance Analysis*, Wiley, 1991 (2^k r designs are Ch. 18 per the slide file name "k_182kr"; the chapter number is not verified against the book).
- **Measuring only a subset of the 2^k states:** performance-influence models write performance as "Π(c) = β₀ + Σ φᵢ(c(i)) + Σ Φᵢ..ⱼ(c(i)..c(j))" with explicit interaction terms. Interactions are added hierarchically: "We first start adding only the individual option influences and then add interactions as candidates containing options that have been found already to contribute to performance."
  - Sampling: Option-Wise (linear cost) and Pair-Wise ("ensures all two-way interactions appear without confounding", quadratic cost) for binary options; Plackett–Burman, "a specific type of fractional factorial design", for numeric options.
  - Validation on held-out sets: more than 10,000 random configurations; real-system error of 10–19%, "only slightly above the measurement bias".
  - Citation: N. Siegmund, A. Grebhahn, S. Apel & C. Kästner, "Performance-influence models for highly configurable systems", ESEC/FSE 2015, pp. 284–294 — [PDF](https://www.se.cs.uni-saarland.de/publications/docs/SGA+15.pdf)
- **Surrogate-based ablation:** "we replace expensive algorithm runs with cheap predictions obtained from a model-based surrogate" (random forests trained on previously measured configurations). Provides "uncertainty bounds in the ablation path". Speedups of 33–14,727× with importance rankings comparable to racing-ablation. Citation: A. Biedenkapp, M. Lindauer, K. Eggensperger, F. Hutter, C. Fawcett & H. H. Hoos, "Efficient Parameter Importance Analysis via Ablation with Surrogates", AAAI 2017 — [PDF](https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/17-AAAI-Surrogate-Ablation.pdf)

### Inferences
- **[Fix: order-dependent decomposition]** With k = 3 components (bytes, overlap, rest), the full oracle factorial is only 2³ = 8 states. Measure all 8 on every machine; at batch-1 decode this is cheap.
  - Then report **both** (a) Jain-style effects and interaction terms with CIs (on log time if effects compose multiplicatively, on seconds if additively), and (b) exact Shapley shares, i.e. the average marginal contribution over all 3! = 6 orders. Shapley shares sum exactly to the total gap, which removes the "measured in one order" objection.
  - A standard identity (not verified in-session) is useful here. If the Möbius/Harsanyi dividends m(S) are computed from the 8 states, then φᵢ = Σ_{S∋i} m(S)/|S|. Each Shapley share is the main effect plus half of each two-way interaction plus a third of the three-way interaction. Showing the interaction terms explicitly tells reviewers how much the attribution order mattered.
- If k is larger (for example 4–5 oracle switches), 16–32 states per machine may still be feasible. Otherwise use a resolution-IV or V fractional factorial, or pair-wise sampling à la Siegmund, fit main effects plus two-way interactions, and validate the fitted model on held-out states, reporting the error as Siegmund et al. do.
- Report the per-order decompositions as a robustness exhibit (all 6 orders as a small-multiples bar chart). This follows Rule 12 and makes the order dependence visible rather than hidden.
- Bootstrap the Shapley shares with the hierarchical scheme in Q3 so every share carries a machine-level CI.

### Gaps
- I found no systems or MLSys paper that reports a *measured* Shapley decomposition of a performance gap across optimisation switches. The canonical sources are from economics (Shorrocks) and AutoML (HyperSHAP, Fawcett & Hoos). The paper would be a relatively early systems use; that is a positive, but it needs careful explanation.
- I could not fetch Shorrocks's full text, so its exact wording on path dependence is not quoted.

---

## Q3. Hierarchical bootstrap, mixed-effects models and variance components with machine-level and run-level variance; how many machines?

### Takeaway
The benchmarking literature treats nested levels (machine or VM instance → process or run → iteration) as random effects. The variance of a grand mean is Σ σᵢ²/∏ nₖ, so adding repetitions at a low level (problems) cannot shrink the high-level (machine) term. Ignoring a level gives CIs that are too narrow.

The prescribed methods are:
- **Hierarchical bootstrap:** resample machines first, then units within each machine (Kalibera & Jones; Saravanan et al.).
- **Mixed-effects model** with a variance component per level.
- **Pre-aggregation** to the highest independent unit.

On the number of machines:
- Kalibera & Jones say their method "would not work well" with fewer than about 5 top-level repetitions.
- A 2026 ECE/CS playbook gives n ≥ 10 as a practical floor and n ≥ 30 as comfortable.
- Cloud studies show inter-instance variation is large and is substantially explained by the CPU model, which supports stratifying by CPU class.

### Cited Findings
- **Random-effects model and variance formula (Kalibera & Jones):** "the times measured in a single execution are randomly distributed, with a mean that is also a random variable". Total variance σ² = Σ σᵢ², and var(Ȳ) = Σ σᵢ²/∏ₖ₌ᵢⁿ⁺¹ nₖ (Eq. 10).
  - Optimal per-level repetition counts: n₁ = ⌈√(c₁T₁²/T₂²)⌉ and nᵢ = ⌈√((cᵢ/cᵢ₋₁)Tᵢ²/Tᵢ₊₁²)⌉, where cᵢ is the cost of one repetition at level i.
  - Hierarchical bootstrap: resample top-level units with replacement, then resample lower levels independently within each; use at least 1000 iterations and take the α/2 and 1−α/2 quantiles.
  - Effect-size CIs for ratios of means use Fieller's theorem or a bootstrap. Recommended reporting form: "system A is faster than system B by 5.5% ± 2.5%, with 95% confidence".
  - Top level: "This method would not work well for very small repetition numbers (say, below 5…)".
  - Source: T. Kalibera & R. Jones, "Quantifying Performance Changes with Effect Size Confidence Intervals" — [arXiv 2007.10899](https://arxiv.org/pdf/2007.10899); [Kent repository](https://kar.kent.ac.uk/30809). The fetch tool attributed this to TOPLAS 2013. I believe it is actually a University of Kent technical report (2012), with the condensed peer-reviewed version being "Rigorous Benchmarking in Reasonable Time", ISMM 2013. The venue attribution and the ISMM DOI are UNVERIFIED; the ISMM 2013 paper's existence is attested by [bookmarks.offog.org](https://bookmarks.offog.org/ats/reproducibility).
- **Hierarchical bootstrap vs pooled statistics (Saravanan et al.):** "First, we sample with replacement from the subjects in the group. Then, for the subjects selected, we then sample with replacement from the individual neurons."
  - Traditional pooled tests had false-positive rates above 45% on hierarchical data. The hierarchical bootstrap stayed near 5%, and was conservative on truly independent data (0.66%).
  - It has more power than "summarized metric" approaches. LMMs control Type-I error similarly and are marginally better at very small sizes.
  - The authors prefer the bootstrap because it needs "fewer implementation choices by the user".
  - Caveat: "The bootstrap assumes that the data collected captures essential characteristics of the population distribution." No minimum number of clusters is given.
  - Citation: V. Saravanan, G. J. Berman & S. J. Sober, "Application of the hierarchical bootstrap to multi-level data in neuroscience", *Neurons, Behavior, Data analysis, and Theory* (NBDT), 2020 — [arXiv 2007.07797](https://arxiv.org/pdf/2007.07797); [NBDT](https://nbdt.scholasticahq.com/article/13927-application-of-the-hierarchical-bootstrap-to-multi-level-data-in-neuroscience); [PMC7906290](https://pmc.ncbi.nlm.nih.gov/articles/PMC7906290)
- **Recent ECE/CS playbook (B. Krishnamachari, USC, arXiv 2605.00428, 2026):**
  - "Summarize each independent run by one number … then compute statistics across those run-level summaries."
  - For nested data: "Either pre-aggregate to the highest independent level (the simplest fix), use a clustered or block bootstrap that resamples whole runs/topologies, or fit a hierarchical/mixed-effects model that has variance components for each level."
  - "A practical floor is n ≥ 10 seeds; n ≥ 30 is comfortable; for tail metrics (p95, p99) you generally want substantially more."
  - Bootstrap as default "because it makes the fewest assumptions". "Always lead with the effect size". Benjamini–Hochberg for multiple comparisons.
  - "Pre-commit to the regimes (loads, distributions, sizes) before looking at any results, list them in the paper, and report all of them."
  - Source: [arXiv 2605.00428](https://arxiv.org/pdf/2605.00428). This is a single-author arXiv preprint, not peer-reviewed.
- **Cloud and multi-machine variability evidence:**
  - **Uta et al. (NSDI 2020):** "it can take 70 repetitions or more to achieve 95% confidence intervals within 1% of the measured median". "76% of the properly specified studies use no more than 15 repetitions". Recommend nonparametric statistics, CONFIRM to check CI convergence, and "create a fresh set of VMs for every experiment". Measurement-to-measurement variability reached 33% (HPCCloud) and 114% (Google Cloud). Authors: A. Uta, A. Custura, D. Duplyakin, I. Jimenez, J. Rellermeyer, C. Maltzahn, R. Ricci & A. Iosup, "Is Big Data Performance Reproducible in Modern Cloud Networks?" — [arXiv 1912.09256](https://arxiv.org/pdf/1912.09256); [USENIX page](https://www.usenix.org/conference/nsdi20/presentation/uta)
  - **Maricq et al. (OSDI 2018), "Taming Performance Variability":** about 900,000 measurements from 835 servers over 10 months. They build "a statistical model that can be used to understand how representative an individual server is of the general population" and release tools that recommend experiment parameters. Authors: A. Maricq, D. Duplyakin, I. Jimenez, C. Maltzahn, R. Stutsman & R. Ricci — [USENIX](https://www.usenix.org/node/222562)
  - **Leitner & Cito (ACM TOIT 2016):** "in 63 of 82 configurations … we experienced a cRSD of more than 5%" between instances of the same type. Controlling for the specific CPU model cut CPU-benchmark cRSD from 12.81% to 0.15–5.45% (EC2 m1.small). Within-instance variance for CPU-bound work on non-bursting instances was "very low". Citation: P. Leitner & J. Cito, "Patterns in the Chaos – a Study of Performance Variation and Predictability in Public IaaS Clouds" — [arXiv 1411.2429](https://arxiv.org/pdf/1411.2429)
  - **Duet benchmarking (Bulej et al., ICPE 2020):** running the two compared workloads concurrently on the same VM "avoids bias by equalizing probability of interference". Accuracy improved 2.3–12.5× (mean 5.03×) on ScalaBench/DaCapo and 23.8–82.4× (mean 37.4×) on SPEC CPU 2017. Uses a nonparametric percentile bootstrap of the geometric mean of paired ratios. Citation: L. Bulej, V. Horký, P. Tůma, F. Farquet & A. Prokopec, DOI 10.1145/3358960.3379132 — [arXiv 2001.05811](https://arxiv.org/pdf/2001.05811)
  - **Bouthillier et al. (MLSys 2021), "Accounting for Variance in Machine Learning Benchmarks":** "As many sources of variation as possible should be randomized whenever possible". Ignoring variance sources gives about 90% false negatives in their setting. Proposes the probability of improvement P(A>B) ≥ γ with γ = 0.75 — [arXiv 2103.03098](https://arxiv.org/pdf/2103.03098); [MLSys proceedings](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html)

### Inferences
- **[Fix: intervals cover only problem resampling]** By Kalibera's var(Ȳ) formula, if between-machine differences are 5–10× the within-machine spread, the machine variance component dominates. More problems on one host cannot fix this, and current CIs are anti-conservative for any claim about the hardware class.
  - Design: ≥ 5 rentals per CPU class (Kalibera's floor). Aim for ~10 if budget allows (Krishnamachari's floor). Report the absolute number of distinct physical hosts.
- **Crossed design.** The same 30 problems run on every machine, so problems are *crossed* with machines, not nested in them. Two options:
  - A two-way (machine × problem) bootstrap that resamples machines and problems independently. I believe this is Owen's "pigeonhole bootstrap", *Ann. Appl. Stat.* 2007, but that is UNVERIFIED.
  - A linear mixed model, y ~ config × cell + CPU-class (fixed) + (1 | machine) + (1 | problem) + (1 | machine:config). It directly reports the **variance components** the reviewers asked for.
- **Absolute levels vs paired effects.** What matters for decomposition shares and oracle gains is the machine × configuration interaction variance, not the machine main effect, because configurations are paired within a machine (as in Duet benchmarking). Report two kinds of CI:
  - (a) for absolute seconds per token, wide and machine-dominated;
  - (b) for within-machine paired effects (oracle gains, Shapley shares), possibly much tighter.
  This could turn the reviewers' 5–10× concern into a finding: "absolute level varies by host; the decomposition is stable across hosts".
- **Small top-level n.** With only 5 machines, a top-level percentile bootstrap has few distinct resamples and under-covers (a general small-sample bootstrap property, not verified in-session). Report a t-based CI on machine-level means with (m−1) degrees of freedom, or a REML mixed-model CI, alongside the bootstrap, and state which is primary.
- **Stratify by CPU class.** Leitner & Cito show that the CPU model explains much of the inter-instance spread. Treat CPU class as a fixed effect and machine within class as random.
- **Multiplicity in 565 pre-registered clauses.** Use Benjamini–Hochberg (Krishnamachari), and lead with effect sizes plus CIs rather than hit counts.

### Gaps
- I could not fetch Maricq et al.'s full PDF (redirect and provenance block), so their concrete "how many servers" recommendations are not quoted.
- I found no source giving a GPU-specific minimum number of hosts for consumer-GPU rentals. The ≥5 / ≥10 numbers are borrowed from general benchmarking and experimental-design sources.
- The peer-reviewed venue for Kalibera & Jones's effect-size report (tech report vs TOPLAS vs ISMM 2013) is not verified.

---

## Q4. Exemplar MLSys / OSDI / SC / ISPASS measurement and bound papers: what made them succeed, and their structure

### Takeaway
The measurement-flavoured MLSys papers that won awards pair a simple, explicit analytical model or bound with a realised system result:
- **Ivanov et al. 2021 (Outstanding Paper):** data-movement bound, then a 1.30× speedup.
- **Pope et al. 2023 (Outstanding Paper):** a "simple analytical model" for partitioning, then a new latency/MFU Pareto frontier.

Pure-characterisation papers that succeeded (Blalock et al. 2020) did so through a broad, systematic survey, a concrete checklist and a released tool (ShrinkBench).

Recent MoE/LLM bound papers present rooflines whose ceilings are *profiled* (MoE-Lightning's Hierarchical Roofline) or explicitly analytic, not a mix. The Empirical Roofline Tool's documentation explains why datasheet peaks are not achievable ceilings.

### Cited Findings
- **Blalock, Gonzalez Ortiz, Frankle & Guttag, "What is the State of Neural Network Pruning?", MLSys 2020:**
  - Survey of 81 papers. "Over 1/4" compared to no other pruning method, and about 1/2 compared to at most one.
  - Section structure: Introduction → Overview → Lessons from Literature → Missing Controlled Comparisons → Further Barriers → Summary and Recommendations → ShrinkBench → Conclusion.
  - 18 figures and 1 table, about 20 pages including appendices A–D (appendix B is a checklist).
  - Checklist item: "Data includes clearly defined error bars and a measure of central tendency (e.g., mean) and variation (e.g., standard deviation)."
  - Sources: [arXiv 2003.03033](https://arxiv.org/pdf/2003.03033); [MLSys proceedings](https://proceedings.mlsys.org/paper_files/paper/2020/hash/6c44dc73014d66ba49b28d483a8f8b0d-Abstract.html)
- **Pope et al., "Efficiently Scaling Transformer Inference", MLSys 2023 Outstanding Paper Award:** "We develop a simple analytical model for inference efficiency to select the best multi-dimensional partitioning techniques…". It reports 29 ms/token low-batch latency (int8) and 76% MFU on PaLM 540B — [MLSys 2023 poster page](https://mlsys.org/virtual/2023/poster/2463); [proceedings](https://proceedings.mlsys.org/paper_files/paper/2023/hash/c4be71ab8d24cdfb45e3d06dbfca2780-Abstract-mlsys2023.html)
- **Ivanov et al., MLSys 2021 Outstanding Paper:** diagnosis that training "has now become memory-bound", then a recipe that reduces data movement by up to 22.91%, giving 1.30× (encoder layer) and 1.19× (BERT) — [arXiv 2007.00072](https://arxiv.org/abs/2007.00072v3); [MLSys 2021 Best Papers](https://mlsys.org/Conferences/2021/BestPapers)
- **MoE-Lightning (Cao, Schafhalter, Gonzalez, Liu, Liu, Zaharia, Griggs, Sheng & Stoica, 2024):** a Hierarchical Roofline Model for CPU–GPU MoE offloading. It has compute roofs, local memory roofs and cross-level (CPU↔GPU) roofs: "P_x^i = min(P_peak^i, B_peak^i × I_x^i, B_peak^{j,i} × I_x^j)". Policies come from "theoretically calculated computation flops and bytes with profiled peak performance and memory bandwidth for the hardware". HRM figures for Mixtral attention and FFN mark turning and balance points — [arXiv 2411.11217](https://arxiv.org/html/2411.11217v1). The venue (ASPLOS 2025) is my belief, UNVERIFIED in-session.
- **Empirical vs datasheet ceilings (LBNL Empirical Roofline Tool):** "Even if the machine characteristics can be estimated, these are theoretical maximums and there may exist no code that can achieve them." ERT measures bandwidth at each memory level and peak FLOP rates empirically — [ERT page](https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/software/ert)
- **Recent LLM decode characterisation (arXiv 2605.30571; title and authors not retrieved):**
  - Uses peak HBM bandwidth and an "analytic memory floor".
  - Spans four GPUs (H100 SXM5, A100-80GB, L40S, L4) and 44 cells.
  - Isolates kernel-launch overhead with CUDA Graphs and concludes the memory-bandwidth account is "true but incomplete".
  - Source: [arXiv 2605.30571](https://arxiv.org/abs/2605.30571). Metadata is unverified, and it is a direct comparator for the "where the seconds go" framing on datacenter GPUs.
- **Bouthillier et al., MLSys 2021:** a methodology paper on variance in ML benchmarks accepted at MLSys. This shows MLSys accepts papers whose contribution is statistical rigour in measurement — [MLSys proceedings](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html)

### Inferences
- **[Fix: no realisable mechanism recovering ≥ 1/3 of the oracle gain]** Every award-winning exemplar found turns its bound or diagnosis into a realised gain (Ivanov 1.30×, Pope's Pareto frontier). A pure "the oracle says X is available" paper is closer to Blalock-style work, which succeeded only with a field-wide survey, a checklist and a tool. The highest-leverage fix is one concrete, deployable mechanism that recovers a stated fraction of the oracle gain, with a CI across machines.
- **[Fix: bound mixes probed and datasheet ceilings]** Follow ERT and MoE-Lightning:
  - Measure every ceiling with microbenchmarks on each rented host: GPU DRAM bandwidth, PCIe H2D, CPU DRAM bandwidth, and CPU GEMV throughput.
  - Report the per-host ceiling distribution.
  - Show the datasheet number only as a reference line, labelled separately. If a datasheet number must enter the bound, state which term and why.
  - A hierarchical-roofline plot (GPU-HBM, PCIe and CPU-DRAM roofs) is a recognised visual for exactly this CPU+GPU MoE setting.
- **[Fix: calibrated "law" does not beat a naive model held-out]** Siegmund et al. (Q2) show the expected standard: fit on training configurations, report error on held-out configurations against the measurement-noise floor. If the law cannot beat the naive model out of sample:
  - demote it to a descriptive fit, or
  - report held-out error for both, with the machine as the held-out unit (leave-one-machine-out), and claim only where it wins.
- **[Fix: clarity]** The Blalock structure (lessons → controlled comparisons → recommendations → tool) shows that a measurement paper can carry many exhibits if each exhibit answers one named question. A practical rule: one headline exhibit per research question, with numbers in tables or figures rather than in sentences. Coined terms should be cut to a few, each defined in one glossary table on first use.

### Gaps
- Exhibit counts and page structure were verified only for Blalock et al. (18 figures, 1 table, about 20 pages including appendices). I did not verify counts for Ivanov 2021, Pope 2023 or MoE-Lightning.
- I did not retrieve MLPerf Inference (Reddi et al., ISCA 2020) or the original Roofline paper (Williams, Waterman & Patterson, CACM 2009) in this session, so neither is cited here.
- I found no MLSys reviewer guideline that specifically defines a "measurement/characterisation" paper bar.

---

## Q5. MLSys 2027 call for papers: dates, page limit, artifact evaluation, tracks and criteria

### Takeaway
The official mlsys.org dates page shows a paper deadline of **30 Oct 2026, 12:00 PM PDT** (24 days after today, 2026-10-06), reviews on 18 Jan 2027, author responses due 21 Jan 2027 and notifications on 28 Feb 2027. The conference runs 21–25 June 2027. A third-party aggregator gives the location as Bellevue, WA, 10 pages plus unlimited references and appendices, and Research and Industrial tracks only.

The criteria are "novelty, quality, interest, and impact". Artifact evaluation is voluntary and does not affect acceptance (2026 rules). There is no dedicated measurement or benchmark track, but "ML benchmarks, datasets, and tooling" is a listed topic.

### Cited Findings
- **Official 2027 dates** (verbatim from mlsys.org "current/Dates", which the page identifies as 2027): submissions open "Oct 10 '26 12:00 PM PDT"; paper submission deadline "Oct 30 '26 12:00 PM PDT"; author reviews available "Jan 18 '27 12:00 PM PST"; author responses due "Jan 21 '27 12:00 PM PST"; author notifications "Feb 28 '27 01:00 PM PST". Young Professionals Symposium Mon June 21; main conference Tue June 22–Thu June 24; Industry Day Fri June 25 — [mlsys.org Dates](https://mlsys.org/Conferences/current/Dates)
- **Conflicting deadline time:** mldeadlines lists the 2027 deadline as "30 October 2026, 20:00:00 UTC" — [mldeadlines](https://mldeadlines.com/conference/mlsys/). 12:00 PDT is 19:00 UTC (US daylight time ends 1 Nov 2026), so the aggregator is one hour later than the official page. Treat the official page as authoritative.
- **Location, page limit and chairs (third-party):**
  - "June 21–25, 2027 · Bellevue, Washington"; "10-page papers plus unlimited references and appendices".
  - Two OpenReview routes: Research Track ("novelty, quality, interest, impact") and Industrial Track ("impact, lessons and experiences from building real-world, large-scale ML systems").
  - General Chair Joseph Gonzalez; Program Chairs Rashmi Vinayak and Song Han.
  - Source: [BERI event page](https://www.beri.net/events/mlsys-2027). This is a third-party page; I could not fetch the official 2027 CFP page because of tool provenance limits.
- **2026 CFP rules (most recent official CFP text fetched):**
  - Main text "up to 10 pages long, not including references".
  - Appendices allowed, but "reviewers are not required to read these".
  - Double-blind, with "authors are required to make a good-faith effort to anonymize their submissions" and "Institutional affiliations must also be anonymized".
  - Voluntary artifact evaluation under the "ACM Artifact Review and Badging policy", which "will not influence the final decision regarding the papers".
  - Author response period; Research and Industrial (new in 2026) tracks; topics include "ML benchmarks, datasets, and tooling".
  - Source: [MLSys 2026 CFP](https://mlsys.org/Conferences/2026/CallForPapers)

### Inferences
- With 24 days to the deadline, the methodological fixes need to be scoped: the machine-level resampling (≥ 5 rentals per CPU class), the full 2³ oracle factorial per machine, and re-probed ceilings. All three are run-time-bounded rather than engineering-bounded, so they are plausible within the window if rentals start immediately. A realisable mechanism recovering at least 1/3 of the oracle gain is the riskiest item.
- Reviewers are not required to read appendices, so the machine-level CIs, the Shapley/factorial exhibit and the measured-ceiling bound must sit in the 10-page main text. Pre-registration scoring details, per-order decompositions and per-host ceiling tables can go in the appendix.
- Artifact evaluation does not affect acceptance, but a released harness plus raw per-machine data supports Rule 9 and strengthens the reproducibility story for a measurement paper.
- Double-blind: the "Independent Researcher" affiliation must be anonymised in the submission.

### Gaps
- The official MLSys 2027 CFP text (page limit, template year, artifact evaluation schedule, any change to review criteria or a new track) was not fetched directly. The 2027 page limit and location rest on a third-party page plus the 2026 CFP.
- The camera-ready date and the artifact-evaluation dates for 2027 were not listed on the official dates page.

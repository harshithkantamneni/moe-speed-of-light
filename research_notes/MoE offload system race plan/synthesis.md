## Research Synthesis: Option C, racing our MoE offload system on one rented machine

**Method:** Structured synthesis of six desk-research notes (code reads, docs, papers, public APIs), plus three spot checks in code | **Participants:** 6 source notes (prevalence counted as "N of 6 notes")
**Date:** notes and synthesis dated 28 Sep 2026 | **Researcher:** synthesis sub-agent for Harshith Kantamneni (independent researcher, cloud-only)

**Source key.** BASE = `our_system_baseline.md`, ENT = `race_entrants.md`, ECO = `llamacpp_ecosystem.md`, HEAD = `headroom_techniques.md`, FAIR = `fair_race_methodology.md`, MACH = `race_machine.md` (all in this folder). NOV = `reports/MoE hybrid decode novelty check.md`; PAPER = `paper/paper.tex`; CHK1–3 = checks run for this synthesis (see Methodology Notes). § and Q follow each note's own numbering. **"My estimate"** marks arithmetic of mine on note-sourced inputs. None of it is calibrated; it sets expectations and gate thresholds only.

### Executive Summary

Option C is worth running only as a pre-registered race whose headline is where each system sits against the speed-of-light at equal GPU memory. No mechanism in our system is new: FreeToken has shipped a GPU-signalled, polled CPU hand-off since 11 Aug 2026, which I confirmed in its source. So a "better system" claim can rest only on measured speed.

Race on a whole-machine Vast.ai RTX 5090 + Ryzen 9 9950X/9950X3D desktop with ≥126 GB RAM, not on the shared Verda slice, for four reasons:
- It is fairer: no co-tenants on the memory bus.
- It costs 2–5× less per hour than the Verda options.
- It engages KTransformers' AVX-512 BF16 path and our AVX-512 helper kernel.
- It is the hardware the accessible-AI mission is about.

On that desktop, DRAM delivers roughly 50–70 GB/s and PCIe 5 about 49–52 GB/s.
- **Against tuned llama.cpp** we should win widely on gpt-oss-120b.
- **Against FreeToken** we will likely lose or at best tie. When PCIe is about as fast as DRAM, fetching an expert costs about as much as running it on the CPU, and FreeToken's pooled cache and hybrid split exploit that.

A Day-4 gate decides whether the one-week sprint chases FreeToken or only the llama.cpp family. The gate runs ours vs the best llama.cpp-family build vs FreeToken, at three budgets with three launches each.

First, fix the confirmed LM-head omission in the case-study speed-of-light. It understates every fraction in the paper: ours goes from 47–64% to about 54–76%, and llama.cpp from 33–45% to about 37–53%.

### Coordinator findings, checked against the notes

| # | Finding | Verdict | Evidence and additions |
|---|---|---|---|
| 1 | The case-study SoL omits the LM head | **Confirmed** | **Sources:** BASE §1, §4.1, §7.<br>**CHK1:** `scripts/sol_a10.py` passes D = 687,012,096 B for gpt-oss-20b. That equals `dense_bytes` in `data/gguf_bytes.json`, whose `head_bytes` (615,329,280 B) is left out. By contrast, `mosl/perfmodel.py` `Workload.dense_bytes` adds `lm_head_params` and `scripts/preregister.py` adds `head_bytes`, so the audit's physical bound is unaffected. CHK1 also found that the saved 038 SoL used `--bc 166.1` (I6's measured read), so host bandwidth is not a second error.<br>**Two corrections exist:**<br>- bytes + head: SoL 161.6 / 183.2 tok/s (20b / Q4_K_M);<br>- the Nsight split: 156.4 / 163.7 tok/s.<br>BASE's 54–76% / 37–53% use the Nsight split.<br>**Inconsistency:** BASE gives 54–76% in §1 and 54–74% in §7. Regenerate the fractions by script rather than copying them by hand. |
| 2 | Rebase is cheap (base 2145525a, 22 commits behind) | **Confirmed** | `git apply --check` is clean on 4da6337 (27 Sep), with offsets only; not compiled (BASE §1, §2.3; ECO Q1).<br>Apply **only** the expert-cache patch: it already contains the overlap hunks, so applying the sched patch after it fails (ECO Q1).<br>CHK3: the "llama.cpp 4d86b2f … master" that HEAD and FAIR read is the local `expert-cache` branch head (2145525a + 16 patch commits, "ec-bench: --no-mmap"), not upstream master. |
| 3 | FreeToken already ships a GPU-signalled, polled hand-off | **Confirmed** | ENT Q2, Q5.<br>**CHK2** (FreeToken 0d652e7, `moe/cpu_executor.py`, `layers/moe.py`): the GPU raises a mapped-pinned "ready" flag with `cuStreamWriteValue64`, a CPU coordinator polls it, and the GPU waits on "done" with `cuStreamWaitValue64`.<br>**Remaining implementation differences:**<br>- **Hand-off per layer.** FreeToken submits and waits on *every* MoE layer, with −1 ids skipped inside the CPU kernel. Our GPU skips the hand-off when no expert of the layer runs on the CPU.<br>- **Data movement.** FreeToken moves activations and results with D2H/H2D copies on the copy engine. We use kernel stores and loads on pinned memory.<br>- **Waiting.** FreeToken waits without occupying an SM. We use a 4-block spin kernel (BASE §2.1). |
| 4 | Lead over llama.cpp shrinks on fast hosts; fixed costs ≈ half the excess | **Confirmed, with a scope caveat** | **Sources:** BASE §1, §5.4.<br>**Fixed costs:** 53% of the excess in the waterfall (O 37% + helper per-request 16%), or 46% in the component view (O 37% + 9%) (BASE §1, §4.3).<br>**Scope:** the scenarios are 150/250/400 GB/s server hosts. The recommended desktop (≈50–70 GB/s, MACH §2; FreeToken's 9950X3D measured B_H 53.8 GB/s, FAIR Q3) lies outside them. The 400 GB/s rows also assume ≥40–48 fast cores (BASE §5.4). |
| 5 | Verda slice vs Vast desktop | **Resolved for MACH** (D1) | ENT's own conditional plan is to "rent a Vast 5090 on a Zen 4/5 or Sapphire Rapids+ host" (ENT Q7). |
| 6 | KTransformers can run on a Ryzen 9950X | **Supported, but risky** | **Requirement:** the BF16 backend needs AVX512F+BW+BF16, "AMD Zen 4+" (ENT Q2).<br>**Known crash:** issue #1754 segfaulted on *exactly* a Ryzen 9950X3D + RTX 5090 single-NUMA desktop. It was closed as user-resolved with a local `worker_pool.h` patch. #1891 and #1929 later changed NUMA handling; whether v0.7.1 is fixed is unverified (ENT Q2).<br>**Other limits:** no gpt-oss support, so KT races Qwen3-30B BF16 only (ENT Q3). It needs a source build for sm_120 and pins torch 2.9.1 (ENT Q6). |
| 7 | A go/no-go gate before the sprint | **Adopted** | Thresholds in D6. |
| 8 | Flag changes: `--no-mmap` → `--load-mode none`; `--fit off` | **Confirmed, plus four more traps** | **Confirmed:** #28334 (9 Sep) deprecated `--no-mmap`; use `-lm/--load-mode none`, or `llama-bench --mmap 0`. `--fit` is on by default with a 1 GiB margin and min ctx 4096 (ECO Q1, Q4; FAIR Q1).<br>**Four more traps:**<br>- Whether `llama-bench` applies `--fit` is unverified (FAIR Q1 gaps).<br>- ik's `-ncmoe` now offloads the *last* N layers (ECO Q3).<br>- leloch v2's `auto` mode refuses a single GPU (ECO Q2).<br>- Shipped #27861 can silently no-op (ENT Q1, Q6). |

### Key Themes

#### Theme 1: Host memory bandwidth decides the race, and hosts differ a lot
**Prevalence:** 6 of 6 notes

**Summary:** Every note treats host DRAM bandwidth as the first-order variable, and in particular how much of it a given host actually delivers. It sets our margin over llama.cpp, the noise level, and which design wins.

**Supporting Evidence:**
- "Our advantage then depends mainly on host bandwidth" — BASE §5.4
- "Pinned vCPUs isolate compute, not memory bandwidth." — MACH §1
- "Effective read bandwidth may be only ~50–70 GB/s" (4-DIMM AM5/LGA1851 desktops) — MACH §2
- "cloud VMs expose just 46–59% of datasheet in STREAM" — HEAD Q2
- "the most common culprit for MoE TG underperformance" (XMP/EXPO disabled) — ECO Q4
- "The only AMD datapoints on a 5090 are from dual-socket Zen 5 EPYC 9355 hosts." — ENT Q2

FAIR Q1 adds a requirement: record STREAM at the thread counts used.

**Implication:**
- Measure read bandwidth and pinned H2D on the race host first, and print them next to every number.
- Rent a host with no co-tenants.
- Treat BASE's 150–400 GB/s projections as not covering a desktop.

#### Theme 2: What is left to win is fixed cost per layer and per call, not bytes
**Prevalence:** 4 of 6 notes (BASE, HEAD, ENT, ECO)

**Summary:** Two fixed costs were about half of the A10 excess over the bound:
- **O**, a per-cached-layer overhead of 26–38 µs;
- **f**, the helpers' per-request cost of 18–50 µs.

Such constants weigh more on faster GPUs and less on slow hosts.

**Supporting Evidence:**
- "It is paid on every cached layer" (O) — BASE §1
- "These costs are constant in time, so they take a larger share on faster GPUs." — HEAD Q1
- "Stock fusions probably won't fire on the patched graph." — ECO Q1
- FreeToken's host-func path "pays ~30-50us of callback dispatch latency per call" — ENT Q2

**Implication:** O and f are the levers we can act on. O has never been profiled ("the 012 nsys files are empty", BASE §4.2), so the sprint starts with nsys. On a slow-DRAM desktop their share shrinks, because CPU miss work dominates the step.

#### Theme 3: The mechanism is not new; the defensible claim is measured standing on equal footing
**Prevalence:** 4 of 6 notes (ENT, FAIR, HEAD, ECO), with one dissent inside ECO

**Summary:** The prior art already covers the pieces:
- FreeToken's hand-off predates our patch.
- Zero-copy loads, prefetch-pause flags and fused fills exist in SeqMoE and FreeToken (HEAD Q5).

What nobody has run is a third-party, equal-VRAM race scored against a bound.

**Supporting Evidence:**
- "That handshake has been public since 11 Aug 2026, before the user's patch was committed" — ENT Q5
- "The paper should cite FreeToken and differentiate on measured hand-off latency, not claim the mechanism." — ENT Q5
- "No named paper has all five." — FAIR Q3. The five are: one host; measured-VRAM equalisation with components enumerated; bit-aligned weights; harmonic summaries with CIs; a speed-of-light score.
- Dissent: "The overlap plus the GPU-signalled mailbox is the novel systems contribution" — ECO Q5 (rejected, see C1)

**Implication:** The deliverable is the protocol plus the scored table; our system is one entrant among several.

#### Theme 4: Baselines are easy to handicap by accident
**Prevalence:** 4 of 6 notes (FAIR, ECO, ENT, BASE)

**Summary:** Several current tool defaults silently change placement or memory: auto-fit, SGLang's memory fraction, the load mode, ik's layer order, leloch's auto mode and #27861's init guard. Our own A10 baseline was ec-bench's `--ncmoe` inside the patched build, not stock llama-bench.

**Supporting Evidence:**
- "Two tool defaults on current stacks silently break "equal GPU memory" unless overridden" — FAIR Q1
- "The PR #27861 cache as shipped is a strawman." — ENT Q5
- "Blocker: the silent no-op bug in shipped #27861" — ENT Q6
- On the A10, `--no-mmap` alone moved llama.cpp by −13.4% to +1.7%, and the baseline was never checked against stock tools — BASE §7

**Implication:**
- Tune the baselines harder than our own system.
- Log every auto-sizer value.
- Add "as-published" baseline rows next to the tuned rows (FAIR Q3).

#### Theme 5: FreeToken is the entrant to beat, and the fair race machine is its home ground
**Prevalence:** 4 of 6 notes (ENT, HEAD, FAIR, MACH)

**Supporting Evidence:**
- "FreeToken is the entrant most likely to beat both llama.cpp and the user's system." — ENT Q5
- FreeToken's own "RTX 5090 desktop" testbed is a Ryzen 9 9950X3D with B_P 49.0 and B_H 53.8 GB/s — FAIR Q3 table
- A community run served gpt-oss-120b at 127.1 tok/s with 40.4% of experts resident in ~29 GB on a 5090, fetch-only `offload` — ENT Q5
- FreeToken keeps cache control on the GPU and does one fused fill per step — HEAD Q3, Q5

**Implication:** On this host PCIe ≈ DRAM, so the reuse break-even r* = p/(b−a) is about 1.1–1.3.
- That is my estimate, using BASE §5.4's formula with B_P 49–52 and B_C 54–60 GB/s.
- For comparison, r* is about 11 on the A10 and about 3 at 150 GB/s (BASE §5.4).
- A fetched expert pays for itself after about one reuse, so fetch plus a pooled cache is structurally favoured here.
- Our CPU-miss design is favoured on server hosts.
- Plan for a loss or a tie against FreeToken on 120b.

#### Theme 6: "Lossless" must be shown against a measured noise floor, not by identical tokens
**Prevalence:** 4 of 6 notes (FAIR, ECO, BASE, ENT)

**Supporting Evidence:**
- Identical greedy output "is the wrong pass criterion for cross-engine or cross-placement comparisons in BF16" — FAIR Q2
- #25952: "Results are not claimed bit-identical" — ECO Q1
- A10 fidelity, cache vs llama.cpp: top-1 agreement 0.981–0.993, ΔNLL −0.99% to +1.06%. gpt-oss-120b on arm D reaches only 0.890–0.897 because that text is off-distribution — BASE §3.3
- No raceable system is bit-identical to llama.cpp, and KT's Expert Deferral is lossy — ENT Q3

**Implication:** Run FAIR's verification steps V0–V2:
- V0: weight hashes.
- V1: teacher-forced KLD and top-1 agreement against a noise floor.
- V2: descriptive free-running greedy match.

Pre-register the thresholds, and use own-text (S) prompts for 120b.

#### Theme 7: Blackwell porting has sharp edges for every entrant, ours included
**Prevalence:** 4 of 6 notes (ECO, ENT, BASE, MACH)

**Supporting Evidence:**
- Any single-token `MUL_MAT_ID` kernel other than mmvq on sm_120 "would index out of bounds on −1" — BASE §2.3
- nvcc 13.2 miscompiles IQ1_S/IQ2_S/IQ3_S on sm_120, and CUDA 13.1 MMQ segfaults were reported. Build `120a` with CUDA 12.8/12.9 — ECO Q1, Q4
- KT's wheels cover SM 80/86/89/90 only; FreeToken needs driver r580+ and a CUDA 13 `nvcc`; KT #2081 is an FP8 assert on SM_120 — ENT Q2
- "Record, don't lock, GPU clocks" (Vast Docker) — MACH §2

**Implication:**
- Budget days for builds.
- Give each system its own venv.
- Run `test-backend-ops` on the rented card.
- Patch our BF16 path before it can race.

#### Theme 8: The A10 headline sits inside baseline noise; repeats are mandatory
**Prevalence:** 3 of 6 notes (BASE, FAIR, MACH)

**Supporting Evidence:**
- "The phase-4.5 jobs have **no repeats**." — BASE §3.4. llama.cpp moved −7.8% to +1.6% between instances, and the 1.66× top of the paper's range comes from a slow llama.cpp run.
- "On cloud VMs, launch-to-launch variance is plausibly the dominant term." — FAIR Q1
- Interleave the systems (A B C D …) with a bandwidth probe and a clock log before each block — MACH §2

**Implication:**
- Run ≥3 launches × ≥5 requests per cell.
- Summarise rates as ΣN/ΣT, with bootstrap CIs.
- Report the A10 range from medians; instance I5 alone gives 1.37–1.60× (BASE §3.4).

### Contradictions between notes, and how I resolve them

| # | Conflict | Resolution |
|---|---|---|
| C1 | **Novelty.** ECO Q5 calls "the overlap plus the GPU-signalled mailbox" the novel contribution. NOV rates the mailbox "partly new" and its GPU-decided skip "new". ENT Q2/Q5 finds FreeToken's handshake public since 11 Aug 2026. | **ENT is right**, and CHK2 confirms it in the source.<br>The overlap patch is not even part of the raced system: it "only matters for split-graph mode, not the mailbox" (BASE §2.1 item 9; "moot in mailbox mode", BASE §2.2). ECO Q5 itself quotes the draft PR: "Stock `--n-cpu-moe` does not benefit."<br>The per-layer skip, kernel-driven transfers and spin-vs-memop wait are implementation differences to measure, not claims to make. |
| C2 | **Race machine.** ENT assumes a Verda 1× RTX PRO 6000 (30 threads, 90 GB). MACH rejects it: it is 1/8 of an 8-GPU host whose CPU, DRAM and NUMA layout are unpublished. MACH recommends a Vast whole-machine 5090 desktop instead. | **MACH** (see D1). ENT's own fallback also points to Vast Zen 4/5 hosts. |
| C3 | **RAM.** MACH wants ≥180 GB "for 130 GB models". BASE §5.5 fits every listed model in one engine within 90 GB (120b pins 61.1 GB). ENT Q2 calls Qwen3.6 BF16 (~70 GB) tight at 90 GB. | No chosen model exceeds ~70 GB, so **≥126 GB suffices**. Above that, prefer measured bandwidth over capacity, because 4-DIMM boards derate DDR5 (MACH §2). |
| C4 | **System constants.**<br>HEAD models the system with the GPU at 38% and the CPU at 60% of peak, 21 µs per CPU expert and an 84 µs/layer hand-off. From these it predicts a fall to 39–46% of SoL and ranks "one job per layer for all misses" first.<br>BASE measures something different: the GPU expert GEMV at η_v 0.73; helpers at 78–81% of measured read; f per *request* (50/19/25 µs), already one request per layer in four chunked phases (since job 018); O 26–38 µs/layer. The 84 µs was the phase-2 split-graph figure (BASE §4.4, job 012). | **Use BASE.** HEAD's constants are the third-party M4 fit (η_g 0.382, η_c 0.599, τ_e 20.7 µs; BASE §5.3) plus the old split-graph cache.<br>HEAD's direction survives, with two changes:<br>- its lever 1 becomes "cut f's four barrier phases";<br>- its lever 2 is ≈0 on the A10 by HEAD's own rule ("≈0 if already ~80%"), to be rechecked on Zen 5. |
| C5 | **Fusions.** HEAD Q6 says "It is unknown whether the system's branch already includes upstream's Nov-2025 CUDA fusions." ECO Q1 dates the base 2145525a to 26 Sep 2026. | Stock layers do have them. Cached layers go through `build_moe_ffn_ec` (BASE §2.3) and probably miss #25952/#28432 (ECO Q1). HEAD's lever 3 becomes "make the fusions fire on cached layers". |
| C6 | **Master hash.** HEAD reads "llama.cpp 4d86b2f" and FAIR "commit 4d86b2f5 (master …)"; ECO and BASE use master 4da6337. | **CHK3:** 4d86b2f is the local expert-cache branch head. The flag facts HEAD and FAIR took from it predate 2145525a (#28334 landed 9 Sep), so they hold. Re-verify them on the frozen race base 4da6337. |
| C7 | **Figures that moved.** BASE gives 54–76% in §1 and 54–74% in §7. The paper gives 47–64% and 1.31–1.66×; NOV gives 46–63% and 1.41–1.81×. | NOV used the older, capped windows (PAPER, Pitfalls: "Prefill caps put decode steps in the prompt"). Use the paper's numbers with the head correction regenerated by script, and report medians across instances. |
| C8 | **KT host.** FAIR Q3 says run KT "on an AMX host"; ENT Q2/Q7 and NOV say AVX-512 BF16 is enough. | For batch-1 decode, KT's own paper prefers AVX-512 ("For decode … a lightweight AVX-512 kernel beats AMX", HEAD Q2). AMX matters for prefill and for the lossy AMXINT paths. A Zen 5 host is therefore fair for a decode race; say so in the paper. NUMA-aware tensor parallelism (up to 1.63×) needs two sockets, so it is moot on a single-NUMA desktop (ENT Q2). |
| C9 | **Projection regime.** BASE §5.4 projects at 150–400 GB/s; MACH's host delivers ~50–70 GB/s. | Not a data conflict: the projections simply do not cover the race host. Rerun `project.py` after the calibration hour (BASE §5.4 list) and pre-register its output. |
| C10 | **Hoefler survey.** FAIR's session summary says 17 of 95 papers reported variation; the prior notes say 15 of 95 (FAIR Q1). | Read Table 1 of the SC'15 PDF before quoting either number to SPCL. |
| C11 | **Race workload.** ENT Q6: "One working day … covers two models × 4–5 budgets × 5 entrants". FAIR Q1: ≥3 launches × ≥5 requests and ≥3 budgets. | **Compatible if the matrix is cut** to 2 headline models × 3 budgets, KT on one model only, and all three FreeToken modes at the middle budget only. My estimate: about 40 cells × ~8 min ≈ 5–6 h, plus calibration, thread sweeps and VRAM-fit search, for 8–10 h in total. That fits MACH §4's 5–10 h. |

### Insights → Opportunities

| Insight | Opportunity | Impact | Effort |
|---|---|---|---|
| The case-study SoL omits the LM head; the corrected fractions are higher for both systems (BASE §7; CHK1) | Fix the scripts and regenerate `table_a10`/`numbers4`. Add a test that ties the SoL scripts' dense bytes to `perfmodel`. | High | Low (hours) |
| On a desktop, PCIe ≈ DRAM (r* ≈ 1.1–1.3, my estimate), and PCIe DMA and the CPU helpers read the *same* DRAM | (a) Retune κ and admission in simulation for the race host.<br>(b) Add a shared-DRAM term to the physical bound: `max(…, (CPU-read + loaded bytes)/B_DRAM)`. Proposition 1 (PAPER §4) treats PCIe and CPU reads as independent. The extra term keeps the bound valid and tightens it on desktops (my inference).<br>(c) State the regime result: CPU execution of misses pays on server hosts, fetch on desktops. | High | Low–Med |
| No third-party equal-VRAM race exists (ENT Q5), and no MoE-offload paper scores itself against a bound (FAIR Q5) | Make the protocol and the bound-scored table the headline, whoever wins | High | Med |
| O and f are about half of the A10 excess, but O has never been profiled (BASE §4.2–4.3) | Before changing code, take an nsys trace of the cache with two controls: static policy with a DFA-derived INIT, and one C = E−1 layer (BASE §8.1) | High | Low |
| FreeToken and we differ in three implementation choices (CHK2) | Microbenchmark the per-layer hand-off on the race GPU: ours, a memop wait, and host-func. No public µs figure exists for a graph host node's wake-up (HEAD Q3 gaps), and NOV priority 2 asks for this ablation. | Med | Med |
| The recommended host class is FreeToken's own desktop testbed (FAIR Q3) | Replicate FreeToken's desktop numbers on its own platform class, using ACM's "reproduced/replicated" terms (FAIR Q4) | Med–High | Low |
| Our margin over static placement is largest on big experts ("gpt-oss-120b benefits most", BASE §5.4), and 120B on a desktop is the mission's showcase | Make gpt-oss-120b MXFP4 the headline model | High | Low |
| Tool defaults handicap baselines (Theme 4) | Publish a tuned equal-VRAM llama.cpp recipe per model and budget, for home users and maintainers | Med | Low |
| BF16 is the only Qwen3 format that FreeToken, KT and llama.cpp share (ENT Q3), and our BF16 GPU path is unpatched (BASE §2.3) | Patch negative ids in the BF16 `MUL_MAT_ID` kernel and add an op test; otherwise race Qwen3 BF16 without us | Med | Low–Med |
| gpt-oss-120b has no all-GPU run on a 32 GB card to anchor the implementation-relative SoL (BASE §3.2, §7) | Use as headline denominator the physical SoL with same-day measured bandwidths (FAIR Q5's "SoL_measured"). Report the implementation-relative SoL as secondary, with 120b's T0 from gpt-oss-20b's all-GPU run plus nsys (±5%, as on the A10). | Med | Low |
| Maintainers want small, sanitizer-clean pieces and proof against `--mmap` (ECO Q2, Q5) | Upstream the harness, recipes and small fixes, and present the patch as a research artifact | Low–Med | Low |

### Audiences for the result (template: user segments)

| Audience | Characteristics | Needs | Weight in this plan |
|---|---|---|---|
| **SPCL / ETH AI Center reviewers** | Read for method. Hoefler & Belli's rules: show bounds (R11), CIs (R5), harmonic means for rates (R3), document every factor (R9) (FAIR Q1). Decide on material submitted by 27 Oct. | A correct bound (head fix), CIs, pre-registration, losses reported honestly, no claim that the mechanism is new | Highest: the only audience with a deadline |
| **People running models at home** (the mission) | RTX 4090/5090 with 64–128 GB desktops; use llama.cpp/Ollama and Q4-class GGUFs (ENT Q5; ECO Q4) | "Which engine and flags for gpt-oss-120b on my 5090, and how fast?"; reproducible commands | Medium-high; it is also why the desktop is the right race machine |
| **llama.cpp maintainers** (am17an, ggerganov, JohannesGaessler, pwilkin) | Treat expert caching as a maintenance burden. Ask for RFCs, small PRs, scheduler changes that pass the sanitizer, and proof against `--mmap`. AI use must be disclosed (ECO Q2, Q5). | Tuned equal-VRAM recipes, a reproducible harness and small fixes, not a ~3,000-line patch | Medium |
| **Competing systems' authors** (FlashML/FreeToken, kvcache-ai/KT, NVIDIA pipelined sharding, leloch/csantiago78/everett6) | Active repos. Claims: FreeToken 1.5–2.3×, KT 1.25–1.93× (NOV) | Exact versions and commands; the chance to supply their best configuration; "not reproduced under stated conditions" rather than "wrong" (FAIR Q4) | Medium: they can discredit the race if not consulted |
| MLSys/ISPASS reviewers (secondary) | Expect two platforms, a strong state-of-the-art baseline and ablations (NOV) | Host-callback ablation, a second platform | Low within this 2-week plan |

### Where we can win, where we will likely lose (blunt)

This is for the recommended desktop: RTX 5090 + Zen 5, about 50–70 GB/s DRAM and 49–52 GB/s PCIe. Rows marked "my estimate" use these note-sourced inputs:
- per-token hit rates (BASE §6, S arm);
- 13.25 MB MXFP4 experts;
- helpers at ≈0.8 of read bandwidth, f ≈ 50 µs, O ≈ 35 µs/layer (BASE §5.4);
- 120b all-GPU on a 5090 ≈ 210 tok/s (BASE §5.4);
- FreeToken's 9950X3D bandwidths (FAIR Q3).

| Opponent | Model / budget | Expectation | Confidence | Why |
|---|---|---|---|---|
| Tuned llama.cpp / ik (static `-ncmoe`/`-ot`) | gpt-oss-120b MXFP4, 12.5–40% | **Win, wide.** My estimate is ≈2–3× on this host: roughly 45–50 vs ≈20 tok/s at 12.5%, and ≈100–110 vs ≈30–35 at 40–45%. | Medium-high | The A10 gave 1.56–1.60× (BASE §3.4). Projections are 1.43–1.69× at 250 GB/s and rise as host bandwidth falls (BASE §5.4). Static placement runs every selected expert of each CPU layer, while the cache runs only misses. The risk is baseline tuning, which moved llama.cpp by up to 13% (BASE §7). |
| Same | Qwen3-30B Q4_K_M | **Win, smaller** (roughly 1.3–1.6×) | Medium | The A10 gave 1.36–1.66× (BASE §3.4) and the 5090 projection is 1.29–1.60× (BASE §5.4). O on 48 layers was 41–60% of the Q4_K_M excess (BASE §4.3). |
| Same | Qwen3-30B BF16 | Unknown until our BF16 path works; likely a win if it does | Low | The GPU kernel path for BF16 is untested and unpatched (BASE §2.3) |
| Community caches (#27861 + everett6 fix; leloch v2) | Qwen3 Q4_K_M, 120b | Likely win | Medium-low | leloch v2 reports only +2.1% on Qwen3-30B-A3B over cache-off (ECO Q2). everett6's 1.49× was not measured at equal VRAM (ENT Q4, Q5). |
| Pipelined sharding v2.0.3 | 120b, Qwen3 GGUF | Likely win at small budgets; possibly a tie near 32 GB | Medium | Its b6097 base lacks later speed-ups (ENT Q6). Its paper's 0 GB llama.cpp row measured 25.7 vs a predicted 49.5 tok/s (ENT Q5). Its 32 G Qwen3 Q4_0 row, 158.6 tok/s, is close to all-GPU (ENT Q5). |
| KTransformers (deferral 0) | Qwen3-30B BF16 | A toss-up. We may win at small budgets (static vs dynamic placement) and lose at large ones. | Low | SeqMoE measured KT at 34.2 vs llama.cpp static at 30.9 tok/s on a 4090 (ENT Q5). KT's own batched data: dynamic update wins at 10% of experts, frequency placement at 90% (NOV). Its AVX-512 BF16 kernel is strong (ENT Q2). |
| **FreeToken** (offload/hybrid) | 120b at ~40% | **Likely loss by ~10–30%; a tie is possible** | Medium-low | The community measured 127.1 tok/s at 40.4% / ~29 GB (ENT Q5). My estimate for ours is ≈95–110 tok/s. FreeToken's pooled all-layer cache gets more hits (+0.4–2.1 pp from pooling, BASE §6), and fetching is as cheap as CPU work here (Theme 5). |
| FreeToken hybrid | 120b at 12.5–25% | Likely loss or tie | Low | Both systems are DRAM-bound. FreeToken's hybrid serves *current* misses by fetch and CPU at once (CHK2: `copy_missing` before the GEMM, CPU in parallel). Our admissions help only two steps later (BASE §2.1). |
| FreeToken | Qwen3-30B BF16 | Toss-up | Low | With 48 layers, per-layer fixed cost decides. Ours: O is 31–38 µs/layer. FreeToken: three D2H copies, one H2D copy and several torch ops per layer (CHK2). Neither is measured on this host. |
| FreeToken | Qwen3.6-35B-A3B BF16 | **Likely loss** | Medium | We are projected at only 0.78–1.02× even against llama.cpp (BASE §1). A merged-gate/up GGUF silently disables mailbox mode (BASE §2.3). It is FreeToken's home model: 77.1 tok/s at 23.8 GB on a 5090 (ENT Q5). |
| Physical SoL | all | Everyone stays far from it | High | The audit's median system is far from the bound (PAPER §7, finding 4). GPU efficiency falls on the 5090, e.g. 38% of datasheet for Qwen3 Q4_K_M (HEAD Q1). |

**Blunt reading:**
- **The bankable win:** the fastest llama.cpp-family offload at equal VRAM on gpt-oss-120b, by a wide margin.
- **The loss to plan for:** FreeToken on gpt-oss-120b on this desktop.
- **The probable loss:** Qwen3.6-35B-A3B.
- **Not measured, declared out of scope:** prefill/TTFT, batch > 1 and speculative decoding (NOV priority 7).
- A server host would likely favour our design, but no server run fits this plan honestly (see D1).

### Recommendations

**Priority order**
1. **[High]** Fix the SoL LM-head omission and regenerate the case-study numbers before anything else. Every fraction in the paper is wrong, and bounds are what SPCL reads for (BASE §7; CHK1; FAIR Q5).
2. **[High]** Race on a whole-machine Vast 5090 + Zen 5 desktop, pre-registered, with a Day-4 gate (D1, D6; MACH §2; FAIR Q4).
3. **[High]** Make gpt-oss-120b MXFP4 at three equal-VRAM budgets the headline, against tuned llama.cpp/ik and FreeToken (D3; ENT Q3, Q7).
4. **[High]** Claim measured standing, not mechanism (D10; ENT Q5; CHK2).
5. **[Medium]** Race Qwen3-30B BF16 with KT (conditional), and include ours only if our BF16 path passes (D2, D3).
6. **[Medium]** Sprint order: measure, then policy, host-side O, f, GPU-side O, then contention (D7).
7. **[Medium]** Email the competitors' authors with logs by 13 Oct (FAIR Q4).
8. **[Lower]** Optional extras: a Verda clock-locked ncu check, Qwen3.6, Pipelined sharding, a community cache, and MoE-Infinity only if hours remain.
9. **[Lower]** Upstream recipes and small pieces, not the patch (ECO Q2).

**Decision summary**

| Decision | Recommendation |
|---|---|
| Race machine | Vast whole-machine RTX 5090 + Ryzen 9 9950X/9950X3D, DDR5, PCIe 5.0 x16, ≥126 GB, driver ≥580 |
| Fallback | Another 5090 + Zen 4/5 desktop with ≥126 GB. Then an RTX PRO 6000 WS/Max-Q on Zen 5, capped at 32 GB. Never an Arrow Lake host. |
| Entrants in | llama.cpp mainline (tuned), ik_llama.cpp, ours, FreeToken |
| Conditional | KTransformers (Qwen3 BF16 only), one community cache, Pipelined sharding |
| Out | MoE-Infinity (unless time remains), HybriMoE, 2512.16473, Fiddler, DALI, SeqMoE, 2606.10493, PR #28414, FATE |
| Models | gpt-oss-120b MXFP4 (headline); Qwen3-30B-A3B-Instruct-2507 BF16 (headline 2); Qwen3-30B Q4_K_M (llama.cpp family only); gpt-oss-20b (calibration); Qwen3.6 BF16 (stretch) |
| Budgets | 12.5 / 25 / ~40% of expert bytes, equalised on measured NVML peak |
| Gate | Day 4. GO-FULL if R_ll ≥ 1.25 at all budgets and R_ft ≥ 0.85 at ≥1 budget; NO-GO if R_ll < 1.10 at ≥2 budgets |
| Money | ≈$30–42 central; optional +$4–6 for Verda; cap $100 |

#### D1. Race machine, and fallbacks

**Choose:** a Vast.ai whole-machine host (`gpu_frac = 1`) with:
- an RTX 5090;
- a Ryzen 9 9950X or 9950X3D, DDR5, PCIe 5.0 x16;
- ≥126 GB RAM;
- driver ≥580, verified, reliability ≥0.99 preferred.

Rent it only for the gate (~5 h) and the race (~8–10 h). Develop on a cheaper Zen 5 host of the same class.

**Candidates** (snapshot of 28 Sep, MACH §2):

| CPU | RAM | Price | Status | Notes |
|---|---|---|---|---|
| 9950X | 255 GB | $0.735/h | verified | Japan, rel 0.999, 6.4 Gb/s, **3-day maximum** |
| 9950X3D | 255 GB | $0.936/h | verified | Sweden, rel 0.960, 0.94 Gb/s, 22 days |
| 9950X | 128 GB | $0.804/h | verified | Sweden, rel 0.999 |
| 9950X | 128 GB | $0.843/h | verified | India, VM-enabled, so clocks may be lockable |

Supply is thin, so re-query the offers the day before (MACH §2).

**Selection rule.** Give each candidate a 15-minute qualification (MACH §4):
- `lscpu` flags, including `avx512_bf16` and `avx512_vnni`;
- `numactl -H`;
- the link reported by `nvidia-smi -q` (Gen5 x16);
- a read-bandwidth sweep over 1–16 threads;
- pinned H2D bandwidth;
- Vast's `gpu_mem_bw` ≥1400.

Pick the highest stable read bandwidth. On a tie, prefer the 9950X3D, because it is FreeToken's own desktop testbed class (FAIR Q3).

**Why this machine:**
- **Fairness.** No co-tenant shares the memory bus, which is MACH's first fairness criterion. The Verda 1× shares DRAM controllers with 7 other GPU tenants (MACH §1).
- **Noise.** Neighbour load on a shared host is invisible to the renter (MACH §1).
- **RAM.** ≥126 GB covers every chosen model (BASE §5.5).
- **Entrant support.** Zen 5 AVX-512 engages KT's BF16 backend and our AVX-512 VNNI MXFP4 helper kernel (ENT Q2; BASE §2.3).
- **Cost.** $0.74–0.94/h, against $1.96/h for Verda 1× and $3.93/h for Verda 2× (MACH §1).
- **Mission.** A 5090 desktop is hardware people own, and the published systems target this class (MACH §2).
- **Natural offload.** 32 GB of VRAM makes offload natural for 120b, so no artificial cap is needed (MACH §1; BASE §5.5).

**What the choice costs:**
- no clock locking in Docker, so record clocks and interleave runs (MACH §2);
- thin supply;
- derated 4-DIMM DDR5, so the result is desktop-specific and must say so.

**Fallbacks, in order:**
1. Another whole-machine 5090 + Zen 4/5 desktop with ≥126 GB; about 10 hosts at $0.51–0.88/h (MACH §2).
2. An RTX PRO 6000 WS or Max-Q on a **Zen 5** desktop: the Ryzen 9 9900X with 191 GB at $1.403/h, or the 9950X3D Max-Q with 186 GB at $1.162/h (MACH §2).
   - Cap every entrant at 32 GB of measured NVML peak.
   - Use a ballast process for any system that cannot be capped (FAIR Q1).
   - Avoid the Core Ultra (Arrow Lake) hosts MACH lists first: they have no AVX-512, so KT drops to AVX2 and our helpers fall back to ggml `vec_dot` (MACH §2; ENT Q2; BASE §2.3).
3. Only if Vast is impossible: the RunPod RTX PRO 6000 (188 GB, 16 vCPU, $2.09/h, container, unknown CPU), with the same caps. RunPod's 5090 has only 35 GB RAM and cannot hold 120b (MACH §2).

**Verda:** not for the headline. Optionally rent 1× on-demand for ≤3 h (≈$4–6) for a clock-locked nsys/ncu run of the GPU-side cache kernels (MACH §1, §4). It could later supply a server-host datapoint, labelled as a noisy shared slice.

**Dev host:** a whole-machine Zen 5 box, so the helper kernel behaves as on the race host (MACH §3).
- **5090 + Ryzen 7 9700X, 62 GB, $0.534/h, verified:** enough for gpt-oss-20b and Qwen3 Q4_K_M.
- **5090 + 9950X, 126 GB, $0.507/h, unverified:** if it qualifies, it can also host 120b and BF16 work.
- Destroy the dev instance between sessions; idle storage is MACH §4's top cost risk.

#### D2. Entrants

**In:**
- **llama.cpp mainline, frozen at 4da6337.** Take the best of `-ncmoe K` (first N layers) and `-ot` (last K layers). Use `--fit off`, sweep `-lm none` vs mmap and threads, and test `GGML_CUDA_GRAPH_OPT=1` once. Keep repack on (ECO Q4). Add a labelled "as-published" `-ngl` row (FAIR Q3).
- **ik_llama.cpp at adce16f.** Keep its defaults (`-fmoe`, `-ooae`), no `-rtr`, explicit `-ot`. Report max(mainline, ik) as the best llama.cpp-family result (ECO Q3).
- **Ours:** the expert-cache patch alone on 4da6337.
- **FreeToken:**
  - run v0.1.3 ("released") and main 0d652e7 ("current") (ENT Q1);
  - run `ft bench bw` first, then offload, cpu and hybrid;
  - set `--moe-cache-size`, `--num-tokens` and `--memory-ratio` explicitly, since auto-sizing failed on Blackwell (ENT Q5);
  - log whether flag-sync or host-func was selected (ENT Q6);
  - add a 6-thread, NUMA-pinned row, FreeToken's own discipline (FAIR Q3).

**Conditional:**
- **KTransformers kt-kernel v0.7.1 + sglang-kt:** Qwen3-30B BF16 only, deferral 0. A deferral-2 row may be added, labelled lossy.
  - It enters if the sm_120 source build and a single-NUMA run succeed within 1 engineer-day. Try `--kt-threadpool-count 1`, then a `worker_pool.h` patch if needed (ENT Q2, Q6).
  - Otherwise report its status as "does not run" with the effort log (FAIR Q4).
- **One community cache:** #27861 + everett6's `0006` on `bccbacd`, or leloch v2 with `--moe-cache on`/N plus the repack comparison. It enters within ≤3 h, and only if cache allocation is verified in the logs (ENT Q6; ECO Q2).
- **Pipelined sharding v2.0.3-mlsys26:** 120b and Qwen3 GGUF, with `-mva` set to the llama.cpp configuration's measured MB. It enters if the Linux build works within ≤4 h. It is an audited weak-baseline row, so it doubles as an audit replication (ENT Q5, Q6).

**Out:**
- **MoE-Infinity:** only if hours remain; its code differs from its paper (ENT Q1).
- **HybriMoE, 2512.16473, Fiddler:** no sm_120 path, obsolete models, or not enough RAM (ENT Q2, Q6).
- **DALI, SeqMoE, 2606.10493:** no code. Report their numbers only as "reported, rescaled to the race host by the model", clearly labelled (FAIR Q3).
- **PR #28414** (prefill-only) and **FATE** (fetch-only; extraordinary claims) (ENT Q1).

#### D3. Models and quantizations

1. **gpt-oss-120b, native MXFP4 experts** (ggml-org GGUF 63.4 GB; openai HF checkpoint for FreeToken). **Headline.**
   - Tier A: dequantize and compare the expert tensors (FAIR Q2 V0).
   - Check that attention precision matches between GGUF and HF (ENT Q3 gap).
   - All entrants except KT. Use S-arm (own-text) prompts, because arm D is off-distribution for 120b (BASE §3.3).
2. **Qwen3-30B-A3B-Instruct-2507 BF16** (HF safetensors, plus a GGUF made with `--outtype bf16` and hash-verified). **Headline 2.**
   - It is the only format FreeToken, KT and llama.cpp share exactly (ENT Q3).
   - Ours joins only if the BF16 negative-id patch passes `test-backend-ops` and the fidelity check (D5).
3. **Qwen3-30B Q4_K_M:** a llama.cpp-family-only table. It gives continuity with the A10 and is what home users run. Never mix it with Tier A rows (FAIR Q1).
4. **gpt-oss-20b MXFP4:** calibration only. It fits all-GPU, which yields η, t_layer and t_e for both gpt-oss models (same expert kernel) (BASE §4.1, §5.4).
5. **Stretch: Qwen3.6-35B-A3B BF16.** FreeToken plus tuned llama.cpp only, to replicate FreeToken's audited row (77.1 tok/s at 23.8 GB; ENT Q5). Ours joins only if its GGUF keeps separate gate and up tensors and reaches `build_moe_ffn_ec` (BASE §8.5).

**Excluded:**
- FP8: llama.cpp dequantizes it, so expert bytes differ (FAIR Q1).
- NVFP4.
- IQ1_S/IQ2_S/IQ3_S: miscompiled on sm_120 (ECO Q1).
- Q8_0 in cross-engine rows.

**Disk needed:** ≈285 GB, or +140 GB with Qwen3.6 (sizes from BASE §5.5 and ENT Q2).

#### D4. Expert budgets and the equal-memory protocol

**Budget definition.** A budget is the measured NVML peak of the whole process tree at a fixed maximum context, e.g. 4096 tokens with f16 KV.
- Anchor each budget to a llama.cpp `-ncmoe` layer step.
- Fit every other system to ≤ that peak by iterative search.
- Report the knob value and a breakdown: experts, dense, KV, other (FAIR Q1).

**gpt-oss-120b** (L 36, E 128), per BASE §5.2, §5.4:

| Budget | Slots C per layer | llama.cpp `-ncmoe` | VRAM | Note |
|---|---|---|---|---|
| 12.5% | 16 | 32 | ≈10.9 GB | |
| 25% | 32 | 27 | ≈18.5 GB | |
| ~40–45% | 57 | 20 | ≈30.7 GB | Tight on 32 GB; step down one or two layers if it does not fit. The FreeToken community point was 40.4% at ~29 GB (ENT Q5). |

**Qwen3-30B BF16** (L 48, E 128, ≈9.4 MB/expert, ≈1.2 GB/layer; ENT Q4):
- 12.5% → `-ncmoe` 42; 25% → 36; ~40% → 29.
- All fit 32 GB with ≈3 GB dense BF16. That dense figure is my estimate.

**Bounds per system.** Score FreeToken against the pooled bound, and the per-layer systems against the per-layer bound (ENT Q4; PAPER §4, which has per-layer and pooled variants). Report both the system's own-placement bound and the optimal-placement bound (FAIR Q5).

**CPU regimes.** Run two, both on the same 16 physical cores:
- (a) equal and fixed;
- (b) each system at its best thread count.

Add FreeToken's 6-thread row. Apply one SMT policy to all systems (FAIR Q1).

#### D5. Must-dos before the gate (Days 1–3, mostly $0)

1. **SoL fix.**
   - In `sol_a10.py`/`sol_windows.py`, set D = `dense_bytes + head_bytes`, as `preregister.py` and `perfmodel` do.
   - Rerun for 038/039 with `--bc 166.1`, and regenerate `table_a10`/`numbers4`, the abstract and the case-study text.
   - Report the Nsight split as a sensitivity.
   - Add a regression test tying the SoL scripts' bytes to `data/gguf_bytes.json` dense + head (BASE §4.1, §7).
2. **Pre-registration** as a tagged commit, before renting the race host (FAIR Q4). Include:
   - the protocol, budgets and prompt token IDs;
   - the fidelity and gate thresholds;
   - the analysis script;
   - predictions from `project.py`, updated after the Day-4 calibration and before the race.
3. **Build and correctness.**
   - Build ours on 4da6337 for sm_120 with CUDA 12.8/12.9 and `120a`.
   - Run `test-backend-ops`: the 18 EC cases plus `MUL_MAT_ID` (BASE §3.1; ECO Q1).
   - Confirm that single-token MXFP4/Q4_K `MUL_MAT_ID` on sm_120 goes through mmvq. Patch negative ids in the BF16 kernel, or drop ours from the Qwen3 BF16 race (BASE §2.3).
   - Raise `LLAMA_EC_TIMEOUT_MS`: it is converted at 2 GHz (BASE §2.2).
   - Set the helper count and pinning explicitly. The default, hardware threads − 2, gives 30 helpers on 16 cores, and helper pinning pairs helpers on SMT siblings (BASE §2.2, §2.3).
4. **Harness.** One client for llama-server, FreeToken and SGLang-KT (FAIR Q1–Q2):
   - identical token IDs, greedy decoding, EOS ignored, N = 256 tokens;
   - TPOT excluding the first token;
   - an NVML sampler, plus a read-bandwidth probe and a clock/power log per block;
   - ΣN/ΣT rates with bootstrap CIs.

   Check that our cache runs under `llama-server`; so far it has only been measured under ec-bench.
5. **Author emails.** Draft them for FreeToken, KT and the NVIDIA pipelined-sharding authors, and send them with logs by about 13 Oct (FAIR Q4).

#### D6. Go/no-go gate

**G0: build and fidelity** (end of Day 3, dev host).

Pass requires all of:
- D5.3 passes;
- our cache runs under `llama-server`;
- on gpt-oss-20b S-arm text, teacher-forced top-1 agreement with stock llama.cpp is ≥0.98 and |ΔNLL| ≤1.0%. The A10 levels were 0.981–0.993 and −0.99…+1.06% (BASE §3.3).

If it still fails after 1.5 days of fixing, drop system work and run the race without us.

**G1: standing** (Day 4, race-class host, ~5 h, ≈$5).

Protocol:
- gpt-oss-120b MXFP4, S-arm prompts;
- budgets 12.5 / 25 / ~40%;
- each system at its best thread count from a quick sweep;
- ≥3 interleaved launches × ≥5 requests × 256 tokens.

Run the calibration hour in the same session (BASE §5.4 items 1–5):
- read-bandwidth and pinned H2D sweeps;
- a gpt-oss-20b all-GPU run with nsys;
- a cpubench thread sweep;
- a cache run at two budgets with stats (this gives O);
- an `-ncmoe` sweep.

Also record `asyncEngineCount`.

Define:
- **R_ll** = ours / max(mainline, ik);
- **R_ft** = ours / the best FreeToken mode.

| Outcome | Condition | Action |
|---|---|---|
| **GO-FULL** | R_ll ≥ 1.25 at all three budgets **and** R_ft ≥ 0.85 at ≥1 budget | Full sprint (D7), aimed at FreeToken |
| **GO-LIMITED** | R_ll ≥ 1.25 at ≥2 budgets, R_ft < 0.85 everywhere | ≤3 sprint days on O and f only. Frame the result as "fastest llama.cpp-family, X% of SoL; FreeToken leads". Put the saved days into the protocol and the paper. |
| **NO-GO** | R_ll < 1.10 at ≥2 budgets, **or** 120b S-arm top-1 < 0.98 | Stop system work. Spend the week on competitor replication, harness, fidelity and writing. Our system stays in as a measured entrant. |
| FreeToken not running after 6 h | – | Decide on R_ll alone. Spend ≤1 more day on FreeToken. Log its status and contact the authors (FAIR Q4). |

**Why these thresholds:**
- **0.85:** the planning figure for the sprint is +10–20% on this host (D7), so only a gap of ≤15% is closable.
- **1.25:** the A10's lowest instance-level ratio was 1.31, and llama.cpp run noise reached 7.8% (BASE §3.4). Every projection for a slower host is higher (BASE §5.4). Falling below 1.25 would mean something is wrong.
- **1.10:** roughly baseline noise plus the fusion difference (ECO Q1).

#### D7. Sprint order (one week), with expected gains

Every change is A/B tested with ≥3 interleaved runs. Keep it only if it gains ≥2% with non-overlapping CIs and passes the op-test and teacher-forced fidelity regressions.

| Order | Item | What | Planning gain on the desktop (my estimate) | Gain on 150–400 GB/s hosts (notes) | Effort | Condition |
|---|---|---|---|---|---|---|
| 0 (before G1, every entrant) | Threads and placement | For us: one helper per physical core across both CCDs, no SMT siblings, explicit pinning. For the others: a thread sweep. | 0–15% (0 if already near read bandwidth) | +10–20% if at 60% of STREAM (HEAD Q6 rank 2) | Hours | Always |
| 1 | Profile | nsys with CUDA-graph node tracing plus OS runtime; static-INIT and C = E−1 controls; `asyncEngineCount` | Decides items 3–7 | – | 0.5 day | Always (BASE §8.1) |
| 2 | Policy retune in simulation | κ for r* ≈ 1.1–1.3 (A10 ≈ 11), half-life, `MAX_ADMIT`, pooled vs per-layer. Uses `ecsim_fast` + `cachesim` + `project.py` on CPU. | 0–5% (DFA hits are within ±2.2 pp of LRU, BASE §6) | 0–10% (HEAD Q6 rank 5) | Hours, $0 | Always (BASE §8.4) |
| 3 | O, host side | Take the policy loop off the critical path (admissions publish two steps later anyway); cache device pointers instead of calling `cudaHostGetDevicePointer` per call; one launch per admitted expert, not per tensor; skip the maps H2D when unchanged (BASE §2.1, §4.2) | 2–6% | Part of "O = 0": +9–45% (BASE §5.4) | 1 day | If nsys shows host gaps |
| 4 | f, helpers | Merge the four barrier phases: quantize once; fused gate+up across the request's experts; per-expert completion counters before down; weighted partial sums straight into the mailbox. Target ≤5–10 µs, down from 50 µs (gpt-oss) and 18–26 µs (Qwen3). | 3–7% | "O = f = 0" adds +5–25% over O = 0 (BASE §5.4); HEAD rank 1 | 1.5–2 days | Always |
| 5 | O, GPU side | Fold the `get_rows` maps and `ec_req` into the router/top-k kernel; merge `ec_wait` into the weighted sum; let #25952/#28432 fire on cached layers | 3–8% | +5–19% (HEAD Q6 rank 3) | 2–3 days | If nsys shows cache-path kernels dominate O |
| 6 | Admission vs DRAM/copy contention | Hold admissions while a layer's helpers run (SeqMoE-style pause flag); chunk fills to 1–4 MB | 0–5% | 2–10% (HEAD Q6 rank 4) | 1–2 days | If nsys or the probe show interference |
| 7 | Zero-copy GPU execution of some misses | Run one miss per layer on the GPU, from pinned memory over PCIe, while the helpers run the rest. This is FreeToken-hybrid-style aggregation. | 0–20% at 12.5–25%; ≈0 at 40% | The bound's "balance" term, 27% of the A10 excess (BASE §4.3) | 2–3 days | Only if helpers pull clearly less than STREAM read and G1's gap to FreeToken is concentrated at small budgets. It replaces item 5. |
| – | Not this week | Prediction/prefetch (0–15%, 3–5 days); PDL or a megakernel (weeks); Expert Deferral (lossy) | – | HEAD Q4, Q6 | – | – |

**Schedule:**
- S1: items 1–2.
- S2: item 3.
- S3–S4: item 4.
- S5: item 5, 6 or 7, as nsys and G1 dictate.
- S6: freeze; run the regressions; update and tag the predictions.
- S7: buffer.

The combined planning figure on the desktop is +10–20%. On server hosts, BASE projects more.

#### D8. Race-day protocol (short)

- **One continuous session.** Run the qualification probe first. Warm the page cache. Interleave the systems in randomised blocks, with a bandwidth probe and a clock/power log per block (MACH §2; FAIR Q1).
- **Cells.** Each cell gets ≥3 fresh launches × ≥5 requests. Continue until the 95% CI half-width is ≤2–3%. Report cold and steady-state results for the dynamic caches (FAIR Q1).
- **Fidelity (V0–V2).** Take the noise floor from benign llama.cpp reconfigurations: all-GPU for 20b, two `-ncmoe` splits, and `GGML_CUDA_MOE_WEIGHTED_REDUCTION=0` (FAIR Q2; ECO Q1).
- **Scores.** For every cell report absolute tok/s with a CI, P90/P99 TBT, the fraction of SoL_measured, and the fraction of the implementation-relative SoL (FAIR Q1, Q5).

#### D9. Framing by outcome

| Outcome | Definition | Headline | Must also say |
|---|---|---|---|
| **Win** | Ours is fastest at equal measured VRAM on ≥2 of 3 budgets for 120b, with non-overlapping CIs | "On a 5090 + Ryzen desktop (measured X GB/s DRAM, Y GB/s PCIe), at equal GPU memory, [ours] decodes gpt-oss-120b at A tok/s, B% of the measured speed-of-light, vs FreeToken C and tuned llama.cpp D." | Attribute the win to measured implementation choices via ablations: the per-layer skip, lower O and f. Cite FreeToken's handshake. Scope: this host class; BASE §5.4 says margins differ on server hosts. |
| **Tie** | CIs overlap, or within ±5% | "Two independent designs, a Python serving engine with a pooled cache and fetch, and a C++ llama.cpp per-layer cache with CPU misses, land within x% of each other at ~B% of the bound." | Present it as evidence for the bound: strong systems cluster at the same fraction, and the rest is host DRAM plus per-layer fixed cost. It links to the audit's finding that published 1.5–2.3× gains mostly reflect weak baselines. |
| **Loss to FreeToken** (likely) | FreeToken faster on ≥2 budgets | "FreeToken is the fastest system we measured, at B% of SoL. Ours is the fastest llama.cpp-family system, at x× tuned llama.cpp." | Use the bound to decompose our gap: hit rate (pooled vs per-layer), O, f, and fetch vs CPU at PCIe ≈ DRAM. Say whether FreeToken's paper numbers were replicated on its own platform class. For the application, the yardstick is the contribution, and this outcome confirms it predicts. |
| **Loss to llama.cpp** (should be caught at G1) | R_ll < 1.10 | Report it plainly. | The cache's fixed costs exceed its hit-rate savings on this host. Drop the system from the headline. |

**Per audience:**
- **SPCL:** lead with the bound, the pre-registration and the CIs.
- **Home users:** a recipe table, "which engine and flags at which VRAM".
- **Maintainers:** the tuned recipes and any upstream bugs found.
- **Competitors:** a status block per system: reproduced, runs but not reproduced, does not run, or not runnable (FAIR Q4).

#### D10. What to say about novelty

Proposed wording for the paper and application:

> "Our system uses no new mechanism. GPU expert caches with CPU-executed misses appear in HybriMoE, 2512.16473, DALI, KTransformers, FreeToken and three llama.cpp contributors' branches; FreeToken has shipped a GPU-signalled, polled CPU hand-off since 11 August 2026; decayed-frequency admission is in llama.cpp PRs #26563/#26824; zero-copy loads from mapped host memory are in SeqMoE. We race it as one entrant under the same protocol as the others. Three implementation choices differ from FreeToken's (a device-side per-layer skip of the hand-off, activations and results moved by kernel loads/stores rather than copy-engine transfers, and a spinning wait kernel rather than stream memory operations); we report their measured effect where we ablated them and claim nothing where we did not."

Sources: NOV claims 4a–4f; ENT Q2, Q5; CHK2.

**What we do claim:**
- the time-domain bound, the validated model and the audit (NOV; PAPER);
- the race protocol, as far as we found: FAIR Q3's five-point combination "no named paper has all five", and ENT Q5's observation that no third-party equal-VRAM comparison exists;
- the measured standings and the bound's explanation of them.

**What we drop:** the phrase "novel systems contribution" from ECO Q5, and the GPU-decided skip as a novelty claim. NOV itself notes reviewers will call the skip "a natural consequence" of the mailbox.

#### Schedule and budget (2 weeks from Mon 29 Sep)

| When | What | Where | Cost |
|---|---|---|---|
| 29 Sep | SoL fix and regenerated numbers; pre-registration draft; harness skeleton; policy simulation with FreeToken's 9950X3D bandwidths as placeholders | Local | $0 |
| 30 Sep – 1 Oct | Build ours, llama.cpp and ik; op tests; server-mode check; BF16 patch; FreeToken (CUDA 13 venv); KT source build; community cache; pipelined-sharding build | Dev host, ~10 h | $5–6 |
| 2 Oct | Qualification, calibration hour, **G1** | Race-class host, ~5 h | $4–5 |
| 3–9 Oct | Sprint (D7), or the measurement-only path | Dev host, ~12–15 h | $6–8 |
| 10 Oct | Freeze; update and tag predictions | Local | $0 |
| 11–12 Oct | **Race** | Race host, 8–10 h | $6–9 |
| 13 Oct | Analysis and scoring; emails to authors with logs | Local | $0 |
| 14–24 Oct | Paper revision (race section, corrected case study) and application; fold in author replies | Local | $0 |
| ≤26 Oct | arXiv post (ETH deadline 27 Oct 16:00 CET) | – | – |

**Totals:**
- storage and downloads $6–10, plus a $4 buffer;
- **≈$30–42** in all, within the $25–40 target at the low end and well under the $100 cap;
- optional Verda check +$4–6.

These rates come from MACH §3–4.

### Top risks and mitigations

| # | Risk | Evidence | Mitigation |
|---|---|---|---|
| 1 | The SoL head bug reaches a submission | BASE §7; CHK1 | Fix on Day 1, add a regression test, regenerate by script. Also check the Qwen3 Q8_0 and 120b T0 extrapolations (BASE §3.2). |
| 2 | Race-host supply dries up, or its contract ends mid-race | 3 qualifying 5090 offers, one capped at 3 days (MACH §2) | Rent only for the gate and the race; re-target by `machine_id`; keep the D1 fallback list; do dev on the same class. |
| 3 | The desktop regime is uncalibrated and projections mislead | BASE §5.4 covers 150–400 GB/s only; the desktop gives ~50–70 GB/s (MACH §2) | Run the calibration hour at G1, then re-run and pre-register `project.py`. Every expectation in this synthesis is a placeholder. |
| 4 | FreeToken install or runtime fails | #554 hash mismatch, Blackwell auto-sizing failure, memop probe (ENT Q5, Q6) | Pin versions; separate CUDA 13 venv; explicit cache and KV sizes; log the handshake mode; 1-day budget, then report status and email the authors. |
| 5 | KT segfaults on a single-NUMA Zen desktop | #1754 on exactly a 9950X3D + 5090 (ENT Q2) | `--kt-threadpool-count 1`; a local `worker_pool.h` patch; torch 2.9.1 venv; 1-day budget. |
| 6 | Our patch breaks on Blackwell (−1 ids in kernels other than mmvq, the 2 GHz timeout, spin kernel under CUDA graphs, SMT helper pairing) | BASE §2.2, §2.3 | Op tests on the card; BF16 patch; raise the timeout; explicit helper pinning. Stay on MXFP4/Q4_K where the id handling is patched. |
| 7 | Accusations of untuned baselines (anti-pattern AP2) | FAIR Q1, Q3; Theme 4 | Tune llama.cpp and ik harder than ours: load mode, threads, `-ot` last-K, `GRAPH_OPT`. Log each system's tuning budget; add as-published rows. |
| 8 | Noise without clock locking | llama.cpp moved −7.8% to +1.6% on the A10 (BASE §3.4); Vast Docker cannot lock clocks (MACH §2) | Interleave; per-block bandwidth and clock logs; ≥3 launches × ≥5 requests; CI ≤2–3%. |
| 9 | Disputes over equivalence (fusions are not bit-identical; greedy trajectories diverge) | ECO Q1; FAIR Q2 | V0–V2 with a pre-registered noise-floor multiple (e.g. ≤3×), and a baseline run with `GGML_CUDA_MOE_WEIGHTED_REDUCTION=0`. |
| 10 | Schedule: author contact needs 1–2 weeks before posting | FAIR Q4 | Race by 12 Oct; emails with logs on 13 Oct; post around 24–26 Oct. |
| 11 | Appearance of favouring our own system | FAIR Q4 ("State in the paper that the author's own system is in the race.") | Tagged pre-registration before renting; same protocol and equal or larger tuning budgets for competitors. |
| 12 | Novelty overclaim | ENT Q5; CHK2 | Use the D10 wording; scrub "novel" from the abstract, the case study and the upstream PR draft. |
| 13 | Cost overrun from idle storage or re-downloads | MACH §4 | Destroy dev instances between sessions; keep only 20b and Q4_K_M on dev; use `hf_transfer`; check `inet_down_cost`. |
| 14 | No all-GPU run can anchor the 120b implementation-relative SoL on 32 GB | BASE §3.2, §7 (A10 T0 extrapolation disagreed by 1 ms) | Headline SoL_measured instead; derive 120b's implementation-relative SoL from 20b's all-GPU run plus nsys, and state ±5%. |
| 15 | Upstream PRs collide mid-plan | #26167, #29184, #29181 (ECO Q1) | Freeze the base at 4da6337 and record it. |
| 16 | Container limits: nsys or ncu permissions, pinning 61 GB inside Docker | Not covered by the notes; the user's memory notes RunPod blocks ncu | Check on the dev host on Day 2; use Verda for ncu if needed. |

### Questions only the user can answer

1. **Vast.ai access.** Do you have, or will you create, a funded Vast.ai account? Is Docker with root but no clock locking acceptable? The providers you have access to rank below it for this race: RunPod's 5090 has 35 GB RAM, Lambda has no sm_120, and Verda's is a shared slice (MACH §2).
2. **leloch.** Are you "leloch", the author of llama.cpp RFC #24528? NOV asks for explicit confirmation, and the answer changes how the prior art is cited.
3. **The race's role in the application.** Is it the headline, or supporting evidence for the bound? The paper currently keeps our system "only as a system under test" (PAPER §2). Option C should add a race section, not re-centre the paper on the system, unless you want otherwise.
4. **Contacting authors.** Will you email the FreeToken, KT and NVIDIA authors around 13 Oct under your own name ("Independent Researcher"), and hold the arXiv post until about 24 Oct for replies?
5. **Qwen3.6-35B-A3B.** In or out? Including it is the honest replication of FreeToken's claim. It is a probable loss for us and adds ~140 GB of downloads.
6. **Money and renting twice.** Do you approve ≈$42 plus the optional Verda check (≈$6), and renting the race-class host twice (gate and race)?
7. **MLSys 2027 (30 Oct).** Is it in scope? If yes, the host-callback ablation (NOV priority 2) competes with sprint time.
8. **Your time.** How many hands-on hours per day can you give the builds? FreeToken, KT and Blackwell toolchain debugging are the main time sinks.
9. **Public pre-registration.** May the pre-registration be public before the race? It reveals the plan and our predictions.
10. **Upstream.** Should an RFC or discussion go to llama.cpp before 27 Oct? AI use must be disclosed, and new contributors are limited to one open PR (ECO Q2).
11. **Base commit.** Freeze on 4da6337 (current), or stay on 2145525a for continuity with the A10 case study?

### Questions for further research (empirical, to settle by Day 4)

- What read bandwidth and pinned H2D does the chosen desktop deliver? How many helper threads saturate it?
- Is FreeToken's memop handshake active in a Vast container, or does its probe fall back to host-func (ENT Q2 gaps)?
- Does KT v0.7.1 run single-NUMA on Zen 5 without the #1754 patch?
- Do #25952/#28432 fire on cached layers (ECO Q1 gaps)?
- How is O split between kernels, hand-off, host policy and admissions (BASE §4.2)?
- Does the spin kernel measurably take SMs or clocks away from the GEMVs (HEAD Q3 gaps)?
- What is the `asyncEngineCount` of the 5090? Do fills contend with helper DRAM reads (HEAD Q5 gaps)?
- Does our cache behave the same under `llama-server` as under ec-bench?
- Qwen3.6's GGUF layout: are gate and up merged (BASE §8.5)?

### Methodology Notes

**Inputs.** All six notes were read in full (≈3,080 lines), plus the novelty-check report and `paper/paper.tex`. The two JSON snapshots were not re-analysed.

**Checks run here** (no GPU, no rentals, no authenticated APIs, no files modified other than this one):
- **CHK1:** `scripts/sol_a10.py`, `scripts/sol_windows.py`, `mosl/perfmodel.py` and `data/gguf_bytes.json` confirm the head omission. The saved `prereg/a10_windows_038/sol_{D,S}.json` carry `B_C` = 166.1 GB/s.
- **CHK2:** FreeToken at 0d652e7, read from the session scratchpad clone (`python/freetoken/moe/cpu_executor.py`, `python/freetoken/layers/moe.py`). It confirms the memop handshake, the unconditional per-layer submit/sync, D2H/H2D copies for activations and results, and on-demand fetch before the GEMM in hybrid mode.
- **CHK3:** in the scratchpad llama.cpp clone, 4d86b2f is the head of the local `expert-cache` branch on 2145525a.

**Limitations:**
- **Prevalence counts** count the notes that bring direct evidence on a theme. They are not independent votes: all six notes were written for this project, often from the same sources, and GitHub threads were read through summarising fetches (ENT, ECO and FAIR say so).
- **Desktop estimates.** Every performance expectation for the desktop is my arithmetic on note-sourced constants, and none is calibrated. In particular, the FreeToken comparison mixes a community run on an unstated host (127.1 tok/s) with my model of ours.
- **Supply snapshots.** Vast offers are one snapshot from 28 Sep (MACH).
- **Conflict of interest.** The synthesis is written for the owner of one of the entrants. The recommendations counter this by pre-registration and extra baseline tuning, not by assuming it away.

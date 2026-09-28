# Controlling, emulating or normalizing host memory bandwidth for CPU-side llama.cpp MoE decode on rented servers

Scope: llama.cpp decode where some layers' experts run on the CPU from host RAM (`--n-cpu-moe`). Published results are mostly desktop dual-channel DDR4/DDR5 (~50-100 GB/s); rentals are often 8-12 channel servers (200-600 GB/s) or VMs that get part of a socket. Researched September 2026. Dates are flagged where known; material from before 2024 is marked [OLDER].

## 1. Hardware bandwidth throttling (Intel RDT MBA, AMD MBA via Linux resctrl): mechanism, precision, and use without host root

### Takeaway
Both vendors expose a memory-bandwidth cap through Linux `resctrl`, but it does not work the same way on each. AMD's is a closed-loop cap set in 1/8 GB/s units and enforced per L3/CCX domain. Intel's is a coarse, nonlinear, per-core "delay" percentage, which can be turned into an MB/s target with the kernel's `mba_MBps` software controller. Mounting or configuring resctrl needs host root. KVM guests generally do not see RDT at all, and a container can only be placed in a resctrl group that the host set up. In practice, then, MBA is available on a rental only if it is bare metal with root, or if the provider configures it.

### Cited Findings
**Linux resctrl interface (current kernel docs)**
- MBA granularity is exposed in `info/MB/min_bandwidth` and `info/MB/bandwidth_gran`: "The allocated b/w percentage is rounded off to the next control step available on the hardware"; the available steps are `min_bw + N * bw_gran`. — [Linux kernel resctrl docs](https://docs.kernel.org/filesystems/resctrl.html)
- Intel: "The bandwidth throttling is a core specific mechanism on some of Intel SKUs". The `thread_throttle_mode` file reports either "max" (the smallest percentage of the two SMT threads is applied to both) or "per-thread". The `delay_linear` file "Indicates if the delay scale is linear or non-linear" and is informational only. — [Linux kernel resctrl docs](https://docs.kernel.org/filesystems/resctrl.html)
- AMD: "The allocated resources are in multiples of one eighth GB/s" (e.g., `MB:1=16` = 2 GB/s). — [Linux kernel resctrl docs](https://docs.kernel.org/filesystems/resctrl.html)
- The `mba_MBps` mount option enables a "Software Controller (mba_sc)" that "reads the actual bandwidth using MBM counters and adjust the memory bandwidth percentages to ensure: 'actual bandwidth < user specified bandwidth'". Each group can pick which MBM event drives this loop (`mba_MBps_event`). — [Linux kernel resctrl docs](https://docs.kernel.org/filesystems/resctrl.html)
- Monitoring: `mbm_total_bytes` / `mbm_local_bytes` count memory traffic per group, which is useful for verifying achieved bandwidth even without throttling. With BMEC (AMD), the event mask is configurable (default total=0x7f, local=0x15). — [Linux kernel resctrl docs](https://docs.kernel.org/filesystems/resctrl.html)
- Requirements: "CONFIG_X86_CPU_RESCTRL and the x86 /proc/cpuinfo flag bits"; mounting is `mount -t resctrl resctrl [-o options] /sys/fs/resctrl`, which needs root. — [Linux kernel resctrl docs](https://docs.kernel.org/filesystems/resctrl.html)

**AMD EPYC (Genoa 9004 / Turin 9005)**
- The cap is enforced per CCX/L3 domain, not per core: "bandwidth ceilings are enforced per CCX". Threads sharing a Class of Service (COS) in the same domain "share the resource limits defined for that class", and ceilings are "shared within a QoS domain (CCX), but not across different CCXs". — [AMD PQOS White Paper for EPYC 9004/9005](https://docs.amd.com/api/khub/documents/VuNrmUG_yfhPVgGYcFlkZg/content)
- Units are 1/8 GB/s (e.g., 256 → 32 GB/s), with a maximum value of 2048 (256 GB/s). It works as "a closed-loop control system" that "tracks memory bandwidth usage per COS and dynamically adjusts throttle levels" at 128 µs intervals. — [AMD PQOS White Paper](https://docs.amd.com/api/khub/documents/VuNrmUG_yfhPVgGYcFlkZg/content)
- "MBA (or SMBA) must be enabled in BIOS or firmware". Support is detected via CPUID 0x8000_0020 EBX bit 1. The paper presents per-VM limits via COS IDs as a host/hypervisor-side use case. — [AMD PQOS White Paper](https://docs.amd.com/api/khub/documents/VuNrmUG_yfhPVgGYcFlkZg/content)
- Socket-wide control is new and not yet mainstream. The "Global Memory Bandwidth Allocation (GMBA)" patch series would "Bound DRAM bandwidth for groups of threads that span multiple L3 QoS domains, rather than being per-L3 like MBA", using an "NPS-node" control domain. It targets Zen 6 (AMD spec Rev 1.00, March 2026); the series was at v2 on 2026-04-24. — [LWN: x86/resctrl GMBA](https://lwn.net/Articles/1069444/)
- Zen 2 Rome (EPYC 7502) was found to offer "Small granularity" and a "Wide range for the throttling settings" with "Strong effect of the bandwidth throttling on application". Very restrictive settings "may show effect on latency". — [Using Bandwidth Throttling to Quantify Application Sensitivity to Heterogeneous Memory, MCHPC@SC 2021 slides](https://passlab.github.io/mchpc/mchpc2021/presentation/104.pdf) [OLDER platform, 2021]

**Intel Xeon**
- On Cascade Lake Xeon Gold 6230, MBA offered only "coarse 10% throttling steps" with "Little effect of the bandwidth throttling on application", needing a "very restrictive setting to observe an effect". — [MCHPC 2021 slides](https://passlab.github.io/mchpc/mchpc2021/presentation/104.pdf) [OLDER, 2021]
- On a Cascade Lake Xeon Silver, only MBA delay values 70, 80 and 90 were significant limiters (10-60 gave negligible reduction; 20 and 30 were erroneous). Delay 80 gave a 26% bandwidth reduction and delay 90 a 50% reduction. The effect depends on workload: reads without L2 writebacks were "the most permissive limitation". — [Sohal et al., "Assessing Intel's Memory Bandwidth Allocation for resource limitation in real-time systems" (arXiv 2206.14637)](https://arxiv.org/pdf/2206.14637) [OLDER, 2022]
- "A Closer Look at Intel Resource Director Technology (RDT)" (RTNS 2022) is the standard deeper characterization of Intel MBA. The ACM page returned 403 and the content was not verified here. — [Sohal et al., RTNS 2022 PDF](https://cs-people.bu.edu/rmancuso/files/papers/CloserLookRDT_RTNS22.pdf)

**Containers and VMs (no host root)**
- Kernel maintainer position in 2019 (after an AWS guest issue): "assuming RDT is not going to be supported in a guest, we need a proper fix to disable it when in a guest". The proposed fix skips RDT init when `X86_FEATURE_HYPERVISOR` is set. — [LKML, Petkov/Chatre, Jan 2019](https://lkml.iu.edu/hypermail/linux/kernel/1901.1/01820.html) [OLDER, 2019; current KVM guest behavior not re-verified]
- In the libvirt/KVM model, MBA is applied from the host: `<memorytune>` "can control allocations for memory bandwidth using the resctrl on the host" for chosen vCPUs (libvirt ≥4.7.0). The guest is not involved. — [libvirt Domain XML format](https://libvirt.org/formatdomain.html)
- Containers: runc gained Intel RDT/MBA support (the OCI `intelRdt` config) — [runc PR #1632](https://github.com/opencontainers/runc/pull/1632). containerd/CRI-O use goresctrl "RDT classes", which are configured host-side on a host-mounted resctrl ("Partitions and classes must be configured at the host level"). MBA can be given as a percentage or as MBps when resctrl is mounted with `-o mba_MBps`. — [intel/goresctrl rdt.md](https://github.com/intel/goresctrl/blob/main/doc/rdt.md)

### Inferences
- On a typical rented VM (KVM, no host access), MBA is effectively unavailable. On a container rental (e.g., Docker on a shared host), it is usable only if the provider exposes `/sys/fs/resctrl` or places the container in an RDT class, which is not something to expect. Bare-metal rentals with root are the only reliable MBA path. A quick check: `grep -oE 'rdt_a|cat_l3|mba' /proc/cpuinfo`, `ls /sys/fs/resctrl`, `mount | grep resctrl`.
- AMD MBA is the better tool for emulation where it is available: absolute GB/s units plus closed-loop enforcement. It is per-CCX, though, so a total cap of X GB/s for llama.cpp threads spread over N CCDs needs X/N per domain. The imbalance across CCDs that this causes is not modeled.
- Intel percentage MBA should not be used to target a GB/s value directly. Use `mba_MBps` (MBM-feedback software controller) and verify with MBM counters or a read benchmark in the same group.
- MBA adds delay or queueing to requests. It does not lower idle latency toward desktop levels, so it reproduces bandwidth but not latency (see Section 2).

### Gaps
- No primary measurements were found of Intel MBA precision on Sapphire/Emerald/Granite Rapids (Xeon 4th-6th gen), nor of any "MBA 2.0" behavior. Published characterizations are Skylake/Cascade Lake era.
- No quantitative "requested vs achieved GB/s" accuracy data for AMD Genoa/Turin MBA was extracted (the AMD white paper shows scaling plots, but the numbers were not captured).
- Could not verify whether any major GPU rental marketplace (RunPod, Vast.ai, Lambda, Verda, etc.) exposes resctrl to tenants. No source found.
- GMBA's upstream merge status as of September 2026 is unknown (last seen at v2 in April 2026).

## 2. Topology tricks: numactl/NUMA binding, AMD NPS, Intel SNC, and limiting threads/cores. Bandwidth, latency, and BIOS needs

### Takeaway
NPS4 (AMD) or SNC (Intel) plus `numactl` binding can cut one NUMA node's bandwidth to desktop-like levels (~100-190 GB/s per node), but both are BIOS settings, and a VM's NUMA layout is fixed by the provider. Two things need no root: pinning threads to a single AMD CCD, which caps reads at the Infinity Fabric link (~64 GB/s GMI-narrow, ~100 GB/s GMI-wide), and limiting thread count, which caps bandwidth through per-core concurrency limits. Both, however, also change compute capacity. None of these restores desktop latency: server DRAM latency is ~130-140 ns at best (NPS1/NPS4) versus ~70-85 ns on desktops.

### Cited Findings
**Bandwidth per node / per CCD**
- EPYC 9355P (Zen 5, 12× DDR5-5200): in NPS4, local-node bandwidth was 117.33 GB/s (107 GB/s to remote nodes). Per-CCD read bandwidth over GMI-Wide was 99.8 GB/s, versus 62.5 GB/s on a desktop Ryzen 9 9900X (GMI-Narrow). Cross-NUMA penalties added "20-30 ns at worst", and worst-case unloaded latencies stayed "under 140 ns". — [Chips and Cheese, "AMD's EPYC 9355P: Inside a 32 Core Zen 5 Server Chip" (2025-09-30)](https://chipsandcheese.com/p/amds-epyc-9355p-inside-a-32-core)
- EPYC 9575F on a server rented from Verda: NPS1 linear read bandwidth was 479 GB/s. In NPS0, "DRAM latency rises to over 220 ns" versus ~130 ns in NPS1, "nearly 90 ns penalty". — [Chips and Cheese, "Evaluating Uniform Memory Access Mode on AMD's Turin"](https://chipsandcheese.com/p/evaluating-uniform-memory-access)
- Desktop Zen 4 (Ryzen 9 7950X3D, DDR5-5600): the IFOP link is 32 B/cycle read and 16 B/cycle write per CCD. Idle DRAM latency is 82-83 ns. One CCD with 8 bandwidth threads reached "nearly 64 GB/s". A single core reached ~50 GB/s read. Latency exceeded 400 ns under heavy single-CCD load. Zen 2 desktop (3950X, DDR4) idle latency was 71.7 ns with 24-25 GB/s per core. — [Chips and Cheese, "Pushing AMD's Infinity Fabric to its Limits" (Nov 2024)](https://chipsandcheese.com/p/pushing-amds-infinity-fabric-to-its)
- Intel Xeon 6980P (dual socket, 24× DDR5-6400) with SNC enabled (3 nodes/socket): read-only bandwidth was ~188.6 GB/s intra-node, ~94.1 GB/s to another node in the same socket, and ~93.6 GB/s cross-socket. MLC "ALL Reads" peak for the whole system was ~1.13 TB/s. — [llama.cpp Discussion #12088, ubergarm (Feb-Mar 2025)](https://github.com/ggml-org/llama.cpp/discussions/12088)
- On that Xeon 6980P, DeepSeek R1 Q2_K_XL decode on one socket was 7.7 tok/s with SNC=Enable and 8.9 tok/s with SNC=Disable. Using both sockets degraded single-stream performance. Advice given: "as much RAM bandwidth crammed into a single NUMA node". — [llama.cpp Discussion #12088](https://github.com/ggml-org/llama.cpp/discussions/12088)
- Genoa NPS2 socket: 185.3 GB/s measured of 230.4 GB/s theoretical (80%). Turin NPS1 socket: 378.3 of 409.6 GB/s (92%). EPYC 9654 NPS1 socket: 359.9 of 460.8 GB/s (78%). All were measured with `likwid-bench` load (read) kernel. — [llama.cpp Discussion #11733, fairydreaming (2025-02-18)](https://github.com/ggml-org/llama.cpp/discussions/11733)

**Per-core concurrency (why limiting cores limits bandwidth)**
- Per-core bandwidth is limited by Little's law: "48*64 Bytes/130ns = 23.6 GB/s" per Sapphire Rapids core (48 outstanding L2 misses, ~130 ns latency). The measured single-core rate was ~19-23.6 GB/s. This is measured on Xeon Max with HBM, but the concurrency bound applies to DDR5 as well. — [McCalpin, "Bandwidth Limits in the Intel Xeon Max Processors", ISC 2023](https://www.ixpug.org/images/docs/ISC23/McCalpin_SPR_BW_limits_2023-05-24_final.pdf) [2023]
- SNC4 "Fewer average hops" / "No die-to-die crossings" reduces average latency. — [McCalpin ISC 2023](https://www.ixpug.org/images/docs/ISC23/McCalpin_SPR_BW_limits_2023-05-24_final.pdf)

### Inferences
- BIOS-only knobs: AMD NPS and Intel SNC need firmware access and a reboot. On bare-metal rentals they depend on the provider's BIOS choice; in VMs, the guest sees whatever vNUMA the provider exposes. Record `numactl -H` and `lscpu` on every rental.
- No-root knobs: `numactl --cpunodebind/--membind` (only useful if more than one node is exposed), `taskset`/`--cpu-mask` and llama.cpp `-t` / `--cpu-range`. On an AMD host where vCPU→CCD mapping is visible (check `/sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list`), pinning llama.cpp's CPU threads to one CCD caps reads at the GMI link. That is ~64 GB/s on GMI-narrow parts (like a single-CCD desktop Ryzen), or ~100 GB/s on GMI-wide parts. This setup is structurally close to a single-CCD desktop Ryzen: 8 cores, 32 MB L3, IF-limited reads. It still keeps server latency and lower server clocks. It is only valid if vCPUs are pinned 1:1 to physical cores, which many VMs do not guarantee.
- Thread limiting on Intel servers gives roughly 20-25 GB/s per core, so ~4 cores ≈ desktop bandwidth. The same change cuts compute to 4 cores, however, and Section 6 shows llama.cpp decode is not purely bandwidth-bound at low core counts. So a thread limit is a confounded bandwidth control, not an emulation of a desktop.
- Latency cannot be emulated downward. Desktop idle DRAM latency is ~70-85 ns; server latency is ~130-140 ns (NPS1/NPS4) and >220 ns (NPS0). MoE decode with small per-expert matrices has many synchronization points per token (Section 4), so the latency and core-to-core gap may matter beyond bandwidth. This is an unquantified bias that makes a bandwidth-matched server run slower than the desktop it is meant to match.

### Gaps
- No direct measurement was found of llama.cpp decode on a server restricted to a desktop-equivalent bandwidth, compared with an actual desktop at the same bandwidth. That is exactly the validation experiment this plan would need.
- Idle DDR5 latency for Intel Xeon 6 / SPR with DDR5 (not HBM) was not captured here. The McCalpin figure is for Xeon Max HBM.
- Dual-channel Intel desktop (Raptor/Arrow Lake) latency and bandwidth figures were not collected.
- The Chips and Cheese summary claimed Zen 4 dual-CCD bandwidth "exceeds 100 GB/s" with DDR5-5600 (89.6 GB/s theoretical). This is inconsistent with DRAM limits, likely a summarization error, and was not used.

## 3. Interference: co-running a bandwidth hog (e.g., STREAM on other cores) to leave a chosen bandwidth

### Takeaway
Co-runners that steal bandwidth are a recognized method, notably "Bandwidth Bandit" (CGO 2013), but they are used to measure sensitivity to contention, not to reproduce a smaller machine. They raise loaded latency, and the bandwidth left to llama.cpp is not set directly (it comes from arbitration). They also use cores and can pollute shared caches unless designed not to. The result is a "busy big server", which is not a desktop. This is less faithful than MBA or CCD pinning.

### Cited Findings
- Bandwidth Bandit "accesses data in carefully chosen sequential and random access patterns that avoid polluting the cache hierarchy", deliberately using "a very limited number of sets in the cache", to steal controlled bandwidth and profile whether applications are latency-sensitive or bandwidth-sensitive. It found "little correlation between bandwidth consumption and sensitivity". Published at CGO 2013 (companion PACT 2012). — [Uppsala UART, Bandwidth Bandit](https://www.it.uu.se/research/group/uart/measurement/shared_resource_sensitivity/bandwidth_bandit.html) [OLDER, 2012-2013, method still cited]
- In a quad-core test, applying MBA delays 80/90 to three interfering cores gave the victim the same latency as having only two unthrottled interferers. Interferer-based setups are therefore sensitive to the number and intensity of co-runners. — [arXiv 2206.14637](https://arxiv.org/pdf/2206.14637) [2022]
- Loaded latency rises steeply when multiple bandwidth threads compete (e.g., >400 ns on one Zen 4 CCD under 5 bandwidth threads). — [Chips and Cheese, Infinity Fabric](https://chipsandcheese.com/p/pushing-amds-infinity-fabric-to-its)

### Inferences
- A STREAM co-runner leaves llama.cpp a bandwidth share that depends on the relative request rates of the two programs. That share has to be measured during the run (MBM counters with resctrl, or `perf` uncore/DF counters, which usually need root or relaxed `perf_event_paranoid`). It cannot be set in advance.
- The co-runner imposes higher loaded latency on llama.cpp than a desktop at the same achieved bandwidth would see. A desktop running llama.cpp alone is also near saturation, so some queueing is realistic, but the server's queueing sits on top of its already higher idle latency. The bias direction is "server slower than an equivalent desktop".
- Useful as a sensitivity probe: sweep co-runner intensity, measure llama.cpp tok/s against llama.cpp's own achieved bandwidth, and check for linearity. It is not useful as a stand-alone emulation of a desktop.

### Gaps
- No LLM-inference study was found that used a co-runner to emulate lower bandwidth and validated it against real lower-bandwidth hardware.
- Intel MLC's loaded-latency (injection-delay) mode is a standard controlled-bandwidth generator. The MLC download page (v3.13, 2026-08-18) did not document its modes or privilege requirements, so details are unverified here — [Intel MLC](https://www.intel.com/content/www/us/en/download/736633/intel-memory-latency-checker-intel-mlc.html).

## 4. Normalization instead of emulation: does llama.cpp CPU token generation scale linearly with measured bandwidth?

### Takeaway
Within a single desktop, decode tok/s tracks measured bandwidth closely (R² > 0.97 across DDR5 4800-6200). Across platforms and at high bandwidth, the scaling breaks down for MoE. Small expert matrices add fixed per-token synchronization and compute costs, so doubling bandwidth with a second socket gave DeepSeek-style MoE only 3-11% more speed, while dense 70B gained 81-90%. Achieved fractions of measured bandwidth range from ~30% to ~95% depending on ISA and quant. A single-factor "tok/s ÷ GB/s" normalization is therefore biased in favor of low-bandwidth desktops. A two-term model (bytes/BW_eff + fixed overhead), fitted on the rental itself, is more defensible.

### Cited Findings
- On an i5-13600KF, DDR5 at 4800→6000→6200 MT/s moved AIDA-measured bandwidth from ~70,000 to ~87,000 MB/s. Decode rose 20-23% (Mistral 7B Q6_K: 9.42→11.34 tok/s; Llama 3.1 8B F16: 3.86→4.74 tok/s), with a reported linear fit R² > 0.97. — [Saplin, "DDR5 Speed, CPU and LLM Inference", dev.to (Oct 2024, edited Feb 2025)](https://dev.to/maximsaplin/ddr5-speed-and-llm-inference-3cdn)
- Dual-socket vs single-socket EPYC scaling, which doubles bandwidth: Llama-3.1 70B F16 gained 181-190%, Mixtral 8x22B Q8_0 146-185%, DeepSeek R1 103-111%. The poster attributes the poor scaling to small matrices (7168×2048): "additional synchronization overhead (barriers etc) completely negates" the gain. Large matrices (28672×8192) scale 163-187%, small ones 105-110%. — [llama.cpp Discussion #11733, fairydreaming (Feb 2025)](https://github.com/ggml-org/llama.cpp/discussions/11733) (NUMA effects are confounded with bandwidth here.)
- EPYC 9654 with 1× vs 2× DIMMs: 1.5 vs 4 tok/s on Llama-2 13B Q4_0 (anecdotal). A 32-core EPYC 7502P reached 7.85 tok/s. "Inference does not benefit from SMT, in fact it hurts it." — [llama.cpp Discussion #3167 (2023-2024)](https://github.com/ggml-org/llama.cpp/discussions/3167) [OLDER]
- Neoverse-N2 (96 cores, ~435 GB/s STREAM Triad): decode reached only ~30-63% of STREAM (UD-IQ2_M 129 GB/s effective, Q4_K 236 GB/s, Q4_K_M 275 GB/s). Throughput kept rising past the 32-thread bandwidth saturation point up to 64 threads, with ">99% of cycles in ggml compute". The issue claims x86 reaches 85-95% on identical workloads (asserted, not independently verified). — [llama.cpp Issue #25976](https://github.com/ggml-org/llama.cpp/issues/25976)
- Xeon 6980P single socket (SNC off), DeepSeek R1 Q2_K_XL at 8.9 tok/s: the poster notes a ~19 tok/s theoretical ceiling from 225 GB/s, so actual decode is "substantially below" it. — [llama.cpp Discussion #12088](https://github.com/ggml-org/llama.cpp/discussions/12088)
- Community hybrid-offload rigs describe expert decode as "dominated by the CPU reading expert weights at RAM bandwidth while the GPUs idle" (e.g., 2×RTX 3090 + 8-channel DDR4 ~116 GB/s), but give no measured GB/s for the expert matmuls. — [llama.cpp Discussion #24528 (June-Aug 2026)](https://github.com/ggml-org/llama.cpp/discussions/24528)

### Inferences
- Model: t_token ≈ Σ(CPU expert bytes)/BW_eff + t_fixed. BW_eff is itself capped by per-core compute when there are few cores or a weak ISA, and t_fixed covers barriers, GPU layers, PCIe and sampling. On a desktop, t_fixed is a small share; on a 400+ GB/s server the bandwidth term shrinks and t_fixed dominates. Dividing server tok/s by a bandwidth ratio therefore under-predicts desktop speed if the server is scaled down, or equivalently overstates how "bandwidth-efficient" the desktop is.
- The Saplin data (fitted over only ~70-87 GB/s) supports linearity locally at desktop bandwidths. It does not support extrapolating from 400 GB/s down to 80 GB/s.
- Worked check from the Saplin data: 5.94 GB × 11.34 tok/s ≈ 67 GB/s effective against ~87 GB/s AIDA, about 77%. AIDA's metric (read vs copy) was not specified in the summary.
- Recommended protocol (inference): on each rental, (a) measure a read-only bandwidth curve against thread count with the same pinning as llama.cpp; (b) run llama.cpp at several thread counts, and with MBA/CCD pinning where available, to get tok/s at several achieved bandwidths; (c) fit t_token = a·bytes + b; (d) compare the fitted `a` (implied BW_eff) and `b` to published desktop points rather than dividing raw tok/s. Where possible, report the CPU-expert-only time (from per-op profiling) and normalize that rather than end-to-end tok/s, since GPU/PCIe time does not scale with host bandwidth.

### Gaps
- No systematic public dataset was found that pairs llama.cpp decode tok/s with measured (not nominal) bandwidth across many desktops and servers for the same MoE model and quant.
- Data specific to `--n-cpu-moe` (hybrid GPU + CPU experts) showing the achieved CPU expert bandwidth in GB/s was not found; Discussion #24528 explicitly lacks it.
- The "85-95% on x86" figure in Issue #25976 is unsourced within the issue.

## 5. Which microbenchmark best predicts llama.cpp's achieved CPU bandwidth (STREAM Triad vs Copy vs pure-read vs Intel MLC vs likwid)?

### Takeaway
llama.cpp decode almost only reads weights, so the right comparator is a read-only kernel (likwid-bench `load`, MLC "ALL Reads", or a custom read kernel) run with the same thread count and pinning. STREAM under-reports actual bus traffic: it ignores write-allocate, so the real traffic is 1.5× the reported figure for Copy and 1.33× for Triad. STREAM Copy compiled with GCC may also be silently turned into `memcpy` with non-temporal stores. That can explain the project's A100-VM result, where llama.cpp beat STREAM Triad and roughly matched Copy.

### Cited Findings
- STREAM counts "16 bytes per iteration for Copy and Scale, and 24 bytes per iteration for Add and Triad", assuming no write-allocate. With write-allocate the real traffic is ×1.5 (Copy/Scale) and ×1.33 (Add/Triad). Recommendations: use `-ffreestanding` "to prevent memcpy() substitution", pin threads (likwid-pin), and size arrays ≥4× the last-level cache. Non-temporal stores avoid write-allocate (icc emits them; GCC does not by default). "STREAM is not an automatic benchmark framework that you can run and expect to give the right answer." — [Georg Hager's Blog, "The McCalpin STREAM benchmark: How to do it right and interpret the results"](https://blogs.fau.de/hager/archives/8263) [2019]
- Read-only and STREAM results diverge by memory type. Xeon Max HBM showed "3.5x higher STREAM BW" but only "2.3x higher Read BW" than DDR5, so STREAM-to-read ratios are platform dependent. — [McCalpin ISC 2023](https://www.ixpug.org/images/docs/ISC23/McCalpin_SPR_BW_limits_2023-05-24_final.pdf)
- llama.cpp community CPU-inference bandwidth studies used likwid-bench `load` (read) as the reference; see the Genoa/Turin table in Section 2 at 78-92% of theoretical. — [llama.cpp Discussion #11733](https://github.com/ggml-org/llama.cpp/discussions/11733)
- Xeon 6980P characterizations used Intel MLC ("ALL Reads" ~1.13 TB/s, plus per-node read-only matrices). — [llama.cpp Discussion #12088](https://github.com/ggml-org/llama.cpp/discussions/12088)
- The Neoverse-N2 analysis used STREAM Triad as its denominator (435 GB/s). That makes "% of bandwidth" figures from different sources non-comparable unless the kernel is stated. — [llama.cpp Issue #25976](https://github.com/ggml-org/llama.cpp/issues/25976)
- Intel MLC current version is 3.13 (2026-08-18). — [Intel MLC download page](https://www.intel.com/content/www/us/en/download/736633/intel-memory-latency-checker-intel-mlc.html)

### Inferences
- Why llama.cpp can exceed STREAM Triad: with write-allocate, Triad's reported GB/s is about 3/4 of real traffic, and the mixed read/write stream also costs DRAM turnaround efficiency. llama.cpp weight streaming is ~100% reads. If the project's STREAM binary let GCC turn Copy into `memcpy` (non-temporal stores, no write-allocate), then reported Copy ≈ real traffic, which is why Copy ≈ llama.cpp. Check by rebuilding STREAM with `-ffreestanding` or inspecting the assembly for `memcpy` calls. A pure-read kernel should come out at or above both.
- Best predictor for this workload: a read-only kernel with the same number of threads, the same core pinning and the same NUMA policy as llama.cpp's CPU threads, preferably swept over thread count. Candidates:
  - `likwid-bench -t load_avx512` / `load_avx`. This is a userspace benchmark; the likwid-perfctr counters need MSR/perf access, but likwid-bench does not.
  - MLC `--max_bandwidth` / "ALL Reads".
  - A few-line OpenMP read-reduction kernel.

  Also report STREAM Triad (with the write-allocate caveat) for comparability with the literature.
- For both desktop reference points and rentals, record the measured read bandwidth, not the nominal channels × MT/s figure. Single-CCD Ryzen desktops are capped near ~62-64 GB/s by the IF link regardless of DDR5 speed (Section 2), so nominal dual-channel DDR5-6000 (96 GB/s) would misstate their bandwidth by ~50%.

### Gaps
- No published side-by-side of llama.cpp achieved bandwidth against STREAM Copy, Triad and read-only on the same machine was found. The project's own A100-VM result appears to be novel evidence.
- The Intel community thread "MLC v3.6 Single-core 'ALL Reads' bandwidth lower than 'Stream-triad like' BW" suggests single-core anomalies but was not read — [Intel Community](https://community.intel.com/t5/Software-Tuning-Performance/Intel-MLC-v3-6-Single-core-quot-ALL-Reads-quot-bandwidth-lower/td-p/1164540).
- Whether MLC runs correctly without root, and on AMD CPUs, was not verified (the download page is silent).

## 6. Does core count or ISA matter separately from bandwidth for llama.cpp MoE on CPU?

### Takeaway
Yes. Decode is not purely bandwidth-bound at low core counts or with weaker SIMD paths. Desktop evidence shows 3 threads giving 73% of 6-thread speed, and server/Arm evidence shows speed still rising after bandwidth saturates. MoE's small matrices add synchronization overhead that grows with thread count. AMX gave only ~5% on Q8_0 decode. A rental with the right bandwidth but a different core count or ISA than the desktop is therefore not equivalent, and thread count must be swept.

### Cited Findings
- i5-13600KF (desktop, DDR5): "3 cores/threads demonstrated 73% of 6 cores/threads speed". 12 threads gave only a "7.6% boost over 6 core baseline". — [Saplin, dev.to (Oct 2024)](https://dev.to/maximsaplin/ddr5-speed-and-llm-inference-3cdn)
- Xeon 6985P-C (GCE, 48 vCPU, HT off, Jan 2026): decode speedup from 4 to 24 threads was 3.2-4.4× (54-73% parallel efficiency). Gains shrink at 16-24 threads. "Q8_0-style quantization typically aligns better with x86 acceleration" (AVX-512/AMX). — [Intel Community blog, "Optimizing SLMs on Intel Xeon Processors: A llama.cpp Performance Study" (2026-01-21)](https://community.intel.com/t5/Blogs/Tech-Innovation/Artificial-Intelligence-AI/Optimizing-SLMs-on-Intel-Xeon-Processors-A-llama-cpp-Performance/post/1734305)
- Neoverse-N2: decode throughput kept rising past bandwidth saturation (32 threads) to a peak at 64 threads. Profiling put >99% of cycles in ggml compute, with GEMV kernels (`ggml_gemv_q8_0_4x8_q8_0` 34%, `ggml_gemv_q4_K_8x8_q8_K` 21%) dominating. Lower-bit quants (IQ2_M) got a lower bandwidth fraction (~30%) than Q4_K_M (~63%), i.e., dequant cost per byte matters. — [llama.cpp Issue #25976](https://github.com/ggml-org/llama.cpp/issues/25976)
- Xeon 6980P: explicitly enabling AMX "possibly showed about 5% improvement for Q8" (5.43 → 5.63 tok/s). ik_llama.cpp was ~12% faster than mainline on the same hardware, so kernel implementation also matters. — [llama.cpp Discussion #12088](https://github.com/ggml-org/llama.cpp/discussions/12088)
- Synchronization overhead on small MoE matrices "completely negates" the gains from extra hardware. — [llama.cpp Discussion #11733](https://github.com/ggml-org/llama.cpp/discussions/11733)
- SMT hurts: use physical-core counts. — [llama.cpp Discussion #3167](https://github.com/ggml-org/llama.cpp/discussions/3167) [OLDER]

### Inferences
- Record for every rental and every reference desktop: physical cores used by llama.cpp's CPU threads, the ISA path compiled/detected (llama.cpp prints `system_info` with AVX2/AVX512/AVX512_VNNI/AMX flags), CPU clock under load, and the quant type of the CPU-resident experts. Low-bit quants (IQ2/IQ3) are more likely to be compute-bound per byte than Q4_K/Q8_0.
- Low server clocks plus high latency mean a bandwidth-matched server slice with the same core count as a desktop will likely be slower per core. The fair comparison is "tok/s at the thread count that maximizes tok/s", measured against a read-bandwidth curve at those same thread counts.
- Many published desktop results come from Intel client CPUs without AVX-512 or from Zen 4/5 desktops with AVX-512; servers usually have AVX-512 (and AMX on Intel). The ISA mismatch should be treated as a covariate. This is general knowledge and was not sourced in this pass.

### Gaps
- No controlled study was found that isolates ISA (AVX2 vs AVX-512 vs AMX) at fixed bandwidth for llama.cpp MoE decode.
- No quantitative data was found on how barrier/synchronization cost in llama.cpp's CPU threadpool scales with thread count and core-to-core latency (server mesh/IF vs desktop ring). That is the likely mechanism behind the poor MoE bandwidth scaling but remains unmeasured.
- Clock-frequency effects (server all-core ~2-3.x GHz vs desktop 5+ GHz) on decode at fixed bandwidth: no source found.

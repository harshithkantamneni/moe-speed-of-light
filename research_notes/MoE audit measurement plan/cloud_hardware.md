# Renting desktop-class consumer/workstation GPU hosts (Sept 2026)

Method note: besides web sources, the Vast.ai public offers endpoint (`https://console.vast.ai/api/v0/bundles/`, no login needed) was queried on 2026-09-27 for single-GPU, rentable, on-demand offers of each target GPU. The API returns about 64 offers per query, so several sort orders were merged: 905 unique offers in total (299 RTX 4090, 320 RTX 5090, 228 RTX 3090, 15 RTX 4080/4080S, 10 RTX A6000, 22 A100). The raw trimmed snapshot is saved next to this file as `vast_snapshot_2026-09-27.json`. CPUs were sorted into classes with a regex. DESKTOP means Ryzen (not Threadripper), Core i3/i5/i7/i9 or Core Ultra. HEDT means Threadripper, Intel X-series or Xeon W. SERVER means EPYC or Xeon Scalable/E5. Vast marketplace offers change hour to hour, so all counts are a single point-in-time snapshot. Citations to "[Vast API snapshot]" refer to that query.

## 1. Which providers rent these GPUs by the hour in 2026, and at what price?

### Takeaway
Vast.ai has the widest hourly supply of RTX 4090/5090/3090 by far, and the cheapest desktop-class hosts: about $0.34–0.60/h for a 4090 and $0.43–0.83/h for a 5090 on a desktop CPU. RunPod, TensorDock, CloudRift and Salad also rent consumer cards. Hyperbolic currently rents only H100/H200/B200, and Lambda's public pricing page no longer lists an A6000.

### Cited Findings
- **Vast.ai, all single-GPU on-demand offers.**
  - RTX 4090: min $0.136, median $0.496/h (n=299).
  - RTX 5090: min $0.336, median $0.629/h (n=320).
  - RTX 3090: min $0.121, median $0.206/h (n=228).
  - RTX 4080: median $0.269 (n=11).
  - RTX A6000: $0.287–0.537, median $0.456 (n=10).
  - A100 40 GB (PCIe or SXM4 slices): $0.43–0.93/h.
  - Source: [Vast API snapshot](https://console.vast.ai/api/v0/bundles/)
- **Vast.ai, desktop-CPU hosts only, whole machine rented (gpu_frac = 1).**
  - RTX 4090 with at least 64 GB RAM: 52 offers, min $0.336, median $0.489/h. Verified hosts with reliability of at least 0.98: 22 offers, median $0.482.
  - RTX 5090 with at least 64 GB RAM: 56 offers, min $0.433, median $0.626/h. Verified with reliability of at least 0.98: 22 offers, median $0.61.
  - RTX 3090 with at least 64 GB RAM: 12 offers, min $0.149, median $0.335/h.
  - Source: [Vast API snapshot](https://console.vast.ai/api/v0/bundles/)
- **Vast's own pricing pages** advertise RTX 4090 "from $0.14/hr" and RTX 5090 "from $0.27/hr". A 2026-09-27 multi-cloud price scrape likewise found the 4090 floor at $0.14/h and the 5090 floor at $0.21/h, both on Vast. — [Vast 4090 page](https://vast.ai/pricing/gpu/RTX-4090); [Vast 5090 page](https://vast.ai/pricing/gpu/RTX-5090); [fastgpu dev.to, Sep 2026](https://dev.to/fastgpu/what-it-costs-to-rent-an-h100-b200-or-rtx-4090-in-september-2026-live-prices-from-28-gpu-clouds-12n3)
- **Watch out for modded 48 GB RTX 4090s.** On Vast, 32 of the 299 "RTX 4090" offers are 48 GB cards (gpu_ram about 49 GB), median $0.674/h, mostly in California, Washington and Japan. Filter on gpu_ram < 30 GB to get stock 24 GB cards. — [Vast API snapshot](https://console.vast.ai/api/v0/bundles/)
- **RunPod pricing page** (fetched 2026-09-27) lists the following. The page summary labelled these "Community Cloud", but the numbers match the *new Secure Cloud* rates reported below, so the tier is ambiguous.

  | GPU | Price/h | vCPU | RAM |
  |---|---|---|---|
  | RTX 5090 | $0.99 | 9 | 35 GB |
  | RTX 4090 | $0.74 | 6 | 41 GB |
  | RTX 3090 | $0.50 | 16 | 125 GB |
  | RTX A6000 | $0.53 | 9 | 50 GB |
  | A100 PCIe 80 GB | $1.59 | – | – |
  | A100 SXM 80 GB | $1.59 | – | – |

  Per-second billing. — [RunPod pricing](https://www.runpod.io/pricing)
- **RunPod raised Secure Cloud prices on 2026-09-20.** Examples: RTX 4090 $0.69 → $0.74, A100 PCIe $1.39 → $1.59. RTX 3090, A6000 and 5090 did not change. Community Cloud prices were unchanged, and Community is now "cheaper than Secure Cloud on all 21 GPU types, by 8 percent to 56 percent". — [UsagePricing, 2026-09-20](https://www.usagepricing.com/blueprint/activity/runpod-2026-09-20-secure-cloud-price-hike)
- **An aggregator lists RunPod's RTX 5090 at $0.69/h** on-demand (9 vCPU / 35 GB). This conflicts with the $0.99 on RunPod's page and is probably the Community vs Secure difference. A third-party blog title claims the 4090 costs $0.34/h on Community Cloud (not verified). — [getdeploying RTX 5090, updated 2026-09-28](https://getdeploying.com/gpus/nvidia-rtx-5090); [synpixcloud](https://www.synpixcloud.com/blog/rtx-4090-cloud-rental-worth-it)
- **Other RTX 5090 hourly sellers (aggregator, 2026-09-28):**

  | Provider | Price/h | vCPU | RAM |
  |---|---|---|---|
  | HyperAI | $0.35 | 16 | 40 GB |
  | GPUhub | $0.46 | 25 | 90 GB |
  | Vast | $0.41 | 24 | 63 GB |
  | Lium | $0.53 | 24 | 63 GB |
  | SwissGPU | $0.60 | 16 | 32 GB |
  | Nova Cloud | $0.66 | 22 | 84 GB |
  | Runcrate | $0.72 | 12 | 120 GB |
  | Sesterce | $0.71 | 12 | 120 GB |
  | Oblivus | $0.90 | 14 | 80 GB |
  | Salad | $0.50 on-demand / $0.25 lowest priority | 4 | 8 GB |

  — [getdeploying RTX 5090](https://getdeploying.com/gpus/nvidia-rtx-5090)
- **TensorDock.** Aggregator listing: RTX 4090 $0.46/h, RTX 3090 $0.31, A6000 $0.56, A100 PCIe 80 GB $1.61 (shown with 16 vCPU / 32 GB), no RTX 5090. TensorDock describes itself as a marketplace of independent hosts in 100+ locations with per-second billing (listing updated 2026-09-18). TensorDock's own 4090 page says "from $0.37/hr" on-demand and $0.20 spot. — [getdeploying TensorDock](https://getdeploying.com/tensordock); [TensorDock 4090 page](https://www.tensordock.com/gpu-4090.html)
- **CloudRift.** RTX 4090 $0.39/h, RTX 5090 $0.60/h, A100 SXM4 80 GB $1.05/h. Per-second billing, "no minimum spend". The pricing page does not state vCPU, RAM or host CPU. — [CloudRift pricing](https://www.cloudrift.ai/pricing)
- **SaladCloud** (new prices effective 2026-09-12, four priority tiers from High to Lowest):

  | GPU | Price/h (High → Lowest) |
  |---|---|
  | RTX 5090 | $0.500 → $0.250 |
  | RTX 4090 | $0.330 → $0.160 |
  | RTX 4080 | $0.230 → $0.110 |
  | RTX 3090 | $0.170 → $0.090 |

  GPU container groups are "not charged for vCPU or RAM". — [Salad blog, price changes Sept 2026](https://blog.salad.com/saladcloud-price-changes-september-2026/)
- **Lambda.** The on-demand pricing page (fetched 2026-09-27) lists only B200 ($6.69), H100 SXM ($3.99), A100 SXM 80 GB ($2.79), A100 SXM 40 GB ($1.99/GPU-h, shown in a 124 vCPU / 1,800 GiB multi-GPU configuration) and V100 ($0.79). No RTX A6000, A10 or single-GPU consumer cards appear. — [Lambda pricing](https://lambda.ai/pricing)
- **Hyperbolic** currently lists only H100 SXM ($3.19), H200 ($3.99) and B200 ($5.99). No RTX 4090/5090/3090/A6000/A100 (updated 2026-09-27). — [ComputePrices: Hyperbolic](https://computeprices.com/providers/hyperbolic)

### Inferences
- For the target GPUs, Vast.ai is the only marketplace with enough desktop-class supply to choose a host platform deliberately: about 100 desktop-CPU offers each for the 4090 and 5090. RunPod, TensorDock and CloudRift sell a GPU SKU, not a host class.
- The A6000 and A100 40 GB are now mostly server-hosted and scarce. Vast had 10 A6000 and about 14 A100-40 GB single-GPU offers, nearly all on Xeon Gold or EPYC. They work as "server host" controls, not as desktop reproductions.

### Gaps
- RunPod Community Cloud prices for the 4090/5090/3090/A6000 could not be read directly: the pricing page render was ambiguous about tier. Check the RunPod console, where the user already has an account.
- Not researched in the budget: Paperspace/DigitalOcean (historically A6000/A4000, no 4090) and Shadeform (mostly a data-center GPU aggregator). Treat both as unverified for Sept 2026.
- Whether Lambda still sells single-GPU A6000 or A100 40 GB instances in its console: the public pricing page no longer shows them.

## 2. Can the renter see host CPU, RAM, memory channels/speed, PCIe and disk before renting?

### Takeaway
Only Vast.ai shows the host in detail before renting. You see the CPU model, core count, RAM, motherboard, PCIe generation, lanes and measured bandwidth, disk model and bandwidth, network speed, reliability and verification status, all searchable. No provider exposes DIMM count, channel count or memory speed. Those have to be inferred from CPU and motherboard, then measured after launch.

### Cited Findings
- Vast's search API exposes these fields, all filterable: `cpu_name`, `cpu_cores`, `cpu_cores_effective`, `cpu_ghz`, `cpu_ram`, `gpu_frac`, `gpu_lanes`, `pci_gen`, `pcie_bw`, `dlperf`, `mobo_name`, `disk_bw`, `disk_space`, `inet_down`/`inet_up`, `reliability`, `verified`, `vms_enabled` and datacenter/hosting type. No memory-channel or DIMM-speed field is listed. — [Vast docs: search offers](https://docs.vast.ai/api-reference/search/search-offers)
- Actual API records include measured values such as `pcie_bw` (for example 47.8 GB/s median on desktop RTX 5090 hosts running PCIe 5.0 x16; 22.6 GB/s median on desktop 4090 hosts), `disk_bw`, `inet_down` and `mobo_name` (for example "ProArt X870E-CREATOR WIFI", "B650 EAGLE AX", "Z790 UD"). — [Vast API snapshot](https://console.vast.ai/api/v0/bundles/)
- **What `cpu_ram` means is unclear.**
  - The docs describe it as the machine's total RAM. That is how the doc page was summarized; the wording was not seen directly.
  - The API data suggests it is the offer's share of the machine. An EPYC 7B13 host offered at gpu_frac 0.083 shows 52 GB, and a Ryzen 9 5950X at gpu_frac 0.5 shows 31 GB. `cpu_cores_effective` is also exactly gpu_frac × cpu_cores.
  - Sources: [Vast docs](https://docs.vast.ai/api-reference/search/search-offers) vs [Vast API snapshot](https://console.vast.ai/api/v0/bundles/)
- RunPod publishes one typical vCPU/RAM figure per GPU type (for example 4090 = 6 vCPU / 41 GB; 3090 = 16 vCPU / 125 GB). Its docs "Choose a Pod" page gives sizing guidance but does not describe any CPU-model visibility or filter. — [RunPod pricing](https://www.runpod.io/pricing); [RunPod docs: choose a pod](https://docs.runpod.io/pods/choose-a-pod)
- TensorDock advertises "à la carte resource allocation" of vCPU, RAM and storage on KVM VMs. Its 4090 page names the host as "2x AMD EPYC 75F3 with 128 combined threads". — [TensorDock 4090 page](https://www.tensordock.com/gpu-4090.html)
- Salad lets you request vCPU (1–16) and RAM (1–60 GB) per container group. The docs do not describe selecting nodes by CPU or RAM spec, and when several GPU classes are selected "the system will assign the first available GPU class". — [Salad docs: container groups](https://docs.salad.com/container-engine/explanation/container-groups/container-groups)
- CloudRift's and Hyperbolic's public pricing pages give no host CPU details. — [CloudRift pricing](https://www.cloudrift.ai/pricing); [ComputePrices: Hyperbolic](https://computeprices.com/providers/hyperbolic)

### Inferences
- Channel count can be inferred reliably from the CPU platform (general platform knowledge, not sourced here):
  - Ryzen AM4/AM5, Intel LGA1700/1851 (12th–14th gen, Core Ultra 200): 2 channels.
  - Threadripper (non-PRO): 4 channels.
  - Threadripper PRO: 8 channels.
  - EPYC Rome/Milan: 8 per socket. EPYC Genoa/Turin: 12 per socket.
  - Xeon Scalable: 6–8 per socket.
- Memory *speed* cannot be inferred and matters a lot. For example, a desktop at 126 GB or 189 GB is almost certainly 4 DIMMs (2 DIMMs per channel), which usually runs slower than 2-DIMM configurations. Plan to run a STREAM or Intel MLC style bandwidth test in the first minutes of each rental, and release hosts that fall outside the target band. Per-second billing makes this cheap.
- `mobo_name` is a useful secondary check: B650/X670/X870/B850/Z790/Z890 boards confirm a desktop platform.
- In unprivileged containers, `dmidecode` (DIMM speed and population) will probably fail. That is expected but not tested.

### Gaps
- No provider was found that lists memory speed or DIMM population before rental.
- Could not confirm whether RunPod's console lets you filter Community Cloud machines by minimum RAM or vCPU, or shows the CPU model, before deploying. From the documentation found, the host CPU appears only after launch (`lscpu` inside the pod).

## 3. Are consumer-GPU hosts real desktops (dual-channel) or servers (EPYC/Xeon)?

### Takeaway
Hosts are mixed. On Vast, roughly a third of 4090/5090 offers sit on real desktop platforms (Ryzen 7000/9000, Core i9-13/14th gen, Core Ultra 9) and more than half on EPYC/Xeon servers. Most server offers are slices of multi-GPU machines, where other tenants share the memory bus. Salad is the only provider whose fleet is by definition consumer gaming PCs, but it caps RAM at 60 GB and runs containers inside WSL2 on Windows. TensorDock's showcased hosts are dual-EPYC servers.

### Cited Findings
- **Vast host-class counts for single-GPU offers** (desktop = D, HEDT = H, server = S). Source: [Vast API snapshot](https://console.vast.ai/api/v0/bundles/)
  - RTX 4090: D 98, H 22, S 167, unknown 12.
  - RTX 5090: D 117, H 23, S 170+, unknown 10.
  - RTX 3090: D 46, H 22, S 151.
  - RTX 4080/4080S: D 7, H 2, S 6.
  - RTX A6000: 9 of 10 server or Xeon W. The one "desktop" is an i9-10940X, which is actually quad-channel HEDT.
  - A100: all server.
- **Desktop-CPU platform mix.** Source: [Vast API snapshot](https://console.vast.ai/api/v0/bundles/)
  - RTX 5090: AM5 Ryzen 7000/9000 81, Intel 12th–14th gen / Core Ultra 23, AM4 10. Most common CPUs: Ryzen 7 7800X3D (19), Ryzen 9 9950X (12), Ryzen 9 7950X (11).
  - RTX 4090: AM4 37, AM5 30, Intel 12th–14th gen / Ultra 17, older Intel 14.
- **Server offers are mostly multi-GPU slices** (gpu_frac < 1): 143 of 167 server-class 4090 offers, 152 of 170 for the 5090, 136 of 151 for the 3090. Typical server CPUs are EPYC 7B13/7K62/7702/7B12/7C13 and Xeon Platinum 8352V. — [Vast API snapshot](https://console.vast.ai/api/v0/bundles/)
- **Desktop hosts are mostly whole machines.** 84 of 97 desktop-CPU 4090 offers and 99 of 116 desktop-CPU 5090 offers have gpu_frac = 1 (the renter gets the whole box). — [Vast API snapshot](https://console.vast.ai/api/v0/bundles/)
- **Vast hosts range widely:** "from tier 4 datacenters with extensive physical and operational security down to individual hobbyists renting out a few machines in their home". — [Vast FAQ](https://cdn.vast.ai/faq/)
- **Salad nodes are "consumer gaming PCs"**, and "the container workload jobs are run inside of WSL". — [Salad support: What is WSL](https://support.salad.com/article/265-what-is-wsl)
- **TensorDock's 4090 page** describes 8×4090 servers on "2x AMD EPYC 75F3". — [TensorDock 4090 page](https://www.tensordock.com/gpu-4090.html)
- **RunPod** describes Community Cloud as "individual compute providers" connected peer-to-peer, and Secure Cloud as "T3/T4 data centers". It does not say what CPUs back either. — [RunPod docs: pods overview](https://docs.runpod.io/pods/overview)

### Inferences
- Reproducing the papers' desktop dual-channel host class is practical only on Vast, filtering on `cpu_name` plus `gpu_frac = 1`. On RunPod, TensorDock or CloudRift the host class is luck of the draw, and it leans toward servers.
- Server slices are a double confound for host-RAM-bound MoE decoding:
  - They have 8–12 channels, so much higher aggregate bandwidth than the papers' desktops.
  - Neighbouring tenants on the same machine share that bandwidth, so results are noisy.
  - If a server contrast is wanted, rent the whole machine or use a single-GPU server.
- Salad matches the "real home desktop" class best, but the WSL2/Hyper-V layer, the 60 GB RAM cap, and no choice of CPU or memory make it poorly suited to controlled measurements.

### Gaps
- No community write-up or benchmark study was found that reports the actual host CPU or RAM behind RunPod Community Cloud 4090/5090 pods in 2026. Searches of Reddit and forum content returned nothing specific.
- No published LLM-inference benchmarking study was found that rented consumer GPUs and reported host memory bandwidth.

## 4. Maximum host RAM with a single RTX 4090/5090 on each provider

### Takeaway
On Vast, desktop hosts top out at about 186–189 GB (4 × 48 GB DDR5 on Z790/Z890/X870E) and commonly offer 123–126 GB. That covers the 32–128 GB requirement on genuine dual-channel platforms. Salad caps at 60 GB, and RunPod lists 35–41 GB for its 4090/5090 SKUs. Server slices can go higher but are not desktop-class.

### Cited Findings
- **Largest desktop-CPU RTX 4090 hosts on Vast:**
  - Core i9-14900KF, Z790 UD, 189 GB, PCIe 4.0 x16, $0.601/h, unverified, Romania.
  - Core Ultra 9 285K, Z890, 188 GB, $0.601/h, unverified, Romania.
  - Ryzen 9 5950X, B550, 126 GB, $0.496, Finland. Downlink only 49 Mbps.
  - Core i9-10900KF, 126 GB, $0.468, verified, Japan.
  - Core i9-14900K, 126 GB, $0.402, verified, India.
  - Ryzen 5 3600, 126 GB, $0.376.
  - At least 120 GB: 12 whole-machine desktop offers, median $0.549/h. Only 2 of those are verified with reliability of at least 0.98.
  - Source: [Vast API snapshot](https://console.vast.ai/api/v0/bundles/)
- **Largest desktop-CPU RTX 5090 hosts on Vast:**
  - Ryzen 9 9950X, ProArt X870E, 186 GB, PCIe 5.0 x16 at 49.6 GB/s, $0.827/h, verified, Netherlands.
  - Core Ultra 9 285K, 188 GB, $0.868, unverified.
  - Ryzen 9 7950X, 125 GB, $0.669, verified, Korea.
  - Ryzen 9 7900X, B850 AI TOP, 125 GB, $0.499, unverified, New Jersey.
  - Several Ryzen 9 9950X / 9700X hosts at 123 GB for $0.60–0.80.
  - At least 120 GB: 14 offers, median $0.672/h.
  - Source: [Vast API snapshot](https://console.vast.ai/api/v0/bundles/)
- **RTX 3090 desktop hosts on Vast** max out at about 94 GB (i9-14900KF). There are no desktop 3090 offers with 120 GB or more. — [Vast API snapshot](https://console.vast.ai/api/v0/bundles/)
- **Server-class single-GPU offers can be much larger**, for example a 4090 on an EPYC 7773X with 882 GB ($2.32/h) or 5090 slices on EPYC 7B13 with 252 GB ($0.67–1.02/h). — [Vast API snapshot](https://console.vast.ai/api/v0/bundles/)
- **RunPod:** RTX 4090 is listed with 41 GB RAM, RTX 5090 with 35 GB, RTX 3090 with 125 GB, A6000 with 50 GB. — [RunPod pricing](https://www.runpod.io/pricing)
- **Salad** has a hard maximum of "Memory (RAM) (1-60GB)" per container. — [Salad docs](https://docs.salad.com/container-engine/explanation/container-groups/container-groups)
- **Aggregator RAM figures** for other 5090 sellers: Runcrate/Sesterce 120 GB, GPUhub 90 GB, Nova Cloud 84 GB, Oblivus 80 GB. Host CPUs are not stated. — [getdeploying RTX 5090](https://getdeploying.com/gpus/nvidia-rtx-5090)

### Inferences
- Models needing up to about 96 GB of host RAM can run on many desktop 4090/5090 hosts. Models needing 120–128 GB have only a handful of desktop candidates: about 12–14 per GPU type, several unverified. Plan a fallback, either a large-RAM Threadripper (HEDT, 4–8 channels, flagged as not desktop-class) or accept a 4-DIMM desktop running at reduced memory speed.
- RunPod's 4090/5090 SKUs, at around 40 GB, fall below the experiment's lower RAM requirement for the larger models.

### Gaps
- TensorDock and CloudRift maximum RAM per single consumer GPU was not found on public pages.

## 5. Access level: container vs VM, clock locking, governor, numactl, counters

### Takeaway
RunPod pods and Vast's default instances are unprivileged Docker containers. You are root inside the container but cannot lock GPU clocks, and GPU and uncore counters are generally unavailable. Vast "VM" instances (about 7% of offers, including a few desktop-CPU 4090/5090 hosts) and TensorDock/Lambda KVM VMs give real root in a guest OS. Clock locking and ncu are then plausible, but the host CPU governor and DRAM counters remain out of reach on every rental option found.

### Cited Findings
- **Vast default instances:** "Clients are isolated to unprivileged docker containers and only have access to their own data." — [Vast FAQ](https://cdn.vast.ai/faq/)
- **Vast VM instances** provide "full system access and hardware counters" for "CUDA Performance Profiling", plus ptrace, kernel modules and custom drivers. The trade-offs are "Longer instance creation and boot times", "Higher disk space requirements" and "Limited machine selection". — [Vast docs: Linux virtual machines](https://docs.vast.ai/linux-virtual-machines)
- **Vast VM-enabled supply is small.** In the snapshot, 66 of 905 single-GPU offers are VM-enabled (37 are 4090, 19 are 5090, 9 are 3090). Desktop-CPU examples:
  - 4090 on Ryzen 5 9600X, 62 GB, $0.456, verified.
  - 4090 on Ryzen 5 7600X, 62 GB, $0.459, verified.
  - 4090 on Ryzen 5 3600, 126 GB, $0.376.
  - 5090 on Core i5-12400F, 94 GB, $0.604, verified.
  - 5090 on Ryzen 9 5950X, 126 GB, $0.562.
  - 5090 on Ryzen 9 7900X, 93 GB, $0.711.
  - Source: [Vast API snapshot](https://console.vast.ai/api/v0/bundles/)
- **RunPod pods are Docker containers** ("Runpod runs Docker for you"). — [RunPod docs: pods overview](https://docs.runpod.io/pods/overview)
- **Clock locking is root-only.** An open-source benchmarking PR states that `nvidia-smi -pm`, `-lgc`, `-ac` and `-pl` are root-only, and that "RunPod Pods, Vast's Docker and Modal's gVisor sandbox cannot pin a clock". The author notes this was exercised only against a fake shell, not real hardware. — [AdvayMonga/inference-server PR #30](https://github.com/AdvayMonga/inference-server/pull/30)
- **TensorDock** provides KVM virtualization with "root access and a dedicated GPU passed through". — [TensorDock 4090 page](https://www.tensordock.com/gpu-4090.html)
- **Salad** runs workloads inside WSL on Windows consumer PCs. — [Salad support: What is WSL](https://support.salad.com/article/265-what-is-wsl)

### Inferences
- **In a container on any provider:**
  - `numactl` runs, but desktop hosts are single-NUMA anyway.
  - Changing the CPU frequency governor and `nvidia-smi -lgc` will fail.
  - `perf` uncore/IMC counters (for direct DRAM-bandwidth measurement) will be blocked.
  - Measure host bandwidth with a user-space benchmark (STREAM or MLC-style) rather than counters.
- **In a VM (Vast VM, TensorDock, Lambda):**
  - With the GPU passed through, the guest driver owns the GPU, so `-lgc` and ncu should work. The user has already confirmed this on Lambda.
  - The CPU governor, turbo and DRAM timing stay under host control.
  - The guest sees a virtual NUMA/CPU topology.
  - DRAM bandwidth seen by the guest should be close to native, but verify with a benchmark.
- For desktop-class runs needing GPU clock control, the only found option is a Vast VM-enabled desktop host. Supply is thin: a handful of offers at any moment.

### Gaps
- No primary provider documentation was found stating explicitly whether `nvidia-smi -lgc` succeeds in Vast VMs or TensorDock VMs. The PR above is untested and second-hand. Test in the first minutes of a rental.
- Whether Vast Docker containers allow `perf_event_open` at all (for core, not uncore, counters) was not found.

## 6. Practical limits: minimum duration, storage, bandwidth, availability, reliability, new-account requirements

### Takeaway
None of the candidates found has a minimum rental time; all bill per second or per minute. On Vast, the practical risks are hosts with slow downlinks (some under 100 Mbps, which turns a 100 GB download into hours), unverified or deverified hosts, and continuous storage charges until the instance is deleted. A new Vast account needs only email verification and a $5 minimum credit.

### Cited Findings
- **Vast new account:** "verify your email address" before renting, then add credit by card, BitPay or Crypto.com with a "$5" minimum. Disk size "cannot be changed later". Fresh image pulls can take 10–60 minutes. "Use Stop to pause GPU billing (storage still accrues charges)". — [Vast quickstart](https://docs.vast.ai/guides/get-started/quickstart)
- **Vast storage and bandwidth charges:** storage is charged "for every single second your instance exists, regardless of what state it is in", and bandwidth "for every byte sent or received". — [Vast FAQ](https://cdn.vast.ai/faq/)
- **Vast snapshot medians for 4090 offers:** storage $0.20/GB-month, download about $0.0054/GB (about $5.5/TB), downlink about 911 Mbps. Some desktop hosts report 45–112 Mbps (for example the Finland 5950X/7950X boxes, and a Ryzen 5 3600 in Germany at 91 Mbps). — [Vast API snapshot](https://console.vast.ai/api/v0/bundles/)
- **Vast host verification:** 524 verified, 226 deverified and 155 unverified offers across the snapshot. — [Vast API snapshot](https://console.vast.ai/api/v0/bundles/)
- **Vast reliability tracking:** Vast tracks "disconnects, outages, and other errors" to estimate host reliability, and "it can take months for providers to accumulate trust and verified status". — [Vast FAQ](https://cdn.vast.ai/faq/)
- **RunPod:**
  - The pods overview page says pods are "billed by the minute with no fees for ingress/egress"; the pricing page says per-second billing.
  - Storage:
    - Container disk and running volume disk: $0.10/GB-month.
    - Idle volume disk: $0.20/GB-month.
    - Network volume: $0.07/GB-month (standard, under 1 TB) or $0.14/GB-month (high-performance).
  - Network volumes persist and can be shared across pods.
  - Sources: [RunPod docs: pods overview](https://docs.runpod.io/pods/overview); [RunPod pricing](https://www.runpod.io/pricing)
- **CloudRift:** "Billing is per second of active runtime", no minimum spend. — [CloudRift pricing](https://www.cloudrift.ai/pricing)
- **TensorDock:** pre-paid, with "automatic server deletion when balance approaches zero". Per-second billing. — [TensorDock 4090 page](https://www.tensordock.com/gpu-4090.html); [getdeploying TensorDock](https://getdeploying.com/tensordock)
- **Salad:** per-second billing. Docs recommend "3+ replicas during testing", reflecting interruptible consumer nodes. — [Salad pricing](https://salad.com/pricing/); [Salad docs](https://docs.salad.com/container-engine/explanation/container-groups/container-groups)

### Inferences
- On Vast, filter on `inet_down` of at least 500 Mbps. Models of tens to hundreds of GB also argue for pulling weights from Hugging Face directly on each host, rather than keeping a long-lived volume on a host you may not rent again.
- Delete, don't just stop, instances between sessions to avoid storage charges.
- Prefer verified hosts with reliability of at least 0.98. Several of the largest-RAM desktop hosts (the Romania i9-14900KF / Ultra 9 285K machines) are unverified, so keep a fallback host in mind.

### Gaps
- TensorDock and CloudRift minimum deposits and new-account verification steps were not found.
- RunPod Community Cloud host availability for 4090/5090 in Sept 2026 was not assessed.

## 7. A realistic $100 plan

### Takeaway
About $45–60 buys a solid multi-platform set on Vast.ai, leaving 40% headroom for failed hosts and re-runs. The set: a DDR5 desktop 4090, an AM5 PCIe-5 5090, a DDR4 AM4 3090 or 4090, a roughly 189 GB desktop for the largest model, one whole-machine EPYC server as the high-bandwidth contrast, and a short Lambda VM session for clock-locked or ncu work.

### Cited Findings
Prices below are from the [Vast API snapshot](https://console.vast.ai/api/v0/bundles/) unless noted.
- Desktop 4090, whole machine, at least 64 GB, verified, reliability of at least 0.98: median $0.482/h (22 offers). Cheapest at least-64 GB desktop 4090: $0.336/h.
- Desktop 5090, whole machine, at least 64 GB, verified, reliability of at least 0.98: median $0.61/h (22 offers). 80 of about 100 whole-machine desktop 5090 offers are PCIe 5.0 x16.
- Desktop 3090, whole machine, at least 64 GB: $0.149–0.44/h.
- Large-RAM desktop: 4090 on i9-14900KF / Ultra 9 285K with 188–189 GB at $0.601/h (unverified), or 5090 on Ryzen 9 9950X with 186 GB at $0.827/h (verified).
- Server contrast: A100 PCIe 40 GB on EPYC 7F72 with 252 GB, whole machine, $0.548/h (verified). Or 4090/5090 on whole-machine EPYC hosts at about $0.5–0.9/h.
- Vast storage at about $0.20/GB-month is negligible for a few days. Download costs about $5.5/TB.

### Inferences
Hours are assumptions: roughly 10–20 hours of measurement per host class, plus 15-minute qualification runs.

| Rental | Hours | Rate ($/h) | Est. cost |
|---|---|---|---|
| Qualification sweep: ~8 candidate hosts × 15 min (lscpu, STREAM/MLC, nvidia-smi -q, clock-lock test), drop outliers | 2 | ~0.5 | ~$1 |
| 4090 desktop, AM5 or Raptor Lake DDR5, ≥96 GB, PCIe 4.0 x16 | 20 | 0.45–0.55 | ~$10 |
| 5090 desktop, AM5 Ryzen 9, DDR5, ≥96–123 GB, PCIe 5.0 x16 | 20 | 0.60–0.70 | ~$13 |
| DDR4 desktop (AM4 Ryzen 9 5950X/5900X), 3090 or 4090, 64–126 GB | 10 | 0.30–0.50 | ~$4 |
| Largest-model desktop, ~186–189 GB (i9-14900KF 4090 or 9950X 5090) | 8 | 0.60–0.83 | ~$6 |
| Server contrast, whole-machine EPYC (A100-40GB on EPYC 7F72 252 GB, or EPYC 4090/5090) | 8 | 0.55–0.75 | ~$5 |
| Lambda VM session for clock-locked/ncu cross-check (price depends on what the console offers) | 3–4 | ~2 | ~$8 |
| Storage + data transfer (several × 100–200 GB downloads) | – | – | ~$3–5 |
| **Total** | | | **≈ $50–52** |

- Leave the remaining approximately $50 for hosts that fail qualification, preemption or disconnect, and repeat runs on a second host of the same class to estimate between-host variance.
- If GPU clock locking on the desktop hosts matters, substitute a VM-enabled Vast desktop host for the 4090 or 5090 rows at a similar price ($0.38–0.71/h in the snapshot). Supply is only a few offers, so check it right before the run.

### Gaps
- Actual achieved DRAM bandwidth of any specific Vast desktop host is unknown until launch. The DIMM configuration is never shown, which is why the qualification sweep is budgeted.
- Lambda's current single-GPU price is not on the public page, so the Lambda line is a placeholder.

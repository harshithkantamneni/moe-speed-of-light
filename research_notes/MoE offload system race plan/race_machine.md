# Race machine and development machine for the MoE offload-system race (as of 28 Sep 2026)

Method note. This extends `../MoE audit measurement plan/cloud_hardware.md` and `bandwidth_control.md`; facts already established there are referenced, not repeated. New primary data collected on 2026-09-28, all from public, unauthenticated endpoints (no accounts, no rentals, no credentials):
- **Verda public API**: `GET https://api.verda.com/v1/instance-types` (70 types), plus `/v1/volume-types`, `/v1/cluster-types` and `/v1/container-types`, which turned out to be public too. `/v1/instance-availability`, `/v1/locations` and `/v1/images` return `unauthorized_request`, so live stock could not be checked. The raw instance-types response is saved as `verda_instance_types_2026-09-28.json` next to this file.
- **Verda docs corpus** (`https://docs.verda.com/llms-full.txt`, 14.5k lines) and the public Discourse forum (`forum.verda.com`, read via its JSON API).
- **Vast.ai public offers API** (`https://console.vast.ai/api/v0/bundles/`), queried for single-GPU, rentable, on-demand offers of RTX 5090, RTX PRO 6000 WS / S / Max-Q, 5080, 5070 Ti, 5070 and 5060 Ti. Six sort orders were merged per GPU, giving 696 unique offers. The trimmed snapshot is saved as `vast_snapshot_2026-09-28_blackwell.json`. It is a single point in time; Vast supply changes hourly. Citations below to "[Vast API 09-28]" refer to it.

Correction to prior notes: `bandwidth_control.md` §2 says "EPYC 9575F on a server rented from Verda: NPS1 linear read bandwidth was 479 GB/s". Re-reading the source, 479 GB/s is the **EPYC 9355P in NPS1**, used as the comparison chip. The Verda machine was **2× EPYC 9575F with 8× B200, run as a VM in NPS0** ("24 memory controllers providing uniform memory access"), where "DRAM latency rises to over 220 ns", a "nearly 90 ns penalty". — [Chips and Cheese, 2025-11-26](https://old.chipsandcheese.com/2025/11/26/evaluating-uniform-memory-access-mode-on-amds-turin-ft-verda-formerly-datacrunch-io/); [newer URL](https://chipsandcheese.com/p/evaluating-uniform-memory-access)

---

## 1. Verda RTX PRO 6000 instances: hardware, software, access, storage, network, spot, billing, API

### Takeaway
Verda's `1RTXPRO6000.30V` is a KVM VM with root. It gets one **RTX PRO 6000 Blackwell Server Edition (96 GB, 1597 GB/s)**, 30 pinned vCPUs and 90 GB RAM, sized as **exactly 1/8 of an 8-GPU host** (240 vCPU / 720 GB). CUDA 13.0–13.2 images and an R580 driver are available, `ncu` works out of the box, and billing is prepaid in 10-minute increments. However, **the host CPU model, DRAM configuration, NUMA exposure and PCIe link width are not published anywhere**, and the other 7 GPU slots' tenants share the host's memory controllers. It is therefore a poor fit for a race whose result is dominated by host memory bandwidth, and 90 GB RAM is below the 130 GB model size. The 2× type (180 GB, $3.93/h) fixes RAM but not the shared-host problem.

### Cited Findings
**Instance shapes and prices (public API, 2026-09-28)**

| Type | vCPU | RAM | GPU | On-demand $/h | Spot $/h |
|---|---|---|---|---|---|
| 1RTXPRO6000.30V | 30 | 90 GB | 1× RTX PRO 6000 96 GB | 1.964 | 0.982 |
| 2RTXPRO6000.60V | 60 | 180 GB | 2× | 3.928 | 1.964 |
| 4RTXPRO6000.120V | 120 | 360 GB | 4× | 7.856 | 3.928 |
| 8RTXPRO6000.240V | 240 | 720 GB | 8× | 15.71 | 7.856 |
| 1RTXPRO6000.30V.CC (confidential) | 30 | 90 GB | 1× | 2.003 | 1.002 |

— [Verda API instance-types](https://api.verda.com/v1/instance-types)

- **All types are described as "Dedicated Hardware Instance".** `dynamic_price` is null and `max_dynamic_price` equals the on-demand price. The old price-history endpoint now returns "This endpoint has been removed. Dynamic pricing is no longer supported." — [Verda API instance-types](https://api.verda.com/v1/instance-types); [price-history endpoint](https://api.verda.com/v1/instance-types/price-history)
- **Prices have risen since launch.** At launch (Sept 2025) RTX PRO 6000 was advertised at "$1.39/h (fixed pricing) → $0.68/h (dynamic) → $0.17/h (spot)". — [Verda/DataCrunch on X](https://x.com/DataCrunch_io/status/1967949111728251347)
- **GPU edition:** Verda's product page says "Blackwell Server Edition", 96 GB GDDR7, and lists "1597 GB/s". — [Verda RTX PRO 6000 page](https://verda.com/rtx-pro-6000)
- **NVIDIA Server Edition spec:** 1597 GB/s, "Up to 600W (configurable)", "Support for PCI Express Gen 5", 24,064 CUDA cores, passive dual-slot. — [NVIDIA RTX PRO 6000 Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)
- **Host CPU: not published for RTX PRO 6000 nodes.** Neither the product page nor the API names it. The page lists only "30 CPU threads" and "90 GB". — [Verda RTX PRO 6000 page](https://verda.com/rtx-pro-6000); [Verda API](https://api.verda.com/v1/instance-types)
  - *Indirect evidence:* the confidential-computing (CC) RTX PRO 6000 VMs use AMD SEV-SNP, and the attestation walkthrough fetches the AMD CA chain with `snpguest fetch ca pem ... turin`. So CC RTX PRO 6000 hosts are EPYC Turin (9005). The same page shows GPU "Hardware Model: GB20X", "Driver Version: 580.126.09". — [Verda docs: Confidential Computing](https://docs.verda.com/cpu-and-gpu-instances/confidential-computing/)
  - *Other Verda node CPUs published in `cluster-types` node_details:* B200/B300/GB300 nodes are "AMD Turin CPU" (240 or 128 cores); H200 nodes are "AMD Genoa CPU" (176 cores). — [Verda API cluster-types](https://api.verda.com/v1/cluster-types)
  - *CPU varies by site:* for CPU-only VMs Verda staff wrote: "The CPU you get varies between sites and also within sites. At the moment we don't have a way to limit deploys of instances to a certain kind. Some CPU servers run EPYC 9655 as visible in lscpu from inside the VM. The vCPUs are pinned and not shared. We use host-passthrough for the CPU." (Nov 2025). The API nonetheless labels `CPU.360V.1440G` "AMD EPYC" / display_name "Genoa", so the label and reality can differ. — [Verda forum #73](https://forum.verda.com/t/73); [Verda API](https://api.verda.com/v1/instance-types)
- **Memory type, channels, bandwidth, NUMA exposure and PCIe link as seen in the RTX PRO 6000 VM:** no source found. The only Verda host with published memory behaviour is the B200 VM: 2× EPYC 9575F in NPS0, >220 ns DRAM latency. — [Chips and Cheese, 2025-11-26](https://old.chipsandcheese.com/2025/11/26/evaluating-uniform-memory-access-mode-on-amds-turin-ft-verda-formerly-datacrunch-io/)
- **OS images and CUDA.** The 1RTXPRO6000.30V `supported_os` list includes:
  - `24.04.cuda13.0`, `24.04.cuda13.1`, `24.04.cuda13.2` (+ `.docker` variants)
  - `26.04.cuda13.0` through `26.04.cuda13.2`
  - `ubuntu-24.04-cuda-13.0-open`
  - `24.04.base` / `26.04.base`, plus a `24.04.cuda13.0.verda-gpureset` image

  The B200 deploy warning advises "Ubuntu 24.04 + CUDA 12.8 Open" or the open driver, so Blackwell needs the open kernel module. — [Verda API instance-types](https://api.verda.com/v1/instance-types)
- **Driver branch.** Serverless Containers use "NVIDIA driver version 580.x.x (R580)" across all GPU pools, and CC RTX PRO 6000 shows 580.126.09. No explicit driver number is published for the ordinary VM images, but a CUDA 13.x image implies driver ≥ 580 (general CUDA/driver compatibility knowledge, not Verda-sourced). — [Verda docs: Containers overview](https://docs.verda.com/containers/overview/); [Verda docs: Confidential Computing](https://docs.verda.com/cpu-and-gpu-instances/confidential-computing/)
- **Root and GPU profiling.**
  - "Verda GPU instances are configured by default to allow non-admin users to access NVIDIA GPU hardware performance counters". Current images ship CUDA, `nvcc`, `nsys` and `ncu`. `NVreg_RestrictProfilingToAdminUsers` is 0.
  - The docs show the user editing `/etc/modprobe.d/` and running `sudo reboot`, so the guest owns the NVIDIA kernel module.
  - Default users are `ubuntu`/`root`-style; `/usr/local/cuda/bin` is on root's PATH.
  - Source: [Verda docs: Profile with Nsight](https://docs.verda.com/cpu-and-gpu-instances/profile-with-nsight/); [Verda docs: troubleshooting SSH](https://docs.verda.com/cpu-and-gpu-instances/troubleshooting-ssh-connection-issues/)
- **`nvidia-smi -lgc` / `-pm` clock locking:** Verda documents nothing on it. No user report found.
- **Storage.**
  - NVMe block volume: $0.20/GB-month (`cps_per_gb` 7.61e-8, i.e. about $0.082/h per 300 GB). Listed as "burst_bandwidth" 2500, "continuous_bandwidth" 2000, iops "100k". HDD is $0.05/GB-month, but "It is no longer possible to create instances and volumes based on HDD".
  - Storage is in GiB.
  - Deleted volumes sit in trash for 96 h and are billed if restored.
  - If the prepaid balance hits zero, "your instances will be discontinued and your volumes (data) will be deleted".
  - The API instance `storage` field says "dynamic", meaning OS/data volumes are sized by the user. No free local NVMe is documented for single instances; the 7 TB local NVMe is only mentioned for Instant Cluster workers.
  - Sources: [Verda API volume-types](https://api.verda.com/v1/volume-types); [Verda release notes](https://docs.verda.com/welcome-to-verda/release-notes/); [Verda docs: Deleting storage](https://docs.verda.com/storage/deleting-storage/); [Verda docs: Pricing and billing](https://docs.verda.com/welcome-to-verda/pricing-and-billing/); [Verda docs: Instant Clusters](https://docs.verda.com/clusters/instant-clusters/)
- **Locations:** FIN-01, FIN-02 and FIN-03, all Helsinki. ICE-01 was discontinued from 18 Dec 2025. — [Verda docs: Locations](https://docs.verda.com/welcome-to-verda/locations-and-sustainability/); [Verda forum #79](https://forum.verda.com/t/79)
- **Network and Hugging Face download speed.**
  - The Instant Cluster docs say "The uplink to the Internet is symmetric 2 Gb/s". The `cluster-types` API says "Uplink 5 Gbit/s". Neither covers single VM instances. — [Verda docs: Instant Clusters](https://docs.verda.com/clusters/instant-clusters/); [Verda API cluster-types](https://api.verda.com/v1/cluster-types)
  - For HF downloads, Verda staff recommend `HF_HUB_ENABLE_HF_TRANSFER=1` and caching to NVMe/SFS. They note that very large single files (~45 GB) "make the downloads quite slow". No measured MB/s was given. — [Verda forum #40](https://forum.verda.com/t/40)
- **Spot behaviour.**
  - "Spot instances ... can be evicted by Verda at any point without warning". — [Verda docs: Set up an instance](https://docs.verda.com/cpu-and-gpu-instances/set-up-a-gpu-instance/)
  - Staff (May 2026): "Spot instance will run as long as there aren't any on-demand request for that specific instance". The status page shows uptime, not eviction rates. — [Verda forum #85](https://forum.verda.com/t/85)
  - For serverless spot there is no SIGTERM: "they can be deleted immediately without warning" (Nov 2025). — [Verda forum #72](https://forum.verda.com/t/72)
  - Since 2026-02-03, a spot create can set `on_spot_discontinue`: `keep_detached` | `move_to_trash` | `delete_permanently` for its volumes. — [Verda docs: API changes](https://docs.verda.com/welcome-to-verda/release-notes/verda-api-changes/)
- **Billing:** "Pay As You Go ... billing calculated in pre-paid 10-minute increments. If a resource is terminated before the billed period ends, the unused portion is automatically refunded". Deleting the instance stops compute billing; kept volumes continue to bill. — [Verda docs: Pricing and billing](https://docs.verda.com/welcome-to-verda/pricing-and-billing/); [Verda forum #66](https://forum.verda.com/t/66)
- **API authentication.**
  - Cloud API credentials are a Client ID + Client Secret pair, created in Console → Credentials. The secret is shown once, and credentials are tied to the creating team member. — [Verda docs: API credentials](https://docs.verda.com/welcome-to-verda/api-credentials/)
  - The API reference documents an OAuth2 token endpoint (`/v1/oauth2/token`) with `grant_type: client_credentials` (and `refresh_token`), and example responses show `expires_in: 3600`. — [Verda API reference](https://api.verda.com/v1/docs)
  - The same credentials work in the `verda` CLI (which has `verda availability --spot`, `verda instance-types --spot`, `verda cost estimate`), Terraform/OpenTofu, dstack and SkyPilot. — [Verda docs: API credentials](https://docs.verda.com/welcome-to-verda/api-credentials/); [Verda docs corpus (CLI)](https://docs.verda.com/llms-full.txt)
- **Boot and provisioning time.**
  - Docs: "get a CPU or GPU instance up and running within a few minutes". No measured figure. — [Verda docs: Set up an instance](https://docs.verda.com/cpu-and-gpu-instances/set-up-a-gpu-instance/)
  - One incident (Oct 2025): `POST /v1/instances` for 1RTXPRO6000.30V timed out (>30 s) during a FIN-03 service restart of about 30 min. — [Verda forum #64](https://forum.verda.com/t/64)
- **Capacity scarcity** has been reported (Oct 2025): "difficult to find spot instances or even on-demand". — [Verda forum #71](https://forum.verda.com/t/71)
- **Quotas** cover GPUs, spot GPUs, vCPUs and NVMe storage. Increase requests are "typically reviewed within one business day". — [Verda docs: Release notes / quotas](https://docs.verda.com/welcome-to-verda/quotas/)

### Inferences
- **The 1× RTX PRO 6000 VM is almost certainly one of eight slices of a single 8-GPU server.** The 8× type is exactly 8 times the 1× type (240 vCPU / 720 GB). All eight GPU tenants therefore share one host's DRAM controllers, and possibly its PCIe root complexes. Pinned vCPUs isolate compute, not memory bandwidth. For a race decided by host DRAM bandwidth, neighbour load makes CPU-expert throughput run-to-run noisy, and the noise is invisible to the renter.
- **If the RTX PRO 6000 hosts are dual-socket Turin**, as the CC evidence and the 240-vCPU shape suggest, a 30-vCPU slice is roughly 15 physical cores, about 2 CCDs.
  - Per-CCD GMI links cap reads at roughly 64–100 GB/s per CCD (see `bandwidth_control.md` §2).
  - If Verda also runs these hosts in NPS0 like its B200 nodes, idle DRAM latency is ~220 ns versus ~80 ns on a desktop.
  - That is a very different "CPU" from the consumer desktops the published offloading systems target, and the actual bandwidth is unknown until measured.
- **90 GB RAM is not enough for 130 GB models.** The 2× type (180 GB, $3.93/h) is enough, but it doubles the price and carries a second GPU entrants must be told to ignore.
- **A 96 GB GPU changes the regime.** With 96 GB VRAM, a 60–130 GB model mostly fits on-GPU, so an offloading race run "naturally" on this card would under-weight the CPU side. Every entrant would need the same imposed VRAM budget, e.g. 24 or 32 GB, and not every system exposes such a knob.
- **Verda's strengths are software, not hardware:** root in a VM, the guest owns the driver, CUDA 13 images, `ncu` without sudo, clean billing, a scriptable API. That makes it a good second opinion for a clock-locked / Nsight measurement of the GPU-side kernels. It is not the primary race machine.

### Gaps
- Host CPU model, DIMM configuration, NPS/NUMA setting and PCIe link width behind non-CC RTX PRO 6000 instances. Not published, and not found in any third-party write-up or forum post. It can only be learned by renting: `lscpu`, `numactl -H`, `nvidia-smi -q | grep -A3 Link`, a read-bandwidth sweep.
- Whether `nvidia-smi -lgc` / `-lmc` / `-pl` succeed in the Verda VM. Likely, since the guest owns the kernel module, but unverified.
- Measured boot time, HF download throughput for single instances, egress/ingress fees (none found in the docs corpus; absence is not proof), and spot eviction frequency for RTX PRO 6000 (Verda publishes none).
- Current live stock of 1RTXPRO6000.30V: the availability endpoint needs auth.

---

## 2. Alternatives for the race, and which is the fairest single machine

### Takeaway
The fairest option within reach is a **whole-machine Vast.ai host (gpu_frac = 1, so no neighbours on the memory bus) with an RTX 5090 on a DDR5 AM5/LGA1851 desktop carrying ~192–256 GB RAM**. It runs a driver ≥ 580 (CUDA 13.x), PCIe 5.0 x16, and costs ~$0.74–0.94/h. It matches the consumer-desktop class the published systems target, fits 130 GB models, and gives every entrant the same 16–24 cores and stable bandwidth.

The catches:
- **Supply is thin.** Only 3 such 5090 offers existed on 28 Sep, and the best one allows at most 3 days' rental.
- **Vast Docker gives root in a container, without GPU clock locking.**

Fallbacks, in order:
1. The same class with an RTX PRO 6000 WS (9 hosts with ≥180 GB, $1.14–1.83/h), with VRAM capped for all entrants.
2. A 126–128 GB DDR5 5090 desktop (about 10 hosts, $0.51–0.88/h), restricting the model to ≤~100 GB.

RunPod (5090 has only 35 GB RAM; PRO 6000 is $2.09/h in a container, with the CPU unknown) and Lambda (no sm_120 GPUs) are not competitive for this race.

### Cited Findings
**Vast.ai supply (28 Sep 2026)** — [Vast API 09-28](https://console.vast.ai/api/v0/bundles/)

| GPU | Offers | Price min / median ($/h) | Whole machine | VM-enabled | Driver-major counts |
|---|---|---|---|---|---|
| RTX 5090 | 294 | 0.387 / 0.652 | 115 | 13 | 570: 12; 580: 96; 590: 11; 595: 142; 610: 24; 615: 9 (96% ≥ 580) |
| RTX PRO 6000 WS | 53 | 1.002 / 1.403 | 26 | 1 | – |
| RTX PRO 6000 S (Server Ed.) | 30 | 0.736 / 1.524 | 0 | – | – |
| RTX PRO 6000 Max-Q | 59 | 0.527 / 1.308 | 20 | – | – |

- The RTX 5090 CPU mix is 92 desktop, 27 HEDT, 166 server, 9 unknown.
- All 30 RTX PRO 6000 S offers are fractional server slices.

**Whole-machine, DDR5, ≥180 GB RAM offers (all single-GPU on-demand)** — [Vast API 09-28](https://console.vast.ai/api/v0/bundles/)

| GPU | Price ($/h) | CPU | RAM | Board | PCIe | Driver / CUDA | Status | Reliability | Downlink | Location | Max duration |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RTX 5090 | 0.735 | Ryzen 9 9950X | 255 GB | MAG X870 TOMAHAWK | 5.0 x16, 52 GB/s | 595.71.05 / 13.2 | verified | 0.999 | 6.4 Gb/s | Japan | **3 days** |
| RTX 5090 | 0.936 | Ryzen 9 9950X3D | 255 GB | X870E Taichi | 5.0 x16, 55 GB/s | 580.178.04 / 13.0 | verified | 0.960 | 0.94 Gb/s | Sweden | 22 days |
| RTX 5090 | 0.868 | Core Ultra 9 285K | 193 GB | – | 5.0 x16, 40 GB/s | – | unverified | 0.968 | – | Romania | – |
| RTX PRO 6000 WS | 1.136 | Core Ultra 7 265K | 257 GB | – | – | – | unverified | 0.955 | – | – | – |
| RTX PRO 6000 WS | 1.256 | Core Ultra 9 285K | 255 GB | – | – | – | verified | 0.897 | – | Nevada | – |
| RTX PRO 6000 WS | 1.336 | Core Ultra 9 285K | 255 GB | – | – | – | verified | 0.995 | – | Nevada | 183 days |
| RTX PRO 6000 WS | 1.403 | Ryzen 9 9900X | 191 GB | – | – | – | verified | – | – | – | – |
| RTX PRO 6000 WS | 1.469 | Core Ultra 7 270K Plus | 257 GB | – | – | – | verified | – | – | – | – |
| RTX PRO 6000 WS | 1.522 | Xeon w9-3475X | 257 GB | – | – | – | verified | – | – | – | – |
| RTX PRO 6000 WS | 1.828 | Core Ultra 9 285K | 193 GB | – | – | – | – | – | – | – | – |
| RTX PRO 6000 Max-Q | 1.162 | Ryzen 9 9950X3D | 186 GB | – | – | – | – | – | – | – | – |
| RTX PRO 6000 Max-Q | 1.390 | Ryzen 7 7700 | 192 GB | – | – | – | – | – | – | – | – |

The Xeon w9-3475X host is 8-channel HEDT. Dashes mean the field was not recorded here.

- **Whole-machine DDR5 5090 desktops with 126–128 GB:**

  | CPU | Price ($/h) | Status | Notes |
  |---|---|---|---|
  | 9950X | 0.507 | unverified | Vietnam |
  | 7950X3D | 0.642 | unverified | – |
  | 9950X | 0.804 | verified | rel 0.999, Sweden |
  | 9950X | 0.843 | verified | **VM-enabled**, India |
  | Core Ultra 5 250K Plus | 0.884 | verified | – |
  | Threadripper 9960X | 0.642 | unverified | 4-channel DDR5 |
  | Threadripper 7960X | 1.004 | – | 4-channel DDR5 |

- **Vast measures GPU memory bandwidth per offer** (`gpu_mem_bw`). RTX 5090 is ~1450 GB/s and RTX PRO 6000 WS ~1400 GB/s. A few hosts read far lower, for example 610 GB/s on a PRO 6000 WS and 633–752 GB/s on some 5090s, which suggests throttled or misconfigured cards. — [Vast API 09-28](https://console.vast.ai/api/v0/bundles/)
- **Access on Vast.**
  - Default instances are "unprivileged docker containers". — [Vast FAQ](https://cdn.vast.ai/faq/)
  - VM instances give "Full control over the virtual environment", systemd and ptrace, at the cost of "Longer instance creation and boot times", "Higher disk space requirements" and "Limited machine selection". — [Vast docs: VMs](https://docs.vast.ai/linux-virtual-machines)
  - Earlier notes record Vast's claim of "full system access and hardware counters" in VMs, and a second-hand, untested claim that "Vast's Docker ... cannot pin a clock". — see `cloud_hardware.md` §5
- **Vast on-demand is not preemptible during the contract:**
  - On-demand has "Guaranteed resources for the rental period".
  - However, "When the rental contract ends, hosts may extend the rental or stop the instance". Check "the maximum duration before renting (shown on offer cards)".
  - Interruptible instances "May be paused if outbid or if on-demand requested".
  - Source: [Vast docs: Instance types](https://docs.vast.ai/documentation/instances/choosing/instance-types)
- **Vast costs.** Storage is charged while the instance exists, even when stopped; "Stop" pauses only GPU billing. The median storage cost across 5090 offers is $0.25/GB-month (the top candidate is $0.13). Download is a median $5.33/TB. — [Vast quickstart](https://docs.vast.ai/guides/get-started/quickstart); [Vast API 09-28](https://console.vast.ai/api/v0/bundles/)
- **RunPod** (Secure Cloud, per-second billing):
  - RTX Pro 6000: 96 GB, 188 GB RAM, 16 vCPU, $2.09/h.
  - RTX 5090: 35 GB RAM, 9 vCPU, $0.99/h.
  - RTX 4090: 41 GB RAM, $0.74/h.
  - Container disk $0.10/GB-month; network storage $0.07/GB-month.
  - Source: [RunPod pricing](https://www.runpod.io/pricing)
  - Pods are Docker containers, and the host CPU is not shown before launch. — see `cloud_hardware.md` §2/§5
- **Lambda's public on-demand list** has no sm_120 part (B200 $6.69, H100, A100, V100 in the 2026-09-27 fetch). — [Lambda pricing](https://lambda.ai/pricing), via `cloud_hardware.md` §1
- **Other RTX PRO 6000 sellers** (aggregator, updated 2026-09-27, 1 GPU):

  | Provider | $/h | vCPU | RAM | Spot |
  |---|---|---|---|---|
  | Nebius | 1.80 | 24 | 218 GB | – |
  | Jarvislabs | 1.89 | 28 | 160 GB | $0.99 |
  | Cloudzy | 1.93 | 24 | 200 GB | – |
  | Hetzner | 2.19 | 24 | 256 GB | – |
  | Lium | 1.19 | 16 | 142 GB | – |
  | UpCloud | 1.88 | 16 | 80 GB | – |
  | Verda | 1.95 | 30 | 90 GB | $0.97 |

  — [getdeploying RTX PRO 6000](https://getdeploying.com/gpus/nvidia-rtx-pro-6000)
- **Hetzner's RTX PRO 6000 box (GEX131)** is a monthly dedicated server, reported at about €889–1,199/month. — [Hetzner GEX131](https://www.hetzner.com/dedicated-rootserver/gex131/); [gpuhosted review](https://gpuhosted.com/en/hetzner-gpu-review/)

### Inferences
**What "fair" requires, and how the options compare:**

1. **No co-tenants on the memory bus.** Only whole-machine rentals guarantee this: Vast `gpu_frac = 1`, or a dedicated server. Verda 1×/2× slices, RunPod pods and Vast fractional offers all share host DRAM with strangers.
2. **RAM ≥ ~1.3× the largest model.** 130 GB weights plus OS, pinned buffers and any duplicate expert copies means ≥ ~180 GB. Only the 186–257 GB desktop/HEDT hosts qualify, or Verda 2× (180 GB, marginal).
3. **Enough cores, the same for everyone.** A 16-core Zen 5 (9950X, full-width AVX-512) or a 24-core Arrow Lake (no AVX-512) is typical of the published systems' test desktops.
   - KTransformers' AMX paths will not engage on either. Its AVX-512 path will engage on Zen 5.
   - This is general knowledge of those code bases, not sourced here; confirm in each system's docs.
4. **Root.** Needed to set up each system, and ideally to lock GPU clocks. Vast containers give root without clock locking; Vast VM hosts and Verda give both.

**Recommendation for the race:**
- Rent the Vast whole-machine 5090 + DDR5 + ~255 GB class, e.g. the 9950X/255 GB Japan host or the 9950X3D/255 GB Sweden host.
- Run every entrant in one continuous rental, interleaved (A B C D A B C D…), with a read-bandwidth probe (likwid-bench load or an OpenMP read kernel) and `nvidia-smi -q -d CLOCK,POWER,PERFORMANCE` logged before each block, so drift shows up.
- Record, don't lock, GPU clocks; note that the host CPU governor is outside anyone's control on every option.

**Caveats on this class:**
- **4-DIMM AM5/LGA1851 boards usually run DDR5 below rated speed.** AMD's 2DPC guidance is general knowledge, not sourced here. Effective read bandwidth may be only ~50–70 GB/s, and single-CCD caps also apply (`bandwidth_control.md` §2).
  - This is representative of a real consumer box with 192–256 GB. It is not the "DDR5-6000 2-DIMM" figure some papers quote, so report the measured GB/s alongside every result.
- **Supply is the main risk.** There are 3 qualifying 5090 hosts now, and the best one caps rentals at 3 days. Re-query the API the day before the race; `machine_id` is stable, so a specific box can be re-targeted.
  - Fallback A: an RTX PRO 6000 WS on a 255 GB desktop (e.g. Core Ultra 9 285K, rel 0.995, 183-day duration, $1.336/h), with all entrants held to the same VRAM cap.
  - Fallback B: a 128 GB 5090 desktop, racing a ≤~100 GB model.
- **Verda 2RTXPRO6000.60V** is the only option in the user's existing accounts that meets the RAM bar with root and clock control. At $3.93/h it costs 4–5× more, and it is a shared server slice with unknown CPU and NUMA. Use it for a GPU-side, clock-locked cross-check, not the headline race.
- RunPod RTX PRO 6000 (188 GB, $2.09/h) meets the RAM bar but has only 16 vCPUs on an unknown, probably shared, server CPU in a container. It is less fair and more expensive than Vast whole-machine.

### Gaps
- Whether `nvidia-smi -lgc` works in Vast VM instances, and whether any VM-enabled whole-machine desktop with ≥180 GB exists. None did on 28 Sep; the only VM-enabled DDR5 desktop 5090 had 128 GB.
- Actual DRAM speed and DIMM population of any specific Vast host. Never exposed; measure after launch.
- RunPod Community Cloud 5090/PRO 6000 RAM limits and host CPUs were not verifiable from public pages.
- Whether all four entrants (FreeToken, NVIDIA pipelined sharding, KTransformers, ik_llama.cpp) support a hard VRAM cap, which matters if a 96 GB card is used. Not researched here.

---

## 3. Development iteration: cheapest Blackwell (sm_120) with ≥32 GB host RAM for ~20–30 h

### Takeaway
Vast.ai on-demand is the cheapest and carries no preemption risk during the contract:
- **Bare minimum:** an RTX 5060 Ti with 32–64 GB RAM at **$0.11–0.18/h**, verified, reliability ≥ 0.97, driver ≥ 580. These are mostly fractional slices on DDR4 or PCIe x8 hosts, which is fine for kernel correctness work but wrong for PCIe/host-bandwidth tuning.
- **Better value for this project:** a whole-machine RTX 5080 or 5090 on a DDR5 desktop with PCIe 5.0 x16 at **$0.31–0.39/h (5080, 62–64 GB)** or **$0.51–0.54/h (5090, 62–126 GB)**. Its host behaves like the race machine.

Verda spot RTX PRO 6000 ($0.98/h) costs 2–8× more and can be evicted without warning. It only makes sense when you need root plus clock locking plus `ncu` on sm_120.

### Cited Findings
**Cheapest sm_120 offers with ≥32 GB host RAM, driver ≥ 580, verified, reliability ≥ 0.97, downlink ≥ 300 Mb/s** — [Vast API 09-28](https://console.vast.ai/api/v0/bundles/)

| GPU | Qualifying offers | Min $/h | Median $/h | Cheapest offer |
|---|---|---|---|---|
| RTX 5060 Ti | 41 | 0.107 | 0.177 | Threadripper 1950X, 32 GB, ½ machine, PCIe 3.0 x8 |
| RTX 5070 | 7 | 0.162 | 0.189 | whole Ryzen 9 5950X box, 64 GB, VM-enabled, DDR4 |
| RTX 5070 Ti | 9 | 0.213 | 0.270 | – |
| RTX 5080 | 24 | 0.250 | 0.336 | – |
| RTX 5090 | 138 | 0.410 | 0.735 | – |

**Whole-machine DDR5 dev hosts (driver ≥ 580, reliability ≥ 0.95)** — [Vast API 09-28](https://console.vast.ai/api/v0/bundles/)

| GPU | CPU | RAM | Price ($/h) | Status | Driver / CUDA | Max duration |
|---|---|---|---|---|---|---|
| RTX 5080 | Ryzen 7 9800X3D | 62 GB | 0.309 | unverified | 580.142 / 13.0 | 1 day |
| RTX 5080 | Core Ultra 7 265K | 64 GB | 0.387 | verified, rel 0.996 | – | 249 days |
| RTX 5090 | Ryzen 9 9950X | 126 GB | 0.507 | unverified | – | 25 days |
| RTX 5090 | Ryzen 7 9700X | 62 GB | 0.534 | verified, rel 0.998 | – | 94 days |
| RTX 5090 | Core Ultra 9 285K | 64 GB | 0.534 | – | – | 511 days |

- There are 54 such 5090 offers in total, with a median of $0.728/h.
- **No whole-machine DDR5 RTX 5060 Ti / 5070 / 5070 Ti offers** passed the driver ≥ 580 and reliability ≥ 0.95 filters.
- **Vast interruption model:** on-demand is guaranteed for the rental period, but the host may stop the instance when the contract (offer max duration) ends. Interruptible/bid instances can be paused if outbid. — [Vast docs: Instance types](https://docs.vast.ai/documentation/instances/choosing/instance-types)
- **Verda spot is $0.982/h** for 1× RTX PRO 6000 (30 vCPU / 90 GB). It "can be evicted by Verda at any point without warning", and runs "as long as there aren't any on-demand request for that specific instance". — [Verda API](https://api.verda.com/v1/instance-types); [Verda docs](https://docs.verda.com/cpu-and-gpu-instances/set-up-a-gpu-instance/); [Verda forum #85](https://forum.verda.com/t/85)
- **Verda's only other cheap GPU**, 1A6000.10V (RTX A6000, $0.64 / $0.32 spot), is Ampere (sm_86), not Blackwell. B200/B300 are sm_100/sm_103 and cost $6.85+/h. — [Verda API](https://api.verda.com/v1/instance-types)
- **RunPod RTX 5090 Secure** is $0.99/h with only 35 GB RAM. — [RunPod pricing](https://www.runpod.io/pricing)

### Inferences
- **For runtime and offload-scheduler work,** where PCIe transfer overlap, pinned-memory copies and host read bandwidth all matter, develop on the same host class as the race: a whole-machine 5090 on AM5 DDR5 with PCIe 5.0 x16, at ~$0.51–0.60/h. The same kernels and tuning constants then transfer directly.
- **For pure sm_120 kernel correctness** (Triton/CUDA tile kernels, FP4/FP8 paths), a 5060 Ti slice at $0.11–0.18/h is the floor. Caveats:
  - The 5060 Ti itself is a PCIe 5.0 x8 part (general knowledge).
  - Its 448 GB/s-class GDDR7 (Vast `gpu_mem_bw` about 366–369 GB/s) makes GPU-side timings unrepresentative.
- **Interruption risk ranking:**
  1. Vast on-demand: none within the contract, but check the max duration and prefer verified hosts with rel ≥ 0.99.
  2. Verda on-demand: none.
  3. Verda spot: unannounced eviction at any time; set `on_spot_discontinue: keep_detached` so the volume survives.
  4. Vast interruptible: paused when outbid.
- **Keep a small persistent setup.** A build script plus a GGUF cache is cheaper than a big idle volume: Verda NVMe at $0.20/GB-month and Vast at $0.07–0.40/GB-month both bill while idle.

### Gaps
- No public data on Vast host behaviour when an on-demand contract nears its end date (notice period, grace).
- No Verda RTX PRO 6000 spot eviction statistics.

---

## 4. Cost estimate: ~5–10 h race + ~20–30 h development, including storage and data transfer

### Takeaway
The recommended Vast-only plan costs about **$23–41** (central ≈ $30), inside the $25–40 target except at the very top of every range. It covers development on a whole-machine DDR5 5090, the race on a 255 GB DDR5 5090, host qualification, storage and downloads. Adding a Verda clock-locked cross-check adds ~$4–8. Doing development and race entirely on Verda would cost ~$45–120, breaching the $40 target and at the top end the $100 cap.

### Cited Findings
Prices used:
- Vast rates: [Vast API 09-28](https://console.vast.ai/api/v0/bundles/)
- Verda rates and storage ($0.20/GB-month, i.e. $0.000274/GB-h): [Verda API instance-types](https://api.verda.com/v1/instance-types); [Verda API volume-types](https://api.verda.com/v1/volume-types)
- Verda 10-minute prepaid billing with refund: [Verda docs: Pricing and billing](https://docs.verda.com/welcome-to-verda/pricing-and-billing/)
- Vast storage billed while the instance exists; download median $5.33/TB: [Vast quickstart](https://docs.vast.ai/guides/get-started/quickstart); [Vast API 09-28](https://console.vast.ai/api/v0/bundles/)

### Inferences
Hours and data sizes below are assumptions. The data assumption is three model families at 60–130 GB each, with two formats per entrant (GGUF plus safetensors/BF16 for KTransformers/FreeToken, where needed): ~400–600 GB downloaded to the race host, ~100–200 GB to the dev host.

**Plan A (recommended, Vast only)**

| Item | Hours | Rate | Cost |
|---|---|---|---|
| Dev: whole-machine 5090, AM5 DDR5, 62–126 GB, PCIe 5.0 x16 | 20–30 | $0.51–0.60 | $10–18 |
| Host qualification: 3–4 candidate race hosts × 15 min (lscpu, read-BW sweep, `nvidia-smi -q`, `pcie_bw`) | ~1 | ~$0.8 | ~$1 |
| Race: whole-machine 5090, DDR5, 255 GB (9950X/9950X3D) | 5–10 | $0.735–0.936 | $4–9 |
| Storage: dev 250 GB × 20–30 h + race 700 GB × 7–12 h at $0.13–0.25/GB-month | – | – | ~$2–5 |
| Downloads: ~0.6–0.8 TB at ~$5.3/TB (host-dependent) | – | – | ~$3–4 |
| Buffer for a failed host / re-run | – | – | ~$3 |
| **Total** | | | **≈ $23–41** (central ≈ $30) |

- **Plan A′ (cheaper dev):** use a 5080 at $0.31–0.39/h, or a 5060 Ti slice at $0.11–0.18/h, for the first ~15 h of kernel work. This saves ~$4–8 and gives a total of ≈ $18–32.
- **Optional Verda cross-check:** 1RTXPRO6000.30V on-demand for 2–3 h at $1.964/h (≈ $4–6), plus a 200 GB NVMe volume for ~3 h (≈ $0.2), for clock-locked `ncu`/`nsys` of the GPU-side kernels. Delete the volume afterwards; kept for a week, 200 GB costs ≈ $9.2.
- **Plan V (all on Verda, for comparison):**

  | Item | Calculation | Cost |
  |---|---|---|
  | Dev on spot 1× | 20–30 h × $0.982 | $20–29 |
  | Race on 2× (for 180 GB RAM) | 5–10 h × $3.928 | $20–39 |
  | 300 GB NVMe kept ~7 days | – | ≈ $14 |
  | **Total** | | **≈ $54–82** |

  Add re-runs after spot evictions on top of that. Using the 1× type for the race ($10–20) only works for ≤~80 GB models.
- **Largest cost risks:**
  1. Leaving volumes or instances idle. On Verda a 300 GB volume costs $0.08/h, about $2/day. On Vast a stopped instance still bills storage.
  2. Re-downloading models on every new host. Pull with `hf_transfer`, and filter Vast hosts for `inet_down` ≥ 1 Gb/s.
  3. The race host's contract ending mid-week. Rent it only on race day.

### Gaps
- Exact per-host Vast download pricing varies (median $5.33/TB, max $53/TB), so check `inet_down_cost` on the chosen offer.
- Verda egress/ingress charges are not documented in the docs corpus. Assumed zero but unverified.
- Model sizes per entrant format (e.g., whether FreeToken or NVIDIA's pipelined sharding need BF16 safetensors in addition to GGUF) were not researched here and drive the download line.

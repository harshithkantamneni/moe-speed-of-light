"""Job 124 offer listing (jobs/124_slowlink@vast.sh): rentable on-demand single-GPU RTX 5090 and RTX 4090 offers, with
the host rule's fields and the expected cost that orders them. Run by the controller before each launch; its output is
committed and pushed (results/124_offers_<time>.txt) before the launch it serves.

    python3 list124.py VASTPY LEDGER

QUALIFIES marks an offer that meets the job's rule: whole machine (gpu_frac 1), at least 90 GB of host memory, at
least 160 GB of disk, at least 300 Mb/s down, a download price of at most $0.010/GB, at most $0.90/h (RTX 5090) or
$0.80/h (RTX 4090), a consumer desktop CPU (Ryzen 3/5/7/9 that is not a Threadripper, Core i3/i5/i7/i9, Core Ultra)
that is an Intel part on an RTX 5090 (an RTX 4090 may have either vendor's), not in the rental ledger. Expected cost:
$/h x (1.6 h + the hours to download 65 GB at the listed speed); 1.6 h is the job's work after the download on a slow-
link host (the routing recording, then three rounds of ten arms at C = 14 and one at C = 32, as projected for job 124
by scripts/p2/job124_design.py on main). Machines rented in any earlier launch (their offers in the ledger and in a
committed listing), machines already run in this job and the two-hosts-per-provider rule are applied at launch; V0
checks the GPU itself on the machine, V0c the card, the CPU and the population, VR the link-to-CPU ratio (below 0.6)."""
import datetime
import importlib.util
import json
import re
import sys
import urllib.parse

spec = importlib.util.spec_from_file_location("vast", sys.argv[1]); V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
led = {r.get("offer") for r in json.load(open(sys.argv[2]))}
CAP = {"RTX 5090": 0.90, "RTX 4090": 0.80}
NET_CAP = 0.010
rows = []
for name in CAP:
    q = {"rentable": {"eq": True}, "rented": {"eq": False}, "num_gpus": {"eq": 1}, "type": "on-demand",
         "order": [["dph_total", "asc"]], "limit": 400, "gpu_name": {"eq": name}, "cpu_ram": {"gte": 64000}}
    res = V.call("GET", "/bundles/?q=" + urllib.parse.quote(json.dumps(q)), auth=False)
    rows += res.get("offers", res) if isinstance(res, dict) else res


def consumer(cpu):
    if "Threadripper" in cpu or "EPYC" in cpu or "Xeon" in cpu:
        return False
    return bool(re.search(r"Ryzen [3579] ", cpu) or re.search(r"Core\S* ?(i[3579]|Ultra)", cpu) or re.search(r"\bi[3579]-\d{4,5}", cpu))


def intel(cpu):
    return bool(re.search(r"Intel|Core", cpu)) and not re.search(r"AMD|Ryzen", cpu)


def expected(o):
    down = max(1.0, o.get("inet_down") or 1.0)   # Mb/s
    return o["dph_total"] * (1.6 + 65 * 8000 / down / 3600)


print("#LISTED " + datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") + " cards RTX 5090, RTX 4090")
for o in sorted(rows, key=expected):
    cpu = (o.get("cpu_name") or "").strip()
    g = o.get("gpu_name") or ""
    ok = ((o.get("gpu_frac") or 0) >= 0.999 and (o.get("cpu_ram") or 0) >= 90000 and consumer(cpu)
          and (g != "RTX 5090" or intel(cpu)) and (o.get("inet_down") or 0) >= 300 and (o.get("disk_space") or 0) >= 160
          and (o.get("inet_down_cost") or 0) <= NET_CAP and g in CAP and o["dph_total"] <= CAP[g] and o["id"] not in led)
    print(f"{o['id']} machine {o.get('machine_id')} host {o.get('host_id')} {cpu[:32]} cores {o.get('cpu_cores')} "
          f"ram {round((o.get('cpu_ram') or 0) / 1000)} disk {round(o.get('disk_space') or 0)} pcie {o.get('pcie_bw')} "
          f"down {round(o.get('inet_down') or 0)} netcost {o.get('inet_down_cost')} geo {o.get('geolocation')} "
          f"dph {round(o['dph_total'], 3)} gpu {g.replace(' ', '_')} gpu_frac {o.get('gpu_frac')} expected ${expected(o):.2f}"
          f"{' IN_LEDGER' if o['id'] in led else ''}{' QUALIFIES' if ok else ''}")

"""Job 121 offer listing: rentable on-demand single-GPU RTX 5090 and RTX 4090 offers, with the host rule's fields and
the expected cost that orders them (jobs/121_ownfetch@vast.sh).

    python3 list121.py VASTPY LEDGER

QUALIFIES marks an offer that meets the job's rule: whole machine (gpu_frac 1), at least 90 GB of host memory, at least
160 GB of disk, at least 300 Mb/s down, a consumer desktop CPU (Ryzen 3/5/7/9 that is not a Threadripper, Core
i3/i5/i7/i9, Core Ultra), not in the rental ledger, and at most $1.10/h (RTX 5090) or $0.70/h (RTX 4090). Expected
cost: $/h x (2.6 h + the hours to download 65 GB at the listed speed). Machines whose GPU failed V0 earlier and the
two-valid-machines-per-host rule are applied at launch. Amendment (before host b): REPEAT_OK marks an offer that meets
every rule but the ledger's, on a machine that was a valid host of jobs 109-120 (jobs/ec2/repeat_machines121.txt); one is
taken only when no new offer qualifies (jobs/121_ownfetch@vast.sh, the amendment)."""
import datetime
import importlib.util
import json
import re
import sys
import urllib.parse

spec = importlib.util.spec_from_file_location("vast", sys.argv[1]); V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
led = {r.get("offer") for r in json.load(open(sys.argv[2]))}
import os
rep = {l.strip() for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "repeat_machines121.txt"))
       if l.strip() and not l.startswith("#")}
CAP = {"RTX 5090": 1.10, "RTX 4090": 0.70}
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


def expected(o):
    down = max(1.0, o.get("inet_down") or 1.0)   # Mb/s
    return o["dph_total"] * (2.6 + 65 * 8000 / down / 3600)


print("#LISTED " + datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") + " cards RTX 5090, RTX 4090")
for o in sorted(rows, key=expected):
    cpu = (o.get("cpu_name") or "").strip()
    g = o.get("gpu_name") or ""
    rest = ((o.get("gpu_frac") or 0) >= 0.999 and (o.get("cpu_ram") or 0) >= 90000 and consumer(cpu)
            and (o.get("inet_down") or 0) >= 300 and (o.get("disk_space") or 0) >= 160 and g in CAP and o["dph_total"] <= CAP[g])
    ok = rest and o["id"] not in led and str(o.get("machine_id")) not in rep
    rep_ok = rest and str(o.get("machine_id")) in rep
    print(f"{o['id']} machine {o.get('machine_id')} host {o.get('host_id')} {cpu[:32]} cores {o.get('cpu_cores')} "
          f"ram {round((o.get('cpu_ram') or 0) / 1000)} disk {round(o.get('disk_space') or 0)} pcie {o.get('pcie_bw')} "
          f"down {round(o.get('inet_down') or 0)} geo {o.get('geolocation')} dph {round(o['dph_total'], 3)} "
          f"gpu {g.replace(' ', '_')} gpu_frac {o.get('gpu_frac')} expected ${expected(o):.2f}"
          f"{' IN_LEDGER' if o['id'] in led else ''}{' QUALIFIES' if ok else ''}{' REPEAT_OK' if rep_ok else ''}")

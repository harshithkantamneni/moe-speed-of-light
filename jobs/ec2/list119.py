"""Jobs 119 and 120 offer listing: rentable on-demand single-GPU offers of one card, with the host rule's fields and the
expected cost that orders them (jobs/119_replication@vast.sh, jobs/120_secondcard2@vast.sh).

    python3 list119.py VASTPY LEDGER 5090|4090

QUALIFIES marks an offer that meets the job's rule: whole machine (gpu_frac 1), at least 96 GB of host memory, a
consumer desktop CPU (Ryzen 3/5/7/9 that is not a Threadripper, Core i3/i5/i7/i9, Core Ultra), not in the rental
ledger, not Vast host 716820, and for 5090 (job 119) at least 240 GB of disk, 300 Mb/s down and at most $1.25/h, for
4090 (job 120) at least 130 GB, 100 Mb/s and at most $0.80/h. Expected cost: $/h x (fixed hours + the hours to download the
job's models at the listed speed): 5090 3.0 h + 124 GB, 4090 2.0 h + 63 GB. The two-valid-machines-per-host rule is
applied at launch from the ledger and the results."""
import datetime
import importlib.util
import json
import re
import sys
import urllib.parse

spec = importlib.util.spec_from_file_location("vast", sys.argv[1]); V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
led = {r.get("offer") for r in json.load(open(sys.argv[2]))}
card = sys.argv[3]
rule = {"5090": dict(name="RTX 5090", disk=240, dph=1.25, hours=3.0, gb=124, down=300),
        "4090": dict(name="RTX 4090", disk=130, dph=0.80, hours=2.0, gb=63, down=100)}[card]
q = {"rentable": {"eq": True}, "rented": {"eq": False}, "num_gpus": {"eq": 1}, "type": "on-demand",
     "order": [["dph_total", "asc"]], "limit": 400, "gpu_name": {"eq": rule["name"]}, "cpu_ram": {"gte": 64000}}
res = V.call("GET", "/bundles/?q=" + urllib.parse.quote(json.dumps(q)), auth=False)
rows = res.get("offers", res) if isinstance(res, dict) else res


def consumer(cpu):
    if "Threadripper" in cpu or "EPYC" in cpu or "Xeon" in cpu:
        return False
    return bool(re.search(r"Ryzen [3579] ", cpu) or re.search(r"Core\S* ?(i[3579]|Ultra)", cpu) or re.search(r"\bi[3579]-\d{4,5}", cpu))


def expected(o):
    down = max(1.0, o.get("inet_down") or 1.0)   # Mb/s
    return o["dph_total"] * (rule["hours"] + rule["gb"] * 8000 / down / 3600)


print("#LISTED " + datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") + f" card {card}")
for o in sorted(rows, key=expected):
    cpu = (o.get("cpu_name") or "").strip()
    ok = ((o.get("gpu_frac") or 0) >= 0.999 and (o.get("cpu_ram") or 0) >= 96000 and consumer(cpu)
          and (o.get("inet_down") or 0) >= rule["down"] and o["id"] not in led and o.get("host_id") != 716820
          and (o.get("disk_space") or 0) >= rule["disk"] and o["dph_total"] <= rule["dph"])
    print(f"{o['id']} machine {o.get('machine_id')} host {o.get('host_id')} {cpu[:32]} cores {o.get('cpu_cores')} "
          f"ram {round((o.get('cpu_ram') or 0) / 1000)} disk {round(o.get('disk_space') or 0)} pcie {o.get('pcie_bw')} "
          f"down {round(o.get('inet_down') or 0)} geo {o.get('geolocation')} dph {round(o['dph_total'], 3)} "
          f"gpu_frac {o.get('gpu_frac')} expected ${expected(o):.2f}{' IN_LEDGER' if o['id'] in led else ''}{' QUALIFIES' if ok else ''}")

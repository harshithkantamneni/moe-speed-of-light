"""Job 118 offer listing: rentable on-demand single-GPU RTX 5090 offers with at least 120 GB of host memory and 150 GB of
disk, cheapest first, with the host rule's fields (jobs/118_thirdcpu@vast.sh). Usage: python3 list118.py VASTPY LEDGER"""
import importlib.util, json, re, sys, urllib.parse, datetime
spec = importlib.util.spec_from_file_location("vast", sys.argv[1]); V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
led = {r.get("offer") for r in json.load(open(sys.argv[2]))}
q = {"rentable": {"eq": True}, "rented": {"eq": False}, "num_gpus": {"eq": 1}, "type": "on-demand",
     "order": [["dph_total", "asc"]], "limit": 300, "gpu_name": {"eq": "RTX 5090"}, "disk_space": {"gte": 150},
     "cpu_ram": {"gte": 120000}}
res = V.call("GET", "/bundles/?q=" + urllib.parse.quote(json.dumps(q)), auth=False)
rows = res.get("offers", res) if isinstance(res, dict) else res
rows.sort(key=lambda o: o["dph_total"])
print("#LISTED " + datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
for o in rows:
    cpu = (o.get("cpu_name") or "").strip()
    third = bool(re.search(r"Ryzen [3579] (PRO )?[789]\d{3}", cpu))   # a Ryzen 7000, 8000 or 9000 (Zen 4 or Zen 5) desktop part
    ok = (o.get("gpu_frac") or 0) >= 0.999 and o["dph_total"] <= 1.10 and (o.get("inet_down") or 0) >= 140 \
        and third and o.get("host_id") != 716820 and o["id"] not in led
    print(f"{o['id']} machine {o.get('machine_id')} host {o.get('host_id')} {cpu[:30]} eff {o.get('cpu_cores_effective')} "
          f"cores {o.get('cpu_cores')} ram {round((o.get('cpu_ram') or 0) / 1000)} disk {round(o.get('disk_space') or 0)} "
          f"pcie {o.get('pcie_bw')} down {o.get('inet_down')} geo {o.get('geolocation')} dph {round(o['dph_total'], 3)} "
          f"gpu_frac {o.get('gpu_frac')}{' third-family' if third else ''}{' IN_LEDGER' if o['id'] in led else ''}"
          f"{' QUALIFIES' if ok else ''}")

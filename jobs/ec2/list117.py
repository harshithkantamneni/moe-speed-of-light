"""Job 117 offer listing: rentable on-demand single-GPU offers of the 80 GB+ cards, cheapest first, with the host
rule's fields. Usage: python3 list117.py VASTPY LEDGER"""
import importlib.util, json, re, sys, urllib.parse, datetime
spec = importlib.util.spec_from_file_location("vast", sys.argv[1]); V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
led = {r.get("offer") for r in json.load(open(sys.argv[2]))}
names = ["A100 SXM4", "A100 PCIE", "A100X", "H100 SXM", "H100 PCIE", "H100 NVL", "RTX PRO 6000 WS", "RTX PRO 6000 S"]
rows = []
for gname in names:
    q = {"rentable": {"eq": True}, "rented": {"eq": False}, "num_gpus": {"eq": 1}, "type": "on-demand",
         "order": [["dph_total", "asc"]], "limit": 200, "gpu_name": {"eq": gname}, "disk_space": {"gte": 190}}
    res = V.call("GET", "/bundles/?q=" + urllib.parse.quote(json.dumps(q)), auth=False)
    rows += res.get("offers", res) if isinstance(res, dict) else res
rows.sort(key=lambda o: o["dph_total"])
print("#LISTED " + datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
for o in rows:
    cpu = (o.get("cpu_name") or "").strip()
    desk = bool(re.search(r"Core|Ryzen", cpu)) and not re.search(r"Threadripper|EPYC|Xeon", cpu, re.I)
    ok = (o.get("gpu_frac") or 0) >= 0.999 and o["dph_total"] <= 1.60 and (o.get("inet_down") or 0) >= 300 \
        and (o.get("gpu_ram") or 0) >= 79000 and (o.get("cuda_max_good") or 0) >= 12.8 and (o.get("cpu_ram") or 0) >= 120000 \
        and (o.get("disk_space") or 0) >= 190 and desk
    print(f"{o['id']} {o.get('gpu_name')} gpu_ram {o.get('gpu_ram')} machine {o.get('machine_id')} host {o.get('host_id')} "
          f"{cpu[:28]} ram {round((o.get('cpu_ram') or 0) / 1000)} disk {round(o.get('disk_space') or 0)} "
          f"down {o.get('inet_down')} cuda {o.get('cuda_max_good')} dph {round(o['dph_total'], 3)} gpu_frac {o.get('gpu_frac')}"
          f"{' desktop' if desk else ''}{' IN_LEDGER' if o['id'] in led else ''}{' QUALIFIES' if ok else ''}")

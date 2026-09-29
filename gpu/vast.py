"""Credential-free Vast.ai job control.

The rented machine never receives a credential: it clones the public `gpu` branch (sparse: jobs/ and gpu/ only),
runs jobs/<job>.sh through gpu/vast_boot.sh, and prints its results directory to the container log as base64 lines
between markers. This controller reads them back through Vast's log API, unpacks them into the local gpu-branch
clone, and destroys the machine. Only this controller holds VAST_API_KEY (from the environment); it is never
printed, written to disk, or sent anywhere except console.vast.ai.

    set -a && . ~/.secrets/keys.env && set +a
    python gpu/vast.py offers --gpu "RTX 5090" --min-ram 120 --whole --max-price 1.0
    python gpu/vast.py launch <offer_id> <job> [--disk 120]
    python gpu/vast.py status | logs <id> | fetch <id> <job> | destroy <id> | spend | balance
"""
import argparse
import base64
import hashlib
import io
import json
import os
import re
import ssl
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://console.vast.ai/api/v0"
REPO = "https://github.com/harshithkantamneni/moe-speed-of-light"
GPU_CLONE = os.environ.get("GPU_CLONE", "/home/claude/gpu-branch")
LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vast_ledger.json")
IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
CA = "/root/.ccr/ca-bundle.crt"
CTX = ssl.create_default_context(cafile=CA) if os.path.exists(CA) else ssl.create_default_context()


def _key():
    k = os.environ.get("VAST_API_KEY")
    if not k:
        sys.exit("VAST_API_KEY is not set (set -a && . ~/.secrets/keys.env && set +a)")
    return k


def call(method, path, body=None, auth=True, raw_url=None, tries=4):
    url = raw_url or API + path
    data = json.dumps(body).encode() if body is not None else None
    for i in range(tries):
        req = urllib.request.Request(url, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if auth and raw_url is None:
            req.add_header("Authorization", "Bearer " + _key())
        try:
            with urllib.request.urlopen(req, context=CTX, timeout=60) as r:
                txt = r.read().decode("utf-8", "replace")
                return json.loads(txt) if raw_url is None else txt
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:300]
            if e.code == 410 and "deprecated" in msg and raw_url is None and url.startswith(API):
                url = API.replace("/api/v0", "/api/v1") + path   # Vast moved some endpoints to v1
                continue
            if e.code in (429, 500, 502, 503, 504) and i < tries - 1:
                time.sleep(3 * (i + 1))
                continue
            raise SystemExit(f"HTTP {e.code} on {method} {path or '(result url)'}: {msg}")
        except urllib.error.URLError as e:
            if i < tries - 1:
                time.sleep(3 * (i + 1))
                continue
            raise SystemExit(f"network error on {method} {path}: {e}")


# ---------------------------------------------------------------- ledger (local spend estimate)
def ledger():
    return json.load(open(LEDGER)) if os.path.exists(LEDGER) else []


def save_ledger(L):
    json.dump(L, open(LEDGER, "w"), indent=1)


def spent(L=None, now=None):
    now = now or time.time()
    return sum(e["dph"] * ((e.get("stop") or now) - e["start"]) / 3600 + e.get("net_gb", 0) * e.get("net_cost", 0)
               for e in (L if L is not None else ledger()))


# ---------------------------------------------------------------- commands
def offers(a):
    q = {"rentable": {"eq": True}, "rented": {"eq": False}, "num_gpus": {"eq": a.num_gpus},
         "type": "on-demand", "order": [["dph_total", "asc"]], "limit": 400}
    if a.gpu:
        q["gpu_name"] = {"eq": a.gpu}
    if a.min_ram:
        q["cpu_ram"] = {"gte": a.min_ram * 1000}
    if a.max_price:
        q["dph_total"] = {"lte": a.max_price}
    if a.verified:
        q["verified"] = {"eq": True}
    if a.min_cuda:
        q["cuda_max_good"] = {"gte": a.min_cuda}
    if a.min_disk:
        q["disk_space"] = {"gte": a.min_disk}
    res = call("GET", "/bundles/?q=" + urllib.parse.quote(json.dumps(q)), auth=False)
    rows = res.get("offers", res) if isinstance(res, dict) else res
    out = []
    for o in rows:
        if a.whole and (o.get("gpu_frac") or 0) < 0.999:
            continue
        if a.cpu and not re.search(a.cpu, o.get("cpu_name") or "", re.I):
            continue
        if a.min_down and (o.get("inet_down") or 0) < a.min_down:
            continue
        if (o.get("inet_down_cost") or 0) > a.max_net_cost:
            continue
        out.append(o)
    print(f"{'offer':>9} {'gpu':<14} {'cpu':<28} {'ram':>5} {'cores':>5} {'$/h':>6} {'rel':>5} {'down':>6} "
          f"{'$/GBdn':>6} {'cuda':>5} {'maxdays':>7} {'pcie':>5} ver")
    for o in out[: a.n]:
        print(f"{o['id']:>9} {o.get('gpu_name', ''):<14.14} {(o.get('cpu_name') or '').strip():<28.28} "
              f"{(o.get('cpu_ram') or 0) / 1000:>5.0f} {o.get('cpu_cores_effective') or 0:>5.0f} {o['dph_total']:>6.3f} "
              f"{o.get('reliability2') or 0:>5.3f} {o.get('inet_down') or 0:>6.0f} {o.get('inet_down_cost') or 0:>6.4f} "
              f"{o.get('cuda_max_good') or 0:>5.1f} {(o.get('duration') or 0) / 86400:>7.1f} {o.get('pcie_bw') or 0:>5.1f} "
              f"{(o.get('verification') or '')[:4]}")
    print(f"({len(out)} matching offers)")


def boot_cmd(job):
    return ("set -x; export DEBIAN_FRONTEND=noninteractive; "
            "(command -v git && command -v curl) >/dev/null || "
            "(apt-get update -qq && apt-get install -y -qq git curl ca-certificates >/dev/null); "
            f"git clone -q --depth 1 --filter=blob:none --sparse -b gpu {REPO} /w && cd /w && "
            "git sparse-checkout set jobs gpu && "
            f"exec bash /w/gpu/vast_boot.sh {job}")


def launch(a):
    L = ledger()
    cap = float(os.environ.get("VAST_CAP", a.cap))
    if spent(L) + a.expect_hours * a.expect_dph > cap:
        sys.exit(f"refusing: estimated spend {spent(L):.2f} + this job would exceed the cap ${cap:.2f}")
    if not os.path.exists(os.path.join(GPU_CLONE, "jobs", a.job + ".sh")):
        sys.exit(f"no jobs/{a.job}.sh in {GPU_CLONE} (it must also be pushed to the public gpu branch)")
    offer = call("GET", "/bundles/?q=" + urllib.parse.quote(json.dumps({"id": {"eq": a.offer}})), auth=False)
    rows = offer.get("offers", []) if isinstance(offer, dict) else offer
    dph = rows[0]["dph_total"] if rows else a.expect_dph
    net_cost = (rows[0].get("inet_down_cost") or 0) if rows else 0
    body = {"client_id": "me", "image": a.image, "disk": a.disk, "runtype": "args",
            "args": ["bash", "-c", boot_cmd(a.job)], "label": a.label or a.job[:40], "cancel_unavail": True}
    if a.env:
        body["env"] = a.env
    r = call("PUT", f"/asks/{a.offer}/", body)
    iid = r.get("new_contract")
    if not iid:
        sys.exit(f"launch failed: {json.dumps(r)[:300]}")
    L.append({"id": iid, "offer": a.offer, "job": a.job, "dph": dph, "net_cost": net_cost, "net_gb": a.net_gb,
              "start": time.time(), "stop": None,
              "gpu": rows[0].get("gpu_name") if rows else None, "cpu": rows[0].get("cpu_name") if rows else None})
    save_ledger(L)
    print(f"launched instance {iid} for {a.job} at ${dph:.3f}/h (ledger estimate so far ${spent(L):.2f}, cap ${cap:.2f})")


def instances():
    r = call("GET", "/instances/")
    return r.get("instances", []) if isinstance(r, dict) else r


def status(a):
    rows = instances()
    if a.id:
        rows = [x for x in rows if x.get("id") == a.id]
    for x in rows:
        up = (time.time() - (x.get("start_date") or time.time())) / 3600
        print(f"{x.get('id')} {x.get('label')!s:<34.34} status={x.get('actual_status')} state={x.get('cur_state')} "
              f"intended={x.get('intended_status')} ${x.get('dph_total') or 0:.3f}/h up={up:.2f}h "
              f"{x.get('gpu_name')} | {(x.get('status_msg') or '')[:90]!r}")
    if not rows:
        print("no instances")


def get_logs(iid, tail):
    r = call("PUT", f"/instances/request_logs/{iid}/", {"tail": str(tail)})
    url = r.get("result_url")
    if not url:
        raise SystemExit(f"no log url: {json.dumps(r)[:200]}")
    for i in range(20):
        try:
            txt = call("GET", None, raw_url=url, tries=1)
            if txt.strip():
                return txt
        except SystemExit:
            pass
        time.sleep(3)
    raise SystemExit("log file never became available")


def logs(a):
    txt = get_logs(a.id, a.tail)
    lines = [l for l in txt.splitlines() if not l.startswith("@@R ")]
    print("\n".join(lines[-a.show:]))


def parse_result(txt, job):
    """Last complete BEGIN..END block for `job`: returns (header dict, tgz bytes)."""
    lines = txt.splitlines()
    ends = [i for i, l in enumerate(lines) if l.startswith(f"@@RESULT_END {job}")]
    for e in reversed(ends):
        b = max((i for i, l in enumerate(lines[:e]) if l.startswith(f"@@RESULT_BEGIN {job}")), default=None)
        if b is None:
            continue
        hdr = dict(kv.split("=", 1) for kv in lines[b].split()[2:] if "=" in kv)
        chunks = {}
        for l in lines[b + 1:e]:
            m = re.match(r"@@R (\d+) (\S*)$", l)
            if m:
                chunks[int(m.group(1))] = m.group(2)
        n = int(hdr["lines"])
        if len(chunks) != n or set(chunks) != set(range(1, n + 1)):
            continue
        try:
            blob = base64.b64decode("".join(chunks[i] for i in range(1, n + 1)), validate=True)
        except ValueError:
            continue
        if hashlib.sha256(blob).hexdigest() != hdr["sha256"]:
            continue
        return hdr, blob
    return None, None


def fetch(a):
    txt = get_logs(a.id, a.tail)
    hdr, blob = parse_result(txt, a.job)
    if blob is None:
        started = "@@START" in txt
        hb = [l for l in txt.splitlines() if l.startswith("@@HB")]
        sys.exit(f"no complete result yet (started={started}, last heartbeat: {hb[-1] if hb else 'none'})")
    dest = os.path.join(GPU_CLONE, "results")
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as t:
        for m in t.getmembers():
            if not m.name.startswith(a.job + "/") and m.name != a.job or ".." in m.name:
                sys.exit(f"unexpected path in results archive: {m.name}")
        t.extractall(dest)
    print(f"fetched {a.job}: rc={hdr.get('rc')} {hdr.get('bytes')} bytes -> {dest}/{a.job}")


def destroy(a):
    r = call("DELETE", f"/instances/{a.id}/")
    L = ledger()
    for e in L:
        if e["id"] == a.id and not e.get("stop"):
            e["stop"] = time.time()
    save_ledger(L)
    print(f"destroy {a.id}: {r.get('success', r)}; ledger estimate ${spent(L):.2f}")


def spend(a):
    L = ledger()
    for e in L:
        h = ((e.get("stop") or time.time()) - e["start"]) / 3600
        print(f"{e['id']} {e['job']:<32.32} {e.get('gpu')!s:<10.10} ${e['dph']:.3f}/h {h:5.2f} h "
              f"{'running' if not e.get('stop') else 'stopped'} ${e['dph'] * h + e.get('net_gb', 0) * e.get('net_cost', 0):.2f}")
    print(f"estimated total ${spent(L):.2f}")


def balance(a):
    u = call("GET", "/users/current/")
    print(f"credit ${float(u.get('credit') or 0):.2f}  balance ${float(u.get('balance') or 0):.2f}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    o = sp.add_parser("offers")
    o.add_argument("--gpu"); o.add_argument("--num-gpus", type=int, default=1)
    o.add_argument("--min-ram", type=float, help="GB"); o.add_argument("--max-price", type=float)
    o.add_argument("--whole", action="store_true"); o.add_argument("--verified", action="store_true")
    o.add_argument("--min-cuda", type=float); o.add_argument("--min-disk", type=float)
    o.add_argument("--min-down", type=float, help="Mb/s"); o.add_argument("--max-net-cost", type=float, default=0.02)
    o.add_argument("--cpu", help="regex on cpu_name"); o.add_argument("-n", type=int, default=25)
    l = sp.add_parser("launch")
    l.add_argument("offer", type=int); l.add_argument("job")
    l.add_argument("--disk", type=float, default=80); l.add_argument("--image", default=IMAGE)
    l.add_argument("--label"); l.add_argument("--env")
    l.add_argument("--cap", type=float, default=10.0); l.add_argument("--expect-hours", type=float, default=1.0)
    l.add_argument("--expect-dph", type=float, default=0.5); l.add_argument("--net-gb", type=float, default=0.0)
    s = sp.add_parser("status"); s.add_argument("id", type=int, nargs="?")
    g = sp.add_parser("logs"); g.add_argument("id", type=int); g.add_argument("--tail", type=int, default=400)
    g.add_argument("--show", type=int, default=40)
    f = sp.add_parser("fetch"); f.add_argument("id", type=int); f.add_argument("job")
    f.add_argument("--tail", type=int, default=20000)
    d = sp.add_parser("destroy"); d.add_argument("id", type=int)
    sp.add_parser("spend"); sp.add_parser("balance")
    a = ap.parse_args()
    {"offers": offers, "launch": launch, "status": status, "logs": logs, "fetch": fetch, "destroy": destroy,
     "spend": spend, "balance": balance}[a.cmd](a)


if __name__ == "__main__":
    main()

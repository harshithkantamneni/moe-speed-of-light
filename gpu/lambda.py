"""Minimal Lambda Cloud control: launch an instance whose cloud-init starts the job
runner, show status, terminate. Reads LAMBDA_API_KEY, LAMBDA_SSH_KEY_NAME and
GH_REPO_TOKEN from the environment; never prints them."""
import base64, json, os, sys, urllib.request

API = "https://cloud.lambda.ai/api/v1"
KEY = os.environ["LAMBDA_API_KEY"]


def call(method, path, body=None):
    req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json", "User-Agent": "moe-sol-runner/1.0",
                                          "Authorization": "Basic " + base64.b64encode(f"{KEY}:".encode()).decode()})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def user_data():
    runner = open(os.path.join(os.path.dirname(__file__), "runner.sh")).read()
    b64 = base64.b64encode(runner.encode()).decode()
    tok = os.environ["GH_REPO_TOKEN"]
    return ("#!/bin/bash\numask 077\n"
            f"printf '%s' '{tok}' > /root/.runner_token\n"
            f"echo '{b64}' | base64 -d > /opt/runner.sh && chmod 700 /opt/runner.sh\n"
            "nohup /opt/runner.sh > /var/log/runner.log 2>&1 &\n")


def launch(itype, region, name):
    body = {"region_name": region, "instance_type_name": itype, "ssh_key_names": [os.environ["LAMBDA_SSH_KEY_NAME"]],
            "name": name, "user_data": user_data()}
    return call("POST", "/instance-operations/launch", body)


def status():
    out = []
    for i in call("GET", "/instances").get("data", []):
        out.append({k: i.get(k) for k in ("id", "name", "status", "ip", "region")} | {"type": i["instance_type"]["name"],
                   "usd_per_h": i["instance_type"]["price_cents_per_hour"] / 100})
    return out


def terminate(ids):
    return call("POST", "/instance-operations/terminate", {"instance_ids": ids})


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "launch":
        print(json.dumps(launch(sys.argv[2], sys.argv[3], sys.argv[4]), indent=1))
    elif cmd == "status":
        print(json.dumps(status(), indent=1))
    elif cmd == "terminate":
        print(json.dumps(terminate(sys.argv[2:]), indent=1))

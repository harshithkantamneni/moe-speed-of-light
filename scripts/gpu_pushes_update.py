"""Merge the newest push events of the public gpu branch (GitHub activity API, through the gh client) into
prereg/gpu_pushes.json, keyed by (before, after, ts); existing records are kept as they are.

    python scripts/gpu_pushes_update.py
"""
import json
import os
import subprocess

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = os.path.join(ROOT, "prereg", "gpu_pushes.json")
REPO = "harshithkantamneni/moe-speed-of-light"


def main():
    old = json.load(open(P))
    seen = {(p["before"], p["after"], p["ts"]) for p in old}
    out = subprocess.run(["gh", "api", f"repos/{REPO}/activity?ref=refs/heads/gpu&per_page=100"], capture_output=True, text=True,
                         check=True).stdout
    new = [dict(before=x["before"], after=x["after"], ts=x["timestamp"], type=x["activity_type"]) for x in json.loads(out)]
    add = [p for p in new if (p["before"], p["after"], p["ts"]) not in seen]
    json.dump(sorted(old + add, key=lambda p: p["ts"]), open(P, "w"), indent=1)
    print(f"{len(add)} new push records; {len(old) + len(add)} in all")


if __name__ == "__main__":
    main()

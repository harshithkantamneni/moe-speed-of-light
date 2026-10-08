"""Move every record of a minadm_plan.py plan one step earlier (job 112's bypassplanS).

The engine (patch oracle5, unpaced background copies) publishes a copy issued after step g at the start of step g + 2.
A plan record of step s is the admission of an expert served at s for its next use; issued after s it lands at s + 2,
late when that use is s + 1. Issued after s - 1 instead, the copy overlaps step s (where the expert, still loading, is
served by the CPU: the first read) and is published at s + 1, as MIN with bypass assumes. A record at a sequence's
first step stays where it is (the oracle's window ends with the sequence). Holds are unchanged.

    python plan_shift.py la.bin plan.bin out.bin
Writes out.bin and out.bin.json (counts; the checks below must hold, else the script exits 2 and writes nothing).
"""
import json
import sys

import numpy as np

DT = np.dtype([("step", "<i4"), ("layer", "<i2"), ("expert", "<i2"), ("hold", "<i4")])


def main():
    la, src, dst = sys.argv[1:4]
    meta = json.load(open(la + ".json"))
    L, k = int(meta["n_layer"]), int(meta["k"])
    raw = np.fromfile(la, np.int16)
    stride = 2 + L * (2 * k + 24)
    seq = raw[: len(raw) // stride * stride].reshape(-1, stride)[:, 0].astype(np.int64)
    first = np.r_[True, seq[1:] != seq[:-1]]
    p = np.fromfile(src, DT)
    s = p["step"].astype(np.int64)
    move = (s > 0) & ~first[s]
    q = p.copy()
    q["step"] = np.where(move, s - 1, s)
    q = q[np.argsort(q["step"], kind="stable")]
    ok = (len(q) == len(p) and bool(np.all(np.diff(q["step"]) >= 0)) and int(q["step"].min()) >= 0
          and bool(np.all(np.sort(p["step"].astype(np.int64) - move) == np.sort(q["step"].astype(np.int64)))))
    info = dict(records=int(len(p)), moved=int(move.sum()), kept=int((~move).sum()), steps=int(len(seq)),
                sequences=int(first.sum()), checks_ok=ok)
    print(json.dumps(info))
    if not ok:
        sys.exit(2)
    q.tofile(dst)
    json.dump(info, open(dst + ".json", "w"))


if __name__ == "__main__":
    main()

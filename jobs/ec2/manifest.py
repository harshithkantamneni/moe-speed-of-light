"""Results manifest of a job, written at the end of the run (jobs 088+):

    python3 manifest.py OUT JOB T0_EPOCH "one-line summary" [key=value ...]

Writes OUT/manifest.json: the job, start/end time and elapsed seconds, every file in OUT with its size (the result
channel drops files over 8 MB and caps the archive at 3.5 MB, so sizes are worth having), the bs1 runs (label, launch,
problems measured, mean tok/s), the run log (OUT/runs.txt: "label launch rc seconds"), skipped steps (OUT/skipped.txt),
the selection and tables files when present, extra key=value fields, and the summary line. Never raises: a manifest
that cannot be completed is written with what it has.
"""
import json
import os
import statistics as st
import sys
import time


def main():
    out, job, t0, summary = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    extra = dict(kv.split("=", 1) for kv in sys.argv[5:] if "=" in kv)
    m = {"job": job, "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
         "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "elapsed_s": int(time.time() - t0),
         "summary": summary, **extra}
    try:
        m["files"] = {f: os.path.getsize(os.path.join(out, f)) for f in sorted(os.listdir(out))
                      if os.path.isfile(os.path.join(out, f))}
    except OSError as e:
        m["files_error"] = repr(e)
    try:
        p = os.path.join(out, "bs1.jsonl")
        if os.path.exists(p):
            by = {}
            for line in open(p):
                r = json.loads(line)
                by.setdefault((r["label"], r["launch"]), []).append(r["decode_tok_s"])
            m["bs1"] = [{"label": k[0], "launch": k[1], "n": len(v), "mean_tok_s": round(st.mean(v), 3)} for k, v in sorted(by.items())]
    except Exception as e:
        m["bs1_error"] = repr(e)
    for name, key in (("runs.txt", "runs"), ("skipped.txt", "skipped"), ("selection.txt", "selection"), ("tables.txt", "tables"),
                      ("gate.txt", "gate")):
        p = os.path.join(out, name)
        if os.path.exists(p):
            m[key] = [l.rstrip("\n") for l in open(p) if l.strip()]
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(m, f, indent=1)
    print(f"manifest: {len(m.get('files', {}))} files, {m['elapsed_s']} s; {summary}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # never fail the job on the manifest
        print(f"manifest failed: {e!r}")

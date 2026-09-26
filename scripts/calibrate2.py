"""Two-run calibration protocol (pre-registered after the A10 transfer test).

The A10 test (prereg/a10/scored.md) showed that the model's structure
transfers but its efficiencies do not. This protocol fixes the remedy BEFORE
it is tested on a second platform:

  1. Measure gpt-oss-20b MXFP4 twice: all experts on the GPU (n_cpu_moe=0) and
     CPU-only (-ngl 0), with the same llama-bench settings as the sweep.
  2. Solve eta_g from run 1 and eta_c (relative to STREAM Triad) from run 2;
     tau_e stays at the M4 value (20.7 us) and tau at 0. Nothing else is fitted.
  3. Predict every other configuration of the sweep.

    python scripts/calibrate2.py <platform_001 dir> <sweep dir> <out dir>
"""
import json
import os
import statistics as st
import sys

from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preregister import SWEEP, platform, step_time
from mosl.gguf_bytes import GGUFS, account

CAL_MODEL = "gpt-oss-20b-mxfp4"
TAU_E_US = 20.712923301117442


def measured(sdir):
    rows = [json.loads(l) for l in open(os.path.join(sdir, "sweep.jsonl")) if l.strip()]
    n_main = sum(1 for l in open(os.path.join(sdir, "order.txt")) if l.strip())
    return {(r["model"], str(r["n_cpu_moe"])): st.median(r["samples_ts"])
            for r in rows if r.get("samples_ts") and r["seq"] <= n_main}


def calibrate(plat, meas):
    g = account(*GGUFS[CAL_MODEL]); g.file_key = CAL_MODEL
    p = dict(eta_g=0.5, eta_c=0.6, tau_e_us=TAU_E_US, tau_us=0.0)
    t_gpu, t_cpu = 1 / meas[(CAL_MODEL, "0")], 1 / meas[(CAL_MODEL, "cpu")]
    p["eta_g"] = brentq(lambda e: step_time(g, 0, plat, dict(p, eta_g=e)) - t_gpu, 0.01, 2.0)
    p["eta_c"] = brentq(lambda e: step_time(g, "cpu", plat, dict(p, eta_c=e)) - t_cpu, 0.01, 3.0)
    return p


def main(pdir, sdir, out):
    plat, meas = platform(pdir), measured(sdir)
    p = calibrate(plat, meas)
    rows = []
    for name, sweep in SWEEP.items():
        g = account(*GGUFS[name]); g.file_key = name
        for n in sweep:
            key = (name, str(n))
            if name == CAL_MODEL and str(n) in ("0", "cpu"):
                continue
            pr = 1 / step_time(g, n, plat, p)
            m = meas.get(key)
            rows.append(dict(model=name, n_cpu_moe=str(n), pred=pr, meas=m, ape=None if m is None else abs(pr - m) / m))
    apes = [r["ape"] for r in rows if r["ape"] is not None]
    res = dict(params=p, n=len(apes), median_ape=st.median(apes), mape=sum(apes) / len(apes), max_ape=max(apes),
               pass_=st.median(apes) <= 0.10, rows=rows)
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "calibrate2.json"), "w"), indent=1)
    print(f"eta_g={p['eta_g']:.3f} eta_c/STREAM={p['eta_c']:.3f}  held-out n={res['n']} median APE {100*res['median_ape']:.1f}% "
          f"MAPE {100*res['mape']:.1f}% max {100*res['max_ape']:.1f}%  (pass <=10%: {res['pass_']})")


if __name__ == "__main__":
    main(*sys.argv[1:])

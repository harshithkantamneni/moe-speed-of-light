"""Job 100: predict, from this host's probe alone and before any timed run, the time per token of the in-step states
whose counters do not depend on the host (aa, w4, w16, fetch at gpt-oss C = 14 and 32), with the calibrated time model

    T = G + max(X_c / B_c, X_p / B_p, (X_c + X_p) / B_cp)

and constants frozen before launch: G = the median of the per-host-cell fits of jobs 095-099 (14 hosts), and the
counters = the job 099 panel's mean misses and in-step fetches per step (equal across its ten hosts to 0.05 per step).
B_c, B_p, B_cp come from this host's probe (fetch_table.bandwidths). Writes JSON to stdout.

    python3 predict_100.py concur.txt H
"""
import json
import sys
import time

from fetch_table import bandwidths

S = 13253760                       # gpt-oss-120b bytes per expert
G = {14: 4.56e-3, 32: 4.53e-3}     # s per token, frozen (median over 14 hosts of jobs 095-099)
COUNT = {                          # (misses, in-step fetches) per step, job 099 panel means
    14: {"aa": (58.365, 58.365), "w4": (48.679, 47.103), "w16": (40.198, 29.456), "fetch": (39.738, 19.799)},
    32: {"aa": (28.080, 28.080), "w4": (25.244, 24.347), "w16": (20.476, 19.598), "fetch": (16.442, 9.683)},
}


def main():
    txt = open(sys.argv[1]).read()
    H = int(sys.argv[2])
    bc, bp, bb = bandwidths(txt, H)
    out = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "B_c": bc, "B_p": bp, "B_cp": bb,
           "link_over_cpu": bp / bc, "G_ms": {str(c): 1e3 * g for c, g in G.items()}, "pred_ms": {}}
    for C, states in COUNT.items():
        for st, (mi, fe) in states.items():
            xc, xp = (mi - fe) * S, fe * S
            t = G[C] + max(xc / (bc * 1e9), xp / (bp * 1e9), (xc + xp) / (bb * 1e9))
            out["pred_ms"][f"C{C}_{st}"] = 1e3 * t
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

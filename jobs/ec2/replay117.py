"""Job 117's replay, before any timed run: reads per token of each configuration on this machine's own routing trace
(ec-bench --lookahead), with vmap.py (= main's scripts/value_map.py at 9a93c14): the engine's counters must equal it
(validity V3). Also writes the small files that leave the machine: the trace's selected experts (uint8) and the first
problems of the forecast.

    python replay117.py LA.bin FORECAST.bin C OUT.json SMALL.npz
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forecast_file as FF  # noqa: E402
import vmap as V  # noqa: E402


def main(la, fc, C, out, small):
    C = int(C)
    act, seq, meta = V.load_la(la)
    E = int(meta["n_expert"])
    rows, fseq = FF.read(fc)
    T = act.shape[0]
    res = dict(records=int(T), C=C, E=E, forecast_rows=int(rows.shape[0]),
               forecast_seq_equal=bool(rows.shape[0] == T and np.array_equal(fseq, seq)))
    r = {}
    r["aa"] = V.replay(act, seq, E, C, -1, True, kappa=-1e9)[0]
    r["fetch"] = V.replay(act, seq, E, C, 0, False)[0]
    r["w4"] = V.replay(act, seq, E, C, 4, True, kappa=-1e9)[0]
    r["r7"] = V.replay(act, seq, E, C, 4, True, 0.7, True, 1, kappa=-1e9)[0]
    for lag in (0, 1, 2):
        r[f"f{lag}"] = V.replay_fc(act, seq, E, C, FF.shifted(rows, fseq, seq, lag), 4)
    m, a = V.replay(act, seq, E, C, -1, True, kappa=1.0)
    r["online_k1"] = m + a
    res["reads_per_token"] = r
    gap = r["aa"] - r["fetch"]
    res["read_share_aa_to_fetch"] = {k: (r["aa"] - r[k]) / gap for k in ("w4", "r7", "f0", "f1", "f2")}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res))
    # the first two problems (records with sequence id 0 or 1), uint8 with 255 = none, for re-checks off the machine
    n2 = int((seq < 2).sum())
    np.savez_compressed(small, act=act[:n2].astype(np.uint8), seq=seq[:n2].astype(np.int16), n_expert=E,
                        fc=(rows[:n2].astype(np.int16) % 256).astype(np.uint8), fc_seq=fseq[:n2].astype(np.int16))


if __name__ == "__main__":
    main(*sys.argv[1:])

"""Engine counters of the CPU test against vmap.replay / replay_fc on the engine's own trace."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forecast_file as FF  # noqa: E402
import vmap as V  # noqa: E402

O, C = sys.argv[1], int(sys.argv[2])
act, seq, meta = V.load_la(f"{O}/la.bin")
E = int(meta["n_expert"])
ok = True


def eng(name):
    s = json.load(open(f"{O}/{name}.json"))
    return s["misses"] / max(1, s["steps"]), s


def same(name, ref):
    global ok
    e, s = eng(name)
    eq = abs(e - ref) < 1e-9
    ok &= eq
    print(f"{name:14s} engine {e:.4f} replay {ref:.4f} {'EQUAL' if eq else 'DIFFERENT'}  forecast rows {s.get('forecast_rows')} "
          f"lag {s.get('forecast_lag')} mismatch {s.get('forecast_seq_mismatch')} policy {s['policy']}")


same("exact_w4", V.replay(act, seq, E, C, 4, True, kappa=-1e9)[0])
same("aa", V.replay(act, seq, E, C, -1, True, kappa=-1e9)[0])
same("r7_w4", V.replay(act, seq, E, C, 4, True, 0.7, True, 1, kappa=-1e9)[0])
rows_x, fseq_x = FF.read(f"{O}/fc_exact.bin")
same("fc_exact_l0", V.replay_fc(act, seq, E, C, FF.shifted(rows_x, fseq_x, seq, 0), 4))
same("fc_exact_l1", V.replay_fc(act, seq, E, C, FF.shifted(rows_x, fseq_x, seq, 1), 4))
rows, fseq = FF.read(f"{O}/fc_rand.bin")
for lag in (0, 1, 2):
    same(f"fc_rand_l{lag}", V.replay_fc(act, seq, E, C, FF.shifted(rows, fseq, seq, lag), 4))
# the exact forecast at lag 0 is the exact window
e0, _ = eng("fc_exact_l0")
ew, _ = eng("exact_w4")
print("exact forecast == exact window:", e0 == ew)
ok &= e0 == ew
print("ALL EQUAL" if ok else "MISMATCH")
sys.exit(0 if ok else 1)

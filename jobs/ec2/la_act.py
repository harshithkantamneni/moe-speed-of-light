"""The routing alone from an ec-bench lookahead file (job 121+): every step's selected experts per layer, as int16 in a
compressed npz (seq, step, act [T, L, k], n_expert), small enough for the result channel; the full lookahead file
(router predictions from earlier hidden states) is deleted on the machine.

    python3 la_act.py la_X.bin route_X.npz
"""
import json
import sys

import numpy as np


def main():
    src, dst = sys.argv[1], sys.argv[2]
    h = json.load(open(src + ".json"))
    L, k = h["n_layer"], h["k"]
    per = 2 * k + 24
    a = np.fromfile(src, dtype=np.int16).reshape(-1, 2 + L * per)
    act = a[:, 2:].reshape(-1, L, per)[..., :k].copy()
    np.savez_compressed(dst, seq=a[:, 0].copy(), step=a[:, 1].copy(), act=act, n_expert=np.int32(h["n_expert"]),
                        n_layer=np.int32(L), k=np.int32(k))
    print(f"routes {dst}: {act.shape[0]} steps x {L} layers x {k}, {len(np.unique(a[:, 0]))} sequences")


if __name__ == "__main__":
    main()

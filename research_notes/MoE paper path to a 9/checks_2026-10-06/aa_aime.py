import json, sys
import numpy as np
sys.path.insert(0, "/home/claude/moe-speed-of-light")
from scripts.foresight import _pol, INF
RES = "/home/claude/gpu-branch/results"
AIME = {"gpt-oss-120b": f"{RES}/084c_gptoss_trace@vast/route_aime25_gptoss.npz",
        "qwen3-30b-a3b": f"{RES}/084b_vram_rerun@vast/route_aime25_qwen3.npz"}
lo = json.load(open("/home/claude/moe-speed-of-light/prereg/learned_offline.json"))
for key, Cs in (("gpt-oss-120b", (14, 32, 51)), ("qwen3-30b-a3b", (16, 32, 56))):
    z = np.load(AIME[key]); R3 = np.ascontiguousarray(z["act"].astype(np.int64).transpose(1, 0, 2)); L, T, k = R3.shape; E = int(z["n_expert"])
    for C in Cs:
        aa = 0
        for l in range(L):
            c, p = _pol(np.ascontiguousarray(R3[l]), E, C, -1, 16.0, -1e18); aa += c + p
        aa /= T
        r = lo[f"{key} C{C}"]
        f, o, lr, dep = r["dfa-fetch"], r["opt"], r["learned"], r["dfa"]
        print(f"{key} C={C}: deployed {dep:.2f} dfa-fetch {f:.2f} always-admit {aa:.2f} learned {lr:.2f} opt {o:.2f} | of fetch->opt gap: AA {100*(f-aa)/(f-o):.1f}%  learned {100*(f-lr)/(f-o):.1f}% | AA fewer than dfa-fetch {100*(f-aa)/f:.1f}%, learned fewer {100*(f-lr)/f:.1f}%, learned vs AA {100*(aa-lr)/aa:.1f}%")

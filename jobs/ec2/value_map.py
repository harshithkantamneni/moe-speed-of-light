"""What foresight of a given horizon and accuracy is worth, in host reads, replayed exactly as the engine's oracle
plans it (llama-expert-cache.cpp, oracle_plan_fetch with LLAMA_EC_ORACLE_HYBRID / _RECALL / _FILL / _SEED).

At decode step g the engine plans this step's misses before the step from the trace's record of it (exact: the
router of each layer gives it when the decision is made), and sees the routing of records g+1 .. g+W of the same
sequence through a degraded view: each expert of a future record is kept with probability r, decided by a hash of
(seed, g, tt, layer, position) that this file replays bit for bit; a dropped expert is replaced by a uniformly drawn
one (fill: precision = recall = r) or left out (no fill: recall r, precision 1). Two plans:

  min     MIN with bypass within the window (the engine's `fetch` oracle): misses seen in the window, soonest first,
          each into a free slot or the slot of the resident seen furthest if that is after the miss's own next use.
  hybrid  the online policy (decayed frequency, half-life 16, kappa) with the window on top, as scripts/foresight.py's
          _pol: misses in request order; victim = a resident not requested now that is not seen in the window (lowest
          decayed score) or else the one seen furthest; a seen miss is admitted unless the victim is seen sooner; an
          unseen miss only if its score beats an unseen victim's by kappa. W = -1 is the online policy itself.

Every miss is read from host memory once (admissions are fetched in the step), so reads = misses. The cache starts
empty and is carried across the problems of one context, as in the engine.

    python scripts/value_map.py check DIR          # the engine's stats in DIR against this replay (CPU test)
    python scripts/value_map.py map                # prereg/value_map.json on the AIME-25 routing of jobs 084b/084c
"""
import json
import os
import sys

import numpy as np
from numba import njit

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
INF = np.int64(2**62)
M64 = 0xFFFFFFFFFFFFFFFF


@njit(cache=True)
def _splitmix(z):
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(cache=True)
def _future(act, g, tt, il, i, E, r, fill, seed):
    e = act[tt, il, i]
    if r >= 1.0 or e < 0:
        return e
    x = (np.uint64(seed) ^ (np.uint64(g) * np.uint64(0x9E3779B97F4A7C15)) ^ (np.uint64(tt) * np.uint64(0xC2B2AE3D27D4EB4F))
         ^ (np.uint64(il) * np.uint64(0x165667B19E3779F9)) ^ (np.uint64(i) * np.uint64(0xD6E8FEB86659FD93)))
    h = _splitmix(x)
    u = np.float64(h >> np.uint64(11)) * (1.0 / 9007199254740992.0)
    if u < r:
        return e
    if fill:
        return np.int64(_splitmix(h) % np.uint64(E))
    return -1


@njit(cache=True)
def _layer(act, seqs, il, E, C, W, hybrid, r, fill, seed, hl, kappa, tab):
    T, L, k = act.shape
    resident = np.full(C, -1, np.int64)
    slot_of = np.full(E, -1, np.int64)
    score = np.zeros(E)
    last = np.zeros(E, np.int64)
    misses = 0
    forced = 0
    nu = np.empty(E, np.int64)
    sel = np.zeros(E, np.bool_)
    res = np.empty(C, np.int64)
    taken = np.zeros(C, np.bool_)
    pe = np.empty(k, np.int64)
    ps = np.empty(k, np.int64)
    missed = np.empty(k, np.int64)
    for g in range(T):
        nu[:] = INF
        if W < 0:
            t_end = g + 1
        elif W > 0:
            t_end = min(T, g + 1 + W)
        else:
            t_end = T
        nseen = 0
        tt = g + 1
        while tt < t_end and seqs[tt] == seqs[g] and nseen < E:
            for i in range(k):
                e = _future(act, g, tt, il, i, E, r, fill, seed)
                if e >= 0 and e < E and nu[e] == INF:
                    nu[e] = tt
                    nseen += 1
            tt += 1
        sel[:] = False
        for i in range(k):
            sel[act[g, il, i]] = True
        for i in range(k):
            if slot_of[act[g, il, i]] < 0:
                misses += 1
        res[:] = resident
        taken[:] = False
        n = 0
        if hybrid:
            for i in range(k):
                e = act[g, il, i]
                if slot_of[e] >= 0:
                    continue
                slot = -1
                for s in range(C):
                    if res[s] < 0 and not taken[s]:
                        slot = s
                        break
                if slot < 0:
                    vbeyond = False
                    vkey = 0.0
                    for s in range(C):
                        c = res[s]
                        if c < 0 or taken[s] or sel[c]:
                            continue
                        if nu[c] == INF:
                            dt = g - last[c]
                            key = score[c] * (tab[dt] if dt < tab.shape[0] else (0.5 ** (1.0 / hl)) ** dt)
                            if (not vbeyond) or key < vkey:
                                slot = s
                                vbeyond = True
                                vkey = key
                        elif not vbeyond:
                            key = float(nu[c])
                            if slot < 0 or key > vkey:
                                slot = s
                                vkey = key
                    if slot < 0:
                        continue
                    if nu[e] == INF:
                        dt = g - last[e]
                        se = score[e] * (tab[dt] if dt < tab.shape[0] else (0.5 ** (1.0 / hl)) ** dt) + 1.0
                        if not (vbeyond and se > vkey + kappa):
                            continue
                    elif (not vbeyond) and float(nu[e]) >= vkey:
                        continue
                res[slot] = e
                taken[slot] = True
                pe[n] = e
                ps[n] = slot
                n += 1
        else:
            nm = 0
            for i in range(k):
                e = act[g, il, i]
                if slot_of[e] < 0 and nu[e] < INF:
                    # stable insertion by next use (libstdc++ sorts <= 16 elements by insertion sort)
                    j = nm
                    while j > 0 and nu[missed[j - 1]] > nu[e]:
                        missed[j] = missed[j - 1]
                        j -= 1
                    missed[j] = e
                    nm += 1
            for q in range(nm):
                e = missed[q]
                slot = -1
                for s in range(C):
                    if res[s] < 0 and not taken[s]:
                        slot = s
                        break
                if slot < 0:
                    furthest = np.int64(-1)
                    for s in range(C):
                        c = res[s]
                        if c < 0 or taken[s] or sel[c]:
                            continue
                        if nu[c] > furthest:
                            furthest = nu[c]
                            slot = s
                    if slot < 0 or furthest <= nu[e]:
                        break
                res[slot] = e
                taken[slot] = True
                pe[n] = e
                ps[n] = slot
                n += 1
        forced += n
        for q in range(n):
            e = pe[q]
            s = ps[q]
            v = resident[s]
            if v >= 0:
                slot_of[v] = -1
            resident[s] = e
            slot_of[e] = s
        for i in range(k):
            e = act[g, il, i]
            dt = g - last[e]
            score[e] = score[e] * (tab[dt] if dt < tab.shape[0] else (0.5 ** (1.0 / hl)) ** dt) + 1.0
            last[e] = g
    return misses, forced


def decay_table(hl=16.0, n=4096):
    d = 0.5 ** (1.0 / hl)
    t = np.empty(n)
    v = 1.0
    for i in range(n):
        t[i] = v
        v *= d
    return t


def replay(act, seqs, E, C, W=0, hybrid=False, r=1.0, fill=True, seed=0, hl=16.0, kappa=1.0):
    """act [T, L, k] int64, seqs [T]. Returns (reads per token, admissions per token), summed over layers."""
    T, L, k = act.shape
    tab = decay_table(hl)
    m = f = 0
    for il in range(L):
        a, b = _layer(act, seqs, il, E, C, int(W), bool(hybrid), float(r), bool(fill), int(seed) & M64, hl, float(kappa), tab)
        m += a
        f += b
    return m / T, f / T


def load_la(path):
    meta = json.load(open(path + ".json"))
    L, k = int(meta["n_layer"]), int(meta["k"])
    raw = np.fromfile(path, np.int16)
    stride = 2 + L * (2 * k + 24)
    rec = raw[: len(raw) // stride * stride].reshape(-1, stride)
    act = np.stack([rec[:, 2 + l * (2 * k + 24): 2 + l * (2 * k + 24) + k] for l in range(L)], axis=1).astype(np.int64)
    return act, rec[:, 0].astype(np.int64), meta


def parse_cfg(cfg):
    kv = dict(p.split("=", 1) for p in cfg.split(":") if "=" in p)
    return dict(C=int(kv["slots"]), W=int(kv.get("oracle_w", 0)), hybrid=kv.get("oracle_hybrid", "0") == "1",
                r=float(kv.get("oracle_recall", 1.0)), fill=kv.get("oracle_fill", "1") == "1",
                seed=int(float(kv.get("oracle_seed", 0))), kappa=float(kv.get("kappa", 0.0)),
                hl=float(kv.get("half_life", 16.0)), stats=kv.get("stats"))


def check(d):
    """d holds la.bin (+.json), configs.txt (one engine configuration string per line) and the stats files."""
    act, seqs, meta = load_la(os.path.join(d, "la.bin"))
    E = int(meta.get("n_expert", 0)) or int(act.max()) + 1
    ok = True
    for line in open(os.path.join(d, "configs.txt")):
        line = line.strip()
        if not line:
            continue
        c = parse_cfg(line)
        s = json.load(open(c["stats"]))
        st = max(1, s["steps"])
        eng = s["misses"] / st
        sim, adm = replay(act[: s["steps"]], seqs[: s["steps"]], E, c["C"], c["W"], c["hybrid"], c["r"], c["fill"], c["seed"], c["hl"], c["kappa"])
        same = abs(eng - sim) < 1e-9
        ok &= same
        print(f"{os.path.basename(c['stats']):18s} engine reads/tok {eng:.4f} replay {sim:.4f} forced {s.get('oracle_forced', 0)/st:.4f} vs {adm:.4f} {'EQUAL' if same else 'DIFFERENT'}  plan {s.get('oracle_plan_us_per_step', 0)} us/step")
    print("ALL EQUAL" if ok else "MISMATCH")
    return ok


TRACES = {"gpt-oss-120b": ("084c_gptoss_trace@vast/route_aime25_gptoss.npz", (14, 32, 51)),
          "qwen3-30b-a3b": ("084b_vram_rerun@vast/route_aime25_qwen3.npz", (16, 32, 56))}
WS = (-1, 1, 2, 4, 8, 16, 32, 0)
RS = (1.0, 0.9, 0.7, 0.5, 0.3)


def build_map(results, out, seeds=(1, 2, 3)):
    res = {}
    for model, (rel, Cs) in TRACES.items():
        z = np.load(os.path.join(results, rel))
        act = z["act"].astype(np.int64)
        seqs = z["seq"].astype(np.int64)
        E = int(z["n_expert"])
        for C in Cs:
            cell = {}
            online = replay(act, seqs, E, C, -1, True)[0]
            opt = replay(act, seqs, E, C, 0, False)[0]
            cell["online_hybrid"] = online
            cell["min_fetch"] = opt
            cell["exact"] = {str(W): replay(act, seqs, E, C, W, True)[0] for W in WS if W != -1}
            for fill in (True, False):
                tag = "fill" if fill else "drop"
                cell[tag] = {}
                for W in (4, 8, 16, 0):
                    for r in RS[1:]:
                        v = [replay(act, seqs, E, C, W, True, r, fill, s)[0] for s in seeds]
                        cell[tag][f"W{W}_r{r}"] = float(np.mean(v))
            cell["min_fetch_noisy"] = {f"r{r}": float(np.mean([replay(act, seqs, E, C, 0, False, r, True, s)[0] for s in seeds])) for r in RS[1:]}
            gap = online - opt
            cell["share"] = {k: (online - v) / gap for k, v in cell["exact"].items()}
            for tag in ("fill", "drop"):
                cell[f"share_{tag}"] = {k: (online - v) / gap for k, v in cell[tag].items()}
            res[f"{model} C{C}"] = cell
            print(model, C, f"online {online:.2f} opt {opt:.2f}", {k: round(v, 2) for k, v in cell["share"].items()},
                  {k: round(v, 2) for k, v in cell["share_fill"].items() if k.startswith("W0") or k.startswith("W8")}, flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "check":
        sys.exit(0 if check(sys.argv[2]) else 1)
    if sys.argv[1] == "map":
        build_map("/home/claude/gpu-branch/results", os.path.join(ROOT, "prereg", "value_map.json"))

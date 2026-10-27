"""Job 124's scorer, registered with its header (jobs/124_slowlink@vast.sh pins this file by its sha256).

It reads one results directory and nothing else: the hosts' own records (ec_g_C*_r*.jsonl, st_g_C*_r*_<arm>.json,
concur.txt, gpu.csv, cpu.txt / lscpu.txt, cores.txt, v0.txt, validity.txt, gate.txt, mode.txt, DONE, NOT_STARTED.txt).
Standard library only; no repository import; nothing it reads is executed. The constants of the frozen model, the
clause thresholds and the frozen counters of clause C0 are in this file.

    python3 -I scorer.py RESULTS_DIR [--job 124|123] [--out SCORECARD.json]

--job 123 scores job 123's hosts (results/123*_confirm@vast) under job 123's header (its clauses C0, T1-T6, P1-P4 and
its decision), with the frozen counters that job 123's rebuilt scorer used; --job 124 (the default) scores this job's
hosts (results/124*_slowlink@vast) under this job's header.

Self-test (run before this file was committed, recorded here and in the job header): pointed at job 123's results
(branch gpu at 5d25868a, results/123*_confirm@vast) with --job 123, this scorer reproduces job 123's rebuilt scorecard
(main: prereg/p2/scorecard_123.json, written by scripts/p2/job123.py from the engine table): all 8,348 values agree
(8,347 exactly, 1 within 1.3e-15 relative; none differs, none missing), every clause's verdict and the decision
(confirmed); and so do the job 123 numbers recovery/known-state.md records (median |error| 1.7%, 90th percentile 6.0%,
90 cells on 5 machines, T6 40 of 40 pairs in 8 contrasts, pairs ordered by the model / the hit rate / raw reads 94.9 /
76.4 / 80.0%, the per-token form's T1 2.6%). Written by scripts/p2/job124_selftest.py on main
(prereg/p2/job124_selftest.md).

The frozen model (job 123's header, unchanged): T_pred = max(G + sum over layer-steps of max(f s_p, c s_c, [0 < f < n]
n s_b), root(G + X_in s_p, A s_p), root(G + (X_c + X_in) s_b, A s_b)), root(a, v) = (a + v + sqrt((a + v)^2 - 4 v G))
/ 2; X_c = M - F, X_in = F + P; s_x = S / B_x, S = 13,253,760 bytes; a layer-step's n misses split into f link copies
and c = n - f CPU reads, f = table[n] of the arm's own fetch table (scaled to the fetches the engine made in that
layer) for the deployed cache and LRU without fetch-on-admit, binomially by F_l / M_l for the oracles and
fetch-on-admit; a layer-ahead prefetch rides on the previous layer's link side. G = 4.413 ms on an RTX 5090 (job 123's
constant); on an RTX 4090, G_4090_MS below (the same definition, the median Nsight GPU time over 13 RTX 4090s). B_c,
B_p, B_both: the machine's first probe (its own concur.txt: V0 admits only GPUs never rented before) at the run's
helper count, read as jobs/ec2/fetch_table.py reads it. Error = T_pred / T - 1; a cell is a machine x arm x budget.

How runs are read (as the engine table of paper 2, scripts/p2/engine_table.py, reads them): an arm-round is one
configuration in one round file; it counts when every problem of the host's set ran at the full decode length and V3
holds (its loss per token within 1% of base's in the same file); T = the mean over counted rounds of (sum of decode ms
/ decoded tokens); counters per token = the mean over counted rounds of the stats file's totals / steps; per-layer
counters from the first counted round. A host is valid when it started, is not a smoke run, recorded no failed gate
(a line "V0 / V0c / VR / V1 / V2 ... FAIL" in v0.txt or validity.txt) and has a counted run; a GPU already valid in the
job does not count again.
"""
import argparse
import json
import math
import os
import re
import statistics

S_BYTES = 13253760.0          # gpt-oss-120b MXFP4 expert (job 123's header)
G_5090_MS = 4.413             # job 123's header: median Nsight GPU time of gpt-oss-120b on 24 RTX 5090s
V3_TOL = 0.01

# ---- BEGIN GENERATED
# Written by scripts/p2/job124_design.py (main) from prereg/p2: do not edit by hand.
G_4090_MS = 5.328962499999999   # gpt-oss-120b on an RTX 4090: median over 13 machines of the Nsight GPU time (time_model.json G_table)
FROZEN_123 = {
 "aa|C14|fetch=0,0,1,2,3:fetch_on_admit=1:kappa=-1000000000:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 58.3654296875,
 "aa|C32|fetch=0,0,1,2,3:fetch_on_admit=1:kappa=-1000000000:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 28.0798828125,
 "base0|C14|fetch=0:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 70.3240234375,
 "base0|C32|fetch=0:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 39.812109375,
 "baselaw|C14|fetch=0,0,1,1,2:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 63.6060546875,
 "baselaw|C14|fetch=0,0,1,2,2:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 63.0708984375,
 "baselaw|C14|fetch=0,0,1,2,3:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 63.1078125,
 "baselaw|C32|fetch=0,0,1,1,2:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 34.7353515625,
 "baselaw|C32|fetch=0,0,1,2,2:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 34.023828125,
 "base|C14|fetch=0,0,1,2,3:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 63.1078125,
 "base|C32|fetch=0,0,1,2,3:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 33.8248046875,
 "both3p|C14|chunk=16777216:fetch=0,0,1,2,3:kappa=1:mailbox=1:max_pending=64:oracle=file:oracle_bypass=0:oracle_fetch=1:oracle_lead=3:oracle_w=0:overlap=1:paced=1:policy=oracle:tdec=1": 46.666796875,
 "both3p|C32|chunk=16777216:fetch=0,0,1,2,3:kappa=1:mailbox=1:max_pending=64:oracle=file:oracle_bypass=0:oracle_fetch=1:oracle_lead=3:oracle_w=0:overlap=1:paced=1:policy=oracle:tdec=1": 21.834765625,
 "bypass0|C14|fetch=0:kappa=1:mailbox=1:oracle=file:oracle_bypass=1:oracle_w=0:overlap=1:paced=0:policy=oracle:tdec=1": 63.852734375,
 "bypass0|C32|fetch=0:kappa=1:mailbox=1:oracle=file:oracle_bypass=1:oracle_w=0:overlap=1:paced=0:policy=oracle:tdec=1": 27.57109375,
 "fetchplan|C14|fetch=0,0,1,2,3:kappa=1:mailbox=1:oracle=file:oracle_bypass=1:oracle_fetch=1:oracle_plan=file:oracle_w=0:overlap=1:paced=0:policy=oracle:tdec=1": 40.7267578125,
 "fetchplan|C32|fetch=0,0,1,2,3:kappa=1:mailbox=1:oracle=file:oracle_bypass=1:oracle_fetch=1:oracle_plan=file:oracle_w=0:overlap=1:paced=0:policy=oracle:tdec=1": 17.3107421875,
 "fetch|C14|fetch=0,0,1,2,3:kappa=1:mailbox=1:oracle=file:oracle_bypass=1:oracle_fetch=1:oracle_w=0:overlap=1:paced=0:policy=oracle:tdec=1": 39.7453125,
 "fetch|C32|fetch=0,0,1,2,3:kappa=1:mailbox=1:oracle=file:oracle_bypass=1:oracle_fetch=1:oracle_w=0:overlap=1:paced=0:policy=oracle:tdec=1": 16.4640625
}
FROZEN_124 = {
 "aa|C14|fetch=0,0,1,2,3:fetch_on_admit=1:kappa=-1000000000:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 58.3654296875,
 "aa|C32|fetch=0,0,1,2,3:fetch_on_admit=1:kappa=-1000000000:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 28.0798828125,
 "base0|C14|fetch=0:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 70.3240234375,
 "base0|C32|fetch=0:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 39.800390625,
 "baselaw|C14|fetch=0,0,0,0,0:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 70.3240234375,
 "baselaw|C14|fetch=0,0,0,0,1:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 69.421484375,
 "baselaw|C14|fetch=0,0,0,1,1:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 67.34697265624999,
 "baselaw|C14|fetch=0,0,1,1,0:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 65.1244140625,
 "baselaw|C14|fetch=0,0,1,1,1:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 63.84521484375,
 "baselaw|C14|fetch=0,0,1,1,2:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 63.6060546875,
 "baselaw|C14|fetch=0,0,1,2,2:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 63.0708984375,
 "baselaw|C14|fetch=0,0,1,2,3:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 63.1078125,
 "baselaw|C32|fetch=0,0,0,0,0:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 39.812109375,
 "baselaw|C32|fetch=0,0,0,0,1:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 39.41171875,
 "baselaw|C32|fetch=0,0,0,1,1:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 38.078125,
 "baselaw|C32|fetch=0,0,1,1,0:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 35.3587890625,
 "baselaw|C32|fetch=0,0,1,1,1:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 34.94951171875,
 "baselaw|C32|fetch=0,0,1,1,2:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 34.7353515625,
 "baselaw|C32|fetch=0,0,1,2,2:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 34.023828125,
 "baselaw|C32|fetch=0,0,1,2,3:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 33.8248046875,
 "base|C14|fetch=0,0,1,2,3:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 63.1078125,
 "base|C32|fetch=0,0,1,2,3:kappa=1:mailbox=1:oracle_w=0:overlap=1:paced=0:policy=dfa:tdec=1": 33.8248046875,
 "both3p|C14|chunk=16777216:fetch=0,0,1,2,3:kappa=1:mailbox=1:max_pending=64:oracle=file:oracle_bypass=0:oracle_fetch=1:oracle_lead=3:oracle_w=0:overlap=1:paced=1:policy=oracle:tdec=1": 46.68046875,
 "both3p|C32|chunk=16777216:fetch=0,0,1,2,3:kappa=1:mailbox=1:max_pending=64:oracle=file:oracle_bypass=0:oracle_fetch=1:oracle_lead=3:oracle_w=0:overlap=1:paced=1:policy=oracle:tdec=1": 21.8330078125,
 "bypass0|C14|fetch=0:kappa=1:mailbox=1:oracle=file:oracle_bypass=1:oracle_w=0:overlap=1:paced=0:policy=oracle:tdec=1": 63.834375,
 "bypass0|C32|fetch=0:kappa=1:mailbox=1:oracle=file:oracle_bypass=1:oracle_w=0:overlap=1:paced=0:policy=oracle:tdec=1": 27.57109375,
 "fetchplan|C14|fetch=0,0,1,2,3:kappa=1:mailbox=1:oracle=file:oracle_bypass=1:oracle_fetch=1:oracle_plan=file:oracle_w=0:overlap=1:paced=0:policy=oracle:tdec=1": 40.7267578125,
 "fetchplan|C32|fetch=0,0,1,2,3:kappa=1:mailbox=1:oracle=file:oracle_bypass=1:oracle_fetch=1:oracle_plan=file:oracle_w=0:overlap=1:paced=0:policy=oracle:tdec=1": 17.3107421875,
 "fetch|C14|fetch=0,0,1,2,3:kappa=1:mailbox=1:oracle=file:oracle_bypass=1:oracle_fetch=1:oracle_w=0:overlap=1:paced=0:policy=oracle:tdec=1": 39.7453125,
 "fetch|C32|fetch=0,0,1,2,3:kappa=1:mailbox=1:oracle=file:oracle_bypass=1:oracle_fetch=1:oracle_w=0:overlap=1:paced=0:policy=oracle:tdec=1": 16.4640625
}
# ---- END GENERATED

# clause thresholds (job 123's header for both jobs; job 124's own clauses after them)
TH = dict(C0_tol=0.01, C0_tol_paced=0.05, C0_spread=0.01, T1=0.07, T2=0.11, T3_within=0.10, T3_share=0.70,
          T4_min_arms=6, T4_rho=0.6, T5_gap=0.01, T5_share=0.85, T6_hit=0.005, T6_gap=0.10, T6_share=0.85,
          T6_min_pairs=5, P_min_machines=3, P2_ratio=1.10, P4_share=0.80, min_valid=3,
          # job 124 only
          VR_max=0.6, F_share=2 / 3, F_min_cells=3, F3_share=0.80, H_min_pairs=10, K_within=0.10, K_share=0.70,
          K_min_cells=5)
FROZEN_ARMS = ("base", "aa", "fetch", "fetchplan", "both3p", "base0", "bypass0", "baselaw")
PACED_ARMS = ("both3p",)
AGREE_ARMS = ("hitopt", "lru")
ONLINE_ARMS = ("baselaw", "base", "lru", "aa", "base0")          # the arms that need no recorded future routing
TWO_PATH_ARMS = ("fetch", "fetchplan")                          # MIN read once and Few: link or CPU in the step

PROFILES = {
    "123": dict(dir_re=r"^123[a-z]\d?_confirm@vast$", frozen="FROZEN_123", G={"5090": G_5090_MS}),
    "124": dict(dir_re=r"^124[a-z]\d?_slowlink@vast$", frozen="FROZEN_124", G=None),
}


# --- reading ---------------------------------------------------------------------------------------------------------
def rd(p):
    try:
        with open(p, errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def jload(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def num(pat, txt, f=float):
    m = re.search(pat, txt)
    return f(m.group(1)) if m else None


def kv(cfg):
    """the configuration's pairs as the engine applies them: in order, a repeated key keeps its last value"""
    d = {}
    for x in (cfg or "").split(":"):
        if "=" in x:
            k, v = x.split("=", 1)
            d.pop(k, None)
            d[k] = v
    return d


def config_key(cfg):
    d = kv(cfg)
    return ":".join(f"{k}={'file' if k in ('oracle', 'oracle_plan') else d[k]}" for k in sorted(d)
                    if k not in ("stats", "slots", "helpers"))


def short_cpu(cpu):
    c = re.sub(r"\(R\)|\(TM\)|®|™", "", cpu or "")
    c = re.sub(r"\b(AMD|Intel|Processor|CPU)\b|\d+-Cores?|@.*", "", c)
    c = re.sub(r"\bRyzen Threadripper\b", "Threadripper", c)
    c = re.sub(r"\s+", " ", c).strip()
    return c or None


def vendor(cpu):
    c = cpu or ""
    return "AMD" if re.search(r"AMD|Ryzen|EPYC|Threadripper", c) else "Intel" if re.search(r"Intel|Core|Xeon", c) else None


def card_class(card):
    c = card or ""
    return "5090" if "5090" in c else "4090" if "4090" in c else None


def bandwidths(txt, helpers):
    """jobs/ec2/fetch_table.py bandwidths: B_c at the helper count (linear between probed thread counts), B_p the
    largest 16 MB x 64-block zero-copy read, B_both the median CPU + zero-copy sum"""
    by = {}
    for t, v in re.findall(r"cpu_read_gbs t=(\d+) ([\d.]+)", txt):
        by.setdefault(int(t), []).append(float(v))
    pts = sorted((t, statistics.mean(v)) for t, v in by.items())
    zc = [float(x) for x in re.findall(r"pcie_zerocopy_gbs chunk=16MB blocks=64 ([\d.]+)", txt)]
    both = [float(z) for z in re.findall(r"concurrent t=\d+ pcie=zerocopy64 cpu_gbs [\d.]+ pcie_gbs [\d.]+ sum ([\d.]+)", txt)]
    if not pts or not zc or not both or helpers is None:
        return None
    if helpers <= pts[0][0]:
        bc = pts[0][1]
    elif helpers >= pts[-1][0]:
        bc = pts[-1][1]
    else:
        bc = None
        for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
            if t0 <= helpers <= t1:
                bc = v0 + (v1 - v0) * (helpers - t0) / (t1 - t0)
                break
    return bc, max(zc), statistics.median(both)


def mean(v):
    return sum(v) / len(v) if v else None


def records(p):
    out = []
    for line in rd(p).splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and all(k in r for k in ("config", "seq", "n_decode", "decode_ms")):
            out.append(r)
    return out


def read_host(d):
    name = os.path.basename(d)
    txt = {f: rd(os.path.join(d, f)) for f in ("v0.txt", "validity.txt", "cpu.txt", "lscpu.txt", "gate.txt",
                                               "NOT_STARTED.txt", "DONE", "mode.txt", "gpu.csv", "cores.txt",
                                               "nvidia-smi-q.txt", "concur.txt")}
    fails = [l.strip() for f in ("v0.txt", "validity.txt") for l in txt[f].splitlines()
             if re.match(r"(V0c?|VR|V1|V2)\b.*FAIL", l)]
    cpu_full = None
    for f in ("lscpu.txt", "cpu.txt"):
        cpu_full = num(r"Model name:\s*(.+)", txt[f], str)
        if cpu_full:
            cpu_full = cpu_full.strip()
            break
    cpu_rep = num(r"Model name:\s*(.+)", txt["cpu.txt"], str)
    g = txt["gpu.csv"].splitlines()
    card = g[1].split(",")[0].strip() if len(g) > 1 and "," in g[1] else num(r"Product Name\s*:\s*(.+)", txt["nvidia-smi-q.txt"], str)
    uuid = num(r"GPU UUID\s*:\s*(\S+)", txt["nvidia-smi-q.txt"], str) or num(r"GPU UUID (GPU-\S+?);", txt["v0.txt"], str)
    return dict(host=name, started=not txt["NOT_STARTED.txt"], smoke="SMOKE=1" in txt["mode.txt"],
                cpu=cpu_rep.strip() if cpu_rep else None, cpu_model=cpu_full, card=card, card_class=card_class(card),
                vendor=vendor(cpu_full), uuid=uuid, gate_failures=fails,
                ratio_gate=(re.search(r"VR: B_p / B_c = ([\d.]+)", txt["gate.txt"]) or [None, None])[1],
                not_started=txt["NOT_STARTED.txt"].strip() or None, done=txt["DONE"].strip() or None,
                cores=num(r"usable physical cores (\d+)", txt["cores.txt"], int),
                plan_dropped="NO PLAN: fetchplan dropped" in txt["validity.txt"],
                probe=txt["concur.txt"] if "cpu_read_gbs" in txt["concur.txt"] else "")


ARM_RE = re.compile(r"^st_([a-z])_(?:C(\d+)_)?(?:r(\d+)([AB]?)_)?")


def host_runs(d, h):
    """the host's runs: one per (process tag, budget, arm), as the engine table groups arm-rounds"""
    inst = []
    for f in sorted(os.listdir(d)):
        m = re.match(r"ec_([a-z])(?:_C(\d+))?(?:_r(\d+)([AB]?))?\.jsonl$", f)
        if not m:
            continue
        groups = {}
        for r in records(os.path.join(d, f)):
            c = kv(r["config"])
            label = os.path.basename(c["stats"])[:-len(".json")] if c.get("stats", "").endswith(".json") else ""
            if label:
                groups.setdefault((r["config"], label), []).append(r)
        for (cfg, label), rows in groups.items():
            mm = ARM_RE.match(label)
            if not mm:
                continue
            arm = label[mm.end():]
            rnd = f"r{m.group(3)}{m.group(4)}" if m.group(3) else (f"r{mm.group(3)}{mm.group(4)}" if mm.group(3) else "r1")
            st = jload(os.path.join(d, label + ".json"))
            inst.append(dict(file=f, cfg=cfg, tag=mm.group(1), arm=arm, round=rnd, rows=rows,
                             stats=st if isinstance(st, dict) and "misses" in st else None))
    text_of = lambda rows: "+".join(sorted({r.get("domain") or "?" for r in rows}))  # noqa: E731
    expect, by = {}, {}
    for i in inst:
        e = expect.setdefault((i["tag"], text_of(i["rows"])), [set(), 0])
        e[0] |= {r["seq"] for r in i["rows"]}
        e[1] = max(e[1], max(r["n_decode"] for r in i["rows"]))
        C = int(kv(i["cfg"])["slots"]) if kv(i["cfg"]).get("slots") else None
        by.setdefault((i["tag"], C, i["arm"]), []).append(i)
    base_loss = {}
    for (tag, C, arm), ins in by.items():
        if arm == "base" or arm.endswith("_base"):
            for i in ins:
                base_loss.setdefault((i["file"], C), mean([r["nll_sum"] / r["nll_n"] for r in i["rows"] if r.get("nll_n")]))
    runs = []
    for (tag, C, arm), ins in sorted(by.items(), key=lambda x: (x[0][0], x[0][1] or 0, x[0][2])):
        ins = sorted(ins, key=lambda i: (i["round"], i["file"]))
        counted, notes = [], []
        for i in ins:
            e = expect[(tag, text_of(i["rows"]))]
            seqs = [r["seq"] for r in i["rows"]]
            complete = sorted(seqs) == sorted(e[0]) and all(r["n_decode"] == e[1] for r in i["rows"])
            lo = mean([r["nll_sum"] / r["nll_n"] for r in i["rows"] if r.get("nll_n")])
            bl = base_loss.get((i["file"], C), next((v for (f_, c_), v in sorted(base_loss.items(), key=str) if f_ == i["file"]), None))
            v3 = None if bl is None or lo is None else abs(lo / bl - 1) <= V3_TOL
            if not complete:
                notes.append(f"{i['file']} {i['round']}: incomplete ({len(set(seqs))} of {len(e[0])} problems)")
            elif v3 is False:
                notes.append(f"{i['file']} {i['round']}: V3 failed (loss {lo:.4f} vs base {bl:.4f})")
            else:
                counted.append(i)
        c0 = kv(ins[0]["cfg"])
        st0 = next((i["stats"] for i in counted if i["stats"]), None) or next((i["stats"] for i in ins if i["stats"]), None)
        reasons = [] if counted else ["no counted round" + (": " + "; ".join(notes) if notes else "")]
        if c0.get("oracle_plan") and st0 is not None and not st0.get("oracle_plan"):
            reasons.append("plan not read: the engine ran greedy MIN")
        times = [sum(r["decode_ms"] for r in i["rows"]) / sum(r["n_decode"] for r in i["rows"]) for i in counted]
        cts = []
        for i in counted:
            s = i["stats"]
            if not s:
                continue
            n = max(1, s.get("steps") or 0)
            M, F, P, A, Hh = (s.get(k) or 0 for k in ("misses", "fetches", "prefetches", "admits", "hits"))
            cts.append(dict(M=M / n, F=F / n, P=P / n, A=A / n, host_reads=(M + A + P) / n,
                            hit_rate=Hh / (Hh + M) if Hh + M else None, X_c=(M - F) / n, X_in=(F + P) / n))
        first = next((i for i in counted if i["stats"]), None)
        run = dict(tag=tag, C=C, arm=arm, config_string=re.sub(r":?stats=[^:]*", "", ins[0]["cfg"]),
                   fetch_table=c0.get("fetch") or (st0 or {}).get("fetch_table"),
                   helpers=int(c0["helpers"]) if c0.get("helpers") else (st0 or {}).get("helpers"),
                   n_rounds=len(counted), time_ms=mean(times), round_notes=notes, reasons=reasons,
                   n_problems=len({r["seq"] for i in ins for r in i["rows"]}),
                   layers=dict(steps=first["stats"]["steps"], layers=[
                       dict(miss_hist=L.get("miss_hist"), fetches=L.get("fetches", 0), prefetches=L.get("prefetches", 0))
                       for L in first["stats"].get("layers") or []]) if first else None,
                   run_id=f"{h['host']}|teacher-forced|{tag}|C{C}|{arm}")
        for k in ("M", "F", "P", "A", "host_reads", "hit_rate", "X_c", "X_in"):
            v = [c[k] for c in cts if c[k] is not None]
            run[k] = mean(v) if v else None
        runs.append(run)
    return runs


# --- the frozen model ----------------------------------------------------------------------------------------------------
def root(a, v, G):
    return (a + v + math.sqrt(max((a + v) ** 2 - 4.0 * v * G, 0.0))) / 2.0


def t_pred(run, G, S=S_BYTES):
    d = kv(run["config_string"])
    binom = d.get("policy") == "oracle" or d.get("fetch_on_admit") == "1"
    tab = [int(x) for x in re.findall(r"\d+", run["fetch_table"] or d.get("fetch", "0"))]
    sc, sp, sb = (S / run[k] / 1e6 for k in ("B_c", "B_p", "B_both"))
    steps, Ls = run["layers"]["steps"], run["layers"]["layers"]
    crit = G
    for l, lay in enumerate(Ls):
        h = lay["miss_hist"]
        M_l = sum(n * x for n, x in enumerate(h))
        F_l = lay["fetches"]
        q = Ls[l + 1]["prefetches"] / steps if l + 1 < len(Ls) else 0.0
        p0 = math.floor(q)
        pmix = [(1 - (q - p0), p0), (q - p0, p0 + 1)]
        if binom:
            phi = F_l / M_l if M_l else 0.0
        else:
            E_l = sum(x * min(n, tab[n] if n < len(tab) else 0) for n, x in enumerate(h))
            rho = F_l / E_l if E_l else 0.0
        for n, x in enumerate(h):
            if not x:
                continue
            split = ([(math.comb(n, f) * phi ** f * (1 - phi) ** (n - f), f) for f in range(n + 1)] if binom else
                     [(1.0, min(n, min(n, tab[n] if n < len(tab) else 0) * rho))])
            for pf, f in split:
                for pp, p in pmix:
                    if pf > 0 and pp > 0:
                        c, ff = n - f, f + p
                        crit += x / steps * pf * pp * max(ff * sp, c * sc, (c + ff) * sb if (c > 0 and ff > 0) else 0.0)
    Xc, Xin, A = run["X_c"], run["X_in"], run["A"]
    return max(crit, root(G + Xin * sp, A * sp, G), root(G + (Xc + Xin) * sb, A * sb, G))


def t_token(run, G, S=S_BYTES, background=True):
    """the per-token two-path form (the registered competitor of K2) and, without background copies, the law"""
    sc, sp, sb = (S / run[k] / 1e6 for k in ("B_c", "B_p", "B_both"))
    Xc, Xin, A = run["X_c"], run["X_in"], run["A"]
    tok = G + max(Xin * sp, Xc * sc, (Xc + Xin) * sb if (Xc > 0 and Xin > 0) else 0.0)
    if not background:
        return tok
    return max(tok, root(G + Xin * sp, A * sp, G), root(G + (Xc + Xin) * sb, A * sb, G))


# --- statistics ------------------------------------------------------------------------------------------------------
def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num_ = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num_ / den if den else None


def percentile(v, p):
    """linear interpolation between order statistics (numpy's default)"""
    v = sorted(v)
    x = (len(v) - 1) * p / 100
    lo = math.floor(x)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (x - lo)


def verdict(holds, tested=True):
    return "untested" if not tested else ("holds" if holds else "fails")


def form_clauses(cells, hosts, key):
    """T1-T6 for one prediction column (descriptive for the forms the header did not register)"""
    e = [abs(c[key] / c["T"] - 1) for c in cells]
    per = [statistics.median(abs(c[key] / c["T"] - 1) for c in cells if c["host"] == h) for h in hosts]
    rhos, prs = [], []
    for h in hosts:
        for C in sorted({c["C"] for c in cells if c["host"] == h}):
            cc = [c for c in cells if c["host"] == h and c["C"] == C]
            if len(cc) >= TH["T4_min_arms"]:
                rhos.append(spearman([c["T"] for c in cc], [c[key] for c in cc]))
            for i in range(len(cc)):
                for k in range(i + 1, len(cc)):
                    fa, sl = (cc[i], cc[k]) if cc[i]["T"] <= cc[k]["T"] else (cc[k], cc[i])
                    if sl["T"] / fa["T"] - 1 <= TH["T5_gap"]:
                        continue
                    opp = abs(fa["hit"] - sl["hit"]) > TH["T6_hit"] and ((fa["hit"] > sl["hit"]) != (fa[key] < sl[key]))
                    gap = max(fa[key], sl[key]) / min(fa[key], sl[key]) - 1
                    prs.append((fa[key] < sl[key], opp and gap >= TH["T6_gap"]))
    t5 = sum(r for r, _ in prs) / len(prs)
    p6 = [r for r, t in prs if t]
    out = dict(T1=statistics.median(e), T2_max=max(per), T3=sum(x <= TH["T3_within"] for x in e) / len(e),
               T4_min=min(rhos) if rhos else None, T5=t5, T6=(sum(p6) / len(p6)) if p6 else None, T6_pairs=len(p6))
    out["pass"] = dict(T1=out["T1"] <= TH["T1"], T2=out["T2_max"] <= TH["T2"], T3=out["T3"] >= TH["T3_share"],
                       T4=out["T4_min"] is not None and out["T4_min"] >= TH["T4_rho"], T5=t5 >= TH["T5_share"],
                       T6=None if len(p6) < TH["T6_min_pairs"] else out["T6"] >= TH["T6_share"])
    out["failed"] = sorted(k for k, v in out["pass"].items() if v is False)
    k_cells = [abs(c[key] / c["T"] - 1) for c in cells if c["arm"] in TWO_PATH_ARMS]
    out["K_cells"] = len(k_cells)
    out["K_share"] = sum(x <= TH["K_within"] for x in k_cells) / len(k_cells) if k_cells else None
    return out


# --- scoring -----------------------------------------------------------------------------------------------------------
def score(results, job):
    prof = PROFILES[job]
    frozen_tab = globals()[prof["frozen"]]
    dirs = sorted(x for x in os.listdir(results) if re.match(prof["dir_re"], x) and os.path.isdir(os.path.join(results, x)))
    hosts_info, cells, seen_gpu = [], [], {}
    for name in dirs:
        d = os.path.join(results, name)
        h = read_host(d)
        h["valid"], h["why_not"] = False, None
        runs = host_runs(d, h) if h["started"] else []
        valid_runs = [r for r in runs if not r["reasons"]]
        if not h["started"]:
            h["why_not"] = "not started"
        elif h["smoke"]:
            h["why_not"] = "smoke run"
        elif h["gate_failures"]:
            h["why_not"] = "; ".join(h["gate_failures"])
        elif not valid_runs:
            h["why_not"] = "no counted run"
        elif h["uuid"] and h["uuid"] in seen_gpu:
            h["why_not"] = f"GPU already valid in this job ({seen_gpu[h['uuid']]})"
        else:
            h["valid"] = True
            if h["uuid"]:
                seen_gpu[h["uuid"]] = name
        G = (prof["G"] or {"5090": G_5090_MS, "4090": G_4090_MS}).get(h["card_class"])
        h["G_ms"] = G
        if h["valid"] and G is None:
            h["valid"], h["why_not"] = False, f"no G for the card {h['card']}"
        h["runs"] = len(runs)
        h["counted_runs"] = len(valid_runs)
        if h["valid"]:
            for r in sorted(valid_runs, key=lambda r: (r["C"], r["arm"])):
                b = bandwidths(h["probe"], r["helpers"])
                if not b or not r["layers"]:
                    continue
                r.update(B_c=b[0], B_p=b[1], B_both=b[2])
                tp = t_pred(r, G)
                cells.append(dict(host=name, machine=h["uuid"], cpu=short_cpu(h["cpu_model"]), vendor=h["vendor"],
                                  card=h["card_class"], G_ms=G, ratio=b[1] / b[0], helpers=r["helpers"], arm=r["arm"],
                                  C=r["C"], T=r["time_ms"], T_pred=tp, err=tp / r["time_ms"] - 1, T_token=t_token(r, G),
                                  T_law=t_token(r, G, background=False), hit=r["hit_rate"], reads=r["host_reads"],
                                  M=r["M"], F=r["F"], P=r["P"], A=r["A"], X_c=r["X_c"], X_in=r["X_in"],
                                  rounds=r["n_rounds"], config_key=config_key(r["config_string"]), B_c=b[0], B_p=b[1],
                                  B_both=b[2], probe_dir=name, run_id=r["run_id"], fetch_table=r["fetch_table"]))
            hr = [c["ratio"] for c in cells if c["host"] == name]
            h["ratio_probe"] = statistics.median(hr) if hr else None
            if job == "124":   # the population as the scorer reads it (reported; the gates on the machine decide)
                h["population_check"] = dict(
                    ratio_below=h["ratio_probe"] is not None and h["ratio_probe"] < TH["VR_max"],
                    card=h["card_class"] == "4090" or (h["card_class"] == "5090" and h["vendor"] == "Intel"))
        del h["probe"]
        hosts_info.append(h)
    for c in cells:
        c["err_token"] = c["T_token"] / c["T"] - 1
        c["err_law"] = c["T_law"] / c["T"] - 1
    hosts = sorted({c["host"] for c in cells})
    res = {}

    def cell(h, arm, C):
        return next((c for c in cells if c["host"] == h and c["arm"] == arm and c["C"] == C), None)

    # C0: counters are not timing (job 124: the paced arm apart, C0p)
    c0_rows, frozen_used = [], {}
    split_paced = job == "124"
    ok_core, ok_paced = True, True
    for c in cells:
        if c["arm"] not in FROZEN_ARMS:
            continue
        key = f"{c['arm']}|C{c['C']}|{c['config_key']}"
        fv = frozen_tab.get(key)
        if fv is None:
            c0_rows.append(dict(host=c["host"], arm=c["arm"], C=c["C"], reads=c["reads"], frozen=None, note="no frozen counters"))
            continue
        tol = TH["C0_tol_paced"] if c["arm"] in PACED_ARMS else TH["C0_tol"]
        dev = c["reads"] / fv - 1
        ok = abs(dev) <= tol
        if split_paced and c["arm"] in PACED_ARMS:
            ok_paced &= ok
        else:
            ok_core &= ok
        frozen_used[key] = fv
        c0_rows.append(dict(host=c["host"], arm=c["arm"], C=c["C"], reads=c["reads"], frozen=fv, deviation=dev, tol=tol, ok=ok))
    missing, dropped = [], {x["host"] for x in hosts_info if x["plan_dropped"]}
    for arm in FROZEN_ARMS:
        if arm == "baselaw":
            continue
        for h in hosts:
            if arm == "fetchplan" and h in dropped:
                continue     # the header's rule: no plan, fetchplan dropped on that host (recorded on the machine)
            for C in sorted({c["C"] for c in cells if c["host"] == h}):
                if not cell(h, arm, C):
                    missing.append(f"{h} {arm} C{C}")
    agree = {}
    for arm in AGREE_ARMS:
        for C in sorted({c["C"] for c in cells}):
            v = [c["reads"] for c in cells if c["arm"] == arm and c["C"] == C]
            if len(v) >= 2:
                sp_ = max(v) / min(v) - 1
                agree[f"{arm} C{C}"] = dict(spread=sp_, machines=len(v), ok=sp_ <= TH["C0_spread"])
                ok_core &= sp_ <= TH["C0_spread"]
    ok_core &= not missing
    if not split_paced:
        ok_core &= ok_paced
    devs = [abs(x["deviation"]) for x in c0_rows if x.get("frozen") and (not split_paced or x["arm"] not in PACED_ARMS)]
    res["C0"] = dict(verdict=verdict(ok_core, bool(cells)), frozen_checks=c0_rows, frozen_values=frozen_used,
                     missing_arms=missing, agreement=agree, max_frozen_deviation=max(devs) if devs else None)
    if split_paced:
        pd_ = [x for x in c0_rows if x.get("frozen") and x["arm"] in PACED_ARMS]
        res["C0p"] = dict(verdict=verdict(ok_paced, bool(pd_)), checks=pd_,
                          max_deviation=max((abs(x["deviation"]) for x in pd_), default=None))
    if not cells:
        return dict(job=job, hosts=hosts_info, valid_hosts=[], cells=[], clauses=res, decision="untested",
                    note="no valid host")
    # T1-T3
    errs = [abs(c["err"]) for c in cells]
    t1 = statistics.median(errs)
    res["T1"] = dict(verdict=verdict(t1 <= TH["T1"]), median_abs_err=t1, cells=len(cells))
    per = {h: statistics.median(abs(c["err"]) for c in cells if c["host"] == h) for h in hosts}
    res["T2"] = dict(verdict=verdict(all(v <= TH["T2"] for v in per.values())), per_machine=per)
    w = sum(e <= TH["T3_within"] for e in errs) / len(errs)
    res["T3"] = dict(verdict=verdict(w >= TH["T3_share"]), share_within=w,
                     outside=[dict(host=c["host"], arm=c["arm"], C=c["C"], err=c["err"]) for c in cells
                              if abs(c["err"]) > TH["T3_within"]])
    # T4
    t4, t4_ok = {}, True
    for h in hosts:
        for C in sorted({c["C"] for c in cells if c["host"] == h}):
            cc = [c for c in cells if c["host"] == h and c["C"] == C]
            if len(cc) >= TH["T4_min_arms"]:
                rho = spearman([c["T"] for c in cc], [c["T_pred"] for c in cc])
                t4[f"{h} C{C}"] = dict(arms=len(cc), rho=rho, ok=rho is not None and rho >= TH["T4_rho"])
                t4_ok &= t4[f"{h} C{C}"]["ok"]
    res["T4"] = dict(verdict=verdict(t4_ok, bool(t4)), cells=t4, min_rho=min(v["rho"] for v in t4.values()) if t4 else None)
    # T5, T6 and the orderings
    pairs = []
    for h in hosts:
        for C in sorted({c["C"] for c in cells if c["host"] == h}):
            cc = [c for c in cells if c["host"] == h and c["C"] == C]
            for i in range(len(cc)):
                for k in range(i + 1, len(cc)):
                    fa, sl = (cc[i], cc[k]) if cc[i]["T"] <= cc[k]["T"] else (cc[k], cc[i])
                    if sl["T"] / fa["T"] - 1 <= TH["T5_gap"]:
                        continue
                    hit_opp = (abs(fa["hit"] - sl["hit"]) > TH["T6_hit"] and
                               ((fa["hit"] > sl["hit"]) != (fa["T_pred"] < sl["T_pred"])))
                    pgap = max(fa["T_pred"], sl["T_pred"]) / min(fa["T_pred"], sl["T_pred"]) - 1
                    pairs.append(dict(host=h, C=C, fast=fa["arm"], slow=sl["arm"], gap=sl["T"] / fa["T"] - 1,
                                      model_right=fa["T_pred"] < sl["T_pred"], hit_right=fa["hit"] > sl["hit"],
                                      reads_right=fa["reads"] < sl["reads"], token_right=fa["T_token"] < sl["T_token"],
                                      law_right=fa["T_law"] < sl["T_law"], t6=hit_opp and pgap >= TH["T6_gap"],
                                      pred_gap=pgap))
    share = lambda k, ps: sum(p[k] for p in ps) / len(ps) if ps else None  # noqa: E731
    t5 = share("model_right", pairs)
    res["T5"] = dict(verdict=verdict(t5 is not None and t5 >= TH["T5_share"], bool(pairs)), share=t5, pairs=len(pairs))
    p6 = [p for p in pairs if p["t6"]]
    t6s = share("model_right", p6)
    res["T6"] = dict(verdict=verdict(t6s is not None and t6s >= TH["T6_share"], len(p6) >= TH["T6_min_pairs"]),
                     share=t6s, pairs=len(p6), right=sum(p["model_right"] for p in p6),
                     machines=len({p["host"] for p in p6}),
                     contrasts=sorted({f"{p['fast']} vs {p['slow']} C{p['C']}" for p in p6}),
                     contrast_arm_pairs=len({(min(p["fast"], p["slow"]), max(p["fast"], p["slow"])) for p in p6}),
                     contrast_arm_pairs_by_budget=len({(min(p["fast"], p["slow"]), max(p["fast"], p["slow"]), p["C"])
                                                       for p in p6}))
    # P1-P4 (job 123's; in job 124 secondary)
    p1, n_p1 = [], 0
    for h in hosts:
        if all(cell(h, a_, 14) for a_ in ("hitopt", "both3p", "fetch")):
            n_p1 += 1
        for other, C in (("both3p", 14), ("both3p", 32), ("fetch", 14)):
            x, y = cell(h, "hitopt", C), cell(h, other, C)
            if x and y:
                p1.append(dict(host=h, C=C, vs=other, hitopt_T=x["T"], other_T=y["T"], ok=x["T"] > y["T"]))
    res["P1"] = dict(verdict=verdict(all(p["ok"] for p in p1), n_p1 >= TH["P_min_machines"]), comparisons=p1, machines=n_p1)
    p2 = []
    for h in hosts:
        x, y = cell(h, "aa", 14), cell(h, "base", 14)
        if x and y and x["T_pred"] / y["T_pred"] >= TH["P2_ratio"]:
            p2.append(dict(host=h, pred_ratio=x["T_pred"] / y["T_pred"], ratio=x["T"] / y["T"], ok=x["T"] > y["T"]))
    res["P2"] = dict(verdict=verdict(all(p["ok"] for p in p2), bool(p2)), machines=p2,
                     pred_ratios={h: cell(h, "aa", 14)["T_pred"] / cell(h, "base", 14)["T_pred"] for h in hosts
                                  if cell(h, "aa", 14) and cell(h, "base", 14)})
    p3, n_p3 = [], 0
    for h in hosts:
        got = False
        for C in (14, 32):
            x, y = cell(h, "lru", C), cell(h, "base", C)
            if x and y:
                got = True
                p3.append(dict(host=h, C=C, ratio=x["T"] / y["T"], ok=x["T"] > y["T"]))
        n_p3 += got
    res["P3"] = dict(verdict=verdict(all(p["ok"] for p in p3), n_p3 >= TH["P_min_machines"]), comparisons=p3)
    p4 = []
    for h in hosts:
        for C in (14, 32):
            x, y = cell(h, "hitopt", C), cell(h, "base", C)
            if x and y:
                p4.append(dict(host=h, C=C, measured=x["T"] / y["T"], predicted=x["T_pred"] / y["T_pred"],
                               ok=(x["T"] > y["T"]) == (x["T_pred"] > y["T_pred"])))
    p4s = share("ok", p4)
    res["P4"] = dict(verdict=verdict(p4s is not None and p4s >= TH["P4_share"], bool(p4)), share=p4s, cells=p4)
    if job == "124":
        res.update(clauses_124(cells, hosts, pairs, cell))
    # decision
    if job == "123":
        core = ("C0", "T1", "T2", "T3", "T4", "T5", "T6", "P1")
        if len(hosts) < TH["min_valid"]:
            decision = "untested"
        elif any(res[k]["verdict"] == "fails" for k in core + ("P2",)):
            decision = "refuted"
        elif any(res[k]["verdict"] == "untested" for k in core):
            decision = "inconclusive"
        else:
            decision = "confirmed"
    else:
        core = ("C0", "T1", "T2", "T3", "T4", "T5", "T6", "K1")
        if len(hosts) < TH["min_valid"]:
            decision = "untested"
        elif any(res[k]["verdict"] == "fails" for k in core):
            decision = "refuted"
        elif any(res[k]["verdict"] == "untested" for k in core):
            decision = "inconclusive"
        else:
            decision = "confirmed"
        b = res["F3"]["verdict"]
        res["boundary"] = dict(
            verdict="untested" if b == "untested" or (res["F1"]["verdict"] == "untested" and res["F2"]["verdict"] == "untested")
            else ("confirmed" if b == "fails" and "holds" in (res["F1"]["verdict"], res["F2"]["verdict"]) else "not confirmed"),
            rule="the model's fetch decision is wrong on slow-link Intel desktops: F1 or F2 holds and F3 fails")
    errs_t = [abs(c["err"]) for c in cells]
    desc = dict(
        p90_abs_err=sorted(errs_t)[max(0, math.ceil(0.9 * len(errs_t)) - 1)],
        p90_abs_err_interpolated=percentile(errs_t, 90),
        pairs_model=t5, pairs_hit=share("hit_right", pairs), pairs_reads=share("reads_right", pairs),
        pairs_token=share("token_right", pairs), pairs_law=share("law_right", pairs),
        token_form=form_clauses(cells, hosts, "T_token"), law_form=form_clauses(cells, hosts, "T_law"),
        frozen_form_check=form_clauses(cells, hosts, "T_pred"),
        outside_10=dict(arms=sorted({c["arm"] for c in cells if abs(c["err"]) > TH["T3_within"]}),
                        min=min((abs(c["err"]) for c in cells if abs(c["err"]) > TH["T3_within"]), default=None),
                        max=max((abs(c["err"]) for c in cells if abs(c["err"]) > TH["T3_within"]), default=None),
                        signs=sorted({"under" if c["err"] < 0 else "over" for c in cells if abs(c["err"]) > TH["T3_within"]})),
        others_range=[min(c["err"] for c in cells if abs(c["err"]) <= TH["T3_within"]),
                      max(c["err"] for c in cells if abs(c["err"]) <= TH["T3_within"])] if any(
            abs(c["err"]) <= TH["T3_within"] for c in cells) else None,
        other_arms_range=[min(c["err"] for c in cells if c["arm"] != "hitopt"),
                          max(c["err"] for c in cells if c["arm"] != "hitopt")],
        hitopt_range=[min(c["err"] for c in cells if c["arm"] == "hitopt"),
                      max(c["err"] for c in cells if c["arm"] == "hitopt")] if any(c["arm"] == "hitopt" for c in cells) else None,
        ratio_range=[min(c["ratio"] for c in cells), max(c["ratio"] for c in cells)])
    return dict(job=job, constants=dict(G_ms=prof["G"] or {"5090": G_5090_MS, "4090": G_4090_MS}, S_bytes=S_BYTES),
                thresholds=TH, valid_hosts=hosts, hosts=hosts_info, cells=cells, clauses=res, decision=decision,
                descriptive=desc, pairs=pairs)


def clauses_124(cells, hosts, pairs, cell):
    out = {}
    intel = [h for h in hosts if next(c["vendor"] for c in cells if c["host"] == h) == "Intel"]

    def sign_rows(arm, hs, law_only=False):
        rows = []
        for h in hs:
            for C in (14, 32):
                x, y = cell(h, arm, C), cell(h, "base0", C)
                if not x or not y:
                    continue
                if law_only and not any(int(v) for v in re.findall(r"\d+", x["fetch_table"] or "0")):
                    continue     # an all-zero law table: the arm is base0 under another name
                rows.append(dict(host=h, C=C, arm=arm, table=x["fetch_table"], measured=x["T"] / y["T"],
                                 predicted=x["T_pred"] / y["T_pred"], fetches_lose=x["T"] > y["T"],
                                 model_right=(x["T"] > y["T"]) == (x["T_pred"] > y["T_pred"])))
        return rows

    f1, f2 = sign_rows("base", intel), sign_rows("baselaw", intel, law_only=True)
    s1 = sum(r["fetches_lose"] for r in f1) / len(f1) if f1 else None
    s2 = sum(r["fetches_lose"] for r in f2) / len(f2) if f2 else None
    out["F1"] = dict(verdict=verdict(s1 is not None and s1 >= TH["F_share"], len(f1) >= TH["F_min_cells"]), share=s1,
                     cells=f1, rule="Intel cells: base0 faster than base (the fixed table's in-step fetches lose)")
    out["F2"] = dict(verdict=verdict(s2 is not None and s2 >= TH["F_share"], len(f2) >= TH["F_min_cells"]), share=s2,
                     cells=f2, rule="Intel cells, law table not all zero: base0 faster than baselaw")
    f3 = f1 + f2
    s3 = sum(r["model_right"] for r in f3) / len(f3) if f3 else None
    out["F3"] = dict(verdict=verdict(s3 is not None and s3 >= TH["F3_share"], len(f3) >= TH["F_min_cells"]), share=s3,
                     cells=len(f3), right=sum(r["model_right"] for r in f3),
                     fixed_right=sum(r["model_right"] for r in f1), law_right=sum(r["model_right"] for r in f2),
                     rule="Eq. (1)'s sign of the fetches right on the F1 and F2 cells")
    amd = [h for h in hosts if h not in intel]
    f4 = sign_rows("base", amd) + sign_rows("baselaw", amd, law_only=True)
    out["F4"] = dict(verdict=verdict(bool(f4) and sum(r["model_right"] for r in f4) / len(f4) >= TH["F3_share"], bool(f4)),
                     share=sum(r["model_right"] for r in f4) / len(f4) if f4 else None, cells=f4,
                     rule="(exploratory) AMD cells: Eq. (1)'s sign of the fetches right")
    on = [p for p in pairs if p["fast"] in ONLINE_ARMS and p["slow"] in ONLINE_ARMS]
    hs = sum(p["hit_right"] for p in on) / len(on) if on else None
    ms = sum(p["model_right"] for p in on) / len(on) if on else None
    out["H1"] = dict(verdict=verdict(on and hs <= ms, len(on) >= TH["H_min_pairs"]), pairs=len(on), hit_share=hs,
                     model_share=ms, token_share=sum(p["token_right"] for p in on) / len(on) if on else None,
                     reads_share=sum(p["reads_right"] for p in on) / len(on) if on else None,
                     rule="online pairs: the hit rate orders no more of them than T_pred")
    kc = [c for c in cells if c["arm"] in TWO_PATH_ARMS]
    k1 = sum(abs(c["err"]) <= TH["K_within"] for c in kc) / len(kc) if kc else None
    k2 = sum(abs(c["err_token"]) <= TH["K_within"] for c in kc) / len(kc) if kc else None
    out["K1"] = dict(verdict=verdict(k1 is not None and k1 >= TH["K_share"], len(kc) >= TH["K_min_cells"]), share=k1,
                     cells=len(kc), rule="fetch and fetchplan cells: at least 70% within 10% under Eq. (1)")
    out["K2"] = dict(verdict=verdict(k2 is not None and k2 < TH["K_share"], len(kc) >= TH["K_min_cells"]), share=k2,
                     cells=len(kc), rule="the per-token two-path form: fewer than 70% of the same cells within 10%")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--job", default="124", choices=sorted(PROFILES))
    ap.add_argument("--out")
    a = ap.parse_args()
    o = score(a.results, a.job)
    body = json.dumps(o, indent=1, sort_keys=True) + "\n"
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(body)
    print(f"job {a.job}: {len(o['valid_hosts'])} valid hosts, {len(o['cells'])} cells; decision {o['decision']}")
    for k in sorted(o["clauses"]):
        print(f"  {k}: {o['clauses'][k]['verdict']}")


if __name__ == "__main__":
    main()

"""Apply the independent source re-check (data/audit/verify_batch{1,2,3}.jsonl) to the normalized audit rows.

Writes data/audit/normalized_v2.jsonl (the input of scripts/audit.py) and data/audit/verification.md. Every change
is listed with its reason; the v1 file is kept for provenance.

    python scripts/apply_verification.py
"""
import copy
import json

ETA_C = (0.5, 0.7)   # achieved/peak host read bandwidth range used to turn a measured bandwidth into a peak band


def main():
    rows = [json.loads(l) for l in open("data/audit/normalized.jsonl")]
    ver = {}
    for b in (1, 2, 3):
        for l in open(f"data/audit/verify_batch{b}.jsonl"):
            r = json.loads(l)
            ver[r["id"]] = r
    out, log, excluded = [], [], []
    for r in rows:
        r = copy.deepcopy(r)
        v = ver.get(r["id"])
        ch = []
        if r["id"].startswith("promoe-"):          # Fig. 11(b) re-read (bar outline was counted): about -0.3 tok/s
            c = v["corrections"]
            r["system_tok_s"] = c["system_tok_s"]["source"]
            for k, cc in c.items():
                if k.startswith("baselines["):
                    i = int(k.split("[")[1].split("]")[0])
                    r["baselines"][i]["tok_s"] = cc["source"]
            r["bw_pcie_band"] = [31.5, 31.5]
            ch.append("Fig. 11(b) values re-read (-0.3 tok/s); PCIe 4.0 x16")
        if r["id"].startswith("cpugpucollab-"):
            r["metric"] = v["corrections"]["metric"]["source"]
            r["end_to_end_unknown"] = True
            ch.append("metric: single-request throughput, decode-only vs end-to-end unknown")
        if r["id"] == "freetoken-qwen3.6-35b-a3b-nvfp4-rtx4060laptop-w2":
            r["b_dense"] = 7.8
            ch.append("dense weights FP8/NVFP4 in nvidia/Qwen3.6-35B-A3B-NVFP4: b_dense 16 -> 7.8")
        if r["id"].startswith("pipeshard-qwen3-235b"):
            r["b_exp"] = r["b_dense"] = 2.92
            ch.append("unsloth Q2_K files (artifact README): 2.92 bits/weight, not 2.625")
        if r["id"] == "pipeshard-qwen3-30b-a3b-q4_0-cli3-rtx5090-8G":
            for b in r["baselines"]:
                if "cmoe" in b["name"]:
                    b["tok_s"] = 26.1
            ch.append("-cmoe baseline read from Fig. 3 bar (1.23x): 26.1 tok/s, not 26.75")
        if r["id"] == "lcpp-d24528-xashr-laguna-s-2.1-q6k-rtx5090":
            for b in r["baselines"]:
                if b["name"].startswith("llama.cpp"):
                    b["class"] = "llama.cpp --n-cpu-moe / -ot partial expert offload"
                    b["note"] = "auto-fit placement: '-fit on --no-mmap'"
            r["ctx"] = 4096
            ch.append("upstream baseline used auto-fit (-fit on): partial expert offload; ctx 4096 assumed (long generation)")
        if r["id"] == "lcpp-pr27861-sdroege-qwen3.8-flash-next-q4kxl-r9700":
            for b in r["baselines"]:
                b["equal_vram"] = "approximately"
            ch.append("baseline --fit on fills VRAM like the cache run: equal VRAM, approximately")
        if r["id"] == "lcpp-pr27861-sissyhistorian-qwen3-30b-a3b-q4km-rx7600":
            r["model"] = "Qwen3-30B-A3B Q4_K_M derivative (architecture assumed identical)"
            ch.append("model is an unnamed Q4_K_M derivative")
        if r["id"] == "spmoe-deepseekv2lite-a100-gpumem39g-b1":
            excluded.append(dict(id=r["id"], reason="effectively all experts in VRAM (31.4 GB model at ~38 GB budget); host traffic ~0"))
            log.append((r["id"], ["excluded: effectively all-in-VRAM"]))
            continue
        # FreeToken rented servers: 6-8 CPU threads per run; the measured host bandwidth, not the datasheet peak, applies
        m = r.get("host_bw_measured")
        if r["system"] == "FreeToken" and m and m < 0.5 * r["bw_cpu"][0]:
            r["bw_cpu"] = [round(m / ETA_C[1], 1), round(m / ETA_C[0], 1)]
            ch.append(f"host band from the measured {m} GB/s (runs capped at 6-8 threads): {r['bw_cpu']} GB/s peak-equivalent")
        if r["system"] == "FreeToken" and (r.get("ctx") or 0) < 1024:
            r["ctx"] = 4096
            ch.append("ctx 4096 assumed (long chain-of-thought / agentic workloads)")
        if v:
            r["verification"] = dict(verdict=v["verdict"], unverifiable=v.get("unverifiable", []), notes=v.get("notes", ""))
        out.append(r)
        if ch:
            log.append((r["id"], ch))
    with open("data/audit/normalized_v2.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    with open("data/audit/excluded.jsonl", "a") as f:
        for e in excluded:
            f.write(json.dumps(dict(e, category="verification")) + "\n")
    L = ["# Source re-check of the adjudicable audit rows\n",
         f"Three independent checkers re-read every source ({len(ver)} rows): "
         f"{sum(v['verdict'] == 'ok' for v in ver.values())} ok, "
         f"{sum(v['verdict'] == 'corrections' for v in ver.values())} with corrections, "
         f"{sum(v['verdict'] == 'exclude' for v in ver.values())} excluded by the checkers. Changes applied:\n"]
    for i, ch in log:
        L.append(f"- `{i}`: " + "; ".join(ch))
    open("data/audit/verification.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

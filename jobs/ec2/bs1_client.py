"""Single-stream (bs=1) decode through a server's OpenAI chat API, measured the way FreeToken's own benchmark
(FlashML-org/FreeToken 0d652e7, benchmarks/bench_decode_moe.py) measures it, so that llama.cpp, the expert cache and
FreeToken are timed by one client on one prompt set with one sampling setup.

Reused from their script, unchanged: the AIME-25 prompt (load_problem), the sampling resolution (resolve_sampling:
the checkpoint's generation_config, else temperature 1.0 / top_p 0.95 / top_k 64), their `ft serve` command line
(serve_cmd), readiness wait and shutdown, and the per-problem protocol (one warm-up request on the problem, then the
measured request, ignore_eos, max_tokens = D). The metric is theirs: decode_tok_s = (completion_tokens - 1) /
(t_last_event - t_first_event). The request body is theirs plus `extra` fields (for llama-server: min_p 0, so its
default min_p 0.05 does not change the sampler, and a fixed seed). Additions: llama-server's own `timings` are kept,
and GPU memory in use is read from nvidia-smi before the server stops.

    # a FreeToken server, spawned with their serve_cmd:
    python bs1_client.py --ft offload --cache-rate 0.25 --model /work/models/gpt-oss-120b --label ft_offload_r0.25 ...
    # any other server: a shell command with {port}; readiness is GET /health == {"status": "ok"}
    python bs1_client.py --cmd "llama-server ... --port {port}" --model /work/models/gpt-oss-120b --label ...
"""
import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

FT = os.environ.get("FT_DIR", "/work/ft")
sys.path.insert(0, f"{FT}/benchmarks")
import bench_decode_moe as B  # noqa: E402  (stdlib-only at import time)


def stream(origin, model_id, problem, sampling, decode, extra):
    """B.stream_generate, plus `extra` body fields and the final chunk's `timings` (llama-server)."""
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": problem}],
        "max_tokens": decode,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": True},
        **sampling,
        **extra,
    }
    req = urllib.request.Request(f"{origin}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    stamps, pieces, usage, timings = [], [], None, None
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.strip()
            if not line or not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:"):].strip()
            if payload == b"[DONE]":
                break
            now = time.perf_counter()
            chunk = json.loads(payload)
            if chunk.get("error"):
                raise RuntimeError(f"server error mid-stream: {chunk['error']}")
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("timings"):
                timings = chunk["timings"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                text = delta.get("reasoning_content") or delta.get("content")
                if text:
                    stamps.append(now)
                    pieces.append(text)
    if usage is None and timings:
        usage = {"completion_tokens": timings.get("predicted_n"), "prompt_tokens": timings.get("prompt_n")}
    if usage is None:
        raise RuntimeError("stream ended without usage")
    return {"t0": t0, "stamps": stamps, "text": "".join(pieces), "usage": usage, "timings": timings}


def vram_used_gib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        return float(out.split()[0]) / 1024
    except Exception:
        return None


def wait_health(origin, proc, log_path, timeout, mode="health"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode} during startup; log {log_path}")
        try:
            if mode == "models":
                with urllib.request.urlopen(f"{origin}/v1/models", timeout=5) as r:
                    if json.load(r).get("data"):
                        return
                time.sleep(1.0)
                continue
            with urllib.request.urlopen(f"{origin}/health", timeout=5) as r:
                if json.load(r).get("status") == "ok":
                    return
        except (OSError, ValueError, urllib.error.HTTPError):
            pass
        time.sleep(1.0)
    raise RuntimeError(f"server not ready after {timeout:.0f}s; log {log_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF checkpoint dir (sampling resolution; FreeToken's model)")
    ap.add_argument("--ft", help="FreeToken backend: offload|hybrid|cpu (spawns `ft serve` via their serve_cmd)")
    ap.add_argument("--cache-rate", type=float)
    ap.add_argument("--mem-ratio", type=float, default=0.9, help="FreeToken --memory-ratio (their bench's default 0.9)")
    ap.add_argument("--cmd", help="shell command for any other server, with {port}")
    ap.add_argument("--extra", default="{}", help="JSON fields added to every request body")
    ap.add_argument("--problems", default="0")
    ap.add_argument("--decode", type=int, default=256)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--aime", default=os.environ.get("FREETOKEN_AIME25_JSONL"))
    ap.add_argument("--label", required=True)
    ap.add_argument("--meta", default="{}", help="JSON fields copied into every row (system, budget, ...)")
    ap.add_argument("--json", required=True)
    ap.add_argument("--log", required=True, help="server log file")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--warmup", default="same", help="same: FreeToken's protocol (a warm-up request on each problem "
                    "before measuring it); once: one warm-up request on --warmup-prompt at server start, then every "
                    "problem measured once, in order, with the cache carried across problems (a session)")
    ap.add_argument("--warmup-prompt", help="file with the held-out warm-up prompt text (--warmup once)")
    ap.add_argument("--hybrid-fetch", type=int, default=-1, help="FreeToken --moe-hybrid-max-fetch (-1: auto)")
    ap.add_argument("--launch", type=int, default=0, help="launch index, copied into every row")
    ap.add_argument("--ready", default="health", help="health: GET /health == {status: ok} (llama-server); models: GET "
                    "/v1/models answers (SGLang, whose /health runs a generation)")
    ap.add_argument("--wrap", help="command prefix for the server (e.g. `nsys launch --session-new=S ...`)")
    ap.add_argument("--before-cmd", help="shell command run after the warm-up, before the first measured problem "
                    "(e.g. `nsys start --session=S ...`)")
    ap.add_argument("--stop-wait", type=float, default=120, help="seconds to wait for the server to exit after SIGINT")
    ap.add_argument("--after-cmd", help="shell command run right after the first measured problem (e.g. `nsys stop`)")
    a = ap.parse_args()
    sampling, src = B.resolve_sampling(a.model, a.greedy)
    extra, meta = json.loads(a.extra), json.loads(a.meta)
    port = B.free_port()
    origin = f"http://127.0.0.1:{port}"
    if a.ft:
        ns = argparse.Namespace(model=a.model, decode=a.decode, mem_ratio=a.mem_ratio, no_graph=False,
                                hybrid_fetch=a.hybrid_fetch, gpu=None, cache=0, cache_rate=a.cache_rate)
        cmd, shell = B.serve_cmd(ns, a.ft, port), False
    else:
        cmd, shell = a.cmd.format(port=port), True
    if a.wrap:
        cmd = (a.wrap + " " + cmd) if shell else shlex.split(a.wrap) + cmd
    print(f"[bs1] {a.label}: sampling {sampling} <- {src}; cmd {cmd}", flush=True)
    log_f = open(a.log, "wb")
    t_start = time.perf_counter()
    proc = subprocess.Popen(cmd, shell=shell, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True)
    rows = []
    try:
        if a.ft:
            B.wait_ready(origin, proc, a.log, a.timeout)
        else:
            wait_health(origin, proc, a.log, a.timeout, a.ready)
        load_s = time.perf_counter() - t_start
        with urllib.request.urlopen(f"{origin}/v1/models", timeout=10) as r:
            model_id = json.load(r)["data"][0]["id"]
        if a.warmup == "once":
            wp = open(a.warmup_prompt).read().strip()
            stream(origin, model_id, wp, sampling, a.decode, extra)                 # one held-out warm-up
        for i, p in enumerate(int(x) for x in a.problems.split(",")):
            problem, answer = B.load_problem(a.aime, p)
            try:
                if a.warmup == "same":
                    stream(origin, model_id, problem, sampling, a.decode, extra)   # warm-up, as theirs
                if i == 0 and a.before_cmd:
                    rc = subprocess.run(a.before_cmd, shell=True).returncode
                    print(f"[bs1] {a.label} before-cmd rc={rc}", flush=True)
                    time.sleep(3)
                r = stream(origin, model_id, problem, sampling, a.decode, extra)
                if i == 0 and a.after_cmd:
                    t_a = time.perf_counter()
                    rc = subprocess.run(a.after_cmd, shell=True).returncode
                    print(f"[bs1] {a.label} after-cmd rc={rc} in {time.perf_counter() - t_a:.0f} s", flush=True)
            except Exception as e:  # keep the other problems
                print(f"[bs1] {a.label} problem {p} failed: {e!r}", flush=True)
                continue
            stamps, usage = r["stamps"], r["usage"]
            completion = usage["completion_tokens"]
            steps = completion - 1
            dt = stamps[-1] - stamps[0] if len(stamps) >= 2 else 0.0
            gaps = sorted((y - x) * 1e3 for x, y in zip(stamps, stamps[1:])) or [0.0]
            row = {"label": a.label, **meta, "launch": a.launch, "warmup": a.warmup, "problem": p,
                   "profiled": bool(i == 0 and a.before_cmd),
                   "prompt_tokens": usage.get("prompt_tokens"), "output_sha1_full": hashlib.sha1(r["text"].encode()).hexdigest(),
                   "decode_steps": steps, "decode_tok_s": steps / dt if dt > 0 else 0.0,
                   "ms_per_token": dt / steps * 1e3 if steps > 0 else 0.0,
                   "event_ms_p50": gaps[len(gaps) // 2], "event_ms_p99": gaps[min(len(gaps) - 1, int(len(gaps) * 0.99))],
                   "ttft_ms": (stamps[0] - r["t0"]) * 1e3 if stamps else None, "events": len(stamps),
                   "completion_tokens": completion, "sampling": sampling, "extra": extra,
                   "output_sha1": hashlib.sha1(r["text"].encode()).hexdigest()[:12], "load_s": load_s,
                   "server_timings": r["timings"], "text_head": r["text"][:160]}
            rows.append(row)
            print(f"[bs1] {a.label} problem {p}: {row['decode_tok_s']:.2f} tok/s ({row['ms_per_token']:.2f} ms/token, "
                  f"p50 {row['event_ms_p50']:.2f} p99 {row['event_ms_p99']:.2f}, {completion} tokens)", flush=True)
        v = vram_used_gib()
        for row in rows:
            row["vram_used_gib"] = v
    finally:
        if a.ft:
            B.stop_server(proc)
        else:
            try:
                os.killpg(proc.pid, signal.SIGINT)   # llama-server shuts down cleanly (writes LLAMA_EC_STATS)
                proc.wait(timeout=a.stop_wait)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            time.sleep(3)
        log_f.close()
    with open(a.json, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())

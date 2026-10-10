#!/usr/bin/env python3
"""OpenAI-compatible chat-completions load benchmark, mirroring vllm bench serve.

The reference `1Cat-vLLM/vllm/benchmarks/serve.py` fires N chat-completions requests at an
OpenAI-compatible endpoint. It measures --request-rate (Poisson arrivals) or
--max-concurrency (at most C requests in flight, the rest queued), then reports request /
output / total token throughput and TTFT / TPOT / ITL / E2E latents (mean, p50, p90). This
reproduces the same measurement surface against the NInfer serve without the vllm dependency
(the serve is OpenAI-compatible at /v1/chat/completions and reports usage.completion_tokens).

Prompts are a fixed-length passage repeated to reach the requested token count; the exact
token length the serve actually processes comes from the response's usage.prompt_tokens.
"""

from __future__ import annotations

import argparse
import http.client
import json
import statistics
import threading
import time

DEFAULT_MODEL = "qwen3.8-27b"

PROMPT_FILLER = (
    "The compiler lowers the fused attention pass into a single flat dispatch that "
    "schedules the prefill slab and the decode slab side by side. Each lane owns its "
    "paged KV blocks and the collective allreduce carries the row-parallel joins. "
)


def make_prompt(input_len: int) -> str:
    """A deterministic ~input_len-token prompt (real prose repeated to grow the length)."""
    if input_len <= 0:
        raise ValueError("input_len must be positive")
    words = PROMPT_FILLER.split()
    while len(" ".join(words)) < input_len * 3:
        words = words + WORDS_BACKFILL.split()
    return " ".join(words)


WORDS_BACKFILL = (
    "attention prefill decode mixed tensor-parallel width tensor row column shard slab page "
    "block KV cache ring buffer allreduce collective kernel launch capture replay graph "
)


def stream_chat_completions(url: str, model: str, prompt: str, max_tokens: int,
                            timeout: float) -> dict:
    """POST one streaming /v1/chat/completions request; return latents + token counts."""
    parsed = __import__("urllib.parse", fromlist=["urlsplit"]).urlsplit(url)
    host, port = parsed.hostname, (parsed.port or 80)
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
    }).encode()
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    conn.request("POST", "/v1/chat/completions", body=body,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        detail = resp.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {resp.status}: {detail[:300]}")

    t0 = time.monotonic()
    ttft = 0.0
    chunk_ts: list[float] = []
    completion_tokens = 0
    prompt_tokens = 0
    finished = False
    for raw in resp:
        if not raw:
            continue
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        usage = obj.get("usage")
        if usage:
            completion_tokens = int(usage.get("completion_tokens", completion_tokens))
            prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if content and (ttft == 0.0 and not chunk_ts):
            ttft = time.monotonic() - t0
        if delta.get("finish_reason"):
            finished = True
        if content:
            chunk_ts.append(time.monotonic())
    e2e = time.monotonic() - t0
    itls = [b - a for a, b in zip(chunk_ts, chunk_ts[1:])]
    if not ttft:
        ttft = e2e
    tpot = ((e2e - ttft) / max(0, completion_tokens - 1)) if completion_tokens > 1 else 0.0
    return {"ttft": ttft, "e2e": e2e, "itls": itls, "tpot": tpot,
            "completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens,
            "finished": finished}


def one_request(args, model, prompt, prompt_len_hint, worker) -> dict:
    return stream_chat_completions(args.url, model, prompt, args.output_len, args.timeout)


def bench(args, model, prompt):
    gate = threading.Semaphore(args.max_concurrency if args.max_concurrency else
                               args.num_prompts)
    fire = threading.Event()
    results: list[dict] = []
    lock = threading.Lock()

    def worker(i):
        with gate:
            fire.wait()
        try:
            r = one_request(args, model, prompt, len(prompt) // 4, i)
        except Exception as exc:  # noqa: BLE001
            r = {"error": str(exc), "ttft": 0.0, "e2e": 0.0, "itls": [], "tpot": 0.0,
                 "completion_tokens": 0, "prompt_tokens": 0}
        with lock:
            results.append(r)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.num_prompts)]
    for t in threads:
        t.start()
    t_start = time.monotonic()
    fire.set()
    for t in threads:
        t.join()
    t_end = time.monotonic()
    dur = t_end - t_start
    return results, dur


def latents(vals):
    if not vals:
        return 0.0, 0.0, 0.0, 0.0
    vals_s = sorted(vals)
    p50 = vals_s[len(vals_s) // 2]
    p90 = vals_s[min(len(vals_s) - 1, int(0.9 * len(vals_s)))]
    return (statistics.mean(vals) * 1000, p50 * 1000, p90 * 1000, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-concurrency", type=int, default=4,
                        help="client-side semaphore cap (server's active-request limit)")
    parser.add_argument("--input-len", type=int, default=2048)
    parser.add_argument("--output-len", type=int, default=256, help="--max_tokens cap")
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()

    prompt = make_prompt(args.input_len)
    results, dur = bench(args, args.model, prompt)
    ok = [r for r in results if r.get("success", "error" not in r)]
    prompts = sum(int(r["prompt_tokens"]) for r in ok)
    outputs = sum(int(r["completion_tokens"]) for r in ok)
    req_thr = len(ok) / dur
    output_thr = outputs / dur
    total_thr = (prompts + outputs) / dur
    ttfts = latents([r["ttft"] for r in ok])
    tpots = latents([r["tpot"] for r in ok])
    e2e = latents([r["e2e"] for r in ok])
    itls = [f for r in ok for f in r["itls"]]
    itl_thr = latents(itls)

    print("Benchmark configuration")
    print(f"  URL                  : {args.url}")
    print(f"  model id             : {args.model}")
    print(f"  num prompts          : {len(results)}")
    print(f"  max concurrency      : {args.max_concurrency}")
    print(f"  input (requested)    : {args.input_len} tokens")
    print(f"  output token cap     : {args.output_len}")
    print()
    print("Aggregates")
    print(f"  request thr (req/s)  : {req_thr:9.4f}")
    print(f"  output thr (tok/s)   : {output_thr:9.3f}")
    print(f"  total thr (tok/s)    : {total_thr:9.3f}")
    print("Latency (ms)")
    print(f"  TTFT mean / p50 / p90: {ttfts[0]:9.2f} / {ttfts[1]:9.2f} / {ttfts[2]:9.2f}")
    print(f"  TPOT mean / p50 / p90: {tpots[0]:9.2f} / {tpots[1]:9.2f} / {tpots[2]:9.2f}")
    print(f"  ITL  mean / p50 / p90: {itl_thr[0]:9.2f} / {itl_thr[1]:9.2f} / {itl_thr[2]:9.2f}")
    print(f"  E2E  mean / p50 / p90: {e2e[0]:9.2f} / {e2e[1]:9.2f} / {e2e[2]:9.2f}")
    print()
    print(f"  completed / all      : {len(ok)} / {len(results)}")


if __name__ == "__main__":
    main()
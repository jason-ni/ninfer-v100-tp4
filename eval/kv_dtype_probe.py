#!/usr/bin/env python3
"""KV-cache dtype quality probe: needle retrieval + reasoning at multiple context lengths.

Usage:
    python kv_dtype_probe.py --url http://127.0.0.1:8080/v1 --model qwen3.8-27b \
        --out results_iq4nl.jsonl [--context-lengths 8000,32000,64000,128000]

Generates synthetic needle-in-haystack prompts (no external corpus needed),
measures retrieval accuracy, and records raw outputs for later diffing.
"""

import argparse
import json
import random
import string
import sys
import time
import urllib.request
import urllib.error

FILLER_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the old stone bridge.",
    "A researcher published a paper on quantum entanglement in superconducting circuits.",
    "The annual revenue of the company increased by twelve percent compared to last year.",
    "She walked through the garden and noticed that the roses had finally bloomed.",
    "The algorithm converges in logarithmic time when the input is well-conditioned.",
    "He remembered the summer they spent building the treehouse behind the barn.",
    "The treaty was signed after months of negotiations between the two delegations.",
    "Light bends around massive objects, a phenomenon confirmed by gravitational lensing.",
    "The chef added a pinch of saffron to the risotto just before serving.",
    "Students gathered in the lecture hall to hear the keynote on artificial intelligence.",
    "The river flooded its banks for the third time in as many years this decade.",
    "A new species of deep-sea fish was discovered near the hydrothermal vents.",
    "The software update fixed a critical memory leak in the rendering pipeline.",
    "They hiked the mountain trail and reached the summit before noon on a clear day.",
    "The orchestra performed a symphony that lasted almost three hours in total.",
    "Farmers in the region switched to drip irrigation to conserve groundwater supplies.",
    "The ancient manuscript was carbon-dated to the twelfth century with high confidence.",
    "A startup raised funding to develop autonomous delivery drones for rural areas.",
    "The patient recovered fully after the minimally invasive surgical procedure.",
    "Waves crash against the cliff face sending spray high into the salty air above.",
]


def make_filler(n_tokens_approx: int, rng: random.Random) -> str:
    """Generate ~n_tokens_approx tokens of filler text (~4 chars/token heuristic)."""
    target_chars = n_tokens_approx * 4
    parts = []
    total = 0
    while total < target_chars:
        s = rng.choice(FILLER_SENTENCES)
        parts.append(s)
        total += len(s) + 1
    return " ".join(parts)


def make_needle(depth_pct: int, rng: random.Random) -> tuple[str, str]:
    """Create a unique needle sentence and a question about it. Returns (needle, question)."""
    color = rng.choice(["crimson", "azure", "emerald", "golden", "violet", "scarlet"])
    animal = rng.choice(["otter", "heron", "falcon", "badger", "lynx", "marten"])
    city = rng.choice(["Oslo", "Kyoto", "Reykjavik", "Marrakech", "Wellington", "Helsinki"])
    number = rng.randint(100, 9999)
    month = rng.choice(["March", "July", "November", "February", "September", "January"])

    needle = (
        f"The secret code word is {color} {animal} {number}. "
        f"This was decided in {month} during the meeting in {city}."
    )
    question = "What is the secret code word mentioned in the document?"
    return needle, question


def build_prompt(haystack_tokens: int, depth_pct: int, rng: random.Random) -> tuple[str, str, str]:
    """Build a needle-in-haystack prompt. Returns (full_prompt, needle, expected_answer)."""
    filler_before_ratio = depth_pct / 100.0
    total_filler = haystack_tokens
    before_tokens = int(total_filler * filler_before_ratio)
    after_tokens = total_filler - before_tokens

    before = make_filler(before_tokens, rng) if before_tokens > 0 else ""
    after = make_filler(after_tokens, rng) if after_tokens > 0 else ""
    needle, question = make_needle(depth_pct, rng)

    doc = f"{before} {needle} {after}".strip()
    prompt = (
        f"Read the following document carefully, then answer the question.\n\n"
        f"--- DOCUMENT START ---\n{doc}\n--- DOCUMENT END ---\n\n"
        f"Question: {question}\n"
        f"Answer with the exact code word sentence only."
    )
    return prompt, needle, question


def call_api(base_url: str, model: str, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> dict:
    url = f"{base_url}/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()}"}
    except Exception as e:
        return {"error": str(e)}


def check_retrieval(response: dict, needle: str) -> bool:
    """Check if the needle's key phrase appears in the response."""
    if "error" in response:
        return False
    choices = response.get("choices", [])
    if not choices:
        return False
    content = choices[0].get("message", {}).get("content", "") or ""
    # Extract the code word from the needle: "X Y N"
    import re
    m = re.search(r"is (\w+ \w+ \d+)\.", needle)
    if not m:
        return False
    codeword = m.group(1)
    return codeword.lower() in content.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Base URL, e.g. http://127.0.0.1:8080/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--context-lengths", default="8000,32000,64000,128000",
                    help="Comma-separated haystack token targets")
    ap.add_argument("--depths", default="10,30,50,70,90",
                    help="Comma-separated needle depth percentages")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ctx_lens = [int(x) for x in args.context_lengths.split(",")]
    depths = [int(x) for x in args.depths.split(",")]
    rng = random.Random(args.seed)

    results = []
    total = len(ctx_lens) * len(depths)
    done = 0

    print(f"KV dtype probe: {args.url} model={args.model}")
    print(f"Context lengths: {ctx_lens}, Depths: {depths}, Total tests: {total}")
    print("-" * 60)

    for ctx in ctx_lens:
        for depth in depths:
            prompt, needle, question = build_prompt(ctx, depth, rng)
            t0 = time.time()
            resp = call_api(args.url, args.model, prompt)
            elapsed = time.time() - t0

            retrieved = check_retrieval(resp, needle)
            content = ""
            if "choices" in resp and resp["choices"]:
                content = resp["choices"][0].get("message", {}).get("content", "") or ""

            entry = {
                "context_tokens": ctx,
                "depth_pct": depth,
                "retrieved": retrieved,
                "elapsed_s": round(elapsed, 2),
                "needle": needle,
                "response_content": content[:500],
                "prompt_tokens": resp.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": resp.get("usage", {}).get("completion_tokens", 0),
                "error": resp.get("error"),
            }
            results.append(entry)
            done += 1

            status = "HIT" if retrieved else ("ERR" if entry["error"] else "MISS")
            print(f"[{done}/{total}] ctx={ctx:>6} depth={depth:>2}%  {status}  "
                  f"({entry['prompt_tokens']} tok in, {elapsed:.1f}s)")
            if entry["error"]:
                print(f"         error: {entry['error'][:200]}")

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    hits = sum(1 for r in results if r["retrieved"])
    errors = sum(1 for r in results if r["error"])
    print("-" * 60)
    print(f"Result: {hits}/{len(results)} retrieved, {errors} errors")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
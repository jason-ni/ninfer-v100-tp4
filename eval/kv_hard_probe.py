#!/usr/bin/env python3
"""Harder KV-cache quality probe: multi-hop, distractors, counting, negation.

These tests stress attention precision more than simple needle retrieval.
KV quantization errors that survive a single-needle test will show up here.
"""

import argparse
import json
import random
import re
import sys
import time
import urllib.request
import urllib.error

FILLER = [
    "The quarterly report showed steady growth across all business divisions.",
    "Researchers at the university published findings on protein folding mechanisms.",
    "The city council approved the new zoning regulations for downtown development.",
    "A study found that regular exercise reduces the risk of cardiovascular disease.",
    "The software team migrated the legacy system to the cloud platform last month.",
    "Students participated in the annual science fair presenting innovative projects.",
    "The harbor master reported calm seas and favorable winds for shipping operations.",
    "The museum acquired a rare manuscript dating back to the fourteenth century.",
    "Engineers tested the new alloy under extreme temperature conditions for durability.",
    "The orchestra rehearsed the symphony for three weeks before the premiere night.",
    "Farmers reported a bountiful harvest despite the unusual weather patterns this year.",
    "The legal team reviewed the contract clauses and identified several ambiguities.",
    "The hospital implemented a new triage protocol to reduce patient waiting times.",
    "The archaeological team uncovered pottery shards from an ancient trading post.",
    "The airline announced new routes connecting major cities across three continents.",
]


def filler(n_tokens, rng):
    target = n_tokens * 4
    parts, total = [], 0
    while total < target:
        s = rng.choice(FILLER)
        parts.append(s)
        total += len(s) + 1
    return " ".join(parts)


def build_multi_hop(ctx_tokens, rng):
    """Two facts buried at different depths; question requires both."""
    fact1_city = rng.choice(["Zurich", "Oslo", "Nairobi", "Seoul", "Lima", "Athens"])
    fact1_amount = rng.randint(100, 999) * 1000
    fact2_city = rng.choice(["Toronto", "Bogota", "Hanoi", "Vienna", "Cairo", "Auckland"])
    fact2_amount = rng.randint(100, 999) * 1000
    while fact2_city == fact1_city:
        fact2_city = rng.choice(["Toronto", "Bogota", "Hanoi", "Vienna", "Cairo", "Auckland"])

    fact1 = f"The budget allocation for {fact1_city} was {fact1_amount} dollars."
    fact2 = f"The budget allocation for {fact2_city} was {fact2_amount} dollars."

    d1 = rng.uniform(0.15, 0.35)
    d2 = rng.uniform(0.65, 0.85)
    t1 = int(ctx_tokens * d1)
    t2 = int(ctx_tokens * d2)

    doc = f"{filler(t1, rng)} {fact1} {filler(t2 - t1 - len(fact1), rng)} {fact2} {filler(ctx_tokens - t2 - len(fact2), rng)}"
    q = f"What is the total combined budget allocation for {fact1_city} and {fact2_city}? Give the sum in dollars."
    expected = str(fact1_amount + fact2_amount)
    return doc, q, expected, f"{fact1_city}={fact1_amount}, {fact2_city}={fact2_amount}"


def build_distractor(ctx_tokens, rng):
    """Correct needle plus a near-identical wrong one."""
    real_code = f"ALPHA-{rng.randint(1000,9999)}-XRAY"
    fake_code = f"ALPHA-{rng.randint(1000,9999)}-XRAY"
    while fake_code == real_code:
        fake_code = f"ALPHA-{rng.randint(1000,9999)}-XRAY"

    real = f"The authorization code for Project Meridian is {real_code}."
    fake = f"The authorization code for Project Volcano is {fake_code}."

    d_real = rng.uniform(0.2, 0.4)
    d_fake = rng.uniform(0.6, 0.8)
    t_real = int(ctx_tokens * d_real)
    t_fake = int(ctx_tokens * d_fake)

    doc = f"{filler(t_real, rng)} {real} {filler(t_fake - t_real - len(real), rng)} {fake} {filler(ctx_tokens - t_fake - len(fake), rng)}"
    q = "What is the authorization code for Project Meridian? Answer with just the code."
    return doc, q, real_code, f"distractor={fake_code}"


def build_counting(ctx_tokens, rng):
    """Scatter N occurrences of a keyword; ask how many times it appears."""
    keyword = rng.choice(["luminance", "catalyst", "meridian", "fluxgate", "cryostat"])
    count = rng.randint(3, 7)
    sentences = [filler(1, rng) for _ in range(max(count + 2, 5))]
    for i in range(count):
        idx = rng.randint(0, len(sentences) - 1)
        sentences[idx] = f"The {keyword} measurement was recorded at station {chr(65+i)}."
    doc = f"{filler(int(ctx_tokens * 0.3), rng)} " + " ".join(sentences) + f" {filler(int(ctx_tokens * 0.3), rng)}"
    q = f"How many times does the word '{keyword}' appear in the document? Answer with just the number."
    return doc, q, str(count), f"keyword={keyword}"


def build_negation(ctx_tokens, rng):
    """Ask about something NOT in the document. Correct answer: not mentioned."""
    absent_fruit = rng.choice(["dragonfruit", "lychee", "persimmon", "rambutan", "longan"])
    doc = filler(ctx_tokens, rng)
    q = f"Does the document mention {absent_fruit}? Answer yes or no."
    return doc, q, "no", f"absent={absent_fruit}"


def call_api(base_url, model, prompt, max_tokens=256, temperature=0.0):
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


def check(entry_type, response, expected, meta):
    if "error" in response:
        return False, response["error"][:200]
    choices = response.get("choices", [])
    if not choices:
        return False, "no choices"
    content = (choices[0].get("message", {}).get("content", "") or "").strip()

    if entry_type == "multi_hop":
        return expected in content, content[:300]
    elif entry_type == "distractor":
        return expected.lower() in content.lower(), content[:300]
    elif entry_type == "counting":
        m = re.search(r'\b(\d+)\b', content)
        if m:
            return m.group(1) == expected, content[:300]
        return False, content[:300]
    elif entry_type == "negation":
        return expected in content.lower(), content[:300]
    return False, content[:300]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--context-lengths", default="32000,64000,128000")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ctx_lens = [int(x) for x in args.context_lengths.split(",")]
    rng = random.Random(args.seed)
    builders = [
        ("multi_hop", build_multi_hop),
        ("distractor", build_distractor),
        ("counting", build_counting),
        ("negation", build_negation),
    ]

    results = []
    total = len(ctx_lens) * len(builders)
    done = 0
    print(f"Hard KV probe: {args.url}")
    print(f"Contexts: {ctx_lens}, Test types: {[b[0] for b in builders]}, Total: {total}")
    print("-" * 60)

    for ctx in ctx_lens:
        for name, builder in builders:
            doc, q, expected, meta = builder(ctx, rng)
            prompt = f"Read the document carefully.\n\n--- DOCUMENT ---\n{doc}\n--- END ---\n\n{q}"
            t0 = time.time()
            resp = call_api(args.url, args.model, prompt)
            elapsed = time.time() - t0
            passed, detail = check(name, resp, expected, meta)
            ptok = resp.get("usage", {}).get("prompt_tokens", 0)

            entry = {
                "type": name, "context_tokens": ctx, "passed": passed,
                "elapsed_s": round(elapsed, 2), "prompt_tokens": ptok,
                "expected": expected, "detail": detail, "meta": meta,
                "error": resp.get("error"),
            }
            results.append(entry)
            done += 1
            tag = "PASS" if passed else ("ERR" if entry["error"] else "FAIL")
            print(f"[{done}/{total}] {name:>12} ctx={ctx:>6}  {tag}  ({ptok} tok, {elapsed:.1f}s)")
            if not passed and not entry["error"]:
                print(f"         expected={expected}  got={detail[:150]}")

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    by_type = {}
    for r in results:
        by_type.setdefault(r["type"], [0, 0])
        by_type[r["type"]][1] += 1
        if r["passed"]:
            by_type[r["type"]][0] += 1
    print("-" * 60)
    for t, (p, n) in sorted(by_type.items()):
        print(f"  {t:>12}: {p}/{n}")
    total_p = sum(1 for r in results if r["passed"])
    print(f"  {'TOTAL':>12}: {total_p}/{len(results)}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
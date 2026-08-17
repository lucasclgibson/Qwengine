#!/usr/bin/env python3
"""Measure a running qwengine (or any OpenAI-compatible server) on a fixed
prompt set. Reports end-to-end tok/s as a client sees it, including HTTP."""
import json, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MAXT = int(sys.argv[2]) if len(sys.argv) > 2 else 120
MODEL = sys.argv[3] if len(sys.argv) > 3 else "qwengine"

PROMPTS = [
    ("reasoning", "Explain in three sentences why the ocean appears blue."),
    ("code",      "Write a short Python function that reverses a linked list."),
    ("factual",   "The capital of France is"),
    ("chat",      "Give me three tips for sleeping better."),
]

def call(prompt, think=True):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": MAXT, "enable_thinking": think}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    dt = time.time() - t0
    n = d["usage"]["completion_tokens"]
    return n, dt, d["choices"][0]["message"]["content"]

print(f"{'prompt':<10} {'tokens':>7} {'seconds':>8} {'tok/s':>8}")
tot_n = tot_t = 0
for name, p in PROMPTS:
    call(p)                                  # warm
    n, dt, _ = call(p)
    tot_n += n; tot_t += dt
    print(f"{name:<10} {n:>7} {dt:>8.2f} {n/dt:>8.2f}")
print(f"{'overall':<10} {tot_n:>7} {tot_t:>8.2f} {tot_n/tot_t:>8.2f}")

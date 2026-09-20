#!/usr/bin/env python3
"""Benchmark and compare Splash Q4 vs Splash Q8 on identical prompts."""

import json
import time
import urllib.request
import urllib.error
import sys

PROMPTS = [
    {
        "id": "math_logic",
        "category": "Math & Logic",
        "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Show your step-by-step algebraic derivation, followed by the final answer."
    },
    {
        "id": "coding",
        "category": "Coding & Algorithms",
        "prompt": "Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges overlapping intervals. It must be clean, optimal in O(N log N) time, handle empty lists and negative coordinates, and include type annotations and docstring."
    },
    {
        "id": "spatial_constraint",
        "category": "Constraint Reasoning",
        "prompt": "Three friends (Alice, Bob, Charlie) are sitting in a row of 3 chairs numbered 1 to 3 from left to right. Constraints:\n1. Alice never sits next to Bob.\n2. Charlie is to the right of Alice (higher chair number).\nWho sits in chair 1, chair 2, and chair 3? Give a logical deduction step by step."
    },
    {
        "id": "technical_explanation",
        "category": "Domain Knowledge",
        "prompt": "Explain clearly the respective roles and architectural differences between FlashAttention, PagedAttention, and Speculative Decoding in high-performance LLM inference engines."
    },
    {
        "id": "executive_summary",
        "category": "Nuanced Writing",
        "prompt": "In exactly three clear, professional sentences, explain why memory bandwidth (rather than raw compute TFLOPS) is the dominant latency bottleneck during autoregressive token generation in transformer LLMs."
    }
]

ENDPOINTS = [
    {
        "name": "Splash Q4",
        "url": "http://127.0.0.1:8000/v1/chat/completions",
        "model": "incoai/Qwen3.8-27B-Splash",
    },
    {
        "name": "Splash Q8",
        "url": "http://127.0.0.1:8001/v1/chat/completions",
        "model": "incoai/Qwen3.8-27B-Splash-Q8",
    }
]

def query_endpoint(endpoint, prompt, max_tokens=600):
    payload = {
        "model": endpoint["model"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint["url"],
        data=data,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            t1 = time.perf_counter()
            body = json.loads(raw)
            return body, t1 - t0
    except Exception as e:
        return {"error": str(e)}, 0.0

def main():
    print("=" * 80)
    print("STARTING SPLASH Q4 vs SPLASH Q8 HEAD-TO-HEAD BENCHMARK")
    print("=" * 80)
    
    results = []
    
    for item in PROMPTS:
        print(f"\n" + "-" * 80)
        print(f"PROMPT [{item['category']}]: {item['id']}")
        print(f"Query: {item['prompt'][:100]}...")
        print("-" * 80)
        
        prompt_res = {"id": item["id"], "category": item["category"], "runs": {}}
        
        for ep in ENDPOINTS:
            print(f"\n--- Testing {ep['name']} ---")
            resp, elapsed = query_endpoint(ep, item["prompt"])
            
            if "error" in resp:
                print(f"ERROR on {ep['name']}: {resp['error']}")
                prompt_res["runs"][ep["name"]] = {"error": resp["error"]}
                continue
                
            choice = resp["choices"][0]["message"]
            content = choice.get("content") or ""
            reasoning = choice.get("reasoning_content") or ""
            usage = resp.get("usage", {})
            metrics = resp.get("metrics", {})
            lat = metrics.get("request_latency", {})
            
            ttft_ms = lat.get("ttft_ms", 0.0)
            tok_per_sec = lat.get("stream_tokens_per_second", 0.0)
            completion_tokens = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            
            print(f"Status: OK | Total Time: {elapsed:.2f}s | TTFT: {ttft_ms:.1f}ms | Throughput: {tok_per_sec:.2f} tok/s")
            print(f"Tokens: Prompt={prompt_tokens}, Completion={completion_tokens}")
            if reasoning:
                print(f"Reasoning snippet: {reasoning[:120].strip()}...")
            print(f"Output snippet: {content[:200].strip()}...")
            
            prompt_res["runs"][ep["name"]] = {
                "elapsed": elapsed,
                "ttft_ms": ttft_ms,
                "tok_per_sec": tok_per_sec,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "reasoning": reasoning,
                "content": content,
                "finish_reason": resp["choices"][0].get("finish_reason")
            }
            
        results.append(prompt_res)
        
    out_file = "/tmp/splash_q4_vs_q8_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 80)
    print(f"Benchmark complete. Full raw results saved to {out_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()

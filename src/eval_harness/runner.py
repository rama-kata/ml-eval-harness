"""Runs inference against an Ollama model and collects metrics."""

import json
import time
from pathlib import Path

import httpx


def load_dataset(dataset_path: str) -> list[dict]:
    """Load a JSONL dataset. Each line: {"prompt": "...", "expected": "..."}"""
    items = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def query_ollama(model: str, prompt: str, ollama_url: str) -> tuple[str, float]:
    """Send a prompt to Ollama, return (response_text, latency_seconds)."""
    start = time.perf_counter()
    resp = httpx.post(
        f"{ollama_url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.0, "num_predict": 256}},
        timeout=120.0,
    )
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    return resp.json()["response"].strip(), elapsed


def score_response(response: str, expected: str) -> dict:
    """Basic exact-match and contains scoring."""
    normalized_resp = response.lower().strip()
    normalized_exp = expected.lower().strip()
    return {
        "exact_match": normalized_resp == normalized_exp,
        "contains": normalized_exp in normalized_resp,
    }


def run_eval(
    model: str,
    dataset_path: str,
    metrics: list[str],
    ollama_url: str,
) -> dict:
    """Run a full evaluation pass. Returns aggregated results."""
    dataset = load_dataset(dataset_path)
    total = len(dataset)
    exact_matches = 0
    contains_matches = 0
    latencies = []

    for i, item in enumerate(dataset, 1):
        response, latency = query_ollama(model, item["prompt"], ollama_url)
        scores = score_response(response, item["expected"])

        exact_matches += int(scores["exact_match"])
        contains_matches += int(scores["contains"])
        latencies.append(latency)

        print(f"  [{i}/{total}] latency={latency:.2f}s exact={scores['exact_match']}")

    return {
        "total": total,
        "accuracy": exact_matches / total if total else 0,
        "contains_rate": contains_matches / total if total else 0,
        "avg_latency_ms": (sum(latencies) / len(latencies) * 1000) if latencies else 0,
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] * 1000 if latencies else 0,
        "total_latency_s": sum(latencies),
    }

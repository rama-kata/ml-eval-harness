"""Runs inference against an Ollama model and collects metrics."""

import json
import time

import httpx

from eval_harness.judge import judge_response
from eval_harness.metrics import score_response

CONCISE_WRAPPER = "Answer the following in as few words as possible.\n\n{prompt}"


def load_dataset(dataset_path: str) -> list[dict]:
    """Load a JSONL dataset. Each line: {"prompt": "...", "expected": "...", "system_prompt": "..."}"""
    items = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def query_ollama(
    model: str,
    prompt: str,
    ollama_url: str,
    system_prompt: str | None = None,
) -> tuple[str, float]:
    """Send a prompt to Ollama, return (response_text, latency_seconds)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 256},
    }
    if system_prompt:
        payload["system"] = system_prompt

    start = time.perf_counter()
    resp = httpx.post(
        f"{ollama_url}/api/generate",
        json=payload,
        timeout=120.0,
    )
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    return resp.json()["response"].strip(), elapsed


def run_eval(
    model: str,
    dataset_path: str,
    metrics: list[str],
    ollama_url: str,
    concise: bool = False,
    judge_model: str | None = None,
) -> dict:
    """Run a full evaluation pass. Returns aggregated results."""
    dataset = load_dataset(dataset_path)
    total = len(dataset)
    exact_matches = 0
    contains_matches = 0
    token_f1_sum = 0.0
    judge_correct = 0
    judge_partial = 0
    latencies = []
    details = []

    for i, item in enumerate(dataset, 1):
        prompt = item["prompt"]
        if concise:
            prompt = CONCISE_WRAPPER.format(prompt=prompt)

        system_prompt = item.get("system_prompt")

        response, latency = query_ollama(model, prompt, ollama_url, system_prompt)
        scores = score_response(response, item["expected"])

        exact_matches += int(scores["exact_match"])
        contains_matches += int(scores["contains"])
        token_f1_sum += scores["token_f1"]
        latencies.append(latency)

        detail = {
            "prompt": item["prompt"],
            "expected": item["expected"],
            "response": response,
            "latency_s": round(latency, 3),
            **scores,
        }

        # LLM-as-judge
        if judge_model:
            verdict = judge_response(
                prompt=item["prompt"],
                expected=item["expected"],
                response=response,
                judge_model=judge_model,
                ollama_url=ollama_url,
            )
            detail["judge_verdict"] = verdict["verdict"]
            detail["judge_reason"] = verdict["reason"]
            if verdict["verdict"] == "correct":
                judge_correct += 1
            elif verdict["verdict"] == "partial":
                judge_partial += 1

        details.append(detail)

        status = f"exact={scores['exact_match']} f1={scores['token_f1']:.2f}"
        if judge_model:
            status += f" judge={detail['judge_verdict']}"
        print(f"  [{i}/{total}] latency={latency:.2f}s {status}")

    results = {
        "total": total,
        "accuracy": exact_matches / total if total else 0,
        "contains_rate": contains_matches / total if total else 0,
        "avg_token_f1": token_f1_sum / total if total else 0,
        "avg_latency_ms": (sum(latencies) / len(latencies) * 1000) if latencies else 0,
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] * 1000 if latencies else 0,
        "total_latency_s": sum(latencies),
        "details": details,
    }

    if judge_model:
        results["judge_model"] = judge_model
        results["judge_correct_rate"] = judge_correct / total if total else 0
        results["judge_partial_rate"] = judge_partial / total if total else 0

    return results

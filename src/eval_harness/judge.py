"""LLM-as-judge scoring via a second Ollama model."""

import json

import httpx

JUDGE_PROMPT = """You are an evaluation judge. Given a question, the expected answer, and a model's response, determine if the response is correct.

Question: {prompt}
Expected answer: {expected}
Model response: {response}

Score the response:
- "correct" if the response contains the right answer, even if verbose
- "partial" if the response is partially right or close
- "wrong" if the response is incorrect

Respond in JSON only:
{{"verdict": "correct|partial|wrong", "reason": "one sentence explanation"}}"""


def judge_response(
    prompt: str,
    expected: str,
    response: str,
    judge_model: str,
    ollama_url: str,
) -> dict:
    """Ask a judge model to evaluate a response."""
    judge_input = JUDGE_PROMPT.format(
        prompt=prompt,
        expected=expected,
        response=response,
    )
    resp = httpx.post(
        f"{ollama_url}/api/generate",
        json={
            "model": judge_model,
            "prompt": judge_input,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 128},
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    raw = resp.json()["response"].strip()

    try:
        # Try to extract JSON from the response
        start = raw.index("{")
        end = raw.rindex("}") + 1
        parsed = json.loads(raw[start:end])
        return {
            "verdict": parsed.get("verdict", "unknown"),
            "reason": parsed.get("reason", ""),
        }
    except (ValueError, json.JSONDecodeError):
        return {"verdict": "unknown", "reason": f"Could not parse judge output: {raw[:200]}"}

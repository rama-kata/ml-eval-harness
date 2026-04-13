"""Metric functions for eval runs."""

import re
import string


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def tokens_per_second(total_tokens: int, total_latency_s: float) -> float:
    if total_latency_s == 0:
        return 0.0
    return total_tokens / total_latency_s


def normalize_text(text: str) -> str:
    """Lowercase, strip articles/punctuation/extra whitespace."""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def normalized_exact_match(response: str, expected: str) -> bool:
    return normalize_text(response) == normalize_text(expected)


def normalized_contains(response: str, expected: str) -> bool:
    return normalize_text(expected) in normalize_text(response)


def token_f1(response: str, expected: str) -> float:
    """Word-overlap F1 between response and expected answer."""
    pred_tokens = set(normalize_text(response).split())
    gold_tokens = set(normalize_text(expected).split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return f1_score(precision, recall)


def score_response(response: str, expected: str) -> dict:
    """Run all deterministic metrics on a single response."""
    return {
        "exact_match": normalize_text(response) == normalize_text(expected),
        "contains": normalized_contains(response, expected),
        "token_f1": token_f1(response, expected),
    }

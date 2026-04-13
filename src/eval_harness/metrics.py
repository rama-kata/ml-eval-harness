"""Additional metric functions for eval runs."""


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def tokens_per_second(total_tokens: int, total_latency_s: float) -> float:
    if total_latency_s == 0:
        return 0.0
    return total_tokens / total_latency_s

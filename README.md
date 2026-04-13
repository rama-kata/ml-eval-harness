# ml-eval-harness

Lightweight evaluation harness for comparing local LLMs via Ollama.

## Setup

```bash
pip install -e ".[dev]"
```

## Usage

Run an evaluation:
```bash
evalrun evaluate qwen2.5:3b datasets/sample_qa.jsonl
```

Compare results across models:
```bash
evalrun compare
```

## Dataset Format

JSONL with one object per line:
```json
{"prompt": "What is the capital of France?", "expected": "Paris"}
```

## Metrics

- **accuracy** — exact match after normalization
- **contains_rate** — expected answer appears in response
- **avg_latency_ms** — mean response time
- **p95_latency_ms** — 95th percentile response time

# ml-eval-harness

Lightweight evaluation harness for comparing local LLMs via Ollama.

**[Live Dashboard](https://rama-kata.github.io/ml-eval-harness/)** — interactive ECharts dashboard with heatmaps, latency analysis, accuracy scatter plots, and per-item drill-down.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

Run an evaluation:
```bash
evalrun evaluate qwen2.5:3b datasets/sample_qa.jsonl
```

With concise prompts (requests short answers):
```bash
evalrun evaluate qwen2.5:3b datasets/sample_qa.jsonl --concise
```

With LLM-as-judge (uses a second model to grade correctness):
```bash
evalrun evaluate qwen2.5:3b datasets/sample_qa.jsonl --concise --judge qwen2.5:7b
```

Compare results across models:
```bash
evalrun compare
```

View per-item details for a specific run:
```bash
evalrun details 1
```

Generate visualizations:
```bash
evalrun visualize
```

## Dataset Format

JSONL with one object per line:
```json
{"prompt": "What is the capital of France?", "expected": "Paris"}
{"prompt": "Explain briefly.", "expected": "...", "system_prompt": "You are concise."}
```

## Metrics

| Metric | Description |
|--------|-------------|
| **Exact Match** | Normalized text equality (strips articles, punctuation, case) |
| **Contains** | Expected answer appears within response |
| **Token F1** | Word-overlap F1 between response and expected |
| **Judge** | LLM-as-judge verdict (correct / partial / wrong) |
| **Latency** | avg and p95 response time |

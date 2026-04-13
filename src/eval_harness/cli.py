"""CLI entry point for the eval harness."""

import json

import click

from eval_harness.runner import run_eval
from eval_harness.storage import ResultsDB


@click.group()
def main():
    """ml-eval-harness: compare local LLMs on structured tasks."""


@main.command()
@click.argument("model")
@click.argument("dataset", type=click.Path(exists=True))
@click.option("--metrics", "-m", default="accuracy,latency", help="Comma-separated metrics")
@click.option("--output", "-o", default="results.db", help="SQLite output file")
@click.option("--ollama-url", default="http://localhost:11434", help="Ollama API base URL")
@click.option("--concise", is_flag=True, help="Wrap prompts to request short answers")
@click.option("--judge", default=None, help="Ollama model to use as LLM judge (e.g. qwen2.5:7b)")
def evaluate(
    model: str, dataset: str, metrics: str, output: str,
    ollama_url: str, concise: bool, judge: str | None,
):
    """Run evaluation for MODEL on DATASET."""
    metric_list = [m.strip() for m in metrics.split(",")]
    db = ResultsDB(output)

    click.echo(f"Evaluating {model} on {dataset}")
    click.echo(f"Metrics: {metric_list}")
    if concise:
        click.echo("Mode: concise (short answer prompts)")
    if judge:
        click.echo(f"Judge: {judge}")

    results = run_eval(
        model=model,
        dataset_path=dataset,
        metrics=metric_list,
        ollama_url=ollama_url,
        concise=concise,
        judge_model=judge,
    )

    db.save_run(model=model, dataset=dataset, results=results)
    click.echo(f"\nResults saved to {output}")

    # Print summary
    click.echo(f"\n--- Summary ---")
    click.echo(f"Exact match:  {results['accuracy']:.1%}")
    click.echo(f"Contains:     {results['contains_rate']:.1%}")
    click.echo(f"Avg token F1: {results['avg_token_f1']:.3f}")
    click.echo(f"Avg latency:  {results['avg_latency_ms']:.0f}ms")
    if judge:
        click.echo(f"Judge correct: {results['judge_correct_rate']:.1%}")
        click.echo(f"Judge partial: {results['judge_partial_rate']:.1%}")


@main.command()
@click.option("--db", default="results.db", help="SQLite results file")
def compare(db: str):
    """Compare results across models."""
    results_db = ResultsDB(db)
    runs = results_db.get_all_runs()

    if not runs:
        click.echo("No results found.")
        return

    for run in runs:
        line = (f"{run['model']:>30s} | {run['dataset']:>20s} | "
                f"exact={run['accuracy']:.1%} contains={run['contains_rate']:.1%} "
                f"f1={run['avg_token_f1']:.3f} | {run['avg_latency_ms']:.0f}ms")
        if run.get('judge_correct_rate') is not None:
            line += f" | judge={run['judge_correct_rate']:.1%}"
        click.echo(line)


@main.command()
@click.argument("run_id", type=int)
@click.option("--db", default="results.db", help="SQLite results file")
def details(run_id: int, db: str):
    """Show per-item details for a specific run."""
    results_db = ResultsDB(db)
    run = results_db.get_run(run_id)
    if not run:
        click.echo(f"Run {run_id} not found.")
        return

    click.echo(f"Run {run_id}: {run['model']} on {run['dataset']}")
    click.echo(f"{'─' * 80}")

    raw = json.loads(run["raw_results"])
    for i, d in enumerate(raw.get("details", []), 1):
        click.echo(f"\n[{i}] {d['prompt']}")
        click.echo(f"  Expected: {d['expected']}")
        click.echo(f"  Response: {d['response'][:120]}")
        click.echo(f"  exact={d['exact_match']} contains={d['contains']} f1={d['token_f1']:.2f}")
        if "judge_verdict" in d:
            click.echo(f"  Judge: {d['judge_verdict']} — {d['judge_reason']}")

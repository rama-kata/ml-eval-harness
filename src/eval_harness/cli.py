"""CLI entry point for the eval harness."""

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
def evaluate(model: str, dataset: str, metrics: str, output: str, ollama_url: str):
    """Run evaluation for MODEL on DATASET."""
    metric_list = [m.strip() for m in metrics.split(",")]
    db = ResultsDB(output)

    click.echo(f"Evaluating {model} on {dataset}")
    click.echo(f"Metrics: {metric_list}")

    results = run_eval(
        model=model,
        dataset_path=dataset,
        metrics=metric_list,
        ollama_url=ollama_url,
    )

    db.save_run(model=model, dataset=dataset, results=results)
    click.echo(f"Results saved to {output}")


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
        click.echo(f"{run['model']:>30s} | {run['dataset']:>20s} | "
                    f"accuracy={run['accuracy']:.3f} | latency={run['avg_latency_ms']:.0f}ms")

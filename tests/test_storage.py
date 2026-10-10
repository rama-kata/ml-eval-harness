import json

from click.testing import CliRunner

from eval_harness.cli import main
from eval_harness.storage import ResultsDB


def test_save_external_run_keeps_text_metrics_null(tmp_path):
    db = ResultsDB(str(tmp_path / "r.db"))
    db.save_external_run("router:knn", "intents", 0.9, [10.0, 20.0, 30.0],
                         details=[{"id": 1}, {"id": 2}, {"id": 3}], extra={"p50_ms": 20})
    run = db.get_run(1)
    assert run["accuracy"] == 0.9
    assert run["avg_latency_ms"] == 20.0
    assert run["total_items"] == 3
    assert run["contains_rate"] is None and run["avg_token_f1"] is None
    assert json.loads(run["raw_results"])["extra"] == {"p50_ms": 20}


def test_import_cli_and_compare(tmp_path):
    results = tmp_path / "run.json"
    results.write_text(json.dumps({"model": "agent:qwen3", "dataset": "scenarios",
                                   "accuracy": 0.75, "latencies_ms": [1000, 3000]}))
    db = str(tmp_path / "r.db")
    runner = CliRunner()
    out = runner.invoke(main, ["import", str(results), "--db", db])
    assert out.exit_code == 0, out.output
    out = runner.invoke(main, ["compare", "--db", db])
    assert "accuracy=75.0%" in out.output


def test_import_rejects_missing_keys(tmp_path):
    results = tmp_path / "run.json"
    results.write_text(json.dumps({"model": "x"}))
    out = CliRunner().invoke(main, ["import", str(results), "--db", str(tmp_path / "r.db")])
    assert out.exit_code != 0 and "missing keys" in out.output

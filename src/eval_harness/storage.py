"""SQLite storage for eval results."""

import json
import sqlite3
from datetime import datetime, timezone


class ResultsDB:
    def __init__(self, path: str = "results.db"):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                dataset TEXT NOT NULL,
                accuracy REAL,
                contains_rate REAL,
                avg_latency_ms REAL,
                p95_latency_ms REAL,
                total_latency_s REAL,
                total_items INTEGER,
                raw_results TEXT,
                created_at TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def save_run(self, model: str, dataset: str, results: dict):
        self.conn.execute(
            """INSERT INTO runs
               (model, dataset, accuracy, contains_rate, avg_latency_ms,
                p95_latency_ms, total_latency_s, total_items, raw_results, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                model,
                dataset,
                results["accuracy"],
                results["contains_rate"],
                results["avg_latency_ms"],
                results["p95_latency_ms"],
                results["total_latency_s"],
                results["total"],
                json.dumps(results),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def get_all_runs(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT model, dataset, accuracy, avg_latency_ms, created_at "
            "FROM runs ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

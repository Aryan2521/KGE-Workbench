"""SQLite persistence for reproducible KGE benchmark runs."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
DATABASE_PATH = Path(
    os.environ.get("KGE_DATABASE_PATH", PROJECT_DIR / "benchmark_results.db")
)


def get_db_connection() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


@contextmanager
def db_connection():
    connection = get_db_connection()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db() -> None:
    """Create the new run schema without modifying the legacy experiments table."""
    with db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                model_plugin_version TEXT,
                uncertainty_model TEXT NOT NULL,
                dataset TEXT NOT NULL,
                dataset_plugin_version TEXT,
                evaluation_protocol_version TEXT,
                status TEXT NOT NULL CHECK(status IN ('queued', 'running', 'completed', 'failed')),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                started_at TEXT,
                completed_at TEXT,
                seed INTEGER NOT NULL,
                config_hash TEXT NOT NULL,
                config_json TEXT NOT NULL,
                code_fingerprint TEXT,
                dataset_fingerprint TEXT,
                uncertainty_protocol TEXT,
                device TEXT,
                full_evaluation INTEGER NOT NULL DEFAULT 0,
                train_sample_size INTEGER,
                eval_sample_size INTEGER,
                mean_rank REAL,
                mrr REAL,
                hits_at_1 REAL,
                hits_at_3 REAL,
                hits_at_10 REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                mse REAL,
                mae REAL,
                ece REAL,
                brier REAL,
                nll REAL,
                train_seconds REAL,
                eval_seconds REAL,
                total_seconds REAL,
                metrics_json TEXT,
                artifact_dir TEXT,
                error_message TEXT
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_dataset_model ON runs(dataset, model_type)"
        )
        existing_columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
        for column in ("model_plugin_version", "dataset_plugin_version", "evaluation_protocol_version"):
            if column not in existing_columns:
                connection.execute(f"ALTER TABLE runs ADD COLUMN {column} TEXT")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS run_evaluators (
                run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                evaluator_id TEXT NOT NULL,
                evaluator_version TEXT NOT NULL,
                PRIMARY KEY (run_id, evaluator_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS run_metrics (
                run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                evaluator_id TEXT NOT NULL,
                metric_id TEXT NOT NULL,
                label TEXT NOT NULL,
                direction TEXT NOT NULL CHECK(direction IN ('higher', 'lower')),
                metric_value REAL NOT NULL,
                PRIMARY KEY (run_id, evaluator_id, metric_id)
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_run_metrics_run ON run_metrics(run_id)")


def create_run(config: dict[str, Any], config_hash: str, artifact_dir: str) -> int:
    with db_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO runs (
                experiment_name, model_type, model_plugin_version, uncertainty_model,
                dataset, dataset_plugin_version, evaluation_protocol_version, status,
                seed, config_hash, config_json, uncertainty_protocol,
                full_evaluation, train_sample_size, eval_sample_size, artifact_dir
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                config["experiment_name"],
                config["model_type"],
                config.get("model_plugin_version"),
                "Fuzzy" if config["model_type"] == "FuzzyTransE" else "None",
                config["dataset"],
                config.get("dataset_plugin_version"),
                config.get("evaluation_protocol_version"),
                config["seed"],
                config_hash,
                json.dumps(config, sort_keys=True),
                config["uncertainty_protocol"],
                int(config["full_evaluation"]),
                config["train_sample_size"],
                config["eval_sample_size"],
                artifact_dir,
            ),
        )
        run_id = int(cursor.lastrowid)
        evaluator_versions = config.get("evaluator_versions", {})
        connection.executemany(
            "INSERT INTO run_evaluators (run_id, evaluator_id, evaluator_version) VALUES (?, ?, ?)",
            [
                (run_id, evaluator_id, str(evaluator_versions.get(evaluator_id, "unknown")))
                for evaluator_id in config.get("evaluators", [])
            ],
        )
        return run_id


def mark_run_running(run_id: int) -> None:
    with db_connection() as connection:
        connection.execute(
            "UPDATE runs SET status = 'running', started_at = CURRENT_TIMESTAMP WHERE id = ?",
            (run_id,),
        )


def complete_run(run_id: int, result: dict[str, Any]) -> None:
    metrics = result["metrics"]
    with db_connection() as connection:
        connection.execute(
            """
            UPDATE runs SET
                status = 'completed', completed_at = CURRENT_TIMESTAMP,
                code_fingerprint = ?, dataset_fingerprint = ?, device = ?,
                mean_rank = ?, mrr = ?, hits_at_1 = ?, hits_at_3 = ?, hits_at_10 = ?,
                precision = ?, recall = ?, f1_score = ?, mse = ?, mae = ?, ece = ?,
                brier = ?, nll = ?, train_seconds = ?, eval_seconds = ?, total_seconds = ?,
                metrics_json = ?, error_message = NULL
            WHERE id = ?
            """,
            (
                result.get("code_fingerprint"),
                result.get("dataset_fingerprint"),
                result.get("device"),
                metrics.get("mean_rank"),
                metrics.get("mrr"),
                metrics.get("hits_at_1"),
                metrics.get("hits_at_3"),
                metrics.get("hits_at_10"),
                metrics.get("precision"),
                metrics.get("recall"),
                metrics.get("f1_score"),
                metrics.get("mse"),
                metrics.get("mae"),
                metrics.get("ece"),
                metrics.get("brier"),
                metrics.get("nll"),
                result.get("train_seconds"),
                result.get("eval_seconds"),
                result.get("total_seconds"),
                json.dumps(metrics, sort_keys=True),
                run_id,
            ),
        )
        connection.execute("DELETE FROM run_metrics WHERE run_id = ?", (run_id,))
        definitions = result.get("metric_definitions", {})
        connection.executemany(
            """
            INSERT INTO run_metrics (
                run_id, evaluator_id, metric_id, label, direction, metric_value
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    definitions.get(metric_id, {}).get("evaluator_id", "legacy"),
                    metric_id,
                    definitions.get(metric_id, {}).get("label", metric_id.replace("_", " ").title()),
                    definitions.get(metric_id, {}).get("direction", "higher"),
                    float(value),
                )
                for metric_id, value in metrics.items()
                if value is not None
            ],
        )


def fail_run(run_id: int, message: str) -> None:
    with db_connection() as connection:
        connection.execute(
            """
            UPDATE runs SET status = 'failed', completed_at = CURRENT_TIMESTAMP,
                error_message = ? WHERE id = ?
            """,
            (message[-4000:], run_id),
        )


def get_run(run_id: int) -> dict[str, Any] | None:
    with db_connection() as connection:
        row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        value = _deserialize_row(row)
        _attach_dynamic_metrics(connection, [value])
    return value


def get_runs(limit: int = 250) -> list[dict[str, Any]]:
    with db_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        values = [_deserialize_row(row) for row in rows]
        _attach_dynamic_metrics(connection, values)
    return values


def delete_runs(run_ids: list[int]) -> int:
    normalized = sorted({int(run_id) for run_id in run_ids if int(run_id) > 0})
    if not normalized:
        return 0
    placeholders = ",".join("?" for _ in normalized)
    with db_connection() as connection:
        cursor = connection.execute(
            f"DELETE FROM runs WHERE id IN ({placeholders})", normalized
        )
        return int(cursor.rowcount)


def _deserialize_row(row: sqlite3.Row) -> dict[str, Any]:
    value = dict(row)
    value["full_evaluation"] = bool(value["full_evaluation"])
    for key in ("config_json", "metrics_json"):
        raw = value.pop(key, None)
        value[key.removesuffix("_json")] = json.loads(raw) if raw else None
    return value


def _attach_dynamic_metrics(connection: sqlite3.Connection, runs: list[dict[str, Any]]) -> None:
    if not runs:
        return
    run_map = {run["id"]: run for run in runs}
    placeholders = ",".join("?" for _ in run_map)
    metric_rows = connection.execute(
        f"SELECT * FROM run_metrics WHERE run_id IN ({placeholders}) ORDER BY evaluator_id, metric_id",
        list(run_map),
    ).fetchall()
    evaluator_rows = connection.execute(
        f"SELECT * FROM run_evaluators WHERE run_id IN ({placeholders}) ORDER BY evaluator_id",
        list(run_map),
    ).fetchall()
    for run in runs:
        run.setdefault("metrics", run.get("metrics") or {})
        run["metric_definitions"] = {}
        run["evaluators"] = []
    for row in metric_rows:
        run = run_map[row["run_id"]]
        run["metrics"][row["metric_id"]] = row["metric_value"]
        run[row["metric_id"]] = row["metric_value"]
        run["metric_definitions"][row["metric_id"]] = {
            "label": row["label"],
            "direction": row["direction"],
            "evaluator_id": row["evaluator_id"],
        }
    for row in evaluator_rows:
        run_map[row["run_id"]]["evaluators"].append({
            "id": row["evaluator_id"], "version": row["evaluator_version"]
        })


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DATABASE_PATH}")

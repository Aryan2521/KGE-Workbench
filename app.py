"""Local web application for reproducible KGE benchmark execution."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template, request

import database
from plugin_registry import PluginError, PluginParameterError, apply_parameter_schema, registry


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = Path(os.environ.get("KGE_ARTIFACT_DIR", PROJECT_DIR / "artifacts"))
RUNNER_PATH = PROJECT_DIR / "benchmark_runner.py"
DEFAULT_THESIS_DIR = Path.home() / "Documents" / "Thesis" / "Thesis new"
THESIS_MODEL_FILES = ("transe.py", "RFM_TransE.py", "utils.py")

DATASET_DESCRIPTIONS = {
    "Synthetic": "Generated family-relationship graph used for fast, controlled experiments.",
    "CoDEx-S": "Compact, diverse knowledge graph extracted from Wikidata for reproducible link prediction.",
    "CoDEx-M": "Medium-scale CoDEx benchmark with broader entity and relation coverage.",
    "FB15k": "Freebase subset containing general-world entities and many relation types.",
    "FB15k-237": "Leakage-reduced Freebase benchmark with inverse relation pairs removed.",
    "WN18RR": "WordNet benchmark focused on lexical and semantic relationships.",
    "YAGO3-10": "Large real-world benchmark derived from YAGO, restricted to frequent entities.",
}


DEFAULTS: dict[str, Any] = {
    "model_type": "TransE",
    "dataset": "Synthetic",
    "evaluators": ["link_prediction", "triple_classification", "calibration"],
    "evaluator_parameters": {},
    "seed": 42,
    "embedding_dimension": 50,
    "learning_rate": 0.01,
    "batch_size": 1024,
    "epochs": 10,
    "margin": 1.5,
    "p_norm": 1,
    "train_fraction": 1.0,
    "validation_fraction": 1.0,
    "test_fraction": 1.0,
    "train_sample_size": 0,
    "eval_sample_size": 0,
    "validation_sample_size": 0,
    "test_sample_size": 0,
    "full_evaluation": True,
    "candidate_chunk_size": 4096,
    "membership_function": "exponential",
    "fuzziness_type": "relation-specific",
    "uncertainty_level": "low",
    "uncertainty_protocol": "controlled_soft_label_injection",
    "soft_label_pos": 0.95,
    "soft_label_neg": 0.05,
    "noise_level": 0.0,
    "alpha": 0.1,
    "fuzzy_bias": 0.0,
    "lambda_fuzzy": 0.1,
    "synthetic_families": 20,
}

CORE_PARAMETERS = {
    "seed": ("integer", 0, 2_147_483_647),
    "embedding_dimension": ("integer", 2, 2048),
    "batch_size": ("integer", 1, 1_000_000),
    "epochs": ("integer", 1, 100_000),
    "p_norm": ("integer", 1, 2),
    "train_sample_size": ("integer", 0, 10_000_000),
    "eval_sample_size": ("integer", 0, 1_000_000),
    "validation_sample_size": ("integer", 0, 1_000_000),
    "test_sample_size": ("integer", 0, 1_000_000),
    "candidate_chunk_size": ("integer", 128, 1_000_000),
    "synthetic_families": ("integer", 2, 10_000),
    "learning_rate": ("number", 1e-8, 10.0),
    "train_fraction": ("number", 1e-12, 1.0),
    "validation_fraction": ("number", 1e-12, 1.0),
    "test_fraction": ("number", 1e-12, 1.0),
    "margin": ("number", 0.0, 1000.0),
    "soft_label_pos": ("number", 0.0, 1.0),
    "soft_label_neg": ("number", 0.0, 1.0),
    "noise_level": ("number", 0.0, 1.0),
    "alpha": ("number", 1e-6, 1000.0),
    "fuzzy_bias": ("number", -1000.0, 1000.0),
    "lambda_fuzzy": ("number", 0.0, 1000.0),
}


class ParameterError(ValueError):
    def __init__(self, field: str, message: str):
        self.field = field
        super().__init__(message)


def public_core_parameters() -> dict[str, dict[str, Any]]:
    return {
        name: {"type": kind, "minimum": low, "maximum": high, "default": DEFAULTS[name]}
        for name, (kind, low, high) in CORE_PARAMETERS.items()
    }


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    try:
        if isinstance(value, bool) or value is None or str(value).strip() == "" or (isinstance(value, float) and not value.is_integer()):
            raise ValueError
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ParameterError(name, f"{name} must be an integer") from error
    if not minimum <= parsed <= maximum:
        raise ParameterError(name, f"{name} must be between {minimum} and {maximum}")
    return parsed


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    try:
        if isinstance(value, bool) or value is None or str(value).strip() == "":
            raise ValueError
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ParameterError(name, f"{name} must be numeric") from error
    if not math.isfinite(parsed):
        raise ParameterError(name, f"{name} must be a finite number")
    if not minimum <= parsed <= maximum:
        raise ParameterError(name, f"{name} must be between {minimum} and {maximum}")
    return parsed


def validate_config(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    config = {**DEFAULTS, **payload}
    name = str(config.get("experiment_name", "")).strip()
    if not name:
        raise ValueError("experiment_name is required")
    if len(name) > 160:
        raise ValueError("experiment_name must be at most 160 characters")
    registry.refresh()
    try:
        model_plugin = registry.get("models", str(config["model_type"]))
        dataset_plugin = registry.get("datasets", str(config["dataset"]))
        apply_parameter_schema(config, model_plugin)
        apply_parameter_schema(config, dataset_plugin)
        evaluator_ids = config.get("evaluators")
        if not isinstance(evaluator_ids, list) or not evaluator_ids:
            raise PluginError("Select at least one evaluator")
        if len(evaluator_ids) != len(set(evaluator_ids)):
            raise PluginError("Evaluator selections must be unique")
        evaluator_plugins = [registry.get("evaluators", str(evaluator_id)) for evaluator_id in evaluator_ids]
        registry.validate_compatibility(model_plugin, dataset_plugin, evaluator_plugins)
        evaluator_parameters = config.get("evaluator_parameters", {})
        if not isinstance(evaluator_parameters, dict):
            raise PluginError("evaluator_parameters must be an object")
        for evaluator in evaluator_plugins:
            parameters = evaluator_parameters.setdefault(evaluator.plugin_id, {})
            if not isinstance(parameters, dict):
                raise PluginError(f"Parameters for {evaluator.plugin_id} must be an object")
            apply_parameter_schema(parameters, evaluator)
    except PluginParameterError as error:
        raise ParameterError(error.field, str(error)) from error
    except PluginError as error:
        raise ValueError(str(error)) from error
    if config["membership_function"] not in {"exponential", "sigmoid", "gaussian"}:
        raise ValueError("Unsupported membership function")
    if config["fuzziness_type"] not in {"global", "relation-specific"}:
        raise ValueError("Unsupported fuzziness type")
    if config["uncertainty_level"] not in {"low", "medium", "high", "custom"}:
        raise ValueError("Unsupported uncertainty level")

    config.update(
        experiment_name=name,
        seed=_integer(config["seed"], "seed", 0, 2_147_483_647),
        embedding_dimension=_integer(config["embedding_dimension"], "embedding_dimension", 2, 2048),
        batch_size=_integer(config["batch_size"], "batch_size", 1, 1_000_000),
        epochs=_integer(config["epochs"], "epochs", 1, 100_000),
        p_norm=_integer(config["p_norm"], "p_norm", 1, 2),
        train_sample_size=_integer(config["train_sample_size"], "train_sample_size", 0, 10_000_000),
        eval_sample_size=_integer(config["eval_sample_size"], "eval_sample_size", 0, 1_000_000),
        validation_sample_size=_integer(config["validation_sample_size"], "validation_sample_size", 0, 1_000_000),
        test_sample_size=_integer(config["test_sample_size"], "test_sample_size", 0, 1_000_000),
        candidate_chunk_size=_integer(config["candidate_chunk_size"], "candidate_chunk_size", 128, 1_000_000),
        synthetic_families=_integer(config["synthetic_families"], "synthetic_families", 2, 10_000),
        learning_rate=_number(config["learning_rate"], "learning_rate", 1e-8, 10.0),
        train_fraction=_number(config["train_fraction"], "train_fraction", 1e-12, 1.0),
        validation_fraction=_number(config["validation_fraction"], "validation_fraction", 1e-12, 1.0),
        test_fraction=_number(config["test_fraction"], "test_fraction", 1e-12, 1.0),
        margin=_number(config["margin"], "margin", 0.0, 1000.0),
        soft_label_pos=_number(config["soft_label_pos"], "soft_label_pos", 0.0, 1.0),
        soft_label_neg=_number(config["soft_label_neg"], "soft_label_neg", 0.0, 1.0),
        noise_level=_number(config["noise_level"], "noise_level", 0.0, 1.0),
        alpha=_number(config["alpha"], "alpha", 1e-6, 1000.0),
        fuzzy_bias=_number(config["fuzzy_bias"], "fuzzy_bias", -1000.0, 1000.0),
        lambda_fuzzy=_number(config["lambda_fuzzy"], "lambda_fuzzy", 0.0, 1000.0),
        full_evaluation=bool(config["full_evaluation"]),
    )
    if config["uncertainty_level"] == "custom" and config["soft_label_pos"] < config["soft_label_neg"]:
        raise ParameterError("soft_label_pos", "Positive label must be greater than or equal to negative label")

    thesis_dir = Path(
        str(
            config.get("thesis_dir")
            or os.environ.get("KGE_THESIS_DIR")
            or DEFAULT_THESIS_DIR
        )
    ).expanduser()
    thesis_dir = thesis_dir.resolve()
    required_thesis_files = set(model_plugin.manifest.get("thesis_files", []))
    required_thesis_files.update(dataset_plugin.manifest.get("thesis_files", []))
    for evaluator in evaluator_plugins:
        required_thesis_files.update(evaluator.manifest.get("thesis_files", []))
    missing = [name for name in sorted(required_thesis_files) if not (thesis_dir / name).is_file()]
    if missing:
        raise ValueError(
            f"Thesis new model is unavailable at {thesis_dir}; missing: {', '.join(missing)}"
        )
    config["thesis_dir"] = str(thesis_dir)
    config["model_plugin_version"] = model_plugin.manifest["version"]
    config["dataset_plugin_version"] = dataset_plugin.manifest["version"]
    config["model_plugin_manifest"] = model_plugin.public_manifest()
    config["dataset_plugin_manifest"] = dataset_plugin.public_manifest()
    config["evaluator_manifests"] = [evaluator.public_manifest() for evaluator in evaluator_plugins]
    config["evaluator_versions"] = {
        evaluator.plugin_id: evaluator.manifest["version"] for evaluator in evaluator_plugins
    }
    config["evaluation_protocol_version"] = "reprokge-eval-v4-full-splits"
    return config


def config_hash(config: dict[str, Any]) -> str:
    material = {
        key: value
        for key, value in config.items()
        if key not in {"experiment_name", "thesis_dir"}
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=32)
def _scan_dataset_splits(split_dir_value: str) -> dict[str, Any] | None:
    """Count dataset facts directly from the exact thesis split files."""
    split_dir = Path(split_dir_value)
    split_counts: dict[str, int] = {}
    entities: set[str] = set()
    relations: set[str] = set()
    for split_name, filename in (("train", "train.txt"), ("validation", "valid.txt"), ("test", "test.txt")):
        path = split_dir / filename
        if not path.is_file():
            return None
        count = 0
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    parts = line.split()
                if len(parts) < 3:
                    continue
                entities.update((parts[0], parts[2]))
                relations.add(parts[1])
                count += 1
        split_counts[split_name] = count
    return {
        "entities": len(entities),
        "relations": len(relations),
        "train_triples": split_counts["train"],
        "validation_triples": split_counts["validation"],
        "test_triples": split_counts["test"],
        "total_triples": sum(split_counts.values()),
        "source": "Exact thesis dataset files",
    }


def dataset_info_for_run(run: dict[str, Any]) -> dict[str, Any]:
    dataset = run["dataset"]
    config = run.get("config") or {}
    artifact_dir = Path(run.get("artifact_dir") or "")
    result_path = artifact_dir / "result.json"
    if result_path.is_file():
        try:
            recorded = json.loads(result_path.read_text(encoding="utf-8")).get("dataset_info")
            if recorded:
                return {
                    "name": dataset,
                    "description": DATASET_DESCRIPTIONS.get(dataset, "Knowledge graph benchmark dataset."),
                    "source": "Recorded with this run",
                    **recorded,
                }
        except (OSError, json.JSONDecodeError):
            pass

    if dataset != "Synthetic":
        thesis_dir = Path(config.get("thesis_dir") or DEFAULT_THESIS_DIR)
        scanned = _scan_dataset_splits(str((thesis_dir / "data" / dataset).resolve()))
        if scanned:
            return {
                "name": dataset,
                "description": DATASET_DESCRIPTIONS.get(dataset, "Knowledge graph benchmark dataset."),
                **scanned,
            }

    info: dict[str, Any] = {
        "name": dataset,
        "description": DATASET_DESCRIPTIONS.get(dataset, "Knowledge graph benchmark dataset."),
        "entities": None,
        "relations": None,
        "train_triples": None,
        "validation_triples": None,
        "test_triples": None,
        "total_triples": None,
        "source": "Run metadata",
    }
    log_path = artifact_dir / "run.log"
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(f"Dataset {dataset}:"):
                numbers = [int(token) for token in line.replace(",", "").split() if token.isdigit()]
                if len(numbers) >= 2:
                    info["entities"], info["relations"] = numbers[:2]
            elif line.startswith("Training triples:"):
                numbers = [int(token) for token in line.replace(";", " ").split() if token.isdigit()]
                if numbers:
                    info["train_triples"] = numbers[0]
                    if len(numbers) > 1:
                        info["evaluated_triples"] = numbers[1]
    return info


def remove_run_artifacts(artifact_dir_value: str | None) -> bool:
    """Delete one run directory only when it is safely contained by ARTIFACT_ROOT."""
    if not artifact_dir_value:
        return False
    artifact_dir = Path(artifact_dir_value).expanduser().resolve()
    artifact_root = ARTIFACT_ROOT.expanduser().resolve()
    if artifact_dir == artifact_root or artifact_root not in artifact_dir.parents:
        raise ValueError("Refusing to delete an artifact path outside the run artifact directory")
    if not artifact_dir.exists():
        return False
    if not artifact_dir.is_dir():
        raise ValueError("Run artifact path is not a directory")
    shutil.rmtree(artifact_dir)
    return True


def execute_run(run_id: int, config_path: Path, result_path: Path, log_path: Path) -> None:
    database.mark_run_running(run_id)
    command = [
        sys.executable,
        "-u",
        str(RUNNER_PATH),
        "--config",
        str(config_path),
        "--result",
        str(result_path),
    ]
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
            process = subprocess.run(
                command,
                cwd=PROJECT_DIR,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if process.returncode != 0:
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            database.fail_run(run_id, f"Runner exited with code {process.returncode}\n{log_tail}")
            return
        result = json.loads(result_path.read_text(encoding="utf-8"))
        database.complete_run(run_id, result)
    except Exception as error:  # keep worker failure visible in the database
        database.fail_run(run_id, f"{type(error).__name__}: {error}")


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__)
    if test_config:
        app.config.update(test_config)
    database.init_db()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def index():
        return render_template("landing.html")

    @app.get("/experiment")
    def experiment_page():
        return render_template(
            "index.html",
            initial_section="run-section",
            page_title="Run a real benchmark",
            page_description="Execute an installed model plugin with a recorded configuration and fixed seed.",
        )

    @app.get("/tracking")
    def tracking_page():
        return render_template(
            "index.html",
            initial_section="tracking-section",
            page_title="Tracking and comparison",
            page_description="Filter, compare, and export reproducible experiment runs.",
        )

    @app.get("/compare")
    def compare_page():
        return render_template("compare.html")

    @app.get("/api/runs")
    def list_runs():
        return jsonify(database.get_runs())

    @app.get("/api/plugins")
    def installed_plugins():
        try:
            registry.refresh()
            return jsonify({**registry.public_catalog(), "core_parameters": public_core_parameters()})
        except PluginError as error:
            return jsonify({"error": str(error)}), 500

    @app.delete("/api/runs")
    def delete_run_records():
        payload = request.get_json(silent=True) or {}
        delete_all = payload.get("all") is True
        required_confirmation = "DELETE ALL" if delete_all else "DELETE SELECTED"
        if payload.get("confirmation") != required_confirmation:
            return jsonify({"error": "Deletion confirmation was not accepted"}), 400

        if delete_all:
            targets = database.get_runs(limit=1_000_000)
        else:
            raw_ids = payload.get("ids")
            if not isinstance(raw_ids, list) or not raw_ids:
                return jsonify({"error": "Select at least one run to delete"}), 400
            try:
                run_ids = sorted({int(run_id) for run_id in raw_ids if int(run_id) > 0})
            except (TypeError, ValueError):
                return jsonify({"error": "Run IDs must be positive integers"}), 400
            if len(run_ids) > 250:
                return jsonify({"error": "At most 250 runs can be deleted at once"}), 400
            targets = [run for run_id in run_ids if (run := database.get_run(run_id))]
            if len(targets) != len(run_ids):
                return jsonify({"error": "One or more selected runs no longer exist"}), 404

        if not targets:
            return jsonify({"deleted": 0, "artifact_directories_deleted": 0})
        active = [run["id"] for run in targets if run["status"] in {"queued", "running"}]
        if active:
            return jsonify({
                "error": "Active runs cannot be deleted. Wait for them to finish first.",
                "active_run_ids": active,
            }), 409

        deleted = database.delete_runs([run["id"] for run in targets])
        artifacts_deleted = 0
        warnings: list[str] = []
        for run in targets:
            try:
                artifacts_deleted += int(remove_run_artifacts(run.get("artifact_dir")))
            except (OSError, ValueError) as error:
                warnings.append(f"Run #{run['id']} artifacts were retained: {error}")
        return jsonify({
            "deleted": deleted,
            "artifact_directories_deleted": artifacts_deleted,
            "warnings": warnings,
        })

    @app.get("/api/runs/<int:run_id>")
    def get_run(run_id: int):
        run = database.get_run(run_id)
        return (jsonify(run), 200) if run else (jsonify({"error": "Run not found"}), 404)

    @app.get("/api/runs/<int:run_id>/dataset-info")
    def get_run_dataset_info(run_id: int):
        run = database.get_run(run_id)
        if not run:
            return jsonify({"error": "Run not found"}), 404
        return jsonify(dataset_info_for_run(run))

    @app.post("/api/runs")
    def start_run():
        try:
            config = validate_config(request.get_json(silent=True) or {})
        except ParameterError as error:
            return jsonify({"error": str(error), "field": error.field}), 400
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        digest = config_hash(config)
        provisional_dir = ARTIFACT_ROOT / f"pending-{time.time_ns()}"
        run_id = database.create_run(config, digest, str(provisional_dir))
        artifact_dir = ARTIFACT_ROOT / f"run-{run_id:05d}"
        artifact_dir.mkdir(parents=True, exist_ok=False)
        config_path = artifact_dir / "config.json"
        result_path = artifact_dir / "result.json"
        log_path = artifact_dir / "run.log"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
        with database.db_connection() as connection:
            connection.execute(
                "UPDATE runs SET artifact_dir = ? WHERE id = ?", (str(artifact_dir), run_id)
            )
        worker = threading.Thread(
            target=execute_run,
            args=(run_id, config_path, result_path, log_path),
            daemon=True,
            name=f"kge-run-{run_id}",
        )
        worker.start()
        return jsonify({"id": run_id, "status": "queued"}), 202

    @app.get("/api/runs/<int:run_id>/events")
    def run_events(run_id: int):
        run = database.get_run(run_id)
        if not run:
            return jsonify({"error": "Run not found"}), 404
        log_path = Path(run["artifact_dir"]) / "run.log"

        def generate():
            offset = 0
            while True:
                if log_path.exists():
                    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
                        handle.seek(offset)
                        while True:
                            line = handle.readline()
                            if not line:
                                break
                            offset = handle.tell()
                            yield f"data: {line.rstrip()}\n\n"
                current = database.get_run(run_id)
                if current and current["status"] in {"completed", "failed"}:
                    yield f"data: [FINISHED] {current['status']}:{run_id}\n\n"
                    break
                time.sleep(0.25)

        return Response(generate(), mimetype="text/event-stream")

    @app.get("/api/runs.csv")
    def export_runs():
        runs = database.get_runs(limit=10_000)
        fields = [
            "id", "experiment_name", "model_type", "model_plugin_version",
            "dataset", "dataset_plugin_version", "evaluation_protocol_version", "status", "seed",
            "mean_rank", "mrr", "hits_at_1", "hits_at_3", "hits_at_10",
            "precision", "recall", "f1_score", "mse", "mae", "ece", "brier",
            "nll", "train_seconds", "eval_seconds", "total_seconds", "device",
            "config_hash", "code_fingerprint", "dataset_fingerprint", "created_at",
        ]
        dynamic_metric_ids = sorted({
            metric_id for run in runs for metric_id in (run.get("metrics") or {})
            if metric_id not in fields
        })
        fields.extend(dynamic_metric_ids)
        fields.append("evaluators")
        export_rows = []
        for run in runs:
            row = {**run, **(run.get("metrics") or {})}
            row["evaluators"] = ", ".join(
                f"{item['id']}@{item.get('version', 'unknown')}" for item in run.get("evaluators", [])
            )
            export_rows.append(row)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(export_rows)
        return Response(
            buffer.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=kge-benchmark-runs.csv"},
        )

    @app.get("/api/health")
    def health():
        thesis_dir = Path(os.environ.get("KGE_THESIS_DIR", DEFAULT_THESIS_DIR)).expanduser().resolve()
        missing = [name for name in THESIS_MODEL_FILES if not (thesis_dir / name).is_file()]
        return jsonify({
            "status": "ok",
            "database": str(database.DATABASE_PATH),
            "thesis_dir": str(thesis_dir),
            "thesis_model": "RFM-TransE (Thesis new)",
            "thesis_ready": not missing,
            "missing_files": missing,
            "installed_models": len(registry.models()),
            "installed_datasets": len(registry.datasets()),
            "installed_evaluators": len(registry.evaluators()),
        })

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)

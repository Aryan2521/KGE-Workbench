import tempfile
import unittest
from pathlib import Path

import database


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.original_path = database.DATABASE_PATH
        database.DATABASE_PATH = Path(self.tempdir.name) / "test.db"
        database.init_db()

    def tearDown(self):
        database.DATABASE_PATH = self.original_path
        self.tempdir.cleanup()

    def test_completed_run_retains_provenance_and_metrics(self):
        config = {
            "experiment_name": "test run",
            "model_type": "TransE",
            "dataset": "Synthetic",
            "seed": 42,
            "uncertainty_protocol": "controlled_soft_label_injection",
            "full_evaluation": False,
            "train_sample_size": 0,
            "eval_sample_size": 10,
        }
        run_id = database.create_run(config, "config-hash", self.tempdir.name)
        database.mark_run_running(run_id)
        database.complete_run(
            run_id,
            {
                "metrics": {
                    "mean_rank": 1.0, "mrr": 1.0, "hits_at_1": 1.0,
                    "hits_at_3": 1.0, "hits_at_10": 1.0, "precision": 1.0,
                    "recall": 1.0, "f1_score": 1.0, "mse": 0.1,
                    "mae": 0.2, "ece": 0.03, "brier": 0.1, "nll": 0.4,
                },
                "device": "cpu",
                "code_fingerprint": "code-hash",
                "dataset_fingerprint": "data-hash",
                "train_seconds": 1.2,
                "eval_seconds": 0.8,
                "total_seconds": 2.0,
            },
        )
        run = database.get_run(run_id)
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["hits_at_1"], 1.0)
        self.assertEqual(run["code_fingerprint"], "code-hash")
        self.assertEqual(run["config"]["seed"], 42)

    def test_delete_runs_removes_only_selected_records(self):
        config = {
            "experiment_name": "delete me", "model_type": "TransE",
            "dataset": "Synthetic", "seed": 42,
            "uncertainty_protocol": "controlled_soft_label_injection",
            "full_evaluation": False, "train_sample_size": 0, "eval_sample_size": 10,
        }
        first = database.create_run(config, "hash-1", self.tempdir.name)
        config["experiment_name"] = "keep me"
        second = database.create_run(config, "hash-2", self.tempdir.name)
        self.assertEqual(database.delete_runs([first]), 1)
        self.assertIsNone(database.get_run(first))
        self.assertIsNotNone(database.get_run(second))

    def test_arbitrary_evaluator_metric_is_stored_without_schema_change(self):
        config = {
            "experiment_name": "custom metric", "model_type": "TransE",
            "dataset": "Synthetic", "seed": 42,
            "uncertainty_protocol": "controlled_soft_label_injection",
            "full_evaluation": False, "train_sample_size": 0, "eval_sample_size": 10,
            "evaluators": ["rule_consistency"],
            "evaluator_versions": {"rule_consistency": "2.0"},
        }
        run_id = database.create_run(config, "hash", self.tempdir.name)
        database.complete_run(run_id, {
            "metrics": {"rule_satisfaction": 0.875},
            "metric_definitions": {
                "rule_satisfaction": {
                    "label": "Rule Satisfaction",
                    "direction": "higher",
                    "evaluator_id": "rule_consistency",
                }
            },
        })
        run = database.get_run(run_id)
        self.assertEqual(run["rule_satisfaction"], 0.875)
        self.assertEqual(run["metric_definitions"]["rule_satisfaction"]["direction"], "higher")
        self.assertEqual(run["evaluators"], [{"id": "rule_consistency", "version": "2.0"}])


if __name__ == "__main__":
    unittest.main()

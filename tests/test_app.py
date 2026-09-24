import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import app
import database


class AppValidationTests(unittest.TestCase):
    def test_validation_rejects_unsupported_model(self):
        with self.assertRaisesRegex(ValueError, "model"):
            app.validate_config({"experiment_name": "bad", "model_type": "ImaginaryKGE"})

    def test_config_hash_ignores_display_name(self):
        first = app.validate_config({"experiment_name": "first"})
        second = app.validate_config({"experiment_name": "second"})
        self.assertEqual(app.config_hash(first), app.config_hash(second))

    def test_health_endpoint(self):
        client = app.app.test_client()
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")
        self.assertTrue(response.get_json()["thesis_ready"])
        self.assertIn("Thesis new", response.get_json()["thesis_dir"])

    def test_plugin_catalog_exposes_installed_models_datasets_and_evaluators(self):
        response = app.app.test_client().get("/api/plugins")
        self.assertEqual(response.status_code, 200)
        catalog = response.get_json()
        self.assertIn("TransE", {plugin["id"] for plugin in catalog["models"]})
        self.assertIn("FuzzyTransE", {plugin["id"] for plugin in catalog["models"]})
        self.assertIn("Synthetic", {plugin["id"] for plugin in catalog["datasets"]})
        self.assertEqual(
            {"link_prediction", "triple_classification", "calibration"},
            {plugin["id"] for plugin in catalog["evaluators"]},
        )

    def test_configuration_records_selected_evaluator_versions(self):
        config = app.validate_config({
            "experiment_name": "ranking only",
            "evaluators": ["link_prediction"],
        })
        self.assertEqual(config["evaluators"], ["link_prediction"])
        self.assertEqual(config["evaluator_versions"], {"link_prediction": "1.0.0"})
        self.assertEqual(config["evaluation_protocol_version"], "reprokge-eval-v4-full-splits")

    def test_default_config_targets_thesis_new_model(self):
        config = app.validate_config({"experiment_name": "new model path"})
        self.assertEqual(Path(config["thesis_dir"]), app.DEFAULT_THESIS_DIR.resolve())
        self.assertEqual(config["fuzziness_type"], "relation-specific")
        self.assertTrue(config["full_evaluation"])
        self.assertEqual(config["train_sample_size"], 0)
        self.assertEqual(config["train_fraction"], 1.0)

    def test_published_numeric_defaults_pass_server_validation(self):
        rules = app.public_core_parameters()
        config = app.validate_config({"experiment_name": "defaults"})
        for name, rule in rules.items():
            self.assertGreaterEqual(rule["default"], rule["minimum"], name)
            self.assertLessEqual(rule["default"], rule["maximum"], name)
            self.assertEqual(config[name], rule["default"], name)

    def test_numeric_errors_identify_field(self):
        for field, value in (("train_fraction", 0), ("alpha", float("nan")),
                             ("epochs", 1.5), ("learning_rate", "")):
            with self.subTest(field=field):
                response = app.app.test_client().post("/api/runs", json={
                    "experiment_name": "invalid", field: value,
                })
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json()["field"], field)

    def test_landing_page_links_to_separate_workspaces(self):
        client = app.app.test_client()
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'href="/experiment"', response.data)
        self.assertIn(b'href="/tracking"', response.data)
        self.assertNotIn(b'id="benchmark-form"', response.data)

    def test_workspace_routes_render_the_expected_initial_section(self):
        client = app.app.test_client()
        experiment = client.get("/experiment")
        tracking = client.get("/tracking")
        self.assertEqual(experiment.status_code, 200)
        self.assertEqual(tracking.status_code, 200)
        self.assertIn(b'data-initial-section="run-section"', experiment.data)
        self.assertIn(b'data-initial-section="tracking-section"', tracking.data)
        self.assertIn(b'id="learning_rate" value="0.01" min="0.00000001" step="any"', experiment.data)
        self.assertIn(b'id="evaluator-options"', experiment.data)
        self.assertIn(b'id="delete-selected-button"', tracking.data)
        self.assertIn(b'id="delete-all-button"', tracking.data)
        self.assertIn(b'id="delete-modal"', tracking.data)
        script = client.get("/static/app.js")
        self.assertIn(b'setInterval(() => loadRuns(), 1500)', script.data)
        self.assertIn(b'The tracking table is up to date.', script.data)
        script.close()

    def test_comparison_page_loads_visual_comparison_assets(self):
        client = app.app.test_client()
        response = client.get("/compare?ids=1,2,3,4")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="ranking-chart"', response.data)
        self.assertIn(b'id="dataset-grid"', response.data)
        self.assertIn(b'two to four runs', response.data)
        self.assertIn(b'/static/compare.js', response.data)
        script = client.get("/static/compare.js")
        self.assertEqual(script.status_code, 200)
        self.assertIn(b'ids.length < 2 || ids.length > 4', script.data)
        self.assertIn(b'direction-winner-', script.data)
        self.assertIn(b'performs best', script.data)
        script.close()

    def test_dataset_split_scanner_counts_entities_relations_and_triples(self):
        with TemporaryDirectory() as directory:
            split_dir = Path(directory)
            (split_dir / "train.txt").write_text("alice\tknows\tbob\nbob\tlikes\tcarol\n", encoding="utf-8")
            (split_dir / "valid.txt").write_text("carol\tknows\talice\n", encoding="utf-8")
            (split_dir / "test.txt").write_text("alice\tlikes\tcarol\n", encoding="utf-8")
            info = app._scan_dataset_splits(str(split_dir))
        self.assertEqual(info["entities"], 3)
        self.assertEqual(info["relations"], 2)
        self.assertEqual(info["train_triples"], 2)
        self.assertEqual(info["validation_triples"], 1)
        self.assertEqual(info["test_triples"], 1)
        self.assertEqual(info["total_triples"], 4)

    def test_delete_endpoint_removes_record_and_contained_artifacts(self):
        original_database = database.DATABASE_PATH
        original_artifacts = app.ARTIFACT_ROOT
        try:
            with TemporaryDirectory() as directory:
                root = Path(directory)
                database.DATABASE_PATH = root / "runs.db"
                app.ARTIFACT_ROOT = root / "artifacts"
                app.ARTIFACT_ROOT.mkdir()
                database.init_db()
                config = app.validate_config({"experiment_name": "disposable"})
                artifact_dir = app.ARTIFACT_ROOT / "run-00001"
                artifact_dir.mkdir()
                (artifact_dir / "run.log").write_text("finished", encoding="utf-8")
                run_id = database.create_run(config, app.config_hash(config), str(artifact_dir))
                database.fail_run(run_id, "test failure")
                response = app.app.test_client().delete(
                    "/api/runs",
                    json={"ids": [run_id], "confirmation": "DELETE SELECTED"},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()["deleted"], 1)
                self.assertIsNone(database.get_run(run_id))
                self.assertFalse(artifact_dir.exists())
        finally:
            database.DATABASE_PATH = original_database
            app.ARTIFACT_ROOT = original_artifacts


if __name__ == "__main__":
    unittest.main()

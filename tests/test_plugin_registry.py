import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from plugin_registry import PluginRegistry, apply_parameter_schema


class PluginRegistryTests(unittest.TestCase):
    def test_discovers_local_plugin_without_application_edits(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plugin_dir = root / "models" / "example"
            plugin_dir.mkdir(parents=True)
            (plugin_dir / "adapter.py").write_text("class Adapter:\n    pass\n", encoding="utf-8")
            manifest = {
                "id": "example",
                "name": "Example model",
                "version": "1.0",
                "entrypoint": "adapter.py:Adapter",
                "parameters": {"dropout": {"type": "number", "default": 0.2, "minimum": 0, "maximum": 1}},
            }
            (plugin_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            discovered = PluginRegistry(root)
            plugin = discovered.get("models", "example")
            config = {}
            apply_parameter_schema(config, plugin)
            self.assertEqual(plugin.load_adapter_class().__name__, "Adapter")
            self.assertEqual(config["dropout"], 0.2)

    def test_compatibility_uses_model_capabilities_and_dataset_features(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = {
                ("models", "model"): {
                    "id": "model", "name": "Model", "version": "1", "entrypoint": "adapter.py:Adapter",
                    "capabilities": {"score_triples": True}, "parameters": {},
                },
                ("datasets", "plain"): {
                    "id": "plain", "name": "Plain", "version": "1", "entrypoint": "adapter.py:Adapter",
                    "features": ["triples"], "parameters": {},
                },
                ("evaluators", "temporal"): {
                    "id": "temporal", "name": "Temporal", "version": "1", "entrypoint": "adapter.py:Adapter",
                    "required_model_capabilities": ["score_triples"],
                    "required_dataset_features": ["triples", "timestamps"], "parameters": {},
                },
            }
            for (kind, plugin_id), manifest in manifests.items():
                plugin_dir = root / kind / plugin_id
                plugin_dir.mkdir(parents=True)
                (plugin_dir / "adapter.py").write_text("class Adapter:\n    pass\n", encoding="utf-8")
                (plugin_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            discovered = PluginRegistry(root)
            with self.assertRaisesRegex(ValueError, "timestamps"):
                discovered.validate_compatibility(
                    discovered.get("models", "model"),
                    discovered.get("datasets", "plain"),
                    [discovered.get("evaluators", "temporal")],
                )


if __name__ == "__main__":
    unittest.main()

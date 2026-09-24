"""Discovery and validation for trusted local ReproKGE plugins."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PLUGIN_ROOT = PROJECT_DIR / "plugins"
PLUGIN_KINDS = ("models", "datasets", "evaluators")


class PluginError(ValueError):
    """Raised when an installed plugin is missing or malformed."""


class PluginParameterError(PluginError):
    def __init__(self, field: str, message: str):
        self.field = field
        super().__init__(message)


@dataclass(frozen=True)
class PluginDefinition:
    kind: str
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]

    @property
    def plugin_id(self) -> str:
        return str(self.manifest["id"])

    def public_manifest(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.manifest.items()
            if key not in {"entrypoint", "source_files", "thesis_files", "source_file", "source_class"}
        }

    def load_adapter_class(self):
        entrypoint = str(self.manifest["entrypoint"])
        module_ref, separator, class_name = entrypoint.partition(":")
        if not separator or not module_ref or not class_name:
            raise PluginError(f"{self.plugin_id}: entrypoint must use module:Class")
        if module_ref.endswith(".py"):
            module_path = (self.root / module_ref).resolve()
            if self.root.resolve() not in module_path.parents:
                raise PluginError(f"{self.plugin_id}: adapter must remain inside its plugin directory")
            if not module_path.is_file():
                raise PluginError(f"{self.plugin_id}: adapter file does not exist: {module_ref}")
            suffix = hashlib.sha256(str(module_path).encode()).hexdigest()[:12]
            module_name = f"reprokge_plugin_{self.kind}_{self.plugin_id}_{suffix}"
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                raise PluginError(f"{self.plugin_id}: could not load adapter module")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_ref)
        try:
            return getattr(module, class_name)
        except AttributeError as error:
            raise PluginError(f"{self.plugin_id}: adapter class {class_name} was not found") from error

    def create_adapter(self, context: dict[str, Any] | None = None):
        return self.load_adapter_class()(self, context or {})

    def fingerprint(self, extra_paths: list[Path] | None = None) -> str:
        paths = [self.manifest_path]
        module_ref = str(self.manifest["entrypoint"]).partition(":")[0]
        if module_ref.endswith(".py"):
            paths.append((self.root / module_ref).resolve())
        else:
            spec = importlib.util.find_spec(module_ref)
            if spec and spec.origin:
                paths.append(Path(spec.origin).resolve())
        paths.extend(extra_paths or [])
        digest = hashlib.sha256()
        for path in sorted(set(paths), key=str):
            digest.update(str(path).encode())
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        return digest.hexdigest()


class PluginRegistry:
    def __init__(self, root: Path = DEFAULT_PLUGIN_ROOT):
        self.root = Path(root)
        self._plugins: dict[str, dict[str, PluginDefinition]] = {kind: {} for kind in PLUGIN_KINDS}
        self.refresh()

    def refresh(self) -> None:
        discovered: dict[str, dict[str, PluginDefinition]] = {kind: {} for kind in PLUGIN_KINDS}
        for kind in discovered:
            kind_dir = self.root / kind
            if not kind_dir.is_dir():
                continue
            for manifest_path in sorted(kind_dir.glob("*/manifest.json")):
                definition = self._read_manifest(kind, manifest_path)
                if definition.plugin_id in discovered[kind]:
                    raise PluginError(f"Duplicate {kind[:-1]} plugin ID: {definition.plugin_id}")
                discovered[kind][definition.plugin_id] = definition
        self._plugins = discovered

    def _read_manifest(self, kind: str, manifest_path: Path) -> PluginDefinition:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PluginError(f"Invalid plugin manifest {manifest_path}: {error}") from error
        for field in ("id", "name", "version", "entrypoint"):
            if not isinstance(manifest.get(field), str) or not manifest[field].strip():
                raise PluginError(f"{manifest_path}: {field} must be a non-empty string")
        parameters = manifest.get("parameters", {})
        if not isinstance(parameters, dict):
            raise PluginError(f"{manifest_path}: parameters must be an object")
        for name, schema in parameters.items():
            if not isinstance(schema, dict):
                raise PluginError(f"{manifest_path}: {name} must have a parameter schema")
            if "default" in schema:
                probe = {name: schema["default"]}
                try:
                    apply_parameter_schema(probe, PluginDefinition(kind, manifest_path.parent, manifest_path, {**manifest, "parameters": {name: schema}}))
                except PluginError as error:
                    raise PluginError(f"{manifest_path}: invalid default for {name}: {error}") from error
        return PluginDefinition(kind, manifest_path.parent, manifest_path, manifest)

    def get(self, kind: str, plugin_id: str) -> PluginDefinition:
        try:
            return self._plugins[kind][plugin_id]
        except KeyError as error:
            singular = kind[:-1] if kind.endswith("s") else kind
            raise PluginError(f"Unknown {singular} plugin: {plugin_id}") from error

    def models(self) -> list[PluginDefinition]:
        return list(self._plugins["models"].values())

    def datasets(self) -> list[PluginDefinition]:
        return list(self._plugins["datasets"].values())

    def evaluators(self) -> list[PluginDefinition]:
        return list(self._plugins["evaluators"].values())

    def public_catalog(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "models": [plugin.public_manifest() for plugin in self.models()],
            "datasets": [plugin.public_manifest() for plugin in self.datasets()],
            "evaluators": [plugin.public_manifest() for plugin in self.evaluators()],
        }

    def compatible_evaluators(self, model: PluginDefinition, dataset: PluginDefinition) -> list[PluginDefinition]:
        capabilities = model.manifest.get("capabilities", {})
        if isinstance(capabilities, list):
            capabilities = {name: True for name in capabilities}
        features = set(dataset.manifest.get("features", []))
        compatible = []
        for evaluator in self.evaluators():
            required_capabilities = evaluator.manifest.get("required_model_capabilities", [])
            required_features = set(evaluator.manifest.get("required_dataset_features", []))
            if all(capabilities.get(name) for name in required_capabilities) and required_features <= features:
                compatible.append(evaluator)
        return compatible

    def validate_compatibility(
        self,
        model: PluginDefinition,
        dataset: PluginDefinition,
        evaluators: list[PluginDefinition],
    ) -> None:
        capabilities = model.manifest.get("capabilities", {})
        if isinstance(capabilities, list):
            capabilities = {name: True for name in capabilities}
        features = set(dataset.manifest.get("features", []))
        for evaluator in evaluators:
            missing_capabilities = [
                name for name in evaluator.manifest.get("required_model_capabilities", [])
                if not capabilities.get(name)
            ]
            missing_features = sorted(set(evaluator.manifest.get("required_dataset_features", [])) - features)
            if missing_capabilities or missing_features:
                reasons = []
                if missing_capabilities:
                    reasons.append(f"model capabilities: {', '.join(missing_capabilities)}")
                if missing_features:
                    reasons.append(f"dataset features: {', '.join(missing_features)}")
                raise PluginError(
                    f"{evaluator.manifest['name']} is incompatible with {model.manifest['name']} "
                    f"and {dataset.manifest['name']}; missing {'; '.join(reasons)}"
                )


def apply_parameter_schema(config: dict[str, Any], plugin: PluginDefinition) -> None:
    """Apply defaults and validate plugin-declared scalar parameters in-place."""
    for name, schema in plugin.manifest.get("parameters", {}).items():
        if not isinstance(schema, dict):
            raise PluginError(f"{plugin.plugin_id}: parameter {name} must be an object")
        value = config.get(name, schema.get("default"))
        if value is None and schema.get("required"):
            raise PluginParameterError(name, f"{name} is required by {plugin.manifest['name']}")
        if value is None:
            continue
        data_type = schema.get("type", "string")
        try:
            if data_type == "integer":
                if isinstance(value, bool) or str(value).strip() == "" or (isinstance(value, float) and not value.is_integer()):
                    raise ValueError
                value = int(value)
            elif data_type == "number":
                if isinstance(value, bool) or str(value).strip() == "":
                    raise ValueError
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError
            elif data_type == "boolean":
                value = bool(value)
            elif data_type == "string":
                value = str(value)
            else:
                raise PluginError(f"{plugin.plugin_id}: unsupported parameter type {data_type}")
        except (TypeError, ValueError) as error:
            raise PluginParameterError(name, f"{name} must be {data_type}") from error
        if "minimum" in schema and value < schema["minimum"]:
            raise PluginParameterError(name, f"{name} must be at least {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise PluginParameterError(name, f"{name} must be at most {schema['maximum']}")
        choices = schema.get("enum")
        if choices is not None and value not in choices:
            raise PluginParameterError(name, f"{name} must be one of: {', '.join(map(str, choices))}")
        config[name] = value


registry = PluginRegistry()

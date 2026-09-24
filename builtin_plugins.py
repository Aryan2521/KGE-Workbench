"""Built-in adapters implementing the public ReproKGE plugin contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from synthetic_dataset import SyntheticKG


def load_source(path: Path, namespace: str):
    spec = importlib.util.spec_from_file_location(namespace, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load plugin source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[namespace] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class DatasetBundle:
    splits: dict[str, Any]
    features: set[str]
    metadata: dict[str, Any]
    mappings: dict[str, Any]
    known_triples: set[tuple[int, int, int]]
    source_files: list[Path]

    @property
    def train(self):
        return self.splits["train"]

    @property
    def validation(self):
        return self.splits["validation"]

    @property
    def test(self):
        return self.splits["test"]

    @property
    def num_entities(self) -> int:
        return int(self.metadata["entities"])

    @property
    def num_relations(self) -> int:
        return int(self.metadata["relations"])


class ThesisModelAdapter:
    """Base contract for models whose implementations live in Thesis new."""

    def __init__(self, definition, context: dict[str, Any]):
        self.definition = definition
        self.thesis_dir = Path(context["thesis_dir"])
        self.score_direction = definition.manifest.get("score_direction", "lower")
        self.capabilities = definition.manifest.get("capabilities", {})

    def source_files(self) -> list[Path]:
        return [self.thesis_dir / self.definition.manifest["source_file"]]

    def build(self, dataset: DatasetBundle, config: dict[str, Any], device: torch.device):
        source_file = self.thesis_dir / self.definition.manifest["source_file"]
        suffix = hashlib.sha256(str(source_file).encode()).hexdigest()[:12]
        module = load_source(source_file, f"reprokge_model_{suffix}")
        model_class = getattr(module, self.definition.manifest["source_class"])
        arguments = {
            "num_entities": dataset.num_entities,
            "num_relations": dataset.num_relations,
            "embedding_dim": config["embedding_dimension"],
            "margin": config["margin"],
            "p_norm": config["p_norm"],
        }
        if self.definition.plugin_id == "FuzzyTransE":
            arguments.update(
                alpha=config["alpha"],
                lambda_fuzzy=config["lambda_fuzzy"],
                membership_function=config["membership_function"],
                fuzzy_bias=config["fuzzy_bias"],
            )
        return model_class(**arguments).to(device)

    def create_optimizer(self, model, config: dict[str, Any]):
        return torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    def train_batch(self, model, positive, negative, positive_labels, negative_labels, optimizer) -> float:
        return float(model.train_step(*positive, *negative, positive_labels, negative_labels, optimizer))

    def score(self, model, heads, relations, tails):
        return model(heads, relations, tails)

    def confidence(self, model, heads, relations, tails, config: dict[str, Any]):
        if self.definition.plugin_id == "FuzzyTransE":
            return model.fuzzy_score(heads, relations, tails)
        return torch.exp(-config["alpha"] * model(heads, relations, tails))

    def save_checkpoint(self, model, path: Path) -> None:
        torch.save(model.state_dict(), path)


class SyntheticDatasetAdapter:
    def __init__(self, definition, context: dict[str, Any]):
        self.definition = definition

    def load(self, config: dict[str, Any]) -> DatasetBundle:
        graph = SyntheticKG(config["synthetic_families"], config["seed"])
        train, validation, test = graph.get_splits(seed=config["seed"])
        return DatasetBundle(
            splits={"train": list(train), "validation": list(validation), "test": list(test)},
            features=set(self.definition.manifest.get("features", [])),
            metadata={"entities": len(graph.entities), "relations": len(graph.relations)},
            mappings={"entity_to_id": graph.entity_to_id, "relation_to_id": graph.relation_to_id},
            known_triples=set(graph.triples),
            source_files=[Path(__file__).resolve().parent / "synthetic_dataset.py"],
        )

    def fingerprint(self, config: dict[str, Any], bundle: DatasetBundle) -> str:
        material = f"SyntheticKG|families={config['synthetic_families']}|seed={config['seed']}"
        return hashlib.sha256(material.encode()).hexdigest()


class ThesisDatasetAdapter:
    def __init__(self, definition, context: dict[str, Any]):
        self.definition = definition
        self.thesis_dir = Path(context["thesis_dir"])

    def load(self, config: dict[str, Any]) -> DatasetBundle:
        utils_path = self.thesis_dir / "utils.py"
        suffix = hashlib.sha256(str(utils_path).encode()).hexdigest()[:12]
        real_kg = load_source(utils_path, f"reprokge_dataset_utils_{suffix}").RealKG
        graph = real_kg(self.definition.plugin_id, data_dir=str(self.thesis_dir / "data"))
        source_files = [self.thesis_dir / "data" / self.definition.plugin_id / name for name in ("train.txt", "valid.txt", "test.txt")]
        train, validation, test = list(graph.train_triples), list(graph.val_triples), list(graph.test_triples)
        return DatasetBundle(
            splits={"train": train, "validation": validation, "test": test},
            features=set(self.definition.manifest.get("features", [])),
            metadata={"entities": len(graph.entity_to_id), "relations": len(graph.relation_to_id)},
            mappings={"entity_to_id": graph.entity_to_id, "relation_to_id": graph.relation_to_id},
            known_triples=set(train + validation + test),
            source_files=source_files,
        )

    def fingerprint(self, config: dict[str, Any], bundle: DatasetBundle) -> str:
        del config
        digest = hashlib.sha256()
        for path in bundle.source_files:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        return digest.hexdigest()

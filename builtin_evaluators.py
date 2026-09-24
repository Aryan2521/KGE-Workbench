"""Built-in evaluator adapters composed by the experiment runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class BaseEvaluator:
    def __init__(self, definition, context: dict[str, Any]):
        self.definition = definition
        self.context = context

    def source_files(self) -> list[Path]:
        return [Path(__file__).resolve()]


class LinkPredictionEvaluator(BaseEvaluator):
    def evaluate(self, model, model_adapter, dataset, evaluation, config, device):
        from benchmark_runner import ranking_metrics

        return ranking_metrics(
            model,
            evaluation["test"],
            dataset.known_triples,
            dataset.num_entities,
            device,
            config["candidate_chunk_size"],
            model_adapter,
        )


class TripleClassificationEvaluator(BaseEvaluator):
    def evaluate(self, model, model_adapter, dataset, evaluation, config, device):
        del dataset, config
        from benchmark_runner import classification_metrics

        return classification_metrics(
            model,
            evaluation["validation"],
            evaluation["validation_negatives"],
            evaluation["test"],
            evaluation["test_negatives"],
            device,
            model_adapter,
        )


class CalibrationEvaluator(BaseEvaluator):
    def evaluate(self, model, model_adapter, dataset, evaluation, config, device):
        del dataset
        from benchmark_runner import calibration_metrics

        return calibration_metrics(
            model,
            evaluation["test"],
            evaluation["test_negatives"],
            config,
            device,
            model_adapter,
        )

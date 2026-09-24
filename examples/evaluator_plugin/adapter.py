"""Starting point for a supervisor-supplied evaluator or benchmark."""

from pathlib import Path


class EvaluatorAdapter:
    def __init__(self, definition, context):
        self.definition = definition
        self.context = context

    def source_files(self) -> list[Path]:
        return [Path(__file__).resolve()]

    def evaluate(self, model, model_adapter, dataset, evaluation, config, device):
        del model, model_adapter, dataset, evaluation, config, device
        # Return exactly the metric IDs declared in manifest.json.
        raise NotImplementedError("Calculate the benchmark metrics here")

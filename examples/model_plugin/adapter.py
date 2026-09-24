"""Starting point for a supervisor-supplied model adapter."""

from pathlib import Path

import torch


class ModelAdapter:
    def __init__(self, definition, context):
        self.definition = definition
        self.context = context
        self.score_direction = definition.manifest["score_direction"]
        self.capabilities = definition.manifest["capabilities"]

    def source_files(self) -> list[Path]:
        return [Path(__file__).resolve()]

    def build(self, dataset, config, device):
        raise NotImplementedError("Construct your torch.nn.Module here")

    def create_optimizer(self, model, config):
        return torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    def train_batch(self, model, positive, negative, positive_labels, negative_labels, optimizer):
        del positive_labels, negative_labels
        optimizer.zero_grad()
        positive_scores = self.score(model, *positive)
        negative_scores = self.score(model, *negative)
        loss = torch.nn.functional.softplus(negative_scores - positive_scores).mean()
        loss.backward()
        optimizer.step()
        return float(loss.item())

    # Optional: implement train(model, dataset, config, device) when the model
    # needs a task-specific loop instead of ReproKGE's triple-training loop.

    def score(self, model, heads, relations, tails):
        return model(heads, relations, tails)

    def save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path)

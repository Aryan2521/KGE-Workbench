"""Structured runner for models and datasets discovered through ReproKGE plugins."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

from plugin_registry import registry


def log(message: str) -> None:
    print(message, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_split(
    triples: list[tuple[int, int, int]], fraction: float, size: int, seed: int
):
    """Select within one official split without mutating it or global RNG state."""
    if not triples:
        raise ValueError("Cannot benchmark an empty official split")
    count = max(1, math.ceil(len(triples) * fraction))
    if size > 0:
        count = min(count, size)
    if count >= len(triples):
        return list(triples)
    indices = sorted(random.Random(seed).sample(range(len(triples)), count))
    return [triples[index] for index in indices]


def corrupt_triples(
    triples: list[tuple[int, int, int]],
    num_entities: int,
    known: set[tuple[int, int, int]],
    rng: random.Random,
) -> list[tuple[int, int, int]]:
    negatives = []
    for h, r, t in triples:
        for _ in range(100):
            candidate = (rng.randrange(num_entities), r, t) if rng.random() < 0.5 else (h, r, rng.randrange(num_entities))
            if candidate not in known:
                negatives.append(candidate)
                break
        else:
            raise RuntimeError("Could not generate a negative triple after 100 attempts")
    return negatives


def uncertainty_values(config: dict) -> tuple[float, float, float]:
    level = config["uncertainty_level"]
    presets = {
        "low": (0.95, 0.05, 0.0),
        "medium": (0.80, 0.20, 0.15),
        "high": (0.60, 0.40, 0.35),
    }
    if level in presets:
        return presets[level]
    return config["soft_label_pos"], config["soft_label_neg"], config["noise_level"]


def fuzzy_labels(size: int, positive: bool, config: dict, device: torch.device) -> torch.Tensor:
    pos_value, neg_value, noise = uncertainty_values(config)
    labels = torch.full((size,), pos_value if positive else neg_value, device=device)
    if noise > 0:
        mask = torch.rand(size, device=device) < noise
        labels[mask] = torch.rand(int(mask.sum().item()), device=device)
    return labels


def as_tensors(triples: list[tuple[int, int, int]], device: torch.device):
    return tuple(
        torch.tensor([triple[index] for triple in triples], dtype=torch.long, device=device)
        for index in range(3)
    )


def train_model(model, train: list[tuple[int, int, int]], known: set, config: dict, device: torch.device, adapter=None, num_entities: int | None = None) -> float:
    optimizer = adapter.create_optimizer(model, config) if adapter else torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    rng = random.Random(config["seed"] + 11)
    start = time.perf_counter()
    epochs = config["epochs"]
    for epoch in range(1, epochs + 1):
        rng.shuffle(train)
        losses = []
        model.train()
        for offset in range(0, len(train), config["batch_size"]):
            positives = train[offset : offset + config["batch_size"]]
            if not positives:
                continue
            entity_count = num_entities if num_entities is not None else model.num_entities
            negatives = corrupt_triples(positives, entity_count, known, rng)
            pos_h, pos_r, pos_t = as_tensors(positives, device)
            neg_h, neg_r, neg_t = as_tensors(negatives, device)
            positive_tensors = (pos_h, pos_r, pos_t)
            negative_tensors = (neg_h, neg_r, neg_t)
            positive_labels = fuzzy_labels(len(positives), True, config, device)
            negative_labels = fuzzy_labels(len(negatives), False, config, device)
            if adapter:
                loss = adapter.train_batch(
                    model, positive_tensors, negative_tensors, positive_labels, negative_labels, optimizer
                )
            else:
                loss = model.train_step(
                    *positive_tensors, *negative_tensors, positive_labels, negative_labels, optimizer
                )
            losses.append(float(loss))
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            average = float(np.mean(losses)) if losses else math.nan
            log(f"Epoch {epoch}/{epochs} | Loss: {average:.6f}")
    return time.perf_counter() - start


def positive_indexes(known: set[tuple[int, int, int]]):
    tails: dict[tuple[int, int], set[int]] = defaultdict(set)
    heads: dict[tuple[int, int], set[int]] = defaultdict(set)
    for h, r, t in known:
        tails[(h, r)].add(t)
        heads[(r, t)].add(h)
    return tails, heads


def candidate_rank(
    model,
    triple: tuple[int, int, int],
    replace_head: bool,
    positives: set[int],
    num_entities: int,
    device: torch.device,
    chunk_size: int,
    adapter=None,
) -> int:
    h, r, t = triple
    higher_is_better = bool(adapter and adapter.score_direction == "higher")
    score = adapter.score if adapter else lambda current_model, heads, relations, tails: current_model(heads, relations, tails)
    with torch.no_grad():
        target = float(score(model, *as_tensors([triple], device))[0].item())
        better = 0
        target_entity = h if replace_head else t
        for start in range(0, num_entities, chunk_size):
            stop = min(num_entities, start + chunk_size)
            candidates = torch.arange(start, stop, dtype=torch.long, device=device)
            relations = torch.full_like(candidates, r)
            if replace_head:
                scores = score(model, candidates, relations, torch.full_like(candidates, t))
            else:
                scores = score(model, torch.full_like(candidates, h), relations, candidates)
            for positive in positives:
                if positive != target_entity and start <= positive < stop:
                    scores[positive - start] = -torch.inf if higher_is_better else torch.inf
            better += int(torch.sum(scores > target if higher_is_better else scores < target).item())
    return better + 1


def ranking_metrics(model, test: list, known: set, num_entities: int, device: torch.device, chunk_size: int, adapter=None):
    tails, heads = positive_indexes(known)
    ranks = []
    total = len(test)
    for index, triple in enumerate(test, start=1):
        h, r, t = triple
        ranks.append(candidate_rank(model, triple, False, tails[(h, r)], num_entities, device, chunk_size, adapter))
        ranks.append(candidate_rank(model, triple, True, heads[(r, t)], num_entities, device, chunk_size, adapter))
        if index == 1 or index == total or index % max(1, total // 10) == 0:
            log(f"Ranking evaluation {index}/{total}")
    values = np.asarray(ranks, dtype=float)
    return {
        "mean_rank": float(values.mean()),
        "mrr": float(np.mean(1.0 / values)),
        "hits_at_1": float(np.mean(values <= 1)),
        "hits_at_3": float(np.mean(values <= 3)),
        "hits_at_10": float(np.mean(values <= 10)),
    }


def raw_scores(model, triples: list, device: torch.device, batch_size: int = 4096, adapter=None) -> np.ndarray:
    values = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(triples), batch_size):
            batch = triples[offset : offset + batch_size]
            tensors = as_tensors(batch, device)
            score = adapter.score(model, *tensors) if adapter else model(*tensors)
            values.append(score.detach().cpu().numpy())
    return np.concatenate(values) if values else np.asarray([], dtype=float)


def distances(model, triples: list, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    """Backward-compatible alias for the original lower-is-better metric tests."""
    return raw_scores(model, triples, device, batch_size)


def best_threshold(positive: np.ndarray, negative: np.ndarray, higher_is_better: bool = False) -> float:
    combined = np.concatenate([positive, negative])
    if not len(combined):
        return 0.0
    candidates = np.unique(np.quantile(combined, np.linspace(0, 1, min(201, len(combined)))))
    best = (-1.0, float(candidates[0]))
    for threshold in candidates:
        positive_prediction = positive >= threshold if higher_is_better else positive <= threshold
        negative_prediction = negative >= threshold if higher_is_better else negative <= threshold
        tp = int(np.sum(positive_prediction))
        fp = int(np.sum(negative_prediction))
        fn = int(np.sum(~positive_prediction))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best[0]:
            best = (f1, float(threshold))
    return best[1]


def classification_metrics(model, valid_pos, valid_neg, test_pos, test_neg, device, adapter=None):
    higher_is_better = bool(adapter and adapter.score_direction == "higher")
    threshold = best_threshold(
        raw_scores(model, valid_pos, device, adapter=adapter),
        raw_scores(model, valid_neg, device, adapter=adapter),
        higher_is_better,
    )
    positive = raw_scores(model, test_pos, device, adapter=adapter)
    negative = raw_scores(model, test_neg, device, adapter=adapter)
    positive_prediction = positive >= threshold if higher_is_better else positive <= threshold
    negative_prediction = negative >= threshold if higher_is_better else negative <= threshold
    tp = int(np.sum(positive_prediction))
    fn = int(np.sum(~positive_prediction))
    fp = int(np.sum(negative_prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1_score": f1}


def confidence_scores(model, triples: list, config: dict, device: torch.device, adapter=None) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for offset in range(0, len(triples), 4096):
            batch = triples[offset : offset + 4096]
            h, r, t = as_tensors(batch, device)
            if adapter:
                score = adapter.confidence(model, h, r, t, config)
            elif hasattr(model, "fuzzy_score"):
                score = model.fuzzy_score(h, r, t)
            else:
                score = torch.exp(-config["alpha"] * model(h, r, t))
            outputs.append(score.detach().cpu().numpy())
    return np.clip(np.concatenate(outputs), 1e-7, 1 - 1e-7)


def calibration_metrics(model, test_pos, test_neg, config: dict, device: torch.device, adapter=None):
    predictions = np.concatenate(
        [confidence_scores(model, test_pos, config, device, adapter), confidence_scores(model, test_neg, config, device, adapter)]
    )
    pos_value, neg_value, noise = uncertainty_values(config)
    rng = np.random.default_rng(config["seed"] + 97)
    targets = np.concatenate(
        [np.full(len(test_pos), pos_value), np.full(len(test_neg), neg_value)]
    )
    if noise > 0:
        mask = rng.random(len(targets)) < noise
        targets[mask] = rng.random(int(mask.sum()))
    errors = predictions - targets
    ece = 0.0
    boundaries = np.linspace(0, 1, 11)
    for index in range(10):
        selected = (predictions >= boundaries[index]) & (predictions < boundaries[index + 1])
        if index == 9:
            selected |= predictions == 1.0
        if np.any(selected):
            ece += float(np.mean(selected) * abs(np.mean(predictions[selected]) - np.mean(targets[selected])))
    nll = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    mse = float(np.mean(errors**2))
    return {
        "mse": mse,
        "mae": float(np.mean(np.abs(errors))),
        "ece": ece,
        "brier": mse,
        "nll": float(nll),
    }


def run(config: dict, artifact_dir: Path | None = None) -> dict:
    started = time.perf_counter()
    thesis_dir = Path(config["thesis_dir"]).expanduser().resolve()
    model_plugin = registry.get("models", config["model_type"])
    dataset_plugin = registry.get("datasets", config["dataset"])
    evaluator_plugins = [registry.get("evaluators", evaluator_id) for evaluator_id in config["evaluators"]]
    registry.validate_compatibility(model_plugin, dataset_plugin, evaluator_plugins)
    context = {"thesis_dir": thesis_dir}
    model_adapter = model_plugin.create_adapter(context)
    dataset_adapter = dataset_plugin.create_adapter(context)
    set_seed(config["seed"])
    device = select_device()
    log(f"Device: {device}")
    log(f"Model plugin: {model_plugin.plugin_id} v{model_plugin.manifest['version']}")
    log(f"Dataset plugin: {dataset_plugin.plugin_id} v{dataset_plugin.manifest['version']}")
    log(f"Evaluators: {', '.join(plugin.plugin_id for plugin in evaluator_plugins)}")
    log(f"Thesis source: {thesis_dir}")

    bundle = dataset_adapter.load(config)
    is_triple_dataset = "triples" in bundle.features
    train = list(bundle.train) if is_triple_dataset else bundle.train
    valid = list(bundle.validation) if is_triple_dataset else bundle.validation
    test = list(bundle.test) if is_triple_dataset else bundle.test
    num_entities, num_relations = bundle.num_entities, bundle.num_relations
    split_sizes = {
        "train": len(train),
        "validation": len(valid),
        "test": len(test),
    }
    dataset_info = {
        **bundle.metadata,
        "entities": num_entities,
        "relations": num_relations,
        "split_sizes": split_sizes,
        "features": sorted(bundle.features),
    }
    if is_triple_dataset:
        dataset_info.update({
            "train_triples": split_sizes["train"],
            "validation_triples": split_sizes["validation"],
            "test_triples": split_sizes["test"],
            "total_triples": sum(split_sizes.values()),
        })
    log(f"Dataset {config['dataset']}: {num_entities} entities, {num_relations} relations")
    known = bundle.known_triples
    valid_neg, test_neg = [], []
    if is_triple_dataset:
        train = select_split(
            train, config["train_fraction"], config["train_sample_size"], config["seed"]
        )
        if not config["full_evaluation"]:
            validation_cap = config["validation_sample_size"] or config["eval_sample_size"]
            test_cap = config["test_sample_size"] or config["eval_sample_size"]
            valid = select_split(valid, config["validation_fraction"], validation_cap, config["seed"] + 1)
            test = select_split(test, config["test_fraction"], test_cap, config["seed"] + 2)
        valid_neg = corrupt_triples(valid, num_entities, known, random.Random(config["seed"] + 31))
        test_neg = corrupt_triples(test, num_entities, known, random.Random(config["seed"] + 41))
        log(f"Training triples: {len(train)}; evaluation triples: {len(test)}")
    else:
        log(f"Training items: {len(train)}; evaluation items: {len(test)}")

    model = model_adapter.build(bundle, config, device)
    custom_train = getattr(model_adapter, "train", None)
    if callable(custom_train):
        training_started = time.perf_counter()
        reported_seconds = custom_train(model, bundle, config, device)
        measured_seconds = time.perf_counter() - training_started
        train_seconds = float(reported_seconds) if reported_seconds is not None else measured_seconds
    elif is_triple_dataset:
        train_seconds = train_model(model, train, known, config, device, model_adapter, num_entities)
    else:
        raise ValueError(
            f"Model {model_plugin.plugin_id} must implement adapter.train() for non-triple dataset {dataset_plugin.plugin_id}"
        )
    log(f"Training completed in {train_seconds:.2f}s")
    checkpoint_name = None
    if artifact_dir and model_plugin.manifest.get("capabilities", {}).get("save_checkpoint"):
        checkpoint_name = "model.pt"
        model_adapter.save_checkpoint(model, artifact_dir / checkpoint_name)
        log(f"Checkpoint written: {checkpoint_name}")

    evaluation_started = time.perf_counter()
    metrics: dict[str, float] = {}
    metric_definitions: dict[str, dict] = {}
    executed_evaluators = []
    evaluation = {
        "train": train,
        "validation": valid,
        "test": test,
        "validation_negatives": valid_neg,
        "test_negatives": test_neg,
    }
    for evaluator_plugin in evaluator_plugins:
        evaluator = evaluator_plugin.create_adapter(context)
        log(f"Running evaluator: {evaluator_plugin.manifest['name']}")
        evaluator_metrics = evaluator.evaluate(model, model_adapter, bundle, evaluation, config, device)
        declared = evaluator_plugin.manifest.get("metrics", {})
        undeclared = sorted(set(evaluator_metrics) - set(declared))
        if undeclared:
            raise ValueError(
                f"Evaluator {evaluator_plugin.plugin_id} returned undeclared metrics: {', '.join(undeclared)}"
            )
        duplicates = sorted(set(metrics) & set(evaluator_metrics))
        if duplicates:
            raise ValueError(f"Duplicate metric IDs across evaluators: {', '.join(duplicates)}")
        numeric_metrics = {name: float(value) for name, value in evaluator_metrics.items()}
        non_finite = sorted(name for name, value in numeric_metrics.items() if not math.isfinite(value))
        if non_finite:
            raise ValueError(
                f"Evaluator {evaluator_plugin.plugin_id} returned non-finite metrics: {', '.join(non_finite)}"
            )
        metrics.update(numeric_metrics)
        for metric_id, definition in declared.items():
            if metric_id in evaluator_metrics:
                metric_definitions[metric_id] = {
                    **definition,
                    "evaluator_id": evaluator_plugin.plugin_id,
                    "evaluator_version": evaluator_plugin.manifest["version"],
                }
        executed_evaluators.append({
            "id": evaluator_plugin.plugin_id,
            "name": evaluator_plugin.manifest["name"],
            "version": evaluator_plugin.manifest["version"],
            "fingerprint": evaluator_plugin.fingerprint(evaluator.source_files()),
        })
    eval_seconds = time.perf_counter() - evaluation_started
    total_seconds = time.perf_counter() - started
    log(f"Evaluation completed in {eval_seconds:.2f}s")
    for name, value in metrics.items():
        log(f"METRIC {name}={value:.8f}")
    return {
        "metrics": metrics,
        "metric_definitions": metric_definitions,
        "device": str(device),
        "code_fingerprint": model_plugin.fingerprint(model_adapter.source_files()),
        "dataset_fingerprint": dataset_adapter.fingerprint(config, bundle),
        "model_plugin": {"id": model_plugin.plugin_id, "version": model_plugin.manifest["version"]},
        "dataset_plugin": {"id": dataset_plugin.plugin_id, "version": dataset_plugin.manifest["version"]},
        "evaluators": executed_evaluators,
        "evaluation_protocol_version": "reprokge-eval-v4-full-splits",
        "checkpoint": checkpoint_name,
        "dataset_info": dataset_info,
        "train_seconds": train_seconds,
        "eval_seconds": eval_seconds,
        "total_seconds": total_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result_path = Path(args.result)
    result = run(config, result_path.parent)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    log(f"Structured result written to {args.result}")


if __name__ == "__main__":
    main()

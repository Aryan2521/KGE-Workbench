"""Supervisor-facing utilities for installed ReproKGE plugins."""

from __future__ import annotations

import argparse
import json

from plugin_registry import PluginError, registry


def validate(kind: str, plugin_id: str) -> None:
    plugin = registry.get(kind, plugin_id)
    adapter_class = plugin.load_adapter_class()
    required_by_kind = {
        "models": ("build", "create_optimizer", "train_batch", "score", "source_files"),
        "datasets": ("load", "fingerprint"),
        "evaluators": ("evaluate", "source_files"),
    }
    required = required_by_kind[kind]
    missing = [name for name in required if not callable(getattr(adapter_class, name, None))]
    if kind == "models" and plugin.manifest.get("capabilities", {}).get("predict_confidence"):
        if not callable(getattr(adapter_class, "confidence", None)):
            missing.append("confidence")
    if kind == "models" and plugin.manifest.get("capabilities", {}).get("save_checkpoint"):
        if not callable(getattr(adapter_class, "save_checkpoint", None)):
            missing.append("save_checkpoint")
    if missing:
        raise PluginError(f"{plugin_id}: adapter is missing methods: {', '.join(missing)}")
    print(f"OK {kind[:-1]} {plugin.plugin_id} v{plugin.manifest['version']} -> {adapter_class.__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and validate trusted local ReproKGE plugins")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list-models")
    subparsers.add_parser("list-datasets")
    subparsers.add_parser("list-evaluators")
    for command in ("validate-model", "validate-dataset", "validate-evaluator"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("plugin_id")
    smoke = subparsers.add_parser("smoke-test")
    smoke.add_argument("--model", default="TransE")
    smoke.add_argument("--dataset", default="Synthetic")
    smoke.add_argument("--evaluators", nargs="+", default=["link_prediction", "triple_classification", "calibration"])
    args = parser.parse_args()
    try:
        if args.command == "list-models":
            print(json.dumps([plugin.public_manifest() for plugin in registry.models()], indent=2))
        elif args.command == "list-datasets":
            print(json.dumps([plugin.public_manifest() for plugin in registry.datasets()], indent=2))
        elif args.command == "list-evaluators":
            print(json.dumps([plugin.public_manifest() for plugin in registry.evaluators()], indent=2))
        elif args.command == "validate-model":
            validate("models", args.plugin_id)
        elif args.command == "validate-dataset":
            validate("datasets", args.plugin_id)
        elif args.command == "validate-evaluator":
            validate("evaluators", args.plugin_id)
        else:
            from app import validate_config
            from benchmark_runner import run

            config = validate_config({
                "experiment_name": "plugin contract smoke test",
                "model_type": args.model,
                "dataset": args.dataset,
                "evaluators": args.evaluators,
                "synthetic_families": 2,
                "epochs": 1,
                "embedding_dimension": 8,
                "batch_size": 32,
                "train_sample_size": 32,
                "eval_sample_size": 3,
            })
            result = run(config)
            print(json.dumps({
                "status": "ok",
                "model": args.model,
                "dataset": args.dataset,
                "evaluators": args.evaluators,
                "metrics": result["metrics"],
            }, indent=2))
    except PluginError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

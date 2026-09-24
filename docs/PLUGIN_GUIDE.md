# Extending ReproKGE

ReproKGE discovers reviewed local plugins when the server starts or `/api/plugins` is requested. Installing an extension means copying one directory into `plugins/models/`, `plugins/datasets/`, or `plugins/evaluators/`. The browser never accepts or executes uploaded Python.

## Add a model

1. Copy `examples/model_plugin` to `plugins/models/<your-model-id>`.
2. Update the manifest ID, name, version, description, capabilities, and parameters.
3. Implement the adapter methods in `adapter.py`.
4. Run `python manage.py validate-model <your-model-id>`.
5. Run a small compatible smoke test and restart the server.

The standard triple-training contract is:

- `build(dataset, config, device)` — construct and return the model.
- `create_optimizer(model, config)` — return the optimizer.
- `train_batch(model, positive, negative, positive_labels, negative_labels, optimizer)` — perform one update and return a finite scalar loss.
- `score(model, heads, relations, tails)` — return one raw score per triple.
- `source_files()` — return every source file that must be included in provenance.
- `save_checkpoint(model, path)` — required when `save_checkpoint` is declared.

For temporal, attributed, rule-based, or otherwise task-specific models, optionally implement `train(model, dataset, config, device)`. This replaces the standard triple-training loop and may return elapsed seconds or `None`. The evaluator still receives the trained model and complete dataset bundle.

When `predict_confidence` is declared, implement `confidence(...)` and return probabilities in `[0, 1]`. Set `score_direction` to `higher` or `lower`; compatible evaluators use that declaration rather than assuming a TransE distance.

## Add a dataset

1. Copy `examples/dataset_plugin` to `plugins/datasets/<your-dataset-id>`.
2. Put the data in that directory or deliberately reference a reviewed local source.
3. Declare the dataset's features and parameters in its manifest.
4. Implement `load(config)` and `fingerprint(config, bundle)`.
5. Run `python manage.py validate-dataset <your-dataset-id>`.

`load` returns a `builtin_plugins.DatasetBundle` with:

- `splits` — named training, validation, and test data.
- `features` — capabilities such as `triples`, `timestamps`, `confidence_labels`, or `text`.
- `metadata` — entity/relation counts and any useful dataset facts displayed with results.
- `mappings` — optional vocabularies and ID mappings.
- `known_triples` — required by filtered link prediction for triple datasets.
- `source_files` — files covered by the dataset fingerprint.

Standard KGE triple IDs must be contiguous integers starting at zero. Validation and test entities and relations should be represented in the training vocabulary. Other data shapes are allowed when the selected model supplies its own `train` method and the selected evaluators declare matching features.

## Add an evaluator or benchmark

1. Copy `examples/evaluator_plugin` to `plugins/evaluators/<your-evaluator-id>`.
2. Declare required model capabilities, required dataset features, metric IDs, labels, and direction.
3. Implement `evaluate(model, model_adapter, dataset, evaluation, config, device)`.
4. Run `python manage.py validate-evaluator <your-evaluator-id>`.
5. Run `python manage.py smoke-test --model <model> --dataset <dataset> --evaluators <your-evaluator-id>`.

The adapter returns a dictionary of finite numeric values. Every returned metric must be declared in the manifest. Metric IDs may not collide across evaluators selected for one run. No database or dashboard edit is needed: metric definitions, values, comparison rows, CSV columns, evaluator versions, evaluator fingerprints, and parameter controls are all dynamic.

The `evaluation` argument contains the resolved `train`, `validation`, and `test` data. For standard triples it also contains `validation_negatives` and `test_negatives`. An evaluator can instead use custom splits or metadata from `dataset`.

## Compatibility declarations

Evaluator manifests are the compatibility boundary:

```json
"required_model_capabilities": ["score_triples"],
"required_dataset_features": ["triples", "test_split"]
```

The experiment page shows only compatible evaluators. The server and runner repeat the check, so an incompatible combination cannot be started through a stale client or hand-edited configuration.

Examples:

- Link prediction requires triple scoring and entity/relation ID splits.
- Calibration requires a model that predicts confidence.
- A temporal benchmark can require `score_temporal` and `timestamps`.
- A confidence-labelled benchmark can require `predict_confidence` and `confidence_labels`.

## Manifest parameters

Model, dataset, and evaluator forms support `integer`, `number`, `string`, and `boolean`. Add `default`, `minimum`, `maximum`, `step`, or `enum` where appropriate. Values are validated again by the server.

```json
"parameters": {
  "dropout": {
    "label": "Dropout",
    "type": "number",
    "default": 0.2,
    "minimum": 0,
    "maximum": 1,
    "step": 0.05
  }
}
```

## Provenance and reproducibility

Every run records the model, dataset, and evaluator plugin versions and fingerprints; the evaluation protocol; the fully resolved configuration; dataset fingerprint and metadata; raw log; dynamic metrics; and, when supported, a model checkpoint. Increment a plugin version whenever behavior changes.

Useful commands:

```bash
python manage.py list-models
python manage.py list-datasets
python manage.py list-evaluators
python manage.py validate-model your_model_id
python manage.py validate-dataset your_dataset_id
python manage.py validate-evaluator your_evaluator_id
python manage.py smoke-test --model TransE --dataset Synthetic --evaluators link_prediction
```

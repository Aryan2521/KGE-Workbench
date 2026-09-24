# ReproKGE Evaluation Framework

ReproKGE is the bachelor-project evaluation layer that complements the uncertainty-aware KGE thesis implementation. The thesis owns the TransE and RFM-TransE research models; this project owns repeatable execution, corrected evaluation, provenance, experiment tracking, comparison, and export.

The dashboard never creates simulated metrics. Every completed row corresponds to a real training and evaluation process and retains its configuration and raw log in an artifact directory.

## Research question

How can deterministic and uncertainty-aware knowledge graph embedding models be compared fairly and reproducibly across link prediction, classification, calibration, robustness, and computational cost?

## Implemented scope

- A trusted local plugin registry for models, datasets, and evaluator/benchmark modules.
- Direct adapters for `TransE` and the newer `RFMTransE` class in `Thesis new`.
- Capability-aware evaluation for lower-is-better and higher-is-better model scores.
- Manifest-generated model and dataset parameter controls in the web interface.
- Capability/feature compatibility checks that show only valid model-dataset-evaluator combinations.
- Synthetic, CoDEx-S/M, FB15k, FB15k-237, WN18RR, and YAGO3-10 datasets already present in the thesis directory.
- Correct filtered MR, MRR, and Hits@1/3/10.
- Precision, recall, and F1 using a validation-derived threshold.
- MSE, MAE, ECE, Brier score, and soft-label negative log-likelihood.
- Fixed random seeds and explicit sampled/full evaluation modes.
- Configuration, thesis-code, and dataset SHA-256 fingerprints.
- Structured JSON configurations/results and raw run logs.
- SQLite experiment tracking, a dedicated visual comparison page for two to four runs, and CSV export.
- Live tracking refresh while jobs are active, with comparison selections preserved across status updates.
- Dynamic metric storage and comparison, so a new evaluator does not require a database or dashboard rewrite.
- Versioned evaluator provenance and optional trained-model checkpoints in run artifacts.
- Confirmed deletion of selected runs or all inactive runs, including their contained artifact folders; active jobs are protected.
- Dataset context calculated from the exact thesis split files, including entity, relation, and triple counts.
- Local-only web server. Arbitrary code execution from the browser has been removed.

## Important methodological note

The included benchmark datasets do not contain ground-truth fact confidence values. Calibration is therefore evaluated with a **controlled soft-label injection protocol**. Low, medium, and high settings alter positive/negative soft targets and inject a recorded amount of label noise. Results must be described as robustness/calibration under controlled uncertainty—not as performance against naturally observed confidence labels.

## Setup

Python 3.11 or newer is recommended. The current machine already has the required packages.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Point the application at the thesis code:

```bash
export KGE_THESIS_DIR="/absolute/path/to/Thesis new"
python app.py
```

Open `http://127.0.0.1:5001`. The server intentionally listens only on the local machine.

Optional paths:

```bash
export KGE_DATABASE_PATH="/absolute/path/to/benchmark_results.db"
export KGE_ARTIFACT_DIR="/absolute/path/to/artifacts"
```

If `KGE_THESIS_DIR` is not set, the application uses `~/Documents/Thesis/Thesis new`. It validates `transe.py`, `RFM_TransE.py`, and `utils.py` before accepting a run, and records that exact source path and its SHA-256 fingerprint with every experiment.

## Recommended evaluation procedure

1. Start with the Synthetic dataset and 5–10 epochs as a smoke test.
2. Run paired TransE and RFM-TransE configurations with the same dataset, seed, sample sizes, embedding dimension, margin, and training budget.
3. Repeat each configuration with at least three seeds.
4. Compare low, medium, and high controlled uncertainty separately.
5. Use full evaluation for final reported tables. Clearly label exploratory sampled runs.
6. Export the tracking table to CSV and retain each run's artifact directory.

For large datasets, full filtered ranking can take a long time because every test triple is scored against every entity. Use sampled evaluation while developing, then schedule final full runs deliberately.

## Project structure

```text
app.py                 Local Flask API and background run manager
benchmark_runner.py    Training, corrected metrics, and structured result writer
plugin_registry.py     Model/dataset/evaluator discovery, compatibility, schemas, and fingerprints
builtin_plugins.py     Adapters for Thesis new models and existing datasets
builtin_evaluators.py  Built-in link prediction, classification, and calibration adapters
plugins/models/        Installed trusted model plugins
plugins/datasets/      Installed trusted dataset plugins
plugins/evaluators/    Installed trusted benchmark/evaluator plugins
examples/              Copyable model, dataset, and evaluator plugin templates
docs/PLUGIN_GUIDE.md   Extension contract and installation instructions
manage.py              Plugin listing and validation commands
database.py            Reproducible SQLite run schema
templates/index.html   Dashboard
templates/compare.html Dedicated run comparison page
static/                 Browser logic and styling
tests/                  Unit and API tests
artifacts/run-*/        Per-run config, log, and result JSON
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover ranking metrics, threshold selection, uncertainty presets, configuration hashing and validation, plugin compatibility, arbitrary dynamic metrics, dataset-stat counting, comparison-page availability, database persistence, and API availability.

## Adding supervisor models, datasets, or benchmarks

Plugins are installed as reviewed local directories; the browser does not execute uploaded code. Start from the templates in `examples/`, follow `docs/PLUGIN_GUIDE.md`, and validate the result:

```bash
python manage.py list-models
python manage.py list-datasets
python manage.py list-evaluators
python manage.py validate-model your_model_id
python manage.py validate-dataset your_dataset_id
python manage.py validate-evaluator your_evaluator_id
```

The experiment form is generated from each manifest, and metrics are stored dynamically. A correctly installed plugin appears without changes to Flask, HTML, JavaScript, the runner, or the database. Models that do not use standard triple negative sampling can provide their own adapter-level training method. See `docs/PLUGIN_GUIDE.md` for the contracts and examples.

## What changed from the first prototype

- Removed generated/mock experiment results.
- Removed unsupported DistMult, ComplEx, probabilistic, and evidential UI claims.
- Removed browser-submitted arbitrary Python execution.
- Replaced regex parsing with a structured `result.json` contract.
- Stopped mapping Hits@5 to Hits@10 and stopped estimating Hits@1 from MRR.
- Added real provenance, timing, error states, and artifact retention.
- Added comparison warnings when selected runs use different protocols.

## Current limitations and next research extensions

- New model families still require a reviewed adapter implementing the documented training and evaluation capabilities.
- Rule-regularization experiments are not exposed in the first clean version.
- Jobs are local background processes, not a distributed queue.
- Natural-confidence evaluation requires a confidence-labelled dataset and evaluator plugin with a separately documented protocol.
- Statistical summaries across seed groups can be added after the final experiment matrix is agreed.

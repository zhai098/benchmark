# Benchmark Repository

This repository contains the benchmark pipeline, the annotation web app, evaluation and prompt-building utilities, and the datasets and generated artifacts used across those workflows.

## Top-Level Layout

- [annotation_app](annotation_app)
  Flask annotation backend, templates, static assets, persisted annotation data, and tests.
- [frontend](frontend)
  Next.js frontend shell and Playwright E2E tests.
- [benchmark_core](benchmark_core)
  Shared Python modules and canonical path definitions used by the main pipeline and moved scripts.
- [tools](tools)
  One-off and workflow-oriented scripts, grouped by function.
- [datasets](datasets)
  Stable source datasets and long-lived benchmark inputs.
- [workflow_data](workflow_data)
  Derived JSONL/JSON files grouped by workflow stage.
- [artifacts](artifacts)
  Analysis outputs, generated previews, archives, and model-output directories.
- [deploy](deploy)
  Deployment-related files.
- [scripts](scripts)
  Utility shell scripts for local development and service control.

Root-level Python entrypoints kept for compatibility:

- [main.py](main.py)
- [runner.py](runner.py)
- [generate.py](generate.py)
- [judge.py](judge.py)
- [run_log_aware_pipeline.py](run_log_aware_pipeline.py)

## Directory Guide

### `benchmark_core/`

Shared modules used across the repo:

- [config.py](benchmark_core/config.py)
- [data_process.py](benchmark_core/data_process.py)
- [log_reference.py](benchmark_core/log_reference.py)
- [metrics.py](benchmark_core/metrics.py)
- [paths.py](benchmark_core/paths.py)
- [prompt.py](benchmark_core/prompt.py)
- [utils.py](benchmark_core/utils.py)

`benchmark_core.paths` is the canonical place for repository-relative paths after the reorganization.

### `tools/`

Scripts are grouped by workflow instead of living at the repository root:

- [tools/claims](tools/claims)
  Claim segmentation, claim dependency packing/merging, and annotation-data builders.
- [tools/evaluation](tools/evaluation)
  Question-quality builders, strict-difficulty builders, scoring analysis, and summary visualizers.
- [tools/prompts](tools/prompts)
  Prompt packers, multisample-solution prompt builders, alternative-solution prompt builders, and solve-the-problem prompt builders.
- [tools/manual_review](tools/manual_review)
  Manual verification HTML/server tools.
- [tools/annotation_exports](tools/annotation_exports)
  Annotation-data export and merge helpers.
- [tools/prefix](tools/prefix)
  Prefix generation and inspection helpers.
- [tools/data](tools/data)
  Generic JSONL and utility scripts.
- [tools/analysis](tools/analysis)
  Miscellaneous analysis and experimentation scripts.

### `datasets/`

Stable source data:

- [datasets/omni_math/Omni_MATH](datasets/omni_math/Omni_MATH)
- [datasets/high_difficulty](datasets/high_difficulty)
- [datasets/sampled_cases](datasets/sampled_cases)

### `workflow_data/`

Workflow-specific derived files:

- [workflow_data/claims](workflow_data/claims)
- [workflow_data/question_quality](workflow_data/question_quality)
- [workflow_data/strict_difficulty](workflow_data/strict_difficulty)
- [workflow_data/annotation_exports](workflow_data/annotation_exports)
- [workflow_data/alt_solutions](workflow_data/alt_solutions)
- [workflow_data/manual_review](workflow_data/manual_review)
- [workflow_data/prefix](workflow_data/prefix)

### `artifacts/`

Generated outputs and previews:

- [artifacts/analysis](artifacts/analysis)
- [artifacts/model_outputs](artifacts/model_outputs)
- [artifacts/previews](artifacts/previews)
- [artifacts/archives](artifacts/archives)

## Annotation App

Backend:

```bash
python annotation_app/app.py
```

Backend tests:

```bash
pytest annotation_app/tests -q
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Production build:

```bash
cd frontend
npm run build
```

Playwright E2E:

```bash
cd frontend
npm run test:e2e
```

## Common Patterns After Reorganization

- If a script needs repository-relative defaults, prefer importing from [benchmark_core.paths](benchmark_core/paths.py).
- Stable inputs belong under `datasets/`.
- Intermediate workflow JSONL/JSON outputs belong under `workflow_data/`.
- Generated previews, analysis charts, archives, and model outputs belong under `artifacts/`.
- New one-off scripts should generally go under the appropriate `tools/<group>/` directory instead of the repository root.

## Notes

- Many historical files were moved out of the repository root. If you have old local commands that relied on root-level JSONL paths, update them to the new canonical locations or use the defaults now embedded in the moved scripts.
- The root directory is intentionally kept small: main apps, core entrypoints, and repository metadata only.

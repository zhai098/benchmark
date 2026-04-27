# Generate Pipeline Test Data

This directory contains the stable purified annotation input used by
`test/test_generate_pipeline.py` and by manual `generate.py` pipeline tests.

## Source

The backup files here were copied from:

```text
annotation_app/data/annotations/___/dev-1775126623662-xze9d4/workflow_outputs/purified/purified_cases.jsonl
annotation_app/data/annotations/___/dev-1775126623662-xze9d4/workflow_outputs/purified/purified_summary.json
```

Those purified rows were extracted from the annotation JSON files in:

```text
annotation_app/data/annotations/___/dev-1775126623662-xze9d4/
```

The extraction path is:

```text
annotation JSON logs -> benchmark_core.log_reference.purify_annotations_folder
                     -> workflow_outputs/purified/purified_cases.jsonl
```

The original benchmark problem text and standard solution metadata were joined
from `Config["Input_path"]`, which currently points to:

```text
datasets/omni_math/Omni_MATH/Omni_MATH_Human_Segmented_100_1.jsonl
```

## Contents

- `purified_cases.jsonl`: 10 purified cases. Eight have structured annotation
  reference steps, claims, and step dependencies.
- `purified_summary.json`: summary of the purification run.

## Quick Validation

Run the unit tests that check the `generate.py` path consumes annotation
reference steps rather than falling back to benchmark segments:

```bash
pytest test/test_generate_pipeline.py -q
```

## Manual Model Test Flow

Use this input for a small end-to-end generation test:

```bash
python generate.py \
  --input_path generate_pipeline_test_data/purified_cases.jsonl \
  --out_root runs/generate_pipeline_smoke \
  --tag MODEL_TAG \
  --max_cases 10 \
  --use_vllm_local
```

For API-backed generation, omit `--use_vllm_local`:

```bash
python generate.py \
  --input_path generate_pipeline_test_data/purified_cases.jsonl \
  --out_root runs/generate_pipeline_smoke \
  --tag MODEL_TAG \
  --max_cases 10
```

The generation file will be written under:

```text
runs/generate_pipeline_smoke/<model_name>_MODEL_TAG/gen_only.jsonl
```

Then run judging on that generation file:

```bash
python judge.py \
  --gen_file runs/generate_pipeline_smoke/<model_name>_MODEL_TAG/gen_only.jsonl \
  --run_dir runs/generate_pipeline_smoke/<model_name>_MODEL_TAG
```

If you only need packed judge prompts for a separate batch runner:

```bash
python tools/prompts/pack_prompt.py \
  --gen_file runs/generate_pipeline_smoke/<model_name>_MODEL_TAG/gen_only.jsonl \
  --out_dir runs/generate_pipeline_smoke/<model_name>_MODEL_TAG/packed_prompts \
  --max_cases 10 \
  --write_all
```

## Testing Different Models

Before each run, update `benchmark_core/config.py`:

- `reasoning_model`
- `reasoning_model_params`
- `reasoning_sampling_params`
- `reasoning_model_gpus`
- `tag`

Use a distinct `--tag` and/or `--out_root` for each model so outputs do not
overwrite each other.

Example:

```bash
python generate.py \
  --input_path generate_pipeline_test_data/purified_cases.jsonl \
  --out_root runs/mistral_small_4 \
  --tag mistral_small_4 \
  --max_cases 10 \
  --use_vllm_local

python generate.py \
  --input_path generate_pipeline_test_data/purified_cases.jsonl \
  --out_root runs/qwen_candidate \
  --tag qwen_candidate \
  --max_cases 10 \
  --use_vllm_local
```

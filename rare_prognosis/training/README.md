# RarePrognosis

## Overview

We provide the training pipeline for rare disease prognosis prediction, covering three sub-tasks: overall outcome, functional status, and symptom burden. For methodological details, please refer to the paper.

## Pipeline

```
Step 1: LLM Generation       →  prognosis_prediction_output.json per model per case
Step 2: Data Preparation     →  S1 CSVs, model directories, train/test splits
Step 3: Feature Engineering  →  features.{train,test}.csv per sub-task
Step 4: Training             →  GBDT stacking models (.pkl)
Step 5: Inference            →  Predictions written to S1 CSVs
```

## Quick Start

Generate the LLM predictions first (Step 1), then prepare data, build features,
train, and infer. Ground-truth labels come from each case's `prognosis_new.json`
under `--case-root`.

```bash
bash rare_prognosis/training/run_pipeline.sh \
    --python /path/to/python \
    --case-root /data/case_output \
    --llm-root /data/llm \
    --train-ids /data/splits/train.json \
    --test-ids /data/splits/test.json \
    --cv-folds 5
```

| Flag | Holds |
| --- | --- |
| `--case-root` | cases with GT labels (`<case>/prognosis_new.json`) |
| `--llm-root`  | per-model LLM predictions (`<model>/<case>/prognosis_prediction_output.json`) |



## Step-by-Step Usage

### Step 1: LLM Generation

[`generate_llm_outputs.py`](generate_llm_outputs.py) calls one model per invocation
and writes `<llm-root>/<model>/<case>/prognosis_prediction_output.json`.

The Prognosis ensemble uses these 12 model/output-directory identifiers:

`Claude-Haiku-4.5`, `DeepSeek-R1`, `DeepSeek-V3.2-exp`,
`Gemini-2.5-Flash`, `GPT-3.5-Turbo`, `GPT-4o-mini`, `GPT-5`, `o3-mini`,
`Qwen3-14B`, `Qwen3-235B-Instruct`, `Qwen3-32B`, and `Qwen3-8B`.

```bash
python -m rare_prognosis.training.generate_llm_outputs \
    /data/case_output /data/llm \
    --model MODEL_NAME \
    --config llm_config.json
```

Replace `MODEL_NAME` with each required ensemble identifier listed above. It
must match either a `model` value or an entry in `tags` in `llm_config.json`.

### Step 2: Data Preparation

[`prepare_data.py`](prepare_data.py) converts raw case outputs and LLM predictions into S1-format CSVs and the directory structure expected by downstream scripts.

```bash
python -m rare_prognosis.training.prepare_data \
    --case-root /data/case_output \
    --llm-root /data/llm \
    --out-dir /data/prepared
```

### Step 3: Feature Engineering

[`build_features.py`](build_features.py) constructs stacking features from multi-model outputs.

```bash
python -m rare_prognosis.training.build_features \
    --rareprognosis-root /data/prepared/rareprognosis \
    --models-root /data/prepared/models \
    --train-ids /data/prepared/dataset/train_case_ids.json \
    --test-ids /data/prepared/dataset/test_case_ids.json \
    --out-dir /data/prepared/features
```

### Step 4: Training

[`train_models.py`](train_models.py) trains a `GradientBoostingClassifier` per sub-task with 5-fold StratifiedKFold OOF evaluation. 

```bash
python -m rare_prognosis.training.train_models \
    --features-dir /data/prepared/features \
    --out-dir /data/prepared/trained_models \
    --seed 42 \
    --cv-folds 5
```

### Step 5: Inference

[`infer_models.py`](infer_models.py) loads trained model bundles, builds per-case features, and writes averaged ensemble predictions to S1 CSVs.

```bash
python -m rare_prognosis.training.infer_models \
    --rareprognosis-root /data/prepared/rareprognosis \
    --models-root /data/prepared/models \
    --train-ids /data/prepared/dataset/train_case_ids.json \
    --test-ids /data/prepared/dataset/test_case_ids.json \
    --models-dir /data/prepared/trained_models
```


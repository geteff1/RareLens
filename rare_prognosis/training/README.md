# RarePrognosis

## Overview

RarePrognosis asks multiple LLMs to predict three long-term prognosis categories and trains a Gradient Boosting stacking ensemble:

```text
case data
  -> multi-LLM prognosis generation
  -> categorical and explanation features
  -> stratified Gradient Boosting models
  -> fold-averaged inference
```

The three tasks are `overall_outcome`, `functional_status`, and `symptom_burden`.

## Code Demo

### 1. Configure LLM endpoints

Copy `llm_config.example.json` to `llm_config.json` and replace the endpoint and key placeholders. All LLM calls made by the demo scripts read model names, endpoints, and credentials from this JSON file.

The Prognosis ensemble uses these 12 model tags:

```text
Claude-Haiku-4.5, DeepSeek-R1, DeepSeek-V3.2-exp,
Gemini-2.5-Flash, GPT-3.5-Turbo, GPT-4o-mini, GPT-5, o3-mini,
Qwen3-14B, Qwen3-235B-Instruct, Qwen3-32B, Qwen3-8B
```

### 2. Run the 10-case demo

Run commands from the repository root. The scripts deterministically select 10 eligible cases from `data_500` with seed 42, use 8 for training and 2 for testing, generate fresh LLM predictions, and execute the complete training pipeline.

Windows PowerShell:

```powershell
.\rare_prognosis\training\reproduce_prognosis.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Models "Claude-Haiku-4.5,DeepSeek-R1,DeepSeek-V3.2-exp,Gemini-2.5-Flash,GPT-3.5-Turbo,GPT-4o-mini,GPT-5,o3-mini,Qwen3-14B,Qwen3-235B-Instruct,Qwen3-32B,Qwen3-8B"
```

macOS / Linux / WSL Bash:

Create the environment on that machine first; a `.venv` copied from Windows cannot run on macOS or Linux.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

bash rare_prognosis/training/reproduce_prognosis.sh \
  --python .venv/bin/python \
  --models Claude-Haiku-4.5,DeepSeek-R1,DeepSeek-V3.2-exp,Gemini-2.5-Flash,GPT-3.5-Turbo,GPT-4o-mini,GPT-5,o3-mini,Qwen3-14B,Qwen3-235B-Instruct,Qwen3-32B,Qwen3-8B
```


## Step-by-Step Reproduction

The commands below are the authoritative workflow for a user-provided cohort.

```text
<CASE_ROOT>        case directories containing prognosis inputs and labels
<SPLIT_ROOT>       train.json, test.json, and all.json
<LLM_OUTPUT_ROOT>  per-model generated prognosis JSON files
<FEATURE_ROOT>     generated feature CSVs
<MODEL_ROOT>       trained stacking model bundles
<RESULT_ROOT>      task-specific result.csv files
```

### Step 0: Prepare cases and splits

Every selected case must contain:

```text
<CASE_ROOT>/<case_id>/prognosis_prediction.json
<CASE_ROOT>/<case_id>/prognosis_new.json
```

`<SPLIT_ROOT>/train.json` and `test.json` must be disjoint JSON arrays of case IDs. Create `all.json` as their de-duplicated union. Use a case-level split so records from one patient cannot cross partitions.

### Step 1: Generate prognosis predictions

Run once per model tag. `--model` may be either a `model` value or a tag defined in `llm_config.json`; the endpoint and credentials are loaded from that file.

```bash
python -m rare_prognosis.training.generate_llm_outputs \
  <CASE_ROOT> <LLM_OUTPUT_ROOT> \
  --model MODEL_TAG \
  --config llm_config.json \
  --case-ids <SPLIT_ROOT>/all.json
```

Each result is written to `<LLM_OUTPUT_ROOT>/MODEL_TAG/<case_id>/prognosis_prediction_output.json`.

### Step 2: Prepare the data

```bash
python -m rare_prognosis.training.prepare_data \
  --case-root <CASE_ROOT> \
  --llm-root <LLM_OUTPUT_ROOT> \
  --result-root <RESULT_ROOT> \
  --train-ids <SPLIT_ROOT>/train.json \
  --test-ids <SPLIT_ROOT>/test.json
```

### Step 3: Build features

```bash
python -m rare_prognosis.training.build_features \
  --results-root <RESULT_ROOT> \
  --models-root <LLM_OUTPUT_ROOT> \
  --train-ids <SPLIT_ROOT>/train.json \
  --test-ids <SPLIT_ROOT>/test.json \
  --out-dir <FEATURE_ROOT>
```

### Step 4: Train the models

```bash
python -m rare_prognosis.training.train_models \
  --features-dir <FEATURE_ROOT> \
  --out-dir <MODEL_ROOT> \
  --seed 42 \
  --cv-folds 5
```

### Step 5: Run standalone inference

```bash
python -m rare_prognosis.training.infer_models \
  --results-root <RESULT_ROOT> \
  --models-root <LLM_OUTPUT_ROOT> \
  --train-ids <SPLIT_ROOT>/train.json \
  --test-ids <SPLIT_ROOT>/test.json \
  --models-dir <MODEL_ROOT>
```
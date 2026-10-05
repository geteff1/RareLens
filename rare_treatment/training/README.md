# RareTreatment

## Overview

RareTreatment generates treatment candidates with multiple LLMs and trains a five-fold XGBoost learning-to-rank ensemble.

```text
case data
  -> multi-LLM treatment generation
  -> external LLM-as-judge scores (method described in the paper)
  -> selected-case data preparation
  -> feature engineering
  -> five-fold XGBoost ranker
  -> fold-averaged test predictions
```

## Code Demo

### 1. Configure LLM endpoints

Configure `llm_config.json` following the format in `llm_config.example.json`, and replace the endpoint and key placeholders. Model generation reads this file. 

### 2. Run the 10-case demo

Run commands from the repository root. The Treatment ensemble uses these 12 models:

```text
Claude-Haiku-4.5, DeepSeek-R1, DeepSeek-V3.2-exp, Gemini-2.5-Flash,
GPT-3.5-Turbo, GPT-4o-mini, GPT-5, o3-mini, Qwen3-14B,
Qwen3-235B-Instruct, Qwen3-32B, Qwen3-8B
```

Windows PowerShell:

```powershell
.\rare_treatment\training\reproduce_treatment.ps1 `
  -Python .\.venv\Scripts\python.exe `
  -Models "Claude-Haiku-4.5,DeepSeek-R1,DeepSeek-V3.2-exp,Gemini-2.5-Flash,GPT-3.5-Turbo,GPT-4o-mini,GPT-5,o3-mini,Qwen3-14B,Qwen3-235B-Instruct,Qwen3-32B,Qwen3-8B"
```

macOS / Linux / WSL Bash:

Create the virtual environment on that machine; a Windows `.venv` cannot run on macOS or Linux.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

bash rare_treatment/training/reproduce_treatment.sh \
  --python .venv/bin/python \
  --models Claude-Haiku-4.5,DeepSeek-R1,DeepSeek-V3.2-exp,Gemini-2.5-Flash,GPT-3.5-Turbo,GPT-4o-mini,GPT-5,o3-mini,Qwen3-14B,Qwen3-235B-Instruct,Qwen3-32B,Qwen3-8B
```

The demo selects 10 eligible `data_500` cases deterministically with seed 42 and writes an 8-case training split and a disjoint 2-case test split.
Both scripts force Hugging Face, Transformers, and Datasets offline mode.


## Step-by-Step Reproduction

The following workflow is for a user-provided cohort.

```text
<CASE_ROOT>        source case directories
<SPLIT_ROOT>       train.json, test.json, and all.json
<LLM_OUTPUT_ROOT>  generated per-model treatment plans
<SCORE_ROOT>       externally generated per-model judge scores (not included)
<FEATURE_ROOT>     generated feature CSVs
<MODEL_ROOT>       trained fold models
<RESULT_ROOT>      ranked test predictions
```

### Step 0: Prepare cases and splits

Every selected case requires:

```text
<CASE_ROOT>/<case_id>/treatment_plan.json
<CASE_ROOT>/<case_id>/treatment_outcome.json
```

Optional reference material may be placed at:

```text
<CASE_ROOT>/<case_id>/treatment_knowledge.json
```

`train.json` and `test.json` must be disjoint JSON arrays of case IDs. `all.json` must be their de-duplicated union. At least five training cases are required by the default five-fold configuration.

### Step 1: Generate treatment candidates

Run once per model. `MODEL_TAG` may be a `model` value or an entry in `tags` in `llm_config.json`.

```bash
python -m rare_treatment.training.generate_llm_outputs \
  <CASE_ROOT> <LLM_OUTPUT_ROOT> \
  --model MODEL_TAG \
  --config llm_config.json \
  --case-ids <SPLIT_ROOT>/all.json
```

### Step 2: Provide judge scores

The LLM-as-judge implementation and prompt are intentionally not included in this repository. Reproduce the evaluation procedure specified in the paper for every generated model and case, then provide its JSON output under this contract:

```text
<SCORE_ROOT>/<MODEL_TAG>/<case_id>/treatment_score.json
```

Conceptual placeholder only:

```text
for each model tag and case:
    prediction = read generated treatment plan
    reference = read treatment_outcome.json and optional treatment_knowledge.json
    score = paper_defined_llm_judge(prediction, reference)
    write score to the matching path above
```

The score JSON must follow the fields and semantics described in the paper because feature construction consumes those scores as supervision. 

### Step 3: Build features

```bash
python -m rare_treatment.training.build_features \
  --case_root <CASE_ROOT> \
  --llm_root <LLM_OUTPUT_ROOT> \
  --score_root <SCORE_ROOT> \
  --train_ids <SPLIT_ROOT>/train.json \
  --test_ids <SPLIT_ROOT>/test.json \
  --out_dir <FEATURE_ROOT> \
  --num_gpus 0
```

### Step 4: Train the ranker

```bash
python -m rare_treatment.training.train_ranker \
  --data-dir <FEATURE_ROOT> \
  --out-dir <MODEL_ROOT> \
  --results-dir <RESULT_ROOT> \
  --objective rank:ndcg \
  --n-splits 5 \
  --target-k 3 \
  --save-models \
  --force-cpu
```

### Step 5: Standalone inference

```bash
python -m rare_treatment.training.infer_ranker \
  --model-dir <MODEL_ROOT>/models \
  --test-csv <FEATURE_ROOT>/features_test.csv \
  --out-dir <RESULT_ROOT>/inference
```

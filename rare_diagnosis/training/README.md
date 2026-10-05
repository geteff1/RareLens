# RareDiagnosis

## Overview
RareDiagnosis generates diagnostic candidates with multiple LLMs and trains an XGBoost learning-to-rank ensemble for primary or follow-up visits.

```text
case data
  -> multi-LLM diagnosis generation
  -> optional local OrphaCode retrieval + gpt-5-nano disambiguation
  -> external LLM-as-judge scores (method described in the paper)
  -> feature engineering
  -> five-fold XGBoost ranker
  -> fold-averaged inference
```

## Code Demo
### 1. Configure LLM endpoints

Configure `llm_config.json` following the format in `llm_config.example.json`, and replace the endpoint and key placeholders. Model generation and optional OrphaCode disambiguation read this file.

### 2. Run a 10-case demo

Run commands from the repository root.

The Diagnosis ensemble uses the same 11 model tags for primary and follow-up:

```text
Claude-Haiku-4.5, DeepSeek-R1, Gemini-2.5-Flash, GPT-3.5-Turbo,
GPT-4o-mini, GPT-5, o3-mini, Qwen3-14B, Qwen3-235B-Instruct,
Qwen3-32B, Qwen3-8B
```

Windows PowerShell — primary:

```powershell
.\rare_diagnosis\training\reproduce_diag.ps1 `
  -VisitType primary `
  -Python .\.venv\Scripts\python.exe `
  -Models "Claude-Haiku-4.5,DeepSeek-R1,Gemini-2.5-Flash,GPT-3.5-Turbo,GPT-4o-mini,GPT-5,o3-mini,Qwen3-14B,Qwen3-235B-Instruct,Qwen3-32B,Qwen3-8B"
```

Windows PowerShell — follow-up:

```powershell
.\rare_diagnosis\training\reproduce_diag.ps1 `
  -VisitType followup `
  -Python .\.venv\Scripts\python.exe `
  -Models "Claude-Haiku-4.5,DeepSeek-R1,Gemini-2.5-Flash,GPT-3.5-Turbo,GPT-4o-mini,GPT-5,o3-mini,Qwen3-14B,Qwen3-235B-Instruct,Qwen3-32B,Qwen3-8B"
```

macOS / Linux / WSL Bash — primary:

Create the environment on that machine first; a `.venv` copied from Windows cannot run on macOS.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

bash rare_diagnosis/training/reproduce_diag.sh \
  --visit-type primary \
  --python .venv/bin/python \
  --models Claude-Haiku-4.5,DeepSeek-R1,Gemini-2.5-Flash,GPT-3.5-Turbo,GPT-4o-mini,GPT-5,o3-mini,Qwen3-14B,Qwen3-235B-Instruct,Qwen3-32B,Qwen3-8B
```

macOS / Linux / WSL Bash — follow-up:

```bash
bash rare_diagnosis/training/reproduce_diag.sh \
  --visit-type followup \
  --python .venv/bin/python \
  --models Claude-Haiku-4.5,DeepSeek-R1,Gemini-2.5-Flash,GPT-3.5-Turbo,GPT-4o-mini,GPT-5,o3-mini,Qwen3-14B,Qwen3-235B-Instruct,Qwen3-32B,Qwen3-8B
```

The default demo deterministically selects 10 eligible cases with seed 42 and saves an 8-case training split and 2-case test split. 

Both scripts set Hugging Face/Transformers offline mode and therefore use the local model cache.

## Step-by-Step Reproduction
The commands below are the authoritative workflow for a user-provided cohort.

```text
<CASE_ROOT>        case directories and diagnosis.json
<SPLIT_ROOT>       train.json, test.json, and all.json
<LLM_OUTPUT_ROOT>  per-model generated candidates
<SCORE_ROOT>       externally generated per-model judge scores (not included)
<FEATURE_ROOT>     generated feature CSVs
<MODEL_ROOT>       trained fold models and predictions
<RAG_VECTOR_CACHE_DIR>  optional prebuilt OrphaCode vector cache
```

### Step 0: Prepare cases and splits

Primary cases require:

```text
<CASE_ROOT>/<case_id>/primary_consultation.json
<CASE_ROOT>/<case_id>/diagnosis.json
```

Follow-up cases additionally require:

```text
<CASE_ROOT>/<case_id>/follow_up_consultation.json
```

`<SPLIT_ROOT>/train.json` and `test.json` must be disjoint JSON arrays of case IDs. Create `<SPLIT_ROOT>/all.json` as their de-duplicated union. Generation uses `all.json`, while feature construction uses the train and test files separately. Use a case-level split so records from one patient cannot cross partitions. At least five training cases are required by the default five-fold training configuration.

### Step 1: Generate diagnosis candidates

Run once per model. `--model` may be a `model` value or tag in `llm_config.json`; use these same 11 public tags for both visit stages:

```text
Claude-Haiku-4.5, DeepSeek-R1, Gemini-2.5-Flash, GPT-3.5-Turbo,
GPT-4o-mini, GPT-5, o3-mini, Qwen3-14B, Qwen3-235B-Instruct,
Qwen3-32B, Qwen3-8B
```

Primary command:

```bash
python -m rare_diagnosis.training.generate_llm_outputs \
  <CASE_ROOT> <LLM_OUTPUT_ROOT> \
  --model MODEL_TAG \
  --config llm_config.json \
  --case-ids <SPLIT_ROOT>/all.json \
  --visit-type primary \
  --enable-orphacode-rag \
  --rag-ontology-path rare_diagnosis/training/orphanet_hierarchy.json \
  --rag-vector-cache-dir <RAG_VECTOR_CACHE_DIR>
```

For follow-up reproduction, first generate primary outputs for the same model and IDs, then run:

```bash
python -m rare_diagnosis.training.generate_llm_outputs \
  <CASE_ROOT> <LLM_OUTPUT_ROOT> \
  --model MODEL_TAG \
  --config llm_config.json \
  --case-ids <SPLIT_ROOT>/all.json \
  --visit-type followup \
  --enable-orphacode-rag \
  --rag-ontology-path rare_diagnosis/training/orphanet_hierarchy.json \
  --rag-vector-cache-dir <RAG_VECTOR_CACHE_DIR>
```

### Step 2: Provide judge scores

The LLM-as-judge implementation and prompt are intentionally not included in this repository. Reproduce the evaluation procedure specified in the paper for every generated model and case, then provide its JSON output under the following contract:

```text
primary:  <SCORE_ROOT>/<MODEL_TAG>/<case_id>/primary_diagnosis_score.json
follow-up:<SCORE_ROOT>/<MODEL_TAG>/<case_id>/followup_diagnosis_score.json
```

Conceptual placeholder only:

```text
for each model tag and case:
    prediction = read generated diagnosis candidate
    reference = read case diagnosis.json
    score = paper_defined_llm_judge(prediction, reference)
    write score to the matching path above
```

The score JSON must follow the fields and semantics described in the paper because feature construction consumes those scores as supervision. 

### Step 3: Build features
Primary:

```bash
python -m rare_diagnosis.training.build_features_primary \
  --query_root <CASE_ROOT> \
  --primary_models_root <LLM_OUTPUT_ROOT> \
  --score_root <SCORE_ROOT> \
  --models MODEL_A,MODEL_B \
  --train_ids <SPLIT_ROOT>/train.json \
  --test_ids <SPLIT_ROOT>/test.json \
  --out_dir <FEATURE_ROOT>
```

Follow-up:

```bash
python -m rare_diagnosis.training.build_features_followup \
  --query_root <CASE_ROOT> \
  --primary_models_root <LLM_OUTPUT_ROOT> \
  --score_root <SCORE_ROOT> \
  --models MODEL_A,MODEL_B \
  --train_ids <SPLIT_ROOT>/train.json \
  --test_ids <SPLIT_ROOT>/test.json \
  --out_dir <FEATURE_ROOT>
```

### Step 4: Train the ranker

Primary:

```bash
python -m rare_diagnosis.training.train_ranker \
  --input-dir <FEATURE_ROOT> \
  --config rare_diagnosis/training/best_hyperopt_config_primary.json \
  --out-dir <MODEL_ROOT>
```

For follow-up, replace the config with `best_hyperopt_config_followup.json`.

### Step 5: Standalone inference

```bash
python -m rare_diagnosis.training.infer_ranker \
  --input-dir <FEATURE_ROOT> \
  --model-dir <MODEL_ROOT>/models \
  --config rare_diagnosis/training/best_hyperopt_config_primary.json \
  --out-dir <MODEL_ROOT>/inference
```

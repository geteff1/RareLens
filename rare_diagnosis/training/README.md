# RareDiagnosis

## Overview

We provide the training pipeline for rare disease diagnosis ranking, supporting both primary consultation and follow-up visit stages. For methodological details, please refer to the paper.

## Pipeline

```
Step 0: RAG Cache (optional)   →  FAISS vector index for OrphaCode resolution
Step 1: LLM Generation         →  per-model diagnosis outputs + OrphaCode mapping
Step 2: Feature Engineering     →  features.{train,test}.csv (56+ features per candidate)
Step 3: XGBoost Training        →  GroupKFold CV ranker (rank:ndcg)
```

## Quick Start

Generate the LLM outputs first (Step 1), then build features and train the ranker.
The per-model LLM outputs (`--llm-root`) and judge scores (`--score-root`, the
ground truth) must both exist before running the commands below.

```bash
# Primary stage
bash rare_diagnosis/training/reproduce_diag.sh \
    --python /path/to/python \
    --visit-type primary \
    --case-root /data/cases \
    --score-root /data/scores \
    --llm-root /data/llm_outputs \
    --train-ids /data/splits/train.json \
    --test-ids /data/splits/test.json

# Follow-up stage 
bash rare_diagnosis/training/reproduce_diag.sh \
    --visit-type followup \
    --case-root /data/cases \
    --score-root /data/scores \
    --llm-root /data/llm_outputs \
    --train-ids /data/splits/train.json \
    --test-ids /data/splits/test.json
```

| Flag | Holds |
| --- | --- |
| `--case-root`  | raw cases (`<case>/primary_consultation.json`) |
| `--llm-root`   | per-model LLM outputs (`<model>/<case>/…`) |
| `--score-root` | per-model judge scores (GT) |

> The judge scores (the ground truth used here) are produced by LLM-as-judge
> evaluation following the method described in the paper. The scoring code is not
> included in this repository — please refer to the paper to reproduce them.

The per-model output filename differs by stage and is handled automatically; override
with `--primary-fname` / `--gt-fname` if needed. Run feature extraction on CPU with
`--num-gpus 0` if GPU workers fail.

## Step-by-Step Usage

### Step 1: LLM Generation

[`generate_llm_outputs.py`](generate_llm_outputs.py) queries LLMs to produce top-5 diagnoses per case. Each diagnosis is optionally enriched with an OrphaCode via semantic retrieval ([`orphacode_rag.py`](orphacode_rag.py)).

The first two arguments are positional (`input_folder` `output_folder`), and `--model` runs **one** model per invocation — loop over models to produce the multi-LLM outputs.

The Diagnosis ensemble uses these 11 model/output-directory names:

`Claude-Haiku-4.5`, `DeepSeek-R1`, `Gemini-2.5-Flash`, `GPT-3.5-Turbo`,
`GPT-4o-mini`, `GPT-5`, `o3-mini`, `Qwen3-14B`, `Qwen3-235B-Instruct`,
`Qwen3-32B`, and `Qwen3-8B`.

```bash
python -m rare_diagnosis.training.generate_llm_outputs \
    /data/query /data/llm_outputs \
    --model MODEL_NAME \
    --config llm_config.json \
    --visit-type primary

# With OrphaCode RAG enrichment
python -m rare_diagnosis.training.generate_llm_outputs \
    /data/query /data/llm_outputs \
    --model MODEL_NAME \
    --config llm_config.json \
    --visit-type primary \
    --enable-orphacode-rag \
    --rag-ontology-path rare_diagnosis/training/orphanet_hierarchy.json
```

Replace `MODEL_NAME` with each required ensemble identifier listed above. It
must match either a `model` value or an entry in `tags` in `llm_config.json`.
Run the command separately for each required visit stage. Do not mix primary
and follow-up generation in the same output root because the generator's output
filenames overlap.

### Step 2: Feature Engineering

[`build_features_primary.py`](build_features_primary.py) and [`build_features_followup.py`](build_features_followup.py) construct ranking features per candidate from multi-model outputs.

> Downloads `pritamdeka/S-PubMedBert-MS-MARCO` (semantic features) from HuggingFace on
> first run — see the main README's *Feature-engineering models* note for the mirror /
> pre-cache and the `--num_gpus 0 --workers 2` CPU fallback if GPU workers crash.

```bash
# Primary stage
python -m rare_diagnosis.training.build_features_primary \
    --query_root /data/query \
    --primary_models_root /data/llm_outputs \
    --gt_root /data/scores \
    --train_ids /data/splits/train.json \
    --test_ids /data/splits/test.json \
    --out_dir /data/features/primary

# Follow-up stage (includes diagnostic test results)
python -m rare_diagnosis.training.build_features_followup \
    --query_root /data/query \
    --primary_models_root /data/llm_outputs \
    --gt_root /data/scores \
    --train_ids /data/splits/train.json \
    --test_ids /data/splits/test.json \
    --out_dir /data/features/followup
```

### Step 3: XGBoost Training

[`train_ranker.py`](train_ranker.py) trains an XGBoost LTR model with GroupKFold (5-fold) cross-validation. Monotonicity constraints are auto-inferred from feature names.

```bash
python -m rare_diagnosis.training.train_ranker \
    --input-dir /data/features/primary \
    --config rare_diagnosis/training/best_hyperopt_config_primary.json \
    --out-dir /data/models/primary \
    --use-gpu
```

Standalone inference with trained models:

```bash
python -m rare_diagnosis.training.infer_ranker \
    --input-dir /data/features/primary \
    --model-dir /data/models/primary/models \
    --config rare_diagnosis/training/best_hyperopt_config_primary.json \
    --out-dir /data/inference_output
```


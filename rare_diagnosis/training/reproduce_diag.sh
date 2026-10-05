#!/usr/bin/env bash
# End-to-end RareDiagnosis code demo on data_500.
# Creates a deterministic split, generates diagnoses, accepts externally supplied
# paper-defined judge scores, builds features, and trains the five-fold ranker.

set -euo pipefail

# Reuse the local Hugging Face model cache. SentenceTransformer otherwise sends
# online HEAD checks even for models that are already available locally.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

_pwd() { pwd -W 2>/dev/null || pwd; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && _pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && _pwd)"

PYTHON="${PYTHON:-python}"
VISIT_TYPE="primary"
MODELS_CSV=""
CASE_ROOT="${REPO_ROOT}/data_500"
CONFIG_PATH="${REPO_ROOT}/llm_config.json"
NUM_CASES=10
TRAIN_COUNT=8
SEED=42
TRAIN_IDS=""
TEST_IDS=""
OUTPUT_ROOT=""
SCORE_ROOT=""
NUM_WORKERS=4
FEATURE_WORKERS=2
FEATURE_GPUS=0
USE_RAG=1
USE_GPU=0

usage() {
    cat <<'EOF'
Usage: bash rare_diagnosis/training/reproduce_diag.sh --models MODEL[,MODEL...] [OPTIONS]

Required:
  --models LIST          Comma-separated llm_config.json tags (for example GPT-5,o3-mini)

Options:
  --visit-type TYPE     primary (default) or followup
  --case-root DIR       Case directory (default: data_500)
  --config PATH         LLM config (default: llm_config.json)
  --num-cases N         Auto-split case count (default: 10)
  --train-count N       Auto-split training count (default: 8)
  --seed N              Auto-split random seed (default: 42)
  --train-ids PATH      Existing train case-ID JSON; requires --test-ids
  --test-ids PATH       Existing test case-ID JSON; requires --train-ids
  --output-root DIR     Demo output directory
  --score-root DIR      External paper-defined judge-score directory
  --python PATH         Python interpreter (default: python)
  --num-workers N       Concurrent LLM requests per model (default: 4)
  --feature-workers N   Feature worker processes (default: 2)
  --feature-gpus N      Feature GPUs (default: 0)
  --no-rag              Disable OrphaCode RAG enrichment
  --use-gpu             Use GPU for XGBoost training
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --visit-type) VISIT_TYPE="$2"; shift 2 ;;
        --models) MODELS_CSV="$2"; shift 2 ;;
        --case-root) CASE_ROOT="$2"; shift 2 ;;
        --config) CONFIG_PATH="$2"; shift 2 ;;
        --num-cases) NUM_CASES="$2"; shift 2 ;;
        --train-count) TRAIN_COUNT="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --train-ids) TRAIN_IDS="$2"; shift 2 ;;
        --test-ids) TEST_IDS="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --score-root) SCORE_ROOT="$2"; shift 2 ;;
        --python) PYTHON="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --feature-workers) FEATURE_WORKERS="$2"; shift 2 ;;
        --feature-gpus) FEATURE_GPUS="$2"; shift 2 ;;
        --no-rag) USE_RAG=0; shift ;;
        --use-gpu) USE_GPU=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ "${VISIT_TYPE}" != "primary" && "${VISIT_TYPE}" != "followup" ]]; then
    echo "[ERROR] --visit-type must be primary or followup." >&2
    exit 1
fi
if [[ -z "${MODELS_CSV}" ]]; then
    echo "[ERROR] --models is required." >&2
    usage >&2
    exit 1
fi
if [[ ! -d "${CASE_ROOT}" || ! -f "${CONFIG_PATH}" ]]; then
    echo "[ERROR] Missing case root or LLM config." >&2
    exit 1
fi
if [[ -n "${TRAIN_IDS}" && -z "${TEST_IDS}" ]] || [[ -z "${TRAIN_IDS}" && -n "${TEST_IDS}" ]]; then
    echo "[ERROR] --train-ids and --test-ids must be provided together." >&2
    exit 1
fi
IFS=',' read -r -a MODELS <<< "${MODELS_CSV}"
"${PYTHON}" -c 'import json,sys; cfg=json.load(open(sys.argv[1],encoding="utf-8")); names={str(x.get("model")) for x in cfg}; names.update(t for x in cfg for t in x.get("tags",[])); missing=[x for x in sys.argv[2:] if x not in names]; assert not missing,f"models absent from config: {missing}"' "${CONFIG_PATH}" "${MODELS[@]}"

# Validate local dependencies before any paid LLM call.
"${PYTHON}" -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO", device="cpu"); print("Local feature semantic model ready.")'
RAG_CACHE_DIR="$(cd "${SCRIPT_DIR}/.." && _pwd)/orphacode_rag_cache"
if [[ "${USE_RAG}" -eq 1 ]]; then
    [[ -d "${RAG_CACHE_DIR}" ]] || { echo "[ERROR] OrphaCode RAG cache does not exist: ${RAG_CACHE_DIR}" >&2; exit 1; }
    "${PYTHON}" -c 'from sentence_transformers import CrossEncoder, SentenceTransformer; SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu"); CrossEncoder("ncbi/MedCPT-Cross-Encoder"); print("Local OrphaCode RAG models ready.")'
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/diagnosis_demo_${VISIT_TYPE}}"
SPLIT_ROOT="${OUTPUT_ROOT}/splits"
LLM_ROOT="${OUTPUT_ROOT}/llm_outputs"
SCORE_ROOT="${SCORE_ROOT:-${OUTPUT_ROOT}/judge_scores}"
FEATURE_ROOT="${OUTPUT_ROOT}/features"
MODEL_ROOT="${OUTPUT_ROOT}/model"
RESULT_ROOT="${OUTPUT_ROOT}/results"
mkdir -p "${SPLIT_ROOT}" "${LLM_ROOT}" "${FEATURE_ROOT}" "${MODEL_ROOT}" "${RESULT_ROOT}"

if [[ -z "${TRAIN_IDS}" ]]; then
    TRAIN_IDS="${SPLIT_ROOT}/train.json"
    TEST_IDS="${SPLIT_ROOT}/test.json"
    "${PYTHON}" -c 'import json,random,sys; from pathlib import Path; root=Path(sys.argv[1]); stage=sys.argv[2]; n=int(sys.argv[3]); nt=int(sys.argv[4]); seed=int(sys.argv[5]); out=Path(sys.argv[6]); req=["primary_consultation.json","diagnosis.json"]+(["follow_up_consultation.json"] if stage=="followup" else []); ids=sorted(p.name for p in root.iterdir() if p.is_dir() and all((p/f).is_file() for f in req)); random.Random(seed).shuffle(ids); assert 5<=nt<n<=len(ids),f"require 5 <= train-count < num-cases <= {len(ids)}"; chosen=ids[:n]; out.mkdir(parents=True,exist_ok=True); (out/"all.json").write_text(json.dumps(chosen,indent=2),encoding="utf-8"); (out/"train.json").write_text(json.dumps(chosen[:nt],indent=2),encoding="utf-8"); (out/"test.json").write_text(json.dumps(chosen[nt:],indent=2),encoding="utf-8")' "${CASE_ROOT}" "${VISIT_TYPE}" "${NUM_CASES}" "${TRAIN_COUNT}" "${SEED}" "${SPLIT_ROOT}"
else
    SOURCE_TRAIN_IDS="${TRAIN_IDS}"
    SOURCE_TEST_IDS="${TEST_IDS}"
    TRAIN_IDS="${SPLIT_ROOT}/train.json"
    TEST_IDS="${SPLIT_ROOT}/test.json"
    "${PYTHON}" -c 'import json,sys; from pathlib import Path; train=list(map(str,json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))); test=list(map(str,json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))); overlap=set(train)&set(test); assert len(train)>=5,"at least five training cases are required"; assert test,"test split must not be empty"; assert not overlap,f"train/test overlap: {sorted(overlap)}"; out=Path(sys.argv[3]); out.mkdir(parents=True,exist_ok=True); (out/"train.json").write_text(json.dumps(train,indent=2),encoding="utf-8"); (out/"test.json").write_text(json.dumps(test,indent=2),encoding="utf-8"); (out/"all.json").write_text(json.dumps(train+test,indent=2),encoding="utf-8")' "${SOURCE_TRAIN_IDS}" "${SOURCE_TEST_IDS}" "${SPLIT_ROOT}"
fi
ALL_IDS="${SPLIT_ROOT}/all.json"
"${PYTHON}" -c 'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); stage=sys.argv[2]; ids=list(map(str,json.loads(Path(sys.argv[3]).read_text(encoding="utf-8")))); req=["primary_consultation.json","diagnosis.json"]+(["follow_up_consultation.json"] if stage=="followup" else []); bad=[cid for cid in ids if not all((root/cid/f).is_file() for f in req)]; assert not bad,f"ineligible or missing cases: {bad}"' "${CASE_ROOT}" "${VISIT_TYPE}" "${ALL_IDS}"

echo "RareDiagnosis code demo"
echo "  stage:       ${VISIT_TYPE}"
echo "  cases:       ${ALL_IDS}"
echo "  train/test:  ${TRAIN_IDS} / ${TEST_IDS}"
echo "  models:      ${MODELS_CSV}"
echo "  output:      ${OUTPUT_ROOT}"

RAG_ARGS=()
if [[ "${USE_RAG}" -eq 1 ]]; then
    RAG_ARGS=(--enable-orphacode-rag --rag-ontology-path "${SCRIPT_DIR}/orphanet_hierarchy.json" --rag-vector-cache-dir "${RAG_CACHE_DIR}")
fi

for MODEL_TAG in "${MODELS[@]}"; do
    if [[ "${VISIT_TYPE}" == "followup" ]]; then
        "${PYTHON}" -m rare_diagnosis.training.generate_llm_outputs \
            "${CASE_ROOT}" "${LLM_ROOT}" --model "${MODEL_TAG}" \
            --config "${CONFIG_PATH}" --case-ids "${ALL_IDS}" \
            --visit-type primary --num-workers "${NUM_WORKERS}" "${RAG_ARGS[@]}"
    fi

    "${PYTHON}" -m rare_diagnosis.training.generate_llm_outputs \
        "${CASE_ROOT}" "${LLM_ROOT}" --model "${MODEL_TAG}" \
        --config "${CONFIG_PATH}" --case-ids "${ALL_IDS}" \
        --visit-type "${VISIT_TYPE}" --num-workers "${NUM_WORKERS}" "${RAG_ARGS[@]}"

done

# The LLM-as-judge implementation is intentionally not distributed. Reproduce
# the paper's evaluation procedure, then provide the score files and rerun.
# Candidate generation is resumable, so existing outputs are reused.
SCORE_FILE="primary_diagnosis_score.json"
[[ "${VISIT_TYPE}" == "followup" ]] && SCORE_FILE="followup_diagnosis_score.json"
MISSING_SCORES="$("${PYTHON}" -c 'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); ids=json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")); fname=sys.argv[3]; models=sys.argv[4:]; missing=[str(root/m/c/fname) for m in models for c in ids if not (root/m/c/fname).is_file()]; print("\n".join(missing))' "${SCORE_ROOT}" "${ALL_IDS}" "${SCORE_FILE}" "${MODELS[@]}")"
if [[ -n "${MISSING_SCORES}" ]]; then
    echo "[ERROR] LLM-as-judge code is not included. Reproduce the judge from the paper and provide score JSONs before feature construction." >&2
    echo "[ERROR] Expected e.g.: $(head -n 1 <<< "${MISSING_SCORES}")" >&2
    exit 1
fi

if [[ "${VISIT_TYPE}" == "primary" ]]; then
    FEATURE_MODULE="rare_diagnosis.training.build_features_primary"
    BEST_CONFIG="${SCRIPT_DIR}/best_hyperopt_config_primary.json"
    EXTRA_FEATURE_ARGS=()
else
    FEATURE_MODULE="rare_diagnosis.training.build_features_followup"
    BEST_CONFIG="${SCRIPT_DIR}/best_hyperopt_config_followup.json"
    EXTRA_FEATURE_ARGS=(--followup_fname followup_diagnosis_orphacode.json)
fi

"${PYTHON}" -m "${FEATURE_MODULE}" \
    --query_root "${CASE_ROOT}" --primary_models_root "${LLM_ROOT}" \
    --score_root "${SCORE_ROOT}" --train_ids "${TRAIN_IDS}" \
    --test_ids "${TEST_IDS}" --out_dir "${FEATURE_ROOT}" \
    --models "${MODELS_CSV}" \
    --ontology_path "${SCRIPT_DIR}/orphanet_hierarchy.json" \
    --num_gpus "${FEATURE_GPUS}" --workers "${FEATURE_WORKERS}" \
    "${EXTRA_FEATURE_ARGS[@]}"

TRAIN_GPU_ARGS=()
if [[ "${USE_GPU}" -eq 1 ]]; then
    TRAIN_GPU_ARGS=(--use-gpu)
fi
"${PYTHON}" -m rare_diagnosis.training.train_ranker \
    --input-dir "${FEATURE_ROOT}" --config "${BEST_CONFIG}" \
    --out-dir "${MODEL_ROOT}" --results-dir "${RESULT_ROOT}" "${TRAIN_GPU_ARGS[@]}"

echo "Demo complete: ${OUTPUT_ROOT}"

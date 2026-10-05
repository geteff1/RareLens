#!/usr/bin/env bash
# RareTreatment demo: generation plus externally supplied paper-defined scores.

set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

_pwd() { pwd -W 2>/dev/null || pwd; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && _pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && _pwd)"

PYTHON="${PYTHON:-python}"
MODELS_CSV=""
CASE_ROOT="${REPO_ROOT}/data_500"
CONFIG_PATH="${REPO_ROOT}/llm_config.json"
NUM_CASES=10
TRAIN_COUNT=8
SEED=42
TRAIN_IDS=""
TEST_IDS=""
OUTPUT_ROOT="${REPO_ROOT}/outputs/treatment_demo"
SCORE_ROOT=""
NUM_WORKERS=4
FEATURE_GPUS=0
USE_GPU=0

usage() {
    cat <<'EOF'
Usage: bash rare_treatment/training/reproduce_treatment.sh --models MODEL[,MODEL...] [OPTIONS]

Required:
  --models LIST          Comma-separated llm_config.json model tags

Options:
  --case-root DIR        Case directory (default: data_500)
  --config PATH          LLM config (default: llm_config.json)
  --num-cases N          Auto-split case count (default: 10)
  --train-count N        Auto-split training count (default: 8)
  --seed N               Auto-split random seed (default: 42)
  --train-ids PATH       Existing train case-ID JSON; requires --test-ids
  --test-ids PATH        Existing test case-ID JSON; requires --train-ids
  --output-root DIR      Demo output directory
  --score-root DIR       External paper-defined judge-score directory
  --python PATH          Python interpreter (default: python)
  --num-workers N        Concurrent generation/judge requests (default: 4)
  --feature-gpus N       Feature GPUs (default: 0)
  --use-gpu              Allow XGBoost GPU training
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
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
        --feature-gpus) FEATURE_GPUS="$2"; shift 2 ;;
        --use-gpu) USE_GPU=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

[[ -n "${MODELS_CSV}" ]] || { echo "[ERROR] --models is required." >&2; exit 1; }
[[ -d "${CASE_ROOT}" ]] || { echo "[ERROR] Case root does not exist: ${CASE_ROOT}" >&2; exit 1; }
[[ -f "${CONFIG_PATH}" ]] || { echo "[ERROR] LLM config does not exist: ${CONFIG_PATH}" >&2; exit 1; }
if [[ -n "${TRAIN_IDS}" && -z "${TEST_IDS}" ]] || [[ -z "${TRAIN_IDS}" && -n "${TEST_IDS}" ]]; then
    echo "[ERROR] --train-ids and --test-ids must be provided together." >&2
    exit 1
fi

IFS=',' read -r -a MODELS <<< "${MODELS_CSV}"
"${PYTHON}" -c 'import json,sys; cfg=json.load(open(sys.argv[1],encoding="utf-8")); names={str(x.get("model")) for x in cfg}; names.update(t for x in cfg for t in x.get("tags",[])); missing=[x for x in sys.argv[2:] if x not in names]; assert not missing,f"models absent from config: {missing}"' "${CONFIG_PATH}" "${MODELS[@]}"

# Fail before paid API calls if the local feature models are unavailable.
"${PYTHON}" -c 'from huggingface_hub import snapshot_download; from sentence_transformers import SentenceTransformer,CrossEncoder; s=snapshot_download("pritamdeka/S-PubMedBert-MS-MARCO",local_files_only=True); n=snapshot_download("cross-encoder/nli-deberta-v3-large",local_files_only=True); SentenceTransformer(s,device="cpu"); CrossEncoder(n,device="cpu"); print("Local Treatment feature models ready.")'

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
    "${PYTHON}" -c 'import json,random,sys; from pathlib import Path; root=Path(sys.argv[1]); n=int(sys.argv[2]); nt=int(sys.argv[3]); seed=int(sys.argv[4]); out=Path(sys.argv[5]); ids=sorted(p.name for p in root.iterdir() if p.is_dir() and (p/"treatment_plan.json").is_file() and (p/"treatment_outcome.json").is_file()); assert 5<=nt<n<=len(ids),f"require 5 <= train-count < num-cases <= {len(ids)}"; random.Random(seed).shuffle(ids); chosen=ids[:n]; out.mkdir(parents=True,exist_ok=True); (out/"all.json").write_text(json.dumps(chosen,indent=2),encoding="utf-8"); (out/"train.json").write_text(json.dumps(chosen[:nt],indent=2),encoding="utf-8"); (out/"test.json").write_text(json.dumps(chosen[nt:],indent=2),encoding="utf-8")' "${CASE_ROOT}" "${NUM_CASES}" "${TRAIN_COUNT}" "${SEED}" "${SPLIT_ROOT}"
else
    SOURCE_TRAIN_IDS="${TRAIN_IDS}"
    SOURCE_TEST_IDS="${TEST_IDS}"
    TRAIN_IDS="${SPLIT_ROOT}/train.json"
    TEST_IDS="${SPLIT_ROOT}/test.json"
    "${PYTHON}" -c 'import json,sys; from pathlib import Path; train=list(map(str,json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))); test=list(map(str,json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))); overlap=set(train)&set(test); assert len(train)>=5,"at least five training cases are required"; assert test,"test split must not be empty"; assert len(set(train))==len(train),"train split contains duplicate case IDs"; assert len(set(test))==len(test),"test split contains duplicate case IDs"; assert not overlap,f"train/test overlap: {sorted(overlap)}"; out=Path(sys.argv[3]); out.mkdir(parents=True,exist_ok=True); (out/"train.json").write_text(json.dumps(train,indent=2),encoding="utf-8"); (out/"test.json").write_text(json.dumps(test,indent=2),encoding="utf-8"); (out/"all.json").write_text(json.dumps(train+test,indent=2),encoding="utf-8")' "${SOURCE_TRAIN_IDS}" "${SOURCE_TEST_IDS}" "${SPLIT_ROOT}"
fi
ALL_IDS="${SPLIT_ROOT}/all.json"
"${PYTHON}" -c 'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); ids=list(map(str,json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))); bad=[cid for cid in ids if not (root/cid/"treatment_plan.json").is_file() or not (root/cid/"treatment_outcome.json").is_file()]; assert not bad,f"ineligible or missing treatment cases: {bad}"' "${CASE_ROOT}" "${ALL_IDS}"

echo "RareTreatment code demo"
echo "  cases:       ${ALL_IDS}"
echo "  train/test:  ${TRAIN_IDS} / ${TEST_IDS}"
echo "  models:      ${MODELS_CSV}"
echo "  output:      ${OUTPUT_ROOT}"

for MODEL_TAG in "${MODELS[@]}"; do
    "${PYTHON}" -m rare_treatment.training.generate_llm_outputs \
        "${CASE_ROOT}" "${LLM_ROOT}" --model "${MODEL_TAG}" \
        --config "${CONFIG_PATH}" --case-ids "${ALL_IDS}" \
        --num-workers "${NUM_WORKERS}"

done

# The LLM-as-judge implementation is intentionally not distributed. Reproduce
# the paper's evaluation procedure, then provide the score files and rerun.
# Candidate generation is resumable, so existing outputs are reused.
MISSING_SCORES="$("${PYTHON}" -c 'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); ids=json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")); models=sys.argv[3:]; missing=[str(root/m/c/"treatment_score.json") for m in models for c in ids if not (root/m/c/"treatment_score.json").is_file()]; print("\n".join(missing))' "${SCORE_ROOT}" "${ALL_IDS}" "${MODELS[@]}")"
if [[ -n "${MISSING_SCORES}" ]]; then
    echo "[ERROR] LLM-as-judge code is not included. Reproduce the judge from the paper and provide score JSONs before feature construction." >&2
    echo "[ERROR] Expected e.g.: $(head -n 1 <<< "${MISSING_SCORES}")" >&2
    exit 1
fi

"${PYTHON}" -m rare_treatment.training.build_features \
    --case_root "${CASE_ROOT}" --llm_root "${LLM_ROOT}" --score_root "${SCORE_ROOT}" \
    --train_ids "${TRAIN_IDS}" --test_ids "${TEST_IDS}" \
    --out_dir "${FEATURE_ROOT}" --num_gpus "${FEATURE_GPUS}"

TRAIN_ARGS=()
if [[ "${USE_GPU}" -eq 0 ]]; then
    TRAIN_ARGS=(--force-cpu)
fi
"${PYTHON}" -m rare_treatment.training.train_ranker \
    --data-dir "${FEATURE_ROOT}" --out-dir "${MODEL_ROOT}" \
    --results-dir "${RESULT_ROOT}" --objective rank:ndcg \
    --n-splits 5 --target-k 3 \
    --save-models "${TRAIN_ARGS[@]}"

echo "Demo complete: ${OUTPUT_ROOT}"

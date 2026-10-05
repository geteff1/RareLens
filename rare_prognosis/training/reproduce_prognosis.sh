#!/usr/bin/env bash
# End-to-end RarePrognosis code demo on a deterministic subset of data_500.

set -euo pipefail

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
OUTPUT_ROOT="${REPO_ROOT}/outputs/prognosis_demo"
NUM_WORKERS=4
CV_FOLDS=5
TASK="all"

usage() {
    cat <<'EOF'
Usage: bash rare_prognosis/training/reproduce_prognosis.sh --models MODEL[,MODEL...] [OPTIONS]

Required:
  --models LIST          Comma-separated llm_config.json model names or tags

Options:
  --case-root DIR       Case directory (default: data_500)
  --config PATH         LLM config (default: llm_config.json)
  --num-cases N         Auto-split case count (default: 10)
  --train-count N       Auto-split training count (default: 8)
  --seed N              Auto-split and model seed (default: 42)
  --train-ids PATH      Existing train case-ID JSON; requires --test-ids
  --test-ids PATH       Existing test case-ID JSON; requires --train-ids
  --output-root DIR     Demo output directory (default: outputs/prognosis_demo)
  --python PATH         Python interpreter (default: python)
  --num-workers N       Concurrent LLM requests per model (default: 4)
  --cv-folds N          Maximum stratified CV folds (default: 5)
  --task TASK           overall_outcome|functional_status|symptom_burden|all
  -h, --help            Show this help
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
        --python) PYTHON="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --cv-folds) CV_FOLDS="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

[[ -n "${MODELS_CSV}" ]] || { echo "[ERROR] --models is required." >&2; usage >&2; exit 1; }
[[ -d "${CASE_ROOT}" ]] || { echo "[ERROR] Case root does not exist: ${CASE_ROOT}" >&2; exit 1; }
[[ -f "${CONFIG_PATH}" ]] || { echo "[ERROR] LLM config does not exist: ${CONFIG_PATH}" >&2; exit 1; }
if [[ -n "${TRAIN_IDS}" && -z "${TEST_IDS}" ]] || [[ -z "${TRAIN_IDS}" && -n "${TEST_IDS}" ]]; then
    echo "[ERROR] --train-ids and --test-ids must be provided together." >&2
    exit 1
fi
case "${TASK}" in
    overall_outcome|functional_status|symptom_burden|all) ;;
    *) echo "[ERROR] Invalid --task: ${TASK}" >&2; exit 1 ;;
esac

IFS=',' read -r -a MODELS <<< "${MODELS_CSV}"
"${PYTHON}" -c 'import json,sys; cfg=json.load(open(sys.argv[1],encoding="utf-8")); names={str(x.get("model")) for x in cfg}; names.update(t for x in cfg for t in x.get("tags",[])); missing=[x.strip() for x in sys.argv[2:] if x.strip() not in names]; assert not missing,f"models absent from config: {missing}"' "${CONFIG_PATH}" "${MODELS[@]}"
"${PYTHON}" -c 'import numpy,openai,sklearn; print("Python dependencies ready.")'

SPLIT_ROOT="${OUTPUT_ROOT}/splits"
LLM_ROOT="${OUTPUT_ROOT}/llm_outputs"
FEATURE_ROOT="${OUTPUT_ROOT}/features"
MODEL_ROOT="${OUTPUT_ROOT}/model"
RESULT_ROOT="${OUTPUT_ROOT}/results"
mkdir -p "${SPLIT_ROOT}" "${LLM_ROOT}" "${FEATURE_ROOT}" "${MODEL_ROOT}" "${RESULT_ROOT}"

if [[ -z "${TRAIN_IDS}" ]]; then
    TRAIN_IDS="${SPLIT_ROOT}/train.json"
    TEST_IDS="${SPLIT_ROOT}/test.json"
    "${PYTHON}" -c 'import json,random,sys; from pathlib import Path; root=Path(sys.argv[1]); n=int(sys.argv[2]); nt=int(sys.argv[3]); seed=int(sys.argv[4]); out=Path(sys.argv[5]); req=["prognosis_prediction.json","prognosis_new.json"]; ids=sorted(p.name for p in root.iterdir() if p.is_dir() and all((p/f).is_file() for f in req)); assert 2<=nt<n<=len(ids),f"require 2 <= train-count < num-cases <= {len(ids)}"; random.Random(seed).shuffle(ids); chosen=ids[:n]; out.mkdir(parents=True,exist_ok=True); [(out/name).write_text(json.dumps(vals,indent=2),encoding="utf-8") for name,vals in (("all.json",chosen),("train.json",chosen[:nt]),("test.json",chosen[nt:]))]' "${CASE_ROOT}" "${NUM_CASES}" "${TRAIN_COUNT}" "${SEED}" "${SPLIT_ROOT}"
else
    SOURCE_TRAIN_IDS="${TRAIN_IDS}"
    SOURCE_TEST_IDS="${TEST_IDS}"
    TRAIN_IDS="${SPLIT_ROOT}/train.json"
    TEST_IDS="${SPLIT_ROOT}/test.json"
    "${PYTHON}" -c 'import json,sys; from pathlib import Path; train=list(map(str,json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))); test=list(map(str,json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))); overlap=set(train)&set(test); assert len(train)>=2,"at least two training cases are required"; assert test,"test split must not be empty"; assert not overlap,f"train/test overlap: {sorted(overlap)}"; out=Path(sys.argv[3]); out.mkdir(parents=True,exist_ok=True); [(out/name).write_text(json.dumps(vals,indent=2),encoding="utf-8") for name,vals in (("train.json",train),("test.json",test),("all.json",train+test))]' "${SOURCE_TRAIN_IDS}" "${SOURCE_TEST_IDS}" "${SPLIT_ROOT}"
fi
ALL_IDS="${SPLIT_ROOT}/all.json"
"${PYTHON}" -c 'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); ids=list(map(str,json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))); req=["prognosis_prediction.json","prognosis_new.json"]; bad=[cid for cid in ids if not all((root/cid/f).is_file() for f in req)]; assert not bad,f"ineligible or missing cases: {bad}"' "${CASE_ROOT}" "${ALL_IDS}"

echo "RarePrognosis code demo"
echo "  cases:       ${ALL_IDS}"
echo "  train/test:  ${TRAIN_IDS} / ${TEST_IDS}"
echo "  models:      ${MODELS_CSV}"
echo "  output:      ${OUTPUT_ROOT}"

for MODEL_TAG in "${MODELS[@]}"; do
    MODEL_TAG="${MODEL_TAG//[[:space:]]/}"
    "${PYTHON}" -m rare_prognosis.training.generate_llm_outputs \
        "${CASE_ROOT}" "${LLM_ROOT}" --model "${MODEL_TAG}" \
        --config "${CONFIG_PATH}" --case-ids "${ALL_IDS}" \
        --num-workers "${NUM_WORKERS}"
done

"${PYTHON}" -m rare_prognosis.training.prepare_data \
    --case-root "${CASE_ROOT}" --llm-root "${LLM_ROOT}" \
    --result-root "${RESULT_ROOT}" --train-ids "${TRAIN_IDS}" --test-ids "${TEST_IDS}"
"${PYTHON}" -m rare_prognosis.training.build_features \
    --results-root "${RESULT_ROOT}" --models-root "${LLM_ROOT}" \
    --train-ids "${TRAIN_IDS}" --test-ids "${TEST_IDS}" --out-dir "${FEATURE_ROOT}" --task "${TASK}"
"${PYTHON}" -m rare_prognosis.training.train_models \
    --features-dir "${FEATURE_ROOT}" --out-dir "${MODEL_ROOT}" \
    --task "${TASK}" --seed "${SEED}" --cv-folds "${CV_FOLDS}"
"${PYTHON}" -m rare_prognosis.training.infer_models \
    --results-root "${RESULT_ROOT}" --models-root "${LLM_ROOT}" \
    --train-ids "${TRAIN_IDS}" --test-ids "${TEST_IDS}" --models-dir "${MODEL_ROOT}" --task "${TASK}"

echo "Demo complete: ${OUTPUT_ROOT}"

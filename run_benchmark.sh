#!/usr/bin/env bash
set -euo pipefail

DOMINO_CODE="${DOMINO_CODE:?set DOMINO_CODE to the released Domino code directory}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the target model path or HF id}"
DRAFT_PATH="${DRAFT_PATH:?set DRAFT_PATH to the Domino draft path or HF id}"

DATASET="${DATASET:-gsm8k}"
MAX_SAMPLES="${MAX_SAMPLES:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
BUDGETS="${BUDGETS:-16}"
METHODS="${METHODS:-chain,marg,cond}"
NODE_TOPK="${NODE_TOPK:-8}"
CORR_TOPM="${CORR_TOPM:-64}"
PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-runs/$(date +%Y%m%d_%H%M%S)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/run.log"
OUTJSONL="${OUT_DIR}/tps_${DATASET}_T${TEMPERATURE}.jsonl"
SMOKE_FLAG=""
[ "${SMOKE:-0}" = "1" ] && SMOKE_FLAG="--smoke"

{
  echo "=== DominoTree benchmark ==="
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "model: ${MODEL_PATH}"
  echo "draft: ${DRAFT_PATH}"
  echo "dataset: ${DATASET} samples: ${MAX_SAMPLES} max_new_tokens: ${MAX_NEW_TOKENS}"
  echo "methods: ${METHODS} budgets: ${BUDGETS} node_topk: ${NODE_TOPK} corr_topm: ${CORR_TOPM}"
  command -v nvidia-smi >/dev/null && nvidia-smi || true
  echo
  "${PYTHON}" "${SCRIPT_DIR}/benchmark.py" \
    --model-name-or-path "${MODEL_PATH}" \
    --draft-name-or-path "${DRAFT_PATH}" \
    --domino-code "${DOMINO_CODE}" \
    --dataset "${DATASET}" \
    --max-samples "${MAX_SAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --budgets "${BUDGETS}" \
    --methods "${METHODS}" \
    --node-topk "${NODE_TOPK}" \
    --corr-topm "${CORR_TOPM}" \
    --out "${OUTJSONL}" \
    ${SMOKE_FLAG}
} 2>&1 | tee "${LOG}"

echo
echo "Output JSONL: ${OUTJSONL}"
echo "Log: ${LOG}"

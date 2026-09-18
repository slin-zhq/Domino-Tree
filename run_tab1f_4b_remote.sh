#!/usr/bin/env bash
# Collect the Qwen3-4B fused-frontier Table 1 data: AR + DominoTree at node budgets
# 16/32/64, GPU-native fused builder, all 8 datasets x 3 temperatures.
#
# This is the exact collection behind results/raw/tab1f_4b/ (see results/README.md).
# It is a driver around benchmark.py, not a new benchmark -- same harness as
# run_benchmark.sh, just looped over datasets/temperatures/budgets with resumability
# (a cell that already has valid rows for every expected method is skipped).
#
# Run it wherever the model/draft/Domino-code paths below resolve (a single GPU box,
# local or remote -- there is nothing RunPod- or SSH-specific here; if you are driving
# it from another machine, `ssh` in yourself first, or wrap this script in your own
# orchestration).
set -Eeuo pipefail

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the Qwen3-4B target model path or HF id}"
DRAFT_PATH="${DRAFT_PATH:?set DRAFT_PATH to the Qwen3-4B-Domino-b16 draft path or HF id}"
DOMINO_CODE="${DOMINO_CODE:?set DOMINO_CODE to the released Domino code directory}"
DOMINOTREE_FRONTIER_SRC="${DOMINOTREE_FRONTIER_SRC:?set DOMINOTREE_FRONTIER_SRC to sglang_dominotree/src/dominotree_sglang/tree/frontier.py}"

PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-results/raw_repro/tab1f_4b/out}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:-0}"
DATASETS=(gsm8k math500 aime25 humaneval mbpp livecodebench mt-bench alpaca)

mkdir -p "${OUT_DIR}"
rm -f "${OUT_DIR}/../DONE" "${OUT_DIR}/../FAILED"
exec >>"${OUT_DIR}/../run.log" 2>&1

on_error() {
    rc=$?
    echo "[$(date -Is)] FAILED (exit=$rc)"
    touch "${OUT_DIR}/../FAILED"
    exit "$rc"
}
trap on_error ERR

# Do not join a GPU occupied by another process. Re-check immediately before launch.
apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk '$1 ~ /^[0-9]+$/')
if [ -n "$apps" ]; then
    echo "[$(date -Is)] GPU $GPU is occupied by PID(s): $apps"
    exit 2
fi

export DOMINOTREE_FRONTIER_SRC
export DOMINOTREE_BUILDER_GRU_TABLE=1

validate_rows() {
    local f=$1
    shift
    local expected actual n method
    expected=$(IFS=,; echo "$*")
    actual=$(jq -r '[.[].method] | unique | sort | join(",")' "$f")
    if [ "$actual" != "$expected" ]; then
        echo "[$(date -Is)] unexpected labels in $f: got=$actual expected=$expected"
        return 1
    fi
    for method in "$@"; do
        n=$(jq -r --arg method "$method" '[.[] | select(.method == $method) | .sample_idx] | unique | length' "$f")
        if [ "$n" -ne 50 ]; then
            echo "[$(date -Is)] bad sample count in $f for $method: $n (expected 50)"
            return 1
        fi
    done
    echo "[$(date -Is)] row check $f"
    jq -r 'group_by(.method)[] | "  \(.[0].method): \(length) rows, \([.[].sample_idx] | unique | length) samples"' "$f"
}

run_case() {
    local f=$1 methods=$2 budgets=$3 temp=$4 dataset=$5
    shift 5
    if [ -s "$f" ] && validate_rows "$f" "$@"; then
        echo "[$(date -Is)] [skip-valid] $dataset T=$temp methods=$methods budgets=$budgets"
        return
    fi
    rm -f "$f"
    echo "[$(date -Is)] START $dataset T=$temp methods=$methods budgets=$budgets"
    (
        cd "$SCRIPT_DIR"
        CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u benchmark.py \
            --model-name-or-path "$MODEL_PATH" \
            --draft-name-or-path "$DRAFT_PATH" \
            --domino-code "$DOMINO_CODE" \
            --methods "$methods" --budgets "$budgets" --corr-topm 64 --node-topk 8 \
            --builder frontier --max-samples 50 --max-new-tokens 2048 \
            --temperature "$temp" --dataset "$dataset" --out "$f"
    )
    validate_rows "$f" "$@"
}

# Stage A: one process per dataset; AR is intentionally collected once while
# all three DominoTree budgets are emitted as distinguishable method labels.
for ds in "${DATASETS[@]}"; do
    run_case "$OUT_DIR/${ds}_T0.0.jsonl" ar,dominotree 16,32,64 0.0 "$ds" \
        ar dominotree@16 dominotree@32 dominotree@64
done

# Stage B: AR is temperature-independent and is deliberately not repeated.
for temp in 0.5 1.0; do
    for ds in "${DATASETS[@]}"; do
        run_case "$OUT_DIR/${ds}_T${temp}.jsonl" dominotree 32,64 "$temp" "$ds" \
            dominotree@32 dominotree@64
    done
done

touch "${OUT_DIR}/../DONE"
trap - ERR
echo "[$(date -Is)] TAB1F_4B DONE"

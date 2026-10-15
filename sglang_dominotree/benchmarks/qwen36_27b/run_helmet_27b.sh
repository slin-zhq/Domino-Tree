#!/usr/bin/env bash
# Qwen3.6-27B HELMET long-context sweep: 5 methods x {infbench_sum, multi_lexsum} x
# {8K, 16K, 32K}, n=50/cell, HELMET's own 1200-token generation cap, bs=1, T=0.
# Reuses ../helmet/helmet_longctx.py unmodified.
#
# PREREQUISITE -- build this target's own prompt set first (its chat template differs
# from Qwen3-4B/8B, so their prompts/ directories are NOT reusable here):
#
#   git clone https://github.com/princeton-nlp/HELMET ~/HELMET
#   python ../helmet/helmet_source_dump.py --helmet-repo ~/HELMET \
#     --out-dir ./helmet_src --tasks infbench_sum,multi_lexsum \
#     --length-bins 8192,16384,32768 --limit 100
#   python ../helmet/helmet_prep.py --out-dir ./prompts/Qwen3.6-27B \
#     --tasks infbench_sum,multi_lexsum --length-bins 8192,16384,32768 --limit 100 \
#     --model-path "$MODEL" --helmet-data-glob "./helmet_src/{task}_{bin}.jsonl" \
#     --prompt-field helmet_prompt
#
# See ../helmet/README.md for what each step does and why (it is the SAME two-step
# prep as the Qwen3-4B/8B bundle; only --model-path / --out-dir change).
#
# Required env: MODEL, DRAFT_DOMINO, DRAFT_DFLASH, PROMPTS (the --out-dir above).
# Optional env: PY, OUT, METHODS, NPR (n-prompts per cell, default 50).
#
# Usage:
#   MODEL=/path/Qwen3.6-27B DRAFT_DOMINO=/path/qwen3.6-27B-domino \
#   DRAFT_DFLASH=/path/Qwen3.6-27B-DFlash PROMPTS=./prompts/Qwen3.6-27B \
#     setsid bash run_helmet_27b.sh > orch_helmet_27b.log 2>&1 < /dev/null &
set -uo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset DOMINOTREE_DIAG_FLAT_GDN
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_serverlib_27b.sh"
MODEL="${MODEL:?set MODEL (target model)}"
PROMPTS="${PROMPTS:?set PROMPTS (dir built by helmet_prep.py for THIS model)}"
PY="${PY:-python}"
LABEL="$(basename "$MODEL")"
OUT="${OUT:-$HERE/out/$LABEL/helmet}"
METHODS="${METHODS:-dominotree domino_chain dflash mtp ar}"
NPR="${NPR:-50}"

log(){ echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] helmet_27b: $*"; }
mkdir -p "$OUT/logs"

for M in $METHODS; do
  F="$OUT/$M/helmet.jsonl"
  if [ -s "$F" ]; then log "skip $M (exists)"; continue; fi
  log "================ METHOD=$M ================"
  ENVS=()
  if [ "$M" = domino_chain ] || [ "$M" = dominotree ]; then
    ENVS+=(SGLANG_PLUGINS=dominotree DRAFT_DOMINO="${DRAFT_DOMINO:-}")
  fi
  [ "$M" = dflash ] && ENVS+=(DRAFT_DFLASH="${DRAFT_DFLASH:-}")

  start_server_27b "$M" "$OUT/logs/server_${M}.log" "${ENVS[@]}" MODEL="$MODEL" PY="$PY" || continue
  mkdir -p "$OUT/$M"
  # The driver derives helmet.jsonl.prompts.jsonl from helmet.jsonl.part.
  # Promote the sidecar first and the final summary last (the resume marker).
  run_client_27b "$PY" -u "$HERE/../helmet/helmet_longctx.py" --host 127.0.0.1 --port "$PORT27B" \
      --method "$M" --model-label "$LABEL" --prompts-dir "$PROMPTS" \
      --tasks infbench_sum,multi_lexsum --length-bins 8192,16384,32768 \
      --n-prompts "$NPR" --temperature 0.0 --request-timeout 1800 --out "$F.part" \
      >> "$OUT/logs/driver_${M}.log" 2>&1 \
    && mv "$OUT/$M/helmet.jsonl.prompts.jsonl" "$OUT/$M/helmet.prompts.jsonl" \
    && mv "$F.part" "$F" \
    && log "done $M: $(wc -l < "$F") rows" \
    || log "!! $M helmet driver failed (see logs/driver_${M}.log)"
  stop_server_27b
done
log "================ ALL DONE ================"

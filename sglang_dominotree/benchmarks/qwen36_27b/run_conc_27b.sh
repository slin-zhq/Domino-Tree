#!/usr/bin/env bash
# Qwen3.6-27B concurrency sweep: 5 methods, each at its own highest crash-free admission
# cap, gsm8k:128/mbpp:128/mt-bench:80 x concurrency {1,2,4,8,16,32}, 512 tokens, T=0.
# Reuses ../concurrency/bench_concurrency.py unmodified.
#
# Required env: MODEL, DRAFT_DOMINO, DRAFT_DFLASH (same as run_bs1_27b.sh).
# Optional env: PY, OUT, METHODS, CAPS (default "32 16 8", tried high to low per method).
#
# Usage:
#   MODEL=/path/Qwen3.6-27B DRAFT_DOMINO=/path/qwen3.6-27B-domino \
#   DRAFT_DFLASH=/path/Qwen3.6-27B-DFlash \
#     setsid bash run_conc_27b.sh > orch_conc_27b.log 2>&1 < /dev/null &
set -uo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset DOMINOTREE_DIAG_FLAT_GDN
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_serverlib_27b.sh"
MODEL="${MODEL:?set MODEL (target model)}"
PY="${PY:-python}"
LABEL="$(basename "$MODEL")"
OUT="${OUT:-$HERE/out/$LABEL/conc}"
METHODS="${METHODS:-dominotree domino_chain dflash mtp ar}"
CAPS="${CAPS:-32 16 8}"

log(){ echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] conc_27b: $*"; }
mkdir -p "$OUT/logs"

for M in $METHODS; do
  F="$OUT/$M/$M.jsonl"
  if [ -s "$F" ]; then log "skip $M (exists)"; continue; fi
  log "================ METHOD=$M ================"
  ENVS=()
  if [ "$M" = domino_chain ] || [ "$M" = dominotree ]; then
    ENVS+=(SGLANG_PLUGINS=dominotree DRAFT_DOMINO="${DRAFT_DOMINO:-}")
  fi
  [ "$M" = dflash ] && ENVS+=(DRAFT_DFLASH="${DRAFT_DFLASH:-}")

  cap=""
  for C in $CAPS; do
    start_server_27b "$M" "$OUT/logs/server_${M}_cap${C}.log" "${ENVS[@]}" \
        MODEL="$MODEL" PY="$PY" MEMFRAC=0.85 MAXRUN="$C" CGMAXBS=8 \
      && { cap="$C"; break; }
  done
  [ -n "$cap" ] || { log "!! $M did not boot at any cap in {$CAPS}"; continue; }
  mkdir -p "$OUT/$M"; echo "$cap" > "$OUT/$M/admission_cap.txt"
  log "$M admission cap = $cap"

  run_client_27b "$PY" -u "$HERE/../concurrency/bench_concurrency.py" --host 127.0.0.1 --port "$PORT27B" \
      --method "$M" --model-path "$MODEL" --tasks gsm8k:128,mbpp:128,mt-bench:80 \
      --concurrencies 1,2,4,8,16,32 --max-new-tokens 512 --warmup-requests 1 \
      --warmup-max-new-tokens 256 --skip-first 0 --temperature 0.0 --timeout-s 1800 \
      --out-jsonl "$F.part" --out-md "$OUT/$M/$M.md" \
      >> "$OUT/logs/driver_${M}.log" 2>&1 \
    && mv "$F.part" "$F" && log "done $M: $(wc -l < "$F") cells" \
    || log "!! $M concurrency driver failed (see logs/driver_${M}.log)"
  stop_server_27b
done
log "================ ALL DONE ================"

#!/usr/bin/env bash
# Single-stream (bs=1) serving sweep: 5 methods x 8 datasets x 3 temperatures.
# Reproduces the paper's SGLang bs=1 table (per-dataset detail in the appendix).
#
# One server per method, drained between methods; all methods share identical
# serving flags (see ../launch_server.sh).
#
# Required env:
#   MODEL          target model path or HF id (also used for the chat template)
#   DRAFT_DOMINO   Domino draft  (domino_chain + dominotree)
#   DRAFT_DFLASH   DFlash draft
#   DRAFT_EAGLE3   EAGLE-3 draft
# Optional env:
#   TP        tensor-parallel size          default 1
#   PY        python executable             default python
#   OUT       output root                   default ./out/<basename MODEL>
#   DATASETS  default gsm8k,math500,aime25,humaneval,mbpp,livecodebench,mt-bench,alpaca
#   TEMPS     default 0.0,0.5,1.0
#   NSAMPLES  prompts per dataset           default 50
#   MAXNEW    max new tokens                default 2048
#   MEMFRAC   --mem-fraction-static         default 0.7  (the validated bs=1 value)
#   MAXRUN    admission cap                 default 64   (the validated bs=1 value;
#             irrelevant to bs=1 timing -- only one request is ever in flight -- but
#             kept identical to the runs that produced the paper's numbers)
#   METHODS   default "dominotree domino_chain dflash eagle3 ar"
#
# Usage (detach; monitor with tail -f orch_bs1.log):
#   MODEL=Qwen/Qwen3-4B DRAFT_DOMINO=./Qwen3-4B-Domino-b16 \
#   DRAFT_DFLASH=./Qwen3-4B-DFlash-b16 DRAFT_EAGLE3=./Qwen3-4B_eagle3 \
#     setsid bash run_bs1_all.sh > orch_bs1.log 2>&1 < /dev/null &
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:?set MODEL (target model)}"
TP="${TP:-1}"
PY="${PY:-python}"
LABEL="$(basename "$MODEL")"
OUT="${OUT:-$HERE/out/$LABEL}"
DATASETS="${DATASETS:-gsm8k,math500,aime25,humaneval,mbpp,livecodebench,mt-bench,alpaca}"
TEMPS="${TEMPS:-0.0,0.5,1.0}"
NSAMPLES="${NSAMPLES:-50}"
MAXNEW="${MAXNEW:-2048}"
MEMFRAC="${MEMFRAC:-0.7}"
MAXRUN="${MAXRUN:-64}"
METHODS="${METHODS:-dominotree domino_chain dflash eagle3 ar}"

declare -A PORTS=( [dominotree]=31500 [domino_chain]=31501 [dflash]=31502 [eagle3]=31503 [ar]=31504 )
log(){ echo "[$(date '+%F %T')] bs1: $*"; }

# Kill the server on $1 and wait for GPU memory to settle before the next launch.
drain(){
  local port="$1"
  [ -n "$port" ] && pkill -9 -f "[s]glang.launch_server.*${port}" 2>/dev/null || true
  for _ in $(seq 1 18); do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    [ -z "$used" ] && break
    [ "$used" -lt 2000 ] && break
    sleep 5
  done
  sleep 3
}

mkdir -p "$OUT"
log "MODEL=$MODEL TP=$TP DATASETS=$DATASETS TEMPS=$TEMPS N=$NSAMPLES MAXNEW=$MAXNEW"

PREV_PORT=""
for M in $METHODS; do
  P=${PORTS[$M]:?no port mapped for method $M}
  log "================ METHOD=$M PORT=$P ================"
  drain "$PREV_PORT"

  log "launch server $M (TP=$TP MEMFRAC=$MEMFRAC MAXRUN=$MAXRUN)"
  setsid env METHOD="$M" PORT="$P" MODEL="$MODEL" TP="$TP" PY="$PY" \
      MEMFRAC="$MEMFRAC" MAXRUN="$MAXRUN" \
      DRAFT_DOMINO="${DRAFT_DOMINO:-}" DRAFT_DFLASH="${DRAFT_DFLASH:-}" \
      DRAFT_EAGLE3="${DRAFT_EAGLE3:-}" \
      bash "$HERE/../launch_server.sh" > "$HERE/server_${M}.log" 2>&1 < /dev/null &

  ok=0
  for _ in $(seq 1 180); do
    curl -fsS "http://127.0.0.1:$P/health" >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  if [ "$ok" -ne 1 ]; then
    log "!! SERVER $M FAILED TO BOOT (900s) -- see server_${M}.log; skipping"
    echo boot_fail > "$HERE/status_${M}.txt"; PREV_PORT=$P; continue
  fi
  log "server $M HEALTHY"

  # Sanity: is speculation actually on? (accept length must exceed 1)
  if [ "$M" != "ar" ]; then
    SA=$(curl -fsS "http://127.0.0.1:$P/generate" -H 'Content-Type: application/json' \
        -d '{"text":"Explain gravity in one short paragraph.","sampling_params":{"temperature":0,"max_new_tokens":64}}' \
        | $PY -c 'import sys,json;print(json.load(sys.stdin)["meta_info"].get("spec_accept_length"))' 2>/dev/null)
    log "sanity spec_accept_length=$SA (expect > 1.0)"
  fi

  for DS in ${DATASETS//,/ }; do
    for T in ${TEMPS//,/ }; do
      OUTF="$OUT/$M/${DS}_T${T}.jsonl"
      mkdir -p "$(dirname "$OUTF")"
      if [ -s "$OUTF" ]; then log "skip $M/$DS/T$T (exists)"; continue; fi
      log "run $M $DS T=$T -> $OUTF"
      env CUDA_VISIBLE_DEVICES="" "$PY" -u "$HERE/bench_bs1.py" \
          --host 127.0.0.1 --port "$P" --method "$M" \
          --dataset "$DS" --temperature "$T" --max-samples "$NSAMPLES" \
          --max-new-tokens "$MAXNEW" --model-path "$MODEL" \
          --request-timeout 1800 --out "$OUTF" \
          >> "$HERE/run_${M}.log" 2>&1 || log "!! $M $DS T=$T exited nonzero (see run_${M}.log)"
    done
  done
  echo done > "$HERE/status_${M}.txt"
  PREV_PORT=$P
done

drain "$PREV_PORT"
log "================ ALL DONE ================"
echo ALLDONE > "$HERE/status_all.txt"

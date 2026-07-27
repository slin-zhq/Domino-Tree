#!/usr/bin/env bash
# HELMET long-context bs=1 orchestrator: runs all 5 methods in turn with drain-safe
# server swaps, one aggregated row per (task, length_bin) per method.
#
# Self-contained (no site-specific dependencies): the drain step is inlined below.
# PREREQ: prompts materialised by helmet_prep.py under $PROMPTS (see README.md).
#
# Env vars (paths have no site defaults — set them):
#   MODEL      target model path/name                         (required)
#   DRAFT_DOMINO   Domino draft (used by domino_chain + dominotree)   (required for those)
#   DRAFT_DFLASH   DFlash draft                                (required for dflash)
#   DRAFT_EAGLE3   EAGLE-3 draft                               (required for eagle3)
#   TP         tensor-parallel size            default 1  (use 2 for a big model / 2 GPUs)
#   PY         python executable               default python
#   MODEL_LABEL row `model` field              default basename(MODEL)
#   PROMPTS    prep output root                default ./prompts/<MODEL_LABEL>
#   OUT        output root                     default ./out/<MODEL_LABEL>
#   TASKS      default infbench_sum,multi_lexsum
#   BINS       default 8192,16384,32768
#   NPR        prompts per cell (prefix)       default 50
#   CLAMP      cap generation (cost knob)      default empty (faithful HELMET cap)
#   METHODS    default "dominotree domino_chain dflash eagle3 ar"
#
# Usage (detach the whole orchestrator; monitor via tail -f orch.log):
#   MODEL=Qwen/Qwen3-8B TP=2 DRAFT_DOMINO=./Qwen3-8B-Domino-b16 \
#   DRAFT_DFLASH=./Qwen3-8B-DFlash-b16 DRAFT_EAGLE3=./Qwen3-8B_eagle3 \
#     setsid bash run_helmet_all.sh > orch.log 2>&1 < /dev/null &
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:?set MODEL (target model)}"
TP="${TP:-1}"
PY="${PY:-python}"
MODEL_LABEL="${MODEL_LABEL:-$(basename "$MODEL")}"
PROMPTS="${PROMPTS:-$HERE/prompts/$MODEL_LABEL}"
OUT="${OUT:-$HERE/out/$MODEL_LABEL}"
TASKS="${TASKS:-infbench_sum,multi_lexsum}"
BINS="${BINS:-8192,16384,32768}"
NPR="${NPR:-50}"
CLAMP="${CLAMP:-}"
MEMFRAC="${MEMFRAC:-0.85}"
PREFILL_GRAPH="${PREFILL_GRAPH:-on}"   # set 'disabled' if a method OOMs in prefill (fair: same for all)
METHODS="${METHODS:-dominotree domino_chain dflash eagle3 ar}"

# KV ceiling we need: longest bin + generation headroom (summ cap 1200 + slack).
MAXBIN=$(echo "$BINS" | tr ',' '\n' | sort -n | tail -1)
NEED_TOK=$(( MAXBIN + 1500 ))

declare -A PORTS=( [dominotree]=31600 [domino_chain]=31601 [dflash]=31602 [eagle3]=31603 [ar]=31604 )
draft_for(){ case "$1" in dflash) echo "${DRAFT_DFLASH:-}";; eagle3) echo "${DRAFT_EAGLE3:-}";;
  domino_chain|dominotree) echo "${DRAFT_DOMINO:-}";; *) echo "";; esac; }

log(){ echo "[$(date '+%F %T')] orch: $*"; }

# --- inline drain: kill the server on $1 and wait for GPU memory to settle ---
drain(){
  local port="$1"
  [ -n "$port" ] && pkill -9 -f "[s]glang.launch_server.*${port}" 2>/dev/null || true
  # wait until GPU memory drops below a small threshold, or give up after ~90s.
  for _ in $(seq 1 18); do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    [ -z "$used" ] && break
    [ "$used" -lt 2000 ] && break
    sleep 5
  done
  sleep 3
}

cd "$HERE"; mkdir -p "$OUT"
log "MODEL=$MODEL TP=$TP LABEL=$MODEL_LABEL TASKS=$TASKS BINS=$BINS NPR=$NPR CLAMP='${CLAMP:-none}' NEED_TOK=$NEED_TOK"
[ -d "$PROMPTS" ] || { log "!! PROMPTS dir $PROMPTS missing — run helmet_prep.py first (README Step 2)"; exit 3; }

PREV_PORT=""
for M in $METHODS; do
  P=${PORTS[$M]:?no port mapped for method $M}
  log "================ METHOD=$M PORT=$P ================"
  rm -f "status_${M}.txt"
  drain "$PREV_PORT"

  DRAFT="$(draft_for "$M")"
  log "launch server $M (TP=$TP MEMFRAC=$MEMFRAC PREFILL_GRAPH=$PREFILL_GRAPH MAXRUN=4 CGMAXBS=4)"
  setsid env METHOD=$M PORT=$P MODEL="$MODEL" TP="$TP" DRAFT="$DRAFT" PY="$PY" \
      MEMFRAC="$MEMFRAC" PREFILL_GRAPH="$PREFILL_GRAPH" MAXRUN=4 CGMAXBS=4 \
      bash "$HERE/launch_helmet_server.sh" > "$HERE/server_${M}.log" 2>&1 < /dev/null &

  ok=0
  for _ in $(seq 1 180); do
    curl -fsS "http://127.0.0.1:$P/health" >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  if [ "$ok" -ne 1 ]; then
    log "!! SERVER $M FAILED TO BOOT (900s) — see server_${M}.log; skipping"
    echo boot_fail > "status_${M}.txt"; PREV_PORT=$P; continue
  fi
  log "server $M HEALTHY"

  MAXTOK=$(grep -aoiE "max_total_num_tokens[^0-9]*[0-9]+" "$HERE/server_${M}.log" \
           | grep -aoE "[0-9]+" | tail -1); MAXTOK=${MAXTOK:-0}
  if [ "$MAXTOK" -ge "$NEED_TOK" ]; then log "KV pool=$MAXTOK >= $NEED_TOK => $MAXBIN cell FITS";
  else log "!! KV pool=$MAXTOK < $NEED_TOK => longest bin may be REJECTED/TRUNCATED"; fi
  echo "max_total_num_tokens=$MAXTOK need=$NEED_TOK" > "kvpool_${M}.txt"

  if [ "$M" != "ar" ]; then
    SA=$(curl -fsS "http://127.0.0.1:$P/generate" -H 'Content-Type: application/json' \
        -d '{"text":"Explain gravity in one short paragraph.","sampling_params":{"temperature":0,"max_new_tokens":64}}' \
        | $PY -c 'import sys,json;print(json.load(sys.stdin)["meta_info"].get("spec_accept_length"))' 2>/dev/null)
    log "sanity spec_accept_length=$SA (expect > 1.0)"
    echo "sanity_spec_accept_length=$SA" >> "kvpool_${M}.txt"
  fi

  CLAMP_ARG=(); [ -n "$CLAMP" ] && CLAMP_ARG=(--gen-cap-clamp "$CLAMP")
  log "driver $M -> $OUT/${M}/helmet.jsonl"
  env CUDA_VISIBLE_DEVICES="" \
      "$PY" -u "$HERE/helmet_longctx.py" --host 127.0.0.1 --port "$P" \
        --method "$M" --model-label "$MODEL_LABEL" \
        --prompts-dir "$PROMPTS" --tasks "$TASKS" --length-bins "$BINS" \
        --n-prompts "$NPR" --temperature 0.0 --request-timeout 1800 "${CLAMP_ARG[@]}" \
        --out "$OUT/${M}/helmet.jsonl" > "$HERE/run_${M}.log" 2>&1
  rc=$?
  log "driver $M exit=$rc rows=$(wc -l < "$OUT/${M}/helmet.jsonl" 2>/dev/null || echo 0)"
  echo "done rc=$rc" > "status_${M}.txt"
  PREV_PORT=$P
done

drain "$PREV_PORT"
log "================ ALL DONE ================"
echo ALLDONE > status_all.txt

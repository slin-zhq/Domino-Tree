#!/usr/bin/env bash
# Concurrency (goodput) sweep: 5 methods x 3 datasets x concurrency 1..32.
# Reproduces the paper's SGLang concurrency table (per-dataset detail in the appendix).
#
# EACH METHOD RUNS AT ITS OWN ADMISSION CAP. Set CAP_<METHOD> to the highest
# crash-free `--max-running-requests` for that method on your GPU (find them with
# find_caps.sh, or start from the values in README.md). Everything else -- flags,
# datasets, prompts, generation length -- is identical across methods.
#
# Required env:
#   MODEL          target model path or HF id
#   DRAFT_DOMINO   Domino draft  (domino_chain + dominotree)
#   DRAFT_DFLASH   DFlash draft
#   DRAFT_EAGLE3   EAGLE-3 draft
# Optional env:
#   TP        tensor-parallel size    default 1
#   PY        python executable       default python
#   OUT       output root             default ./out/<basename MODEL>
#   TASKS     default gsm8k:128,mbpp:128,mt-bench:80
#   CONC      default 1,2,4,8,16,32
#   MAXNEW    max new tokens          default 512
#   MEMFRAC   --mem-fraction-static   default 0.7 at TP=1, 0.85 at TP>1 (validated)
#   CAP_AR CAP_EAGLE3 CAP_DFLASH CAP_DOMINO_CHAIN CAP_DOMINOTREE   default 32
#   METHODS   default "ar dominotree domino_chain dflash eagle3"
#
# Usage (detach; monitor with tail -f orch_conc.log):
#   MODEL=Qwen/Qwen3-8B TP=2 CAP_DOMINOTREE=8 CAP_DOMINO_CHAIN=16 \
#   CAP_DFLASH=16 CAP_EAGLE3=16 CAP_AR=32 \
#   DRAFT_DOMINO=./Qwen3-8B-Domino-b16 DRAFT_DFLASH=./Qwen3-8B-DFlash-b16 \
#   DRAFT_EAGLE3=./Qwen3-8B_eagle3 \
#     setsid bash run_conc_all.sh > orch_conc.log 2>&1 < /dev/null &
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:?set MODEL (target model)}"
TP="${TP:-1}"
PY="${PY:-python}"
LABEL="$(basename "$MODEL")"
OUT="${OUT:-$HERE/out/$LABEL}"
TASKS="${TASKS:-gsm8k:128,mbpp:128,mt-bench:80}"
CONC="${CONC:-1,2,4,8,16,32}"
MAXNEW="${MAXNEW:-512}"
# Validated: 0.7 at TP=1 -- on a 16 GB card the tree's KV-move kernel needs transient
# headroom at high concurrency and OOMs above that; 0.85 at TP>1, where sharded weights
# leave more of each card for KV. Override if your GPU differs (same value for ALL methods).
if [ "$TP" -gt 1 ]; then MEMFRAC="${MEMFRAC:-0.85}"; else MEMFRAC="${MEMFRAC:-0.7}"; fi
METHODS="${METHODS:-ar dominotree domino_chain dflash eagle3}"

declare -A PORTS=( [dominotree]=31660 [domino_chain]=31661 [dflash]=31662 [eagle3]=31663 [ar]=31664 )
cap_for(){ case "$1" in
  ar)           echo "${CAP_AR:-32}";;
  eagle3)       echo "${CAP_EAGLE3:-32}";;
  dflash)       echo "${CAP_DFLASH:-32}";;
  domino_chain) echo "${CAP_DOMINO_CHAIN:-32}";;
  dominotree)   echo "${CAP_DOMINOTREE:-32}";;
esac; }
log(){ echo "[$(date '+%F %T')] conc: $*"; }

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
log "MODEL=$MODEL TP=$TP TASKS=$TASKS CONC=$CONC MAXNEW=$MAXNEW"

PREV_PORT=""
for M in $METHODS; do
  P=${PORTS[$M]:?no port mapped for method $M}
  CAP="$(cap_for "$M")"
  log "================ METHOD=$M PORT=$P CAP=$CAP ================"
  drain "$PREV_PORT"

  setsid env METHOD="$M" PORT="$P" MODEL="$MODEL" TP="$TP" PY="$PY" \
      MEMFRAC="$MEMFRAC" MAXRUN="$CAP" CGMAXBS="$CAP" \
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
    echo "boot_fail cap=$CAP" > "$HERE/status_${M}.txt"; PREV_PORT=$P; continue
  fi
  log "server $M HEALTHY (cap=$CAP)"

  if [ "$M" != "ar" ]; then
    SA=$(curl -fsS "http://127.0.0.1:$P/generate" -H 'Content-Type: application/json' \
        -d '{"text":"Explain gravity in one short paragraph.","sampling_params":{"temperature":0,"max_new_tokens":64}}' \
        | $PY -c 'import sys,json;print(json.load(sys.stdin)["meta_info"].get("spec_accept_length"))' 2>/dev/null)
    log "sanity spec_accept_length=$SA (expect > 1.0)"
  fi

  mkdir -p "$OUT/$M"
  env CUDA_VISIBLE_DEVICES="" "$PY" -u "$HERE/bench_concurrency.py" \
      --host 127.0.0.1 --port "$P" --method "$M" --model-path "$MODEL" \
      --tasks "$TASKS" --concurrencies "$CONC" \
      --max-new-tokens "$MAXNEW" --warmup-requests 1 --warmup-max-new-tokens 256 \
      --skip-first 0 --temperature 0.0 --timeout-s 1800 \
      --out-jsonl "$OUT/$M/${M}.jsonl" --out-md "$OUT/$M/${M}.md" \
      > "$HERE/run_${M}.log" 2>&1
  rc=$?
  cells=$(wc -l < "$OUT/$M/${M}.jsonl" 2>/dev/null || echo 0)
  log "driver $M exit=$rc cells=$cells"
  # The cap is part of the result: record it next to the data.
  echo "done rc=$rc cap=$CAP cells=$cells" > "$HERE/status_${M}.txt"
  PREV_PORT=$P
done

drain "$PREV_PORT"
log "================ ALL DONE ================"
echo ALLDONE > "$HERE/status_all.txt"
